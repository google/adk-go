// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package web

import (
	"io"
	"net"
	"net/http"
	"testing"
)

func TestH2CFlag(t *testing.T) {
	for _, tc := range []struct {
		name    string
		args    []string
		wantH2C bool
	}{
		{
			name: "disabled by default",
		},
		{
			name:    "enabled",
			args:    []string{"--h2c"},
			wantH2C: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			launcher := NewLauncher().(*webLauncher)
			if _, err := launcher.Parse(tc.args); err != nil {
				t.Fatalf("Parse(%v) failed: %v", tc.args, err)
			}

			srv := launcher.buildHTTPServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Request-Protocol", r.Proto)
				w.WriteHeader(http.StatusNoContent)
			}))
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatalf("net.Listen() failed: %v", err)
			}
			serveErr := make(chan error, 1)
			go func() {
				serveErr <- srv.Serve(listener)
			}()
			t.Cleanup(func() {
				if err := srv.Close(); err != nil {
					t.Errorf("server Close() failed: %v", err)
				}
				if err := <-serveErr; err != http.ErrServerClosed {
					t.Errorf("server Serve() error = %v, want %v", err, http.ErrServerClosed)
				}
			})

			url := "http://" + listener.Addr().String()
			assertProtocol(t, http.DefaultClient, url, 1)

			h2cProtocols := new(http.Protocols)
			h2cProtocols.SetUnencryptedHTTP2(true)
			h2cClient := &http.Client{
				Transport: &http.Transport{Protocols: h2cProtocols},
			}
			t.Cleanup(h2cClient.CloseIdleConnections)

			resp, err := h2cClient.Get(url)
			if !tc.wantH2C {
				if err == nil {
					if closeErr := resp.Body.Close(); closeErr != nil {
						t.Errorf("response body Close() failed: %v", closeErr)
					}
					t.Fatalf("h2c request unexpectedly succeeded with protocol %q", resp.Proto)
				}
				return
			}
			if err != nil {
				t.Fatalf("h2c request failed: %v", err)
			}
			defer func() {
				if err := resp.Body.Close(); err != nil {
					t.Errorf("response body Close() failed: %v", err)
				}
			}()
			if resp.ProtoMajor != 2 {
				t.Errorf("h2c response protocol = %q, want HTTP/2", resp.Proto)
			}
			if got := resp.Header.Get("X-Request-Protocol"); got != "HTTP/2.0" {
				t.Errorf("handler request protocol = %q, want %q", got, "HTTP/2.0")
			}
		})
	}
}

func assertProtocol(t *testing.T, client *http.Client, url string, wantMajor int) {
	t.Helper()

	resp, err := client.Get(url)
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Errorf("response body Close() failed: %v", err)
		}
	}()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("reading response body failed: %v", err)
	}
	if resp.ProtoMajor != wantMajor {
		t.Errorf("response protocol = %q, want HTTP/%d", resp.Proto, wantMajor)
	}
}

func TestHostBinding(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{
			// The web server must default to loopback-only so it cannot
			// accidentally be exposed to the network.
			name: "loopback by default",
			want: "127.0.0.1:8080",
		},
		{
			name: "explicit loopback",
			args: []string{"--host", "127.0.0.1"},
			want: "127.0.0.1:8080",
		},
		{
			name: "all interfaces",
			args: []string{"--host", "0.0.0.0"},
			want: "0.0.0.0:8080",
		},
		{
			name: "IPv6 loopback",
			args: []string{"--host", "::1"},
			want: "[::1]:8080",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			launcher := NewLauncher().(*webLauncher)
			if _, err := launcher.Parse(tc.args); err != nil {
				t.Fatalf("Parse(%v) failed: %v", tc.args, err)
			}
			srv := launcher.buildHTTPServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusNoContent)
			}))
			if got := srv.Addr; got != tc.want {
				t.Errorf("server Addr = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestWebURL(t *testing.T) {
	for _, tc := range []struct {
		name string
		host string
		port int
		want string
	}{
		{name: "localhost", host: "localhost", port: 8080, want: "http://localhost:8080"},
		// Loopback-ish hosts are normalized to "localhost" so the displayed
		// URL matches the ADK Web UI backend origin (http://localhost:8080/api)
		// and avoids a browser CORS mismatch.
		{name: "IPv4 loopback", host: "127.0.0.1", port: 8080, want: "http://localhost:8080"},
		{name: "IPv6 loopback", host: "::1", port: 8080, want: "http://localhost:8080"},
		{name: "all interfaces IPv4", host: "0.0.0.0", port: 8080, want: "http://localhost:8080"},
		{name: "all interfaces IPv6", host: "::", port: 8080, want: "http://localhost:8080"},
		// Non-loopback configured hosts are left untouched.
		{name: "custom hostname", host: "example.com", port: 8080, want: "http://example.com:8080"},
		{name: "custom IP", host: "192.168.1.10", port: 8080, want: "http://192.168.1.10:8080"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := &webLauncher{config: &webConfig{host: tc.host, port: tc.port}}
			if got := w.webURL(); got != tc.want {
				t.Errorf("webURL() = %q, want %q", got, tc.want)
			}
		})
	}
}
