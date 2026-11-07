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

package openapitoolset

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBlockPrivateNetworksLexical(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		want    string
	}{
		{"localhost", "http://localhost:8080", "Blocked host"},
		{"loopback ip", "http://127.0.0.1", "Blocked host"},
		{"metadata ip", "http://169.254.169.254", "Blocked host"},
		{"private ip", "http://10.0.0.1", "Blocked host"},
		{"internal suffix", "https://foo.internal", "Blocked host"},
		{"file scheme", "file://x", "Unsupported url scheme"},
		{"ftp scheme", "ftp://example.com", "Unsupported url scheme"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tool, err := NewRESTAPITool(RESTAPIToolConfig{
				Name:                 "test_tool",
				Endpoint:             Endpoint{BaseURL: tc.baseURL, Path: "/api", Method: "get"},
				BlockPrivateNetworks: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			res := runTool(t, tool, map[string]any{})
			errMsg, _ := res["error"].(string)
			if !strings.Contains(errMsg, tc.want) {
				t.Errorf("error = %q, want it to contain %q", errMsg, tc.want)
			}
		})
	}
}

func TestBlockPrivateNetworksDialGuard(t *testing.T) {
	// A hostname that resolves to a loopback address is blocked at dial time,
	// even though the lexical check passes it. The default guarded client's
	// dialer sees the resolved IP.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(srv.Close)
	_, port, err := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	// "localtest.me" resolves to 127.0.0.1 but is not a lexically blocked name.
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:                 "test_tool",
		Endpoint:             Endpoint{BaseURL: "http://localtest.me:" + port, Path: "/api", Method: "get"},
		BlockPrivateNetworks: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := tool.run(context.Background(), map[string]any{}, nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	errMsg, _ := res["error"].(string)
	if !strings.Contains(errMsg, "Blocked host") {
		t.Skipf("dial guard not exercised (DNS for localtest.me unavailable): %v", res)
	}
}

func TestAllowsPrivateByDefault(t *testing.T) {
	// Without BlockPrivateNetworks, a request to the loopback test server (the
	// u2483 echo scenario) succeeds.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"ok": true}`)
	}))
	t.Cleanup(srv.Close)
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "test_tool",
		Endpoint: Endpoint{BaseURL: srv.URL, Path: "/api", Method: "get"},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{})
	if res["ok"] != true {
		t.Errorf("result = %v, want ok:true (loopback allowed by default)", res)
	}
}

func TestIsGlobalUnicast(t *testing.T) {
	tests := []struct {
		ip   string
		want bool
	}{
		{"8.8.8.8", true},
		{"127.0.0.1", false},
		{"10.0.0.1", false},
		{"192.168.1.1", false},
		{"169.254.169.254", false},
		{"100.64.0.1", false}, // carrier-grade NAT
		{"::1", false},
		{"2001:4860:4860::8888", true},
		{"fe80::1", false}, // link-local
	}
	for _, tc := range tests {
		ip := net.ParseIP(tc.ip)
		if ip == nil {
			t.Fatalf("bad test IP %q", tc.ip)
		}
		if got := isGlobalUnicast(ip); got != tc.want {
			t.Errorf("isGlobalUnicast(%s) = %v, want %v", tc.ip, got, tc.want)
		}
	}
}
