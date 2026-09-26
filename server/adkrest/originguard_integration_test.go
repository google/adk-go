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

package adkrest_test

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/gorilla/websocket"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
)

// The scenarios in these tests come from issue #1154, which reported that the
// ADK REST server accepts a /run_live WebSocket and REST calls from a page that
// reaches it by rebinding its own DNS name onto the loopback address.

const guardTestApp = "test-agent"

// newGuardServer serves the REST API over loopback and declares that bind, so
// the DNS-rebinding check is armed the way it is on a real loopback server.
// allowedOrigins goes to the server as configured.
func newGuardServer(t *testing.T, allowedOrigins ...string) *httptest.Server {
	t.Helper()

	root, err := agent.New(agent.Config{Name: guardTestApp, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  session.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
		ArtifactService: artifact.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(root),
		AllowedOrigins:  allowedOrigins,
		BindHost:        "127.0.0.1",
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)
	return ts
}

// hostOf returns the host:port a test server listens on.
func hostOf(t *testing.T, ts *httptest.Server) string {
	t.Helper()
	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatalf("url.Parse(%q) error = %v", ts.URL, err)
	}
	return u.Host
}

// dialRunLive opens a /run_live WebSocket with the given Host and Origin
// headers and returns the handshake status. A rebound page controls both, so
// the two are set independently of the address actually dialled.
func dialRunLive(t *testing.T, ts *httptest.Server, host, origin string) int {
	t.Helper()

	addr := hostOf(t, ts)
	header := http.Header{}
	if origin != "" {
		header.Set("Origin", origin)
	}
	dialer := websocket.Dialer{
		// Dial the real listener whatever Host says, which is what rebound DNS
		// does: the name resolves here and the browser sends the name it knows.
		NetDial: func(network, _ string) (net.Conn, error) {
			return net.Dial(network, addr)
		},
	}

	wsURL := "ws://" + host + "/run_live?appName=" + guardTestApp + "&userId=u1&sessionId=s1"
	conn, resp, err := dialer.DialContext(t.Context(), wsURL, header)
	if conn != nil {
		t.Cleanup(func() { _ = conn.Close() })
	}
	if resp == nil {
		t.Fatalf("Dial(%q) returned no response: %v", wsURL, err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp.StatusCode
}

func TestRunLiveRejectsRebindingAndForeignOrigins(t *testing.T) {
	ts := newGuardServer(t)
	loopback := hostOf(t, ts)

	for _, tc := range []struct {
		name       string
		host       string
		origin     string
		wantStatus int
	}{
		{
			// The PoC's DNS-rebinding case. gorilla's default check passes it,
			// because the page controls Origin and Host alike.
			name:       "rebound origin and host",
			host:       "evil.com" + portOf(t, loopback),
			origin:     "http://evil.com" + portOf(t, loopback),
			wantStatus: http.StatusForbidden,
		},
		{
			// Rebinding without an Origin at all: a browser omits it on a
			// request it considers same-origin, which a rebound page's is.
			name:       "rebound host with no origin",
			host:       "evil.com" + portOf(t, loopback),
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "cross-site browser",
			host:       loopback,
			origin:     "http://evil.com",
			wantStatus: http.StatusForbidden,
		},
		{
			// A non-browser client on this machine. It sends no Origin, and
			// nothing about the request distinguishes it from a legitimate CLI,
			// so the loopback bind is what keeps it local.
			name:       "loopback client with no origin",
			host:       loopback,
			wantStatus: http.StatusSwitchingProtocols,
		},
		{
			name:       "same-origin browser",
			host:       loopback,
			origin:     "http://" + loopback,
			wantStatus: http.StatusSwitchingProtocols,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := dialRunLive(t, ts, tc.host, tc.origin); got != tc.wantStatus {
				t.Errorf("/run_live handshake status = %d, want %d", got, tc.wantStatus)
			}
		})
	}
}

// newUndeclaredBindServer serves the REST API without saying what it binds,
// which is what the web launcher does today: it binds every interface, so the
// Host check stays off and the Origin check is on its own.
func newUndeclaredBindServer(t *testing.T, allowedOrigins ...string) *httptest.Server {
	t.Helper()

	root, err := agent.New(agent.Config{Name: guardTestApp, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(root),
		AllowedOrigins: allowedOrigins,
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)
	return ts
}

// TestRunLiveRejectsRebindingWithoutDeclaredBind is the case the web launcher
// is in until it binds loopback: nothing declares the bind, so the Host check
// is off. A WebSocket handshake always carries Origin, so the Origin check
// refuses a rebound page on its own.
func TestRunLiveRejectsRebindingWithoutDeclaredBind(t *testing.T) {
	// The launcher always passes -webui_address, so the allowlist is never
	// empty in practice and the check has to hold with one configured. This is
	// where the guard diverges from adk-python, which skips it once origins are
	// configured.
	ts := newUndeclaredBindServer(t, "http://localhost:8080")
	rebound := "evil.com" + portOf(t, hostOf(t, ts))

	got := dialRunLive(t, ts, rebound, "http://"+rebound)
	if want := http.StatusForbidden; got != want {
		t.Errorf("/run_live handshake status = %d, want %d", got, want)
	}
}

// TestUndeclaredBindServesForeignHosts pins the other side of that: with no
// declared bind, a request naming a host we do not recognize is served. A
// sidecar proxy, an nginx proxy_pass to 127.0.0.1 and Debian's 127.0.1.1
// hostname all arrive over loopback under such a Host, and none of them carries
// an Origin, so refusing them would take down ordinary traffic.
func TestUndeclaredBindServesForeignHosts(t *testing.T) {
	ts := newUndeclaredBindServer(t)

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, ts.URL+"/list-apps", nil)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext() error = %v", err)
	}
	req.Host = "adk.internal"

	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatalf("GET /list-apps error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if want := http.StatusOK; resp.StatusCode != want {
		t.Errorf("GET /list-apps status = %d, want %d", resp.StatusCode, want)
	}
}

func TestRunLiveAcceptsConfiguredOrigin(t *testing.T) {
	// A web UI served by a separate dev server, the reason -webui_address
	// exists. gorilla's default check rejects this, since Origin and Host
	// differ.
	const devUI = "http://localhost:4200"
	ts := newGuardServer(t, devUI)

	got := dialRunLive(t, ts, hostOf(t, ts), devUI)
	if want := http.StatusSwitchingProtocols; got != want {
		t.Errorf("/run_live handshake status = %d, want %d", got, want)
	}
}

func TestRESTRejectsRebindingAndForeignOrigins(t *testing.T) {
	ts := newGuardServer(t)
	loopback := hostOf(t, ts)

	for _, tc := range []struct {
		name       string
		host       string
		origin     string
		wantStatus int
	}{
		{
			name:       "rebound host",
			host:       "evil.com",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "cross-site browser",
			host:       loopback,
			origin:     "http://evil.com",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "same-origin browser",
			host:       loopback,
			origin:     "http://" + loopback,
			wantStatus: http.StatusOK,
		},
		{
			name:       "loopback client with no origin",
			host:       loopback,
			wantStatus: http.StatusOK,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, ts.URL+"/list-apps", nil)
			if err != nil {
				t.Fatalf("http.NewRequestWithContext() error = %v", err)
			}
			req.Host = tc.host
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}

			resp, err := ts.Client().Do(req)
			if err != nil {
				t.Fatalf("GET /list-apps error = %v", err)
			}
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != tc.wantStatus {
				t.Errorf("GET /list-apps status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
		})
	}
}

// TestWildcardOriginDisablesBothChecks pins the documented escape hatch, which
// an operator needs when the server is deliberately reachable under a name we
// cannot know.
func TestWildcardOriginDisablesBothChecks(t *testing.T) {
	ts := newGuardServer(t, "*")

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, ts.URL+"/list-apps", nil)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext() error = %v", err)
	}
	req.Host = "anything.example.com"
	req.Header.Set("Origin", "http://evil.com")

	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatalf("GET /list-apps error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if want := http.StatusOK; resp.StatusCode != want {
		t.Errorf("GET /list-apps status = %d, want %d", resp.StatusCode, want)
	}
}

// portOf returns the ":port" suffix of a host:port, for building a rebound Host
// that names the same port the listener is on.
func portOf(t *testing.T, hostPort string) string {
	t.Helper()
	i := strings.LastIndex(hostPort, ":")
	if i < 0 {
		t.Fatalf("host %q has no port", hostPort)
	}
	return hostPort[i:]
}
