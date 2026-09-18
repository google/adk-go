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

package originguard

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// loopbackBind is the config a server that binds 127.0.0.1 builds, which is
// what arms the DNS-rebinding check.
var loopbackBind = Config{BindHost: "127.0.0.1"}

// on returns a request to path as it would arrive on a connection accepted at
// localAddr, which is what the DNS-rebinding check reads. An empty localAddr
// leaves it unset, standing for a request that did not come from an
// [net/http.Server].
func on(localAddr, host string) *http.Request {
	r := httptest.NewRequest(http.MethodGet, "/list-apps", nil)
	r.Host = host
	if localAddr != "" {
		ctx := context.WithValue(r.Context(), http.LocalAddrContextKey, &net.TCPAddr{
			IP:   net.ParseIP(localAddr),
			Port: 8080,
		})
		r = r.WithContext(ctx)
	}
	return r
}

func TestCheckDNSRebinding(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bindHost string
		origins  []string
		host     string
		want     error
	}{
		{
			// The attack from issue #1154: a page on evil.com whose DNS flips
			// to 127.0.0.1. The browser considers the request same-origin and
			// so sends no Origin on a GET, leaving Host as the only tell.
			name:     "rebound host",
			bindHost: "127.0.0.1",
			host:     "evil.com:8080",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "rebound host on an IPv6 loopback bind",
			bindHost: "::1",
			host:     "evil.com:8080",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "rebound host on a bind named localhost",
			bindHost: "localhost",
			host:     "evil.com:8080",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "loopback host",
			bindHost: "127.0.0.1",
			host:     "127.0.0.1:8080",
		},
		{
			name:     "localhost",
			bindHost: "127.0.0.1",
			host:     "localhost:8080",
		},
		{
			// Host names are case-insensitive and may carry a root dot.
			name:     "LocalHost with a root dot",
			bindHost: "127.0.0.1",
			host:     "LocalHost.:8080",
		},
		{
			// A server on a routable address is legitimately reachable under
			// whatever name resolves to it, so the check stays off.
			name:     "any host on a routable bind",
			bindHost: "0.0.0.0",
			host:     "adk.example.com",
		},
		{
			// A bind we were not told about is not ours to guess. This is the
			// case that keeps a sidecar proxy, an nginx proxy_pass to
			// 127.0.0.1, and Debian's 127.0.1.1 hostname working: all three
			// arrive over loopback under a Host we would otherwise refuse.
			name: "undeclared bind",
			host: "adk.example.com",
		},
		{
			// Listing an origin vouches for its host, which is how a loopback
			// bind reachable through a same-machine reverse proxy keeps working.
			name:     "configured host",
			bindHost: "127.0.0.1",
			origins:  []string{"http://adk.internal"},
			host:     "adk.internal",
		},
		{
			name:     "unconfigured host with other origins configured",
			bindHost: "127.0.0.1",
			origins:  []string{"http://adk.internal"},
			host:     "evil.com",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "wildcard turns the check off",
			bindHost: "127.0.0.1",
			origins:  []string{"*"},
			host:     "evil.com",
		},
		{
			// Host has no list form, so a comma is header smuggling rather than
			// a client. net/http rejects two Host headers before we see them,
			// but permits a comma inside the one value.
			name:     "comma-separated host list",
			bindHost: "127.0.0.1",
			host:     "localhost:8080,evil.com",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "comma-separated list of allowed hosts",
			bindHost: "127.0.0.1",
			origins:  []string{"http://adk.internal"},
			host:     "adk.internal,evil.com",
			want:     ErrHostNotAllowed,
		},
		{
			// HTTP/1.1 and HTTP/2 both require Host, so an empty one is not a
			// request a browser can make.
			name:     "empty host",
			bindHost: "127.0.0.1",
		},
		{
			// The port-stripping must not read this as the loopback address
			// with a port, which is the whole reason a malformed authority
			// comes back whole.
			name:     "loopback address as a subdomain label",
			bindHost: "127.0.0.1",
			host:     "127.0.0.1:8080.evil.com",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "loopback name as a subdomain label",
			bindHost: "127.0.0.1",
			host:     "localhost.evil.com",
			want:     ErrHostNotAllowed,
		},
		{
			name:     "bracketed IPv6 loopback",
			bindHost: "::1",
			host:     "[::1]:8080",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{BindHost: tc.bindHost, AllowedOrigins: tc.origins}
			// The connection arrives over loopback in every case, so that the
			// declared bind host is the only thing that can arm the check.
			got := New(cfg).Check(on("127.0.0.1", tc.host))
			if !errors.Is(got, tc.want) {
				t.Errorf("Check() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCheckOriginHeader(t *testing.T) {
	for _, tc := range []struct {
		name    string
		origins []string
		local   string
		host    string
		origin  string
		want    error
	}{
		{
			// No Origin is a non-browser client, which this check was never
			// able to constrain. adk-python skips it on the same grounds.
			name:  "absent origin",
			local: "127.0.0.1",
			host:  "localhost:8080",
		},
		{
			name:   "same origin",
			local:  "127.0.0.1",
			host:   "localhost:8080",
			origin: "http://localhost:8080",
		},
		{
			name:   "cross origin with nothing configured",
			local:  "127.0.0.1",
			host:   "localhost:8080",
			origin: "http://localhost:4200",
			want:   ErrOriginNotAllowed,
		},
		{
			name:    "cross origin that is configured",
			origins: []string{"http://localhost:4200"},
			local:   "127.0.0.1",
			host:    "localhost:8080",
			origin:  "http://localhost:4200",
		},
		{
			name:    "bare host:port configured is read as http",
			origins: []string{"localhost:4200"},
			local:   "127.0.0.1",
			host:    "localhost:8080",
			origin:  "http://localhost:4200",
		},
		{
			// Browsers lowercase the host in Origin, so a configured origin
			// spelled with capitals has to match one that is not.
			name:    "configured origin with a mixed-case host",
			origins: []string{"http://LocalHost:4200"},
			local:   "127.0.0.1",
			host:    "localhost:8080",
			origin:  "http://localhost:4200",
		},
		{
			name:   "mixed-case host in a same-origin request",
			local:  "127.0.0.1",
			host:   "LocalHost:8080",
			origin: "http://localhost:8080",
		},
		{
			name:    "configured origin over https does not match http",
			origins: []string{"https://localhost:4200"},
			local:   "127.0.0.1",
			host:    "localhost:8080",
			origin:  "http://localhost:4200",
			want:    ErrOriginNotAllowed,
		},
		{
			name:    "wildcard allows any origin",
			origins: []string{"*"},
			local:   "127.0.0.1",
			host:    "localhost:8080",
			origin:  "http://evil.com",
		},
		{
			// A remote page cannot forge Host, so a same-origin comparison
			// against it is sound once the connection is not loopback.
			name:   "cross origin on a routable connection",
			local:  "192.168.1.5",
			host:   "adk.example.com",
			origin: "http://evil.com",
			want:   ErrOriginNotAllowed,
		},
		{
			name:   "same origin on a routable connection",
			local:  "192.168.1.5",
			host:   "adk.example.com",
			origin: "http://adk.example.com",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := on(tc.local, tc.host)
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			p := New(Config{AllowedOrigins: tc.origins})

			if got := p.Check(r); !errors.Is(got, tc.want) {
				t.Errorf("Check() = %v, want %v", got, tc.want)
			}
			// CheckOrigin is the WebSocket upgrader's view of the same
			// question, so the two must never disagree about an Origin.
			if got, want := p.CheckOrigin(r), tc.want == nil; got != want {
				t.Errorf("CheckOrigin() = %v, want %v", got, want)
			}
		})
	}
}

// TestCheckOriginRebindingWithoutHostGuard pins the case the Host guard cannot
// reach: gorilla's default check passes a rebound page because it controls both
// Origin and Host, and CheckOrigin has to refuse it on its own.
func TestCheckOriginRebindingWithoutHostGuard(t *testing.T) {
	r := on("127.0.0.1", "evil.com:8080")
	r.Header.Set("Origin", "http://evil.com:8080")

	if New(Config{}).CheckOrigin(r) {
		t.Error("CheckOrigin() = true for a rebound origin, want false")
	}
	// Configuring the origin is the operator saying they meant it.
	if !New(Config{AllowedOrigins: []string{"http://evil.com:8080"}}).CheckOrigin(r) {
		t.Error("CheckOrigin() = false for a configured origin, want true")
	}
}

func TestCheckForwardedHeaders(t *testing.T) {
	for _, tc := range []struct {
		name    string
		headers map[string]string
		want    error
	}{
		{
			name: "same origin behind an X-Forwarded proxy",
			headers: map[string]string{
				"Origin":            "https://adk.internal",
				"X-Forwarded-Host":  "adk.internal",
				"X-Forwarded-Proto": "https",
			},
		},
		{
			name: "same origin behind an RFC 7239 proxy",
			headers: map[string]string{
				"Origin":    "https://adk.internal",
				"Forwarded": `for=192.0.2.1;host="adk.internal";proto=https`,
			},
		},
		{
			name: "forwarded headers naming a different origin",
			headers: map[string]string{
				"Origin":           "https://evil.com",
				"X-Forwarded-Host": "adk.internal",
			},
			want: ErrOriginNotAllowed,
		},
		{
			// Half a Forwarded header says nothing about the scheme, so the
			// X-Forwarded-* pair below it is what the origin is built from.
			name: "Forwarded with no proto falls back",
			headers: map[string]string{
				"Origin":            "https://adk.internal",
				"Forwarded":         "for=192.0.2.1",
				"X-Forwarded-Host":  "adk.internal",
				"X-Forwarded-Proto": "https",
			},
		},
		{
			// A WebSocket behind a proxy is forwarded as ws, and a browser
			// reports the page's scheme in Origin, never ws.
			name: "ws proto compares as http",
			headers: map[string]string{
				"Origin":            "http://adk.internal",
				"X-Forwarded-Host":  "adk.internal",
				"X-Forwarded-Proto": "ws",
			},
		},
		{
			name: "wss proto compares as https",
			headers: map[string]string{
				"Origin":            "https://adk.internal",
				"X-Forwarded-Host":  "adk.internal",
				"X-Forwarded-Proto": "wss",
			},
		},
		{
			// A proxy that appends rather than replaces leaves a list, and the
			// client-facing hop is the first element.
			name: "first element of a forwarded list wins",
			headers: map[string]string{
				"Origin":            "https://adk.internal",
				"X-Forwarded-Host":  "adk.internal, internal-lb",
				"X-Forwarded-Proto": "https, http",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// No origins configured, so the same-origin comparison is the only
			// thing that can accept these, and the forwarded headers are the
			// only record of what the browser addressed. Host is the internal
			// address the proxy dialled, which is not what the browser saw.
			r := on("192.168.1.5", "10.0.0.7:8080")
			for name, value := range tc.headers {
				r.Header.Set(name, value)
			}
			if got := New(Config{}).Check(r); !errors.Is(got, tc.want) {
				t.Errorf("Check() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestLoopbackBindBehindLocalProxy covers the deployment the Host guard would
// otherwise break: a server on loopback reached through a reverse proxy on the
// same machine, which puts its own hostname in Host. Listing the proxy's origin
// is what vouches for that hostname.
func TestLoopbackBindBehindLocalProxy(t *testing.T) {
	r := on("127.0.0.1", "adk.internal")
	r.Header.Set("Origin", "https://adk.internal")
	r.Header.Set("X-Forwarded-Host", "adk.internal")
	r.Header.Set("X-Forwarded-Proto", "https")

	if got := New(Config{BindHost: "127.0.0.1", AllowedOrigins: []string{"https://adk.internal"}}).Check(r); got != nil {
		t.Errorf("Check() = %v, want nil", got)
	}
	if got := New(loopbackBind).Check(r); !errors.Is(got, ErrHostNotAllowed) {
		t.Errorf("Check() without the proxy origin = %v, want %v", got, ErrHostNotAllowed)
	}
}

// TestForwardedHeadersCannotDefeatHostGuard pins that the rebinding check reads
// only the real Host header. A page may set X-Forwarded-Host and Forwarded on a
// same-origin fetch, so honouring them there would hand the attacker the guard.
func TestForwardedHeadersCannotDefeatHostGuard(t *testing.T) {
	for _, header := range []string{"X-Forwarded-Host", "Forwarded"} {
		t.Run(header, func(t *testing.T) {
			r := on("127.0.0.1", "evil.com:8080")
			if header == "Forwarded" {
				r.Header.Set(header, `host=localhost;proto=http`)
			} else {
				r.Header.Set(header, "localhost")
			}
			if got := New(loopbackBind).Check(r); !errors.Is(got, ErrHostNotAllowed) {
				t.Errorf("Check() = %v, want %v", got, ErrHostNotAllowed)
			}
		})
	}
}

// TestDeclaredBindOverridesTheConnection pins that one declared bind address
// settles both checks on its own, and that a wildcard one does not. An operator
// who says the server is exposed must not have either check re-imposed by
// whichever interface a connection happened to arrive on, and one who says it
// is on loopback must not lose them the other way. Naming every interface says
// neither, so it has to leave the server exactly as guarded as saying nothing:
// the two "wildcard bind" cases and the two "nothing declared" cases below
// answer alike.
func TestDeclaredBindOverridesTheConnection(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bindHost string
		local    string
		want     error
	}{
		{
			// Declared exposed, connection over loopback: someone on this
			// machine browsing a server that is open to the network anyway.
			name:     "routable bind, loopback connection",
			bindHost: "192.168.1.5",
			local:    "127.0.0.1",
		},
		{
			// A wildcard bind is every interface, loopback included, so the
			// connection is the only thing that says where this one came from.
			name:     "wildcard bind, loopback connection",
			bindHost: "0.0.0.0",
			local:    "127.0.0.1",
			want:     ErrOriginNotAllowed,
		},
		{
			name:     "wildcard bind written as a bare port, loopback connection",
			bindHost: ":8080",
			local:    "127.0.0.1",
			want:     ErrOriginNotAllowed,
		},
		{
			name:     "IPv6 wildcard bind, loopback connection",
			bindHost: "[::]",
			local:    "127.0.0.1",
			want:     ErrOriginNotAllowed,
		},
		{
			name:     "wildcard bind, routable connection",
			bindHost: "0.0.0.0",
			local:    "192.168.1.5",
		},
		{
			// Declared loopback, connection reported as routable. Only this
			// machine can reach the socket whatever the address says, so both
			// checks stay on and the Host one refuses this first.
			name:     "loopback bind, routable connection",
			bindHost: "127.0.0.1",
			local:    "192.168.1.5",
			want:     ErrHostNotAllowed,
		},
		{
			name:  "nothing declared, loopback connection",
			local: "127.0.0.1",
			want:  ErrOriginNotAllowed,
		},
		{
			name:  "nothing declared, routable connection",
			local: "192.168.1.5",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A page claiming a non-loopback origin, same-origin by the
			// browser's reckoning, which is what rebinding produces.
			r := on(tc.local, "adk.example.com")
			r.Header.Set("Origin", "http://adk.example.com")

			got := New(Config{BindHost: tc.bindHost}).Check(r)
			if !errors.Is(got, tc.want) {
				t.Errorf("Check() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestOverlongOriginIsRefusedUnparsed pins the bound on work done before the
// caller is authorized. net/http accepts a header up to MaxHeaderBytes, and
// parsing one of those on every request is the cost an unauthenticated caller
// could otherwise impose.
func TestOverlongOriginIsRefusedUnparsed(t *testing.T) {
	r := on("192.168.1.5", "adk.example.com")
	r.Header.Set("Origin", "http://adk.example.com/"+strings.Repeat("a", maxOriginLen))

	if got := New(Config{}).Check(r); !errors.Is(got, ErrOriginNotAllowed) {
		t.Errorf("Check() = %v, want %v", got, ErrOriginNotAllowed)
	}
	// The same origin within the bound is the one this server serves.
	r.Header.Set("Origin", "http://adk.example.com")
	if got := New(Config{}).Check(r); got != nil {
		t.Errorf("Check() = %v, want nil", got)
	}
}

// TestCheckOverTLS pins that the request's own scheme, not a hardcoded http,
// is what a same-origin comparison uses.
func TestCheckOverTLS(t *testing.T) {
	for _, tc := range []struct {
		name   string
		origin string
		want   error
	}{
		{name: "https origin over TLS", origin: "https://adk.internal"},
		{name: "http origin over TLS", origin: "http://adk.internal", want: ErrOriginNotAllowed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := on("192.168.1.5", "adk.internal")
			r.TLS = &tls.ConnectionState{}
			r.Header.Set("Origin", tc.origin)

			if got := New(Config{}).Check(r); !errors.Is(got, tc.want) {
				t.Errorf("Check() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestNewIgnoresEmptyOrigins pins that padding in the configured list does not
// count as configuration. An empty entry that did would read as an operator
// having named their origins, and switch off the loopback-origin rule that a
// server with none relies on.
func TestNewIgnoresEmptyOrigins(t *testing.T) {
	r := on("127.0.0.1", "evil.com:8080")
	r.Header.Set("Origin", "http://evil.com:8080")

	if New(Config{AllowedOrigins: []string{"", "  "}}).CheckOrigin(r) {
		t.Error("CheckOrigin() = true for a rebound origin, want false")
	}
}

func TestMiddleware(t *testing.T) {
	for _, tc := range []struct {
		name       string
		host       string
		wantStatus int
		wantBody   string
	}{
		{
			name:       "allowed request reaches the handler",
			host:       "localhost:8080",
			wantStatus: http.StatusOK,
			wantBody:   "served",
		},
		{
			name:       "rebound request is refused",
			host:       "evil.com:8080",
			wantStatus: http.StatusForbidden,
			wantBody:   "Forbidden: host not allowed\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler := New(loopbackBind).Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if _, err := w.Write([]byte("served")); err != nil {
					t.Errorf("Write() error = %v", err)
				}
			}))

			rr := httptest.NewRecorder()
			handler.ServeHTTP(rr, on("127.0.0.1", tc.host))

			if got := rr.Code; got != tc.wantStatus {
				t.Errorf("status = %d, want %d", got, tc.wantStatus)
			}
			if got := rr.Body.String(); got != tc.wantBody {
				t.Errorf("body = %q, want %q", got, tc.wantBody)
			}
		})
	}
}

func TestNormalizeOrigin(t *testing.T) {
	for _, tc := range []struct {
		addr string
		want string
	}{
		{addr: "", want: ""},
		{addr: "*", want: "*"},
		{addr: "  localhost:8080  ", want: "http://localhost:8080"},
		{addr: "localhost", want: "http://localhost"},
		{addr: "http://localhost:8080/", want: "http://localhost:8080"},
		{addr: "https://ui.example.com/app?x=1", want: "https://ui.example.com"},
		{addr: "://nonsense", want: "://nonsense"},
	} {
		t.Run(tc.addr, func(t *testing.T) {
			if got := NormalizeOrigin(tc.addr); got != tc.want {
				t.Errorf("NormalizeOrigin(%q) = %q, want %q", tc.addr, got, tc.want)
			}
		})
	}
}

func TestStripPort(t *testing.T) {
	for _, tc := range []struct {
		host string
		want string
	}{
		{host: "localhost:8080", want: "localhost"},
		{host: "localhost", want: "localhost"},
		{host: "[::1]:8080", want: "::1"},
		{host: "[::1]", want: "::1"},
		{host: "::1", want: "::1"},
		// A malformed authority comes back whole, so no caller can read the
		// leading label as the host.
		{host: "127.0.0.1:8080.evil.com", want: "127.0.0.1:8080.evil.com"},
		{host: "[::1", want: "[::1"},
		{host: "localhost:", want: "localhost:"},
		{host: "localhost:http", want: "localhost:http"},
	} {
		t.Run(tc.host, func(t *testing.T) {
			if got := stripPort(tc.host); got != tc.want {
				t.Errorf("stripPort(%q) = %q, want %q", tc.host, got, tc.want)
			}
		})
	}
}
