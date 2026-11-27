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

package credredirect

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync"
	"testing"
)

// TestSameTarget covers which redirects may carry the credential, across the
// scheme and port combinations which the end-to-end tests in mcptoolset and
// remoteagent/v2 do not reach: an upgrade to https, default ports written out
// or left implicit, and schemes that are neither http nor https.
func TestSameTarget(t *testing.T) {
	tests := []struct {
		name     string
		from, to string
		want     bool
	}{
		{name: "identical", from: "https://h/rpc", to: "https://h/other", want: true},
		{name: "default port made explicit", from: "https://h/rpc", to: "https://h:443/rpc", want: true},
		{name: "default http port made explicit", from: "http://h/rpc", to: "http://h:80/rpc", want: true},
		{name: "host case differs", from: "https://Host/rpc", to: "https://host/rpc", want: true},
		{name: "scheme upgraded to https, same explicit port", from: "http://h:8080/rpc", to: "https://h:8080/rpc", want: true},
		{name: "scheme upgraded to https, both ports implicit", from: "http://h/rpc", to: "https://h/rpc", want: true},
		{name: "scheme upgraded to https, explicit default ports", from: "http://h:80/rpc", to: "https://h:443/rpc", want: true},
		{name: "scheme upgraded to https onto a different port", from: "http://h/rpc", to: "https://h:9443/rpc", want: false},
		{name: "scheme downgraded with implicit ports", from: "https://h/rpc", to: "http://h/rpc", want: false},
		{name: "unrelated scheme", from: "https://h/rpc", to: "ftp://h/rpc", want: false},
		// Only http to https counts as an upgrade: either half alone is not one.
		{name: "http to an unrelated scheme", from: "http://h/rpc", to: "ftp://h/rpc", want: false},
		{name: "an unrelated scheme to https", from: "ftp://h/rpc", to: "https://h/rpc", want: false},
		// An explicit default port on one side and an implicit one on the other
		// is still each scheme's own default.
		{name: "scheme upgraded to https, explicit http default to implicit", from: "http://h:80/rpc", to: "https://h/rpc", want: true},
		{name: "scheme downgraded to http", from: "https://h/rpc", to: "http://h/rpc", want: false},
		// The port is the same on both sides, only one side writes it out.
		{name: "scheme upgraded to https, implicit http default to explicit", from: "http://h/rpc", to: "https://h:80/rpc", want: true},
		{name: "scheme upgraded to https, explicit https default to implicit", from: "http://h:443/rpc", to: "https://h/rpc", want: true},
		{name: "different host", from: "https://h/rpc", to: "https://other/rpc", want: false},
		{name: "different port", from: "https://h:8443/rpc", to: "https://h:9443/rpc", want: false},
		// net/http writes the Host header and a proxy's request target from the
		// punycode form of the name as written, and dials the IDNA lookup
		// form, which maps characters first. Each of the next seven pairs
		// sends a different Host header and is refused for that, whichever
		// name the mapping tables of the Go release in use make it dial.
		{name: "sharp s and capital sharp s", from: "https://fa\u00df.example/rpc", to: "https://fa\u1e9e.example/rpc", want: false},
		{name: "ss and capital sharp s", from: "https://fass.example/rpc", to: "https://fa\u1e9e.example/rpc", want: false},
		{name: "final sigma and sigma", from: "https://\u03b5\u03bb\u03bb\u03ac\u03c2.example/rpc", to: "https://\u03b5\u03bb\u03bb\u03ac\u03c3.example/rpc", want: false},
		{name: "soft hyphen inside the name", from: "https://api.example/rpc", to: "https://ap\u00adi.example/rpc", want: false},
		{name: "fullwidth letter", from: "https://fass.example/rpc", to: "https://\uff46ass.example/rpc", want: false},
		{name: "long s", from: "https://fass.example/rpc", to: "https://fa\u017fs.example/rpc", want: false},
		{name: "kelvin sign", from: "https://fak.example/rpc", to: "https://fa\u212a.example/rpc", want: false},
		// A Unicode name already in the form the mapping produces is the
		// same target as its punycode form.
		{name: "unicode host to its punycode form", from: "https://fa\u00df.example/rpc", to: "https://xn--fa-hia.example/rpc", want: true},
		{name: "punycode host to its unicode form", from: "https://xn--fa-hia.example/rpc", to: "https://fa\u00df.example/rpc", want: true},
		// Same Host header, but the first is dialed as the name its mapping
		// produces and the second as written.
		{name: "capital sharp s to its punycode form", from: "https://fa\u1e9e.example/rpc", to: "https://xn--fa-64s.example/rpc", want: false},
		{name: "ascii case differs in a unicode host", from: "https://M\u00fcnchen.example/rpc", to: "https://m\u00fcnchen.example/rpc", want: true},
		// The Host headers differ: punycode keeps the case as written.
		{name: "unicode case differs", from: "https://M\u00dcNCHEN.example/rpc", to: "https://m\u00fcnchen.example/rpc", want: false},
		// An ASCII name is taken as written, so the empty punycode label the
		// IDNA profiles would drop still tells the two apart.
		{name: "trailing dot and an empty punycode label", from: "https://mcp.example.com./rpc", to: "https://mcp.example.com.xn--/rpc", want: false},
		// Names the lookup profile rejects, an underscore next to a non-ASCII
		// letter, still have a punycode form, and an ASCII name needs none.
		{name: "underscore and sharp s, same name", from: "https://fa\u00df_x.example/rpc", to: "https://fa\u00df_x.example/rpc", want: true},
		{name: "underscore and sharp s against capital sharp s", from: "https://fa\u00df_x.example/rpc", to: "https://fa\u1e9e_x.example/rpc", want: false},
		{name: "ascii underscore names differ", from: "https://mcp_server/rpc", to: "https://evil_host/rpc", want: false},
		// A punycode label net/http cannot encode has no Host header, so
		// the request could not be sent, and the hop is refused.
		{name: "invalid punycode label beside a unicode one", from: "https://xn--zzz-.fa\u00df.example/rpc", to: "https://xn--zzz-.fa\u00df.example/rpc", want: false},
		// A Google API host and its mTLS name are the same service: a
		// client presenting a certificate is sent to the mTLS name.
		{name: "google api host to its mtls name", from: "https://agent.googleapis.com/rpc", to: "https://agent.mtls.googleapis.com/rpc", want: true},
		{name: "mtls name to its google api host", from: "https://agent.mtls.googleapis.com/rpc", to: "https://agent.googleapis.com:443/rpc", want: true},
		{name: "mtls twin, mixed case", from: "https://Agent.GoogleAPIs.com/rpc", to: "https://agent.MTLS.googleapis.com/rpc", want: true},
		{name: "mtls name of another service", from: "https://agent.googleapis.com/rpc", to: "https://other.mtls.googleapis.com/rpc", want: false},
		{name: "mtls twin on another port", from: "https://agent.googleapis.com/rpc", to: "https://agent.mtls.googleapis.com:8443/rpc", want: false},
		{name: "mtls twin over http", from: "http://agent.googleapis.com/rpc", to: "http://agent.mtls.googleapis.com/rpc", want: false},
		{name: "mtls twin upgraded to https", from: "http://agent.googleapis.com/rpc", to: "https://agent.mtls.googleapis.com/rpc", want: false},
		{name: "mtls suffix outside googleapis", from: "https://agent.example.com/rpc", to: "https://agent.mtls.example.com/rpc", want: false},
		{name: "mtls twin upgraded to https, same explicit port", from: "http://agent.googleapis.com:443/rpc", to: "https://agent.mtls.googleapis.com:443/rpc", want: false},
		// strings.ToLower folds the Kelvin sign into k, which would make
		// these twins although they send different Host headers.
		{name: "kelvin sign in the mtls twin", from: "https://agentk.googleapis.com/rpc", to: "https://agent\u212a.mtls.googleapis.com/rpc", want: false},
		{name: "kelvin sign in the plain twin", from: "https://agent\u212a.googleapis.com/rpc", to: "https://agentk.mtls.googleapis.com/rpc", want: false},
		{name: "empty label and its mtls name", from: "https://.googleapis.com/rpc", to: "https://.mtls.googleapis.com/rpc", want: false},
		{name: "bare mtls host to bare googleapis", from: "https://googleapis.com/rpc", to: "https://mtls.googleapis.com/rpc", want: false},
		// Names the client certificate transport never moves a request to.
		{name: "multi-region host and an mtls name it does not have", from: "https://agent.us.rep.googleapis.com/rpc", to: "https://agent.us.rep.mtls.googleapis.com/rpc", want: false},
		{name: "mtls host and a doubled mtls name", from: "https://mtls.googleapis.com/rpc", to: "https://mtls.mtls.googleapis.com/rpc", want: false},
		{name: "ipv6 literal, same", from: "https://[::1]:8443/rpc", to: "https://[::1]:8443/rpc", want: true},
		{name: "ipv6 literals differ", from: "https://[::1]:8443/rpc", to: "https://[2001:db8::1]:8443/rpc", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			from, err := url.Parse(tc.from)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.from, err)
			}
			to, err := url.Parse(tc.to)
			if err != nil {
				t.Fatalf("url.Parse(%q) error = %v", tc.to, err)
			}
			if got := sameTarget(from, to); got != tc.want {
				t.Errorf("sameTarget(%q, %q) = %v, want %v", tc.from, tc.to, got, tc.want)
			}
		})
	}
}

// TestCheck covers Check's two comparisons across a chain, and the request
// fields other than the URL that decide where the request goes on the wire.
func TestCheck(t *testing.T) {
	tests := []struct {
		name    string
		via     []string
		to      string
		host    string // req.Host, the Host header when set
		opaque  string
		path    string // replaces the parsed path when set
		wantErr bool
	}{
		{name: "same origin", via: []string{"https://h/a"}, to: "https://h/b"},
		{name: "cross host", via: []string{"https://h/a"}, to: "https://other/b", wantErr: true},
		{name: "upgrade then downgrade", via: []string{"http://h/a", "https://h/b"}, to: "http://h/c", wantErr: true},
		{name: "google api host to its mtls name", via: []string{"https://agent.googleapis.com/a"}, to: "https://agent.mtls.googleapis.com/b"},
		// Only the comparison with the original request refuses this one:
		// the two names count as one host between https URLs alone.
		{name: "upgrade then move to the mtls name", via: []string{"http://agent.googleapis.com/a", "https://agent.googleapis.com/b"}, to: "https://agent.mtls.googleapis.com/c", wantErr: true},
		{name: "same origin across a chain", via: []string{"http://h/a", "https://h/b"}, to: "https://h:443/c"},
		{name: "host header names the same host", via: []string{"http://h/a"}, to: "http://h/b", host: "h"},
		{name: "host header names another host", via: []string{"http://h/a"}, to: "http://h/b", host: "evil.test", wantErr: true},
		{name: "host header names another port", via: []string{"http://h:8080/a"}, to: "http://h:8080/b", host: "h:9090", wantErr: true},
		// net/http sends the header string as given, so a respelling of the
		// same name is a different Host on the wire.
		{name: "host header brackets the name", via: []string{"http://h/a"}, to: "http://h/b", host: "[h]", wantErr: true},
		{name: "host header writes the default port out", via: []string{"http://h/a"}, to: "http://h/b", host: "h:80", wantErr: true},
		{name: "host header folds the port into an ipv6 literal", via: []string{"http://[2001:db8::1]:8080/a"}, to: "http://[2001:db8::1]:8080/b", host: "2001:db8::1:8080", wantErr: true},
		{name: "host header matches an ipv6 literal with its port", via: []string{"http://[2001:db8::1]:8080/a"}, to: "http://[2001:db8::1]:8080/b", host: "[2001:db8::1]:8080"},
		{name: "host header is the unicode name net/http would encode", via: []string{"http://fa\u00df.example:8080/a"}, to: "http://fa\u00df.example:8080/b", host: "fa\u00df.example:8080"},
		{name: "host header is the punycode of the unicode url", via: []string{"http://fa\u00df.example/a"}, to: "http://fa\u00df.example/b", host: "xn--fa-hia.example"},
		{name: "host header is another spelling of the unicode url", via: []string{"http://fa\u00df.example/a"}, to: "http://fa\u00df.example/b", host: "fa\u1e9e.example", wantErr: true},
		{name: "host header cannot be encoded", via: []string{"http://h/a"}, to: "http://h/b", host: "xn--zzz-.fa\u00df.example", wantErr: true},
		{name: "opaque url", via: []string{"http://h/a"}, to: "http://h/b", opaque: "//evil.test/b", wantErr: true},
		{name: "path without a leading slash", via: []string{"http://h/a"}, to: "http://h/b", path: ".evil.test/b", wantErr: true},
		{name: "path turning the host into userinfo", via: []string{"http://h:8080/a"}, to: "http://h:8080/b", path: "@evil.test/b", wantErr: true},
		{name: "empty path", via: []string{"http://h/a"}, to: "http://h", path: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var via []*http.Request
			for _, raw := range tc.via {
				u, err := url.Parse(raw)
				if err != nil {
					t.Fatal(err)
				}
				via = append(via, &http.Request{URL: u})
			}
			u, err := url.Parse(tc.to)
			if err != nil {
				t.Fatal(err)
			}
			if tc.path != "" {
				u.Path = tc.path
			}
			u.Opaque = tc.opaque
			err = Check(&http.Request{URL: u, Host: tc.host}, via)
			if (err != nil) != tc.wantErr {
				t.Errorf("Check() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

// TestHostName pins the Host header name. The dial name has no literals here:
// it follows the IDNA mapping tables, which change between Go releases, so
// TestHostNamesMatchNetHTTP checks it against net/http instead.
func TestHostName(t *testing.T) {
	tests := []struct {
		host string
		want string
	}{
		{host: "api.example", want: "api.example"},
		// ASCII is taken as written: the IDNA profiles would drop the label.
		{host: "mcp.example.com.xn--", want: "mcp.example.com.xn--"},
		{host: "fa\u00df.example", want: "xn--fa-hia.example"},
		// Encoded as written, with no mapping first.
		{host: "fa\u1e9e.example", want: "xn--fa-64s.example"},
		{host: "ap\u00adi.example", want: "xn--api-hfa.example"},
		{host: "fa\u00df_x.example", want: "xn--fa_x-wna.example"},
	}
	for _, tc := range tests {
		t.Run(tc.host, func(t *testing.T) {
			got, ok := hostName(&url.URL{Host: tc.host})
			if !ok || got != tc.want {
				t.Errorf("hostName(%q) = %q, %v, want %q, true", tc.host, got, ok, tc.want)
			}
		})
	}
}

// TestHostNamesMatchNetHTTP checks dialName and hostName against net/http
// itself rather than against literals: a client whose dialer records the
// address it is given and a server that records the Host header it receives.
// dialName reads the IDNA tables of golang.org/x/net in go.mod and net/http
// reads the copy it vendors, so for the names listed a difference between the
// two fails here.
func TestHostNamesMatchNetHTTP(t *testing.T) {
	var mu sync.Mutex
	var dialed, hostHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hostHeader = r.Host
		mu.Unlock()
	}))
	defer srv.Close()
	port := srv.Listener.Addr().(*net.TCPAddr).Port
	client := &http.Client{Transport: &http.Transport{
		// Every request dials, so each one records its own address.
		DisableKeepAlives: true,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			mu.Lock()
			dialed = addr
			mu.Unlock()
			return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
		},
	}}
	for _, host := range []string{
		"api.example",
		"mcp.example.com.xn--",
		"fa\u00df.example",
		"xn--fa-hia.example",
		"fa\u1e9e.example",
		"ap\u00adi.example",
		"M\u00dcNCHEN.example",
		"fa\u00df_x.example",
	} {
		t.Run(host, func(t *testing.T) {
			u := &url.URL{Scheme: "http", Host: net.JoinHostPort(host, strconv.Itoa(port)), Path: "/"}
			resp, err := client.Do(&http.Request{Method: http.MethodGet, URL: u, Header: http.Header{}})
			if err != nil {
				t.Fatalf("Do(%s) error = %v", u, err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Error(err)
			}
			mu.Lock()
			defer mu.Unlock()
			if want := net.JoinHostPort(dialName(u), strconv.Itoa(port)); dialed != want {
				t.Errorf("net/http dialed %q, dialName gives %q", dialed, want)
			}
			name, ok := hostName(u)
			if want := net.JoinHostPort(name, strconv.Itoa(port)); !ok || hostHeader != want {
				t.Errorf("net/http sent Host %q, hostName gives %q, %v", hostHeader, want, ok)
			}
		})
	}
}
