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
	"net/http"
	"net/url"
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
		// form, which maps characters first. Two names are the same only
		// when the written form agrees, so a spelling that only dials the
		// same address is refused, and a Unicode name equals its punycode.
		{name: "sharp s and capital sharp s", from: "https://fa\u00df.example/rpc", to: "https://fa\u1e9e.example/rpc", want: false},
		{name: "ss and capital sharp s", from: "https://fass.example/rpc", to: "https://fa\u1e9e.example/rpc", want: false},
		{name: "final sigma and sigma", from: "https://\u03b5\u03bb\u03bb\u03ac\u03c2.example/rpc", to: "https://\u03b5\u03bb\u03bb\u03ac\u03c3.example/rpc", want: false},
		{name: "soft hyphen inside the name", from: "https://api.example/rpc", to: "https://ap\u00adi.example/rpc", want: false},
		{name: "fullwidth letter", from: "https://fass.example/rpc", to: "https://\uff46ass.example/rpc", want: false},
		{name: "long s", from: "https://fass.example/rpc", to: "https://fa\u017fs.example/rpc", want: false},
		{name: "kelvin sign", from: "https://fak.example/rpc", to: "https://fa\u212a.example/rpc", want: false},
		{name: "unicode host to its punycode form", from: "https://fa\u00df.example/rpc", to: "https://xn--fa-hia.example/rpc", want: true},
		{name: "punycode host to its unicode form", from: "https://xn--fa-hia.example/rpc", to: "https://fa\u00df.example/rpc", want: true},
		// Same Host header, but net/http dials fass.example for the first
		// and, with the lookup profile rejecting the decoded label, the
		// written name for the second.
		{name: "capital sharp s to its punycode form", from: "https://fa\u1e9e.example/rpc", to: "https://xn--fa-64s.example/rpc", want: false},
		{name: "ascii case differs in a unicode host", from: "https://M\u00fcnchen.example/rpc", to: "https://m\u00fcnchen.example/rpc", want: true},
		// Both dial one address, but the Host headers differ.
		{name: "unicode case differs", from: "https://M\u00dcNCHEN.example/rpc", to: "https://m\u00fcnchen.example/rpc", want: false},
		// An ASCII name is taken as written, so the empty punycode label the
		// IDNA profiles would drop still tells the two apart.
		{name: "trailing dot and an empty punycode label", from: "https://mcp.example.com./rpc", to: "https://mcp.example.com.xn--/rpc", want: false},
		// Names the lookup profile rejects, an underscore next to a non-ASCII
		// letter, still have a punycode form, and an ASCII name needs none.
		{name: "underscore and sharp s, same name", from: "https://fa\u00df_x.example/rpc", to: "https://fa\u00df_x.example/rpc", want: true},
		{name: "underscore and sharp s against capital sharp s", from: "https://fa\u00df_x.example/rpc", to: "https://fa\u1e9e_x.example/rpc", want: false},
		{name: "ascii underscore names differ", from: "https://mcp_server/rpc", to: "https://evil_host/rpc", want: false},
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
		{name: "same origin across a chain", via: []string{"http://h/a", "https://h/b"}, to: "https://h:443/c"},
		{name: "host header names the same host", via: []string{"http://h/a"}, to: "http://h/b", host: "h"},
		{name: "host header names the same host and default port", via: []string{"http://h/a"}, to: "http://h/b", host: "h:80"},
		{name: "host header names another host", via: []string{"http://h/a"}, to: "http://h/b", host: "evil.test", wantErr: true},
		{name: "host header names another port", via: []string{"http://h:8080/a"}, to: "http://h:8080/b", host: "h:9090", wantErr: true},
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

// TestHostNames pins the two names net/http derives from a host, each on its
// own, since sameTarget only shows whether they agree.
func TestHostNames(t *testing.T) {
	tests := []struct {
		host     string
		wantDial string
		wantHost string
	}{
		{host: "api.example", wantDial: "api.example", wantHost: "api.example"},
		// ASCII is taken as written: the IDNA profiles would drop the label.
		{host: "mcp.example.com.xn--", wantDial: "mcp.example.com.xn--", wantHost: "mcp.example.com.xn--"},
		{host: "fa\u00df.example", wantDial: "xn--fa-hia.example", wantHost: "xn--fa-hia.example"},
		// Mapped before encoding for the dial, encoded as written for Host.
		{host: "fa\u1e9e.example", wantDial: "fass.example", wantHost: "xn--fa-64s.example"},
		{host: "ap\u00adi.example", wantDial: "api.example", wantHost: "xn--api-hfa.example"},
		// The lookup profile rejects the underscore, so the dial falls back
		// to the written name, as net/http's does.
		{host: "fa\u00df_x.example", wantDial: "fa\u00df_x.example", wantHost: "xn--fa_x-wna.example"},
	}
	for _, tc := range tests {
		t.Run(tc.host, func(t *testing.T) {
			u := &url.URL{Host: tc.host}
			if got := dialName(u); got != tc.wantDial {
				t.Errorf("dialName(%q) = %q, want %q", tc.host, got, tc.wantDial)
			}
			got, ok := hostName(u)
			if !ok || got != tc.wantHost {
				t.Errorf("hostName(%q) = %q, %v, want %q, true", tc.host, got, ok, tc.wantHost)
			}
		})
	}
}
