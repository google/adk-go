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

// Package credredirect implements the redirect policy for an HTTP client whose
// transport applies a credential on every request. Such a transport runs below
// the layer where net/http strips Authorization on a cross-host redirect, so the
// client's CheckRedirect is the only thing bounding where the credential
// travels.
package credredirect

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/idna"

	"google.golang.org/adk/v2/internal/googlemtls"
)

// MaxRedirects is net/http's default cap on requests per redirect chain, which
// a non-nil CheckRedirect replaces rather than extends. Check does not apply
// it, because one caller applies it conditionally.
const MaxRedirects = 10

// Check refuses a redirect that leaves the original request's host or
// downgrades its scheme, and one whose request would address another host on
// the wire than its URL names. A Google API host and its mTLS name count as
// one host between https URLs (see mtlsTwins). It does not cap the number of
// hops.
func Check(req *http.Request, via []*http.Request) error {
	// Check against the original request, then against the hop just taken.
	// The previous hop is what refuses a chain climbing to https and then
	// dropping back to http. The original is what refuses one climbing to
	// https and then moving to the host's mTLS name, which counts as the
	// same host only between two https URLs. The order only decides which
	// origin a refusal names.
	for _, from := range []*url.URL{via[0].URL, via[len(via)-1].URL} {
		if !sameTarget(from, req.URL) {
			return fmt.Errorf("refusing redirect from %s to %s", origin(from), origin(req.URL))
		}
	}
	if !addressesOwnHost(req) {
		return fmt.Errorf("refusing redirect to %s: the request's Host header, opaque URL or path addresses another host", origin(req.URL))
	}
	return nil
}

// addressesOwnHost reports whether req reaches the host its URL names on the
// wire. net/http sends Host from req.Host when it is set, and a forward proxy
// takes the whole target from the request line, which an opaque URL or a path
// without a leading slash can point at another host. A request net/http built
// to follow a redirect passes; one a caller's CheckRedirect edited may not.
func addressesOwnHost(req *http.Request) bool {
	if req.URL.Opaque != "" {
		return false
	}
	if path := req.URL.EscapedPath(); path != "" && !strings.HasPrefix(path, "/") {
		return false
	}
	if req.Host == "" {
		return true
	}
	// net/http writes the Host header string as given, so it has to be the
	// one it would write from the URL, byte for byte after punycode. Parsing
	// it as a host would let a respelling through, brackets around a name
	// or a port folded into an IPv6 literal, that reaches a front end or a
	// proxy as another name.
	want, ok := hostHeader(req.URL.Host)
	if !ok {
		return false
	}
	got, ok := hostHeader(req.Host)
	return ok && got == want
}

// hostHeader returns the Host header net/http writes for a host[:port]
// string, as httpguts.PunycodeHostPort derives it: ASCII as given, otherwise
// the name's punycode with the port kept, and false when it has none.
func hostHeader(hostport string) (string, bool) {
	if isASCII(hostport) {
		return hostport, true
	}
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = hostport, ""
	}
	ascii, err := idna.ToASCII(host)
	if err != nil {
		return "", false
	}
	if port == "" {
		return ascii, true
	}
	return net.JoinHostPort(ascii, port), true
}

// sameTarget reports whether it is safe to replay the credential on a
// redirect from one URL to the other: every name net/http derives from the
// host must be identical, and the scheme may only stay the same or be upgraded
// from http to https, which strictly improves confidentiality. The one pair of
// different names it accepts is the one mtlsTwins describes.
func sameTarget(from, to *url.URL) bool {
	if mtlsTwins(from, to) {
		return true
	}
	if !strings.EqualFold(dialName(from), dialName(to)) {
		return false
	}
	fromHost, ok := hostName(from)
	if !ok {
		return false
	}
	toHost, ok := hostName(to)
	if !ok || !strings.EqualFold(fromHost, toHost) {
		return false
	}
	switch {
	case strings.EqualFold(from.Scheme, to.Scheme):
		return defaultedPort(from) == defaultedPort(to)
	case strings.EqualFold(from.Scheme, "http") && strings.EqualFold(to.Scheme, "https"):
		// The default port differs between the two schemes, so an upgrade
		// keeps the same endpoint when both sides use their own scheme's
		// default, or when both name the same port, written out or implied.
		return isDefaultPort(from) && isDefaultPort(to) || defaultedPort(from) == defaultedPort(to)
	default:
		return false
	}
}

// mtlsTwins reports whether from and to are a Google API host and its
// .mtls.googleapis.com name, both https on the same port. A client that
// presents a client certificate is sent to the mTLS name, so a server may
// redirect between the two, and both serve the same API. It also reports true
// for one ASCII name on both sides, which the rest of sameTarget accepts as
// well.
func mtlsTwins(from, to *url.URL) bool {
	if !strings.EqualFold(from.Scheme, "https") || !strings.EqualFold(to.Scheme, "https") ||
		defaultedPort(from) != defaultedPort(to) {
		return false
	}
	return googlemtls.Twins(from.Hostname(), to.Hostname())
}

// dialName returns the address net/http dials for u, falling back to the
// written name when the lookup mapping fails, as net/http does.
//
// net/http derives two names from a URL's host, and a redirect is the same
// target only when both agree. It dials the IDNA lookup form, which maps
// characters before encoding, and it writes the Host header and a forward
// proxy's request target from the punycode form of the name as written. So
// two spellings can share a dial address and still carry different Host
// headers, as a name with a soft hyphen or a fullwidth letter does against
// the plain one. Which spellings map together follows the IDNA tables of the
// Go release in use. A name the lookup profile rejects is dialed as written
// while its Host header is still encoded. An ASCII name is taken as written
// by both, so an empty punycode label is not dropped, and a Unicode name
// that is already in its mapped form compares equal to its punycode form.
func dialName(u *url.URL) string {
	host := u.Hostname()
	if isASCII(host) {
		return host
	}
	if ascii, err := idna.Lookup.ToASCII(host); err == nil {
		return ascii
	}
	return host
}

// hostName returns the name net/http writes into the Host header for u, as
// httpguts.PunycodeHostPort derives it, and false when it has none.
func hostName(u *url.URL) (string, bool) {
	host := u.Hostname()
	if isASCII(host) {
		return host, true
	}
	ascii, err := idna.ToASCII(host)
	return ascii, err == nil
}

func isASCII(s string) bool {
	for _, c := range []byte(s) {
		if c >= 0x80 {
			return false
		}
	}
	return true
}

// defaultedPort returns the URL's port with the scheme's default filled in, so
// that https://example.com and https://example.com:443 compare equal.
func defaultedPort(u *url.URL) string {
	if port := u.Port(); port != "" {
		return port
	}
	switch strings.ToLower(u.Scheme) {
	case "https":
		return "443"
	case "http":
		return "80"
	}
	return ""
}

// isDefaultPort reports whether the URL leaves its scheme's port implicit or
// names the default explicitly.
func isDefaultPort(u *url.URL) bool {
	port := u.Port()
	if port == "" {
		return true
	}
	return port == defaultedPort(&url.URL{Scheme: u.Scheme})
}

func origin(u *url.URL) string { return u.Scheme + "://" + u.Host }
