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
	"net/http"
	"net/url"
	"strings"
)

// MaxRedirects is net/http's default cap on requests per redirect chain, which
// a non-nil CheckRedirect replaces rather than extends. Check does not apply
// it, because one caller applies it conditionally.
const MaxRedirects = 10

// Check refuses a redirect that leaves the original request's host or
// downgrades its scheme. It does not cap the number of hops.
func Check(req *http.Request, via []*http.Request) error {
	// Check against the original request and against the hop just taken. The
	// original stops a chain of individually-safe hops laundering the
	// credential away from the original host; the previous hop stops a chain
	// climbing to https and then dropping back to http, which the original
	// alone would accept.
	for _, from := range []*url.URL{via[0].URL, via[len(via)-1].URL} {
		if !sameTarget(from, req.URL) {
			return fmt.Errorf("refusing redirect from %s to %s", origin(from), origin(req.URL))
		}
	}
	return nil
}

// sameTarget reports whether it is safe to replay the credential on a
// redirect from one URL to the other: the host must be identical, and the
// scheme may only stay the same or be upgraded from http to https, which
// strictly improves confidentiality.
func sameTarget(from, to *url.URL) bool {
	if !strings.EqualFold(from.Hostname(), to.Hostname()) {
		return false
	}
	switch {
	case strings.EqualFold(from.Scheme, to.Scheme):
		return defaultedPort(from) == defaultedPort(to)
	case strings.EqualFold(from.Scheme, "http") && strings.EqualFold(to.Scheme, "https"):
		// The default port differs between the two schemes, so an upgrade
		// cannot be judged by comparing defaulted ports. It keeps the same
		// endpoint when both sides use their own scheme's default, or when both
		// name the same explicit port.
		return isDefaultPort(from) && isDefaultPort(to) || from.Port() == to.Port()
	default:
		return false
	}
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
