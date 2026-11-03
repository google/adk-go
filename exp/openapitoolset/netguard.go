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
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"

	"google.golang.org/adk/v2/internal/version"
)

// Timeouts for the default HTTP client. The connect timeout is short so an
// unreachable host fails fast; the overall timeout is generous so a slow but
// legitimate API still completes, matching adk-python's read/write budget.
const (
	connectTimeout = 10 * time.Second
	requestTimeout = 600 * time.Second
)

// errBlockedHost is returned by the dial guard when a request targets a
// non-global address. It carries the host so the error names what was blocked.
type errBlockedHost struct {
	host string
}

func (e *errBlockedHost) Error() string { return "Blocked host: " + e.host }

// userAgent is the User-Agent a tool sends, naming ADK, its version and the
// tool, as adk-python's RestApiTool does.
func userAgent(toolName string) string {
	return fmt.Sprintf("%s (tool: %s)", version.TrackingHeader(), toolName)
}

// defaultHTTPClient returns the client used when the caller supplies none. When
// blockPrivate is set, its dialer refuses any connection to a non-global IP, so
// the check runs against the address actually dialed and also covers a hostname
// that resolves to a private address (DNS rebinding included).
func defaultHTTPClient(blockPrivate bool) *http.Client {
	if !blockPrivate {
		return &http.Client{Timeout: requestTimeout}
	}
	dialer := &net.Dialer{Timeout: connectTimeout}
	dialer.Control = func(_, address string, _ syscall.RawConn) error {
		host, _, err := net.SplitHostPort(address)
		if err != nil {
			host = address
		}
		if ip := net.ParseIP(host); ip != nil && !isGlobalUnicast(ip) {
			return &errBlockedHost{host: host}
		}
		return nil
	}
	return &http.Client{
		Timeout:   requestTimeout,
		Transport: &http.Transport{DialContext: dialer.DialContext},
	}
}

// isGlobalUnicast reports whether ip is a globally routable unicast address,
// excluding loopback, private, link-local (including the cloud metadata range)
// and other non-global ranges.
func isGlobalUnicast(ip net.IP) bool {
	if !ip.IsGlobalUnicast() {
		return false
	}
	if ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast() {
		return false
	}
	// Carrier-grade NAT (100.64.0.0/10): routable but not a public address.
	if ip4 := ip.To4(); ip4 != nil && ip4[0] == 100 && ip4[1]&0xc0 == 0x40 {
		return false
	}
	return true
}

// allowedSchemes are the URL schemes a tool may call.
var allowedSchemes = map[string]bool{"http": true, "https": true}

// blockedHostnames always designate the local machine or a metadata endpoint.
var blockedHostnames = map[string]bool{
	"localhost":     true,
	"metadata":      true,
	"metadata.goog": true,
}

// blockedHostnameSuffixes are reserved for loopback, link-local and internal
// networks.
var blockedHostnameSuffixes = []string{".localhost", ".local", ".internal", ".metadata.goog"}

// checkURL applies the lexical SSRF checks that do not need DNS: the scheme
// must be http or https, a hostname must be present, and the hostname must not
// name the local machine or internal infrastructure. It is the part of
// adk-python's _validate_url that holds whoever resolves DNS, so it runs even
// with a caller-supplied client that the dial guard cannot reach. It returns an
// *inputValidationError so the message reaches the model.
func checkURL(rawURL string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return &inputValidationError{fmt.Sprintf("Invalid url: %s", rawURL)}
	}
	if !allowedSchemes[strings.ToLower(parsed.Scheme)] {
		return &inputValidationError{fmt.Sprintf("Unsupported url scheme: %s", rawURL)}
	}
	host := parsed.Hostname()
	if host == "" {
		return &inputValidationError{fmt.Sprintf("URL is missing a hostname: %s", rawURL)}
	}
	if isBlockedHostname(host) {
		return &inputValidationError{"Blocked host: " + host}
	}
	if ip := net.ParseIP(host); ip != nil && !isGlobalUnicast(ip) {
		return &inputValidationError{"Blocked host: " + host}
	}
	return nil
}

func isBlockedHostname(host string) bool {
	h := strings.ToLower(strings.TrimRight(host, "."))
	if blockedHostnames[h] {
		return true
	}
	for _, suffix := range blockedHostnameSuffixes {
		if strings.HasSuffix(h, suffix) {
			return true
		}
	}
	return false
}

// requestErrorMessage turns a transport error into a message for the model
// without leaking the full request URL. A timeout and a blocked host are named;
// any other error is reduced to its cause, which carries the dialed address but
// not the URL's query.
func requestErrorMessage(err error) string {
	var blocked *errBlockedHost
	if errors.As(err, &blocked) {
		return blocked.Error()
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return "Request timed out."
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "Request timed out."
	}
	// Strip the *url.Error wrapper, whose Error() prints the full request URL.
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return fmt.Sprintf("the request could not be completed: %v", urlErr.Err)
	}
	return fmt.Sprintf("the request could not be completed: %v", err)
}
