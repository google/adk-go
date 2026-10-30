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

package typesafe

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// statusOverloaded is the HTTP status code TypeSafe returns when the service is
// overloaded and asks callers to back off and retry. Callers observe it through
// [ErrOverloaded] and [APIError.StatusCode].
const statusOverloaded = 529

var (
	// ErrInvalidConfig is returned when [Config] or TYPESAFE_BASE_URL is invalid,
	// or when a zero-value [Classifier] not constructed with [New] is called.
	ErrInvalidConfig = errors.New("typesafe: invalid configuration")
	// ErrMissingAPIKey is returned by [Classifier.Classify] when neither
	// [Config.APIKey] nor TYPESAFE_API_KEY is set.
	ErrMissingAPIKey = errors.New("typesafe: missing API key")
	// ErrMalformedResponse is returned when TypeSafe returns a 2xx response whose
	// body does not match the requested questions or value bounds.
	ErrMalformedResponse = errors.New("typesafe: malformed API response")
	// ErrAuthentication is returned for HTTP 401 and 403 responses.
	ErrAuthentication = errors.New("typesafe: authentication failed")
	// ErrBadRequest is returned for HTTP 400 and 422 responses.
	ErrBadRequest = errors.New("typesafe: bad request")
	// ErrRateLimited is returned when HTTP 429 persists after all retry attempts.
	ErrRateLimited = errors.New("typesafe: rate limited")
	// ErrOverloaded is returned when HTTP 529 persists after all retry attempts.
	ErrOverloaded = errors.New("typesafe: service overloaded")
	// ErrAPIStatus is returned for any other non-2xx HTTP response.
	ErrAPIStatus = errors.New("typesafe: unexpected API status")
)

// APIError represents a non-2xx HTTP response from the TypeSafe API.
//
// Unlike adk-python's TypesafeClassifier, which includes a slice of the
// response body in RuntimeError, Error formats only the HTTP status code, a
// sanitized request ID header when present, and the underlying sentinel error
// so response payloads and customer content are never echoed.
type APIError struct {
	StatusCode int
	RequestID  string
	Err        error
}

// Error formats StatusCode, RequestID (when present), and Err without exposing
// response bodies or request URLs.
func (e *APIError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "typesafe: request failed with status %d", e.StatusCode)
	if e.RequestID != "" {
		fmt.Fprintf(&b, " [request_id=%s]", e.RequestID)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %s", strings.TrimPrefix(e.Err.Error(), "typesafe: "))
	}
	return b.String()
}

// Unwrap returns the sentinel error classifying StatusCode.
func (e *APIError) Unwrap() error {
	return e.Err
}

func newAPIError(resp *http.Response) *APIError {
	var sentinel error
	switch resp.StatusCode {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		sentinel = ErrBadRequest
	case http.StatusUnauthorized, http.StatusForbidden:
		sentinel = ErrAuthentication
	case http.StatusTooManyRequests:
		sentinel = ErrRateLimited
	case statusOverloaded:
		sentinel = ErrOverloaded
	default:
		sentinel = ErrAPIStatus
	}
	reqID := sanitizeHeaderID(resp.Header.Get("X-Request-Id"))
	if reqID == "" {
		reqID = sanitizeHeaderID(resp.Header.Get("Request-Id"))
	}
	return &APIError{
		StatusCode: resp.StatusCode,
		RequestID:  reqID,
		Err:        sentinel,
	}
}

const maxRequestIDLen = 128

func sanitizeHeaderID(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || len(raw) > maxRequestIDLen {
		return ""
	}
	for i := range len(raw) {
		c := raw[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
			c == '-' || c == '_' || c == '.' || c == ':' {
			continue
		}
		return ""
	}
	return raw
}

// sanitizeTransportError strips all *url.Error layers (http.Client.Do can wrap
// an inner *url.Error inside an outer *url.Error) so the full request URL, with
// its path and any query string, is not included in returned error strings. The
// underlying network error is kept as is; it may still name the host, as in
// "dial tcp: lookup api.example.com: no such host".
func sanitizeTransportError(err error) error {
	var op string
	for {
		var urlErr *url.Error
		if !errors.As(err, &urlErr) {
			break
		}
		if op == "" {
			op = urlErr.Op
		}
		if urlErr.Err == nil {
			err = errors.New("unknown transport error")
			break
		}
		err = urlErr.Err
	}
	if op != "" {
		return fmt.Errorf("typesafe: transport error (%s): %w", op, err)
	}
	return fmt.Errorf("typesafe: transport error: %w", err)
}

// isRetryableTransportError mirrors adk-python's _RETRY_ERRORS
// (TimeoutException, NetworkError, RemoteProtocolError), excluding context
// cancellation, proxy connect failures, non-transient DNS/address errors, TLS
// certificate errors, and local URL/protocol configuration errors.
func isRetryableTransportError(err error) bool {
	if errors.Is(err, context.Canceled) {
		return false
	}
	// Per-attempt timeout elapsed (caller context cancellation is checked in doAttempt).
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	// Server closed the connection or truncated the HTTP response mid-flight.
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return dnsErr.IsTimeout || dnsErr.IsTemporary
	}
	var addrErr *net.AddrError
	if errors.As(err, &addrErr) {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return opErr.Op != "proxyconnect"
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "stream error: stream ID ") ||
		strings.Contains(msg, "http2: server sent GOAWAY")
}
