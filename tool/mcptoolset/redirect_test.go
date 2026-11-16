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

package mcptoolset

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// TestAuthCheckRedirectChains drives authCheckRedirect with chains the
// end-to-end tests cannot build without TLS: a hop after the chain moved to
// https, which only the comparison against the previous hop refuses, and the
// GET chain the SDK uses for its server-to-client stream.
func TestAuthCheckRedirectChains(t *testing.T) {
	allowAll := func(*http.Request, []*http.Request) error { return nil }
	retarget := func(req *http.Request, _ []*http.Request) error { req.URL.Host = "other"; return nil }
	tests := []struct {
		name   string
		method string
		// reqMethod, when set, is the method of the hop under test. A 303
		// turns a POST into a GET, so a chain can carry both.
		reqMethod string
		via       []string
		to        string
		next      func(*http.Request, []*http.Request) error
		wantErr   string // "" for allowed, "use last response" for the unfollowed 3xx
	}{
		{name: "downgrade after an upgrade, with a permissive hook", method: http.MethodPost, via: []string{"http://h/a", "https://h/b"}, to: "http://h/c", next: allowAll, wantErr: "refusing redirect"},
		{name: "stays on https after an upgrade, with a permissive hook", method: http.MethodPost, via: []string{"http://h/a", "https://h/b"}, to: "https://h/c", next: allowAll},
		{name: "downgrade after an upgrade, no hook", method: http.MethodPost, via: []string{"http://h/a", "https://h/b"}, to: "http://h/c", wantErr: "refusing redirect"},
		// The original request was http, so only the copy of the previous hop
		// catches a hook that drops the scheme back down.
		{name: "hook downgrades the scheme after an upgrade", method: http.MethodPost, via: []string{"http://h/a", "https://h/b"}, to: "https://h/c", next: func(req *http.Request, _ []*http.Request) error { req.URL.Scheme = "http"; return nil }, wantErr: "refusing redirect"},
		{name: "cross host on a POST chain", method: http.MethodPost, via: []string{"https://h/a"}, to: "https://other/a", wantErr: "refusing redirect"},
		{name: "cross host on a GET chain", method: http.MethodGet, via: []string{"https://h/a"}, to: "https://other/a", wantErr: "use last response"},
		{name: "cross host on a GET chain, with a permissive hook", method: http.MethodGet, via: []string{"https://h/a"}, to: "https://other/a", next: allowAll, wantErr: "use last response"},
		// A hook that retargets the request is refused after it ran, and on
		// a GET chain that refusal is the unfollowed 3xx as well.
		{name: "hook retargets a POST chain", method: http.MethodPost, via: []string{"https://h/a"}, to: "https://h/b", next: retarget, wantErr: "refusing redirect"},
		{name: "hook retargets a GET chain", method: http.MethodGet, via: []string{"https://h/a"}, to: "https://h/b", next: retarget, wantErr: "use last response"},
		{name: "hook sets the Host header on a GET chain", method: http.MethodGet, via: []string{"https://h/a"}, to: "https://h/b", next: func(req *http.Request, _ []*http.Request) error { req.Host = "other"; return nil }, wantErr: "use last response"},
		// The chain's first request decides: a POST chain a 303 turned into
		// GET still reports the refusal as an error.
		{name: "cross host after a 303 turned POST into GET", method: http.MethodPost, reqMethod: http.MethodGet, via: []string{"https://h/a", "https://h/b"}, to: "https://other/c", wantErr: "refusing redirect"},
		// And a hook that rewrites via[0].Method cannot change that.
		{name: "hook rewrites the chain's first method", method: http.MethodPost, via: []string{"https://h/a"}, to: "https://h/b", next: func(req *http.Request, via []*http.Request) error {
			via[0].Method = http.MethodGet
			req.URL.Host = "other"
			return nil
		}, wantErr: "refusing redirect"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqMethod := tc.method
			if tc.reqMethod != "" {
				reqMethod = tc.reqMethod
			}
			var via []*http.Request
			for i, raw := range tc.via {
				u, err := url.Parse(raw)
				if err != nil {
					t.Fatal(err)
				}
				// Only the chain's first request keeps the original method.
				method := reqMethod
				if i == 0 {
					method = tc.method
				}
				via = append(via, &http.Request{Method: method, URL: u})
			}
			u, err := url.Parse(tc.to)
			if err != nil {
				t.Fatal(err)
			}
			err = authCheckRedirect(tc.next)(&http.Request{Method: reqMethod, URL: u}, via)
			switch tc.wantErr {
			case "":
				if err != nil {
					t.Errorf("error = %v, want nil", err)
				}
			case "use last response":
				// net/http recognizes the sentinel by identity, so a wrapped
				// one would not do.
				if err != http.ErrUseLastResponse {
					t.Errorf("error = %v, want http.ErrUseLastResponse", err)
				}
			default:
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error = %v, want it to contain %q", err, tc.wantErr)
				}
			}
		})
	}
}

// TestRedactRedirectTarget drives the redaction over hand-built error chains:
// the shape net/http and the SDK produce, plus the shapes that must be left
// alone or must keep their outer messages.
func TestRedactRedirectTarget(t *testing.T) {
	refused := refuseRedirect(http.MethodPost, errors.New("refusing redirect from http://h to http://other"))
	leak := &url.Error{Op: "Post", URL: "http://other/cb?code=SECRET", Err: refused}
	tests := []struct {
		name string
		err  error
		want string // "" leaves err unchanged
	}{
		{name: "bare url.Error", err: leak, want: refused.Error()},
		{name: "wrapped once", err: fmt.Errorf("sending: %w", leak), want: "sending: " + refused.Error()},
		{name: "wrapped twice with another error beside it", err: fmt.Errorf("calling: %w", fmt.Errorf("sending: %w: %w", errors.New("rejected by transport"), leak)), want: "calling: sending: rejected by transport: " + refused.Error()},
		// A layer whose message does not quote the inner error, such as a
		// custom type, is dropped in favor of the redacted error.
		{name: "opaque layer", err: opaqueErr{leak}, want: refused.Error()},
		// Nothing to redact: no refusal under it, or a url.Error around some
		// other error even with a refusal elsewhere in the chain.
		{name: "no refusal", err: fmt.Errorf("x: %w", &url.Error{Op: "Get", URL: "http://h/?q=1", Err: errors.New("boom")})},
		{name: "refusal not under the url.Error", err: fmt.Errorf("%w: %w", &url.Error{Op: "Get", URL: "http://h/?q=1", Err: errors.New("boom")}, refused)},
		{name: "plain error", err: errors.New("plain")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := redactRedirectTarget(tc.err)
			if tc.want == "" {
				if got != tc.err {
					t.Errorf("redactRedirectTarget() = %v, want the error unchanged", got)
				}
				return
			}
			if got.Error() != tc.want {
				t.Errorf("redactRedirectTarget() = %q, want %q", got.Error(), tc.want)
			}
			if strings.Contains(got.Error(), "SECRET") {
				t.Errorf("redactRedirectTarget() = %q still carries the Location", got.Error())
			}
			if !errors.Is(got, refused) {
				t.Errorf("redactRedirectTarget() lost the refusal from the chain")
			}
		})
	}
}

type opaqueErr struct{ err error }

func (e opaqueErr) Error() string { return "opaque" }
func (e opaqueErr) Unwrap() error { return e.err }
