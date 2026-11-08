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
	tests := []struct {
		name    string
		method  string
		via     []string
		to      string
		next    func(*http.Request, []*http.Request) error
		wantErr string // "" for allowed, "use last response" for the unfollowed 3xx
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
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var via []*http.Request
			for _, raw := range tc.via {
				u, err := url.Parse(raw)
				if err != nil {
					t.Fatal(err)
				}
				via = append(via, &http.Request{Method: tc.method, URL: u})
			}
			u, err := url.Parse(tc.to)
			if err != nil {
				t.Fatal(err)
			}
			err = authCheckRedirect(tc.next)(&http.Request{Method: tc.method, URL: u}, via)
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
