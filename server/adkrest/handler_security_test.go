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
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
)

// defaultPayloadLimit is the default body limit the package promises. It is
// deliberately hard-coded rather than derived from
// adkrest.DefaultMaxPayloadSize: a fixture sized from the constant under test
// shrinks with it, so changing the constant would leave the test green.
const defaultPayloadLimit = 10 << 20

// TestServerRejectsOversizedBody verifies the server rejects request bodies
// larger than ServerConfig.MaxPayloadSize. It is written to fail if the
// MaxBytesMiddleware is removed from NewServer: without the limit, the body
// below is valid JSON and the request would succeed.
func TestServerRejectsOversizedBody(t *testing.T) {
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		MaxPayloadSize: 1024,
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	// A valid JSON body that is far larger than the 1 KiB limit. "state" is a
	// map so the body decodes cleanly when the limit is not enforced.
	payload := fmt.Sprintf(`{"state": {"padding": %q}}`, strings.Repeat("a", 256*1024))
	req := httptest.NewRequest(http.MethodPost, "/apps/myapp/users/u1/sessions", bytes.NewBufferString(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if got, want := rec.Code, http.StatusBadRequest; got != want {
		t.Fatalf("oversized body status = %d, want %d", got, want)
	}
	if got, want := rec.Body.String(), "http: request body too large\n"; got != want {
		t.Fatalf("oversized body response = %q, want %q", got, want)
	}
}

// TestMaxBytesMiddlewarePinsTheDefaultLimit uses the middleware directly to pin
// both the value of the default limit and the boundary: a body of exactly the
// limit is accepted, one more byte is not.
func TestMaxBytesMiddlewarePinsTheDefaultLimit(t *testing.T) {
	handler := adkrest.MaxBytesMiddleware(0)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.ReadAll(r.Body); err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))

	send := func(size int) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(bytes.Repeat([]byte("a"), size)))
		handler.ServeHTTP(rec, req)
		return rec
	}

	if rec := send(defaultPayloadLimit); rec.Code != http.StatusNoContent {
		t.Fatalf("body of exactly the default limit: status %d, want %d (%s)", rec.Code, http.StatusNoContent, rec.Body.String())
	}
	rec := send(defaultPayloadLimit + 1)
	if got, want := rec.Code, http.StatusBadRequest; got != want {
		t.Fatalf("body one byte over the default limit: status %d, want %d", got, want)
	}
	if got, want := rec.Body.String(), "http: request body too large\n"; got != want {
		t.Fatalf("body one byte over the default limit: response %q, want %q", got, want)
	}
}

// TestServerUsesDefaultMaxPayloadSize verifies that a MaxPayloadSize of 0 or
// below selects the default limit. The small body below pins the fallback:
// without it, MaxBytesReader would be installed with a limit of 0 and reject
// that request too.
func TestServerUsesDefaultMaxPayloadSize(t *testing.T) {
	for _, tc := range []struct {
		name string
		max  int64
	}{
		{name: "zero", max: 0},
		{name: "negative", max: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, err := adkrest.NewServer(adkrest.ServerConfig{
				SessionService: session.InMemoryService(),
				MaxPayloadSize: tc.max,
			})
			if err != nil {
				t.Fatalf("NewServer: %v", err)
			}

			send := func(padding int) *httptest.ResponseRecorder {
				payload := fmt.Sprintf(`{"state": {"padding": %q}}`, strings.Repeat("a", padding))
				req := httptest.NewRequest(http.MethodPost, "/apps/myapp/users/u1/sessions", bytes.NewBufferString(payload))
				req.Header.Set("Content-Type", "application/json")
				rec := httptest.NewRecorder()
				srv.ServeHTTP(rec, req)
				return rec
			}

			// A small, valid body succeeds, so the fallback is not a limit of 0.
			if rec := send(1024); rec.Code != http.StatusOK {
				t.Fatalf("small body with default limit: status %d, want %d (%s)", rec.Code, http.StatusOK, rec.Body.String())
			}

			// A valid JSON body slightly above the hard-coded default is
			// rejected, so the fallback is that default and not unlimited.
			rec := send(defaultPayloadLimit + 4096)
			if got, want := rec.Code, http.StatusBadRequest; got != want {
				t.Fatalf("over-default body: status %d, want %d", got, want)
			}
			if got, want := rec.Body.String(), "http: request body too large\n"; got != want {
				t.Fatalf("over-default body response = %q, want %q", got, want)
			}
		})
	}
}
