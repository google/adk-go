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

package authz_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/server/authz"
)

// ctxWithUser returns a context carrying an authenticated identity for userID,
// the way authn.Middleware populates it before a handler runs.
func ctxWithUser(userID string) context.Context {
	return authn.WithCaller(context.Background(), &authn.Caller{UserID: userID})
}

// TestStrictCanActAsUser is the core authorization check: Strict permits a
// caller to act only as the user its authenticated identity names, and rejects
// every other user. A rejection must wrap [authz.ErrUnauthorized] so callers can
// classify it with errors.Is.
func TestStrictCanActAsUser(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
		// pathUserID is the user the request is trying to act as (from the URL).
		pathUserID string
		wantAllow  bool
	}{
		{
			name:       "matching user is allowed",
			ctx:        ctxWithUser("alice"),
			pathUserID: "alice",
			wantAllow:  true,
		},
		{
			name:       "different user is denied",
			ctx:        ctxWithUser("alice"),
			pathUserID: "bob",
			wantAllow:  false,
		},
		{
			name:       "authenticated user cannot act as a different empty-string user",
			ctx:        ctxWithUser("alice"),
			pathUserID: "",
			wantAllow:  false,
		},
		{
			name:       "empty identity cannot act as a named user",
			ctx:        ctxWithUser(""),
			pathUserID: "alice",
			wantAllow:  false,
		},
		{
			// Documents a sharp edge: Strict compares UserIDs verbatim, so an
			// identity with an empty UserID (as authn.Noop yields) matches a
			// request for the empty-string user. Real routes carry a non-empty
			// user_id, so this never authorizes a genuine request, but it is why
			// Strict must not be paired with an authenticator that leaves UserID
			// empty.
			name:       "empty identity matches the empty-string user",
			ctx:        ctxWithUser(""),
			pathUserID: "",
			wantAllow:  true,
		},
	}

	s := authz.NewStrict()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.CanActAsUser(tt.ctx, tt.pathUserID)
			if tt.wantAllow {
				if err != nil {
					t.Errorf("CanActAsUser() = %v, want nil (caller may act as this user)", err)
				}
				return
			}
			if err == nil {
				t.Fatal("CanActAsUser() = nil, want an error (caller must not act as this user)")
			}
			if !errors.Is(err, authz.ErrUnauthorized) {
				t.Errorf("CanActAsUser() error = %v, want it to wrap authz.ErrUnauthorized", err)
			}
		})
	}
}

// TestStrictDeniesWithoutCallerIdentity checks Strict fails closed when the context
// carries no authenticated identity at all: a request that reached authorization
// with no identity (for example Strict wired without an authenticator) must be
// denied, not allowed, and must not panic on a nil context.
func TestStrictDeniesWithoutCallerIdentity(t *testing.T) {
	s := authz.NewStrict()

	tests := []struct {
		name string
		ctx  context.Context
	}{
		{name: "background context has no identity", ctx: context.Background()},
		{name: "nil context", ctx: nil}, //nolint:staticcheck // exercising the nil-context guard on purpose
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.CanActAsUser(tt.ctx, "alice")
			if err == nil {
				t.Fatal("CanActAsUser() = nil, want an error when no identity is present")
			}
			if !errors.Is(err, authz.ErrUnauthorized) {
				t.Errorf("CanActAsUser() error = %v, want it to wrap authz.ErrUnauthorized", err)
			}
		})
	}
}

// TestNoopAllowsEverything checks Noop is a permit-all authorizer: it returns
// nil for any user regardless of whether the context carries an identity.
func TestNoopAllowsEverything(t *testing.T) {
	n := authz.NewNoop()

	tests := []struct {
		name       string
		ctx        context.Context
		pathUserID string
	}{
		{name: "matching caller's identity", ctx: ctxWithUser("alice"), pathUserID: "alice"},
		{name: "mismatched caller's identity is still allowed", ctx: ctxWithUser("alice"), pathUserID: "bob"},
		{name: "no caller's identity is still allowed", ctx: context.Background(), pathUserID: "alice"},
		{name: "empty user is allowed", ctx: context.Background(), pathUserID: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := n.CanActAsUser(tt.ctx, tt.pathUserID); err != nil {
				t.Errorf("CanActAsUser() = %v, want nil (Noop permits everything)", err)
			}
		})
	}
}

// TestWriteHTTPStatusForAuthError pins the HTTP response an authorization
// failure produces: 403 Forbidden (the caller is authenticated but not
// permitted), never 401. The supplied error's text is appended when present.
func TestWriteHTTPStatusForAuthError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantBody string
	}{
		{
			name:     "with error appends its message",
			err:      (authz.NewStrict()).CanActAsUser(ctxWithUser("alice"), "bob"),
			wantBody: "forbidden: " + authz.ErrUnauthorized.Error(),
		},
		{
			name:     "nil error writes a bare message",
			err:      nil,
			wantBody: "forbidden",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			authz.WriteHTTPStatusForAuthError(rr, tt.err)

			if got, want := rr.Code, http.StatusForbidden; got != want {
				t.Errorf("status = %d, want %d (authorization failure is Forbidden, not Unauthorized)", got, want)
			}
			if got := strings.TrimSpace(rr.Body.String()); !strings.Contains(got, tt.wantBody) {
				t.Errorf("body = %q, want it to contain %q", got, tt.wantBody)
			}
		})
	}
}
