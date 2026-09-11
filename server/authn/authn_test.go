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

package authn

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestContextRoundTrip(t *testing.T) {
	ctx := WithCaller(context.Background(), &Caller{UserID: "alice"})

	id, ok := CallerFromContext(ctx)
	if !ok {
		t.Fatalf("CallerFromContext() ok = false, want true")
	}
	if got, want := id.UserID, "alice"; got != want {
		t.Errorf("Caller.UserID = %q, want %q", got, want)
	}
	if got, want := userIDFromContext(ctx), "alice"; got != want {
		t.Errorf("UserIDFromContext() = %q, want %q", got, want)
	}
}

func TestContextAbsent(t *testing.T) {
	if _, ok := CallerFromContext(context.Background()); ok {
		t.Errorf("CallerFromContext(empty) ok = true, want false")
	}
	if got := userIDFromContext(context.Background()); got != "" {
		t.Errorf("UserIDFromContext(empty) = %q, want empty", got)
	}
	// A nil context must not panic.
	if got := userIDFromContext(nil); got != "" { //nolint:staticcheck // exercising the nil guard on purpose
		t.Errorf("UserIDFromContext(nil) = %q, want empty", got)
	}
}

func TestWithCallerNilIsNoOp(t *testing.T) {
	ctx := WithCaller(context.Background(), nil)
	if _, ok := CallerFromContext(ctx); ok {
		t.Errorf("CallerFromContext() ok = true after WithIdentity(nil), want false")
	}
}

// echoUserID is a handler that writes the authenticated user id from the
// context, so a test can assert the middleware exposed it downstream.
func echoUserID(w http.ResponseWriter, r *http.Request) {
	_, _ = fmt.Fprint(w, userIDFromContext(r.Context()))
}

func TestMiddlewareRejectsMissingCredentials(t *testing.T) {
	auth := NewHeader("non-existing-header")
	called := false
	next := http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true })
	h := Middleware(auth)(next)

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/", nil))

	if called {
		t.Errorf("next handler ran on an unauthenticated request, want it blocked")
	}
	if got, want := rr.Code, http.StatusUnauthorized; got != want {
		t.Fatalf("status = %d, want %d", got, want)
	}
}

func TestMiddlewareInternalErrorIs500(t *testing.T) {
	auth := NewCustom(func(r *http.Request) (*Caller, error) {
		return nil, errors.New("issuer unreachable")
	})

	h := Middleware(auth)(http.HandlerFunc(echoUserID))

	r := httptest.NewRequest(http.MethodGet, "/", nil)
	r.Header.Set("Authorization", "Bearer x")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, r)

	if got, want := rr.Code, http.StatusInternalServerError; got != want {
		t.Errorf("status = %d, want %d", got, want)
	}
}

func TestMiddlewareNilIsPassThrough(t *testing.T) {
	h := Middleware(nil)(http.HandlerFunc(echoUserID))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/", nil))

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Errorf("status = %d, want %d", got, want)
	}
	if got := rr.Body.String(); got != "" {
		t.Errorf("body = %q, want empty (no identity without an authenticator)", got)
	}
}

// TestMiddlewareCallerNilNoErrorIsRejected covers a misbehaving authenticator
// that returns neither an caller's identity nor an error: the request must be rejected,
// not let through unauthenticated.
func TestMiddlewareCallerNilNoErrorIsRejected(t *testing.T) {
	auth := NewCustom(func(r *http.Request) (*Caller, error) {
		return nil, nil
	})
	called := false
	next := http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true })
	h := Middleware(auth)(next)

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/", nil))

	if called {
		t.Errorf("next handler ran despite a nil identity, want it blocked")
	}
	if got, want := rr.Code, http.StatusUnauthorized; got != want {
		t.Errorf("status = %d, want %d", got, want)
	}
}

// userIDFromContext returns the authenticated user ID carried by ctx, or the
// empty string when the request was not authenticated.
func userIDFromContext(ctx context.Context) string {
	if id, ok := CallerFromContext(ctx); ok {
		return id.UserID
	}
	return ""
}
