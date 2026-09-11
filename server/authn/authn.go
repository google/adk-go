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

// Package authn provides inbound HTTP authentication for the ADK REST API.
//
// It defines a small [Authenticator] seam that different providers implement
// and a [Middleware] that gates authenticated endpoints and answers 401 when a request carries no valid
// credentials
//
// A server wires an [Authenticator] through adkrest.ServerConfig. Endpoints are
// authenticated by default and only those explicitly marked public (for example
// /health and /version) are reachable without credentials.
package authn

import (
	"context"
	"errors"
	"net/http"
)

// ErrUnauthenticated reports that a request carried no valid credentials. An
// [Authenticator] returns it, or an error wrapping it, to make [Middleware]
// answer 401 Unauthorized. Any other error is treated as an internal provider
// failure and answered 500.
var ErrUnauthenticated = errors.New("authn: unauthenticated")

// Caller is the authenticated principal resolved from a request.
type Caller struct {
	// UserID is the stable identifier of the caller. Middleware puts it on the
	// request context, where it can be read with [CallerFromContext].
	UserID string
	// Claims carries optional provider-specific attributes (email, roles, token
	// scopes, ...). It may be nil.
	Claims map[string]any
}

// Authenticator authenticates an inbound HTTP request. Implementations back
// different providers/schemes. Use [NewCustom] to adapt a plain function.
type Authenticator interface {
	// Authenticate verifies the request's credentials and returns the
	// authenticated identity. It returns an error wrapping [ErrUnauthenticated]
	// when the request has no valid credentials for this provider.
	Authenticate(r *http.Request) (*Caller, error)
}

// callerKey is the context key under which an [Caller] is stored.
type callerKey struct{}

// WithCaller returns a copy of ctx carrying id. A nil id is ignored and ctx
// is returned unchanged.
func WithCaller(ctx context.Context, id *Caller) context.Context {
	if id == nil {
		return ctx
	}
	return context.WithValue(ctx, callerKey{}, id)
}

// CallerFromContext returns the [Caller] carried by ctx, reporting false
// when the request was not authenticated.
func CallerFromContext(ctx context.Context) (*Caller, bool) {
	if ctx == nil {
		return nil, false
	}
	id, ok := ctx.Value(callerKey{}).(*Caller)
	return id, ok && id != nil
}

// Middleware returns HTTP middleware that authenticates every request
// before passing it to the next handler, and answers 401 (or 500 on an internal
// provider failure) when authentication fails. On success it stores the
// resolved identity on the request context, where downstream handlers read it
// with [CallerFromContext].
//
// A nil a yields pass-through middleware, so a caller can wire Middleware
// unconditionally.
func Middleware(a Authenticator) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		if a == nil {
			return next
		}
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id, err := a.Authenticate(r)
			if err != nil {
				writeAuthError(w, err)
				return
			}
			if id == nil {
				// A provider that reports neither caller's identity nor error is
				// treated as a decline rather than silently letting the
				// request through unauthenticated.
				writeAuthError(w, ErrUnauthenticated)
				return
			}
			next.ServeHTTP(w, r.WithContext(WithCaller(r.Context(), id)))
		})
	}
}

// writeAuthError answers a failed authentication. A credential problem is a
// 401; anything else is an internal provider failure and a 500.
func writeAuthError(w http.ResponseWriter, err error) {
	if !errors.Is(err, ErrUnauthenticated) {
		// A provider-internal failure (for example an unreachable token
		// issuer), not a credential problem. Keep the detail out of the
		// response; a provider is expected to log its own errors.
		http.Error(w, "authentication failed", http.StatusInternalServerError)
		return
	}

	http.Error(w, "unauthorized", http.StatusUnauthorized)
}
