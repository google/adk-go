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

package method

import (
	"context"
	"strings"
)

// authenticatedUserIDKey is the context key used to store the authenticated
// user identity extracted from the platform request (e.g. a verified JWT sub
// claim or a Vertex AI service-account principal injected by middleware).
type authenticatedUserIDKey struct{}

// WithAuthenticatedUserID returns a new context carrying the authenticated
// user identity.  Middleware that validates the inbound credential SHOULD call
// this before handing the context to any method handler.
func WithAuthenticatedUserID(ctx context.Context, userID string) context.Context {
	return context.WithValue(ctx, authenticatedUserIDKey{}, userID)
}

// authenticatedUserID retrieves the authenticated user identity from ctx.
// The second return value is true only when an authenticated identity was
// previously stored by WithAuthenticatedUserID.
func authenticatedUserID(ctx context.Context) (string, bool) {
	v, ok := ctx.Value(authenticatedUserIDKey{}).(string)
	if !ok || v == "" {
		return "", false
	}
	return v, true
}

// resolveUserID returns the user identity that should be used for a session
// operation.
//
//   - If an authenticated identity is present in ctx it is returned and the
//     caller-supplied value is ignored, preventing BOLA/IDOR.
//   - Otherwise the caller-supplied value is used as-is (preserves backward
//     compatibility with single-tenant / local-dev deployments where no auth
//     middleware is configured).
func resolveUserID(ctx context.Context, callerSupplied string) string {
	if authed, ok := authenticatedUserID(ctx); ok {
		return authed
	}
	return callerSupplied
}

// sanitizeExternalState removes keys with the "app:" and "user:" prefixes from
// a state map supplied by an external caller.
//
// These prefixes are interpreted by the session service as writes into shared
// app-wide and per-user state stores.  Allowing untrusted callers to set them
// directly would let a caller poison state that is visible to other users or
// to the same user across unrelated sessions.  Only trusted server-side code
// paths should be able to write prefixed keys.
func sanitizeExternalState(state map[string]any) map[string]any {
	if len(state) == 0 {
		return state
	}
	sanitized := make(map[string]any, len(state))
	for k, v := range state {
		if strings.HasPrefix(k, "app:") || strings.HasPrefix(k, "user:") {
			// Drop keys that target shared state stores.
			continue
		}
		sanitized[k] = v
	}
	return sanitized
}
