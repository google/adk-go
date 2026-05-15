// Copyright 2025 Google LLC
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

package openai

import "context"

// Context keys for session management
type contextKey string

const (
	// SessionIDKey is the context key for session ID
	SessionIDKey contextKey = "sessionID"
)

// WithSessionID creates a new context with the given session ID.
//
// Example:
//
//	ctx := openai.WithSessionID(context.Background(), "user-session-123")
//	model.GenerateContent(ctx, req, false)
func WithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, SessionIDKey, sessionID)
}

// GetSessionID retrieves the session ID from the context.
// Returns empty string if not found.
func GetSessionID(ctx context.Context) string {
	if sessionID, ok := ctx.Value(SessionIDKey).(string); ok && sessionID != "" {
		return sessionID
	}
	return ""
}

// MustGetSessionID retrieves session ID from context or panics if not found.
// Use this when session ID is absolutely required.
func MustGetSessionID(ctx context.Context) string {
	sessionID := GetSessionID(ctx)
	if sessionID == "" {
		panic("session ID not found in context - use openai.WithSessionID()")
	}
	return sessionID
}

// HasSessionID checks if context contains a session ID.
func HasSessionID(ctx context.Context) bool {
	return GetSessionID(ctx) != ""
}
