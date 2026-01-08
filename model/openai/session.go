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

import (
	"context"
	"log"

	"github.com/google/uuid"
)

// Context keys for session management
type contextKey string

const (
	// SessionIDKey is the context key for session ID
	SessionIDKey contextKey = "sessionID"

	// Alternative keys that might be used by different frameworks
	sessionIDKeyAlt1 contextKey = "session_id"
	sessionIDKeyAlt2 contextKey = "SessionID"
)

// SessionConfig holds configuration for session management.
type SessionConfig struct {
	// Logger for session-related warnings (optional)
	Logger *log.Logger
	// DisableAutoGeneration disables automatic UUID generation for missing sessionID
	DisableAutoGeneration bool
}

var (
	// defaultSessionConfig is used when no config is provided
	defaultSessionConfig = &SessionConfig{
		Logger:                nil,
		DisableAutoGeneration: false,
	}
)

// WithSessionID creates a new context with the given session ID.
// This is the recommended way to pass session ID to the OpenAI adapter.
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

// extractSessionID extracts session ID from context with multiple fallback strategies.
// Priority order:
// 1. SessionIDKey ("sessionID")
// 2. Alternative keys ("session_id", "SessionID")
// 3. Auto-generated UUID (with warning if enabled)
// 4. Empty string (if auto-generation disabled)
func extractSessionID(ctx context.Context, cfg *SessionConfig) string {
	if cfg == nil {
		cfg = defaultSessionConfig
	}

	// Try primary key
	if sessionID, ok := ctx.Value(SessionIDKey).(string); ok && sessionID != "" {
		return sessionID
	}

	// Try alternative keys for compatibility
	alternativeKeys := []contextKey{sessionIDKeyAlt1, sessionIDKeyAlt2}
	for _, key := range alternativeKeys {
		if sessionID, ok := ctx.Value(key).(string); ok && sessionID != "" {
			if cfg.Logger != nil {
				cfg.Logger.Printf("WARNING: Using alternative session key '%s', prefer openai.WithSessionID()", key)
			}
			return sessionID
		}
	}

	// No session ID found - auto-generate or return empty
	if !cfg.DisableAutoGeneration {
		generatedID := uuid.NewString()
		if cfg.Logger != nil {
			cfg.Logger.Printf("WARNING: No session ID in context, auto-generated: %s", generatedID)
		}
		return generatedID
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

// extractSessionIDWithLogging is a helper that extracts session ID with logging enabled.
// This is used internally by the OpenAI adapter.
func extractSessionIDWithLogging(ctx context.Context, logger *log.Logger) string {
	cfg := &SessionConfig{
		Logger:                logger,
		DisableAutoGeneration: false,
	}
	return extractSessionID(ctx, cfg)
}

// extractSessionIDStrict extracts session ID without auto-generation.
// Returns empty string if not found. No warnings logged.
func extractSessionIDStrict(ctx context.Context) string {
	cfg := &SessionConfig{
		Logger:                nil,
		DisableAutoGeneration: true,
	}
	return extractSessionID(ctx, cfg)
}
