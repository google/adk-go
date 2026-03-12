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
	"testing"
)

func TestWithSessionID(t *testing.T) {
	ctx := context.Background()
	sessionID := "test-session-123"

	ctx = WithSessionID(ctx, sessionID)

	retrieved := GetSessionID(ctx)
	if retrieved != sessionID {
		t.Errorf("Expected session ID '%s', got '%s'", sessionID, retrieved)
	}
}

func TestGetSessionID_NotFound(t *testing.T) {
	ctx := context.Background()

	retrieved := GetSessionID(ctx)
	if retrieved != "" {
		t.Errorf("Expected empty string, got '%s'", retrieved)
	}
}

func TestHasSessionID(t *testing.T) {
	ctx := context.Background()

	if HasSessionID(ctx) {
		t.Error("Expected HasSessionID to return false for empty context")
	}

	ctx = WithSessionID(ctx, "test-session")

	if !HasSessionID(ctx) {
		t.Error("Expected HasSessionID to return true after setting session ID")
	}
}

func TestMustGetSessionID_Success(t *testing.T) {
	ctx := WithSessionID(context.Background(), "test-session")

	sessionID := MustGetSessionID(ctx)
	if sessionID != "test-session" {
		t.Errorf("Expected 'test-session', got '%s'", sessionID)
	}
}

func TestMustGetSessionID_Panic(t *testing.T) {
	ctx := context.Background()

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when session ID not found")
		}
	}()

	MustGetSessionID(ctx)
}
