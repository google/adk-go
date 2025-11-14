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

package toolinternal

import (
	"testing"

	"google.golang.org/adk/agent"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestToolContext(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{})

	if _, ok := toolCtx.(agent.ReadonlyContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a ReadonlyContext", toolCtx)
	}
	if _, ok := toolCtx.(agent.CallbackContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a CallbackContext", toolCtx)
	}
	if got, ok := toolCtx.(agent.InvocationContext); ok {
		t.Errorf("ToolContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestInternalArtifacts_NilSafe(t *testing.T) {
	// Create invocation context without artifact service
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{
		Artifacts: nil,
	})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{})

	artifacts := toolCtx.Artifacts()
	// artifacts will be nil when service not configured

	// Attempting to call methods on nil should be safe (won't panic)
	// but will return errors
	t.Run("List returns error", func(t *testing.T) {
		_, err := artifacts.List(t.Context())
		if err == nil {
			t.Error("Expected error from List(), got nil")
		}
		expectedMsg := "artifact service not configured"
		if err != nil && err.Error() != expectedMsg {
			t.Errorf("Expected error %q, got: %v", expectedMsg, err)
		}
	})

	t.Run("Load returns error", func(t *testing.T) {
		_, err := artifacts.Load(t.Context(), "test.txt")
		if err == nil {
			t.Error("Expected error from Load(), got nil")
		}
		expectedMsg := "artifact service not configured"
		if err != nil && err.Error() != expectedMsg {
			t.Errorf("Expected error %q, got: %v", expectedMsg, err)
		}
	})

	t.Run("Save returns error", func(t *testing.T) {
		_, err := artifacts.Save(t.Context(), "test.txt", nil)
		if err == nil {
			t.Error("Expected error from Save(), got nil")
		}
		expectedMsg := "artifact service not configured"
		if err != nil && err.Error() != expectedMsg {
			t.Errorf("Expected error %q, got: %v", expectedMsg, err)
		}
	})
}
