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

func TestToolContext_Confirmation(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := NewToolContextWithToolName(inv, "fn1", actions, "test_tool")

	hint := "This is a test confirmation"
	payload := map[string]any{"key": "value"}

	// Initially, no confirmation should be requested
	if actions.ConfirmationRequest != nil {
		t.Errorf("ConfirmationRequest should be nil initially, got: %v", actions.ConfirmationRequest)
	}

	// Request confirmation
	err := toolCtx.RequestConfirmation(hint, payload)
	if err == nil {
		t.Errorf("Expected RequestConfirmation to return an error to indicate confirmation is required")
	}

	// Check that confirmation request was stored in actions
	if actions.ConfirmationRequest == nil {
		t.Error("ConfirmationRequest should not be nil after calling RequestConfirmation")
	} else {
		if actions.ConfirmationRequest.Hint != hint {
			t.Errorf("Expected hint %q, got %q", hint, actions.ConfirmationRequest.Hint)
		}
		if actions.ConfirmationRequest.ToolName != "test_tool" {
			t.Errorf("Expected tool name %q, got %q", "test_tool", actions.ConfirmationRequest.ToolName)
		}
		if len(actions.ConfirmationRequest.Payload) != 1 || actions.ConfirmationRequest.Payload["key"] != "value" {
			t.Errorf("Payload was not stored correctly: %v", actions.ConfirmationRequest.Payload)
		}
	}

	// Try to request another confirmation - should fail
	err = toolCtx.RequestConfirmation("Another request", map[string]any{})
	if err == nil {
		t.Error("Expected second call to RequestConfirmation to fail")
	}
}

func TestToolContext_ConfirmationNoToolName(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := NewToolContext(inv, "fn1", actions)

	hint := "This is a test confirmation"
	payload := map[string]any{"key": "value"}

	// Request confirmation - should fail because tool name is missing
	err := toolCtx.RequestConfirmation(hint, payload)
	if err == nil {
		t.Error("Expected RequestConfirmation to return an error when tool name is missing")
	}

	// Check that no confirmation request was stored
	if actions.ConfirmationRequest != nil {
		t.Error("ConfirmationRequest should be nil when RequestConfirmation fails")
	}
}

