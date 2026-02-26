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
	"strings"
	"testing"

	"google.golang.org/adk/agent"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestToolContext(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{}, nil)

	if _, ok := toolCtx.(agent.ReadonlyContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a ReadonlyContext", toolCtx)
	}
	if _, ok := toolCtx.(agent.CallbackContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a CallbackContext", toolCtx)
	}
}

func TestRequestConfirmation_SetsSkipSummarization(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := NewToolContext(inv, "fn1", actions, nil)

	err := toolCtx.RequestConfirmation("please confirm", map[string]any{"key": "value"})
	if err != nil {
		t.Fatalf("RequestConfirmation returned unexpected error: %v", err)
	}

	if !actions.SkipSummarization {
		t.Error("RequestConfirmation did not set SkipSummarization to true")
	}

	if actions.RequestedToolConfirmations == nil {
		t.Fatal("RequestConfirmation did not set RequestedToolConfirmations")
	}
	tc, ok := actions.RequestedToolConfirmations["fn1"]
	if !ok {
		t.Fatal("RequestConfirmation did not set confirmation for function call ID 'fn1'")
	}
	if tc.Hint != "please confirm" {
		t.Errorf("expected hint 'please confirm', got %q", tc.Hint)
	}
	if tc.Confirmed {
		t.Error("expected Confirmed to be false")
	}
}

func TestRequestConfirmation_AutoGeneratesIDWhenEmpty(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	// NewToolContext auto-generates a UUID when functionCallID is empty.
	toolCtx := NewToolContext(inv, "", actions, nil)

	err := toolCtx.RequestConfirmation("hint", nil)
	if err != nil {
		t.Fatalf("RequestConfirmation returned unexpected error: %v", err)
	}

	if !actions.SkipSummarization {
		t.Error("SkipSummarization should be set even with auto-generated function call ID")
	}
	if len(actions.RequestedToolConfirmations) != 1 {
		t.Fatalf("expected 1 confirmation entry, got %d", len(actions.RequestedToolConfirmations))
	}
	for _, tc := range actions.RequestedToolConfirmations {
		if tc.Hint != "hint" {
			t.Errorf("expected hint 'hint', got %q", tc.Hint)
		}
		if tc.Confirmed {
			t.Error("expected Confirmed to be false")
		}
	}
}

func TestInternalArtifacts_NilArtifactsService(t *testing.T) {
	// Create a tool context with nil artifacts service
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{
		Artifacts: nil, // Explicitly set to nil to simulate misconfiguration
	})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{}, nil)

	artifacts := toolCtx.Artifacts()

	t.Run("List returns error for nil service", func(t *testing.T) {
		_, err := artifacts.List(t.Context())
		if err == nil {
			t.Fatal("expected error when calling List on nil artifacts service, got nil")
		}
		expectedMsg := "artifact service is not configured"
		if !strings.Contains(err.Error(), expectedMsg) {
			t.Errorf("expected error message to contain %q, got %q", expectedMsg, err.Error())
		}
	})

	t.Run("Load returns error for nil service", func(t *testing.T) {
		_, err := artifacts.Load(t.Context(), "test.txt")
		if err == nil {
			t.Fatal("expected error when calling Load on nil artifacts service, got nil")
		}
		expectedMsg := "artifact service is not configured"
		if !strings.Contains(err.Error(), expectedMsg) {
			t.Errorf("expected error message to contain %q, got %q", expectedMsg, err.Error())
		}
	})

	t.Run("LoadVersion returns error for nil service", func(t *testing.T) {
		_, err := artifacts.LoadVersion(t.Context(), "test.txt", 1)
		if err == nil {
			t.Fatal("expected error when calling LoadVersion on nil artifacts service, got nil")
		}
		expectedMsg := "artifact service is not configured"
		if !strings.Contains(err.Error(), expectedMsg) {
			t.Errorf("expected error message to contain %q, got %q", expectedMsg, err.Error())
		}
	})

	t.Run("Save returns error for nil service", func(t *testing.T) {
		_, err := artifacts.Save(t.Context(), "test.txt", nil)
		if err == nil {
			t.Fatal("expected error when calling Save on nil artifacts service, got nil")
		}
		expectedMsg := "artifact service is not configured"
		if !strings.Contains(err.Error(), expectedMsg) {
			t.Errorf("expected error message to contain %q, got %q", expectedMsg, err.Error())
		}
	})
}
