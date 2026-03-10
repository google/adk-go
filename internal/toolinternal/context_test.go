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
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{}, nil)

	if _, ok := toolCtx.(agent.ReadonlyContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a ReadonlyContext", toolCtx)
	}
	if _, ok := toolCtx.(agent.CallbackContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a CallbackContext", toolCtx)
	}
}

func TestRequestConfirmation_SetsExitLoop(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := NewToolContext(inv, "fn1", actions, nil)

	err := toolCtx.RequestConfirmation("please confirm", map[string]any{"key": "value"})
	if err != nil {
		t.Fatalf("RequestConfirmation returned unexpected error: %v", err)
	}

	if !actions.ExitLoop {
		t.Error("RequestConfirmation did not set ExitLoop to true")
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

	if !actions.ExitLoop {
		t.Error("ExitLoop should be set even with auto-generated function call ID")
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

func TestExitLoop(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := NewToolContext(inv, "fn1", actions, nil)

	if actions.ExitLoop {
		t.Error("ExitLoop should be false initially")
	}

	toolCtx.ExitLoop()

	if !actions.ExitLoop {
		t.Error("ExitLoop should be true after calling ExitLoop()")
	}
	// SkipSummarization should also be set for backward compatibility
	if !actions.SkipSummarization {
		t.Error("SkipSummarization should be true for backward compatibility")
	}
}
