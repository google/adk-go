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

package agent_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/agent"
)

// TestNewReadonlyContext_DelegatesReads verifies that the
// ReadonlyContext's read methods delegate to the wrapped
// InvocationContext (through Branch and InvocationID as canaries).
func TestNewReadonlyContext_DelegatesReads(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{
		Branch:       "test-branch",
		InvocationID: "test-id",
	})
	readonly := agent.NewReadonlyContext(inv)

	if got := readonly.Branch(); got != "test-branch" {
		t.Errorf("ReadonlyContext.Branch() = %q, want %q", got, "test-branch")
	}
	if got := readonly.InvocationID(); got != "test-id" {
		t.Errorf("ReadonlyContext.InvocationID() = %q, want %q", got, "test-id")
	}
}

// TestNewReadonlyContext_ToolMethodsReturnZero verifies the "mix"
// policy from the unified Context API: when a tool-only capability is
// invoked from a non-tool context, pollable accessors return zero
// values and mutating actions return ErrOutsideToolCall.
func TestNewReadonlyContext_ToolMethodsReturnZero(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{})
	readonly := agent.NewReadonlyContext(inv)

	if got := readonly.FunctionCallID(); got != "" {
		t.Errorf("ReadonlyContext.FunctionCallID() = %q, want empty", got)
	}
	if got := readonly.Actions(); got != nil {
		t.Errorf("ReadonlyContext.Actions() = %v, want nil", got)
	}
	if got := readonly.ToolConfirmation(); got != nil {
		t.Errorf("ReadonlyContext.ToolConfirmation() = %v, want nil", got)
	}
	if err := readonly.RequestConfirmation("hint", nil); !errors.Is(err, agent.ErrOutsideToolCall) {
		t.Errorf("ReadonlyContext.RequestConfirmation() = %v, want %v", err, agent.ErrOutsideToolCall)
	}
}

// TestNewCallbackContext_ToolMethodsReturnZero verifies the same mix
// policy on a CallbackContext-flavoured Context. Callbacks have access
// to the writable State and Artifacts surface but are not a tool-call
// site, so tool-only capabilities still return zero values / errors.
func TestNewCallbackContext_ToolMethodsReturnZero(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{})
	cb := agent.NewCallbackContext(inv)

	if got := cb.FunctionCallID(); got != "" {
		t.Errorf("CallbackContext.FunctionCallID() = %q, want empty", got)
	}
	if got := cb.ToolConfirmation(); got != nil {
		t.Errorf("CallbackContext.ToolConfirmation() = %v, want nil", got)
	}
	if err := cb.RequestConfirmation("hint", nil); !errors.Is(err, agent.ErrOutsideToolCall) {
		t.Errorf("CallbackContext.RequestConfirmation() = %v, want %v", err, agent.ErrOutsideToolCall)
	}
}
