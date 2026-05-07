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
	"testing"

	"google.golang.org/adk/agent"
)

// TestNewReadonlyContext_NotAnInvocationContext verifies that the
// ReadonlyContext returned by NewReadonlyContext does not accidentally
// satisfy the wider InvocationContext interface — clients holding a
// ReadonlyContext should not be able to type-assert their way to
// methods like EndInvocation, Memory, or WithContext.
func TestNewReadonlyContext_NotAnInvocationContext(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{})
	readonly := agent.NewReadonlyContext(inv)

	if got, ok := readonly.(agent.InvocationContext); ok {
		t.Errorf("ReadonlyContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

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
