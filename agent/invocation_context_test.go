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
	"context"
	"testing"

	"google.golang.org/adk/agent"
)

type testKey struct{}

// TestNewInvocationContext_GeneratesID verifies that an
// InvocationContext built without an explicit ID gets a non-empty,
// "e-"-prefixed identifier auto-assigned.
func TestNewInvocationContext_GeneratesID(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{})
	id := inv.InvocationID()
	if id == "" {
		t.Fatal("NewInvocationContext: InvocationID is empty, want non-empty auto-generated ID")
	}
	if got := id[:2]; got != "e-" {
		t.Errorf("NewInvocationContext: InvocationID = %q, want prefix %q", id, "e-")
	}
}

// TestNewInvocationContext_KeepsExplicitID verifies that an explicit
// InvocationID in params is preserved verbatim.
func TestNewInvocationContext_KeepsExplicitID(t *testing.T) {
	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{
		InvocationID: "explicit-id",
	})
	if got := inv.InvocationID(); got != "explicit-id" {
		t.Errorf("NewInvocationContext: InvocationID = %q, want %q", got, "explicit-id")
	}
}

// TestInvocationContext_WithContext verifies that WithContext returns a
// new context whose embedded context.Context is the one passed in,
// while preserving the rest of the invocation params (here: Branch).
func TestInvocationContext_WithContext(t *testing.T) {
	baseCtx := t.Context()
	inv := agent.NewInvocationContext(baseCtx, agent.InvocationContextParams{
		Branch: "test-branch",
	})

	key := testKey{}
	val := "val"
	got := inv.WithContext(context.WithValue(baseCtx, key, val))

	if got.Value(key) != val {
		t.Errorf("WithContext() did not update embedded context")
	}
	if got.Branch() != "test-branch" {
		t.Errorf("WithContext() lost Branch param: got %q, want %q", got.Branch(), "test-branch")
	}
	if got.InvocationID() != inv.InvocationID() {
		t.Errorf("WithContext() lost InvocationID: got %q, want %q", got.InvocationID(), inv.InvocationID())
	}
}

// TestInvocationContext_WithAgent verifies that WithAgent returns a new
// context with the Agent overridden, while preserving other params and
// without mutating the receiver.
func TestInvocationContext_WithAgent(t *testing.T) {
	parentAgent := mustNewAgent(t, "parent")
	childAgent := mustNewAgent(t, "child")

	inv := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{
		Agent:  parentAgent,
		Branch: "test-branch",
	})

	got := inv.WithAgent(childAgent)

	if got.Agent() != childAgent {
		t.Errorf("WithAgent(): Agent = %v, want %v", got.Agent(), childAgent)
	}
	if got.Branch() != "test-branch" {
		t.Errorf("WithAgent() lost Branch param: got %q, want %q", got.Branch(), "test-branch")
	}
	// Receiver unchanged:
	if inv.Agent() != parentAgent {
		t.Errorf("WithAgent() mutated receiver: Agent = %v, want %v (parent)", inv.Agent(), parentAgent)
	}
}

// TestInvocationContext_EndInvocationIsolation verifies that ending the
// invocation on a derived context (from WithAgent) does not bleed back
// to the receiver — each branch has its own lifecycle flag.
func TestInvocationContext_EndInvocationIsolation(t *testing.T) {
	parent := agent.NewInvocationContext(t.Context(), agent.InvocationContextParams{})
	child := parent.WithAgent(mustNewAgent(t, "child"))

	child.EndInvocation()

	if parent.Ended() {
		t.Error("parent.Ended() = true after only child.EndInvocation(); lifecycle leaked")
	}
	if !child.Ended() {
		t.Error("child.Ended() = false after child.EndInvocation()")
	}
}

func mustNewAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name})
	if err != nil {
		t.Fatalf("agent.New(%q): %v", name, err)
	}
	return a
}
