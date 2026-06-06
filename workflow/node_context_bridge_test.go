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

package workflow

import (
	"context"
	"testing"

	"google.golang.org/adk/agent"
)

// TestWithNodeBridge_SurvivesDerivedContext pins the core property
// the bridge exists for: a node context stashed via withNodeBridge
// must survive arbitrary downstream context.Context derivations
// (context.WithCancel, context.WithValue, ...) because the
// scheduler -> AgentNode -> LlmAgent -> Flow -> tool.Context chain
// produces exactly those derivations along the way (per-node
// WithCancel in scheduleResumedNode; telemetry span ctx wrap in
// Flow.handleFunctionCalls; the embedded context.Context in every
// InvocationContext built along the path).
//
// If this test regresses, NodeContextFromGoContext(toolCtx)
// lookups in runnable tools will silently return (nil, false) at
// runtime and the tools will fall back to their no-node-context
// path.
func TestWithNodeBridge_SurvivesDerivedContext(t *testing.T) {
	sentinel := agent.NewNodeContext(newMockCtx(t), "p", "1", nil, nil, nil)
	parent := withNodeBridge(context.Background(), &nodeBridge{ctx: sentinel})

	// Simulate a WithCancel (NewInvocationContext does effectively
	// this when called with the per-node nodeCtx).
	derived, cancel := context.WithCancel(parent)
	defer cancel()

	// And a WithValue layer on top (telemetry span ctx does this).
	derived = context.WithValue(derived, struct{ k string }{"span"}, "stub")

	got, ok := NodeContextFromGoContext(derived)
	if !ok {
		t.Fatal("node context lost across WithCancel + WithValue derivations")
	}
	if got != sentinel {
		t.Errorf("Derived context returned wrong node context: got %p, want %p", got, sentinel)
	}
}
