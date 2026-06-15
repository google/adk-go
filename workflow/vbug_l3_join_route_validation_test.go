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

// FINDING L3 — conditional edge into a JoinNode is not rejected.
//
// Bug: a JoinNode is a fan-in barrier that releases only after EVERY
// declared predecessor has completed. Routing a CONDITIONAL edge into a
// JoinNode is therefore a configuration error: when the route does not
// match, that predecessor never fires, so the barrier can never release
// and the join (and everything downstream) silently dead-ends. The
// JoinNode docs explicitly call this "a configuration error", yet
// validateWorkflow has no check for it, so New(...) accepts the graph.
//
// Expected correct behavior: New(...) should reject a graph in which any
// incoming edge to a JoinNode carries a concrete (conditional) Route,
// returning a non-nil validation error.
//
// This test currently FAILS, demonstrating the bug: New returns nil for
// a graph whose p1 -> join edge is conditional.

package workflow

import (
	"testing"

	"google.golang.org/adk/agent"
)

// TestVbugL3_ConditionalEdgeIntoJoinRejected builds an otherwise-valid
// graph in which a JoinNode has one conditional predecessor (p1, via a
// StringRoute) and one unconditional predecessor (p2). The graph passes
// every existing validation check, so the ONLY thing that could reject
// it is a (missing) "no conditional edge into a JoinNode" rule.
func TestVbugL3_ConditionalEdgeIntoJoinRejected(t *testing.T) {
	p1 := NewFunctionNode("p1",
		func(agent.Context, any) (string, error) { return "p1", nil },
		defaultNodeConfig)
	p2 := NewFunctionNode("p2",
		func(agent.Context, any) (string, error) { return "p2", nil },
		defaultNodeConfig)
	join := NewJoinNode("join")
	handler := NewFunctionNode("handler",
		func(agent.Context, any) (string, error) { return "h", nil },
		defaultNodeConfig)

	// p1 -> join is CONDITIONAL (StringRoute). If the route never
	// matches, p1 never completes, and the join barrier — which waits
	// for BOTH p1 and p2 — can never release.
	edges := []Edge{
		{From: Start, To: p1},
		{From: Start, To: p2},
		{From: p1, To: join, Route: StringRoute("x")},
		{From: p2, To: join},
		{From: join, To: handler},
	}

	_, err := New("wf", edges)
	if err == nil {
		t.Errorf("New accepted a workflow with a conditional edge (p1 -> join) into JoinNode %q; "+
			"want a non-nil validation error rejecting conditional routing into a fan-in barrier", join.Name())
	}
}
