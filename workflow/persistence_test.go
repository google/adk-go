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
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// sliceEvents adapts a []*session.Event to session.Events for tests.
type sliceEvents []*session.Event

func (e sliceEvents) Len() int                { return len(e) }
func (e sliceEvents) At(i int) *session.Event { return e[i] }
func (e sliceEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

func TestCollectNodeOutputs_Ownership(t *testing.T) {
	nodes := map[string]Node{"p": newDummyNode("p"), "q": newDummyNode("q")}
	for _, tc := range []struct {
		name         string
		path         string
		outputFor    []string
		partial      bool
		workflowName string
		basePath     *string
		want         map[string]any
	}{
		{name: "own", path: "root@1/p@1", want: map[string]any{"p": "result"}},
		{name: "discarded_child", path: "root@1/p@1/q@1", outputFor: []string{"root@1/p@1/q@1"}, want: map[string]any{}},
		{name: "delegated_child", path: "root@1/p@1/q@1", outputFor: []string{"root@1/p@1/q@1", "root@1/p@1"}, want: map[string]any{"p": "result"}},
		{name: "legacy_author", want: map[string]any{"p": "result"}},
		{name: "partial", path: "root@1/p@1", partial: true, want: map[string]any{}},
		{name: "same_workflow_and_node_name", path: "p@1/p@1", workflowName: "p", want: map[string]any{"p": "result"}},
		{name: "unprefixed_dynamic_child", path: "p@1/q@1", workflowName: "p", basePath: new(string), want: map[string]any{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := &session.Event{Author: "p", Output: "result", NodeInfo: &session.NodeInfo{Path: tc.path, OutputFor: tc.outputFor}}
			ev.Partial = tc.partial
			if tc.basePath != nil {
				ev.CustomMetadata = map[string]any{workflowBasePathPrefix + tc.workflowName: *tc.basePath}
			}
			outputs, _ := collectNodeOutputs(sliceEvents{ev}, nodes, "", tc.workflowName)
			if !reflect.DeepEqual(outputs, tc.want) {
				t.Fatal("output was attributed to the wrong graph node")
			}
		})
	}
}

func TestReconstructRunState_WorkflowNameMatchesNode(t *testing.T) {
	ask := NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
	}, NodeConfig{})
	done := NewFunctionNode("done", func(agent.Context, any) (string, error) { return "result", nil }, NodeConfig{})
	wf, err := New("done", []Edge{{From: Start, To: ask}, {From: Start, To: done}})
	if err != nil {
		t.Fatal(err)
	}
	for _, prefixed := range []bool{false, true} {
		t.Run(map[bool]string{false: "RunNode", true: "Run"}[prefixed], func(t *testing.T) {
			ctx := newMockCtx(t)
			seq := wf.RunNode(agent.Promote(ctx), nil)
			if prefixed {
				seq = wf.Run(ctx)
			}
			var events sliceEvents
			for ev, err := range seq {
				if err != nil {
					t.Fatal(err)
				}
				events = append(events, ev)
			}
			state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
			if err != nil {
				t.Fatal(err)
			}
			if nodeState(t, state, "ask").Status != NodeWaiting || nodeState(t, state, "done").Output != "result" {
				t.Fatal("workflow name shadowed a graph node's event ownership")
			}
		})
	}
}

func TestReconstructRunState_LatestOutcome(t *testing.T) {
	ask, sibling := newDummyNode("ask"), newDummyNode("sibling")
	wf, err := New("root", []Edge{{From: Start, To: ask}, {From: Start, To: sibling}})
	if err != nil {
		t.Fatal(err)
	}
	output := func() *session.Event {
		return &session.Event{Author: "sibling", Output: "result", Branch: "root.sibling"}
	}
	failure := func() *session.Event {
		ev := &session.Event{Author: "sibling", Branch: "root.sibling"}
		ev.ErrorCode = workflowNodeFailureCode
		return ev
	}
	completed := func() *session.Event {
		ev := &session.Event{Author: "sibling", Branch: "root.sibling"}
		ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: true}
		return ev
	}
	for _, tc := range []struct {
		name   string
		events sliceEvents
		status NodeStatus
		output any
	}{
		{name: "output_then_failure", events: sliceEvents{output(), failure()}, status: NodeFailed},
		{name: "failure_then_result", events: sliceEvents{failure(), output()}, status: NodeCompleted, output: "result"},
		{name: "failure_then_nil_success", events: sliceEvents{failure(), completed()}, status: NodeCompleted},
		{name: "nil_success", events: sliceEvents{completed()}, status: NodeCompleted},
		{name: "later_nil_success_clears_old_output", events: sliceEvents{output(), completed()}, status: NodeCompleted},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := append(sliceEvents{&session.Event{Author: "ask", LongRunningToolIDs: []string{"approval"}}}, tc.events...)
			// Session services round-trip event metadata through JSON.
			data, err := json.Marshal(events)
			if err != nil {
				t.Fatal(err)
			}
			var stored sliceEvents
			if err := json.Unmarshal(data, &stored); err != nil {
				t.Fatal(err)
			}
			state, err := wf.ReconstructRunState(fakeSession{events: stored}, "")
			if err != nil {
				t.Fatal(err)
			}
			ns := nodeState(t, state, "sibling")
			if ns.Status != tc.status || ns.Output != tc.output || ns.Branch != "root.sibling" {
				t.Fatal("latest outcome was not reconstructed with its output and branch")
			}
			if tc.status == NodeFailed && state.completed["sibling"] {
				t.Fatal("failed node would suppress a later retry")
			}
		})
	}
}

func TestReconstructRunState_PreviouslyResumedReentry(t *testing.T) {
	c := NewFunctionNode("c", func(agent.Context, any) (string, error) { return "result", nil }, NodeConfig{RerunOnResume: new(true)})
	wf, err := New("root", []Edge{{From: Start, To: c}})
	if err != nil {
		t.Fatal(err)
	}
	raised := &session.Event{Author: "c", LongRunningToolIDs: []string{"approval"}}
	resolved := &session.Event{Author: "user"}
	resolved.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": "approved"}, genai.RoleUser)
	resolved.Content.Parts[0].FunctionResponse.ID = "approval"
	for _, outcome := range []string{"not_rerun_yet", "already_finished", "partial_result"} {
		t.Run(outcome, func(t *testing.T) {
			events := sliceEvents{raised, resolved}
			want := NodePending
			if outcome != "not_rerun_yet" {
				ev := &session.Event{Author: "c", Output: "result"}
				ev.Partial = outcome == "partial_result"
				events = append(events, ev)
				if !ev.Partial {
					want = NodeCompleted
				}
			}
			state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
			if err != nil {
				t.Fatal(err)
			}
			if nodeState(t, state, "c").Status != want {
				t.Fatal("previously resolved interrupt changed the latest node outcome")
			}
		})
	}
}

// A re-entry node with several incoming edges must rerun with the output
// of the predecessor that triggered it. The order the edges are declared
// in does not say which one that was.
func TestReconstructRunState_RerunInputFromTrigger(t *testing.T) {
	a, b, c := newDummyNode("a"), newDummyNode("b"), newDummyNode("c")
	ask := NewFunctionNode("ask", func(agent.Context, any) (any, error) { return nil, nil }, NodeConfig{RerunOnResume: new(true)})
	// The events are stamped the way the scheduler stamps them: the node
	// path on every event, and on an output event the emitter's own path
	// in OutputFor too.
	nodeEvent := func(node string) *session.Event {
		return &session.Event{NodeInfo: &session.NodeInfo{Path: node + "@1"}}
	}
	finished := func(node, output string, routes ...string) *session.Event {
		ev := nodeEvent(node)
		ev.Output, ev.Routes = output, routes
		ev.NodeInfo.OutputFor = []string{ev.NodeInfo.Path}
		return ev
	}
	routed := func(node string, routes ...string) *session.Event {
		ev := nodeEvent(node)
		ev.Routes = routes
		return ev
	}
	completed := func(node string) *session.Event {
		ev := nodeEvent(node)
		ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: true}
		return ev
	}
	failed := func(node string) *session.Event {
		ev := nodeEvent(node)
		ev.CustomMetadata = map[string]any{workflowNodeOutcomeKey: workflowNodeFailureOutcome}
		return ev
	}
	delegated := func(node string, output any) *session.Event {
		ev := completed(node)
		ev.CustomMetadata[workflowDelegatedOutputKey] = output
		return ev
	}
	raised := func(node, id string) *session.Event {
		ev := nodeEvent(node)
		ev.LongRunningToolIDs = []string{id}
		return ev
	}
	answered := func(id string) *session.Event {
		ev := &session.Event{Author: "user"}
		ev.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": "approved"}, genai.RoleUser)
		ev.Content.Parts[0].FunctionResponse.ID = id
		return ev
	}
	// under moves an event below base, the path of the workflow run.
	under := func(base string, ev *session.Event) *session.Event {
		ev.NodeInfo.Path = base + "/" + ev.NodeInfo.Path
		for i, path := range ev.NodeInfo.OutputFor {
			ev.NodeInfo.OutputFor[i] = base + "/" + path
		}
		return ev
	}
	withError := func(ev *session.Event) *session.Event {
		ev.ErrorCode = "SOME_ERROR"
		return ev
	}
	partial := func(ev *session.Event) *session.Event {
		ev.Partial = true
		return ev
	}
	earlier := func(ev *session.Event) *session.Event {
		ev.InvocationID = "earlier"
		return ev
	}
	// routedElsewhere is the graph of the reported case: a comes first
	// among the edges into ask but routes around it through b.
	routedElsewhere := []Edge{
		{From: Start, To: a},
		{From: a, To: ask, Route: StringRoute("go")},
		{From: a, To: b, Route: StringRoute("skip")},
		{From: b, To: ask},
	}
	// validateLoop sends a to b and back until a stops routing "retry".
	validateLoop := []Edge{
		{From: Start, To: a},
		{From: Start, To: ask},
		{From: a, To: b, Route: StringRoute("retry")},
		{From: b, To: a, Route: Default},
		{From: a, To: ask, Route: Default},
	}
	// secondVisit lets c send a through a second visit after a already
	// triggered ask.
	secondVisit := []Edge{
		{From: Start, To: a},
		{From: Start, To: c},
		{From: c, To: a, Route: Default},
		{From: Start, To: ask},
		{From: a, To: ask, Route: Default},
		{From: a, To: b, Route: StringRoute("skip")},
	}
	for _, tc := range []struct {
		name     string
		workflow string
		edges    []Edge
		history  []*session.Event
		// pause is what ask itself recorded. By default it raised
		// "approval" and got the answer.
		pause       []*session.Event
		wantStatus  NodeStatus
		wantTrigger string
		wantInput   any
	}{
		{
			name:        "predecessor_routed_elsewhere",
			edges:       routedElsewhere,
			history:     []*session.Event{finished("a", "A", "skip"), finished("b", "B")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// Both edges into ask have fired, a's on the first visit and
			// b's on the way back around the loop.
			name: "last_predecessor_to_finish",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: ask, To: b, Route: StringRoute("again")},
				{From: b, To: ask},
			},
			history:     []*session.Event{finished("a", "A", "go"), finished("b", "B")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			name: "predecessor_routed_to_node",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{finished("a", "A", "go")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			name: "default_route_not_taken",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history:     []*session.Event{finished("a", "A", "skip")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			name: "default_route_taken",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: b, To: ask, Route: StringRoute("go")},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
				{From: a, To: c},
			},
			history:     []*session.Event{finished("b", "B", "stop"), finished("a", "A", "other")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			name: "default_route_without_routing_event",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: b, To: ask, Route: StringRoute("go")},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history:     []*session.Event{finished("b", "B", "stop"), finished("a", "A")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// An edge with a concrete route does not fire for a node that
			// emitted no routing event at all.
			name: "concrete_route_without_routing_event",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{finished("a", "A")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// Finishing without output still fires the edge.
			name: "last_predecessor_finished_without_output",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: ask, To: b, Route: StringRoute("again")},
				{From: b, To: ask},
			},
			history:     []*session.Event{finished("a", "A", "go"), completed("b")},
			wantTrigger: "b",
			wantInput:   nil,
		},
		{
			// The comparison is by finish order. b is neither the first
			// nor the last edge declared.
			name: "finish_order_not_declaration_order",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: Start, To: c},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: b, To: ask},
				{From: c, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{finished("a", "A", "go"), finished("c", "C", "go"), finished("b", "B")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// a routed "retry" on its first visit and nothing on its
			// second. The old route must not hide the Default edge that
			// the second visit took.
			name:        "route_of_earlier_visit",
			edges:       validateLoop,
			history:     []*session.Event{finished("a", "A1", "retry"), finished("b", "B"), finished("a", "A2")},
			wantTrigger: "a",
			wantInput:   "A2",
		},
		{
			// The same loop with the route on its own event, ahead of
			// the output it belongs to.
			name:        "route_of_earlier_visit_on_its_own_event",
			edges:       validateLoop,
			history:     []*session.Event{routed("a", "retry"), finished("a", "A1"), finished("b", "B"), finished("a", "A2")},
			wantTrigger: "a",
			wantInput:   "A2",
		},
		{
			// The first visit took the Default edge into ask. The second
			// one routed back to b and ask has not been reached again.
			name:        "route_of_latest_visit",
			edges:       validateLoop,
			history:     []*session.Event{finished("a", "A1"), finished("b", "B"), routed("a", "retry"), finished("a", "A2")},
			wantTrigger: "a",
			wantInput:   "A1",
		},
		{
			// c recorded its output before a's first visit and triggered
			// the second one later. Two outputs are two visits.
			name:        "second_output_starts_a_visit",
			edges:       secondVisit,
			history:     []*session.Event{finished("c", "C"), finished("a", "A1", "skip"), finished("a", "A2")},
			wantTrigger: "a",
			wantInput:   "A2",
		},
		{
			// Same for two routing events.
			name:  "second_route_starts_a_visit",
			edges: secondVisit,
			history: []*session.Event{
				finished("c", "C"),
				routed("a", "other"), finished("a", "A1"),
				routed("a", "skip"), finished("a", "A2"),
			},
			wantTrigger: "a",
			wantInput:   "A1",
		},
		{
			// a sent itself around again and paused on the second visit.
			// The first visit completed and fired its edge into ask.
			name: "self_loop",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: c},
				{From: a, To: a, Route: StringRoute("again")},
				{From: a, To: ask},
				{From: c, To: ask, Route: Default},
			},
			history:     []*session.Event{finished("c", "C"), finished("a", "A", "again"), raised("a", "other")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// a keeps routing back to itself and never takes its Default
			// edge into ask, so ask was triggered over its START edge.
			name: "self_loop_never_taking_the_edge_to_node",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: a, Route: StringRoute("again")},
				{From: a, To: ask, Route: Default},
			},
			history:     []*session.Event{finished("a", "A1", "again"), finished("a", "A2", "again")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// A Default edge back to the node. The route and the output
			// of the second visit sit on one event and stay together.
			name: "self_loop_on_default_route",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: a, Route: Default},
				{From: a, To: ask, Route: StringRoute("done")},
			},
			history:     []*session.Event{finished("a", "A1"), finished("a", "A2", "done")},
			wantTrigger: "a",
			wantInput:   "A2",
		},
		{
			// Attempt 1 emitted a route and failed. The retry emitted
			// none and took the Default edge.
			name: "route_of_failed_attempt",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history:     []*session.Event{routed("a", "skip"), failed("a"), finished("a", "A")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// A node may emit its output before its routing event.
			name: "route_after_output",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{finished("a", "A"), routed("a", "go")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// A node that routed but recorded no result has not completed.
			name: "route_without_result",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{routed("a", "go")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// An error event that carries output is a result, as it is
			// for the rest of the scan.
			name: "output_on_error_event",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: StringRoute("go")},
			},
			history:     []*session.Event{withError(finished("a", "A", "go"))},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// The route of a dynamic child is not the route of the node
			// that ran it. The workflow shares its name with that node,
			// so the paths only resolve once the base is stripped.
			name:     "route_of_dynamic_child",
			workflow: "a",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history: []*session.Event{
				under("a@1/a@1", routed("child", "skip")),
				under("a@1", finished("a", "A")),
			},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// A dynamic node that hands its output over to a child
			// records the result on its completion record.
			name:        "delegated_output",
			edges:       routedElsewhere,
			history:     []*session.Event{finished("a", "A", "skip"), delegated("b", "B")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// With nothing delegated the scheduler adds its own record.
			// Both belong to one visit and the route applies to it.
			name: "delegated_nil_output",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: ask},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history:     []*session.Event{routed("a", "skip"), delegated("a", nil), completed("a")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// A route emitted in an earlier run of the same session says
			// nothing about the edges taken in this one.
			name: "routing_from_earlier_invocation",
			edges: []Edge{
				{From: Start, To: a},
				{From: b, To: ask},
				{From: a, To: ask, Route: Default},
				{From: a, To: b, Route: StringRoute("skip")},
			},
			history:     []*session.Event{earlier(routed("a", "skip")), finished("a", "A")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// A predecessor whose latest attempt failed did not trigger
			// anything, whatever it emitted before failing.
			name: "failed_predecessor",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: Start, To: ask},
			},
			history:     []*session.Event{finished("a", "A", "go"), failed("a")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// b emitted output and then paused on its own interrupt. A
			// waiting node schedules no successors, so c triggered ask.
			name: "paused_predecessor",
			edges: []Edge{
				{From: Start, To: c},
				{From: Start, To: b},
				{From: c, To: ask},
				{From: b, To: ask, Route: Default},
			},
			history:     []*session.Event{finished("c", "C"), finished("b", "B"), raised("b", "other")},
			wantTrigger: "c",
			wantInput:   "C",
		},
		{
			// Once b got its answer and finished, its edge did fire.
			name: "paused_predecessor_finished_after_answer",
			edges: []Edge{
				{From: Start, To: c},
				{From: Start, To: b},
				{From: c, To: ask},
				{From: b, To: ask, Route: Default},
			},
			history:     []*session.Event{finished("c", "C"), raised("b", "other"), answered("other"), finished("b", "B")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// Sending the same answer again changes nothing.
			name: "paused_predecessor_answered_twice",
			edges: []Edge{
				{From: Start, To: c},
				{From: Start, To: b},
				{From: c, To: ask},
				{From: b, To: ask, Route: Default},
			},
			history: []*session.Event{
				finished("c", "C"), raised("b", "other"), answered("other"),
				finished("b", "B"), answered("other"),
			},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// a asks under the same ID on every visit, like a node that
			// builds the ID from the invocation ID. The answer to the
			// second pause ends that pause too, and the rerun that
			// follows fires into ask.
			name:  "same_interrupt_id_on_later_visit",
			edges: secondVisit,
			history: []*session.Event{
				raised("a", "q"), answered("q"), finished("a", "A1"),
				completed("c"),
				raised("a", "q"), answered("q"), finished("a", "A2"),
			},
			wantTrigger: "a",
			wantInput:   "A2",
		},
		{
			// a triggered ask on its first visit. Its second visit paused
			// and fired nothing, but that does not undo the first one.
			name:        "predecessor_paused_on_later_visit",
			edges:       secondVisit,
			history:     []*session.Event{finished("a", "A1"), completed("c"), raised("a", "other")},
			wantTrigger: "a",
			wantInput:   "A1",
		},
		{
			// Same, with a second visit that failed.
			name:        "predecessor_failed_on_later_visit",
			edges:       secondVisit,
			history:     []*session.Event{finished("a", "A1"), finished("c", "C"), failed("a")},
			wantTrigger: "a",
			wantInput:   "A1",
		},
		{
			// The second visit of a was triggered by the answer to b's
			// pause, which b hands on without an output event.
			name: "predecessor_paused_on_visit_triggered_by_an_answer",
			edges: []Edge{
				{From: Start, To: a},
				{From: Start, To: b},
				{From: b, To: a, Route: Default},
				{From: Start, To: ask},
				{From: a, To: ask, Route: Default},
			},
			history:     []*session.Event{finished("a", "A"), raised("b", "other"), answered("other"), raised("a", "third")},
			wantTrigger: "a",
			wantInput:   "A",
		},
		{
			// c recorded its output early. The completion record of a
			// still closes the first visit, so the pause is a second one.
			name:        "completion_record_ends_the_visit",
			edges:       secondVisit,
			history:     []*session.Event{finished("c", "C"), completed("a"), raised("a", "other")},
			wantTrigger: "a",
			wantInput:   nil,
		},
		{
			// Same, with a second visit that routed away from ask.
			name:        "predecessor_routed_elsewhere_on_later_visit",
			edges:       secondVisit,
			history:     []*session.Event{finished("a", "A1"), finished("c", "C"), finished("a", "A2", "skip")},
			wantTrigger: "a",
			wantInput:   "A1",
		},
		{
			name: "partial_event",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: Start, To: ask},
			},
			history:     []*session.Event{finished("a", "A", "skip"), partial(finished("a", "A", "go"))},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			name: "only_start_edge_fired",
			edges: []Edge{
				{From: Start, To: a},
				{From: a, To: ask, Route: StringRoute("go")},
				{From: Start, To: ask},
			},
			history:     []*session.Event{finished("a", "A", "skip")},
			wantTrigger: Start.Name(),
			wantInput:   "seed",
		},
		{
			// One of two interrupts answered: ask reruns with the answer
			// it has and must get the same input again.
			name:        "partial_resume",
			edges:       routedElsewhere,
			history:     []*session.Event{finished("a", "A", "skip"), finished("b", "B")},
			pause:       []*session.Event{raised("ask", "approval"), raised("ask", "second"), answered("approval")},
			wantTrigger: "b",
			wantInput:   "B",
		},
		{
			// The rerun after the answer failed. A retry needs the input
			// as well.
			name:        "failed_after_answer",
			edges:       routedElsewhere,
			history:     []*session.Event{finished("a", "A", "skip"), finished("b", "B")},
			pause:       []*session.Event{raised("ask", "approval"), answered("approval"), failed("ask")},
			wantStatus:  NodeFailed,
			wantTrigger: "b",
			wantInput:   "B",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wf, err := New(tc.workflow, tc.edges)
			if err != nil {
				t.Fatal(err)
			}
			seed := &session.Event{Author: "user"}
			seed.Content = genai.NewContentFromText("seed", genai.RoleUser)
			pause := tc.pause
			if pause == nil {
				pause = []*session.Event{raised("ask", "approval"), answered("approval")}
			}
			events := append(sliceEvents{seed}, tc.history...)
			events = append(events, pause...)
			for _, ev := range events {
				if ev.InvocationID == "" {
					ev.InvocationID = "current"
				}
			}

			state, err := wf.ReconstructRunState(fakeSession{events: events}, "current")
			if err != nil {
				t.Fatal(err)
			}
			wantStatus := tc.wantStatus
			if wantStatus == 0 {
				wantStatus = NodePending
			}
			ns := nodeState(t, state, "ask")
			if ns.Status != wantStatus || ns.TriggeredBy != tc.wantTrigger || ns.Input != tc.wantInput {
				t.Errorf("ask = status %v, triggered by %q with input %v, want status %v, triggered by %q with input %v",
					ns.Status, ns.TriggeredBy, ns.Input, wantStatus, tc.wantTrigger, tc.wantInput)
			}
		})
	}
}

func TestResume_FailedSiblingRequiresMatchedInterrupt(t *testing.T) {
	var attempts atomic.Int32
	ask := newDummyNode("ask")
	failed := NewFunctionNode("failed", func(agent.Context, any) (string, error) {
		attempts.Add(1)
		return "retried", nil
	}, NodeConfig{})
	wf, err := New("root", []Edge{{From: Start, To: ask}, {From: Start, To: failed}})
	if err != nil {
		t.Fatal(err)
	}
	state := NewRunState()
	state.Nodes["ask"] = &NodeState{Status: NodeWaiting, Interrupts: []string{"approval"}}
	state.Nodes["failed"] = &NodeState{Status: NodeFailed}
	var runErr error
	for _, err := range wf.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"unknown": "response"}) {
		if err != nil {
			runErr = err
		}
	}
	if !errors.Is(runErr, ErrNothingToResume) || attempts.Load() != 0 {
		t.Fatal("unmatched response retried a failed sibling")
	}
}

func TestReconstructRunState_CancellationAfterInterrupt(t *testing.T) {
	ask := newDummyNode("ask")
	wf := mustNew(t, []Edge{{From: Start, To: ask}})
	raised := &session.Event{Author: "ask", LongRunningToolIDs: []string{"approval"}}
	resolved := &session.Event{Author: "user"}
	resolved.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"response": "approved"}, genai.RoleUser)
	resolved.Content.Parts[0].FunctionResponse.ID = "approval"
	cancelled := &session.Event{Author: "ask"}
	cancelled.ErrorCode = workflowNodeCancelledCode
	for _, answered := range []bool{false, true} {
		t.Run(map[bool]string{false: "pause_not_consumed", true: "reentry_cancelled"}[answered], func(t *testing.T) {
			events := sliceEvents{raised}
			want := NodeWaiting
			if answered {
				events = append(events, resolved)
				want = NodeFailed
			}
			events = append(events, cancelled)
			state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
			if err != nil {
				t.Fatal(err)
			}
			if nodeState(t, state, "ask").Status != want {
				t.Fatal("cancellation confused an outstanding pause with failed re-entry")
			}
		})
	}
}

func TestResume_FailedRequesterReceivesResponse(t *testing.T) {
	var called atomic.Bool
	ask := NewFunctionNode("ask", func(ctx agent.Context, _ any) (string, error) {
		called.Store(true)
		if value, ok := ctx.ResumedInput("approval"); !ok || value != "corrected" {
			t.Error("failed requester did not receive its response on retry")
		}
		return "result", nil
	}, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: ask}})
	state := NewRunState()
	state.Nodes["ask"] = &NodeState{Status: NodeFailed, ResumedInputs: map[string]any{"approval": "approved"}}
	for _, err := range wf.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"approval": "corrected"}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !called.Load() {
		t.Fatal("matched response did not retry the failed requester")
	}
}

func TestResume_InvalidFailedResponseDoesNotRetry(t *testing.T) {
	var called atomic.Bool
	ask := newDummyNode("ask")
	failed := NewFunctionNode("failed", func(agent.Context, any) (any, error) {
		called.Store(true)
		return nil, nil
	}, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: ask}, {From: Start, To: failed}})
	state := NewRunState()
	state.Nodes["ask"] = &NodeState{Status: NodeWaiting, Interrupts: []string{"ask"}}
	state.Nodes["failed"] = &NodeState{Status: NodeFailed, ResumedInputs: map[string]any{"approval": 1}, interruptSchemas: map[string]*jsonschema.Schema{"approval": {Type: "integer"}}}
	var rejected bool
	for _, err := range wf.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"ask": "approved", "approval": "invalid"}) {
		if errors.Is(err, ErrInvalidResumeResponse) {
			rejected = true
		}
	}
	if !rejected || called.Load() {
		t.Fatal("invalid corrected response retried a failed requester")
	}
}

func TestScheduler_NoOutputCompletionConsumerExit(t *testing.T) {
	n := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: n}})
	sawCompletion := false
	for ev, err := range wf.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatal(err)
		}
		sawCompletion = ev.CustomMetadata[workflowNodeCompletedKey] == true
		break
	}
	if !sawCompletion {
		t.Fatal("silent node did not produce a completion record")
	}
}

func TestScheduler_NoOutputActivationClearsPreviousResult(t *testing.T) {
	n := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: n}})
	s := newScheduler(agent.Promote(newMockCtx(t)), wf.graph, 0)
	s.state.Nodes["silent"] = &NodeState{Status: NodeCompleted, Output: "old result"}
	s.scheduleNode(n, nil, "", "")
	var sawCompletion bool
	s.run(func(ev *session.Event, err error) bool {
		if err != nil {
			t.Error(err)
		}
		if ev != nil && ev.CustomMetadata[workflowNodeCompletedKey] == true {
			sawCompletion = true
		}
		return true
	}, false)
	s.wg.Wait()
	if ns := s.state.Nodes["silent"]; !sawCompletion || ns.Status != NodeCompleted || ns.Output != nil {
		t.Fatal("silent activation retained a previous activation's result")
	}
}

func TestWorkflow_CancelledDynamicChildDoesNotReplayOutput(t *testing.T) {
	for _, tc := range []struct {
		name         string
		nested       bool
		contextError bool
	}{
		{name: "child"},
		{name: "nested_child", nested: true},
		{name: "nested_context_error", nested: true, contextError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			paused, childEmitted := make(chan struct{}), make(chan struct{})
			var attempts, failures, stableAttempts atomic.Int32
			stable := NewFunctionNode("stable", func(agent.Context, any) (string, error) {
				stableAttempts.Add(1)
				return "stable", nil
			}, NodeConfig{})
			q := NewDynamicNode("q", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				if attempts.Add(1) > 1 {
					return "retried", nil
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "partial"
				if err := emit(ev); err != nil {
					return nil, err
				}
				close(childEmitted)
				<-ctx.Done()
				return nil, nil
			}, NodeConfig{})
			if tc.nested {
				leaf := q
				q = NewDynamicNode("mid", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
					out, err := RunNode[any](ctx, leaf, in)
					if tc.contextError && ctx.Err() != nil {
						return nil, ctx.Err()
					}
					return out, err
				}, NodeConfig{})
			}
			c := NewDynamicNode("c", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
				if _, err := RunNode[string](ctx, stable, in); err != nil {
					return nil, err
				}
				return RunNode[any](ctx, q, in)
			}, NodeConfig{})
			b := NewDynamicNode("b", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "b", Message: "approve"})
			}, NodeConfig{})
			failure := errors.New("scripted failure")
			f := NewDynamicNode("f", func(agent.Context, any, func(*session.Event) error) (any, error) {
				if failures.Add(1) > 1 {
					return nil, nil
				}
				<-paused
				<-childEmitted
				return nil, failure
			}, NodeConfig{})
			join := NewJoinNode("join")
			wf, err := New("root", []Edge{{From: Start, To: b}, {From: Start, To: c}, {From: Start, To: f}, {From: b, To: join}, {From: c, To: join}})
			if err != nil {
				t.Fatal(err)
			}
			ctx := newMockCtx(t)
			var events sliceEvents
			var sawFailure bool
			for ev, err := range wf.Run(ctx) {
				if err != nil {
					sawFailure = errors.Is(err, failure)
					continue
				}
				events = append(events, ev)
				if ev.RequestedInput != nil {
					close(paused)
				}
			}
			if !sawFailure {
				t.Fatal("fixture did not cancel the dynamic child")
			}
			ctx.sess = &eventsSession{events: events}
			state, err := wf.ReconstructRunState(ctx.sess, ctx.InvocationID())
			if err != nil {
				t.Fatal(err)
			}
			resumeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: new("root@1")})
			var sawJoin bool
			for ev, err := range wf.Resume(resumeCtx, state, map[string]any{"b": "approved"}) {
				if err != nil {
					t.Fatal(err)
				}
				if out, ok := ev.Output.(map[string]any); ok {
					sawJoin = true
					if !reflect.DeepEqual(out, map[string]any{"b": "approved", "c": "retried"}) {
						t.Error("cancelled child output was replayed as a successful result")
					}
				}
			}
			if !sawJoin || attempts.Load() != 2 || stableAttempts.Load() != 1 {
				t.Fatal("cancelled child was not executed again on resume")
			}
		})
	}
}

// fakeSession backs ReconstructRunState, which reads only Events(); the
// rest are stubs.
type fakeSession struct {
	events session.Events
}

func (s fakeSession) ID() string                { return "test-session" }
func (s fakeSession) AppName() string           { return "test-app" }
func (s fakeSession) UserID() string            { return "test-user" }
func (s fakeSession) State() session.State      { return nil }
func (s fakeSession) Events() session.Events    { return s.events }
func (s fakeSession) LastUpdateTime() time.Time { return time.Time{} }

func modelEvent(path, text string, messageAsOutput bool) *session.Event {
	ev := &session.Event{
		NodeInfo: &session.NodeInfo{Path: path, MessageAsOutput: messageAsOutput},
	}
	ev.LLMResponse.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: text}},
	}
	return ev
}

// Resume derives output from the model message when an event is
// flagged MessageAsOutput with no explicit Output (adk-python parity).
func TestCollectNodeOutputs_MessageAsOutput(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", true)}

	outputs, completed := collectNodeOutputs(events, nodes, "", "")

	if got, want := outputs["talky"], "Hello, world!"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
	if !completed["talky"] {
		t.Errorf("completed[talky] = false, want true")
	}
}

func TestCollectNodeOutputs_MessageNotFlagged(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", false)}

	outputs, _ := collectNodeOutputs(events, nodes, "", "")

	if _, ok := outputs["talky"]; ok {
		t.Errorf("outputs[talky] = %#v, want absent", outputs["talky"])
	}
}

func TestCollectNodeOutputs_ExplicitOutputWins(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	ev := modelEvent("talky", "from message", true)
	ev.Output = "explicit"
	events := sliceEvents{ev}

	outputs, _ := collectNodeOutputs(events, nodes, "", "")

	if got, want := outputs["talky"], "explicit"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
}

// A delegated child's output is attributed on resume to the static
// owners of every path in OutputFor, so a delegating ancestor recovers
// it without re-emitting (adk-python output_for parity).
func TestCollectNodeOutputs_OutputForAttributesAncestors(t *testing.T) {
	nodes := map[string]Node{
		"child": newDummyNode("child"),
		"outer": newDummyNode("outer"),
	}

	ev := &session.Event{
		Output: "delegated",
		NodeInfo: &session.NodeInfo{
			Path:      "child/gc@1",
			OutputFor: []string{"child/gc@1", "child", "outer"},
		},
	}

	outputs, _ := collectNodeOutputs(sliceEvents{ev}, nodes, "", "")

	if got, want := outputs["child"], "delegated"; got != want {
		t.Errorf("outputs[child] = %#v, want %q", got, want)
	}
	if got, want := outputs["outer"], "delegated"; got != want {
		t.Errorf("outputs[outer] = %#v, want %q (ancestor not attributed)", got, want)
	}
}

func TestEventNodeName(t *testing.T) {
	nodes := map[string]Node{
		"nodeA":  newDummyNode("nodeA"),
		"parent": newDummyNode("parent"),
		"child":  newDummyNode("child"),
	}

	tests := []struct {
		name string
		ev   *session.Event
		want string
	}{
		{
			name: "nil NodeInfo falls back to Author",
			ev:   &session.Event{Author: "authorNode"},
			want: "authorNode",
		},
		{
			name: "empty Path falls back to Author",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: ""}},
			want: "authorNode",
		},
		{
			name: "static node path",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "nodeA"}},
			want: "nodeA",
		},
		{
			name: "node path with invocation ID",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "nodeA@1"}},
			want: "nodeA",
		},
		{
			name: "hierarchical path matches static parent",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "parent/child@1"}},
			want: "parent",
		},
		{
			name: "hierarchical path matches child when parent unknown",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "unknown/child@2"}},
			want: "child",
		},
		{
			name: "path segments not in nodes map falls back to Author",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "unknown/other@1"}},
			want: "authorNode",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := eventNodeName(tc.ev, nodes)
			if got != tc.want {
				t.Errorf("eventNodeName() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestReconstructRunState_InvocationScope verifies rehydration is scoped
// to one logical run: a stable interrupt ID resolved in an earlier
// invocation must not shadow the same ID freshly raised in the current
// invocation (the examples/workflow/hitl_simple re-run bug). It drives
// the public ReconstructRunState so the production resume path is
// exercised end to end, not just the internal scan helper.
func TestReconstructRunState_InvocationScope(t *testing.T) {
	ask := newDummyNode("ask")
	wf, err := New("hitl-rerun", []Edge{{From: Start, To: ask}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	raise := func(invID string) *session.Event {
		return &session.Event{
			Author:             "ask",
			InvocationID:       invID,
			LongRunningToolIDs: []string{"iid"},
		}
	}
	resolve := func(invID string) *session.Event {
		ev := &session.Event{Author: "user", InvocationID: invID}
		ev.Content = &genai.Content{Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "iid",
				Response: map[string]any{"payload": "first"},
			},
		}}}
		return ev
	}

	// Both runs reuse interrupt ID "iid": run 1 (inv1) raised and
	// resolved, run 2 (inv2) freshly raised.
	sess := fakeSession{events: sliceEvents{raise("inv1"), resolve("inv1"), raise("inv2")}}

	// Scanning inv2, the prior run's resolution is out of scope, so the
	// reused ID counts as unresolved and "ask" rehydrates as still waiting.
	state, err := wf.ReconstructRunState(sess, "inv2")
	if err != nil {
		t.Fatalf("ReconstructRunState(inv2) error = %v", err)
	}
	if ns := nodeState(t, state, "ask"); ns.Status != NodeWaiting ||
		len(ns.Interrupts) != 1 || ns.Interrupts[0] != "iid" {
		t.Errorf("inv2 ask = %+v, want NodeWaiting on interrupt [iid]", ns)
	}

	// inv1's own resolution is in scope, so the same node resolves —
	// proving the scope isolates runs rather than merely hiding history.
	state, err = wf.ReconstructRunState(sess, "inv1")
	if err != nil {
		t.Fatalf("ReconstructRunState(inv1) error = %v", err)
	}
	if ns := nodeState(t, state, "ask"); ns.Status != NodeCompleted || len(ns.Interrupts) != 0 {
		t.Errorf("inv1 ask = %+v, want NodeCompleted with no pending interrupts", ns)
	}
}

func nodeState(t *testing.T, state *RunState, name string) *NodeState {
	t.Helper()
	if state == nil {
		t.Fatalf("ReconstructRunState returned nil state, want node %q", name)
	}
	ns := state.Nodes[name]
	if ns == nil {
		t.Fatalf("no NodeState for %q; nodes = %+v", name, state.Nodes)
	}
	return ns
}

// A completed sibling must retain its output and branch for a join after
// another node pauses. Historical runs and unfinished siblings must not
// become completed predecessors of the current invocation.
func TestReconstructRunState_CompletedJoinPredecessors(t *testing.T) {
	ask, done, old, streaming := newDummyNode("ask"), newDummyNode("done"), newDummyNode("old"), newDummyNode("streaming")
	join := NewJoinNode("join")
	wf, err := New("root", []Edge{{From: Start, To: ask}, {From: Start, To: done}, {From: Start, To: old}, {From: Start, To: streaming}, {From: ask, To: join}, {From: done, To: join}})
	if err != nil {
		t.Fatal(err)
	}
	for _, messageOutput := range []bool{false, true} {
		t.Run(map[bool]string{false: "explicit_output", true: "message_output"}[messageOutput], func(t *testing.T) {
			completed := modelEvent("done", "saved", messageOutput)
			completed.InvocationID, completed.Branch = "current", "outer@1.done@1"
			if !messageOutput {
				completed.Output = "saved"
			}
			pending := &session.Event{Author: "ask", InvocationID: "current", Output: "intermediate", LongRunningToolIDs: []string{"approval"}}
			previous := &session.Event{Author: "old", InvocationID: "previous", Output: "obsolete"}
			unfinished := modelEvent("streaming", "partial", false)
			unfinished.InvocationID = "current"
			state, err := wf.ReconstructRunState(fakeSession{events: sliceEvents{previous, completed, unfinished, pending}}, "current")
			if err != nil {
				t.Fatal(err)
			}
			ns := nodeState(t, state, "done")
			if ns.Status != NodeCompleted || ns.Output != "saved" || ns.Branch != completed.Branch {
				t.Fatal("completed sibling lost its output, branch or completed status")
			}
			if nodeState(t, state, "ask").Status != NodeWaiting {
				t.Fatal("interrupted node was marked completed")
			}
			for _, name := range []string{"old", "streaming"} {
				if _, exists := state.Nodes[name]; exists {
					t.Fatalf("%s incorrectly restored as completed", name)
				}
			}
		})
	}
}
