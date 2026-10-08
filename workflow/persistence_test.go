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
	rerun := true
	c := NewFunctionNode("c", func(agent.Context, any) (string, error) { return "result", nil }, NodeConfig{RerunOnResume: &rerun})
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
			path := "root@1"
			resumeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path})
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
