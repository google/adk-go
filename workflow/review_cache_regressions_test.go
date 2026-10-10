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
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

func TestSubScheduler_RehydrateCache_AncestorOutcomes(t *testing.T) {
	const parent = "root@1/c@1/mid@1"
	const leaf = parent + "/leaf@1"
	output := func(path string) *session.Event {
		return &session.Event{NodeInfo: &session.NodeInfo{Path: path}, Output: "result"}
	}
	marker := func(path, outcome string, failedPaths ...string) *session.Event {
		ev := &session.Event{NodeInfo: &session.NodeInfo{Path: path}}
		ev.CustomMetadata = map[string]any{"adk.workflow.node_outcome": outcome}
		if failedPaths != nil {
			paths := make([]any, len(failedPaths))
			for i, path := range failedPaths {
				paths[i] = path
			}
			ev.CustomMetadata[workflowFailedChildPathsKey] = paths
		}
		return ev
	}
	legacy := &session.Event{NodeInfo: &session.NodeInfo{Path: "root@1/c@1"}}
	legacy.ErrorCode = workflowNodeCancelledCode
	emptyPaths := marker("root@1/c@1", "cancelled")
	emptyPaths.CustomMetadata[workflowFailedChildPathsKey] = []any{}
	emptyFailure := marker(parent, "failed")
	emptyFailure.CustomMetadata[workflowFailedChildPathsKey] = []any{}
	nullPaths := marker(parent, "failed")
	nullPaths.CustomMetadata[workflowFailedChildPathsKey] = []any(nil)
	wrongType := marker(parent, "failed")
	wrongType.CustomMetadata[workflowFailedChildPathsKey] = "invalid"
	invalidPaths := marker("root@1/c@1", "cancelled")
	invalidPaths.CustomMetadata[workflowFailedChildPathsKey] = []any{"", false}
	parentOutput := output(parent)
	parentOutput.ErrorCode = "finish_reason"
	uncommitted := &session.Event{NodeInfo: &session.NodeInfo{Path: leaf}}
	uncommitted.CustomMetadata = map[string]any{workflowDelegatedOutputKey: "result"}
	partial := marker(parent, "cancelled")
	partial.Partial = true
	for _, tc := range []struct {
		name   string
		events sliceEvents
		want   map[string]any
	}{
		{name: "recorded_parent_path", events: sliceEvents{output(leaf), marker("root@1/c@1", "cancelled", parent)}, want: map[string]any{}},
		{name: "recorded_ancestor_path", events: sliceEvents{output(leaf), marker("root@1", "failed", "root@1/c@1")}, want: map[string]any{}},
		{name: "direct_parent_failure", events: sliceEvents{output(leaf), marker(parent, "failed")}, want: map[string]any{}},
		{name: "ancestor_unknown_cancel", events: sliceEvents{output(leaf), marker("root@1/c@1", "cancelled")}, want: map[string]any{}},
		{name: "ancestor_empty_paths", events: sliceEvents{output(leaf), emptyPaths}, want: map[string]any{leaf: "result"}},
		{name: "direct_parent_empty_failure", events: sliceEvents{output(leaf), emptyFailure}, want: map[string]any{leaf: "result"}},
		{name: "direct_parent_null_paths", events: sliceEvents{output(leaf), nullPaths}, want: map[string]any{}},
		{name: "direct_parent_invalid_type", events: sliceEvents{output(leaf), wrongType}, want: map[string]any{}},
		{name: "ancestor_invalid_paths", events: sliceEvents{output(leaf), invalidPaths}, want: map[string]any{}},
		{name: "legacy_ancestor_cancel", events: sliceEvents{output(leaf), legacy}, want: map[string]any{}},
		{name: "failed_descendant_subtree", events: sliceEvents{output(parent + "/branch@1/leaf@1"), output(parent + "/stable@1"), marker("root@1/c@1", "failed", parent+"/branch@1")}, want: map[string]any{parent + "/stable@1": "result"}},
		{name: "known_sibling_failure", events: sliceEvents{output(leaf), marker("root@1/c@1", "cancelled", "root@1/c@1/other@1")}, want: map[string]any{leaf: "result"}},
		{name: "path_boundary", events: sliceEvents{output(leaf), marker("root@1/c@10", "cancelled")}, want: map[string]any{leaf: "result"}},
		{name: "recorded_path_boundary", events: sliceEvents{output(leaf), marker("root@1/c@1", "cancelled", "root@1/c@1/mid@10")}, want: map[string]any{leaf: "result"}},
		{name: "partial_marker", events: sliceEvents{output(leaf), partial}, want: map[string]any{leaf: "result"}},
		{name: "parent_output_with_error_code", events: sliceEvents{output(leaf), parentOutput}, want: map[string]any{leaf: "result"}},
		{name: "unknown_outcome", events: sliceEvents{output(leaf), marker(parent, "progress")}, want: map[string]any{leaf: "result"}},
		{name: "uncommitted_delegation", events: sliceEvents{uncommitted}, want: map[string]any{}},
		{name: "later_success", events: sliceEvents{output(leaf), marker("root@1/c@1", "cancelled", parent), output(leaf)}, want: map[string]any{leaf: "result"}},
		{name: "child_failure", events: sliceEvents{output(leaf), marker(leaf, "failed")}, want: map[string]any{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, ev := range tc.events {
				ev.InvocationID = "test-invocation-id"
			}
			other := marker(parent, "cancelled")
			other.InvocationID = "different-invocation"
			tc.events = append(tc.events, other)
			// Exercise the persisted metadata representation, not just Go maps.
			data, err := json.Marshal(tc.events)
			if err != nil {
				t.Fatal("history serialization failed")
			}
			var persisted sliceEvents
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal("history deserialization failed")
			}
			for _, events := range []sliceEvents{tc.events, persisted} {
				ctx := newMockCtx(t)
				ctx.sess = &eventsSession{events: events}
				sub := newDynamicSubScheduler(agent.Promote(ctx), parent, noopEmit).(*dynamicSubScheduler)
				if !reflect.DeepEqual(sub.resultByPath, tc.want) {
					t.Fatal("cache did not respect outcome path ancestry")
				}
			}
		})
	}
}

func TestSubScheduler_FailureInventory(t *testing.T) {
	ctx := newMockCtx(t)
	sub := newDynamicSubScheduler(agent.Promote(ctx), "parent", noopEmit).(*dynamicSubScheduler)
	failure := errors.New("scripted failure")
	var calls atomic.Int32
	retry := NewFunctionNode("retry", func(agent.Context, any) (string, error) {
		if calls.Add(1) == 1 {
			return "", failure
		}
		return "ok", nil
	}, NodeConfig{})
	opts := runNodeOptions{customRunID: "stable"}
	if _, err := sub.runNode(retry, nil, opts); !errors.Is(err, failure) {
		t.Fatal("fixture did not fail")
	}
	if _, err := sub.runNode(retry, nil, opts); err != nil {
		t.Fatal("fixture did not recover")
	}
	failed := NewFunctionNode("failed", func(agent.Context, any) (string, error) { return "", failure }, NodeConfig{})
	_, failedErr := sub.runNode(failed, nil, opts)
	if !errors.Is(failedErr, failure) {
		t.Fatal("fixture did not fail its second child")
	}
	if _, hit := sub.awaitOrLead("parent/inflight@1"); hit {
		t.Fatal("fixture did not register an in-flight path")
	}
	inventory := sub.childFailures()
	node := newDummyNode("parent")
	wf := mustNew(t, []Edge{{From: Start, To: node}})
	sched := newScheduler(agent.Promote(ctx), wf.graph, 0)
	sched.state.Nodes["parent"] = &NodeState{Status: NodeFailed}
	ev := sched.completionEvent("parent", &nodeRun{nodePath: "parent"}, inventory)
	paths, ok := ev.CustomMetadata[workflowFailedChildPathsKey].([]any)
	if !ok || len(paths) != 2 {
		t.Fatal("inventory lost an uncertain path or retained duplicate/recovered paths")
	}
	set := map[any]bool{}
	for _, path := range paths {
		set[path] = true
	}
	if !set["parent/failed@stable"] || !set["parent/inflight@1"] {
		t.Fatal("inventory did not cover both failed and in-flight children")
	}
	sub.finishRun("parent/inflight@1", runResult{out: "ok"}, false, failureInventory{})
}

func TestWorkflowNode_DelegatedControlOwnership(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		name := "nonterminal"
		if terminal {
			name = "terminal"
		}
		t.Run(name, func(t *testing.T) {
			compute := NewFunctionNode("compute", func(agent.Context, any) (string, error) { return "42", nil }, NodeConfig{})
			parent, err := NewDynamicNodeWithSchema("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
				_, err := RunNode[string](ctx, compute, in, WithUseAsOutput())
				return nil, err
			}, nil, &jsonschema.Schema{Type: "integer"}, NodeConfig{})
			if err != nil {
				t.Fatal("fixture construction failed")
			}
			edges := []Edge{{From: Start, To: parent}}
			if !terminal {
				done := NewFunctionNode("done", func(agent.Context, any) (any, error) { return nil, nil }, NodeConfig{})
				edges = append(edges, Edge{From: parent, To: done})
			}
			node, err := NewWorkflowNode("sub", edges)
			if err != nil {
				t.Fatal("fixture construction failed")
			}
			var got any
			for ev, err := range node.Run(agent.Promote(newMockCtx(t)), nil) {
				if err != nil {
					t.Fatal("nested workflow failed")
				}
				if ev.Output != nil {
					got = ev.Output
				}
			}
			var want any
			if terminal {
				want = float64(42)
			}
			if got != want {
				t.Fatal("nested workflow confused delegated control ownership or schema projection")
			}
		})
	}
}

func TestResume_RejectedResponseConsumerExit(t *testing.T) {
	for trial := 0; trial < 20; trial++ {
		var started, ended atomic.Int32
		entered := make(chan struct{}, 10)
		var edges []Edge
		state := NewRunState()
		responses := map[string]any{"invalid": "wrong"}
		for i := 0; i < 10; i++ {
			name := fmt.Sprintf("valid%d", i)
			node := NewDynamicNode(name, func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				started.Add(1)
				defer ended.Add(1)
				entered <- struct{}{}
				return nil, emit(session.NewEvent(ctx, ctx.InvocationID()))
			}, NodeConfig{})
			edges = append(edges, Edge{From: Start, To: node})
			state.Nodes[name] = &NodeState{Status: NodePending, ResumedInputs: map[string]any{name: "ok"}}
			responses[name] = "ok"
		}
		invalid := NewDynamicNode("invalid", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
		edges = append(edges, Edge{From: Start, To: invalid})
		state.Nodes["invalid"] = &NodeState{Status: NodeFailed, ResumedInputs: map[string]any{"invalid": 1}, interruptSchemas: map[string]*jsonschema.Schema{"invalid": {Type: "integer"}}}
		wf := mustNew(t, edges)
		base, cancel := context.WithCancel(t.Context())
		ctx := newMockCtx(t)
		ctx.Context = base
		var rejected, scheduled bool
		for _, err := range wf.Resume(agent.Promote(ctx), state, responses) {
			if errors.Is(err, ErrInvalidResumeResponse) {
				rejected = true
				select {
				case <-entered:
					scheduled = true
				case <-time.After(50 * time.Millisecond):
					// Map iteration can encounter the invalid node first.
				}
				break
			}
		}
		if !rejected {
			cancel()
			t.Fatal("fixture did not reject the failed requester's response")
		}
		if !scheduled {
			cancel()
			continue
		}
		if ended.Load() != started.Load() {
			cancel()
			t.Fatal("consumer exit left scheduled producers blocked")
		}
		cancel()
		return
	}
	t.Fatal("fixture did not exercise a producer scheduled before rejection")
}

func TestSubScheduler_WaitForOutputHasNoFailureMarker(t *testing.T) {
	wait := true
	child := NewDynamicNode("wait", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{WaitForOutput: &wait})
	ctx := newMockCtx(t)
	var events sliceEvents
	sub := newDynamicSubScheduler(agent.Promote(ctx), "parent", func(ev *session.Event) error { events = append(events, ev); return nil }).(*dynamicSubScheduler)
	if _, err := sub.runNode(child, nil, runNodeOptions{}); !errors.Is(err, ErrNodeWaitingForOutput) {
		t.Fatal("fixture did not park waiting for output")
	}
	for _, ev := range events {
		if workflowNodeOutcome(ev) != "" {
			t.Fatal("waiting for output was recorded as a failure")
		}
	}
	inventory := sub.childFailures()
	if !inventory.known || inventory.paths == nil || len(inventory.paths) != 0 {
		t.Fatal("waiting for output contaminated the failure inventory")
	}
}

func TestSubScheduler_DelegatedControlOwnPath(t *testing.T) {
	for _, tc := range []struct {
		name       string
		descendant bool
		completed  bool
		want       any
	}{
		{name: "own_complete", completed: true, want: "value"},
		{name: "own_uncommitted"},
		{name: "descendant_complete", descendant: true, completed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			child := NewDynamicNode("child", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				path := ctx.Path()
				if tc.descendant {
					path += "/other@1"
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.NodeInfo = &session.NodeInfo{Path: path}
				ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: tc.completed, workflowDelegatedOutputKey: "value"}
				return nil, emit(ev)
			}, NodeConfig{})
			sub := newDynamicSubScheduler(agent.Promote(newMockCtx(t)), "parent", noopEmit).(*dynamicSubScheduler)
			out, err := sub.runNode(child, nil, runNodeOptions{})
			if err != nil || out != tc.want {
				t.Fatal("dynamic result did not respect delegated completion ownership")
			}
		})
	}
}

func TestWorkflow_DelegatedOutputSchemaRejected(t *testing.T) {
	child := NewFunctionNode("child", func(agent.Context, any) (string, error) { return "not a number", nil }, NodeConfig{})
	parent, err := NewDynamicNodeWithSchema("parent", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		_, err := RunNode[string](ctx, child, in, WithUseAsOutput())
		return nil, err
	}, nil, &jsonschema.Schema{Type: "integer"}, NodeConfig{})
	if err != nil {
		t.Fatal("fixture construction failed")
	}
	wf := mustNew(t, []Edge{{From: Start, To: parent}})
	var failed bool
	for _, err := range wf.Run(newMockCtx(t)) {
		failed = failed || errors.Is(err, ErrNodeFailed)
	}
	if !failed {
		t.Fatal("invalid delegated result bypassed the parent's output schema")
	}
}

func TestReconstructRunState_DelegatedCompletion(t *testing.T) {
	parent := NewDynamicNode("p", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: parent}})
	raised := &session.Event{Author: "p", LongRunningToolIDs: []string{"approval"}, NodeInfo: &session.NodeInfo{Path: "root@1/p@1"}}
	delegated := &session.Event{Output: "computed", NodeInfo: &session.NodeInfo{Path: "root@1/p@1/compute@1", OutputFor: []string{"root@1/p@1"}}}
	resolved := &session.Event{Author: "user"}
	resolved.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"result": "approved"}, genai.RoleUser)
	resolved.Content.Parts[0].FunctionResponse.ID = "approval"
	finished := &session.Event{Author: "p", NodeInfo: &session.NodeInfo{Path: "root@1/p@1"}}
	finished.CustomMetadata = map[string]any{workflowNodeCompletedKey: true, workflowDelegatedOutputKey: "computed"}
	failed := &session.Event{Author: "p", NodeInfo: &session.NodeInfo{Path: "root@1/p@1"}}
	failed.CustomMetadata = map[string]any{workflowNodeOutcomeKey: workflowNodeFailureOutcome}
	for _, tc := range []struct {
		name   string
		events sliceEvents
		status NodeStatus
		output any
		cached bool
	}{
		{name: "answer_invalidates_delegated_output", events: sliceEvents{raised, delegated, resolved}, status: NodePending},
		{name: "own_output_before_answer_is_not_finished", events: sliceEvents{raised, {Author: "p", Output: "old", NodeInfo: &session.NodeInfo{Path: "root@1/p@1"}}, resolved}, status: NodePending},
		{name: "delegation_after_answer_is_not_completion", events: sliceEvents{raised, resolved, delegated}, status: NodePending, cached: true},
		{name: "delegation_does_not_recover_failure", events: sliceEvents{raised, resolved, failed, delegated}, status: NodeFailed, cached: true},
		{name: "completed_with_delegation", events: sliceEvents{raised, resolved, delegated, finished}, status: NodeCompleted, output: "computed"},
		{name: "duplicate_answer_keeps_completion", events: sliceEvents{raised, resolved, delegated, finished, resolved}, status: NodeCompleted, output: "computed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.events)
			if err != nil {
				t.Fatal("history serialization failed")
			}
			var persisted sliceEvents
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal("history deserialization failed")
			}
			state, err := wf.ReconstructRunState(fakeSession{events: persisted}, "")
			if err != nil {
				t.Fatal("state reconstruction failed")
			}
			ns := nodeState(t, state, "p")
			if ns.Status != tc.status || ns.Output != tc.output {
				t.Fatal("delegation was confused with execution completion")
			}
			if tc.status == NodePending {
				outputs, _ := collectNodeOutputs(persisted, buildNodesByName(wf.graph), "", "root")
				if _, cached := outputs["p"]; cached != tc.cached {
					t.Fatal("output collection did not track the resume boundary")
				}
			}
		})
	}
}

func TestWorkflow_CancelledParallelNestedChildren(t *testing.T) {
	emittedA, emittedB, paused := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var attemptsA, attemptsB, failures atomic.Int32
	mkMid := func(name string, emitted chan struct{}, attempts *atomic.Int32) Node {
		leaf := NewDynamicNode("leaf", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			if attempts.Add(1) > 1 {
				return "retried-" + name, nil
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = "partial-" + name
			if err := emit(ev); err != nil {
				return nil, err
			}
			close(emitted)
			<-ctx.Done()
			return nil, nil
		}, NodeConfig{})
		return NewDynamicNode(name, func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
			return RunNode[any](ctx, leaf, in)
		}, NodeConfig{})
	}
	midA, midB := mkMid("midA", emittedA, &attemptsA), mkMid("midB", emittedB, &attemptsB)
	c := NewDynamicNode("c", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		type result struct {
			out any
			err error
		}
		a, b := make(chan result, 1), make(chan result, 1)
		go func() { out, err := RunNode[any](ctx, midA, in); a <- result{out, err} }()
		go func() { out, err := RunNode[any](ctx, midB, in); b <- result{out, err} }()
		ra, rb := <-a, <-b
		if ra.err != nil {
			return nil, ra.err
		}
		if rb.err != nil {
			return nil, rb.err
		}
		return map[string]any{"a": ra.out, "b": rb.out}, nil
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
		<-emittedA
		<-emittedB
		return nil, failure
	}, NodeConfig{})
	join := NewJoinNode("join")
	wf, err := New("root", []Edge{{From: Start, To: b}, {From: Start, To: c}, {From: Start, To: f}, {From: b, To: join}, {From: c, To: join}})
	if err != nil {
		t.Fatal("workflow execution failed")
	}
	ctx := newMockCtx(t)
	var history sliceEvents
	for ev, err := range wf.Run(ctx) {
		if err != nil {
			continue
		}
		history = append(history, ev)
		if ev.RequestedInput != nil {
			close(paused)
		}
	}
	ctx.sess = &eventsSession{events: history}
	state, err := wf.ReconstructRunState(ctx.sess, ctx.InvocationID())
	if err != nil {
		t.Fatal("workflow execution failed")
	}
	path := "root@1"
	resumeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path})
	for _, err := range wf.Resume(resumeCtx, state, map[string]any{"b": "approved"}) {
		if err != nil {
			t.Fatal("workflow execution failed")
		}
	}
	if attemptsA.Load() != 2 || attemptsB.Load() != 2 {
		t.Fatalf("leaf execution counts a=%d b=%d; both cancelled leaves must retry", attemptsA.Load(), attemptsB.Load())
	}
}

func TestReconstructRunState_DelegatedSchemaConversion(t *testing.T) {
	q := NewFunctionNode("q", func(agent.Context, any) (string, error) { return "42", nil }, NodeConfig{})
	p, err := NewDynamicNodeWithSchema("p", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		_, err := RunNode[string](ctx, q, in, WithUseAsOutput())
		return nil, err
	}, nil, &jsonschema.Schema{Type: "integer"}, NodeConfig{})
	if err != nil {
		t.Fatal("fixture construction failed")
	}
	b := NewDynamicNode("b", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "b", Message: "approve"})
	}, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: p}, {From: Start, To: b}})
	var events sliceEvents
	for ev, err := range wf.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatal("initial run failed")
		}
		events = append(events, ev)
	}
	state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
	if err != nil {
		t.Fatal("reconstruction failed")
	}
	want, err := p.ValidateOutput("42")
	if err != nil {
		t.Fatal("fixture validation failed")
	}
	ns := nodeState(t, state, "p")
	if !reflect.DeepEqual(ns.Output, want) {
		t.Fatalf("restored delegated output type %T, want validated type %T", ns.Output, want)
	}
}

func TestWorkflowNode_NilDelegationIsNotTerminalOutput(t *testing.T) {
	for _, oneOutput := range []bool{false, true} {
		t.Run(map[bool]string{false: "both_nil", true: "one_output"}[oneOutput], func(t *testing.T) {
			mk := func(name string, out any) Node {
				child := NewDynamicNode("child", func(agent.Context, any, func(*session.Event) error) (any, error) { return out, nil }, NodeConfig{})
				return NewDynamicNode(name, func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
					_, err := RunNode[any](ctx, child, in, WithUseAsOutput())
					return nil, err
				}, NodeConfig{})
			}
			a := mk("a", nil)
			var value any
			if oneOutput {
				value = "result"
			}
			b := mk("b", value)
			node, err := NewWorkflowNode("sub", []Edge{{From: Start, To: a}, {From: Start, To: b}})
			if err != nil {
				t.Fatal("fixture construction failed")
			}
			for _, err := range node.Run(agent.Promote(newMockCtx(t)), nil) {
				if err != nil {
					t.Fatal("nil delegation was counted as a terminal output")
				}
			}
		})
	}
}

func TestSubScheduler_NilDelegationWaitsForOutput(t *testing.T) {
	wait := true
	silent := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	child := NewDynamicNode("child", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		_, err := RunNode[any](ctx, silent, in, WithUseAsOutput())
		return nil, err
	}, NodeConfig{WaitForOutput: &wait})
	var history sliceEvents
	sub := newDynamicSubScheduler(agent.Promote(newMockCtx(t)), "parent", func(ev *session.Event) error { history = append(history, ev); return nil }).(*dynamicSubScheduler)
	_, err := sub.runNode(child, nil, runNodeOptions{})
	if !errors.Is(err, ErrNodeWaitingForOutput) {
		t.Fatal("nil delegation bypassed WaitForOutput")
	}
	for _, ev := range history {
		if ev.CustomMetadata[workflowNodeCompletedKey] == true {
			t.Fatal("nil delegation recorded completion for a parked child")
		}
	}
}

func TestSubScheduler_NilDelegationCompletedCache(t *testing.T) {
	var calls atomic.Int32
	silent := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	child := NewDynamicNode("child", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
		calls.Add(1)
		_, err := RunNode[any](ctx, silent, in, WithUseAsOutput())
		return nil, err
	}, NodeConfig{})
	ctx := newMockCtx(t)
	var history sliceEvents
	sub := newDynamicSubScheduler(agent.Promote(ctx), "parent", func(ev *session.Event) error { history = append(history, ev); return nil }).(*dynamicSubScheduler)
	if out, err := sub.runNode(child, nil, runNodeOptions{}); err != nil || out != nil {
		t.Fatal("nil delegation did not complete successfully")
	}
	ctx.sess = &eventsSession{events: history}
	resumed := newDynamicSubScheduler(agent.Promote(ctx), "parent", noopEmit).(*dynamicSubScheduler)
	if out, err := resumed.runNode(child, nil, runNodeOptions{}); err != nil || out != nil || calls.Load() != 1 {
		t.Fatal("proven nil completion was not replayed")
	}
}

func TestReconstructRunState_CancellationAnsweredLater(t *testing.T) {
	parent := NewDynamicNode("parent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: parent}})
	raised := &session.Event{Author: "parent", LongRunningToolIDs: []string{"approval"}}
	cancelled := &session.Event{Author: "parent"}
	cancelled.ErrorCode = workflowNodeCancelledCode
	resolved := &session.Event{Author: "user"}
	resolved.Content = genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"result": "approved"}, genai.RoleUser)
	resolved.Content.Parts[0].FunctionResponse.ID = "approval"
	state, err := wf.ReconstructRunState(fakeSession{events: sliceEvents{raised, cancelled, resolved}}, "")
	if err != nil || nodeState(t, state, "parent").Status != NodePending {
		t.Fatal("legacy cancellation overrode the later answer to an outstanding pause")
	}
}

func TestWorkflow_NilDelegationPreservesOwnOutput(t *testing.T) {
	for _, nested := range []bool{false, true} {
		t.Run(map[bool]string{false: "direct", true: "workflow_node"}[nested], func(t *testing.T) {
			silent := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
			p := NewDynamicNode("p", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "own-result"
				if err := emit(ev); err != nil {
					return nil, err
				}
				_, err := RunNode[any](ctx, silent, in, WithUseAsOutput())
				return nil, err
			}, NodeConfig{})
			producer := p
			if nested {
				var err error
				producer, err = NewWorkflowNode("sub", []Edge{{From: Start, To: p}})
				if err != nil {
					t.Fatal("fixture construction failed")
				}
			}
			var received any
			d := NewFunctionNode("d", func(_ agent.Context, in any) (any, error) { received = in; return nil, nil }, NodeConfig{})
			ask := NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
			}, NodeConfig{})
			wf := mustNew(t, []Edge{{From: Start, To: producer}, {From: producer, To: d}, {From: Start, To: ask}})
			var history sliceEvents
			for ev, err := range wf.Run(newMockCtx(t)) {
				if err != nil {
					t.Fatal("nil completion discarded a real output")
				}
				history = append(history, ev)
			}
			if received != "own-result" {
				t.Fatal("downstream did not receive the real output")
			}
			data, err := json.Marshal(history)
			if err != nil || json.Unmarshal(data, &history) != nil {
				t.Fatal("event round trip failed")
			}
			state, err := wf.ReconstructRunState(fakeSession{events: history}, "")
			if err != nil || nodeState(t, state, producer.Name()).Output != "own-result" {
				t.Fatal("history lost the real output after a nil completion")
			}
		})
	}
}

func TestSubScheduler_NilDelegationPreservesOwnOutput(t *testing.T) {
	var calls atomic.Int32
	silent := NewDynamicNode("silent", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
	child := NewDynamicNode("child", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
		calls.Add(1)
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "own-result"
		if err := emit(ev); err != nil {
			return nil, err
		}
		_, err := RunNode[any](ctx, silent, in, WithUseAsOutput())
		return nil, err
	}, NodeConfig{})
	ctx := newMockCtx(t)
	var history sliceEvents
	sub := newDynamicSubScheduler(agent.Promote(ctx), "parent", func(ev *session.Event) error { history = append(history, ev); return nil }).(*dynamicSubScheduler)
	if out, err := sub.runNode(child, nil, runNodeOptions{}); err != nil || out != "own-result" {
		t.Fatal("nil completion discarded the child's real output")
	}
	ctx.sess = &eventsSession{events: history}
	resumed := newDynamicSubScheduler(agent.Promote(ctx), "parent", noopEmit).(*dynamicSubScheduler)
	if out, err := resumed.runNode(child, nil, runNodeOptions{}); err != nil || out != "own-result" || calls.Load() != 1 {
		t.Fatal("nil completion overwrote the child's cached real output")
	}
}

func TestSubScheduler_NilCompletionControlIsNotOutput(t *testing.T) {
	wait := true
	child := NewDynamicNode("child", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: true, workflowDelegatedOutputKey: nil}
		return nil, emit(ev)
	}, NodeConfig{WaitForOutput: &wait})
	sub := newDynamicSubScheduler(agent.Promote(newMockCtx(t)), "parent", noopEmit).(*dynamicSubScheduler)
	_, err := sub.runNode(child, nil, runNodeOptions{})
	if !errors.Is(err, ErrNodeWaitingForOutput) {
		t.Fatal("nil completion was mistaken for a child output")
	}
}
