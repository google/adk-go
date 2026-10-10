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
	"sync/atomic"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
)

// runNode queues its completion before ending its span. Waiting for this
// exporter puts an event-free completion ahead of Resume's cleanup without
// depending on goroutine timing or sleeping.
type cleanupCompletionExporter struct {
	completed chan struct{}
}

func (e *cleanupCompletionExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	for _, span := range spans {
		if span.Name() == "invoke_node valid" {
			e.completed <- struct{}{}
		}
	}
	return nil
}

func (*cleanupCompletionExporter) Shutdown(context.Context) error { return nil }

func TestResume_RejectedResponseCompletionDoesNotDispatch(t *testing.T) {
	for _, retry := range []bool{false, true} {
		t.Run(map[bool]string{false: "success_without_event", true: "retryable_error_without_event"}[retry], func(t *testing.T) {
			exporter := &cleanupCompletionExporter{completed: make(chan struct{}, 2)}
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
			telemetry.OverrideTracerForTesting(t, provider)
			// A caller-goroutine panic is not covered by runNode's recover.
			defer func() {
				if recover() != nil {
					t.Fatal("cleanup panicked on the caller goroutine")
				}
			}()
			for trial := 0; trial < 32; trial++ {
				var validCalls, successorCalls atomic.Int32
				cfg := NodeConfig{}
				if retry {
					cfg.RetryConfig = &RetryConfig{MaxAttempts: 2}
				}
				valid := NewDynamicNode("valid", func(agent.Context, any, func(*session.Event) error) (any, error) {
					if validCalls.Add(1) == 1 && retry {
						return nil, errors.New("scripted parent failure")
					}
					return nil, nil
				}, cfg)
				next := NewFunctionNode("next", func(agent.Context, any) (any, error) {
					successorCalls.Add(1)
					return nil, nil
				}, NodeConfig{})
				invalid := NewDynamicNode("invalid", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{})
				wf := mustNew(t, []Edge{{From: Start, To: valid}, {From: valid, To: next}, {From: Start, To: invalid}})
				newState := func() *RunState {
					state := NewRunState()
					state.Nodes["valid"] = &NodeState{Status: NodePending, ResumedInputs: map[string]any{"valid": "ok"}}
					state.Nodes["invalid"] = &NodeState{Status: NodeFailed, ResumedInputs: map[string]any{"invalid": 1}, interruptSchemas: map[string]*jsonschema.Schema{"invalid": {Type: "integer"}}}
					return state
				}
				state := newState()
				rejected, scheduled := false, false
				for _, err := range wf.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"valid": "ok", "invalid": "wrong"}) {
					if errors.Is(err, ErrInvalidResumeResponse) {
						rejected = true
						if state.Nodes["valid"].Status == NodeRunning {
							scheduled = true
							select {
							case <-exporter.completed:
							case <-t.Context().Done():
								t.Fatal("producer did not queue its completion")
							}
						}
						break
					}
				}
				if !rejected {
					t.Fatal("fixture did not reject the invalid response")
				}
				if !scheduled {
					// Map iteration can validate the invalid node before scheduling.
					continue
				}
				if validCalls.Load() != 1 || successorCalls.Load() != 0 {
					t.Fatal("cleanup dispatched a retry or successor after consumer exit")
				}
				// Reload the last persisted state: cleanup events were not consumed.
				for _, err := range wf.Resume(agent.Promote(newMockCtx(t)), newState(), map[string]any{"valid": "ok", "invalid": 42}) {
					if err != nil {
						t.Fatal("corrected response could not resume")
					}
				}
				if validCalls.Load() != 2 || successorCalls.Load() != 1 {
					t.Fatal("corrected response repeated or skipped the successor")
				}
				return
			}
			t.Fatal("fixture did not schedule a valid node before rejection")
		})
	}
}

func TestScheduler_InitialDrainDoesNotDispatch(t *testing.T) {
	for _, tc := range []struct {
		name    string
		failure error
		event   bool
	}{
		{name: "queued_success"},
		{name: "queued_retryable_failure", failure: errors.New("scripted parent failure")},
		{name: "queued_event_and_completion", event: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() != nil {
					t.Fatal("draining queued completion panicked")
				}
			}()
			var successorCalls atomic.Int32
			n := NewDynamicNode("valid", func(agent.Context, any, func(*session.Event) error) (any, error) { return nil, nil }, NodeConfig{RetryConfig: &RetryConfig{MaxAttempts: 2}})
			next := NewFunctionNode("next", func(agent.Context, any) (any, error) { successorCalls.Add(1); return nil, nil }, NodeConfig{})
			wf := mustNew(t, []Edge{{From: Start, To: n}, {From: n, To: next}})
			ctx := agent.Promote(newMockCtx(t))
			s := newScheduler(ctx, wf.graph, 0)
			s.state.Nodes["valid"] = &NodeState{Status: NodeRunning}
			s.runsByName["valid"] = &nodeRun{nodePath: "valid@1"}
			processed := make(chan struct{})
			if tc.event {
				s.eventQueue <- eventItem{nodeName: "valid", ev: session.NewEvent(ctx, ctx.InvocationID()), processed: processed}
			}
			s.eventQueue <- completionItem{nodeName: "valid", err: tc.failure}
			s.cancelAll()
			var yields int
			s.run(func(*session.Event, error) bool { yields++; return false }, true)
			s.wg.Wait()
			if yields != 0 || successorCalls.Load() != 0 || len(s.retryTimers) != 0 || len(s.runsByName) != 0 {
				t.Fatal("initial drain yielded or dispatched new work")
			}
			if tc.event {
				select {
				case <-processed:
				default:
					t.Fatal("initial drain did not release the producer handshake")
				}
			}
		})
	}
}

func TestScheduler_InitialDrainDoesNotFinalize(t *testing.T) {
	a := NewFunctionNode("a", func(agent.Context, any) (any, error) { return nil, nil }, NodeConfig{})
	b := NewFunctionNode("b", func(agent.Context, any) (any, error) { return nil, nil }, NodeConfig{})
	wf := mustNew(t, []Edge{{From: Start, To: a}, {From: Start, To: b}})
	s := newScheduler(agent.Promote(newMockCtx(t)), wf.graph, 0)
	s.state.Nodes["a"] = &NodeState{Status: NodeCompleted, Output: "a"}
	s.state.Nodes["b"] = &NodeState{Status: NodeCompleted, Output: "b"}
	s.cancelAll()
	s.run(func(*session.Event, error) bool { t.Fatal("initial drain finalized or yielded"); return false }, true)
}

func TestSubScheduler_EmptyFailureInventoryPersists(t *testing.T) {
	ctx := newTopLevelCtx(t)
	sub := newDynamicSubScheduler(ctx, "parent", noopEmit).(*dynamicSubScheduler)
	inventory := sub.childFailures()
	if !inventory.known || inventory.paths == nil || len(inventory.paths) != 0 {
		t.Fatal("known empty inventory was lost")
	}
	n := newDummyNode("parent")
	wf := mustNew(t, []Edge{{From: Start, To: n}})
	sched := newScheduler(ctx, wf.graph, 0)
	sched.state.Nodes["parent"] = &NodeState{Status: NodeFailed}
	ev := sched.completionEvent("parent", &nodeRun{nodePath: "parent"}, inventory)
	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatal("failure record serialization failed")
	}
	var persisted session.Event
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatal("failure record deserialization failed")
	}
	paths, ok := persisted.CustomMetadata[workflowFailedChildPathsKey].([]any)
	if !ok || paths == nil || len(paths) != 0 {
		t.Fatal("empty inventory was omitted or persisted as null")
	}
	unknown := sched.completionEvent("parent", &nodeRun{nodePath: "parent"}, failureInventory{})
	if _, ok := unknown.CustomMetadata[workflowFailedChildPathsKey]; ok {
		t.Fatal("unknown inventory was recorded as known")
	}
}

func TestSubScheduler_RecoveredNestedFailureInventory(t *testing.T) {
	const mid = "parent/mid@1"
	const leaf = mid + "/leaf@1"
	const otherLeaf = mid + "/other@1"
	const sibling = "parent/mid@10/leaf@1"
	for _, tc := range []struct {
		name  string
		paths []string
	}{
		{name: "recovered"},
		{name: "known_empty", paths: []string{}},
		{name: "new_failure", paths: []string{otherLeaf}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sub := newDynamicSubScheduler(newTopLevelCtx(t), "parent", noopEmit).(*dynamicSubScheduler)
			sub.failedPaths[sibling] = struct{}{}
			failure := errors.New("scripted child failure")
			sub.finishRun(mid, runResult{err: failure}, true, failureInventory{known: true, paths: []string{leaf}})
			if _, ok := sub.failedPaths[leaf]; !ok || len(sub.failedPaths) != 2 {
				t.Fatal("nested inventory was widened or lost")
			}
			res := runResult{out: "recovered"}
			var inventory failureInventory
			if tc.paths != nil {
				res.err = failure
				inventory = failureInventory{known: true, paths: tc.paths}
			}
			sub.finishRun(mid, res, true, inventory)
			if _, ok := sub.failedPaths[sibling]; !ok || len(sub.failedPaths) != 1+len(tc.paths) {
				t.Fatal("new child inventory retained stale failures or cleared another subtree")
			}
			for _, path := range tc.paths {
				if _, ok := sub.failedPaths[path]; !ok {
					t.Fatal("new child inventory omitted a failed path")
				}
			}
		})
	}
}

func TestSubScheduler_UnrecordedChildFailureRemainsConservative(t *testing.T) {
	for _, recorded := range []bool{true, false} {
		t.Run(map[bool]string{true: "recorded_inventory", false: "rejected_marker"}[recorded], func(t *testing.T) {
			const parent = "parent"
			const midPath = parent + "/mid@stable"
			const chargePath = midPath + "/charge@charge"
			ctx := newMockCtx(t)
			ctx.sess = &eventsSession{events: sliceEvents{}}
			var history sliceEvents
			sub := newDynamicSubScheduler(agent.Promote(ctx), parent, func(ev *session.Event) error {
				if workflowNodeOutcome(ev) != "" && ev.NodeInfo.Path == midPath && !recorded {
					return errors.New("scripted marker rejection")
				}
				history = append(history, ev)
				return nil
			}).(*dynamicSubScheduler)
			charge := NewFunctionNode("charge", func(agent.Context, any) (string, error) { return "charged", nil }, NodeConfig{})
			failure := errors.New("scripted nested body failure")
			mid := NewDynamicNode("mid", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
				if _, err := RunNode[string](ctx, charge, in, WithRunID("charge")); err != nil {
					return nil, err
				}
				partial := session.NewEvent(ctx, ctx.InvocationID())
				partial.Output = "partial"
				partial.NodeInfo = &session.NodeInfo{Path: ctx.Path()}
				if err := emit(partial); err != nil {
					return nil, err
				}
				return nil, failure
			}, NodeConfig{})
			_, err := sub.runNode(mid, nil, runNodeOptions{customRunID: "stable"})
			if !errors.Is(err, failure) {
				t.Fatal("fixture did not preserve the nested failure cause")
			}
			n := newDummyNode(parent)
			wf := mustNew(t, []Edge{{From: Start, To: n}})
			sched := newScheduler(agent.Promote(ctx), wf.graph, 0)
			sched.state.Nodes[parent] = &NodeState{Status: NodeFailed}
			history = append(history, sched.completionEvent(parent, &nodeRun{nodePath: parent}, sub.childFailures()))
			data, err := json.Marshal(history)
			if err != nil {
				t.Fatal("history serialization failed")
			}
			var persisted sliceEvents
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal("history deserialization failed")
			}
			ctx.sess = &eventsSession{events: persisted}
			rebuilt := newDynamicSubScheduler(agent.Promote(ctx), parent, noopEmit).(*dynamicSubScheduler)
			if _, ok := rebuilt.resultByPath[midPath]; ok {
				t.Fatal("failed child's own partial output survived")
			}
			if _, ok := rebuilt.resultByPath[chargePath]; ok != recorded {
				t.Fatal("unknown failure scope was trusted or known child completion was discarded")
			}
		})
	}
}
