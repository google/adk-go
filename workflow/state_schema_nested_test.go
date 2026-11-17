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

package workflow_test

import (
	"errors"
	"iter"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

// itemNode is a custom node whose Run receives the node's input.
type itemNode struct {
	workflow.BaseNode
	run func(ctx agent.Context, in any, yield func(*session.Event, error) bool)
}

func newItemNode(name string, run func(ctx agent.Context, in any, yield func(*session.Event, error) bool)) *itemNode {
	return &itemNode{BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{}), run: run}
}

func (n *itemNode) Run(ctx agent.Context, in any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, in, yield)
	}
}

// orchestrator returns a dynamic node "orch" that writes name = parentName,
// runs child, and returns "done", or the child's error.
func orchestrator(parentName string, child workflow.Node) workflow.Node {
	return workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if err := ctx.State().Set("name", parentName); err != nil {
			return "", err
		}
		if _, err := workflow.RunNode[any](ctx, child, nil); err != nil {
			return "", err
		}
		return "done", nil
	}, workflow.NodeConfig{})
}

func newParallelWorker(t *testing.T, wrapped workflow.Node, maxConcurrency int) workflow.Node {
	t.Helper()
	pw, err := workflow.NewParallelWorker("pw", wrapped, maxConcurrency, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewParallelWorker() error = %v", err)
	}
	return pw
}

// newItemsWorkflow runs a parallel worker over items under schema S.
func newItemsWorkflow(t *testing.T, worker workflow.Node, items []any) *workflow.Workflow {
	t.Helper()
	seed := workflow.NewFunctionNode("seed", func(agent.Context, any) ([]any, error) {
		return items, nil
	}, workflow.NodeConfig{})
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, seed, worker), workflow.WithStateSchema(stateSchemaS(t)))
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}
	return wf
}

func pathOf(ev *session.Event) string {
	if ev.NodeInfo == nil {
		return ""
	}
	return ev.NodeInfo.Path
}

func TestDynamicChild_WritesValidatedAndAttributed(t *testing.T) {
	t.Run("accepted write is the child's", func(t *testing.T) {
		child := workflow.NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			return "c", ctx.State().Set("counter", 3)
		}, workflow.NodeConfig{})
		wf := newStateWorkflow(t, orchestrator("parent", child), workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["counter"] != 3 || stored["name"] != "parent" {
			t.Errorf("stored counter = %v and name = %v, want 3 and parent", stored["counter"], stored["name"])
		}
		withCounter := eventsWithKey(events, "counter")
		if len(withCounter) != 1 {
			t.Fatalf("%d caller-visible events carry counter, want 1", len(withCounter))
		}
		if got := pathOf(withCounter[0]); got != "wf@1/orch@1/child@1" {
			t.Errorf("counter event path = %q, want wf@1/orch@1/child@1", got)
		}
		if diff := cmp.Diff(map[string]any{"counter": 3}, withCounter[0].Actions.StateDelta); diff != "" {
			t.Errorf("child event StateDelta mismatch (-want +got):\n%s", diff)
		}
		withName := eventsWithKey(events, "name")
		if len(withName) != 1 {
			t.Fatalf("%d caller-visible events carry name, want 1", len(withName))
		}
		if got := pathOf(withName[0]); got != "wf@1/orch@1" {
			t.Errorf("name event path = %q, want wf@1/orch@1", got)
		}
	})
	t.Run("refused write fails the orchestrator", func(t *testing.T) {
		child := workflow.NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			if err := ctx.State().Set("counter", 3); err != nil {
				return "", err
			}
			return "", ctx.State().Set("unknown", 1)
		}, workflow.NodeConfig{})
		wf := newStateWorkflow(t, orchestrator("parent", child), workflow.WithStateSchema(stateSchemaS(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)

		if se := asStateSchemaError(t, err); se.Key != "unknown" {
			t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
		}
		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("run error = %v, want it to wrap ErrNodeFailed", err)
		}
		if stored["counter"] != 3 {
			t.Errorf("stored counter = %v, want 3", stored["counter"])
		}
		if _, ok := stored["name"]; ok {
			t.Errorf("stored name = %v, want it absent", stored["name"])
		}
	})
	t.Run("refused event delta is not emitted", func(t *testing.T) {
		child := newStateNode("child", func(ctx agent.Context, yield func(*session.Event, error) bool) {
			yield(outputEvent(ctx, "c", map[string]any{"arbitrary_key": 1}), nil)
		})
		wf := newStateWorkflow(t, orchestrator("parent", child), workflow.WithStateSchema(stateSchemaS(t)))

		events, stored, err := runStateWorkflow(t, wf, nil)

		if se := asStateSchemaError(t, err); se.Key != "arbitrary_key" {
			t.Errorf("StateSchemaError.Key = %q, want arbitrary_key", se.Key)
		}
		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("run error = %v, want it to wrap ErrNodeFailed", err)
		}
		if got := eventsWithKey(events, "arbitrary_key"); len(got) != 0 {
			t.Errorf("%d caller-visible events carry arbitrary_key, want 0", len(got))
		}
		if _, ok := stored["arbitrary_key"]; ok {
			t.Errorf("stored state has arbitrary_key")
		}
	})
}

func TestDynamicChild_TrailingEventPath(t *testing.T) {
	child := newStateNode("child", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("counter", 2); err != nil {
			yield(nil, err)
		}
	})
	wf := newStateWorkflow(t, orchestrator("p", child), workflow.WithStateSchema(stateSchemaS(t)))

	events, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if stored["counter"] != 2 || stored["name"] != "p" {
		t.Errorf("stored counter = %v and name = %v, want 2 and p", stored["counter"], stored["name"])
	}
	withCounter := eventsWithKey(events, "counter")
	if len(withCounter) != 1 {
		t.Fatalf("%d caller-visible events carry counter, want 1", len(withCounter))
	}
	if got := pathOf(withCounter[0]); got != "wf@1/orch@1/child@1" {
		t.Errorf("child's trailing event path = %q, want wf@1/orch@1/child@1", got)
	}
	if diff := cmp.Diff(map[string]any{"counter": 2}, withCounter[0].Actions.StateDelta); diff != "" {
		t.Errorf("child's trailing event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"name": "p"}, eventWithOutput(t, events, "done").Actions.StateDelta); diff != "" {
		t.Errorf("orchestrator's terminal event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestDynamicChild_FailureDropsOrchestratorPendingKeepsChild(t *testing.T) {
	errBoom := errors.New("boom")
	child := workflow.NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		if err := ctx.State().Set("name", "c"); err != nil {
			return "", err
		}
		return "", errBoom
	}, workflow.NodeConfig{})
	orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if err := ctx.State().Set("counter", 1); err != nil {
			return "", err
		}
		return workflow.RunNode[string](ctx, child, nil)
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, orch, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	if stored["name"] != "c" {
		t.Errorf("stored name = %v, want c", stored["name"])
	}
	if _, ok := stored["counter"]; ok {
		t.Errorf("stored counter = %v, want it absent", stored["counter"])
	}
}

// A dynamic child run with an isolation scope sends its trailing state-only
// event in that scope.
func TestDynamicChild_TrailingEventCarriesIsolationScope(t *testing.T) {
	child := newStateNode("child", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("counter", 3); err != nil {
			yield(nil, err)
		}
	})
	orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if _, err := workflow.RunNode[any](ctx, child, nil, workflow.WithIsolationScope("s1")); err != nil {
			return "", err
		}
		return "done", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, orch, workflow.WithStateSchema(stateSchemaS(t)))

	events, _, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	withCounter := eventsWithKey(events, "counter")
	if len(withCounter) != 1 {
		t.Fatalf("%d caller-visible events carry counter, want 1", len(withCounter))
	}
	if got := withCounter[0].IsolationScope; got != "s1" {
		t.Errorf("child's trailing event IsolationScope = %q, want s1", got)
	}
}

// A dynamic child that fails because its own dynamic child failed sends none
// of its pending writes.
func TestDynamicChild_GrandchildFailureDropsChildPending(t *testing.T) {
	errBoom := errors.New("boom")
	grandchild := workflow.NewFunctionNode("grandchild", func(agent.Context, any) (string, error) {
		return "", errBoom
	}, workflow.NodeConfig{})
	child := workflow.NewDynamicNode("child", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if err := ctx.State().Set("name", "c"); err != nil {
			return "", err
		}
		return workflow.RunNode[string](ctx, grandchild, nil)
	}, workflow.NodeConfig{})
	orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return workflow.RunNode[string](ctx, child, nil)
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, orch, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	if _, ok := stored["name"]; ok {
		t.Errorf("stored name = %v, want it absent", stored["name"])
	}
}

func TestParallelWorker_ItemWrites(t *testing.T) {
	var arrived, othersDone sync.WaitGroup
	arrived.Add(3)
	othersDone.Add(2)
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		i := in.(int)
		arrived.Done()
		arrived.Wait()
		if i != 2 {
			if err := ctx.State().Set("item:"+strconv.Itoa(i), i); err != nil {
				yield(nil, err)
				return
			}
		}
		switch i {
		case 0:
			// Item 0 finishes last, so a merge in finish order lets its
			// counter win over item 1's.
			othersDone.Wait()
			time.Sleep(20 * time.Millisecond)
			if err := ctx.State().Set("counter", 1); err != nil {
				yield(nil, err)
				return
			}
			yield(outputEvent(ctx, i, nil), nil)
		case 1:
			defer othersDone.Done()
			yield(outputEvent(ctx, i, map[string]any{"counter": 4}), nil)
		default:
			defer othersDone.Done()
			if !yield(outputEvent(ctx, i, nil), nil) {
				return
			}
			// A write after the item's last event.
			if err := ctx.State().Set("item:2", i); err != nil {
				yield(nil, err)
			}
		}
	})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0, 1, 2})

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	want := map[string]any{"item:0": 0, "item:1": 1, "item:2": 2, "counter": 4}
	if diff := cmp.Diff(want, stored); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

func TestParallelWorker_ItemViolationFailsRun(t *testing.T) {
	item := workflow.NewFunctionNode("item", func(ctx agent.Context, in int) (int, error) {
		if in == 1 {
			return 0, ctx.State().Set("unknown", 1)
		}
		return in, nil
	}, workflow.NodeConfig{})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0, 1, 2})

	_, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if _, ok := stored["unknown"]; ok {
		t.Errorf("stored state has unknown")
	}
}

func TestParallelWorker_ItemEventDeltaValidated(t *testing.T) {
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		yield(outputEvent(ctx, in, map[string]any{"unknown": 1}), nil)
	})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0})

	_, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if _, ok := stored["unknown"]; ok {
		t.Errorf("stored state has unknown")
	}
}

func TestParallelWorker_ItemEventDeltaCarried(t *testing.T) {
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		i := in.(int)
		yield(outputEvent(ctx, i, map[string]any{"item:" + strconv.Itoa(i): i}), nil)
	})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0, 1})

	events, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if stored["item:0"] != 0 || stored["item:1"] != 1 {
		t.Errorf("stored item:0 = %v and item:1 = %v, want 0 and 1", stored["item:0"], stored["item:1"])
	}
	var aggregated *session.Event
	for _, ev := range events {
		if _, ok := ev.Output.([]any); ok {
			aggregated = ev
		}
	}
	if aggregated == nil {
		t.Fatal("no caller-visible event has the worker's aggregated output")
	}
	if diff := cmp.Diff(map[string]any{"item:0": 0, "item:1": 1}, aggregated.Actions.StateDelta); diff != "" {
		t.Errorf("aggregated event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestParallelWorker_FailureKeepsItemWrites(t *testing.T) {
	errBoom := errors.New("boom")
	failingChild := workflow.NewFunctionNode("child", func(agent.Context, any) (string, error) {
		return "", errBoom
	}, workflow.NodeConfig{})
	tests := []struct {
		name    string
		wrapped workflow.Node
	}{
		{
			name: "item returns an error",
			wrapped: workflow.NewFunctionNode("item", func(ctx agent.Context, in int) (int, error) {
				if in == 1 {
					return 0, errBoom
				}
				return in, ctx.State().Set("counter", 1)
			}, workflow.NodeConfig{}),
		},
		{
			name: "item's dynamic child fails",
			wrapped: workflow.NewDynamicNode("item", func(ctx agent.Context, in int, _ func(*session.Event) error) (int, error) {
				if in == 1 {
					_, err := workflow.RunNode[string](ctx, failingChild, nil)
					return 0, err
				}
				return in, ctx.State().Set("counter", 1)
			}, workflow.NodeConfig{}),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// One item at a time, so item 0 has finished before item 1 fails.
			wf := newItemsWorkflow(t, newParallelWorker(t, tc.wrapped, 1), []any{0, 1})

			_, stored, err := runStateWorkflow(t, wf, nil)

			if !errors.Is(err, errBoom) {
				t.Errorf("run error = %v, want errBoom", err)
			}
			if stored["counter"] != 1 {
				t.Errorf("stored counter = %v, want 1", stored["counter"])
			}
		})
	}
}

// An item that fails after it wrote keeps its write.
func TestParallelWorker_FailedItemKeepsOwnWrites(t *testing.T) {
	errBoom := errors.New("boom")
	item := newItemNode("item", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("counter", 1); err != nil {
			yield(nil, err)
			return
		}
		yield(nil, errBoom)
	})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0})

	_, stored, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	if stored["counter"] != 1 {
		t.Errorf("stored counter = %v, want 1", stored["counter"])
	}
}

// A failed worker sends the artifact delta of an item that finished before
// the failure, even when the item wrote no state.
func TestParallelWorker_FailureKeepsItemArtifactDelta(t *testing.T) {
	errBoom := errors.New("boom")
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in == 1 {
			yield(nil, errBoom)
			return
		}
		if err := agent.MergeArtifactDelta(ctx, map[string]int64{"a0.txt": 1}); err != nil {
			yield(nil, err)
			return
		}
		yield(outputEvent(ctx, in, nil), nil)
	})
	// One item at a time, so item 0 has finished before item 1 fails.
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 1), []any{0, 1})

	events, _, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	carried := eventsWithArtifact(events, "a0.txt")
	if len(carried) != 1 {
		t.Fatalf("%d caller-visible events carry artifact a0.txt, want 1", len(carried))
	}
	if got := carried[0].Actions.ArtifactDelta["a0.txt"]; got != 1 {
		t.Errorf("ArtifactDelta[a0.txt] = %d, want 1", got)
	}
}

// An item that is retried keeps the writes of its failed attempt as well as
// those of the attempt that succeeds.
func TestParallelWorker_RetriedItemKeepsEveryAttemptsWrites(t *testing.T) {
	errRetry := errors.New("retry me")
	var attempts int
	item := newItemNode("item", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		attempts++
		if attempts == 1 {
			if err := ctx.State().Set("counter", 1); err != nil {
				yield(nil, err)
				return
			}
			yield(nil, errRetry)
			return
		}
		if err := ctx.State().Set("name", "x"); err != nil {
			yield(nil, err)
			return
		}
		yield(outputEvent(ctx, "ok", nil), nil)
	})
	pw, err := workflow.NewParallelWorker("pw", item, 0, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{
		MaxAttempts:  2,
		InitialDelay: time.Millisecond,
		ShouldRetry:  func(err error) bool { return errors.Is(err, errRetry) },
	}})
	if err != nil {
		t.Fatalf("NewParallelWorker() error = %v", err)
	}
	wf := newItemsWorkflow(t, pw, []any{0})

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if stored["counter"] != 1 || stored["name"] != "x" {
		t.Errorf("stored counter = %v and name = %v, want 1 and x", stored["counter"], stored["name"])
	}
}

// An item parked in its retry delay when a sibling fails keeps the writes of
// the attempt that failed.
func TestParallelWorker_ItemWaitingToRetryKeepsWritesOnSiblingFailure(t *testing.T) {
	errBoom := errors.New("boom")
	errRetry := errors.New("retry me")
	written := make(chan struct{})
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in.(int) == 1 {
			<-written
			yield(nil, errBoom)
			return
		}
		if err := ctx.State().Set("counter", 1); err != nil {
			yield(nil, err)
			return
		}
		close(written)
		yield(nil, errRetry)
	})
	// The retry delay is long enough that only the sibling's failure ends it.
	pw, err := workflow.NewParallelWorker("pw", item, 0, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{
		MaxAttempts:  2,
		InitialDelay: time.Hour,
		ShouldRetry:  func(err error) bool { return errors.Is(err, errRetry) },
	}})
	if err != nil {
		t.Fatalf("NewParallelWorker() error = %v", err)
	}
	wf := newItemsWorkflow(t, pw, []any{0, 1})

	_, stored, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	if stored["counter"] != 1 {
		t.Errorf("stored counter = %v, want 1", stored["counter"])
	}
}

func TestParallelWorker_CanceledItemDiscardsPending(t *testing.T) {
	errBoom := errors.New("boom")
	started := make(chan struct{})
	item := newItemNode("item", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		if in.(int) == 1 {
			<-started
			yield(nil, errBoom)
			return
		}
		if err := ctx.State().Set("counter", 1); err != nil {
			yield(nil, err)
			return
		}
		close(started)
		// The sibling's failure cancels this item, which then ends without
		// an error of its own.
		<-ctx.Done()
	})
	wf := newItemsWorkflow(t, newParallelWorker(t, item, 0), []any{0, 1})

	_, stored, err := runStateWorkflow(t, wf, nil)

	if !errors.Is(err, errBoom) {
		t.Errorf("run error = %v, want errBoom", err)
	}
	if _, ok := stored["counter"]; ok {
		t.Errorf("stored counter = %v, want it absent", stored["counter"])
	}
}

// A node of a nested workflow without a schema of its own checks its writes
// against the outer schema at the write: Set returns the error, which the node
// may handle and carry on.
func TestStateSchema_NestedWorkflowWithoutSchemaInheritsOuter(t *testing.T) {
	var setErr error
	writer := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		setErr = ctx.State().Set("unknown", 1)
		return "ok", nil
	}, workflow.NodeConfig{})
	inner, err := workflow.NewWorkflowNode("inner", workflow.Chain(workflow.Start, writer))
	if err != nil {
		t.Fatalf("NewWorkflowNode() error = %v", err)
	}
	wf := newStateWorkflow(t, inner, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	var se *workflow.StateSchemaError
	if !errors.As(setErr, &se) {
		t.Fatalf("Set(unknown) error = %v, want a *workflow.StateSchemaError", setErr)
	}
	if se.Key != "unknown" || se.Schema != "PipelineSchema" {
		t.Errorf("StateSchemaError Key = %q and Schema = %q, want unknown and PipelineSchema", se.Key, se.Schema)
	}
	if _, ok := stored["unknown"]; ok {
		t.Errorf("stored state has unknown")
	}
}

// newWorkflowAgentNode wraps a workflow agent "innerwf" that runs inner in an
// AgentNode.
func newWorkflowAgentNode(t *testing.T, inner workflow.Node) workflow.Node {
	t.Helper()
	wa, err := workflowagent.New(workflowagent.Config{
		Name:  "innerwf",
		Edges: workflow.Chain(workflow.Start, inner),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	an, err := workflow.NewAgentNode(wa, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	return an
}

// A workflow agent run by an AgentNode takes the enclosing workflow's schema,
// so its node's State().Set refuses an undeclared key at the write, and the
// node's earlier accepted write is stored.
func TestStateSchema_AgentNodeOverWorkflowAgentChecksAtWrite(t *testing.T) {
	var setErr error
	wentOn := false
	writer := workflow.NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		if err := ctx.State().Set("counter", 2); err != nil {
			return "", err
		}
		setErr = ctx.State().Set("unknown", 1)
		if setErr != nil {
			return "", setErr
		}
		wentOn = true
		return "ok", nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, newWorkflowAgentNode(t, writer), workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	var se *workflow.StateSchemaError
	if !errors.As(setErr, &se) || se.Key != "unknown" {
		t.Errorf("Set(unknown) error = %v, want a *workflow.StateSchemaError for key unknown", setErr)
	}
	if wentOn {
		t.Errorf("writer went on past the refused Set")
	}
	if got := asStateSchemaError(t, err); got != se {
		t.Errorf("run error = %v, want the error Set returned", err)
	}
	if diff := cmp.Diff(map[string]any{"counter": 2}, stored); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

// An event state delta a workflow agent's node yields inside an AgentNode is
// refused against the enclosing workflow's schema, and nothing is stored.
func TestStateSchema_AgentNodeOverWorkflowAgentEventDeltaRefused(t *testing.T) {
	writer := newStateNode("writer", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		yield(outputEvent(ctx, "ok", map[string]any{"unknown": 1}), nil)
	})
	wf := newStateWorkflow(t, newWorkflowAgentNode(t, writer), workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if len(stored) != 0 {
		t.Errorf("stored state = %v, want empty", stored)
	}
}

// Events of a workflow run under a context with no state validator are not
// checked by that workflow's runners, so the enclosing activation, which has a
// validator, checks them when they reach it.
func TestStateSchema_InnerRunnerWithoutValidatorCheckedByOuter(t *testing.T) {
	writer := newStateNode("writer", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		yield(outputEvent(ctx, "ok", map[string]any{"unknown": 1}), nil)
	})
	inner, err := workflow.New("inner", workflow.Chain(workflow.Start, writer))
	if err != nil {
		t.Fatalf("workflow.New(inner) error = %v", err)
	}
	host := newStateNode("host", func(ctx agent.Context, yield func(*session.Event, error) bool) {
		var none agent.StateValidator
		unchecked := ctx.WithDelta(&agent.CommonContextDelta{StateValidator: &none})
		for ev, err := range inner.RunNode(unchecked, nil) {
			if !yield(ev, err) || err != nil {
				return
			}
		}
	})
	wf := newStateWorkflow(t, host, workflow.WithStateSchema(stateSchemaS(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	if se := asStateSchemaError(t, err); se.Key != "unknown" {
		t.Errorf("StateSchemaError.Key = %q, want unknown", se.Key)
	}
	if _, ok := stored["unknown"]; ok {
		t.Errorf("stored state has unknown")
	}
}

// A generator tool runs as a dynamic child of the node that hosts the LLM
// agent, so its writes are checked against the workflow's schema like any
// other dynamic child's: a refused write fails the tool, which reports the
// error to the model, and none of the tool's writes is stored.
func TestDynamicChild_GeneratorToolWritesChecked(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Title:      "PipelineSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"counter": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	gen, err := generatortool.New(functiontool.Config{Name: "gen", Description: "writes state"},
		func(ctx agent.Context, _ struct{}, _ func(*session.Event) error) (string, error) {
			if err := ctx.State().Set("unknown", 1); err != nil {
				return "", err
			}
			return "ok", ctx.State().Set("counter", 7)
		})
	if err != nil {
		t.Fatalf("generatortool.New() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "gen", Args: map[string]any{}}}}},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: llm, Tools: []tool.Tool{gen}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	host := workflow.NewDynamicNode("host", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		for ev, err := range llmagent.RunLLMAgentAsNode(a, ctx, nil) {
			if err != nil {
				return nil, err
			}
			if err := emit(ev); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}, workflow.NodeConfig{})
	wf := newStateWorkflow(t, host, workflow.WithStateSchema(schema))

	events, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if len(stored) != 0 {
		t.Errorf("stored state = %v, want empty", stored)
	}
	var responses []map[string]any
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "gen" {
				responses = append(responses, p.FunctionResponse.Response)
			}
		}
	}
	if len(responses) != 1 {
		t.Fatalf("%d function responses of gen reached the caller, want 1", len(responses))
	}
	const wantErr = `workflow: state key "unknown" is not declared in the state schema "PipelineSchema"; declared: [counter]`
	if got, _ := responses[0]["error"].(string); !strings.Contains(got, wantErr) {
		t.Errorf("gen's function response error = %q, want it to contain %q", got, wantErr)
	}
}
