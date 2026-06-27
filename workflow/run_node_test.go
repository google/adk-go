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
	"errors"
	"iter"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestRunNode_ErrInvalidRunNodeContext_OnStaticContext(t *testing.T) {
	t.Skip()
	ctx := agent.NewNodeContext(newMockCtx(t), nil) // no subScheduler attached
	_, err := RunNode[string](ctx, newStubNode("c", "x"), nil)
	if !errors.Is(err, ErrInvalidRunNodeContext) {
		t.Errorf("err = %v, want ErrInvalidRunNodeContext", err)
	}
}

// TestRunNode_DynamicNodeUnderScheduler checks that a dynamic node run
// by the main scheduler receives a RunNode-capable agent.Context
// (carrying the sub-scheduler) so its body can schedule a child via
// RunNode.
func TestRunNode_DynamicNodeUnderScheduler(t *testing.T) {
	child := newStubNode("c", "ok")

	var (
		gotOut string
		gotErr error
	)
	// orch is a static graph node that is itself a dynamic node,
	// mirroring an AgentNode run by the main scheduler.
	orch := NewDynamicNode[string, string]("orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			gotOut, gotErr = RunNode[string](ctx, child, nil)
			return gotOut, gotErr
		},
		NodeConfig{},
	)

	w, err := New("root", Chain(Start, orch))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}
	for _, err := range w.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatalf("workflow.Run error: %v", err)
		}
	}

	if gotErr != nil {
		t.Fatalf("RunNode err = %v, want nil", gotErr)
	}
	if gotOut != "ok" {
		t.Errorf("RunNode output = %q, want %q", gotOut, "ok")
	}
}

func TestRunNode_ReturnsTypedOutput(t *testing.T) {
	child := newStubNode("c", "hello")
	got := runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if got != "hello" {
		t.Errorf("RunNode output = %q, want %q", got, "hello")
	}
}

func TestRunNode_OutputTypeMismatch(t *testing.T) {
	child := newStubNode("c", 42) // emits int
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil) // expects string
	})
	if err == nil {
		t.Fatal("expected error for type mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "does not satisfy expected") {
		t.Errorf("err = %v, want type-mismatch message", err)
	}
}

func TestRunNode_PropagatesErrNodeInterrupted(t *testing.T) {
	asker := newRequestInputNode("asker", "approve?")
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, asker, nil)
	})
	if !errors.Is(err, ErrNodeInterrupted) {
		t.Errorf("err = %v, want errors.Is ErrNodeInterrupted", err)
	}
}

func TestRunNode_WaitForOutputChildWithNoOutput_ParksParent(t *testing.T) {
	// A WaitForOutput child that finishes without producing output must
	// park the parent (ErrNodeInterrupted), not falsely complete it with
	// the zero value. Mirrors adk-python ctx.run_node(raise_on_wait=True).
	child := newWaitForOutputNode("waiter")
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if !errors.Is(err, ErrNodeInterrupted) {
		t.Errorf("err = %v, want errors.Is ErrNodeInterrupted", err)
	}
}

func TestRunNode_WaitForOutputChildWithOutput_Completes(t *testing.T) {
	// A WaitForOutput child that does produce output completes normally;
	// the gate must only fire on missing output.
	child := newWaitForOutputWithValueNode("waiter", "done")
	got := runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if got != "done" {
		t.Errorf("RunNode output = %q, want %q", got, "done")
	}
}

func TestRunNode_NoWaitForOutputChildWithNoOutput_ReturnsZero(t *testing.T) {
	// Without WaitForOutput, a child that emits no output still completes
	// and yields the zero value — the gate must not change this default.
	child := newStubNode("c", nil)
	got, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("RunNode output = %q, want zero value", got)
	}
}

func TestRunNode_PropagatesErrNodeFailed(t *testing.T) {
	failer := newFailingNode("failer", errors.New("boom"))
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, failer, nil)
	})
	if !errors.Is(err, ErrNodeFailed) {
		t.Errorf("err = %v, want errors.Is ErrNodeFailed", err)
	}
}

func TestRunNode_WithRunID_AppearsInChildPath(t *testing.T) {
	child := newStubNode("processor", "")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithRunID("order-42"))
	})
	if got, want := child.lastPath, "orch/processor@order-42"; got != want {
		t.Errorf("child Path() = %q, want %q", got, want)
	}
}

func TestRunNode_WithRunID_InvalidRejected(t *testing.T) {
	child := newStubNode("c", "")
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithRunID("123")) // purely numeric
	})
	if !errors.Is(err, ErrInvalidRunID) {
		t.Errorf("err = %v, want errors.Is ErrInvalidRunID", err)
	}
}

func TestRunNode_NilChildOutput_ReturnsZero(t *testing.T) {
	child := newStubNode("c", nil)
	got := runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if got != "" {
		t.Errorf("nil child output → OUT = %q, want \"\" (zero)", got)
	}
}

func TestRunNode_DefaultInheritsParentBranch(t *testing.T) {
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	// MockInvocationContext yields Branch() == "", and the
	// orchestrator inherits it, so the child must also see "".
	if got := child.lastBranch; got != "" {
		t.Errorf("child Branch() = %q, want \"\" (inherits parent at root)", got)
	}
}

func TestRunNode_WithUseSubBranch_AppendsSegment(t *testing.T) {
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithUseSubBranch())
	})
	// Parent branch is "" (root); sub-branch is bare "<name>@<runID>".
	// Auto-counter assigns runID "1" for the first call.
	if got, want := child.lastBranch, "c@1"; got != want {
		t.Errorf("child Branch() = %q, want %q (use_sub_branch at root)", got, want)
	}
}

func TestRunNode_WithUseSubBranch_PlusCustomRunID(t *testing.T) {
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithUseSubBranch(), WithRunID("order-42"))
	})
	if got, want := child.lastBranch, "c@order-42"; got != want {
		t.Errorf("child Branch() = %q, want %q", got, want)
	}
}

func TestRunNode_WithOverrideBranch_ReplacesBase(t *testing.T) {
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithOverrideBranch("custom_branch"))
	})
	if got, want := child.lastBranch, "custom_branch"; got != want {
		t.Errorf("child Branch() = %q, want %q", got, want)
	}
}

func TestRunNode_WithOverrideBranch_PlusUseSubBranch_AppendsToOverride(t *testing.T) {
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil,
			WithOverrideBranch("base"),
			WithUseSubBranch())
	})
	if got, want := child.lastBranch, "base.c@1"; got != want {
		t.Errorf("child Branch() = %q, want %q (override is base, use_sub_branch appends)", got, want)
	}
}

func TestRunNode_WithOverrideBranch_Empty_TreatedAsNoOverride(t *testing.T) {
	// Empty WithOverrideBranch is a no-op (per WithOverrideBranch
	// godoc): even combined with WithUseSubBranch the sub-branch
	// derives off the inherited parent branch.
	child := newStubNode("c", "x")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil,
			WithOverrideBranch(""),
			WithUseSubBranch())
	})
	if got, want := child.lastBranch, "c@1"; got != want {
		t.Errorf("child Branch() = %q, want %q (empty override is a no-op)", got, want)
	}
}

func TestRunNode_WithUseAsOutput_ChildOutputBecomesParentOutput(t *testing.T) {
	child := newStubNode("c", "child_value")
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil, WithUseAsOutput()); err != nil {
				return "", err
			}
			return "parent_value", nil
		},
		NodeConfig{},
	)
	events := drainDynamic(t, n, "")
	// Full suppression: the delegated output is carried on the child's
	// own event; the parent emits no terminal event.
	if got := outputBearingPaths(events); !reflect.DeepEqual(got, []string{"orch/c@1"}) {
		t.Errorf("paths of events with Output = %v, want exactly [\"orch/c@1\"]", got)
	}
	if got := parentTerminalOutput(t, events, "orch/c@1"); got != "child_value" {
		t.Errorf("delegated child Output = %v, want %q", got, "child_value")
	}
	// The child event is stamped OutputFor with its own path plus the
	// delegating parent, so resume attributes the output to both.
	if got := outputForAtPath(events, "orch/c@1"); !reflect.DeepEqual(got, []string{"orch/c@1", "orch"}) {
		t.Errorf("OutputFor = %v, want [orch/c@1 orch]", got)
	}
}

func TestRunNode_WithUseAsOutput_MultiLevelStampsAllAncestors(t *testing.T) {
	// grandchild delegates up through child to the top orchestrator;
	// the single output event is stamped for the whole chain.
	grandchild := newStubNode("gc", "deep_value")
	child := NewDynamicNode[string, string](
		"mid",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, grandchild, nil, WithUseAsOutput())
		},
		NodeConfig{},
	)
	top := NewDynamicNode[string, string](
		"top",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, child, nil, WithUseAsOutput())
		},
		NodeConfig{},
	)
	events := drainDynamic(t, top, "")
	// One output event, carried on the grandchild, suppressing both
	// delegating ancestors.
	if got := outputBearingPaths(events); !reflect.DeepEqual(got, []string{"top/mid@1/gc@1"}) {
		t.Errorf("output-bearing paths = %v, want [top/mid@1/gc@1]", got)
	}
	if got := outputForAtPath(events, "top/mid@1/gc@1"); !reflect.DeepEqual(got, []string{"top/mid@1/gc@1", "top/mid@1", "top"}) {
		t.Errorf("OutputFor = %v, want [top/mid@1/gc@1 top/mid@1 top]", got)
	}
}

func TestRunNode_WithUseAsOutput_MessageAsOutputChildBecomesParentOutput(t *testing.T) {
	// A delegated child whose message IS its output (NodeInfo.
	// MessageAsOutput, no explicit Output — like an LlmAgent node)
	// promotes its model text to the parent's terminal Output.
	child := newMessageAsOutputNode("c", "child_text")
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil, WithUseAsOutput()); err != nil {
				return "", err
			}
			return "parent_value", nil
		},
		NodeConfig{},
	)
	events := drainDynamic(t, n, "")
	// Full suppression: the child's own event carries the output (via
	// MessageAsOutput); the parent emits nothing.
	if got, ok := derivedOutputAtPath(events, "orch/c@1"); !ok || got != "child_text" {
		t.Errorf("delegated child derived output = %v (ok=%v), want %q", got, ok, "child_text")
	}
}

func TestRunNode_WithUseAsOutput_MessageAsOutputEmptyTextIsValidOutput(t *testing.T) {
	// Empty model text under MessageAsOutput is a valid output ("",
	// not "no output"), matching adk-python. The parent's terminal
	// Output must be the empty string, not nil.
	child := newMessageAsOutputNode("c", "")
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil, WithUseAsOutput()); err != nil {
				return "", err
			}
			return "parent_value", nil
		},
		NodeConfig{},
	)
	events := drainDynamic(t, n, "")
	if got, ok := derivedOutputAtPath(events, "orch/c@1"); !ok || got != "" {
		t.Errorf("delegated child derived output = %#v (ok=%v), want empty string", got, ok)
	}
}

func TestRunNode_WithUseAsOutput_SecondDelegationReturnsError(t *testing.T) {
	c1 := newStubNode("c1", "v1")
	c2 := newStubNode("c2", "v2")
	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		if _, err := RunNode[string](ctx, c1, nil, WithUseAsOutput()); err != nil {
			return "", err
		}
		_, err := RunNode[string](ctx, c2, nil, WithUseAsOutput())
		return "", err
	})
	if !errors.Is(err, ErrOutputAlreadyDelegated) {
		t.Errorf("err = %v, want errors.Is ErrOutputAlreadyDelegated", err)
	}
}

func TestRunNode_WithRunID_IdempotentReplay(t *testing.T) {
	child := newCountingStubNode("c", "the_value")
	got1, got2 := "", ""
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		var err error
		got1, err = RunNode[string](ctx, child, nil, WithRunID("stable-id"))
		if err != nil {
			return "", err
		}
		got2, err = RunNode[string](ctx, child, nil, WithRunID("stable-id"))
		return "", err
	})
	if got1 != "the_value" || got2 != "the_value" {
		t.Errorf("RunNode outputs = (%q, %q), want both %q", got1, got2, "the_value")
	}
	if got := child.runCount(); got != 1 {
		t.Errorf("child.Run invocations = %d, want 1", got)
	}
}

func TestRunNode_WithRunID_AndUseAsOutput_IdempotentReplay(t *testing.T) {
	child := newCountingStubNode("c", "delegated_value")
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil,
				WithRunID("stable-id"), WithUseAsOutput()); err != nil {
				return "", err
			}
			if _, err := RunNode[string](ctx, child, nil,
				WithRunID("stable-id"), WithUseAsOutput()); err != nil {
				return "", err
			}
			return "parent_value", nil
		},
		NodeConfig{},
	)
	events := drainDynamic(t, n, "")
	if got := child.runCount(); got != 1 {
		t.Errorf("child.Run invocations = %d, want 1", got)
	}
	// Full suppression: the child's event carries the delegated output;
	// the cached replay re-emits nothing and the parent stays silent.
	if got, ok := derivedOutputAtPath(events, "orch/c@stable-id"); !ok || got != "delegated_value" {
		t.Errorf("delegated child output = %v (ok=%v), want %q", got, ok, "delegated_value")
	}
}

func TestRunNode_SequentialFanOut_PerSibling_DistinctBranches(t *testing.T) {
	// Two children scheduled sequentially with WithUseSubBranch get
	// distinct sub-branches via the auto-counter — child name + "@1",
	// child name + "@2", etc. (counter is per child name, not global).
	c1 := newStubNode("c", "first")
	// re-use the same node twice to exercise the per-name counter.
	var got []string
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		if _, err := RunNode[string](ctx, c1, nil, WithUseSubBranch()); err != nil {
			return "", err
		}
		got = append(got, c1.lastBranch)
		if _, err := RunNode[string](ctx, c1, nil, WithUseSubBranch()); err != nil {
			return "", err
		}
		got = append(got, c1.lastBranch)
		return "", nil
	})
	if len(got) != 2 {
		t.Fatalf("got %d branches, want 2 (orchestrator may have errored mid-loop)", len(got))
	}
	want := []string{"c@1", "c@2"}
	if got[0] != want[0] || got[1] != want[1] {
		t.Errorf("observed branches = %v, want %v "+
			"(per-(name) auto-counter must produce distinct sub-branches)",
			got, want)
	}
}

func TestRunNode_IsolationScopeFromNodePath(t *testing.T) {
	// WithIsolationScopeFromNodePath scopes the child under its full
	// node path on both the child context and its emitted events, so a
	// task-mode node is isolated from peers (b/514866119).
	child := newStubNode("c", "v")
	events := drainDynamic(t, NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil, WithIsolationScopeFromNodePath()); err != nil {
				return "", err
			}
			return "", nil
		},
		NodeConfig{},
	), "")

	if got, want := child.lastScope, "orch/c@1"; got != want {
		t.Errorf("child ctx IsolationScope = %q, want %q", got, want)
	}
	if got := isolationScopeAtPath(events, "orch/c@1"); got != "orch/c@1" {
		t.Errorf("emitted event IsolationScope = %q, want %q", got, "orch/c@1")
	}
}

func TestRunNode_IsolationScope_Explicit(t *testing.T) {
	child := newStubNode("c", "v")
	events := drainDynamic(t, NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[string](ctx, child, nil, WithIsolationScope("fc-123")); err != nil {
				return "", err
			}
			return "", nil
		},
		NodeConfig{},
	), "")

	if got, want := child.lastScope, "fc-123"; got != want {
		t.Errorf("child ctx IsolationScope = %q, want %q", got, want)
	}
	if got := isolationScopeAtPath(events, "orch/c@1"); got != "fc-123" {
		t.Errorf("emitted event IsolationScope = %q, want %q", got, "fc-123")
	}
}

func TestRunNode_ExplicitScopeTakesPrecedenceOverNodePath(t *testing.T) {
	child := newStubNode("c", "v")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		_, err := RunNode[string](ctx, child, nil,
			WithIsolationScope("fc-123"), WithIsolationScopeFromNodePath())
		return "", err
	})
	if got, want := child.lastScope, "fc-123"; got != want {
		t.Errorf("child ctx IsolationScope = %q, want %q (explicit must win)", got, want)
	}
}

func TestRunNode_NoIsolationScope_InheritsParent(t *testing.T) {
	// Without a scope option the child inherits the parent's scope,
	// which is empty for an unscoped orchestrator.
	child := newStubNode("c", "v")
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		_, err := RunNode[string](ctx, child, nil)
		return "", err
	})
	if got := child.lastScope; got != "" {
		t.Errorf("child ctx IsolationScope = %q, want empty (inherited)", got)
	}
}

func TestRunNode_IsolationScope_DistinctPerNodePath(t *testing.T) {
	// Two children sharing a name but at distinct run positions get
	// distinct scopes, the uniqueness guarantee from b/514866119.
	child := newStubNode("c", "v")
	var scopes []string
	runInOrchestrator[string](t, func(ctx agent.Context) (string, error) {
		if _, err := RunNode[string](ctx, child, nil, WithIsolationScopeFromNodePath()); err != nil {
			return "", err
		}
		scopes = append(scopes, child.lastScope)
		if _, err := RunNode[string](ctx, child, nil, WithIsolationScopeFromNodePath()); err != nil {
			return "", err
		}
		scopes = append(scopes, child.lastScope)
		return "", nil
	})
	if want := []string{"orch/c@1", "orch/c@2"}; !reflect.DeepEqual(scopes, want) {
		t.Errorf("scopes = %v, want %v", scopes, want)
	}
}

// --- test helpers ---

// isolationScopeAtPath returns the IsolationScope of the last event
// stamped with nodePath.
func isolationScopeAtPath(events []*session.Event, nodePath string) string {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.NodeInfo != nil && ev.NodeInfo.Path == nodePath {
			return ev.IsolationScope
		}
	}
	return ""
}

// runInOrchestrator drives orchestratorFn inside a dynamic node so that
// the RunNode calls inside have a valid agent.Context + sub-scheduler.
func runInOrchestrator[OUT any](t *testing.T, orchestratorFn func(agent.Context) (OUT, error)) OUT {
	t.Helper()
	got, err := runInOrchestratorWithErr[OUT](t, orchestratorFn)
	if err != nil {
		t.Fatalf("orchestrator error: %v", err)
	}
	return got
}

func runInOrchestratorWithErr[OUT any](t *testing.T, orchestratorFn func(agent.Context) (OUT, error)) (OUT, error) {
	t.Helper()
	var (
		got    OUT
		gotErr error
	)
	n := NewDynamicNode[string, OUT](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (OUT, error) {
			got, gotErr = orchestratorFn(ctx)
			if gotErr != nil {
				return got, gotErr
			}
			return got, nil
		},
		NodeConfig{},
	)
	_, runErr := drainDynamicWithErr(t, n, "")
	if runErr != nil {
		return got, runErr
	}
	return got, gotErr
}

// outputBearingPaths returns NodeInfo.Path of every event whose
// Output is non-nil, preserving order.
func outputBearingPaths(events []*session.Event) []string {
	var paths []string
	for _, ev := range events {
		if ev.Output == nil {
			continue
		}
		var path string
		if ev.NodeInfo != nil {
			path = ev.NodeInfo.Path
		}
		paths = append(paths, path)
	}
	return paths
}

// parentTerminalOutput returns the Output of the last event
// stamped with parentPath.
// outputForAtPath returns NodeInfo.OutputFor of the event at nodePath.
func outputForAtPath(events []*session.Event, nodePath string) []string {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.NodeInfo != nil && ev.NodeInfo.Path == nodePath {
			return ev.NodeInfo.OutputFor
		}
	}
	return nil
}

// derivedOutputAtPath returns the output the event at nodePath carries,
// via childEventOutput (explicit Output or MessageAsOutput-derived).
func derivedOutputAtPath(events []*session.Event, nodePath string) (any, bool) {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.NodeInfo != nil && ev.NodeInfo.Path == nodePath {
			return childEventOutput(ev)
		}
	}
	return nil, false
}

func parentTerminalOutput(t *testing.T, events []*session.Event, parentPath string) any {
	t.Helper()
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.NodeInfo != nil && ev.NodeInfo.Path == parentPath {
			return ev.Output
		}
	}
	t.Fatalf("no event with NodeInfo.Path == %q found among %d events", parentPath, len(events))
	return nil
}

// countingStubNode is a stubNode that counts Run invocations so
// cache-hit tests can assert the child was not re-executed.
type countingStubNode struct {
	*stubNode
	mu    sync.Mutex
	calls int
}

func newCountingStubNode(name string, out any) *countingStubNode {
	return &countingStubNode{stubNode: newStubNode(name, out)}
}

func (n *countingStubNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	n.mu.Lock()
	n.calls++
	n.mu.Unlock()
	return n.stubNode.Run(ctx, input)
}

func (n *countingStubNode) runCount() int {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.calls
}

// TestRunNode_ConcurrentChildren_NoRace runs several children from
// separate goroutines (the WithUseSubBranch pattern). They all emit
// through one shared yield, so without serialization this races the
// range-over-func loop state and panics the iterator. A start gate
// releases the goroutines together to maximize the overlap. Each
// goroutine recovers its panic so the failure is reported even without
// -race; -race is the reliable signal.
func TestRunNode_ConcurrentChildren_NoRace(t *testing.T) {
	const n = 8

	var (
		mu     sync.Mutex
		panics []any
	)
	errs := make([]error, n)

	orch := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(n)
			for i := 0; i < n; i++ {
				// Distinct child + run-id per goroutine so the only shared
				// mutable state is the parent yield path; a shared run-id
				// would instead exercise the idempotency-cache race.
				child := newStubNode("child", "out")
				go func(i int) {
					defer wg.Done()
					defer func() {
						if r := recover(); r != nil {
							mu.Lock()
							panics = append(panics, r)
							mu.Unlock()
						}
					}()
					<-start
					_, errs[i] = RunNode[string](ctx, child, nil,
						WithUseSubBranch(), WithRunID("c"+strconv.Itoa(i)))
				}(i)
			}
			close(start)
			wg.Wait()
			return "done", errors.Join(errs...)
		},
		NodeConfig{},
	)

	events, err := drainDynamicWithErr(t, orch, "")

	mu.Lock()
	gotPanics := append([]any(nil), panics...)
	mu.Unlock()
	if len(gotPanics) > 0 {
		t.Fatalf("%d recovered panic(s) from concurrent children sharing one yield; first: %v",
			len(gotPanics), gotPanics[0])
	}
	if err != nil {
		t.Fatalf("orchestrator error: %v", err)
	}

	// n child outputs forwarded up + the parent's own terminal output.
	outputs := 0
	for _, ev := range events {
		if ev.Output != nil {
			outputs++
		}
	}
	if want := n + 1; outputs != want {
		t.Errorf("output-bearing events = %d, want %d", outputs, want)
	}
}

// waitForOutputNode has WaitForOutput=true and emits a state-only event
// (no Output), modeling an LlmAgent task/chat node that has not yet
// produced its final output.
type waitForOutputNode struct{ BaseNode }

func newWaitForOutputNode(name string) *waitForOutputNode {
	t := true
	return &waitForOutputNode{BaseNode: NewBaseNode(name, "", NodeConfig{WaitForOutput: &t})}
}

func (n *waitForOutputNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{}, nil) // state-only: no Output, no RequestedInput
	}
}

// waitForOutputWithValueNode has WaitForOutput=true and does emit an
// Output, so RunNode must complete it normally.
type waitForOutputWithValueNode struct {
	BaseNode
	out any
}

func newWaitForOutputWithValueNode(name string, out any) *waitForOutputWithValueNode {
	t := true
	return &waitForOutputWithValueNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{WaitForOutput: &t}),
		out:      out,
	}
}

func (n *waitForOutputWithValueNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	out := n.out
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{Output: out}, nil)
	}
}
