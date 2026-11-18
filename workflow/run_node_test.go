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
	"errors"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestRunNode_ErrInvalidRunNodeContext_OnStaticContext(t *testing.T) {
	ctx := agent.NewContext(newMockCtx(t)) // no subScheduler attached
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

// gatedRunCountNode counts Run invocations and blocks until release is
// closed, so a test can hold the leader inside Run while another caller
// reaches the single-flight gate. outErr fails the run; out, when
// non-nil, is emitted as the child's output.
type gatedRunCountNode struct {
	BaseNode
	runs    atomic.Int64
	release chan struct{}
	out     any
	outErr  error
}

func (n *gatedRunCountNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		<-n.release
		if n.outErr != nil {
			yield(nil, n.outErr)
			return
		}
		if n.out != nil {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = n.out
			yield(ev, nil)
		}
	}
}

// gatedInterruptNode is gatedRunCountNode's HITL counterpart: it blocks
// until release is closed, then asks for input.
type gatedInterruptNode struct {
	BaseNode
	runs    atomic.Int64
	release chan struct{}
}

func (n *gatedInterruptNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		<-n.release
		yield(&session.Event{
			RequestedInput: &session.RequestInput{InterruptID: "iid", Message: "ask"},
		}, nil)
	}
}

// runConcurrentDup issues two RunNode calls sharing one WithRunID against
// child, holding the leader inside Run until the second caller is durably
// parked on the gate, then releases both. Results are indexed leader-first.
// synctest.Wait is what makes the overlap a rendezvous instead of a race
// against a deadline.
func runConcurrentDup(t *testing.T, child *gatedRunCountNode) (outs [2]string, errs [2]error) {
	t.Helper()
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			var wg sync.WaitGroup
			wg.Go(func() {
				outs[0], errs[0] = RunNode[string](ctx, child, "x", WithRunID("dup"))
			})
			synctest.Wait() // leader is inside child.Run, parked on release

			wg.Go(func() {
				outs[1], errs[1] = RunNode[string](ctx, child, "x", WithRunID("dup"))
			})
			synctest.Wait() // second caller is parked on the gate, not merely started

			close(child.release)
			wg.Wait()
			return "", nil
		},
		NodeConfig{},
	)
	if _, err := drainDynamicWithErr(t, n, ""); err != nil {
		t.Fatalf("orchestrator Run error: %v", err)
	}
	return outs, errs
}

// TestRunNode_WithRunID_ConcurrentSingleFlight: two concurrent RunNode
// calls sharing one WithRunID resolve the same childPath, so the child
// runs once and both callers receive its output.
func TestRunNode_WithRunID_ConcurrentSingleFlight(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := &gatedRunCountNode{
			BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
			release:  make(chan struct{}),
			out:      "shared_value",
		}
		outs, errs := runConcurrentDup(t, child)

		if got := child.runs.Load(); got != 1 {
			t.Errorf("child ran %d times, want exactly 1", got)
		}
		for i := range outs {
			if errs[i] != nil {
				t.Errorf("caller %d: unexpected error: %v", i, errs[i])
			}
			if outs[i] != "shared_value" {
				t.Errorf("caller %d: output = %q, want %q", i, outs[i], "shared_value")
			}
		}
	})
}

// TestRunNode_WithRunID_ConcurrentSharesInterrupt: concurrent callers of
// one HITL child produce a single RequestedInput, so the user is asked to
// approve once rather than once per caller.
func TestRunNode_WithRunID_ConcurrentSharesInterrupt(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := &gatedInterruptNode{
			BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
			release:  make(chan struct{}),
		}
		var errs [2]error
		n := NewDynamicNode[string, string](
			"orch",
			func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
				var wg sync.WaitGroup
				wg.Go(func() {
					_, errs[0] = RunNode[string](ctx, child, "x", WithRunID("dup"))
				})
				synctest.Wait()

				wg.Go(func() {
					_, errs[1] = RunNode[string](ctx, child, "x", WithRunID("dup"))
				})
				synctest.Wait()

				close(child.release)
				wg.Wait()
				return "", nil
			},
			NodeConfig{},
		)
		events, _ := drainDynamicWithErr(t, n, "")

		if got := child.runs.Load(); got != 1 {
			t.Errorf("child ran %d times, want exactly 1 (an interrupt is shared, not re-run)", got)
		}
		var asks int
		for _, ev := range events {
			if ev.RequestedInput != nil {
				asks++
			}
		}
		if asks != 1 {
			t.Errorf("emitted %d RequestedInput events, want exactly 1", asks)
		}
		for i, err := range errs {
			if !errors.Is(err, ErrNodeInterrupted) {
				t.Errorf("caller %d: error = %v, want ErrNodeInterrupted", i, err)
			}
		}
	})
}

// TestRunNode_WithRunID_ConcurrentSharesFailure: a caller waiting on a
// leader whose child fails receives the leader's error instead of running
// the child again, so side effects are not repeated on the failure path.
func TestRunNode_WithRunID_ConcurrentSharesFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := &gatedRunCountNode{
			BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
			release:  make(chan struct{}),
			outErr:   errors.New("boom"),
		}
		_, errs := runConcurrentDup(t, child)

		if got := child.runs.Load(); got != 1 {
			t.Errorf("child ran %d times, want exactly 1 (a failure is shared, not re-run)", got)
		}
		for i, err := range errs {
			if !errors.Is(err, ErrNodeFailed) {
				t.Errorf("caller %d: error = %v, want ErrNodeFailed", i, err)
			}
		}
	})
}

// TestRunNode_WithRunID_SequentialRerunAfterFailure: sharing is scoped to
// one run. A failure is not cached, so a later sequential call with
// the same WithRunID runs the child again.
func TestRunNode_WithRunID_SequentialRerunAfterFailure(t *testing.T) {
	child := &gatedRunCountNode{
		BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
		release:  make(chan struct{}),
		outErr:   errors.New("boom"),
	}
	close(child.release)

	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			for range 2 {
				if _, err := RunNode[string](ctx, child, "x", WithRunID("dup")); !errors.Is(err, ErrNodeFailed) {
					t.Errorf("error = %v, want ErrNodeFailed", err)
				}
			}
			return "", nil
		},
		NodeConfig{},
	)
	if _, err := drainDynamicWithErr(t, n, ""); err != nil {
		t.Fatalf("orchestrator Run error: %v", err)
	}
	if got := child.runs.Load(); got != 2 {
		t.Errorf("child ran %d times, want 2 (a failure must not be cached)", got)
	}
}

// TestRunNode_WithRunID_WaiterUnblocksOnCancel: a caller parked on the gate
// is released when the invocation is canceled, rather than waiting out a
// leader whose child is still working.
func TestRunNode_WithRunID_WaiterUnblocksOnCancel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := &gatedRunCountNode{
			BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
			release:  make(chan struct{}),
			out:      "shared_value",
		}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		var waiterErr error
		n := NewDynamicNode[string, string](
			"orch",
			func(nc agent.Context, _ string, _ func(*session.Event) error) (string, error) {
				var wg sync.WaitGroup
				wg.Go(func() {
					_, _ = RunNode[string](nc, child, "x", WithRunID("dup"))
				})
				synctest.Wait()

				wg.Go(func() {
					_, waiterErr = RunNode[string](nc, child, "x", WithRunID("dup"))
				})
				synctest.Wait()

				cancel()
				synctest.Wait() // the waiter returns without the leader finishing

				if !errors.Is(waiterErr, context.Canceled) {
					t.Errorf("waiter error = %v, want context.Canceled", waiterErr)
				}
				close(child.release)
				wg.Wait()
				return "", nil
			},
			NodeConfig{},
		)
		for range n.Run(agent.NewContext(&MockInvocationContext{Context: ctx}), "") { //nolint:errcheck // the node body holds the assertions and cancels its own context
		}
	})
}

// gatedPanicNode blocks until release is closed, then panics. out, when
// non-nil, is emitted as an output event before the panic, so a test can
// check that a partial output is not mistaken for a completed run.
type gatedPanicNode struct {
	BaseNode
	runs    atomic.Int64
	release chan struct{}
	out     any
}

func (n *gatedPanicNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		<-n.release
		if n.out != nil {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = n.out
			yield(ev, nil)
		}
		panic("child exploded")
	}
}

// TestRunNode_WithRunID_PanicNotCachedAsSuccess: a child that panics did
// not complete, so nothing is cached for its childPath and a later call
// with the same WithRunID runs it again. The outcome is published from a
// defer, which a panic also runs, so a guard is what stops a partial
// output being replayed as a finished run.
func TestRunNode_WithRunID_PanicNotCachedAsSuccess(t *testing.T) {
	child := &gatedPanicNode{
		BaseNode: NewBaseNode("boom", "", NodeConfig{}),
		release:  make(chan struct{}),
		out:      "partial_value",
	}
	close(child.release)

	var (
		secondOut      string
		secondErr      error
		secondPanicked bool
	)
	n := NewDynamicNode[string, string](
		"orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			func() {
				defer func() { _ = recover() }()
				_, _ = RunNode[string](ctx, child, "x", WithRunID("dup"))
			}()
			func() {
				defer func() {
					if r := recover(); r != nil {
						secondPanicked = true
					}
				}()
				secondOut, secondErr = RunNode[string](ctx, child, "x", WithRunID("dup"))
			}()
			return "", nil
		},
		NodeConfig{},
	)
	if _, err := drainDynamicWithErr(t, n, ""); err != nil {
		t.Fatalf("orchestrator Run error: %v", err)
	}
	if got := child.runs.Load(); got != 2 {
		t.Errorf("child ran %d times, want 2 (a panic must not be cached)", got)
	}
	// The re-run panics again, so the second call never returns.
	if !secondPanicked {
		t.Errorf("second call returned (%q, %v) instead of re-running the child", secondOut, secondErr)
	}
}

// TestRunNode_WithRunID_ConcurrentSharesPanicAsFailure: a caller waiting
// on a leader whose child panics is handed a failure. Without the guard
// it receives the leader's named results as they stood when the panic
// hit, and reports success with the partial output.
func TestRunNode_WithRunID_ConcurrentSharesPanicAsFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		child := &gatedPanicNode{
			BaseNode: NewBaseNode("boom", "", NodeConfig{}),
			release:  make(chan struct{}),
			out:      "partial_value",
		}
		var (
			waiterOut string
			waiterErr error
		)
		n := NewDynamicNode[string, string](
			"orch",
			func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
				var wg sync.WaitGroup
				wg.Go(func() {
					defer func() { _ = recover() }()
					_, _ = RunNode[string](ctx, child, "x", WithRunID("dup"))
				})
				synctest.Wait() // leader is inside child.Run, parked on release

				wg.Go(func() {
					waiterOut, waiterErr = RunNode[string](ctx, child, "x", WithRunID("dup"))
				})
				synctest.Wait() // waiter is parked on the gate

				close(child.release)
				wg.Wait()
				return "", nil
			},
			NodeConfig{},
		)
		if _, err := drainDynamicWithErr(t, n, ""); err != nil {
			t.Fatalf("orchestrator Run error: %v", err)
		}
		if got := child.runs.Load(); got != 1 {
			t.Errorf("child ran %d times, want exactly 1", got)
		}
		if !errors.Is(waiterErr, ErrNodeFailed) {
			t.Errorf("waiter got (%q, %v), want an ErrNodeFailed outcome", waiterOut, waiterErr)
		}
	})
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
	for _, ev := range slices.Backward(events) {
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
	for _, ev := range slices.Backward(events) {
		if ev.NodeInfo != nil && ev.NodeInfo.Path == nodePath {
			return ev.NodeInfo.OutputFor
		}
	}
	return nil
}

// derivedOutputAtPath returns the output the event at nodePath carries,
// via childEventOutput (explicit Output or MessageAsOutput-derived).
func derivedOutputAtPath(events []*session.Event, nodePath string) (any, bool) {
	for _, ev := range slices.Backward(events) {
		if ev.NodeInfo != nil && ev.NodeInfo.Path == nodePath {
			return childEventOutput(ev)
		}
	}
	return nil, false
}

func parentTerminalOutput(t *testing.T, events []*session.Event, parentPath string) any {
	t.Helper()
	for _, ev := range slices.Backward(events) {
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
			for i := range n {
				// Distinct run-id per goroutine so every child actually
				// runs and the only shared mutable state is the parent
				// yield path; a shared run-id would collapse the goroutines
				// into one execution.
				child := newStubNode("child", "out")
				wg.Go(func() {
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
				})
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
	return &waitForOutputNode{BaseNode: NewBaseNode(name, "", NodeConfig{WaitForOutput: new(true)})}
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
	return &waitForOutputWithValueNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{WaitForOutput: new(true)}),
		out:      out,
	}
}

func (n *waitForOutputWithValueNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	out := n.out
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{Output: out}, nil)
	}
}

// TestRunNode_DynamicChildRecoversOwnContext is an end-to-end check that a
// child of a dynamic-node activation can recover its own agent.Context — the
// path runnable tools rely on to reach the sub-scheduler from the context.
func TestRunNode_DynamicChildRecoversOwnContext(t *testing.T) {
	var captured agent.Context

	inner := newFnNode("inner", func(ctx agent.Context) (any, error) {
		captured = ctx
		return "ok", nil
	})

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, inner, nil)
		},
		NodeConfig{},
	)

	if _, err := drainDynamicWithErr(t, orchestrator, ""); err != nil {
		t.Fatalf("orchestrator error: %v", err)
	}
	if captured == nil {
		t.Fatal("inner did not recover any agent.Context")
	}
}

// fnNode adapts a func(agent.Context) callback into a Node that emits
// a single Output event on success.
type fnNode struct {
	BaseNode
	fn func(agent.Context) (any, error)
}

func newFnNode(name string, fn func(agent.Context) (any, error)) *fnNode {
	return &fnNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		fn:       fn,
	}
}

func (n *fnNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		out, err := n.fn(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(&session.Event{Output: out}, nil)
	}
}

// childObs is what one run of a child saw of its own context.
type childObs struct {
	RunID, Path string
}

// observingChild records the run id and path each run of its node sees; the
// n-th run outputs "call-<n>".
type observingChild struct {
	mu  sync.Mutex
	obs []childObs
}

func (c *observingChild) node(name string) Node {
	return newFnNode(name, func(ctx agent.Context) (any, error) {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.obs = append(c.obs, childObs{RunID: ctx.RunID(), Path: ctx.Path()})
		return "call-" + strconv.Itoa(len(c.obs)), nil
	})
}

func (c *observingChild) seen() []childObs {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.obs)
}

// runToolOfAgentNode runs workflow "wf" whose first node is an agent node "a",
// followed by downstream. The agent's run calls tool once with a tool context
// of its own, with call id call_1, as an LLM agent's flow calls a tool.
func runToolOfAgentNode(t *testing.T, tool func(toolCtx agent.Context), downstream ...Node) ([]*session.Event, error) {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "a",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				tool(agent.NewToolContext(ic, "call_1", nil, nil))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	n, err := NewAgentNode(a, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	w, err := New("wf", Chain(append([]Node{Start, n}, downstream...)...))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	mock, _ := newSessionMockCtx(t)
	return collect(w.Run(mock))
}

// sessionToolCtx is a tool context with call id call_1 outside any workflow:
// it carries a session and no sub-scheduler.
func sessionToolCtx(t *testing.T) agent.Context {
	t.Helper()
	mock, _ := newSessionMockCtx(t)
	return agent.NewToolContext(mock, "call_1", nil, nil)
}

// deltaEvent is an event that carries state and artifact deltas.
func deltaEvent(state map[string]any, artifacts map[string]int64) *session.Event {
	return &session.Event{Actions: session.EventActions{StateDelta: state, ArtifactDelta: artifacts}}
}

// hasText reports whether an event in events has a text part equal to text.
func hasText(events []*session.Event, text string) bool {
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.Text == text {
				return true
			}
		}
	}
	return false
}

// Fails if RunNode from a context without a replaying sub-scheduler caches a
// child's output, shares a run, numbers runs past 1, refuses a numeric run id,
// accepts a run id with a path separator, attaches a scheduler to the caller's
// context, puts a leading slash on a root path, loses or misplaces a delta,
// loses the SkipSummarization of a child that fails, lets a child pause where
// nothing can answer, or refuses a context it serves;
// or if RunNode serves the context of a workflow node, or a tool call through
// a dynamic node's context replays.
func TestRunNode_StandaloneAndFallbackContract(t *testing.T) {
	t.Run("agent_node_calls", func(t *testing.T) {
		var child observingChild
		var outs []string
		var errs []error
		_, err := runToolOfAgentNode(t, func(tc agent.Context) {
			for range 2 {
				out, err := RunNode[string](tc, child.node("child"), nil)
				outs = append(outs, out)
				errs = append(errs, err)
			}
		})
		if err != nil {
			t.Fatalf("workflow run error = %v", err)
		}
		if diff := cmp.Diff([]error{nil, nil}, errs, cmpopts.EquateErrors()); diff != "" {
			t.Fatalf("RunNode errors mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"call-1", "call-2"}, outs); diff != "" {
			t.Errorf("RunNode outputs mismatch (-want +got):\n%s", diff)
		}
		want := []childObs{{RunID: "1", Path: "wf@1/a@1/child@1"}, {RunID: "1", Path: "wf@1/a@1/child@1"}}
		if diff := cmp.Diff(want, child.seen()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("fallback_empty_path", func(t *testing.T) {
		tc := sessionToolCtx(t)
		var child observingChild
		out1, err1 := RunNode[string](tc, child.node("child"), nil)
		out2, err2 := RunNode[string](tc, child.node("child"), nil)
		if err1 != nil || err2 != nil {
			t.Fatalf("RunNode errors = %v, %v, want nil", err1, err2)
		}
		if out1 != "call-1" || out2 != "call-2" {
			t.Errorf("RunNode outputs = %q, %q, want call-1, call-2", out1, out2)
		}
		want := []childObs{{RunID: "1", Path: "child@1"}, {RunID: "1", Path: "child@1"}}
		if diff := cmp.Diff(want, child.seen()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("custom_numeric_run_id", func(t *testing.T) {
		var child observingChild
		if _, err := RunNode[string](sessionToolCtx(t), child.node("child"), nil, WithRunID("42")); err != nil {
			t.Fatalf("RunNode(WithRunID(42)) error = %v, want nil", err)
		}
		if diff := cmp.Diff([]childObs{{RunID: "42", Path: "child@42"}}, child.seen()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("reserved_separators", func(t *testing.T) {
		for _, id := range []string{"a/b", "a@b"} {
			var child observingChild
			_, err := RunNode[string](sessionToolCtx(t), child.node("child"), nil, WithRunID(id))
			if !errors.Is(err, ErrInvalidRunID) {
				t.Errorf("RunNode(WithRunID(%q)) error = %v, want ErrInvalidRunID", id, err)
			}
			if got := child.seen(); len(got) != 0 {
				t.Errorf("RunNode(WithRunID(%q)) ran the child: %v", id, got)
			}
		}
	})

	t.Run("fallback_does_not_attach", func(t *testing.T) {
		tc := sessionToolCtx(t)
		if _, err := RunNode[string](tc, newStubNode("child", "x"), nil); err != nil {
			t.Fatalf("RunNode() error = %v, want nil", err)
		}
		if ss := tc.SubScheduler(); ss != nil {
			t.Errorf("SubScheduler() after RunNode = %v, want nil", ss)
		}
	})

	t.Run("deltas_to_actions", func(t *testing.T) {
		tc := sessionToolCtx(t)
		child := newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			partial := deltaEvent(map[string]any{"partial": 9}, nil)
			partial.Partial = true
			if !yield(partial, nil) {
				return
			}
			committed := deltaEvent(map[string]any{"committed": 3}, map[string]int64{"doc.txt": 1})
			committed.Output = "ok"
			yield(committed, nil)
		})
		if _, err := RunNode[string](tc, child, nil); err != nil {
			t.Fatalf("RunNode() error = %v, want nil", err)
		}
		if diff := cmp.Diff(map[string]any{"committed": 3}, tc.Actions().StateDelta); diff != "" {
			t.Errorf("Actions().StateDelta mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]int64{"doc.txt": 1}, tc.Actions().ArtifactDelta); diff != "" {
			t.Errorf("Actions().ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_actions", func(t *testing.T) {
		mock, _ := newSessionMockCtx(t)
		ctx := agent.NewContext(mock)
		child := newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			ev := deltaEvent(map[string]any{"k": 1}, map[string]int64{"doc.txt": 1})
			ev.Output = "ok"
			yield(ev, nil)
		})
		out, err := RunNode[string](ctx, child, nil)
		if err != nil || out != "ok" {
			t.Errorf("RunNode() = %q, %v, want ok, nil", out, err)
		}
	})

	t.Run("request_for_input_refused", func(t *testing.T) {
		tc := sessionToolCtx(t)
		child := newHitlNode("child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if !yield(deltaEvent(map[string]any{"before": 1}, nil), nil) {
				return
			}
			yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ask", Message: "region?"}), nil)
		})
		_, err := RunNode[string](tc, child, nil)
		if !errors.Is(err, ErrInvalidRunNodeContext) || !errors.Is(err, ErrNodeFailed) {
			t.Errorf("RunNode() error = %v, want one wrapping ErrInvalidRunNodeContext and ErrNodeFailed", err)
		}
		if diff := cmp.Diff(map[string]any{"before": 1}, tc.Actions().StateDelta); diff != "" {
			t.Errorf("Actions().StateDelta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("canceled", func(t *testing.T) {
		mock, _ := newSessionMockCtx(t)
		canceled, cancel := context.WithCancel(mock.Context)
		cancel()
		mock.Context = canceled
		_, err := RunNode[string](agent.NewToolContext(mock, "call_1", nil, nil), newStubNode("child", "x"), nil)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("RunNode() error = %v, want context.Canceled", err)
		}
	})

	t.Run("concurrent_calls", func(t *testing.T) {
		tc := sessionToolCtx(t)
		child := newHitlNode("child", func(_ agent.Context, in any, yield func(*session.Event, error) bool) {
			yield(&session.Event{Output: fmt.Sprintf("w%v", in)}, nil)
		})
		outs := make([]string, 4)
		errs := make([]error, 4)
		var wg sync.WaitGroup
		for i := range 4 {
			wg.Go(func() { outs[i], errs[i] = RunNode[string](tc, child, i) })
		}
		wg.Wait()
		if diff := cmp.Diff(make([]error, 4), errs, cmpopts.EquateErrors()); diff != "" {
			t.Fatalf("RunNode errors mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"w0", "w1", "w2", "w3"}, outs); diff != "" {
			t.Errorf("RunNode outputs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("tool_context_isolation_scope", func(t *testing.T) {
		mock, _ := newSessionMockCtx(t)
		mock.isolationScope = "fc_9"
		child := newStubNode("child", "x")
		if _, err := RunNode[string](agent.NewToolContext(mock, "call_1", nil, nil), child, nil); err != nil {
			t.Fatalf("RunNode() error = %v, want nil", err)
		}
		if got := child.lastScope; got != "fc_9" {
			t.Errorf("child IsolationScope() = %q, want fc_9", got)
		}
	})

	t.Run("actions_with_nil_maps", func(t *testing.T) {
		tc := sessionToolCtx(t)
		tc.Actions().StateDelta = nil
		tc.Actions().ArtifactDelta = nil
		child := newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			yield(deltaEvent(map[string]any{"committed": 3}, map[string]int64{"doc.txt": 1}), nil)
		})
		if _, err := RunNode[string](tc, child, nil); err != nil {
			t.Fatalf("RunNode() error = %v, want nil", err)
		}
		if diff := cmp.Diff(map[string]any{"committed": 3}, tc.Actions().StateDelta); diff != "" {
			t.Errorf("Actions().StateDelta mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]int64{"doc.txt": 1}, tc.Actions().ArtifactDelta); diff != "" {
			t.Errorf("Actions().ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
	})

	// textChild emits a text event and then outputs child-out.
	textChild := func(runs *atomic.Int32) Node {
		return newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			runs.Add(1)
			text := &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("child-text", genai.RoleModel)}}
			if !yield(text, nil) {
				return
			}
			yield(&session.Event{Output: "child-out"}, nil)
		})
	}
	// runInFunctionNode runs workflow "wf" whose only node is the function
	// node "host", whose body is body.
	runInFunctionNode := func(t *testing.T, body func(ctx agent.Context)) []*session.Event {
		t.Helper()
		host := NewFunctionNode("host", func(ctx agent.Context, _ any) (string, error) {
			body(ctx)
			return "host-out", nil
		}, NodeConfig{})
		w, err := New("wf", Chain(Start, host))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		mock, _ := newSessionMockCtx(t)
		events, err := collect(w.Run(mock))
		if err != nil {
			t.Fatalf("workflow run error = %v", err)
		}
		return events
	}

	t.Run("static_function_node_refused", func(t *testing.T) {
		var runs atomic.Int32
		var runErr error
		events := runInFunctionNode(t, func(ctx agent.Context) {
			_, runErr = RunNode[string](ctx, textChild(&runs), nil)
		})
		if !errors.Is(runErr, ErrInvalidRunNodeContext) {
			t.Errorf("RunNode() error = %v, want ErrInvalidRunNodeContext", runErr)
		}
		if got := runs.Load(); got != 0 {
			t.Errorf("child runs = %d, want 0", got)
		}
		if hasText(events, "child-text") {
			t.Errorf("the child's text event reached the workflow's events")
		}
	})

	t.Run("tool_context_of_a_function_node_served", func(t *testing.T) {
		var runs atomic.Int32
		var out string
		var runErr error
		runInFunctionNode(t, func(ctx agent.Context) {
			out, runErr = RunNode[string](agent.NewToolContext(ctx, "fc_1", nil, nil), textChild(&runs), nil)
		})
		if runErr != nil || out != "child-out" {
			t.Errorf("RunNode() = %q, %v, want child-out, nil", out, runErr)
		}
		if got := runs.Load(); got != 1 {
			t.Errorf("child runs = %d, want 1", got)
		}
	})

	t.Run("bare_context_with_session_served", func(t *testing.T) {
		mock, _ := newSessionMockCtx(t)
		var runs atomic.Int32
		out, err := RunNode[string](agent.NewContext(mock), textChild(&runs), nil)
		if err != nil || out != "child-out" {
			t.Errorf("RunNode() = %q, %v, want child-out, nil", out, err)
		}
		if got := runs.Load(); got != 1 {
			t.Errorf("child runs = %d, want 1", got)
		}
	})

	// runInDynamicNode runs workflow "wf" whose only node is the dynamic node
	// "orch", whose body is body.
	runInDynamicNode := func(t *testing.T, body func(ctx agent.Context)) {
		t.Helper()
		orch := NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			body(ctx)
			return nil, nil
		}, NodeConfig{})
		w, err := New("wf", Chain(Start, orch))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		mock, _ := newSessionMockCtx(t)
		if _, err := collect(w.Run(mock)); err != nil {
			t.Fatalf("workflow run error = %v", err)
		}
	}

	t.Run("tool_context_on_replaying_scheduler", func(t *testing.T) {
		var child observingChild
		var outs []string
		var errs []error
		runInDynamicNode(t, func(ctx agent.Context) {
			tc := agent.NewToolContext(ctx, "fc_1", nil, nil)
			for _, opts := range [][]RunNodeOption{nil, nil, {WithRunID("42")}} {
				out, err := RunNode[string](tc, child.node("child"), nil, opts...)
				outs = append(outs, out)
				errs = append(errs, err)
			}
		})
		if diff := cmp.Diff([]error{nil, nil, nil}, errs, cmpopts.EquateErrors()); diff != "" {
			t.Fatalf("RunNode errors mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"call-1", "call-2", "call-3"}, outs); diff != "" {
			t.Errorf("RunNode outputs mismatch (-want +got):\n%s", diff)
		}
		want := []childObs{
			{RunID: "1", Path: "wf@1/orch@1/child@1"},
			{RunID: "1", Path: "wf@1/orch@1/child@1"},
			{RunID: "42", Path: "wf@1/orch@1/child@42"},
		}
		if diff := cmp.Diff(want, child.seen()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("dynamic_body_keeps_replay", func(t *testing.T) {
		var child observingChild
		runInDynamicNode(t, func(ctx agent.Context) {
			for range 2 {
				if _, err := RunNode[string](ctx, child.node("child"), nil); err != nil {
					t.Errorf("RunNode() error = %v", err)
				}
			}
		})
		want := []childObs{{RunID: "1", Path: "wf@1/orch@1/child@1"}, {RunID: "2", Path: "wf@1/orch@1/child@2"}}
		if diff := cmp.Diff(want, child.seen()); diff != "" {
			t.Errorf("child runs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("agent_node_parallel_tool_calls", func(t *testing.T) {
		// Two tool calls of one agent run the same child at the same time,
		// both under run id 1. Each runs the child: no run in flight is
		// shared, as each adk-python tool call has a scheduler of its own.
		var runs atomic.Int32
		started := make(chan struct{}, 2)
		release := make(chan struct{})
		child := newFnNode("child", func(agent.Context) (any, error) {
			runs.Add(1)
			started <- struct{}{}
			<-release
			return "ok", nil
		})
		errs := make([]error, 2)
		a, err := agent.New(agent.Config{
			Name: "a",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {
					var wg sync.WaitGroup
					for i, id := range []string{"call_1", "call_2"} {
						tc := agent.NewToolContext(ic, id, nil, nil)
						wg.Go(func() { _, errs[i] = RunNode[string](tc, child, nil) })
					}
					// A call that shares the other's run never starts the
					// child, so stop waiting for it after a while.
					timeout := time.After(10 * time.Second)
				wait:
					for range 2 {
						select {
						case <-started:
						case <-timeout:
							break wait
						}
					}
					close(release)
					wg.Wait()
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		n, err := NewAgentNode(a, NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode() error = %v", err)
		}
		w, err := New("wf", Chain(Start, n))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		mock, _ := newSessionMockCtx(t)
		if _, err := collect(w.Run(mock)); err != nil {
			t.Fatalf("workflow run error = %v", err)
		}
		if diff := cmp.Diff([]error{nil, nil}, errs, cmpopts.EquateErrors()); diff != "" {
			t.Fatalf("RunNode errors mismatch (-want +got):\n%s", diff)
		}
		if got := runs.Load(); got != 2 {
			t.Errorf("child runs = %d, want 2", got)
		}
	})

	t.Run("concurrent_deltas", func(t *testing.T) {
		// Concurrent calls on one context merge into the same actions; run
		// under -race.
		tc := sessionToolCtx(t)
		child := newHitlNode("child", func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
			ctx.Actions().SkipSummarization = true
			yield(deltaEvent(map[string]any{fmt.Sprintf("k%v", in): in}, nil), nil)
		})
		var wg sync.WaitGroup
		for i := range 4 {
			wg.Go(func() {
				if _, err := RunNode[any](tc, child, i); err != nil {
					t.Errorf("RunNode(%d) error = %v", i, err)
				}
			})
		}
		wg.Wait()
		want := map[string]any{"k0": 0, "k1": 1, "k2": 2, "k3": 3}
		if diff := cmp.Diff(want, tc.Actions().StateDelta); diff != "" {
			t.Errorf("Actions().StateDelta mismatch (-want +got):\n%s", diff)
		}
		if !tc.Actions().SkipSummarization {
			t.Error("Actions().SkipSummarization = false, want true")
		}
	})

	t.Run("skip_summarization_of_failed_child", func(t *testing.T) {
		tc := sessionToolCtx(t)
		child := newFnNode("child", func(ctx agent.Context) (any, error) {
			ctx.Actions().SkipSummarization = true
			return nil, errors.New("boom")
		})
		_, err := RunNode[any](tc, child, nil)
		if !errors.Is(err, ErrNodeFailed) {
			t.Errorf("RunNode() error = %v, want one wrapping ErrNodeFailed", err)
		}
		if !tc.Actions().SkipSummarization {
			t.Error("Actions().SkipSummarization = false, want true")
		}
	})

	t.Run("dynamic_child_agent_node_takes_delegated_tool_output", func(t *testing.T) {
		// A tool child run with WithUseAsOutput delegates its output to the
		// agent node, so the agent node, run as a dynamic child, returns it.
		a, err := agent.New(agent.Config{
			Name: "a",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {
					tc := agent.NewToolContext(ic, "call_1", nil, nil)
					if _, err := RunNode[string](tc, newStubNode("child", "delegated"), nil, WithUseAsOutput()); err != nil {
						t.Errorf("tool RunNode() error = %v", err)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		node, err := NewAgentNode(a, NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode() error = %v", err)
		}
		var out any
		var runErr error
		runInDynamicNode(t, func(ctx agent.Context) {
			out, runErr = RunNode[any](ctx, node, nil)
		})
		if runErr != nil || out != "delegated" {
			t.Errorf("RunNode(agent node) = %v, %v, want delegated, nil", out, runErr)
		}
	})
}

// inputSchemaNode outputs its input's "count" as text and counts its runs; its
// input schema requires "count" to be an integer.
type inputSchemaNode struct {
	BaseNode
	runs atomic.Int32
}

func newInputSchemaNode(t *testing.T, name string) *inputSchemaNode {
	t.Helper()
	schema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"count": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	return &inputSchemaNode{BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, schema, nil)}
}

func (n *inputSchemaNode) Run(_ agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		m, _ := input.(map[string]any)
		yield(&session.Event{Output: fmt.Sprint(m["count"])}, nil)
	}
}

// Fails if RunNode runs a child whose input fails the child's input schema,
// or claims the parent's output for it first, on a replaying scheduler and on
// an agent node's standalone one.
func TestRunNode_ValidatesChildInputSchema(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, body func(ctx agent.Context))
	}{
		{
			name: "replaying",
			run: func(t *testing.T, body func(ctx agent.Context)) {
				_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
					body(ctx)
					return "", nil
				})
				if err != nil {
					t.Fatalf("orchestrator error = %v", err)
				}
			},
		},
		{
			name: "standalone",
			run: func(t *testing.T, body func(ctx agent.Context)) {
				if _, err := runToolOfAgentNode(t, body); err != nil {
					t.Fatalf("workflow run error = %v", err)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			child := newInputSchemaNode(t, "child")
			valid := newInputSchemaNode(t, "valid")
			var badErr, goodErr error
			var good any
			tc.run(t, func(ctx agent.Context) {
				_, badErr = RunNode[any](ctx, child, map[string]any{"count": "bad"}, WithUseAsOutput())
				good, goodErr = RunNode[any](ctx, valid, map[string]any{"count": 3}, WithUseAsOutput())
			})

			if got := child.runs.Load(); got != 0 {
				t.Errorf("child runs = %d, want 0", got)
			}
			if _, ok := errors.AsType[*NodeRunError](badErr); !ok || !errors.Is(badErr, ErrInputValidation) {
				t.Errorf("RunNode(invalid input) error = %v, want a *NodeRunError wrapping ErrInputValidation", badErr)
			}
			if badErr == nil || !strings.Contains(badErr.Error(), `input validation failed for node "child"`) {
				t.Errorf("RunNode(invalid input) error = %v, want it to name node child", badErr)
			}
			if goodErr != nil || good != "3" {
				t.Errorf("RunNode(valid input, WithUseAsOutput()) = %v, %v, want 3, nil", good, goodErr)
			}
		})
	}
}

// Fails if a child run by a tool of an agent node is not stamped as the
// agent node's output under WithUseAsOutput, a second such child is not
// refused before it runs, or the child's isolation scope does not follow
// WithIsolationScopeFromNodePath and WithIsolationScope.
func TestRunNode_StandaloneUseAsOutputAndIsolationScope(t *testing.T) {
	child1 := newStubNode("child1", "one")
	child2 := newCountingStubNode("child2", "two")
	child3 := newStubNode("child3", "three")
	var err1, err2, err3 error
	// A downstream node keeps "a" from being a terminal node, whose output
	// would also be the workflow's.
	events, err := runToolOfAgentNode(t, func(tc agent.Context) {
		_, err1 = RunNode[string](tc, child1, nil, WithUseAsOutput(), WithIsolationScopeFromNodePath())
		_, err2 = RunNode[string](tc, child2, nil, WithUseAsOutput())
		_, err3 = RunNode[string](tc, child3, nil, WithIsolationScopeFromNodePath(), WithIsolationScope("explicit"))
	}, newStubNode("b", nil))
	if err != nil {
		t.Fatalf("workflow run error = %v", err)
	}

	if err1 != nil {
		t.Fatalf("first RunNode(WithUseAsOutput()) error = %v, want nil", err1)
	}
	if diff := cmp.Diff([]string{"wf@1/a@1/child1@1", "wf@1/a@1"}, outputForAtPath(events, "wf@1/a@1/child1@1")); diff != "" {
		t.Errorf("OutputFor of child1's event mismatch (-want +got):\n%s", diff)
	}
	if got := child1.lastScope; got != "wf@1/a@1/child1@1" {
		t.Errorf("child1 IsolationScope() = %q, want wf@1/a@1/child1@1", got)
	}
	if !errors.Is(err2, ErrOutputAlreadyDelegated) {
		t.Errorf("second RunNode(WithUseAsOutput()) error = %v, want ErrOutputAlreadyDelegated", err2)
	}
	if got := child2.runCount(); got != 0 {
		t.Errorf("child2 runs = %d, want 0", got)
	}
	if err3 != nil || child3.lastScope != "explicit" {
		t.Errorf("RunNode(both scope options) error = %v and child3 IsolationScope() = %q, want nil and explicit", err3, child3.lastScope)
	}
}

// textEvent is a model text event.
func textEvent(text string) *session.Event {
	return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}}
}

// Fails if the events of an agent and of a child that one of its tools runs
// reach the consumer of AgentNode.Run at the same time. The consumer of an
// iterator is not concurrent, so it appends to a plain slice; run under -race.
func TestAgentNode_ToolChildEventsSerializedWithAgentEvents(t *testing.T) {
	const perSource = 50
	child := newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
		for i := range perSource {
			if !yield(textEvent(fmt.Sprintf("child-%d", i)), nil) {
				return
			}
		}
		yield(&session.Event{Output: "ok"}, nil)
	})
	var toolErr error
	a, err := agent.New(agent.Config{
		Name: "a",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var wg sync.WaitGroup
				wg.Go(func() { _, toolErr = RunNode[any](agent.NewToolContext(ic, "call_1", nil, nil), child, nil) })
				for i := range perSource {
					if !yield(textEvent(fmt.Sprintf("agent-%d", i)), nil) {
						break
					}
				}
				wg.Wait()
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	node, err := NewAgentNode(a, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	mock, _ := newSessionMockCtx(t)

	var events []*session.Event
	for ev, err := range node.Run(agent.NewContext(mock), nil) {
		if err != nil {
			t.Fatalf("AgentNode.Run error = %v", err)
		}
		events = append(events, ev)
	}

	if toolErr != nil {
		t.Errorf("RunNode() error = %v, want nil", toolErr)
	}
	var agentTexts, childTexts int
	for _, ev := range events {
		if ev.Content == nil || len(ev.Content.Parts) == 0 {
			continue
		}
		switch text := ev.Content.Parts[0].Text; {
		case strings.HasPrefix(text, "agent-"):
			agentTexts++
		case strings.HasPrefix(text, "child-"):
			childTexts++
		}
	}
	if agentTexts != 50 || childTexts != 50 {
		t.Errorf("events seen = %d of the agent and %d of the child, want 50 and 50", agentTexts, childTexts)
	}
}

// Fails if AgentNode.Run yields to a consumer that has stopped. The consumer
// stops on the first event, which is the event of a child that the agent's
// tool runs; the tool then fails and the agent reports that error, which
// reaches AgentNode.Run after its consumer has gone.
func TestAgentNode_ConsumerStopsWhileToolChildRuns(t *testing.T) {
	child := newHitlNode("child", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(textEvent("child-text"), nil)
	})
	var toolErr error
	a, err := agent.New(agent.Config{
		Name: "a",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				_, toolErr = RunNode[any](agent.NewToolContext(ic, "call_1", nil, nil), child, nil)
				yield(nil, toolErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	node, err := NewAgentNode(a, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	mock, _ := newSessionMockCtx(t)

	for _, err := range node.Run(agent.NewContext(mock), nil) {
		if err != nil {
			t.Fatalf("AgentNode.Run error = %v, want the child's event first", err)
		}
		break
	}

	if !errors.Is(toolErr, context.Canceled) {
		t.Errorf("RunNode() error = %v, want one wrapping context.Canceled", toolErr)
	}
}
