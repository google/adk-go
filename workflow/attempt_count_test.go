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
	"fmt"
	"iter"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/functiontool"
)

// attemptLog collects, per key, the attempt numbers that node bodies read.
// Node bodies run on scheduler goroutines, so it is guarded by a mutex.
type attemptLog struct {
	mu   sync.Mutex
	seen map[string][]int
}

func (l *attemptLog) add(key string, n int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.seen == nil {
		l.seen = make(map[string][]int)
	}
	l.seen[key] = append(l.seen[key], n)
}

func (l *attemptLog) get(key string) []int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]int(nil), l.seen[key]...)
}

var errAttempt = errors.New("attempt failed")

// TestScheduler_AttemptCount_EndsAtTheBudget checks that a node that always
// fails reads one number per run its retry budget grants, from 1 up to the
// last attempt.
//
// Fails if startNode reports ns.Attempt without adding 1 (budget 2 reads
// [1, 1]) or adds 2.
func TestScheduler_AttemptCount_EndsAtTheBudget(t *testing.T) {
	tests := []struct {
		name  string
		retry *RetryConfig
		want  []int
	}{
		{name: "no retry config", retry: nil, want: []int{1}},
		{name: "budget 1", retry: retryAlways(1), want: []int{1}},
		{name: "budget 2", retry: retryAlways(2), want: []int{1, 2}},
		{name: "budget 3", retry: retryAlways(3), want: []int{1, 2, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var log attemptLog
			n := newFaninNode("n", NodeConfig{RetryConfig: tt.retry}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				log.add("n", agent.AttemptCount(ctx))
				yield(nil, errAttempt)
			})
			w, err := New("wf", Chain(Start, n))
			if err != nil {
				t.Fatal(err)
			}
			ctx, _ := faninCtx(t, faninBudget)

			res := faninRun(t, w.Run(ctx))

			if !errors.Is(res.err, errAttempt) {
				t.Errorf("Run error = %v, want %v", res.err, errAttempt)
			}
			if diff := cmp.Diff(tt.want, log.get("n")); diff != "" {
				t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestScheduler_AttemptCount_NewActivationAfterRetryAndParkStartsAtOne
// checks that a node that used a retry and then parked for WaitForOutput
// starts its next activation at 1. The graph is the one of
// TestScheduler_WaitForOutput_AttemptBudgetResetsPerActivation: gate fails,
// parks, fails again and then succeeds.
//
// Fails if startNode computes the number before it resets NodeState.Attempt
// for a new activation: gate reads [1, 2, 2, 3].
func TestScheduler_AttemptCount_NewActivationAfterRetryAndParkStartsAtOne(t *testing.T) {
	var gateCalls atomic.Int32
	var log attemptLog
	gateStarted := make(chan struct{})
	wait := true
	a := newFaninNode("a", NodeConfig{}, faninEmitter("A", "r"))
	b := newFaninNode("b", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		faninWait(gateStarted, faninBudget)
		faninEmit(ctx, yield, "B", "r")
	})
	gate := newFaninNode("gate", NodeConfig{WaitForOutput: &wait, RetryConfig: retryAlways(2)}, func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		log.add("gate", agent.AttemptCount(ctx))
		switch gateCalls.Add(1) {
		case 1:
			close(gateStarted)
			yield(nil, errAttempt)
		case 2:
			faninEmit(ctx, yield, nil) // the first activation parks
		case 3:
			yield(nil, errAttempt)
		default:
			faninEmit(ctx, yield, fmt.Sprintf("open:%v", in))
		}
	})
	w, err := New("wf", []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: gate, Route: StringRoute("r")},
		{From: b, To: gate, Route: StringRoute("r")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 2, 1, 2}, log.get("gate")); diff != "" {
		t.Errorf("gate attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_AttemptCount_ResumeStartsAtOne checks that a node that
// paused for input on its second attempt reads 1 when the resume turn
// re-enters it, as adk-python, which does not persist the count.
//
// Fails if the failed attempts of the first turn are carried into the
// resumed run's NodeState.Attempt: the resumed run reads 3.
func TestScheduler_AttemptCount_ResumeStartsAtOne(t *testing.T) {
	var calls atomic.Int32
	var log attemptLog
	rerun := true
	n := newFaninNode("n", NodeConfig{RerunOnResume: &rerun, RetryConfig: retryAlways(3)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		log.add("n", agent.AttemptCount(ctx))
		if _, ok := ctx.ResumedInput("ask"); ok {
			log.add("resumed", int(calls.Add(1)))
			faninEmit(ctx, yield, "done")
			return
		}
		if calls.Add(1) == 1 {
			yield(nil, errAttempt)
			return
		}
		faninAsk(ctx, yield, "ask")
	})
	w, err := New("wf", Chain(Start, n))
	if err != nil {
		t.Fatal(err)
	}
	conv := startFaninConv(t, w)

	res := conv.reply(map[string]any{"ask": "yes"})

	if res.err != nil {
		t.Errorf("resume error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 2, 1}, log.get("n")); diff != "" {
		t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]int{3}, log.get("resumed")); diff != "" {
		t.Errorf("calls that saw the resume input mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_AttemptCount_RetriedWorkflowNodeRestartsItsInnerNodes checks
// that the inner nodes of a retried nested workflow read 1 on the first
// attempt of their own activation, not the attempt of the node around them.
// On the outer node's second attempt its context reads 2, so an inner node
// that inherited it would read 2.
//
// Fails if startNode writes the number only when NodeState.Attempt > 0: head
// reads [1, 2].
func TestScheduler_AttemptCount_RetriedWorkflowNodeRestartsItsInnerNodes(t *testing.T) {
	var tailCalls atomic.Int32
	var log attemptLog
	head := newFaninNode("head", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		log.add("head", agent.AttemptCount(ctx))
		faninEmit(ctx, yield, "H")
	})
	tail := newFaninNode("tail", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		log.add("tail", agent.AttemptCount(ctx))
		if tailCalls.Add(1) == 1 {
			yield(nil, errAttempt)
			return
		}
		faninEmit(ctx, yield, "T")
	})
	inner, err := NewWorkflowNode("inner", Chain(Start, head, tail))
	if err != nil {
		t.Fatal(err)
	}
	inner.BaseNode = NewBaseNode("inner", "", NodeConfig{RetryConfig: retryAlways(2)})
	w, err := New("wf", Chain(Start, inner))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 1}, log.get("head")); diff != "" {
		t.Errorf("head attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]int{1, 1}, log.get("tail")); diff != "" {
		t.Errorf("tail attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestScheduler_AttemptCount_ConcurrentNodesCountSeparately checks that two
// nodes retrying at the same time each read their own numbers.
//
// Fails if the number comes from state the scheduler shares between nodes
// instead of the node's own NodeState.
func TestScheduler_AttemptCount_ConcurrentNodesCountSeparately(t *testing.T) {
	var xCalls, yCalls atomic.Int32
	var log attemptLog
	x := newFaninNode("x", NodeConfig{RetryConfig: retryAlways(3)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		log.add("x", agent.AttemptCount(ctx))
		if xCalls.Add(1) <= 2 {
			yield(nil, errAttempt)
			return
		}
		faninEmit(ctx, yield, nil)
	})
	y := newFaninNode("y", NodeConfig{RetryConfig: retryAlways(2)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		log.add("y", agent.AttemptCount(ctx))
		if yCalls.Add(1) == 1 {
			yield(nil, errAttempt)
			return
		}
		faninEmit(ctx, yield, nil)
	})
	w, err := New("wf", []Edge{{From: Start, To: x}, {From: Start, To: y}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 2, 3}, log.get("x")); diff != "" {
		t.Errorf("x attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]int{1, 2}, log.get("y")); diff != "" {
		t.Errorf("y attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestParallelWorker_AttemptCount_PerItem checks that each item of a
// parallel worker counts its own attempts. Item c reads its number only
// after b's third run has started, so a counter shared by the items would
// show in c.
//
// Fails if runWorker does not give each attempt its number (b reads
// [1, 1, 1]), gives the count of failures without adding 1 (b reads
// [1, 1, 2]), or counts across items (c reads 3).
func TestParallelWorker_AttemptCount_PerItem(t *testing.T) {
	var bCalls atomic.Int32
	var log attemptLog
	bThird := make(chan struct{})
	wrapped := NewFunctionNode("work", func(ctx agent.Context, item string) (string, error) {
		if item == "c" {
			faninWait(bThird, faninBudget)
		}
		log.add(item, agent.AttemptCount(ctx))
		if item != "b" {
			return item, nil
		}
		switch bCalls.Add(1) {
		case 1, 2:
			return "", errAttempt
		default:
			close(bThird)
			return item, nil
		}
	}, NodeConfig{})
	pw, err := NewParallelWorker("parallel", wrapped, 0, NodeConfig{RetryConfig: retryAlways(3)})
	if err != nil {
		t.Fatal(err)
	}
	ctx := agent.NewContext(newMockCtx(t))

	var runErr error
	for _, err := range pw.Run(ctx, []any{"a", "b", "c"}) {
		if err != nil && runErr == nil {
			runErr = err
		}
	}

	if runErr != nil {
		t.Errorf("Run error = %v, want nil", runErr)
	}
	for item, want := range map[string][]int{"a": {1}, "b": {1, 2, 3}, "c": {1}} {
		if diff := cmp.Diff(want, log.get(item)); diff != "" {
			t.Errorf("item %s attempt numbers mismatch (-want +got):\n%s", item, diff)
		}
	}
}

// TestDynamicNode_AttemptCount_ChildReadsOneUnderRetriedParent checks that a
// child run with RunNode reads 1 on every attempt of its retried parent.
//
// Fails if the child's context copies the parent's number: the child reads
// [1, 2, 3].
func TestDynamicNode_AttemptCount_ChildReadsOneUnderRetriedParent(t *testing.T) {
	var parentCalls atomic.Int32
	var log attemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		log.add("child", agent.AttemptCount(ctx))
		return "C", nil
	}, NodeConfig{})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		log.add("parent", agent.AttemptCount(ctx))
		out, err := RunNode[string](ctx, child, nil)
		if err != nil {
			return "", err
		}
		if parentCalls.Add(1) < 3 {
			return "", errAttempt
		}
		return out, nil
	}, NodeConfig{RetryConfig: retryAlways(3)})
	w, err := New("wf", Chain(Start, parent))
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 2, 3}, log.get("parent")); diff != "" {
		t.Errorf("parent attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]int{1, 1, 1}, log.get("child")); diff != "" {
		t.Errorf("child attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestToolNode_AttemptCount_ToolReadsNodeAttempt checks that the tool a
// ToolNode runs reads the number of the node attempt that runs it, as
// adk-python's ToolNode hands the tool the node's own context.
//
// Fails if the tool context stops reporting the number of the context it
// was built from: the tool reads [1, 1, 1].
func TestToolNode_AttemptCount_ToolReadsNodeAttempt(t *testing.T) {
	var calls atomic.Int32
	var log attemptLog
	flaky, err := functiontool.New(functiontool.Config{Name: "flaky", Description: "fails twice"},
		func(ctx agent.Context, _ struct{}) (string, error) {
			log.add("tool", agent.AttemptCount(ctx))
			if calls.Add(1) <= 2 {
				return "", errAttempt
			}
			return "ok", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	n, err := NewToolNode(flaky, NodeConfig{RetryConfig: retryAlways(3)})
	if err != nil {
		t.Fatal(err)
	}
	src := newFaninNode("src", NodeConfig{}, faninEmitter(map[string]any{}))
	w, err := New("wf", Chain(Start, src, n))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 2, 3}, log.get("tool")); diff != "" {
		t.Errorf("tool attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestAgentNode_AttemptCount_AgentRunReadsOne checks that a context inside
// an agent's run reads 1 on every attempt of the retried AgentNode that runs
// the agent, as adk-python builds a fresh context for each agent run.
//
// Fails if the AgentNode hands the agent a context that carries the node's
// number and agent.Run does not reset it to 1: the agent reads [1, 2, 3].
func TestAgentNode_AttemptCount_AgentRunReadsOne(t *testing.T) {
	var calls atomic.Int32
	var log attemptLog
	a, err := agent.New(agent.Config{
		Name: "flaky_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				log.add("agent", agent.AttemptCount(agent.NewToolContext(ctx, "", nil, nil)))
				if calls.Add(1) <= 2 {
					yield(nil, errAttempt)
					return
				}
				faninEmit(agent.NewContext(ctx), yield, "done")
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	n, err := NewAgentNode(a, NodeConfig{RetryConfig: retryAlways(3)})
	if err != nil {
		t.Fatal(err)
	}
	w, err := New("wf", Chain(Start, n))
	if err != nil {
		t.Fatal(err)
	}
	ctx, _ := faninCtx(t, faninBudget)

	res := faninRun(t, w.Run(ctx))

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if diff := cmp.Diff([]int{1, 1, 1}, log.get("agent")); diff != "" {
		t.Errorf("agent attempt numbers mismatch (-want +got):\n%s", diff)
	}
}
