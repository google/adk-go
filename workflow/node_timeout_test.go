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
	"sync/atomic"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
)

// blockUntilDone waits for the node's context and returns its error, as a
// node that honors cancellation does.
func blockUntilDone(ctx agent.Context, _ any) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

// firstError runs seq to the end and returns the first error it yielded.
func firstError(seq iter.Seq2[*session.Event, error]) error {
	var first error
	for _, err := range seq {
		if err != nil && first == nil {
			first = err
		}
	}
	return first
}

// timedNode is a function node named name whose NodeConfig.Timeout is timeout.
func timedNode(name string, timeout time.Duration, fn func(agent.Context, any) (string, error)) *FunctionNode {
	return NewFunctionNode(name, fn, NodeConfig{Timeout: timeout})
}

// withTimeout sets n's NodeConfig.Timeout to timeout and returns n.
func withTimeout(n *WorkflowNode, timeout time.Duration) *WorkflowNode {
	n.BaseNode = NewBaseNode(n.Name(), "", NodeConfig{Timeout: timeout})
	return n
}

// boundedSeededCtx is newSeededMockCtx under a 10-second deadline, so that a
// node timeout that never fires fails the test instead of hanging it.
func boundedSeededCtx(t *testing.T) *MockInvocationContext {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	t.Cleanup(cancel)
	m := newSeededMockCtx(t)
	m.Context = ctx
	return m
}

func TestNodeTimeoutError_Text(t *testing.T) {
	var nilErr *NodeTimeoutError
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"tenth of a second", &NodeTimeoutError{NodeName: "SlowNode", Timeout: 100 * time.Millisecond}, `node "SlowNode" timed out after 0.1 seconds`},
		{"fifty milliseconds", &NodeTimeoutError{NodeName: "inner_wf", Timeout: 50 * time.Millisecond}, `node "inner_wf" timed out after 0.05 seconds`},
		{"whole seconds", &NodeTimeoutError{NodeName: "n", Timeout: 5 * time.Second}, `node "n" timed out after 5 seconds`},
		{"one and a half seconds", &NodeTimeoutError{NodeName: "n", Timeout: 1500 * time.Millisecond}, `node "n" timed out after 1.5 seconds`},
		{"a microsecond", &NodeTimeoutError{NodeName: "n", Timeout: time.Microsecond}, `node "n" timed out after 1e-06 seconds`},
		{"ninety minutes", &NodeTimeoutError{NodeName: "n", Timeout: 90 * time.Minute}, `node "n" timed out after 5400 seconds`},
		{"name with a quote and a newline", &NodeTimeoutError{NodeName: "a\"b\nc", Timeout: time.Second}, `node "a\"b\nc" timed out after 1 seconds`},
		{"zero value", &NodeTimeoutError{}, `node "" timed out after 0 seconds`},
		{"nil receiver", nilErr, `<nil>`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNodeTimeoutError_UnwrapAlwaysReportsDeadlineExceeded(t *testing.T) {
	var nilErr *NodeTimeoutError
	built := &NodeTimeoutError{NodeName: "n", Timeout: time.Second}
	wrapped := &NodeRunError{ChildName: "c", Cause: fmt.Errorf("%w: %w", ErrNodeFailed, built)}

	if got := errors.Unwrap(built); got != context.DeadlineExceeded {
		t.Errorf("Unwrap of a hand-built error = %v, want context.DeadlineExceeded itself", got)
	}
	if got := errors.Unwrap(&NodeTimeoutError{}); got != context.DeadlineExceeded {
		t.Errorf("Unwrap of the zero value = %v, want context.DeadlineExceeded itself", got)
	}
	if got := nilErr.Unwrap(); got != nil {
		t.Errorf("Unwrap of a nil receiver = %v, want nil", got)
	}
	if !errors.Is(wrapped, context.DeadlineExceeded) {
		t.Error("errors.Is(NodeRunError{ErrNodeFailed + timeout}, DeadlineExceeded) = false, want true")
	}
	var got *NodeTimeoutError
	if !errors.As(wrapped, &got) || got != built {
		t.Errorf("errors.As through NodeRunError found %v, want the same *NodeTimeoutError", got)
	}
}

func TestScheduler_NodeTimeoutErrorNamesNodeAndLimit(t *testing.T) {
	slow := timedNode("slow", 50*time.Millisecond, blockUntilDone)
	w := mustNew(t, []Edge{{From: Start, To: slow}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	got, ok := err.(*NodeTimeoutError)
	if !ok {
		t.Fatalf("Run error = %T (%v), want *NodeTimeoutError itself, not a wrapper", err, err)
	}
	if got.NodeName != "slow" || got.Timeout != 50*time.Millisecond {
		t.Errorf("Run error = {NodeName: %q, Timeout: %v}, want {\"slow\", 50ms}", got.NodeName, got.Timeout)
	}
	if cause := errors.Unwrap(got); cause != context.DeadlineExceeded {
		t.Errorf("cause = %v, want context.DeadlineExceeded itself", cause)
	}
}

func TestScheduler_NestedWorkflowTimeoutNamesTheNodeWhoseLimitFired(t *testing.T) {
	tests := []struct {
		name         string
		innerWFLimit time.Duration
		slowLimit    time.Duration
		wantText     string
		wantName     string
		wantLimit    time.Duration
	}{
		{"nested workflow limit fires first", 50 * time.Millisecond, 5 * time.Second, `node "inner_wf" timed out after 0.05 seconds`, "inner_wf", 50 * time.Millisecond},
		{"inner node limit fires first", 5 * time.Second, 30 * time.Millisecond, `node "slow_node" timed out after 0.03 seconds`, "slow_node", 30 * time.Millisecond},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slow := timedNode("slow_node", tt.slowLimit, blockUntilDone)
			inner, err := NewWorkflowNode("inner_wf", []Edge{{From: Start, To: slow}})
			if err != nil {
				t.Fatal(err)
			}
			withTimeout(inner, tt.innerWFLimit)
			outer, err := New("outer_wf", []Edge{{From: Start, To: inner}})
			if err != nil {
				t.Fatal(err)
			}

			err = firstError(outer.Run(boundedSeededCtx(t)))

			if err == nil || err.Error() != tt.wantText {
				t.Errorf("Run error text = %v, want %q", err, tt.wantText)
			}
			got, ok := err.(*NodeTimeoutError)
			if !ok {
				t.Fatalf("Run error = %T, want *NodeTimeoutError itself", err)
			}
			if got.NodeName != tt.wantName || got.Timeout != tt.wantLimit {
				t.Errorf("Run error = {%q, %v}, want {%q, %v}", got.NodeName, got.Timeout, tt.wantName, tt.wantLimit)
			}
		})
	}
}

func TestScheduler_BodyDeadlineErrorOnLeafWithTimeoutIsNodeTimeout(t *testing.T) {
	bodyErr := fmt.Errorf("fetch: %w", context.DeadlineExceeded)
	n := timedNode("fetch_node", 5*time.Second, func(agent.Context, any) (string, error) { return "", bodyErr })
	w := mustNew(t, []Edge{{From: Start, To: n}})

	var got *NodeTimeoutError
	err := firstError(w.Run(boundedSeededCtx(t)))

	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "fetch_node" || got.Timeout != 5*time.Second {
		t.Errorf("error = {%q, %v}, want {\"fetch_node\", 5s}", got.NodeName, got.Timeout)
	}
	if cause := errors.Unwrap(got); cause != bodyErr {
		t.Errorf("cause = %v, want the body's error %q", cause, bodyErr)
	}
}

func TestScheduler_BodyDeadlineErrorStaysUntypedWithoutOwnLeafLimit(t *testing.T) {
	bodyErr := fmt.Errorf("fetch: %w", context.DeadlineExceeded)
	fail := func(agent.Context, any) (string, error) { return "", bodyErr }

	t.Run("leaf node without a timeout", func(t *testing.T) {
		w := mustNew(t, []Edge{{From: Start, To: timedNode("fetch_node", 0, fail)}})
		err := firstError(w.Run(boundedSeededCtx(t)))
		if err != bodyErr {
			t.Errorf("Run error = %v (%T), want the body's error unchanged", err, err)
		}
	})
	t.Run("nested workflow with a timeout", func(t *testing.T) {
		inner, err := NewWorkflowNode("inner_wf", []Edge{{From: Start, To: timedNode("fetch_node", 0, fail)}})
		if err != nil {
			t.Fatal(err)
		}
		withTimeout(inner, 5*time.Second)
		outer := mustNew(t, []Edge{{From: Start, To: inner}})
		got := firstError(outer.Run(boundedSeededCtx(t)))
		if _, ok := errors.AsType[*NodeTimeoutError](got); ok {
			t.Errorf("Run error = %v, want the body's error, not a node timeout", got)
		}
		if !errors.Is(got, bodyErr) {
			t.Errorf("Run error = %v, want it to wrap the body's error", got)
		}
	})
	t.Run("parallel worker with a timeout", func(t *testing.T) {
		pw, err := NewParallelWorker("parallel", timedNode("wrapped", 0, fail), 0, NodeConfig{Timeout: 5 * time.Second})
		if err != nil {
			t.Fatal(err)
		}
		items := NewFunctionNode("items", func(agent.Context, any) ([]any, error) { return []any{1}, nil }, defaultNodeConfig)
		outer := mustNew(t, []Edge{{From: Start, To: items}, {From: items, To: pw}})
		got := firstError(outer.Run(boundedSeededCtx(t)))
		if _, ok := errors.AsType[*NodeTimeoutError](got); ok {
			t.Errorf("Run error = %v, want the body's error, not a node timeout", got)
		}
		if !errors.Is(got, bodyErr) {
			t.Errorf("Run error = %v, want it to wrap the body's error", got)
		}
	})
}

func TestScheduler_OrdinaryErrorOnLeafWithTimeoutStaysUntyped(t *testing.T) {
	errBoom := errors.New("boom")
	n := timedNode("n", 5*time.Second, func(agent.Context, any) (string, error) { return "", errBoom })
	w := mustNew(t, []Edge{{From: Start, To: n}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	if err != errBoom {
		t.Errorf("Run error = %v (%T), want the node's own error unchanged", err, err)
	}
	if _, ok := errors.AsType[*NodeTimeoutError](err); ok {
		t.Errorf("Run error = %v, want no *NodeTimeoutError before the timeout elapses", err)
	}
}

func TestScheduler_UnrelatedErrorAfterDeadlineStaysUntyped(t *testing.T) {
	errBoom := errors.New("boom")
	n := timedNode("slow", 30*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "", errBoom
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	if err != errBoom {
		t.Errorf("Run error = %v (%T), want the node's own error unchanged", err, err)
	}
}

func TestScheduler_NodeReturningAnotherNodesTimeoutErrorKeepsIt(t *testing.T) {
	other := &NodeTimeoutError{NodeName: "upstream", Timeout: time.Second}
	n := timedNode("caller", 5*time.Second, func(agent.Context, any) (string, error) {
		return "", fmt.Errorf("call upstream: %w", other)
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	var got *NodeTimeoutError
	err := firstError(w.Run(boundedSeededCtx(t)))

	if !errors.As(err, &got) || got != other {
		t.Errorf("Run error = %v, want the upstream *NodeTimeoutError, not one for \"caller\"", err)
	}
}

func TestScheduler_RetryPredicateSeesNodeTimeoutError(t *testing.T) {
	var attempts atomic.Int32
	var seen []string
	n := NewFunctionNode("slow", func(ctx agent.Context, in any) (string, error) {
		attempts.Add(1)
		return blockUntilDone(ctx, in)
	}, NodeConfig{
		Timeout: 30 * time.Millisecond,
		RetryConfig: &RetryConfig{
			MaxAttempts:  2,
			InitialDelay: time.Millisecond,
			ShouldRetry: func(err error) bool {
				if typed, ok := errors.AsType[*NodeTimeoutError](err); ok {
					seen = append(seen, typed.NodeName)
				}
				return true
			},
		},
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	var final *NodeTimeoutError
	err := firstError(w.Run(boundedSeededCtx(t)))

	if len(seen) != 1 || seen[0] != "slow" {
		t.Errorf("ShouldRetry saw *NodeTimeoutError for %v, want [slow]", seen)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("attempts = %d, want 2", got)
	}
	if !errors.As(err, &final) || final.NodeName != "slow" || final.Timeout != 30*time.Millisecond {
		t.Errorf("Run error = %v, want the last attempt's *NodeTimeoutError for \"slow\" (30ms)", err)
	}
}

func TestScheduler_NodeSpanRecordsTheNodeTimeoutOnlyForTheNodesOwnDeadline(t *testing.T) {
	tests := []struct {
		name       string
		invocation time.Duration
		nodeLimit  time.Duration
		wantStatus string
	}{
		{"node limit fires", 5 * time.Second, 50 * time.Millisecond, `node "slow" timed out after 0.05 seconds`},
		{"invocation deadline fires first", 40 * time.Millisecond, 5 * time.Second, "context deadline exceeded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spanExp := tracetest.NewInMemoryExporter()
			telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))
			mock := boundedSeededCtx(t)
			parent, cancel := context.WithTimeout(mock.Context, tt.invocation)
			defer cancel()
			w := mustNew(t, []Edge{{From: Start, To: timedNode("slow", tt.nodeLimit, blockUntilDone)}})

			if err := firstError(w.Run(mock.WithContext(parent).(*MockInvocationContext))); err == nil {
				t.Fatal("Run error = nil, want a timeout")
			}

			var got string
			for _, sp := range spanExp.GetSpans() {
				if sp.Name == "invoke_node slow" {
					got = sp.Status.Description
				}
			}
			if got != tt.wantStatus {
				t.Errorf("span status description = %q, want %q", got, tt.wantStatus)
			}
		})
	}
}

func TestScheduler_NodeCutOffAtTheHandshakeReportsItsTimeout(t *testing.T) {
	n := NewEmittingFunctionNode("emit", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		if err := emit(session.NewEvent(ctx, ctx.InvocationID())); err != nil {
			return "", err
		}
		return "done", nil
	}, NodeConfig{Timeout: 50 * time.Millisecond})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	var err error
	slept := false
	for ev, evErr := range w.Run(boundedSeededCtx(t)) {
		if evErr != nil {
			if err == nil {
				err = evErr
			}
			continue
		}
		// The first event is the one the node emits. The consumer holds it
		// past the node's deadline, so the node is still waiting for the
		// event to be taken when the deadline fires.
		if ev != nil && !slept {
			slept = true
			time.Sleep(200 * time.Millisecond)
		}
	}

	got, ok := err.(*NodeTimeoutError)
	if !ok {
		t.Fatalf("Run error = %T (%v), want *NodeTimeoutError itself", err, err)
	}
	if got.NodeName != "emit" || got.Timeout != 50*time.Millisecond {
		t.Errorf("Run error = {%q, %v}, want {\"emit\", 50ms}", got.NodeName, got.Timeout)
	}
}

// TestNodeTimeoutSetsNoContextCause pins that a node's deadline ends its
// context with the plain deadline error as its cause, so code that compares
// context.Cause with context.DeadlineExceeded (plugin/agentanalytics does)
// keeps seeing the node's timeout as an ordinary deadline.
func TestNodeTimeoutSetsNoContextCause(t *testing.T) {
	var cause error
	n := timedNode("slow", 30*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		cause = context.Cause(ctx)
		return "", ctx.Err()
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	if err := firstError(w.Run(boundedSeededCtx(t))); err == nil {
		t.Fatal("Run error = nil, want a timeout")
	}

	if cause != context.DeadlineExceeded {
		t.Errorf("context.Cause of the node's context after its deadline = %v (%T), want context.DeadlineExceeded itself", cause, cause)
	}
}

// TestScheduler_NodeTimeoutCauseIsTheBareDeadlineError pins the documented
// Unwrap of a timeout the node's own deadline caused: the deadline error
// itself, even when the body returned it wrapped.
func TestScheduler_NodeTimeoutCauseIsTheBareDeadlineError(t *testing.T) {
	n := timedNode("slow", 30*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "", fmt.Errorf("slow: %w", ctx.Err())
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	got, ok := err.(*NodeTimeoutError)
	if !ok {
		t.Fatalf("Run error = %T (%v), want *NodeTimeoutError itself", err, err)
	}
	if cause := errors.Unwrap(got); cause != context.DeadlineExceeded {
		t.Errorf("cause = %v (%T), want context.DeadlineExceeded itself, not the body's wrapper", cause, cause)
	}
}

func TestParallelWorker_ItemSpanRecordsTheNodeTimeout(t *testing.T) {
	spanExp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))
	pw := mustParallelWorker(t, timedNode("wrapped", 50*time.Millisecond, blockUntilDone), 0, defaultNodeConfig)

	if _, err := runParallelWorker(t, t.Context(), pw, []any{1}); err == nil {
		t.Fatal("Run error = nil, want a timeout")
	}

	var got string
	for _, sp := range spanExp.GetSpans() {
		if sp.Name == "invoke_node wrapped" {
			got = sp.Status.Description
		}
	}
	if want := `node "wrapped" timed out after 0.05 seconds`; got != want {
		t.Errorf("item span status description = %q, want %q", got, want)
	}
}

// A negative Timeout means no timeout, and the node still runs under the
// scheduler's cancel, so a failing sibling stops it.
func TestScheduler_NegativeTimeoutNodeIsCanceledWhenASiblingFails(t *testing.T) {
	errBoom := errors.New("boom")
	started := make(chan struct{})
	var waiterErr error
	waiter := timedNode("waiter", -time.Second, func(ctx agent.Context, _ any) (string, error) {
		close(started)
		<-ctx.Done()
		waiterErr = ctx.Err()
		return "", waiterErr
	})
	failer := timedNode("failer", 0, func(agent.Context, any) (string, error) {
		<-started
		return "", errBoom
	})
	w := mustNew(t, []Edge{{From: Start, To: failer}, {From: Start, To: waiter}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	if !errors.Is(err, errBoom) {
		t.Errorf("Run error = %v, want the failing sibling's error", err)
	}
	if waiterErr != context.Canceled {
		t.Errorf("waiter's context error = %v, want context.Canceled (the sibling's failure, not the 10 s test bound)", waiterErr)
	}
}
