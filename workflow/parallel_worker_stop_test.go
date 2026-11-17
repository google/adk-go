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
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// stoppedItemNode waits until its context is done and then either
// yields one output event, counting the runs in which its yield
// returned true, or fails with ctx.Err().
type stoppedItemNode struct {
	BaseNode
	fail    bool
	started chan struct{}
	starts  atomic.Int32
	wentOn  atomic.Int32
}

func (n *stoppedItemNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if n.starts.Add(1) == 1 {
			close(n.started)
		}
		<-ctx.Done()
		if n.fail {
			yield(nil, ctx.Err())
			return
		}
		if yield(&session.Event{Output: 1}, nil) {
			n.wentOn.Add(1)
		}
	}
}

// TestParallelWorker_StoppedItemDoesNotRunAgain checks that an item
// whose context is canceled neither goes on past its wrapped node's next
// event nor starts the wrapped node again, even with retries left.
func TestParallelWorker_StoppedItemDoesNotRunAgain(t *testing.T) {
	tests := []struct {
		name string
		fail bool
	}{
		{name: "event", fail: false},
		{name: "error", fail: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := range 100 {
				wrapped := &stoppedItemNode{
					BaseNode: NewBaseNode("stopped_item", "", defaultNodeConfig),
					fail:     tt.fail,
					started:  make(chan struct{}),
				}
				pw, err := NewParallelWorker("parallel", wrapped, 0, NodeConfig{
					RetryConfig: &RetryConfig{MaxAttempts: 5, InitialDelay: 0},
				})
				if err != nil {
					t.Fatal(err)
				}

				ctx, cancel := context.WithCancel(t.Context())
				exCtx := agent.NewContext(&MockInvocationContext{Context: ctx})
				done := make(chan struct{})
				var gotErr error
				var gotOutput bool
				go func() {
					defer close(done)
					for ev, err := range pw.Run(exCtx, []any{"item"}) {
						if err != nil {
							gotErr = err
						}
						if _, ok := extractOutput(ev); ok {
							gotOutput = true
						}
					}
				}()

				<-wrapped.started
				cancel()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatalf("iteration %d: timeout waiting for Run to return after the cancel", i)
				}

				if got := wrapped.starts.Load(); got != 1 {
					t.Errorf("iteration %d: wrapped node started %d times, want 1", i, got)
				}
				if got := wrapped.wentOn.Load(); got != 0 {
					t.Errorf("iteration %d: wrapped node went on past its event after the cancel %d times, want 0", i, got)
				}
				if !errors.Is(gotErr, context.Canceled) {
					t.Errorf("iteration %d: Run() error = %v, want %v", i, gotErr, context.Canceled)
				}
				if gotOutput {
					t.Errorf("iteration %d: Run() yielded an output event, want none", i)
				}
				if t.Failed() {
					return
				}
			}
		})
	}
}

// dispatchHookCtx lets a test run code when ParallelWorker.Run derives an
// item's context: Run derives its worker context with WithAgentContext,
// which keeps the hook, and each item's context with WithDelta.
type dispatchHookCtx struct {
	agent.Context
	withDelta func(inner agent.Context, d *agent.CommonContextDelta) agent.Context
}

func (c *dispatchHookCtx) WithAgentContext(ctx context.Context) agent.Context {
	return &dispatchHookCtx{Context: c.Context.WithAgentContext(ctx), withDelta: c.withDelta}
}

func (c *dispatchHookCtx) WithDelta(d *agent.CommonContextDelta) agent.Context {
	return c.withDelta(c.Context, d)
}

// TestParallelWorker_UncappedStartsEveryItemAfterSiblingFailure checks
// that without a concurrency cap every item starts, even one dispatched
// after a sibling failed, and that the failed sibling stops it through
// its canceled context. The dispatch of the second item waits until the
// first item's failure has canceled the worker context, so the third
// item is dispatched after the failure in every run.
func TestParallelWorker_UncappedStartsEveryItemAfterSiblingFailure(t *testing.T) {
	errB := errors.New("error b")
	var started atomic.Int32
	var cSawCancel atomic.Bool
	wrapped := NewFunctionNode("fails_or_waits", func(ctx agent.Context, input string) (string, error) {
		started.Add(1)
		if input == "b" {
			return "", errB
		}
		<-ctx.Done()
		if input == "c" {
			cSawCancel.Store(true)
		}
		return "", ctx.Err()
	}, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	dispatched := 0
	exCtx := &dispatchHookCtx{
		Context: agent.NewContext(newMockCtx(t)),
		withDelta: func(inner agent.Context, d *agent.CommonContextDelta) agent.Context {
			dispatched++
			if dispatched == 2 {
				select {
				case <-inner.Done():
				case <-time.After(5 * time.Second):
					t.Error("timeout waiting for item b's failure to cancel the worker context")
				}
			}
			return inner.WithDelta(d)
		},
	}

	var gotErr error
	for _, err := range pw.Run(exCtx, []any{"b", "held", "c"}) {
		if err != nil {
			gotErr = err
		}
	}

	if !errors.Is(gotErr, errB) {
		t.Errorf("Run() error = %v, want %v", gotErr, errB)
	}
	if got := started.Load(); got != 3 {
		t.Errorf("items started = %d, want 3", got)
	}
	if !cSawCancel.Load() {
		t.Error("item c did not observe the cancel caused by item b's failure")
	}
}

// TestParallelWorker_UncappedStartsNoItemWhenRunContextIsDone checks that
// without a concurrency cap no item starts when the context Run is called
// with is already canceled.
func TestParallelWorker_UncappedStartsNoItemWhenRunContextIsDone(t *testing.T) {
	var started atomic.Int32
	wrapped := NewFunctionNode("counts_calls", func(ctx agent.Context, input string) (string, error) {
		started.Add(1)
		return input, nil
	}, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	exCtx := agent.NewContext(&MockInvocationContext{Context: ctx})

	var gotErr error
	var gotOutput bool
	for ev, err := range pw.Run(exCtx, []any{"a", "b", "c", "d", "e"}) {
		if err != nil {
			gotErr = err
		}
		if _, ok := extractOutput(ev); ok {
			gotOutput = true
		}
	}

	if got := started.Load(); got != 0 {
		t.Errorf("items started = %d, want 0", got)
	}
	if !errors.Is(gotErr, context.Canceled) {
		t.Errorf("Run() error = %v, want %v", gotErr, context.Canceled)
	}
	if gotOutput {
		t.Error("Run() yielded an output event, want none")
	}
}
