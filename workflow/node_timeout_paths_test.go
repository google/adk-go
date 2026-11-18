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
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// dynamicWorkflow is a workflow whose only node is a dynamic node named
// "orch" with config cfg, whose body is body.
func dynamicWorkflow(t *testing.T, cfg NodeConfig, body func(ctx agent.Context) (string, error)) *Workflow {
	t.Helper()
	orch := NewDynamicNode[any, string]("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return body(ctx)
	}, cfg)
	return mustNew(t, []Edge{{From: Start, To: orch}})
}

// runParallelWorker runs pw over items under ctx, bounded by 10 seconds so
// that an item timeout that never fires fails the test instead of hanging
// it, and returns the worker's output and its first error.
func runParallelWorker(t *testing.T, ctx context.Context, pw *ParallelWorker, items []any) (any, error) {
	t.Helper()
	bounded, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	var out any
	var first error
	for ev, err := range pw.Run(agent.NewContext(&MockInvocationContext{Context: bounded}), items) {
		if err != nil && first == nil {
			first = err
		}
		if o, ok := extractOutput(ev); ok {
			out = o
		}
	}
	return out, first
}

func mustParallelWorker(t *testing.T, wrapped Node, maxConcurrency int, cfg NodeConfig) *ParallelWorker {
	t.Helper()
	pw, err := NewParallelWorker("parallel", wrapped, maxConcurrency, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return pw
}

func TestParallelWorker_CallerStopIsNotAnItemTimeout(t *testing.T) {
	t.Run("caller deadline", func(t *testing.T) {
		wrapped := NewFunctionNode("wrapped", ignoresCancellation(nil), NodeConfig{Timeout: 5 * time.Second})
		ctx, cancel := context.WithTimeout(t.Context(), 40*time.Millisecond)
		defer cancel()

		_, err := runParallelWorker(t, ctx, mustParallelWorker(t, wrapped, 0, defaultNodeConfig), []any{1})

		if _, ok := errors.AsType[*NodeTimeoutError](err); ok {
			t.Errorf("Run error = %v, want the caller's deadline, not a timeout of \"wrapped\"", err)
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Run error = %v, want context.DeadlineExceeded", err)
		}
	})
	t.Run("caller cancel", func(t *testing.T) {
		wrapped := NewFunctionNode("wrapped", ignoresCancellation(nil), NodeConfig{Timeout: 5 * time.Second})
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		time.AfterFunc(30*time.Millisecond, cancel)

		_, err := runParallelWorker(t, ctx, mustParallelWorker(t, wrapped, 0, defaultNodeConfig), []any{1})

		if _, ok := errors.AsType[*NodeTimeoutError](err); ok {
			t.Errorf("Run error = %v, want the caller's cancel, not a timeout of \"wrapped\"", err)
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Run error = %v, want context.Canceled", err)
		}
	})
}

func TestRunNode_ChildTimeoutFailsWithNodeTimeoutError(t *testing.T) {
	child := timedNode("slow_child", 40*time.Millisecond, blockUntilDone)
	w := dynamicWorkflow(t, defaultNodeConfig, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})

	err := firstError(w.Run(boundedSeededCtx(t)))

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "slow_child" || got.Timeout != 40*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"slow_child\", 40ms}", got.NodeName, got.Timeout)
	}
	var runErr *NodeRunError
	if !errors.As(err, &runErr) || runErr.ChildName != "slow_child" {
		t.Errorf("Run error = %v, want a *NodeRunError for child \"slow_child\"", err)
	}
	if !errors.Is(err, ErrNodeFailed) {
		t.Errorf("errors.Is(%v, ErrNodeFailed) = false, want true", err)
	}
}

func TestRunNode_ChildIgnoringContextFailsWithNodeTimeoutError(t *testing.T) {
	child := timedNode("deaf_child", 40*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "late", nil
	})
	w := dynamicWorkflow(t, defaultNodeConfig, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})

	err := firstError(w.Run(boundedSeededCtx(t)))

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "deaf_child" || got.Timeout != 40*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"deaf_child\", 40ms}", got.NodeName, got.Timeout)
	}
}

func TestRunNode_OrchestratorTimeoutBeatsLongerChildTimeout(t *testing.T) {
	child := timedNode("child", 5*time.Second, blockUntilDone)
	w := dynamicWorkflow(t, NodeConfig{Timeout: 40 * time.Millisecond}, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})

	err := firstError(w.Run(boundedSeededCtx(t)))

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "orch" || got.Timeout != 40*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"orch\", 40ms}", got.NodeName, got.Timeout)
	}
}

func TestRunNode_ChildBodyDeadlineErrorDoesNotTimeOutTheOrchestrator(t *testing.T) {
	bodyErr := fmt.Errorf("fetch: %w", context.DeadlineExceeded)
	child := timedNode("fetch_child", 0, func(agent.Context, any) (string, error) { return "", bodyErr })
	w := dynamicWorkflow(t, NodeConfig{Timeout: 5 * time.Second}, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil)
	})

	err := firstError(w.Run(boundedSeededCtx(t)))

	if _, ok := errors.AsType[*NodeTimeoutError](err); ok {
		t.Errorf("Run error = %v, want the child's failure, not a timeout of \"orch\"", err)
	}
	if !errors.Is(err, ErrNodeFailed) {
		t.Errorf("errors.Is(%v, ErrNodeFailed) = false, want true", err)
	}
	if !errors.Is(err, bodyErr) {
		t.Errorf("Run error = %v, want it to wrap the child's error", err)
	}
}

func TestParallelWorker_WrappedTimeoutFailsRunWithNodeTimeoutError(t *testing.T) {
	wrapped := timedNode("wrapped", 30*time.Millisecond, blockUntilDone)
	pw := mustParallelWorker(t, wrapped, 0, defaultNodeConfig)

	_, err := runParallelWorker(t, t.Context(), pw, []any{1, 2, 3, 4, 5, 6, 7, 8})

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "wrapped" || got.Timeout != 30*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"wrapped\", 30ms}", got.NodeName, got.Timeout)
	}
}

func TestParallelWorker_WrappedNodeIgnoringContextFailsWithNodeTimeoutError(t *testing.T) {
	wrapped := NewFunctionNode("wrapped", ignoresCancellation(nil), NodeConfig{Timeout: 30 * time.Millisecond})
	pw := mustParallelWorker(t, wrapped, 0, defaultNodeConfig)

	_, err := runParallelWorker(t, t.Context(), pw, []any{1})

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "wrapped" || got.Timeout != 30*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"wrapped\", 30ms}", got.NodeName, got.Timeout)
	}
}

func TestParallelWorker_TimedOutItemIsRetriedWithAFreshTimeout(t *testing.T) {
	var calls atomic.Int32
	wrapped := timedNode("wrapped", 60*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		if calls.Add(1) == 1 {
			<-ctx.Done()
			return "", ctx.Err()
		}
		select {
		case <-time.After(20 * time.Millisecond):
			return "ok", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	pw := mustParallelWorker(t, wrapped, 0, NodeConfig{RetryConfig: &RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond}})

	out, err := runParallelWorker(t, t.Context(), pw, []any{1})
	if err != nil {
		t.Fatalf("Run error = %v, want nil", err)
	}
	if want := []any{"ok"}; !reflect.DeepEqual(out, want) {
		t.Errorf("Run output = %#v, want %#v", out, want)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("calls = %d, want 2", got)
	}
}

func TestParallelWorker_WorkerTimeoutBeatsLongerWrappedTimeout(t *testing.T) {
	pw := mustParallelWorker(t, timedNode("wrapped", 5*time.Second, blockUntilDone), 0, NodeConfig{Timeout: 40 * time.Millisecond})
	items := NewFunctionNode("items", func(agent.Context, any) ([]any, error) { return []any{1, 2}, nil }, defaultNodeConfig)
	w := mustNew(t, []Edge{{From: Start, To: items}, {From: items, To: pw}})

	err := firstError(w.Run(boundedSeededCtx(t)))

	var got *NodeTimeoutError
	if !errors.As(err, &got) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", err)
	}
	if got.NodeName != "parallel" || got.Timeout != 40*time.Millisecond {
		t.Errorf("error = {%q, %v}, want {\"parallel\", 40ms}", got.NodeName, got.Timeout)
	}
}

func TestNonPositiveTimeoutMeansNoLimit(t *testing.T) {
	tests := []struct {
		name    string
		timeout time.Duration
	}{
		{"zero", 0},
		{"negative", -time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := timedNode("wrapped", tt.timeout, func(ctx agent.Context, _ any) (string, error) {
				select {
				case <-time.After(60 * time.Millisecond):
					return "ok", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			})
			pw := mustParallelWorker(t, wrapped, 0, defaultNodeConfig)

			out, err := runParallelWorker(t, t.Context(), pw, []any{1})
			if err != nil {
				t.Fatalf("Run error = %v, want nil", err)
			}
			if want := []any{"ok"}; !reflect.DeepEqual(out, want) {
				t.Errorf("Run output = %#v, want %#v", out, want)
			}
		})
	}
}

func TestNodeTimeoutContextIsReleasedWhenTheAttemptEnds(t *testing.T) {
	t.Run("dynamic child", func(t *testing.T) {
		var childCtx agent.Context
		child := timedNode("child", time.Minute, func(ctx agent.Context, _ any) (string, error) {
			childCtx = ctx
			return "ok", nil
		})
		var afterReturn error
		w := dynamicWorkflow(t, defaultNodeConfig, func(ctx agent.Context) (string, error) {
			out, err := RunNode[string](ctx, child, nil)
			afterReturn = childCtx.Err()
			return out, err
		})

		if err := firstError(w.Run(boundedSeededCtx(t))); err != nil {
			t.Fatalf("Run error = %v, want nil", err)
		}
		if afterReturn != context.Canceled {
			t.Errorf("child context error after RunNode returned = %v, want context.Canceled", afterReturn)
		}
	})
	t.Run("worker item", func(t *testing.T) {
		// With one item at a time, item 2 runs while the worker is still
		// running, after item 1's attempt ended.
		firstItemCtx := make(chan agent.Context, 1)
		wrapped := NewFunctionNode("wrapped", func(ctx agent.Context, in int) (bool, error) {
			if in == 1 {
				firstItemCtx <- ctx
				return false, nil
			}
			first := <-firstItemCtx
			select {
			case <-first.Done():
				return first.Err() == context.Canceled, nil
			case <-time.After(time.Second):
				return false, nil
			}
		}, NodeConfig{Timeout: time.Minute})
		pw := mustParallelWorker(t, wrapped, 1, defaultNodeConfig)

		out, err := runParallelWorker(t, t.Context(), pw, []any{1, 2})
		if err != nil {
			t.Fatalf("Run error = %v, want nil", err)
		}
		if want := []any{false, true}; !reflect.DeepEqual(out, want) {
			t.Errorf("Run output = %#v, want %#v (item 2 saw item 1's context canceled)", out, want)
		}
	})
}
