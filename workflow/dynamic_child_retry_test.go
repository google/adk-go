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
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
)

// childAttempt is one run of a child as the child saw it.
type childAttempt struct {
	Path    string
	RunID   string
	Attempt int
}

// childAttemptLog collects the runs of children from node goroutines.
type childAttemptLog struct {
	mu    sync.Mutex
	items []childAttempt
}

// add records ctx's run and returns its attempt number.
func (l *childAttemptLog) add(ctx agent.Context) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := agent.AttemptCount(ctx)
	l.items = append(l.items, childAttempt{Path: ctx.Path(), RunID: ctx.RunID(), Attempt: n})
	return n
}

func (l *childAttemptLog) get() []childAttempt {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.items)
}

func (l *childAttemptLog) attempts() []int {
	var out []int
	for _, a := range l.get() {
		out = append(out, a.Attempt)
	}
	return out
}

func (l *childAttemptLog) paths() []string {
	var out []string
	for _, a := range l.get() {
		out = append(out, a.Path)
	}
	return out
}

// retryNoDelay retries every error up to maxAttempts times without waiting.
func retryNoDelay(maxAttempts int) *RetryConfig {
	return &RetryConfig{MaxAttempts: maxAttempts, BackoffFactor: 1, ShouldRetry: func(error) bool { return true }}
}

// callResult is what one RunNode call in a parent body returned.
type callResult struct {
	out  any
	err  error
	skip bool // the caller's Actions().SkipSummarization after the call
}

// callingParent returns a dynamic node "parent" that runs child once with
// RunNode and reports the call's result on the returned channel.
func callingParent(child Node, opts ...RunNodeOption) (Node, <-chan callResult) {
	got := make(chan callResult, 1)
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		out, err := RunNode[any](ctx, child, nil, opts...)
		got <- callResult{out: out, err: err, skip: ctx.Actions() != nil && ctx.Actions().SkipSummarization}
		return out, err
	}, NodeConfig{})
	return parent, got
}

// retrySession is a faninSession with a state, so that nodes can write
// state; the writes reach the test through the events' state deltas.
type retrySession struct {
	*faninSession
	state retryState
}

func (s *retrySession) State() session.State { return &s.state }

// retryState is a session.State that is safe for concurrent use.
type retryState struct {
	mu   sync.Mutex
	data map[string]any
}

func (s *retryState) Get(k string) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.data[k]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s *retryState) Set(k string, v any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data == nil {
		s.data = map[string]any{}
	}
	s.data[k] = v
	return nil
}

func (s *retryState) All() iter.Seq2[string, any] {
	s.mu.Lock()
	data := maps.Clone(s.data)
	s.mu.Unlock()
	return maps.All(data)
}

// runParent runs parent as the only node of workflow "wf" under a session
// that records every yielded event.
func runParent(t *testing.T, parent Node, opts ...Option) faninResult {
	t.Helper()
	ctx, _ := faninCtx(t, faninBudget)
	return runParentCtx(t, parent, ctx, opts...)
}

// runParentCtx is runParent under ctx, which the test may cancel.
func runParentCtx(t *testing.T, parent Node, ctx *MockInvocationContext, opts ...Option) faninResult {
	t.Helper()
	w, err := New("wf", Chain(Start, parent), opts...)
	if err != nil {
		t.Fatal(err)
	}
	sess := &retrySession{faninSession: &faninSession{}}
	ctx.sess = sess
	return faninRunObserved(t, w.Run(ctx), sess.append)
}

// receive returns the result the parent reported, failing the test when it
// reported none.
func receive(t *testing.T, got <-chan callResult) callResult {
	t.Helper()
	select {
	case r := <-got:
		return r
	default:
		t.Fatal("the parent did not report a RunNode result")
		return callResult{}
	}
}

// describeAt describes, in stream order, each event at path: "failed" for a
// failure record, "output=<v>" for an output and "state:k=v,..." for a state
// delta, joined by spaces.
func describeAt(events []*session.Event, path string) []string {
	var out []string
	for _, ev := range events {
		if ev.Partial || ev.NodeInfo == nil || ev.NodeInfo.Path != path {
			continue
		}
		out = append(out, describeEvent(ev))
	}
	return out
}

func describeEvent(ev *session.Event) string {
	var parts []string
	if workflowNodeOutcome(ev) == workflowNodeFailureOutcome {
		parts = append(parts, "failed")
	}
	if ev.Output != nil {
		parts = append(parts, fmt.Sprintf("output=%v", ev.Output))
	}
	if len(ev.Actions.StateDelta) > 0 {
		var kv []string
		for _, k := range slices.Sorted(maps.Keys(ev.Actions.StateDelta)) {
			kv = append(kv, fmt.Sprintf("%s=%v", k, ev.Actions.StateDelta[k]))
		}
		parts = append(parts, "state:"+strings.Join(kv, ","))
	}
	return strings.Join(parts, " ")
}

// storedState is the state a session service stores from events: the state
// deltas of the non-partial events, applied in order.
func storedState(events []*session.Event) map[string]any {
	state := map[string]any{}
	for _, ev := range events {
		if !ev.Partial {
			maps.Copy(state, ev.Actions.StateDelta)
		}
	}
	return state
}

const childPathUnderParent = "wf@1/parent@1/child@1"

// TestRunNode_Child_RetriedUntilItSucceeds checks that one RunNode call runs
// a failing child again as its RetryConfig allows, under one path and run
// id, with one failure record per failed attempt, and returns the output of
// the attempt that succeeds.
//
// Fails if runNode runs the child once, if every attempt reads attempt
// number 1, or if the failure record is sent once per call.
func TestRunNode_Child_RetriedUntilItSucceeds(t *testing.T) {
	var log childAttemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		if log.add(ctx) < 3 {
			return "", errAttempt
		}
		return "ok", nil
	}, NodeConfig{RetryConfig: retryAlways(5)})
	parent, got := callingParent(child)

	res := runParent(t, parent)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if r := receive(t, got); r.out != "ok" || r.err != nil {
		t.Errorf("RunNode = (%v, %v), want (ok, nil)", r.out, r.err)
	}
	want := []childAttempt{
		{Path: childPathUnderParent, RunID: "1", Attempt: 1},
		{Path: childPathUnderParent, RunID: "1", Attempt: 2},
		{Path: childPathUnderParent, RunID: "1", Attempt: 3},
	}
	if diff := cmp.Diff(want, log.get()); diff != "" {
		t.Errorf("child runs mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"failed", "failed", "output=ok"}, describeAt(res.events, childPathUnderParent)); diff != "" {
		t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"output=ok"}, describeAt(res.events, "wf@1/parent@1")); diff != "" {
		t.Errorf("events at the parent's path mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_Child_GivesUpAfterMaxAttempts checks that a child that fails
// every attempt fails RunNode with the last attempt's error after
// MaxAttempts attempts.
//
// Fails if the call returns the first attempt's error, or if the budget is
// off by one.
func TestRunNode_Child_GivesUpAfterMaxAttempts(t *testing.T) {
	var log childAttemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		return "", fmt.Errorf("boom %d", log.add(ctx))
	}, NodeConfig{RetryConfig: retryAlways(2)})
	parent, got := callingParent(child)

	runParent(t, parent)

	err := receive(t, got).err
	if !errors.Is(err, ErrNodeFailed) {
		t.Fatalf("RunNode error = %v, want ErrNodeFailed", err)
	}
	var nre *NodeRunError
	if !errors.As(err, &nre) {
		t.Fatalf("RunNode error = %v, want a *NodeRunError", err)
	}
	if nre.ChildName != "child" || nre.ChildPath != childPathUnderParent || nre.RunID != "1" {
		t.Errorf("NodeRunError = {%q, %q, %q}, want {child, %s, 1}", nre.ChildName, nre.ChildPath, nre.RunID, childPathUnderParent)
	}
	if !strings.Contains(err.Error(), "boom 2") || strings.Contains(err.Error(), "boom 1") {
		t.Errorf("RunNode error = %q, want the second attempt's error only", err)
	}
	if diff := cmp.Diff([]int{1, 2}, log.attempts()); diff != "" {
		t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_Child_RetryPolicy checks that RunNode decides on another
// attempt as ShouldRetry does: no config, a budget of at most one attempt or
// a predicate that refuses the error stop the call, and a refused input is
// not an attempt.
//
// Fails if the loop ignores MaxAttempts or the predicate, or if input
// validation moves into the attempt.
func TestRunNode_Child_RetryPolicy(t *testing.T) {
	errRetryable := errors.New("retryable")
	errFatal := errors.New("fatal")
	tests := []struct {
		name         string
		cfg          *RetryConfig
		errs         []error // the error of attempt n is errs[n-1], or the last one
		wantAttempts []int
		wantErr      error
	}{
		{name: "nil_config", cfg: nil, errs: []error{errAttempt}, wantAttempts: []int{1}, wantErr: errAttempt},
		{name: "zero_config", cfg: &RetryConfig{}, errs: []error{errAttempt}, wantAttempts: []int{1}, wantErr: errAttempt},
		{name: "max_attempts_one", cfg: retryNoDelay(1), errs: []error{errAttempt}, wantAttempts: []int{1}, wantErr: errAttempt},
		{name: "max_attempts_negative", cfg: retryNoDelay(-1), errs: []error{errAttempt}, wantAttempts: []int{1}, wantErr: errAttempt},
		{
			name: "predicate_stops_on_unmatched_error",
			cfg: &RetryConfig{MaxAttempts: 4, BackoffFactor: 1, ShouldRetry: func(err error) bool {
				return errors.Is(err, errRetryable)
			}},
			errs:         []error{errRetryable, errFatal},
			wantAttempts: []int{1, 2},
			wantErr:      errFatal,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var log childAttemptLog
			child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
				n := log.add(ctx)
				return "", tc.errs[min(n, len(tc.errs))-1]
			}, NodeConfig{RetryConfig: tc.cfg})
			parent, got := callingParent(child)

			res := runParent(t, parent)

			if err := receive(t, got).err; !errors.Is(err, tc.wantErr) || !errors.Is(err, ErrNodeFailed) {
				t.Errorf("RunNode error = %v, want ErrNodeFailed wrapping %v", err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.wantAttempts, log.attempts()); diff != "" {
				t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
			}
			if got, want := len(describeAt(res.events, childPathUnderParent)), len(tc.wantAttempts); got != want {
				t.Errorf("%d events at the child's path, want %d failure records", got, want)
			}
		})
	}
	t.Run("input_validation_not_retried", func(t *testing.T) {
		var calls int
		child, err := NewFunctionNodeWithSchema("child", func(agent.Context, any) (any, error) {
			calls++
			return nil, nil
		}, &jsonschema.Schema{Type: "integer"}, nil, NodeConfig{RetryConfig: DefaultRetryConfig()})
		if err != nil {
			t.Fatal(err)
		}
		got := make(chan error, 1)
		parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			out, err := RunNode[any](ctx, child, "not-an-int")
			got <- err
			return out, err
		}, NodeConfig{})

		res := runParent(t, parent)

		if err := <-got; !errors.Is(err, ErrInputValidation) {
			t.Errorf("RunNode error = %v, want ErrInputValidation", err)
		}
		if calls != 0 {
			t.Errorf("child ran %d times, want 0", calls)
		}
		if evs := describeAt(res.events, childPathUnderParent); len(evs) != 0 {
			t.Errorf("events at the child's path = %q, want none", evs)
		}
	})
}

// TestRunNode_Child_BackoffBetweenAttempts checks that RunNode waits
// CalculateDelay of the number of failed attempts before each retry.
//
// Fails if the call does not wait, or waits a constant delay.
func TestRunNode_Child_BackoffBetweenAttempts(t *testing.T) {
	var mu sync.Mutex
	var starts []time.Time
	child := NewFunctionNode("child", func(agent.Context, any) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		starts = append(starts, time.Now())
		if len(starts) < 3 {
			return "", errAttempt
		}
		return "ok", nil
	}, NodeConfig{RetryConfig: &RetryConfig{
		MaxAttempts: 3, InitialDelay: 40 * time.Millisecond, BackoffFactor: 2, ShouldRetry: func(error) bool { return true },
	}})
	parent, got := callingParent(child)

	runParent(t, parent)

	if r := receive(t, got); r.err != nil {
		t.Fatalf("RunNode error = %v, want nil", r.err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(starts) != 3 {
		t.Fatalf("child ran %d times, want 3", len(starts))
	}
	if gap := starts[1].Sub(starts[0]); gap < 40*time.Millisecond {
		t.Errorf("gap before attempt 2 = %v, want at least 40ms", gap)
	}
	if gap := starts[2].Sub(starts[1]); gap < 80*time.Millisecond {
		t.Errorf("gap before attempt 3 = %v, want at least 80ms", gap)
	}
}

// TestRunNode_Child_CancelDuringDelayStopsTheChild checks that a
// cancellation of the dynamic node's context while RunNode waits to retry
// ends the call with ErrNodeFailed and the context's error, and starts no
// further attempt.
//
// Fails if the wait ignores the context, or if a delay that has elapsed wins
// over a context that is done.
func TestRunNode_Child_CancelDuringDelayStopsTheChild(t *testing.T) {
	t.Run("cancel_during_long_delay", func(t *testing.T) {
		ctx, cancel := faninCtx(t, faninBudget)
		var log childAttemptLog
		child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			log.add(ctx)
			time.AfterFunc(50*time.Millisecond, cancel)
			return "", errAttempt
		}, NodeConfig{RetryConfig: &RetryConfig{
			MaxAttempts: 3, InitialDelay: time.Hour, BackoffFactor: 1, ShouldRetry: func(error) bool { return true },
		}})
		parent, got := callingParent(child)

		runParentCtx(t, parent, ctx)

		err := receive(t, got).err
		if !errors.Is(err, ErrNodeFailed) || !errors.Is(err, context.Canceled) {
			t.Errorf("RunNode error = %v, want ErrNodeFailed wrapping context.Canceled", err)
		}
		var nre *NodeRunError
		if !errors.As(err, &nre) || nre.ChildPath != childPathUnderParent {
			t.Errorf("RunNode error = %#v, want a *NodeRunError at %s", err, childPathUnderParent)
		}
		if diff := cmp.Diff([]int{1}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("already_canceled_with_zero_delay", func(t *testing.T) {
		// With a zero delay and a done context, a select over both picks
		// either at random, so one run proves little.
		for range 20 {
			ctx, cancel := faninCtx(t, faninBudget)
			var log childAttemptLog
			child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
				log.add(ctx)
				cancel()
				return "", errAttempt
			}, NodeConfig{RetryConfig: retryNoDelay(3)})
			parent, got := callingParent(child)

			runParentCtx(t, parent, ctx)

			if err := receive(t, got).err; !errors.Is(err, ErrNodeFailed) || !errors.Is(err, context.Canceled) {
				t.Fatalf("RunNode error = %v, want ErrNodeFailed wrapping context.Canceled", err)
			}
			if diff := cmp.Diff([]int{1}, log.attempts()); diff != "" {
				t.Fatalf("attempt numbers mismatch (-want +got):\n%s", diff)
			}
		}
	})
}

// TestRunNode_Child_TimeoutIsPerAttempt checks that each attempt of a child
// gets the whole NodeConfig.Timeout and that a timed-out attempt is retried.
//
// Fails if the timeout bounds the call instead of the attempt, or if a
// timed-out attempt is not retried.
func TestRunNode_Child_TimeoutIsPerAttempt(t *testing.T) {
	t.Run("recovers_on_second_attempt", func(t *testing.T) {
		var log childAttemptLog
		child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			if log.add(ctx) == 1 {
				<-ctx.Done()
				return "", ctx.Err()
			}
			// Longer than what a 50ms budget shared with attempt 1 leaves.
			time.Sleep(30 * time.Millisecond)
			return "ok", nil
		}, NodeConfig{Timeout: 50 * time.Millisecond, RetryConfig: retryNoDelay(3)})
		parent, got := callingParent(child)

		res := runParent(t, parent)

		if r := receive(t, got); r.out != "ok" || r.err != nil {
			t.Errorf("RunNode = (%v, %v), want (ok, nil)", r.out, r.err)
		}
		if diff := cmp.Diff([]int{1, 2}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"failed", "output=ok"}, describeAt(res.events, childPathUnderParent)); diff != "" {
			t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("exhausts_timeouts", func(t *testing.T) {
		var log childAttemptLog
		child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			log.add(ctx)
			<-ctx.Done()
			return "", ctx.Err()
		}, NodeConfig{Timeout: 40 * time.Millisecond, RetryConfig: retryNoDelay(2)})
		parent, got := callingParent(child)

		runParent(t, parent)

		err := receive(t, got).err
		var nte *NodeTimeoutError
		if !errors.Is(err, ErrNodeFailed) || !errors.As(err, &nte) {
			t.Fatalf("RunNode error = %v, want ErrNodeFailed wrapping a *NodeTimeoutError", err)
		}
		if nte.NodeName != "child" || nte.Timeout != 40*time.Millisecond {
			t.Errorf("NodeTimeoutError = {%q, %v}, want {child, 40ms}", nte.NodeName, nte.Timeout)
		}
		if diff := cmp.Diff([]int{1, 2}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestRunNode_Child_AttemptIsolation checks that what one attempt of a child
// produced does not reach the next: its output, its SkipSummarization, its
// long-running tool calls and its pending state writes, which it sends on
// its own event before its failure record.
//
// Fails if any of these lives in the call instead of the attempt, or if the
// failed attempt's writes are lost or sent after its failure record.
func TestRunNode_Child_AttemptIsolation(t *testing.T) {
	t.Run("partial_output_does_not_carry", func(t *testing.T) {
		var log childAttemptLog
		child := newFaninNode("child", NodeConfig{RetryConfig: retryNoDelay(2)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if log.add(ctx) == 1 {
				if !faninEmit(ctx, yield, "stale") {
					return
				}
				yield(nil, errAttempt)
			}
		})
		parent, got := callingParent(child)

		runParent(t, parent)

		if r := receive(t, got); r.out != nil || r.err != nil {
			t.Errorf("RunNode = (%v, %v), want (<nil>, nil)", r.out, r.err)
		}
	})
	t.Run("skip_summarization_comes_from_last_attempt", func(t *testing.T) {
		tests := []struct {
			name     string
			skipOn   int
			wantSkip bool
		}{
			{name: "set_by_failed_attempt", skipOn: 1, wantSkip: false},
			{name: "set_by_final_attempt", skipOn: 2, wantSkip: true},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var log childAttemptLog
				child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
					n := log.add(ctx)
					if n == tc.skipOn {
						ctx.Actions().SkipSummarization = true
					}
					if n == 1 {
						return "", errAttempt
					}
					return "ok", nil
				}, NodeConfig{RetryConfig: retryNoDelay(2)})
				parent, got := callingParent(child)

				runParent(t, parent)

				r := receive(t, got)
				if r.err != nil {
					t.Fatalf("RunNode error = %v, want nil", r.err)
				}
				if r.skip != tc.wantSkip {
					t.Errorf("caller's SkipSummarization = %v, want %v", r.skip, tc.wantSkip)
				}
			})
		}
	})
	t.Run("long_running_ids_do_not_carry", func(t *testing.T) {
		var log childAttemptLog
		child := newFaninNode("child", NodeConfig{RetryConfig: retryNoDelay(2)}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			if log.add(ctx) == 1 {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "t"}}}}
				ev.LongRunningToolIDs = []string{"lr-1"}
				if !yield(ev, nil) {
					return
				}
				yield(nil, errAttempt)
				return
			}
			faninEmit(ctx, yield, "clean")
		})
		parent, got := callingParent(child, WithRaiseOnWait())

		runParent(t, parent)

		if r := receive(t, got); r.out != "clean" || r.err != nil {
			t.Errorf("RunNode = (%v, %v), want (clean, nil)", r.out, r.err)
		}
	})
	t.Run("failed_attempt_writes_flushed_before_its_failure_record", func(t *testing.T) {
		var log childAttemptLog
		child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
			n := log.add(ctx)
			if err := ctx.State().Set("k", fmt.Sprintf("v%d", n)); err != nil {
				return "", err
			}
			if n == 1 {
				return "", errAttempt
			}
			return "ok", nil
		}, NodeConfig{RetryConfig: retryNoDelay(2)})
		parent, got := callingParent(child, WithIsolationScope("scope-a"))

		res := runParent(t, parent)

		if r := receive(t, got); r.err != nil {
			t.Fatalf("RunNode error = %v, want nil", r.err)
		}
		want := []string{"state:k=v1", "failed", "output=ok state:k=v2"}
		if diff := cmp.Diff(want, describeAt(res.events, childPathUnderParent)); diff != "" {
			t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
		}
		for _, ev := range res.events {
			if _, ok := ev.Actions.StateDelta["k"]; ok && ev.IsolationScope != "scope-a" {
				t.Errorf("event with k=%v has IsolationScope %q, want scope-a", ev.Actions.StateDelta["k"], ev.IsolationScope)
			}
		}
	})
}

// TestRunNode_Child_RetryEmitsSpanPerAttempt checks that each attempt of a
// child has an invoke_node span of its own, marked failed when the attempt
// failed, and that a cached result has one too.
//
// Fails if the span covers the call instead of the attempt, or if an
// attempt's span is not ended or not marked.
func TestRunNode_Child_RetryEmitsSpanPerAttempt(t *testing.T) {
	spanExp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))
	var log childAttemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		if log.add(ctx) == 1 {
			return "", errors.New("boom")
		}
		return "ok", nil
	}, NodeConfig{RetryConfig: retryNoDelay(3)})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if _, err := RunNode[string](ctx, child, nil, WithRunID("k")); err != nil {
			return "", err
		}
		return RunNode[string](ctx, child, nil, WithRunID("k"))
	}, NodeConfig{})

	res := runParent(t, parent)

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	var got []codes.Code
	for _, s := range spanExp.GetSpans() {
		if s.Name == "invoke_node child" {
			got = append(got, s.Status.Code)
		}
	}
	if diff := cmp.Diff([]codes.Code{codes.Error, codes.Unset, codes.Unset}, got); diff != "" {
		t.Errorf("invoke_node child span statuses mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_Child_StateSchemaFailureIsRetried checks that an attempt whose
// state write the state schema refuses fails and is retried.
//
// Fails if the refusal ends the call without a retry decision.
func TestRunNode_Child_StateSchemaFailureIsRetried(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"count": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	var log childAttemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		if log.add(ctx) < 3 {
			return "", ctx.State().Set("count", "not-an-int")
		}
		return "ok", ctx.State().Set("count", 7)
	}, NodeConfig{RetryConfig: retryNoDelay(3)})
	parent, got := callingParent(child)

	res := runParent(t, parent, WithStateSchema(schema))

	if r := receive(t, got); r.out != "ok" || r.err != nil {
		t.Errorf("RunNode = (%v, %v), want (ok, nil)", r.out, r.err)
	}
	if diff := cmp.Diff([]int{1, 2, 3}, log.attempts()); diff != "" {
		t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"count": 7}, storedState(res.events)); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_Child_OutputSchemaFailureIsRetried checks that an attempt
// whose output the child's output schema refuses fails and is retried, that
// the refused output is never sent while the attempt's state writes are, and
// that the call that runs out of attempts returns the refusal wrapped once.
//
// Fails if the refusal gets no retry decision, if the refused output is
// sent, if the attempt's writes are lost, or if the refusal is wrapped twice.
func TestRunNode_Child_OutputSchemaFailureIsRetried(t *testing.T) {
	t.Run("retries_until_valid", func(t *testing.T) {
		var log childAttemptLog
		child, err := NewFunctionNodeWithSchema("child", func(ctx agent.Context, _ any) (any, error) {
			n := log.add(ctx)
			if err := ctx.State().Set(fmt.Sprintf("c_attempt_%d", n), n); err != nil {
				return nil, err
			}
			if n < 3 {
				return "not_an_int", nil
			}
			return 99, nil
		}, nil, &jsonschema.Schema{Type: "integer"}, NodeConfig{RetryConfig: retryNoDelay(3)})
		if err != nil {
			t.Fatal(err)
		}
		parent, got := callingParent(child)

		res := runParent(t, parent)

		if r := receive(t, got); r.out != 99 || r.err != nil {
			t.Errorf("RunNode = (%v, %v), want (99, nil)", r.out, r.err)
		}
		if diff := cmp.Diff([]int{1, 2, 3}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
		want := []string{"state:c_attempt_1=1", "failed", "state:c_attempt_2=2", "failed", "output=99 state:c_attempt_3=3"}
		if diff := cmp.Diff(want, describeAt(res.events, childPathUnderParent)); diff != "" {
			t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("exhausts_max_attempts", func(t *testing.T) {
		var log childAttemptLog
		child, err := NewFunctionNodeWithSchema("child", func(ctx agent.Context, _ any) (any, error) {
			log.add(ctx)
			return "bad", nil
		}, nil, &jsonschema.Schema{Type: "integer"}, NodeConfig{RetryConfig: retryNoDelay(2)})
		if err != nil {
			t.Fatal(err)
		}
		parent, got := callingParent(child)

		res := runParent(t, parent)

		err = receive(t, got).err
		var nre *NodeRunError
		if !errors.Is(err, ErrNodeFailed) || !errors.As(err, &nre) || nre.ChildPath != childPathUnderParent {
			t.Fatalf("RunNode error = %v, want a *NodeRunError at %s wrapping ErrNodeFailed", err, childPathUnderParent)
		}
		if !strings.Contains(err.Error(), `output validation failed for node "child"`) {
			t.Errorf("RunNode error = %q, want the output refusal", err)
		}
		if got := strings.Count(err.Error(), ErrNodeFailed.Error()); got != 1 {
			t.Errorf("RunNode error = %q holds %q %d times, want 1", err, ErrNodeFailed, got)
		}
		if diff := cmp.Diff([]int{1, 2}, log.attempts()); diff != "" {
			t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"failed", "failed"}, describeAt(res.events, childPathUnderParent)); diff != "" {
			t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestRunNode_Child_RefusedOutputStoresNothingOfItsOwnEvent checks that an
// event whose output the child's output schema refuses is dropped whole: its
// own state delta is not stored, while the writes the attempt made through
// its context are.
//
// Fails if the refused event's writes are admitted before its output is
// validated.
func TestRunNode_Child_RefusedOutputStoresNothingOfItsOwnEvent(t *testing.T) {
	schema, err := (&jsonschema.Schema{Type: "integer"}).Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	var log childAttemptLog
	child := &faninNode{
		BaseNode: NewBaseNodeWithSchemas("child", "", NodeConfig{RetryConfig: retryNoDelay(3)}, nil, schema),
		run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			k := log.add(ctx)
			if err := ctx.State().Set(fmt.Sprintf("pending_%d", k), k); err != nil {
				yield(nil, err)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = "bad"
			if k == 3 {
				ev.Output = 42
			}
			ev.Actions.StateDelta = map[string]any{fmt.Sprintf("own_%d", k): k}
			ev.Actions.ArtifactDelta = map[string]int64{fmt.Sprintf("own_art_%d", k): int64(k)}
			yield(ev, nil)
		},
	}
	parent, got := callingParent(child)

	res := runParent(t, parent)

	if r := receive(t, got); r.out != 42 || r.err != nil {
		t.Errorf("RunNode = (%v, %v), want (42, nil)", r.out, r.err)
	}
	want := map[string]any{"pending_1": 1, "pending_2": 2, "pending_3": 3, "own_3": 3}
	if diff := cmp.Diff(want, storedState(res.events)); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
	storedArt := map[string]int64{}
	for _, ev := range res.events {
		if !ev.Partial {
			maps.Copy(storedArt, ev.Actions.ArtifactDelta)
		}
	}
	if diff := cmp.Diff(map[string]int64{"own_art_3": 3}, storedArt); diff != "" {
		t.Errorf("stored artifact delta mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_Child_RetriedParentRunsChildWithFullBudget checks that a
// parent retried after its child ran out of attempts runs the child again
// under the same run id with its whole budget.
//
// Fails if the child's attempt count carries over to the parent's next
// attempt, or if the parent's retry is served from a cache.
func TestRunNode_Child_RetriedParentRunsChildWithFullBudget(t *testing.T) {
	var log childAttemptLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		log.add(ctx)
		return "", errAttempt
	}, NodeConfig{RetryConfig: retryNoDelay(2)})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return RunNode[string](ctx, child, nil)
	}, NodeConfig{RetryConfig: retryAlways(2)})

	res := runParent(t, parent)

	if !errors.Is(res.err, ErrNodeFailed) {
		t.Errorf("Run error = %v, want ErrNodeFailed", res.err)
	}
	want := []childAttempt{
		{Path: childPathUnderParent, RunID: "1", Attempt: 1},
		{Path: childPathUnderParent, RunID: "1", Attempt: 2},
		{Path: childPathUnderParent, RunID: "1", Attempt: 1},
		{Path: childPathUnderParent, RunID: "1", Attempt: 2},
	}
	if diff := cmp.Diff(want, log.get()); diff != "" {
		t.Errorf("child runs mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"failed", "failed", "failed", "failed"}, describeAt(res.events, childPathUnderParent)); diff != "" {
		t.Errorf("events at the child's path mismatch (-want +got):\n%s", diff)
	}
}

// TestRunNode_RetriedDynamicChild_SubChildFailureReplaysSiblings checks that
// a retried dynamic child whose sub-child failed starts its sub-children at
// run id 1 again: the completed sibling is served from history and the
// failed sub-child runs again under its first path. The failed attempt's
// record names the failed sub-child, and its pending writes are dropped.
//
// Fails if the sub-children's run ids keep counting across the attempts, or
// if the failure record lacks the failed sub-child.
func TestRunNode_RetriedDynamicChild_SubChildFailureReplaysSiblings(t *testing.T) {
	var log childAttemptLog
	sib := NewFunctionNode("sib", func(ctx agent.Context, _ any) (string, error) {
		log.add(ctx)
		return "s", nil
	}, NodeConfig{})
	var leafCalls int
	leaf := NewFunctionNode("leaf", func(ctx agent.Context, _ any) (string, error) {
		log.add(ctx)
		leafCalls++
		if leafCalls == 1 {
			return "", errAttempt
		}
		return "l", nil
	}, NodeConfig{})
	var midAttempts attemptLog
	middle := NewDynamicNode("middle", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		midAttempts.add("middle", agent.AttemptCount(ctx))
		s, err := RunNode[string](ctx, sib, nil)
		if err != nil {
			return "", err
		}
		if agent.AttemptCount(ctx) == 1 {
			if err := ctx.State().Set("mid_pending", "drop_me"); err != nil {
				return "", err
			}
		}
		l, err := RunNode[string](ctx, leaf, nil)
		if err != nil {
			return "", err
		}
		return s + ":" + l, nil
	}, NodeConfig{RetryConfig: retryNoDelay(3)})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return RunNode[string](ctx, middle, nil)
	}, NodeConfig{})

	res := runParent(t, parent)

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	const mid = "wf@1/parent@1/middle@1"
	if diff := cmp.Diff([]int{1, 2}, midAttempts.get("middle")); diff != "" {
		t.Errorf("middle's attempt numbers mismatch (-want +got):\n%s", diff)
	}
	want := []childAttempt{
		{Path: mid + "/sib@1", RunID: "1", Attempt: 1},
		{Path: mid + "/leaf@1", RunID: "1", Attempt: 1},
		{Path: mid + "/leaf@1", RunID: "1", Attempt: 1},
	}
	if diff := cmp.Diff(want, log.get()); diff != "" {
		t.Errorf("sub-child runs mismatch (-want +got):\n%s", diff)
	}
	var failedPaths []any
	for _, ev := range res.events {
		if ev.NodeInfo != nil && ev.NodeInfo.Path == mid && workflowNodeOutcome(ev) == workflowNodeFailureOutcome {
			failedPaths = toAnySlice(ev.CustomMetadata[workflowFailedChildPathsKey])
		}
	}
	if diff := cmp.Diff([]any{mid + "/leaf@1"}, failedPaths); diff != "" {
		t.Errorf("failed child paths on middle's failure record mismatch (-want +got):\n%s", diff)
	}
	if v, ok := storedState(res.events)["mid_pending"]; ok {
		t.Errorf("stored mid_pending = %v, want it absent", v)
	}
}

// toAnySlice returns v as a []any whether it holds []any or []string.
func toAnySlice(v any) []any {
	switch s := v.(type) {
	case []any:
		return s
	case []string:
		out := make([]any, len(s))
		for i, p := range s {
			out[i] = p
		}
		return out
	}
	return nil
}

// TestRunNode_RetriedDynamicChild_OwnErrorRunsSubChildrenAgain checks that a
// retried dynamic child that failed with its own error keeps counting its
// sub-children's run ids, so a completed sub-child runs again under a new
// path.
//
// Fails if the sub-children's run ids start at 1 again after every failure.
func TestRunNode_RetriedDynamicChild_OwnErrorRunsSubChildrenAgain(t *testing.T) {
	var log childAttemptLog
	leaf := NewFunctionNode("leaf", func(ctx agent.Context, _ any) (string, error) {
		log.add(ctx)
		return "l", nil
	}, NodeConfig{})
	middle := NewDynamicNode("middle", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if _, err := RunNode[string](ctx, leaf, nil); err != nil {
			return "", err
		}
		if agent.AttemptCount(ctx) < 3 {
			return "", errAttempt
		}
		return "m", nil
	}, NodeConfig{RetryConfig: retryNoDelay(3)})
	parent, got := callingParent(middle)

	runParent(t, parent)

	if r := receive(t, got); r.out != "m" || r.err != nil {
		t.Errorf("RunNode = (%v, %v), want (m, nil)", r.out, r.err)
	}
	const mid = "wf@1/parent@1/middle@1"
	want := []string{mid + "/leaf@1", mid + "/leaf@2", mid + "/leaf@3"}
	if diff := cmp.Diff(want, log.paths()); diff != "" {
		t.Errorf("leaf paths mismatch (-want +got):\n%s", diff)
	}
}
