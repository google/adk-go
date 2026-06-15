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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

var upperNode = NewFunctionNode("upper", func(ctx agent.Context, input string) (string, error) {
	return strings.ToUpper(input), nil
}, defaultNodeConfig)

func TestParallelWorker_Run(t *testing.T) {
	tests := []struct {
		name           string
		maxConcurrency int
		input          any
		wrapped        Node
		wantOutput     []any
		wantErr        bool
		errSubstr      string
	}{
		{
			name:           "Success",
			maxConcurrency: 0,
			input:          []any{"a", "b", "c"},
			wrapped:        upperNode,
			wantOutput:     []any{"A", "B", "C"},
			wantErr:        false,
		},
		{
			name:           "EmptyList",
			maxConcurrency: 0,
			input:          []any{},
			wrapped:        upperNode,
			wantOutput:     []any{},
			wantErr:        false,
		},
		{
			name:           "InvalidInput_NotSlice",
			maxConcurrency: 0,
			input:          "not a slice",
			wrapped:        upperNode,
			wantErr:        true,
			errSubstr:      "expects a slice input",
		},
		{
			name:           "WorkerError",
			maxConcurrency: 0,
			input:          []any{"a", "b", "c"},
			wrapped: NewFunctionNode("error_node", func(ctx agent.Context, input string) (string, error) {
				if input == "b" {
					return "", errors.New("failed on b")
				}
				return input, nil
			}, defaultNodeConfig),
			wantErr:   true,
			errSubstr: "failed on b",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pw, err := NewParallelWorker("parallel", tc.wrapped, tc.maxConcurrency, defaultNodeConfig)
			if err != nil {
				t.Fatal(err)
			}

			mockCtx := newMockCtx(t)
			exCtx := agent.NewNodeContext(mockCtx, nil)
			events := pw.Run(exCtx, tc.input)

			var gotOutput []any
			var gotErr error

			for ev, err := range events {
				if err != nil {
					gotErr = err
					break
				}
				if out, ok := extractOutput(ev); ok {
					gotOutput = out.([]any)
				}
			}

			if tc.wantErr {
				if gotErr == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.errSubstr != "" && !strings.Contains(gotErr.Error(), tc.errSubstr) {
					t.Errorf("expected error containing %q, got %v", tc.errSubstr, gotErr)
				}
			} else {
				if gotErr != nil {
					t.Fatalf("unexpected error: %v", gotErr)
				}
				if diff := cmp.Diff(tc.wantOutput, gotOutput); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestParallelWorker_Concurrency(t *testing.T) {
	var counter int32
	blockCh := make(chan struct{})

	startedCh := make(chan struct{}, 4)
	wrapped := NewFunctionNode("blocking", func(ctx agent.Context, input int) (int, error) {
		atomic.AddInt32(&counter, 1)
		startedCh <- struct{}{}
		<-blockCh
		return input, nil
	}, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 2, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewNodeContext(mockCtx, nil)
	input := []any{1, 2, 3, 4}

	done := make(chan struct{})
	go func() {
		for range pw.Run(exCtx, input) {
		}
		close(done)
	}()

	// Wait for 2 workers to start.
	// We expect at most 2 workers to start because maxConcurrency is 2.
	<-startedCh
	<-startedCh

	c := atomic.LoadInt32(&counter)
	if c != 2 {
		t.Errorf("expected counter to be 2, got %d", c)
	}

	// Verify no 3rd worker started
	select {
	case <-startedCh:
		t.Error("expected only 2 workers to start, but more did")
	default:
	}

	// Unblock workers
	close(blockCh)

	// Wait for completion
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for execution to complete")
	}

	// After unblocking, all workers should have run
	c = atomic.LoadInt32(&counter)
	if c != 4 {
		t.Errorf("expected final counter to be 4, got %d", c)
	}
}

func TestParallelWorker_SuppressIntermediateEvents(t *testing.T) {
	wrapped := NewFunctionNode("wrapped", func(ctx agent.Context, input any) (any, error) { return input, nil }, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewNodeContext(mockCtx, nil)
	input := []any{1, 2}

	events := pw.Run(exCtx, input)

	nonOutputCount := 0
	hasAggregatedOutput := false

	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out, ok := extractOutput(ev); ok {
			hasAggregatedOutput = true
			wantOutput := []any{1, 2}
			if diff := cmp.Diff(wantOutput, out); diff != "" {
				t.Errorf("output mismatch (-want +got):\n%s", diff)
			}
		} else if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "progress" {
			nonOutputCount++
		}
	}

	if nonOutputCount != 0 {
		t.Errorf("expected 0 progress events, got %d", nonOutputCount)
	}
	if !hasAggregatedOutput {
		t.Error("expected final aggregated output event")
	}
}

func TestParallelWorker_WorkflowIntegration(t *testing.T) {
	splitFn := func(ctx agent.Context, input string) ([]any, error) {
		parts := strings.Split(input, ",")
		var res []any
		for _, p := range parts {
			res = append(res, p)
		}
		return res, nil
	}
	splitNode := NewFunctionNode("split", splitFn, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", upperNode, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	joinFn := func(ctx agent.Context, input []any) (string, error) {
		var strs []string
		for _, v := range input {
			strs = append(strs, v.(string))
		}
		return strings.Join(strs, ":"), nil
	}
	joinNode := NewFunctionNode("join", joinFn, defaultNodeConfig)

	edges := []Edge{
		{From: Start, To: splitNode},
		{From: splitNode, To: pw},
		{From: pw, To: joinNode},
	}

	w := mustNew(t, edges)

	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "a,b,c"}},
	}

	events := w.Run(mockCtx)

	var lastOutput any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if out, ok := extractOutput(ev); ok {
			lastOutput = out
		}
	}

	wantOutput := "A:B:C"
	if lastOutput != wantOutput {
		t.Errorf("expected output %q, got %v", wantOutput, lastOutput)
	}
}

func TestNewParallelWorker_ErrorOnWrappedRetryConfig(t *testing.T) {
	wrapped := NewFunctionNode("wrapped", func(ctx agent.Context, input any) (any, error) { return input, nil }, NodeConfig{RetryConfig: DefaultRetryConfig()})

	_, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err == nil {
		t.Fatal("expected error when wrapped node has RetryConfig, got nil")
	}
	if !strings.Contains(err.Error(), "cannot have RetryConfig") {
		t.Errorf("expected error containing 'cannot have RetryConfig', got %v", err)
	}
}

func TestParallelWorker_Retry(t *testing.T) {
	var mu sync.Mutex
	attempts := make(map[string]int)

	wrapped := NewFunctionNode("retry_node", func(ctx agent.Context, input string) (string, error) {
		mu.Lock()
		attempts[input]++
		count := attempts[input]
		mu.Unlock()

		if input == "b" && count <= 2 {
			return "", errors.New("temporary failure")
		}
		return input, nil
	}, defaultNodeConfig)

	rc := DefaultRetryConfig()
	rc.MaxAttempts = 3
	rc.InitialDelay = 0
	rc.MaxDelay = 0
	rc.Jitter = 0
	rc.ShouldRetry = func(err error) bool { return true }

	pw, err := NewParallelWorker("parallel", wrapped, 0, NodeConfig{RetryConfig: rc})
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewNodeContext(mockCtx, nil)
	input := []any{"a", "b"}

	events := pw.Run(exCtx, input)

	var gotOutput []any
	var gotErr error

	for ev, err := range events {
		if err != nil {
			gotErr = err
			break
		}
		if out, ok := extractOutput(ev); ok {
			gotOutput = out.([]any)
		}
	}

	if gotErr != nil {
		t.Fatalf("unexpected error: %v", gotErr)
	}

	wantOutput := []any{"a", "b"}
	if diff := cmp.Diff(wantOutput, gotOutput); diff != "" {
		t.Errorf("output mismatch (-want +got):\n%s", diff)
	}

	mu.Lock()
	countA := attempts["a"]
	countB := attempts["b"]
	mu.Unlock()

	if countA != 1 {
		t.Errorf("expected 1 attempt for 'a', got %d", countA)
	}
	if countB != 3 {
		t.Errorf("expected 3 attempts for 'b', got %d", countB)
	}
}

func TestParallelWorker_FailFast(t *testing.T) {
	var workerCCancelled int32
	cancelledCh := make(chan struct{})

	wrapped := NewFunctionNode("fail_fast_node", func(ctx agent.Context, input string) (string, error) {
		if input == "b" {
			return "", errors.New("error b")
		}
		if input == "c" {
			// Block until cancelled
			<-ctx.Done()
			atomic.StoreInt32(&workerCCancelled, 1)
			close(cancelledCh)
			return "", ctx.Err()
		}
		return input, nil
	}, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	input := []any{"a", "b", "c"}
	exCtx := agent.NewNodeContext(mockCtx, nil)
	events := pw.Run(exCtx, input)

	var gotErr error
	for _, err := range events {
		if err != nil {
			gotErr = err
			break
		}
	}

	if gotErr == nil {
		t.Fatal("expected error, got nil")
	}

	if gotErr.Error() != "error b" {
		t.Errorf("expected error 'error b', got %v", gotErr)
	}

	// Wait for worker C to observe cancellation
	select {
	case <-cancelledCh:
		// Good
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for worker C to be cancelled")
	}

	if atomic.LoadInt32(&workerCCancelled) != 1 {
		t.Error("expected worker c to be cancelled")
	}
}

func TestParallelWorker_CancelDuringExecution(t *testing.T) {
	blockCh := make(chan struct{})
	startedCh := make(chan struct{}, 2)
	wrapped := NewFunctionNode("blocking", func(ctx agent.Context, input any) (any, error) {
		startedCh <- struct{}{}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-blockCh:
			return input, nil
		}
	}, defaultNodeConfig)

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	mockCtx := &MockInvocationContext{Context: ctx}
	exCtx := agent.NewNodeContext(mockCtx, nil)
	input := []any{1, 2}

	done := make(chan struct{})
	var gotErr error
	var hasFinalResult bool

	go func() {
		for ev, err := range pw.Run(exCtx, input) {
			if err != nil {
				gotErr = err
			}
			if _, ok := extractOutput(ev); ok {
				hasFinalResult = true
			}
		}
		close(done)
	}()

	// Wait for workers to start before cancelling
	<-startedCh
	<-startedCh
	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for cancellation to complete")
	}

	if gotErr == nil {
		t.Error("expected error on cancellation, got nil")
	}
	if !errors.Is(gotErr, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", gotErr)
	}
	if hasFinalResult {
		t.Error("expected no final result to be yielded")
	}

	close(blockCh)
}

func TestParallelWorker_ConcurrentMultiOutputOrder(t *testing.T) {
	releaseA := make(chan struct{})
	releaseB := make(chan struct{})
	releaseC := make(chan struct{})
	startedCh := make(chan struct{}, 3)

	wrapped := &delayedMultiOutputTestNode{
		releaseChans: map[string]chan struct{}{
			"a": releaseA,
			"b": releaseB,
			"c": releaseC,
		},
		startedCh: startedCh,
	}

	pw, err := NewParallelWorker("parallel", wrapped, 0, defaultNodeConfig)
	if err != nil {
		t.Fatal(err)
	}

	mockCtx := newMockCtx(t)
	exCtx := agent.NewNodeContext(mockCtx, nil)
	input := []any{"a", "b", "c"}

	done := make(chan struct{})
	var gotOutput []any
	go func() {
		events := pw.Run(exCtx, input)
		for ev, err := range events {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if out, ok := extractOutput(ev); ok {
				gotOutput = out.([]any)
			}
		}
		close(done)
	}()

	// Wait for all workers to start
	<-startedCh
	<-startedCh
	<-startedCh

	// Release in reverse order to simulate out-of-order completion
	close(releaseC)
	close(releaseB)
	close(releaseA)

	<-done

	wantOutput := []any{
		[]any{"a", "a_2"},
		[]any{"b", "b_2"},
		[]any{"c", "c_2"},
	}

	if diff := cmp.Diff(wantOutput, gotOutput); diff != "" {
		t.Errorf("output mismatch (-want +got):\n%s", diff)
	}
}

type delayedMultiOutputTestNode struct {
	BaseNode
	releaseChans map[string]chan struct{}
	startedCh    chan struct{}
}

func (n *delayedMultiOutputTestNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		s := input.(string)

		if n.startedCh != nil {
			n.startedCh <- struct{}{}
		}
		if ch, ok := n.releaseChans[s]; ok {
			<-ch
		}

		ev1 := session.NewEvent(ctx.InvocationID())
		ev1.Output = s
		if !yield(ev1, nil) {
			return
		}

		ev2 := session.NewEvent(ctx.InvocationID())
		ev2.Output = fmt.Sprintf("%v_2", s)
		yield(ev2, nil)
	}
}

func (n *delayedMultiOutputTestNode) Name() string        { return "delayed_multi_output" }
func (n *delayedMultiOutputTestNode) Description() string { return "" }
func (n *delayedMultiOutputTestNode) Config() NodeConfig  { return defaultNodeConfig }

func TestParallelWorker_SchedulerDoesNotRetryOnFailure(t *testing.T) {
	var wrappedAttempts int32

	wrapped := NewFunctionNode("worker", func(ctx agent.Context, input string) (string, error) {
		atomic.AddInt32(&wrappedAttempts, 1)
		return "", errors.New("persistent failure")
	}, defaultNodeConfig)

	rc := DefaultRetryConfig()
	rc.MaxAttempts = 2
	rc.InitialDelay = 0
	rc.MaxDelay = 0
	rc.Jitter = 0

	pw, err := NewParallelWorker("parallel", wrapped, 0, NodeConfig{RetryConfig: rc})
	if err != nil {
		t.Fatal(err)
	}

	splitFn := func(ctx agent.Context, input string) ([]any, error) {
		return []any{input}, nil
	}
	splitNode := NewFunctionNode("split", splitFn, defaultNodeConfig)

	edges := []Edge{
		{From: Start, To: splitNode},
		{From: splitNode, To: pw},
	}

	w := mustNew(t, edges)

	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "a"}},
	}

	events := w.Run(mockCtx)

	var gotErr error
	for _, err := range events {
		if err != nil {
			gotErr = err
			break
		}
	}

	if gotErr == nil {
		t.Fatal("expected error, got nil")
	}

	if atomic.LoadInt32(&wrappedAttempts) != 2 {
		t.Errorf("expected 2 attempts for wrapped node, got %d (scheduler likely retried the node)", atomic.LoadInt32(&wrappedAttempts))
	}
}
