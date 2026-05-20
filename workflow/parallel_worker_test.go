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
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

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
			wrapped: &parallelTestNode{
				BaseNode: NewBaseNode("upper", "", NodeConfig{}),
				fn: func(ctx agent.InvocationContext, input any) (any, error) {
					return strings.ToUpper(input.(string)), nil
				},
			},
			wantOutput: []any{"A", "B", "C"},
			wantErr:    false,
		},
		{
			name:           "EmptyList",
			maxConcurrency: 0,
			input:          []any{},
			wrapped: &parallelTestNode{
				BaseNode: NewBaseNode("upper", "", NodeConfig{}),
				fn: func(ctx agent.InvocationContext, input any) (any, error) {
					return strings.ToUpper(input.(string)), nil
				},
			},
			wantOutput: []any{},
			wantErr:    false,
		},
		{
			name:           "InvalidInput_NotSlice",
			maxConcurrency: 0,
			input:          "not a slice",
			wrapped: &parallelTestNode{
				BaseNode: NewBaseNode("upper", "", NodeConfig{}),
				fn: func(ctx agent.InvocationContext, input any) (any, error) {
					return strings.ToUpper(input.(string)), nil
				},
			},
			wantErr:   true,
			errSubstr: "expects a slice input",
		},
		{
			name:           "WorkerError",
			maxConcurrency: 0,
			input:          []any{"a", "b", "c"},
			wrapped: &parallelTestNode{
				BaseNode: NewBaseNode("error_node", "", NodeConfig{}),
				fn: func(ctx agent.InvocationContext, input any) (any, error) {
					s := input.(string)
					if s == "b" {
						return "", errors.New("failed on b")
					}
					return s, nil
				},
			},
			wantErr:   true,
			errSubstr: "failed on b",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pw := NewParallelWorker("parallel", tc.wrapped, tc.maxConcurrency, NodeConfig{})

			mockCtx := newMockCtx(t)
			events := pw.Run(mockCtx, tc.input)

			var gotOutput []any
			var gotErr error

			for ev, err := range events {
				if err != nil {
					gotErr = err
					break
				}
				if ev != nil && ev.Actions.StateDelta != nil {
					if out, ok := ev.Actions.StateDelta["output"]; ok {
						gotOutput = out.([]any)
					}
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

	wrapped := &parallelTestNode{
		BaseNode: NewBaseNode("blocking", "", NodeConfig{}),
		fn: func(ctx agent.InvocationContext, input any) (any, error) {
			atomic.AddInt32(&counter, 1)
			<-blockCh
			return input, nil
		},
	}

	pw := NewParallelWorker("parallel", wrapped, 2, NodeConfig{})

	mockCtx := newMockCtx(t)
	input := []any{1, 2, 3, 4}

	done := make(chan struct{})
	go func() {
		for range pw.Run(mockCtx, input) {
		}
		close(done)
	}()

	// Wait a bit for workers to start.
	// We expect at most 2 workers to start because maxConcurrency is 2.
	time.Sleep(100 * time.Millisecond)

	c := atomic.LoadInt32(&counter)
	if c != 2 {
		t.Errorf("expected counter to be 2, got %d", c)
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
	wrapped := &customEventNode{}

	pw := NewParallelWorker("parallel", wrapped, 0, NodeConfig{})

	mockCtx := newMockCtx(t)
	input := []any{1, 2}

	events := pw.Run(mockCtx, input)

	nonOutputCount := 0
	hasAggregatedOutput := false

	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev != nil {
			if out, ok := ev.Actions.StateDelta["output"]; ok {
				hasAggregatedOutput = true
				wantOutput := []any{1, 2}
				if diff := cmp.Diff(wantOutput, out); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			} else if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "progress" {
				nonOutputCount++
			}
		}
	}

	if nonOutputCount != 0 {
		t.Errorf("expected 0 progress events, got %d", nonOutputCount)
	}
	if !hasAggregatedOutput {
		t.Error("expected final aggregated output event")
	}
}

type parallelTestNode struct {
	BaseNode
	fn func(ctx agent.InvocationContext, input any) (any, error)
}

func (n *parallelTestNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		out, err := n.fn(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}
		ev := session.NewEvent(ctx.InvocationID())
		ev.Actions.StateDelta["output"] = out
		yield(ev, nil)
	}
}

type customEventNode struct {
	BaseNode
}

func (n *customEventNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev1 := session.NewEvent(ctx.InvocationID())
		ev1.Content = &genai.Content{Parts: []*genai.Part{{Text: "progress"}}}
		if !yield(ev1, nil) {
			return
		}

		ev2 := session.NewEvent(ctx.InvocationID())
		ev2.Actions.StateDelta["output"] = input
		yield(ev2, nil)
	}
}

func (n *customEventNode) Name() string        { return "custom" }
func (n *customEventNode) Description() string { return "" }
func (n *customEventNode) Config() NodeConfig  { return NodeConfig{} }

func TestParallelWorker_MultipleOutputsPerWorker(t *testing.T) {
	wrapped := &multiOutputTestNode{}

	pw := NewParallelWorker("parallel", wrapped, 0, NodeConfig{})

	mockCtx := newMockCtx(t)
	input := []any{"a", "b"}

	events := pw.Run(mockCtx, input)

	var gotOutput []any
	var gotErr error

	for ev, err := range events {
		if err != nil {
			gotErr = err
			break
		}
		if ev != nil && ev.Actions.StateDelta != nil {
			if out, ok := ev.Actions.StateDelta["output"]; ok {
				gotOutput = out.([]any)
			}
		}
	}

	if gotErr != nil {
		t.Fatalf("unexpected error: %v", gotErr)
	}

	wantOutput := []any{
		[]any{"a", "a_2"},
		[]any{"b", "b_2"},
	}

	if diff := cmp.Diff(wantOutput, gotOutput); diff != "" {
		t.Errorf("output mismatch (-want +got):\n%s", diff)
	}
}

type multiOutputTestNode struct {
	BaseNode
}

func (n *multiOutputTestNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev1 := session.NewEvent(ctx.InvocationID())
		ev1.Actions.StateDelta["output"] = input
		if !yield(ev1, nil) {
			return
		}

		ev2 := session.NewEvent(ctx.InvocationID())
		ev2.Actions.StateDelta["output"] = fmt.Sprintf("%v_2", input)
		yield(ev2, nil)
	}
}

func (n *multiOutputTestNode) Name() string        { return "multi_output" }
func (n *multiOutputTestNode) Description() string { return "" }
func (n *multiOutputTestNode) Config() NodeConfig  { return NodeConfig{} }
