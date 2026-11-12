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

package llminternal

import (
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type mockLiveSession struct {
	sendFunc func(agent.LiveRequest) error
}

func (m *mockLiveSession) Send(req agent.LiveRequest) error {
	if m.sendFunc != nil {
		return m.sendFunc(req)
	}
	return nil
}

func (m *mockLiveSession) Close() error { return nil }

func TestHandleFunctionCalls_Streaming(t *testing.T) {
	type Args struct {
		Count int `json:"count"`
	}

	handler := func(ctx agent.Context, args Args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			for i := 0; i < args.Count; i++ {
				if !yield(fmt.Sprintf("chunk %d", i), nil) {
					return
				}
			}
		}
	}

	streamTool, err := functiontool.NewStreaming(functiontool.Config{
		Name:        "test_stream",
		Description: "streams chunks",
	}, handler)
	if err != nil {
		t.Fatal(err)
	}

	toolsDict := map[string]tool.Tool{
		"test_stream": streamTool,
	}

	resp := &model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_1",
						Name: "test_stream",
						Args: map[string]any{"count": 3},
					},
				},
			},
			Role: "model",
		},
	}

	t.Run("Live Mode (Streaming)", func(t *testing.T) {
		var receivedChunks []string
		var mu sync.Mutex
		var wg sync.WaitGroup
		wg.Add(3) // We expect 3 chunks

		mockSess := &mockLiveSession{
			sendFunc: func(req agent.LiveRequest) error {
				mu.Lock()
				defer mu.Unlock()
				if req.Content != nil && len(req.Content.Parts) > 0 {
					receivedChunks = append(receivedChunks, req.Content.Parts[0].Text)
				}
				wg.Done()
				return nil
			},
		}

		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		})

		flow := &Flow{
			Tools: []tool.Tool{streamTool},
		}

		mergedEvent, err := flow.handleFunctionCalls(invCtx, toolsDict, resp, nil, mockSess)
		if err != nil {
			t.Fatal(err)
		}

		// Verify immediate response is pending
		if mergedEvent == nil {
			t.Fatal("expected non-nil mergedEvent")
		}
		if len(mergedEvent.LLMResponse.Content.Parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(mergedEvent.LLMResponse.Content.Parts))
		}
		respPart := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse
		if respPart == nil {
			t.Fatal("expected FunctionResponse part")
		}
		status, ok := respPart.Response["status"].(string)
		if !ok || status != "The function is running asynchronously and the results are pending." {
			t.Errorf("unexpected status: %v", respPart.Response["status"])
		}

		// Wait for background streaming to complete
		wg.Wait()

		wantChunks := []string{
			"Function test_stream returned: chunk 0",
			"Function test_stream returned: chunk 1",
			"Function test_stream returned: chunk 2",
		}
		if diff := cmp.Diff(wantChunks, receivedChunks); diff != "" {
			t.Errorf("unexpected chunks (-want +got):\n%s", diff)
		}
	})

	t.Run("Non-Live Mode (Aggregation)", func(t *testing.T) {
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		})

		flow := &Flow{
			Tools: []tool.Tool{streamTool},
		}

		mergedEvent, err := flow.handleFunctionCalls(invCtx, toolsDict, resp, nil, nil)
		if err != nil {
			t.Fatal(err)
		}

		if mergedEvent == nil {
			t.Fatal("expected non-nil mergedEvent")
		}
		if len(mergedEvent.LLMResponse.Content.Parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(mergedEvent.LLMResponse.Content.Parts))
		}
		respPart := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse
		if respPart == nil {
			t.Fatal("expected FunctionResponse part")
		}
		result, ok := respPart.Response["result"].(string)
		if !ok || result != "chunk 0chunk 1chunk 2" {
			t.Errorf("unexpected result: %v", respPart.Response["result"])
		}
	})
}

// A streaming tool that fails reports the failure the way a non-streaming
// one does: the tool error callbacks see the error, and the model reads
// what they answer, or a generic message when none of them answers. The
// error a callback sees carries the panic value and the stack of a
// recovered panic, which the model never reads.
func TestHandleFunctionCalls_StreamingToolErrors(t *testing.T) {
	panicking := func(agent.Context, struct{}) iter.Seq2[string, error] {
		return func(func(string, error) bool) { panic("boom") }
	}
	for _, tc := range []struct {
		name string
		// handler is the streaming tool.
		handler functiontool.StreamingFunc[struct{}]
		// callback is the tool error callback, when the case has one.
		callback OnToolErrorCallback
		// want is the function response the model reads.
		want map[string]any
		// wantSeen is a substring the error the callback saw must contain.
		wantSeen string
		// wantAfterTool says whether the after-tool callback has to run.
		wantAfterTool bool
	}{
		{
			name:          "panic is hidden from the model",
			handler:       panicking,
			want:          map[string]any{"error": "Invoking `test_stream()` failed with an internal error."},
			wantAfterTool: true,
		},
		{
			name: "returned error reaches the model",
			handler: func(agent.Context, struct{}) iter.Seq2[string, error] {
				return func(yield func(string, error) bool) { yield("", errors.New("order not found")) }
			},
			want:          map[string]any{"error": "order not found"},
			wantAfterTool: true,
		},
		{
			name:    "the callback sees the panic and the stack",
			handler: panicking,
			callback: func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return nil, nil
			},
			want:          map[string]any{"error": "Invoking `test_stream()` failed with an internal error."},
			wantSeen:      "boom",
			wantAfterTool: true,
		},
		{
			name:    "a callback that answers replaces the error",
			handler: panicking,
			callback: func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return map[string]any{"retry": "please try again"}, nil
			},
			want:          map[string]any{"retry": "please try again"},
			wantAfterTool: true,
		},
		{
			name:    "a callback that fails replaces the error",
			handler: panicking,
			callback: func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return nil, errors.New("callback failed")
			},
			want:          map[string]any{"error": "callback failed"},
			wantAfterTool: true,
		},
		{
			name: "a successful streaming tool runs the after-tool callback",
			handler: func(agent.Context, struct{}) iter.Seq2[string, error] {
				return func(yield func(string, error) bool) {
					yield("chunk ", nil)
					yield("two", nil)
				}
			},
			want:          map[string]any{"result": "chunk two"},
			wantAfterTool: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			streamTool, err := functiontool.NewStreaming(functiontool.Config{Name: "test_stream"}, tc.handler)
			if err != nil {
				t.Fatal(err)
			}
			resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "test_stream", Args: map[string]any{}},
			}}}}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				InvocationID: "inv_1",
				Agent:        &mockAgent{name: "agent_1"},
			})

			flow := &Flow{}
			var seen error
			afterToolCalls := 0
			flow.AfterToolCallbacks = []AfterToolCallback{
				func(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					afterToolCalls++
					return nil, nil
				},
			}
			if tc.callback != nil {
				callback := tc.callback
				flow.OnToolErrorCallbacks = []OnToolErrorCallback{
					func(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
						seen = err
						return callback(ctx, t, args, err)
					},
				}
			}

			mergedEvent, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, resp, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse.Response
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("FunctionResponse mismatch (-want +got):\n%s", diff)
			}
			if ran, want := afterToolCalls > 0, tc.wantAfterTool; ran != want {
				t.Errorf("after-tool callback ran = %t; want %t", ran, want)
			}
			if tc.wantSeen != "" {
				if seen == nil {
					t.Fatal("the callback did not receive an error")
				}
				if !strings.Contains(seen.Error(), tc.wantSeen) {
					t.Errorf("the error the callback saw = %q, want it to contain %q", seen, tc.wantSeen)
				}
				var panicErr *toolinternal.PanicError
				if !errors.As(seen, &panicErr) {
					t.Errorf("the error the callback saw = %v, want a *toolinternal.PanicError", seen)
				}
			}
		})
	}
}

// The callback chain runs only after the tool's iterator has unwound. A
// callback that runs inside the loop would see a tool that has not run its
// deferred cleanup yet, so a callback taking the resource the tool releases
// in that cleanup would deadlock; a non-streaming tool's callbacks likewise
// run after Run has returned.
func TestHandleFunctionCalls_StreamingToolCallbacksRunAfterTheToolReturns(t *testing.T) {
	var finished bool
	var toolFinishedWhenCalled []bool

	streamTool, err := functiontool.NewStreaming(functiontool.Config{Name: "test_stream"}, func(agent.Context, struct{}) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			defer func() { finished = true }()
			yield("", errors.New("order not found"))
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	flow := &Flow{
		OnToolErrorCallbacks: []OnToolErrorCallback{
			func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				toolFinishedWhenCalled = append(toolFinishedWhenCalled, finished)
				return nil, nil
			},
		},
		AfterToolCallbacks: []AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				toolFinishedWhenCalled = append(toolFinishedWhenCalled, finished)
				return nil, nil
			},
		},
	}

	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "test_stream", Args: map[string]any{}},
	}}}}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})

	if _, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, resp, nil, nil); err != nil {
		t.Fatal(err)
	}

	if len(toolFinishedWhenCalled) != 2 {
		t.Fatalf("callbacks that ran = %d; want 2", len(toolFinishedWhenCalled))
	}
	for i, seen := range toolFinishedWhenCalled {
		if !seen {
			t.Errorf("callback %d ran before the tool iterator finished", i)
		}
	}
}

// A successful non-live stream runs the after-tool chain with the completed
// result and a nil error, so a plugin that resets its failure count there
// treats a streaming success like a non-streaming one.
func TestHandleFunctionCalls_SuccessfulStreamingToolRunsAfterTool(t *testing.T) {
	streamTool, err := functiontool.NewStreaming(functiontool.Config{Name: "test_stream"}, func(agent.Context, struct{}) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			yield("chunk ", nil)
			yield("two", nil)
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotResult map[string]any
	var gotErr error
	flow := &Flow{
		AfterToolCallbacks: []AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
				gotResult, gotErr = result, err
				return nil, nil
			},
		},
	}

	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "test_stream", Args: map[string]any{}},
	}}}}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})

	mergedEvent, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, resp, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if gotErr != nil {
		t.Errorf("after-tool callback error = %v; want nil", gotErr)
	}
	if diff := cmp.Diff(map[string]any{"result": "chunk two"}, gotResult); diff != "" {
		t.Errorf("after-tool callback result mismatch (-want +got):\n%s", diff)
	}
	got := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse.Response
	if diff := cmp.Diff(map[string]any{"result": "chunk two"}, got); diff != "" {
		t.Errorf("FunctionResponse mismatch (-want +got):\n%s", diff)
	}
}

func TestHandleFunctionCalls_LiveControlPlane(t *testing.T) {
	type Args struct {
		DelayMS int `json:"delay_ms"`
	}

	var cancelCount int
	var cancelMu sync.Mutex

	handler := func(ctx agent.Context, args Args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			for i := 0; ; i++ {
				time.Sleep(time.Millisecond)
				select {
				case <-ctx.Done():
					cancelMu.Lock()
					cancelCount++
					cancelMu.Unlock()
					return
				default:
				}
				if !yield(fmt.Sprintf("number %d", i), nil) {
					cancelMu.Lock()
					cancelCount++
					cancelMu.Unlock()
					return
				}
			}
		}
	}

	streamTool, err := functiontool.NewStreaming(functiontool.Config{
		Name:        "count_forever",
		Description: "counts indefinitely",
	}, handler)
	if err != nil {
		t.Fatal(err)
	}

	toolsDict := map[string]tool.Tool{
		"count_forever": streamTool,
	}

	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})

	flow := &Flow{
		Tools: []tool.Tool{streamTool},
	}

	respStart := &model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_forever_1",
						Name: "count_forever",
						Args: map[string]any{"delay_ms": 1},
					},
				},
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_forever_2",
						Name: "count_forever",
						Args: map[string]any{"delay_ms": 1},
					},
				},
			},
			Role: "model",
		},
	}

	liveSess := newLiveSessionImpl()
	liveSess.activeTools = make(map[string][]activeTask)

	go func() {
		for range liveSess.inputCh {
		}
	}()

	_, err = flow.handleFunctionCalls(invCtx, toolsDict, respStart, nil, liveSess)
	if err != nil {
		t.Fatal(err)
	}

	liveSess.mu.Lock()
	tasks, exists := liveSess.activeTools["count_forever"]
	liveSess.mu.Unlock()
	if !exists || len(tasks) != 2 {
		t.Fatalf("expected 2 active tasks, found: %v", tasks)
	}

	respStop := &model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_stop_1",
						Name: "stop_streaming",
						Args: map[string]any{"function_name": "count_forever"},
					},
				},
			},
			Role: "model",
		},
	}

	mergedStopEvent, err := flow.handleFunctionCalls(invCtx, toolsDict, respStop, nil, liveSess)
	if err != nil {
		t.Fatal(err)
	}

	if mergedStopEvent == nil {
		t.Fatal("expected non-nil mergedStopEvent")
	}
	respPart := mergedStopEvent.LLMResponse.Content.Parts[0].FunctionResponse
	if respPart == nil {
		t.Fatal("expected FunctionResponse part")
	}
	status, ok := respPart.Response["status"].(string)
	if !ok || status != "Successfully stopped all running instances of count_forever" {
		t.Errorf("unexpected stop status: %s", status)
	}

	time.Sleep(50 * time.Millisecond)

	cancelMu.Lock()
	gotCancels := cancelCount
	cancelMu.Unlock()
	if gotCancels != 2 {
		t.Errorf("expected exactly 2 goroutines to be canceled, got: %d", gotCancels)
	}

	liveSess.mu.Lock()
	tasksAfter := liveSess.activeTools["count_forever"]
	liveSess.mu.Unlock()
	if len(tasksAfter) != 0 {
		t.Errorf("expected registry to be empty after cancellation, got: %v", tasksAfter)
	}
}
