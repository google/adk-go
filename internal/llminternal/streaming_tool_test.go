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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
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

func TestHandleFunctionCalls_StreamingToolErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		handler functiontool.StreamingFunc[struct{}]
		want    string
	}{
		{
			name: "panic is hidden from the model",
			handler: func(agent.Context, struct{}) iter.Seq2[string, error] {
				return func(func(string, error) bool) { panic("boom") }
			},
			want: "Invoking `test_stream()` failed with an internal error.",
		},
		{
			name: "returned error reaches the model",
			handler: func(agent.Context, struct{}) iter.Seq2[string, error] {
				return func(yield func(string, error) bool) { yield("", errors.New("order not found")) }
			},
			want: "order not found",
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

			mergedEvent, err := (&Flow{}).handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, resp, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse.Response["error"]; got != tc.want {
				t.Errorf("FunctionResponse error = %q, want %q", got, tc.want)
			}
		})
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

// argsCapturingStreamTool sends the args RunStream is called with to gotArgs.
type argsCapturingStreamTool struct {
	gotArgs chan any
}

func (t *argsCapturingStreamTool) Name() string        { return "capture_stream" }
func (t *argsCapturingStreamTool) Description() string { return "captures its args" }
func (t *argsCapturingStreamTool) IsLongRunning() bool { return false }

func (t *argsCapturingStreamTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name(), Description: t.Description()}
}

func (t *argsCapturingStreamTool) RunStream(_ agent.Context, args any) iter.Seq2[string, error] {
	t.gotArgs <- args
	return func(yield func(string, error) bool) {}
}

// A live-mode call without args reaches the streaming tool as an empty map.
func TestHandleFunctionCalls_NoArgsReachLiveStreamingToolAsEmptyMap(t *testing.T) {
	st := &argsCapturingStreamTool{gotArgs: make(chan any, 1)}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
	resp := &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: st.Name()}}},
	}}

	_, err := (&Flow{}).handleFunctionCalls(invCtx, map[string]tool.Tool{st.Name(): st}, resp, nil, &mockLiveSession{})
	if err != nil {
		t.Fatal(err)
	}

	select {
	case got := <-st.gotArgs:
		if diff := cmp.Diff(any(map[string]any{}), got); diff != "" {
			t.Errorf("live streaming tool args mismatch (-want +got):\n%s", diff)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("streaming tool was not run")
	}
}

// noRunTool is a tool that is neither a function tool nor a streaming one.
type noRunTool struct{}

func (noRunTool) Name() string        { return "no_run" }
func (noRunTool) Description() string { return "has no Run method" }
func (noRunTool) IsLongRunning() bool { return false }

// A call without args to a tool that cannot be run reaches the on-tool-error
// callback as an empty map.
func TestHandleFunctionCalls_NoArgsReachOnToolErrorForNonFunctionTool(t *testing.T) {
	var gotArgs map[string]any
	called := false
	flow := &Flow{
		OnToolErrorCallbacks: []OnToolErrorCallback{
			func(_ agent.Context, _ tool.Tool, args map[string]any, _ error) (map[string]any, error) {
				called, gotArgs = true, args
				return map[string]any{"handled": true}, nil
			},
		},
	}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
	resp := &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "no_run"}}},
	}}

	_, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"no_run": noRunTool{}}, resp, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !called {
		t.Fatal("on-tool-error callback not called")
	}
	if diff := cmp.Diff(map[string]any{}, gotArgs); diff != "" {
		t.Errorf("on-tool-error args mismatch (-want +got):\n%s", diff)
	}
}
