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
	"fmt"
	"iter"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
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

	handler := func(ctx tool.Context, args Args) iter.Seq2[string, error] {
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

func TestHandleFunctionCalls_LiveControlPlane(t *testing.T) {
	type Args struct {
		DelayMS int `json:"delay_ms"`
	}

	var cancelCount int
	var cancelMu sync.Mutex

	handler := func(ctx tool.Context, args Args) iter.Seq2[string, error] {
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
		t.Errorf("expected exactly 2 goroutines to be cancelled, got: %d", gotCancels)
	}

	liveSess.mu.Lock()
	tasksAfter := liveSess.activeTools["count_forever"]
	liveSess.mu.Unlock()
	if len(tasksAfter) != 0 {
		t.Errorf("expected registry to be empty after cancellation, got: %v", tasksAfter)
	}
}
