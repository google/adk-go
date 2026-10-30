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
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
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

// fixedToolset hands out the same tools on every call.
type fixedToolset []tool.Tool

func (fixedToolset) Name() string { return "fixed" }

func (ts fixedToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return ts, nil }

// TestHandleFunctionCalls_LiveConfirmationGate checks that a streaming call the
// confirmation gate stops in live mode is refused on the calling goroutine. The
// model gets an error rather than a pending status, the confirmation request is
// on the response event, and the stream never starts. Each response carries two
// calls, which the flow runs concurrently, so -race sees two gates and two
// response events being built at once.
func TestHandleFunctionCalls_LiveConfirmationGate(t *testing.T) {
	type Args struct{}
	const toolName = "gated_stream"

	newStreamingTool := func(t *testing.T, cfg functiontool.Config, handlerRan *atomic.Bool) tool.Tool {
		t.Helper()
		cfg.Name = toolName
		st, err := functiontool.NewStreaming(cfg, func(agent.Context, Args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) {
				handlerRan.Store(true)
				yield("chunk", nil)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		return st
	}
	wrap := func(t *testing.T, inner tool.Tool, requireConfirmation bool) tool.Tool {
		t.Helper()
		tools, err := tool.WithConfirmation(fixedToolset{inner}, requireConfirmation, nil).Tools(nil)
		if err != nil {
			t.Fatal(err)
		}
		return tools[0]
	}

	tests := []struct {
		name          string
		build         func(t *testing.T, handlerRan *atomic.Bool) tool.Tool
		confirmations map[string]*toolconfirmation.ToolConfirmation
		// wantErr is the sentinel each call's error response wraps. Nil means
		// the call is allowed and streams.
		wantErr       error
		wantRequested bool
	}{
		{
			name: "tool requires confirmation",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return newStreamingTool(t, functiontool.Config{RequireConfirmation: true}, handlerRan)
			},
			wantErr:       tool.ErrConfirmationRequired,
			wantRequested: true,
		},
		{
			name: "provider requires confirmation",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return newStreamingTool(t, functiontool.Config{
					RequireConfirmationProvider: func(Args) bool { return true },
				}, handlerRan)
			},
			wantErr:       tool.ErrConfirmationRequired,
			wantRequested: true,
		},
		{
			name: "toolset requires confirmation",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return wrap(t, newStreamingTool(t, functiontool.Config{}, handlerRan), true)
			},
			wantErr:       tool.ErrConfirmationRequired,
			wantRequested: true,
		},
		{
			name: "tool requires confirmation inside a toolset that does not",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return wrap(t, newStreamingTool(t, functiontool.Config{RequireConfirmation: true}, handlerRan), false)
			},
			wantErr:       tool.ErrConfirmationRequired,
			wantRequested: true,
		},
		{
			name: "rejected",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return wrap(t, newStreamingTool(t, functiontool.Config{RequireConfirmation: true}, handlerRan), true)
			},
			confirmations: map[string]*toolconfirmation.ToolConfirmation{
				"call_1": {Confirmed: false},
				"call_2": {Confirmed: false},
			},
			wantErr: tool.ErrConfirmationRejected,
		},
		{
			name: "confirmed",
			build: func(t *testing.T, handlerRan *atomic.Bool) tool.Tool {
				return wrap(t, newStreamingTool(t, functiontool.Config{RequireConfirmation: true}, handlerRan), true)
			},
			confirmations: map[string]*toolconfirmation.ToolConfirmation{
				"call_1": {Confirmed: true},
				"call_2": {Confirmed: true},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var handlerRan atomic.Bool
			gated := tt.build(t, &handlerRan)

			sent := make(chan string, 2)
			liveSess := &mockLiveSession{sendFunc: func(req agent.LiveRequest) error {
				sent <- req.Content.Parts[0].Text
				return nil
			}}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				InvocationID: "inv_1",
				Agent:        &mockAgent{name: "agent_1"},
			})
			resp := &model.LLMResponse{Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: toolName, Args: map[string]any{}}},
					{FunctionCall: &genai.FunctionCall{ID: "call_2", Name: toolName, Args: map[string]any{}}},
				},
			}}

			flow := &Flow{Tools: []tool.Tool{gated}}
			ev, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{toolName: gated}, resp, tt.confirmations, liveSess)
			if err != nil {
				t.Fatal(err)
			}
			if ev == nil {
				t.Fatal("handleFunctionCalls() returned no event")
			}

			for _, part := range ev.Content.Parts {
				fr := part.FunctionResponse
				if fr == nil {
					t.Fatalf("response part has no FunctionResponse: %+v", part)
				}
				if tt.wantErr == nil {
					if fr.Response["status"] == nil {
						t.Errorf("response for %s = %v, want a pending status", fr.ID, fr.Response)
					}
					continue
				}
				if status, ok := fr.Response["status"]; ok {
					t.Errorf("response for %s has status %q, want an error and no pending status", fr.ID, status)
				}
				errText, _ := fr.Response["error"].(string)
				if !strings.Contains(errText, tt.wantErr.Error()) {
					t.Errorf("response for %s error = %q, want it to contain %q", fr.ID, errText, tt.wantErr.Error())
				}
			}

			var wantRequested []string
			if tt.wantRequested {
				wantRequested = []string{"call_1", "call_2"}
			}
			gotRequested := slices.Sorted(maps.Keys(ev.Actions.RequestedToolConfirmations))
			if diff := cmp.Diff(wantRequested, gotRequested); diff != "" {
				t.Errorf("RequestedToolConfirmations call IDs mismatch (-want +got):\n%s", diff)
			}
			if ev.Actions.SkipSummarization != tt.wantRequested {
				t.Errorf("SkipSummarization = %t, want %t", ev.Actions.SkipSummarization, tt.wantRequested)
			}

			if tt.wantErr != nil {
				if handlerRan.Load() {
					t.Error("the stream ran for a call the gate stopped")
				}
				return
			}
			for range 2 {
				select {
				case got := <-sent:
					if want := "Function " + toolName + " returned: chunk"; got != want {
						t.Errorf("model received %q, want %q", got, want)
					}
				case <-time.After(10 * time.Second):
					t.Fatal("a confirmed call did not stream to the model")
				}
			}
		})
	}
}

// TestHandleFunctionCalls_LiveStreamErrorReachesModel checks that a live
// streaming call that fails after its pending response tells the model so.
// Otherwise the model keeps waiting on output that is never coming.
func TestHandleFunctionCalls_LiveStreamErrorReachesModel(t *testing.T) {
	type Args struct{}
	const toolName = "failing_stream"
	st, err := functiontool.NewStreaming(functiontool.Config{Name: toolName}, func(agent.Context, Args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			if !yield("partial", nil) {
				return
			}
			yield("", errors.New("backend unavailable"))
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	sent := make(chan string, 3)
	liveSess := &mockLiveSession{sendFunc: func(req agent.LiveRequest) error {
		sent <- req.Content.Parts[0].Text
		return nil
	}}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
	resp := &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: toolName, Args: map[string]any{}}}},
	}}

	flow := &Flow{Tools: []tool.Tool{st}}
	if _, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{toolName: st}, resp, nil, liveSess); err != nil {
		t.Fatal(err)
	}

	want := []string{
		"Function failing_stream returned: partial",
		"Function failing_stream failed: backend unavailable",
	}
	var got []string
	for range want {
		select {
		case text := <-sent:
			got = append(got, text)
		case <-time.After(10 * time.Second):
			t.Fatalf("model received %q, want %q", got, want)
		}
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("content sent to the model mismatch (-want +got):\n%s", diff)
	}
}

// TestHandleFunctionCalls_LiveGateRunsOnce checks that a live streaming call is
// gated once, on the calling goroutine. Gating it again in the background
// stream calls the confirmation provider a second time, and a provider that
// answers differently then records its request after the response event has
// been built.
func TestHandleFunctionCalls_LiveGateRunsOnce(t *testing.T) {
	type Args struct{}
	const toolName = "provider_gated_stream"
	var providerCalls atomic.Int32
	st, err := functiontool.NewStreaming(functiontool.Config{
		Name: toolName,
		// Allows the first check of a call and asks for confirmation on any later one.
		RequireConfirmationProvider: func(Args) bool { return providerCalls.Add(1) > 1 },
	}, func(agent.Context, Args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) { yield("chunk", nil) }
	})
	if err != nil {
		t.Fatal(err)
	}

	sent := make(chan string, 2)
	liveSess := &mockLiveSession{sendFunc: func(req agent.LiveRequest) error {
		sent <- req.Content.Parts[0].Text
		return nil
	}}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
	resp := &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: toolName, Args: map[string]any{}}}},
	}}

	flow := &Flow{Tools: []tool.Tool{st}}
	if _, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{toolName: st}, resp, nil, liveSess); err != nil {
		t.Fatal(err)
	}

	select {
	case got := <-sent:
		if want := "Function " + toolName + " returned: chunk"; got != want {
			t.Errorf("model received %q, want %q", got, want)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the allowed call did not stream to the model")
	}
	if got := providerCalls.Load(); got != 1 {
		t.Errorf("confirmation provider called %d times, want 1", got)
	}
}
