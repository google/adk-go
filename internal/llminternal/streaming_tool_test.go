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
	"sync/atomic"
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
			for i := range args.Count {
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

// customStreamTool is a streaming tool that is not built with
// functiontool.NewStreaming, so nothing recovers its panic before the flow.
type customStreamTool struct {
	stream iter.Seq2[string, error]
}

func (customStreamTool) Name() string                            { return "custom_stream" }
func (customStreamTool) Description() string                     { return "custom stream" }
func (customStreamTool) IsLongRunning() bool                     { return false }
func (customStreamTool) Declaration() *genai.FunctionDeclaration { return nil }

func (c customStreamTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return c.stream
}

func TestHandleFunctionCalls_CustomStreamingToolPanic(t *testing.T) {
	toolsDict := map[string]tool.Tool{"custom_stream": customStreamTool{
		stream: func(func(string, error) bool) { _ = any(12345.0).(string) },
	}}
	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "custom_stream", Args: map[string]any{}},
	}}}}
	newInvCtx := func(t *testing.T) agent.InvocationContext {
		return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		})
	}

	t.Run("non-live", func(t *testing.T) {
		mergedEvent, err := (&Flow{}).handleFunctionCalls(newInvCtx(t), toolsDict, resp, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := "Invoking `custom_stream()` failed with an internal error."
		if got := mergedEvent.LLMResponse.Content.Parts[0].FunctionResponse.Response["error"]; got != want {
			t.Errorf("FunctionResponse error = %q, want %q", got, want)
		}
	})

	t.Run("live", func(t *testing.T) {
		liveSess := newLiveSessionImpl()
		liveSess.activeTools = make(map[string][]activeTask)
		if _, err := (&Flow{}).handleFunctionCalls(newInvCtx(t), toolsDict, resp, nil, liveSess); err != nil {
			t.Fatal(err)
		}
		// Wait for the goroutine that runs the tool to finish. An unrecovered
		// panic in it ends the test process.
		deadline := time.Now().Add(5 * time.Second)
		for {
			liveSess.mu.Lock()
			running := len(liveSess.activeTools["custom_stream"])
			liveSess.mu.Unlock()
			if running == 0 {
				return
			}
			if time.Now().After(deadline) {
				t.Fatal("the streaming tool is still registered after it panicked")
			}
			time.Sleep(time.Millisecond)
		}
	})
}

// A panic in the loop that reads the stream is the reader's own, so it is not
// turned into a tool error.
func TestRecoverStreamPanicPropagatesConsumerPanic(t *testing.T) {
	streamTool := customStreamTool{
		stream: func(yield func(string, error) bool) { yield("chunk", nil) },
	}
	original := errors.New("consumer panic")
	var got any
	func() {
		defer func() { got = recover() }()
		for range recoverStreamPanic(streamTool, nil, nil) { //nolint:errcheck // the body panics on the first chunk; the test asserts the recovered value
			panic(original)
		}
	}()
	if got != original {
		t.Errorf("recovered panic has type %T, want the original consumer panic value", got)
	}
}

// A consumer that stops reading stops the stream it reads.
func TestRecoverStreamPanicStopsStreamWhenConsumerStops(t *testing.T) {
	var stopped bool
	streamTool := customStreamTool{
		stream: func(yield func(string, error) bool) { stopped = !yield("first", nil) },
	}
	for range recoverStreamPanic(streamTool, nil, nil) { //nolint:errcheck // the consumer stops at the first chunk; the test asserts that the stream is told
		break
	}
	if !stopped {
		t.Error("the stream was not told that its consumer stopped")
	}
}

// A panic after the stream has yielded is recovered like one before it.
func TestRecoverStreamPanicRecoversPanicAfterAYield(t *testing.T) {
	streamTool := customStreamTool{
		stream: func(yield func(string, error) bool) {
			if !yield("chunk", nil) {
				return
			}
			panic("boom")
		},
	}
	var chunks int
	var errs []error
	for _, err := range recoverStreamPanic(streamTool, nil, nil) {
		if err != nil {
			errs = append(errs, err)
			continue
		}
		chunks++
	}
	if chunks != 1 || len(errs) != 1 {
		t.Fatalf("stream yielded %d chunks and %d errors, want one chunk and one error", chunks, len(errs))
	}
	if _, ok := errs[0].(*toolinternal.PanicError); !ok {
		t.Errorf("stream yielded an error of type %T, want *toolinternal.PanicError", errs[0])
	}
	if text := errs[0].Error(); !strings.HasPrefix(text, `panic in tool "custom_stream": boom`) || !strings.Contains(text, "\nstack: ") || !strings.Contains(text, "runtime/debug.Stack") {
		t.Error("the stream error does not carry the tool name, the panic value and the stack")
	}
}

// A panic the stream raises after its consumer stopped cannot be yielded, so it
// propagates, as it does for a streaming function tool.
func TestRecoverStreamPanicPropagatesCleanupPanicAfterStop(t *testing.T) {
	original := errors.New("cleanup panic")
	streamTool := customStreamTool{
		stream: func(yield func(string, error) bool) {
			defer func() { panic(original) }()
			yield("chunk", nil)
		},
	}
	var got any
	func() {
		defer func() { got = recover() }()
		for range recoverStreamPanic(streamTool, nil, nil) { //nolint:errcheck // the consumer stops at the first chunk; the test asserts the recovered value
			break
		}
	}()
	if got != original {
		t.Errorf("recovered panic has type %T, want the original cleanup panic value", got)
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

// argEditingStreamTool sets input_str in the args RunStream is called with and
// streams the new value.
type argEditingStreamTool struct{}

func (argEditingStreamTool) Name() string        { return "edit_stream" }
func (argEditingStreamTool) Description() string { return "edits its args" }
func (argEditingStreamTool) IsLongRunning() bool { return false }

func (t argEditingStreamTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name(), Description: t.Description()}
}

func (argEditingStreamTool) RunStream(_ agent.Context, args any) iter.Seq2[string, error] {
	args.(map[string]any)["input_str"] = "modified_input"
	return func(yield func(string, error) bool) { yield("modified_input", nil) }
}

// A live streaming tool's edits to its args do not reach the function call
// event.
func TestHandleFunctionCalls_LiveStreamingToolArgEditsDoNotReachTheCallEvent(t *testing.T) {
	sent := make(chan struct{}, 1)
	liveSess := &mockLiveSession{sendFunc: func(agent.LiveRequest) error {
		sent <- struct{}{}
		return nil
	}}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
	st := argEditingStreamTool{}
	call := &genai.FunctionCall{ID: "call_1", Name: st.Name(), Args: map[string]any{"input_str": "original"}}
	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: call}}}}

	_, err := (&Flow{}).handleFunctionCalls(invCtx, map[string]tool.Tool{st.Name(): st}, resp, nil, liveSess)
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("streaming tool sent no result")
	}
	if diff := cmp.Diff(map[string]any{"input_str": "original"}, call.Args); diff != "" {
		t.Errorf("function call args mismatch (-want +got):\n%s", diff)
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

// streamRecorder records the arguments a streaming tool ran with.
type streamRecorder struct {
	mu   sync.Mutex
	runs []string
}

func (r *streamRecorder) record(msg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.runs = append(r.runs, msg)
}

func (r *streamRecorder) calls() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.runs...)
}

// A streaming tool goes through the same callback pipeline as a plain function
// tool: a before-tool callback can answer the call or rewrite its arguments,
// and the outcome of the call reaches the on-tool-error and after-tool
// callbacks.
func TestHandleFunctionCalls_StreamingToolCallbacks(t *testing.T) {
	type Args struct {
		Msg string `json:"msg"`
	}

	// newStreamTool returns a streaming tool that records what it ran with and
	// then yields its argument back, or fails with failWith.
	newStreamTool := func(t *testing.T, failWith error) (tool.Tool, *streamRecorder) {
		t.Helper()
		rec := &streamRecorder{}
		streamTool, err := functiontool.NewStreaming(functiontool.Config{
			Name:        "test_stream",
			Description: "streams chunks",
		}, func(_ agent.Context, args Args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) {
				rec.record(args.Msg)
				if failWith != nil {
					yield("", failWith)
					return
				}
				yield(args.Msg, nil)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		return streamTool, rec
	}

	call := &model.LLMResponse{Content: &genai.Content{
		Role: "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID:   "call_1",
			Name: "test_stream",
			Args: map[string]any{"msg": "sent"},
		}}},
	}}

	// run calls the tool and returns the function response for the model.
	run := func(t *testing.T, flow *Flow, streamTool tool.Tool, liveSess agent.LiveSession) map[string]any {
		t.Helper()
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		})
		event, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, call, nil, liveSess)
		if err != nil {
			t.Fatal(err)
		}
		if event == nil {
			t.Fatal("handleFunctionCalls returned no event")
		}
		if len(event.LLMResponse.Content.Parts) != 1 {
			t.Fatalf("got %d parts, want 1", len(event.LLMResponse.Content.Parts))
		}
		part := event.LLMResponse.Content.Parts[0].FunctionResponse
		if part == nil {
			t.Fatal("event part is not a function response")
		}
		return part.Response
	}

	t.Run("before-tool callback answers instead of the tool", func(t *testing.T) {
		streamTool, rec := newStreamTool(t, nil)
		flow := &Flow{BeforeToolCallbacks: []BeforeToolCallback{
			func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return map[string]any{"denied": true}, nil
			},
		}}

		got := run(t, flow, streamTool, nil)

		if diff := cmp.Diff(map[string]any{"denied": true}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
		if runs := rec.calls(); len(runs) != 0 {
			t.Errorf("tool ran %v, want it not to run", runs)
		}
	})

	t.Run("before-tool callback fails the call", func(t *testing.T) {
		streamTool, rec := newStreamTool(t, nil)
		flow := &Flow{BeforeToolCallbacks: []BeforeToolCallback{
			func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return nil, errors.New("not allowed")
			},
		}}

		got := run(t, flow, streamTool, nil)

		if diff := cmp.Diff(map[string]any{"error": "not allowed"}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
		if runs := rec.calls(); len(runs) != 0 {
			t.Errorf("tool ran %v, want it not to run", runs)
		}
	})

	t.Run("before-tool callback rewrites the arguments", func(t *testing.T) {
		streamTool, rec := newStreamTool(t, nil)
		flow := &Flow{BeforeToolCallbacks: []BeforeToolCallback{
			func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
				args["msg"] = "rewritten"
				return nil, nil
			},
		}}

		got := run(t, flow, streamTool, nil)

		if diff := cmp.Diff(map[string]any{"result": "rewritten"}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"rewritten"}, rec.calls()); diff != "" {
			t.Errorf("tool arguments mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("after-tool callback sees the result", func(t *testing.T) {
		streamTool, _ := newStreamTool(t, nil)
		var gotResult map[string]any
		var gotErr error
		flow := &Flow{AfterToolCallbacks: []AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
				gotResult, gotErr = result, err
				return map[string]any{"wrapped": result["result"]}, nil
			},
		}}

		got := run(t, flow, streamTool, nil)

		if diff := cmp.Diff(map[string]any{"result": "sent"}, gotResult); diff != "" {
			t.Errorf("after-tool result mismatch (-want +got):\n%s", diff)
		}
		if gotErr != nil {
			t.Errorf("after-tool err = %v, want nil", gotErr)
		}
		if diff := cmp.Diff(map[string]any{"wrapped": "sent"}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("on-tool-error callback answers a failing stream", func(t *testing.T) {
		streamTool, _ := newStreamTool(t, errors.New("order not found"))
		var gotErr error
		flow := &Flow{OnToolErrorCallbacks: []OnToolErrorCallback{
			func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
				gotErr = err
				return map[string]any{"handled": true}, nil
			},
		}}

		got := run(t, flow, streamTool, nil)

		if gotErr == nil || gotErr.Error() != "order not found" {
			t.Errorf("on-tool-error err = %v, want \"order not found\"", gotErr)
		}
		if diff := cmp.Diff(map[string]any{"handled": true}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("live mode: before-tool callback keeps the stream from starting", func(t *testing.T) {
		streamTool, rec := newStreamTool(t, nil)
		var sends atomic.Int64
		sess := &mockLiveSession{sendFunc: func(agent.LiveRequest) error {
			sends.Add(1)
			return nil
		}}
		flow := &Flow{BeforeToolCallbacks: []BeforeToolCallback{
			func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return map[string]any{"denied": true}, nil
			},
		}}

		got := run(t, flow, streamTool, sess)

		if diff := cmp.Diff(map[string]any{"denied": true}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
		if runs := rec.calls(); len(runs) != 0 {
			t.Errorf("tool ran %v, want it not to run", runs)
		}
		if n := sends.Load(); n != 0 {
			t.Errorf("live session received %d requests, want 0", n)
		}
	})

	t.Run("live mode: after-tool callback sees the pending status", func(t *testing.T) {
		streamTool, _ := newStreamTool(t, nil)
		var sent sync.WaitGroup
		sent.Add(1)
		sess := &mockLiveSession{sendFunc: func(agent.LiveRequest) error {
			sent.Done()
			return nil
		}}
		var gotResult map[string]any
		flow := &Flow{AfterToolCallbacks: []AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				gotResult = result
				return nil, nil
			},
		}}

		got := run(t, flow, streamTool, sess)
		sent.Wait()

		want := map[string]any{"status": "The function is running asynchronously and the results are pending."}
		if diff := cmp.Diff(want, gotResult); diff != "" {
			t.Errorf("after-tool result mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
	})
}

// A live streaming tool is registered before the after-tool callbacks run, so
// a stop_streaming call issued in the same batch can still find it, and it
// does not start when one of those callbacks fails the call.
func TestHandleFunctionCalls_LiveStreamingToolAfterCallback(t *testing.T) {
	type Args struct {
		Msg string `json:"msg"`
	}

	newLive := func(t *testing.T) *liveSessionImpl {
		t.Helper()
		sess := newLiveSessionImpl()
		sess.activeTools = make(map[string][]activeTask)
		go func() {
			for range sess.inputCh {
			}
		}()
		return sess
	}

	call := &model.LLMResponse{Content: &genai.Content{
		Role: "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID:   "call_1",
			Name: "test_stream",
			Args: map[string]any{"msg": "sent"},
		}}},
	}}

	run := func(t *testing.T, flow *Flow, streamTool tool.Tool, sess *liveSessionImpl) map[string]any {
		t.Helper()
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			InvocationID: "inv_1",
			Agent:        &mockAgent{name: "agent_1"},
		})
		event, err := flow.handleFunctionCalls(invCtx, map[string]tool.Tool{"test_stream": streamTool}, call, nil, sess)
		if err != nil {
			t.Fatal(err)
		}
		return event.LLMResponse.Content.Parts[0].FunctionResponse.Response
	}

	t.Run("registered while the after-tool callbacks run", func(t *testing.T) {
		streamTool, err := functiontool.NewStreaming(functiontool.Config{
			Name:        "test_stream",
			Description: "streams chunks",
		}, func(_ agent.Context, args Args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) { yield(args.Msg, nil) }
		})
		if err != nil {
			t.Fatal(err)
		}
		sess := newLive(t)
		var registered bool
		flow := &Flow{AfterToolCallbacks: []AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				sess.mu.Lock()
				registered = len(sess.activeTools["test_stream"]) == 1
				sess.mu.Unlock()
				return nil, nil
			},
		}}

		run(t, flow, streamTool, sess)

		if !registered {
			t.Error("streaming tool was not registered while the after-tool callback ran")
		}
	})

	t.Run("after-tool callback failure keeps the stream from starting", func(t *testing.T) {
		var ran atomic.Bool
		streamTool, err := functiontool.NewStreaming(functiontool.Config{
			Name:        "test_stream",
			Description: "streams chunks",
		}, func(_ agent.Context, args Args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) {
				ran.Store(true)
				yield(args.Msg, nil)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		sess := newLive(t)
		flow := &Flow{AfterToolCallbacks: []AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				return nil, errors.New("not allowed")
			},
		}}

		got := run(t, flow, streamTool, sess)

		if diff := cmp.Diff(map[string]any{"error": "not allowed"}, got); diff != "" {
			t.Errorf("function response mismatch (-want +got):\n%s", diff)
		}
		if ran.Load() {
			t.Error("streaming tool ran, want it not to start")
		}
		sess.mu.Lock()
		left := len(sess.activeTools["test_stream"])
		sess.mu.Unlock()
		if left != 0 {
			t.Errorf("%d streaming tools still registered, want 0", left)
		}
	})
}
