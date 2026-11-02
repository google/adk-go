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
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func TestFunctionToolPanicBoundary(t *testing.T) {
	for _, tc := range []struct {
		name     string
		panics   bool
		override string
		want     string
	}{
		{name: "panic", panics: true, want: streamingToolInternalError("lookup")},
		{name: "returned error", want: "order not found"},
		{name: "wrapped panic", panics: true, override: "wrap", want: streamingToolInternalError("lookup")},
		{name: "error callback result", panics: true, override: "result", want: "recovered"},
		{name: "error callback error", panics: true, override: "error", want: "retry later"},
		{name: "after callback result", panics: true, override: "after result", want: "recovered"},
		{name: "after callback error", panics: true, override: "after error", want: "retry later"},
	} {
		for _, live := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/live=%t", tc.name, live), func(t *testing.T) {
				type args struct{}
				type result struct{}
				lookup, err := functiontool.New(functiontool.Config{Name: "lookup"}, func(agent.Context, args) (result, error) {
					if tc.panics {
						panic("private-panic-value")
					}
					return result{}, errors.New("order not found")
				})
				if err != nil {
					t.Fatal("cannot construct tool")
				}
				callbackSawDetails := false
				afterSawDetails := false
				f := &Flow{OnToolErrorCallbacks: []OnToolErrorCallback{func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
					callbackSawDetails = strings.Contains(err.Error(), "private-panic-value") && strings.Contains(err.Error(), "stack:")
					switch tc.override {
					case "wrap":
						return nil, fmt.Errorf("callback context: %w", err)
					case "result":
						return map[string]any{"result": "recovered"}, nil
					case "error":
						return nil, errors.New("retry later")
					}
					return nil, nil
				}}, AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
					afterSawDetails = err != nil && strings.Contains(err.Error(), "private-panic-value") && strings.Contains(err.Error(), "stack:")
					switch tc.override {
					case "after result":
						return map[string]any{"result": "recovered"}, nil
					case "after error":
						return nil, errors.New("retry later")
					}
					return nil, nil
				}}}
				ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
				response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}}}}
				var liveSession agent.LiveSession
				if live {
					liveSession = &mockLiveSession{}
				}
				event, err := f.handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": lookup}, response, nil, liveSession)
				if err != nil {
					t.Fatal("unexpected flow error")
				}
				key := "error"
				if tc.override == "result" || tc.override == "after result" {
					key = "result"
				}
				if event.Content.Parts[0].FunctionResponse.Response[key] != tc.want {
					t.Fatal("incorrect model-facing response")
				}
				if tc.panics && !callbackSawDetails {
					t.Fatal("error callback lost panic diagnostics")
				}
				if tc.panics && tc.override != "result" && tc.override != "error" && !afterSawDetails {
					t.Fatal("after callback lost panic diagnostics")
				}
			})
		}
	}
}

// A caller can capture a framework error without importing its internal type.
func TestStreamingHandlerReyieldedFrameworkError(t *testing.T) {
	type args struct{}
	source, err := functiontool.NewStreaming(functiontool.Config{Name: "source", RequireConfirmation: true}, func(agent.Context, args) iter.Seq2[string, error] {
		return func(func(string, error) bool) { t.Error("unapproved source handler ran") }
	})
	if err != nil {
		t.Fatal("cannot construct source")
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
	stream := source.(interface {
		RunStream(agent.Context, any) iter.Seq2[string, error]
	})
	var captured error
	for _, err := range stream.RunStream(agent.NewToolContext(ctx, "source_call", &session.EventActions{}, nil), map[string]any{}) {
		captured = err
	}
	if !errors.Is(captured, tool.ErrConfirmationRequired) {
		t.Fatal("did not capture a framework error")
	}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"}, func(agent.Context, args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) { yield("", captured) }
	})
	if err != nil {
		t.Fatal("cannot construct lookup")
	}
	for _, live := range []bool{false, true} {
		t.Run(fmt.Sprintf("live=%t", live), func(t *testing.T) {
			response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}}}}
			sent := make(chan agent.LiveRequest, 1)
			var liveSession agent.LiveSession
			if live {
				liveSession = &mockLiveSession{sendFunc: func(r agent.LiveRequest) error { sent <- r; return nil }}
			}
			event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": lookup}, response, nil, liveSession)
			if err != nil {
				t.Fatal("unexpected flow error")
			}
			content := event.Content
			if live {
				select {
				case r := <-sent:
					content = r.Content
				case <-time.After(3 * time.Second):
					t.Fatal("missing follow-up")
				}
			}
			if content.Parts[0].FunctionResponse.Response["error"] != streamingToolInternalError("lookup") {
				t.Fatal("handler re-yielded a trusted framework error")
			}
		})
	}
}
