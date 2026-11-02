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
	"google.golang.org/adk/v2/internal/toolerror"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func TestStreamingToolFailureReporting(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"},
		func(_ agent.Context, a args) iter.Seq2[string, error] {
			return func(func(string, error) bool) { panic("lookup failed for " + a.Value) }
		})
	if err != nil {
		t.Fatal(err)
	}
	tools := map[string]tool.Tool{"lookup": lookup}
	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{"value": "user-secret-123"}},
	}}}}
	newCtx := func(t *testing.T) agent.InvocationContext {
		return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{InvocationID: "inv", Agent: &mockAgent{name: "a"}})
	}
	wantErr := streamingToolInternalError("lookup")

	t.Run("non-live", func(t *testing.T) {
		ev, err := (&Flow{}).handleFunctionCalls(newCtx(t), tools, resp, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		msg, _ := ev.Content.Parts[0].FunctionResponse.Response["error"].(string)
		if msg != wantErr {
			t.Errorf("error = %q, want %q", msg, wantErr)
		}
		if strings.Contains(msg, "user-secret-123") || strings.Contains(msg, "runtime/debug.Stack") {
			t.Errorf("function response sent to the model (%d bytes) contains the panic value or a stack trace", len(msg))
		}
	})

	t.Run("live", func(t *testing.T) {
		sent := make(chan agent.LiveRequest, 1)
		sess := &mockLiveSession{sendFunc: func(r agent.LiveRequest) error { sent <- r; return nil }}
		if _, err := (&Flow{}).handleFunctionCalls(newCtx(t), tools, resp, nil, sess); err != nil {
			t.Fatal(err)
		}
		select {
		case req := <-sent:
			if req.Content == nil || len(req.Content.Parts) == 0 || req.Content.Parts[0].FunctionResponse == nil {
				t.Fatalf("follow-up missing FunctionResponse: %#v", req.Content)
			}
			fr := req.Content.Parts[0].FunctionResponse
			if fr.ID != "call_1" || fr.Name != "lookup" {
				t.Errorf("FunctionResponse id/name = %q/%q, want call_1/lookup", fr.ID, fr.Name)
			}
			msg, _ := fr.Response["error"].(string)
			if msg != wantErr {
				t.Errorf("error = %q, want %q", msg, wantErr)
			}
			if strings.Contains(msg, "user-secret-123") || strings.Contains(msg, "runtime/debug.Stack") {
				t.Errorf("live follow-up contains panic value or stack trace")
			}
		case <-time.After(2 * time.Second):
			t.Error("the model got the pending response but no follow-up after the tool failed")
		}
	})
}

// A custom implementation cannot import ADK's internal handler-error marker.
type failingCustomStream struct {
	tool.Tool
	failure error
}

func (t *failingCustomStream) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name()}
}

func (t *failingCustomStream) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) { yield("", t.failure) }
}

func TestCustomStreamingFailureIsSanitized(t *testing.T) {
	for _, failure := range []error{errors.New("private-handler-data"), fmt.Errorf("private-handler-data: %w", tool.ErrConfirmationRejected)} {
		for _, live := range []bool{false, true} {
			t.Run(fmt.Sprintf("live=%t/wrapped=%t", live, errors.Is(failure, tool.ErrConfirmationRejected)), func(t *testing.T) {
				custom := &failingCustomStream{Tool: &mockFunctionTool{name: "lookup"}, failure: failure}
				ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
				response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}}}}
				sent := make(chan agent.LiveRequest, 1)
				var liveSession agent.LiveSession
				if live {
					liveSession = &mockLiveSession{sendFunc: func(r agent.LiveRequest) error { sent <- r; return nil }}
				}
				event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": custom}, response, nil, liveSession)
				if err != nil {
					t.Fatal("unexpected flow error")
				}
				content := event.Content
				if live {
					select {
					case request := <-sent:
						content = request.Content
					case <-time.After(3 * time.Second):
						t.Fatal("missing failure follow-up")
					}
				}
				if content.Parts[0].FunctionResponse.Response["error"] != streamingToolInternalError("lookup") {
					t.Fatal("custom handler error was not sanitized")
				}
			})
		}
	}
}

func TestStreamingFrameworkErrorsPreserved(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	for _, tc := range []struct {
		name         string
		input        map[string]any
		confirmation *toolconfirmation.ToolConfirmation
		want         string
	}{
		{name: "approval required", input: map[string]any{"value": "v"}, want: tool.ErrConfirmationRequired.Error()},
		{name: "approval rejected", input: map[string]any{"value": "v"}, confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false}, want: tool.ErrConfirmationRejected.Error()},
		{name: "argument conversion", input: map[string]any{"value": 1}, want: "string"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handlerCalled := false
			lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup", RequireConfirmation: true}, func(agent.Context, args) iter.Seq2[string, error] {
				handlerCalled = true
				return func(func(string, error) bool) {}
			})
			if err != nil {
				t.Fatal("cannot construct tool")
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
			response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: tc.input}}}}}
			confirmations := map[string]*toolconfirmation.ToolConfirmation{"call_1": tc.confirmation}
			event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": lookup}, response, confirmations, nil)
			if err != nil {
				t.Fatal("unexpected flow error")
			}
			msg, _ := event.Content.Parts[0].FunctionResponse.Response["error"].(string)
			if !strings.Contains(msg, tc.want) {
				t.Fatal("framework error message lost")
			}
			if handlerCalled {
				t.Fatal("handler ran before approval or argument validation")
			}
		})
	}
}

type singleStreamingToolset struct{ value tool.Tool }

func (s *singleStreamingToolset) Name() string { return "streaming" }
func (s *singleStreamingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{s.value}, nil
}

func TestStreamingConfirmationWrapperErrors(t *testing.T) {
	for _, tc := range []struct {
		name         string
		confirmation *toolconfirmation.ToolConfirmation
		want         string
	}{
		{name: "required", want: tool.ErrConfirmationRequired.Error()},
		{name: "rejected", confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false}, want: tool.ErrConfirmationRejected.Error()},
		{name: "approved handler fails", confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}, want: streamingToolInternalError("lookup")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
			custom := &failingCustomStream{Tool: &mockFunctionTool{name: "lookup"}, failure: errors.New("private-handler-data")}
			wrapped, err := tool.WithConfirmation(&singleStreamingToolset{value: custom}, true, nil).Tools(agent.NewToolContext(ctx, "call_1", &session.EventActions{}, tc.confirmation))
			if err != nil {
				t.Fatal("cannot wrap tool")
			}
			response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}}}}
			event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": wrapped[0]}, response, map[string]*toolconfirmation.ToolConfirmation{"call_1": tc.confirmation}, nil)
			if err != nil {
				t.Fatal("unexpected flow error")
			}
			message, _ := event.Content.Parts[0].FunctionResponse.Response["error"].(string)
			if !strings.Contains(message, tc.want) {
				t.Fatal("incorrect wrapped streaming error")
			}
		})
	}
}

func TestStreamingHandlerCannotExposeFrameworkError(t *testing.T) {
	type args struct{}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"}, func(agent.Context, args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			yield("", &toolerror.BeforeHandler{Err: errors.New("private-handler-data")})
		}
	})
	if err != nil {
		t.Fatal("cannot construct tool")
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
	response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}}}}
	event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": lookup}, response, nil, nil)
	if err != nil {
		t.Fatal("unexpected flow error")
	}
	if event.Content.Parts[0].FunctionResponse.Response["error"] != streamingToolInternalError("lookup") {
		t.Fatal("handler exposed a framework error")
	}
}
