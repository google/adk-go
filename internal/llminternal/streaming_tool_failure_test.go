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
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// liveToolFailure returns the follow-up, after verifying the pending event.
func liveToolFailure(t *testing.T, lookup tool.Tool, input map[string]any, confirmation *toolconfirmation.ToolConfirmation) *genai.FunctionResponse {
	t.Helper()
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
	response := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: lookup.Name(), Args: input}}}}}
	sent := make(chan agent.LiveRequest, 1)
	sess := &mockLiveSession{sendFunc: func(r agent.LiveRequest) error { sent <- r; return nil }}
	event, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{lookup.Name(): lookup}, response, map[string]*toolconfirmation.ToolConfirmation{"call_1": confirmation}, sess)
	if err != nil {
		t.Fatal("unexpected flow error")
	}
	if event.Content.Parts[0].FunctionResponse.Response["status"] == nil {
		t.Fatal("missing pending event")
	}
	select {
	case request := <-sent:
		if request.Content == nil || len(request.Content.Parts) != 1 || request.Content.Parts[0].FunctionResponse == nil {
			t.Fatal("missing failure response")
		}
		result := request.Content.Parts[0].FunctionResponse
		if result.ID != "call_1" || result.Name != lookup.Name() {
			t.Fatal("incorrect failure attribution")
		}
		return result
	case <-time.After(3 * time.Second):
		t.Fatal("missing live failure follow-up")
		return nil
	}
}

func TestStreamingToolFailureReporting(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"}, func(_ agent.Context, a args) iter.Seq2[string, error] {
		return func(func(string, error) bool) { panic("lookup failed for " + a.Value) }
	})
	if err != nil {
		t.Fatal(err)
	}
	result := liveToolFailure(t, lookup, map[string]any{"value": "user-secret-123"}, nil)
	if result.Response["error"] != streamingToolInternalError("lookup") {
		t.Fatal("live panic was not sanitized")
	}
}

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
		t.Run(fmt.Sprintf("wrapped=%t", errors.Is(failure, tool.ErrConfirmationRejected)), func(t *testing.T) {
			custom := &failingCustomStream{Tool: &mockFunctionTool{name: "lookup"}, failure: failure}
			result := liveToolFailure(t, custom, map[string]any{}, nil)
			if result.Response["error"] != streamingToolInternalError("lookup") {
				t.Fatal("live custom handler error was not sanitized")
			}
		})
	}
}

// liveFrameworkErrorMessage tests preflight marking and the live classifier
// synchronously; live approval event-state synchronization is a separate concern.
func liveFrameworkErrorMessage(t *testing.T, lookup tool.Tool, input map[string]any, confirmation *toolconfirmation.ToolConfirmation) string {
	t.Helper()
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
	stream := lookup.(interface {
		RunStream(agent.Context, any) iter.Seq2[string, error]
	})
	for _, err := range stream.RunStream(agent.NewToolContext(ctx, "call_1", &session.EventActions{}, confirmation), input) {
		if err == nil {
			t.Fatal("expected preflight or handler error")
		}
		return streamingToolErrorMessage(lookup.Name(), err)
	}
	t.Fatal("missing streaming error")
	return ""
}

func TestLiveFrameworkErrorClassification(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	for _, tc := range []struct {
		name         string
		input        map[string]any
		confirmation *toolconfirmation.ToolConfirmation
		want         string
	}{
		{name: "approval required", input: map[string]any{"value": "v"}, want: `error tool "lookup" ` + tool.ErrConfirmationRequired.Error()},
		{name: "approval rejected", input: map[string]any{"value": "v"}, confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false}, want: `error tool "lookup" ` + tool.ErrConfirmationRejected.Error()},
		{name: "argument conversion", input: map[string]any{"value": 1}, want: "string"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup", RequireConfirmation: true}, func(agent.Context, args) iter.Seq2[string, error] {
				return func(func(string, error) bool) { t.Error("handler ran before approval or validation") }
			})
			if err != nil {
				t.Fatal(err)
			}
			message := liveFrameworkErrorMessage(t, lookup, tc.input, tc.confirmation)
			if (tc.name == "argument conversion" && !strings.Contains(message, tc.want)) || (tc.name != "argument conversion" && message != tc.want) {
				t.Fatal("live framework error message lost")
			}
		})
	}
}

type singleStreamingToolset struct{ value tool.Tool }

func (s *singleStreamingToolset) Name() string { return "streaming" }
func (s *singleStreamingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return []tool.Tool{s.value}, nil
}

func TestLiveConfirmationWrapperErrorClassification(t *testing.T) {
	for _, tc := range []struct {
		name         string
		confirmation *toolconfirmation.ToolConfirmation
		want         string
	}{
		{name: "required", want: `error tool "lookup" ` + tool.ErrConfirmationRequired.Error()},
		{name: "rejected", confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false}, want: `error tool "lookup" ` + tool.ErrConfirmationRejected.Error()},
		{name: "approved handler fails", confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}, want: streamingToolInternalError("lookup")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
			custom := &failingCustomStream{Tool: &mockFunctionTool{name: "lookup"}, failure: errors.New("private-handler-data")}
			wrapped, err := tool.WithConfirmation(&singleStreamingToolset{value: custom}, true, nil).Tools(agent.NewToolContext(ctx, "call_1", &session.EventActions{}, tc.confirmation))
			if err != nil {
				t.Fatal(err)
			}
			message := liveFrameworkErrorMessage(t, wrapped[0], map[string]any{}, tc.confirmation)
			if message != tc.want {
				t.Fatal("incorrect live wrapped error")
			}
		})
	}
}

// A caller can capture a framework error without importing its internal type.
func TestStreamingHandlerReyieldedFrameworkError(t *testing.T) {
	type args struct{}
	source, err := functiontool.NewStreaming(functiontool.Config{Name: "source", RequireConfirmation: true}, func(agent.Context, args) iter.Seq2[string, error] {
		return func(func(string, error) bool) { t.Error("unapproved source handler ran") }
	})
	if err != nil {
		t.Fatal(err)
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
		t.Fatal("did not capture framework error")
	}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"}, func(agent.Context, args) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) { yield("", captured) }
	})
	if err != nil {
		t.Fatal(err)
	}
	result := liveToolFailure(t, lookup, map[string]any{}, nil)
	if result.Response["error"] != streamingToolInternalError("lookup") {
		t.Fatal("live handler re-yielded a trusted framework error")
	}
}
