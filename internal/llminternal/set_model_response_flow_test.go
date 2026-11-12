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
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

func functionResponseEvent(name string, response map[string]any, stored any) *session.Event {
	return &session.Event{
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: name, Response: response}}},
		}},
		Actions: session.EventActions{SetModelResponse: stored},
	}
}

func TestStructuredFinalEvent_GatesOnActionAndFormatsBareValue(t *testing.T) {
	a := utils.Must(agent.New(agent.Config{Name: "SchemaAgent"}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})

	tests := []struct {
		name     string
		ev       *session.Event
		wantText string // empty: no final event
	}{
		{
			name: "error response with nothing stored",
			ev:   functionResponseEvent("set_model_response", map[string]any{"error": "bad"}, nil),
		},
		{
			name:     "stored empty list is the text, not the result wrapper",
			ev:       functionResponseEvent("set_model_response", map[string]any{"result": []any{}}, []any{}),
			wantText: "[]",
		},
		{
			name:     "stored record list is the text, not the result wrapper",
			ev:       functionResponseEvent("set_model_response", map[string]any{"result": []any{map[string]any{"id": float64(1)}}}, []any{map[string]any{"id": float64(1)}}),
			wantText: `[{"id":1}]`,
		},
		{
			name: "stored value without a set_model_response part",
			ev:   functionResponseEvent("other_tool", map[string]any{"a": float64(1)}, map[string]any{"a": float64(1)}),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := structuredFinalEvent(ctx, tc.ev)
			if err != nil {
				t.Fatalf("structuredFinalEvent() error = %v", err)
			}
			if tc.wantText == "" {
				if got != nil {
					t.Fatalf("structuredFinalEvent() = event with content %+v, want nil", got.Content)
				}
				return
			}
			if got == nil {
				t.Fatal("structuredFinalEvent() = nil, want a final event")
			}
			want := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: tc.wantText}}}
			if diff := cmp.Diff(want, got.Content); diff != "" {
				t.Errorf("final event content mismatch (-want +got):\n%s", diff)
			}
			if got.Author != "SchemaAgent" {
				t.Errorf("final event Author = %q, want %q", got.Author, "SchemaAgent")
			}
		})
	}
}

// itemListSchema is a list-of-records output schema: list[Item] with Item
// {id: int, name: str}.
func itemListSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:  genai.TypeObject,
			Title: "Item",
			Properties: map[string]*genai.Schema{
				"id":   {Type: genai.TypeInteger},
				"name": {Type: genai.TypeString},
			},
			Required: []string{"id", "name"},
		},
	}
}

func setModelResponseCallTurn(id string, args map[string]any) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "set_model_response", Args: args}}},
	}}
}

// newOutputSchemaFlow returns a flow whose only request processor is
// outputSchemaRequestProcessor, run for an LLM agent that has the output
// schema and one other tool, so set_model_response reaches the model the way
// it does in production: packed into the request, not listed in Flow.Tools.
func newOutputSchemaFlow(t *testing.T, m model.LLM, schema *genai.Schema) (*Flow, agent.InvocationContext) {
	t.Helper()
	llmAgent := &mockLLMAgent{
		Agent: utils.Must(agent.New(agent.Config{Name: "SchemaAgent"})),
		s: &State{
			Model:        m,
			OutputSchema: schema,
			Tools:        []tool.Tool{&mockFunctionTool{name: "other_tool"}},
		},
	}
	f := &Flow{
		Model:             m,
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{outputSchemaRequestProcessor},
	}
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{InvocationID: "inv_1", Agent: llmAgent},
	)
	return f, ctx
}

// runFlow collects the events of f.Run, failing on an error or a run that
// does not end.
func runFlow(t *testing.T, f *Flow, ctx agent.InvocationContext) []*session.Event {
	t.Helper()
	const safetyLimit = 20
	var events []*session.Event
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		events = append(events, ev)
		if len(events) > safetyLimit {
			t.Fatalf("Run() did not end after %d events", safetyLimit)
		}
	}
	return events
}

func functionResponseOf(t *testing.T, ev *session.Event) *genai.FunctionResponse {
	t.Helper()
	frs := utils.FunctionResponses(ev.Content)
	if len(frs) != 1 {
		t.Fatalf("event has %d function responses, want 1: %+v", len(frs), ev.Content)
	}
	return frs[0]
}

func TestFlow_SetModelResponseRetryAndFinalEvent(t *testing.T) {
	m := &scriptedThinkingModel{turns: []*model.LLMResponse{
		setModelResponseCallTurn("fc-1", map[string]any{"items": []any{map[string]any{"id": "bad", "name": "one"}}}),
		setModelResponseCallTurn("fc-2", map[string]any{"items": []any{map[string]any{"id": float64(1), "name": "one"}}}),
	}}
	f, ctx := newOutputSchemaFlow(t, m, itemListSchema())
	var toolErrs []error
	f.OnToolErrorCallbacks = []OnToolErrorCallback{func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
		toolErrs = append(toolErrs, err)
		return nil, nil
	}}
	var afterToolSawErr []bool
	f.AfterToolCallbacks = []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
		afterToolSawErr = append(afterToolSawErr, err != nil)
		return nil, nil
	}}

	events := runFlow(t, f, ctx)

	if len(events) != 5 {
		t.Fatalf("Run() yielded %d events, want 5 (call, feedback, call, response, final text)", len(events))
	}
	if m.calls != 2 {
		t.Errorf("model called %d times, want 2", m.calls)
	}

	feedback := functionResponseOf(t, events[1])
	errText, _ := feedback.Response["error"].(string)
	for _, want := range []string{"Validation Error found:", "1 validation error for list[Item]", "0.id", "[type=int_parsing]"} {
		if !strings.Contains(errText, want) {
			t.Errorf("feedback %q does not contain %q", errText, want)
		}
	}
	if events[1].Actions.SetModelResponse != nil {
		t.Errorf("feedback event stored %v, want nothing", events[1].Actions.SetModelResponse)
	}
	if len(toolErrs) != 1 || !strings.HasPrefix(toolErrs[0].Error(), "Validation Error found:") {
		t.Errorf("on-tool-error callbacks saw %v, want one validation error", toolErrs)
	}
	if diff := cmp.Diff([]bool{true, false}, afterToolSawErr); diff != "" {
		t.Errorf("after-tool callbacks saw an error per call (-want +got):\n%s", diff)
	}

	answer := []any{map[string]any{"id": float64(1), "name": "one"}}
	if diff := cmp.Diff(map[string]any{"result": answer}, functionResponseOf(t, events[3]).Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(answer, events[3].Actions.SetModelResponse); diff != "" {
		t.Errorf("stored answer mismatch (-want +got):\n%s", diff)
	}

	wantFinal := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: `[{"id":1,"name":"one"}]`}}}
	if diff := cmp.Diff(wantFinal, events[4].Content); diff != "" {
		t.Errorf("final event content mismatch (-want +got):\n%s", diff)
	}
}

func TestFlow_SetModelResponseBeforeToolCallbackShortCircuits(t *testing.T) {
	m := &scriptedThinkingModel{turns: []*model.LLMResponse{
		setModelResponseCallTurn("fc-1", map[string]any{"items": []any{map[string]any{"id": float64(1), "name": "one"}}}),
		{Content: genai.NewContentFromText("done", "model")},
	}}
	f, ctx := newOutputSchemaFlow(t, m, itemListSchema())
	f.BeforeToolCallbacks = []BeforeToolCallback{func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
		return map[string]any{"custom": "override"}, nil
	}}

	events := runFlow(t, f, ctx)

	if len(events) != 3 {
		t.Fatalf("Run() yielded %d events, want 3 (call, overridden response, text)", len(events))
	}
	if m.calls != 2 {
		t.Errorf("model called %d times, want 2", m.calls)
	}
	if diff := cmp.Diff(map[string]any{"custom": "override"}, functionResponseOf(t, events[1]).Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	if events[1].Actions.SetModelResponse != nil {
		t.Errorf("overridden call stored %v, want nothing", events[1].Actions.SetModelResponse)
	}
	if diff := cmp.Diff(genai.NewContentFromText("done", "model"), events[2].Content); diff != "" {
		t.Errorf("event after the response mismatch (-want +got):\n%s", diff)
	}
}

func TestMergeEventActions_SetModelResponse(t *testing.T) {
	tests := []struct {
		name  string
		base  any
		other any
		want  any
	}{
		{
			name:  "two maps are deep merged",
			base:  map[string]any{"a": 1, "nested": map[string]any{"x": 1}},
			other: map[string]any{"b": 2, "nested": map[string]any{"y": 2}},
			want:  map[string]any{"a": 1, "b": 2, "nested": map[string]any{"x": 1, "y": 2}},
		},
		{
			name:  "nothing stored before takes the later list",
			base:  nil,
			other: []any{"a"},
			want:  []any{"a"},
		},
		{
			name:  "later list replaces earlier list",
			base:  []any{"a"},
			other: []any{"b"},
			want:  []any{"b"},
		},
		{
			name:  "later list replaces earlier map",
			base:  map[string]any{"a": 1},
			other: []any{"b"},
			want:  []any{"b"},
		},
		{
			name:  "later map replaces earlier list",
			base:  []any{"a"},
			other: map[string]any{"b": 2},
			want:  map[string]any{"b": 2},
		},
		{
			name:  "nothing stored later keeps the earlier answer",
			base:  []any{"a"},
			other: nil,
			want:  []any{"a"},
		},
		{
			name:  "later empty list is an answer and replaces",
			base:  []any{"a"},
			other: []any{},
			want:  []any{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := &session.EventActions{SetModelResponse: tc.base}
			other := &session.EventActions{SetModelResponse: tc.other}
			got := mergeEventActions(base, other)
			if diff := cmp.Diff(tc.want, got.SetModelResponse); diff != "" {
				t.Errorf("merged SetModelResponse mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
