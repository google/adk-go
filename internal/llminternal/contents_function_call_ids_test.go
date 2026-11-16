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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/session"
)

const idsAgentName = "test_agent"

// capsModel is a model that declares fixed capabilities.
type capsModel struct {
	model.LLM
	caps model.Capabilities
}

func (m capsModel) Capabilities() model.Capabilities { return m.caps }

// forwardingModel wraps a model and returns the wrapped model's capabilities,
// as model.Capabilities asks of a wrapper.
type forwardingModel struct {
	model.LLM
}

func (m forwardingModel) Capabilities() model.Capabilities {
	return m.LLM.(model.CapabilitiesProvider).Capabilities()
}

// toolExchange returns a user turn, then a call to test_tool with id and its
// response, as the agent records them.
func toolExchange(id string) []*session.Event {
	return []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Call the tool", genai.RoleUser)}},
		callEvent(&genai.FunctionCall{ID: id, Name: "test_tool"}),
		responseEvent(&genai.FunctionResponse{ID: id, Name: "test_tool", Response: map[string]any{"result": "ok"}}),
	}
}

func callEvent(calls ...*genai.FunctionCall) *session.Event {
	content := &genai.Content{Role: genai.RoleModel}
	for _, c := range calls {
		content.Parts = append(content.Parts, &genai.Part{FunctionCall: c})
	}
	return &session.Event{Author: idsAgentName, LLMResponse: model.LLMResponse{Content: content}}
}

func responseEvent(responses ...*genai.FunctionResponse) *session.Event {
	content := &genai.Content{Role: genai.RoleUser}
	for _, r := range responses {
		content.Parts = append(content.Parts, &genai.Part{FunctionResponse: r})
	}
	return &session.Event{Author: idsAgentName, LLMResponse: model.LLMResponse{Content: content}}
}

// requestContents runs ContentsRequestProcessor for an agent whose model is m
// over events and returns the request contents.
func requestContents(t *testing.T, m model.LLM, include llmagent.IncludeContents, events []*session.Event, f *llminternal.Flow) []*genai.Content {
	t.Helper()
	a := utils.Must(llmagent.New(llmagent.Config{Name: idsAgentName, Model: m, IncludeContents: include}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   a,
		Session: &fakeSession{events: events},
	})
	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, f) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor() error = %v", err)
		}
	}
	return req.Contents
}

// functionIDs returns the ID of every function call and function response in
// contents, in order.
func functionIDs(contents []*genai.Content) []string {
	var ids []string
	for _, c := range contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				ids = append(ids, p.FunctionCall.ID)
			}
			if p.FunctionResponse != nil {
				ids = append(ids, p.FunctionResponse.ID)
			}
		}
	}
	return ids
}

func newTestAnthropicModel(t *testing.T) model.LLM {
	t.Helper()
	m, err := anthropicmodel.NewModel(t.Context(), "claude-3-7-sonnet-latest", anthropicmodel.ClientConfig{
		Backend: anthropicmodel.BackendAnthropic,
		APIKey:  "test-anthropic-key",
	})
	if err != nil {
		t.Fatalf("anthropicmodel.NewModel() error = %v", err)
	}
	return m
}

// TestContentsIDs_ProviderIDsSurviveStripping checks that, for a model that
// does not pair calls by ID, only the adk- IDs ADK generated leave the request
// and an ID the provider issued stays (adk-python
// flows/llm_flows/context/_contents.py _copy_content_for_request).
func TestContentsIDs_ProviderIDsSurviveStripping(t *testing.T) {
	t.Parallel()
	events := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Call the tools", genai.RoleUser)}},
		callEvent(&genai.FunctionCall{ID: "adk-1", Name: "a"}, &genai.FunctionCall{ID: "toolu_9", Name: "b"}),
		responseEvent(&genai.FunctionResponse{ID: "adk-1", Name: "a"}, &genai.FunctionResponse{ID: "toolu_9", Name: "b"}),
	}
	m := &testModel{}

	got := functionIDs(requestContents(t, m, llmagent.IncludeContentsDefault, events, &llminternal.Flow{Model: m}))

	want := []string{"", "toolu_9", "", "toolu_9"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("request function IDs mismatch (-want +got):\n%s", diff)
	}
}

// TestContentsIDs_StrippingLeavesSessionEventsAlone checks that removing adk-
// IDs from the request leaves the stored session events with their IDs
// (adk-python unit test test_contents.py, stripping does not mutate session
// events).
func TestContentsIDs_StrippingLeavesSessionEventsAlone(t *testing.T) {
	t.Parallel()
	events := toolExchange("adk-test-call-id")
	m := &testModel{}

	contents := requestContents(t, m, llmagent.IncludeContentsDefault, events, &llminternal.Flow{Model: m})

	if diff := cmp.Diff([]string{"", ""}, functionIDs(contents)); diff != "" {
		t.Errorf("request function IDs mismatch (-want +got):\n%s", diff)
	}
	if got := events[1].Content.Parts[0].FunctionCall.ID; got != "adk-test-call-id" {
		t.Errorf("session call ID = %q, want %q", got, "adk-test-call-id")
	}
	if got := events[2].Content.Parts[0].FunctionResponse.ID; got != "adk-test-call-id" {
		t.Errorf("session response ID = %q, want %q", got, "adk-test-call-id")
	}
}

// TestContentsIDs_ModelDeclarationDecidesKeepOrStrip checks that the request
// keeps adk- IDs exactly when the model's own Capabilities declares
// ClientFunctionCallIDs. A wrapper keeps them when it forwards the wrapped
// model's Capabilities, and a type that only embeds model.LLM strips them.
func TestContentsIDs_ModelDeclarationDecidesKeepOrStrip(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		model func(t *testing.T) model.LLM
		want  []string
	}{
		{
			name:  "declares_client_function_call_ids",
			model: func(*testing.T) model.LLM { return capsModel{caps: model.Capabilities{ClientFunctionCallIDs: true}} },
			want:  []string{"adk-1", "adk-1"},
		},
		{
			name:  "declares_only_output_schema_and_tools",
			model: func(*testing.T) model.LLM { return capsModel{caps: model.Capabilities{OutputSchemaAndTools: true}} },
			want:  []string{"", ""},
		},
		{
			name:  "no_capabilities_provider",
			model: func(*testing.T) model.LLM { return &testModel{} },
			want:  []string{"", ""},
		},
		{
			name:  "anthropicmodel",
			model: newTestAnthropicModel,
			want:  []string{"adk-1", "adk-1"},
		},
		{
			name:  "embedding_only_wrapper_around_anthropicmodel",
			model: func(t *testing.T) model.LLM { return struct{ model.LLM }{newTestAnthropicModel(t)} },
			want:  []string{"", ""},
		},
		{
			name:  "forwarding_wrapper_around_anthropicmodel",
			model: func(t *testing.T) model.LLM { return forwardingModel{newTestAnthropicModel(t)} },
			want:  []string{"adk-1", "adk-1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			m := tt.model(t)

			got := functionIDs(requestContents(t, m, llmagent.IncludeContentsDefault, toolExchange("adk-1"), &llminternal.Flow{Model: m}))

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("request function IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContentsIDs_CurrentTurnPathHonorsDeclaration checks that with
// IncludeContents "none" the request keeps or strips adk- IDs by the same rule
// as with the full history, both when a user turn starts the current turn and
// when no event can start it (adk-python passes preserve_function_call_ids to
// _get_current_turn_contents, flows/llm_flows/context/_contents.py:177).
func TestContentsIDs_CurrentTurnPathHonorsDeclaration(t *testing.T) {
	t.Parallel()
	const id = "adk-test-call-id"
	withUserTurn := toolExchange(id)
	withoutUserTurn := toolExchange(id)[1:]
	declaring := capsModel{caps: model.Capabilities{ClientFunctionCallIDs: true}}
	tests := []struct {
		name   string
		model  model.LLM
		events []*session.Event
		want   []string
	}{
		{name: "user_turn/declaring", model: declaring, events: withUserTurn, want: []string{id, id}},
		{name: "user_turn/not_declaring", model: &testModel{}, events: withUserTurn, want: []string{"", ""}},
		{name: "no_user_turn/declaring", model: declaring, events: withoutUserTurn, want: []string{id, id}},
		{name: "no_user_turn/not_declaring", model: &testModel{}, events: withoutUserTurn, want: []string{"", ""}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := functionIDs(requestContents(t, tt.model, llmagent.IncludeContentsNone, tt.events, &llminternal.Flow{Model: tt.model}))

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("request function IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContentsRequestProcessor_UsesAgentModelOutsideAStep checks that the
// processor reads the agent's own model when the Flow has none, as the other
// request processors do, so a declaring agent model keeps adk- IDs; a nil Flow
// reads the agent's model too.
func TestContentsRequestProcessor_UsesAgentModelOutsideAStep(t *testing.T) {
	t.Parallel()
	declaring := capsModel{caps: model.Capabilities{ClientFunctionCallIDs: true}}
	tests := []struct {
		name string
		flow *llminternal.Flow
	}{
		{name: "empty_flow", flow: &llminternal.Flow{}},
		{name: "nil_flow", flow: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := functionIDs(requestContents(t, declaring, llmagent.IncludeContentsDefault, toolExchange("adk-test-call-id"), tt.flow))

			want := []string{"adk-test-call-id", "adk-test-call-id"}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("request function IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
