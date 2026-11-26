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
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

func TestOutputSchemaFor(t *testing.T) {
	bound := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeString}}}
	declared := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"b": {Type: genai.TypeString}}}
	sub := utils.Must(agent.New(agent.Config{Name: "sub"}))
	parent := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "parent"})), s: &State{}}

	for _, tc := range []struct {
		name      string
		agentName string
		state     *State
		subAgents []agent.Agent
		parent    agent.Agent                     // an LLM agent that lists this one as a sub-agent
		bindTo    func(s *State) (string, *State) // which agent the binding names
		mode      Mode                            // bound for this agent; ModeSingleTurn when unset
		want      *genai.Schema
	}{
		{
			name:      "bound schema",
			agentName: "self",
			state:     &State{},
			want:      bound,
		},
		{
			name:      "chat mode bound for this agent",
			agentName: "self",
			state:     &State{},
			mode:      ModeChat,
		},
		{
			name:      "task mode bound for this agent",
			agentName: "self",
			state:     &State{},
			mode:      ModeTask,
		},
		{
			name:      "declared OutputSchema wins",
			agentName: "self",
			state:     &State{OutputSchema: declared},
			want:      declared,
		},
		{
			name:      "GenerateContentConfig.ResponseSchema counts as declared",
			agentName: "self",
			state:     &State{GenerateContentConfig: &genai.GenerateContentConfig{ResponseSchema: declared}},
		},
		{
			name:      "GenerateContentConfig.ResponseJsonSchema counts as declared",
			agentName: "self",
			state:     &State{GenerateContentConfig: &genai.GenerateContentConfig{ResponseJsonSchema: map[string]any{"type": "object"}}},
		},
		{
			name:      "GenerateContentConfig.ResponseMIMEType counts as declared",
			agentName: "self",
			state:     &State{GenerateContentConfig: &genai.GenerateContentConfig{ResponseMIMEType: "text/plain"}},
		},
		{
			name:      "agent that can transfer",
			agentName: "self",
			state:     &State{},
			subAgents: []agent.Agent{sub},
		},
		{
			name:      "agent that can transfer to its parent",
			agentName: "self",
			state:     &State{},
			parent:    parent,
		},
		{
			name:      "agent under a parent with transfers disallowed",
			agentName: "self",
			state:     &State{DisallowTransferToParent: true, DisallowTransferToPeers: true},
			parent:    parent,
			want:      bound,
		},
		{
			name:      "binding for a same-named agent",
			agentName: "self",
			state:     &State{},
			bindTo:    func(*State) (string, *State) { return "self", &State{} },
		},
		{
			name:      "binding for the same state under another name",
			agentName: "self",
			state:     &State{},
			bindTo:    func(s *State) (string, *State) { return "peer", s },
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bindName, bindState := tc.agentName, tc.state
			if tc.bindTo != nil {
				bindName, bindState = tc.bindTo(tc.state)
			}
			a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: tc.agentName, SubAgents: tc.subAgents})), s: tc.state}
			mode := tc.mode
			if mode == ModeUnset {
				mode = ModeSingleTurn
			}
			base := WithBoundMode(t.Context(), tc.agentName, tc.state, mode)
			if tc.parent != nil {
				base = parentmap.ToContext(base, parentmap.Map{tc.agentName: tc.parent})
			}
			ctx := icontext.NewInvocationContext(
				WithBoundOutputSchema(base, bindName, bindState, bound),
				icontext.InvocationContextParams{Agent: a},
			)

			got, err := outputSchemaFor(ctx, tc.state)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("outputSchemaFor mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Each request gets its own copy of a bound schema, so editing one request's
// schema does not change what the next request is sent.
func TestOutputSchemaForCopiesBoundSchema(t *testing.T) {
	bound := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeString}}}
	state := &State{}
	a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "self"})), s: state}
	ctx := icontext.NewInvocationContext(
		WithBoundOutputSchema(WithBoundMode(t.Context(), "self", state, ModeSingleTurn), "self", state, bound),
		icontext.InvocationContextParams{Agent: a},
	)

	first, err := outputSchemaFor(ctx, state)
	if err != nil {
		t.Fatal(err)
	}
	first.Properties["a"].Description = "edited"
	first.Properties["b"] = &genai.Schema{Type: genai.TypeString}
	second, err := outputSchemaFor(ctx, state)
	if err != nil {
		t.Fatal(err)
	}

	want := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeString}}}
	if diff := cmp.Diff(want, second); diff != "" {
		t.Errorf("second request's schema changed after editing the first (-want +got):\n%s", diff)
	}
}

// A nil binding shadows an outer one, as an inner placement with no schema
// should.
func TestWithBoundOutputSchemaNilShadowsOuter(t *testing.T) {
	bound := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeString}}}
	state := &State{}
	a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "self"})), s: state}
	outer := WithBoundOutputSchema(WithBoundMode(t.Context(), "self", state, ModeSingleTurn), "self", state, bound)
	ctx := icontext.NewInvocationContext(
		WithBoundOutputSchema(outer, "self", state, nil),
		icontext.InvocationContextParams{Agent: a},
	)

	if got, err := outputSchemaFor(ctx, state); err != nil || got != nil {
		t.Fatal("outputSchemaFor did not return a nil schema under a nil inner binding")
	}
}

func TestOutputSchemaForCloneError(t *testing.T) {
	wantErr := errors.New("schema value cannot be marshaled")
	for _, tc := range []struct {
		name   string
		schema *genai.Schema
	}{
		{name: "default", schema: &genai.Schema{Default: &cloneTestJSON{err: wantErr}}},
		{name: "example", schema: &genai.Schema{Example: &cloneTestJSON{err: wantErr}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := &State{}
			a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "self"})), s: state}
			ctx := icontext.NewInvocationContext(
				WithBoundOutputSchema(WithBoundMode(t.Context(), "self", state, ModeSingleTurn), "self", state, tc.schema),
				icontext.InvocationContextParams{Agent: a},
			)

			got, err := outputSchemaFor(ctx, state)
			if got != nil || !errors.Is(err, wantErr) {
				t.Fatal("outputSchemaFor did not preserve the clone error")
			}
		})
	}
}

func TestRequestProcessors_BoundSchemaCloneError(t *testing.T) {
	wantErr := errors.New("schema value cannot be marshaled")
	for _, tc := range []struct {
		name      string
		processor func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]
		tools     []tool.Tool
	}{
		{name: "basic", processor: basicRequestProcessor},
		{name: "output schema", processor: outputSchemaRequestProcessor, tools: []tool.Tool{&mockTool{name: "other_tool"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := &State{Model: &mockLLM{name: "gemini-2.5-flash"}, Tools: tc.tools}
			schema := &genai.Schema{Default: &cloneTestJSON{err: wantErr}}
			a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "self"})), s: state}
			ctx := icontext.NewInvocationContext(
				WithBoundOutputSchema(WithBoundMode(t.Context(), "self", state, ModeSingleTurn), "self", state, schema),
				icontext.InvocationContextParams{Agent: a},
			)
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}
			errorsSeen := 0
			for ev, err := range tc.processor(ctx, req, &Flow{}) {
				if ev != nil || !errors.Is(err, wantErr) {
					t.Fatal("processor did not preserve the clone error")
				}
				errorsSeen++
			}
			if errorsSeen != 1 {
				t.Fatalf("processor returned %d errors, want one", errorsSeen)
			}
			if !cmp.Equal(req, &model.LLMRequest{Config: &genai.GenerateContentConfig{}}) {
				t.Fatal("processor changed the request after the clone failed")
			}
		})
	}
}
