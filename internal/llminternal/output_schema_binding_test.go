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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
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
		want      *genai.Schema
	}{
		{
			name:      "bound schema",
			agentName: "self",
			state:     &State{},
			want:      bound,
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
			base := t.Context()
			if tc.parent != nil {
				base = parentmap.ToContext(base, parentmap.Map{tc.agentName: tc.parent})
			}
			ctx := icontext.NewInvocationContext(
				WithBoundOutputSchema(base, bindName, bindState, bound),
				icontext.InvocationContextParams{Agent: a},
			)

			got := outputSchemaFor(ctx, tc.state)

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
		WithBoundOutputSchema(t.Context(), "self", state, bound),
		icontext.InvocationContextParams{Agent: a},
	)

	first := outputSchemaFor(ctx, state)
	first.Properties["a"].Description = "edited"
	first.Properties["b"] = &genai.Schema{Type: genai.TypeString}
	second := outputSchemaFor(ctx, state)

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
	outer := WithBoundOutputSchema(t.Context(), "self", state, bound)
	ctx := icontext.NewInvocationContext(
		WithBoundOutputSchema(outer, "self", state, nil),
		icontext.InvocationContextParams{Agent: a},
	)

	if got := outputSchemaFor(ctx, state); got != nil {
		t.Errorf("outputSchemaFor = %v under a nil inner binding, want nil", got)
	}
}
