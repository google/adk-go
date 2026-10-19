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

package main

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// scriptedModel replays fixed responses and records the tools declared in
// each request.
type scriptedModel struct {
	responses []*genai.Part
	requests  []*model.LLMRequest
}

func (*scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(m.requests) == len(m.responses) {
			yield(nil, fmt.Errorf("unexpected model call %d", len(m.requests)+1))
			return
		}
		part := m.responses[len(m.requests)]
		m.requests = append(m.requests, req)
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}}}, nil)
	}
}

func declaredNames(req *model.LLMRequest) []string {
	var names []string
	if req.Config != nil {
		for _, t := range req.Config.Tools {
			for _, d := range t.FunctionDeclarations {
				names = append(names, d.Name)
			}
		}
	}
	slices.Sort(names)
	return names
}

func call(name string, args map[string]any) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{Name: name, Args: args}}
}

// TestDiscoveryThroughRunner checks that a gated tool is declared only after
// search_tools returns it, that the model can then call it in the same
// invocation, and that it stays declared in the next turn.
func TestDiscoveryThroughRunner(t *testing.T) {
	m := &scriptedModel{responses: []*genai.Part{
		call("search_tools", map[string]any{"query": "multiply"}),
		call("multiply_numbers", map[string]any{"a": 6, "b": 7}),
		genai.NewPartFromText("42"),
		genai.NewPartFromText("hello"),
	}}
	a, err := newAgent(m)
	if err != nil {
		t.Fatalf("newAgent() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var results []*genai.FunctionResponse
	for _, prompt := range []string{"Multiply 6 by 7.", "Hi"} {
		for event, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText(prompt, genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run(%q) error = %v", prompt, err)
			}
			if event.Content == nil {
				continue
			}
			for _, p := range event.Content.Parts {
				if p.FunctionResponse != nil {
					results = append(results, p.FunctionResponse)
				}
			}
		}
	}

	if len(m.requests) != len(m.responses) {
		t.Fatalf("model called %d times, want %d", len(m.requests), len(m.responses))
	}
	initial := []string{"get_current_time", "search_tools"}
	discovered := []string{"get_current_time", "multiply_numbers", "search_tools"}
	for i, want := range [][]string{initial, discovered, discovered, discovered} {
		if diff := cmp.Diff(want, declaredNames(m.requests[i])); diff != "" {
			t.Errorf("request %d declared tools mismatch (-want +got):\n%s", i+1, diff)
		}
	}
	if len(results) != 2 || results[1].Name != "multiply_numbers" || results[1].Response["value"] != 42.0 {
		t.Errorf("function responses = %+v, want search_tools then multiply_numbers with value 42", results)
	}
}
