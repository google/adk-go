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

package setmodelresponsetool_test

import (
	"context"
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

// callingModel records each request. It answers the first with a single
// set_model_response call carrying args, and any later one with the text
// "again", so a run that the call does not end stops after two requests.
type callingModel struct {
	args     map[string]any
	requests []*model.LLMRequest
}

func (m *callingModel) Name() string { return "calling" }

func (m *callingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.requests = append(m.requests, req)
		if len(m.requests) > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("again", genai.RoleModel)}, nil)
			return
		}
		call := &genai.FunctionCall{Name: setmodelresponsetool.Name, Args: m.args}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: call}}}}, nil)
	}
}

func lookupTool(t *testing.T) tool.Tool {
	t.Helper()
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup_city", Description: "Looks up a city."},
		func(agent.Context, struct{ Name string }) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	return lookup
}

// runAgent runs cfg once with the message "q" and returns the events, or the
// first run error.
func runAgent(t *testing.T, cfg llmagent.Config) ([]agentEvent, error) {
	t.Helper()
	return runAgentContext(t, t.Context(), cfg)
}

// runAgentContext is runAgent with the run's context.
func runAgentContext(t *testing.T, ctx context.Context, cfg llmagent.Config) ([]agentEvent, error) {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	r, err := runner.NewInMemory("app", a)
	if err != nil {
		t.Fatalf("runner.NewInMemory() error: %v", err)
	}
	var events []agentEvent
	for ev, err := range r.Run(ctx, "user", "session", genai.NewContentFromText("q", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return events, err
		}
		events = append(events, agentEvent{text: textOf(ev.Content), final: ev.IsFinalResponse()})
	}
	return events, nil
}

type agentEvent struct {
	text  string
	final bool
}

func textOf(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

// declaredNames lists the function declarations of req in order.
func declaredNames(req *model.LLMRequest) []string {
	var names []string
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

func declarationNamed(t *testing.T, req *model.LLMRequest, name string) *genai.FunctionDeclaration {
	t.Helper()
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			if d.Name == name {
				return d
			}
		}
	}
	t.Fatalf("request declares no %q", name)
	return nil
}

func answerSchema() *genai.Schema {
	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
		Required:   []string{"answer"},
	}
}

// TestConfigTools_ToolEndsTheRun fails if a tool of this package cannot be
// listed in llmagent.Config.Tools (the flow refuses a tool without
// ProcessRequest before calling the model), if ProcessRequest does not put the
// tool's own declaration in the request, or if AsLegacy's ProcessRequest packs
// the JSON form instead of the typed one.
func TestConfigTools_ToolEndsTheRun(t *testing.T) {
	newTool := func(t *testing.T) tool.Tool {
		t.Helper()
		tl, err := setmodelresponsetool.New(answerSchema())
		if err != nil {
			t.Fatal(err)
		}
		return tl
	}
	tests := []struct {
		name      string
		tool      func(t *testing.T) tool.Tool
		args      map[string]any
		wantDecl  string
		wantFinal string
	}{
		{
			name: "new",
			tool: newTool,
			args: map[string]any{"answer": "hi"},
			wantDecl: `{"description":"Set your final response using the required output schema.\n\nUse this tool to provide your final structured answer instead\nof outputting text directly.",` +
				`"name":"set_model_response",` +
				`"parametersJsonSchema":{"properties":{"answer":{"type":"STRING"}},"required":["answer"],"type":"OBJECT"}}`,
			wantFinal: `{"answer":"hi"}`,
		},
		{
			name: "new_from_json_schema",
			tool: func(t *testing.T) tool.Tool {
				t.Helper()
				tl, err := setmodelresponsetool.NewFromJSONSchema(&jsonschema.Schema{
					Type:                 "object",
					AdditionalProperties: &jsonschema.Schema{Type: "integer"},
				})
				if err != nil {
					t.Fatal(err)
				}
				return tl
			},
			args: map[string]any{"response": map[string]any{"a": float64(1)}},
			wantDecl: `{"description":"Set your final response using the required output schema.\n\nUse this tool to provide your final structured answer instead\nof outputting text directly.",` +
				`"name":"set_model_response",` +
				`"parametersJsonSchema":{"properties":{"response":{"additionalProperties":{"type":"integer"},"type":"object"}},"required":["response"],"type":"object"}}`,
			wantFinal: `{"a":1}`,
		},
		{
			name: "as_legacy",
			tool: func(t *testing.T) tool.Tool {
				t.Helper()
				return setmodelresponsetool.AsLegacy(newTool(t))
			},
			args: map[string]any{"answer": "hi"},
			wantDecl: `{"description":"Set your final response using the required output schema.\n\nUse this tool to provide your final structured answer instead\nof outputting text directly.",` +
				`"name":"set_model_response",` +
				`"parameters":{"properties":{"answer":{"type":"STRING"}},"required":["answer"],"type":"OBJECT"}}`,
			wantFinal: `{"answer":"hi"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &callingModel{args: tc.args}

			events, err := runAgent(t, llmagent.Config{
				Name:  "finder",
				Model: llm,
				Tools: []tool.Tool{lookupTool(t), tc.tool(t)},
			})
			if err != nil {
				t.Fatalf("run error: %v", err)
			}

			if len(llm.requests) != 1 {
				t.Fatalf("model called %d times, want 1", len(llm.requests))
			}
			req := llm.requests[0]
			if diff := cmp.Diff([]string{"lookup_city", "set_model_response"}, declaredNames(req)); diff != "" {
				t.Errorf("declarations mismatch (-want +got):\n%s", diff)
			}
			if got := canonicalJSON(t, declarationNamed(t, req, setmodelresponsetool.Name)); got != tc.wantDecl {
				t.Errorf("set_model_response declaration =\n%s\nwant\n%s", got, tc.wantDecl)
			}
			if len(events) == 0 {
				t.Fatal("run yielded no events")
			}
			last := events[len(events)-1]
			if want := (agentEvent{text: tc.wantFinal, final: true}); last != want {
				t.Errorf("last event = %+v, want %+v", last, want)
			}
		})
	}
}

// TestConfigTools_LegacySwitch fails if a listed tool from New ignores
// feature.WithLegacyFunctionDeclarationSchema on the run's context and packs
// the JSON form, or if the typed tool it packs is not the one the flow runs.
func TestConfigTools_LegacySwitch(t *testing.T) {
	smr, err := setmodelresponsetool.New(answerSchema())
	if err != nil {
		t.Fatal(err)
	}
	llm := &callingModel{args: map[string]any{"answer": "hi"}}
	ctx := feature.WithLegacyFunctionDeclarationSchema(t.Context(), true)

	events, err := runAgentContext(t, ctx, llmagent.Config{
		Name:  "finder",
		Model: llm,
		Tools: []tool.Tool{lookupTool(t), smr},
	})
	if err != nil {
		t.Fatalf("run error: %v", err)
	}

	if len(llm.requests) != 1 {
		t.Fatalf("model called %d times, want 1", len(llm.requests))
	}
	const wantDecl = `{"description":"Set your final response using the required output schema.\n\nUse this tool to provide your final structured answer instead\nof outputting text directly.",` +
		`"name":"set_model_response",` +
		`"parameters":{"properties":{"answer":{"type":"STRING"}},"required":["answer"],"type":"OBJECT"}}`
	if got := canonicalJSON(t, declarationNamed(t, llm.requests[0], setmodelresponsetool.Name)); got != wantDecl {
		t.Errorf("set_model_response declaration =\n%s\nwant\n%s", got, wantDecl)
	}
	if len(events) == 0 {
		t.Fatal("run yielded no events")
	}
	if last, want := events[len(events)-1], (agentEvent{text: `{"answer":"hi"}`, final: true}); last != want {
		t.Errorf("last event = %+v, want %+v", last, want)
	}
}

// TestConfigTools_WithOutputSchemaIsDuplicate fails if ProcessRequest drops the
// error of toolutils.PackTool: an agent with an OutputSchema adds its own
// set_model_response tool, so a second one in Tools is a duplicate.
func TestConfigTools_WithOutputSchemaIsDuplicate(t *testing.T) {
	tests := []struct {
		name string
		wrap func(tool.Tool) tool.Tool
	}{
		{name: "new", wrap: func(tl tool.Tool) tool.Tool { return tl }},
		{name: "as_legacy", wrap: setmodelresponsetool.AsLegacy},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			smr, err := setmodelresponsetool.New(answerSchema())
			if err != nil {
				t.Fatal(err)
			}
			llm := &callingModel{args: map[string]any{"answer": "hi"}}

			_, err = runAgent(t, llmagent.Config{
				Name:         "finder",
				Model:        llm,
				Tools:        []tool.Tool{lookupTool(t), tc.wrap(smr)},
				OutputSchema: answerSchema(),
			})

			const want = `duplicate tool: "set_model_response"`
			if err == nil || err.Error() != want {
				t.Errorf("run error = %v, want %q", err, want)
			}
			if len(llm.requests) != 0 {
				t.Errorf("model called %d times, want 0", len(llm.requests))
			}
		})
	}
}
