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
	"encoding/json"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

// cities is an output schema: a list of records.
var cities = &genai.Schema{
	Type: genai.TypeArray,
	Items: &genai.Schema{
		Type:  genai.TypeObject,
		Title: "City",
		Properties: map[string]*genai.Schema{
			"name":       {Type: genai.TypeString},
			"population": {Type: genai.TypeInteger},
		},
		Required: []string{"name", "population"},
	},
}

// The tool declares one parameter, items, for a list of records.
func ExampleNew() {
	tl, err := setmodelresponsetool.New(cities)
	if err != nil {
		log.Fatal(err)
	}
	decl := tl.(interface {
		Declaration() *genai.FunctionDeclaration
	}).Declaration()

	params, err := json.Marshal(decl.ParametersJsonSchema)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(decl.Name)
	fmt.Println(string(params))
	// Output:
	// set_model_response
	// {"type":"object","properties":{"items":{"type":"array","items":{"$ref":"#/$defs/City"}}},"$defs":{"City":{"type":"object","properties":{"name":{"type":"string"},"population":{"type":"integer"}},"title":"City","required":["name","population"]}},"required":["items"]}
}

// scriptedModel answers each request with the next set_model_response call
// of its script.
type scriptedModel struct {
	answers []map[string]any
	turn    int
}

func (m *scriptedModel) Name() string { return "scripted" }

// Capabilities declares that the model cannot take an output schema and tools
// in one request, so an agent with tools gives it set_model_response instead.
func (m *scriptedModel) Capabilities() model.Capabilities {
	return model.Capabilities{OutputSchemaAndTools: false}
}

func (m *scriptedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		call := &genai.FunctionCall{Name: setmodelresponsetool.Name, Args: m.answers[m.turn]}
		m.turn++
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: call}}}}, nil)
	}
}

// An agent with an output schema and a tool, whose model does not declare
// model.Capabilities.OutputSchemaAndTools, answers through set_model_response.
// An answer that does not match the schema goes back to the model as the
// feedback in the function response, and a valid answer, with "709000"
// converted to a number, becomes the agent's final text.
func Example_agentWithTools() {
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup_city", Description: "Looks up a city."},
		func(agent.Context, struct{ Name string }) (map[string]any, error) { return nil, nil })
	if err != nil {
		log.Fatal(err)
	}
	llm := &scriptedModel{answers: []map[string]any{
		{"items": []any{map[string]any{"name": "Oslo"}}},
		{"items": []any{map[string]any{"name": "Oslo", "population": "709000"}}},
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:         "city_finder",
		Model:        llm,
		Tools:        []tool.Tool{lookup},
		OutputSchema: cities,
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.NewInMemory("app", a)
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("Which cities?", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Println("call:", p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Printf("response: %v\n", p.FunctionResponse.Response)
			case p.Text != "":
				fmt.Println("final:", p.Text)
			}
		}
	}
	// Output:
	// call: set_model_response
	// response: map[error:Validation Error found:
	// 1 validation error for list[City]
	// 0.population
	//   Field required [type=missing]
	// Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types.]
	// call: set_model_response
	// response: map[result:[map[name:Oslo population:709000]]]
	// final: [{"name":"Oslo","population":709000}]
}
