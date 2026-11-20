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

package feature_test

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// declarationModel prints which form of the set_model_response declaration
// it receives, and answers with a valid call of that tool so the run ends.
type declarationModel struct{}

func (declarationModel) Name() string { return "scripted" }

func (declarationModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	for _, t := range req.Config.Tools {
		for _, d := range t.FunctionDeclarations {
			if d.Name != "set_model_response" {
				continue
			}
			form, params := "ParametersJsonSchema", d.ParametersJsonSchema
			if d.Parameters != nil {
				form, params = "Parameters", d.Parameters
			}
			b, err := json.Marshal(params)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("%s: %s\n", form, b)
		}
	}
	args := map[string]any{"name": "Ada", "home": map[string]any{"city": "London"}}
	call := genai.NewContentFromFunctionCall("set_model_response", args, genai.RoleModel)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: call}, nil)
	}
}

// The switch travels in the context given to runner.Run. With it on, the
// set_model_response tool an agent adds declares its parameters as a typed
// *genai.Schema in Parameters, with every nested object inline and no
// titles. Without it, the declaration is a JSON schema in
// ParametersJsonSchema, and a titled nested object appears once under
// "$defs".
func ExampleWithLegacyFunctionDeclarationSchema() {
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup_person", Description: "Looks up a person."},
		func(agent.Context, struct{ Name string }) (map[string]any, error) { return nil, nil })
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "person_finder",
		Model: declarationModel{},
		Tools: []tool.Tool{lookup},
		OutputSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
				"home": {
					Type:       genai.TypeObject,
					Title:      "Address",
					Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
					Required:   []string{"city"},
				},
			},
			Required: []string{"name", "home"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.NewInMemory("app", a)
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("Who is Ada?", genai.RoleUser)
	for _, ctx := range []context.Context{
		context.Background(),
		feature.WithLegacyFunctionDeclarationSchema(context.Background(), true),
	} {
		for _, err := range r.Run(ctx, "user", "session", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
	// Output:
	// ParametersJsonSchema: {"type":"object","properties":{"home":{"$ref":"#/$defs/Address"},"name":{"type":"string"}},"$defs":{"Address":{"type":"object","properties":{"city":{"type":"string"}},"title":"Address","required":["city"]}},"required":["name","home"]}
	// Parameters: {"properties":{"home":{"properties":{"city":{"type":"STRING"}},"required":["city"],"type":"OBJECT"},"name":{"type":"STRING"}},"required":["name","home"],"type":"OBJECT"}
}
