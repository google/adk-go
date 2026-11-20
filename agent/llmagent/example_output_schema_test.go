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

package llmagent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// answersOnceModel prints the parameters of the set_model_response
// declaration it receives and answers with one call of that tool.
type answersOnceModel struct {
	args map[string]any
}

func (*answersOnceModel) Name() string { return "scripted" }

func (m *answersOnceModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	for _, t := range req.Config.Tools {
		for _, d := range t.FunctionDeclarations {
			if d.Name == "set_model_response" {
				params, err := json.Marshal(d.ParametersJsonSchema)
				if err != nil {
					log.Fatal(err)
				}
				fmt.Println("declared:", string(params))
			}
		}
	}
	call := genai.NewContentFromFunctionCall("set_model_response", m.args, genai.RoleModel)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: call}, nil)
	}
}

// OutputJsonSchema takes a raw JSON schema, so it can describe a map with
// additionalProperties, which a *genai.Schema cannot. With a tool beside it,
// on a model that does not declare model.Capabilities.OutputSchemaAndTools,
// the schema reaches the model as the "response" parameter of
// set_model_response. The answer is passed through without validation: it is
// the function response, it is stored in the event's
// Actions.SetModelResponse, and it becomes the agent's final text.
func ExampleConfig_outputJsonSchema() {
	lookup, err := functiontool.New(functiontool.Config{Name: "count_words", Description: "Counts words."},
		func(agent.Context, struct{ Text string }) (map[string]any, error) { return nil, nil })
	if err != nil {
		log.Fatal(err)
	}
	llm := &answersOnceModel{args: map[string]any{"response": map[string]any{"cat": 2, "dog": 1}}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "word_counter",
		Model: llm,
		Tools: []tool.Tool{lookup},
		OutputJsonSchema: &jsonschema.Schema{
			Type:                 "object",
			AdditionalProperties: &jsonschema.Schema{Type: "integer"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.NewInMemory("app", a)
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("Count the animals.", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionResponse != nil:
				fmt.Printf("response: %v\n", p.FunctionResponse.Response)
				fmt.Printf("stored: %v\n", ev.Actions.SetModelResponse)
			case p.Text != "":
				fmt.Println("final:", p.Text)
			}
		}
	}
	// Output:
	// declared: {"type":"object","properties":{"response":{"type":"object","additionalProperties":{"type":"integer"}}},"required":["response"]}
	// response: map[cat:2 dog:1]
	// stored: map[cat:2 dog:1]
	// final: {"cat":2,"dog":1}
}
