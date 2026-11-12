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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

func ExampleNew() {
	// The agent's output schema: a list of records.
	schema := &genai.Schema{
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
	tl, err := setmodelresponsetool.New(schema)
	if err != nil {
		panic(err)
	}
	// The flow runs the tool through these methods.
	ft := tl.(interface {
		Declaration() *genai.FunctionDeclaration
		Run(ctx agent.Context, args any) (map[string]any, error)
	})

	// In an agent run the flow supplies the tool context.
	ic := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
	ctx := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)

	params, _ := json.Marshal(ft.Declaration().ParametersJsonSchema)
	fmt.Println("parameters:", string(params))

	_, err = ft.Run(ctx, map[string]any{"items": []any{map[string]any{"name": "Oslo"}}})
	fmt.Println(err)

	result, err := ft.Run(ctx, map[string]any{"items": []any{
		map[string]any{"name": "Oslo", "population": "709000"},
	}})
	if err != nil {
		panic(err)
	}
	fmt.Println("result:", result)
	fmt.Println("stored:", ctx.Actions().SetModelResponse)
	// Output:
	// parameters: {"type":"object","properties":{"items":{"type":"array","items":{"$ref":"#/$defs/City"}}},"$defs":{"City":{"type":"object","properties":{"name":{"type":"string"},"population":{"type":"integer"}},"title":"City","required":["name","population"]}},"required":["items"]}
	// Validation Error found:
	// 1 validation error for list[City]
	// 0.population
	//   Field required [type=missing]
	// Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types.
	// result: map[result:[map[name:Oslo population:709000]]]
	// stored: [map[name:Oslo population:709000]]
}
