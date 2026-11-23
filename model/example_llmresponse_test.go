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

package model_test

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// FunctionCalls lists the function calls a model response asks for, skipping
// its other parts. The same method is available on a session.Event.
func ExampleLLMResponse_FunctionCalls() {
	resp := &model.LLMResponse{Content: &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{
			{Text: "Let me check both cities."},
			{FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
			{FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "Rome"}}},
		},
	}}
	for _, fc := range resp.FunctionCalls() {
		fmt.Println(fc.Name, fc.Args["city"])
	}
	fmt.Println(len(resp.FunctionResponses()))
	// Output:
	// get_weather Paris
	// get_weather Rome
	// 0
}
