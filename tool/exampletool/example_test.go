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

package exampletool_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/tool/exampletool"
)

// keywordProvider returns the stored examples whose input text shares a word
// with the user query.
type keywordProvider struct {
	examples []*exampletool.Example
}

func (p *keywordProvider) Examples(_ context.Context, query string) ([]*exampletool.Example, error) {
	var matched []*exampletool.Example
	for _, ex := range p.examples {
		for _, word := range strings.Fields(query) {
			if strings.Contains(ex.Input.Parts[0].Text, word) {
				matched = append(matched, ex)
				break
			}
		}
	}
	return matched, nil
}

// This example configures the example tool with a Provider instead of a fixed
// list. On each turn, the tool's ProcessRequest joins the text of the user
// message into a query, calls Provider.Examples with it, and appends the
// returned examples to the system instruction of the model request.
func ExampleNew_provider() {
	provider := &keywordProvider{
		examples: []*exampletool.Example{
			{
				Input:  genai.NewContentFromText("What is 2 + 2?", genai.RoleUser),
				Output: []*genai.Content{genai.NewContentFromText("4", genai.RoleModel)},
			},
			{
				Input:  genai.NewContentFromText("What is the capital of France?", genai.RoleUser),
				Output: []*genai.Content{genai.NewContentFromText("Paris", genai.RoleModel)},
			},
		},
	}

	et, err := exampletool.New(exampletool.ExampleToolConfig{
		Provider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(et.Name())

	// The tool makes this call with the user's text on each turn.
	examples, err := provider.Examples(context.Background(), "capital of Spain")
	if err != nil {
		log.Fatal(err)
	}
	for _, ex := range examples {
		fmt.Println(ex.Input.Parts[0].Text)
	}
	// Output:
	// example_tool
	// What is the capital of France?
}
