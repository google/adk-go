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

package anthropicmodel

import (
	"fmt"
	"testing"

	"google.golang.org/genai"
)

func benchmarkDeclarations(n int) []*genai.FunctionDeclaration {
	decls := make([]*genai.FunctionDeclaration, n)
	for i := range n {
		decls[i] = &genai.FunctionDeclaration{
			Name:        fmt.Sprintf("tool_%02d", i),
			Description: fmt.Sprintf("Executes benchmark tool %d with structured parameters.", i),
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"query": {
						Type:        genai.TypeString,
						Description: "Search query string.",
					},
					"limit": {
						Type:        genai.TypeInteger,
						Description: "Maximum number of results.",
					},
					"tags": {
						Type: genai.TypeArray,
						Items: &genai.Schema{
							Type: genai.TypeString,
						},
					},
					"filter": {
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"include_archived": {Type: genai.TypeBoolean},
							"min_score":        {Type: genai.TypeNumber},
							"region": {
								Type: genai.TypeString,
								Enum: []string{"us", "eu", "apac"},
							},
						},
						Required: []string{"region"},
					},
				},
				Required: []string{"query"},
			},
		}
	}
	return decls
}

// BenchmarkConvertTools measures converting 20 function declarations into
// Anthropic tool definitions.
func BenchmarkConvertTools(b *testing.B) {
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{FunctionDeclarations: benchmarkDeclarations(20)},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		tools, _, err := buildToolsAndChoice(cfg, nil)
		if err != nil {
			b.Fatalf("buildToolsAndChoice failed: %v", err)
		}
		if len(tools) != 20 {
			b.Fatalf("len(tools) = %d, want 20", len(tools))
		}
	}
}
