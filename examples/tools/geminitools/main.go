// Copyright 2025 Google LLC
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

// Package main demonstrates how to use Gemini's built-in tools for grounding
// and enhanced capabilities. These tools operate internally within the model
// and do not require local code execution.
//
// This example shows:
//   - URLContext: Retrieve and analyze content from URLs
//   - GoogleMaps: Ground responses with Google Maps data (commented)
//   - EnterpriseWebSearch: Web search with enterprise compliance (commented)
//   - VertexAiSearch: Custom Vertex AI Search integration (commented)
//
// Usage:
//
//	export GOOGLE_API_KEY=your-api-key
//	go run main.go console
//
// Example prompts:
//   - "What does https://golang.org/doc/ say about getting started?"
//   - "Summarize the content at https://github.com/google/adk-go"
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create an agent with built-in Gemini tools for enhanced grounding
	tools := []tool.Tool{
		// URLContext retrieves content from URLs to inform responses
		geminitool.URLContext{},

		// Uncomment to add Google Maps grounding for location-based queries
		// geminitool.GoogleMaps{},

		// Uncomment to add enterprise-compliant web search
		// geminitool.EnterpriseWebSearch{},

		// For Vertex AI Search, configure with your data store or search engine:
		// &geminitool.VertexAiSearch{
		// 	DataStoreID: "projects/YOUR_PROJECT/locations/us/collections/default/dataStores/YOUR_DATASTORE",
		// 	MaxResults:  genai.Ptr(int32(5)),
		// },
		// OR with search engine:
		// &geminitool.VertexAiSearch{
		// 	SearchEngineID: "projects/YOUR_PROJECT/locations/us/collections/default/engines/YOUR_ENGINE",
		// 	Filter:         "category: ANY(\"persona_A\")",
		// 	MaxResults:     genai.Ptr(int32(10)),
		// },
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "research_assistant",
		Model:       model,
		Description: "A research assistant that can retrieve and analyze content from URLs",
		Instruction: "You are a helpful research assistant. When given URLs, retrieve their content " +
			"and provide detailed, accurate summaries and analysis. Always cite the sources you use.",
		Tools: tools,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
