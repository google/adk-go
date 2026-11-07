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

// Package provides a quickstart ADK agent.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/exp/vertexairag"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
)

func main() {
	ctx := context.Background()

	ragCorpus := os.Getenv("RAG_CORPUS")
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if ragCorpus == "" {
		log.Fatal("RAG_CORPUS is not set. Please set it to a valid corpus - `projects/{project}/locations/{location}/ragCorpora/{ragCorporaId}`.")
	}
	if project == "" {
		log.Fatal("GOOGLE_CLOUD_PROJECT is not set. Please set it to a valid project.")
	}
	if location == "" {
		log.Fatal("GOOGLE_CLOUD_LOCATION is not set. Please set it to a valid location.")
	}

	model, err := gemini.NewModel(ctx, "gemini-3.5-flash", &genai.ClientConfig{
		Backend:  genai.BackendEnterprise,
		Project:  project,
		Location: location,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	vrag, err := vertexairag.New(vertexairag.Config{
		Description: "Searches rag",
		RAGResources: []*genai.VertexRAGStoreRAGResource{
			{RAGCorpus: ragCorpus},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create vertexairag tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "helpful_agent",
		Model:       model,
		Description: "Helpful Agent",
		Instruction: "use retrival tool for search first, and google_search if nothing is found",
		Tools:       []tool.Tool{vrag, geminitool.GoogleSearch{}},
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
