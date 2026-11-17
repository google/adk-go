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

// Package main provides an example ADK agent that grounds its answers on a
// Vertex AI RAG corpus using the exp/vertexairag tool.
//
// Vertex RAG grounding runs on the Vertex AI backend, so the model, project,
// location and corpus are configured through the environment:
//
//	GOOGLE_CLOUD_PROJECT  the GCP project
//	GOOGLE_CLOUD_LOCATION the region, e.g. us-central1
//	RAG_CORPUS            the fully-qualified corpus resource name,
//	                      projects/.../locations/.../ragCorpora/...
//	GOOGLE_MODEL          optional Gemini model (default gemini-3.5-flash)
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
	"google.golang.org/adk/v2/exp/vertexairagtool"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
)

func main() {
	ctx := context.Background()

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	corpus := os.Getenv("RAG_CORPUS")
	if project == "" || location == "" || corpus == "" {
		log.Fatal("set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION and RAG_CORPUS")
	}

	modelName := os.Getenv("GOOGLE_MODEL")
	if modelName == "" {
		modelName = "gemini-3.5-flash"
	}

	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: location,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	ragTool, err := vertexairagtool.New(vertexairagtool.Config{
		Description: "Retrieves relevant passages from the documentation corpus.",
		RAGResources: []*genai.VertexRAGStoreRAGResource{
			{RAGCorpus: corpus},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create RAG tool: %v", err)
	}

	llmAgent, err := llmagent.New(llmagent.Config{
		Name:        "rag_assistant",
		Model:       m,
		Description: "Agent that answers questions grounded on a Vertex AI RAG corpus.",
		Instruction: "You are a helpful assistant. Use the retrieval tool to ground " +
			"your answers on the documentation corpus, and cite what you find.",
		Tools: []tool.Tool{ragTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(llmAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
