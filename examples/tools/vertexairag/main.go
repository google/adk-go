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
//	GOOGLE_MODEL          optional Gemini model (default gemini-2.5-flash)
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/vertexairag"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
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
		modelName = "gemini-2.5-flash"
	}

	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: location,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	ragTool, err := vertexairag.New(vertexairag.Config{
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

	appName, userID := "rag_app", "test_user"
	sessionService := session.InMemoryService()
	resp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	currentSession := resp.Session

	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          llmAgent,
		SessionService: sessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	fmt.Println("Ask a question about the corpus. Ctrl-C to quit.")
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nUser -> ")

		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		fmt.Print("\nAgent -> ")
		streamingMode := agent.StreamingModeSSE
		for event, err := range r.Run(ctx, userID, currentSession.ID(), userMsg, agent.RunConfig{
			StreamingMode: streamingMode,
		}) {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
				continue
			}
			if event.LLMResponse.Content == nil {
				continue
			}
			for _, p := range event.LLMResponse.Content.Parts {
				// In SSE streaming mode, print only partial responses to avoid
				// repeating the final aggregated message.
				if streamingMode != agent.StreamingModeSSE || event.LLMResponse.Partial {
					fmt.Print(p.Text)
				}
			}
		}
	}
}
