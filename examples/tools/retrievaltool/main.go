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

package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/server/restapi/services"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/retrievaltool"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	modelName := "gemini-2.0-flash-001"
	if v := os.Getenv("MODEL_NAME"); v != "" {
		modelName = v
	}
	location := "us-central1"
	if v := os.Getenv("LOCATION"); v != "" {
		location = v
	}
	model, err := gemini.NewModel(ctx, modelName, 
		&genai.ClientConfig{
			Backend:  genai.BackendVertexAI,
			Project:  os.Getenv("PROJECT_ID"),
			Location: location,
		})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Need to create RAG corpus by https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart#run-rag
	ragCorpus := os.Getenv("RAG_CORPUS")
	if ragCorpus == "" {
		log.Fatalf("RAG_CORPUS environment variable is required")
	}

	similarityTopKStr := "10"
	if v := os.Getenv("SIMILARITY_TOP_K"); v != "" {
		similarityTopKStr = v
	}
	similarityTopKVal, err := strconv.ParseInt(similarityTopKStr, 10, 32)
	if err != nil {
		log.Fatalf("failed to parse SIMILARITY_TOP_K: %v", err)
	}
	similarityTopK := int32(similarityTopKVal)

	vectorDistanceThresholdStr := "0.6"
	if v := os.Getenv("VECTOR_DISTANCE_THRESHOLD"); v != "" {
		vectorDistanceThresholdStr = v
	}
	vectorDistanceThreshold, err := strconv.ParseFloat(vectorDistanceThresholdStr, 64)
	if err != nil {
		log.Fatalf("failed to parse VECTOR_DISTANCE_THRESHOLD: %v", err)
	}

	ragStore := &genai.VertexRAGStore{
		RAGCorpora:              []string{ragCorpus},
		SimilarityTopK:          &similarityTopK,
		VectorDistanceThreshold: &vectorDistanceThreshold,
	}

	askVertexRetrieval, err := retrievaltool.NewVertexAIRAG(
		"retrieve_rag_documentation",
		"Use this tool to retrieve documentation and reference materials for the question from the RAG corpus",
		ragStore,
	)
	if err != nil {
		log.Fatalf("Failed to create retrievaltool: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "ask_rag_agent",
		Model:       model,
		Description: "Agent that answers questions using RAG-based document retrieval",
		Instruction: `You are a helpful assistant that can retrieve documentation and reference materials to answer questions.

When answering questions:
1. Use the retrieve_rag_documentation tool to search for relevant information from the knowledge base
2. Base your answers on the retrieved documentation
3. If the retrieved information is insufficient, clearly state what additional information might be needed
4. Always cite or reference the source of your information when possible`,
		Tools: []tool.Tool{
			askVertexRetrieval,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &adk.Config{
		AgentLoader: services.NewSingleAgentLoader(rootAgent),
	}

	l := full.NewLauncher()
	err = l.Execute(ctx, config, os.Args[1:])
	if err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
