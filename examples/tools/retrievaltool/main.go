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

	model, err := gemini.NewModel(ctx, "gemini-2.0-flash-001", &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  os.Getenv("PROJECT_ID"),
		Location: "us-central1",
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Need to create RAG corpus by https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart#run-rag
	ragCorpus := os.Getenv("RAG_CORPUS")
	if ragCorpus == "" {
		log.Fatalf("RAG_CORPUS environment variable is required")
	}

	similarityTopK := int32(10)
	vectorDistanceThreshold := 0.6

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
