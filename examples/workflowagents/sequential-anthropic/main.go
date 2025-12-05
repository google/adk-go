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

// Package demonstrates a workflow agent that runs Anthropic-powered sub-agents sequentially.
//
// This example implements a simple "expand and summarise" pipeline where:
// 1. First agent expands on a topic with detailed information
// 2. Second agent summarises the expanded content
//
// This example supports two backends:
//
// 1. Direct Anthropic API (default):
//   - Set ANTHROPIC_API_KEY to your API key
//
// 2. Vertex AI:
//   - Set ANTHROPIC_USE_VERTEX=true
//   - Set GOOGLE_CLOUD_PROJECT to your Google Cloud project ID
//   - Set GOOGLE_CLOUD_LOCATION to the Vertex AI region (e.g., global, us-east5, europe-west1)
//   - Uses Application Default Credentials (ADC) for authentication:
//     gcloud auth application-default login
//
// Run with:
//
//	godotenv go run ./examples/workflowagents/sequential-anthropic
package main

import (
	"context"
	"log"
	"os"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/anthropic"
)

func main() {
	ctx := context.Background()

	// Create the Anthropic model
	model, err := anthropic.NewModel(ctx, anthropicsdk.Model("claude-opus-4-5@20251101"), nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Stage 1: Researcher agent that expands on a topic
	researcherAgent, err := llmagent.New(llmagent.Config{
		Name:        "researcher_agent",
		Model:       model,
		Description: "Researches and expands on topics with detailed information.",
		Instruction: `You are a knowledgeable researcher. When given a topic or question,
provide a detailed explanation covering key aspects, history, and interesting facts.
Write 3-4 paragraphs of informative content.`,
		OutputKey: "research_content",
	})
	if err != nil {
		log.Fatalf("Failed to create researcher agent: %v", err)
	}

	// Stage 2: Summariser agent that condenses the research
	summariserAgent, err := llmagent.New(llmagent.Config{
		Name:        "summariser_agent",
		Model:       model,
		Description: "Summarises content into concise key points.",
		Instruction: `You are an expert summariser. You will receive detailed research content.
Your task is to distill this into a clear, concise summary with:
- A one-sentence overview
- 3-5 bullet points of key takeaways
- A brief conclusion

**Content to summarise:**
{research_content}`,
		OutputKey: "summary",
	})
	if err != nil {
		log.Fatalf("Failed to create summariser agent: %v", err)
	}

	// Create a sequential agent that runs researcher then summariser
	sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "research_and_summarise_agent",
			Description: "Researches a topic in depth, then provides a concise summary",
			SubAgents:   []agent.Agent{researcherAgent, summariserAgent},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create sequential agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(sequentialAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
