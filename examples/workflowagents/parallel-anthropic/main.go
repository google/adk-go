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

// Package demonstrates a workflow agent that runs Anthropic-powered sub-agents in parallel.
//
// Run with:
//
//	godotenv go run ./examples/workflowagents/parallel-anthropic
//
// Environment variables:
//   - ANTHROPIC_API_KEY: API key for direct Anthropic API access
package main

import (
	"context"
	"log"
	"os"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/anthropic"
)

func main() {
	ctx := context.Background()

	// Create the Anthropic model
	model, err := anthropic.NewModel(ctx, anthropicsdk.ModelClaudeSonnet4_20250514, nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create two LLM agents that will run in parallel
	// Each agent has a different perspective on the same topic

	optimistAgent, err := llmagent.New(llmagent.Config{
		Name:        "optimist_agent",
		Model:       model,
		Description: "An optimistic analyst that focuses on opportunities.",
		Instruction: `You are an optimistic business analyst. When given a topic or question,
provide a brief (2-3 sentences) analysis focusing on the positive aspects, opportunities,
and potential benefits. Be concise but insightful.`,
		OutputKey: "optimist_view",
	})
	if err != nil {
		log.Fatalf("Failed to create optimist agent: %v", err)
	}

	pessimistAgent, err := llmagent.New(llmagent.Config{
		Name:        "pessimist_agent",
		Model:       model,
		Description: "A cautious analyst that focuses on risks.",
		Instruction: `You are a cautious risk analyst. When given a topic or question,
provide a brief (2-3 sentences) analysis focusing on potential risks, challenges,
and things to watch out for. Be concise but thorough.`,
		OutputKey: "pessimist_view",
	})
	if err != nil {
		log.Fatalf("Failed to create pessimist agent: %v", err)
	}

	// Create a parallel agent that runs both analysts simultaneously
	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "dual_perspective_agent",
			Description: "Analyses topics from both optimistic and pessimistic perspectives in parallel",
			SubAgents:   []agent.Agent{optimistAgent, pessimistAgent},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create parallel agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(parallelAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
