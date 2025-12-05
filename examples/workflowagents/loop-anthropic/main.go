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

// Package demonstrates a workflow agent that runs an Anthropic-powered agent in a loop.
//
// Run with:
//
//	godotenv go run ./examples/workflowagents/loop-anthropic
//
// Environment variables:
//   - ANTHROPIC_API_KEY: API key for direct Anthropic API access
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/anthropic"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// StoryInput is the input for the continue_story tool.
type StoryInput struct {
	Direction string `json:"direction" jsonschema:"the direction to take the story (e.g., 'add conflict', 'introduce character', 'resolve plot')"`
}

// StoryOutput is the output of the continue_story tool.
type StoryOutput struct {
	Instruction string `json:"instruction"`
}

// continueStory provides story direction to the agent.
func continueStory(_ tool.Context, input StoryInput) (StoryOutput, error) {
	return StoryOutput{
		Instruction: "Continue the story by: " + input.Direction,
	}, nil
}

func main() {
	ctx := context.Background()

	// Create the Anthropic model
	model, err := anthropic.NewModel(ctx, anthropicsdk.ModelClaudeOpus4_5, nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create a tool for story continuation
	storyTool, err := functiontool.New(functiontool.Config{
		Name:        "continue_story",
		Description: "Get instructions for continuing the story in a specific direction",
	}, continueStory)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	// Create an LLM agent that tells stories
	storyteller, err := llmagent.New(llmagent.Config{
		Name:        "storyteller",
		Model:       model,
		Description: "A creative storyteller that writes short story segments.",
		Instruction: `You are a creative storyteller. Each time you are called, continue the ongoing story with 2-3 sentences.
Build upon what came before, adding new elements while maintaining narrative coherence.
Keep each segment brief but engaging. Do not repeat previous content.
If this is the first segment, begin a new story about an unexpected adventure.`,
		Tools: []tool.Tool{storyTool},
	})
	if err != nil {
		log.Fatalf("Failed to create storyteller agent: %v", err)
	}

	// Create a loop agent that runs the storyteller multiple times
	loopAgent, err := loopagent.New(loopagent.Config{
		MaxIterations: 3,
		AgentConfig: agent.Config{
			Name:        "story_loop",
			Description: "A loop agent that builds a story over multiple iterations using Anthropic Claude",
			SubAgents:   []agent.Agent{storyteller},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create loop agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(loopAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
