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

// Package main demonstrates the use of confirmation in FunctionTools.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type WriteFileArgs struct {
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// writeWithConfirmation simulates a file write operation that requires confirmation
func writeWithConfirmation(ctx tool.Context, args WriteFileArgs) (string, error) {
	// Request confirmation before writing the file
	err := ctx.RequestConfirmation("Writing file: "+args.Filename, map[string]any{
		"filename": args.Filename,
		"content":  args.Content,
	})
	if err != nil {
		// If confirmation was required, the flow will pause and wait for confirmation
		// In this example, we would normally resume after receiving confirmation
		return "", err
	}
	
	// After confirmation is granted, we would write the file
	// For this example, we'll just simulate it
	return fmt.Sprintf("File %s written successfully", args.Filename), nil
}

// staticConfirmationTool is a file operation that always requires confirmation
func staticConfirmationTool(ctx tool.Context, args WriteFileArgs) (string, error) {
	return fmt.Sprintf("Static confirmation tool executed for file: %s", args.Filename), nil
}

func main() {
	ctx := context.Background()
	
	// Create a function tool that requests confirmation dynamically
	dynamicTool, err := functiontool.New(functiontool.Config{
		Name:        "write_file_dynamic",
		Description: "Write content to a file, with dynamic confirmation",
	}, writeWithConfirmation)
	if err != nil {
		log.Fatalf("Failed to create dynamic confirmation tool: %v", err)
	}
	
	// Create a function tool that always requires confirmation via config
	staticTool, err := functiontool.New(functiontool.Config{
		Name:                  "write_file_static",
		Description:          "Write content to a file, with static confirmation requirement",
		RequireConfirmation:  true,
	}, staticConfirmationTool)
	if err != nil {
		log.Fatalf("Failed to create static confirmation tool: %v", err)
	}

	// Create a simple LLM agent with the tools
	llmAgent, err := agent.NewLLMAgent(ctx, agent.LLMAgentConfig{
		Name:  "file_operator",
		Model: &model.MockModel{}, // Use mock model for example
		Tools: []tool.Tool{dynamicTool, staticTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create a runner config
	config := &session.SessionConfig{
		Agents: []agent.Agent{llmAgent},
	}

	// Create a launcher with full capabilities
	l := full.NewLauncher()
	
	// Run the agent
	err = l.ParseAndRun(ctx, config, os.Args[1:], nil)
	if err != nil {
		log.Fatalf("Failed to run: %v", err)
	}
}