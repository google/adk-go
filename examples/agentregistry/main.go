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

// Package main provides an example ADK agent that discovers and consumes
// components from the Google Cloud Agent Registry.
//
// It lists the agents registered in a project/location, then optionally builds
// a sub-agent and/or an MCP toolset from the registry and runs an LLM agent
// composed of them.
//
// Environment variables:
//   - GOOGLE_CLOUD_PROJECT   (required) the Google Cloud project ID
//   - GOOGLE_CLOUD_LOCATION  (required) the location, e.g. "us-central1"
//   - GOOGLE_API_KEY         (required) API key for the Gemini model
//   - AGENT_RESOURCE         (optional) full agent resource name to add as a sub-agent
//   - MCP_SERVER_RESOURCE    (optional) full MCP server resource name to add as a toolset
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agentregistry"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
)

func main() {
	// Create a context that cancels on interrupt (Ctrl+C).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if project == "" || location == "" {
		log.Fatal("set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION")
	}

	registry, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create Agent Registry client: %v", err)
	}

	// Discovery: list the agents registered in this project/location.
	fmt.Println("Registered agents:")
	for a, err := range registry.AllAgents(ctx) {
		if err != nil {
			log.Fatalf("Failed to list agents: %v", err)
		}
		fmt.Printf("  - %s (%s)\n", a.DisplayName, a.Name)
	}

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Optionally build a sub-agent from a registered A2A agent.
	var subAgents []agent.Agent
	if name := os.Getenv("AGENT_RESOURCE"); name != "" {
		sub, err := registry.RemoteAgent(ctx, name)
		if err != nil {
			log.Fatalf("Failed to build remote agent %q: %v", name, err)
		}
		subAgents = append(subAgents, sub)
	}

	// Optionally build a toolset from a registered MCP server.
	var toolsets []tool.Toolset
	if name := os.Getenv("MCP_SERVER_RESOURCE"); name != "" {
		ts, err := registry.McpToolset(ctx, name)
		if err != nil {
			log.Fatalf("Failed to build MCP toolset %q: %v", name, err)
		}
		toolsets = append(toolsets, ts)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "registry_agent",
		Model:       model,
		Description: "Agent composed from Google Cloud Agent Registry components.",
		Instruction: "You are a helpful assistant. Delegate to sub-agents and use tools when appropriate.",
		SubAgents:   subAgents,
		Toolsets:    toolsets,
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
