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

// Package provides a quickstart for Agent Engine deployment
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/agentengine"
	vertexaiMem "google.golang.org/adk/memory/vertexai"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session/vertexai"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	vertexaiutil "google.golang.org/adk/util/vertexai"
)

// Args defines the input structure for the memory search tool.
type Args struct {
	Query string `json:"query" jsonschema:"The query to search for in the memory."`
}

// Result defines the output structure for the memory search tool.
type Result struct {
	Results []string `json:"results"`
}

const (
	stateKeySessionLastUpdateTime = "sessionLastUpdateTime"
)

// memorySearchToolFunc is the implementation of the memory search tool.
// This function demonstrates accessing memory via tool.Context.
func memorySearchToolFunc(tctx tool.Context, args Args) (Result, error) {
	// The SearchMemory function is available on the context.
	searchResults, err := tctx.SearchMemory(tctx, args.Query)
	if err != nil {
		log.Printf("Error searching memory: %v", err)
		return Result{}, fmt.Errorf("failed memory search: %w", err)
	}

	var results []string
	for _, res := range searchResults.Memories {
		if res.Content != nil {
			for _, part := range res.Content.Parts {
				results = append(results, part.Text)
			}
		}
	}
	return Result{Results: results}, nil
}

func main() {
	ctx := context.Background()

	// those values are provided by AgentEngine, visible after the deployment to the container
	// for tesing, simply set those to your GCP project
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	location := os.Getenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
	agentEngineID := os.Getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID")

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  projectID,
		Location: location,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	type Input struct {
		Min int `json:"min"`
		Max int `json:"max"`
	}
	type Output struct {
		Result int `json:"result"`
	}
	handler := func(ctx tool.Context, input Input) (Output, error) {
		return Output{
			Result: input.Min + rand.IntN(input.Max-input.Min+1),
		}, nil
	}
	randomTool, err := functiontool.New(functiontool.Config{
		Name:        "random",
		Description: "Returns a random number between min and max",
	}, handler)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	// Define a tool that can search memory.
	memorySearchTool, err := functiontool.New(
		functiontool.Config{
			Name:        "search_past_conversations",
			Description: "Searches past conversations for relevant information.",
		},
		memorySearchToolFunc,
	)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "ae_agent",
		Model:       model,
		Description: "General helpful agent",
		Instruction: "You are a helpful agent, you should answer any questions you are given. Use 'random' tool to provide random numbers. Use search_past_conversations tool to get the facts about the user",
		Tools: []tool.Tool{
			randomTool,
			memorySearchTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	sessionService, err := vertexai.NewSessionService(
		ctx, vertexai.VertexAIServiceConfig{
			ProjectID:       projectID,
			Location:        location,
			ReasoningEngine: agentEngineID,
		})
	if err != nil {
		log.Fatalf("Failed to create session service: %v", err)
	}

	memService, err := vertexaiMem.NewService(ctx,
		&vertexaiMem.ServiceConfig{
			AgentEngineData: vertexaiutil.AgentEngineData{
				ProjectID:       projectID,
				Location:        location,
				ReasoningEngine: agentEngineID,
			},
			StateKeySessionLastUpdateTime: stateKeySessionLastUpdateTime,
		})
	if err != nil {
		log.Fatalf("Failed to create memory service: %v", err)
	}

	memPlugin, err := plugin.New(plugin.Config{
		Name: "Memory generator",
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			state := ic.Session().State()
			err := state.Set(stateKeySessionLastUpdateTime, ic.Session().LastUpdateTime())
			if err != nil {
				log.Printf("state.Set failed: %v\n", err)
				return nil, err
			}
			return nil, nil
		},
		AfterRunCallback: func(ic agent.InvocationContext) {
			m := ic.Memory()
			if m == nil {
				log.Printf("ic.Memory() is nil\n")
				return
			}
			err := m.AddSessionToMemory(ic, ic.Session())
			if err != nil {
				log.Printf("ic.Memory().AddSessionToMemory failed: %v\n", err)
			}
		},
	})
	if err != nil {
		log.Fatalf("Failed to create plugin: %v", err)
	}

	config := &launcher.Config{
		SessionService: sessionService,
		AgentLoader:    agent.NewSingleLoader(a),
		MemoryService:  memService,
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{
				memPlugin,
			},
		},
	}

	l := agentengine.NewLauncher(agentEngineID)
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
