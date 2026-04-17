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

// Package provides a quickstart ADK agent.
package main

import (
	"context"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")
	apiKey = strings.TrimSpace(apiKey)

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	type ParsedData struct {
		Data       string         `json:"data"`
		Attributes map[string]any `json:"attributes"`
	}

	handler := func(ctx tool.Context, input ParsedData) (ParsedData, error) {
		return input, nil
	}

	parseEventTool, err := functiontool.New(functiontool.Config{
		Name:        "parse_event",
		Description: "parses raw event data",
	}, handler)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "event_processor",
		Model:       model,
		Description: "Agent to process the events from Pub/Sub and Eventarc triggers.",
		Instruction: `
		You are an event-processing agent that handles incoming
		events from Pub/Sub and Eventarc triggers.

		When you receive an event:
		1. Use the "parse_event" tool to extract the event data and attributes.
		2. Analyze the event contents and determine what action to take.
		3. Summarize what you found and what action you would recommend.

		Be concise and structured in your responses.`,
		Tools: []tool.Tool{
			parseEventTool,
		},
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
