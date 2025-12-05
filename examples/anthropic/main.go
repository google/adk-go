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

// Package provides an example ADK agent using Anthropic Claude.
//
// Run with:
//
//	godotenv go run ./examples/anthropic
//
// Environment variables:
//   - ANTHROPIC_API_KEY: API key for direct Anthropic API access
//   - ANTHROPIC_USE_VERTEX: Set to "true" or "1" to use Vertex AI backend
//   - GOOGLE_CLOUD_PROJECT: GCP project ID (required for Vertex AI)
//   - GOOGLE_CLOUD_REGION: GCP region (required for Vertex AI)
package main

import (
	"context"
	"log"
	"os"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/anthropic"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// WeatherInput is the input schema for the weather tool.
type WeatherInput struct {
	City string `json:"city" jsonschema:"the city name to get weather for"`
}

// WeatherOutput is the output schema for the weather tool.
type WeatherOutput struct {
	Weather string `json:"weather"`
}

// getWeather is a simple tool function for the agent to call.
func getWeather(_ tool.Context, input WeatherInput) (WeatherOutput, error) {
	// In a real application, this would call a weather API
	weather := "The weather in " + input.City + " is sunny with a temperature of 22°C."
	return WeatherOutput{Weather: weather}, nil
}

func main() {
	ctx := context.Background()

	// Create the Anthropic model.
	// By default, uses the ANTHROPIC_API_KEY environment variable.
	// Set ANTHROPIC_USE_VERTEX=true to use Vertex AI backend instead.
	model, err := anthropic.NewModel(ctx, anthropicsdk.ModelClaudeHaiku4_5, nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create a function tool for the agent
	weatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
	}, getWeather)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "anthropic_weather_agent",
		Model:       model,
		Description: "Agent to answer questions about the weather in a city using Anthropic Claude.",
		Instruction: "You are a helpful weather assistant. When asked about weather, use the get_weather tool to fetch current conditions. Be concise in your responses.",
		Tools:       []tool.Tool{weatherTool},
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
