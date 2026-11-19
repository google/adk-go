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

// Package demonstrates an ADK agent backed by an OpenAI (or OpenAI-compatible)
// chat model instead of Gemini. The only ADK-specific difference from the
// Gemini quickstart is the model constructor; everything downstream (agents,
// tools, the runner, the launcher) is identical.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	openaimodel "google.golang.org/adk/v2/model/openaimodel"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// defaultModel is used when OPENAI_MODEL is unset. gpt-4o-mini is cheap and
// serves the Responses API that this integration targets.
const defaultModel = "gpt-4o-mini"

type weatherInput struct {
	City string `json:"city"`
}

type weatherOutput struct {
	Report string `json:"report"`
}

// getWeather is a stand-in for a real weather API so the sample runs offline
// once the model call returns. It shows a plain Go function surfaced to an
// OpenAI model as a tool via ADK's function-calling support.
func getWeather(_ agent.Context, in weatherInput) (weatherOutput, error) {
	return weatherOutput{
		Report: fmt.Sprintf("It is currently 22°C and sunny in %s.", in.City),
	}, nil
}

func main() {
	ctx := context.Background()

	// Point at api.openai.com with a key, or at any endpoint that implements the
	// OpenAI Responses API (recent Ollama, LM Studio, vLLM) via OPENAI_BASE_URL.
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if apiKey == "" && baseURL == "" {
		log.Fatal("set OPENAI_API_KEY (for OpenAI) or OPENAI_BASE_URL (for an OpenAI-compatible endpoint)")
	}

	modelName := os.Getenv("OPENAI_MODEL")
	if modelName == "" {
		modelName = defaultModel
	}

	model, err := openaimodel.NewModel(ctx, modelName, &openaimodel.ClientConfig{
		APIKey:  apiKey,
		BaseURL: baseURL,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	weatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Returns the current weather for a given city.",
	}, getWeather)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "openai_weather_agent",
		Model:       model,
		Description: "Answers weather questions using an OpenAI model.",
		Instruction: "You are a helpful assistant. When asked about the weather in a city, call the get_weather tool and report the result.",
		Tools: []tool.Tool{
			weatherTool,
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
