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

// Package demonstrates an ADK agent backed by an OrcaRouter model instead of
// Gemini. OrcaRouter is an OpenAI-compatible AI gateway that exposes many
// models behind a single base URL (https://api.orcarouter.ai/v1). It talks to
// OrcaRouter's Responses API, so it plugs into the same model.LLM wiring as the
// OpenAI example (examples/openai); the only ADK-specific difference is the
// constructor's client config.
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

// defaultModel is used when ORCAROUTER_MODEL is unset. orcarouter/free routes
// to a free model on the gateway and needs no billing details.
const defaultModel = "orcarouter/free"

type weatherInput struct {
	City string `json:"city"`
}

type weatherOutput struct {
	Report string `json:"report"`
}

// getWeather is a stand-in for a real weather API so the sample runs offline
// once the model call returns. It shows a plain Go function surfaced to an
// OrcaRouter model as a tool via ADK's function-calling support.
func getWeather(_ agent.Context, in weatherInput) (weatherOutput, error) {
	return weatherOutput{
		Report: fmt.Sprintf("It is currently 22°C and sunny in %s.", in.City),
	}, nil
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("ORCAROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("set ORCAROUTER_API_KEY (get one at https://www.orcarouter.ai)")
	}

	modelName := os.Getenv("ORCAROUTER_MODEL")
	if modelName == "" {
		modelName = defaultModel
	}

	model, err := openaimodel.NewModel(ctx, modelName, &openaimodel.ClientConfig{
		APIKey:  apiKey,
		BaseURL: "https://api.orcarouter.ai/v1",
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
		Name:        "orcarouter_weather_agent",
		Model:       model,
		Description: "Answers weather questions using an OrcaRouter model.",
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
