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

// Example demonstrating OpenAI adapter with LM Studio, Ollama, or OpenAI API.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/openai"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// WeatherInput defines the input for the weather tool
type WeatherInput struct {
	Location string `json:"location"`
}

// WeatherOutput defines the output for the weather tool
type WeatherOutput struct {
	Location    string `json:"location"`
	Temperature string `json:"temperature"`
	Condition   string `json:"condition"`
	Humidity    string `json:"humidity"`
}

// Weather tool for demonstration
func getWeather(ctx tool.Context, input WeatherInput) (WeatherOutput, error) {
	// Mock weather data
	return WeatherOutput{
		Location:    input.Location,
		Temperature: "22°C",
		Condition:   "Sunny",
		Humidity:    "65%",
	}, nil
}

func main() {
	ctx := context.Background()

	// Configuration for different providers:
	// 1. LM Studio (default):
	//    BaseURL: "http://localhost:1234/v1"
	//    APIKey: "" (not needed)
	//
	// 2. Ollama:
	//    BaseURL: "http://localhost:11434/v1"
	//    APIKey: "" (not needed)
	//
	// 3. OpenAI:
	//    BaseURL: "https://api.openai.com/v1"
	//    APIKey: os.Getenv("OPENAI_API_KEY")
	//
	// 4. Remote LM Studio via ngrok:
	//    BaseURL: "https://your-ngrok-url.ngrok.io/v1"
	//    APIKey: "" (not needed)

	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:1234/v1" // Default to LM Studio
	}

	modelName := os.Getenv("OPENAI_MODEL")
	if modelName == "" {
		modelName = "google/gemma-3-12b" // Default model with tool support
	}

	apiKey := os.Getenv("OPENAI_API_KEY") // Optional, for OpenAI API

	log.Printf("Using OpenAI-compatible endpoint: %s", baseURL)
	log.Printf("Using model: %s", modelName)

	// Create OpenAI model adapter
	model, err := openai.NewModel(modelName, &openai.Config{
		BaseURL: baseURL,
		APIKey:  apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create weather tool with proper typing
	weatherTool, err := functiontool.New(
		functiontool.Config{
			Name:        "get_weather",
			Description: "Get the current weather for a specific location",
		},
		getWeather,
	)
	if err != nil {
		log.Fatalf("Failed to create weather tool: %v", err)
	}

	// Create LLM agent with tools
	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_assistant",
		Model:       model,
		Description: "An assistant that can answer questions about weather using local LLM",
		Instruction: `You are a helpful weather assistant. When asked about weather in a location,
use the get_weather tool to fetch the information. Provide friendly and informative responses.`,
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
