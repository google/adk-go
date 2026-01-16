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

package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session/vertexai"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

const (
	modelName = "gemini-2.5-flash"
)

func main() {
	ctx := context.Background()

	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if projectID == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_PROJECT is not set")
	}
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_LOCATION is not set")
	}
	engineID := os.Getenv("VERTEX_ENGINE_ID")
	if engineID == "" {
		log.Fatalf("Env var VERTEX_ENGINE_ID is not set")
	}

	rootAgent, err := createAgent()
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	srvs, err := vertexai.NewSessionService(ctx, vertexai.VertexAIServiceConfig{
		Location:        location,
		ProjectID:       projectID,
		ReasoningEngine: engineID,
	})
	if err != nil {
		log.Fatalf("Failed to create session service: %v", err)
	}

	config := &launcher.Config{
		SessionService: srvs,
		AgentLoader:    agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	err = l.Execute(ctx, config, os.Args[1:])
	if err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func createAgent() (agent.Agent, error) {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{})
	if err != nil {
		return nil, err
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		return nil, err
	}

	return agent, nil
}
