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

// Package main provides an Anthropic-backed ADK agent example.
package main

import (
	"context"
	"log"
	"os"

	anthropicapi "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	adkanthropic "google.golang.org/adk/model/anthropic"
)

func main() {
	ctx := context.Background()

	model, err := adkanthropic.NewModel(ctx, anthropicapi.Model("claude-sonnet-4-20250514"), &adkanthropic.Config{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "anthropic_weather_agent",
		Model:       model,
		Description: "A concise assistant powered by Anthropic Claude.",
		Instruction: "You are a helpful assistant. Respond briefly and accurately.",
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
