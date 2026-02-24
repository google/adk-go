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

// Package provides a quickstart ADK agent with telemetry.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/model/gemini"
	"github.com/sjzsdu/adk-go/telemetry"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/geminitool"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	cfg := llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "Your SOLE purpose is to answer questions about the current time and weather in a specific city. You MUST refuse to answer any questions unrelated to time or weather.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	}

	a, err := llmagent.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	r, err := resource.New(ctx, resource.WithAttributes(
		semconv.ServiceNameKey.String("weather-time-agent"),
		semconv.ServiceVersionKey.String("1.0.0"),
	))
	if err != nil {
		return fmt.Errorf("failed to create resource: %w", err)
	}
	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
		TelemetryOptions: []telemetry.Option{
			telemetry.WithResource(r),
			// Other telemetry options can be added here.
		},
	}

	// Launcher automatically starts the telemetry.
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		return fmt.Errorf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}
