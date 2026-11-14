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

// Command declarative builds a multi-agent system from a YAML document,
// agent-system.yaml: a root LlmAgent, studio, that transfers to two workflow
// sub-agents, writer_critics and llm_router. Routing, loop control and data
// shaping are CEL expressions in the document. The Go code registers only the
// model the document refers to as default.
//
// The loader in system.go and cel.go is part of the example, not an ADK API.
//
// With GOOGLE_API_KEY set, default is Gemini. Otherwise set OPENAI_API_KEY or
// OPENAI_BASE_URL, and OPENAI_MODEL, to use any OpenAI-compatible endpoint,
// such as a local Ollama server. Set AGENT_SYSTEM to the path of another
// document to run that one instead of the embedded copy.
//
//	export GOOGLE_API_KEY=...
//	go run ./examples/declarative/ console
//
// or, with Ollama:
//
//	export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama
//	export OPENAI_MODEL=glm-5.3-flash:cloud
//	go run ./examples/declarative/ console
package main

import (
	"context"
	_ "embed"
	"errors"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/model/openaimodel"
)

//go:embed agent-system.yaml
var agentSystem []byte

func main() {
	log.SetFlags(0)
	ctx := context.Background()

	llm, err := newModel(ctx)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	doc := agentSystem
	if path := os.Getenv("AGENT_SYSTEM"); path != "" {
		if doc, err = os.ReadFile(path); err != nil {
			log.Fatalf("Failed to read agent system: %v", err)
		}
	}

	agents, err := loadSystem(doc, map[string]model.LLM{"default": llm})
	if err != nil {
		log.Fatalf("Failed to load agent system: %v", err)
	}

	// Serve the root agent first and every other declared agent after it, so
	// ADK Web can also open writer_critics or llm_router directly.
	loader, err := agent.NewMultiLoader(agents[0], agents[1:]...)
	if err != nil {
		log.Fatalf("Failed to create agent loader: %v", err)
	}

	cfg := &launcher.Config{AgentLoader: loader}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// newModel returns the model the document calls default: Gemini when
// GOOGLE_API_KEY is set, otherwise the OpenAI-compatible endpoint the OPENAI_*
// variables describe.
func newModel(ctx context.Context) (model.LLM, error) {
	if apiKey := os.Getenv("GOOGLE_API_KEY"); apiKey != "" {
		return gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{APIKey: apiKey})
	}
	apiKey, baseURL, name := os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_BASE_URL"), os.Getenv("OPENAI_MODEL")
	if apiKey == "" && baseURL == "" {
		return nil, errors.New("set GOOGLE_API_KEY, or OPENAI_API_KEY or OPENAI_BASE_URL for an OpenAI-compatible endpoint")
	}
	if name == "" {
		return nil, errors.New("OPENAI_MODEL is required with an OpenAI-compatible endpoint")
	}
	return openaimodel.NewModel(ctx, name, &openaimodel.ClientConfig{
		APIKey:  apiKey,
		BaseURL: baseURL,
		API:     openaimodel.APIChatCompletions,
	})
}
