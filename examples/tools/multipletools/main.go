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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/geminitool"
)

// Package main demonstrates a workaround for using multiple tool types (e.g.,
// Google Search and custom functions) in a single agent. This is necessary
// due to limitations in the genai API. The approach is to wrap agents with
// different tool types into sub-agents, which are then managed by a root agent.
func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	searchAgent, err := llmagent.New(llmagent.Config{
		Name:        "search_agent",
		Model:       model,
		Description: "Does google search.",
		Instruction: "You're a specialist in Google Search.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	type Input struct {
		LineCount int `json:"lineCount"`
	}
	type Output struct {
		Poem string `json:"poem"`
	}
	handler := func(ctx tool.Context, input Input) (Output, error) {
		return Output{
			Poem: strings.Repeat("A line of a poem,", input.LineCount) + "\n",
		}, nil
	}
	poemTool, err := functiontool.New(functiontool.Config{
		Name:        "poem",
		Description: "Returns poem",
	}, handler)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}
	poemAgent, err := llmagent.New(llmagent.Config{
		Name:        "poem_agent",
		Model:       model,
		Description: "returns poem",
		Instruction: "You return poems.",
		Tools: []tool.Tool{
			poemTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "root_agent",
		Model:       model,
		Description: "You can do a google search and generate poems.",
		Instruction: "Answer questions about weather based on google search unless asked for a poem," +
			" for a poem generate it with a tool.",
		Tools: []tool.Tool{
			agenttool.New(searchAgent, nil), agenttool.New(poemAgent, nil),
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
