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

// Research planner: a Gemini agent prompted with the PlanReAct
// template. The model emits /*PLANNING*/, /*REASONING*/, /*ACTION*/,
// /*FINAL_ANSWER*/ sections; the planner post-processes the response
// to mark pre-final-answer parts as Thought=true so downstream UIs
// can hide them by default.
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
	"google.golang.org/adk/planner"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const baseInstruction = `You are a research analyst. When given a topic,
plan your sub-questions, reason step-by-step using the search tool, and
then write a one-paragraph final answer.

Use the search tool whenever you need a fact you don't know.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	planReact := &planner.PlanReAct{}

	search, err := buildSearchTool()
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "research_planner",
		Description: "A research analyst that plans, reasons, and answers using the PlanReAct template.",
		Model:       model,
		// Compose the user instruction with the planner's instruction.
		// The plumbing for native planner integration into LlmAgent
		// arrives in a follow-up; for now we stitch the templates
		// together manually so the example runs as-is.
		Instruction: baseInstruction + "\n\n" + planReact.BuildPlanningInstruction(nil, nil),
		Tools:       []tool.Tool{search},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func buildSearchTool() (tool.Tool, error) {
	type args struct {
		Query string `json:"query"`
	}
	type hit struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	}
	type result struct {
		Hits []hit `json:"hits"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "search",
			Description: "Search the web; returns up to 5 result titles + URLs.",
		},
		func(_ tool.Context, a args) (result, error) {
			// Stubbed catalog. Replace with a real search backend.
			return result{Hits: []hit{
				{Title: "Overview of " + a.Query, URL: "https://example.com/" + a.Query},
				{Title: "Recent advances in " + a.Query, URL: "https://example.com/recent/" + a.Query},
			}}, nil
		},
	)
}
