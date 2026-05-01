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

// dynamic_flow shows an orchestrator agent that, given a user request,
// decides at runtime how to compose a catalog of pre-built agents into a
// flow (sequential, parallel, or nested) and runs it via the run_flow
// tool from package flowtool.
//
// Catalog (registered up front, identities stable):
//
//	researcher    — produces a research brief on the topic
//	drafter       — writes a 200-word article from the brief
//	fact_checker  — flags claims needing verification
//	editor        — polishes a draft into the final article
//
// Example prompt:
//
//	"Write me a short article about hummingbird migration. Run the
//	 researcher first, then in parallel run the drafter and the
//	 fact_checker on the brief, then have the editor merge their
//	 outputs."
//
// The orchestrator emits one run_flow tool call carrying the recursive
// spec; flowtool materialises the flow, runs it, and returns a path-keyed
// outputs map plus the final output.
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
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/flowtool"
)

const (
	researcherInstruction = `You are a research analyst. Given a topic, produce a tight research brief: 5 bullet points and 1 sentence describing what an article should argue. Plain text only.`
	drafterInstruction    = `You are a drafter. Given a research brief, write a 200-word article using every fact in it. Output only the draft.`
	factCheckerInstruction = `You are a fact-checker. Given a research brief, write a one-line verdict per bullet: "CONFIRMED" or "VERIFY: <what to check>". Numbered list, no preamble.`
	editorInstruction      = `You are an editor. Given a draft and fact-check verdicts, produce the final article. If any verdicts say VERIFY, prepend a one-paragraph note listing what needs verification.`

	orchestratorInstruction = `You orchestrate a small team of writing agents.

The catalog of agents available via the run_flow tool:
  - researcher
  - drafter
  - fact_checker
  - editor

Decide a flow shape that fits the user request. Typical pattern for an article:
seq[researcher → parallel(drafter, fact_checker) → editor], where the editor's
input pulls drafter and fact_checker outputs via {{nodes.<path>.output}}
templates.

Always produce ONE call to run_flow with the full recursive spec, then summarise
the final article for the user from the tool's final_output field.`
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	researcher := mustAgent(llmagent.New(llmagent.Config{
		Name:        "researcher",
		Description: "Produces a research brief on the supplied topic.",
		Model:       model,
		Instruction: researcherInstruction,
	}))
	drafter := mustAgent(llmagent.New(llmagent.Config{
		Name:        "drafter",
		Description: "Writes a 200-word article from the brief.",
		Model:       model,
		Instruction: drafterInstruction,
	}))
	factChecker := mustAgent(llmagent.New(llmagent.Config{
		Name:        "fact_checker",
		Description: "Flags claims needing verification.",
		Model:       model,
		Instruction: factCheckerInstruction,
	}))
	editor := mustAgent(llmagent.New(llmagent.Config{
		Name:        "editor",
		Description: "Composes the final article from drafter + fact-checker outputs.",
		Model:       model,
		Instruction: editorInstruction,
	}))

	catalog := map[string]agent.Agent{
		"researcher":   researcher,
		"drafter":      drafter,
		"fact_checker": factChecker,
		"editor":       editor,
	}

	orchestrator, err := llmagent.New(llmagent.Config{
		Name:        "orchestrator",
		Description: "Decides a flow shape and dispatches the writing team.",
		Model:       model,
		Instruction: orchestratorInstruction,
		Tools:       []tool.Tool{flowtool.New(catalog)},
	})
	if err != nil {
		log.Fatalf("orchestrator: %v", err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(orchestrator)}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func mustAgent(a agent.Agent, err error) agent.Agent {
	if err != nil {
		log.Fatal(err)
	}
	return a
}
