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

// content_pipeline is a multi-agent workflow graph: a researcher
// produces a brief, a drafter and a fact_checker run IN PARALLEL on
// that brief, a JoinNode aggregates their outputs, and an editor
// composes the final article. Every vertex is a real Gemini-backed
// LlmAgent wrapped via workflow.FromAgent.
//
//	START → researcher ─┬─ drafter ─┐
//	                    └─ fact_chk ─┴─ join → editor
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
	"google.golang.org/adk/workflow"
)

const researcherInstruction = `You are a research analyst.

The user supplies a topic. Produce a tight research brief:
  - 5 bullet points of relevant facts
  - 2 cited URLs (real ones if you can; placeholders otherwise)
  - 1 sentence describing what an article on this topic should argue

Output the brief as plain text in the format above. No preamble.`

const drafterInstruction = `You are a drafter.

You receive a research brief. Write a 200-word article draft that uses
every fact in the brief and follows the suggested argument. Do NOT
fact-check; trust the brief. Output only the draft.`

const factCheckerInstruction = `You are a fact-checker.

You receive the same research brief the drafter received. For each
bullet point in the brief, write a one-line verdict: "CONFIRMED" if the
claim is plausible without further sourcing, or "VERIFY: <what to
check>" if you'd flag it for an editor.

Output a numbered list, one verdict per bullet. No preamble.`

const editorInstruction = `You are an editor. You receive a JSON-like
object with two keys:
  - "drafter":     the article draft
  - "fact_checker": fact-check verdicts on the source brief

Produce the final article. Apply this editorial rule: if the verdicts
include any "VERIFY:" entries, prepend a one-paragraph note at the top
explaining which claims need verification. Otherwise publish the draft
verbatim. Do not change facts; only restructure or annotate.`

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

	// Wrap each Gemini agent as a workflow node.
	researcherN := workflow.FromAgent(researcher)
	drafterN := workflow.FromAgent(drafter)
	factCheckerN := workflow.FromAgent(factChecker)
	join := workflow.Join("join")
	editorN := workflow.FromAgent(editor)

	wf, err := workflow.New(workflow.Config{
		Name:        "content_pipeline",
		Description: "Researcher → parallel(drafter, fact-checker) → join → editor.",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, researcherN),
			workflow.Connect(researcherN, drafterN),
			workflow.Connect(researcherN, factCheckerN),
			workflow.Connect(drafterN, join),
			workflow.Connect(factCheckerN, join),
			workflow.Connect(join, editorN),
		},
	})
	if err != nil {
		log.Fatalf("workflow.New: %v", err)
	}

	wfAgent, err := wf.AsAgent()
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(wfAgent)}
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

func mustAgent(a agent.Agent, err error) agent.Agent {
	if err != nil {
		log.Fatal(err)
	}
	return a
}
