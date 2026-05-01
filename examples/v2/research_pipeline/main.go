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

// research_pipeline composes a multi-stage research workflow and shows
// how the PlanReAct planner integrates: its instruction goes into the
// LLM system prompt, and ProcessPlanningResponse marks pre-final-answer
// parts as thoughts so they don't leak into the user-visible output.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/planner"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

type Plan struct {
	Topic       string
	Subqueries  []string
	NeedsReview bool
}

type Findings struct {
	Bullets []string
}

func main() {
	plan := workflow.Func("plan",
		func(_ *workflow.NodeContext, topic string) (Plan, error) {
			return Plan{
				Topic: topic,
				Subqueries: []string{
					"what is " + topic + "?",
					"recent advances in " + topic,
					"common pitfalls of " + topic,
				},
				NeedsReview: true,
			}, nil
		})

	search := workflow.Func("search",
		func(_ *workflow.NodeContext, p Plan) (Findings, error) {
			out := Findings{}
			for _, q := range p.Subqueries {
				out.Bullets = append(out.Bullets, "[search] result for: "+q)
			}
			return out, nil
		})

	factCheck := workflow.Func("fact_check",
		func(_ *workflow.NodeContext, p Plan) (Findings, error) {
			return Findings{Bullets: []string{
				"[fact-check] " + p.Topic + ": no contradictions found in 3 cross-checks",
			}}, nil
		})

	join := workflow.Join("join")
	synthesize := workflow.Func("synthesize",
		func(_ *workflow.NodeContext, in any) (string, error) {
			m := in.(map[string]any)
			var b strings.Builder
			fmt.Fprintln(&b, "research summary:")
			for _, name := range []string{"search", "fact_check"} {
				if f, ok := m[name].(Findings); ok {
					for _, x := range f.Bullets {
						fmt.Fprintf(&b, "  - %s\n", x)
					}
				}
			}
			return strings.TrimRight(b.String(), "\n"), nil
		})

	// In a real LlmAgent this is where you'd attach the planner:
	//
	//   plannedAgent, _ := llmagent.New(llmagent.Config{
	//       Name: "synth", Model: gemini, Planner: &planner.PlanReAct{},
	//   })
	//
	// Here we just exercise the planner's prompt + response handling
	// directly so the example runs offline.
	demoPlanner()

	wf, err := workflow.New(workflow.Config{
		Name: "research",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, plan),
			workflow.Connect(plan, search),
			workflow.Connect(plan, factCheck),
			workflow.Connect(search, join),
			workflow.Connect(factCheck, join),
			workflow.Connect(join, synthesize),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "research_demo",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	// Plumb the topic into session state so plan can read it.
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "research"}}}
	_ = msg

	// Use the workflow with a literal topic by registering a seed node.
	// (See refund_approval for the pattern of plumbing typed input via
	// session state; here we just embed the topic directly in a seed.)
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (string, error) {
			return "agent-development-kit", nil
		})

	wf2, _ := workflow.New(workflow.Config{
		Name: "research2",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, plan),
			workflow.Connect(plan, search),
			workflow.Connect(plan, factCheck),
			workflow.Connect(search, join),
			workflow.Connect(factCheck, join),
			workflow.Connect(join, synthesize),
		},
	})
	wfAgent2, _ := wf2.AsAgent()
	r2, _ := runner.New(runner.Config{
		AppName:           "research_demo",
		Agent:             wfAgent2,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	for ev, err := range r2.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Author == "synthesize" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			fmt.Println("\n=== final answer ===")
			fmt.Println(ev.Actions.NodeInfo.Output)
		}
	}
	_ = r
}

func demoPlanner() {
	fmt.Println("=== PlanReAct planner contract ===")
	p := &planner.PlanReAct{}
	inst := p.BuildPlanningInstruction(nil, nil)
	fmt.Println("system instruction injected by planner (first 100 chars):")
	if len(inst) > 100 {
		fmt.Println("  " + inst[:100] + "...")
	} else {
		fmt.Println("  " + inst)
	}

	parts := []*genai.Part{
		{Text: "/*PLANNING*/ enumerate sub-questions"},
		{Text: "/*REASONING*/ each builds on the previous"},
		{Text: "/*FINAL_ANSWER*/ ADK is an SDK for agentic apps."},
	}
	out := p.ProcessPlanningResponse(nil, parts)
	fmt.Println("\nresponse-part annotations after planner post-processing:")
	for i, pp := range out {
		fmt.Printf("  part %d: thought=%v text=%q\n", i, pp.Thought, pp.Text)
	}
	fmt.Println()
}
