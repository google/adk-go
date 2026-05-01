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

// QA evaluator: runs a Gemini agent against an eval set and scores
// each response with a Gemini-backed LLM-as-judge. Demonstrates
// eval.Runner + eval/llmjudge composed against a real agent.
//
// Unlike the other examples in v2, this one is a non-interactive
// command-line tool: it loads an eval set, runs every case, and prints
// a report.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/eval"
	"google.golang.org/adk/eval/llmjudge"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

const subjectInstruction = `You are an FAQ bot for an online electronics
store. Answer in one short sentence.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	subject, err := llmagent.New(llmagent.Config{
		Name:        "faq_bot",
		Description: "FAQ bot under evaluation.",
		Model:       model,
		Instruction: subjectInstruction,
	})
	if err != nil {
		log.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "qa_eval",
		Agent:             subject,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	judge, err := llmjudge.New(llmjudge.Config{LLM: model})
	if err != nil {
		log.Fatal(err)
	}

	set := &eval.Set{
		Name: "faq_qa",
		Cases: []eval.Case{
			{ID: "warranty", Input: "How long is the warranty on a laptop?", ExpectedOutput: "Most laptops carry a 2-year manufacturer warranty."},
			{ID: "returns", Input: "What is the return window?", ExpectedOutput: "We accept returns within 30 days of delivery."},
			{ID: "intl_ship", Input: "Do you ship to Japan?", ExpectedOutput: "Yes, we ship to most countries via DHL."},
			{ID: "pay", Input: "What payment methods are accepted?", ExpectedOutput: "Visa, Mastercard, AmEx, and PayPal."},
		},
	}

	evalRunner := &eval.Runner{
		Agent:     &agentAdapter{r: r},
		Scorer:    judge,
		Threshold: 0.7,
	}
	report, err := evalRunner.Run(ctx, set)
	if err != nil {
		log.Fatal(err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nPASS RATE: %.0f%%   (%d pass / %d fail / %d error)\n",
		report.PassRate()*100, report.PassCount, report.FailCount, report.ErrorCount)
}

// agentAdapter bridges the eval.AgentRunner interface to the runner.
// Each Score call drives one fresh session.
type agentAdapter struct {
	r *runner.Runner
}

func (a *agentAdapter) Run(ctx context.Context, userID, sessionID, input string) (string, error) {
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: input}}}
	var out string
	for ev, err := range a.r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		if ev.Author == "faq_bot" && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text != "" {
					out += p.Text
				}
			}
		}
	}
	return out, nil
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

var _ iter.Seq2[any, any] // keep iter import in case the adapter is extended
