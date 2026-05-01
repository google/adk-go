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

// Customer-support agent whose long sessions auto-summarize via
// EventsCompactionConfig. The summarizer is itself a Gemini-backed
// LlmEventSummarizer.
//
// Note: the launcher's Config does not yet surface app.App, so this
// example wires a runner directly and demonstrates the compaction path
// via an interactive console loop. To use the standard launcher,
// supply the same plugins through cmd.Config.PluginConfig instead.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	adkapp "google.golang.org/adk/app"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

const supportInstruction = `You are a customer-support agent for an
e-commerce store. Be concise and friendly. If you do not have the
order details the user is asking about, ask them for the order id.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// The summarizer model can be the same model or a cheaper one.
	summarizerModel, _ := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})

	supportAgent, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Description: "A customer support agent.",
		Model:       model,
		Instruction: supportInstruction,
	})
	if err != nil {
		log.Fatal(err)
	}

	app, err := adkapp.New(adkapp.App{
		Name:      "support_app",
		RootAgent: supportAgent,
		EventsCompactionConfig: &adkapp.EventsCompactionConfig{
			// Real Gemini-backed summarizer. Compacts older turns into a
			// single synthetic event; the contents-builder folds it in
			// place of the subsumed raw events on the next LLM call.
			Summarizer:         adkapp.NewLlmEventSummarizer(summarizerModel),
			CompactionInterval: 3, // compact every 3 user invocations
			OverlapSize:        1, // keep 1 prior invocation for continuity
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	sessSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		App:               app,
		SessionService:    sessSvc,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== support session (compaction every 3 turns) ===")
	fmt.Println("type your message, blank line to quit")
	fmt.Println()
	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("you> ")
		if !in.Scan() {
			return
		}
		text := in.Text()
		if text == "" {
			return
		}
		msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
		for ev, err := range r.Run(ctx, "alice", "case-77", msg, agent.RunConfig{}) {
			if err != nil {
				fmt.Println("error:", err)
				return
			}
			if ev.Author == "support_agent" && ev.Content != nil {
				for _, p := range ev.Content.Parts {
					if p.Text != "" {
						fmt.Print(p.Text)
					}
				}
				fmt.Println()
			}
			if ev.Actions.Compaction != nil {
				fmt.Printf("[compacted older turns: %s -> %s]\n",
					ev.Actions.Compaction.StartTimestamp.Format("15:04:05"),
					ev.Actions.Compaction.EndTimestamp.Format("15:04:05"))
			}
		}
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}
