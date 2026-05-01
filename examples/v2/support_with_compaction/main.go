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

// support_with_compaction simulates a long customer-support session
// whose older turns auto-summarize via EventsCompactionConfig.
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	adkapp "google.golang.org/adk/app"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/skill" // unused; package import keeps imports tidy when extending
)

// counterSummarizer is a deterministic stand-in for an LLM summarizer.
// Each invocation produces a text summary numbered by call count so we
// can read what compaction is doing without an LLM in the loop.
type counterSummarizer struct {
	calls int
}

func (s *counterSummarizer) MaybeSummarize(_ context.Context, events []*session.Event) (*session.Event, error) {
	if len(events) == 0 {
		return nil, nil
	}
	s.calls++
	first := events[0].Timestamp
	last := events[len(events)-1].Timestamp
	body := fmt.Sprintf("Compacted %d events from %s to %s (call #%d).",
		len(events), first.Format("15:04:05"), last.Format("15:04:05"), s.calls)
	out := session.NewEvent(fmt.Sprintf("compact-%d", s.calls))
	out.Author = "user"
	out.Timestamp = time.Now()
	out.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   first,
		EndTimestamp:     last,
		CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: body}}},
	}
	return out, nil
}

// supportAgent is the simplest possible agent that produces one canned
// response per invocation. Replace with a real LlmAgent in production.
func supportAgent() agent.Agent {
	a, err := agent.New(agent.Config{
		Name:        "support_agent",
		Description: "Pretends to answer customer questions.",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic.InvocationID())
				ev.Author = "support_agent"
				ev.LLMResponse.Content = &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{Text: "ack: " + textOf(ic.UserContent())}},
				}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	return a
}

func main() {
	root := supportAgent()
	app, err := adkapp.New(adkapp.App{
		Name:      "support_demo",
		RootAgent: root,
		EventsCompactionConfig: &adkapp.EventsCompactionConfig{
			Summarizer:         &counterSummarizer{},
			CompactionInterval: 2, // compact every 2 new user invocations
			OverlapSize:        1, // overlap 1 invocation for context continuity
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

	turns := []string{
		"My order #4521 hasn't shipped yet.",
		"Can you give me an estimated arrival date?",
		"I noticed a $25 charge I don't recognize.",
		"It says PENDING from yesterday.",
		"Also can I change my shipping address?",
		"Sure, here's the new address: 100 Market St, SF.",
	}

	ctx := context.Background()
	for i, q := range turns {
		fmt.Printf("=== turn %d (user) ===\n%s\n", i+1, q)
		msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: q}}}
		for ev, err := range r.Run(ctx, "alice", "case-77", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			if ev.Author == "support_agent" {
				fmt.Printf("→ %s\n", textOf(ev.LLMResponse.Content))
			}
		}
		// Compaction runs after the agent loop in runner.Run and appends
		// to the session. Read the session to inspect what landed.
		printCompactions(ctx, sessSvc)
		fmt.Println()
	}
}

func printCompactions(ctx context.Context, svc session.Service) {
	resp, err := svc.Get(ctx, &session.GetRequest{
		AppName: "support_demo", UserID: "alice", SessionID: "case-77",
	})
	if err != nil {
		return
	}
	for ev := range resp.Session.Events().All() {
		if ev.Actions.Compaction == nil {
			continue
		}
		// Only print compactions whose Compaction summary we haven't shown yet.
		if marked[ev.ID] {
			continue
		}
		marked[ev.ID] = true
		fmt.Printf("[compaction event %s] %s\n",
			ev.ID[:8], textOf(ev.Actions.Compaction.CompactedContent))
	}
}

var marked = map[string]bool{}

func textOf(c *genai.Content) string {
	if c == nil {
		return ""
	}
	out := ""
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			out += p.Text
		}
	}
	return out
}

var _ = skill.Skill{} // keep import silent until we extend example
