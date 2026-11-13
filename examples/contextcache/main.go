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

// Package main shows context caching serving an agent's fixed prompt prefix
// from a Gemini cache instead of resending it on every turn.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

var modelName = flag.String("model", "gemini-3.5-flash", "Gemini model to call")

var questions = []string{
	"What is the refund window for hardware orders?",
	"And for software licenses?",
	"Who approves a refund above the standard limit?",
}

func main() {
	flag.Parse()
	log.SetFlags(0)
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return errors.New("GOOGLE_API_KEY is not set")
	}
	llm, err := gemini.NewModel(ctx, *modelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return fmt.Errorf("creating the model: %w", err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Model:       llm,
		Description: "Answers questions from the support handbook.",
		Instruction: "You are a support agent. Answer in one sentence, using only the handbook below.\n\n" + handbook(),
	})
	if err != nil {
		return fmt.Errorf("creating the agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "contextcache",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		ContextCache: &model.ContextCacheConfig{
			CacheIntervals: 5,
			TTL:            5 * time.Minute,
		},
	})
	if err != nil {
		return fmt.Errorf("creating the runner: %w", err)
	}

	for i, q := range questions {
		log.Printf("> %s", q)
		msg := genai.NewContentFromText(q, genai.RoleUser)
		for ev, err := range r.Run(ctx, "user", "session", msg, agent.RunConfig{}) {
			if err != nil {
				return fmt.Errorf("turn %d: %s", i+1, explain(err))
			}
			if ev.Content == nil || ev.Author != a.Name() {
				continue
			}
			for _, p := range ev.Content.Parts {
				log.Print(p.Text)
			}
			log.Printf("[turn %d: %s]\n\n", i+1, describe(ev))
		}
	}
	return nil
}

// explain renders a Gemini API failure as the part that identifies the fix,
// rather than the service's whole response body.
func explain(err error) string {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return fmt.Sprintf("the Gemini API answered %d %s; check GOOGLE_API_KEY and that it can use model %q", apiErr.Code, apiErr.Status, *modelName)
	}
	return err.Error()
}

// describe reports how much of a reply's prompt the cache served.
func describe(ev *session.Event) string {
	var prompt, cached int32
	if u := ev.UsageMetadata; u != nil {
		prompt, cached = u.PromptTokenCount, u.CachedContentTokenCount
	}
	state := "caching off"
	switch m := ev.CacheMetadata; {
	case m.Active():
		state = fmt.Sprintf("cache %s, used by %d invocation(s)", m.CacheName, m.InvocationsUsed)
	case m != nil:
		state = "no cache yet, prefix fingerprinted"
	}
	return fmt.Sprintf("prompt %d tokens, %d from cache, %s", prompt, cached, state)
}

// handbook returns a long, fixed reference text: the part of the prompt that
// is the same on every turn and therefore worth caching. It is long enough to
// clear the minimum cache size of the models the sample is likely to run on.
func handbook() string {
	sections := []struct{ title, body string }{
		{"Refunds", "Hardware orders can be refunded within 30 days of delivery. Software licenses can be refunded within 14 days of purchase if the license key has not been activated. Refunds above 500 dollars need approval from the support lead on duty."},
		{"Shipping", "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and is free on orders above 200 dollars. We do not ship to post office boxes."},
		{"Warranty", "Hardware carries a 2 year limited warranty covering manufacturing defects. Accidental damage is not covered. Warranty claims need the original order number."},
		{"Accounts", "Customers can change their email address once every 30 days. Account deletion is permanent after a 14 day grace period. Two factor authentication can be reset only after identity verification."},
		{"Escalation", "Escalate to the support lead when a customer asks for a refund above the standard limit, reports a security issue, or has contacted support three times about the same order."},
	}
	var b strings.Builder
	for clause := 1; clause <= 24; clause++ {
		for _, s := range sections {
			fmt.Fprintf(&b, "## %s, clause %d\n%s\n\n", s.title, clause, s.body)
		}
	}
	return b.String()
}
