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

package llmagent_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// scriptedResponses stands in for a real model: every call yields the same
// responses, in order, and the number of calls is counted.
type scriptedResponses struct {
	responses []*model.LLMResponse
	calls     int
}

func (*scriptedResponses) Name() string { return "scripted" }

func (m *scriptedResponses) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	return func(yield func(*model.LLMResponse, error) bool) {
		for _, r := range m.responses {
			if !yield(r, nil) {
				return
			}
		}
	}
}

// runAndPrint runs a in a new in-memory session with one user message and
// prints every event the run yields.
func runAndPrint(a agent.Agent, cfg agent.RunConfig) {
	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, cfg) {
		if err != nil {
			log.Fatal(err)
		}
		text := ""
		if ev.Content != nil && len(ev.Content.Parts) > 0 {
			text = ev.Content.Parts[0].Text
		}
		fmt.Printf("partial=%t error=%q text=%q\n", ev.Partial, ev.ErrorCode, text)
	}
}

// An AfterModelCallback that replaces a streamed response keeps the
// response's Partial flag unless it sets Partial itself. Partial: false alone
// does not count as set, so the replacement of a partial chunk stays partial.
// Adding PartialSet: true makes the explicit false count, and the replacement
// becomes a non-partial event.
func ExampleAfterModelCallback() {
	newAgent := func(explicit bool) agent.Agent {
		llm := &scriptedResponses{responses: []*model.LLMResponse{
			{Content: genai.NewContentFromText("secret ", genai.RoleModel), Partial: true},
			{Content: genai.NewContentFromText("secret plan", genai.RoleModel), FinishReason: genai.FinishReasonStop},
		}}
		redact := func(_ agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
			if err != nil || resp == nil || !resp.Partial {
				return nil, err
			}
			return &model.LLMResponse{
				Content:    genai.NewContentFromText("[redacted]", genai.RoleModel),
				Partial:    false,
				PartialSet: explicit,
			}, nil
		}
		a, err := llmagent.New(llmagent.Config{
			Name:                "assistant",
			Model:               llm,
			AfterModelCallbacks: []llmagent.AfterModelCallback{redact},
		})
		if err != nil {
			log.Fatal(err)
		}
		return a
	}
	sse := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}

	fmt.Println("Partial: false")
	runAndPrint(newAgent(false), sse)
	fmt.Println("Partial: false, PartialSet: true")
	runAndPrint(newAgent(true), sse)
	// Output:
	// Partial: false
	// partial=true error="" text="[redacted]"
	// partial=false error="" text="secret plan"
	// Partial: false, PartialSet: true
	// partial=false error="" text="[redacted]"
	// partial=false error="" text="secret plan"
}
