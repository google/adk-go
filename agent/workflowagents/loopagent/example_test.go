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

package loopagent_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/exitlooptool"
)

// loopDocsModel stands in for a model that answers its calls in turn with
// replies, one reply per call.
type loopDocsModel struct {
	replies []*genai.Content
	calls   *int
}

func (loopDocsModel) Name() string { return "loop-docs" }

func (m loopDocsModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		reply := m.replies[*m.calls]
		*m.calls++
		yield(&model.LLMResponse{Content: reply}, nil)
	}
}

// runLoopDocs runs root once in a new session and prints each event.
func runLoopDocs(root agent.Agent) {
	ctx := context.Background()
	sessions := session.InMemoryService()
	if _, err := sessions.Create(ctx, &session.CreateRequest{AppName: "docs", UserID: "user", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "docs", Agent: root, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	for ev, err := range r.Run(ctx, "user", "s1", genai.NewContentFromText("Start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Printf("%s: calls %s\n", ev.Author, p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Printf("%s: %s answered, escalate=%v\n", ev.Author, p.FunctionResponse.Name, ev.Actions.Escalate)
			default:
				fmt.Printf("%s: %s\n", ev.Author, p.Text)
			}
		}
	}
}

// A LoopAgent runs its sub-agents in order, again and again, until one of
// them escalates. The exit_loop tool from package exitlooptool escalates for
// the LLM agent whose model calls it, so the model decides when the loop
// ends. Here the writer's model calls exit_loop on its third call.
func ExampleNew() {
	exitLoop, err := exitlooptool.New()
	if err != nil {
		log.Fatal(err)
	}
	calls := 0
	writer, err := llmagent.New(llmagent.Config{
		Name:        "writer",
		Instruction: "Improve the draft. Call exit_loop when it needs no more work.",
		Model: loopDocsModel{calls: &calls, replies: []*genai.Content{
			genai.NewContentFromText("Draft 1.", genai.RoleModel),
			genai.NewContentFromText("Draft 2.", genai.RoleModel),
			genai.NewContentFromFunctionCall("exit_loop", map[string]any{}, genai.RoleModel),
		}},
		Tools: []tool.Tool{exitLoop},
	})
	if err != nil {
		log.Fatal(err)
	}
	refine, err := loopagent.New(loopagent.Config{
		MaxIterations: 10,
		AgentConfig:   agent.Config{Name: "refine", SubAgents: []agent.Agent{writer}},
	})
	if err != nil {
		log.Fatal(err)
	}

	runLoopDocs(refine)

	// Output:
	// writer: Draft 1.
	// writer: Draft 2.
	// writer: calls exit_loop
	// writer: exit_loop answered, escalate=true
}

// With MaxIterations set, a LoopAgent whose sub-agents never escalate stops
// after that many passes. MaxIterations 0 means no limit.
func ExampleNew_maxIterations() {
	calls := 0
	poller, err := llmagent.New(llmagent.Config{
		Name: "poller",
		Model: loopDocsModel{calls: &calls, replies: []*genai.Content{
			genai.NewContentFromText("Still pending.", genai.RoleModel),
			genai.NewContentFromText("Still pending.", genai.RoleModel),
			genai.NewContentFromText("Still pending.", genai.RoleModel),
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	poll, err := loopagent.New(loopagent.Config{
		MaxIterations: 3,
		AgentConfig:   agent.Config{Name: "poll", SubAgents: []agent.Agent{poller}},
	})
	if err != nil {
		log.Fatal(err)
	}

	runLoopDocs(poll)
	fmt.Println("model calls:", calls)

	// Output:
	// poller: Still pending.
	// poller: Still pending.
	// poller: Still pending.
	// model calls: 3
}
