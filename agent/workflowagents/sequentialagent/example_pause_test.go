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

package sequentialagent_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// oneReplyModel stands in for a model that always answers with reply.
type oneReplyModel struct{ reply *genai.Content }

func (oneReplyModel) Name() string { return "one-reply" }

func (m oneReplyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: m.reply}, nil)
	}
}

// In a resumable run, a sequential agent stops after the sub-agent that
// pauses the invocation on a long-running tool call: the sub-agents after
// it do not start.
func ExampleNew_resumablePause() {
	approval, err := functiontool.New(functiontool.Config{
		Name:          "request_approval",
		Description:   "Asks a manager to approve the order.",
		IsLongRunning: true,
	}, func(agent.Context, struct{}) (map[string]any, error) {
		return nil, nil // The answer comes later.
	})
	if err != nil {
		log.Fatal(err)
	}
	approver, err := llmagent.New(llmagent.Config{
		Name: "approver",
		Model: oneReplyModel{reply: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "approval-1", Name: "request_approval", Args: map[string]any{}}},
		}}},
		Tools: []tool.Tool{approval},
	})
	if err != nil {
		log.Fatal(err)
	}
	shipper, err := llmagent.New(llmagent.Config{
		Name:  "shipper",
		Model: oneReplyModel{reply: genai.NewContentFromText("Shipped.", genai.RoleModel)},
	})
	if err != nil {
		log.Fatal(err)
	}
	order, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "order", SubAgents: []agent.Agent{approver, shipper}},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, resumable := range []bool{false, true} {
		fmt.Println("resumable:", resumable)
		r, err := runner.New(runner.Config{
			AppName:           "orders",
			Agent:             order,
			SessionService:    session.InMemoryService(),
			AutoCreateSession: true,
			Resumable:         resumable,
		})
		if err != nil {
			log.Fatal(err)
		}
		msg := genai.NewContentFromText("Ship order 42", genai.RoleUser)
		for ev, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			if ev.Content == nil {
				continue // An event that records an agent's end.
			}
			if fc := ev.Content.Parts[0].FunctionCall; fc != nil {
				fmt.Printf("  %s: calls %s\n", ev.Author, fc.Name)
				continue
			}
			fmt.Printf("  %s: %s\n", ev.Author, ev.Content.Parts[0].Text)
		}
	}

	// Output:
	// resumable: false
	//   approver: calls request_approval
	//   shipper: Shipped.
	// resumable: true
	//   approver: calls request_approval
}
