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

// transferDocsModel stands in for a model that always answers with reply.
type transferDocsModel struct{ reply *genai.Content }

func (transferDocsModel) Name() string { return "transfer-docs" }

func (m transferDocsModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: m.reply}, nil)
	}
}

// describeTransferDocsEvent prints the author of ev and what it carries.
func describeTransferDocsEvent(ev *session.Event) {
	for _, p := range ev.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			fmt.Printf("%s: calls %s(%v)\n", ev.Author, p.FunctionCall.Name, p.FunctionCall.Args)
		case p.FunctionResponse != nil:
			fmt.Printf("%s: %s answered, transfer to %q\n", ev.Author, p.FunctionResponse.Name, ev.Actions.TransferToAgent)
		default:
			fmt.Printf("%s: %s\n", ev.Author, p.Text)
		}
	}
}

// An LLM agent with sub-agents can hand the conversation to one of them. The
// flow offers its model the transfer_to_agent tool, with the names of the
// agent's transfer targets as the allowed values, and runs the agent the model
// names. The runner sends the next user message to the agent that answered
// last, because it and the agent above it are LLM agents that allow transfer
// to their parent.
func ExampleConfig_subAgents() {
	billing, err := llmagent.New(llmagent.Config{
		Name:        "billing",
		Description: "Handles payments, refunds and invoices.",
		Model:       transferDocsModel{reply: genai.NewContentFromText("Your refund is on its way.", genai.RoleModel)},
	})
	if err != nil {
		log.Fatal(err)
	}
	support, err := llmagent.New(llmagent.Config{
		Name:        "support",
		Description: "Handles technical problems.",
		Model:       transferDocsModel{reply: genai.NewContentFromText("Try signing in again.", genai.RoleModel)},
	})
	if err != nil {
		log.Fatal(err)
	}
	transferToBilling := genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "billing"}, genai.RoleModel)
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Instruction: "Send payment questions to billing and technical problems to support.",
		Model:       transferDocsModel{reply: transferToBilling},
		SubAgents:   []agent.Agent{billing, support},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	if _, err := sessions.Create(ctx, &session.CreateRequest{AppName: "helpdesk", UserID: "user", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "helpdesk", Agent: coordinator, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	for _, msg := range []string{"My payment failed.", "When will I get my money back?"} {
		fmt.Println("user:", msg)
		for ev, err := range r.Run(ctx, "user", "s1", genai.NewContentFromText(msg, genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			describeTransferDocsEvent(ev)
		}
	}

	// Output:
	// user: My payment failed.
	// coordinator: calls transfer_to_agent(map[agent_name:billing])
	// coordinator: transfer_to_agent answered, transfer to "billing"
	// billing: Your refund is on its way.
	// user: When will I get my money back?
	// billing: Your refund is on its way.
}
