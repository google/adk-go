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

package agent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// transferModel answers each request with the next of its scripted turns.
type transferModel struct{ turns []*genai.Content }

func (*transferModel) Name() string { return "scripted" }

func (m *transferModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(m.turns) == 0 {
			yield(nil, errors.New("script exhausted"))
			return
		}
		turn := m.turns[0]
		m.turns = m.turns[1:]
		yield(&model.LLMResponse{Content: turn, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// runTransfer runs a coordinator with the sub-agents billing and support,
// where billing may not transfer to its peers. The coordinator transfers to
// billing, whose model answers with billingTurns.
func runTransfer(billingTurns ...*genai.Content) {
	billing, err := llmagent.New(llmagent.Config{
		Name:                    "billing",
		Description:             "Handles payments.",
		Model:                   &transferModel{turns: billingTurns},
		DisallowTransferToPeers: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	support, err := llmagent.New(llmagent.Config{Name: "support", Description: "Handles technical problems.", Model: &transferModel{}})
	if err != nil {
		log.Fatal(err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Model:     &transferModel{turns: []*genai.Content{genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "billing"}, genai.RoleModel)}},
		SubAgents: []agent.Agent{billing, support},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "helpdesk", Agent: coordinator, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	for ev, err := range r.Run(context.Background(), "user", "s1", genai.NewContentFromText("My login fails.", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			fmt.Println("run failed:", err)
			fmt.Println("invalid transfer target:", errors.Is(err, agent.ErrInvalidTransferTarget))
			return
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Printf("%s calls %s %v\n", ev.Author, p.FunctionCall.Name, p.FunctionCall.Args)
			case p.FunctionResponse != nil:
				resp := fmt.Sprint(p.FunctionResponse.Response)
				if e, ok := p.FunctionResponse.Response["error"].(string); ok {
					resp, _, _ = strings.Cut(e, ";")
					resp = "error: " + resp
				}
				fmt.Printf("%s: response of %s: %s\n", ev.Author, p.FunctionResponse.Name, resp)
			case p.Text != "":
				fmt.Printf("%s: %s\n", ev.Author, p.Text)
			}
		}
	}
}

// A transfer_to_agent call fails the run, with an error that matches
// ErrInvalidTransferTarget, when it names an agent of the tree that the caller
// may not transfer to. A call that names no agent of the tree is answered
// with an error the model sees, and the run goes on.
func ExampleErrInvalidTransferTarget() {
	fmt.Println("-- billing transfers to its peer support, which it may not do")
	runTransfer(genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "support"}, genai.RoleModel))

	fmt.Println("-- billing transfers to refunds, which is no agent of the tree")
	runTransfer(
		genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "refunds"}, genai.RoleModel),
		genai.NewContentFromText("I can help with that myself.", genai.RoleModel),
	)

	// Output:
	// -- billing transfers to its peer support, which it may not do
	// coordinator calls transfer_to_agent map[agent_name:billing]
	// coordinator: response of transfer_to_agent: map[result:<nil>]
	// billing calls transfer_to_agent map[agent_name:support]
	// billing: response of transfer_to_agent: map[result:<nil>]
	// run failed: Agent billing is not allowed to transfer to agent support.
	// invalid transfer target: true
	// -- billing transfers to refunds, which is no agent of the tree
	// coordinator calls transfer_to_agent map[agent_name:billing]
	// coordinator: response of transfer_to_agent: map[result:<nil>]
	// billing calls transfer_to_agent map[agent_name:refunds]
	// billing: response of transfer_to_agent: error: agent "refunds" is not a valid transfer target
	// billing: I can help with that myself.
}
