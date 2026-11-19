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

package exampletool_test

import (
	"context"
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
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/exampletool"
)

// recordingModel stands in for a real model: it answers every request with the
// same text and keeps the last request it received, so an example can show the
// system instruction the example tool built.
type recordingModel struct {
	name    string
	lastReq *model.LLMRequest
}

func (m *recordingModel) Name() string { return m.name }

func (m *recordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText("OK.", genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// runTurn runs an agent with the given model and tool for one user message,
// and returns the part of the system instruction sent to the model that the
// example tool added.
func runTurn(llm *recordingModel, examples tool.Tool, userText string) string {
	ctx := context.Background()
	a, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Model:       llm,
		Instruction: "Help the user with their order.",
		Tools:       []tool.Tool{examples},
	})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText(userText, genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}
	var sys strings.Builder
	for _, part := range llm.lastReq.Config.SystemInstruction.Parts {
		sys.WriteString(part.Text)
	}
	// The example tool appends its block after the agent's own instruction.
	_, block, _ := strings.Cut(sys.String(), "<EXAMPLES>")
	return "<EXAMPLES>" + block
}

// This example gives an agent a fixed list of examples. On every turn the
// tool appends them to the system instruction of the model request. A function
// call in an example is written as a call with its arguments sorted by name;
// for a Gemini model, or a request that names no model, it is in a plain code
// fence. A function response is written in the layout of Go's %v.
func ExampleNew() {
	examples, err := exampletool.New(exampletool.ExampleToolConfig{
		Examples: []*exampletool.Example{
			{
				Input: genai.NewContentFromText("Where is order 4417?", genai.RoleUser),
				Output: []*genai.Content{
					genai.NewContentFromFunctionCall("check_order_status", map[string]any{"order_id": "4417"}, genai.RoleModel),
					genai.NewContentFromFunctionResponse("check_order_status", map[string]any{"status": "shipped"}, genai.RoleUser),
					genai.NewContentFromText("Order 4417 shipped on Tuesday.", genai.RoleModel),
				},
			},
			{
				Input:  genai.NewContentFromText("I want a refund.", genai.RoleUser),
				Output: []*genai.Content{genai.NewContentFromText("Which order number is that for?", genai.RoleModel)},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(runTurn(&recordingModel{name: "gemini-2.5-flash"}, examples, "Where is order 981?"))
	// Output:
	// <EXAMPLES>
	// Begin few-shot
	// The following are examples of user queries and model responses using the available tools.
	//
	// EXAMPLE 1:
	// Begin example
	// [user]
	// Where is order 4417?
	// [model]
	// ```
	// check_order_status(order_id='4417')
	// ```
	// [user]
	// ```
	// &{<nil>  []  check_order_status map[status:shipped]}
	// ```
	// [model]
	// Order 4417 shipped on Tuesday.
	// End example
	//
	// EXAMPLE 2:
	// Begin example
	// [user]
	// I want a refund.
	// [model]
	// Which order number is that for?
	// End example
	//
	// End few-shot
	// <EXAMPLES>
}

// topicProvider returns the examples stored under the first topic word found in
// the user query.
type topicProvider struct {
	examplesByTopic map[string][]*exampletool.Example
}

func (p *topicProvider) Examples(_ context.Context, query string) ([]*exampletool.Example, error) {
	fmt.Printf("provider query: %q\n", query)
	for word := range strings.FieldsSeq(strings.ToLower(query)) {
		if examples, ok := p.examplesByTopic[strings.Trim(word, "?.,!")]; ok {
			return examples, nil
		}
	}
	return nil, nil
}

// This example configures the example tool with a Provider instead of a fixed
// list. On each turn the tool joins the text of the user message into a query,
// calls Provider.Examples with it, and appends the examples it returns to the
// system instruction of the model request. The model is not a Gemini model, so
// the function call is in a tool_code fence.
func ExampleNew_provider() {
	provider := &topicProvider{
		examplesByTopic: map[string][]*exampletool.Example{
			"capital": {{
				Input: genai.NewContentFromText("What is the capital of France?", genai.RoleUser),
				Output: []*genai.Content{
					genai.NewContentFromFunctionCall("get_capital", map[string]any{"country": "France"}, genai.RoleModel),
					genai.NewContentFromText("Paris", genai.RoleModel),
				},
			}},
			"sum": {{
				Input:  genai.NewContentFromText("What is the sum of 2 and 2?", genai.RoleUser),
				Output: []*genai.Content{genai.NewContentFromText("4", genai.RoleModel)},
			}},
		},
	}
	examples, err := exampletool.New(exampletool.ExampleToolConfig{Provider: provider})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(runTurn(&recordingModel{name: "claude-sonnet-4-5"}, examples, "What is the capital of Spain?"))
	// Output:
	// provider query: "What is the capital of Spain?"
	// <EXAMPLES>
	// Begin few-shot
	// The following are examples of user queries and model responses using the available tools.
	//
	// EXAMPLE 1:
	// Begin example
	// [user]
	// What is the capital of France?
	// [model]
	// ```tool_code
	// get_capital(country='France')
	// ```
	// Paris
	// End example
	//
	// End few-shot
	// <EXAMPLES>
}
