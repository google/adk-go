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

package model_test

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
	"google.golang.org/adk/v2/tool/functiontool"
)

// toolCallingModel stands in for a real model. Its first answer calls the
// get_time tool without a function call ID, as Gemini does; its second
// answer is text. It keeps the second request.
type toolCallingModel struct {
	calls  int
	second *model.LLMRequest
}

func (*toolCallingModel) Name() string { return "tool-calling" }

func (m *toolCallingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	content := genai.NewContentFromFunctionCall("get_time", map[string]any{}, genai.RoleModel)
	if m.calls == 2 {
		m.second = req
		content = genai.NewContentFromText("It is noon.", genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// idPairingModel is a toolCallingModel that declares, through
// model.CapabilitiesProvider, that it pairs tool calls with their results by
// ID.
type idPairingModel struct {
	toolCallingModel
}

func (*idPairingModel) Capabilities() model.Capabilities {
	return model.Capabilities{ClientFunctionCallIDs: true}
}

// callIDsInSecondRequest runs one user turn on llm with the get_time tool and
// prints the function call and response IDs of the second request.
func callIDsInSecondRequest(llm model.LLM, second func() *model.LLMRequest) {
	getTime, err := functiontool.New(functiontool.Config{Name: "get_time", Description: "Returns the time."},
		func(agent.Context, struct{}) (string, error) { return "12:00", nil })
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "clock", Model: llm, Tools: []tool.Tool{getTime}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("What time is it?", genai.RoleUser)
	for _, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}
	describe := func(id string) string {
		if strings.HasPrefix(id, "adk-") {
			return "adk-..."
		}
		return fmt.Sprintf("%q", id)
	}
	for _, c := range second().Contents {
		for _, p := range c.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Println("  call ID:", describe(p.FunctionCall.ID))
			case p.FunctionResponse != nil:
				fmt.Println("  response ID:", describe(p.FunctionResponse.ID))
			}
		}
	}
}

// When a model returns a function call without an ID, ADK gives it one that
// starts with "adk-" and stores it in the session. The next request keeps
// that ID only for a model whose Capabilities set ClientFunctionCallIDs; for
// any other model, including one that does not implement
// CapabilitiesProvider, ADK removes the ID from the request copy.
func ExampleCapabilities_clientFunctionCallIDs() {
	pairing := &idPairingModel{}
	fmt.Println("declares ClientFunctionCallIDs:")
	callIDsInSecondRequest(pairing, func() *model.LLMRequest { return pairing.second })

	plain := &toolCallingModel{}
	fmt.Println("declares nothing:")
	callIDsInSecondRequest(plain, func() *model.LLMRequest { return plain.second })
	// Output:
	// declares ClientFunctionCallIDs:
	//   call ID: adk-...
	//   response ID: adk-...
	// declares nothing:
	//   call ID: ""
	//   response ID: ""
}
