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
	"maps"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// plainModel stands in for a real model that does not declare its
// capabilities: it keeps the request it receives and answers with JSON.
type plainModel struct {
	lastReq *model.LLMRequest
}

func (*plainModel) Name() string { return "my-model" }

func (m *plainModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(`{"capital": "Paris"}`, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// capableModel is a plainModel that declares, through
// model.CapabilitiesProvider, that it accepts an output schema together with
// tools.
type capableModel struct {
	plainModel
}

func (*capableModel) Capabilities() model.Capabilities {
	return model.Capabilities{OutputSchemaAndTools: true}
}

// describeRequest runs an agent with an output schema and one tool on llm,
// then prints the tools and the response schema of the request it received.
func describeRequest(llm model.LLM, lastReq func() *model.LLMRequest) {
	type cityArgs struct {
		Country string `json:"country"`
	}
	lookup, err := functiontool.New(functiontool.Config{
		Name:        "lookup_capital",
		Description: "Returns the capital of a country.",
	}, func(_ agent.Context, args cityArgs) (string, error) { return "Paris", nil })
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "capital_agent",
		Model: llm,
		Tools: []tool.Tool{lookup},
		OutputSchema: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{"capital": {Type: genai.TypeString}},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

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
	msg := genai.NewContentFromText("What is the capital of France?", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}

	req := lastReq()
	fmt.Println("tools:", slices.Sorted(maps.Keys(req.Tools)))
	fmt.Println("native response schema:", req.Config.ResponseSchema != nil)
}

// An LLM declares whether it accepts an output schema together with tools by
// implementing CapabilitiesProvider. An agent with both sends the schema
// natively to a model that declares it. A model that does not implement
// CapabilitiesProvider gets this only if it is a Gemini model on the Vertex AI
// backend; any other model, such as plainModel here, gets the
// set_model_response tool instead of a native response schema.
func ExampleCapabilitiesProvider() {
	capable := &capableModel{}
	fmt.Println("declares OutputSchemaAndTools:")
	describeRequest(capable, func() *model.LLMRequest { return capable.lastReq })

	plain := &plainModel{}
	fmt.Println("declares nothing:")
	describeRequest(plain, func() *model.LLMRequest { return plain.lastReq })
	// Output:
	// declares OutputSchemaAndTools:
	// tools: [lookup_capital]
	// native response schema: true
	// declares nothing:
	// tools: [lookup_capital set_model_response]
	// native response schema: false
}
