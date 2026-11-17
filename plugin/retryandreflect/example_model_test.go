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

package retryandreflect_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/retryandreflect"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// modelExampleLLM stands in for a real model: it answers its n-th request
// with the n-th of its responses.
type modelExampleLLM struct {
	responses []*model.LLMResponse
	next      int
}

func (m *modelExampleLLM) Name() string { return "example-model" }

func (m *modelExampleLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.next >= len(m.responses) {
			yield(nil, errors.New("no response left"))
			return
		}
		m.next++
		yield(m.responses[m.next-1], nil)
	}
}

// modelExampleMalformed is the response a model gives when it produced a
// function call it could not complete.
func modelExampleMalformed() *model.LLMResponse {
	return &model.LLMResponse{
		ErrorCode:    "MALFORMED_FUNCTION_CALL",
		ErrorMessage: "the function call was malformed",
		FinishReason: genai.FinishReasonMalformedFunctionCall,
	}
}

// modelExampleRun runs an agent on llm with the plugin p for one user
// message and prints the reflection calls, error codes and text it sees.
func modelExampleRun(llm model.LLM, p *plugin.Plugin) {
	ctx := context.Background()
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: sessions,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("What is the capital of France?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			fmt.Println("run error:", err)
			return
		}
		for _, call := range ev.FunctionCalls() {
			fmt.Println("reflection call, retry_count:", call.Args["retry_count"])
		}
		if ev.ErrorCode != "" {
			fmt.Println("error code:", ev.ErrorCode)
		}
		if ev.Content != nil {
			for _, part := range ev.Content.Parts {
				if part.Text != "" {
					fmt.Println("answer:", part.Text)
				}
			}
		}
	}
}

// A model that fails twice gets a reflection call after each failure and
// recovers on its third attempt.
func ExampleNewModel() {
	p, err := retryandreflect.NewModel()
	if err != nil {
		log.Fatal(err)
	}
	llm := &modelExampleLLM{responses: []*model.LLMResponse{
		modelExampleMalformed(),
		modelExampleMalformed(),
		{Content: genai.NewContentFromText("Paris", genai.RoleModel)},
	}}

	modelExampleRun(llm, p)
	// Output:
	// reflection call, retry_count: 1
	// reflection call, retry_count: 2
	// answer: Paris
}

// With WithModelErrorIfRetryExceeded(false), the failure past the retry
// budget ends the run without an error, and the model's error response is
// the agent's last event.
func ExampleNewModel_throwDisabled() {
	p, err := retryandreflect.NewModel(
		retryandreflect.WithModelMaxRetries(1),
		retryandreflect.WithModelErrorIfRetryExceeded(false),
	)
	if err != nil {
		log.Fatal(err)
	}
	llm := &modelExampleLLM{responses: []*model.LLMResponse{
		modelExampleMalformed(),
		modelExampleMalformed(),
	}}

	modelExampleRun(llm, p)
	// Output:
	// reflection call, retry_count: 1
	// error code: MALFORMED_FUNCTION_CALL
}
