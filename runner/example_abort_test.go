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

package runner_test

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
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// lookupCallModel stands in for a model that answers with a call to the
// slow_lookup tool.
type lookupCallModel struct{}

func (lookupCallModel) Name() string { return "lookup-call" }

func (lookupCallModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "slow_lookup", Args: map[string]any{}}},
		}}}, nil)
	}
}

// Closing the abort signal stops the run. The caller keeps ranging and
// receives the event that answers the call the run left open, which is also
// stored in the session, so the next turn starts from a consistent history.
func ExampleWithAbortSignal() {
	lookup, err := functiontool.New(functiontool.Config{Name: "slow_lookup", Description: "Looks a value up slowly."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			// The abort cancels the context tools run under.
			<-ctx.Done()
			return nil, ctx.Err()
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: lookupCallModel{}, Tools: []tool.Tool{lookup}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	abort := make(chan struct{})
	msg := genai.NewContentFromText("Look it up", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}, runner.WithAbortSignal(abort)) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Printf("%s calls %s\n", ev.Author, p.FunctionCall.Name)
				// The user does not want to wait for the lookup.
				close(abort)
			case p.FunctionResponse != nil:
				fmt.Printf("%s: %s %v\n", ev.ErrorCode, p.FunctionResponse.Name, p.FunctionResponse.Response)
			}
		}
	}

	// Output:
	// assistant calls slow_lookup
	// INVOCATION_ABORTED: slow_lookup map[error:Invocation was aborted by client.]
}
