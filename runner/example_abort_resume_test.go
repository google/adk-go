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
	"sync/atomic"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// bookThenRecoverModel stands in for a model that first calls the book tool
// and then, given the tool's response, answers with text.
type bookThenRecoverModel struct {
	calls atomic.Int32
}

func (*bookThenRecoverModel) Name() string { return "book-then-recover" }

func (m *bookThenRecoverModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		content := genai.NewContentFromText("The booking was canceled; nothing was booked.", genai.RoleModel)
		if m.calls.Add(1) == 1 {
			content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "book", Args: map[string]any{}}},
			}}
		}
		yield(&model.LLMResponse{Content: content}, nil)
	}
}

// An aborted invocation can be resumed by its ID. The abort sealed the call
// the run left open, and the sealed response counts as the call's answer, so
// the resumed run hands it to the model and does not run the tool.
func ExampleWithAbortSignal_resume() {
	var bookings atomic.Int32
	book, err := functiontool.New(functiontool.Config{Name: "book", Description: "Books a table."},
		func(agent.Context, struct{}) (map[string]any, error) {
			bookings.Add(1)
			return map[string]any{"booked": true}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &bookThenRecoverModel{}, Tools: []tool.Tool{book}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		log.Fatal(err)
	}

	abort := make(chan struct{})
	invocationID := ""
	msg := genai.NewContentFromText("Book a table", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}, runner.WithAbortSignal(abort)) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.ErrorCode != "" {
			fmt.Println(ev.ErrorCode)
			continue
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Printf("%s calls %s\n", ev.Author, p.FunctionCall.Name)
				// The user changes their mind before the tool runs.
				invocationID = ev.InvocationID
				close(abort)
			}
		}
	}

	// A resume with no message continues the invocation from its history.
	for ev, err := range r.Run(context.Background(), "user", "session", nil, agent.RunConfig{}, runner.WithInvocationID(invocationID)) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
			fmt.Printf("%s: %s\n", ev.Author, ev.Content.Parts[0].Text)
		}
	}
	fmt.Println("bookings:", bookings.Load())

	// Output:
	// assistant calls book
	// INVOCATION_ABORTED
	// assistant: The booking was canceled; nothing was booked.
	// bookings: 0
}
