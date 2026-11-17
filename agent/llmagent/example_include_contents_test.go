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

// historyModel stands in for a real model. It keeps the contents of the
// last request it received and answers every request with "Noted.".
type historyModel struct{ last []*genai.Content }

func (*historyModel) Name() string { return "history" }

func (m *historyModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.last = req.Contents
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("Noted.", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
	}
}

// secondRequest runs two user turns in one session on an agent with the given
// IncludeContents and prints the contents of the second model request.
func secondRequest(include llmagent.IncludeContents) {
	m := &historyModel{}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m, IncludeContents: include})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	for _, text := range []string{"My name is Ana.", "Summarize this ticket."} {
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for _, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
	for _, c := range m.last {
		fmt.Println(" ", c.Role, c.Parts[0].Text)
	}
}

// IncludeContents decides how much of the session the model sees. By default
// each request carries the conversation so far. With IncludeContentsNone it
// carries only the current turn, which suits an agent whose task does not
// depend on earlier turns; the session itself still keeps every event.
func ExampleConfig_includeContents() {
	fmt.Println("default:")
	secondRequest(llmagent.IncludeContentsDefault)
	fmt.Println("none:")
	secondRequest(llmagent.IncludeContentsNone)
	// Output:
	// default:
	//   user My name is Ana.
	//   model Noted.
	//   user Summarize this ticket.
	// none:
	//   user Summarize this ticket.
}
