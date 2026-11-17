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

// docRecordingModel stands in for a real model. It records every request and
// answers each with the text "Paris".
type docRecordingModel struct{ requests []*model.LLMRequest }

func (*docRecordingModel) Name() string { return "recording" }

func (m *docRecordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.requests = append(m.requests, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("Paris", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
	}
}

// ModelInputContext gives the model contents for one invocation, such as
// facts an application looks up per request, without storing them in the
// session. The model sees them just before the user's message; the session
// holds only the message and the reply.
func ExampleRunConfig_modelInputContext() {
	m := &docRecordingModel{}
	a, err := llmagent.New(llmagent.Config{Name: "helper", Model: m})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions, AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}

	cfg := agent.RunConfig{ModelInputContext: &[]*genai.Content{
		genai.NewContentFromText("The user is in France.", genai.RoleUser),
	}}
	msg := genai.NewContentFromText("What is the capital?", genai.RoleUser)
	for _, err := range r.Run(context.Background(), "user", "s1", msg, cfg) {
		if err != nil {
			log.Fatal(err)
		}
	}

	fmt.Println("request:")
	for _, c := range m.requests[0].Contents {
		fmt.Println(" ", c.Role, c.Parts[0].Text)
	}
	resp, err := sessions.Get(context.Background(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("session:")
	for ev := range resp.Session.Events().All() {
		fmt.Println(" ", ev.Author, ev.Content.Parts[0].Text)
	}
	// Output:
	// request:
	//   user The user is in France.
	//   user What is the capital?
	// session:
	//   user What is the capital?
	//   helper Paris
}
