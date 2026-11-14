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

package contextfilter_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/contextfilter"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// printingModel stands in for a real model: it prints the text of every
// content of each request it receives and replies "ok".
type printingModel struct{}

func (printingModel) Name() string { return "printing" }

func (printingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var texts []string
	for _, c := range req.Contents {
		texts = append(texts, c.Parts[0].Text)
	}
	fmt.Println(texts)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}, nil)
	}
}

// converse runs one session with a runner that has plugin p and sends each
// message in turn.
func converse(p *plugin.Plugin, messages ...string) {
	ctx := context.Background()
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: printingModel{}})
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
	for _, text := range messages {
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
}

// A plugin that keeps the last two invocations sends the whole history until
// the conversation holds three invocations, and from then on only the last
// two. The session itself keeps every event.
func Example() {
	p, err := contextfilter.New(contextfilter.Config{NumInvocationsToKeep: 2})
	if err != nil {
		log.Fatal(err)
	}

	converse(p, "one", "two", "three")
	// Output:
	// [one]
	// [one ok two]
	// [two ok three]
}

// A custom filter decides which contents reach the model. This one drops the
// model's earlier replies.
func ExampleNew_customFilter() {
	userOnly := func(contents []*genai.Content) ([]*genai.Content, error) {
		var kept []*genai.Content
		for _, c := range contents {
			if c.Role == genai.RoleUser {
				kept = append(kept, c)
			}
		}
		return kept, nil
	}
	p, err := contextfilter.New(contextfilter.Config{CustomFilter: userOnly})
	if err != nil {
		log.Fatal(err)
	}

	converse(p, "one", "two", "three")
	// Output:
	// [one]
	// [one two]
	// [one two three]
}
