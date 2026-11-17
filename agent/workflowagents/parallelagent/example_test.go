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

package parallelagent_test

import (
	"context"
	"fmt"
	"iter"
	"log"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// parallelDocsModel stands in for a model that always answers with reply.
type parallelDocsModel struct{ reply string }

func (parallelDocsModel) Name() string { return "parallel-docs" }

func (m parallelDocsModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(m.reply, genai.RoleModel)}, nil)
	}
}

// A ParallelAgent runs its sub-agents at the same time, each on its own
// branch, so a sub-agent does not see the events of its siblings. The order in
// which the branches' events arrive varies from run to run, so the example
// sorts them. Each sub-agent saves its answer under its OutputKey, where an
// agent that runs after the ParallelAgent can read it.
func ExampleNew() {
	weather, err := llmagent.New(llmagent.Config{
		Name:      "weather",
		Model:     parallelDocsModel{reply: "Sunny, 21 degrees."},
		OutputKey: "weather",
	})
	if err != nil {
		log.Fatal(err)
	}
	news, err := llmagent.New(llmagent.Config{
		Name:      "news",
		Model:     parallelDocsModel{reply: "The library opens on Monday."},
		OutputKey: "news",
	})
	if err != nil {
		log.Fatal(err)
	}
	gather, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "gather", SubAgents: []agent.Agent{weather, news}},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	if _, err := sessions.Create(ctx, &session.CreateRequest{AppName: "docs", UserID: "user", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "docs", Agent: gather, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	var lines []string
	for ev, err := range r.Run(ctx, "user", "s1", genai.NewContentFromText("Morning briefing", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		lines = append(lines, fmt.Sprintf("%s (branch %s): %s", ev.Author, ev.Branch, ev.Content.Parts[0].Text))
	}
	slices.Sort(lines)
	for _, l := range lines {
		fmt.Println(l)
	}

	got, err := sessions.Get(ctx, &session.GetRequest{AppName: "docs", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	for _, key := range []string{"weather", "news"} {
		v, err := got.Session.State().Get(key)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("state[%q] = %v\n", key, v)
	}

	// Output:
	// news (branch gather.news): The library opens on Monday.
	// weather (branch gather.weather): Sunny, 21 degrees.
	// state["weather"] = Sunny, 21 degrees.
	// state["news"] = The library opens on Monday.
}
