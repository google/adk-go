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

package planner_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// cannedModel stands in for a real model: it answers every request with the
// same tagged Plan-Re-Act text.
type cannedModel struct{}

func (cannedModel) Name() string { return "canned" }

func (cannedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText("/*PLANNING*/Look up the forecast./*FINAL_ANSWER*/It will be sunny.", genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// An agent configured with a PlanReActPlanner asks the model to plan with
// tags, and the planner turns the tagged reply into a thought part and the
// final answer.
func ExamplePlanReActPlanner() {
	a, err := llmagent.New(llmagent.Config{
		Name:    "weather_agent",
		Model:   cannedModel{},
		Planner: planner.NewPlanReActPlanner(),
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
	msg := genai.NewContentFromText("Will it rain tomorrow?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			fmt.Printf("thought=%t %q\n", p.Thought, p.Text)
		}
	}
	// Output:
	// thought=true "Look up the forecast."
	// thought=false "It will be sunny."
}

// ProcessPlanningResponse can be called directly, without an agent run. It
// marks a part that starts with a planning tag as a thought, strips the tag,
// and keeps the function calls that follow.
func ExamplePlanReActPlanner_ProcessPlanningResponse() {
	parts := []*genai.Part{
		{Text: "/*REASONING*/The user wants the weather."},
		{FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
	}
	got, err := planner.NewPlanReActPlanner().ProcessPlanningResponse(nil, parts)
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range got {
		if p.FunctionCall != nil {
			fmt.Printf("call %s\n", p.FunctionCall.Name)
			continue
		}
		fmt.Printf("thought=%t %q\n", p.Thought, p.Text)
	}
	// Output:
	// thought=true "The user wants the weather."
	// call get_weather
}
