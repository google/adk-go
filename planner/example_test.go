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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// cannedModel stands in for a real model: it answers every request with the
// same parts and keeps the last request it received, so an example can show
// what the planner sent to the model.
type cannedModel struct {
	reply   []*genai.Part
	lastReq *model.LLMRequest
}

func (*cannedModel) Name() string { return "canned" }

func (m *cannedModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: m.reply},
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// systemInstruction returns the text of the system instruction of req.
func systemInstruction(req *model.LLMRequest) string {
	if req.Config == nil || req.Config.SystemInstruction == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range req.Config.SystemInstruction.Parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

// runAgent runs a in a new in-memory session with one user message and calls
// handle for every event the run yields.
func runAgent(a agent.Agent, userText string, handle func(*session.Event)) {
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
	msg := genai.NewContentFromText(userText, genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		handle(ev)
	}
}

// printParts prints the parts of ev, one per line, with their Thought flag.
func printParts(ev *session.Event) {
	if ev.Content == nil {
		return
	}
	for _, p := range ev.Content.Parts {
		fmt.Printf("thought=%t %q\n", p.Thought, p.Text)
	}
}

// An agent configured with a PlanReActPlanner sends the Plan-Re-Act
// instruction to the model with every request, after the agent's own
// instruction. The model answers under the planning tags, and the planner
// turns the tagged reply into a thought part, which holds the plan, and a plain
// part, which holds the final answer.
func ExamplePlanReActPlanner() {
	llm := &cannedModel{reply: []*genai.Part{
		{Text: "/*PLANNING*/Look up the forecast./*FINAL_ANSWER*/It will be sunny."},
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       llm,
		Instruction: "You answer questions about the weather.",
		Planner:     planner.NewPlanReActPlanner(),
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
		for _, p := range ev.Content.Parts {
			fmt.Printf("thought=%t %q\n", p.Thought, p.Text)
		}
	}

	sys := systemInstruction(llm.lastReq)
	fmt.Println("starts with the agent instruction:", strings.HasPrefix(sys, "You answer questions about the weather."))
	for line := range strings.SplitSeq(sys, "\n") {
		if strings.HasPrefix(line, "Follow this format") {
			fmt.Println(line)
		}
	}
	// Output:
	// thought=true "Look up the forecast."
	// thought=false "It will be sunny."
	// starts with the agent instruction: true
	// Follow this format when answering the question: (1) The planning part should be under /*PLANNING*/. (2) The tool code snippets should be under /*ACTION*/, and the reasoning parts should be under /*REASONING*/. (3) The final answer part should be under /*FINAL_ANSWER*/.
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

// A BuiltInPlanner sets the thinking config of every request the agent sends,
// in place of any ThinkingConfig in the agent's GenerateContentConfig. It adds
// no instruction and does not rewrite the response: the thought part comes
// from the model itself.
func ExampleBuiltInPlanner() {
	llm := &cannedModel{reply: []*genai.Part{
		{Text: "The user asks for a capital; Paris is the capital of France.", Thought: true},
		{Text: "The capital of France is Paris."},
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:        "capital_agent",
		Model:       llm,
		Instruction: "You answer questions about capital cities.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](128)},
		},
		Planner: planner.NewBuiltInPlanner(&genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingBudget:  genai.Ptr[int32](1024),
		}),
	})
	if err != nil {
		log.Fatal(err)
	}
	runAgent(a, "What is the capital of France?", printParts)

	tc := llm.lastReq.Config.ThinkingConfig
	fmt.Printf("thinking budget %d, include thoughts %t\n", *tc.ThinkingBudget, tc.IncludeThoughts)
	fmt.Println("planning instruction added:", strings.Contains(systemInstruction(llm.lastReq), "/*PLANNING*/"))
	// Output:
	// thought=true "The user asks for a capital; Paris is the capital of France."
	// thought=false "The capital of France is Paris."
	// thinking budget 1024, include thoughts true
	// planning instruction added: false
}

// stepsPlanner is an instruction-only planner: it implements Planner but not
// RewritingPlanner.
type stepsPlanner struct{}

func (stepsPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) (string, error) {
	return "Before you answer, list the steps you will take.", nil
}

// A planner that implements only Planner adds its instruction to every request,
// after the agent's own instruction and separated from it by a blank line. The
// model's response reaches the events unchanged.
func ExamplePlanner() {
	llm := &cannedModel{reply: []*genai.Part{
		{Text: "1. Boil water. 2. Add pasta. Cook the pasta for ten minutes."},
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:        "cooking_agent",
		Model:       llm,
		Instruction: "You are a cooking assistant.",
		Planner:     stepsPlanner{},
	})
	if err != nil {
		log.Fatal(err)
	}
	runAgent(a, "How do I cook pasta?", printParts)

	sys := systemInstruction(llm.lastReq)
	fmt.Println("starts with the agent instruction:", strings.HasPrefix(sys, "You are a cooking assistant."))
	fmt.Println("ends with the planning instruction:", strings.HasSuffix(sys, "\n\nBefore you answer, list the steps you will take."))
	// Output:
	// thought=false "1. Boil water. 2. Add pasta. Cook the pasta for ten minutes."
	// starts with the agent instruction: true
	// ends with the planning instruction: true
}

// draftPlanner asks the model for a draft and an answer, shows the draft as a
// thought, and keeps the latest draft in session state.
type draftPlanner struct{}

func (draftPlanner) BuildPlanningInstruction(agent.ReadonlyContext, *model.LLMRequest) (string, error) {
	return "Write a draft after 'Draft:' and your answer after 'Answer:'.", nil
}

func (draftPlanner) ProcessPlanningResponse(ctx agent.Context, parts []*genai.Part) ([]*genai.Part, error) {
	var out []*genai.Part
	for _, p := range parts {
		draft, answer, ok := strings.Cut(p.Text, "Answer:")
		if !ok {
			out = append(out, p)
			continue
		}
		draft = strings.TrimSpace(strings.TrimPrefix(draft, "Draft:"))
		if err := ctx.State().Set("last_draft", draft); err != nil {
			return nil, fmt.Errorf("save draft: %w", err)
		}
		out = append(out,
			&genai.Part{Text: draft, Thought: true},
			&genai.Part{Text: strings.TrimSpace(answer)},
		)
	}
	return out, nil
}

// A RewritingPlanner rewrites the parts of each model response. When it
// writes session state, the run yields an event that carries the state change
// and no content, before the event with the rewritten response.
func ExampleRewritingPlanner() {
	llm := &cannedModel{reply: []*genai.Part{
		{Text: "Draft: Haiku, five-seven-five. Answer: An old silent pond."},
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:    "poet",
		Model:   llm,
		Planner: draftPlanner{},
	})
	if err != nil {
		log.Fatal(err)
	}
	runAgent(a, "Write the first line of a haiku.", func(ev *session.Event) {
		if len(ev.Actions.StateDelta) > 0 {
			fmt.Printf("%s: state %v\n", ev.Author, ev.Actions.StateDelta)
		}
		printParts(ev)
	})
	// Output:
	// poet: state map[last_draft:Haiku, five-seven-five.]
	// thought=true "Haiku, five-seven-five."
	// thought=false "An old silent pond."
}
