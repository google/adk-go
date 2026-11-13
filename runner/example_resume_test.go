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
	"slices"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// cannedModel stands in for a model: each call returns the next of its
// responses, whole, in one response.
type cannedModel struct {
	mu        sync.Mutex
	responses []*genai.Content
}

func (*cannedModel) Name() string { return "canned" }

func (m *cannedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		if len(m.responses) == 0 {
			m.mu.Unlock()
			yield(&model.LLMResponse{Content: genai.NewContentFromText("(no response left)", genai.RoleModel)}, nil)
			return
		}
		content := m.responses[0]
		m.responses = m.responses[1:]
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: content}, nil)
	}
}

func exampleCall(id, name string) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}},
	}}
}

// describe prints one line for ev: its author and what it holds.
func describe(ev *session.Event) {
	switch {
	case ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent:
		fmt.Printf("%s: end of agent\n", ev.Author)
	case ev.Actions.AgentState != nil:
		fmt.Printf("%s: agent state %v\n", ev.Author, ev.Actions.AgentState)
	case ev.Content == nil || len(ev.Content.Parts) == 0:
		fmt.Printf("%s: (no content)\n", ev.Author)
	case ev.Content.Parts[0].FunctionCall != nil:
		fmt.Printf("%s: calls %s\n", ev.Author, ev.Content.Parts[0].FunctionCall.Name)
	case ev.Content.Parts[0].FunctionResponse != nil:
		fr := ev.Content.Parts[0].FunctionResponse
		fmt.Printf("%s: %s returned %v\n", ev.Author, fr.Name, fr.Response)
	default:
		fmt.Printf("%s: %s\n", ev.Author, ev.Content.Parts[0].Text)
	}
}

// A resumable runner pauses an invocation on a long-running tool call that
// has no result yet. The application gets the result from elsewhere, here a
// manager's approval, and sends it as a function response, which continues
// the paused invocation.
func ExampleConfig_resumable() {
	approval, err := functiontool.New(functiontool.Config{
		Name:          "request_approval",
		Description:   "Asks a manager to approve a refund.",
		IsLongRunning: true,
	}, func(agent.Context, struct{}) (map[string]any, error) {
		// The request is sent; the answer comes later, so there is no
		// result yet.
		return nil, nil
	})
	if err != nil {
		log.Fatal(err)
	}
	assistant, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &cannedModel{responses: []*genai.Content{
			exampleCall("approval-1", "request_approval"),
			genai.NewContentFromText("The refund is approved and issued.", genai.RoleModel),
		}},
		Tools: []tool.Tool{approval},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "refunds",
		Agent:             assistant,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	fmt.Println("-- run 1")
	var pausedCall *genai.FunctionCall
	var firstInvocation string
	msg := genai.NewContentFromText("Refund order 42", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
		firstInvocation = ev.InvocationID
		for _, id := range ev.LongRunningToolIDs {
			for _, fc := range ev.Content.Parts {
				if fc.FunctionCall != nil && fc.FunctionCall.ID == id {
					pausedCall = fc.FunctionCall
				}
			}
		}
	}

	fmt.Println("-- run 2: the manager approved")
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID:       pausedCall.ID,
		Name:     pausedCall.Name,
		Response: map[string]any{"approved": true},
	}}}}
	for ev, err := range r.Run(ctx, "user", "s1", answer, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
		fmt.Println("  same invocation:", ev.InvocationID == firstInvocation)
	}

	// Output:
	// -- run 1
	// assistant: calls request_approval
	// -- run 2: the manager approved
	// assistant: The refund is approved and issued.
	//   same invocation: true
	// assistant: end of agent
	//   same invocation: true
}

// A resumable invocation can be continued by its ID after the process
// running it stopped. Here the caller stops reading after the model asked
// for a tool, before the tool ran. Resuming the invocation runs that
// function call, which never got a response, and the agent goes on from
// there.
func ExampleWithInvocationID() {
	bookings := 0
	book, err := functiontool.New(functiontool.Config{Name: "book", Description: "Books a table."},
		func(agent.Context, struct{}) (map[string]any, error) {
			bookings++
			return map[string]any{"table": 7}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	assistant, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &cannedModel{responses: []*genai.Content{
			exampleCall("call-1", "book"),
			genai.NewContentFromText("You have table 7.", genai.RoleModel),
		}},
		Tools: []tool.Tool{book},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "bookings",
		Agent:             assistant,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	fmt.Println("-- run 1, stopped after the call")
	invocationID := ""
	msg := genai.NewContentFromText("Book a table for two", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
		invocationID = ev.InvocationID
		break // The process stops here.
	}
	fmt.Println("bookings:", bookings)

	fmt.Println("-- run 2, resumed by invocation ID")
	for ev, err := range r.Run(ctx, "user", "s1", nil, agent.RunConfig{}, runner.WithInvocationID(invocationID)) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
	}
	fmt.Println("bookings:", bookings)

	// Output:
	// -- run 1, stopped after the call
	// assistant: calls book
	// bookings: 0
	// -- run 2, resumed by invocation ID
	// assistant: book returned map[table:7]
	// assistant: You have table 7.
	// assistant: end of agent
	// bookings: 1
}

// A custom agent takes part in resumption by recording its progress in
// events: one whose actions carry AgentState after each step it finishes,
// and one with EndOfAgent set when it is done. When the invocation is
// resumed, the agent reads its last state from the invocation's events and
// skips the steps it already finished. A resume of an invocation whose
// agent has ended yields nothing.
func ExampleConfig_resumable_customAgent() {
	steps := []string{"draft", "review", "publish"}
	pipeline, err := agent.New(agent.Config{
		Name: "pipeline",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				name := ctx.Agent().Name()
				next := 0
				for ev := range ctx.Session().Events().All() {
					if ev.InvocationID != ctx.InvocationID() || ev.Author != name || ev.Actions.AgentState == nil {
						continue
					}
					completed, _ := ev.Actions.AgentState["completed"].(string)
					next = slices.Index(steps, completed) + 1
				}
				for _, step := range steps[next:] {
					done := session.NewEvent(ctx, ctx.InvocationID())
					done.Author = name
					done.Content = genai.NewContentFromText(step+" done", genai.RoleModel)
					if !yield(done, nil) {
						return
					}
					state := session.NewEvent(ctx, ctx.InvocationID())
					state.Author = name
					state.Actions.AgentState = map[string]any{"completed": step}
					if !yield(state, nil) {
						return
					}
				}
				end := session.NewEvent(ctx, ctx.InvocationID())
				end.Author = name
				end.Actions.EndOfAgent = new(bool)
				*end.Actions.EndOfAgent = true
				yield(end, nil)
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "publishing",
		Agent:             pipeline,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	fmt.Println("-- run 1, stopped after the first step")
	invocationID := ""
	msg := genai.NewContentFromText("Publish the post", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
		invocationID = ev.InvocationID
		if ev.Actions.AgentState != nil {
			break // The process stops here.
		}
	}

	for _, run := range []string{"-- run 2, resumed", "-- run 3, resumed again"} {
		fmt.Println(run)
		for ev, err := range r.Run(ctx, "user", "s1", nil, agent.RunConfig{}, runner.WithInvocationID(invocationID)) {
			if err != nil {
				log.Fatal(err)
			}
			describe(ev)
		}
	}

	// Output:
	// -- run 1, stopped after the first step
	// pipeline: draft done
	// pipeline: agent state map[completed:draft]
	// -- run 2, resumed
	// pipeline: review done
	// pipeline: agent state map[completed:review]
	// pipeline: publish done
	// pipeline: agent state map[completed:publish]
	// pipeline: end of agent
	// -- run 3, resumed again
}

// On a runner whose root is an LlmAgent, a Config.Node or a graph workflow,
// a user message sent while a task agent is still working on its task joins
// that task's invocation: it is stored as a new user turn in the task's
// isolation scope, and the task agent receives it. The runner does not
// need to be resumable.
func ExampleRunner_Run_joinTask() {
	booker, err := llmagent.New(llmagent.Config{
		Name:        "flight_booker",
		Description: "Books flights.",
		Mode:        llmagent.ModeTask,
		Model: &cannedModel{responses: []*genai.Content{
			genai.NewContentFromText("Which city?", genai.RoleModel),
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "finish-1", Name: "finish_task", Args: map[string]any{"result": "Flight to Paris booked."},
			}}}},
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		SubAgents: []agent.Agent{booker},
		Model: &cannedModel{responses: []*genai.Content{
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "task-1", Name: "flight_booker", Args: map[string]any{"request": "Book a flight"},
			}}}},
			genai.NewContentFromText("Your flight to Paris is booked.", genai.RoleModel),
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "travel",
		Agent:             coordinator,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	firstInvocation := ""
	for _, text := range []string{"Book me a flight", "Paris"} {
		fmt.Printf("-- user: %s\n", text)
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for ev, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}, runner.WithYieldUserMessage()) {
			if err != nil {
				log.Fatal(err)
			}
			if firstInvocation == "" {
				firstInvocation = ev.InvocationID
			}
			if ev.Author == "user" && ev.Content.Parts[0].Text != "" {
				fmt.Printf("stored in scope %q, first invocation: %t\n", ev.IsolationScope, ev.InvocationID == firstInvocation)
				continue
			}
			describe(ev)
		}
	}

	// Output:
	// -- user: Book me a flight
	// stored in scope "", first invocation: true
	// coordinator: calls flight_booker
	// flight_booker: Which city?
	// -- user: Paris
	// stored in scope "task-1", first invocation: true
	// flight_booker: calls finish_task
	// flight_booker: finish_task returned map[result:Task completed.]
	// user: flight_booker returned map[result:Flight to Paris booked.]
	// coordinator: Your flight to Paris is booked.
}
