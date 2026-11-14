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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// historyModel stands in for a model. It answers every call with the texts
// of the request's user contents joined by " | ", so the output shows which
// user turns reached the model.
type historyModel struct{}

func (historyModel) Name() string { return "history" }

func (historyModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var texts []string
	for _, c := range req.Contents {
		if c.Role != genai.RoleUser {
			continue
		}
		for _, p := range c.Parts {
			if p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(strings.Join(texts, " | "), genai.RoleModel)}, nil)
	}
}

// A runner can run a workflow node instead of an agent. A node root takes the
// user message as a *genai.Content; a function node's result is the Output
// of the event it yields, whose NodeInfo.Path is "<name>@1".
// WithYieldUserMessage makes Run yield the stored user event first.
func ExampleConfig_node() {
	shout := workflow.NewFunctionNode("shout", func(_ agent.Context, msg *genai.Content) (string, error) {
		return strings.ToUpper(msg.Parts[0].Text) + "!", nil
	}, workflow.NodeConfig{})

	r, err := runner.New(runner.Config{
		Node:              shout, // AppName defaults to the node's name.
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}, runner.WithYieldUserMessage()) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Author == "user" {
			fmt.Println("user:", ev.Content.Parts[0].Text)
			continue
		}
		fmt.Printf("%s at %s: %v\n", ev.Author, ev.NodeInfo.Path, ev.Output)
	}

	// Output:
	// user: hello
	// shout at shout@1: HELLO!
}

// RunConfig.CustomMetadata is merged into every event of the run, the stored
// user event included. An event's own key wins over the run's, and keys that
// ADK reserves for itself, such as those starting with "__adk_internal_",
// never reach the events.
func ExampleRunner_Run_customMetadata() {
	// tagger is a plugin that sets its own "source" key on the events it sees.
	tagger, err := plugin.New(plugin.Config{
		Name: "tagger",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.Author != "user" {
				if ev.CustomMetadata == nil {
					ev.CustomMetadata = map[string]any{}
				}
				ev.CustomMetadata["source"] = "agent"
			}
			return nil, nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: historyModel{}})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "support",
		Agent:             a,
		SessionService:    sessions,
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{tagger}},
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{
		"request_id":       "req-42",
		"source":           "web",
		"__adk_internal_x": "dropped",
	}}
	msg := genai.NewContentFromText("hi", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", "s1", msg, cfg) {
		if err != nil {
			log.Fatal(err)
		}
	}

	resp, err := sessions.Get(ctx, &session.GetRequest{AppName: "support", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	for ev := range resp.Session.Events().All() {
		md := ev.CustomMetadata
		fmt.Printf("%s: request_id=%v source=%v internal=%v\n", ev.Author, md["request_id"], md["source"], md["__adk_internal_x"])
	}

	// Output:
	// user: request_id=req-42 source=web internal=<nil>
	// assistant: request_id=req-42 source=agent internal=<nil>
}

// RunConfig.GetSessionConfig limits the events the runner loads from the
// session service at the start of a run. The stored history keeps every
// event; only this run's view of it, and so the model's request, is shorter.
func ExampleRunner_Run_sessionWindow() {
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: historyModel{}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "chat",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	turn := func(text string, cfg agent.RunConfig) {
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for ev, err := range r.Run(ctx, "user", "s1", msg, cfg) {
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("%s -> model saw: %s\n", text, ev.Content.Parts[0].Text)
		}
	}

	turn("one", agent.RunConfig{})
	turn("two", agent.RunConfig{})
	// Load only the two most recent events: the "two" turn and its reply.
	turn("three", agent.RunConfig{GetSessionConfig: &agent.GetSessionConfig{NumRecentEvents: 2}})

	// Output:
	// one -> model saw: one
	// two -> model saw: one | two
	// three -> model saw: two | three
}

// A client that did not see the end of a run can send the same message again
// under the same invocation ID. The runner recognizes the retry by the ID: it
// stores no second user event and runs the agent again in that invocation.
func ExampleWithInvocationID_retry() {
	a, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &cannedModel{responses: []*genai.Content{
			genai.NewContentFromText("first answer", genai.RoleModel),
			genai.NewContentFromText("second answer", genai.RoleModel),
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "chat",
		Agent:             a,
		SessionService:    sessions,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	msg := genai.NewContentFromText("What is the status?", genai.RoleUser)
	for attempt := 1; attempt <= 2; attempt++ {
		for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}, runner.WithInvocationID("inv-1")) {
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("attempt %d: %s in %s\n", attempt, ev.Content.Parts[0].Text, ev.InvocationID)
		}
	}

	resp, err := sessions.Get(ctx, &session.GetRequest{AppName: "chat", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	for ev := range resp.Session.Events().All() {
		describe(ev)
	}

	// Output:
	// attempt 1: first answer in inv-1
	// attempt 2: second answer in inv-1
	// user: What is the status?
	// assistant: first answer
	// assistant: second answer
}

// An OnEventCallback plugin sees every event before the runner stores and
// yields it. Returning an event replaces it: the runner keeps the original's
// ID, invocation ID and timestamp and takes the fields the replacement sets.
// Returning nil keeps the event, with any edits the callback made to it.
func ExampleRunner_Run_onEventPlugin() {
	redact, err := plugin.New(plugin.Config{
		Name: "redact",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.Author == "user" || ev.Content == nil {
				return nil, nil
			}
			text := strings.ReplaceAll(ev.Content.Parts[0].Text, "4111-1111", "[card]")
			return &session.Event{LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText(text, genai.RoleModel),
			}}, nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &cannedModel{responses: []*genai.Content{
			genai.NewContentFromText("Charged card 4111-1111.", genai.RoleModel),
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "shop",
		Agent:             a,
		SessionService:    sessions,
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{redact}},
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	msg := genai.NewContentFromText("Pay for my order", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print("yielded: ")
		describe(ev)
	}

	resp, err := sessions.Get(ctx, &session.GetRequest{AppName: "shop", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	for ev := range resp.Session.Events().All() {
		fmt.Print("stored: ")
		describe(ev)
	}

	// Output:
	// yielded: assistant: Charged card [card].
	// stored: user: Pay for my order
	// stored: assistant: Charged card [card].
}

// A BeforeRunCallback that returns content answers the run without running
// the agent. The runner stores the content as the agent's reply, marks it
// with the "adk:before_run_reply" metadata key, and still runs the after-run
// callbacks.
func ExampleRunner_Run_beforeRunPlugin() {
	closed, err := plugin.New(plugin.Config{
		Name: "closed",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			return genai.NewContentFromText("We are closed today.", genai.RoleModel), nil
		},
		AfterRunCallback: func(agent.InvocationContext) {
			fmt.Println("after run")
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		// The agent never runs, so the model has no response to give.
		Model: &cannedModel{},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "shop",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{closed}},
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("Can I order?", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		describe(ev)
		fmt.Println("before-run reply:", ev.CustomMetadata["adk:before_run_reply"])
	}

	// Output:
	// assistant: We are closed today.
	// before-run reply: true
	// after run
}
