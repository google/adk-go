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
	"errors"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// docToolModel stands in for a real model. Its first answer calls the tool
// "act" and every later answer is the text "done".
type docToolModel struct{ calls int }

func (*docToolModel) Name() string { return "scripted" }

func (m *docToolModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	content := genai.NewContentFromText("done", genai.RoleModel)
	if m.calls == 1 {
		content = genai.NewContentFromFunctionCall("act", map[string]any{}, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// docRunTool runs an agent whose model calls one tool, "act", which runs
// handler with the tool's Context. The runner gets memories as its memory
// service when it is not nil. docRunTool returns the events of the run.
func docRunTool(memories memory.Service, cfg agent.RunConfig, handler func(agent.Context)) []*session.Event {
	act, err := functiontool.New(functiontool.Config{Name: "act", Description: "Acts."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			handler(ctx)
			return map[string]any{}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "helper", Model: &docToolModel{}, Tools: []tool.Tool{act}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		MemoryService:     memories,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	var events []*session.Event
	msg := genai.NewContentFromText("My favorite color is teal.", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "s1", msg, cfg) {
		if err != nil {
			log.Fatal(err)
		}
		events = append(events, ev)
	}
	return events
}

// The name "user" is reserved for the end user's input in session history,
// so New refuses it. Every constructor that builds its agent through New,
// such as llmagent.New and sequentialagent.New, refuses it too. The check is
// case-sensitive.
func ExampleNew_reservedName() {
	_, err := agent.New(agent.Config{Name: "user"})
	fmt.Println(err)
	fmt.Println(errors.Is(err, agent.ErrInvalidAgentConfig))

	_, err = llmagent.New(llmagent.Config{Name: "user"})
	fmt.Println(errors.Is(err, agent.ErrInvalidAgentConfig))

	_, err = agent.New(agent.Config{Name: "User"})
	fmt.Println(err)
	// Output:
	// invalid agent configuration: agent name cannot be `user`, it is reserved for the end user's input
	// true
	// true
	// <nil>
}

// FindAgent searches an agent and its descendants, depth first, and returns
// the agent values that were built, so a lookup of a workflow agent returns
// that workflow agent. FindSubAgent leaves out the agent it is called on.
func ExampleAgent_findAgent() {
	newLeaf := func(name string) agent.Agent {
		a, err := agent.New(agent.Config{Name: name})
		if err != nil {
			log.Fatal(err)
		}
		return a
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
		Name:      "pipeline",
		SubAgents: []agent.Agent{newLeaf("draft"), newLeaf("review")},
	}})
	if err != nil {
		log.Fatal(err)
	}
	root, err := agent.New(agent.Config{Name: "root", SubAgents: []agent.Agent{pipeline}})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(root.FindAgent("pipeline") == pipeline)
	fmt.Println(root.FindAgent("review").Name())
	fmt.Println(root.FindAgent("root") == root)
	fmt.Println(root.FindSubAgent("root") == nil)
	fmt.Println(root.FindAgent("missing") == nil)
	// Output:
	// true
	// review
	// true
	// true
	// true
}

// A tool adds the session it runs in to memory with AddSessionToMemory, even
// though a tool's Context reports no session. Without a memory service the
// call returns an error and does nothing.
func ExampleAddSessionToMemory() {
	memories := memory.InMemoryService()
	docRunTool(memories, agent.RunConfig{}, func(ctx agent.Context) {
		fmt.Println("with a memory service:", agent.AddSessionToMemory(ctx))
	})
	resp, err := memories.SearchMemory(context.Background(), &memory.SearchRequest{
		AppName: "app", UserID: "user", Query: "favorite color",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("found:", resp.Memories[0].Content.Parts[0].Text)

	docRunTool(nil, agent.RunConfig{}, func(ctx agent.Context) {
		fmt.Println("without one:", agent.AddSessionToMemory(ctx))
	})
	// Output:
	// with a memory service: <nil>
	// found: My favorite color is teal.
	// without one: cannot add session to memory: memory service is not available
}

// AddMemory writes memories directly, which only a service that implements
// memory.MemoryAdder accepts. The in-memory service does not, so the error
// matches memory.ErrAddMemoryUnsupported and errors.ErrUnsupported.
func ExampleAddMemory() {
	docRunTool(memory.InMemoryService(), agent.RunConfig{}, func(ctx agent.Context) {
		err := agent.AddMemory(ctx, []*memory.Entry{{
			Content: genai.NewContentFromText("Prefers teal.", genai.RoleUser),
		}}, nil)
		fmt.Println(errors.Is(err, memory.ErrAddMemoryUnsupported))
		fmt.Println(errors.Is(err, errors.ErrUnsupported))
	})
	// Output:
	// true
	// true
}

// A tool or callback reads the run's RunConfig.CustomMetadata with
// CustomMetadata. Keys that ADK reserves for its own metadata are left out,
// and the map is a copy.
func ExampleCustomMetadata() {
	cfg := agent.RunConfig{CustomMetadata: &map[string]any{
		"tenant":                 "acme",
		"__adk_internal_counter": 1,
	}}
	docRunTool(nil, cfg, func(ctx agent.Context) {
		md := agent.CustomMetadata(ctx)
		fmt.Println(md)
		md["tenant"] = "changed"
		fmt.Println(agent.CustomMetadata(ctx)["tenant"])
	})
	// Output:
	// map[tenant:acme]
	// acme
}

// A tool asks the UI host to render a widget beside its function response
// event. A second widget with the same ID in that event is refused.
func ExampleRenderUIWidget() {
	widget := session.UIWidget{
		ID:       "status_dashboard",
		Provider: "mcp",
		Payload:  map[string]any{"resource_uri": "ui://analytics/status"},
	}
	events := docRunTool(nil, agent.RunConfig{}, func(ctx agent.Context) {
		fmt.Println(agent.RenderUIWidget(ctx, widget))
		fmt.Println(agent.RenderUIWidget(ctx, widget))
	})
	for _, ev := range events {
		for _, w := range ev.Actions.RenderUIWidgets {
			fmt.Printf("event of %s carries widget %s from %s: %v\n", ev.Author, w.ID, w.Provider, w.Payload)
		}
	}
	// Output:
	// <nil>
	// UI widget with ID 'status_dashboard' already exists in the current event actions.
	// event of helper carries widget status_dashboard from mcp: map[resource_uri:ui://analytics/status]
}

// docPlainContext is a Context of the caller's own that does not implement
// agent.UIWidgetRenderer.
type docPlainContext struct{ agent.Context }

// RenderUIWidget returns ErrUnsupportedContext for a context that cannot
// render widgets. It matches errors.ErrUnsupported, as the errors of adk-go's
// other optional-interface helpers do.
func ExampleErrUnsupportedContext() {
	err := agent.RenderUIWidget(docPlainContext{}, session.UIWidget{ID: "w"})
	fmt.Println(errors.Is(err, agent.ErrUnsupportedContext))
	fmt.Println(errors.Is(err, errors.ErrUnsupported))
	// Output:
	// true
	// true
}

// docRunWorkflow runs w as the root agent of a runner, in a new in-memory
// session with one user message, and returns the run's first error.
func docRunWorkflow(w *workflow.Workflow) error {
	ctx := context.Background()
	a, err := agent.New(agent.Config{Name: w.Name(), Run: w.Run})
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return err
	}
	msg := genai.NewContentFromText("go", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

// A node whose NodeConfig grants retries reads which attempt it is on with
// AttemptCount: 1 on the first run, one more on each retry.
func ExampleAttemptCount() {
	flaky := workflow.NewFunctionNode("flaky", func(ctx agent.Context, _ any) (string, error) {
		attempt := agent.AttemptCount(ctx)
		fmt.Println("attempt", attempt)
		if attempt < 3 {
			return "", errors.New("service unavailable")
		}
		return "done", nil
	}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 3}})
	w, err := workflow.New("retry_demo", workflow.Chain(workflow.Start, flaky))
	if err != nil {
		log.Fatal(err)
	}
	if err := docRunWorkflow(w); err != nil {
		fmt.Println("run failed:", err)
	}
	// Output:
	// attempt 1
	// attempt 2
	// attempt 3
}
