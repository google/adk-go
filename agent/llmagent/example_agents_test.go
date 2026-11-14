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
	"errors"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// docModel stands in for a real model. It answers each request with the next
// content of its script.
type docModel struct {
	name   string
	script []*genai.Content
	turn   int
}

func (m *docModel) Name() string { return m.name }

func (m *docModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	content := m.script[m.turn]
	m.turn++
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// docToolset lists fixed tools, or fails when err is set.
type docToolset struct {
	name  string
	tools []tool.Tool
	err   error
}

func (s docToolset) Name() string { return s.name }

func (s docToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return s.tools, s.err }

func docTool(name string) tool.Tool {
	t, err := functiontool.New(functiontool.Config{Name: name, Description: "Does " + name + "."},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{}, nil })
	if err != nil {
		log.Fatal(err)
	}
	return t
}

// An agent without a model uses the model of its nearest LLM-agent ancestor
// and skips the workflow agents in between. An agent with no such ancestor
// uses the process-wide default model, which SetDefaultModel changes.
func ExampleCanonicalModel() {
	writer, err := llmagent.New(llmagent.Config{Name: "writer"})
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
		Name:      "pipeline",
		SubAgents: []agent.Agent{writer},
	}})
	if err != nil {
		log.Fatal(err)
	}
	_, err = llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Model:     &docModel{name: "coordinator-model"},
		SubAgents: []agent.Agent{pipeline},
	})
	if err != nil {
		log.Fatal(err)
	}
	m, err := llmagent.CanonicalModel(writer)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("writer:", m.Name())

	loner, err := llmagent.New(llmagent.Config{Name: "loner"})
	if err != nil {
		log.Fatal(err)
	}
	if err := llmagent.SetDefaultModel(&docModel{name: "house-model"}); err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := llmagent.SetDefaultModelByName(llmagent.DefaultModel); err != nil {
			log.Fatal(err)
		}
	}()
	m, err = llmagent.CanonicalModel(loner)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("loner:", m.Name())
	// Output:
	// writer: coordinator-model
	// loner: house-model
}

// A default given by name resolves through the model registry. Package
// model/gemini registers the Gemini names when it is imported, and resolving
// a name creates no client, so this needs no credentials.
func ExampleSetDefaultModelByName() {
	if err := llmagent.SetDefaultModelByName("gemini-2.5-flash"); err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := llmagent.SetDefaultModelByName(llmagent.DefaultModel); err != nil {
			log.Fatal(err)
		}
	}()
	a, err := llmagent.New(llmagent.Config{Name: "assistant"})
	if err != nil {
		log.Fatal(err)
	}
	m, err := llmagent.CanonicalModel(a)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(m.Name(), gemini.IsGeminiConnector(m))
	fmt.Println(llmagent.SetDefaultModelByName(""))
	// Output:
	// gemini-2.5-flash true
	// the default model must be a non-empty name
}

// The agent offers Tools first, then the tools of each toolset in order.
// tool.AsToolset puts a plain tool between toolsets. A toolset whose listing
// fails is skipped.
func ExampleCanonicalTools() {
	a, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Tools: []tool.Tool{docTool("search")},
		Toolsets: []tool.Toolset{
			docToolset{name: "files", tools: []tool.Tool{docTool("read_file"), docTool("write_file")}},
			tool.AsToolset(docTool("summarize")),
			docToolset{name: "broken", err: errors.New("server unreachable")},
			docToolset{name: "mail", tools: []tool.Tool{docTool("send_mail")}},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	tools, err := llmagent.CanonicalTools(nil, a)
	if err != nil {
		log.Fatal(err)
	}
	for _, t := range tools {
		fmt.Println(t.Name())
	}
	// Output:
	// search
	// read_file
	// write_file
	// summarize
	// send_mail
}

// New refuses a GenerateContentConfig that sets Tools, SystemInstruction or
// HTTPOptions.BaseURL, because each has its own place: Config.Tools,
// Config.Instruction, and the model or its client. Other fields, such as
// Temperature, are accepted.
func ExampleNew_generateContentConfig() {
	configs := []*genai.GenerateContentConfig{
		{Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}}},
		{SystemInstruction: genai.NewContentFromText("Be brief.", genai.RoleUser)},
		{HTTPOptions: &genai.HTTPOptions{BaseURL: "https://proxy.example.com"}},
		{Temperature: genai.Ptr[float32](0.2)},
	}
	for _, cfg := range configs {
		_, err := llmagent.New(llmagent.Config{Name: "assistant", GenerateContentConfig: cfg})
		fmt.Println(errors.Is(err, agent.ErrInvalidAgentConfig), err)
	}
	// Output:
	// true invalid agent configuration: all tools must be set via llmagent.Config.Tools, not via llmagent.Config.GenerateContentConfig.Tools
	// true invalid agent configuration: system instruction must be set via llmagent.Config.Instruction, not via llmagent.Config.GenerateContentConfig.SystemInstruction
	// true invalid agent configuration: base URL is a transport setting and must be set on the model or its client, not via llmagent.Config.GenerateContentConfig.HTTPOptions.BaseURL
	// false <nil>
}

// OutputKey stores the agent's text in session state. Text the model gives
// before a tool call is kept: the stored value joins the text of every
// non-partial event of the agent in the run.
func ExampleConfig_outputKey() {
	llm := &docModel{name: "scripted", script: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{
			genai.NewPartFromText("Let me check the forecast. "),
			genai.NewPartFromFunctionCall("get_forecast", map[string]any{}),
		}},
		genai.NewContentFromText("It will be sunny.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:      "forecaster",
		Model:     llm,
		Tools:     []tool.Tool{docTool("get_forecast")},
		OutputKey: "forecast",
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions, AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("Will it rain?", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}
	got, err := sessions.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	v, err := got.Session.State().Get("forecast")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%q\n", v)
	// Output:
	// "Let me check the forecast. It will be sunny."
}
