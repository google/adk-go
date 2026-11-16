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

package codeexecutor_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// geminiStub stands in for a Gemini model that runs code in its own
// code-execution tool: it records the tools of each request and replies with
// the code it ran, the result and a text answer.
type geminiStub struct {
	name  string
	tools [][]*genai.Tool
}

func (m *geminiStub) Name() string { return m.name }

func (m *geminiStub) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	if req.Config != nil {
		m.tools = append(m.tools, req.Config.Tools)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				genai.NewPartFromExecutableCode("print(6 * 7)", genai.LanguagePython),
				genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "42\n"),
				genai.NewPartFromText("6 times 7 is 42."),
			}},
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// runBuiltIn runs one user message through an agent with the built-in
// executor and prints the parts of each event, or the error that ends the run.
func runBuiltIn(llm model.LLM) {
	ctx := context.Background()
	a, err := llmagent.New(llmagent.Config{
		Name:         "calculator",
		Model:        llm,
		CodeExecutor: codeexecutor.NewBuiltInExecutor(),
	})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("What is 6 times 7?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.ExecutableCode != nil:
				fmt.Printf("%s: code %q\n", ev.Author, p.ExecutableCode.Code)
			case p.CodeExecutionResult != nil:
				fmt.Printf("%s: result %s %q\n", ev.Author, p.CodeExecutionResult.Outcome, p.CodeExecutionResult.Output)
			default:
				fmt.Printf("%s: text %q\n", ev.Author, p.Text)
			}
		}
	}
}

// With the built-in executor the agent adds Gemini's code-execution tool to
// each request; the model runs the code and returns it with its result, and
// the agent passes the response on without running anything. A model name
// that is not a Gemini name fails the run before the model is called.
func ExampleNewBuiltInExecutor() {
	gemini := &geminiStub{name: "gemini-2.5-flash"}
	runBuiltIn(gemini)
	for _, tool := range gemini.tools[0] {
		fmt.Printf("request tool: code execution %t\n", tool.CodeExecution != nil)
	}

	other := &geminiStub{name: "claude-sonnet-4"}
	runBuiltIn(other)
	fmt.Println("calls to the other model:", len(other.tools))

	// Output:
	// calculator: code "print(6 * 7)"
	// calculator: result OUTCOME_OK "42\n"
	// calculator: text "6 times 7 is 42."
	// request tool: code execution true
	// error: Gemini code execution tool is not supported for model claude-sonnet-4
	// calls to the other model: 0
}
