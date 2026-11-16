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
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// codeWritingModel stands in for a real model: call n replies with
// replies[n] as text, and the contents of every request are kept.
type codeWritingModel struct {
	replies  []string
	requests [][]*genai.Content
}

func (*codeWritingModel) Name() string { return "code-writer" }

func (m *codeWritingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	n := len(m.requests)
	m.requests = append(m.requests, slices.Clone(req.Contents))
	return func(yield func(*model.LLMResponse, error) bool) {
		if n >= len(m.replies) {
			yield(nil, errors.New("no reply scripted"))
			return
		}
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(m.replies[n], genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// cannedSandbox stands in for a sandbox that runs Python: it knows the
// output of the programs in outputs and reports any other program as failed.
// A real executor runs in.Code.
type cannedSandbox struct{ outputs map[string]string }

func (cannedSandbox) Settings() codeexecutor.Settings { return codeexecutor.Settings{} }

func (s cannedSandbox) ExecuteCode(ctx context.Context, _ agent.InvocationContext, in *codeexecutor.Input) (*codeexecutor.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("execute code: %w", err)
	}
	out, ok := s.outputs[in.Code]
	if !ok {
		return &codeexecutor.Result{Stderr: "NameError\n"}, nil
	}
	return &codeexecutor.Result{Stdout: out}, nil
}

// An agent with a custom code executor runs the first code block of a model
// response itself. It yields the code and the result as events, in place of
// the response, and calls the model again; the second request holds the code
// and the result as fenced text. The default settings add no instruction
// about fences, so the agent's instruction asks for a python block. The
// runner has an artifact service, which running the executor needs.
func ExampleConfig_codeExecutor() {
	ctx := context.Background()
	llm := &codeWritingModel{replies: []string{
		"I will compute it.\n```python\nprint(6 * 7)\n```",
		"6 times 7 is 42.",
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:         "calculator",
		Model:        llm,
		Instruction:  "To compute something, write Python in a ```python block.",
		CodeExecutor: cannedSandbox{outputs: map[string]string{"print(6 * 7)": "42\n"}},
	})
	if err != nil {
		log.Fatal(err)
	}

	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:         "app",
		Agent:           a,
		SessionService:  sessions,
		ArtifactService: artifact.InMemoryService(),
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("What is 6 times 7?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
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

	fmt.Println("model calls:", len(llm.requests))
	if len(llm.requests) < 2 {
		return
	}
	fmt.Println("second request:")
	for _, c := range llm.requests[1] {
		for _, p := range c.Parts {
			fmt.Printf("  %s: %q\n", c.Role, p.Text)
		}
	}
	// Output:
	// calculator: text "I will compute it.\n"
	// calculator: code "print(6 * 7)"
	// calculator: result OUTCOME_OK "Code execution result:\n42\n\n"
	// calculator: text "6 times 7 is 42."
	// model calls: 2
	// second request:
	//   user: "What is 6 times 7?"
	//   model: "I will compute it.\n"
	//   model: "```tool_code\nprint(6 * 7)\n```"
	//   user: "```tool_output\nCode execution result:\n42\n\n\n```"
}
