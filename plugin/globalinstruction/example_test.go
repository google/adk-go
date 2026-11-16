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

package globalinstruction_test

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
	"google.golang.org/adk/v2/plugin/globalinstruction"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/util/instructionutil"
)

// printingModel stands in for a real model: it prints the parts of the system
// instruction of each request it receives and replies "ok".
type printingModel struct{}

func (printingModel) Name() string { return "printing" }

func (printingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	for i, p := range req.Config.SystemInstruction.Parts {
		fmt.Printf("part %d: %s\n", i, p.Text)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}, nil)
	}
}

// runOnce runs one turn of an agent with instruction "Answer briefly." through
// a runner that has plugin p, in a session whose state holds state.
func runOnce(p *plugin.Plugin, state map[string]any) {
	ctx := context.Background()
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: printingModel{}, Instruction: "Answer briefly."})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", State: state})
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
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}
}

// The instruction is a template filled from session state before every model
// call. It becomes the first part of the system instruction; the agent's own
// instructions follow it.
func ExampleNew() {
	p, err := globalinstruction.New(globalinstruction.Config{
		Instruction: "You work for {company}.",
	})
	if err != nil {
		log.Fatal(err)
	}
	runOnce(p, map[string]any{"company": "Example Corp"})
	// Output:
	// part 0: You work for Example Corp.
	// part 1: Answer briefly.
	//
	// You are an agent. Your internal name is "assistant".
}

// A provider computes the instruction for each model call. Its result is used
// as given; it calls instructionutil.InjectSessionState to fill placeholders.
func ExampleNew_provider() {
	p, err := globalinstruction.New(globalinstruction.Config{
		InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
			return instructionutil.InjectSessionState(ctx, "Agent "+ctx.AgentName()+" serves {customer}.")
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	runOnce(p, map[string]any{"customer": "Ada"})
	// Output:
	// part 0: Agent assistant serves Ada.
	// part 1: Answer briefly.
	//
	// You are an agent. Your internal name is "assistant".
}
