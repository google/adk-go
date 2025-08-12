// Copyright 2025 Google LLC
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

package llmagent

import (
	"fmt"
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func New(cfg Config) (*llmAgent, error) {
	a := &llmAgent{
		model:       cfg.Model,
		instruction: cfg.Instruction,

		State: llminternal.State{
			Model:                    cfg.Model,
			DisallowTransferToParent: cfg.DisallowTransferToParent,
		},
	}

	baseAgent, err := agent.New(agent.Config{
		Name:        cfg.Name,
		Description: cfg.Description,
		SubAgents:   cfg.SubAgents,
		BeforeAgent: cfg.BeforeAgent,
		Run:         a.run,
		AfterAgent:  cfg.AfterAgent,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create agent: %w", err)
	}

	a.Agent = baseAgent

	return a, nil
}

type Config struct {
	Name        string
	Description string
	SubAgents   []agent.Agent

	BeforeAgent []agent.Callback
	AfterAgent  []agent.Callback

	BeforeModel []BeforeModelCallback
	Model       llm.Model
	AfterModel  []AfterModelCallback

	Instruction       string
	GlobalInstruction string

	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	IncludeContents string

	InputSchema  *genai.Schema
	OutputSchema *genai.Schema

	// TODO: BeforeTool and AfterTool callbacks
	// TODO: switch to tool.Tool. Right now it's types.Tool to reduce chages.
	Tools []types.Tool
}

type BeforeModelCallback func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error)

type AfterModelCallback func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error)

type llmAgent struct {
	agent.Agent
	llminternal.State

	model       llm.Model
	instruction string
}

func (a *llmAgent) run(ctx agent.Context) iter.Seq2[*session.Event, error] {
	req := &llm.Request{
		Contents: []*genai.Content{
			ctx.UserContent(),
		},
		GenerateConfig: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText(a.instruction, ""),
		},
	}

	return func(yield func(*session.Event, error) bool) {
		// TODO: right now it's generateStream only, we'd need to propagate this from the AgentRunConfig or equivalent.
		for resp, err := range a.model.GenerateStream(ctx, req) {
			// TODO: check if we should stop iterator on the first error from stream or continue yielding next results.
			if err != nil {
				yield(nil, err)
				return
			}

			// TODO: proper event initialization, function calls handling etc.
			ev := session.NewEvent(ctx.InvocationID())
			ev.LLMResponse = resp
			ev.Author = genai.RoleModel

			if !yield(ev, nil) {
				return
			}
		}
	}
}
