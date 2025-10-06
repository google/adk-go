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
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func New(cfg Config) (agent.Agent, error) {
	beforeModel := make([]llminternal.BeforeModelCallback, 0, len(cfg.BeforeModel))
	for _, c := range cfg.BeforeModel {
		beforeModel = append(beforeModel, llminternal.BeforeModelCallback(c))
	}

	afterModel := make([]llminternal.AfterModelCallback, 0, len(cfg.AfterModel))
	for _, c := range cfg.AfterModel {
		afterModel = append(afterModel, llminternal.AfterModelCallback(c))
	}

	a := &llmAgent{
		beforeModel: beforeModel,
		model:       cfg.Model,
		afterModel:  afterModel,
		instruction: cfg.Instruction,

		State: llminternal.State{
			Model:                    cfg.Model,
			GenerateContentConfig:    cfg.GenerateContentConfig,
			Tools:                    cfg.Tools,
			DisallowTransferToParent: cfg.DisallowTransferToParent,
			DisallowTransferToPeers:  cfg.DisallowTransferToPeers,
			OutputSchema:             cfg.OutputSchema,
			IncludeContents:          cfg.IncludeContents,
			Instruction:              cfg.Instruction,
			GlobalInstruction:        cfg.GlobalInstruction,
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

	a.AgentType = agentinternal.TypeLLMAgent

	return a, nil
}

type Config struct {
	Name        string
	Description string
	SubAgents   []agent.Agent

	BeforeAgent []agent.BeforeAgentCallback
	AfterAgent  []agent.AfterAgentCallback

	GenerateContentConfig *genai.GenerateContentConfig

	// BeforeModel callbacks are executed sequentially right before a request is
	// sent to the model.
	//
	// The first callback that returns non-nil LLMResponse/error makes
	// LLMAgent **skip** the actual model call and yields the callback result
	// instead.
	//
	// This provides an opportunity to inspect, log, or modify the `LLMRequest`
	// object. It can also be used to implement caching by returning a cached
	// `LLMResponse`, which would skip the actual model call.
	BeforeModel []BeforeModelCallback
	Model       model.LLM
	// AfterModel callbacks are executed sequentially right after a response is
	// received from the model.
	//
	// The first callback that returns non-nil LLMResponse/error **replaces**
	// the actual model response/error and stops execution of the remaining
	// callbacks.
	//
	// This is the ideal place to log model responses, collect metrics on token
	// usage, or perform post-processing on the raw `LLMResponse`.
	AfterModel []AfterModelCallback

	Instruction       string
	GlobalInstruction string

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// Whether to include contents in the model request.
	// When set to 'none', the model request will not include any contents, such as
	// user messages, tool requests, etc.
	IncludeContents string

	// The input schema when agent is used as a tool.
	InputSchema *genai.Schema
	// The output schema when agent replies.
	//
	// NOTE: when this is set, agent can only reply and cannot use any tools,
	// such as function tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	// TODO: BeforeTool and AfterTool callbacks
	Tools []tool.Tool

	// OutputKey
	// Planner
	// CodeExecutor
	// Examples

	// BeforeToolCallback
	// AfterToolCallback
}

type BeforeModelCallback func(ctx agent.Context, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

type AfterModelCallback func(ctx agent.Context, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

type llmAgent struct {
	agent.Agent
	llminternal.State
	agentState

	beforeModel []llminternal.BeforeModelCallback
	model       model.LLM
	afterModel  []llminternal.AfterModelCallback
	instruction string
}

type agentState = agentinternal.State

func (a *llmAgent) run(ctx agent.Context) iter.Seq2[*session.Event, error] {
	// TODO: branch context?
	ctx = agent.NewContext(ctx, a, ctx.UserContent(), ctx.Artifacts(), ctx.Session(), ctx.Memory(), ctx.Branch())

	f := &llminternal.Flow{
		Model:                a.model,
		RequestProcessors:    llminternal.DefaultRequestProcessors,
		ResponseProcessors:   llminternal.DefaultResponseProcessors,
		BeforeModelCallbacks: a.beforeModel,
		AfterModelCallbacks:  a.afterModel,
	}

	return f.Run(ctx)
}
