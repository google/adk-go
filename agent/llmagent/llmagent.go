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
	"strings"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func New(cfg Config) (agent.Agent, error) {
	beforeModelCallbacks := make([]llminternal.BeforeModelCallback, 0, len(cfg.BeforeModelCallbacks))
	for _, c := range cfg.BeforeModelCallbacks {
		beforeModelCallbacks = append(beforeModelCallbacks, llminternal.BeforeModelCallback(c))
	}

	afterModelCallbacks := make([]llminternal.AfterModelCallback, 0, len(cfg.AfterModelCallbacks))
	for _, c := range cfg.AfterModelCallbacks {
		afterModelCallbacks = append(afterModelCallbacks, llminternal.AfterModelCallback(c))
	}

	a := &llmAgent{
		beforeModelCallbacks: beforeModelCallbacks,
		model:                cfg.Model,
		afterModelCallbacks:  afterModelCallbacks,
		instruction:          cfg.Instruction,
		inputSchema:          cfg.InputSchema,
		outputSchema:         cfg.OutputSchema,

		State: llminternal.State{
			Model:                    cfg.Model,
			GenerateContentConfig:    cfg.GenerateContentConfig,
			Tools:                    cfg.Tools,
			Toolsets:                 cfg.Toolsets,
			DisallowTransferToParent: cfg.DisallowTransferToParent,
			DisallowTransferToPeers:  cfg.DisallowTransferToPeers,
			InputSchema:              cfg.InputSchema,
			OutputSchema:             cfg.OutputSchema,
			IncludeContents:          cfg.IncludeContents,
			Instruction:              cfg.Instruction,
			GlobalInstruction:        cfg.GlobalInstruction,
			OutputKey:                cfg.OutputKey,
		},
	}

	baseAgent, err := agent.New(agent.Config{
		Name:                 cfg.Name,
		Description:          cfg.Description,
		SubAgents:            cfg.SubAgents,
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		Run:                  a.run,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
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

	BeforeAgentCallbacks []agent.BeforeAgentCallback
	AfterAgentCallbacks  []agent.AfterAgentCallback

	GenerateContentConfig *genai.GenerateContentConfig

	// BeforeModelCallbacks are executed sequentially right before a request is
	// sent to the model.
	//
	// The first callback that returns non-nil LLMResponse/error makes
	// LLMAgent **skip** the actual model call and yields the callback result
	// instead.
	//
	// This provides an opportunity to inspect, log, or modify the `LLMRequest`
	// object. It can also be used to implement caching by returning a cached
	// `LLMResponse`, which would skip the actual model call.
	BeforeModelCallbacks []BeforeModelCallback
	Model                model.LLM
	// AfterModelCallbacks are executed sequentially right after a response is
	// received from the model.
	//
	// The first callback that returns non-nil LLMResponse/error **replaces**
	// the actual model response/error and stops execution of the remaining
	// callbacks.
	//
	// This is the ideal place to log model responses, collect metrics on token
	// usage, or perform post-processing on the raw `LLMResponse`.
	AfterModelCallbacks []AfterModelCallback

	Instruction       string
	GlobalInstruction string

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// Whether to include contents in the model request.
	// When set to 'none', the model request will not include any contents, such as
	// user messages, tool requests, etc.
	IncludeContents string

	// TODO(ngeorgy): consider to switch to jsonschema for input and output schema.
	// The input schema when agent is used as a tool.
	InputSchema *genai.Schema
	// The output schema when agent replies.
	//
	// NOTE: when this is set, agent can only reply and cannot use any tools,
	// such as function tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	// TODO: BeforeTool and AfterTool callbacks
	Tools []tool.Tool
	// Toolsets will be used by llmagent to extract tools and pass to the
	// underlying LLM.
	Toolsets []tool.Set

	// OutputKey is an optional parameter to specify the key in session state for the agent output.
	//
	// Typical uses cases are:
	// - Extracts agent reply for later use, such as in tools, callbacks, etc.
	// - Connects agents to coordinate with each other.
	OutputKey string
	// Planner
	// CodeExecutor
	// Examples

	// BeforeToolCallback
	// AfterToolCallback
}

type BeforeModelCallback func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

type AfterModelCallback func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

type llmAgent struct {
	agent.Agent
	llminternal.State
	agentState

	beforeModelCallbacks []llminternal.BeforeModelCallback
	model                model.LLM
	afterModelCallbacks  []llminternal.AfterModelCallback
	instruction          string

	inputSchema  *genai.Schema
	outputSchema *genai.Schema
}

type agentState = agentinternal.State

func (a *llmAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	// TODO: branch context?
	ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:   ctx.Artifacts(),
		Memory:      ctx.Memory(),
		Session:     ctx.Session(),
		Branch:      ctx.Branch(),
		Agent:       a,
		UserContent: ctx.UserContent(),
		RunConfig:   ctx.RunConfig(),
	})

	f := &llminternal.Flow{
		Model:                a.model,
		RequestProcessors:    llminternal.DefaultRequestProcessors,
		ResponseProcessors:   llminternal.DefaultResponseProcessors,
		BeforeModelCallbacks: a.beforeModelCallbacks,
		AfterModelCallbacks:  a.afterModelCallbacks,
	}

	return func(yield func(*session.Event, error) bool) {
		for ev, err := range f.Run(ctx) {
			a.maybeSaveOutputToState(ev)
			if !yield(ev, err) {
				return
			}
		}
	}
}

// maybeSaveOutputToState saves the model output to state if needed. skip if the event
// was authored by some other agent (e.g. current agent transferred to another agent)
func (a *llmAgent) maybeSaveOutputToState(event *session.Event) {
	if event == nil {
		return
	}
	if event.Author != a.Name() {
		// TODO: log "Skipping output save for agent %s: event authored by %s"
		return
	}
	if a.OutputKey != "" && !event.Partial && event.Content != nil && len(event.Content.Parts) > 0 {
		var sb strings.Builder
		for _, part := range event.Content.Parts {
			if part.Text != "" && !part.Thought {
				sb.WriteString(part.Text)
			}
		}
		result := sb.String()

		// TODO: add output schema validation and unmarshalling
		if a.OutputSchema != nil {
			// If the result from the final chunk is just whitespace or empty,
			// it means this is an empty final chunk of a stream.
			// Do not attempt to parse it as JSON.
			if strings.TrimSpace(result) == "" {
				return
			}
		}

		if event.Actions.StateDelta == nil {
			event.Actions.StateDelta = make(map[string]any)
		}

		event.Actions.StateDelta[a.OutputKey] = result
	}
}
