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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// New is a constructor for LLMAgent.
func New(cfg Config) (agent.Agent, error) {
	beforeModelCallbacks := make([]llminternal.BeforeModelCallback, 0, len(cfg.BeforeModelCallbacks))
	for _, c := range cfg.BeforeModelCallbacks {
		beforeModelCallbacks = append(beforeModelCallbacks, llminternal.BeforeModelCallback(c))
	}

	afterModelCallbacks := make([]llminternal.AfterModelCallback, 0, len(cfg.AfterModelCallbacks))
	for _, c := range cfg.AfterModelCallbacks {
		afterModelCallbacks = append(afterModelCallbacks, llminternal.AfterModelCallback(c))
	}

	beforeToolCallbacks := make([]llminternal.BeforeToolCallback, 0, len(cfg.BeforeToolCallbacks))
	for _, c := range cfg.BeforeToolCallbacks {
		beforeToolCallbacks = append(beforeToolCallbacks, llminternal.BeforeToolCallback(c))
	}

	afterToolCallbacks := make([]llminternal.AfterToolCallback, 0, len(cfg.AfterToolCallbacks))
	for _, c := range cfg.AfterToolCallbacks {
		afterToolCallbacks = append(afterToolCallbacks, llminternal.AfterToolCallback(c))
	}

	a := &llmAgent{
		beforeModelCallbacks: beforeModelCallbacks,
		model:                cfg.Model,
		afterModelCallbacks:  afterModelCallbacks,
		beforeToolCallbacks:  beforeToolCallbacks,
		afterToolCallbacks:   afterToolCallbacks,
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
			// TODO: internal type for includeContents
			IncludeContents:           string(cfg.IncludeContents),
			Instruction:               cfg.Instruction,
			InstructionProvider:       llminternal.InstructionProvider(cfg.InstructionProvider),
			GlobalInstruction:         cfg.GlobalInstruction,
			GlobalInstructionProvider: llminternal.InstructionProvider(cfg.GlobalInstructionProvider),
			OutputKey:                 cfg.OutputKey,
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
	a.Config = cfg

	return a, nil
}

// Config of the LLMAgent.
type Config struct {
	// Name must be a non-empty string, unique within the agent tree.
	// Agent name cannot be "user", since it's reserved for end-user's input.
	Name string
	// Description of the agent's capability.
	//
	// LLM uses this to determine whether to delegate control to the agent.
	// One-line description is enough and preferred.
	Description string
	// SubAgents are the child agents that this agent can delegate tasks to.
	// ADK will automatically set a parent of each sub-agent to this agent to
	// allow agent transferring across the tree.
	SubAgents []agent.Agent

	// BeforeAgentCallbacks is a list of callbacks that are called sequentially
	// before the agent starts its run.
	//
	// If any callback returns non-nil content or error, then the agent run and
	// the remaining callbacks will be skipped, and a new event will be created
	// from the content or error of that callback.
	BeforeAgentCallbacks []agent.BeforeAgentCallback
	// AfterAgentCallbacks is a list of callbacks that are called sequentially
	// after the agent has completed its run.
	//
	// If any callback returns non-nil content or error, then a new event will be
	// created from the content or error of that callback and the remaining
	// callbacks will be skipped.
	AfterAgentCallbacks []agent.AfterAgentCallback

	// GenerateContentConfig is for the additional content generation
	// configuration.
	//
	// NOTE: not all fields are usable, e.g. tools must be configured via
	// `tools`.
	//
	// For example: use this config to adjust model temperature, configure
	// safety settings, etc.
	GenerateContentConfig *genai.GenerateContentConfig

	// BeforeModelCallbacks will be called in the order they are provided until
	// there's a callback that returns a non-nil LLMResponse or error. Then
	// actual LLM call is skipped, and the returned response/error is used.
	//
	// This provides an opportunity to inspect, log, or modify the `LLMRequest`
	// object. It can also be used to implement caching by returning a cached
	// `LLMResponse`, which would skip the actual model call.
	BeforeModelCallbacks []BeforeModelCallback
	// Model that is used by the agent.
	Model model.LLM
	// AfterModelCallbacks will be called in the order they are provided until
	// there's a callback that returns a non-nil LLMResponse or error. Then
	// actual LLM response is replaced with the returned response/error.
	//
	// This is the ideal place to log model responses, collect metrics on token
	// usage, or perform post-processing on the raw `LLMResponse`.
	AfterModelCallbacks []AfterModelCallback

	// Instruction is set for the LLM model guiding the agent's behavior.
	//
	// The string is treated as a template:
	//  - There can be placeholders like {key_name} that will be resolved by ADK
	//    at runtime using session state and context.
	//  - key_name must match "^[a-zA-Z_][a-zA-Z0-9_]*$", otherwise it will be
	//    treated as a literal.
	//  - {artifact.key_name} can be used to insert the text content of the
	//    artifact named key_name.
	//
	// If the state variable or artifact does not exist, the agent will raise an
	// error. If you want to ignore the error, you can append a ? to the
	// variable name as in {var?} to make it optional.
	//
	Instruction string
	// InstructionProvider allows to create instructions dynamically based on
	// the agent context.
	//
	// It takes over the Instruction field if both are set.
	InstructionProvider InstructionProvider

	// GlobalInstruction is the instruction for all agents in the entire
	// agent tree.
	//
	// The string is treated as a template:
	//  - There can be placeholders like {key_name} that will be resolved by ADK
	//    at runtime using session state and context.
	//  - key_name must match "^[a-zA-Z_][a-zA-Z0-9_]*$", otherwise it will be
	//    treated as a literal.
	//  - {artifact.key_name} can be used to insert the text content of the
	//    artifact named key_name.
	//
	// If the state variable or artifact does not exist, the agent will raise an
	// error. If you want to ignore the error, you can append a ? to the
	// variable name as in {var?} to make it optional.
	//
	// ONLY the GlobalInstruction in the root agent will take effect.
	//
	// For example: GlobalInstruction can make all agents have a stable identity
	// or personality.
	GlobalInstruction string
	// GlobalInstructionProvider allows to create global instructions
	// dynamically based on the agent context.
	//
	// It takes over the GlobalInstruction field if both are set.
	GlobalInstructionProvider InstructionProvider

	// DisallowTransferToParent prevents transferring to parent agent if LLM
	// decides to.
	DisallowTransferToParent bool
	// DisallowTransferToPeers prevents transferring to peer agents.
	DisallowTransferToPeers bool

	// Whether to include contents (conversation history) in the model request.
	IncludeContents IncludeContents

	// TODO(ngeorgy): consider to switch to jsonschema for input and output schema.
	// The input schema when agent is used as a tool.
	InputSchema *genai.Schema
	// The output schema when agent replies.
	//
	// NOTE: when this is set, agent can only reply and cannot use any tools,
	// such as function tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	// Callbacks are executed in the order they are provided.
	// The execution of the callback chain stops at the first callback that returns a non-nil
	// response
	//   - If a callback returns (map[string]any, nil), it will be used directly as the result of
	//     the tool call. Subsequent [BeforeToolCallback]s in the list are skipped.
	//   - If a callback returns (nil, error), the tool execution is aborted, and the returned
	//     error is propagated. Subsequent [BeforeToolCallback]s are skipped.
	//   - If a callback returns (nil, nil), the execution continues to the next [BeforeToolCallback]
	//     in the sequence.
	BeforeToolCallbacks []BeforeToolCallback
	// Tools available to the agent.
	Tools []tool.Tool
	// Callbacks are executed in the order they are provided.
	// The execution of the callback chain stops at the first callback that returns a non-nil
	// response.
	//   - If a callback returns (map[string]any, nil), it will replace the result returned by
	//     the tool's Run method. Subsequent [AfterToolCallback]s in the list are skipped.
	//   - If a callback returns (nil, error), this error indicates a failure within the
	//     callback itself, and will be propagated. Subsequent [AfterToolCallback]s are skipped.
	//   - If a callback returns (nil, nil), the execution continues to the next [AfterToolCallback]
	//     in the sequence.
	AfterToolCallbacks []AfterToolCallback
	// Toolsets will be used by llmagent to extract tools and pass to the
	// underlying LLM.
	Toolsets []tool.Toolset

	// OutputKey is an optional parameter to specify the key in session state for the agent output.
	//
	// Typical uses cases are:
	// - Extracts agent reply for later use, such as in tools, callbacks, etc.
	// - Connects agents to coordinate with each other.
	OutputKey string
}

// BeforeModelCallback that is called before sending a request to the model.
//
// If it returns non-nil LLMResponse or error, the actual model call is skipped
// and the returned response/error is used.
type BeforeModelCallback func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

// AfterModelCallback that is called after receiving a response from the model.
//
// If it returns non-nil LLMResponse or error, the actual model response/error
// is replaced with the returned response/error.
type AfterModelCallback func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

// BeforeToolCallback is a function type executed before a tool's Run method is invoked.
//
// Parameters:
//   - ctx: The tool.Context for the current tool execution.
//   - tool: The tool.Tool instance that is about to be executed.
//   - args: The original arguments provided to the tool.
type BeforeToolCallback func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error)

// AfterToolCallback is a function type executed after a tool's Run method has completed,
// regardless of whether the tool returned a result or an error.
//
// Parameters:
//   - ctx:    The tool.Context for the tool execution.
//   - tool:   The tool.Tool instance that was executed.
//   - args:   The arguments originally passed to the tool.
//   - result: The result returned by the tool's Run method.
//   - err:    The error returned by the tool's Run method.
type AfterToolCallback func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error)

// IncludeContents controls what parts of prior conversation history is received by llmagent.
type IncludeContents string

const (
	// IncludeContentsNone makes the llmagent operate solely on its current turn (latest user input + any following agent events).
	IncludeContentsNone IncludeContents = "none"
	// IncludeContentsDefault is enabled by default. The llmagent receives the relevant conversation history.
	IncludeContentsDefault IncludeContents = "default"
)

type llmAgent struct {
	agent.Agent
	llminternal.State
	agentState

	beforeModelCallbacks []llminternal.BeforeModelCallback
	model                model.LLM
	afterModelCallbacks  []llminternal.AfterModelCallback
	instruction          string

	beforeToolCallbacks []llminternal.BeforeToolCallback
	afterToolCallbacks  []llminternal.AfterToolCallback

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
		BeforeToolCallbacks:  a.beforeToolCallbacks,
		AfterToolCallbacks:   a.afterToolCallbacks,
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

// InstructionProvider allows to create instructions dynamically. It is called
// on each agent invocation.
//
// NOTE: when InstructionProvider is used, ADK will NOT inject session state
// placeholders into the instruction. You can use
// util/instructionutil.InjectSessionState() helper if this functionality is needed.
type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)
