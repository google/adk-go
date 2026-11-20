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

// Package llmagent provides an LLM-based agent.
// LLM agents use large language models to perform tasks based on instructions, user input,
// deciding on actions to take, and executing actions using available tools or
// delegating to sub agents.
//
// # Model requests and responses
//
// Before each model call, the agent adds the label "adk_agent_name", set to
// its own name, to the request's GenerateContentConfig.Labels, so that billing
// reports can be split per agent. It keeps a label already present under that
// key, for example one set through [agent.RunConfig.Labels] or by a
// BeforeModelCallback. The connector in model/gemini removes every label from
// what it sends to the Gemini Developer API, which does not accept labels; the
// connectors in model/anthropicmodel and model/openaimodel drop this label.
//
// A completed model response that finished with [genai.FinishReasonStop] but
// holds no part a caller can act on (no parts at all, only thought parts, or
// only text that is whitespace) becomes an error event, which ends the agent's
// turn. Its ErrorCode is "MODEL_RETURNED_NO_CONTENT", and its ErrorMessage, if
// the response had none, says whether the response had no parts or only
// thought or whitespace parts. The check runs after the AfterModelCallbacks,
// on the response a callback put in place of the model's, and leaves a
// response that already has an ErrorCode as it is. Under
// [agent.StreamingModeSSE], a response without parts is exempt, because one
// ends every stream.
//
// A thought-only response that did not finish with STOP is not an error: the
// agent calls the model again after it, up to ten thought-only responses in a
// row, and then ends its turn on the last one.
package llmagent

import (
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// New is a constructor for LLMAgent. It returns an error if both
// Config.OutputSchema and Config.OutputJsonSchema are set.
func New(cfg Config) (agent.Agent, error) {
	if cfg.OutputSchema != nil && cfg.OutputJsonSchema != nil {
		return nil, fmt.Errorf("LlmAgent %q: OutputSchema and OutputJsonSchema are mutually exclusive; set at most one", cfg.Name)
	}
	beforeModelCallbacks := make([]llminternal.BeforeModelCallback, 0, len(cfg.BeforeModelCallbacks))
	for _, c := range cfg.BeforeModelCallbacks {
		beforeModelCallbacks = append(beforeModelCallbacks, llminternal.BeforeModelCallback(c))
	}

	afterModelCallbacks := make([]llminternal.AfterModelCallback, 0, len(cfg.AfterModelCallbacks))
	for _, c := range cfg.AfterModelCallbacks {
		afterModelCallbacks = append(afterModelCallbacks, llminternal.AfterModelCallback(c))
	}

	onModelErrorCallbacks := make([]llminternal.OnModelErrorCallback, 0, len(cfg.OnModelErrorCallbacks))
	for _, c := range cfg.OnModelErrorCallbacks {
		onModelErrorCallbacks = append(onModelErrorCallbacks, llminternal.OnModelErrorCallback(c))
	}

	beforeToolCallbacks := make([]llminternal.BeforeToolCallback, 0, len(cfg.BeforeToolCallbacks))
	for _, c := range cfg.BeforeToolCallbacks {
		beforeToolCallbacks = append(beforeToolCallbacks, llminternal.BeforeToolCallback(c))
	}

	afterToolCallbacks := make([]llminternal.AfterToolCallback, 0, len(cfg.AfterToolCallbacks))
	for _, c := range cfg.AfterToolCallbacks {
		afterToolCallbacks = append(afterToolCallbacks, llminternal.AfterToolCallback(c))
	}

	onToolErrorCallback := make([]llminternal.OnToolErrorCallback, 0, len(cfg.OnToolErrorCallbacks))
	for _, c := range cfg.OnToolErrorCallbacks {
		onToolErrorCallback = append(onToolErrorCallback, llminternal.OnToolErrorCallback(c))
	}

	a := &llmAgent{
		model:                 cfg.Model,
		beforeModelCallbacks:  beforeModelCallbacks,
		afterModelCallbacks:   afterModelCallbacks,
		onModelErrorCallbacks: onModelErrorCallbacks,
		beforeToolCallbacks:   beforeToolCallbacks,
		afterToolCallbacks:    afterToolCallbacks,
		onToolErrorCallbacks:  onToolErrorCallback,
		instruction:           cfg.Instruction,
		inputSchema:           cfg.InputSchema,
		outputSchema:          cfg.OutputSchema,

		State: llminternal.State{
			Model:                     cfg.Model,
			Mode:                      cfg.Mode,
			GenerateContentConfig:     cfg.GenerateContentConfig,
			Tools:                     slices.Clone(cfg.Tools),
			Toolsets:                  slices.Clone(cfg.Toolsets),
			DisallowTransferToParent:  cfg.DisallowTransferToParent,
			DisallowTransferToPeers:   cfg.DisallowTransferToPeers,
			InputSchema:               cfg.InputSchema,
			OutputSchema:              cfg.OutputSchema,
			OutputJsonSchema:          cfg.OutputJsonSchema,
			IncludeContents:           llminternal.IncludeContents(cfg.IncludeContents),
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

	// TODO: remove this in favor of the state reveal below.
	a.Agent = baseAgent
	a.AgentType = agentinternal.TypeLLMAgent
	a.Config = cfg

	// TODO: temporary hack to set the LLMAgent type field correctly. Currently, beforeAgentCallback for LLMAgent only
	// sees basic *agent.agent type: http://google3/third_party/golang/adk/agent/agent.go;l=177-201;rcl=869633263
	// So in BeforeAgentCallback, we cannot access llmAgent.State fields.
	// We should remote llminternal.State in favor of agentinternal.State.

	internalAgent, ok := baseAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeLLMAgent
	state.Config = cfg
	// A before-agent callback that returns content short-circuits the run, so its
	// event never reaches llmAgent.run. Give it the same output_key handling.
	state.OnBeforeAgentCallbackEvent = a.maybeSaveOutputToState

	if err := installTaskTools(a); err != nil {
		return nil, err
	}

	return a, nil
}

func installTaskTools(a *llmAgent) error {
	state := llminternal.Reveal(a)

	if state.Mode == llminternal.ModeTask {
		finishTool, err := workflowinternal.NewFinishTaskTool(a)
		if err != nil {
			return fmt.Errorf("installing `finish_task` tool for agent %q: %w", a.Name(), err)
		}
		state.Tools = append(state.Tools, finishTool)
	}

	for _, sub := range a.SubAgents() {
		subInternal, ok := sub.(llminternal.Agent)
		if !ok {
			continue
		}
		subState := llminternal.Reveal(subInternal)
		// A sub-agent that declares no mode is a chat peer, reached via
		// transfer_to_agent rather than through a tool on the parent.
		switch llminternal.ResolveMode(subState.Mode, llminternal.ModeChat) {
		case llminternal.ModeSingleTurn:
			t, err := workflowinternal.NewSingleTurnTool(sub)
			if err != nil {
				return fmt.Errorf(
					"failed to install `single_turn` agent tool for sub-agent %q of %q: %w",
					sub.Name(), a.Name(), err,
				)
			}
			state.Tools = append(state.Tools, t)
		case llminternal.ModeTask:
			t, err := workflowinternal.NewTaskAgentTool(sub)
			if err != nil {
				return fmt.Errorf(
					"failed to install `task` agent tool for sub-agent %q of %q: %w",
					sub.Name(), a.Name(), err,
				)
			}
			state.Tools = append(state.Tools, t)
		}
	}

	return nil
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

	// BeforeAgentCallbacks are called sequentially before the agent starts its
	// run. See [agent.BeforeAgentCallback] for how return values affect execution.
	BeforeAgentCallbacks []agent.BeforeAgentCallback
	// AfterAgentCallbacks are called sequentially after the agent completes its
	// run. See [agent.AfterAgentCallback] for when they are skipped and how return
	// values affect execution.
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

	OnModelErrorCallbacks []OnModelErrorCallback

	// Instruction is set for the LLM model guiding the agent's behavior.
	//
	// The string is treated as a template:
	//  - There can be placeholders like {key_name} that will be resolved by ADK
	//    at runtime using session state and context.
	//  - key_name must be an underscore or a Unicode ID_Start character
	//    followed by Unicode ID_Continue characters, optionally after an
	//    app:, user: or temp: prefix; otherwise the placeholder is left
	//    as literal text.
	//  - {artifact.key_name} can be used to insert the text content of the
	//    artifact named key_name.
	//  - A placeholder directly preceded by $ or \ is treated as a literal,
	//    so "${key_name}", "${{key_name}}" and "\{key_name}" are left as is.
	//
	// If the state variable or artifact does not exist, the agent will raise an
	// error. If you want to ignore the error, you can append a ? to the
	// variable name as in {var?} to make it optional.
	//
	// If templating logic for {} chars is not desired, then InstructionProvider
	// should be used.
	Instruction string
	// InstructionProvider allows to create instructions dynamically based on
	// the agent context.
	//
	// It takes over the Instruction field if both are set.
	//
	// InstructionProvider does not automatically substitute values to {} and
	// treats them as just a raw char.
	// If you need to inject session state variables, use
	// util/instructionutil.InjectSessionState helper.
	InstructionProvider InstructionProvider

	// GlobalInstruction is the instruction for all agents in the entire
	// agent tree.
	//
	// The string is treated as a template:
	//  - There can be placeholders like {key_name} that will be resolved by ADK
	//    at runtime using session state and context.
	//  - key_name must be an underscore or a Unicode ID_Start character
	//    followed by Unicode ID_Continue characters, optionally after an
	//    app:, user: or temp: prefix; otherwise the placeholder is left
	//    as literal text.
	//  - {artifact.key_name} can be used to insert the text content of the
	//    artifact named key_name.
	//  - A placeholder directly preceded by $ or \ is treated as a literal,
	//    so "${key_name}", "${{key_name}}" and "\{key_name}" are left as is.
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
	//
	// Left unset, an agent placed as a single_turn workflow node sees only the
	// current turn. Setting IncludeContentsDefault explicitly keeps the history
	// even there.
	IncludeContents IncludeContents

	// TODO(ngeorgy): consider to switch to jsonschema for input and output schema.
	// The input schema when agent is used as a tool.
	InputSchema *genai.Schema
	// The output schema when agent replies.
	//
	// The schema reaches the model in one of two ways. When the agent has
	// Tools or Toolsets, its Model does not declare
	// [model.Capabilities].OutputSchemaAndTools (see
	// [model.CapabilitiesProvider]), and Mode is not ModeTask, the agent adds
	// the set_model_response tool of package
	// [google.golang.org/adk/v2/tool/setmodelresponsetool] to the request.
	// The model can then call the other tools and gives its final answer
	// through set_model_response, whose valid answer becomes the agent's
	// final text as JSON. An answer that does not match the schema goes
	// back to the model as feedback in the function response, and the model
	// is called again. When the agent has no tools or toolsets, or its
	// model declares OutputSchemaAndTools, and Mode is not ModeTask, the
	// schema goes into the request config as
	// GenerateContentConfig.ResponseSchema. In ModeTask the finish_task tool
	// carries the schema instead.
	OutputSchema *genai.Schema
	// OutputJsonSchema is the output schema when the agent replies, given as a
	// raw JSON schema. It carries what a *genai.Schema cannot, such as
	// additionalProperties. It is mutually exclusive with OutputSchema.
	// It reaches the model by the rule of OutputSchema, with
	// GenerateContentConfig.ResponseJsonSchema in place of ResponseSchema. The
	// set_model_response tool passes an answer for it through without
	// validating it, as adk-python does.
	OutputJsonSchema *jsonschema.Schema

	// Callbacks are executed in the order they are provided.
	// If a callback returns result/error, then the execution of the callback
	// list stops AND the actual tool call is skipped.
	BeforeToolCallbacks []BeforeToolCallback
	// Tools available to the agent. The model is offered the function
	// declaration of each tool. A tool whose Declaration method returns nil is
	// not offered, and a function call to its name gets the tool-not-found
	// result. In live mode (RunLive) the second rule holds only for a tool
	// that registers itself in the request, as toolutils.PackTool does.
	Tools []tool.Tool
	// Callbacks are executed in the order they are provided.
	// If a callback returns result/error, then the execution of the callback
	// list stops and this result/error is returned instead.
	AfterToolCallbacks []AfterToolCallback
	// Toolsets will be used by llmagent to extract tools and pass to the
	// underlying LLM.
	Toolsets []tool.Toolset

	OnToolErrorCallbacks []OnToolErrorCallback

	// OutputKey is an optional parameter to specify the key in session state for the agent output.
	//
	// Typical uses cases are:
	// - Extracts agent reply for later use, such as in tools, callbacks, etc.
	// - Connects agents to coordinate with each other.
	OutputKey string

	// Mode is the delegation mode for this agent.
	//
	// Options:
	//   ModeChat: Standard chat agent reachable via transfer_to_agent.
	//   ModeTask: Task agent that chats with the user to accomplish a task.
	//   ModeSingleTurn: Agents that complete a task without chatting with the user.
	//
	// Default value is ModeChat as a sub-agent, ModeSingleTurn as a node in a workflow.
	Mode Mode
}

// Mode is the delegation mode of an LLMAgent. See [Config.Mode] for details.
type Mode = llminternal.Mode

const (
	// ModeUnset means the mode has not been set explicitly.
	ModeUnset = llminternal.ModeUnset
	// ModeChat is the standard chat agent reachable via transfer_to_agent.
	ModeChat = llminternal.ModeChat
	// ModeTask is a task agent that chats with the user to accomplish a task.
	ModeTask = llminternal.ModeTask
	// ModeSingleTurn is an agent that completes a task without chatting with
	// the user.
	ModeSingleTurn = llminternal.ModeSingleTurn
)

// BeforeModelCallback that is called before sending a request to the model.
//
// If it returns non-nil LLMResponse or error, the actual model call is skipped
// and the returned response/error is used.
type BeforeModelCallback func(ctx agent.Context, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

// AfterModelCallback that is called after receiving a response from the model.
//
// If it returns non-nil LLMResponse or error, the actual model response/error
// is replaced with the returned response/error. A replacement keeps the
// Partial, TurnComplete and UsageMetadata of the response it replaces unless
// it sets them itself. Partial counts as set when it is true or when
// PartialSet is true, so a replacement for a partial response that sets
// Partial: false and PartialSet: true produces a non-partial event, while one
// that sets only Partial: false stays partial. TurnComplete has no such mark:
// a replacement for a response with TurnComplete true always keeps
// TurnComplete true. UsageMetadata counts as set when it is non-nil.
type AfterModelCallback func(ctx agent.Context, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

// OnModelErrorCallback that is called when receiving an error response from the llm model.
//
// If it returns non-nil LLMResponse or error, the actual model response/error
// is replaced with the returned response/error.
type OnModelErrorCallback func(ctx agent.Context, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)

// BeforeToolCallback is executed before a tool's Run method.
//
// Callbacks are executed in the order they are provided.
// If a callback returns a non-nil result or an error:
// - execution of remaining callbacks stops
// - the actual tool call is skipped
// - the returned result is used as the tool result
//
// To modify tool arguments and still run the tool,
// update args in place and return (nil, nil).
// The edits reach the tool and the callbacks that run after this one for the
// same call. They do not reach the function call event, which keeps the
// arguments the model sent in the session and in later model requests.
type BeforeToolCallback func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error)

// AfterToolCallback is a function type executed after a tool's Run method has completed,
// regardless of whether the tool returned a result or an error.
//
// Callbacks are executed in the order they are provided.
// If a callback returns a non-nil result or an error:
//   - execution of remaining callbacks stops
//   - the returned result and/or error is used as the final tool output
type AfterToolCallback func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error)

// OnToolErrorCallback that is called when receiving an error response from tool execution.
//
// If it returns non-nil LLMResponse or error, the actual model response/error
// is replaced with the returned response/error.
type OnToolErrorCallback func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error)

// IncludeContents controls what parts of prior conversation history is received by llmagent.
type IncludeContents string

const (
	// IncludeContentsNone makes the llmagent operate solely on its current turn (latest user input + any following agent events).
	IncludeContentsNone IncludeContents = "none"
	// IncludeContentsDefault is what an unset IncludeContents behaves as for a
	// plain conversational agent: the llmagent receives the relevant
	// conversation history. Setting it explicitly is not the same as leaving
	// the field unset — it also keeps the history for an agent placed as a
	// single_turn workflow node, which otherwise sees only the current turn.
	IncludeContentsDefault IncludeContents = "default"
)

type llmAgent struct {
	agent.Agent
	llminternal.State
	agentState

	beforeModelCallbacks  []llminternal.BeforeModelCallback
	model                 model.LLM
	afterModelCallbacks   []llminternal.AfterModelCallback
	instruction           string
	onModelErrorCallbacks []llminternal.OnModelErrorCallback

	beforeToolCallbacks  []llminternal.BeforeToolCallback
	afterToolCallbacks   []llminternal.AfterToolCallback
	onToolErrorCallbacks []llminternal.OnToolErrorCallback

	inputSchema  *genai.Schema
	outputSchema *genai.Schema
}

type agentState = agentinternal.State

func (a *llmAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	var ag agent.Agent = a
	ctx = ctx.WithICDelta(&agent.InvocationContextDelta{Agent: &ag})

	f := &llminternal.Flow{
		Model:                 a.model,
		RequestProcessors:     llminternal.DefaultRequestProcessors,
		ResponseProcessors:    llminternal.DefaultResponseProcessors,
		BeforeModelCallbacks:  a.beforeModelCallbacks,
		AfterModelCallbacks:   a.afterModelCallbacks,
		OnModelErrorCallbacks: a.onModelErrorCallbacks,
		BeforeToolCallbacks:   a.beforeToolCallbacks,
		AfterToolCallbacks:    a.afterToolCallbacks,
		OnToolErrorCallbacks:  a.onToolErrorCallbacks,
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

func (a *llmAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:      ctx.Artifacts(),
		Memory:         ctx.Memory(),
		Session:        ctx.Session(),
		Branch:         ctx.Branch(),
		IsolationScope: ctx.IsolationScope(),
		Agent:          a,
		UserContent:    ctx.UserContent(),
		RunConfig:      ctx.RunConfig(),
		InvocationID:   ctx.InvocationID(),
	})

	f := &llminternal.Flow{
		Model:                 a.model,
		RequestProcessors:     llminternal.DefaultRequestProcessors,
		ResponseProcessors:    llminternal.DefaultResponseProcessors,
		BeforeModelCallbacks:  a.beforeModelCallbacks,
		AfterModelCallbacks:   a.afterModelCallbacks,
		OnModelErrorCallbacks: a.onModelErrorCallbacks,
		BeforeToolCallbacks:   a.beforeToolCallbacks,
		AfterToolCallbacks:    a.afterToolCallbacks,
		OnToolErrorCallbacks:  a.onToolErrorCallbacks,
	}

	sess, innerIter, err := f.RunLive(ctx)
	if err != nil {
		return nil, nil, err
	}

	wrappedIter := func(yield func(*session.Event, error) bool) {
		for ev, err := range innerIter {
			if err == nil {
				a.maybeSaveOutputToState(ev)
			}
			if !yield(ev, err) {
				return
			}
		}
	}

	return sess, wrappedIter, nil
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
	// A task agent delivers its result through finish_task, which runTask writes
	// to OutputKey. Its text turns (a clarification question, a greeting) are
	// not the result. runTask runs only when the task agent is entered as a node
	// (its RunNode method or RunLLMAgentAsNode), whichever agent sits above it.
	// When it is run through Run (as SequentialAgent, LoopAgent and
	// ParallelAgent do) or RunLive, nothing writes the finish_task output, so
	// OutputKey stays unset. Matches adk-python's __maybe_save_output_to_state.
	if llminternal.Reveal(a).Mode == llminternal.ModeTask {
		return
	}
	if a.OutputKey != "" && event.IsFinalResponse() && event.Content != nil && len(event.Content.Parts) > 0 {
		var sb strings.Builder
		hasTextPart := false
		for _, part := range event.Content.Parts {
			if part.Text != "" && !part.Thought {
				hasTextPart = true
				sb.WriteString(part.Text)
			}
		}
		if !hasTextPart {
			return
		}
		result := sb.String()

		// TODO: add output schema validation and unmarshalling
		if a.OutputSchema != nil || a.OutputJsonSchema != nil {
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

// FindAgent finds a sub-agent by name.
func (a *llmAgent) FindAgent(name string) agent.Agent {
	if a.Name() == name {
		return a
	}
	return a.Agent.FindSubAgent(name)
}

func (a *llmAgent) RunNode(ctx agent.Context, nodeInput any) iter.Seq2[*session.Event, error] {
	return RunLLMAgentAsNode(a, ctx, nodeInput)
}

// InstructionProvider allows to create instructions dynamically. It is called
// on each agent invocation.
//
// NOTE: when InstructionProvider is used, ADK will NOT inject session state
// placeholders into the instruction. You can use
// util/instructionutil.InjectSessionState() helper if this functionality is needed.
type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)
