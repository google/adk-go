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

package agent

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

// AgentOption configures the AgentSpec.
type AgentOption interface {
	apply2AgentSpec(types.Agent) error
}

// LLMAgentOption is an AgentOption that configures an LLMAgent.
// Passing this option to other types of Agent is an error.
type LLMAgentOption interface {
	AgentOption
	apply2LLMAgent(types.Agent) error
}

type optionFunc func(*types.AgentSpec) error

func (o optionFunc) apply2AgentSpec(a types.Agent) error {
	s := a.Spec()
	if s == nil {
		return fmt.Errorf("agent's spec is not configured")
	}
	return o(s)
}

func WithDescription(desc string) AgentOption {
	return optionFunc(func(s *types.AgentSpec) error {
		s.Description = desc
		return nil
	})
}

func WithSubAgents(agents ...types.Agent) AgentOption {
	return optionFunc(func(s *types.AgentSpec) error {
		s.SubAgents = agents
		return nil
	})
}

func WithBeforeAgentCallbacks(callbacks ...types.BeforeAgentCallback) AgentOption {
	return optionFunc(func(s *types.AgentSpec) error {
		s.BeforeAgentCallbacks = callbacks
		return nil
	})
}

func WithAfterAgentCallbacks(callbacks ...types.AfterAgentCallback) AgentOption {
	return optionFunc(func(s *types.AgentSpec) error {
		s.AfterAgentCallbacks = callbacks
		return nil
	})
}

type llmOptionFunc func(*LLMAgent) error

func (o llmOptionFunc) apply2LLMAgent(a types.Agent) error {
	llmAgent, ok := a.(*LLMAgent)
	if !ok {
		return fmt.Errorf("cannot apply to non-LLMAgent")
	}
	return o(llmAgent)
}

func (o llmOptionFunc) apply2AgentSpec(a types.Agent) error {
	if _, ok := a.(*LLMAgent); !ok {
		return fmt.Errorf("option cannot apply to non-LLMAgent")
	}
	return nil
}

func WithInstruction(inst string) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.Instruction = inst
		return nil
	})
}

func WithGlobalInstruction(inst string) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.GlobalInstruction = inst
		return nil
	})
}

func WithTools(tools ...types.Tool) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.Tools = tools
		// TODO: check if tools names don't conflict or include reserved names.
		return nil
	})
}

func WithDisallowTransferToParent() LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.DisallowTransferToParent = true
		return nil
	})
}

func WithDisallowTransferToPeers() LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.DisallowTransferToPeers = true
		return nil
	})
}

func WithIncludeContents(v string) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.IncludeContents = v
		return nil
	})
}

func WithInputSchema(s *genai.Schema) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.InputSchema = s
		return nil
	})
}

func WithOutputSchema(s *genai.Schema) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.OutputSchema = s
		return nil
	})
}

func WithBeforeModelCallbacks(callbacks ...BeforeModelCallback) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.BeforeModelCallbacks = callbacks
		return nil
	})
}

func WithAfterModelCallbacks(callbacks ...AfterModelCallback) LLMAgentOption {
	return llmOptionFunc(func(a *LLMAgent) error {
		a.AfterModelCallbacks = callbacks
		return nil
	})
}

// NewLLMAgent returns a new LLMAgent configured with the provided options.
func NewLLMAgent(name string, model types.Model, opts ...AgentOption) (*LLMAgent, error) {
	agentSpec := &types.AgentSpec{Name: name}
	a := &LLMAgent{Model: model, agentSpec: agentSpec}

	// apply Options that are not llmOptions to initialize agentSpec.
	for _, o := range opts {
		if _, ok := o.(LLMAgentOption); ok {
			continue
		}
		if err := o.apply2AgentSpec(a); err != nil {
			return nil, err
		}
	}
	// fully initialize agentSpec.
	if err := agentSpec.Init(a); err != nil {
		return nil, err
	}

	// apply Options that are llmOptions.
	for _, o := range opts {
		llmOption, ok := o.(llmOptionFunc)
		if !ok {
			continue
		}
		if err := llmOption.apply2LLMAgent(a); err != nil {
			return nil, err
		}
	}
	return a, nil
}

// LLMAgent is an LLM-based Agent.
type LLMAgent struct {
	agentSpec *types.AgentSpec

	Model types.Model

	Instruction           string
	GlobalInstruction     string
	Tools                 []types.Tool
	GenerateContentConfig *genai.GenerateContentConfig

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

	// OutputKey
	// Planner
	// CodeExecutor
	// Examples

	// BeforeToolCallback
	// AfterToolCallback
}

type BeforeModelCallback func(ctx context.Context, callbackCtx *types.CallbackContext, llmRequest *types.LLMRequest) (*types.LLMResponse, error)

type AfterModelCallback func(ctx context.Context, callbackCtx *types.CallbackContext, llmResponse *types.LLMResponse, llmResponseError error) (*types.LLMResponse, error)

func (a *LLMAgent) Spec() *types.AgentSpec {
	return a.agentSpec
}

func (a *LLMAgent) Name() string {
	return a.agentSpec.Name
}

func (a *LLMAgent) Description() string {
	return a.agentSpec.Description
}

func (a *LLMAgent) newInvocationContext(ctx context.Context, p *types.InvocationContext) (context.Context, *types.InvocationContext) {
	ctx, c := types.NewInvocationContext(ctx, a, nil, nil, nil, nil)
	if p != nil {
		// copy everything but Agent and internal state.
		c.InvocationID = p.InvocationID
		c.Branch = p.Branch // TODO: why don't we update branch?
		c.UserContent = p.UserContent
		c.RunConfig = p.RunConfig
		c.Session = p.Session
		c.SessionService = p.SessionService
	}
	return ctx, c
}

func (a *LLMAgent) Run(ctx context.Context, parentCtx *types.InvocationContext) iter.Seq2[*types.Event, error] {
	// TODO: Select model (LlmAgent.canonical_model)
	ctx, parentCtx = a.newInvocationContext(ctx, parentCtx)
	flow := &baseFlow{
		Model:                a.Model,
		RequestProcessors:    defaultRequestProcessors,
		ResponseProcessors:   defaultResponseProcessors,
		BeforeModelCallbacks: a.BeforeModelCallbacks,
		AfterModelCallbacks:  a.AfterModelCallbacks,
	}
	return flow.Run(ctx, parentCtx)
}

func (a *LLMAgent) useAutoFlow() bool {
	return len(a.Spec().SubAgents) != 0 || !a.DisallowTransferToParent || !a.DisallowTransferToPeers
}

var _ types.Agent = (*LLMAgent)(nil)

var (
	defaultRequestProcessors = []func(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error{
		basicRequestProcessor,
		authPreprocesssor,
		instructionsRequestProcessor,
		identityRequestProcessor,
		contentsRequestProcessor,
		// Some implementations of NL Planning mark planning contents as thoughts in the post processor.
		// Since these need to be unmarked, NL Planning should be after contentsRequestProcessor.
		nlPlanningRequestProcessor,
		// Code execution should be after contentsRequestProcessor as it mutates the contents
		// to optimize data files.
		codeExecutionRequestProcessor,
		agentTransferRequestProcessor,
	}
	defaultResponseProcessors = []func(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest, resp *types.LLMResponse) error{
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
)

type baseFlow struct {
	Model types.Model

	RequestProcessors  []func(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error
	ResponseProcessors []func(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest, resp *types.LLMResponse) error

	BeforeModelCallbacks []BeforeModelCallback
	AfterModelCallbacks  []AfterModelCallback
}

func (f *baseFlow) Run(ctx context.Context, parentCtx *types.InvocationContext) iter.Seq2[*types.Event, error] {
	return func(yield func(*types.Event, error) bool) {
		for {
			var lastEvent *types.Event
			for ev, err := range f.runOneStep(ctx, parentCtx) {
				if err != nil {
					yield(nil, err)
					return
				}
				// forward the event first.
				if !yield(ev, nil) {
					return
				}
				lastEvent = ev
			}
			if lastEvent == nil || lastEvent.IsFinalResponse() {
				return
			}
			if lastEvent.LLMResponse.Partial {
				// We may have reached max token limit during streaming mode.
				// TODO: handle Partial response in model level. CL 781377328
				yield(nil, fmt.Errorf("TODO: last event is not final"))
				return
			}
		}
	}
}

func (f *baseFlow) runOneStep(ctx context.Context, parentCtx *types.InvocationContext) iter.Seq2[*types.Event, error] {
	return func(yield func(*types.Event, error) bool) {
		req := &types.LLMRequest{Model: f.Model}

		// Preprocess before calling the LLM.
		if err := f.preprocess(ctx, parentCtx, req); err != nil {
			yield(nil, err)
			return
		}

		// Calls the LLM.
		for resp, err := range f.callLLM(ctx, parentCtx, req) {
			if err != nil {
				yield(nil, err)
				return
			}
			if err := f.postprocess(ctx, parentCtx, req, resp); err != nil {
				yield(nil, err)
				return
			}
			// Skip the model response event if there is no content and no error code.
			// This is needed for the code executor to trigger another loop according to
			// adk-python src/google/adk/flows/llm_flows/base_llm_flow.py BaseLlmFlow._postprocess_async.
			if resp.Content == nil && resp.ErrorCode == 0 && !resp.Interrupted {
				continue
			}
			// Build the event and yield.
			modelResponseEvent := f.finalizeModelResponseEvent(parentCtx, resp)
			if !yield(modelResponseEvent, nil) {
				return
			}
			// TODO: generate and yield an auth event if needed.

			// Handle function calls.
			ev, err := handleFunctionCalls(ctx, parentCtx, req.Tools, resp)
			if err != nil {
				yield(nil, err)
				return
			}
			if ev == nil {
				// nothing to yield/process.
				continue
			}
			if !yield(ev, nil) {
				return
			}

			// Actually handle "transfer_to_agent" tool. The function call sets the ev.Actions.TransferToAgent field.
			// We are followng python's execution flow which is
			//   BaseLlmFlow._postprocess_async
			//    -> _postprocess_handle_function_calls_async
			// TODO(hakim): figure out why this isn't handled by the runner.
			if ev.Actions == nil || ev.Actions.TransferToAgent == "" {
				return
			}
			nextAgent := f.agentToRun(parentCtx, ev.Actions.TransferToAgent)
			if nextAgent == nil {
				yield(nil, fmt.Errorf("failed to find agent: %s", ev.Actions.TransferToAgent))
				return
			}
			for ev, err := range nextAgent.Run(ctx, parentCtx) {
				if !yield(ev, err) || err != nil { // forward
					return
				}
			}
		}
	}
}

func (f *baseFlow) finalizeModelResponseEvent(parentCtx *types.InvocationContext, resp *types.LLMResponse) *types.Event {
	// FunctionCall & FunctionResponse matching algorithm assumes non-empty function call IDs
	// but function call ID is optional in genai API and some models do not use the field.
	// Generate function call ids. (see functions.populate_client_function_call_id in python SDK)
	populateClientFunctionCallID(resp.Content)

	ev := types.NewEvent(parentCtx.InvocationID)
	ev.Author = parentCtx.Agent.Spec().Name
	ev.Branch = parentCtx.Branch
	ev.LLMResponse = resp

	// TODO: populate ev.LongRunningToolIDs (see BaseLlmFlow._finalize_model_response_event)

	return ev
}

func (f *baseFlow) preprocess(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
	llmAgent := asLLMAgent(parentCtx.Agent)
	if llmAgent == nil {
		return nil
	}
	// apply request processor functions to the request in the configured order.
	for _, processor := range f.RequestProcessors {
		if err := processor(ctx, parentCtx, req); err != nil {
			return err
		}
	}
	// run processors for tools.
	// TODO: check need/feasibility of running this concurrently.
	for _, t := range llmAgent.Tools {
		toolCtx := &types.ToolContext{
			InvocationContext: parentCtx, // TODO: how to prevent mutation on this?
			EventActions:      &types.EventActions{},
		}
		if err := t.ProcessRequest(ctx, toolCtx, req); err != nil {
			return err
		}
	}
	return nil
}

func (f *baseFlow) callLLM(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) types.LLMResponseStream {
	return func(yield func(*types.LLMResponse, error) bool) {
		for _, callback := range f.BeforeModelCallbacks {
			callbackResponse, callbackErr := callback(ctx, &types.CallbackContext{
				InvocationContext: parentCtx,
			}, req)

			if callbackResponse != nil || callbackErr != nil {
				yield(callbackResponse, callbackErr)
				return
			}
		}

		// TODO: Set _ADK_AGENT_NAME_LABEL_KEY in req.GenerateConfig.Labels
		// to help with slicing the billing reports on a per-agent basis.

		// TODO: RunLive mode when invocation_context.run_config.support_cfc is true.

		for resp, err := range f.Model.GenerateContent(ctx, req, parentCtx.RunConfig != nil && parentCtx.RunConfig.StreamingMode == types.StreamingModeSSE) {
			callbackResp, callbackErr := f.runAfterModelCallbacks(ctx, parentCtx, resp, err)
			// TODO: check if we should stop iterator on the first error from stream or continue yielding next results.
			if callbackErr != nil {
				yield(nil, callbackErr)
				return
			}

			if callbackResp != nil {
				if !yield(callbackResp, nil) {
					return
				}
				continue
			}

			// TODO: check if we should stop iterator on the first error from stream or continue yielding next results.
			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(resp, nil) {
				return
			}
		}
	}
}

func (f *baseFlow) runAfterModelCallbacks(ctx context.Context, parentCtx *types.InvocationContext, llmResp *types.LLMResponse, llmErr error) (*types.LLMResponse, error) {
	for _, callback := range f.AfterModelCallbacks {
		callbackResponse, callbackErr := callback(ctx, &types.CallbackContext{
			InvocationContext: parentCtx,
		}, llmResp, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

func (f *baseFlow) postprocess(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest, resp *types.LLMResponse) error {
	// apply response processor functions to the response in the configured order.
	for _, processor := range f.ResponseProcessors {
		if err := processor(ctx, parentCtx, req, resp); err != nil {
			return err
		}
	}
	return nil
}

func (f *baseFlow) agentToRun(parentCtx *types.InvocationContext, agentName string) types.Agent {
	// NOTE: in python, BaseLlmFlow._get_gent_to_run searches the entire agent
	// tree from the root_agent when processing _postprocess_handle_function_calls_async.
	// I think that is strange. In our version, we check the agents included in transferTarget.
	agents := transferTarget(asLLMAgent(parentCtx.Agent))
	for _, agent := range agents {
		if agent.Spec().Name == agentName {
			return agent
		}
	}
	return nil
}

// handleFunctionCalls calls the functions and returns the function response event.
//
// TODO: accept filters to include/exclude function calls.
// TODO: check feasibility of running tool.Run concurrently.
func handleFunctionCalls(ctx context.Context, parentCtx *types.InvocationContext, toolsDict map[string]types.Tool, resp *types.LLMResponse) (*types.Event, error) {
	var fnResponseEvents []*types.Event

	fnCalls := functionCalls(resp.Content)
	for _, fnCall := range fnCalls {
		tool, ok := toolsDict[fnCall.Name]
		if !ok {
			return nil, fmt.Errorf("unknown tool: %q", fnCall.Name)
		}
		toolCtx := &types.ToolContext{
			InvocationContext: parentCtx,
			FunctionCallID:    fnCall.ID,
			EventActions:      &types.EventActions{},
		}
		// TODO: agent.canonical_before_tool_callbacks
		result, err := tool.Run(ctx, toolCtx, fnCall.Args)
		// genai.FunctionResponse expects to use "output" key to specify function output
		// and "error" key to specify error details (if any). If "output" and "error" keys
		// are not specified, then whole "response" is treated as function output.
		// TODO(hakim): revisit the tool's function signature to handle error from user function better.
		if err != nil {
			result = map[string]any{"error": fmt.Errorf("tool %q failed: %w", tool.Name(), err)}
		}
		// TODO: agent.canonical_after_tool_callbacks
		// TODO: handle long-running tool.
		ev := types.NewEvent(parentCtx.InvocationID)
		ev.LLMResponse = &types.LLMResponse{
			Content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       fnCall.ID,
							Name:     fnCall.Name,
							Response: result,
						},
					},
				},
			},
		}
		ev.Author = parentCtx.Agent.Spec().Name
		ev.Branch = parentCtx.Branch
		ev.Actions = toolCtx.EventActions
		fnResponseEvents = append(fnResponseEvents, ev)
	}
	return mergeParallelFunctionResponseEvents(fnResponseEvents)
}

func mergeParallelFunctionResponseEvents(events []*types.Event) (*types.Event, error) {
	switch len(events) {
	case 0:
		return nil, nil
	case 1:
		return events[0], nil
	}
	var parts []*genai.Part
	var actions *types.EventActions
	for _, ev := range events {
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
			continue
		}
		parts = append(parts, ev.LLMResponse.Content.Parts...)
		actions = mergeEventActions(actions, ev.Actions)
	}
	// reuse events[0]
	ev := events[0]
	ev.LLMResponse = &types.LLMResponse{
		Content: &genai.Content{
			Role:  "user",
			Parts: parts,
		},
	}
	ev.Actions = actions
	return ev, nil
}

func mergeEventActions(base, other *types.EventActions) *types.EventActions {
	// flows/llm_flows/functions.py merge_parallel_function_response_events
	//
	// TODO: merge_parallel_function_response_events creates a "last one wins" scenario
	// except parts and requested_auth_configs. Check with the ADK team about
	// the intention.
	if other == nil {
		return base
	}
	if base == nil {
		return other
	}
	if other.SkipSummarization {
		base.SkipSummarization = true
	}
	if other.TransferToAgent != "" {
		base.TransferToAgent = other.TransferToAgent
	}
	if other.Escalate {
		base.Escalate = true
	}
	if other.StateDelta != nil {
		base.StateDelta = other.StateDelta
	}
	return base
}
