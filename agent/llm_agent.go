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

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

// LLMAgent is an LLM-based Agent.
type LLMAgent struct {
	AgentName        string
	AgentDescription string

	Model adk.Model

	Instruction           string
	GlobalInstruction     string
	Tools                 []adk.Tool
	GenerateContentConfig *genai.GenerateContentConfig

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// Whether to include contents in the model request.
	// When set to 'none', the model request will not include any contents, such as
	// user messages, tool requests, etc.
	IncludeContents string

	// The input schema when agent is used as a tool.
	IntpuSchema *genai.Schema

	// The output schema when agent replies.
	//
	// NOTE: when this is set, agent can only reply and cannot use any tools,
	// such as function tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	ParentAgent adk.Agent
	SubAgents   []adk.Agent

	// OutputKey
	// Planner
	// CodeExecutor
	// Examples

	// BeforeModelCallback
	// AfterModelCallback
	// BeforeToolCallback
	// AfterToolCallback
}

// AddSubAgents adds the agents to the subagent list.
func (a *LLMAgent) AddSubAgents(agents ...adk.Agent) {
	for _, subagent := range agents {
		a.SubAgents = append(a.SubAgents, subagent)
		if s := asLLMAgent(subagent); s != nil {
			s.ParentAgent = a
		}
	}
}

func (a *LLMAgent) Name() string        { return a.AgentName }
func (a *LLMAgent) Description() string { return a.AgentDescription }
func (a *LLMAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
	// TODO: Select model (LlmAgent.canonical_model)

	flow := &baseFlow{
		Model:              a.Model,
		RequestProcessors:  defaultRequestProcessors,
		ResponseProcessors: defaultResponseProcessors,
	}
	return flow.Run(ctx, parentCtx), nil
}

func (a *LLMAgent) useAutoFlow() bool {
	return len(a.SubAgents) != 0 || !a.DisallowTransferToParent || !a.DisallowTransferToPeers
}

var _ adk.Agent = (*LLMAgent)(nil)

var (
	defaultRequestProcessors = []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error{
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
	defaultResponseProcessors = []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error{
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
)

type baseFlow struct {
	Model adk.Model

	RequestProcessors  []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error
	ResponseProcessors []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error
}

func (f *baseFlow) Run(ctx context.Context, parentCtx *adk.InvocationContext) adk.EventStream {
	return func(yield func(*adk.Event, error) bool) {
		for {
			var lastEvent *adk.Event
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

func (f *baseFlow) runOneStep(ctx context.Context, parentCtx *adk.InvocationContext) adk.EventStream {
	return func(yield func(*adk.Event, error) bool) {
		req := &adk.LLMRequest{Model: f.Model}

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
			for ev, err := range f.postprocessHandleFunctionCalls(ctx, parentCtx, req, resp) {
				if !yield(ev, err) || err != nil {
					return
				}
			}
		}
	}
}

func (f *baseFlow) finalizeModelResponseEvent(parentCtx *adk.InvocationContext, resp *adk.LLMResponse) *adk.Event {
	// FunctionCall & FunctionResponse matching algorithm assumes non-empty function call IDs
	// but function call ID is optional in genai API and some models do not use the field.
	// Generate function call ids. (see functions.populate_client_function_call_id in python SDK)
	populateClientFunctionCallID(resp.Content)

	ev := adk.NewEvent(parentCtx.InvocationID)
	ev.Author = parentCtx.Agent.Name()
	ev.Branch = parentCtx.Branch
	ev.LLMResponse = resp

	// TODO: populate ev.LongRunningToolIDs (see BaseLlmFlow._finalize_model_response_event)

	return ev
}

func (f *baseFlow) preprocess(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
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
		toolCtx := &adk.ToolContext{
			InvocationContext: parentCtx, // TODO: how to prevent mutation on this?
			EventActions:      &adk.EventActions{},
		}
		if err := t.ProcessRequest(ctx, toolCtx, req); err != nil {
			return err
		}
	}
	return nil
}

func (f *baseFlow) callLLM(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) adk.LLMResponseStream {
	return func(yield func(*adk.LLMResponse, error) bool) {

		// TODO: run BeforeModelCallback if exists.
		//   if f.BeforeModelCallback != nil {
		//      resp, err := f.BeforeModelCallback(...)
		//      yield(resp, err)
		//      return
		//   }

		// TODO: Set _ADK_AGENT_NAME_LABEL_KEY in req.GenerateConfig.Labels
		// to help with slicing the billing reports on a per-agent basis.

		// TODO: RunLive mode when invocation_context.run_config.support_cfc is true.

		for resp, err := range f.Model.GenerateContent(ctx, req, parentCtx.RunConfig != nil && parentCtx.RunConfig.StreamingMode == adk.StreamingModeSSE) {
			if err != nil {
				yield(nil, err)
				return
			}
			// TODO: run AfterModelCallback if exists.
			if !yield(resp, err) {
				return
			}
		}
	}
}

func (f *baseFlow) postprocess(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error {
	// apply response processor functions to the response in the configured order.
	for _, processor := range f.ResponseProcessors {
		if err := processor(ctx, parentCtx, req, resp); err != nil {
			return err
		}
	}
	return nil
}

func (f *baseFlow) postprocessHandleFunctionCalls(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) adk.EventStream {
	return func(yield func(*adk.Event, error) bool) {
		ev, err := handleFunctionCalls(ctx, parentCtx, req.Tools, resp)
		if err != nil {
			yield(nil, err)
			return
		}
		if ev == nil {
			// nothing to yield.
			return
		}
		// TODO: generate and yield an auth event if needed.

		if !yield(ev, nil) {
			return
		}

		// TODO: handle ev.Actions such as TransferToAgent set by transfer_to_agent tool.
	}
}

// handleFunctionCalls calls the functions and returns the function response event.
//
// TODO: accept filters to include/exclude function calls.
// TODO: check feasibility of running tool.Run concurrently.
func handleFunctionCalls(ctx context.Context, parentCtx *adk.InvocationContext, toolsDict map[string]adk.Tool, resp *adk.LLMResponse) (*adk.Event, error) {
	var fnResponseEvents []*adk.Event

	fnCalls := functionCalls(resp.Content)
	for _, fnCall := range fnCalls {
		tool, ok := toolsDict[fnCall.Name]
		if !ok {
			return nil, fmt.Errorf("unknown tool: %q", fnCall.Name)
		}
		toolCtx := &adk.ToolContext{
			InvocationContext: parentCtx,
			FunctionCallID:    fnCall.ID,
			EventActions:      &adk.EventActions{},
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
		ev := adk.NewEvent(parentCtx.InvocationID)
		ev.LLMResponse = &adk.LLMResponse{
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
		ev.Author = parentCtx.Agent.Name()
		ev.Branch = parentCtx.Branch
		ev.Actions = toolCtx.EventActions
		fnResponseEvents = append(fnResponseEvents, ev)
	}
	return mergeParallelFunctionResponseEvents(fnResponseEvents)
}

func mergeParallelFunctionResponseEvents(events []*adk.Event) (*adk.Event, error) {
	switch len(events) {
	case 0:
		return nil, nil
	case 1:
		return events[0], nil
	}
	var parts []*genai.Part
	var actions *adk.EventActions
	for _, ev := range events {
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
			continue
		}
		parts = append(parts, ev.LLMResponse.Content.Parts...)
		actions = mergeEventActions(actions, ev.Actions)
	}
	// reuse events[0]
	ev := events[0]
	ev.LLMResponse = &adk.LLMResponse{
		Content: &genai.Content{
			Role:  "user",
			Parts: parts,
		},
	}
	ev.Actions = actions
	return ev, nil
}

func mergeEventActions(base, other *adk.EventActions) *adk.EventActions {
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
