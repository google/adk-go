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

package llminternal

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"slices"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type BeforeModelCallback func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error)

type AfterModelCallback func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error)

type Flow struct {
	Model llm.Model

	RequestProcessors    []func(ctx agent.Context, req *llm.Request) error
	ResponseProcessors   []func(ctx agent.Context, req *llm.Request, resp *llm.Response) error
	BeforeModelCallbacks []BeforeModelCallback
	AfterModelCallbacks  []AfterModelCallback
}

var (
	DefaultRequestProcessors = []func(ctx agent.Context, req *llm.Request) error{
		basicRequestProcessor,
		authPreprocesssor,
		instructionsRequestProcessor,
		identityRequestProcessor,
		ContentsRequestProcessor,
		// Some implementations of NL Planning mark planning contents as thoughts in the post processor.
		// Since these need to be unmarked, NL Planning should be after contentsRequestProcessor.
		nlPlanningRequestProcessor,
		// Code execution should be after contentsRequestProcessor as it mutates the contents
		// to optimize data files.
		codeExecutionRequestProcessor,
		AgentTransferRequestProcessor,
	}
	DefaultResponseProcessors = []func(ctx agent.Context, req *llm.Request, resp *llm.Response) error{
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
)

func (f *Flow) Run(ctx agent.Context) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for {
			var lastEvent *session.Event
			for ev, err := range f.runOneStep(ctx) {
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
			if lastEvent == nil || utils.IsFinalResponse(lastEvent) {
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

func (f *Flow) runOneStep(ctx agent.Context) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		req := &llm.Request{}

		// Preprocess before calling the LLM.
		if err := f.preprocess(ctx, req); err != nil {
			yield(nil, err)
			return
		}

		// Calls the LLM.
		for resp, err := range f.callLLM(ctx, req) {
			if err != nil {
				yield(nil, err)
				return
			}
			if err := f.postprocess(ctx, req, resp); err != nil {
				yield(nil, err)
				return
			}
			// Skip the model response event if there is no content and no error code.
			// This is needed for the code executor to trigger another loop according to
			// adk-python src/google/adk/flows/llm_flows/base_llm_flow.py BaseLlmFlow._postprocess_async.
			if resp.Content == nil && resp.ErrorCode == 0 && !resp.Interrupted {
				continue
			}

			// TODO: temporarily convert
			tools := make(map[string]tool.Tool)
			for k, v := range req.Tools {
				tool, ok := v.(tool.Tool)
				if !ok {
					if !yield(nil, fmt.Errorf("unexpected tool type %T for tool %v", v, k)) {
						return
					}
				}
				tools[k] = tool
			}

			// Build the event and yield.
			modelResponseEvent := f.finalizeModelResponseEvent(ctx, resp, tools)
			if !yield(modelResponseEvent, nil) {
				return
			}
			// TODO: generate and yield an auth event if needed.

			// Handle function calls.

			ev, err := handleFunctionCalls(ctx, tools, resp)
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
			if ev.Actions.TransferToAgent == "" {
				return
			}
			nextAgent := f.agentToRun(ctx, ev.Actions.TransferToAgent)
			if nextAgent == nil {
				yield(nil, fmt.Errorf("failed to find agent: %s", ev.Actions.TransferToAgent))
				return
			}
			for ev, err := range nextAgent.Run(ctx) {
				if !yield(ev, err) || err != nil { // forward
					return
				}
			}
		}
	}
}

func (f *Flow) preprocess(ctx agent.Context, req *llm.Request) error {
	llmAgent, ok := ctx.Agent().(Agent)
	if !ok {
		return fmt.Errorf("agent %v is not an LLMAgent", ctx.Agent().Name())
	}

	// apply request processor functions to the request in the configured order.
	for _, processor := range f.RequestProcessors {
		if err := processor(ctx, req); err != nil {
			return err
		}
	}
	// run processors for tools.
	return toolPreprocess(ctx, req, Reveal(llmAgent).Tools)
}

// toolPreprocess runs tool preprocess on the given request
// If a tool set is encountered, it's expanded recursively in DFS fashion.
// TODO: check need/feasibility of running this concurrently.
func toolPreprocess(ctx agent.Context, req *llm.Request, tools []tool.Tool) error {
	for _, t := range tools {
		toolSet, ok := t.(tool.Set)
		if ok {
			tsTools, err := toolSet.Tools(ctx)
			if err != nil {
				return fmt.Errorf("failed to extract tools from the tool set %q: %w", toolSet.Name(), err)
			}

			if err := toolPreprocess(ctx, req, tsTools); err != nil {
				return fmt.Errorf("failed to tool preprocess for tool set %q: %w", toolSet.Name(), err)
			}

			continue
		}

		requestProcessor, ok := t.(toolinternal.RequestProcessor)
		if !ok {
			return fmt.Errorf("tool %q does not implement RequestProcessor() method", t.Name())
		}
		// TODO: how to prevent mutation on this?
		toolCtx := tool.NewContext(ctx, "", &session.Actions{})
		if err := requestProcessor.ProcessRequest(toolCtx, req); err != nil {
			return err
		}
	}

	return nil
}

func (f *Flow) callLLM(ctx agent.Context, req *llm.Request) iter.Seq2[*llm.Response, error] {
	return func(yield func(*llm.Response, error) bool) {
		for _, callback := range f.BeforeModelCallbacks {
			callbackResponse, callbackErr := callback(ctx, req)

			if callbackResponse != nil || callbackErr != nil {
				yield(callbackResponse, callbackErr)
				return
			}
		}

		// TODO: Set _ADK_AGENT_NAME_LABEL_KEY in req.GenerateConfig.Labels
		// to help with slicing the billing reports on a per-agent basis.

		// TODO: RunLive mode when invocation_context.run_config.support_cfc is true.

		gen := func(ctx context.Context, req *llm.Request) iter.Seq2[*llm.Response, error] {
			return func(yield func(*llm.Response, error) bool) {
				resp, err := f.Model.Generate(ctx, req)
				yield(resp, err)
			}
		}
		if runconfig.FromContext(ctx).StreamingMode == runconfig.StreamingModeSSE {
			gen = f.Model.GenerateStream
		}

		for resp, err := range gen(ctx, req) {
			callbackResp, callbackErr := f.runAfterModelCallbacks(ctx, resp, err)
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

func (f *Flow) runAfterModelCallbacks(ctx agent.Context, llmResp *llm.Response, llmErr error) (*llm.Response, error) {
	for _, callback := range f.AfterModelCallbacks {
		callbackResponse, callbackErr := callback(ctx, llmResp, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

func (f *Flow) postprocess(ctx agent.Context, req *llm.Request, resp *llm.Response) error {
	// apply response processor functions to the response in the configured order.
	for _, processor := range f.ResponseProcessors {
		if err := processor(ctx, req, resp); err != nil {
			return err
		}
	}
	return nil
}

func (f *Flow) agentToRun(ctx agent.Context, agentName string) agent.Agent {
	// NOTE: in python, BaseLlmFlow._get_agent_to_run searches the entire agent
	// tree from the root_agent when processing _postprocess_handle_function_calls_async.
	// I think that is strange. In our version, we check the agents included in transferTarget.
	parents := parentmap.FromContext(ctx)
	agents := transferTargets(ctx.Agent(), parents[ctx.Agent().Name()])
	for _, agent := range agents {
		if agent.Name() == agentName {
			return agent
		}
	}
	return nil
}

func (f *Flow) finalizeModelResponseEvent(ctx agent.Context, resp *llm.Response, tools map[string]tool.Tool) *session.Event {
	// FunctionCall & FunctionResponse matching algorithm assumes non-empty function call IDs
	// but function call ID is optional in genai API and some models do not use the field.
	// Generate function call ids. (see functions.populate_client_function_call_id in python SDK)
	utils.PopulateClientFunctionCallID(resp.Content)

	ev := session.NewEvent(ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.LLMResponse = resp

	// Populate ev.LongRunningToolIDs
	ev.LongRunningToolIDs = findLongRunningFunctionCallIDs(resp.Content, tools)

	return ev
}

// findLongRunningFunctionCallIDs iterates over the FunctionCalls and
// returns the callIDs of the long running functions
func findLongRunningFunctionCallIDs(c *genai.Content, tools map[string]tool.Tool) []string {
	set := make(map[string]struct{})
	// Iterate over function calls.
	for _, fc := range utils.FunctionCalls(c) {
		if tool, ok := tools[fc.Name]; ok && fc.ID != "" && tool.IsLongRunning() {
			// If the tool exists and is long-running, add its ID to the set.
			set[fc.ID] = struct{}{}
		}
	}
	// Transform the set (map keys) into a slice.
	return slices.Collect(maps.Keys(set))
}

// handleFunctionCalls calls the functions and returns the function response event.
//
// TODO: accept filters to include/exclude function calls.
// TODO: check feasibility of running tool.Run concurrently.
func handleFunctionCalls(ctx agent.Context, toolsDict map[string]tool.Tool, resp *llm.Response) (*session.Event, error) {
	var fnResponseEvents []*session.Event

	fnCalls := utils.FunctionCalls(resp.Content)
	for _, fnCall := range fnCalls {
		curTool, ok := toolsDict[fnCall.Name]
		if !ok {
			return nil, fmt.Errorf("unknown tool: %q", fnCall.Name)
		}
		funcTool, ok := curTool.(toolinternal.FunctionTool)
		if !ok {
			return nil, fmt.Errorf("tool %q is not a function tool", curTool.Name())
		}
		toolCtx := tool.NewContext(ctx, fnCall.ID, &session.Actions{})
		//toolCtx := tool.
		// TODO: agent.canonical_before_tool_callbacks
		result, err := funcTool.Run(toolCtx, fnCall.Args)
		// genai.FunctionResponse expects to use "output" key to specify function output
		// and "error" key to specify error details (if any). If "output" and "error" keys
		// are not specified, then whole "response" is treated as function output.
		// TODO(hakim): revisit the tool's function signature to handle error from user function better.
		if err != nil {
			result = map[string]any{"error": fmt.Errorf("tool %q failed: %w", curTool.Name(), err)}
		}

		m, ok := result.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool %q failed to convert results to a required type, got %T", curTool.Name(), result)
		}

		// TODO: agent.canonical_after_tool_callbacks
		// TODO: handle long-running tool.
		ev := session.NewEvent(ctx.InvocationID())
		ev.LLMResponse = &llm.Response{
			Content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       fnCall.ID,
							Name:     fnCall.Name,
							Response: m,
						},
					},
				},
			},
		}
		ev.Author = ctx.Agent().Name()
		ev.Branch = ctx.Branch()
		ev.Actions = *toolCtx.EventActions()
		fnResponseEvents = append(fnResponseEvents, ev)
	}
	return mergeParallelFunctionResponseEvents(fnResponseEvents)
}

func mergeParallelFunctionResponseEvents(events []*session.Event) (*session.Event, error) {
	switch len(events) {
	case 0:
		return nil, nil
	case 1:
		return events[0], nil
	}
	var parts []*genai.Part
	var actions *session.Actions
	for _, ev := range events {
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
			continue
		}
		parts = append(parts, ev.LLMResponse.Content.Parts...)
		actions = mergeEventActions(actions, &ev.Actions)
	}
	// reuse events[0]
	ev := events[0]
	ev.LLMResponse = &llm.Response{
		Content: &genai.Content{
			Role:  "user",
			Parts: parts,
		},
	}
	ev.Actions = *actions
	return ev, nil
}

func mergeEventActions(base, other *session.Actions) *session.Actions {
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
