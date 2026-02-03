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
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/plugininternal/plugincontext"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

var ErrModelNotConfigured = errors.New("model not configured; ensure Model is set in llmagent.Config")

type BeforeModelCallback func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

type AfterModelCallback func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

type OnModelErrorCallback func(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)

type BeforeToolCallback func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error)

type AfterToolCallback func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error)

type OnToolErrorCallback func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error)

type Flow struct {
	Model model.LLM

	RequestProcessors     []func(ctx agent.InvocationContext, req *model.LLMRequest) error
	ResponseProcessors    []func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error
	BeforeModelCallbacks  []BeforeModelCallback
	AfterModelCallbacks   []AfterModelCallback
	OnModelErrorCallbacks []OnModelErrorCallback
	BeforeToolCallbacks   []BeforeToolCallback
	AfterToolCallbacks    []AfterToolCallback
	OnToolErrorCallbacks  []OnToolErrorCallback
}

var (
	DefaultRequestProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest) error{
		basicRequestProcessor,
		authPreprocessor,
		instructionsRequestProcessor,
		identityRequestProcessor,
		ContentsRequestProcessor,
		// Some implementations of NL Planning mark planning contents as thoughts in the post processor.
		// Since these need to be unmarked, NL Planning should be after contentsRequestProcessor.
		nlPlanningRequestProcessor,
		// Code execution should be after contentsRequestProcessor as it mutates the contents
		// to optimize data files.
		codeExecutionRequestProcessor,
		outputSchemaRequestProcessor,
		AgentTransferRequestProcessor,
		removeDisplayNameIfExists,
	}
	DefaultResponseProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error{
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
)

func (f *Flow) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
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

func (f *Flow) runOneStep(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if f.Model == nil {
			yield(nil, fmt.Errorf("agent %q: %w", ctx.Agent().Name(), ErrModelNotConfigured))
			return
		}

		req := &model.LLMRequest{
			Model: f.Model.Name(),
		}

		// Preprocess before calling the LLM.
		if err := f.preprocess(ctx, req); err != nil {
			yield(nil, err)
			return
		}
		if ctx.Ended() {
			return
		}
		spans := telemetry.StartTrace(ctx, "call_llm")
		// Create event to pass to callback state delta
		stateDelta := make(map[string]any)
		// Calls the LLM.
		for resp, err := range f.callLLM(ctx, req, stateDelta) {
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
			if resp.Content == nil && resp.ErrorCode == "" && !resp.Interrupted {
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
			modelResponseEvent := f.finalizeModelResponseEvent(ctx, resp, tools, stateDelta)
			telemetry.TraceLLMCall(spans, ctx, req, modelResponseEvent)
			if !yield(modelResponseEvent, nil) {
				return
			}
			// TODO: generate and yield an auth event if needed.

			// Handle function calls.

			ev, err := f.handleFunctionCalls(ctx, tools, resp)
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

			// If the model response is structured, yield it as a final model response event.
			outputSchemaResponse, err := retrieveStructuredModelResponse(ev)
			if err != nil {
				yield(nil, err)
				return
			}
			if outputSchemaResponse != "" {
				if !yield(createFinalModelResponseEvent(ctx, outputSchemaResponse), nil) {
					return
				}
			}
			// Actually handle "transfer_to_agent" tool. The function call sets the ev.Actions.TransferToAgent field.
			// We are following python's execution flow which is
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

func (f *Flow) preprocess(ctx agent.InvocationContext, req *model.LLMRequest) error {
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
	tools := Reveal(llmAgent).Tools
	for _, toolSet := range Reveal(llmAgent).Toolsets {
		tsTools, err := toolSet.Tools(icontext.NewReadonlyContext(ctx))
		if err != nil {
			return fmt.Errorf("failed to extract tools from the tool set %q: %w", toolSet.Name(), err)
		}

		tools = append(tools, tsTools...)
	}

	return toolPreprocess(ctx, req, tools)
}

// toolPreprocess runs tool preprocess on the given request
// If a tool set is encountered, it's expanded recursively in DFS fashion.
// TODO: check need/feasibility of running this concurrently.
func toolPreprocess(ctx agent.InvocationContext, req *model.LLMRequest, tools []tool.Tool) error {
	for _, t := range tools {
		requestProcessor, ok := t.(toolinternal.RequestProcessor)
		if !ok {
			return fmt.Errorf("tool %q does not implement RequestProcessor() method", t.Name())
		}
		// TODO: how to prevent mutation on this?
		toolCtx := toolinternal.NewToolContext(ctx, "", &session.EventActions{})
		if err := requestProcessor.ProcessRequest(toolCtx, req); err != nil {
			return err
		}
	}
	return nil
}

func (f *Flow) callLLM(ctx agent.InvocationContext, req *model.LLMRequest, stateDelta map[string]any) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		pluginManager := pluginManagerFromContext(ctx)
		if pluginManager != nil {
			cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
			callbackResponse, callbackErr := pluginManager.RunBeforeModelCallback(cctx, req)
			if callbackResponse != nil || callbackErr != nil {
				yield(callbackResponse, callbackErr)
				return
			}
		}

		for _, callback := range f.BeforeModelCallbacks {
			cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
			callbackResponse, callbackErr := callback(cctx, req)

			if callbackResponse != nil || callbackErr != nil {
				yield(callbackResponse, callbackErr)
				return
			}
		}

		// TODO: Set _ADK_AGENT_NAME_LABEL_KEY in req.GenerateConfig.Labels
		// to help with slicing the billing reports on a per-agent basis.

		// TODO: RunLive mode when invocation_context.run_config.support_cfc is true.
		useStream := runconfig.FromContext(ctx).StreamingMode == runconfig.StreamingModeSSE

		for resp, err := range f.Model.GenerateContent(ctx, req, useStream) {
			if err != nil {
				cbResp, cbErr := f.runOnModelErrorCallbacks(ctx, req, stateDelta, err)
				if cbErr != nil {
					yield(nil, cbErr)
					return
				}
				if cbResp == nil {
					yield(nil, err)
					return
				}
				resp = cbResp
				err = cbErr
			}
			callbackResp, callbackErr := f.runAfterModelCallbacks(ctx, resp, stateDelta, err)
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

func (f *Flow) runAfterModelCallbacks(ctx agent.InvocationContext, llmResp *model.LLMResponse, stateDelta map[string]any, llmErr error) (*model.LLMResponse, error) {
	pluginManager := pluginManagerFromContext(ctx)
	if pluginManager != nil {
		cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
		callbackResponse, callbackErr := pluginManager.RunAfterModelCallback(cctx, llmResp, llmErr)
		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	for _, callback := range f.AfterModelCallbacks {
		cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
		callbackResponse, callbackErr := callback(cctx, llmResp, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

func (f *Flow) runOnModelErrorCallbacks(ctx agent.InvocationContext, llmReq *model.LLMRequest, stateDelta map[string]any, llmErr error) (*model.LLMResponse, error) {
	pluginManager := pluginManagerFromContext(ctx)
	if pluginManager != nil {
		cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
		callbackResponse, callbackErr := pluginManager.RunOnModelErrorCallback(cctx, llmReq, llmErr)
		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	for _, callback := range f.OnModelErrorCallbacks {
		cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta)
		callbackResponse, callbackErr := callback(cctx, llmReq, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

func (f *Flow) postprocess(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// apply response processor functions to the response in the configured order.
	for _, processor := range f.ResponseProcessors {
		if err := processor(ctx, req, resp); err != nil {
			return err
		}
	}
	return nil
}

func (f *Flow) agentToRun(ctx agent.InvocationContext, agentName string) agent.Agent {
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

func (f *Flow) finalizeModelResponseEvent(ctx agent.InvocationContext, resp *model.LLMResponse, tools map[string]tool.Tool, stateDelta map[string]any) *session.Event {
	// FunctionCall & FunctionResponse matching algorithm assumes non-empty function call IDs
	// but function call ID is optional in genai API and some models do not use the field.
	// Generate function call ids. (see functions.populate_client_function_call_id in python SDK)
	utils.PopulateClientFunctionCallID(resp.Content)

	ev := session.NewEvent(ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.LLMResponse = *resp
	ev.Actions.StateDelta = stateDelta

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

type fakeTool struct {
	name string
}

func (f *fakeTool) Name() string      { return f.name }
func (*fakeTool) Description() string { return "Tool not found" }
func (*fakeTool) IsLongRunning() bool { return false }

var _ tool.Tool = (*fakeTool)(nil)

// newToolNotFoundError creates an error matching the specific Python format
func newToolNotFoundError(toolName string, availableTools []string) error {
	joinedTools := strings.Join(availableTools, ", ")

	return fmt.Errorf(`tool '%s' not found.
Available tools: %s

Possible causes:
  1. LLM hallucinated the function name - review agent instruction clarity
  2. Tool not registered - verify agent.tools list
  3. Name mismatch - check for typos

Suggested fixes:
  - Review agent instruction to ensure tool usage is clear
  - Verify tool is included in agent.tools list
  - Check for typos in function name`, toolName, joinedTools)
}

// handleFunctionCalls calls the functions and returns the function response event.
//
// TODO: accept filters to include/exclude function calls.
// TODO: check feasibility of running tool.Run concurrently.
func (f *Flow) handleFunctionCalls(ctx agent.InvocationContext, toolsDict map[string]tool.Tool, resp *model.LLMResponse) (*session.Event, error) {
	var fnResponseEvents []*session.Event

	fnCalls := utils.FunctionCalls(resp.Content)
	toolNames := slices.Collect(maps.Keys(toolsDict))
	var result map[string]any
	for _, fnCall := range fnCalls {
		toolCtx := toolinternal.NewToolContext(ctx, fnCall.ID, &session.EventActions{StateDelta: make(map[string]any)})

		spans := telemetry.StartTrace(ctx, "execute_tool "+fnCall.Name)
		curTool, found := toolsDict[fnCall.Name]
		if !found {
			err := newToolNotFoundError(fnCall.Name, toolNames)
			result, err = f.runOnToolErrorCallbacks(toolCtx, &fakeTool{name: fnCall.Name}, fnCall.Args, err)
			if err != nil {
				result = map[string]any{"error": err.Error()}
			}
		} else if funcTool, ok := curTool.(toolinternal.FunctionTool); !ok {
			err := newToolNotFoundError(fnCall.Name, toolNames)
			result, err = f.runOnToolErrorCallbacks(toolCtx, &fakeTool{name: fnCall.Name}, fnCall.Args, err)
			if err != nil {
				result = map[string]any{"error": err.Error()}
			}
		} else {
			result = f.callTool(toolCtx, funcTool, fnCall.Args)
		}

		// TODO: handle long-running tool.
		ev := session.NewEvent(ctx.InvocationID())
		ev.LLMResponse = model.LLMResponse{
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
		ev.Author = ctx.Agent().Name()
		ev.Branch = ctx.Branch()
		ev.Actions = *toolCtx.Actions()

		traceTool := curTool
		if traceTool == nil {
			traceTool = &fakeTool{name: fnCall.Name}
		}
		telemetry.TraceToolCall(spans, traceTool, fnCall.Args, ev)

		fnResponseEvents = append(fnResponseEvents, ev)
	}
	mergedEvent, err := mergeParallelFunctionResponseEvents(fnResponseEvents)
	if err != nil {
		return mergedEvent, err
	}
	// this is needed for debug traces of parallel calls
	spans := telemetry.StartTrace(ctx, "execute_tool (merged)")
	telemetry.TraceMergedToolCalls(spans, mergedEvent)
	return mergedEvent, nil
}

func (f *Flow) runOnToolErrorCallbacks(toolCtx tool.Context, tool tool.Tool, fArgs map[string]any, err error) (map[string]any, error) {
	pluginManager := pluginManagerFromContext(toolCtx)
	if pluginManager != nil {
		result, err := pluginManager.RunOnToolErrorCallback(toolCtx, tool, fArgs, err)
		if result != nil || err != nil {
			return result, err
		}
	}
	return f.invokeOnToolErrorCallbacks(toolCtx, tool, fArgs, err)
}

func (f *Flow) callTool(toolCtx tool.Context, tool toolinternal.FunctionTool, fArgs map[string]any) map[string]any {
	var response map[string]any
	var err error
	pluginManager := pluginManagerFromContext(toolCtx)
	if pluginManager != nil {
		response, err = pluginManager.RunBeforeToolCallback(toolCtx, tool, fArgs)
	}
	if response == nil && err == nil {
		response, err = f.invokeBeforeToolCallbacks(toolCtx, tool, fArgs)
	}

	if response == nil && err == nil {
		response, err = tool.Run(toolCtx, fArgs)
	}

	var errorResponse map[string]any
	var cbErr error
	if err != nil && pluginManager != nil {
		errorResponse, cbErr = pluginManager.RunOnToolErrorCallback(toolCtx, tool, fArgs, err)
	}
	if err != nil && errorResponse == nil && cbErr == nil {
		errorResponse, cbErr = f.invokeOnToolErrorCallbacks(toolCtx, tool, fArgs, err)
	}
	if errorResponse != nil || cbErr != nil {
		response = errorResponse
		err = cbErr
	}

	var alteredResponse map[string]any
	var alteredErr error
	if pluginManager != nil {
		alteredResponse, alteredErr = pluginManager.RunAfterToolCallback(toolCtx, tool, fArgs, response, err)
	}
	if alteredResponse == nil && alteredErr == nil {
		alteredResponse, alteredErr = f.invokeAfterToolCallbacks(toolCtx, tool, fArgs, response, err)
	}
	if alteredResponse != nil || alteredErr != nil {
		response = alteredResponse
		err = alteredErr
	}

	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	return response
}

func (f *Flow) invokeBeforeToolCallbacks(toolCtx tool.Context, tool tool.Tool, fArgs map[string]any) (map[string]any, error) {
	for _, callback := range f.BeforeToolCallbacks {
		result, err := callback(toolCtx, tool, fArgs)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	return nil, nil
}

func (f *Flow) invokeAfterToolCallbacks(toolCtx tool.Context, tool toolinternal.FunctionTool, fArgs, fResult map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range f.AfterToolCallbacks {
		result, err := callback(toolCtx, tool, fArgs, fResult, fErr)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	// If no callback returned a result/error, return the original result/error.
	return fResult, fErr
}

func (f *Flow) invokeOnToolErrorCallbacks(toolCtx tool.Context, tool tool.Tool, fArgs map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range f.OnToolErrorCallbacks {
		result, err := callback(toolCtx, tool, fArgs, fErr)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	// If no callback returned a result/error, return the original result/error.
	return nil, fErr
}

func mergeParallelFunctionResponseEvents(events []*session.Event) (*session.Event, error) {
	switch len(events) {
	case 0:
		return nil, nil
	case 1:
		return events[0], nil
	}
	var parts []*genai.Part
	var actions *session.EventActions
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		parts = append(parts, ev.LLMResponse.Content.Parts...)
		actions = mergeEventActions(actions, &ev.Actions)
	}
	// reuse events[0]
	ev := events[0]
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role:  "user",
			Parts: parts,
		},
	}
	ev.Actions = *actions
	return ev, nil
}

func mergeEventActions(base, other *session.EventActions) *session.EventActions {
	// flows/llm_flows/functions.py merge_parallel_function_response_events
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
		base.StateDelta = deepMergeMap(base.StateDelta, other.StateDelta)
	}
	return base
}

func deepMergeMap(dst, src map[string]any) map[string]any {
	if dst == nil {
		dst = make(map[string]any)
	}
	for key, value := range src {
		if srcMap, ok := value.(map[string]any); ok {
			if dstMap, ok := dst[key].(map[string]any); ok {
				dst[key] = deepMergeMap(dstMap, srcMap)
				continue
			}
		}
		dst[key] = value
	}
	return dst
}

func pluginManagerFromContext(ctx context.Context) pluginManager {
	m, ok := ctx.Value(plugincontext.PluginManagerCtxKey).(pluginManager)
	if !ok {
		return nil
	}
	return m
}

type pluginManager interface {
	RunBeforeModelCallback(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)
	RunAfterModelCallback(cctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)
	RunOnModelErrorCallback(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)
	RunBeforeToolCallback(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error)
	RunAfterToolCallback(ctx tool.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error)
	RunOnToolErrorCallback(ctx tool.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error)
}
