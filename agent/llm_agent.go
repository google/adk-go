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
	"bytes"
	"context"
	"fmt"
	"reflect"
	"slices"
	"text/template"

	"github.com/google/adk-go"
	"github.com/google/uuid"
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
	// such asfunction tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	RootAgent   adk.Agent
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

func (a *LLMAgent) Name() string        { return a.AgentName }
func (a *LLMAgent) Description() string { return a.AgentDescription }
func (a *LLMAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
	// TODO: Select model (LlmAgent.canonical_model)
	// TODO: Autoflow Run.

	flow, err := func() (*baseFlow, error) {
		if a.DisallowTransferToParent && a.DisallowTransferToPeers && len(a.SubAgents) == 0 {
			return newSingleFlow(parentCtx)
		}
		return newAutoFlow(parentCtx)
	}()
	if err != nil {
		return nil, err
	}
	return flow.Run(ctx, parentCtx), nil
}

var _ adk.Agent = (*LLMAgent)(nil)

// TODO: Do we want to abstract "Flow" too?

func newSingleFlow(parentCtx *adk.InvocationContext) (*baseFlow, error) {
	llmAgent, ok := parentCtx.Agent.(*LLMAgent)
	if !ok {
		return nil, fmt.Errorf("invalid agent type: %+T", parentCtx.Agent)
	}
	return &baseFlow{
		Model:              llmAgent.Model,
		RequestProcessors:  singleFlowRequestProcessors,
		ResponseProcessors: singleFlowResponseProcessors,
	}, nil
}

/*
	newAutoFlow returns an AutoFlow that is SingleFlow with agent transfer capability.

Agent transfer is allowed in the following direction:

 1. from parent to sub-agent;

 2. from sub-agent to parent;

 3. from sub-agent to its peer agents;

    For peer-agent transfers, it's only enabled when all below conditions are met:

    - The parent agent is also of AutoFlow;
    - `disallow_transfer_to_peer` option of this agent is False (default).

Depending on the target agent flow type, the transfer may be automatically
reversed. The condition is as below:

  - If the flow type of the tranferee agent is also auto, transfee agent will
    remain as the active agent. The transfee agent will respond to the user's
    next message directly.
  - If the flow type of the transfere agent is not auto, the active agent will
    be reversed back to previous agent.
*/
func newAutoFlow(parentCtx *adk.InvocationContext) (*baseFlow, error) {
	llmAgent, ok := parentCtx.Agent.(*LLMAgent)
	if !ok {
		return nil, fmt.Errorf("invalid agent type: %+T", parentCtx.Agent)
	}
	return &baseFlow{
		Model:              llmAgent.Model,
		RequestProcessors:  autoFlowRequestProcessors,
		ResponseProcessors: singleFlowResponseProcessors,
	}, nil
}

var (
	singleFlowRequestProcessors = []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error{
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
	}
	singleFlowResponseProcessors = []func(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error{
		assignMissingFunctionIDProcessor,
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
	autoFlowRequestProcessors = append(slices.Clone(singleFlowRequestProcessors), agentTransferRequestProcessor)
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
				yield(ev, nil)
				lastEvent = ev
			}
			if lastEvent == nil || lastEvent.IsFinalResponse() {
				return
			}
			if lastEvent.LLMResponse.Partial {
				// TODO: hanle this in Model level.
				yield(nil, fmt.Errorf("TODO: last event shouldn't be partial. LLM max output limit may be reached"))
				return
			}
		}

		// TODO: handle Partial response event - LLM max output limit may be reached.
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
			// Skip the model response event if there is no content and no error code.
			// This is needed for the code executor to trigger another loop according to
			// adk-python src/google/adk/flos/llm_flows/base_llm_flow.py BaseLlmFlow._postprocess_sync.
			if resp.Content == nil && resp.ErrorCode == 0 && !resp.Interrupted {
				continue
			}
			if err := f.postprocess(ctx, parentCtx, req, resp); err != nil {
				yield(nil, err)
				return
			}
			// Build the event.
			ev := adk.NewEvent(parentCtx.InvocationID)
			ev.Author = parentCtx.Agent.Name()
			ev.Branch = parentCtx.Branch
			ev.LLMResponse = resp
			ev.LongRunningToolIDs = collectLongRunningToolIDs(req, resp)

			// TODO: yield and run function calls???
			if !yield(ev, nil) {
				return
			}

			// Handles function calls.
			fnCallEvent, err := f.handleFunctionCalls(ctx, parentCtx, ev, req)
			if err != nil {
				yield(nil, err)
				return
			}
			if fnCallEvent == nil {
				return
			}
			if !yield(fnCallEvent, nil) {
				return
			}
			// auth_event (BaseLlmFlow._postprocess_handle_function_calls_async)

			// TODO: check ev.Actions.TransferToAgent
			// TODO: handle function calls (postprocessFunctionCalls)
		}
	}
}

func (f *baseFlow) handleFunctionCalls(ctx context.Context, parentCtx *adk.InvocationContext, ev *adk.Event, req *adk.LLMRequest) (*adk.Event, error) {
	resp := ev.LLMResponse
	llmAgent := asLLMAgent(parentCtx.Agent)
	if resp == nil || resp.Content == nil || req == nil || len(llmAgent.Tools) == 0 {
		return nil, nil
	}

	var fns []*genai.FunctionCall
	knownTools := map[string]adk.Tool{}
	for _, tool := range llmAgent.Tools {
		knownTools[tool.Name()] = tool
	}
	for _, p := range resp.Content.Parts {
		if p.FunctionCall == nil {
			continue
		}
		if _, ok := knownTools[p.FunctionCall.Name]; ok {
			fns = append(fns, p.FunctionCall)
		}
	}
	if len(fns) == 0 {
		return nil, nil
	}
	functionResponseEvents := make([]*adk.Event, 0, len(fns))

	// TODO: run fns in parallel?
	for _, fn := range fns {
		tool, ok := knownTools[fn.Name]
		if !ok {
			return nil, fmt.Errorf("unknown tool: %s", fn.Name)
		}
		toolContext := &adk.ToolContext{
			InvocationContext: parentCtx,
			FunctionCallID:    fn.ID,
		}
		// TODO: before_tool_callback.

		result, err := tool.Run(ctx, toolContext, fn.Args)
		if err != nil {
			return nil, fmt.Errorf("tool %s failed to run: %v", tool.Name(), err)
			// TODO: should we include the fn.Args in the error? (that may have sensitive info)
		}
		// TODO: after_tool_callback.

		// TODO: handle long-running tool.

		ev := adk.NewEvent(parentCtx.InvocationID)

		ev.LLMResponse = &adk.LLMResponse{
			Content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       fn.ID,
							Name:     fn.Name,
							Response: result,
						},
					},
				},
			},
		}
		ev.Author = parentCtx.Agent.Name()
		ev.Branch = parentCtx.Branch
		ev.Actions = toolContext.EventActions

		functionResponseEvents = append(functionResponseEvents, ev)
	}
	return mergeParallelFunctionResponseEvents(functionResponseEvents)
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

func generateClientFunctionCallID() string {
	return fmt.Sprintf("adk-%s", uuid.New().String())
}

func collectLongRunningToolIDs(req *adk.LLMRequest, resp *adk.LLMResponse) []string {
	var longRunningTools []string

	knownTools := req.Tools
	if len(knownTools) == 0 {
		return nil
	}
	if resp.Content == nil {
		return nil
	}
	for _, part := range resp.Content.Parts {
		if part.FunctionCall == nil {
			continue
		}
		tool, ok := knownTools[part.FunctionCall.Name]
		if !ok {
			continue
		}
		if _, ok := tool.(adk.LongRunningTool); !ok {
			continue
		}
		longRunningTools = append(longRunningTools, part.FunctionCall.ID)
	}
	return longRunningTools
}

func (f *baseFlow) preprocess(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	llmAgent := asLLMAgent(parentCtx.Agent)

	// apply request processor functions to the request in the configured order.
	for _, processor := range f.RequestProcessors {
		if err := processor(ctx, parentCtx, req); err != nil {
			return err
		}
	}
	// Run processors for tools.
	for _, tool := range llmAgent.Tools {
		if err := tool.ProcessRequest(ctx, &adk.ToolContext{InvocationContext: parentCtx}, req); err != nil {
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

		// TODO: RunLive mode when invocation_context.run_config.support_ctc is true.

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

func basicRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	llmAgent, ok := parentCtx.Agent.(*LLMAgent)
	if !ok {
		return fmt.Errorf("invalid agent type: %+T", parentCtx.Agent)
	}
	req.Model = llmAgent.Model
	req.GenerateConfig = clone(llmAgent.GenerateContentConfig)
	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}
	if llmAgent.OutputSchema != nil {
		req.GenerateConfig.ResponseSchema = llmAgent.OutputSchema
		req.GenerateConfig.ResponseMIMEType = "application/json"
	}
	return nil
}

func clone[M any](src M) M {
	val := reflect.ValueOf(src)

	// Handle nil pointers
	if val.Kind() == reflect.Ptr && val.IsNil() {
		var zero M
		return zero
	}

	// Dereference pointer to get the underlying value
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	// Create a new instance of the same type
	newVal := reflect.New(val.Type()).Elem()

	// Recursively copy fields
	deepCopy(val, newVal)

	// Return as the original type
	return newVal.Addr().Interface().(M)
}

func deepCopy(src, dst reflect.Value) {
	switch src.Kind() {
	case reflect.Struct:
		for i := 0; i < src.NumField(); i++ {
			// Create a copy of the field and set it on the destination struct
			fieldCopy := reflect.New(src.Field(i).Type()).Elem()
			deepCopy(src.Field(i), fieldCopy)
			dst.Field(i).Set(fieldCopy)
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		for i := 0; i < src.Len(); i++ {
			// Create a copy of each element and set it in the new slice
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			deepCopy(src.Index(i), elemCopy)
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Map:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeMap(src.Type()))
		for _, key := range src.MapKeys() {
			// Create copies of the key and value and set them in the new map
			keyCopy := reflect.New(key.Type()).Elem()
			deepCopy(key, keyCopy)
			valCopy := reflect.New(src.MapIndex(key).Type()).Elem()
			deepCopy(src.MapIndex(key), valCopy)
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Ptr:
		if src.IsNil() {
			return
		}
		// Create a new pointer and deep copy the underlying value
		newPtr := reflect.New(src.Elem().Type())
		deepCopy(src.Elem(), newPtr.Elem())
		dst.Set(newPtr)
	default:
		// For basic types, direct assignment is sufficient
		dst.Set(src)
	}
}

func authPreprocesssor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/auth/auth_preprocessor.py)
	return nil
}

func asLLMAgent(agent adk.Agent) *LLMAgent {
	if agent == nil {
		return nil
	}
	if llmAgent, ok := agent.(*LLMAgent); ok {
		return llmAgent
	}
	return nil
}

// instructionsRequestProcessor configures req's instructions and global instructions for LLM flow.
func instructionsRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// reference: adk-python src/google/adk/instruction.py

	llmAgent, ok := parentCtx.Agent.(*LLMAgent)
	if !ok {
		return fmt.Errorf("invalid agent type: %+T", parentCtx.Agent)
	}
	rootAgent := asLLMAgent(llmAgent.RootAgent)
	if rootAgent == nil {
		rootAgent = llmAgent
	}

	// Append global instructions if set.
	if rootAgent != nil && rootAgent.GlobalInstruction != "" {
		// TODO: apply instructions_utils.inject_session_state
		req.AppendInstructions(rootAgent.GlobalInstruction)
	}

	// Append agent's instruction
	if llmAgent.Instruction != "" {
		// TODO: apply instructions_utils.inject_session_state
		req.AppendInstructions(llmAgent.Instruction)
	}

	return nil
}

func identityRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/ad/identity.py)
	return nil
}

func contentsRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/contents.py) - extract function call results, etc.

	return nil
}

func nlPlanningRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/_nl_plnning.py)
	return nil
}

func codeExecutionRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/_code_execution.py)
	return nil
}

// assignMissingFunctionIDPrcessor assigns function call id if it is not set.
func assignMissingFunctionIDProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error {
	if resp == nil || resp.Content == nil {
		return nil
	}
	for _, p := range resp.Content.Parts {
		if p.FunctionCall == nil || p.FunctionCall.ID != "" {
			continue
		}
		p.FunctionCall.ID = generateClientFunctionCallID()
	}
	return nil
}

func nlPlanningResponseProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return nil
}

func codeExecutionResponseProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest, resp *adk.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}

func agentTransferRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	agent := asLLMAgent(parentCtx.Agent)
	if agent == nil {
		return fmt.Errorf("invalid agent type: %+T", parentCtx.Agent)
	}
	targets := transferTarget(agent)
	si, err := buildInstructions(agent, targets)
	if err != nil {
		return err
	}
	req.AppendInstructions(si)
	//     transfer_to_agent_tool = FunctionTool(func=transfer_to_agent)
	// tool_context = ToolContext(invocation_context)
	// await transfer_to_agent_tool.process_llm_request(
	//    tool_context=tool_context, llm_request=llm_request
	//)
	return nil
}

func transferTarget(current *LLMAgent) []adk.Agent {
	var targets []adk.Agent
	targets = append(targets, current.SubAgents...)
	parent := asLLMAgent(current.ParentAgent)
	if parent == nil {
		return targets
	}
	if !current.DisallowTransferToParent {
		targets = append(targets, parent)
	}
	if !current.DisallowTransferToPeers {
		targets = append(targets, parent.SubAgents...)
	}
	return targets
}

func buildInstructions(agent *LLMAgent, targets []adk.Agent) (string, error) {
	tmpl, err := template.New("transfer_prompt").Parse(transferTemplate)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, struct {
		AgentName string
		Parent    adk.Agent
		Targets   []adk.Agent
	}{
		AgentName: agent.Name(),
		Parent:    agent.ParentAgent,
		Targets:   targets,
	}); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// Prompt source:
//  flows/llm_flows/agent_transfer.py _build_target_agents_instructions.

const transferTemplate = `You have a list of other agents to transfer to:
{{range .Targets}}
Agent name: {{.Name}}
Agent description: {{.Description}}
{{end}}
If you are the best to answer the question according to your description, you
can answer it.
If another agent is better for answering the question according to its
description, call '{{.AgentName}}' function to transfer the
question to that agent. When transfering, do not generate any text other than
the function call.
{{if .Parent}}
Your parent agent is {{.Parent.Name}}. If neither the other agents nor
you are best for answering the question according to the descriptions, transfer
to your parent agent. If you don't have parent agent, try answer by yourself.
{{end}}
`
