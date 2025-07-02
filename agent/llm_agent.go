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
	// such asfunction tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	RootAgent adk.Agent
	SubAgents []adk.Agent

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

	if a.DisallowTransferToParent && a.DisallowTransferToPeers && len(a.SubAgents) == 0 {
		flow, err := newSingleFlow(parentCtx)
		if err != nil {
			return nil, err
		}
		return flow.Run(ctx, parentCtx), nil
	} else {
		panic("not implemented")
	}
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
			// adk-python src/google/adk/flos/llm_flows/base_llm_flow.py BaseLlmFlow._postprocess_sync.
			if resp.Content == nil && resp.ErrorCode == 0 && !resp.Interrupted {
				continue
			}
			// Build the event.
			ev := adk.NewEvent(parentCtx.InvocationID)
			ev.Author = parentCtx.Agent.Name()
			ev.Branch = parentCtx.Branch
			ev.LLMResponse = resp

			if !yield(ev, nil) {
				return
			}

			// TODO: populate ev.LongRunningToolIDs (see BaseLlmFlow._finalize_model_response_event)
			// TODO: handle function calls (postprocessFunctionCalls)
		}
	}
}

func (f *baseFlow) preprocess(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// apply request processor functions to the request in the configured order.
	for _, processor := range f.RequestProcessors {
		if err := processor(ctx, parentCtx, req); err != nil {
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
