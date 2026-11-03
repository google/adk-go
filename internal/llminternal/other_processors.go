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
	"iter"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type thinkingConfigApplier interface {
	ApplyThinkingConfig(req *model.LLMRequest)
}

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return
		}
		p := llmAgent.internal().Planner
		if p == nil {
			return
		}

		if applier, ok := p.(thinkingConfigApplier); ok {
			applier.ApplyThinkingConfig(req)
			return
		}

		if inst := p.BuildPlanningInstruction(icontext.NewReadonlyContext(ctx), req); inst != "" {
			utils.AppendInstructions(req, inst)
		}
		removeThoughtFromRequest(req)
	}
}

func removeThoughtFromRequest(req *model.LLMRequest) {
	if req == nil {
		return
	}
	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil {
				part.Thought = false
			}
		}
	}
}

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return func(yield func(*session.Event, error) bool) {}
}

func authPreprocessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/auth/auth_preprocessor.py)
	return func(yield func(*session.Event, error) bool) {}
}

func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
		return nil
	}
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil
	}
	p := llmAgent.internal().Planner
	if p == nil {
		return nil
	}
	if _, ok := p.(thinkingConfigApplier); ok {
		return nil
	}

	// ResponseProcessors return only an error and cannot emit a separate
	// state-delta event if ProcessPlanningResponse modifies session state,
	// unlike adk-python's async generator response processor.
	processedParts := p.ProcessPlanningResponse(icontext.NewCallbackContext(ctx), resp.Content.Parts)
	if len(processedParts) > 0 {
		resp.Content.Parts = processedParts
	}
	return nil
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}
