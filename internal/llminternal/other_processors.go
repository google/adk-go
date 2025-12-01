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
	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/planner"
)

func identityRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/identity.py)
	return nil
}

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	p := getPlanner(ctx)
	if p == nil {
		return nil
	}

	switch planner := p.(type) {
	case *planner.BuiltInPlanner:
		planner.ApplyThinkingConfig(req)
	case *planner.ReActPlanner:
		readonlyContext := icontext.NewReadonlyContext(ctx)
		if planningInstruction := planner.BuildPlanningInstruction(readonlyContext, req); planningInstruction != "" {
			utils.AppendInstructions(req, planningInstruction)
		}

		for _, content := range req.Contents {
			if content.Parts == nil {
				continue
			}
			for _, part := range content.Parts {
				part.Thought = false
			}
		}
	default:
		return nil
	}
	return nil
}

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return nil
}

func authPreprocessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/auth/auth_preprocessor.py)
	return nil
}

func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
		return nil
	}

	p := getPlanner(ctx)
	if p == nil {
		return nil
	}

	// Skip built-in planner response processing
	if _, ok := p.(*planner.BuiltInPlanner); ok {
		return nil
	}

	callbackContext := icontext.NewCallbackContext(ctx)
	if processedParts := p.ProcessPlanningResponse(callbackContext, resp.Content.Parts); processedParts != nil {
		resp.Content.Parts = processedParts
	}
	return nil
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}

// getPlanner returns the planner from the invocation context, or nil if no planner is available.
func getPlanner(ctx agent.InvocationContext) planner.BasePlanner {
	if llmAgent, ok := ctx.Agent().(Agent); ok {
		state := Reveal(llmAgent)
		return state.Planner
	}

	return nil
}
