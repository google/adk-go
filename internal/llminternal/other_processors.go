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
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
)

// identityRequestProcessor sets up identity context for LLM agents in the LLM request.
// It adds system instructions that inform the LLM about the agent's name and description.
func identityRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// reference: adk-python src/google/adk/flows/llm_flows/identity.py

	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil // do nothing.
	}

	// Add identity information to system instructions.
	identityInstructions := make([]string, 0, 2)
	if name := ctx.Agent().Name(); name != "" {
		identityInstructions = append(identityInstructions, fmt.Sprintf(`You are an agent. Your internal name is %q.`, name))
	}
	if description := ctx.Agent().Description(); description != "" {
		identityInstructions = append(identityInstructions, fmt.Sprintf(`The description about you is %q.`, description))
	}

	// Append identity instructions to the system instruction.
	utils.AppendInstructions(req, identityInstructions...)
	return nil
}

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_nl_plnning.py)
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
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return nil
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}
