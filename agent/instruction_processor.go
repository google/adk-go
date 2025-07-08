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

	"github.com/google/adk-go"
)

// instructionsRequestProcessor configures req's instructions and global instructions for LLM flow.
func instructionsRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// reference: adk-python src/google/adk/flows/llm_flows/instructions.py

	llmAgent := asLLMAgent(parentCtx.Agent)
	if llmAgent == nil {
		return nil // do nothing.
	}
	rootAgent := asLLMAgent(rootAgent(llmAgent))
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
