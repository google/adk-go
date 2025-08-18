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
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

// instructionsRequestProcessor configures req's instructions and global instructions for LLM flow.
func instructionsRequestProcessor(ctx agent.Context, req *llm.Request) error {
	// reference: adk-python src/google/adk/flows/llm_flows/instructions.py

	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil // do nothing.
	}

	parents := parentmap.FromContext(ctx)

	rootAgent := asLLMAgent(parents.RootAgent(ctx.Agent()))
	if rootAgent == nil {
		rootAgent = llmAgent
	}

	// Append global instructions if set.
	if rootAgent != nil && rootAgent.internal().GlobalInstruction != "" {
		// TODO: apply instructions_utils.inject_session_state
		appendInstructions(req, llmAgent.internal().GlobalInstruction)
	}

	// Append agent's instruction
	if llmAgent.internal().Instruction != "" {
		// TODO: apply instructions_utils.inject_session_state
		appendInstructions(req, llmAgent.internal().Instruction)
	}

	return nil
}

func appendInstructions(r *llm.Request, instructions ...string) {
	if len(instructions) == 0 {
		return
	}

	inst := strings.Join(instructions, "\n\n")

	if r.GenerateConfig == nil {
		r.GenerateConfig = &genai.GenerateContentConfig{}
	}
	if current := r.GenerateConfig.SystemInstruction; current != nil && len(current.Parts) > 0 && current.Parts[0].Text != "" {
		r.GenerateConfig.SystemInstruction = genai.NewContentFromText(current.Parts[0].Text+"\n\n"+inst, "")
	} else {
		r.GenerateConfig.SystemInstruction = genai.NewContentFromText(inst, "")
	}
}
