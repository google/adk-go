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
	"github.com/google/adk-go/internal/typeutil"
	"google.golang.org/genai"
)

// basicRequestProcessor populates the LLMRequest
// with the agent's LLM generation configs.
func basicRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// reference: adk-python src/google/adk/flows/llm_flows/basic.py

	llmAgent := asLLMAgent(parentCtx.Agent)
	if llmAgent == nil {
		return nil // do nothing.
	}
	req.Model = llmAgent.Model
	req.GenerateConfig = typeutil.Clone(llmAgent.GenerateContentConfig)
	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}
	if llmAgent.OutputSchema != nil {
		req.GenerateConfig.ResponseSchema = llmAgent.OutputSchema
		req.GenerateConfig.ResponseMIMEType = "application/json"
	}
	// TODO: missing features
	//  populate LLMRequest LiveConnectConfig setting
	return nil
}

// asLLMAgent returns LLMAgent if agent is LLMAgent. Otherwise, nil.
func asLLMAgent(agent adk.Agent) *LLMAgent {
	if agent == nil {
		return nil
	}
	if llmAgent, ok := agent.(*LLMAgent); ok {
		return llmAgent
	}
	return nil
}
