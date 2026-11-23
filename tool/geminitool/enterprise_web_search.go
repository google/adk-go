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

package geminitool

import (
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
)

// EnterpriseWebSearch is a built-in Gemini tool that grounds responses in web
// search results under Enterprise compliance controls. It is a model built-in,
// not a function tool: it is never executed locally, and before each model call
// it adds an enterprise_web_search entry to the request.
//
// It is not the same as Vertex AI Search (formerly "Enterprise Search"). See
// https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/web-grounding-enterprise.
//
// ProcessRequest enforces the model policy: a Gemini 1.x model that already
// offers another tool cannot use it, and a non-Gemini model does not support it.
type EnterpriseWebSearch struct{}

// Name implements tool.Tool.
func (EnterpriseWebSearch) Name() string {
	return "enterprise_web_search"
}

// Description implements tool.Tool.
func (EnterpriseWebSearch) Description() string {
	return "Performs web grounding for Enterprise compliance"
}

// IsLongRunning implements tool.Tool.
func (EnterpriseWebSearch) IsLongRunning() bool {
	return false
}

// ProcessRequest adds the enterprise_web_search entry to the request for a
// Gemini model. It mirrors adk-python v2.0.0
// (google/adk/tools/enterprise_search_tool.py): for a Gemini 1.x model whose
// request already offers any tool it fails, and for a non-Gemini model it
// reports the model as unsupported.
func (EnterpriseWebSearch) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if req == nil {
		return errors.New("llm request is nil")
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	if !googlellm.IsGeminiModel(req.Model) {
		return fmt.Errorf("cannot process: Enterprise Web Search tool is not supported for model %s", req.Model)
	}
	if googlellm.IsGemini1Model(req.Model) && len(req.Config.Tools) > 0 {
		return errors.New("cannot process: Enterprise Web Search tool cannot be used with other tools in Gemini 1.x")
	}

	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		EnterpriseWebSearch: &genai.EnterpriseWebSearch{},
	})
	return nil
}
