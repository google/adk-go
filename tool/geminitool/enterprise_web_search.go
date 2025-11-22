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
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// EnterpriseWebSearch is a built-in tool that is automatically invoked by
// Gemini 2 models to perform web grounding with enterprise compliance.
// The tool operates internally within the model and does not require or
// perform local code execution.
type EnterpriseWebSearch struct{}

// Name implements tool.Tool.
func (s EnterpriseWebSearch) Name() string {
	return "enterprise_web_search"
}

// Description implements tool.Tool.
func (s EnterpriseWebSearch) Description() string {
	return "Performs web search with enterprise compliance and security features."
}

// ProcessRequest adds the EnterpriseWebSearch tool to the LLM request.
func (s EnterpriseWebSearch) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return setTool(req, &genai.Tool{
		EnterpriseWebSearch: &genai.EnterpriseWebSearch{},
	})
}

// IsLongRunning implements tool.Tool.
func (t EnterpriseWebSearch) IsLongRunning() bool {
	return false
}
