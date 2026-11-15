// Copyright 2026 Google LLC
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
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
)

// CodeExecution is a built-in tool that lets Gemini 2 models write and run
// Python code. The code runs inside the model service; the tool does not
// perform local code execution.
type CodeExecution struct{}

// Name implements tool.Tool.
func (c CodeExecution) Name() string {
	return "code_execution"
}

// Description implements tool.Tool.
func (c CodeExecution) Description() string {
	return "Lets the model write and run code to solve problems."
}

// ProcessRequest adds the code execution tool to the LLM request.
func (c CodeExecution) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if req == nil {
		return fmt.Errorf("llm request is nil")
	}
	if !googlellm.IsGeminiModel(req.Model) {
		return fmt.Errorf("cannot process: code execution tool is not supported for model %v", req.Model)
	}
	if googlellm.IsGemini1Model(req.Model) {
		return fmt.Errorf("cannot process: code execution tool cannot be used in Gemini 1.x")
	}
	return setTool(req, &genai.Tool{
		CodeExecution: &genai.ToolCodeExecution{},
	})
}

// IsLongRunning implements tool.Tool.
func (c CodeExecution) IsLongRunning() bool {
	return false
}
