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

// URLContext is a built-in tool that is automatically invoked by Gemini 2
// models to retrieve content from URLs and use that content to inform and
// shape its response.
// The tool operates internally within the model and does not require or
// perform local code execution.
type URLContext struct{}

// Name implements tool.Tool.
func (s URLContext) Name() string {
	return "url_context"
}

// Description implements tool.Tool.
func (s URLContext) Description() string {
	return "Retrieves content from URLs to inform and shape model responses."
}

// ProcessRequest adds the URLContext tool to the LLM request.
func (s URLContext) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return setTool(req, &genai.Tool{
		URLContext: &genai.URLContext{},
	})
}

// IsLongRunning implements tool.Tool.
func (t URLContext) IsLongRunning() bool {
	return false
}
