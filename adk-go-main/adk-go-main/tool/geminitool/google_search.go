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

// GoogleSearch is a built-in tool that is automatically invoked by Gemini 2
// models to retrieve search results from Google Search.
// The tool operates internally within the model and does not require or
// perform local code execution.
type GoogleSearch struct{}

// Name implements tool.Tool.
func (s GoogleSearch) Name() string {
	return "google_search"
}

// Description implements tool.Tool.
func (s GoogleSearch) Description() string {
	return "Performs a Google search to retrieve information from the web."
}

// ProcessRequest adds the GoogleSearch tool to the LLM request.
func (s GoogleSearch) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return setTool(req, &genai.Tool{
		GoogleSearch: &genai.GoogleSearch{},
	})
}

// IsLongRunning implements tool.Tool.
func (t GoogleSearch) IsLongRunning() bool {
	return false
}
