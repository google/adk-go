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

package tool

import (
	"fmt"
	"strings"

	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

// GoogleSearchTool is a tool that adds google search configuration to the LLM request.
type GoogleSearchTool struct {
	name        string
	description string
	// TODO: temporary the model is the input for the tool
	model llm.Model
}

// Assert that GoogleSearchTool implements adk.Tool
var _ Tool = (*GoogleSearchTool)(nil)

// NewGoogleSearchTool creates a new GoogleSearchTool.
func NewGoogleSearchTool(model llm.Model) *GoogleSearchTool {
	return &GoogleSearchTool{
		name:        "google_search",
		description: "google_search",
		model:       model,
	}
}

// Name implements adk.Tool.
func (t *GoogleSearchTool) Name() string {
	return t.name
}

// Description implements adk.Tool.
func (t *GoogleSearchTool) Description() string {
	return t.description
}

func (t *GoogleSearchTool) Declaration() *genai.FunctionDeclaration {
	return nil
}

// ProcessRequest modifies the LLM request to include the google search tool configuration.
func (t *GoogleSearchTool) ProcessRequest(ctx Context, req *llm.Request) error {
	if req == nil {
		return fmt.Errorf("llm request is nil")
	}

	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}

	var tool *genai.Tool

	if strings.HasPrefix(t.model.Name(), "gemini-1") {
		if len(req.GenerateConfig.Tools) > 0 {
			return fmt.Errorf("google search tool cannot be used with other tools in Gemini 1.x")
		}
		tool = &genai.Tool{
			GoogleSearchRetrieval: &genai.GoogleSearchRetrieval{},
		}
	} else if strings.HasPrefix(t.model.Name(), "gemini-2") {
		tool = &genai.Tool{
			GoogleSearch: &genai.GoogleSearch{},
		}
	} else {
		return fmt.Errorf("google search tool is not supported for model %s", t.model)
	}

	req.GenerateConfig.Tools = append(req.GenerateConfig.Tools, tool)
	return nil
}

// Run is not implemented for this tool, as it's an internal model tool.
func (t *GoogleSearchTool) Run(ctx Context, args any) (any, error) {
	return nil, fmt.Errorf("google search tool runs internally on the model, it can not be run directly")
}
