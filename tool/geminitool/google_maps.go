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
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
)

// GoogleMaps is a built-in tool that is automatically invoked by Gemini 2
// models to retrieve search results from Google Maps.
// The tool operates internally within the model and does not require or
// perform local code execution.
type GoogleMaps struct{}

// Name implements tool.Tool.
func (m GoogleMaps) Name() string {
	return "google_maps"
}

// Description implements tool.Tool.
func (m GoogleMaps) Description() string {
	return "Performs a Google map search to retrieve geographical information from the web."
}

// ProcessRequest adds the GoogleSearch tool to the LLM request.
func (m GoogleMaps) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if !googlellm.IsGeminiModel(req.Model) {
		return fmt.Errorf("cannot process: Google maps tool is not supported for model %v", req.Model)
	}
	if googlellm.IsGemini1Model(req.Model) {
		return fmt.Errorf("cannot process: Google maps tool cannot be used with other tools in Gemini 1.x")
	}
	return setTool(req, &genai.Tool{
		GoogleMaps: &genai.GoogleMaps{},
	})
}

// IsLongRunning implements tool.Tool.
func (m GoogleMaps) IsLongRunning() bool {
	return false
}
