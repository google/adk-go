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

// Package geminitool provides access to Gemini native tools. You can use any
// tool from genai with geminitool.New().
//
// For example, to create a Gemini retrieval tool:
//
//	geminitool.New("data_retrieval", &genai.Tool{
//		Retrieval: &genai.Retrieval{
//			ExternalAPI: &genai.ExternalAPI{
//				Endpoint: ,
//				AuthConfig:
//			},
//		},
//	})
//
// Package also provides default tools like GoogleSearch.
package geminitool

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// New creates  gemini API tool.
func New(name string, t *genai.Tool) tool.Tool {
	return &geminiTool{
		name:  name,
		value: t,
	}
}

// geminiTool is a wrapper around a genai.Tool.
type geminiTool struct {
	name  string
	value *genai.Tool
}

// ProcessRequest adds the Gemini tool to the LLM request.
func (t *geminiTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return setTool(req, t.value)
}

// Name implements tool.Tool.
func (t *geminiTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *geminiTool) Description() string {
	return "Performs a Google search to retrieve information from the web."
}

// IsLongRunning implements tool.Tool.
func (t *geminiTool) IsLongRunning() bool {
	return false
}

func setTool(req *model.LLMRequest, t *genai.Tool) error {
	if req == nil {
		return fmt.Errorf("llm request is nil")
	}

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	req.Config.Tools = append(req.Config.Tools, t)
	return nil
}
