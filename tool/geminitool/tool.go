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

	"google.golang.org/adk/llm"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// New creates  gemini API tool.
func New(name string, t *genai.Tool) tool.Tool {
	return &geminiTool{
		name:  name,
		value: t,
	}
}

type geminiTool struct {
	name  string
	value *genai.Tool
}

func (t *geminiTool) ProcessRequest(ctx tool.Context, req *llm.Request) error {
	if req == nil {
		return fmt.Errorf("llm request is nil")
	}

	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}

	req.GenerateConfig.Tools = append(req.GenerateConfig.Tools, t.value)

	return nil
}

func (t *geminiTool) Name() string {
	return t.name
}

func (t *geminiTool) Description() string {
	return t.name
}
