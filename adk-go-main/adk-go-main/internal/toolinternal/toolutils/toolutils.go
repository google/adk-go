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

// Package tool defines internal-only interfaces and logic for tools.
package toolutils

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

type Tool interface {
	Name() string
	Declaration() *genai.FunctionDeclaration
}

// The PackTool ensures that in case there is a usage of multiple function tools,
// all of them are consolidated into one genai tool that has all the function declarations
// provided by the tools. So, if there is already a tool with a function declaration,
// it appends another to it; otherwise, it creates a new genai tool.
func PackTool(req *model.LLMRequest, tool Tool) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := tool.Name()

	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = tool

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if decl := tool.Declaration(); decl == nil {
		return nil
	}
	// Find an existing genai.Tool with FunctionDeclarations
	var funcTool *genai.Tool
	for _, tool := range req.Config.Tools {
		if tool != nil && tool.FunctionDeclarations != nil {
			funcTool = tool
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{tool.Declaration()},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, tool.Declaration())
	}
	return nil
}
