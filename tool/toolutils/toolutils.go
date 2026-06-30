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

// Package toolutils provides public helpers for packing tool declarations
// into a model.LLMRequest. It allows external code to consolidate tool
// function declarations the same way the built-in ADK tools do, without
// re-implementing the logic.
package toolutils

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// Packable is implemented by any tool that can be packed into an
// model.LLMRequest via PackTool.
type Packable interface {
	Name() string
	Declaration() *genai.FunctionDeclaration
}

// PackTool ensures that in case there is a usage of multiple function tools,
// all of them are consolidated into one genai tool that has all the function declarations
// provided by the tools. So, if there is already a tool with a function declaration,
// it appends another to it; otherwise, it creates a new genai tool.
func PackTool(req *model.LLMRequest, t Packable) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := t.Name()

	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if decl := t.Declaration(); decl == nil {
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
			FunctionDeclarations: []*genai.FunctionDeclaration{t.Declaration()},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, t.Declaration())
	}
	return nil
}
