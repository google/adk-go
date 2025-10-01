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

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// longRunningFunctionTool wraps a Go function.
type longRunningFunctionTool[TArgs, TResults any] struct {
	functionTool *functionTool[TArgs, TResults]
}

// Description implements tool.Tool.
func (f *longRunningFunctionTool[TArgs, TResults]) Description() string {
	return f.functionTool.Description()
}

// Name implements tool.Tool.
func (f *longRunningFunctionTool[TArgs, TResults]) Name() string {
	return f.functionTool.Name()
}

// IsLongRunning implements tool.Tool.
func (f *longRunningFunctionTool[TArgs, TResults]) IsLongRunning() bool {
	return f.functionTool.IsLongRunning()
}

// ProcessRequest implements interfaces.RequestProcessor.
func (f *longRunningFunctionTool[TArgs, TResults]) ProcessRequest(ctx Context, req *model.LLMRequest) error {
	return f.functionTool.ProcessRequest(ctx, req)
}

// FunctionDeclaration implements interfaces.FunctionTool.
func (f *longRunningFunctionTool[TArgs, TResults]) Declaration() *genai.FunctionDeclaration {
	declaration := f.functionTool.Declaration()
	if declaration != nil {
		instruction := "NOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status."
		if declaration.Description != "" {
			declaration.Description += "\n\n" + instruction
		} else {
			declaration.Description = instruction
		}
	}
	return declaration
}

// Run executes the tool with the provided context and yields events.
func (f *longRunningFunctionTool[TArgs, TResults]) Run(ctx Context, args any) (any, error) {
	return f.functionTool.Run(ctx, args)
}

// NewLongRunningFunctionTool creates a new tool with a name, description, and the provided handler.
// Input schema is automatically inferred from the input and output types.
func NewLongRunningFunctionTool[TArgs, TResults any](cfg FunctionToolConfig, handler Function[TArgs, TResults]) (Tool, error) {
	cfg.isLongRunning = true
	innerTool, err := NewFunctionTool(cfg, handler)
	if err != nil {
		return nil, err
	}

	// Use a type assertion to get the concrete type from the interface.
	concreteTool, ok := innerTool.(*functionTool[TArgs, TResults])
	if !ok {
		// This is a safeguard. It should not happen if NewFunctionTool
		// always returns the expected type, but it prevents panics.
		return nil, fmt.Errorf("internal error: NewFunctionTool returned an unexpected type")
	}

	return &longRunningFunctionTool[TArgs, TResults]{
		functionTool: concreteTool,
	}, nil
}

var _ Tool = (*longRunningFunctionTool[any, any])(nil)
