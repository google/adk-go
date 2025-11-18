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

// Package functiontool provides a tool that wraps a Go function.
package functiontool

import (
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// FunctionTool: borrow implementation from MCP go.

// Config is the input to the NewFunctionTool function.
type Config struct {
	// The name of this tool.
	Name string
	// A human-readable description of the tool.
	Description string
	// An optional JSON schema object defining the expected parameters for the tool.
	// If it is nil, FunctionTool tries to infer the schema based on the handler type.
	InputSchema *jsonschema.Schema
	// An optional JSON schema object defining the structure of the tool's output.
	// If it is nil, FunctionTool tries to infer the schema based on the handler type.
	OutputSchema *jsonschema.Schema
	// IsLongRunning makes a FunctionTool a long-running operation.
	IsLongRunning bool
}

// Func represents a Go function that can be wrapped in a tool.
// It takes a tool.Context and a generic argument type, and returns a generic result type.
type Func[TArgs, TResults any] func(tool.Context, TArgs) (TResults, error)

// New creates a new tool with a name, description, and the provided handler.
// Input schema is automatically inferred from the input and output types.
func New[TArgs, TResults any](cfg Config, handler Func[TArgs, TResults]) (tool.Tool, error) {
	// TODO: How can we improve UX for functions that does not require an argument, returns a simple type value, or returns a no result?
	//  https://github.com/modelcontextprotocol/go-sdk/discussions/37
	ischema, err := resolvedSchema[TArgs](cfg.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to infer input schema: %w", err)
	}
	oschema, err := resolvedSchema[TResults](cfg.OutputSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to infer output schema: %w", err)
	}

	return &functionTool[TArgs, TResults]{
		cfg:          cfg,
		inputSchema:  ischema,
		outputSchema: oschema,
		handler:      handler,
	}, nil
}

// functionTool wraps a Go function.
type functionTool[TArgs, TResults any] struct {
	cfg Config

	// A JSON Schema object defining the expected parameters for the tool.
	inputSchema *jsonschema.Resolved
	// A JSON Schema object defining the result of the tool.
	outputSchema *jsonschema.Resolved

	// handler is the Go function.
	handler Func[TArgs, TResults]
}

// Description implements tool.Tool.
func (f *functionTool[TArgs, TResults]) Description() string {
	return f.cfg.Description
}

// Name implements tool.Tool.
func (f *functionTool[TArgs, TResults]) Name() string {
	return f.cfg.Name
}

// IsLongRunning implements tool.Tool.
func (f *functionTool[TArgs, TResults]) IsLongRunning() bool {
	return f.cfg.IsLongRunning
}

// ProcessRequest packs the function tool's declaration into the LLM request.
func (f *functionTool[TArgs, TResults]) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, f)
}

// FunctionDeclaration implements interfaces.FunctionTool.
func (f *functionTool[TArgs, TResults]) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name:        f.Name(),
		Description: f.Description(),
	}
	if f.inputSchema != nil {
		decl.ParametersJsonSchema = f.inputSchema.Schema()
	}
	if f.outputSchema != nil {
		decl.ResponseJsonSchema = f.outputSchema.Schema()
	}

	if f.cfg.IsLongRunning {
		instruction := "NOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status."
		if decl.Description != "" {
			decl.Description += "\n\n" + instruction
		} else {
			decl.Description = instruction
		}
	}

	return decl
}

// Run executes the tool with the provided context and yields events.
func (f *functionTool[TArgs, TResults]) Run(ctx tool.Context, args any) (map[string]any, error) {
	// TODO: Handle function call request from tc.InvocationContext.
	// TODO: Handle panic -> convert to error.
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type, got: %T", args)
	}
	input, err := typeutil.ConvertToWithJSONSchema[map[string]any, TArgs](m, f.inputSchema)
	if err != nil {
		return nil, err
	}
	output, err := f.handler(ctx, input)
	if err != nil {
		return nil, err
	}
	resp, err := typeutil.ConvertToWithJSONSchema[TResults, map[string]any](output, f.outputSchema)
	if err == nil { // all good
		return resp, nil
	}

	// Specs requires the result to be a map (dict in python). python impl allows basic types when building response event
	// functions.py __build_response_event does the following
	// if not isinstance(function_result, dict):
	// 		function_result = {'result': function_result}
	if f.outputSchema != nil {
		if err1 := f.outputSchema.Validate(output); err1 != nil {
			return resp, err // if it fails propagate original err.
		}
	}
	wrappedOutput := map[string]any{"result": output}
	return wrappedOutput, nil
}

// ** NOTE FOR REVIEWERS **
// Initially I started to borrow the design of the MCP ServerTool and
// ToolHandlerFor/ToolHandler [1], but got diverged.
//  * MCP ServerTool provides direct access to mcp.CallToolResult message
//    but we expect Function in our case is a simple wrapper around a Go
//    function, and does not need to worry about how the result is translated
//    in genai.Content.
//  * Function returns only TResults, not (TResults, error). If the user
//    function can return an error, that needs to be included in the output
//    json schema. And for function that never returns an error, I think it
//    gets less uglier.
//  * MCP ToolHandler expects mcp.ServerSession. types.ToolContext may be close
//    to it, but we don't need to expose this to user function
//    (similar to ADK Python FunctionTool [2])
// References
//  [1] MCP SDK https://pkg.go.dev/github.com/modelcontextprotocol/go-sdk@v0.0.0-20250625213837-ff0d746521c4/mcp#ToolHandler
//  [2] ADK Python https://github.com/google/adk-python/blob/04de3e197d7a57935488eb7bfa647c7ab62cd9d9/src/google/adk/tools/function_tool.py#L110-L112

func resolvedSchema[T any](override *jsonschema.Schema) (*jsonschema.Resolved, error) {
	// TODO: check if override schema is compatible with T.
	if override != nil {
		return override.Resolve(nil)
	}
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, err
	}
	return schema.Resolve(nil)
}
