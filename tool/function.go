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
	"context"
	"fmt"

	"github.com/google/adk-go"
	"github.com/google/adk-go/internal/itype"
	"github.com/google/adk-go/internal/typeutil"
	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"google.golang.org/genai"
)

// FunctionTool: borrow implementation from MCP go.
// transfer_to_agent ??
// MCP Tool
// LoadArtifactsTool
// ExitLoopTool
// AgentTool
// LongRunningFunctionTool

// BuiltinCodeExecutionTool
// GoogeSearchTool
// MCPTool

// FunctionToolConfig is the input to the NewFunctionTool function.
type FunctionToolConfig struct {
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
}

// Funtion represents a Go function.
type Function[TArgs, TResults any] func(context.Context, TArgs) TResults

// NewFunctionTool creates a new tool with a name, description, and the provided handler.
// Input schema is automatically inferred from the input and output types.
func NewFunctionTool[TArgs, TResults any](cfg FunctionToolConfig, handler Function[TArgs, TResults]) (*FunctionTool[TArgs, TResults], error) {
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

	return &FunctionTool[TArgs, TResults]{
		cfg:          cfg,
		inputSchema:  ischema,
		outputSchema: oschema,
		handler:      handler,
	}, nil
}

// FunctionTool wraps a Go function.
type FunctionTool[TArgs, TResults any] struct {
	cfg FunctionToolConfig

	// A JSON Schema object defining the expected parameters for the tool.
	inputSchema *jsonschema.Resolved
	// A JSON Schema object defining the result of the tool.
	outputSchema *jsonschema.Resolved

	// handler is the Go function.
	handler Function[TArgs, TResults]
}

var _ adk.Tool = (*FunctionTool[any, any])(nil)
var _ itype.FunctionTool = (*FunctionTool[any, any])(nil)

// Description implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) Description() string {
	return f.cfg.Description
}

// Name implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) Name() string {
	return f.cfg.Name
}

// ProcessRequest implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) ProcessRequest(ctx context.Context, tc *adk.ToolContext, req *adk.LLMRequest) error {
	return req.AppendTools(f)
}

// FunctionDeclaration implements interfaces.FunctionTool.
func (f *FunctionTool[TArgs, TResults]) FunctionDeclaration() *genai.FunctionDeclaration {
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
	return decl
}

// Run executes the tool with the provided context and yields events.
func (f *FunctionTool[TArgs, TResults]) Run(ctx context.Context, tc *adk.ToolContext, args map[string]any) (map[string]any, error) {
	// TODO: Handle function call request from tc.InvocationContext.
	// TODO: Handle panic -> convert to error.
	input, err := typeutil.ConvertToWithJSONSchema[map[string]any, TArgs](args, f.inputSchema)
	if err != nil {
		return nil, err
	}
	output := f.handler(ctx, input)
	return typeutil.ConvertToWithJSONSchema[TResults, map[string]any](output, f.outputSchema)
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
//  * MCP ToolHandler expects mcp.ServerSession. adk.ToolContext may be close
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
	schema, err := jsonschema.For[T]()
	if err != nil {
		return nil, err
	}
	return schema.Resolve(nil)
}
