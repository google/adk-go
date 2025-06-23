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
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/adk-go"
	"github.com/google/adk-go/internal/jsonschema"
	"google.golang.org/genai"
)

// FunctionTool: borrow implementation from MCP go.
// transfer_to_agent ??
// LongRunningFunctionTool
// LoadArtifactsTool
// ExitLoopTool
// AgentTool

// BuiltinCodeExecutionTool
// GoogeSearchTool
// MCPTool

/*
type TransferToAgentTool struct {
}

// ProcessRequest implements adk.Tool.
func (t *TransferToAgentTool) ProcessRequest(ctx context.Context, tc *adk.ToolContext, req *adk.LLMRequest) error {

	panic("unimplemented")
}

// Run implements adk.Tool.
func (t *TransferToAgentTool) Run(ctx context.Context, tc *adk.ToolContext, args any) (any, error) {
	panic("unimplemented")
}

type TransferToAgentToolInput struct {
	AgentName string `json:"agent_name"`
}

// Schema implements adk.Tool.
func (t *TransferToAgentTool) Schema() *jsonschema.Schema {
	schema, err := jsonschema.For[TransferToAgentToolInput]()
	if err != nil {
		panic(err)
	}
	return schema
}

// Description implements adk.Tool.
func (t *TransferToAgentTool) Description() string {
	return `Transfer the question to another agent.

This tool hands off control to another agent when it's more suiltable to answer the user's question according to the agent's decription.`
}

// Name implements adk.Tool.
func (t *TransferToAgentTool) Name() string {
	return "transfer_to_agent"
}

// ProcessRequest implements adk.Tool.

var _ adk.Tool = (*TransferToAgentTool)(nil)
*/

// NewFunctionTool is a helper to make a tool using reflection on the given type parameters.
// When the tool is called, CallToolParams.Arguments will be of type In.
//
// If provided, variadic [ToolOption] values may be used to customize the tool.
//
// The input schema for the tool is extracted from the request type for the
// handler, and used to unmmarshal and validate requests to the handler. This
// schema may be customized using the [Input] option.
// TODO: options.
// TODO: can we codegen TArgs, TResults, ...?
func NewFunctionTool[TArgs, TResults any](name, description string, handler FunctionToolHandler[TArgs, TResults]) *FunctionTool[TArgs, TResults] {
	st, err := newFunctionToolErr(name, description, handler)
	if err != nil {
		panic(fmt.Errorf("NewServerTool(%q): %w", name, err))
	}
	return st
}

func newFunctionToolErr[TArgs, TResults any](name, description string, handler FunctionToolHandler[TArgs, TResults]) (*FunctionTool[TArgs, TResults], error) {
	// TODO: check that In is a struct.
	ischema, err := jsonschema.For[TArgs]()
	if err != nil {
		return nil, err
	}
	oschema, err := jsonschema.For[TResults]()
	if err != nil {
		return nil, err
	}

	t := &FunctionTool[TArgs, TResults]{
		name:         name,
		description:  description,
		inputSchema:  ischema,
		outputSchema: oschema,
		handler:      handler,
	}
	// TODO: handle opts.

	if ischema != nil {
		r, err := ischema.Resolve(&jsonschema.ResolveOptions{ValidateDefaults: true})
		if err != nil {
			return nil, err
		}
		t.inputResolved = r
	}
	if oschema != nil {
		r, err := oschema.Resolve(&jsonschema.ResolveOptions{ValidateDefaults: true})
		if err != nil {
			return nil, err
		}
		t.outputResolved = r

	}

	return t, nil

}

// unmarshalSchema unmarshals data into v and validates the result according to
// the given resolved schema.
func unmarshalSchema(data json.RawMessage, resolved *jsonschema.Resolved, v any) error {
	// TODO: use reflection to create the struct type to unmarshal into.
	// Separate validation from assignment.

	// Disallow unknown fields.
	// Otherwise, if the tool was built with a struct, the client could send extra
	// fields and json.Unmarshal would ignore them, so the schema would never get
	// a chance to declare the extra args invalid.
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("unmarshaling: %w", err)
	}
	// TODO: test with nil args.
	if resolved != nil {
		if err := resolved.ApplyDefaults(v); err != nil {
			return fmt.Errorf("applying defaults from \n\t%s\nto\n\t%s:\n%w", schemaJSON(resolved.Schema()), data, err)
		}
		if err := resolved.Validate(v); err != nil {
			return fmt.Errorf("validating\n\t%s\nagainst\n\t %s:\n %w", data, schemaJSON(resolved.Schema()), err)
		}
	}
	return nil
}

// schemaJSON returns the JSON value for s as a string, or a string indicating an error.
func schemaJSON(s *jsonschema.Schema) string {
	m, err := json.Marshal(s)
	if err != nil {
		return fmt.Sprintf("<!%s>", err)
	}
	return string(m)
}

type FunctionTool[TArgs, TResults any] struct {
	// The name of the tool.
	name string
	// This can be used by clients to improve the LLM's understanding of available
	// tools. It can be thought of like a "hint" to the model.
	description string
	// A JSON Schema object defining the expected parameters for the tool.
	inputSchema *jsonschema.Schema
	// A JSON Schema object defining the result of the tool.
	outputSchema *jsonschema.Schema

	handler FunctionToolHandler[TArgs, TResults]
	// Resolved tool schemas. Set in Server.AddToolsErr.
	inputResolved, outputResolved *jsonschema.Resolved

	isLongRunning bool
}

// IsLongRunningTool implements LongRunningTool.
func (f *FunctionTool[TArgs, TResults]) IsLongRunningTool() bool {
	return f.isLongRunning
}

// Description implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) Description() string {
	return f.description
}

// Name implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) Name() string {
	return f.name
}

// ProcessRequest implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) ProcessRequest(ctx context.Context, tc *adk.ToolContext, req *adk.LLMRequest) error {
	if f.name == "" {
		return nil // TODO: return error?
	}
	decl := &genai.FunctionDeclaration{
		Name:                 f.name,
		Description:          f.description,
		ParametersJsonSchema: f.inputSchema,
		ResponseJsonSchema:   f.outputSchema,
	}

	req.GenerateConfig.Tools = append(req.GenerateConfig.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{decl},
	})
	return nil
}

// Run implements adk.Tool.
func (f *FunctionTool[TArgs, TResults]) Run(ctx context.Context, tc *adk.ToolContext, args map[string]any) (map[string]any, error) {
	// hack. genai.FunctionCall sends map[string]any.
	rawArgs, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("args is not json serializable: %w", err)
	}
	var typedArgs TArgs
	if rawArgs != nil {
		if err := unmarshalSchema(rawArgs, f.inputResolved, &typedArgs); err != nil {
			return nil, err
		}
	}

	resp, err := f.handler(ctx, tc, typedArgs)
	if err != nil {
		return nil, err
	}

	// hack. genai.FunctionResponse expects map[string]any.

	respMsg, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	var respMap map[string]any
	if err := json.Unmarshal(respMsg, &respMap); err != nil {
		return nil, err
	}
	return respMap, nil
}

var _ adk.Tool = (*FunctionTool[any, any])(nil)
var _ adk.LongRunningTool = (*FunctionTool[any, any])(nil)

type FunctionToolHandler[TArgs, TResults any] func(context.Context, *adk.ToolContext, TArgs) (TResults, error)
