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

package functiontool

import (
	"fmt"
	"reflect"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// nonStructInputWrapper wraps a functionTool with a non-struct input type,
// converting the LLM-provided map[string]any with an "input" key to the expected type.
type nonStructInputWrapper[TArgs, TResults any] struct {
	cfg          Config
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
	handler      Func[TArgs, TResults]
}

// wrapNonStructInput wraps a functionTool if its input type is not a struct.
// This is necessary for llmagent compatibility, as the LLM always provides arguments
// as map[string]any with an "input" key for non-struct types.
func wrapNonStructInput[TArgs, TResults any](
	cfg Config,
	inputSchema *jsonschema.Resolved,
	outputSchema *jsonschema.Resolved,
	handler Func[TArgs, TResults],
) (tool.Tool, error) {
	// Check if TArgs is a struct type
	var zeroArgs TArgs
	if reflect.TypeOf(zeroArgs).Kind() != reflect.Struct {
		// Non-struct type: wrap it
		return &nonStructInputWrapper[TArgs, TResults]{
			cfg:          cfg,
			inputSchema:  inputSchema,
			outputSchema: outputSchema,
			handler:      handler,
		}, nil
	}

	// Struct type: return the regular functionTool
	return &functionTool[TArgs, TResults]{
		cfg:          cfg,
		inputSchema:  inputSchema,
		outputSchema: outputSchema,
		handler:      handler,
	}, nil
}

// Description implements tool.Tool.
func (w *nonStructInputWrapper[TArgs, TResults]) Description() string {
	return w.cfg.Description
}

// Name implements tool.Tool.
func (w *nonStructInputWrapper[TArgs, TResults]) Name() string {
	return w.cfg.Name
}

// IsLongRunning implements tool.Tool.
func (w *nonStructInputWrapper[TArgs, TResults]) IsLongRunning() bool {
	return w.cfg.IsLongRunning
}

// ProcessRequest implements tool.Tool.
func (w *nonStructInputWrapper[TArgs, TResults]) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, w)
}

// Declaration implements FunctionTool interface.
// It modifies the schema to expect {"input": <value>} format.
func (w *nonStructInputWrapper[TArgs, TResults]) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name:        w.Name(),
		Description: w.Description(),
	}

	// Create a wrapper schema that expects {"input": <value>} format
	wrappedSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"input": {},
		},
		Required: []string{"input"},
	}

	// Copy the original input schema properties to the "input" property.
	// `inputSchema` is expected to be non-nil (inferred by New), so assign directly.
	wrappedSchema.Properties["input"] = w.inputSchema.Schema()
	decl.ParametersJsonSchema = wrappedSchema
	if w.outputSchema != nil {
		decl.ResponseJsonSchema = w.outputSchema.Schema()
	}

	if w.cfg.IsLongRunning {
		instruction := "NOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status."
		if decl.Description != "" {
			decl.Description += "\n\n" + instruction
		} else {
			decl.Description = instruction
		}
	}

	return decl
}

// Run implements FunctionTool.
// It unwraps the {"input": <value>} format and calls the handler.
func (w *nonStructInputWrapper[TArgs, TResults]) Run(ctx tool.Context, args any) (map[string]any, error) {
	// Extract the map
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type, got: %T", args)
	}

	// Extract the "input" key
	inputVal, ok := m["input"]
	if !ok {
		return nil, fmt.Errorf("missing required 'input' argument")
	}

	// Convert to TArgs - use JSON marshaling to handle the conversion.
	// Do NOT pass the original resolved input schema to ConvertToWithJSONSchema here,
	// because resolvedSchema validation expects a map[string]any and will fail for
	// primitive JSON values (e.g. string). Validation is handled by the declaration
	// schema sent to the model; at runtime we only need to convert the value.
	input, err := typeutil.ConvertToWithJSONSchema[any, TArgs](inputVal, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert input: %w", err)
	}

	// Call the handler
	output, err := w.handler(ctx, input)
	if err != nil {
		return nil, err
	}

	// Convert output to map[string]any
	resp, err := typeutil.ConvertToWithJSONSchema[TResults, map[string]any](output, w.outputSchema)
	if err == nil {
		return resp, nil
	}

	// If conversion fails and outputSchema is set, validate and return a
	// specific validation error to aid debugging.
	if w.outputSchema != nil {
		if err1 := w.outputSchema.Validate(output); err1 != nil {
			return nil, fmt.Errorf("output validation failed: %w", err1)
		}
	}
	wrappedOutput := map[string]any{"result": output}
	return wrappedOutput, nil
}
