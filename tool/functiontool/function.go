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
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"runtime/debug"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
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

	// RequireConfirmation flags whether this tool must always ask for user confirmation
	// before execution. If set to true, the ADK framework will automatically initiate
	// a Human-in-the-Loop (HITL) confirmation request when this tool is invoked.
	RequireConfirmation bool

	// RequireConfirmationProvider allows for dynamic determination of whether
	// user confirmation is needed. This field is a function called at runtime to decide if
	// a confirmation request should be sent. The function takes the tool's input parameters as arguments.
	// This provider offers more flexibility than the static RequireConfirmation flag,
	// enabling conditional confirmation based on the invocation details.
	// If set, this often takes precedence over the RequireConfirmation flag.
	//
	// Required signature for a provider function:
	// func(toolInput ToolArgs) (bool)
	// where ToolArgs is the input type of your go function
	// Returning true means confirmation is required.
	RequireConfirmationProvider any

	// ResponseScheduling is copied onto every function response of the tool.
	// See tool.ResponseScheduler. The zero value sets none. It applies to tools
	// built with New and with NewStreaming.
	ResponseScheduling genai.FunctionResponseScheduling

	// Sequential makes each call of the tool start after every earlier call of
	// its model turn has returned. See tool.SequentialRunner. The zero value
	// leaves the calls of one turn concurrent. NewStreaming returns an error
	// that wraps ErrInvalidArgument when it is set.
	Sequential bool
}

// Func represents a Go function that can be wrapped in a tool.
// It takes a agent.Context and a generic argument type, and returns a generic result type.
type Func[TArgs, TResults any] func(agent.Context, TArgs) (TResults, error)

// ErrInvalidArgument indicates the input parameter type is invalid.
var ErrInvalidArgument = errors.New("invalid argument")

// New creates a new tool with a name, description, and the provided handler.
// Input schema is automatically inferred from the input and output types.
func New[TArgs, TResults any](cfg Config, handler Func[TArgs, TResults]) (tool.Tool, error) {
	// TODO: How can we improve UX for functions that does not require an argument, returns a simple type value, or returns a no result?
	// https://github.com/modelcontextprotocol/go-sdk/discussions/37

	var zeroArgs TArgs
	argsType := reflect.TypeOf(zeroArgs)
	for argsType != nil && argsType.Kind() == reflect.Pointer {
		argsType = argsType.Elem()
	}
	if argsType == nil || (argsType.Kind() != reflect.Struct && argsType.Kind() != reflect.Map) {
		return nil, fmt.Errorf("input must be a struct or a map or a pointer to those types, but received: %v: %w", argsType, ErrInvalidArgument)
	}

	ischema, err := resolvedSchema[TArgs](cfg.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to infer input schema: %w", err)
	}
	oschema, err := resolvedSchema[TResults](cfg.OutputSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to infer output schema: %w", err)
	}

	var confirmWrapper func(TArgs) bool

	if cfg.RequireConfirmationProvider != nil {
		// Attempt to cast the interface directly to the function signature
		fn, ok := cfg.RequireConfirmationProvider.(func(TArgs) bool)
		if !ok {
			return nil, fmt.Errorf("error RequireConfirmationProvider must be a function with signature func(%T) bool", *new(TArgs))
		}
		confirmWrapper = fn
	}

	return &functionTool[TArgs, TResults]{
		cfg:                         cfg,
		inputSchema:                 ischema,
		outputSchema:                oschema,
		handler:                     handler,
		requireConfirmation:         cfg.RequireConfirmation,
		requireConfirmationProvider: confirmWrapper,
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

	requireConfirmation bool

	requireConfirmationProvider func(TArgs) bool
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

// RunsSequentially implements tool.SequentialRunner.
func (f *functionTool[TArgs, TResults]) RunsSequentially() bool {
	return f.cfg.Sequential
}

// ResponseScheduling implements tool.ResponseScheduler.
func (f *functionTool[TArgs, TResults]) ResponseScheduling() genai.FunctionResponseScheduling {
	return f.cfg.ResponseScheduling
}

// ProcessRequest packs the function tool's declaration into the LLM request,
// in the form the request's typed-declaration switch and backend ask for.
func (f *functionTool[TArgs, TResults]) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	base := f.Declaration()
	decl, err := requestDeclaration(ctx, base, schemaOf(f.inputSchema), schemaOf(f.outputSchema))
	if err != nil {
		return fmt.Errorf("function tool %q declaration: %w", f.Name(), err)
	}
	if decl == base {
		return toolutils.PackTool(req, f)
	}
	return toolutils.PackTool(req, &declaredFunctionTool[TArgs, TResults]{functionTool: f, decl: decl})
}

// Declaration implements interfaces.FunctionTool.
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
func (f *functionTool[TArgs, TResults]) Run(ctx agent.Context, args any) (result map[string]any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = toolinternal.NewPanicError(f.Name(), r, debug.Stack())
		}
	}()

	output, early, err := f.invoke(ctx, args)
	if err != nil || early != nil {
		return early, err
	}
	if toolinternal.CarriesPart(output) {
		if m, ok := partMap(output); ok {
			return m, nil
		}
		return map[string]any{"result": output}, nil
	}
	resp, converted, err := f.convert(output)
	if err != nil || converted {
		return resp, err
	}
	// Specs requires the result to be a map (dict in python). python impl allows basic types when building response event
	// functions.py __build_response_event does the following
	// if not isinstance(function_result, dict):
	// 		function_result = {'result': function_result}
	return map[string]any{"result": output}, nil
}

// RunRaw implements toolinternal.RawResultTool. It handles the arguments as
// Run does and returns the handler's output without wrapping a non-map value:
// an output holding a *genai.Part at any depth of maps, slices and arrays as
// it is (a map with string keys copied into a new map[string]any), a nil
// output as an untyped nil, an output that converts to a JSON object as that
// map[string]any, and any other output as it is.
func (f *functionTool[TArgs, TResults]) RunRaw(ctx agent.Context, args any) (result any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = toolinternal.NewPanicError(f.Name(), r, debug.Stack())
		}
	}()

	output, early, err := f.invoke(ctx, args)
	if err != nil {
		return nil, err
	}
	if early != nil {
		return early, nil
	}
	if toolinternal.CarriesPart(output) {
		if m, ok := partMap(output); ok {
			return m, nil
		}
		return output, nil
	}
	if isNil(output) {
		return nil, nil
	}
	resp, converted, err := f.convert(output)
	if err != nil {
		return nil, err
	}
	if converted {
		return resp, nil
	}
	return output, nil
}

// invoke runs everything up to and including the handler: argument handling,
// confirmation and the handler itself. early is the result to answer with
// instead of running the handler, when mandatory arguments are missing.
func (f *functionTool[TArgs, TResults]) invoke(ctx agent.Context, args any) (output TResults, early map[string]any, err error) {
	m, ok := args.(map[string]any)
	if !ok {
		return output, nil, fmt.Errorf("unexpected args type, got: %T", args)
	}
	// A nil map keeps the validator's refusal of a JSON null.
	if m != nil {
		m = dropUndeclared(schemaOf(f.inputSchema), m)
		if missing := missingMandatory(schemaOf(f.inputSchema), m); len(missing) > 0 {
			return output, map[string]any{"error": missingArgsText(f.Name(), missing)}, nil
		}
	}
	input, err := typeutil.ConvertToWithJSONSchema[map[string]any, TArgs](m, f.inputSchema)
	if err != nil {
		return output, nil, err
	}

	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return output, nil, fmt.Errorf("error tool %q %w", f.Name(), tool.ErrConfirmationRejected)
		}
	} else {
		requireConfirmation := f.requireConfirmation

		// Only run the potentially expensive provider if the static flag didn't already trigger it
		// Provider takes precedence/overrides:
		if f.requireConfirmationProvider != nil {
			requireConfirmation = f.requireConfirmationProvider(input)
		}

		if requireConfirmation {
			err := ctx.RequestConfirmation(
				fmt.Sprintf("Please approve or reject the tool call %s() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					f.Name()), nil)
			if err != nil {
				return output, nil, err
			}
			ctx.Actions().SkipSummarization = true
			return output, nil, fmt.Errorf("error tool %q %w", f.Name(), tool.ErrConfirmationRequired)
		}
	}

	output, err = f.handler(ctx, input)
	return output, nil, err
}

// convert converts an output that holds no *genai.Part to a map[string]any
// through its JSON form. converted is false when the output is not a JSON
// object; the caller then answers with the output itself. An output
// encoding/json cannot represent is an error, because it cannot be sent, and
// so is one an explicit OutputSchema rejects.
func (f *functionTool[TArgs, TResults]) convert(output TResults) (resp map[string]any, converted bool, err error) {
	resp, err = typeutil.ConvertToWithJSONSchema[TResults, map[string]any](output, f.outputSchema)
	if err == nil {
		return resp, true, nil
	}
	if _, mErr := json.Marshal(output); mErr != nil {
		return nil, false, mErr
	}
	// A schema inferred from TResults describes the type, not what encoding/json
	// emits for it.
	if f.cfg.OutputSchema != nil {
		if err1 := typeutil.ValidateWithJSONSchema(output, f.outputSchema); err1 != nil {
			return resp, false, err // if it fails propagate original err.
		}
	}
	return nil, false, nil
}

// partMap returns a map with string keys as a new map[string]any holding the
// same values, so that *genai.Part values inside it survive.
func partMap(v any) (map[string]any, bool) {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Map || rv.Type().Key().Kind() != reflect.String {
		return nil, false
	}
	m := make(map[string]any, rv.Len())
	for it := rv.MapRange(); it.Next(); {
		m[it.Key().String()] = it.Value().Interface()
	}
	return m, true
}

// isNil reports whether v is nil, or a nil pointer, map, slice or interface.
func isNil(v any) bool {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() {
		return true
	}
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return rv.IsNil()
	}
	return false
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
//  * MCP ToolHandler expects mcp.ServerSession. agent.Context may be close
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
