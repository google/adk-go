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

package workflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// FunctionNode wraps a custom function.
type FunctionNode struct {
	BaseNode
	fn              func(ctx agent.InvocationContext, input any) (any, error)
	stateFieldNames []string
}

// StateFieldNames returns the list of state keys this node consumes.
func (n *FunctionNode) StateFieldNames() []string {
	return n.stateFieldNames
}

// NewFunctionNode creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNode[IN, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error), cfg NodeConfig) *FunctionNode {
	return newFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, nil, nil, cfg)
}

// NewFunctionNodeWithSchema creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNodeWithSchema[IN, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*FunctionNode, error) {
	var ischema *jsonschema.Resolved
	var err error
	if inputSchema != nil {
		ischema, err = inputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving input schema: %w", err)
		}
	}

	var oschema *jsonschema.Resolved
	if outputSchema != nil {
		oschema, err = outputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving output schema: %w", err)
		}
	}

	return newFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, ischema, oschema, cfg), nil
}

// NewFunctionNodeFromState returns a FunctionNode whose user function
// receives a struct-of-parameters loaded from ctx.state. Field names
// default to the Go field name; override with a struct tag
// `state:"my_key"`. A field tagged `state:"node_input"` (or named
// "NodeInput") receives the raw predecessor output instead.
//
// Params must be a struct type.
func NewFunctionNodeFromState[Params, OUT any](
	name string,
	fn func(ctx agent.InvocationContext, p Params) (OUT, error),
	cfg NodeConfig,
) (*FunctionNode, error) {
	paramType := reflect.TypeOf(*new(Params))
	if paramType == nil || paramType.Kind() != reflect.Struct {
		return nil, fmt.Errorf("params must be a struct type")
	}

	var nodeInputField reflect.StructField
	var hasNodeInput bool
	var stateFieldNames []string

	type fieldBinding struct {
		fieldName string
		stateKey  string
	}
	var bindings []fieldBinding

	for i := 0; i < paramType.NumField(); i++ {
		field := paramType.Field(i)
		if !field.IsExported() {
			continue
		}

		tag := field.Tag.Get("state")
		isNodeInput := false
		stateKey := field.Name

		if tag != "" {
			stateKey = tag
			if tag == "node_input" {
				isNodeInput = true
			}
		} else if field.Name == "NodeInput" {
			isNodeInput = true
		}

		if isNodeInput {
			if hasNodeInput {
				return nil, fmt.Errorf("multiple node_input fields found")
			}
			hasNodeInput = true
			nodeInputField = field
		} else {
			bindings = append(bindings, fieldBinding{
				fieldName: field.Name,
				stateKey:  stateKey,
			})
			stateFieldNames = append(stateFieldNames, stateKey)
		}
	}

	var ischema *jsonschema.Resolved
	if hasNodeInput {
		rawSchema, err := jsonschema.ForType(nodeInputField.Type, nil)
		if err != nil {
			return nil, fmt.Errorf("generating schema for node_input: %w", err)
		}
		ischema, err = rawSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("resolving input schema: %w", err)
		}
	}

	oschemaRaw, err := jsonschema.For[OUT](nil)
	if err != nil {
		return nil, fmt.Errorf("generating output schema: %w", err)
	}
	oschema, err := oschemaRaw.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("resolving output schema: %w", err)
	}

	wrappedFn := func(ctx agent.InvocationContext, input any) (any, error) {
		sessionState := ctx.Session().State()
		stateMap := make(map[string]any)

		for _, b := range bindings {
			val, err := sessionState.Get(b.stateKey)
			if err != nil {
				return nil, fmt.Errorf("missing state value for required field %q (state key %q): %w", b.fieldName, b.stateKey, err)
			}
			stateMap[b.fieldName] = val
		}

		if hasNodeInput {
			stateMap[nodeInputField.Name] = input
		}

		p, err := typeutil.ConvertToWithJSONSchema[any, Params](stateMap, nil)
		if err != nil {
			var typeErr *json.UnmarshalTypeError
			if errors.As(err, &typeErr) {
				parts := strings.Split(typeErr.Field, ".")
				fieldName := parts[len(parts)-1]
				if hasNodeInput && fieldName == nodeInputField.Name {
					return nil, fmt.Errorf("new function node from state: invalid input type for node_input: %w", err)
				}
				return nil, fmt.Errorf("failed to convert state value to field %q: %w", fieldName, err)
			}
			return nil, fmt.Errorf("failed to convert state to Params: %w", err)
		}

		output, err := fn(ctx, p)
		if err != nil {
			return output, err
		}
		if oschema != nil {
			if err := typeutil.ValidateWithJSONSchema(output, oschema); err != nil {
				return nil, fmt.Errorf("function node %s: validation failed for output %T: %w", name, new(OUT), err)
			}
		}
		return output, nil
	}

	return &FunctionNode{
		BaseNode:        NewBaseNodeWithSchemas(name, "", cfg, ischema, oschema),
		fn:              wrappedFn,
		stateFieldNames: stateFieldNames,
	}, nil
}

// newFunctionNodeWithResolvedSchemas is an internal constructor that consumes already resolved schemas.
func newFunctionNodeWithResolvedSchemas[IN, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Resolved, cfg NodeConfig) *FunctionNode {
	wrappedFn := func(ctx agent.InvocationContext, input any) (any, error) {
		var output OUT
		var err error
		if input == nil {
			var zero IN
			output, err = fn(ctx, zero)
		} else {
			typedInput, ok := input.(IN)
			if !ok {
				// Fallback to the json-like input types that cannot be converted by the standard type assertion.
				// E.g. tool nodes return map[string]any as input and user may define a struct as the target type.
				typedInput, err = typeutil.ConvertToWithJSONSchema[any, IN](input, inputSchema)
				if err != nil {
					return nil, fmt.Errorf("new function node: invalid input type, expected %T: %w", new(IN), err)
				}
			}
			output, err = fn(ctx, typedInput)
		}

		if err != nil {
			return output, err
		}

		if outputSchema != nil {
			validateErr := typeutil.ValidateWithJSONSchema(output, outputSchema)
			if validateErr != nil {
				return nil, fmt.Errorf("function node %s: validation failed for output %T: %w", name, new(OUT), validateErr)
			}
		}

		return output, nil
	}

	return &FunctionNode{
		BaseNode: NewBaseNodeWithSchemas(name, "", cfg, inputSchema, outputSchema),
		fn:       wrappedFn,
	}
}

// Run executes the function node with the given input and returns an iterator over events.
func (n *FunctionNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		output, err := n.fn(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Output = output
		if s, ok := output.(string); ok {
			event.Content = &genai.Content{
				Parts: []*genai.Part{{Text: s}},
			}
		}
		yield(event, nil)
	}
}
