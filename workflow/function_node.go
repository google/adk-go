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

// EmittingFunctionFn is the streaming variant of a FunctionNode body:
// it may emit intermediate events before returning the terminal
// output. The shape mirrors DynamicFn (see dynamic_node.go) without a
// sub-scheduler.
//
// To pause for human input, emit a NewRequestInputEvent and return
// (zero, ErrNodeInterrupted); the engine parks the node and routes the
// resume payload per NodeConfig.RerunOnResume.
type EmittingFunctionFn[IN, OUT any] = func(
	ctx agent.Context, input IN, emit func(*session.Event) error,
) (OUT, error)

// FunctionNode wraps a custom function.
//
// Exactly one of fn or emittingFn is set per node.
type FunctionNode struct {
	BaseNode
	fn              func(ctx agent.Context, input any) (any, error)
	emittingFn      func(ctx agent.Context, input any, emit func(*session.Event) error) (any, error)
	stateFieldNames []string
}

// StateFieldNames returns the list of state keys this node consumes.
func (n *FunctionNode) StateFieldNames() []string {
	return n.stateFieldNames
}

// NewFunctionNode creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNode[IN, OUT any](name string, fn func(ctx agent.Context, input IN) (OUT, error), cfg NodeConfig) *FunctionNode {
	return newFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, nil, nil, cfg)
}

// NewFunctionNodeWithSchema creates a new node wrapping a custom function using generics to automatically infer input and output types.
func NewFunctionNodeWithSchema[IN, OUT any](name string, fn func(ctx agent.Context, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*FunctionNode, error) {
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

// NewEmittingFunctionNode wraps fn as a streaming FunctionNode whose
// body may emit intermediate events (HITL prompts, progress, state
// deltas) before returning. See EmittingFunctionFn for the pause
// contract.
func NewEmittingFunctionNode[IN, OUT any](name string, fn EmittingFunctionFn[IN, OUT], cfg NodeConfig) *FunctionNode {
	return newEmittingFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, nil, nil, cfg)
}

// NewEmittingFunctionNodeWithSchema is the explicit-schema variant of
// NewEmittingFunctionNode. A nil schema is inferred from the
// corresponding generic type.
func NewEmittingFunctionNodeWithSchema[IN, OUT any](name string, fn EmittingFunctionFn[IN, OUT], inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*FunctionNode, error) {
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
	return newEmittingFunctionNodeWithResolvedSchemas[IN, OUT](name, fn, ischema, oschema, cfg), nil
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

	wrappedFn := func(ctx agent.Context, input any) (any, error) {
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
		return output, nil
	}

	return &FunctionNode{
		BaseNode:        NewBaseNodeWithSchemas(name, "", cfg, ischema, oschema),
		fn:              wrappedFn,
		stateFieldNames: stateFieldNames,
	}, nil
}

// newFunctionNodeWithResolvedSchemas is an internal constructor that consumes already resolved schemas.
func newFunctionNodeWithResolvedSchemas[IN, OUT any](name string, fn func(ctx agent.Context, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Resolved, cfg NodeConfig) *FunctionNode {
	wrappedFn := func(ctx agent.Context, input any) (any, error) {
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
				// ConvertToWithJSONSchema also validates against inputSchema; the explicit Validate below covers
				// the assertion-hit path so schema constraints (e.g. maxLength) are enforced regardless.
				typedInput, err = typeutil.ConvertToWithJSONSchema[any, IN](input, inputSchema)
				if err != nil {
					return nil, fmt.Errorf("new function node: invalid input type, expected %T: %w", new(IN), err)
				}
			} else if inputSchema != nil {
				if err = typeutil.ValidateWithJSONSchema(typedInput, inputSchema); err != nil {
					return nil, fmt.Errorf("function node %s: validation failed for input %T: %w", name, new(IN), err)
				}
			}
			output, err = fn(ctx, typedInput)
		}

		if err != nil {
			return output, err
		}

		return output, nil
	}

	return &FunctionNode{
		BaseNode: NewBaseNodeWithSchemas(name, "", cfg, inputSchema, outputSchema),
		fn:       wrappedFn,
	}
}

// newEmittingFunctionNodeWithResolvedSchemas is the internal
// constructor for the streaming variant.
func newEmittingFunctionNodeWithResolvedSchemas[IN, OUT any](name string, fn EmittingFunctionFn[IN, OUT], inputSchema, outputSchema *jsonschema.Resolved, cfg NodeConfig) *FunctionNode {
	wrappedFn := func(ctx agent.Context, input any, emit func(*session.Event) error) (any, error) {
		var typedInput IN
		if input != nil {
			t, ok := input.(IN)
			if !ok {
				// Fallback to the json-like input types that cannot be converted by the standard type assertion.
				// E.g. tool nodes return map[string]any as input and user may define a struct as the target type.
				// ConvertToWithJSONSchema also validates against inputSchema; the explicit Validate below covers
				// the assertion-hit path so schema constraints (e.g. maxLength) are enforced regardless.
				converted, err := typeutil.ConvertToWithJSONSchema[any, IN](input, inputSchema)
				if err != nil {
					return nil, fmt.Errorf("new function node: invalid input type, expected %T: %w", new(IN), err)
				}
				t = converted
			} else if inputSchema != nil {
				if err := typeutil.ValidateWithJSONSchema(t, inputSchema); err != nil {
					return nil, fmt.Errorf("function node %s: validation failed for input %T: %w", name, new(IN), err)
				}
			}
			typedInput = t
		}

		output, err := fn(ctx, typedInput, emit)
		if err != nil {
			return nil, err
		}
		return output, nil
	}

	return &FunctionNode{
		BaseNode:   NewBaseNodeWithSchemas(name, "", cfg, inputSchema, outputSchema),
		emittingFn: wrappedFn,
	}
}

// Run executes the function node with the given input and returns an iterator over events.
func (n *FunctionNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if n.emittingFn != nil {
			n.runEmitting(ctx, input, yield)
			return
		}
		output, err := n.fn(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		// If the return type is already a *session.Event, yield it directly.
		// This ensures that a custom FunctionNode can explicitly route its output
		// (via event.Routes) to select conditional successors.
		if ev, ok := output.(*session.Event); ok {
			if ev.InvocationID == "" {
				ev.InvocationID = ctx.InvocationID()
			}
			yield(ev, nil)
			return
		}

		event := session.NewEventWithContext(ctx, ctx.InvocationID())
		if c, ok := output.(*genai.Content); ok {
			event.Content = c
		} else if c, ok := output.(genai.Content); ok {
			event.Content = &c
		} else {
			event.Output = output
		}
		yield(event, nil)
	}
}

// runEmitting executes the streaming variant, mirroring
// dynamicNode.Run: ErrNodeInterrupted swallows the result (the pause
// event was already emitted), any other error fails the node, and a
// nil output suppresses the terminal event.
func (n *FunctionNode) runEmitting(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
	emit := makeEmit(yield, ctx)
	output, err := n.emittingFn(ctx, input, emit)
	if err != nil {
		if errors.Is(err, ErrNodeInterrupted) {
			return
		}
		yield(nil, err)
		return
	}
	if output == nil {
		return
	}
	event := session.NewEventWithContext(ctx, ctx.InvocationID())
	event.Output = output
	if s, ok := output.(string); ok {
		event.Content = &genai.Content{
			Parts: []*genai.Part{{Text: s}},
		}
	}
	yield(event, nil)
}
