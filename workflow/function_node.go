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
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// FunctionNode wraps a custom function.
type FunctionNode struct {
	baseNode
	fn func(ctx agent.InvocationContext, input any) (any, error)
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

// newFunctionNodeWithResolvedSchemas is an internal constructor that consumes already resolved schemas.
func newFunctionNodeWithResolvedSchemas[IN, OUT any](name string, fn func(ctx agent.InvocationContext, input IN) (OUT, error), inputSchema, outputSchema *jsonschema.Resolved, cfg NodeConfig) *FunctionNode {
	wrappedFn := func(ctx agent.InvocationContext, input any) (any, error) {
		if input == nil {
			var zero IN
			return fn(ctx, zero)
		}
		typedInput, ok := input.(IN)
		if !ok {
			// Fallback to the json-like input types that cannot be converted by the standard type assertion.
			// E.g. tool nodes return map[string]any as input and user may define a struct as the target type.
			var err error
			typedInput, err = typeutil.ConvertToWithJSONSchema[any, IN](input, inputSchema)
			if err != nil {
				return nil, fmt.Errorf("new function node: invalid input type, expected %T: %v", new(IN), err)
			}
		}
		output, err := fn(ctx, typedInput)
		if err != nil {
			return output, err
		}

		if outputSchema != nil {
			validateErr := outputSchema.Validate(output)
			if validateErr != nil {
				return nil, fmt.Errorf("function node %s: validation failed for output %T: %v", name, new(OUT), validateErr)
			}
		}

		return output, nil
	}

	if cfg.RetryConfig != nil {
		cfg.RetryConfig.applyDefaults()
	}

	return &FunctionNode{
		baseNode: baseNode{name: name, config: cfg},
		fn:       wrappedFn,
	}
}

func (n *FunctionNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		output, err := n.fn(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Actions.StateDelta["output"] = output
		if s, ok := output.(string); ok {
			event.Content = &genai.Content{
				Parts: []*genai.Part{{Text: s}},
			}
		}
		yield(event, nil)
	}
}
