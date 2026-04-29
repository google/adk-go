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
	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// toolNode wraps a tool from the tool package.
type toolNode[Input, Output any] struct {
	baseNode
	tool         tool.Tool
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
}

// NewToolNodeWithSchemasTyped creates a new node wrapping a tool with explicitly provided schemas.
// If a schema is nil, it will be inferred from the corresponding generic type Input or Output.
func NewToolNodeWithSchemasTyped[Input, Output any](t tool.Tool, inputSchema, outputSchema *jsonschema.Schema) (Node, error) {
	if t == nil {
		return nil, fmt.Errorf("tool cannot be nil")
	}
	ischema, err := resolvedSchema[Input](inputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving input schema for tool %q: %w", t.Name(), err)
	}
	oschema, err := resolvedSchema[Output](outputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving output schema for tool %q: %w", t.Name(), err)
	}

	return &toolNode[Input, Output]{
		baseNode:     baseNode{name: t.Name(), description: t.Description()},
		tool:         t,
		inputSchema:  ischema,
		outputSchema: oschema,
	}, nil
}

// NewToolNodeWithSchemas is a convenience wrapper for NewToolNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
func NewToolNodeWithSchemas(t tool.Tool, inputSchema, outputSchema *jsonschema.Schema) (Node, error) {
	return NewToolNodeWithSchemasTyped[any, any](t, inputSchema, outputSchema)
}

// NewToolNodeTyped creates a new node wrapping a tool using generics to
// automatically infer input and output schemas from the provided types.
func NewToolNodeTyped[Input, Output any](t tool.Tool) (Node, error) {
	return NewToolNodeWithSchemasTyped[Input, Output](t, nil, nil)
}

// NewToolNode creates a new node wrapping a tool. Input and output schemas
// are inferred as 'any'.
func NewToolNode(t tool.Tool) (Node, error) {
	return NewToolNodeTyped[any, any](t)
}

func (n *toolNode[Input, Output]) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		runnable, ok := n.tool.(interface {
			Run(ctx tool.Context, args any) (map[string]any, error)
		})
		if !ok {
			yield(nil, fmt.Errorf("tool %q (type %T) is not directly runnable in workflow node", n.tool.Name(), n.tool))
			return
		}

		toolInput, err := typeutil.ConvertToWithJSONSchema[any, any](input, n.inputSchema)
		if err != nil {
			yield(nil, fmt.Errorf("converting input for tool %q: %w", n.tool.Name(), err))
			return
		}

		eventActions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
		toolCtx := toolinternal.NewToolContext(ctx, uuid.NewString(), eventActions, nil)

		output, err := runnable.Run(toolCtx, toolInput)
		if err != nil {
			yield(nil, fmt.Errorf("tool %q execution failed: %w", n.tool.Name(), err))
			return
		}

		typedOutput, err := typeutil.ConvertToWithJSONSchema[any, Output](output, n.outputSchema)
		if err != nil {
			// If outputSchema is not set or direct conversion failed, functiontool might have wrapped
			// the result in a "result" key (common for basic types).
			if val, ok := output["result"]; ok {
				// If we have a "result" key but it can't be converted
				typedOutput, err = typeutil.ConvertToWithJSONSchema[any, Output](val, n.outputSchema)
				if err != nil {
					// Try to convert to the type directly if that's some basic type.
					if v, ok := val.(Output); ok {
						typedOutput = v
					} else {
						yield(nil, fmt.Errorf("converting tool %q output to %T (from \"result\" key): %w", n.tool.Name(), *new(Output), err))
						return
					}
				}
			} else {
				yield(nil, fmt.Errorf("converting tool %q output to %T: %w", n.tool.Name(), *new(Output), err))
				return
			}
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Actions = *eventActions
		event.Actions.StateDelta["output"] = typedOutput

		// If output is a string, set it as content for convenience (similar to FunctionNode).
		if s, ok := any(typedOutput).(string); ok {
			event.Content = &genai.Content{
				Parts: []*genai.Part{{Text: s}},
			}
		}

		yield(event, nil)
	}
}

// resolvedSchema is a helper to infer schema from type T.
func resolvedSchema[T any](override *jsonschema.Schema) (*jsonschema.Resolved, error) {
	if override != nil {
		return override.Resolve(nil)
	}
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, err
	}
	return schema.Resolve(nil)
}
