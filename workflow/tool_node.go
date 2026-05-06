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
type toolNode struct {
	baseNode
	tool         tool.Tool
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
	config       NodeConfig
}

type runnableTool interface {
	Run(ctx tool.Context, args any) (map[string]any, error)
}

// newToolNodeWithSchemasTyped creates a new node wrapping a tool with explicitly provided schemas.
// If a schema is nil, it will be inferred from the corresponding generic type Input or Output.
func newToolNodeWithSchemasTyped[Input, Output any](t tool.Tool, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (Node, error) {
	if t == nil {
		return nil, fmt.Errorf("tool cannot be nil")
	}
	ischema, err := resolvedSchema[Input](inputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving input schema for tool %q: %w", t.Name(), err)
	}
	if ischema == nil {
		return nil, fmt.Errorf("resolved input schema for tool %q is nil", t.Name())
	}
	oschema, err := resolvedSchema[Output](outputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving output schema for tool %q: %w", t.Name(), err)
	}
	if oschema == nil {
		return nil, fmt.Errorf("resolved output schema for tool %q is nil", t.Name())
	}

	if _, ok := t.(runnableTool); !ok {
		return nil, fmt.Errorf("tool %q (type %T) is not directly runnable in workflow node", t.Name(), t)
	}

	if cfg.RetryConfig != nil {
		cfg.RetryConfig.applyDefaults()
	}

	return &toolNode{
		baseNode:     baseNode{name: t.Name(), description: t.Description(), config: cfg},
		tool:         t,
		inputSchema:  ischema,
		outputSchema: oschema,
	}, nil
}

// NewToolNodeWithSchemas is a convenience wrapper for NewToolNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
func NewToolNodeWithSchemas(t tool.Tool, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (Node, error) {
	return newToolNodeWithSchemasTyped[any, any](t, inputSchema, outputSchema, cfg)
}

// NewToolNodeTyped creates a new node wrapping a tool using generics to
// automatically infer input and output schemas from the provided types.
func NewToolNodeTyped[Input, Output any](t tool.Tool, cfg NodeConfig) (Node, error) {
	return newToolNodeWithSchemasTyped[Input, Output](t, nil, nil, cfg)
}

// NewToolNode creates a new node wrapping a tool. Input and output schemas
// are inferred as 'any'.
func NewToolNode(t tool.Tool, cfg NodeConfig) (Node, error) {
	return NewToolNodeTyped[any, any](t, cfg)
}

func (n *toolNode) runTool(toolCtx tool.Context, input any) (any, error) {
	runnable := n.tool.(runnableTool)
	toolInput, err := typeutil.ConvertToWithJSONSchema[any, any](input, n.inputSchema)
	if err != nil {
		return nil, fmt.Errorf("converting input for tool %q: %w", n.tool.Name(), err)
	}

	output, err := runnable.Run(toolCtx, toolInput)
	if err != nil {
		return nil, fmt.Errorf("tool %q execution failed: %w", n.tool.Name(), err)
	}

	var toolOutput any = output

	// Validate
	if err := n.outputSchema.Validate(output); err != nil {
		if val, ok := output["result"]; ok {
			if err := n.outputSchema.Validate(val); err != nil {
				return nil, fmt.Errorf("converting tool %q output: validation failed for result key: %w", n.tool.Name(), err)
			}
			toolOutput = val
		} else {
			return nil, fmt.Errorf("converting tool %q output: validation failed: %w", n.tool.Name(), err)
		}
	}

	return toolOutput, nil
}

// Run implements the Node interface and executes the tool.
func (n *toolNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		eventActions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
		toolCtx := toolinternal.NewToolContext(ctx, uuid.NewString(), eventActions, nil)

		toolOutput, err := n.runTool(toolCtx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Actions = *eventActions
		event.Actions.StateDelta["output"] = toolOutput

		// If output is a string, set it as content for convenience (similar to FunctionNode).
		if s, ok := toolOutput.(string); ok {
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
