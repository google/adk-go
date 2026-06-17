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
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// ToolNode wraps a tool from the tool package.
type ToolNode struct {
	BaseNode
	tool tool.Tool
}

// runnableTool is the internal interface that Node uses to invoke tools.
type runnableTool interface {
	Run(ctx agent.Context, args any) (map[string]any, error)
}

// newToolNodeWithSchemasTyped creates a new node wrapping a tool with explicitly provided schemas.
// If a schema is nil, it will be inferred from the corresponding generic type Input or Output.
func newToolNodeWithSchemasTyped[Input, Output any](t tool.Tool, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*ToolNode, error) {
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

	return &ToolNode{
		BaseNode: NewBaseNodeWithSchemas(t.Name(), t.Description(), cfg, ischema, oschema),
		tool:     t,
	}, nil
}

// NewToolNodeWithSchemas is a convenience wrapper for NewToolNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
func NewToolNodeWithSchemas(t tool.Tool, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*ToolNode, error) {
	return newToolNodeWithSchemasTyped[any, any](t, inputSchema, outputSchema, cfg)
}

// NewToolNodeTyped creates a new node wrapping a tool using generics to
// automatically infer input and output schemas from the provided types.
func NewToolNodeTyped[Input, Output any](t tool.Tool, cfg NodeConfig) (*ToolNode, error) {
	return newToolNodeWithSchemasTyped[Input, Output](t, nil, nil, cfg)
}

// NewToolNode creates a new node wrapping a tool. Input and output schemas
// are inferred as 'any'.
func NewToolNode(t tool.Tool, cfg NodeConfig) (*ToolNode, error) {
	return NewToolNodeTyped[any, any](t, cfg)
}

func (n *ToolNode) runTool(toolCtx agent.Context, input any) (any, error) {
	runnable := n.tool.(runnableTool)
	// Upstream nodes (like LLM Agents) frequently produce serialized JSON strings representing
	// structured tool call arguments. Since ToolNodes expect structured key-value mappings (maps)
	// to conform to their input validation schemas, receiving a raw JSON string would cause a crash.
	// If the input is a raw string, we attempt to eagerly unmarshal it as JSON into a map[string]any
	// to make the workflow node fully self-healing and compatible with upstream text outputs.
	if s, ok := input.(string); ok {
		var m map[string]any
		if json.Unmarshal([]byte(s), &m) == nil {
			input = m
		}
	}

	toolInput, err := n.ValidateInput(input)
	if err != nil {
		return nil, fmt.Errorf("converting input for tool %q: %w", n.tool.Name(), err)
	}

	output, err := runnable.Run(toolCtx, toolInput)
	if err != nil {
		return nil, fmt.Errorf("tool %q execution failed: %w", n.tool.Name(), err)
	}

	return output, nil
}

// ValidateOutput validates the tool output against the node's output
// schema, adding a FunctionTool-specific fallback on top of the default
// behavior: when the output is a map of shape {"result": X} that fails
// direct schema validation, it retries against the unwrapped "result"
// value and, on success, returns that unwrapped value.
//
// This override is the home for the {"result": X} convention because it
// is tool-specific; making it a general default could mask genuine
// validation errors in other node types.
func (n *ToolNode) ValidateOutput(out any) (any, error) {
	schema := n.OutputSchema()
	if schema == nil {
		return out, nil
	}
	// Try standard validation first.
	if validated, err := defaultValidateOutput(out, schema); err == nil {
		return validated, nil
	}
	// Fallback: unwrap {"result": X} (FunctionTool convention).
	if m, ok := out.(map[string]any); ok {
		if val, ok := m["result"]; ok {
			if validated, err := defaultValidateOutput(val, schema); err == nil {
				return validated, nil
			}
		}
	}
	// Both attempts failed: return the error from validating the
	// original output, not the one from the {"result": X} fallback,
	// so the caller sees the actual schema mismatch.
	return defaultValidateOutput(out, schema)
}

// Run implements the Node interface and executes the tool.
func (n *ToolNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		eventActions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
		toolCtx := agent.NewToolContext(ctx, uuid.NewString(), eventActions, nil)

		toolOutput, err := n.runTool(toolCtx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		event := session.NewEvent(ctx.InvocationID())
		event.Actions = *eventActions
		event.Output = toolOutput

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
