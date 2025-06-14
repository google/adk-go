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

package adk

import (
	"context"
	"fmt"

	"github.com/google/adk-go/internal/jsonschema"
)

// ToolContext is the tool invocation context.
type ToolContext struct {
	// The invocation context of the tool call.
	InvocationContext *InvocationContext

	// The function call id of the current tool call.
	// This id was returned in the function call event from LLM to identify
	// a function call. If LLM didn't return an id, ADK will assign one t it.
	// This id is used to map function call response to the original function call.
	FunctionCallID string

	// The event actions of the current tool call.
	EventActions *EventActions
}

// TODO: Implement an MCP client as a tool to validate whether
// a concrete type is sufficient for tool use.

// Tool is an ADK tool.
type Tool struct {
	Name        string
	Description string
	InputSchema *jsonschema.Schema
	// TODO: OutputSchema

	rawHandler rawToolHandler
}

// Run executes the tool with the provided context and yields events.
func (t *Tool) Run(ctx context.Context, tc *ToolContext, args map[string]any) (map[string]any, error) {
	return t.rawHandler(ctx, tc, args)
}

// ToolHandler is the execution handler of a tool.
type ToolHandler[In, Out any] func(ctx context.Context, input In) (output Out, err error)

type rawToolHandler func(ctx context.Context, tc *ToolContext, args map[string]any) (map[string]any, error)

// NewTool creates a new tool with a name, description, and the provided handler.
// Input schema is automatically inferred from the input and output types.
func NewTool[In, Out any](name string, description string, handler ToolHandler[In, Out]) *Tool {
	// TODO(jbd): Add ToolOption as variadic arguments to NewTool.
	ischema, err := jsonschema.For[In]()
	if err != nil {
		panic(fmt.Errorf("NewTool(%q): %w", name, err))
	}
	rawHandler := func(ctx context.Context, tc *ToolContext, args map[string]any) (map[string]any, error) {
		panic("not yet implemented")
		// TODO: Handle function call request from tc.InvocationContext.
		// TODO: Unmarshal into input.
		// TODO: Make a call to handler.
		// TODO: Yield events with the output.
	}
	return &Tool{
		Name:        name,
		Description: description,
		InputSchema: ischema,
		rawHandler:  rawHandler,
	}
}
