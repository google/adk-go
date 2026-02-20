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

// Package tool defines the interfaces for tools that can be called by an agent.
// A tool is a piece of code that performs a specific task. You can either define
// your own custom tools or use built-in ones, for example, GoogleSearch.
package tool

import (
	"context"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// Tool defines the interface for a callable tool.
type Tool interface {
	// Name returns the name of the tool.
	Name() string
	// Description returns a description of the tool.
	Description() string
	// IsLongRunning indicates whether the tool is a long-running operation,
	// which typically returns a resource id first and finishes the operation later.
	IsLongRunning() bool
}

// Context defines the interface for the context passed to a tool when it's
// called. It provides access to invocation-specific information and allows
// the tool to interact with the agent's state and memory.
type Context interface {
	agent.CallbackContext
	// FunctionCallID returns the unique identifier of the function call
	// that triggered this tool execution.
	FunctionCallID() string

	// Actions returns the EventActions for the current event. This can be
	// used by the tool to modify the agent's state, transfer to another
	// agent, or perform other actions.
	Actions() *session.EventActions
	// SearchMemory performs a semantic search on the agent's memory.
	SearchMemory(context.Context, string) (*memory.SearchResponse, error)

	// ToolConfirmation returns a handler for checking the Human-in-the-Loop
	// confirmation status for the current tool context. This should be used within a tool's logic
	// *before* performing any sensitive operations that require user approval.
	//
	// Example Usage:
	// if confirmation := ctx.ToolConfirmation(); confirmation == nil {
	//     // Confirmation required, create confirmation or handle appropriately
	//     ctx.RequestConfirmation("hint", payload)
	// }
	//
	// The returned *toolconfirmation.ToolConfirmation object provides methods to check the actual
	// confirmation state.
	ToolConfirmation() *toolconfirmation.ToolConfirmation

	// RequestConfirmation initiates the Human-in-the-Loop (HITL) process to ask the user for approval
	// before the tool proceeds with a specific action. Call this method when a tool needs
	// explicit user consent.
	//
	// This will typically result in the ADK emitting a special event
	// (e.g., a FunctionCall like "adk_request_confirmation") to the client application/UI,
	// prompting the user for a decision.
	//
	// Args:
	//   - hint: A human-readable string explaining why confirmation is needed. This is usually
	//     displayed to the user in the confirmation prompt.
	//   - payload: Any additional data or context about the action requiring confirmation.
	//
	// Returns:
	//   - nil: If the confirmation request was successfully enqueued or initiated within the ADK.
	//     This indicates that the process of asking the user has begun. It does NOT mean the action
	//     is approved. The tool's execution will likely pause or be suspended until the user responds.
	//   - error: If there was a failure in initiating the confirmation process itself (e.g., invalid
	//     arguments, issue with the event system). The request to ask the user has not been sent.
	RequestConfirmation(hint string, payload any) error
}

// Toolset is an interface for a collection of tools. It allows grouping
// related tools together and providing them to an agent.
type Toolset interface {
	// Name returns the name of the toolset.
	Name() string
	// Tools returns a list of tools in the toolset. The provided
	// ReadonlyContext can be used to dynamically determine which tools
	// to return based on the current invocation state.
	Tools(ctx agent.ReadonlyContext) ([]Tool, error)
}

// Predicate is a function which decides whether a tool should be exposed to LLM.
type Predicate func(ctx agent.ReadonlyContext, tool Tool) bool

// StringPredicate is a helper that creates a Predicate from a string slice.
func StringPredicate(allowedTools []string) Predicate {
	m := make(map[string]bool)
	for _, t := range allowedTools {
		m[t] = true
	}

	return func(ctx agent.ReadonlyContext, tool Tool) bool {
		return m[tool.Name()]
	}
}

// FilterToolset returns a Toolset that filters the tools in the given Toolset
// using the given predicate.
func FilterToolset(toolset Toolset, predicate Predicate) Toolset {
	if toolset == nil {
		panic("toolset must not be nil")
	}
	if predicate == nil {
		panic("predicate must not be nil")
	}

	return &filteredToolset{
		toolset:   toolset,
		predicate: predicate,
	}
}

type filteredToolset struct {
	toolset   Toolset
	predicate Predicate
}

func (f *filteredToolset) Name() string {
	return f.toolset.Name()
}

func (f *filteredToolset) Tools(ctx agent.ReadonlyContext) ([]Tool, error) {
	tools, err := f.toolset.Tools(ctx)
	if err != nil {
		return nil, err
	}
	var filtered []Tool
	for _, tool := range tools {
		if f.predicate(ctx, tool) {
			filtered = append(filtered, tool)
		}
	}
	return filtered, nil
}

// ConfirmationProvider defines a function that dynamically determines whether
// a specific tool execution requires user confirmation.
//
// It accepts the tool name and the input parameters as arguments.
// Returning true signals that the system must wait for Human-in-the-Loop (HITL)
// approval before proceeding with the execution.
type ConfirmationProvider func(toolName string, toolInput any) bool

// WithConfirmation wraps a toolset to inject confirmation logic in each tool.
// Only tools within the provided Toolset that implement the `runnableTool` interface
// (i.e., provide a FunctionDeclaration and a Run method) will be wrapped with
// confirmation logic. Other tools will be included in the returned Toolset without
// modification.
func WithConfirmation(ts Toolset, requireConfirmation bool, requireConfirmationProvider ConfirmationProvider) Toolset {
	return &confirmationToolset{
		toolset:                     ts,
		requireConfirmation:         requireConfirmation,
		requireConfirmationProvider: requireConfirmationProvider,
	}
}

type confirmationToolset struct {
	toolset                     Toolset
	requireConfirmation         bool
	requireConfirmationProvider ConfirmationProvider
}

func (c *confirmationToolset) Name() string { return c.toolset.Name() }

func (c *confirmationToolset) Tools(ctx agent.ReadonlyContext) ([]Tool, error) {
	tools, err := c.toolset.Tools(ctx)
	if err != nil {
		return nil, err
	}
	wrappedTools := make([]Tool, 0, len(tools))
	for _, t := range tools {
		if _, ok := t.(runnableTool); ok {
			wrappedTools = append(wrappedTools, &confirmationTool{
				Tool:                t,
				requireConfirmation: c.requireConfirmation,
				provider:            c.requireConfirmationProvider,
			})
		} else {
			// Tools that are not runnableTool are not wrapped.
			wrappedTools = append(wrappedTools, t)
		}
	}

	return wrappedTools, nil
}

// confirmationTool is a wrapper around a tool that adds confirmation logic.
// It implements tool.Tool and adk/internal/toolinternal.FunctionTool and adk/internal/toolinternal.RequestProcessor.
type confirmationTool struct {
	Tool
	requireConfirmation bool
	provider            ConfirmationProvider
}

type runnableTool interface {
	Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx Context, args any) (result map[string]any, err error)
}

func (t *confirmationTool) Declaration() *genai.FunctionDeclaration {
	return t.Tool.(runnableTool).Declaration()
}

func (t *confirmationTool) ProcessRequest(ctx Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func (t *confirmationTool) Run(ctx Context, args any) (map[string]any, error) {
	ft := t.Tool.(runnableTool)

	// Check for Human-in-the-Loop confirmation.
	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return nil, fmt.Errorf("tool %q call is rejected", t.Tool.Name())
		}
	} else {
		requireConfirmation := t.requireConfirmation
		if t.provider != nil {
			requireConfirmation = t.provider(t.Name(), args)
		}

		if requireConfirmation {
			err := ctx.RequestConfirmation(
				fmt.Sprintf("Please approve or reject the tool call %s() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					t.Name()), nil)
			if err != nil {
				return nil, err
			}
			ctx.Actions().SkipSummarization = true
			return nil, fmt.Errorf("tool %q requires confirmation, please approve or reject", t.Name())
		}
	}

	return ft.Run(ctx, args)
}
