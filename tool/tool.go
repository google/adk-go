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

// Package tool defines the interfaces for tools that can be called by an agent.
// A tool is a piece of code that performs a specific task. You can either define
// your own custom tools or use built-in ones, for example, GoogleSearch.
package tool

import (
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// ErrConfirmationRequired indicates that the tool requires confirmation.
var ErrConfirmationRequired = errors.New("requires confirmation, please approve or reject")

// ErrConfirmationRejected indicated that the tool call confirmation rejected.
var ErrConfirmationRejected = errors.New("call is rejected")

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
// Deprecated: use AllowedToolsPredicate instead.
func StringPredicate(allowedTools []string) Predicate {
	return AllowedToolsPredicate(allowedTools)
}

// AllowedToolsPredicate returns a Predicate that allows only the tools with the given names.
func AllowedToolsPredicate(allowedTools []string) Predicate {
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
//
// EXPERIMENTAL: ConfirmationProvider is experimental and not currently in scope for the v1.0 API.
type ConfirmationProvider func(toolName string, toolInput any) bool

// WithConfirmation wraps a toolset to inject confirmation logic in each tool.
// Only tools within the provided Toolset that implement the `runnableTool` interface
// (i.e., provide a FunctionDeclaration and a Run method) will be wrapped with
// confirmation logic. Other tools will be included in the returned Toolset without
// modification.
//
// EXPERIMENTAL: WithConfirmation is experimental and not currently in scope for the v1.0 API.
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
				runnableTool:        t.(runnableTool),
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
	runnableTool
	requireConfirmation bool
	provider            ConfirmationProvider
}

type runnableTool interface {
	Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (result map[string]any, err error)
}

func (t *confirmationTool) Declaration() *genai.FunctionDeclaration {
	return t.runnableTool.Declaration()
}

func (t *confirmationTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if rp, ok := t.runnableTool.(interface {
		ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
	}); ok {
		_, existedBefore := req.Tools[t.Name()]
		if err := rp.ProcessRequest(ctx, req); err != nil {
			return err
		}
		// If the inner tool packed itself into req.Tools during ProcessRequest,
		// replace it with the confirmation wrapper so confirmationTool.Run is invoked.
		if !existedBefore && req.Tools != nil && req.Tools[t.Name()] != nil {
			req.Tools[t.Name()] = t
			return nil
		}
	}
	return toolutils.PackTool(req, t)
}

func (t *confirmationTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	ft := t.runnableTool

	// Check for Human-in-the-Loop confirmation.
	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return nil, fmt.Errorf("error tool %q %w", t.runnableTool.Name(), ErrConfirmationRejected)
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
			return nil, fmt.Errorf("error tool %q %w", t.Name(), ErrConfirmationRequired)
		}
	}

	return ft.Run(ctx, args)
}
