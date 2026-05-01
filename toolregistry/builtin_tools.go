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

package toolregistry

import (
	"errors"
	"fmt"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// ListToolsArgs is the input shape for list_tools. Empty fields match
// everything.
type ListToolsArgs struct {
	// Query is a case-insensitive substring matched against tool name,
	// description, and tags.
	Query string `json:"query,omitempty"`

	// Tags requires every listed tag be present on the tool.
	Tags []string `json:"tags,omitempty"`
}

// ListToolsResult is what list_tools returns to the LLM.
type ListToolsResult struct {
	Tools []Info `json:"tools"`
}

// NewListToolsTool builds a tool the LLM uses to discover what's
// available without paying the FunctionDeclaration cost up front.
//
// The tool returns a ListToolsResult describing matching tools; the LLM
// then chooses which to load via load_tool.
func NewListToolsTool(reg *Registry) (tool.Tool, error) {
	if reg == nil {
		return nil, errors.New("toolregistry: NewListToolsTool: registry must not be nil")
	}
	return functiontool.New[ListToolsArgs, ListToolsResult](
		functiontool.Config{
			Name: "list_tools",
			Description: "List the tools available in the dynamic tool registry. " +
				"Filter by an optional substring query and optional tags. " +
				"Use this BEFORE calling load_tool to discover what is available.",
		},
		func(_ tool.Context, args ListToolsArgs) (ListToolsResult, error) {
			return ListToolsResult{Tools: reg.List(Filter{Query: args.Query, Tags: args.Tags})}, nil
		},
	)
}

// LoadToolArgs is the input shape for load_tool.
type LoadToolArgs struct {
	// Name is the registered tool name to activate.
	Name string `json:"name"`
}

// LoadToolResult is what load_tool returns to the LLM.
type LoadToolResult struct {
	// Loaded reports the names currently active after this call.
	Loaded []string `json:"loaded"`

	// Message is a human-readable summary of the action taken.
	Message string `json:"message"`
}

// NewLoadToolTool builds the tool the LLM uses to activate a specific
// tool by name. Activation persists in session state (under
// StateKeyLoadedTools) so the tool surfaces on the next LLM request.
//
// Calling load_tool on an already-loaded name is a no-op (the message
// reflects this). Unknown names return an error to the LLM.
func NewLoadToolTool(reg *Registry) (tool.Tool, error) {
	if reg == nil {
		return nil, errors.New("toolregistry: NewLoadToolTool: registry must not be nil")
	}
	return functiontool.New[LoadToolArgs, LoadToolResult](
		functiontool.Config{
			Name: "load_tool",
			Description: "Activate a tool by name from the dynamic tool registry. " +
				"After calling this, the tool will be available in subsequent turns. " +
				"Discover available tools by calling list_tools first.",
		},
		func(ctx tool.Context, args LoadToolArgs) (LoadToolResult, error) {
			if args.Name == "" {
				return LoadToolResult{}, errors.New("load_tool: name must not be empty")
			}
			if !reg.Has(args.Name) {
				return LoadToolResult{}, fmt.Errorf("load_tool: tool %q is not registered", args.Name)
			}
			state := ctx.State()
			if state == nil {
				return LoadToolResult{}, errors.New("load_tool: tool context has nil State")
			}
			current, _ := state.Get(StateKeyLoadedTools)
			names := coerceLoadedNames(current)
			for _, n := range names {
				if n == args.Name {
					return LoadToolResult{Loaded: names, Message: fmt.Sprintf("%q already loaded", args.Name)}, nil
				}
			}
			names = append(names, args.Name)
			if err := state.Set(StateKeyLoadedTools, names); err != nil {
				return LoadToolResult{}, fmt.Errorf("load_tool: persist state: %w", err)
			}
			return LoadToolResult{
				Loaded:  names,
				Message: fmt.Sprintf("loaded %q. It will be available on the next turn.", args.Name),
			}, nil
		},
	)
}

// UnloadToolArgs is the input shape for unload_tool.
type UnloadToolArgs struct {
	Name string `json:"name"`
}

// UnloadToolResult is what unload_tool returns.
type UnloadToolResult struct {
	Loaded  []string `json:"loaded"`
	Message string   `json:"message"`
}

// NewUnloadToolTool builds an optional tool the LLM uses to deactivate a
// previously-loaded tool. Useful when a long-lived session has loaded
// many tools and the agent wants to focus the LLM context.
//
// Pass to NewToolset via WithAlwaysOn alongside list_tools / load_tool
// to expose unload to the model.
func NewUnloadToolTool(reg *Registry) (tool.Tool, error) {
	return functiontool.New[UnloadToolArgs, UnloadToolResult](
		functiontool.Config{
			Name:        "unload_tool",
			Description: "Deactivate a previously-loaded tool by name. The tool will not appear in subsequent turns until load_tool is called again.",
		},
		func(ctx tool.Context, args UnloadToolArgs) (UnloadToolResult, error) {
			if args.Name == "" {
				return UnloadToolResult{}, errors.New("unload_tool: name must not be empty")
			}
			state := ctx.State()
			if state == nil {
				return UnloadToolResult{}, errors.New("unload_tool: tool context has nil State")
			}
			current, _ := state.Get(StateKeyLoadedTools)
			names := coerceLoadedNames(current)
			out := make([]string, 0, len(names))
			removed := false
			for _, n := range names {
				if n == args.Name {
					removed = true
					continue
				}
				out = append(out, n)
			}
			if !removed {
				return UnloadToolResult{Loaded: names, Message: fmt.Sprintf("%q was not loaded", args.Name)}, nil
			}
			if err := state.Set(StateKeyLoadedTools, out); err != nil {
				return UnloadToolResult{}, fmt.Errorf("unload_tool: persist state: %w", err)
			}
			return UnloadToolResult{Loaded: out, Message: fmt.Sprintf("unloaded %q", args.Name)}, nil
		},
	)
}
