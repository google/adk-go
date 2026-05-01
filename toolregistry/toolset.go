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
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// Toolset surfaces the always-on discovery tools (list_tools, load_tool)
// and the tools the agent has actively loaded into session state.
//
// Tools is invoked by the LlmAgent on each turn to build the
// FunctionDeclarations sent to the model, so loading takes effect on the
// next LLM request after a load_tool call.
type Toolset struct {
	registry *Registry
	alwaysOn []tool.Tool
}

// ToolsetOption customizes a Toolset.
type ToolsetOption func(*Toolset)

// WithAlwaysOn replaces the default always-on tools (list_tools,
// load_tool) with the supplied list. Pass an empty slice to disable
// discovery entirely (rare — usually only for tests).
func WithAlwaysOn(tools ...tool.Tool) ToolsetOption {
	return func(t *Toolset) { t.alwaysOn = append([]tool.Tool(nil), tools...) }
}

// NewToolset returns a Toolset that pairs reg's discovery tools with any
// loaded tools recorded in session state.
//
// By default the always-on set is [list_tools, load_tool]. Override with
// WithAlwaysOn — for instance to add an unload_tool tool, or to swap in
// custom discovery prompts.
func NewToolset(reg *Registry, opts ...ToolsetOption) *Toolset {
	t := &Toolset{registry: reg}
	for _, opt := range opts {
		opt(t)
	}
	if t.alwaysOn == nil {
		list, _ := NewListToolsTool(reg)
		load, _ := NewLoadToolTool(reg)
		t.alwaysOn = []tool.Tool{list, load}
	}
	return t
}

// Name implements tool.Toolset.
func (t *Toolset) Name() string { return "tool_registry" }

// Tools implements tool.Toolset. Returns the always-on discovery tools
// followed by every tool whose name is recorded in
// state[StateKeyLoadedTools]. Unknown names in state are skipped (so a
// stale state key doesn't crash the agent loop) and a one-line warning
// would be appropriate at runtime — emitted as a debug log per call.
func (t *Toolset) Tools(rctx agent.ReadonlyContext) ([]tool.Tool, error) {
	out := append([]tool.Tool{}, t.alwaysOn...)
	if rctx == nil {
		return out, nil
	}
	state := rctx.ReadonlyState()
	if state == nil {
		return out, nil
	}
	v, err := state.Get(StateKeyLoadedTools)
	if err != nil {
		// ErrStateKeyNotExist is expected on a fresh session.
		return out, nil
	}
	names := coerceLoadedNames(v)
	seen := map[string]bool{}
	// Avoid double-listing always-on tools.
	for _, t := range t.alwaysOn {
		seen[t.Name()] = true
	}
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		tt, err := t.registry.Get(name)
		if err != nil {
			continue
		}
		out = append(out, tt)
	}
	return out, nil
}

// ProcessRequest implements the structural RequestProcessor contract
// the LlmAgent flow uses to wire toolsets into the LLM request. Without
// this method, base_flow.toolsetPreprocess silently skips this toolset
// and the dynamically-loaded tools never reach req.Tools — the LLM
// then sees the load succeed but the tool itself never declared.
//
// The method calls Tools(ctx) to get the currently-loaded set, then
// delegates ProcessRequest to each tool that implements it (every
// functiontool.New tool does).
func (t *Toolset) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	tools, err := t.Tools(ctx)
	if err != nil {
		return fmt.Errorf("toolregistry: enumerating tools: %w", err)
	}
	type processor interface {
		ProcessRequest(ctx tool.Context, req *model.LLMRequest) error
	}
	for _, tt := range tools {
		rp, ok := tt.(processor)
		if !ok {
			return fmt.Errorf("toolregistry: tool %q does not implement ProcessRequest", tt.Name())
		}
		if err := rp.ProcessRequest(ctx, req); err != nil {
			return fmt.Errorf("toolregistry: tool %q ProcessRequest: %w", tt.Name(), err)
		}
	}
	return nil
}

// coerceLoadedNames converts a session state value into a string slice.
// State persistence may serialize through JSON, in which case []string
// arrives as []any. Both shapes are honored.
func coerceLoadedNames(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}
