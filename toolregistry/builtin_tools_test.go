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

package toolregistry_test

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/adk/toolregistry"
)

// stubState is a minimal session.State backed by a map.
type stubState struct {
	m map[string]any
}

func (s *stubState) Get(key string) (any, error) {
	if v, ok := s.m[key]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}
func (s *stubState) Set(key string, val any) error { s.m[key] = val; return nil }
func (s *stubState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

// stubToolContext is a minimal tool.Context for invoking the builtin
// list_tools / load_tool / unload_tool function tools directly.
type stubToolContext struct {
	context.Context
	state    *stubState
	actions  *session.EventActions
	callID   string
	branch   string
}

func (c *stubToolContext) AgentName() string                                                        { return "stub" }
func (c *stubToolContext) AppName() string                                                          { return "test" }
func (c *stubToolContext) Branch() string                                                           { return c.branch }
func (c *stubToolContext) InvocationID() string                                                     { return "inv" }
func (c *stubToolContext) ReadonlyState() session.ReadonlyState                                     { return c.state }
func (c *stubToolContext) State() session.State                                                     { return c.state }
func (c *stubToolContext) SessionID() string                                                        { return "sess" }
func (c *stubToolContext) UserContent() *genai.Content                                              { return nil }
func (c *stubToolContext) UserID() string                                                           { return "u" }
func (c *stubToolContext) Artifacts() agent.Artifacts                                               { return nil }
func (c *stubToolContext) FunctionCallID() string                                                   { return c.callID }
func (c *stubToolContext) Actions() *session.EventActions                                           { return c.actions }
func (c *stubToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error)     { return nil, nil }
func (c *stubToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation                    { return nil }
func (c *stubToolContext) RequestConfirmation(string, any) error                                    { return nil }

var _ tool.Context = (*stubToolContext)(nil)

func TestNewListToolsTool_RejectsNilRegistry(t *testing.T) {
	if _, err := toolregistry.NewListToolsTool(nil); err == nil {
		t.Error("expected error")
	}
}

func TestNewLoadToolTool_RejectsNilRegistry(t *testing.T) {
	if _, err := toolregistry.NewLoadToolTool(nil); err == nil {
		t.Error("expected error")
	}
}

func TestListTools_BuildsCorrectName(t *testing.T) {
	reg := toolregistry.New()
	tt, err := toolregistry.NewListToolsTool(reg)
	if err != nil {
		t.Fatalf("NewListToolsTool: %v", err)
	}
	if tt.Name() != "list_tools" {
		t.Errorf("Name = %q, want list_tools", tt.Name())
	}
}

func TestLoadTool_BuildsCorrectName(t *testing.T) {
	reg := toolregistry.New()
	tt, err := toolregistry.NewLoadToolTool(reg)
	if err != nil {
		t.Fatalf("NewLoadToolTool: %v", err)
	}
	if tt.Name() != "load_tool" {
		t.Errorf("Name = %q, want load_tool", tt.Name())
	}
}

func TestUnloadTool_BuildsCorrectName(t *testing.T) {
	reg := toolregistry.New()
	tt, err := toolregistry.NewUnloadToolTool(reg)
	if err != nil {
		t.Fatalf("NewUnloadToolTool: %v", err)
	}
	if tt.Name() != "unload_tool" {
		t.Errorf("Name = %q, want unload_tool", tt.Name())
	}
}

func TestRegistry_LazyBuilderRunsOnce(t *testing.T) {
	// Smoke check that builder caching works through the public surface.
	reg := toolregistry.New()
	calls := 0
	reg.Register(toolregistry.Info{Name: "lazy"}, func() (tool.Tool, error) {
		calls++
		return mustTool(t, "lazy"), nil
	})
	for i := 0; i < 3; i++ {
		_, _ = reg.Get("lazy")
	}
	if calls != 1 {
		t.Errorf("builder calls = %d, want 1", calls)
	}
}
