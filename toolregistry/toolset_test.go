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
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/toolregistry"
)

// stubReadonlyState is a minimal ReadonlyState backed by an in-memory map.
type stubReadonlyState struct {
	m map[string]any
}

func (s *stubReadonlyState) Get(key string) (any, error) {
	if v, ok := s.m[key]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}
func (s *stubReadonlyState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

// stubReadonlyContext satisfies agent.ReadonlyContext for Toolset.Tools.
type stubReadonlyContext struct {
	context.Context
	state *stubReadonlyState
}

func (s *stubReadonlyContext) AgentName() string                    { return "stub" }
func (s *stubReadonlyContext) AppName() string                      { return "test" }
func (s *stubReadonlyContext) Branch() string                       { return "" }
func (s *stubReadonlyContext) InvocationID() string                 { return "inv" }
func (s *stubReadonlyContext) ReadonlyState() session.ReadonlyState { return s.state }
func (s *stubReadonlyContext) SessionID() string                    { return "sess" }
func (s *stubReadonlyContext) UserContent() *genai.Content          { return nil }
func (s *stubReadonlyContext) UserID() string                       { return "u" }

func TestToolset_DefaultAlwaysOnIncludesListAndLoad(t *testing.T) {
	reg := toolregistry.New()
	ts := toolregistry.NewToolset(reg)
	tools, err := ts.Tools(&stubReadonlyContext{state: &stubReadonlyState{m: map[string]any{}}})
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	names := map[string]bool{}
	for _, tt := range tools {
		names[tt.Name()] = true
	}
	if !names["list_tools"] || !names["load_tool"] {
		t.Errorf("default always-on missing list_tools or load_tool; got %v", names)
	}
}

func TestToolset_AppendsLoadedFromState(t *testing.T) {
	reg := toolregistry.New()
	reg.RegisterTool(mustTool(t, "calc"), toolregistry.Info{Name: "calc"})
	reg.RegisterTool(mustTool(t, "search"), toolregistry.Info{Name: "search"})

	ts := toolregistry.NewToolset(reg)
	rctx := &stubReadonlyContext{state: &stubReadonlyState{m: map[string]any{
		toolregistry.StateKeyLoadedTools: []string{"calc"},
	}}}
	tools, err := ts.Tools(rctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	names := map[string]bool{}
	for _, tt := range tools {
		names[tt.Name()] = true
	}
	if !names["calc"] {
		t.Error("expected calc to be loaded")
	}
	if names["search"] {
		t.Error("search should not be loaded")
	}
}

func TestToolset_AcceptsAnySliceFromState(t *testing.T) {
	// JSON deserialization yields []any; the toolset must coerce.
	reg := toolregistry.New()
	reg.RegisterTool(mustTool(t, "calc"), toolregistry.Info{Name: "calc"})
	ts := toolregistry.NewToolset(reg)
	rctx := &stubReadonlyContext{state: &stubReadonlyState{m: map[string]any{
		toolregistry.StateKeyLoadedTools: []any{"calc"},
	}}}
	tools, _ := ts.Tools(rctx)
	found := false
	for _, tt := range tools {
		if tt.Name() == "calc" {
			found = true
		}
	}
	if !found {
		t.Error("[]any state value should be honored")
	}
}

func TestToolset_SkipsUnknownNames(t *testing.T) {
	reg := toolregistry.New()
	ts := toolregistry.NewToolset(reg)
	rctx := &stubReadonlyContext{state: &stubReadonlyState{m: map[string]any{
		toolregistry.StateKeyLoadedTools: []string{"does_not_exist"},
	}}}
	tools, err := ts.Tools(rctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	for _, tt := range tools {
		if tt.Name() == "does_not_exist" {
			t.Error("unknown tool should be skipped")
		}
	}
}

func TestToolset_NoDuplicateAlwaysOnIfReloadedByName(t *testing.T) {
	// State accidentally lists "list_tools" — should not duplicate.
	reg := toolregistry.New()
	ts := toolregistry.NewToolset(reg)
	rctx := &stubReadonlyContext{state: &stubReadonlyState{m: map[string]any{
		toolregistry.StateKeyLoadedTools: []string{"list_tools"},
	}}}
	tools, _ := ts.Tools(rctx)
	count := 0
	for _, tt := range tools {
		if tt.Name() == "list_tools" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("list_tools count = %d, want 1", count)
	}
}

// TestToolset_ProcessRequest_NoDuplicateOnRepeatPack guards against the
// regression where ProcessRequest re-packed alwaysOn tools that
// toolPreprocess had already packed via the f.Tools path, causing
// PackTool's duplicate check to fire on turn 1 with
// `toolregistry: tool "list_tools" ProcessRequest: duplicate tool: "list_tools"`.
func TestToolset_ProcessRequest_NoDuplicateOnRepeatPack(t *testing.T) {
	reg := toolregistry.New()
	ts := toolregistry.NewToolset(reg)

	// Simulate the state after toolPreprocess has already packed the
	// alwaysOn tools: req.Tools already contains list_tools + load_tool.
	req := &model.LLMRequest{Tools: map[string]any{}}
	preTools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	for _, tt := range preTools {
		req.Tools[tt.Name()] = tt
	}
	wantSize := len(req.Tools)
	if wantSize == 0 {
		t.Fatalf("expected pre-packed alwaysOn tools, got 0")
	}

	if err := ts.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if len(req.Tools) != wantSize {
		t.Errorf("req.Tools size = %d, want %d (no new packs expected)", len(req.Tools), wantSize)
	}
}

// TestToolset_ProcessRequest_PacksMissingTools guards the original
// 2031e50 behavior: tools the agent's cached f.Tools never picked up
// (in production: dynamically loaded via load_tool; here: simulated by
// pre-packing only one alwaysOn) must still reach req.Tools through
// Toolset.ProcessRequest. The dedup-skip path and the pack path go
// through the same code, so this also proves the load_tool flow.
func TestToolset_ProcessRequest_PacksMissingTools(t *testing.T) {
	reg := toolregistry.New()
	ts := toolregistry.NewToolset(reg)

	preTools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(preTools) < 2 {
		t.Fatalf("expected >=2 alwaysOn tools, got %d", len(preTools))
	}

	// Pre-pack only the first alwaysOn tool, leaving the second to be
	// added by ProcessRequest.
	first, second := preTools[0], preTools[1]
	req := &model.LLMRequest{Tools: map[string]any{first.Name(): first}}

	if err := ts.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if _, ok := req.Tools[first.Name()]; !ok {
		t.Errorf("first tool %q dropped from req.Tools", first.Name())
	}
	if _, ok := req.Tools[second.Name()]; !ok {
		t.Errorf("second tool %q not packed by ProcessRequest", second.Name())
	}
}

// Compile-time check that the Toolset satisfies the public interface.
var _ = func() {
	var _ interface {
		Name() string
		Tools(agent.ReadonlyContext) ([]any, error)
	} = nil
}
