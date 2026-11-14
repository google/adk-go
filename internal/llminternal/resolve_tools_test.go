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

package llminternal

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// resolveToolsToolset lists tools, or fails with err when err is set.
type resolveToolsToolset struct {
	name  string
	tools []tool.Tool
	err   error
}

func (s *resolveToolsToolset) Name() string { return s.name }

func (s *resolveToolsToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.tools, nil
}

func toolNames(tools []tool.Tool) []string {
	names := []string{}
	for _, t := range tools {
		names = append(names, t.Name())
	}
	return names
}

func newResolveToolsInvocation(ctx context.Context, t *testing.T, own []tool.Tool, toolsets []tool.Toolset) agent.InvocationContext {
	t.Helper()
	a := &mockLLMAgent{
		Agent: utils.Must(agent.New(agent.Config{Name: "resolve_tools_agent"})),
		s:     &State{Tools: own, Toolsets: toolsets},
	}
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: a})
}

func TestToolProcessor_SkippedToolsetKeepsAnEmptyGroup(t *testing.T) {
	o := &toolsProcessorMockTool{name: "o"}
	w := &toolsProcessorMockTool{name: "w"}
	ctx := newResolveToolsInvocation(t.Context(), t, []tool.Tool{o}, []tool.Toolset{
		&resolveToolsToolset{name: "failing", err: errors.New("mcp down")},
		&resolveToolsToolset{name: "working", tools: []tool.Tool{w}},
	})
	f := &Flow{}

	for _, err := range toolProcessor(ctx, &model.LLMRequest{}, f) {
		if err != nil {
			t.Fatalf("toolProcessor yielded error %v, want the failing toolset skipped", err)
		}
	}

	if got := toolNames(f.Tools); len(got) != 2 || got[0] != "o" || got[1] != "w" {
		t.Errorf("f.Tools = %v, want [o w]", got)
	}
}

func TestToolProcessor_ListingErrorChangesNothing(t *testing.T) {
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	ctx := newResolveToolsInvocation(cancelled, t, []tool.Tool{&toolsProcessorMockTool{name: "o"}}, []tool.Toolset{
		&resolveToolsToolset{name: "working", tools: []tool.Tool{&toolsProcessorMockTool{name: "w"}}},
		&resolveToolsToolset{name: "failing", err: errors.New("mcp down")},
	})
	before := &toolsProcessorMockTool{name: "before"}
	f := &Flow{Tools: []tool.Tool{before}}

	var gotErr error
	for _, err := range toolProcessor(ctx, &model.LLMRequest{}, f) {
		if err != nil {
			gotErr = err
		}
	}

	if !errors.Is(gotErr, context.Canceled) {
		t.Errorf("toolProcessor error = %v, want one wrapping context.Canceled", gotErr)
	}
	if got := toolNames(f.Tools); len(got) != 1 || got[0] != "before" {
		t.Errorf("f.Tools = %v, want [before] (unchanged)", got)
	}
}
