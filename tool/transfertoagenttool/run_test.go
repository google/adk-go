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

package transfertoagenttool_test

import (
	"reflect"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/transfertoagenttool"
)

// newTree returns the root of a tree of two LLM agents, "root" and its
// sub-agent "child".
func newTree(t *testing.T) agent.Agent {
	t.Helper()
	llm := &struct{ model.LLM }{}
	child, err := llmagent.New(llmagent.Config{Name: "child", Model: llm})
	if err != nil {
		t.Fatal(err)
	}
	root, err := llmagent.New(llmagent.Config{Name: "root", Model: llm, SubAgents: []agent.Agent{child}})
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// newThreeLevelTree returns the agents of the tree top -> mid -> leaf.
func newThreeLevelTree(t *testing.T) (top, mid, leaf agent.Agent) {
	t.Helper()
	llm := &struct{ model.LLM }{}
	leaf, err := llmagent.New(llmagent.Config{Name: "leaf", Model: llm})
	if err != nil {
		t.Fatal(err)
	}
	mid, err = llmagent.New(llmagent.Config{Name: "mid", Model: llm, SubAgents: []agent.Agent{leaf}})
	if err != nil {
		t.Fatal(err)
	}
	top, err = llmagent.New(llmagent.Config{Name: "top", Model: llm, SubAgents: []agent.Agent{mid}})
	if err != nil {
		t.Fatal(err)
	}
	return top, mid, leaf
}

// treeContext returns the tool context of a call by self, an agent of root's
// tree, over the invocation context the runner builds: one that carries the
// tree's parent map.
func treeContext(t *testing.T, root, self agent.Agent) agent.Context {
	t.Helper()
	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx := parentmap.ToContext(t.Context(), parents)
	return agent.NewToolContext(icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: self}), "", &session.EventActions{}, nil)
}

// noAgentContext returns a tool context over an invocation context with no
// agent and no parent map.
func noAgentContext(t *testing.T) agent.Context {
	t.Helper()
	return agent.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), "", &session.EventActions{}, nil)
}

func TestNew_RunRecordsReason(t *testing.T) {
	tests := []struct {
		name       string
		args       map[string]any
		wantReason string
	}{
		{name: "reason", args: map[string]any{"agent_name": "target_agent", "transfer_reason": "escalation"}, wantReason: "escalation"},
		{name: "absent", args: map[string]any{"agent_name": "target_agent"}, wantReason: ""},
		{name: "empty", args: map[string]any{"agent_name": "target_agent", "transfer_reason": ""}, wantReason: ""},
		{name: "whitespace", args: map[string]any{"agent_name": "target_agent", "transfer_reason": "  "}, wantReason: "  "},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"target_agent"}, IncludeTransferReason: true})
			ctx := noAgentContext(t)
			got, err := runTool(t, tl, ctx, tc.args)
			if err != nil {
				t.Fatalf("Run(%v) error = %v", tc.args, err)
			}
			if want := map[string]any(nil); !reflect.DeepEqual(got, want) {
				t.Errorf("Run(%v) = %#v, want %#v", tc.args, got, want)
			}
			if got := ctx.Actions().TransferToAgent; got != "target_agent" {
				t.Errorf("TransferToAgent = %q, want %q", got, "target_agent")
			}
			if got := ctx.Actions().TransferReason; got != tc.wantReason {
				t.Errorf("TransferReason = %q, want %q", got, tc.wantReason)
			}
		})
	}
}

func TestNew_RunIgnoresReasonWithoutOption(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"target_agent"}})
	ctx := noAgentContext(t)
	args := map[string]any{"agent_name": "target_agent", "transfer_reason": "x"}
	if _, err := runTool(t, tl, ctx, args); err != nil {
		t.Fatalf("Run(%v) error = %v", args, err)
	}
	if got := ctx.Actions().TransferToAgent; got != "target_agent" {
		t.Errorf("TransferToAgent = %q, want %q", got, "target_agent")
	}
	// adk-python records "x" here, a reason its declaration did not offer.
	if got := ctx.Actions().TransferReason; got != "" {
		t.Errorf("TransferReason = %q, want it unset: the tool does not declare transfer_reason", got)
	}
}

func TestNew_RunIgnoresNonStringReason(t *testing.T) {
	tests := map[string]any{
		"number": 5,
		"bool":   true,
		"list":   []any{"x"},
	}
	for name, reason := range tests {
		t.Run(name, func(t *testing.T) {
			tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"target_agent"}, IncludeTransferReason: true})
			ctx := noAgentContext(t)
			args := map[string]any{"agent_name": "target_agent", "transfer_reason": reason}
			if _, err := runTool(t, tl, ctx, args); err != nil {
				t.Fatalf("Run(%v) error = %v", args, err)
			}
			if got := ctx.Actions().TransferToAgent; got != "target_agent" {
				t.Errorf("TransferToAgent = %q, want %q", got, "target_agent")
			}
			if got := ctx.Actions().TransferReason; got != "" {
				t.Errorf("TransferReason = %q, want it unset for a %T reason", got, reason)
			}
		})
	}
}

func TestNew_RunNameValidation(t *testing.T) {
	root := newTree(t)
	child := root.FindAgent("child")
	top, mid, leaf := newThreeLevelTree(t)
	solo, err := llmagent.New(llmagent.Config{Name: "solo", Model: &struct{ model.LLM }{}})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		names   []string
		tree    agent.Agent // the agent the run's parent map is built from
		self    agent.Agent
		args    any
		wantErr string // empty: the call succeeds and transfers to args' agent_name
	}{
		{name: "listed, not in the tree", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": "listed"}},
		{name: "in the tree, not listed", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": "child"}},
		{name: "root of a sub-agent's tree, not listed", names: []string{"helper"}, tree: root, self: child, args: map[string]any{"agent_name": "root"}},
		{name: "root itself, not listed", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": "root"}},
		{name: "lone agent itself, not listed", names: []string{"x"}, tree: solo, self: solo, args: map[string]any{"agent_name": "solo"}},
		{name: "root two levels up, not listed", names: []string{"x"}, tree: top, self: leaf, args: map[string]any{"agent_name": "top"}},
		{name: "above the run's root, not listed", names: []string{"x"}, tree: mid, self: leaf, args: map[string]any{"agent_name": "top"}},
		{name: "unknown", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": "ghost"}, wantErr: `agent "ghost" is not a valid transfer target; available agents: listed`},
		{name: "unknown with reason", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": "ghost", "transfer_reason": "r"}, wantErr: `agent "ghost" is not a valid transfer target; available agents: listed`},
		{name: "missing argument", names: []string{"listed"}, tree: root, self: root, args: nil, wantErr: "missing argument"},
		{name: "no agent_name", names: []string{"listed"}, tree: root, self: root, args: map[string]any{}, wantErr: "empty agent_name"},
		{name: "empty agent_name", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": ""}, wantErr: "empty agent_name"},
		{name: "non-string agent_name", names: []string{"listed"}, tree: root, self: root, args: map[string]any{"agent_name": 7}, wantErr: "empty agent_name"},
		{name: "not a map", names: []string{"listed"}, tree: root, self: root, args: []string{"listed"}, wantErr: "unexpected args type: []string"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: tc.names, IncludeTransferReason: true})
			ctx := treeContext(t, tc.tree, tc.self)
			_, err := runTool(t, tl, ctx, tc.args)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Run(%v) error = %v, want success", tc.args, err)
				}
				if got, want := ctx.Actions().TransferToAgent, tc.args.(map[string]any)["agent_name"]; got != want {
					t.Errorf("TransferToAgent = %q, want %q", got, want)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Run(%v) error = %v, want one containing %q", tc.args, err, tc.wantErr)
			}
			if got := ctx.Actions(); got.TransferToAgent != "" || got.TransferReason != "" {
				t.Errorf("failed Run(%v) set TransferToAgent = %q, TransferReason = %q, want both unset", tc.args, got.TransferToAgent, got.TransferReason)
			}
		})
	}
}

func TestNew_RunWithoutAgentInContext(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"listed"}})

	listedCtx := noAgentContext(t)
	if _, err := runTool(t, tl, listedCtx, map[string]any{"agent_name": "listed"}); err != nil {
		t.Errorf("Run(agent_name: listed) error = %v, want success", err)
	}
	if got := listedCtx.Actions().TransferToAgent; got != "listed" {
		t.Errorf("TransferToAgent = %q, want %q", got, "listed")
	}

	childCtx := noAgentContext(t)
	if _, err := runTool(t, tl, childCtx, map[string]any{"agent_name": "child"}); err == nil {
		t.Errorf("Run(agent_name: child) succeeded, want an error: a context without an agent has no tree")
	}
	if got := childCtx.Actions().TransferToAgent; got != "" {
		t.Errorf("TransferToAgent = %q, want it unset", got)
	}
}
