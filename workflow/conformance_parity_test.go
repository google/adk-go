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
	"slices"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool/functiontool"
)

// TestWorkflow_PathingParity verifies hierarchical event pathing and OutputFor
// metadata generation across synthetic root wrappers and standard workflows.
func TestWorkflow_PathingParity(t *testing.T) {
	tests := []struct {
		name          string
		wfName        string
		opts          []Option
		wantPath      string
		wantOutputFor []string
	}{
		{
			name:          "root_wrapper_suppresses_prefix",
			wfName:        "synthetic_root",
			opts:          []Option{WithRootWrapper()},
			wantPath:      "leaf",
			wantOutputFor: []string{"leaf"},
		},
		{
			name:          "named_workflow_includes_ancestors",
			wfName:        "parent_wf",
			opts:          nil,
			wantPath:      "parent_wf@1/leaf@1",
			wantOutputFor: []string{"parent_wf@1/leaf@1", "parent_wf@1"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Given a workflow configured with specific options and a terminal leaf node
			leafFn := func(ctx agent.Context, input string) (string, error) {
				return "result", nil
			}
			node := NewFunctionNode("leaf", leafFn, defaultNodeConfig)
			wf, err := New(tc.wfName, []Edge{{From: Start, To: node}}, tc.opts...)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			// When the workflow is executed
			mockCtx := newSeededMockCtx(t)
			var gotPath string
			var gotOutputFor []string
			for ev, err := range wf.Run(mockCtx) {
				if err != nil {
					t.Fatalf("Run() unexpected error = %v", err)
				}
				if ev.NodeInfo != nil && ev.Output != nil {
					gotPath = ev.NodeInfo.Path
					gotOutputFor = ev.NodeInfo.OutputFor
				}
			}

			// Then emitted NodeInfo.Path and OutputFor match expected hierarchical lineage
			if gotPath != tc.wantPath {
				t.Errorf("NodeInfo.Path = %q, want %q", gotPath, tc.wantPath)
			}
			if !slices.Equal(gotOutputFor, tc.wantOutputFor) {
				t.Errorf("NodeInfo.OutputFor = %v, want %v", gotOutputFor, tc.wantOutputFor)
			}
		})
	}
}

// TestWorkflowNode_TerminalOutputCapture verifies that nested WorkflowNodes
// accurately resolve terminal graph nodes from compound hierarchical paths.
func TestWorkflowNode_TerminalOutputCapture(t *testing.T) {
	tests := []struct {
		name       string
		subWfName  string
		subNode    string
		wantOutput any
	}{
		{
			name:       "compound_path_strips_workflow_prefix",
			subWfName:  "sub_wf",
			subNode:    "child_leaf",
			wantOutput: "child_output",
		},
		{
			name:       "nested_hierarchical_path_resolves_terminal",
			subWfName:  "deep/nested_wf",
			subNode:    "leaf_step",
			wantOutput: "nested_output",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Given an outer workflow embedding a nested sub-workflow
			leafFn := func(ctx agent.Context, input string) (string, error) {
				return tc.wantOutput.(string), nil
			}
			childNode := NewFunctionNode(tc.subNode, leafFn, defaultNodeConfig)
			wfNode, err := NewWorkflowNode(tc.subWfName, []Edge{{From: Start, To: childNode}})
			if err != nil {
				t.Fatalf("NewWorkflowNode() error = %v", err)
			}

			outerWf, err := New("outer", []Edge{{From: Start, To: wfNode}})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			// When the outer workflow runs
			mockCtx := newSeededMockCtx(t)
			var gotOutput any
			for ev, err := range outerWf.Run(mockCtx) {
				if err != nil {
					t.Fatalf("Run() unexpected error = %v", err)
				}
				if ev.Output != nil {
					gotOutput = ev.Output
				}
			}

			// Then terminal outputs are successfully forwarded up to the parent
			if gotOutput != tc.wantOutput {
				t.Errorf("captured output = %v, want %v", gotOutput, tc.wantOutput)
			}
		})
	}
}

// TestToolNode_ExplicitNaming verifies explicit node name assignment and fallback
// behavior when instantiating ToolNodes.
func TestToolNode_ExplicitNaming(t *testing.T) {
	dummyTool, err := functiontool.New(functiontool.Config{
		Name:        "default_tool_name",
		Description: "dummy description",
	}, func(ctx agent.Context, in struct{}) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	tests := []struct {
		name         string
		explicitName string
		wantName     string
	}{
		{
			name:         "explicit_name_overrides_default",
			explicitName: "custom_node_name",
			wantName:     "custom_node_name",
		},
		{
			name:         "empty_name_falls_back_to_tool_name",
			explicitName: "",
			wantName:     "default_tool_name",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Given a tool wrapped with NewNamedToolNode
			node, err := NewNamedToolNode(tc.explicitName, dummyTool, defaultNodeConfig)
			if err != nil {
				t.Fatalf("NewNamedToolNode() error = %v", err)
			}

			// Then the resulting node name matches expectations
			if node.Name() != tc.wantName {
				t.Errorf("node.Name() = %q, want %q", node.Name(), tc.wantName)
			}
		})
	}
}
