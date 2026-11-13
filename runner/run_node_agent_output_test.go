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

package runner_test

import (
	"fmt"
	"iter"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// These tests cover what workflow.RunNode returns for an AgentNode child that
// wraps a workflow. The wrapped agent runs without a path, so its outputs are
// stamped in its own namespace. The child's own output must still come back,
// and output from anything below the child must not.

func leafNode(name, value string) workflow.Node {
	return workflow.NewFunctionNode(name, func(_ agent.Context, _ any) (string, error) {
		return value, nil
	}, workflow.NodeConfig{})
}

func newWorkflowAgent(t *testing.T, name string, edges []workflow.Edge) agent.Agent {
	t.Helper()
	a, err := workflowagent.New(workflowagent.Config{Name: name, Edges: edges})
	if err != nil {
		t.Fatalf("workflowagent.New(%q) error = %v", name, err)
	}
	return a
}

func newAgentNode(t *testing.T, a agent.Agent) workflow.Node {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode(%q) error = %v", a.Name(), err)
	}
	return n
}

// newCustomWorkflowAgent returns an agent.New agent whose Run drives a
// workflow named wfName ("" for an anonymous one) that ends in "CUSTOM".
func newCustomWorkflowAgent(t *testing.T, name, wfName string) agent.Agent {
	t.Helper()
	w, err := workflow.New(wfName, workflow.Chain(workflow.Start, leafNode("leaf", "CUSTOM")))
	if err != nil {
		t.Fatalf("workflow.New(%q) error = %v", wfName, err)
	}
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return w.Run(ctx)
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// runDispatched runs child with workflow.RunNode from a dynamic node in a
// workflow agent named "top", and returns what RunNode returned.
func runDispatched(t *testing.T, child workflow.Node, opts ...workflow.RunNodeOption) any {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	var got any
	dispatch := workflow.NewDynamicNode("dispatch",
		func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
			out, err := workflow.RunNode[any](ctx, child, in, opts...)
			got = out
			return out, err
		}, workflow.NodeConfig{})
	top := newWorkflowAgent(t, "top", workflow.Chain(workflow.Start, dispatch))

	r := newNodeTestRunner(t, top, svc)
	drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))
	return got
}

func TestRunNode_AgentNodeChildOutput(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		child func(t *testing.T) workflow.Node
		opts  []workflow.RunNodeOption
		want  any
	}{
		{
			// The leaf is stamped [sub@1/leaf@1 sub@1], not under dispatch.
			name: "workflow agent",
			child: func(t *testing.T) workflow.Node {
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, leafNode("leaf", "LEAF"))))
			},
			want: "LEAF",
		},
		{
			name: "workflow agent with WithUseAsOutput",
			child: func(t *testing.T) workflow.Node {
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, leafNode("leaf", "LEAF"))))
			},
			opts: []workflow.RunNodeOption{workflow.WithUseAsOutput()},
			want: "LEAF",
		},
		{
			// sub's terminal node wraps another workflow agent, so sub's
			// output is sub2's.
			name: "terminal node wraps another workflow agent",
			child: func(t *testing.T) workflow.Node {
				sub2 := newWorkflowAgent(t, "sub2", workflow.Chain(workflow.Start, leafNode("leaf", "INNER")))
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, newAgentNode(t, sub2))))
			},
			want: "INNER",
		},
		{
			// sub's terminal node hands its output to a wrapped workflow
			// agent with WithUseAsOutput.
			name: "terminal dynamic node delegates to a wrapped workflow agent",
			child: func(t *testing.T) workflow.Node {
				sub2 := newAgentNode(t, newWorkflowAgent(t, "sub2", workflow.Chain(workflow.Start, leafNode("leaf", "DELEGATED"))))
				relay := workflow.NewDynamicNode("relay",
					func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
						return workflow.RunNode[any](ctx, sub2, in, workflow.WithUseAsOutput())
					}, workflow.NodeConfig{})
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, relay)))
			},
			want: "DELEGATED",
		},
		{
			name: "custom agent running a differently named workflow",
			child: func(t *testing.T) workflow.Node {
				return newAgentNode(t, newCustomWorkflowAgent(t, "custom", "other"))
			},
			want: "CUSTOM",
		},
		{
			name: "custom agent running an anonymous workflow",
			child: func(t *testing.T) workflow.Node {
				return newAgentNode(t, newCustomWorkflowAgent(t, "custom", ""))
			},
			want: "CUSTOM",
		},
		{
			// sub runs another workflow agent, also named "sub", and drops
			// its result. sub itself ends without an output.
			name: "same-named workflow agent below the child",
			child: func(t *testing.T) workflow.Node {
				inner := newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, leafNode("leaf", "INNER"))))
				discard := workflow.NewDynamicNode("discard",
					func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
						_, err := workflow.RunNode[any](ctx, inner, in)
						return nil, err
					}, workflow.NodeConfig{})
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, discard)))
			},
			want: nil,
		},
		{
			// A node inside sub drives another workflow with its own context
			// and forwards the events. That workflow is below the child: it
			// must not take the paths left for sub. sub's terminal node ends
			// without an output.
			name: "workflow run directly by a node inside the child",
			child: func(t *testing.T) workflow.Node {
				other, err := workflow.New("other", workflow.Chain(workflow.Start, leafNode("leaf", "OTHER")))
				if err != nil {
					t.Fatal(err)
				}
				side := workflow.NewDynamicNode("side",
					func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
						for ev, err := range other.Run(ctx) {
							if err != nil {
								return nil, err
							}
							if err := emit(ev); err != nil {
								return nil, err
							}
						}
						return "SIDE", nil
					}, workflow.NodeConfig{})
				end := workflow.NewFunctionNode("end", func(_ agent.Context, _ any) (any, error) {
					return nil, nil
				}, workflow.NodeConfig{})
				return newAgentNode(t, newWorkflowAgent(t, "sub", workflow.Chain(workflow.Start, side, end)))
			},
			want: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := runDispatched(t, tc.child(t), tc.opts...); got != tc.want {
				t.Errorf("RunNode() = %v, want %v", got, tc.want)
			}
		})
	}
}

// On a resume, workflowagent calls Workflow.Resume with the empty path
// AgentNode gives it, so the leaf is stamped [leaf@1] with no sub@1 root.
// RunNode must still return the leaf's output.
func TestRunNode_AgentNodeChildOutput_AfterResume(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)

	leaf := workflow.NewFunctionNode("leaf", func(_ agent.Context, in any) (string, error) {
		return fmt.Sprintf("LEAF(%v)", in), nil
	}, workflow.NodeConfig{})
	subNode := newAgentNode(t, newWorkflowAgent(t, "sub",
		workflow.Chain(workflow.Start, newHitlAsker("asker", "", false), leaf)))

	var got any
	runs := 0
	dispatch := workflow.NewDynamicNode("dispatch",
		func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			runs++
			out, err := workflow.RunNode[any](ctx, subNode, ctx.UserContent())
			got = out
			return out, err
		}, workflow.NodeConfig{})
	r := newNodeTestRunner(t, newWorkflowAgent(t, "top", workflow.Chain(workflow.Start, dispatch)), svc)

	turn1 := drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID == "" {
		t.Fatalf("turn 1 did not pause; events:\n%s", debugEvents(turn1))
	}
	if got != nil {
		t.Fatalf("RunNode() = %v before the input was answered, want nil", got)
	}

	drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, resumeContent(callID, callName, "yes"), agent.RunConfig{}))
	if runs != 2 {
		t.Fatalf("dispatch ran %d times, want 2 (once per turn)", runs)
	}
	if want := "LEAF(yes)"; got != want {
		t.Errorf("RunNode() after resume = %v, want %q", got, want)
	}
}
