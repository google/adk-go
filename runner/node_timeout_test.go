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
	"context"
	"errors"
	"fmt"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// runNodeRootToError runs n as the runner's root node under a 10-second
// deadline, so that a node timeout that never fires fails the test instead of
// hanging it, and returns the first error the run yields.
func runNodeRootToError(t *testing.T, n workflow.Node) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	r := newNodeRootRunner(t, n, svc)

	var got error
	for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
		if err != nil && got == nil {
			got = err
		}
	}
	return got
}

// workflowAgentNode is an AgentNode with NodeConfig.Timeout timeout over a
// workflow agent named "inner_agent" whose graph is Start -> inner.
func workflowAgentNode(t *testing.T, inner workflow.Node, timeout time.Duration) *workflow.AgentNode {
	t.Helper()
	wa, err := workflowagent.New(workflowagent.Config{
		Name:  "inner_agent",
		Edges: []workflow.Edge{{From: workflow.Start, To: inner}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	an, err := workflow.NewAgentNode(wa, workflow.NodeConfig{Timeout: timeout})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	return an
}

func TestRunner_RootNodeTimeoutFailsWithNodeTimeoutError(t *testing.T) {
	slow := workflow.NewFunctionNode("slow_root", func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}, workflow.NodeConfig{Timeout: 50 * time.Millisecond})

	got := runNodeRootToError(t, slow)

	var typed *workflow.NodeTimeoutError
	if !errors.As(got, &typed) || typed.NodeName != "slow_root" || typed.Timeout != 50*time.Millisecond {
		t.Errorf("Run error = %v (%T), want a *workflow.NodeTimeoutError for \"slow_root\" (50ms)", got, got)
	}
}

func TestRunner_AgentNodeOverWorkflowAgentLeavesBodyDeadlineErrorUntyped(t *testing.T) {
	bodyErr := fmt.Errorf("fetch: %w", context.DeadlineExceeded)

	t.Run("workflow agent", func(t *testing.T) {
		fetch := workflow.NewFunctionNode("fetch", func(agent.Context, any) (string, error) {
			return "", bodyErr
		}, workflow.NodeConfig{})

		got := runNodeRootToError(t, workflowAgentNode(t, fetch, 5*time.Second))

		if typed, ok := errors.AsType[*workflow.NodeTimeoutError](got); ok {
			t.Errorf("Run error = %v, want the body's error, not a timeout of %q", got, typed.NodeName)
		}
		if !errors.Is(got, bodyErr) {
			t.Errorf("Run error = %v, want it to wrap the body's error", got)
		}
	})
	t.Run("plain agent", func(t *testing.T) {
		plain, err := agent.New(agent.Config{
			Name: "plain_agent",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { yield(nil, bodyErr) }
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		an, err := workflow.NewAgentNode(plain, workflow.NodeConfig{Timeout: 5 * time.Second})
		if err != nil {
			t.Fatalf("NewAgentNode() error = %v", err)
		}

		got := runNodeRootToError(t, an)

		var typed *workflow.NodeTimeoutError
		if !errors.As(got, &typed) || typed.NodeName != "plain_agent" || typed.Timeout != 5*time.Second {
			t.Fatalf("Run error = %v (%T), want a *workflow.NodeTimeoutError for \"plain_agent\" (5s)", got, got)
		}
		if cause := errors.Unwrap(typed); cause != bodyErr {
			t.Errorf("cause = %v, want the body's error %q", cause, bodyErr)
		}
	})
}

func TestRunner_AgentNodeOverWorkflowAgentNamesTheInnerNodeThatTimedOut(t *testing.T) {
	slow := workflow.NewFunctionNode("slow", func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}, workflow.NodeConfig{Timeout: 50 * time.Millisecond})

	got := runNodeRootToError(t, workflowAgentNode(t, slow, 5*time.Second))

	if got == nil || got.Error() != `node "slow" timed out after 0.05 seconds` {
		t.Errorf("Run error = %v, want %q", got, `node "slow" timed out after 0.05 seconds`)
	}
}
