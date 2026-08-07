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

package workflow_test

import (
	"slices"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/workflow"
)

// coordinatorToolNames returns the tools a chat coordinator installed for
// its sub-agents. Which tool a sub-agent gets (single_turn vs task) is
// derived from that sub-agent's mode, so this is the user-visible
// consequence of mode resolution.
func coordinatorToolNames(t *testing.T, a agent.Agent) []string {
	t.Helper()
	llmA, ok := a.(llminternal.Agent)
	if !ok {
		t.Fatalf("agent %q is not an LlmAgent", a.Name())
	}
	var names []string
	for _, tl := range llminternal.Reveal(llmA).Tools {
		names = append(names, tl.Name())
	}
	slices.Sort(names)
	return names
}

func declaredMode(t *testing.T, a agent.Agent) llminternal.Mode {
	t.Helper()
	llmA, ok := a.(llminternal.Agent)
	if !ok {
		t.Fatalf("agent %q is not an LlmAgent", a.Name())
	}
	return llminternal.Reveal(llmA).Mode
}

// Adapting an agent into a graph node must not rewrite the agent's own
// declaration. The node's placement is a property of the graph, not of
// the agent, and the same agent instance may be placed elsewhere.
func TestNewAgentNode_DoesNotMutateAgentMode(t *testing.T) {
	t.Parallel()

	a, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	if got := declaredMode(t, a); got != llminternal.ModeUnset {
		t.Fatalf("precondition: declared mode = %q, want unset", got)
	}

	if _, err := workflow.NewAgentNode(a, workflow.NodeConfig{}); err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	if got := declaredMode(t, a); got != llminternal.ModeUnset {
		t.Errorf("declared mode after NewAgentNode = %q, want unset (wrapping must not mutate the agent)", got)
	}
}

// An agent with no declared mode resolves by placement. Resolving one
// placement must not leak into another: a coordinator's view of its
// sub-agent must not depend on whether that sub-agent was wrapped as a
// graph node first.
func TestLlmAgent_SubAgentModeResolution_IsConstructionOrderIndependent(t *testing.T) {
	t.Parallel()

	newWorker := func(t *testing.T) agent.Agent {
		t.Helper()
		sub, err := llmagent.New(llmagent.Config{Name: "worker"})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}
		return sub
	}

	// Order A: the coordinator is built first, then the same agent
	// instance is also placed in a graph.
	subA := newWorker(t)
	coordA, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{subA},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}
	if _, err := workflow.NewAgentNode(subA, workflow.NodeConfig{}); err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	// Order B: the graph placement happens first.
	subB := newWorker(t)
	if _, err := workflow.NewAgentNode(subB, workflow.NodeConfig{}); err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	coordB, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{subB},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	gotA, gotB := coordinatorToolNames(t, coordA), coordinatorToolNames(t, coordB)
	if !slices.Equal(gotA, gotB) {
		t.Errorf("coordinator tools depend on construction order:\n  coordinator-first = %v\n  node-first        = %v", gotA, gotB)
	}
}

// The same agent instance is documented to serve many concurrent
// invocations, so placement resolution must not write to state shared
// across them. Run with -race.
func TestNewAgentNode_ConcurrentWrappingIsRaceFree(t *testing.T) {
	t.Parallel()

	a, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := workflow.NewAgentNode(a, workflow.NodeConfig{}); err != nil {
				t.Errorf("NewAgentNode: %v", err)
			}
		}()
	}
	wg.Wait()
}
