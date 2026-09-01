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
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
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

	// Two sub-agents, because an all-undeclared coordinator installs no tools
	// at all and "no tools" compares equal to "no tools" however the resolution
	// went. The declared one gives the comparison something to be wrong about.
	newSubAgents := func(t *testing.T) (undeclared, declared agent.Agent) {
		t.Helper()
		sub, err := llmagent.New(llmagent.Config{Name: "worker"})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}
		dec, err := llmagent.New(llmagent.Config{Name: "declared_worker", Mode: llmagent.ModeSingleTurn})
		if err != nil {
			t.Fatalf("llmagent.New(declared_worker): %v", err)
		}
		return sub, dec
	}

	// Order A: the coordinator is built first, then the same agent
	// instance is also placed in a graph.
	subA, decA := newSubAgents(t)
	coordA, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{subA, decA},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}
	if _, err := workflow.NewAgentNode(subA, workflow.NodeConfig{}); err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	// Order B: the graph placement happens first.
	subB, decB := newSubAgents(t)
	if _, err := workflow.NewAgentNode(subB, workflow.NodeConfig{}); err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	coordB, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{subB, decB},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	// An undeclared sub-agent is a chat peer in both orders, reached by
	// transfer, so the coordinator installs no delegation tool for it. The
	// declared single_turn one gets a tool in both orders.
	want := []string{"declared_worker"}
	gotA, gotB := coordinatorToolNames(t, coordA), coordinatorToolNames(t, coordB)
	if !slices.Equal(gotA, gotB) {
		t.Errorf("coordinator tools depend on construction order:\n  coordinator-first = %v\n  node-first        = %v", gotA, gotB)
	}
	if !slices.Equal(gotA, want) {
		t.Errorf("coordinator tools = %v, want %v", gotA, want)
	}
}

// The order test above can only see a difference BETWEEN the two orders, so it
// cannot catch a write that both orders make. Building a coordinator must not
// write a mode onto a sub-agent at all: the instance is shared, and a second
// coordinator over the same sub-agent would race this write.
func TestLlmAgent_New_DoesNotMutateSubAgentMode(t *testing.T) {
	t.Parallel()

	sub, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	if got := declaredMode(t, sub); got != llminternal.ModeUnset {
		t.Fatalf("precondition: declared mode = %q, want unset", got)
	}

	if _, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{sub},
	}); err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	if got := declaredMode(t, sub); got != llminternal.ModeUnset {
		t.Errorf("declared mode after being adopted as a sub-agent = %q, want unset "+
			"(building a coordinator must not write to the sub-agent)", got)
	}
}

// The same shared-instance write, raced. Two coordinators built concurrently
// over one sub-agent must not write to it. Run with -race.
func TestLlmAgent_New_ConcurrentCoordinatorsOverOneSubAgent(t *testing.T) {
	t.Parallel()

	sub, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := llmagent.New(llmagent.Config{
				Name:      fmt.Sprintf("coordinator-%d", i),
				Mode:      llmagent.ModeChat,
				SubAgents: []agent.Agent{sub},
			}); err != nil {
				t.Errorf("llmagent.New(coordinator): %v", err)
			}
		}()
	}
	wg.Wait()
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

// raceFreeLLM holds no mutable state, so a race the detector reports under
// the test below is on the agent, not on the model double.
type raceFreeLLM struct{}

func (*raceFreeLLM) Name() string { return "race-free" }

func (*raceFreeLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "answer"}},
		}}, nil)
	}
}

// The wrapping test above covers only the write that construction used to
// make. Placement is resolved again on every run, so build the nodes
// single-threaded and race the runs instead: the write this replaces lived on
// the run path, where it raced the contents processor's read. Run with -race.
func TestOneAgentInstance_ConcurrentInvocationsAreRaceFree(t *testing.T) {
	t.Parallel()

	a, err := llmagent.New(llmagent.Config{Name: "worker", Model: &raceFreeLLM{}})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	var wfs []*workflow.Workflow
	for range 16 {
		node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode: %v", err)
		}
		wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
		if err != nil {
			t.Fatalf("workflow.New: %v", err)
		}
		wfs = append(wfs, wf)
	}

	var wg sync.WaitGroup
	for _, wf := range wfs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, err := range wf.Run(newModeTestCtx(t, a)) {
				if err != nil {
					t.Errorf("workflow.Run: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
