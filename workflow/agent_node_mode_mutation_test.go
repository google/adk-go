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
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
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

func declaredIncludeContents(t *testing.T, a agent.Agent) string {
	t.Helper()
	llmA, ok := a.(llminternal.Agent)
	if !ok {
		t.Fatalf("agent %q is not an LlmAgent", a.Name())
	}
	return llminternal.Reveal(llmA).IncludeContents
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

// raceFreeLLM holds one atomic counter and nothing else, so a race the detector
// reports under the test below is on the agent, not on the model double. The
// counter is what lets the test prove it reached the model at all. Note that
// atomics order the goroutines that touch them, so a conflicting access placed
// AFTER the model call could be masked by it — the write this replaces happened
// before, and is unaffected.
type raceFreeLLM struct{ calls atomic.Int64 }

func (*raceFreeLLM) Name() string { return "race-free" }

func (m *raceFreeLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
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
//
// Scope, because "one instance, concurrent invocations" sounds broader than
// what this races. The agent has no Tools and no Toolsets, so although every
// run does enter the tool processor, it appends nothing there — and that
// append onto the agent's own Tools slice is a separate shared-state hazard
// this change does not touch, so it stays unexercised here. Every goroutine gets
// its own node, workflow and session, so the only objects shared are the agent
// and the model. And all sixteen placements are the same one — an instance
// under a runner-root chat placement and a graph-node single_turn placement at
// the same time is not raced anywhere.
func TestOneAgentInstance_ConcurrentInvocationsAreRaceFree(t *testing.T) {
	t.Parallel()

	const runs = 16

	llm := &raceFreeLLM{}
	a, err := llmagent.New(llmagent.Config{Name: "worker", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	// Both the workflows and the contexts are built here rather than inside the
	// goroutines: newModeTestCtx calls t.Fatalf, which is only valid on the test
	// goroutine, and a fixture failure in a worker would otherwise be reported
	// as something other than a failure.
	var wfs []*workflow.Workflow
	var ctxs []agent.Context
	for range runs {
		node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode: %v", err)
		}
		wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
		if err != nil {
			t.Fatalf("workflow.New: %v", err)
		}
		wfs = append(wfs, wf)
		ctxs = append(ctxs, newModeTestCtx(t, a))
	}

	var wg sync.WaitGroup
	for i, wf := range wfs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, err := range wf.Run(ctxs[i]) {
				if err != nil {
					t.Errorf("workflow.Run: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	// Without this the test would stay green if the runs short-circuited before
	// reaching the contents processor, having opened no race window at all.
	if got := llm.calls.Load(); got != runs {
		t.Errorf("model was called %d time(s), want %d — the runs did not reach the code this races", got, runs)
	}
}

// The sibling of TestLlmAgent_SubAgentModeResolution_IsConstructionOrderIndependent,
// for the dimension that test cannot see. Construction
// order moved two things on the merge base, not one: which delegation tool the
// coordinator installed, and whether the sub-agent was a transfer TARGET.
// Node-first stamped the undeclared agent single_turn, and isUntransferableMode
// reads the declaration, so it dropped off the target list entirely. That list
// is assembled in the request processor rather than in state.Tools, so no
// assertion on tool names can observe it.
func TestLlmAgent_TransferTargets_AreConstructionOrderIndependent(t *testing.T) {
	t.Parallel()

	// coordinatorInstruction builds a chat coordinator over one undeclared
	// sub-agent, optionally placing that sub-agent at a graph node first, runs
	// it, and returns the system instruction the coordinator's model saw.
	coordinatorInstruction := func(t *testing.T, nodeFirst bool) string {
		t.Helper()
		sub, err := llmagent.New(llmagent.Config{Name: "worker", Description: "does the work"})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}
		if nodeFirst {
			if _, err := workflow.NewAgentNode(sub, workflow.NodeConfig{}); err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}
		}
		llm := &capturingLLM{}
		coord, err := llmagent.New(llmagent.Config{
			Name:      "coordinator",
			Mode:      llmagent.ModeChat,
			Model:     llm,
			SubAgents: []agent.Agent{sub},
		})
		if err != nil {
			t.Fatalf("llmagent.New(coordinator): %v", err)
		}
		if !nodeFirst {
			if _, err := workflow.NewAgentNode(sub, workflow.NodeConfig{}); err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}
		}
		r, err := runner.New(runner.Config{
			AppName:           "app",
			Agent:             coord,
			SessionService:    session.InMemoryService(),
			AutoCreateSession: true,
		})
		if err != nil {
			t.Fatalf("runner.New: %v", err)
		}
		for _, err := range r.Run(t.Context(), "u", "s1", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
		}
		if llm.got == nil {
			t.Fatal("the coordinator's model was never called, so this pins nothing")
		}
		return llm.systemInstruction()
	}

	coordFirst := coordinatorInstruction(t, false)
	nodeFirst := coordinatorInstruction(t, true)

	// Guard against both orders being empty, which would satisfy the equality
	// below while proving nothing.
	if !strings.Contains(coordFirst, "transfer_to_agent") {
		t.Fatalf("no transfer instructions in either order, so the comparison is vacuous; got %q", coordFirst)
	}
	if !strings.Contains(coordFirst, "worker") {
		t.Errorf("the undeclared sub-agent is not a transfer target when the coordinator is built first; got %q", coordFirst)
	}
	if coordFirst != nodeFirst {
		t.Errorf("the coordinator's transfer targets depend on construction order:\n coordinator-first = %q\n node-first        = %q", coordFirst, nodeFirst)
	}
}

// CONTRACT 1 covers State.IncludeContents as well as State.Mode, and until now
// only the Mode half was asserted. The wrapper used to write
// state.IncludeContents = "none" onto the shared agent when it ran a
// single_turn placement, which is why a second placement of the same instance
// then hid history everywhere. A guarded reintroduction of that write —
// only when the field is still empty — changes no behaviour the history tests
// observe, because they set IncludeContents explicitly, so nothing but this
// assertion or the race detector would catch it.
func TestAgentNode_Run_DoesNotMutateTheAgentsIncludeContents(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	a, err := llmagent.New(llmagent.Config{Name: "worker", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	if got := declaredIncludeContents(t, a); got != "" {
		t.Fatalf("precondition: IncludeContents = %q, want empty", got)
	}

	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", []workflow.Edge{{From: workflow.Start, To: node}})
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}
	for _, err := range wf.Run(newModeTestCtx(t, a)) {
		if err != nil {
			t.Fatalf("wf.Run: %v", err)
		}
	}
	if llm.got == nil {
		t.Fatal("the model was never called, so the single_turn placement was not exercised")
	}

	if got := declaredIncludeContents(t, a); got != "" {
		t.Errorf("IncludeContents after a single_turn placement = %q, want empty (a run must not mutate the agent)", got)
	}
	if got := declaredMode(t, a); got != llminternal.ModeUnset {
		t.Errorf("declared mode after a single_turn placement = %q, want unset", got)
	}
}

// The whole change, in the shape a user meets it: one placement must not decide
// what the agent is for every placement after it.
//
// An undeclared agent is wrapped in a graph node and run, then the SAME instance
// is used as a runner root. On the merge base those were two separate writes:
// NewAgentNode stamped Mode=single_turn at construction, and the run then
// stamped IncludeContents=none. Both were permanent, on the object, so the
// runner rejected it with "root agent worker must be a chat LlmAgent, but has
// mode single_turn". Measured both ways: the base errors, this runs.
//
// The individual writes are pinned field by field elsewhere. This pins the
// consequence, which is the thing the PR description promises and the only
// place the two placements are exercised in order on one instance.
func TestOneInstance_ANodeRunDoesNotDecideItsNextPlacement(t *testing.T) {
	t.Parallel()

	llm := &raceFreeLLM{}
	a, err := llmagent.New(llmagent.Config{
		Name: "worker", Description: "w", Model: llm, Instruction: "OWN",
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	ctx := newModeTestCtx(t, a)
	for _, err := range node.Run(ctx, nil) {
		if err != nil {
			t.Fatalf("node.Run: %v", err)
		}
	}

	// Guard against a vacuous pass: if nothing ran, the absences below prove
	// nothing. Every sibling in this file carries the same check.
	if llm.calls.Load() == 0 {
		t.Fatal("the model was never called, so the node run did not happen")
	}

	// Nothing the construction or the run did may be visible on the agent.
	state := llminternal.Reveal(a.(llminternal.Agent))
	if state.Mode != llminternal.ModeUnset {
		t.Errorf("Mode after a node run = %q, want unset", state.Mode)
	}
	if state.IncludeContents != "" {
		t.Errorf("IncludeContents after a node run = %q, want empty", state.IncludeContents)
	}

	// And the consequence: the same instance is still usable as a chat root.
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New after a node run: %v", err)
	}
	var runErr error
	for _, err := range r.Run(t.Context(), "u", "s1", genai.NewContentFromText("hello", "user"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
		}
	}
	if runErr != nil {
		t.Errorf("running the same instance as a chat root after a node run: %v", runErr)
	}
}
