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

package sequentialagent_test

import (
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/session"
)

const completionInstructionMarker = "call the task_completed function to exit"

type liveRunnerForTest interface {
	RunLive(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

// recordingInjectedAgent records the injection of each live run. It does not
// implement a plain RunLive, so the sequential agent has to reach it through
// LiveRunnerWithInjection to start it at all.
type recordingInjectedAgent struct {
	agent.Agent
	mu         sync.Mutex
	injections []llminternal.LiveInjection
}

func newRecordingInjectedAgent(name string) *recordingInjectedAgent {
	return &recordingInjectedAgent{Agent: mustAgent(agent.New(agent.Config{Name: name}))}
}

func (r *recordingInjectedAgent) RunLiveWithInjection(_ agent.InvocationContext, injection llminternal.LiveInjection) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	r.mu.Lock()
	r.injections = append(r.injections, injection)
	r.mu.Unlock()
	return &dummyLiveSession{sendChan: make(chan agent.LiveRequest, 1)}, func(func(*session.Event, error) bool) {}, nil
}

func (r *recordingInjectedAgent) recorded() []llminternal.LiveInjection {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]llminternal.LiveInjection(nil), r.injections...)
}

func newLiveContext(t *testing.T, a agent.Agent) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
}

func drainLive(t *testing.T, it iter.Seq2[*session.Event, error]) error {
	t.Helper()
	var gotErr error
	for _, err := range it {
		if err != nil && gotErr == nil {
			gotErr = err
		}
	}
	return gotErr
}

func requireCompletionInjection(t *testing.T, child *recordingInjectedAgent, want int) {
	t.Helper()
	got := child.recorded()
	if len(got) != want {
		t.Fatalf("sub-agent %s got %d injections, want %d", child.Name(), len(got), want)
	}
	for _, injection := range got {
		if injection.Tool == nil || injection.Tool.Name() != "task_completed" {
			t.Errorf("sub-agent %s injection tool = %v, want task_completed", child.Name(), injection.Tool)
		}
		if !strings.Contains(injection.InstructionSuffix, completionInstructionMarker) {
			t.Errorf("sub-agent %s injection suffix = %q, want it to contain %q", child.Name(), injection.InstructionSuffix, completionInstructionMarker)
		}
	}
}

func TestSequentialAgent_RunLive_InjectsEverySubAgent(t *testing.T) {
	first := newRecordingInjectedAgent("child_1")
	second := newRecordingInjectedAgent("child_2")
	seq := newSequentialAgent(t, []agent.Agent{first, second}, "sequential")

	_, it, err := seq.(liveRunnerForTest).RunLive(newLiveContext(t, seq))
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	if err := drainLive(t, it); err != nil {
		t.Fatalf("RunLive() iteration error = %v", err)
	}

	requireCompletionInjection(t, first, 1)
	requireCompletionInjection(t, second, 1)
}

func TestSequentialAgent_RunLive_NestedSequentialInjectsLeafAgents(t *testing.T) {
	leaf := newRecordingInjectedAgent("leaf")
	inner := newSequentialAgent(t, []agent.Agent{leaf}, "inner")
	branch := newRecordingInjectedAgent("branch")
	outer := newSequentialAgent(t, []agent.Agent{branch, inner}, "outer")

	_, it, err := outer.(liveRunnerForTest).RunLive(newLiveContext(t, outer))
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	if err := drainLive(t, it); err != nil {
		t.Fatalf("RunLive() iteration error = %v", err)
	}

	requireCompletionInjection(t, branch, 1)
	requireCompletionInjection(t, leaf, 1)
}

func TestSequentialAgent_RunLive_NonLiveCompositeReportsError(t *testing.T) {
	leaf := newRecordingInjectedAgent("leaf")
	nonLive := mustAgent(agent.New(agent.Config{Name: "non_live", SubAgents: []agent.Agent{leaf}}))
	branch := newRecordingInjectedAgent("branch")
	outer := newSequentialAgent(t, []agent.Agent{branch, nonLive}, "outer")

	_, it, err := outer.(liveRunnerForTest).RunLive(newLiveContext(t, outer))
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	gotErr := drainLive(t, it)
	if gotErr == nil || !strings.Contains(gotErr.Error(), "does not support Live Run") {
		t.Fatalf("RunLive() iteration error = %v, want a Live Run support error", gotErr)
	}

	requireCompletionInjection(t, branch, 1)
	requireCompletionInjection(t, leaf, 0)
}

func TestSequentialAgent_RunLive_DoesNotMutateSubAgentState(t *testing.T) {
	child, err := llmagent.New(llmagent.Config{
		Name:        "child",
		Model:       &FakeLLM{},
		Instruction: "original instruction",
	})
	if err != nil {
		t.Fatal(err)
	}
	seq := newSequentialAgent(t, []agent.Agent{child}, "sequential")

	_, it, err := seq.(liveRunnerForTest).RunLive(newLiveContext(t, seq))
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	// The fake model has no live client, so the run fails; the point is that
	// starting and iterating it leaves the shared state untouched.
	_ = drainLive(t, it)

	state := llminternal.Reveal(child.(llminternal.Agent))
	if len(state.Tools) != 0 {
		t.Errorf("sub-agent tools after RunLive = %v, want none", state.Tools)
	}
	if state.Instruction != "original instruction" {
		t.Errorf("sub-agent instruction after RunLive = %q, want %q", state.Instruction, "original instruction")
	}
}

func TestSequentialAgent_RunLive_ConcurrentCallersEachInject(t *testing.T) {
	const callers = 20
	first := newRecordingInjectedAgent("child_1")
	second := newRecordingInjectedAgent("child_2")
	seq := newSequentialAgent(t, []agent.Agent{first, second}, "sequential")
	ctx := newLiveContext(t, seq)

	start := make(chan struct{})
	var wg sync.WaitGroup
	for range callers {
		wg.Go(func() {
			<-start
			_, it, err := seq.(liveRunnerForTest).RunLive(ctx)
			if err != nil {
				t.Errorf("RunLive() error = %v", err)
				return
			}
			if err := drainLive(t, it); err != nil {
				t.Errorf("RunLive() iteration error = %v", err)
			}
		})
	}
	close(start)
	wg.Wait()

	requireCompletionInjection(t, first, callers)
	requireCompletionInjection(t, second, callers)
}
