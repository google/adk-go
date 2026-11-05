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
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const completionInstructionMarker = "call the task_completed function to exit"

type liveRunnerForTest interface {
	RunLive(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func TestSequentialAgent_RunLive_ConcurrentInjectionIsVisibleToEveryCaller(t *testing.T) {
	for _, tt := range []struct {
		name    string
		parents int
	}{
		{name: "same_parent", parents: 1},
		{name: "shared_children_between_parents", parents: 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for range 5 {
				children := []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2), newCustomAgent(t, 3)}
				parents := make([]agent.Agent, tt.parents)
				for i := range parents {
					parents[i] = newSequentialAgent(t, children, fmt.Sprintf("sequential_%d", i))
				}
				start := make(chan struct{})
				var wg sync.WaitGroup
				for caller := range 20 {
					wg.Go(func() {
						<-start
						seq := parents[caller%len(parents)]
						ctx := &mockInvocationContext{agent: seq, invocationID: "concurrent", ctx: t.Context()}
						_, _, err := seq.(liveRunnerForTest).RunLive(ctx)
						if err != nil {
							t.Errorf("RunLive() error = %v", err)
							return
						}
						// Read after this caller returns, while other callers may still be
						// initializing. A count taken only after wg.Wait misses that edge.
						for i, child := range children {
							state := llminternal.Reveal(child.(llminternal.Agent))
							completed := 0
							for _, item := range state.Tools {
								if item.Name() == "task_completed" {
									completed++
								}
							}
							if completed != 1 {
								t.Errorf("child %d has %d completion tools, want 1", i, completed)
							}
							if count := strings.Count(state.Instruction, completionInstructionMarker); count != 1 {
								t.Errorf("child %d has %d completion instructions, want 1", i, count)
							}
						}
					})
				}
				close(start)
				wg.Wait()
			}
		})
	}
}

type namedCompletionTool struct{}

func (*namedCompletionTool) Name() string        { return "task_completed" }
func (*namedCompletionTool) Description() string { return "caller-provided completion tool" }
func (*namedCompletionTool) IsLongRunning() bool { return false }

func TestSequentialAgent_RunLive_PreservesExistingCompletionTool(t *testing.T) {
	existing := &namedCompletionTool{}
	child, err := llmagent.New(llmagent.Config{
		Name: "child", Model: &FakeLLM{}, Tools: []tool.Tool{existing}, Instruction: "original instruction",
	})
	if err != nil {
		t.Fatal(err)
	}
	seq := newSequentialAgent(t, []agent.Agent{child}, "sequential")
	ctx := &mockInvocationContext{agent: seq, invocationID: "existing", ctx: t.Context()}
	for range 2 {
		if _, _, err := seq.(liveRunnerForTest).RunLive(ctx); err != nil {
			t.Fatalf("RunLive() error = %v", err)
		}
	}
	state := llminternal.Reveal(child.(llminternal.Agent))
	if len(state.Tools) != 1 || state.Tools[0] != existing {
		t.Error("RunLive changed the caller's completion tool")
	}
	if state.Instruction != "original instruction" {
		t.Error("RunLive changed the instruction for a caller-provided completion tool")
	}
}

func TestSequentialAgent_RunLive_AppendsCompletionInstructionToProvider(t *testing.T) {
	errProvider := errors.New("provider failed")
	for _, tt := range []struct {
		name    string
		result  string
		err     error
		wantErr error
		want    string
	}{
		{
			name:   "appends_to_provider_result",
			result: "dynamic instruction",
			want:   "dynamic instruction",
		},
		{
			name:    "passes_provider_error_through",
			result:  "partial",
			err:     errProvider,
			wantErr: errProvider,
			want:    "partial",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			child, err := llmagent.New(llmagent.Config{
				Name:  "child",
				Model: &FakeLLM{},
				InstructionProvider: func(agent.ReadonlyContext) (string, error) {
					return tt.result, tt.err
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			seq := newSequentialAgent(t, []agent.Agent{child}, "sequential")
			ctx := &mockInvocationContext{agent: seq, invocationID: "provider", ctx: t.Context()}
			for range 2 {
				if _, _, err := seq.(liveRunnerForTest).RunLive(ctx); err != nil {
					t.Fatalf("RunLive() error = %v", err)
				}
			}

			// The provider takes over Instruction, so the model is told about
			// task_completed only if the provider's result carries it.
			state := llminternal.Reveal(child.(llminternal.Agent))
			got, err := state.InstructionProvider(icontext.NewReadonlyContext(ctx))
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("InstructionProvider() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr != nil {
				if got != tt.want {
					t.Errorf("InstructionProvider() = %q, want %q", got, tt.want)
				}
				return
			}
			if !strings.HasPrefix(got, tt.want) {
				t.Errorf("InstructionProvider() = %q, want it to start with %q", got, tt.want)
			}
			if count := strings.Count(got, completionInstructionMarker); count != 1 {
				t.Errorf("InstructionProvider() has %d completion instructions, want 1: %q", count, got)
			}
		})
	}
}
