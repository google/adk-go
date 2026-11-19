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

package parallelagent_test

import (
	"slices"
	"sync/atomic"
	"testing"
	"testing/synctest"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// TestParallelAgent_EarlyStopDoesNotReleaseBranchAndResumeRunsToolOnce checks
// that once the caller stops reading at a branch's function call event, the
// ParallelAgent releases no branch, so the branch's tool does not run, and
// that, in a resumable run, resuming the invocation by its ID runs the tool
// once and ends the branch and every agent above it. A non-resumable run has
// no resume, so its case checks the stop only.
//
// The order is forced, not left to the scheduler: the caller waits until every
// goroutine of the run is blocked (synctest.Wait) before it stops, so the
// branch that yielded the call is parked waiting for the event to be handled.
// Fails if the merge loop releases that branch before it checks for the stop:
// the tool then runs before the branches are canceled, giving 1 booking after
// the stop and, in a resumable run, 2 after the resume.
func TestParallelAgent_EarlyStopDoesNotReleaseBranchAndResumeRunsToolOnce(t *testing.T) {
	tests := []struct {
		name      string
		root      func(t *testing.T, par agent.Agent) agent.Agent
		resumable bool
		wantTail  []string
	}{
		{
			name:      "par",
			root:      func(_ *testing.T, par agent.Agent) agent.Agent { return par },
			resumable: true,
			wantTail:  []string{"par:END"},
		},
		{
			name:      "seq[par]",
			root:      sequenceOver,
			resumable: true,
			wantTail:  []string{"par:END", "seq:END"},
		},
		{
			name: "par, not resumable",
			root: func(_ *testing.T, par agent.Agent) agent.Agent { return par },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Everything the run uses is built inside the bubble, so every
			// goroutine it starts blocks only on channels the bubble owns and
			// synctest.Wait can tell when all of them are parked.
			synctest.Test(t, func(t *testing.T) {
				var bookings atomic.Int32
				book, err := functiontool.New(functiontool.Config{Name: "book", Description: "books a table"},
					func(agent.Context, struct{}) (map[string]any, error) {
						bookings.Add(1)
						return map[string]any{"table": 7}, nil
					})
				if err != nil {
					t.Fatalf("functiontool.New() error = %v", err)
				}
				assistant := must(llmagent.New(llmagent.Config{
					Name: "assistant",
					Model: &testutil.MockModel{Responses: []*genai.Content{
						{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "book", Args: map[string]any{}}}}},
						genai.NewContentFromText("You have table 7.", genai.RoleModel),
					}},
					Tools: []tool.Tool{book},
				}))
				open := make(chan struct{})
				close(open)
				sibling := must(llmagent.New(llmagent.Config{Name: "sibling", Model: &gatedLLM{gate: open, text: "sibling done"}}))
				r, sessions := resumeRunner(t, tc.root(t, resumeParallel(t, assistant, sibling)), tc.resumable)

				var invocationID string
				for ev, err := range r.Run(t.Context(), "user", "s1", genai.NewContentFromText("book a table", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("turn 1: Run() error = %v", err)
					}
					if describeTurnEvent(ev) == "assistant:FC(book)" {
						invocationID = ev.InvocationID
						synctest.Wait()
						break
					}
				}
				if invocationID == "" {
					t.Fatal("turn 1: the assistant's function call was not yielded")
				}
				if got := bookings.Load(); got != 0 {
					t.Errorf("turn 1: bookings after the stop = %d, want 0", got)
				}
				stored := storedTurnEvents(t, sessions)
				if !slices.Contains(stored, "assistant:FC(book)") || slices.Contains(stored, "assistant:FR(book)") {
					t.Errorf("turn 1: session = %q, want assistant:FC(book) and no assistant:FR(book)", stored)
				}
				if !tc.resumable {
					return
				}

				var turn2 []string
				for ev, err := range r.Run(t.Context(), "user", "s1", nil, agent.RunConfig{}, runner.WithInvocationID(invocationID)) {
					if err != nil {
						t.Fatalf("turn 2: Run() error = %v", err)
					}
					turn2 = append(turn2, describeTurnEvent(ev))
				}
				if got := bookings.Load(); got != 1 {
					t.Errorf("turn 2: bookings after the resume = %d, want 1", got)
				}
				for _, want := range []string{"assistant:FR(book)", "assistant:You have table 7.", "assistant:END"} {
					if !slices.Contains(turn2, want) {
						t.Errorf("turn 2: events = %q, want %q among them", turn2, want)
					}
				}
				if len(turn2) < len(tc.wantTail) {
					t.Fatalf("turn 2: events = %q, want them to end with %q", turn2, tc.wantTail)
				}
				if diff := cmp.Diff(tc.wantTail, turn2[len(turn2)-len(tc.wantTail):]); diff != "" {
					t.Errorf("turn 2: last events (-want +got):\n%s\nall events: %q", diff, turn2)
				}
			})
		})
	}
}

// sequenceOver returns a sequential agent named "seq" whose only sub-agent is
// sub.
func sequenceOver(t *testing.T, sub agent.Agent) agent.Agent {
	t.Helper()
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{sub}}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}
	return seq
}
