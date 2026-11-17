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
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// TestParallelAgent_FinishedGraphWorkflowSubAgentSkippedOnResume checks that a
// graph workflow agent branch of a resumable ParallelAgent that finished in
// turn 1 records its end there, so that the turn answering a sibling's
// long-running call does not run the workflow again, and the ParallelAgent
// (and a SequentialAgent above it) end in that turn.
//
// Fails if the graph workflow agent yields no end-of-agent event when it
// finishes: d then runs a second time in turn 2 and no par:END follows.
func TestParallelAgent_FinishedGraphWorkflowSubAgentSkippedOnResume(t *testing.T) {
	tests := []struct {
		name      string
		root      func(t *testing.T, par agent.Agent) agent.Agent
		wantTurn1 []string // among turn 1's events
		wantTurn2 []string
	}{
		{
			name:      "par[wf, pauser]",
			root:      func(_ *testing.T, par agent.Agent) agent.Agent { return par },
			wantTurn1: []string{"par:STATE(map[])", "pauser:FC(approve)"},
			wantTurn2: []string{"pauser:approved", "pauser:END", "par:END"},
		},
		{
			name:      "seq[par[wf, pauser]]",
			root:      sequenceOver,
			wantTurn1: []string{"pauser:FC(approve)"},
			wantTurn2: []string{"pauser:approved", "pauser:END", "par:END", "seq:END"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var dRuns atomic.Int32
			wf := textWorkflow(t, "wf", &dRuns)
			r, _ := resumeRunner(t, tc.root(t, resumeParallel(t, wf, approvalPauser(t))), true)

			turn1, errs := runWorkflowTurn(t, r, genai.NewContentFromText("go", genai.RoleUser))
			if len(errs) > 0 {
				t.Fatalf("turn 1: errors = %q, want none", errs)
			}
			if got := dRuns.Load(); got != 1 {
				t.Errorf("turn 1: d ran %d times, want 1", got)
			}
			end := slices.IndexFunc(turn1, func(ev *session.Event) bool { return describeTurnEvent(ev) == "wf:END" })
			if end < 0 {
				t.Errorf("turn 1: events = %q, want a wf:END event", describeTurnEvents(turn1))
			} else if ev := turn1[end]; ev.Branch != "par.wf" || ev.Content != nil {
				t.Errorf("turn 1: wf end event has Branch = %q, Content = %v, want Branch = %q, Content = nil", ev.Branch, ev.Content, "par.wf")
			}
			for _, want := range tc.wantTurn1 {
				if !slices.Contains(describeTurnEvents(turn1), want) {
					t.Errorf("turn 1: events = %q, want %q among them", describeTurnEvents(turn1), want)
				}
			}

			turn2, errs := runWorkflowTurn(t, r, approvalResponse())
			if len(errs) > 0 {
				t.Fatalf("turn 2: errors = %q, want none", errs)
			}
			if diff := cmp.Diff(tc.wantTurn2, describeTurnEvents(turn2)); diff != "" {
				t.Errorf("turn 2: events (-want +got):\n%s", diff)
			}
			if got := dRuns.Load(); got != 1 {
				t.Errorf("after turn 2: d ran %d times, want 1", got)
			}
		})
	}

	// A graph workflow agent branch that paused for input has not ended, so
	// the turn answering the sibling runs it again from the start, and the
	// ParallelAgent does not end. This is the known gap of a graph workflow
	// agent that resumes only on an answer to its own request.
	t.Run("par[wfin, pauser] answered at pauser only", func(t *testing.T) {
		var bRuns atomic.Int32
		wfin := inputRequestWorkflow(t, "wfin", &bRuns)
		r, _ := resumeRunner(t, resumeParallel(t, wfin, approvalPauser(t)), true)

		turn1, errs := runWorkflowTurn(t, r, genai.NewContentFromText("go", genai.RoleUser))
		if len(errs) > 0 {
			t.Fatalf("turn 1: errors = %q, want none", errs)
		}
		if slices.Contains(describeTurnEvents(turn1), "wfin:END") {
			t.Errorf("turn 1: events = %q, want no wfin:END", describeTurnEvents(turn1))
		}

		turn2, errs := runWorkflowTurn(t, r, approvalResponse())
		if len(errs) > 0 {
			t.Fatalf("turn 2: errors = %q, want none", errs)
		}
		if got := bRuns.Load(); got != 2 {
			t.Errorf("after turn 2: b ran %d times, want 2", got)
		}
		if slices.Contains(describeTurnEvents(turn2), "par:END") {
			t.Errorf("turn 2: events = %q, want no par:END", describeTurnEvents(turn2))
		}
	})
}

// textWorkflow returns a graph workflow agent named name with one node: agent
// "d", which counts its runs in runs and yields the text "D".
func textWorkflow(t *testing.T, name string, runs *atomic.Int32) agent.Agent {
	t.Helper()
	d := resumeAgent(t, "d", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		runs.Add(1)
		yield(resumeText(ctx, "D"), nil)
	})
	return singleNodeWorkflow(t, name, d)
}

// inputRequestWorkflow returns a graph workflow agent named name with one
// node: agent "b", which counts its runs in runs and asks for input with
// interrupt ID "b"; once answered it yields the text "B".
func inputRequestWorkflow(t *testing.T, name string, runs *atomic.Int32) agent.Agent {
	t.Helper()
	b := resumeAgent(t, "b", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		runs.Add(1)
		emit := func(ev *session.Event) error {
			if !yield(ev, nil) {
				return errors.New("consumer stopped")
			}
			return nil
		}
		if _, err := workflow.ResumeOrRequestInput(agent.Promote(ctx), emit, session.RequestInput{InterruptID: "b", Message: "approve"}); err != nil {
			return
		}
		yield(resumeText(ctx, "B"), nil)
	})
	return singleNodeWorkflow(t, name, b)
}

// singleNodeWorkflow returns a graph workflow agent named name whose graph
// runs node agent a after the start node.
func singleNodeWorkflow(t *testing.T, name string, a agent.Agent) agent.Agent {
	t.Helper()
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(%q) error = %v", a.Name(), err)
	}
	wf, err := workflowagent.New(workflowagent.Config{Name: name, Edges: []workflow.Edge{{From: workflow.Start, To: node}}})
	if err != nil {
		t.Fatalf("workflowagent.New(%q) error = %v", name, err)
	}
	return wf
}

// approvalPauser returns an LlmAgent named "pauser" that first calls the
// long-running tool "approve" with call ID "lr-1", which pauses a resumable
// invocation, and answers "approved" once the call has a response (see
// approvalResponse).
func approvalPauser(t *testing.T) agent.Agent {
	t.Helper()
	approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "asks for approval", IsLongRunning: true},
		func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return must(llmagent.New(llmagent.Config{
		Name: "pauser",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "approve", Args: map[string]any{}}}}},
			genai.NewContentFromText("approved", genai.RoleModel),
		}},
		Tools: []tool.Tool{approve},
	}))
}

// approvalResponse returns the user content that answers approvalPauser's
// long-running call.
func approvalResponse() *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{ID: "lr-1", Name: "approve", Response: map[string]any{"status": "approved"}},
	}}}
}

// runWorkflowTurn runs one turn of r in session "s1" with message msg and
// returns the events and the errors it yielded.
func runWorkflowTurn(t *testing.T, r *runner.Runner, msg *genai.Content) ([]*session.Event, []string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), resumeTimeout)
	defer cancel()
	var (
		events []*session.Event
		errs   []string
	)
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		events = append(events, ev)
	}
	return events, errs
}

// describeTurnEvents renders each of events with describeTurnEvent.
func describeTurnEvents(events []*session.Event) []string {
	out := make([]string, 0, len(events))
	for _, ev := range events {
		out = append(out, describeTurnEvent(ev))
	}
	return out
}

// storedTurnEvents returns the events stored in session "s1" of sessions,
// other than the user's messages, rendered by describeTurnEvent.
func storedTurnEvents(t *testing.T, sessions session.Service) []string {
	t.Helper()
	got, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var events []*session.Event
	for ev := range got.Session.Events().All() {
		if ev.Author != "user" {
			events = append(events, ev)
		}
	}
	return describeTurnEvents(events)
}

// describeTurnEvent renders ev as "author:END" for an end-of-agent event,
// "author:STATE(<state as %v>)" for an agent-state event, or "author:"
// followed by "FC(name)" for each function call, "FR(name)" for each function
// response and the text of each text part.
func describeTurnEvent(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	if ev.Actions.AgentState != nil {
		return fmt.Sprintf("%s:STATE(%v)", ev.Author, ev.Actions.AgentState)
	}
	var b strings.Builder
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Fprintf(&b, "FC(%s)", p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Fprintf(&b, "FR(%s)", p.FunctionResponse.Name)
			default:
				b.WriteString(p.Text)
			}
		}
	}
	return ev.Author + ":" + b.String()
}
