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
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// workflowAgentName is also the Author stamped on the events the agent emits.
const workflowAgentName = "test_workflow"

// TestRunner_WorkflowHITL_Roundtrip_Handoff exercises the full
// happy-path round-trip through a real runner: turn 1 pauses with
// a RequestInput event, turn 2 carries a FunctionResponse keyed by
// the interrupt's call ID and the workflow finishes by routing the
// response to the asker's successor as its input.
func TestRunner_WorkflowHITL_Roundtrip_Handoff(t *testing.T) {
	ctx := t.Context()

	asker := newHitlAsker("asker", "", false /*rerunOnResume*/)

	var handlerInput atomic.Value
	handler := workflow.NewFunctionNode(
		"handler",
		func(ctx agent.Context, input string) (string, error) {
			handlerInput.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)

	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, handler))

	// Turn 1: fresh message; the workflow pauses at the asker.
	turn1 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		userText("draft"),
		agent.RunConfig{},
	))

	callID, callName := findLongRunningInterrupt(turn1)
	if callID == "" {
		t.Fatalf("turn 1 produced no LongRunning interrupt event; events:\n%s", debugEvents(turn1))
	}
	if callName != workflow.WorkflowInputFunctionCallName {
		t.Errorf("turn 1 interrupt FunctionCall name = %q, want %q", callName, workflow.WorkflowInputFunctionCallName)
	}
	if v := handlerInput.Load(); v != nil {
		t.Errorf("handler ran during turn 1; got input %v, want it not to run", v)
	}

	// Turn 2: resume by sending a FunctionResponse with the
	// captured ID. The runner routes this back to the same
	// workflow agent (matching the response to the prior call by
	// ID), which detects the resume and continues the workflow.
	turn2 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		resumeContent(callID, callName, "approve"),
		agent.RunConfig{},
	))
	if id, _ := findLongRunningInterrupt(turn2); id != "" {
		t.Errorf("turn 2 unexpectedly produced another interrupt: id=%q", id)
	}
	if got, want := handlerInput.Load(), "approve"; got != want {
		t.Errorf("handler input = %v, want %q", got, want)
	}
}

// TestRunner_WorkflowHITL_Roundtrip_ReEntry verifies the same
// round-trip with the asker configured for re-entry mode: instead
// of routing the response to the next node, the engine re-activates
// the asker which observes the response via ResumedInput and emits
// it as an output event for the successor.
func TestRunner_WorkflowHITL_Roundtrip_ReEntry(t *testing.T) {
	ctx := t.Context()

	asker := newHitlAsker("asker", "approval", true /*rerunOnResume*/)

	var handlerInput atomic.Value
	handler := workflow.NewFunctionNode(
		"handler",
		func(ctx agent.Context, input string) (string, error) {
			handlerInput.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)

	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, handler))

	turn1 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		userText("draft"),
		agent.RunConfig{},
	))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID != "approval" {
		t.Fatalf("turn 1 interrupt ID = %q, want %q (asker supplied a stable InterruptID)", callID, "approval")
	}

	drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		resumeContent(callID, callName, "yes"),
		agent.RunConfig{},
	))
	if got, want := handlerInput.Load(), "yes"; got != want {
		t.Errorf("handler input = %v, want %q (re-entry asker should emit the response as its output)", got, want)
	}
}

// TestRunner_WorkflowHITL_FunctionResponseRoutedByID verifies the
// runner-level routing contract: the resume FunctionResponse is matched
// to the prior FunctionCall by ID, and the matched event's Author
// selects which agent gets re-invoked (runner.findAgentToRun).
func TestRunner_WorkflowHITL_FunctionResponseRoutedByID(t *testing.T) {
	ctx := t.Context()

	asker := newHitlAsker("asker", "", false)

	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker))

	turn1 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		userText("x"),
		agent.RunConfig{},
	))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID == "" {
		t.Fatal("no interrupt produced on turn 1")
	}

	// Verify the call's Author is the workflow agent's name. The
	// runner uses Author to locate the sub-agent that issued the
	// call when routing the matching FunctionResponse.
	authoredBy := ""
	for _, ev := range turn1 {
		if ev != nil && ev.Author != "" {
			authoredBy = ev.Author
			break
		}
	}
	if authoredBy != workflowAgentName {
		t.Errorf("interrupt event Author = %q, want %q (used to route the resume back to the right agent)", authoredBy, workflowAgentName)
	}

	// Turn 2: resume; the runner must locate the workflow agent by
	// matching FunctionResponse.ID against the prior call's ID.
	turn2 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		resumeContent(callID, callName, "ok"),
		agent.RunConfig{},
	))
	if id, _ := findLongRunningInterrupt(turn2); id != "" {
		t.Errorf("turn 2 produced a fresh interrupt instead of resuming; id=%q", id)
	}
}

// TestRunner_WorkflowHITL_DynamicOrchestrator_Resume is the
// end-to-end scenario for dynamic-workflow resume + HITL: a dynamic
// orchestrator runs two children sequentially via RunNode. The first
// completes; the second requests human input and suspends the whole
// parent. On resume the parent body re-executes from the top:
//   - the first RunNode re-runs (the per-invocation cache does not
//     carry across turns; each resume is a new invocation, matching
//     adk-python which gates rehydration on invocation_id), and
//   - the second RunNode observes the user's response and the parent
//     completes with its final value.
func TestRunner_WorkflowHITL_DynamicOrchestrator_Resume(t *testing.T) {
	ctx := t.Context()

	const interruptID = "approval"

	var firstChildRuns atomic.Int32
	firstChild := workflow.NewFunctionNode(
		"first_child",
		func(ctx agent.Context, input string) (string, error) {
			firstChildRuns.Add(1)
			return "X:" + input, nil
		},
		workflow.NodeConfig{},
	)

	secondChild := newHitlAsker("second_child", interruptID, true /*rerunOnResume*/)

	var parentOutput atomic.Value
	orchestrator := workflow.NewDynamicNode[string, string](
		"orchestrate",
		func(nc agent.Context, input string, _ func(*session.Event) error) (string, error) {
			x, err := workflow.RunNode[string](nc, firstChild, input, workflow.WithRunID("c1"))
			if err != nil {
				return "", err
			}
			y, err := workflow.RunNode[any](nc, secondChild, nil, workflow.WithRunID("c2"))
			if err != nil {
				return "", err
			}
			decision, _ := y.(string)
			out := x + "|Y:" + decision
			parentOutput.Store(out)
			return out, nil
		},
		workflow.NodeConfig{},
	)

	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, orchestrator))

	turn1 := drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		userText("draft"),
		agent.RunConfig{},
	))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID != interruptID {
		t.Fatalf("turn 1 interrupt ID = %q, want %q; events:\n%s", callID, interruptID, debugEvents(turn1))
	}
	if got := firstChildRuns.Load(); got != 1 {
		t.Fatalf("first child ran %d times on turn 1, want 1", got)
	}

	drainRunner(t, r.Run(
		ctx, nodeTestUser, nodeTestSession,
		resumeContent(callID, callName, "yes"),
		agent.RunConfig{},
	))

	if got := firstChildRuns.Load(); got != 2 {
		t.Errorf("first child ran %d times total, want 2 (resume re-executes the orchestrator under a new invocation)", got)
	}
	if got, want := parentOutput.Load(), "X:draft|Y:yes"; got != want {
		t.Errorf("parent output = %v, want %q (re-run first child + resumed second child)", got, want)
	}
}

// TestRunner_WorkflowHITL_TwoFullCycles_SameSession guards the re-run
// case behind the hitl_simple bug: two complete pause/resume cycles in
// one session must not collide. Each pause gets a fresh auto-generated
// interrupt ID and rehydration is scoped to the resumed run's
// invocation (adk-python parity), so the second resume still routes its
// response to the handler instead of failing with ErrNothingToResume.
func TestRunner_WorkflowHITL_TwoFullCycles_SameSession(t *testing.T) {
	ctx := t.Context()

	asker := newHitlAsker("asker", "", false /*rerunOnResume*/)
	var handlerInput atomic.Value
	handler := workflow.NewFunctionNode(
		"handler",
		func(ctx agent.Context, input string) (string, error) {
			handlerInput.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, handler))

	runCycle := func(answer string) {
		turn1 := drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, userText("draft"), agent.RunConfig{}))
		callID, callName := findLongRunningInterrupt(turn1)
		if callID == "" {
			t.Fatalf("fresh turn produced no interrupt; events:\n%s", debugEvents(turn1))
		}
		drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, resumeContent(callID, callName, answer), agent.RunConfig{}))
		if got, want := handlerInput.Load(), answer; got != want {
			t.Fatalf("handler input = %v, want %q", got, want)
		}
	}

	runCycle("first")
	runCycle("second")
}

// newWorkflowRunner builds a runner driving a workflow agent over the
// given edges, with its session pre-created.
func newWorkflowRunner(t *testing.T, edges []workflow.Edge) *runner.Runner {
	t.Helper()

	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  workflowAgentName,
		Edges: edges,
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}

	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	return newNodeTestRunner(t, wfAgent, svc)
}

// hitlAskerNode is the pause point in the scenarios: it requests human
// input, then on re-entry emits the resumed response as its output.
type hitlAskerNode struct {
	workflow.BaseNode
	interruptID string
}

func newHitlAsker(name, interruptID string, rerunOnResume bool) *hitlAskerNode {
	cfg := workflow.NodeConfig{}
	if rerunOnResume {
		t := true
		cfg.RerunOnResume = &t
	}
	return &hitlAskerNode{
		BaseNode:    workflow.NewBaseNode(name, "", cfg),
		interruptID: interruptID,
	}
}

func (n *hitlAskerNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// On re-entry, hand the response to the successor as output.
		if response, ok := ctx.ResumedInput(n.interruptID); ok {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = response
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: n.interruptID,
			Message:     "please decide",
			Payload:     input,
		}), nil)
	}
}

// resumeContent builds the user-side Content that resumes a
// previously-paused workflow: a single FunctionResponse part whose
// ID/name match the interrupt's FunctionCall, and whose response
// payload is wrapped under the "payload" key (the wire shape the
// workflow agent expects when decoding a resume response).
func resumeContent(callID, callName string, payload any) *genai.Content {
	return &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:   callID,
				Name: callName,
				Response: map[string]any{
					"payload": payload,
				},
			},
		}},
	}
}

// findLongRunningInterrupt returns the ID and name of the first
// FunctionCall flagged in its event's LongRunningToolIDs, or empty
// strings if none.
func findLongRunningInterrupt(events []*session.Event) (id, name string) {
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			fc := p.FunctionCall
			if fc == nil {
				continue
			}
			if slices.Contains(ev.LongRunningToolIDs, fc.ID) {
				return fc.ID, fc.Name
			}
		}
	}
	return "", ""
}

// drainRunner collects all events from a run, failing on any error.
func drainRunner(t *testing.T, seq iter.Seq2[*session.Event, error]) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev, err := range seq {
		if err != nil {
			t.Fatalf("runner yielded error: %v", err)
		}
		out = append(out, ev)
	}
	return out
}

// debugEvents renders events for test failure messages.
func debugEvents(events []*session.Event) string {
	var b strings.Builder
	for i, ev := range events {
		if ev == nil {
			b.WriteString("  <nil>\n")
			continue
		}
		fmt.Fprintf(&b, "  [%02d] [%s]\n", i, strings.Join(eventFields(ev), " "))
	}
	return b.String()
}

// eventFields returns the event's non-empty fields, one token each.
func eventFields(ev *session.Event) []string {
	var fields []string
	if ev.Author != "" {
		fields = append(fields, "author="+ev.Author)
	}
	if len(ev.LongRunningToolIDs) > 0 {
		fields = append(fields, "lrt="+strings.Join(ev.LongRunningToolIDs, ","))
	}
	if ev.RequestedInput != nil {
		fields = append(fields, "requested_input.id="+ev.RequestedInput.InterruptID)
	}
	if ev.Content == nil {
		return fields
	}
	for _, p := range ev.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			fields = append(fields, fmt.Sprintf("fc=%s:%s", p.FunctionCall.Name, p.FunctionCall.ID))
		case p.FunctionResponse != nil:
			fields = append(fields, fmt.Sprintf("fr=%s:%s", p.FunctionResponse.Name, p.FunctionResponse.ID))
		case p.Text != "":
			fields = append(fields, "text="+p.Text)
		}
	}
	return fields
}
