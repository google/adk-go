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
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// hitlAskerNode emits a single RequestInput event and exits. Used
// as the pause point in integration scenarios.
type hitlAskerNode struct {
	workflow.BaseNode
	interruptID string
	rerun       bool
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
		rerun:       rerunOnResume,
	}
}

func (n *hitlAskerNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// In re-entry mode the same node may be re-activated with
		// the response; emit the response as an output event so
		// the successor receives it.
		if response, ok := ctx.ResumedInput(n.interruptID); ok {
			ev := session.NewEvent(ctx.InvocationID())
			ev.Actions.StateDelta["output"] = response
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

// findFunctionCall walks events for the first FunctionCall part
// whose name and ID together identify a HITL interrupt. Returns
// "" / "" when none is found.
func findFunctionCall(events []*session.Event, wantName string) (id, name string) {
	for _, ev := range events {
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			fc := p.FunctionCall
			if fc == nil {
				continue
			}
			if wantName == "" || fc.Name == wantName {
				return fc.ID, fc.Name
			}
		}
	}
	return "", ""
}

// findLongRunningInterrupt walks events for the first interrupt
// signalled the way the runner's generic dispatch detects it: an
// event with a non-empty LongRunningToolIDs whose IDs match a
// FunctionCall in the same event. Returns the matched call's name
// and ID, or ("", "") when none is found. This mirrors how
// adk-python's CLI detects HITL pauses.
func findLongRunningInterrupt(events []*session.Event) (id, name string) {
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 || ev.Content == nil {
			continue
		}
		lrIDs := map[string]struct{}{}
		for _, lr := range ev.LongRunningToolIDs {
			lrIDs[lr] = struct{}{}
		}
		for _, p := range ev.Content.Parts {
			fc := p.FunctionCall
			if fc == nil {
				continue
			}
			if _, ok := lrIDs[fc.ID]; ok {
				return fc.ID, fc.Name
			}
		}
	}
	return "", ""
}

// drainRunner consumes a runner.Run iterator into a slice. Fails
// the test on any non-nil error.
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

// resumeContent builds the user-side Content that resumes a
// previously-paused workflow: a single FunctionResponse part whose
// ID/name match the interrupt's FunctionCall, and whose response
// payload is wrapped under "payload" (the wire shape decoded by
// workflowagent.decodeWorkflowInputResponse).
func resumeContent(callID, callName string, payload any) *genai.Content {
	return &genai.Content{
		Role: string(genai.RoleUser),
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

// newRunnerForWorkflow assembles a runner with a workflow agent
// backed by an in-memory session service. Returns the runner plus
// the user/session IDs to pass into Run.
func newRunnerForWorkflow(t *testing.T, edges []workflow.Edge) (r *runner.Runner, userID, sessionID string) {
	t.Helper()

	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  "test_workflow",
		Edges: edges,
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}

	sessionService := session.InMemoryService()
	r, err = runner.New(runner.Config{
		AppName:           "test_app",
		Agent:             wfAgent,
		SessionService:    sessionService,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r, "test_user", "test_session"
}

// TestRunner_WorkflowHITL_Roundtrip_Handoff exercises the full
// happy-path round-trip through a real runner: turn 1 pauses with
// a RequestInput event, turn 2 carries a FunctionResponse keyed by
// the interrupt's call ID and the workflow finishes by routing the
// response to the asker's successor as its input.
func TestRunner_WorkflowHITL_Roundtrip_Handoff(t *testing.T) {
	ctx := context.Background()

	asker := newHitlAsker("asker", "", false /*rerun*/)

	var handlerInput atomic.Value
	handler := workflow.NewFunctionNode(
		"handler",
		func(ctx agent.InvocationContext, input string) (string, error) {
			handlerInput.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)

	r, userID, sessionID := newRunnerForWorkflow(t, workflow.Chain(workflow.Start, asker, handler))

	// Turn 1: fresh user message; expect the runner to forward
	// the asker's RequestInput event with a FunctionCall part
	// keyed in LongRunningToolIDs, then naturally end the iter.
	turn1 := drainRunner(t, r.Run(
		ctx, userID, sessionID,
		genai.NewContentFromText("draft", genai.RoleUser),
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
	// captured ID. The runner's generic findAgentToRun routes
	// this back to the same workflow agent, which detects the
	// resume in workflowAgent.detectResume and dispatches to
	// Workflow.Resume.
	turn2 := drainRunner(t, r.Run(
		ctx, userID, sessionID,
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
	ctx := context.Background()

	asker := newHitlAsker("asker", "approval", true /*rerun*/)

	var handlerInput atomic.Value
	handler := workflow.NewFunctionNode(
		"handler",
		func(ctx agent.InvocationContext, input string) (string, error) {
			handlerInput.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)

	r, userID, sessionID := newRunnerForWorkflow(t, workflow.Chain(workflow.Start, asker, handler))

	turn1 := drainRunner(t, r.Run(
		ctx, userID, sessionID,
		genai.NewContentFromText("draft", genai.RoleUser),
		agent.RunConfig{},
	))
	callID, callName := findLongRunningInterrupt(turn1)
	if callID != "approval" {
		t.Fatalf("turn 1 interrupt ID = %q, want %q (asker supplied a stable InterruptID)", callID, "approval")
	}

	drainRunner(t, r.Run(
		ctx, userID, sessionID,
		resumeContent(callID, callName, "yes"),
		agent.RunConfig{},
	))
	if got, want := handlerInput.Load(), "yes"; got != want {
		t.Errorf("handler input = %v, want %q (re-entry asker should emit the response as its output)", got, want)
	}
}

// TestRunner_WorkflowHITL_FunctionResponseRoutedByID verifies the
// runner-level routing contract: the second turn's FunctionResponse
// is matched against the prior FunctionCall by ID alone (not by
// content or session-state magic), and the matched event's Author
// is what determines which agent gets re-invoked. Pinning this
// behaviour against accidental regression in runner.findAgentToRun.
func TestRunner_WorkflowHITL_FunctionResponseRoutedByID(t *testing.T) {
	ctx := context.Background()

	asker := newHitlAsker("asker", "", false)

	r, userID, sessionID := newRunnerForWorkflow(t, workflow.Chain(workflow.Start, asker))

	turn1 := drainRunner(t, r.Run(
		ctx, userID, sessionID,
		genai.NewContentFromText("x", genai.RoleUser),
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
	if authoredBy != "test_workflow" {
		t.Errorf("interrupt event Author = %q, want %q (used by findAgentToRun)", authoredBy, "test_workflow")
	}

	// Turn 2: resume; the runner must locate test_workflow by
	// matching FunctionResponse.ID against the prior call's ID
	// in session events.
	turn2 := drainRunner(t, r.Run(
		ctx, userID, sessionID,
		resumeContent(callID, callName, "ok"),
		agent.RunConfig{},
	))
	// Successful resume is signalled by the absence of another
	// interrupt and by the iter completing without error (asserted
	// inside drainRunner).
	if id, _ := findLongRunningInterrupt(turn2); id != "" {
		t.Errorf("turn 2 produced a fresh interrupt instead of resuming; id=%q", id)
	}
}

// debugEvents formats a slice of session.Event into a human-readable
// summary used in test failure messages.
func debugEvents(events []*session.Event) string {
	out := ""
	for i, ev := range events {
		if ev == nil {
			out += "  <nil>\n"
			continue
		}
		out += eventLine(i, ev)
	}
	return out
}

func eventLine(i int, ev *session.Event) string {
	line := ""
	parts := []string{}
	if ev.Author != "" {
		parts = append(parts, "author="+ev.Author)
	}
	if len(ev.LongRunningToolIDs) > 0 {
		parts = append(parts, "lrt="+joinIDs(ev.LongRunningToolIDs))
	}
	if ev.RequestedInput != nil {
		parts = append(parts, "requested_input.id="+ev.RequestedInput.InterruptID)
	}
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				parts = append(parts, "fc="+p.FunctionCall.Name+":"+p.FunctionCall.ID)
			}
			if p.FunctionResponse != nil {
				parts = append(parts, "fr="+p.FunctionResponse.Name+":"+p.FunctionResponse.ID)
			}
			if p.Text != "" {
				parts = append(parts, "text="+p.Text)
			}
		}
	}
	for j, s := range parts {
		if j == 0 {
			line += "  ["
		} else {
			line += " "
		}
		line += s
	}
	if line != "" {
		line += "]"
	}
	return formatIndex(i) + line + "\n"
}

func joinIDs(ids []string) string {
	out := ""
	for i, s := range ids {
		if i > 0 {
			out += ","
		}
		out += s
	}
	return out
}

func formatIndex(i int) string {
	switch {
	case i < 10:
		return "  [0" + itoa(i) + "]"
	default:
		return "  [" + itoa(i) + "]"
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	out := ""
	for i > 0 {
		out = string(rune('0'+i%10)) + out
		i /= 10
	}
	return out
}
