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

package workflowagent_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// TestWorkflowAgent_EndOfAgentEvent_ResumedNodeFinishesWithoutOtherInterrupt
// checks that a resumable graph workflow agent writes its end event when the
// resume that answers one of a re-entry node's two requests lets the node
// finish without asking the other again.
//
// Fails if a node that is no longer waiting, but still lists an interrupt it
// raised earlier, withholds the end event.
func TestWorkflowAgent_EndOfAgentEvent_ResumedNodeFinishesWithoutOtherInterrupt(t *testing.T) {
	rerun := true
	two := newFuncNode("two", workflow.NodeConfig{RerunOnResume: &rerun}, func(ctx agent.Context, yield func(*session.Event, error) bool) {
		if _, ok := ctx.ResumedInput("ask-a"); ok {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "two"
			ev.Output = "done"
			yield(ev, nil)
			return
		}
		if !yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ask-a", Message: "a"}), nil) {
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ask-b", Message: "b"}), nil)
	})
	wf := mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, two)})
	r, _ := newEndRunner(t, wf, true)

	turn1 := runEndTurn(t.Context(), r, userText("go"), nil)
	if len(turn1.errs) != 0 {
		t.Fatalf("turn 1 Run() errors = %v, want none", turn1.errs)
	}
	if diff := cmp.Diff([]string{"ask-a", "ask-b"}, requestedIDs(turn1.events)); diff != "" {
		t.Fatalf("turn 1 input requests (-want +got):\n%s", diff)
	}

	turn2 := runEndTurn(t.Context(), r, testutil.InputResponse("ask-a", "one"), nil)
	if len(turn2.errs) != 0 {
		t.Fatalf("turn 2 Run() errors = %v, want none", turn2.errs)
	}
	if diff := cmp.Diff([]string{"two:", "wf:", "wf:END"}, describeAll(turn2.events)); diff != "" {
		t.Errorf("turn 2 yielded events (-want +got):\n%s", diff)
	}
}
