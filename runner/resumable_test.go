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
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// longRunningAsker returns an LLM agent whose model replies with responses in
// order and that has the long-running tool "approve". After the call the flow
// asks the model again in the same turn, so each call is followed by a reply.
func longRunningAsker(t *testing.T, name string, responses ...*genai.Content) (agent.Agent, *testutil.MockModel) {
	t.Helper()
	approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "approve", IsLongRunning: true},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"status": "pending"}, nil })
	if err != nil {
		t.Fatal(err)
	}
	m := &testutil.MockModel{Responses: responses}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Tools: []tool.Tool{approve}})
	if err != nil {
		t.Fatal(err)
	}
	return a, m
}

func approveCall(id string) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "approve", Args: map[string]any{}}}}}
}

func text(s string) *genai.Content { return genai.NewContentFromText(s, genai.RoleModel) }

// runTurns runs a first turn and then one turn per message in msgs, all in one
// session, and returns how many events with content each author yielded on
// each turn.
func runTurns(t *testing.T, root agent.Agent, resumable bool, msgs ...*genai.Content) []map[string]int {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true, Resumable: resumable})
	if err != nil {
		t.Fatal(err)
	}
	msgs = append([]*genai.Content{genai.NewContentFromText("start", genai.RoleUser)}, msgs...)
	turns := make([]map[string]int, 0, len(msgs))
	for i, msg := range msgs {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		authors := map[string]int{}
		for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
			if err != nil {
				cancel()
				t.Fatalf("turn %d: %v, authors = %v", i, err, authors)
			}
			if ev.Content != nil {
				authors[ev.Author]++
			}
		}
		cancel()
		turns = append(turns, authors)
	}
	return turns
}

// TestRun_ResumableSequenceContinuesAfterAnswer checks that a sequence stops
// on a long-running call and, once the call is answered, continues from the
// caller: the caller gets the answer and the sub-agents after it run, once.
func TestRun_ResumableSequenceContinuesAfterAnswer(t *testing.T) {
	tests := []struct {
		name    string
		newRoot func(t *testing.T, before, asker, other, after agent.Agent) agent.Agent
	}{
		{"sequential", func(t *testing.T, before, asker, other, after agent.Agent) agent.Agent {
			return mustSequential(t, "root", before, asker, other, after)
		}},
		{"nested sequential", func(t *testing.T, before, asker, other, after agent.Agent) agent.Agent {
			return mustSequential(t, "root", before, mustSequential(t, "inner", asker, other), after)
		}},
		{"sequential inside parallel", func(t *testing.T, before, asker, other, after agent.Agent) agent.Agent {
			return mustParallel(t, "root", mustSequential(t, "inner", before, asker, other, after))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			asker, askerModel := longRunningAsker(t, "asker", approveCall("lr1"), text("waiting"), text("approved"))
			var beforeRuns, otherRuns, afterRuns atomic.Int32
			root := tc.newRoot(t, customAgent(t, "before", &beforeRuns, nil), asker, customAgent(t, "other", &otherRuns, nil), customAgent(t, "after", &afterRuns, nil))

			turns := runTurns(t, root, true, resumeContent("lr1", "approve", "ok"))

			want := []map[string]int{
				{"before": 1, "asker": 3}, // call, pending response, "waiting"
				{"asker": 1, "other": 1, "after": 1},
			}
			if diff := cmp.Diff(want, turns); diff != "" {
				t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
			}
			if n := len(askerModel.Requests); n != 3 || !hasFunctionResponse(askerModel.Requests[2].Contents, "lr1") {
				t.Errorf("asker did not get the answer on the resume turn, requests = %d", n)
			}
		})
	}
}

// TestRun_ResumableParallelWaitsForEveryAnswer checks that a parallel agent
// whose sub-agents each made a long-running call holds the sequence above it
// until both calls are answered, and does not run a finished sub-agent again.
func TestRun_ResumableParallelWaitsForEveryAnswer(t *testing.T) {
	a, aModel := longRunningAsker(t, "a", approveCall("a1"), text("waiting"), text("a approved"))
	b, bModel := longRunningAsker(t, "b", approveCall("b1"), text("waiting"), text("b approved"))
	var besideRuns, afterRuns atomic.Int32
	root := mustSequential(t, "root", mustParallel(t, "par", a, b, customAgent(t, "beside", &besideRuns, nil)), customAgent(t, "after", &afterRuns, nil))

	turns := runTurns(t, root, true, resumeContent("a1", "approve", "ok"), resumeContent("b1", "approve", "ok"))

	want := []map[string]int{
		{"a": 3, "b": 3, "beside": 1},
		{"a": 1},
		{"b": 1, "after": 1},
	}
	if diff := cmp.Diff(want, turns); diff != "" {
		t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
	}
	if n := len(aModel.Requests); n != 3 {
		t.Errorf("a called its model %d times, want 3", n)
	}
	if n := len(bModel.Requests); n != 3 || !hasFunctionResponse(bModel.Requests[2].Contents, "b1") {
		t.Errorf("b did not get its answer on the last turn, requests = %d", n)
	}
}

// TestRun_ResumableLoopCountsPassesAcrossTurns checks that a loop paused on a
// long-running call continues the same pass once the call is answered, and
// that MaxIterations counts the passes of earlier turns.
func TestRun_ResumableLoopCountsPassesAcrossTurns(t *testing.T) {
	asker, askerModel := longRunningAsker(t, "asker",
		approveCall("lr1"), text("waiting"), text("approved 1"),
		approveCall("lr2"), text("waiting"), text("approved 2"),
		approveCall("lr3"), text("waiting"))
	var beforeRuns, afterRuns atomic.Int32
	root := mustLoop(t, "root", 2, customAgent(t, "before", &beforeRuns, nil), asker, customAgent(t, "after", &afterRuns, nil))

	turns := runTurns(t, root, true, resumeContent("lr1", "approve", "ok"), resumeContent("lr2", "approve", "ok"))

	want := []map[string]int{
		{"before": 1, "asker": 3},
		{"asker": 4, "after": 1, "before": 1}, // "approved 1", then pass 2 asks again
		{"asker": 1, "after": 1},
	}
	if diff := cmp.Diff(want, turns); diff != "" {
		t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
	}
	if n := len(askerModel.Requests); n != 6 {
		t.Errorf("asker called its model %d times, want 6 (no third pass)", n)
	}
}

// TestRun_ResumableGraphWorkflowInSequence checks that a graph workflow asking
// for input inside a sequence resumes with the answer and the sequence then
// runs the sub-agent after it.
func TestRun_ResumableGraphWorkflowInSequence(t *testing.T) {
	var afterRuns atomic.Int32
	root := mustSequential(t, "root", pausedWorkflow(t, "wf", "b", "d"), customAgent(t, "after", &afterRuns, nil))

	turns := runTurns(t, root, true, resumeContent("b", workflow.WorkflowInputFunctionCallName, "ok"))

	if got := turns[0]["after"]; got != 0 {
		t.Errorf("after ran %d times before the answer, want 0", got)
	}
	if got := turns[1]["after"]; got != 1 {
		t.Errorf("after ran %d times on the resume turn, want 1, authors = %v", got, turns[1])
	}
}

// TestRun_NotResumableKeepsRunning checks that without Config.Resumable a
// sequence still runs the sub-agents after a long-running call on the same
// turn, as before.
func TestRun_NotResumableKeepsRunning(t *testing.T) {
	asker, _ := longRunningAsker(t, "asker", approveCall("lr1"), text("waiting"), text("approved"))
	var afterRuns atomic.Int32
	root := mustSequential(t, "root", asker, customAgent(t, "after", &afterRuns, nil))

	turns := runTurns(t, root, false)

	if got := turns[0]["after"]; got != 1 {
		t.Errorf("after ran %d times on the first turn, want 1", got)
	}
}

// TestRun_ResumableParallelInsideLoop checks that a parallel agent on a later
// pass of a loop waits for the sub-agent that paused on that pass, even though
// the sub-agent finished on an earlier pass.
func TestRun_ResumableParallelInsideLoop(t *testing.T) {
	a, aModel := longRunningAsker(t, "a", text("a done"), approveCall("a2"), text("waiting"), text("a approved"))
	var bRuns atomic.Int32
	root := mustLoop(t, "root", 2, mustParallel(t, "par", a, customAgent(t, "b", &bRuns, nil)))

	turns := runTurns(t, root, true, resumeContent("a2", "approve", "ok"))

	want := []map[string]int{
		{"a": 4, "b": 2}, // pass 1: "a done"; pass 2: call, pending response, "waiting"
		{"a": 1},
	}
	if diff := cmp.Diff(want, turns); diff != "" {
		t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
	}
	if n := len(aModel.Requests); n != 4 || !hasFunctionResponse(aModel.Requests[3].Contents, "a2") {
		t.Errorf("a did not get its answer on the resume turn, requests = %d", n)
	}
}

// TestRun_ResumableTransferResumesSubAgent checks that an LLM agent that
// transferred to a sub-agent before the pause resumes that sub-agent once its
// call is answered, rather than asking its own model again, as adk-python's
// LlmAgent does.
func TestRun_ResumableTransferResumesSubAgent(t *testing.T) {
	asker, askerModel := longRunningAsker(t, "asker", approveCall("lr1"), text("waiting"), text("approved"))
	coordModel := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "asker"}, genai.RoleModel),
	}}
	coord, err := llmagent.New(llmagent.Config{Name: "coord", Model: coordModel, SubAgents: []agent.Agent{asker}})
	if err != nil {
		t.Fatal(err)
	}
	var afterRuns atomic.Int32
	root := mustSequential(t, "root", coord, customAgent(t, "after", &afterRuns, nil))

	turns := runTurns(t, root, true, resumeContent("lr1", "approve", "ok"))

	want := []map[string]int{
		{"coord": 2, "asker": 3}, // transfer call and response; call, pending response, "waiting"
		{"asker": 1, "after": 1},
	}
	if diff := cmp.Diff(want, turns); diff != "" {
		t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
	}
	if n := len(coordModel.Requests); n != 1 {
		t.Errorf("coord called its model %d times, want 1", n)
	}
	if n := len(askerModel.Requests); n != 3 || !hasFunctionResponse(askerModel.Requests[2].Contents, "lr1") {
		t.Errorf("asker did not get the answer on the resume turn, requests = %d", n)
	}
}

// TestRun_ResumableLoopStopsOnEscalation checks that escalation still ends a
// loop in a resumable run.
func TestRun_ResumableLoopStopsOnEscalation(t *testing.T) {
	var runs atomic.Int32
	escalator := customAgent(t, "escalator", &runs, func(_ agent.InvocationContext, ev *session.Event) { ev.Actions.Escalate = true })
	root := mustLoop(t, "root", 3, escalator)

	runTurns(t, root, true)

	if got := runs.Load(); got != 1 {
		t.Errorf("escalator ran %d times, want 1", got)
	}
}

// TestRun_ResumableRootLLMAgent checks that with an LLM agent at the root, the
// answer reaches the agent that made the call, whether that is the root itself
// or a sub-agent it transferred to.
func TestRun_ResumableRootLLMAgent(t *testing.T) {
	t.Run("root made the call", func(t *testing.T) {
		asker, askerModel := longRunningAsker(t, "asker", approveCall("lr1"), text("waiting"), text("approved"))

		turns := runTurns(t, asker, true, resumeContent("lr1", "approve", "ok"))

		if diff := cmp.Diff([]map[string]int{{"asker": 3}, {"asker": 1}}, turns); diff != "" {
			t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
		}
		if n := len(askerModel.Requests); n != 3 || !hasFunctionResponse(askerModel.Requests[2].Contents, "lr1") {
			t.Errorf("asker did not get the answer, requests = %d", n)
		}
	})
	t.Run("root transferred", func(t *testing.T) {
		asker, askerModel := longRunningAsker(t, "asker", approveCall("lr1"), text("waiting"), text("approved"))
		coordModel := &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "asker"}, genai.RoleModel),
		}}
		coord, err := llmagent.New(llmagent.Config{Name: "coord", Model: coordModel, SubAgents: []agent.Agent{asker}})
		if err != nil {
			t.Fatal(err)
		}

		turns := runTurns(t, coord, true, resumeContent("lr1", "approve", "ok"))

		if diff := cmp.Diff([]map[string]int{{"coord": 2, "asker": 3}, {"asker": 1}}, turns); diff != "" {
			t.Errorf("events by author per turn mismatch (-want +got):\n%s", diff)
		}
		if n := len(coordModel.Requests); n != 1 {
			t.Errorf("coord called its model %d times, want 1", n)
		}
		if n := len(askerModel.Requests); n != 3 || !hasFunctionResponse(askerModel.Requests[2].Contents, "lr1") {
			t.Errorf("asker did not get the answer, requests = %d", n)
		}
	})
}

// TestRun_ResumableRecordsCheckpoints pins the checkpoint events a sequence
// and a loop write, in adk-python's shape: the sub-agent each is on, the
// loop's finished passes, and an end-of-agent event when each finishes.
func TestRun_ResumableRecordsCheckpoints(t *testing.T) {
	var aRuns, bRuns atomic.Int32
	root := mustSequential(t, "root", customAgent(t, "a", &aRuns, nil), mustLoop(t, "loop", 1, customAgent(t, "b", &bRuns, nil)))
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: service, AutoCreateSession: true, Resumable: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	got, err := service.Get(t.Context(), &session.GetRequest{AppName: "repro", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	type checkpoint struct {
		Author     string
		State      map[string]any
		EndOfAgent bool
	}
	var checkpoints []checkpoint
	for ev := range got.Session.Events().All() {
		if ev.Actions.AgentState != nil || ev.Actions.EndOfAgent {
			checkpoints = append(checkpoints, checkpoint{ev.Author, ev.Actions.AgentState, ev.Actions.EndOfAgent})
		}
	}
	want := []checkpoint{
		{Author: "root", State: map[string]any{"current_sub_agent": "a"}},
		{Author: "root", State: map[string]any{"current_sub_agent": "loop"}},
		{Author: "loop", State: map[string]any{"current_sub_agent": "b", "times_looped": 0}},
		{Author: "loop", EndOfAgent: true},
		{Author: "root", EndOfAgent: true},
	}
	if diff := cmp.Diff(want, checkpoints); diff != "" {
		t.Errorf("checkpoints mismatch (-want +got):\n%s", diff)
	}
}

// TestRun_ResumableAgentAnswersItsOwnCall checks that an LLM agent that
// transferred earlier in the invocation, here on a loop's first pass, keeps
// the answer to its own call rather than resuming the agent it transferred to.
func TestRun_ResumableAgentAnswersItsOwnCall(t *testing.T) {
	var helperRuns atomic.Int32
	helper := customAgent(t, "helper", &helperRuns, nil)
	approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "approve", IsLongRunning: true},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"status": "pending"}, nil })
	if err != nil {
		t.Fatal(err)
	}
	coordModel := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "helper"}, genai.RoleModel),
		approveCall("lr1"), text("waiting"), text("approved"),
	}}
	coord, err := llmagent.New(llmagent.Config{Name: "coord", Model: coordModel, Tools: []tool.Tool{approve}, SubAgents: []agent.Agent{helper}})
	if err != nil {
		t.Fatal(err)
	}
	root := mustLoop(t, "root", 2, coord)

	runTurns(t, root, true, resumeContent("lr1", "approve", "ok"))

	if got := helperRuns.Load(); got != 1 {
		t.Errorf("helper ran %d times, want 1 (only on the first pass)", got)
	}
	if n := len(coordModel.Requests); n != 4 || !hasFunctionResponse(coordModel.Requests[3].Contents, "lr1") {
		t.Errorf("coord did not get the answer to its own call, requests = %d", n)
	}
}
