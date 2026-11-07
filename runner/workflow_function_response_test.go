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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// TestRun_FunctionResponseResumesWorkflowSubAgent checks that when a sequential
// or parallel agent is the root, a user function response is routed to the
// sub-agent that made the long-running call, on that sub-agent's branch, instead
// of re-running the root.
func TestRun_FunctionResponseResumesWorkflowSubAgent(t *testing.T) {
	tests := []struct {
		name       string
		newRoot    func(caller, other agent.Agent) (agent.Agent, error)
		wantBranch string
		// wantOtherCalls is how many times the other agent calls its model
		// in total. It runs at most once, on the first turn.
		wantOtherCalls int
	}{
		{
			name: "parallel",
			newRoot: func(caller, other agent.Agent) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{caller, other}}})
			},
			wantBranch:     "root.caller",
			wantOtherCalls: 1,
		},
		{
			name: "sequential, caller runs second",
			newRoot: func(caller, other agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{other, caller}}})
			},
			wantBranch:     "",
			wantOtherCalls: 1,
		},
		{
			name: "sequential, caller runs first",
			newRoot: func(caller, other agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{caller, other}}})
			},
			wantBranch:     "",
			wantOtherCalls: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "approve", IsLongRunning: true},
				func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
			if err != nil {
				t.Fatal(err)
			}
			callerModel := &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "approve", Args: map[string]any{}}}}},
				genai.NewContentFromText("approved", genai.RoleModel),
			}}
			otherModel := &testutil.MockModel{Responses: []*genai.Content{
				genai.NewContentFromText("other answer", genai.RoleModel),
			}}
			caller, err := llmagent.New(llmagent.Config{Name: "caller", Model: callerModel, Tools: []tool.Tool{approve}})
			if err != nil {
				t.Fatal(err)
			}
			other, err := llmagent.New(llmagent.Config{Name: "other", Model: otherModel})
			if err != nil {
				t.Fatal(err)
			}
			root, err := tc.newRoot(caller, other)
			if err != nil {
				t.Fatal(err)
			}
			sessionService := session.InMemoryService()
			r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: sessionService})
			if err != nil {
				t.Fatal(err)
			}
			created, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u"})
			if err != nil {
				t.Fatal(err)
			}
			sessionID := created.Session.ID()

			for _, err := range r.Run(ctx, "u", sessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("first Run() error = %v", err)
				}
			}

			response := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "approve", Response: map[string]any{"ok": true}}}}}
			var got []*session.Event
			for ev, err := range r.Run(ctx, "u", sessionID, response, agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("resume Run() error = %v", err)
				}
				got = append(got, ev)
			}

			if len(got) != 1 {
				t.Fatalf("resume yielded %d events, want 1 from the caller", len(got))
			}
			if got[0].Author != "caller" || got[0].Branch != tc.wantBranch {
				t.Errorf("resume event author=%q branch=%q, want author=%q branch=%q", got[0].Author, got[0].Branch, "caller", tc.wantBranch)
			}
			if text := got[0].Content.Parts[0].Text; text != "approved" {
				t.Errorf("resume event text = %q, want %q", text, "approved")
			}
			if n := len(callerModel.Requests); n != 2 {
				t.Fatalf("caller called the model %d times, want 2", n)
			}
			if !hasFunctionResponse(callerModel.Requests[1].Contents, "call-1") {
				t.Errorf("caller's second model request has no function response for call-1")
			}
			if n := len(otherModel.Requests); n != tc.wantOtherCalls {
				t.Errorf("other agent called the model %d times, want %d (it must not re-run on resume)", n, tc.wantOtherCalls)
			}
		})
	}
}

func hasFunctionResponse(contents []*genai.Content, callID string) bool {
	for _, c := range contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.ID == callID {
				return true
			}
		}
	}
	return false
}

// TestRun_FunctionResponseInsideGraphWorkflowResumesWorkflow checks that an
// adk_request_input response to a node of a graph workflow goes through the
// workflow, at the root or nested, so Workflow.Resume runs the downstream nodes.
// Running the node agent on its own would ask for input again.
func TestRun_FunctionResponseInsideGraphWorkflowResumesWorkflow(t *testing.T) {
	tests := []struct {
		name    string
		newRoot func(wf agent.Agent) (agent.Agent, error)
	}{
		{
			name:    "graph workflow root",
			newRoot: func(wf agent.Agent) (agent.Agent, error) { return wf, nil },
		},
		{
			name: "graph workflow under sequential",
			newRoot: func(wf agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{wf}}})
			},
		},
		{
			name: "graph workflow under parallel",
			newRoot: func(wf agent.Agent) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{wf}}})
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
			bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
			wf, err := workflowagent.New(workflowagent.Config{
				Name: "wf", SubAgents: []agent.Agent{b, d},
				Edges: []workflow.Edge{{From: workflow.Start, To: bn}, {From: bn, To: dn}},
			})
			if err != nil {
				t.Fatal(err)
			}
			root, err := tc.newRoot(wf)
			if err != nil {
				t.Fatal(err)
			}
			r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatal(err)
			}
			for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
			}
			sawD := false
			for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("resume error: %v", err)
				}
				if ev.Author == "d" && ev.Content != nil {
					sawD = true
				}
			}
			if !sawD {
				t.Error("resume did not reach downstream node d")
			}
		})
	}
}

// TestRun_FunctionResponseSkipsSiblingsOfGraphWorkflow checks that a composite
// root does not re-run the sub-agents ahead of or beside a graph workflow when
// a node of the workflow gets its adk_request_input response. The LLM sibling
// would fail on a response it never saw the call for. Sub-agents after the
// workflow in a sequence still run.
func TestRun_FunctionResponseSkipsSiblingsOfGraphWorkflow(t *testing.T) {
	tests := []struct {
		name      string
		newRoot   func(subAgents ...agent.Agent) (agent.Agent, error)
		withAfter bool
	}{
		{
			name: "sequential, sibling ahead",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: subAgents}})
			},
		},
		{
			name: "sequential, sub-agent after the workflow runs",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: subAgents}})
			},
			withAfter: true,
		},
		{
			name: "sequential inside sequential",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				inner, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "inner", SubAgents: subAgents}})
				if err != nil {
					return nil, err
				}
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{inner}}})
			},
		},
		{
			name: "parallel",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: subAgents}})
			},
		},
		{
			name: "parallel, sibling after the workflow",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{subAgents[1], subAgents[0]}}})
			},
		},
		{
			name: "loop",
			newRoot: func(subAgents ...agent.Agent) (agent.Agent, error) {
				return loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: subAgents}, MaxIterations: 1})
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			otherModel := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("other answer", genai.RoleModel)}}
			other, err := llmagent.New(llmagent.Config{Name: "other", Model: otherModel})
			if err != nil {
				t.Fatal(err)
			}
			b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
			bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
			wf, err := workflowagent.New(workflowagent.Config{
				Name: "wf", SubAgents: []agent.Agent{b, d},
				Edges: []workflow.Edge{{From: workflow.Start, To: bn}, {From: bn, To: dn}},
			})
			if err != nil {
				t.Fatal(err)
			}
			subAgents := []agent.Agent{other, wf}
			if tc.withAfter {
				subAgents = append(subAgents, joinResumeAgent(t, "after", "A", false))
			}
			root, err := tc.newRoot(subAgents...)
			if err != nil {
				t.Fatal(err)
			}
			r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatal(err)
			}
			for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
			}
			authors := map[string]bool{}
			for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("resume error: %v", err)
				}
				authors[ev.Author] = true
			}
			if !authors["d"] {
				t.Error("resume did not reach downstream node d")
			}
			if authors["other"] {
				t.Error("resume re-ran the sibling other")
			}
			if tc.withAfter && !authors["after"] {
				t.Error("resume did not run the sub-agent after the workflow")
			}
		})
	}
}

// pausedWorkflow returns a graph workflow whose node ask asks for input and
// whose node done runs after the answer.
func pausedWorkflow(t *testing.T, name, ask, done string) agent.Agent {
	t.Helper()
	a, d := joinResumeAgent(t, ask, "ASK", true), joinResumeAgent(t, done, "DONE", false)
	an, dn := joinResumeAgentNode(t, a), joinResumeAgentNode(t, d)
	wf, err := workflowagent.New(workflowagent.Config{
		Name: name, SubAgents: []agent.Agent{a, d},
		Edges: []workflow.Edge{{From: workflow.Start, To: an}, {From: an, To: dn}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return wf
}

// customAgent returns an agent that runs emit each time it runs and counts the
// runs in *runs.
func customAgent(t *testing.T, name string, runs *atomic.Int32, emit func(agent.InvocationContext, *session.Event)) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			runs.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText(name, genai.RoleModel)
			if emit != nil {
				emit(ctx, ev)
			}
			yield(ev, nil)
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func mustSequential(t *testing.T, name string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func mustParallel(t *testing.T, name string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func mustLoop(t *testing.T, name string, maxIterations uint, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}, MaxIterations: maxIterations})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// runResumeTurn runs a first turn and then a turn with msg, and returns how
// many events each author yielded on the second turn. onResume runs between
// the two.
func runResumeTurn(t *testing.T, root agent.Agent, msg *genai.Content, onResume func()) (map[string]int, error) {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("first turn: %v", err)
		}
	}
	if onResume != nil {
		onResume()
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	authors := map[string]int{}
	for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			return authors, err
		}
		authors[ev.Author]++
	}
	return authors, nil
}

// TestRun_FunctionResponseNarrowsOnlyFirstLoopPass checks that a composite
// narrows to the workflow on the first pass of an enclosing loop only: every
// later pass runs its full list.
func TestRun_FunctionResponseNarrowsOnlyFirstLoopPass(t *testing.T) {
	tests := []struct {
		name    string
		newRoot func(t *testing.T, other, wf agent.Agent) agent.Agent
	}{
		{"loop", func(t *testing.T, other, wf agent.Agent) agent.Agent {
			return mustLoop(t, "root", 2, other, wf)
		}},
		{"sequential inside loop", func(t *testing.T, other, wf agent.Agent) agent.Agent {
			return mustLoop(t, "root", 2, mustSequential(t, "inner", other, wf))
		}},
		{"parallel inside loop", func(t *testing.T, other, wf agent.Agent) agent.Agent {
			return mustLoop(t, "root", 2, mustParallel(t, "inner", other, wf))
		}},
		{"loop inside loop", func(t *testing.T, other, wf agent.Agent) agent.Agent {
			return mustLoop(t, "root", 2, mustLoop(t, "inner", 1, other, wf))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var runs atomic.Int32
			other := customAgent(t, "other", &runs, nil)
			root := tc.newRoot(t, other, pausedWorkflow(t, "wf", "b", "d"))
			authors, err := runResumeTurn(t, root, resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), func() { runs.Store(0) })
			if err != nil {
				t.Fatalf("resume error: %v", err)
			}
			if authors["d"] == 0 {
				t.Errorf("resume did not reach downstream node d, authors = %v", authors)
			}
			if got := runs.Load(); got != 1 {
				t.Errorf("other ran %d times on the resume turn, want 1 (second pass only)", got)
			}
		})
	}
}

// TestRun_FunctionResponseEscalatorInLoopTerminates checks that an unbounded
// loop ends on the resume turn when the sub-agent that escalates sits ahead of
// the workflow. Skipping it on every pass would spin until the context ends.
func TestRun_FunctionResponseEscalatorInLoopTerminates(t *testing.T) {
	var runs atomic.Int32
	esc := customAgent(t, "esc", &runs, func(ctx agent.InvocationContext, ev *session.Event) {
		events := ctx.Session().Events()
		for i := range events.Len() {
			if a := events.At(i).Author; a == "b" || a == "d" {
				ev.Actions.Escalate = true
			}
		}
	})
	root := mustLoop(t, "root", 0, mustSequential(t, "inner", esc, pausedWorkflow(t, "wf", "b", "d")))
	authors, err := runResumeTurn(t, root, resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), nil)
	if err != nil {
		t.Fatalf("resume did not terminate: authors = %v, err = %v", authors, err)
	}
	if authors["d"] == 0 {
		t.Errorf("resume did not reach downstream node d, authors = %v", authors)
	}
}

// TestRun_FunctionResponsesSelectEveryCaller checks that every response in the
// message selects a sub-agent, whatever order the parts arrive in.
func TestRun_FunctionResponsesSelectEveryCaller(t *testing.T) {
	const externalCallID = "external-call"
	externalCall := func(_ agent.InvocationContext, ev *session.Event) {
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: externalCallID, Name: "external"}}}}
	}
	askA := resumeContent("a", workflow.WorkflowInputFunctionCallName, "approved").Parts[0]
	askB := resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved").Parts[0]
	answerExternal := resumeContent(externalCallID, "external", "approved").Parts[0]

	tests := []struct {
		name    string
		newRoot func(t *testing.T, wfA, wfB, other, caller agent.Agent) agent.Agent
		parts   []*genai.Part
		wantRan []string
		// wantOther is how many times the sibling other runs on the resume turn.
		wantOther int32
	}{
		{
			name: "parallel resumes both workflows, not the sibling",
			newRoot: func(t *testing.T, wfA, wfB, other, _ agent.Agent) agent.Agent {
				return mustParallel(t, "root", wfA, other, wfB)
			},
			parts:   []*genai.Part{askA, askB},
			wantRan: []string{"c", "d"},
		},
		{
			name: "parallel, parts in reverse order",
			newRoot: func(t *testing.T, wfA, wfB, other, _ agent.Agent) agent.Agent {
				return mustParallel(t, "root", wfA, other, wfB)
			},
			parts:   []*genai.Part{askB, askA},
			wantRan: []string{"c", "d"},
		},
		{
			name: "sequential starts from the lowest index",
			newRoot: func(t *testing.T, wfA, wfB, other, _ agent.Agent) agent.Agent {
				return mustSequential(t, "root", other, wfA, wfB)
			},
			parts:   []*genai.Part{askB, askA},
			wantRan: []string{"c", "d"},
		},
		{
			name: "loop starts from the lowest index",
			newRoot: func(t *testing.T, wfA, wfB, other, _ agent.Agent) agent.Agent {
				return mustLoop(t, "root", 1, other, wfA, wfB)
			},
			parts:   []*genai.Part{askB, askA},
			wantRan: []string{"c", "d"},
		},
		{
			name: "parallel of sequences narrows inside each",
			newRoot: func(t *testing.T, wfA, wfB, other, _ agent.Agent) agent.Agent {
				return mustParallel(t, "root", mustSequential(t, "seqA", other, wfA), wfB)
			},
			parts:   []*genai.Part{askA, askB},
			wantRan: []string{"c", "d"},
		},
		{
			name: "parallel also runs a sub-agent outside a graph that gets a response",
			newRoot: func(t *testing.T, wfA, _, other, caller agent.Agent) agent.Agent {
				return mustParallel(t, "root", wfA, other, caller)
			},
			parts:   []*genai.Part{askA, answerExternal},
			wantRan: []string{"c", "caller"},
		},
		{
			name: "no narrowing without a caller in a graph",
			newRoot: func(t *testing.T, _, _, other, caller agent.Agent) agent.Agent {
				return mustParallel(t, "root", other, caller)
			},
			parts:     []*genai.Part{resumeContent("unknown-call", "external", "approved").Parts[0], answerExternal},
			wantRan:   []string{"caller"},
			wantOther: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var otherRuns, callerRuns atomic.Int32
			other := customAgent(t, "other", &otherRuns, nil)
			caller := customAgent(t, "caller", &callerRuns, externalCall)
			root := tc.newRoot(t, pausedWorkflow(t, "wfA", "a", "c"), pausedWorkflow(t, "wfB", "b", "d"), other, caller)
			msg := &genai.Content{Role: genai.RoleUser, Parts: tc.parts}
			authors, err := runResumeTurn(t, root, msg, func() { otherRuns.Store(0) })
			if err != nil {
				t.Fatalf("resume error: %v, authors = %v", err, authors)
			}
			for _, name := range tc.wantRan {
				if authors[name] == 0 {
					t.Errorf("%s did not run on the resume turn, authors = %v", name, authors)
				}
			}
			if got := otherRuns.Load(); got != tc.wantOther {
				t.Errorf("sibling other ran %d times on the resume turn, want %d", got, tc.wantOther)
			}
		})
	}
}
