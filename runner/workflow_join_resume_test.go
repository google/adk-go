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
	"errors"
	"iter"
	"reflect"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func joinResumeAgent(t *testing.T, name, output string, interrupt bool) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			if interrupt {
				emit := func(ev *session.Event) error {
					if !yield(ev, nil) {
						return errors.New("consumer stopped")
					}
					return nil
				}
				if _, err := workflow.ResumeOrRequestInput(agent.Promote(ctx), emit, session.RequestInput{InterruptID: name, Message: "approve"}); err != nil {
					return
				}
				yield(&session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(output, genai.RoleModel)}}, nil)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText(output, genai.RoleModel)
			yield(ev, nil)
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func newJoinResumeRunner(t *testing.T, edges []workflow.Edge, agents ...agent.Agent) *runner.Runner {
	t.Helper()
	root, err := workflowagent.New(workflowagent.Config{Name: "root", SubAgents: agents, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func joinResumeAgentNode(t *testing.T, a agent.Agent) workflow.Node {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// RunNode output is not the orchestrator's output unless it explicitly delegates.
func TestRunner_WorkflowHITL_DynamicOutputOwnership(t *testing.T) {
	for _, useJoin := range []bool{false, true} {
		for _, delegate := range []bool{false, true} {
			name := map[bool]string{false: "terminal", true: "join"}[useJoin] + "/" + map[bool]string{false: "discarded", true: "delegated"}[delegate]
			t.Run(name, func(t *testing.T) {
				b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
				bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
				q := workflow.NewFunctionNode("q", func(agent.Context, any) (string, error) { return "q-out", nil }, workflow.NodeConfig{})
				p := workflow.NewDynamicNode("p", func(ctx agent.Context, in any, _ func(*session.Event) error) (any, error) {
					var opts []workflow.RunNodeOption
					if delegate {
						opts = append(opts, workflow.WithUseAsOutput())
					}
					_, err := workflow.RunNode[string](ctx, q, in, opts...)
					return nil, err
				}, workflow.NodeConfig{})
				edges := []workflow.Edge{{From: workflow.Start, To: p}, {From: workflow.Start, To: bn}}
				if useJoin {
					join := workflow.NewJoinNode("join")
					edges = append(edges, workflow.Edge{From: p, To: join}, workflow.Edge{From: bn, To: join}, workflow.Edge{From: join, To: dn})
				} else {
					edges = append(edges, workflow.Edge{From: bn, To: dn})
				}
				r := newJoinResumeRunner(t, edges, b, d)
				for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						t.Fatal(err)
					}
				}
				var sawD, sawJoin bool
				var runErr error
				for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
					if err != nil {
						runErr = err
						continue
					}
					if ev.Author == "d" && ev.Content != nil {
						sawD = true
					}
					if out, ok := ev.Output.(map[string]any); ok {
						sawJoin = true
						var wantP any
						if delegate {
							wantP = "q-out"
						}
						if !reflect.DeepEqual(out, map[string]any{"p": wantP, "b": "approved"}) {
							t.Error("join did not preserve the orchestrator's output ownership")
						}
					}
				}
				if useJoin && !sawJoin || !sawD {
					t.Error("resume did not reach the join and downstream node")
				}
				if !useJoin && delegate {
					if !errors.Is(runErr, workflow.ErrMultipleTerminalOutputs) {
						t.Fatal("delegated terminal output was not counted")
					}
				} else if runErr != nil {
					t.Fatal(runErr)
				}
			})
		}
	}
}

// An output emitted before failure must not open the barrier on a later turn.
func TestRunner_WorkflowHITL_FailedSiblingRetries(t *testing.T) {
	for _, child := range []bool{false, true} {
		t.Run(map[bool]string{false: "direct", true: "dynamic_child"}[child], func(t *testing.T) { runFailedSiblingRetry(t, child) })
	}
}

func runFailedSiblingRetry(t *testing.T, child bool) {
	t.Helper()
	b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
	bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
	paused := make(chan struct{})
	failure := errors.New("scripted node failure")
	var attempts atomic.Int32
	fn := func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if attempts.Add(1) != 1 {
			return "C-retry", nil
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "C"
		if err := emit(ev); err != nil {
			return nil, err
		}
		select {
		case <-paused:
			return nil, failure
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	c := workflow.NewDynamicNode("c", fn, workflow.NodeConfig{})
	if child {
		q := workflow.NewDynamicNode("q", fn, workflow.NodeConfig{})
		c = workflow.NewDynamicNode("c", func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
			if _, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "c", Message: "approve"}); err != nil {
				return nil, err
			}
			return workflow.RunNode[any](ctx, q, in)
		}, workflow.NodeConfig{})
	}
	join := workflow.NewJoinNode("join")
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: bn}, {From: workflow.Start, To: c}, {From: bn, To: join}, {From: c, To: join}, {From: join, To: dn}}, b, d)
	var sawFailure, sawPause bool
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			sawFailure = errors.Is(err, failure) || child && errors.Is(err, workflow.ErrNodeFailed)
			continue
		}
		if ev.RequestedInput != nil && !sawPause {
			sawPause = true
			close(paused)
		}
	}
	if child {
		for _, err := range r.Run(t.Context(), "u", "s", resumeContent("c", workflow.WorkflowInputFunctionCallName, "approved C"), agent.RunConfig{}) {
			if err != nil {
				sawFailure = errors.Is(err, workflow.ErrNodeFailed)
			}
		}
	}
	if !sawFailure || !sawPause {
		t.Fatal("fixture did not produce both the pause and sibling failure")
	}
	var sawD, sawJoin bool
	for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if out, ok := ev.Output.(map[string]any); ok {
			sawJoin = true
			if !reflect.DeepEqual(out, map[string]any{"b": "approved", "c": "C-retry"}) {
				t.Error("join consumed output from the failed attempt")
			}
		}
		if ev.Author == "d" && ev.Content != nil {
			sawD = true
		}
	}
	if attempts.Load() != 2 || !sawJoin || !sawD {
		t.Fatal("resume did not retry the failed predecessor before advancing")
	}
}

func TestRunner_WorkflowHITL_PreviouslyResumedSibling(t *testing.T) {
	b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
	bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
	var attempts atomic.Int32
	c := workflow.NewDynamicNode("c", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		attempts.Add(1)
		return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "c", Message: "approve"})
	}, workflow.NodeConfig{})
	join := workflow.NewJoinNode("join")
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: bn}, {From: workflow.Start, To: c}, {From: bn, To: join}, {From: c, To: join}, {From: join, To: dn}}, b, d)
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("c", workflow.WorkflowInputFunctionCallName, "approved C"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Author == "d" {
			t.Fatal("join advanced before both approvals")
		}
	}
	var sawJoin, sawD bool
	for ev, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if out, ok := ev.Output.(map[string]any); ok {
			sawJoin = true
			if !reflect.DeepEqual(out, map[string]any{"b": "approved", "c": "approved C"}) {
				t.Error("join lost the result from the earlier re-entry turn")
			}
		}
		if ev.Author == "d" && ev.Content != nil {
			sawD = true
		}
	}
	if !sawJoin || !sawD || attempts.Load() != 2 {
		t.Fatal("completed re-entry sibling was re-run or blocked the join")
	}
	for _, err := range r.Run(t.Context(), "u", "s", resumeContent("c", workflow.WorkflowInputFunctionCallName, "approved C"), agent.RunConfig{}) {
		if err != nil && !errors.Is(err, workflow.ErrNothingToResume) {
			t.Fatal(err)
		}
	}
	if attempts.Load() != 2 {
		t.Fatal("duplicate answer re-ran a completed re-entry sibling")
	}
}

func TestRunner_WorkflowHITL_TerminalOutputsAcrossTurns(t *testing.T) {
	b, c := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "c", "C", false)
	r := newJoinResumeRunner(t, []workflow.Edge{{From: workflow.Start, To: joinResumeAgentNode(t, b)}, {From: workflow.Start, To: joinResumeAgentNode(t, c)}}, b, c)
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	var runErr error
	for _, err := range r.Run(t.Context(), "u", "s", resumeContent("b", workflow.WorkflowInputFunctionCallName, "approved"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
		}
	}
	if !errors.Is(runErr, workflow.ErrMultipleTerminalOutputs) {
		t.Fatal("resume did not enforce the single-terminal-output rule")
	}
}

func TestRunner_WorkflowHITL_AgentJoinResume(t *testing.T) {
	for _, secondWait := range []bool{false, true} {
		name := "completed_sibling"
		if secondWait {
			name = "two_waiting_siblings"
		}
		t.Run(name, func(t *testing.T) { runAgentJoinResume(t, secondWait) })
	}
}

func runAgentJoinResume(t *testing.T, secondWait bool) {
	t.Helper()
	b := joinResumeAgent(t, "b", "B", true)
	c := joinResumeAgent(t, "c", "C", secondWait)
	d := joinResumeAgent(t, "d", "D", false)
	bNode, err := workflow.NewAgentNode(b, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	cNode, err := workflow.NewAgentNode(c, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	dNode, err := workflow.NewAgentNode(d, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	join := workflow.NewJoinNode("join")
	root, err := workflowagent.New(workflowagent.Config{Name: "root", SubAgents: []agent.Agent{b, c, d}, Edges: []workflow.Edge{
		{From: workflow.Start, To: bNode},
		{From: workflow.Start, To: cNode},
		{From: bNode, To: join},
		{From: cNode, To: join},
		{From: join, To: dNode},
	}})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "repro", Agent: root, SessionService: svc, AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	for ev, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Author == "d" {
			t.Fatal("D executed before approval")
		}
	}
	msg := genai.NewContentFromFunctionResponse(workflow.WorkflowInputFunctionCallName, map[string]any{"response": "approved"}, genai.RoleUser)
	msg.Parts[0].FunctionResponse.ID = "b"
	wantC := "C"
	if secondWait {
		for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
			if err != nil {
				t.Fatal(err)
			}
			if ev.Author == "d" || ev.Output != nil {
				t.Fatal("join advanced while C still waited")
			}
		}
		msg = resumeContent("c", workflow.WorkflowInputFunctionCallName, "approved C")
		wantC = "approved C"
	}
	var sawD, sawJoin bool
	for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Author == "c" {
			t.Fatal("completed C ran again")
		}
		if got, ok := ev.Output.(map[string]any); ok {
			sawJoin = true
			if !reflect.DeepEqual(got, map[string]any{"b": "approved", "c": wantC}) {
				t.Fatal("join output did not match the completed predecessors")
			}
		}
		if ev.Author == "d" {
			sawD = true
		}
	}
	if !sawJoin {
		t.Fatal("join did not emit predecessor outputs")
	}
	if !sawD {
		t.Fatal("D never executed after approval; completed predecessor must survive rehydration")
	}
}
