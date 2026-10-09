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
	"fmt"
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

func run3(t *testing.T, r *runner.Runner, msg *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var evs []*session.Event
	var first error
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			if first == nil {
				first = err
			}
			continue
		}
		evs = append(evs, ev)
	}
	return evs, first
}

// model3 records every request. It answers a confirm_action response with
// text and anything else with a confirm_action call.
type model3 struct{ calls [][]*genai.Content }

func (m *model3) Name() string { return "model3" }

func (m *model3) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls = append(m.calls, append([]*genai.Content(nil), req.Contents...))
	return func(yield func(*model.LLMResponse, error) bool) {
		if n := len(req.Contents); n > 0 {
			for _, p := range req.Contents[n-1].Parts {
				if p.FunctionResponse != nil && p.FunctionResponse.Name == "confirm_action" {
					yield(&model.LLMResponse{Content: genai.NewContentFromText("done", "model")}, nil)
					return
				}
			}
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model")}, nil)
	}
}

func (m *model3) callSees(i int, text string) bool {
	for _, c := range m.calls[i] {
		for _, p := range c.Parts {
			if strings.Contains(p.Text, text) {
				return true
			}
		}
	}
	return false
}

func gated3(t *testing.T, name string, m model.LLM) *workflow.AgentNode {
	t.Helper()
	ct, err := functiontool.New(functiontool.Config{Name: "confirm_action", Description: "x", RequireConfirmation: true},
		func(agent.Context, struct{}) (map[string]string, error) {
			return map[string]string{"result": "executed"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Tools: []tool.Tool{ct}})
	if err != nil {
		t.Fatal(err)
	}
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func approve3(id string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: "adk_request_confirmation", Response: map[string]any{"confirmed": true},
	}}}}
}

func graphRunner3(t *testing.T, edges []workflow.Edge) *runner.Runner {
	t.Helper()
	wf, err := workflowagent.New(workflowagent.Config{Name: workflowAgentName, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	return newNodeTestRunner(t, wf, svc)
}

// contentOnly is a re-entry node whose successful resume emits only content.
type contentOnly struct {
	workflow.BaseNode
	resumes atomic.Int32
}

func (n *contentOnly) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput("q"); ok {
			n.resumes.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText(fmt.Sprintf("handled %v", resp), "model")
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "q", Message: "q"}), nil)
	}
}

// A re-entry node that finished its resume without Output has still consumed
// the reply. Answering an unrelated question afterwards must not run it again.
func TestResumeRound3_ReentryNodeWithoutOutputConsumesReply(t *testing.T) {
	yes := true
	a := &contentOnly{BaseNode: workflow.NewBaseNode("asker", "", workflow.NodeConfig{RerunOnResume: &yes})}
	other := newHitlAsker("other", "io", false)
	osink := workflow.NewFunctionNode("osink", func(_ agent.Context, in any) (any, error) { return in, nil }, workflow.NodeConfig{})
	r := newWorkflowRunner(t, append(workflow.Chain(workflow.Start, a), workflow.Chain(workflow.Start, other, osink)...))
	t1, _ := run3(t, r, userText("start"))
	names := map[string]string{}
	for _, ev := range t1 {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				names[p.FunctionCall.ID] = p.FunctionCall.Name
			}
		}
	}
	if _, err := run3(t, r, resumeContent("q", names["q"], "ok")); err != nil {
		t.Fatalf("answering asker: %v", err)
	}
	if _, err := run3(t, r, resumeContent("q", names["q"], "ok")); err == nil {
		t.Error("duplicate reply returned nil, want ErrNothingToResume")
	}
	if _, err := run3(t, r, resumeContent("q", names["q"], "changed")); err == nil {
		t.Error("changed reply returned nil, want ErrNothingToResume")
	}
	if _, err := run3(t, r, resumeContent("io", names["io"], "other")); err != nil {
		t.Fatalf("answering other: %v", err)
	}
	if got := a.resumes.Load(); got != 1 {
		t.Errorf("asker resumed %d times, want 1 (an unrelated reply re-ran it)", got)
	}
}

type writer3 struct {
	workflow.BaseNode
	runs atomic.Int32
}

func (n *writer3) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = fmt.Sprintf("INPUT-%d", n.runs.Add(1))
		yield(ev, nil)
	}
}

type router3 struct {
	workflow.BaseNode
	runs atomic.Int32
}

func (n *router3) Run(ctx agent.Context, in any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = in
		ev.Routes = []string{"finish"}
		if n.runs.Add(1) == 1 {
			ev.Routes = []string{"loop"}
		}
		yield(ev, nil)
	}
}

// writer -> gated -> router, looping once. After the second approval the
// gated node must still see its second input.
func TestResumeRound3_SingleTurnResumeInLoopKeepsInput(t *testing.T) {
	m := &model3{}
	w := &writer3{BaseNode: workflow.NewBaseNode("writer", "", workflow.NodeConfig{})}
	g := gated3(t, "gated", m)
	rt := &router3{BaseNode: workflow.NewBaseNode("router", "", workflow.NodeConfig{})}
	fin := workflow.NewFunctionNode("fin", func(_ agent.Context, in any) (any, error) { return in, nil }, workflow.NodeConfig{})
	r := graphRunner3(t, []workflow.Edge{
		{From: workflow.Start, To: w},
		{From: w, To: g},
		{From: g, To: rt},
		{From: rt, To: w, Route: workflow.StringRoute("loop")},
		{From: rt, To: fin, Route: workflow.StringRoute("finish")},
	})
	t1, _ := run3(t, r, userText("start"))
	id1, _ := findLongRunningInterrupt(t1)
	t2, err := run3(t, r, approve3(id1))
	if err != nil {
		t.Fatalf("first approval: %v", err)
	}
	id2, _ := findLongRunningInterrupt(t2)
	if _, err := run3(t, r, approve3(id2)); err != nil {
		t.Fatalf("second approval: %v", err)
	}
	if len(m.calls) != 4 {
		t.Fatalf("gated model calls = %d, want 4", len(m.calls))
	}
	if !m.callSees(3, "INPUT-2") {
		t.Errorf("request after the second approval does not contain INPUT-2")
	}
}

// approve (handoff) -> gated. The gated node's first call sees the reply as
// its input. After the confirmation it must still see that input.
func TestResumeRound3_HandoffIntoGatedNodeKeepsInput(t *testing.T) {
	m := &model3{}
	g := gated3(t, "executor", m)
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, newHitlAsker("approve", "ia", false), g))
	t1, _ := run3(t, r, userText("start"))
	_, name := findLongRunningInterrupt(t1)
	t2, err := run3(t, r, resumeContent("ia", name, "go-PAYLOAD"))
	if err != nil {
		t.Fatalf("answering approve: %v", err)
	}
	id, _ := findLongRunningInterrupt(t2)
	if _, err := run3(t, r, approve3(id)); err != nil {
		t.Fatalf("approval: %v", err)
	}
	if len(m.calls) != 2 {
		t.Fatalf("executor model calls = %d, want 2", len(m.calls))
	}
	if !m.callSees(0, "go-PAYLOAD") || !m.callSees(1, "go-PAYLOAD") {
		t.Errorf("go-PAYLOAD in first call = %v, in resume call = %v; want true, true", m.callSees(0, "go-PAYLOAD"), m.callSees(1, "go-PAYLOAD"))
	}
	if m.callSees(1, "adk_request_input") {
		t.Errorf("resume call contains the approve node's adk_request_input call")
	}
}

// A handoff asker with no successor. Once answered, a duplicate or a
// different reply must be rejected.
func TestResumeRound3_TerminalHandoffReplyRejected(t *testing.T) {
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, newHitlAsker("ask", "q", false)))
	t1, _ := run3(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	if _, err := run3(t, r, resumeContent(id, name, "ok")); err != nil {
		t.Fatalf("resume: %v", err)
	}
	_, dup := run3(t, r, resumeContent(id, name, "ok"))
	_, changed := run3(t, r, resumeContent(id, name, "changed"))
	if dup == nil || changed == nil {
		t.Errorf("duplicate reply err = %v, changed reply err = %v; want ErrNothingToResume for both", dup, changed)
	}
}
