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
	"iter"
	"reflect"
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
				if _, err := workflow.ResumeOrRequestInput(agent.Promote(ctx), func(ev *session.Event) error { yield(ev, nil); return nil }, session.RequestInput{InterruptID: name, Message: "approve"}); err != nil {
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
				t.Fatalf("join output = %v", got)
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
