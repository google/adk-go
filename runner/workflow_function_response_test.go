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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
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
// workflow, when the workflow is the root or the only child, so Workflow.Resume
// runs the downstream nodes. Running the node agent on its own would ask for
// input again.
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
