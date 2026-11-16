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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
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
		{
			name: "loop",
			newRoot: func(caller, other agent.Agent) (agent.Agent, error) {
				return loopagent.New(loopagent.Config{
					MaxIterations: 1,
					AgentConfig:   agent.Config{Name: "root", SubAgents: []agent.Agent{other, caller}},
				})
			},
			wantBranch:     "",
			wantOtherCalls: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "approve", IsLongRunning: true},
				func(agent.ToolContext, struct{}) (map[string]any, error) {
					return map[string]any{"status": "pending"}, nil
				})
			if err != nil {
				t.Fatal(err)
			}
			callerModel := &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "approve", Args: map[string]any{}}}}},
				genai.NewContentFromText("pending", genai.RoleModel),
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
			if n := len(callerModel.Requests); n != 3 {
				t.Fatalf("caller called the model %d times, want 3", n)
			}
			if !hasFunctionResponse(callerModel.Requests[2].Contents, "call-1") {
				t.Errorf("caller's third model request has no function response for call-1")
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

func TestRun_FunctionResponsePreservesBranchAcrossOnUserMessageCallback(t *testing.T) {
	ctx := t.Context()
	approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "approve", IsLongRunning: true},
		func(agent.ToolContext, struct{}) (map[string]any, error) {
			return map[string]any{"status": "pending"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	callerModel := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "approve", Args: map[string]any{}}}}},
		genai.NewContentFromText("pending", genai.RoleModel),
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
	root, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{caller, other}}})
	if err != nil {
		t.Fatal(err)
	}
	p, err := plugin.New(plugin.Config{
		Name: "pass-through",
		OnUserMessageCallback: func(_ agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			return msg, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          root,
		SessionService: sessionService,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
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
		t.Fatalf("resume yielded %d events, want 1", len(got))
	}
	if got[0].Author != "caller" || got[0].Branch != "root.caller" {
		t.Errorf("resume event author=%q branch=%q, want author=%q branch=%q", got[0].Author, got[0].Branch, "caller", "root.caller")
	}
}
