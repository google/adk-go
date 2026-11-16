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

package llmagent_test

import (
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// adk-python raises ValueError("Agent 'ghost' not found. ...") from
// LlmAgent.__get_agent_to_run (agents/llm_agent.py) before any model call.
// Fails when the resume lookup drops the error of a name the agent tree does
// not have: the coordinator then calls its model again.
func TestResume_TransferToAgentOutsideTreeFailsRun(t *testing.T) {
	for _, tc := range []struct {
		name  string
		after []string // authors of text events after the transfer
	}{
		{name: "transfer is the last event"},
		{name: "the removed agent ran after the transfer", after: []string{"ghost"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// shell -- coordinator -- helper. The paused invocation holds a
			// transfer of coordinator to "ghost", which the tree does not
			// have, as after a redeploy that removed that sub-agent.
			coordModel := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("coordinator done", genai.RoleModel)}}
			helper := newLLMAgent(t, llmagent.Config{Name: "helper", Model: &testutil.MockModel{}})
			coord := newLLMAgent(t, llmagent.Config{Name: "coordinator", Model: coordModel, SubAgents: []agent.Agent{helper}})
			shell, err := agent.New(agent.Config{
				Name:      "shell",
				SubAgents: []agent.Agent{coord},
				Run:       func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] { return coord.Run(ctx) },
			})
			if err != nil {
				t.Fatal(err)
			}
			svc := session.InMemoryService()
			created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
			if err != nil {
				t.Fatal(err)
			}
			newEvent := func(author string, content *genai.Content) *session.Event {
				ev := session.NewEvent(t.Context(), "inv-1")
				ev.Author = author
				ev.Content = content
				return ev
			}
			call := newEvent("coordinator", &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "t-1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "ghost"},
			}}}})
			resp := newEvent("coordinator", &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: "t-1", Name: "transfer_to_agent", Response: map[string]any{"result": nil},
			}}}})
			resp.Actions.TransferToAgent = "ghost"
			stored := []*session.Event{newEvent("user", genai.NewContentFromText("go", genai.RoleUser)), call, resp}
			for _, author := range tc.after {
				stored = append(stored, newEvent(author, genai.NewContentFromText("working", genai.RoleModel)))
			}
			for _, ev := range stored {
				if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
					t.Fatal(err)
				}
			}
			r, err := runner.New(runner.Config{AppName: "app", Agent: shell, SessionService: svc, Resumable: true})
			if err != nil {
				t.Fatal(err)
			}

			_, err = runUntilError(r.Run(t.Context(), "user", "s", nil, agent.RunConfig{}, runner.WithInvocationID("inv-1")))

			if err == nil || err.Error() != "failed to find agent: ghost" {
				t.Errorf("run error = %v, want %q", err, "failed to find agent: ghost")
			}
			if got := len(coordModel.Requests); got != 0 {
				t.Errorf("coordinator model calls = %d, want 0", got)
			}
		})
	}
}

// runUntilError returns the events stream yields before its first error, and
// that error.
func runUntilError(stream iter.Seq2[*session.Event, error]) ([]*session.Event, error) {
	var events []*session.Event
	for ev, err := range stream {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}
