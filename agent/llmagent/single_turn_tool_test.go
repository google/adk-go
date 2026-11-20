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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// singleTurnSession runs a workflow agent "wf" over one in-memory session.
type singleTurnSession struct {
	t      *testing.T
	runner *runner.Runner
}

func newSingleTurnSession(t *testing.T, node workflow.Node) *singleTurnSession {
	t.Helper()
	root, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, node)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return &singleTurnSession{t: t, runner: r}
}

// run sends msg and returns the turn's events; a run error fails the test.
func (s *singleTurnSession) run(msg *genai.Content, opts ...runner.RunOption) []*session.Event {
	s.t.Helper()
	var events []*session.Event
	for ev, err := range s.runner.Run(s.t.Context(), "u", "s", msg, agent.RunConfig{}, opts...) {
		if err != nil {
			s.t.Fatalf("Run() error = %v", err)
		}
		events = append(events, ev)
	}
	return events
}

func modelFC(id, name string, args map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args}}}}
}

// chatWithHelper is a chat agent "chat" answering from chatModel whose
// single_turn sub-agent "helper" answers from helperModel with tools.
func chatWithHelper(t *testing.T, chatModel, helperModel *testutil.MockModel, tools ...tool.Tool) agent.Agent {
	t.Helper()
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Description: "helps", Model: helperModel, Mode: llmagent.ModeSingleTurn, Tools: tools})
	if err != nil {
		t.Fatalf("llmagent.New(helper) error = %v", err)
	}
	chat, err := llmagent.New(llmagent.Config{Name: "chat", Model: chatModel, Mode: llmagent.ModeChat, SubAgents: []agent.Agent{helper}})
	if err != nil {
		t.Fatalf("llmagent.New(chat) error = %v", err)
	}
	return chat
}

func chatAgentNode(t *testing.T, a agent.Agent) workflow.Node {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	return n
}

// helperResponses returns the function responses of helper in events.
func helperResponses(events []*session.Event) []map[string]any {
	var out []map[string]any
	for _, ev := range events {
		for _, fr := range ev.FunctionResponses() {
			if fr.Name == "helper" {
				out = append(out, fr.Response)
			}
		}
	}
	return out
}

// Fails if a chat agent in a workflow's agent node, or in that node run as a
// dynamic child, cannot call its single_turn sub-agent as a tool, a second
// call returns the first call's answer without running the sub-agent, or the
// sub-agent's events lose their branch or path.
func TestSingleTurnTool_InWorkflowAgentNode(t *testing.T) {
	t.Run("two_calls_in_P1_and_P2", func(t *testing.T) {
		tests := []struct {
			name     string
			place    func(t *testing.T, a agent.Agent) workflow.Node
			wantPath string
		}{
			{name: "P1", place: chatAgentNode, wantPath: "wf@1/chat@1/helper@1"},
			{
				name: "P2",
				place: func(t *testing.T, a agent.Agent) workflow.Node {
					child := chatAgentNode(t, a)
					return workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
						return workflow.RunNode[any](ctx, child, nil)
					}, workflow.NodeConfig{})
				},
				wantPath: "wf@1/orch@1/chat@1/helper@1",
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				chatModel := &testutil.MockModel{Responses: []*genai.Content{
					modelFC("call_1", "helper", map[string]any{"request": "one"}),
					modelFC("call_2", "helper", map[string]any{"request": "two"}),
					genai.NewContentFromText("chat-done", genai.RoleModel),
				}}
				helperModel := &testutil.MockModel{Responses: []*genai.Content{
					genai.NewContentFromText("reply-1", genai.RoleModel),
					genai.NewContentFromText("reply-2", genai.RoleModel),
				}}
				s := newSingleTurnSession(t, tc.place(t, chatWithHelper(t, chatModel, helperModel)))

				events := s.run(genai.NewContentFromText("go", genai.RoleUser))

				if got := len(helperModel.Requests); got != 2 {
					t.Errorf("helper model requests = %d, want 2", got)
				}
				type where struct{ Branch, Path string }
				var got []where
				for _, ev := range events {
					if ev.Author == "helper" {
						var path string
						if ev.NodeInfo != nil {
							path = ev.NodeInfo.Path
						}
						got = append(got, where{Branch: ev.Branch, Path: path})
					}
				}
				want := []where{{Branch: "helper@call_1", Path: tc.wantPath}, {Branch: "helper@call_2", Path: tc.wantPath}}
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("events of helper mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]map[string]any{{"result": "reply-1"}, {"result": "reply-2"}}, helperResponses(events)); diff != "" {
					t.Errorf("function responses of helper mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})
}
