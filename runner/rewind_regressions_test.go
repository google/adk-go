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
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

type slidingEchoSummarizer struct {
	mu   sync.Mutex
	seen []string
}

func (s *slidingEchoSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (compaction.SummarizeResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var parts []string
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p != nil && p.Text != "" {
				parts = append(parts, p.Text)
			}
		}
	}
	joined := strings.Join(parts, " | ")
	s.seen = append(s.seen, joined)
	return compaction.SummarizeResult{
		Content: genai.NewContentFromText("SUMMARY: "+joined, genai.RoleModel),
	}, nil
}

func TestRewind_SlidingWindowCompactionSkipsRewoundTurns(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv1", "hello"),
		seedText("inv1", "agent", "hi"),
		seedUser("inv_to_rewind", "SECRET_REWOUND_CONTENT"),
		seedText("inv_to_rewind", "agent", "acknowledged"),
	)

	summarizer := &slidingEchoSummarizer{}
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromText("turn 2 reply", genai.RoleModel),
		genai.NewContentFromText("turn 3 reply", genai.RoleModel),
	}}
	root, err := llmagent.New(llmagent.Config{Name: "agent", Model: m})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          root,
		SessionService: svc,
		Compaction: &compaction.Config{
			CompactionInterval: 2,
			OverlapSize:        0,
			Summarizer:         summarizer,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Rewind(ctx, nodeTestUser, nodeTestSession, "inv_to_rewind", nil); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("next"), agent.RunConfig{})); len(errs) > 0 {
		t.Fatalf("turn after rewind errors = %v", errs)
	}
	if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("again"), agent.RunConfig{})); len(errs) > 0 {
		t.Fatalf("final turn errors = %v", errs)
	}

	for _, s := range summarizer.seen {
		if strings.Contains(s, "SECRET_REWOUND_CONTENT") {
			t.Errorf("summarizer was fed rewound content: %q", s)
		}
	}
}

func TestRewind_ResolvesInvocationFromLiveEventsOnly(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv1", "prompt 1"),
		seedLongRunningCall("inv1", "root_agent", "call-shared", "ask_human"),
		seedUser("inv2", "prompt 2"),
		seedLongRunningCall("inv2", "root_agent", "call-shared", "ask_human"),
	)
	var gotInvIDs []string
	type args struct{}
	ask, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	root := scriptedLlmAgent(t, "root_agent", scripted(modelText("done")), llmagent.Config{
		Tools: []tool.Tool{ask},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(c agent.Context) (*genai.Content, error) {
				gotInvIDs = append(gotInvIDs, c.InvocationID())
				return nil, nil
			},
		},
	})
	r := newResumeRunner(t, root, svc, true)
	if err := r.Rewind(ctx, nodeTestUser, nodeTestSession, "inv2", nil); err != nil {
		t.Fatalf("Rewind(inv2): %v", err)
	}

	runTurn(t, r, responseContent("call-shared", "ask_human", map[string]any{"ok": true}))
	if diff := cmp.Diff([]string{"inv1"}, gotInvIDs); diff != "" {
		t.Errorf("resolved invocation ID after rewinding inv2 (-want +got):\n%s", diff)
	}
}

func TestRewind_StaleApprovalDoesNotRunWorkflowHandler(t *testing.T) {
	for _, root := range []string{"workflowagent", "node"} {
		t.Run(root, func(t *testing.T) {
			ctx := t.Context()
			asker := newHitlAsker("asker", "", false)
			var handlerRuns atomic.Int32
			handler := workflow.NewFunctionNode("handler", func(_ agent.Context, input string) (string, error) {
				handlerRuns.Add(1)
				return "sent:" + input, nil
			}, workflow.NodeConfig{})
			edges := workflow.Chain(workflow.Start, asker, handler)

			svc := session.InMemoryService()
			newNodeTestSession(t, ctx, svc)
			cfg := runner.Config{AppName: nodeTestApp, SessionService: svc}
			if root == "workflowagent" {
				wfAgent, err := workflowagent.New(workflowagent.Config{Name: workflowAgentName, Edges: edges})
				if err != nil {
					t.Fatal(err)
				}
				cfg.Agent = wfAgent
			} else {
				wf, err := workflow.NewWorkflowNode(workflowAgentName, edges)
				if err != nil {
					t.Fatal(err)
				}
				cfg.Node = wf
			}
			r, err := runner.New(cfg)
			if err != nil {
				t.Fatal(err)
			}

			turn1 := drainRunner(t, r.Run(ctx, nodeTestUser, nodeTestSession, userText("draft"), agent.RunConfig{}))
			callID, callName := findLongRunningInterrupt(turn1)
			if callID == "" {
				t.Fatalf("no interrupt in turn 1")
			}
			inv1 := turn1[0].InvocationID
			if err := r.Rewind(ctx, nodeTestUser, nodeTestSession, inv1, nil); err != nil {
				t.Fatalf("Rewind: %v", err)
			}

			collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, resumeContent(callID, callName, "approve"), agent.RunConfig{}))
			if n := handlerRuns.Load(); n > 0 {
				t.Errorf("handler ran %d time(s) after its approval request was rewound", n)
			}
		})
	}
}

func TestResumableRunner_GraphWorkflowSubAgentRouting(t *testing.T) {
	b, d := joinResumeAgent(t, "b", "B", true), joinResumeAgent(t, "d", "D", false)
	bn, dn := joinResumeAgentNode(t, b), joinResumeAgentNode(t, d)
	wf, err := workflowagent.New(workflowagent.Config{
		Name:      "wf",
		SubAgents: []agent.Agent{b, d},
		Edges:     []workflow.Edge{{From: workflow.Start, To: bn}, {From: bn, To: dn}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "repro",
		Agent:             wf,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
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
		t.Error("resumable runner: resume did not reach downstream node d in WorkflowAgent")
	}
}

func TestNodePathResume_OnUserMessageCallbackKeepsRestoredUserContent(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	var mu sync.Mutex
	var userContents []string
	n := newFuncNode("node", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		mu.Lock()
		text := ""
		if uc := ctx.UserContent(); uc != nil && len(uc.Parts) > 0 {
			text = uc.Parts[0].Text
		}
		userContents = append(userContents, text)
		mu.Unlock()
		ev := session.NewEvent(ctx, ctx.InvocationID())
		if _, ok := ctx.ResumedInput("fc-1"); ok {
			ev.Output = "resumed"
			yield(ev, nil)
			return
		}
		ev.Content = fnCallContent("fc-1", "tool", map[string]any{})
		ev.LongRunningToolIDs = []string{"fc-1"}
		yield(ev, nil)
	})
	p, err := plugin.New(plugin.Config{
		Name: "passthrough_fr",
		OnUserMessageCallback: func(_ agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			if len(msg.Parts) > 0 && msg.Parts[0].FunctionResponse != nil {
				cp := *msg
				return &cp, nil
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := newNodeRootRunner(t, n, svc, p)

	turn1 := runTurn(t, r, userText("original text"))
	runTurn(t, r, responseContent("fc-1", "tool", map[string]any{"v": 1}), runner.WithInvocationID(turn1[0].InvocationID))

	if diff := cmp.Diff([]string{"original text", "original text"}, userContents); diff != "" {
		t.Errorf("node user contents (-want +got):\n%s", diff)
	}
}
