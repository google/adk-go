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
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// joinedRun records how a node-path run saw its turn.
type joinedRun struct {
	invocationID, userContent, input string
}

// joinRecorder records runs. Nodes and callbacks run on other goroutines, so
// it is guarded by a mutex.
type joinRecorder struct {
	mu   sync.Mutex
	runs []joinedRun
}

func (r *joinRecorder) record(ctx agent.InvocationContext, input any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	// A node in a graph workflow receives Start's input as text.
	in, isText := input.(string)
	if !isText {
		c, _ := input.(*genai.Content)
		in = joinText(c)
	}
	r.runs = append(r.runs, joinedRun{invocationID: ctx.InvocationID(), userContent: joinText(ctx.UserContent()), input: in})
}

func (r *joinRecorder) recorded() []joinedRun {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.runs
}

// joinText returns the first part of c as text: its text, or "fr:<id>"
// for a function response; "" for nil.
func joinText(c *genai.Content) string {
	if c == nil || len(c.Parts) == 0 {
		return ""
	}
	if fr := c.Parts[0].FunctionResponse; fr != nil {
		return "fr:" + fr.ID
	}
	return c.Parts[0].Text
}

// seedOpenTask stores the session of a paused task delegation in inv-1: the
// user asked, the coordinator called task_agent (fc-1), and the task agent
// asked back in isolation scope fc-1. It also stores before it a turn of
// inv-old, whose call fc-old is unanswered, and a user turn of inv-caller.
func seedOpenTask(t *testing.T, ctx context.Context, svc session.Service) {
	t.Helper()
	newNodeTestSession(t, ctx, svc)
	ev := func(author, invocationID, scope string, part *genai.Part) *session.Event {
		role := genai.RoleModel
		if author == "user" {
			role = genai.RoleUser
		}
		return &session.Event{
			Author:         author,
			InvocationID:   invocationID,
			IsolationScope: scope,
			LLMResponse:    model.LLMResponse{Content: &genai.Content{Role: role, Parts: []*genai.Part{part}}},
		}
	}
	for _, e := range []*session.Event{
		ev("user", "inv-old", "", &genai.Part{Text: "old start"}),
		ev("coordinator", "inv-old", "", &genai.Part{FunctionCall: &genai.FunctionCall{ID: "fc-old", Name: "ask"}}),
		ev("user", "inv-caller", "", &genai.Part{Text: "caller start"}),
		ev("user", "inv-1", "", &genai.Part{Text: "book a flight"}),
		ev("coordinator", "inv-1", "", &genai.Part{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "task_agent"}}),
		ev("task_agent", "inv-1", "fc-1", &genai.Part{Text: "which city?"}),
	} {
		appendSeedEvent(t, ctx, svc, e)
	}
}

// lastStoredUserEvent returns the newest user event of the node test session.
func lastStoredUserEvent(t *testing.T, ctx context.Context, svc session.Service) *session.Event {
	t.Helper()
	var last *session.Event
	for ev := range storedNodeTestSession(t, ctx, svc).Events().All() {
		if ev.Author == "user" {
			last = ev
		}
	}
	return last
}

func answer(id string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "ask", Response: map[string]any{"ok": true}}}}}
}

// beforeRunRecorder returns a plugin that records each run's invocation and
// user content in rec.
func beforeRunRecorder(t *testing.T, rec *joinRecorder) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "before_run_recorder",
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			rec.record(ctx, nil)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// TestRunner_PausedTaskJoin_NodePathBorrowsAndClassicPathDoesNot pins
// adk-python's task join (workflow/_node_runner_utils.py, run_node_async):
// on the node path a message that resolves to no invocation, sent while a
// task delegation is open, runs under the delegation's invocation as a new
// user turn with its own content. The classic path scopes the message to the
// task but keeps a new invocation (runners.py _append_user_event).
func TestRunner_PausedTaskJoin_NodePathBorrowsAndClassicPathDoesNot(t *testing.T) {
	t.Run("llm_agent_root", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seedOpenTask(t, ctx, svc)
		rec := &joinRecorder{}
		root, err := llmagent.New(llmagent.Config{
			Name:      "coordinator",
			Model:     &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}},
			SubAgents: []agent.Agent{textReplyLlmAgent(t, "task_agent", "ok")},
		})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		r := newNodeTestRunnerWithPlugins(t, root, svc, beforeRunRecorder(t, rec))

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("Paris"), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		if got, want := rec.recorded(), []joinedRun{{invocationID: "inv-1", userContent: "Paris"}}; len(got) != 1 || got[0] != want[0] {
			t.Errorf("before_run saw %+v, want %+v", got, want)
		}
		got := lastStoredUserEvent(t, ctx, svc)
		if got.InvocationID != "inv-1" || got.IsolationScope != "fc-1" || joinText(got.Content) != "Paris" {
			t.Errorf("stored user event = (%q, %q, %q), want (inv-1, fc-1, Paris)", got.InvocationID, got.IsolationScope, joinText(got.Content))
		}
	})

	// adk-python runs a Workflow root as a node (runners.py, Runner.run_async).
	t.Run("graph_workflow_root", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seedOpenTask(t, ctx, svc)
		rec := &joinRecorder{}
		n := newFuncNode("quiet", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
			rec.record(ctx, input)
		})
		wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, n)})
		if err != nil {
			t.Fatalf("workflowagent.New() error = %v", err)
		}
		r := newNodeTestRunner(t, wf, svc)

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("London"), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		if got, want := rec.recorded(), []joinedRun{{invocationID: "inv-1", userContent: "London", input: "London"}}; len(got) != 1 || got[0] != want[0] {
			t.Errorf("node saw %+v, want %+v", got, want)
		}
		got := lastStoredUserEvent(t, ctx, svc)
		if got.InvocationID != "inv-1" || got.IsolationScope != "fc-1" || joinText(got.Content) != "London" {
			t.Errorf("stored user event = (%q, %q, %q), want (inv-1, fc-1, London)", got.InvocationID, got.IsolationScope, joinText(got.Content))
		}
	})

	t.Run("classic_root", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seedOpenTask(t, ctx, svc)
		r := newNodeTestRunner(t, textReplyAgent(t, "custom_root", "ok"), svc)

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, userText("Paris"), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		got := lastStoredUserEvent(t, ctx, svc)
		if got.InvocationID == "inv-1" || got.InvocationID == "" || got.IsolationScope != "fc-1" || joinText(got.Content) != "Paris" {
			t.Errorf("stored user event = (%q, %q, %q), want (a new invocation, fc-1, Paris)", got.InvocationID, got.IsolationScope, joinText(got.Content))
		}
	})
}

// TestRunner_NodePath_TaskJoinGuards pins when a node-path run joins an open
// task delegation and what a joined turn keeps (adk-python
// workflow/_node_runner_utils.py, run_node_async): only a message that
// resolves to no invocation, with no caller ID and no answered function call,
// joins, and a joined turn keeps the borrowed invocation and its own content,
// also when an on-user-message callback replaces it with a function response.
func TestRunner_NodePath_TaskJoinGuards(t *testing.T) {
	idLess := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "t", Response: map[string]any{}}}}}

	tests := []struct {
		name    string
		msg     *genai.Content
		opts    []runner.RunOption
		plugins func(t *testing.T) []*plugin.Plugin
		want    joinedRun
		// wantStored is the newest stored user event: invocation, scope,
		// content.
		wantStored [3]string
	}{
		{
			name:       "plain_message_joins",
			msg:        userText("London"),
			want:       joinedRun{invocationID: "inv-1", userContent: "London", input: "London"},
			wantStored: [3]string{"inv-1", "fc-1", "London"},
		},
		{
			name:       "response_without_id_joins",
			msg:        idLess,
			want:       joinedRun{invocationID: "inv-1", userContent: "fr:", input: "fr:"},
			wantStored: [3]string{"inv-1", "fc-1", "fr:"},
		},
		{
			name: "caller_id_wins",
			msg:  userText("retry text"),
			opts: []runner.RunOption{runner.WithInvocationID("inv-caller")},
			want: joinedRun{invocationID: "inv-caller", userContent: "caller start", input: "caller start"},
			// A retry: nothing is stored.
			wantStored: [3]string{"inv-1", "", "book a flight"},
		},
		{
			name:       "resolved_message_does_not_join",
			msg:        answer("fc-old"),
			want:       joinedRun{invocationID: "inv-old", userContent: "old start", input: "old start"},
			wantStored: [3]string{"inv-old", "fc-1", "fr:fc-old"},
		},
		{
			name: "replacement_keeps_borrowed_id_and_content",
			msg:  userText("London"),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replaceMessagePlugin(t, answer("fc-old"))}
			},
			want:       joinedRun{invocationID: "inv-1", userContent: "fr:fc-old", input: "fr:fc-old"},
			wantStored: [3]string{"inv-1", "fc-1", "fr:fc-old"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc := session.InMemoryService()
			seedOpenTask(t, ctx, svc)
			rec := &joinRecorder{}
			n := newFuncNode("quiet", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
				rec.record(ctx, input)
			})
			var plugins []*plugin.Plugin
			if tc.plugins != nil {
				plugins = tc.plugins(t)
			}
			r := newNodeRootRunner(t, n, svc, plugins...)

			_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, tc.msg, agent.RunConfig{}, tc.opts...))

			if len(errs) > 0 {
				t.Fatalf("Run() errors = %v", errs)
			}
			if got := rec.recorded(); len(got) != 1 || got[0] != tc.want {
				t.Errorf("node saw %+v, want [%+v]", got, tc.want)
			}
			stored := lastStoredUserEvent(t, ctx, svc)
			if got := [3]string{stored.InvocationID, stored.IsolationScope, joinText(stored.Content)}; got != tc.wantStored {
				t.Errorf("newest stored user event = %q, want %q", got, tc.wantStored)
			}
		})
	}

	// adk-python never reaches the join with a response it cannot attribute;
	// lenient resolution gives it a new invocation, as before the join.
	t.Run("unmatched_answer_does_not_join", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seedOpenTask(t, ctx, svc)
		rec := &joinRecorder{}
		n := newFuncNode("quiet", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
			rec.record(ctx, input)
		})
		r := newNodeRootRunner(t, n, svc)

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, answer("fc-none"), agent.RunConfig{}))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		got := rec.recorded()
		if len(got) != 1 || got[0].invocationID == "inv-1" || got[0].invocationID == "" || got[0].userContent != "fr:fc-none" {
			t.Fatalf("node saw %+v, want one run under a new invocation with user content fr:fc-none", got)
		}
		stored := lastStoredUserEvent(t, ctx, svc)
		if stored.InvocationID != got[0].invocationID || joinText(stored.Content) != "fr:fc-none" {
			t.Errorf("newest stored user event = (%q, %q), want (%q, fr:fc-none)", stored.InvocationID, joinText(stored.Content), got[0].invocationID)
		}
	})

	t.Run("message_without_content_does_not_join", func(t *testing.T) {
		ctx := t.Context()
		svc := session.InMemoryService()
		seedOpenTask(t, ctx, svc)
		rec := &joinRecorder{}
		n := newFuncNode("quiet", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
			rec.record(ctx, input)
		})
		r := newNodeRootRunner(t, n, svc)

		_, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, nil, agent.RunConfig{}, runner.WithStateDelta(map[string]any{"k": "v"})))

		if len(errs) > 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		stored := lastStoredUserEvent(t, ctx, svc)
		if stored.InvocationID == "inv-1" || stored.Actions.StateDelta["k"] != "v" {
			t.Errorf("state-delta event = (%q, %v), want a new invocation with k=v", stored.InvocationID, stored.Actions.StateDelta)
		}
	})
}

// TestRunner_NodePath_ResumeAfterJoinedTurnRestoresFirstUserMessage pins a
// ported adk-python flaw: once a turn joined a paused task's invocation, a
// later resume of that invocation runs on the invocation's first user
// message, not on the joined turn (runners.py
// _find_user_message_for_invocation).
func TestRunner_NodePath_ResumeAfterJoinedTurnRestoresFirstUserMessage(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	seedOpenTask(t, ctx, svc)
	rec := &joinRecorder{}
	n := newFuncNode("approver", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
		rec.record(ctx, input)
		if _, ok := ctx.ResumedInput("fc-2"); ok {
			return
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fc-2", Name: "ask"}}}}
		ev.LongRunningToolIDs = []string{"fc-2"}
		yield(ev, nil)
	})
	r := newNodeRootRunner(t, n, svc)

	for i, msg := range []*genai.Content{userText("London"), answer("fc-2")} {
		if _, errs := collectRun(r.Run(ctx, nodeTestUser, nodeTestSession, msg, agent.RunConfig{})); len(errs) > 0 {
			t.Fatalf("turn %d Run() errors = %v", i+1, errs)
		}
	}

	want := []joinedRun{
		{invocationID: "inv-1", userContent: "London", input: "London"},
		{invocationID: "inv-1", userContent: "book a flight", input: "book a flight"},
	}
	got := rec.recorded()
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("approver saw %+v, want %+v", got, want)
	}
}
