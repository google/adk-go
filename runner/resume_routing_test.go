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

package runner

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// routedRun records where an agent ran: its invocation and branch.
type routedRun struct {
	invocationID, branch string
}

// routingRecorder returns an agent named name, with subAgents, that records
// each run in runs and answers with one text event.
func routingRecorder(name string, runs *[]routedRun, subAgents ...agent.Agent) agent.Agent {
	return must(agent.New(agent.Config{
		Name:      name,
		SubAgents: subAgents,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				*runs = append(*runs, routedRun{invocationID: ctx.InvocationID(), branch: ctx.Branch()})
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Branch = ctx.Branch()
				ev.Content = genai.NewContentFromText("ok", genai.RoleModel)
				yield(ev, nil)
			}
		},
	}))
}

// routingEvent returns a seeded event of author on branch in invocationID
// holding part.
func routingEvent(author, branch, invocationID string, part *genai.Part) *session.Event {
	role := genai.RoleModel
	if author == "user" {
		role = genai.RoleUser
	}
	return &session.Event{
		Author:       author,
		Branch:       branch,
		InvocationID: invocationID,
		LLMResponse:  model.LLMResponse{Content: &genai.Content{Role: role, Parts: []*genai.Part{part}}},
	}
}

func routingCall(author, branch, invocationID, id string) *session.Event {
	return routingEvent(author, branch, invocationID, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "tool"}})
}

func routingText(author, branch, invocationID, text string) *session.Event {
	return routingEvent(author, branch, invocationID, &genai.Part{Text: text})
}

// routingAnswers returns a user message answering the calls ids.
func routingAnswers(ids ...string) *genai.Content {
	msg := &genai.Content{Role: genai.RoleUser}
	for _, id := range ids {
		msg.Parts = append(msg.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "tool", Response: map[string]any{"ok": true}}})
	}
	return msg
}

// runRouting seeds a session of svc with events, runs r on it with msg and
// opts, and fails the test on a run error.
func runRouting(t *testing.T, r *Runner, svc session.Service, events []*session.Event, msg *genai.Content, opts ...RunOption) {
	t.Helper()
	createdSession(t, svc, "s1", events)
	for _, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}, opts...) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}
}

// createdSession creates session id for app "app" and user "u1" in svc and
// appends events to it.
func createdSession(t *testing.T, svc session.Service, id string, events []*session.Event) {
	t.Helper()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: id})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	for i, ev := range events {
		if err := svc.AppendEvent(t.Context(), resp.Session, ev); err != nil {
			t.Fatalf("AppendEvent[%d]: %v", i, err)
		}
	}
}

// TestRunner_ClassicPath_FunctionResponseRestoresSubAgentBranch pins where a
// classic-path runner resumes the sub-agent whose call a function response
// answers: on the branch adk-python's restore_branch_from_history gives,
// searched in the invocation the responses resolve to leniently, or in the
// whole session when they resolve to none (runners.py
// _setup_context_for_new_invocation).
func TestRunner_ClassicPath_FunctionResponseRestoresSubAgentBranch(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
		msg    *genai.Content
		want   routedRun
	}{
		{
			// The responses span two invocations, so the run gets a new
			// invocation and the sub-agent's newest branch, not the branch
			// of the call the first response answers.
			name: "responses_of_two_invocations",
			events: []*session.Event{
				routingCall("sub", "root.sub@1", "inv_1", "fc_1"),
				routingCall("sub", "root.sub@2", "inv_2", "fc_2"),
			},
			msg:  routingAnswers("fc_1", "fc_2"),
			want: routedRun{branch: "root.sub@2"},
		},
		{
			// One invocation: its tool message branch and the later
			// invocation's branch are both passed over.
			name: "responses_of_one_invocation",
			events: []*session.Event{
				routingCall("sub", "root.sub@1", "inv_1", "fc_1"),
				routingText("sub", "helper_tool@fc_1", "inv_1", "tool msg"),
				routingText("sub", "root.sub@2", "inv_2", "later"),
			},
			msg:  routingAnswers("fc_1"),
			want: routedRun{invocationID: "inv_1", branch: "root.sub@1"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var runs []routedRun
			root := must(agent.New(agent.Config{Name: "root", SubAgents: []agent.Agent{routingRecorder("sub", &runs)}}))
			svc := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: root, SessionService: svc})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			runRouting(t, r, svc, tc.events, tc.msg)

			if len(runs) != 1 {
				t.Fatalf("sub ran %d times, want 1", len(runs))
			}
			got := runs[0]
			if tc.want.invocationID == "" {
				if got.invocationID == "inv_1" || got.invocationID == "inv_2" {
					t.Errorf("sub ran in invocation %q, want a new one", got.invocationID)
				}
				got.invocationID = ""
			}
			if got != tc.want {
				t.Errorf("sub ran as %+v, want %+v", got, tc.want)
			}
		})
	}
}

// TestRunner_ClassicPath_ResponseToRewoundCallRunsRoot pins that the classic
// path routes a function response by the live events only: when a rewind
// event removes the invocation of the sub-agent's call from them, the root
// runs (adk-python agents/_agent_router.py find_agent_to_run, which reads
// _apply_rewinds).
func TestRunner_ClassicPath_ResponseToRewoundCallRunsRoot(t *testing.T) {
	t.Parallel()
	var rootRuns, subRuns []routedRun
	root := routingRecorder("root", &rootRuns, routingRecorder("sub", &subRuns))
	svc := session.InMemoryService()
	r, err := New(Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rewind := routingText("user", "", "inv_2", "rewind")
	rewind.Actions.RewindBeforeInvocationID = "inv_1"

	runRouting(t, r, svc, []*session.Event{
		routingText("user", "", "inv_1", "start"),
		routingCall("sub", "root.sub@1", "inv_1", "fc_1"),
		rewind,
	}, routingAnswers("fc_1"))

	if len(subRuns) != 0 || len(rootRuns) != 1 {
		t.Errorf("sub ran %d times and root %d times, want 0 and 1", len(subRuns), len(rootRuns))
	}
}

// lastUserEvent returns the newest user event of session "s1" in svc.
func lastUserEvent(t *testing.T, svc session.Service) *session.Event {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	var last *session.Event
	for ev := range resp.Session.Events().All() {
		if ev.Author == "user" {
			last = ev
		}
	}
	if last == nil {
		t.Fatal("no stored user event")
	}
	return last
}

// TestRunner_AppendUserEvent_StampsBranchAndScopeOfAnsweredCall pins
// adk-python's _stamp_event_branch_context (runners.py): the stored user
// event answering a call takes that call's branch, and its isolation scope
// when it has none, where the call is the one the first function response
// answers among the events of the run's invocation.
func TestRunner_AppendUserEvent_StampsBranchAndScopeOfAnsweredCall(t *testing.T) {
	t.Parallel()

	scopedCall := func(scope string) *session.Event {
		ev := routingCall("sub", "root.sub", "inv_1", "fc_1")
		ev.IsolationScope = scope
		return ev
	}
	scopedText := func(scope string) *session.Event {
		ev := routingText("task_agent", "", "inv_1", "q")
		ev.IsolationScope = scope
		return ev
	}
	closed := func(scope string) *session.Event {
		ev := routingEvent("user", "", "inv_1", &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID: "ft", Name: "finish_task", Response: map[string]any{"result": "Task completed."},
		}})
		ev.IsolationScope = scope
		return ev
	}
	firstWithoutID := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{Name: "tool", Response: map[string]any{}}},
		{FunctionResponse: &genai.FunctionResponse{ID: "fc_2", Name: "tool", Response: map[string]any{}}},
	}}

	tests := []struct {
		name       string
		events     []*session.Event
		msg        *genai.Content
		opts       []RunOption
		wantBranch string
		wantScope  string
	}{
		{
			name:       "branch_and_scope_of_the_call",
			events:     []*session.Event{scopedCall("call-scope"), closed("call-scope")},
			msg:        routingAnswers("fc_1"),
			wantBranch: "root.sub",
			wantScope:  "call-scope",
		},
		{
			name:       "open_task_scope_kept",
			events:     []*session.Event{scopedCall("other-scope"), scopedText("active-scope")},
			msg:        routingAnswers("fc_1"),
			wantBranch: "root.sub",
			wantScope:  "active-scope",
		},
		{
			// The responses span two invocations, so the run has a new
			// one, which holds neither call (a ported adk-python flaw).
			name: "call_of_another_invocation",
			events: []*session.Event{
				routingCall("sub", "root.sub", "inv_1", "fc_1"),
				routingCall("sub", "root.sub", "inv_2", "fc_2"),
			},
			msg: routingAnswers("fc_1", "fc_2"),
		},
		{
			// A call ID reused within the invocation is answered by the
			// newest call, as adk-python's reversed search finds it.
			name: "reused_call_id_takes_newest_call",
			events: []*session.Event{
				routingCall("sub", "root.old", "inv_1", "fc_1"),
				routingCall("sub", "root.sub", "inv_1", "fc_1"),
			},
			msg:        routingAnswers("fc_1"),
			wantBranch: "root.sub",
		},
		{
			// A custom agent can store a call without an ID. A response
			// without an ID still matches no call, as adk-python's
			// find_matching_function_call returns none for it.
			name:   "response_without_id_matches_no_call",
			events: []*session.Event{routingCall("sub", "root.sub", "inv_1", "")},
			msg:    routingAnswers(""),
			opts:   []RunOption{WithInvocationID("inv_1")},
		},
		{
			name:   "first_response_without_id",
			events: []*session.Event{routingCall("sub", "root.sub", "inv_1", "fc_2")},
			msg:    firstWithoutID,
			opts:   []RunOption{WithInvocationID("inv_1")},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var runs []routedRun
			root := routingRecorder("root", &runs, routingRecorder("sub", &runs))
			svc := session.InMemoryService()
			r, err := New(Config{AppName: "app", Agent: root, SessionService: svc})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			runRouting(t, r, svc, tc.events, tc.msg, tc.opts...)

			got := lastUserEvent(t, svc)
			if got.Branch != tc.wantBranch || got.IsolationScope != tc.wantScope {
				t.Errorf("stored user event has branch %q and scope %q, want %q and %q", got.Branch, got.IsolationScope, tc.wantBranch, tc.wantScope)
			}
		})
	}
}

// routingModel answers every request with reply.
type routingModel struct{ reply string }

func (routingModel) Name() string { return "routing" }

func (m routingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(m.reply, genai.RoleModel)}, nil)
	}
}

// llmRoutingRunner returns a runner on svc over the agent tree of
// adk-python's _make_agent_tree (tests/unittests/agents/test_agent_router.py):
// root_agent, answering "root reply", with sub_agent1, answering "sub1 reply",
// and non_transferable, answering "non_transferable reply", which disallows
// transfer to its parent.
func llmRoutingRunner(t *testing.T, svc session.Service, resumable bool, plugins ...*plugin.Plugin) *Runner {
	t.Helper()
	sub1 := must(llmagent.New(llmagent.Config{Name: "sub_agent1", Model: routingModel{reply: "sub1 reply"}}))
	nonTransferable := must(llmagent.New(llmagent.Config{
		Name:                     "non_transferable",
		Model:                    routingModel{reply: "non_transferable reply"},
		DisallowTransferToParent: true,
	}))
	root := must(llmagent.New(llmagent.Config{
		Name:      "root_agent",
		Model:     routingModel{reply: "root reply"},
		SubAgents: []agent.Agent{sub1, nonTransferable},
	}))
	r, err := New(Config{
		AppName:        "app",
		Agent:          root,
		SessionService: svc,
		Resumable:      resumable,
		PluginConfig:   PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return r
}

// routedEvents seeds session "s1" of svc with events, runs r on it with msg,
// and returns the events r yields that hold text.
func routedEvents(t *testing.T, r *Runner, svc session.Service, events []*session.Event, msg *genai.Content) []*session.Event {
	t.Helper()
	createdSession(t, svc, "s1", events)
	var got []*session.Event
	for ev, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev != nil && ev.Content != nil && slices.ContainsFunc(ev.Content.Parts, func(p *genai.Part) bool { return p.Text != "" }) {
			got = append(got, ev)
		}
	}
	return got
}

// routedReplies is routedEvents as "<author>: <text>" lines.
func routedReplies(t *testing.T, r *Runner, svc session.Service, events []*session.Event, msg *genai.Content) []string {
	t.Helper()
	var got []string
	for _, ev := range routedEvents(t, r, svc, events, msg) {
		for _, p := range ev.Content.Parts {
			if p.Text != "" {
				got = append(got, ev.Author+": "+p.Text)
			}
		}
	}
	return got
}

// routingNamedCall returns a seeded event of author on branch in
// invocationID holding the call id of the function name.
func routingNamedCall(author, branch, invocationID, id, name string) *session.Event {
	return routingEvent(author, branch, invocationID, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: name}})
}

// routingPause is routingNamedCall with id in LongRunningToolIDs.
func routingPause(author, branch, invocationID, id, name string) *session.Event {
	ev := routingNamedCall(author, branch, invocationID, id, name)
	ev.LongRunningToolIDs = []string{id}
	return ev
}

// routingSeal returns an event of author in invocationID that seals an
// aborted invocation, answering the calls ids, or with no content when ids
// is empty (adk-python events/_abort_events.py).
func routingSeal(author, branch, invocationID string, ids ...string) *session.Event {
	ev := &session.Event{Author: author, Branch: branch, InvocationID: invocationID}
	ev.ErrorCode = "INVOCATION_ABORTED"
	ev.ErrorMessage = "Invocation was aborted by client."
	if len(ids) > 0 {
		ev.Content = routingAnswers(ids...)
	}
	return ev
}

// TestRunner_Run_FunctionCallByUserAuthorFallsBackToScan pins that a call
// whose author is not an agent of the tree does not take the answer: the
// scan picks the agent (adk-python agents/_agent_router.py:186, :193-205). A
// call event with no author is not resolved through its branch either.
func TestRunner_Run_FunctionCallByUserAuthorFallsBackToScan(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		events []*session.Event
		want   []string
	}{
		{
			name:   "falls_back_to_root",
			events: []*session.Event{routingCall("user", "", "inv_1", "c1")},
			want:   []string{"root_agent: root reply"},
		},
		{
			name: "falls_back_to_transferable_sub_agent",
			events: []*session.Event{
				routingText("sub_agent1", "", "inv_1", "earlier"),
				routingCall("user", "", "inv_1", "c1"),
			},
			want: []string{"sub_agent1: sub1 reply"},
		},
		{
			// The call's branch names non_transferable's call fc_tool, which
			// would route to it if the event's empty author were resolved.
			name: "empty_call_author_falls_back_to_root",
			events: []*session.Event{
				routingNamedCall("non_transferable", "", "inv_1", "fc_tool", "sub_workflow_tool"),
				routingCall("", "sub_workflow_tool@fc_tool.inner_input_node@1", "inv_1", "c1"),
			},
			want: []string{"root_agent: root reply"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			svc := session.InMemoryService()
			r := llmRoutingRunner(t, svc, true)

			got := routedReplies(t, r, svc, tc.events, routingAnswers("c1"))

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunner_Run_AbortGuardsSkipRootSealAndUnsealedCallAfterSeal pins the
// abort guards on the node path: the scan passes over a seal, even a
// content-less one by the root (adk-python test_agent_router.py:558), and an
// answer is not routed to its call's author when the last event that is not
// the user's is a seal, even when that seal did not answer the call (a
// ported adk-python flaw, agents/_agent_router.py:177-181, runners.py:178-179).
func TestRunner_Run_AbortGuardsSkipRootSealAndUnsealedCallAfterSeal(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		events []*session.Event
		msg    *genai.Content
		want   []string
	}{
		{
			name: "root_authored_seal_skipped_in_scan",
			events: []*session.Event{
				routingText("sub_agent1", "", "inv_1", "earlier"),
				routingSeal("root_agent", "", "inv_1"),
			},
			msg:  genai.NewContentFromText("next", genai.RoleUser),
			want: []string{"sub_agent1: sub1 reply"},
		},
		{
			name: "last_non_user_seal_hides_earlier_open_call",
			events: []*session.Event{
				routingCall("non_transferable", "", "inv_1", "c1"),
				routingText("root_agent", "", "inv_2", "root earlier"),
				routingSeal("root_agent", "", "inv_2"),
			},
			msg:  routingAnswers("c1"),
			want: []string{"root_agent: root reply"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			svc := session.InMemoryService()
			r := llmRoutingRunner(t, svc, true)

			got := routedReplies(t, r, svc, tc.events, tc.msg)

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// replacingPlugin returns a plugin whose OnUserMessageCallback replaces every
// user message with replacement.
func replacingPlugin(t *testing.T, replacement *genai.Content) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "replace_message",
		OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
			return replacement, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New: %v", err)
	}
	return p
}

// classicRoutingRunner returns a classic-path runner on svc whose root
// agent "root" has the sub-agent "sub"; both record their runs.
func classicRoutingRunner(t *testing.T, svc session.Service, rootRuns, subRuns *[]routedRun, plugins ...*plugin.Plugin) *Runner {
	t.Helper()
	root := routingRecorder("root", rootRuns, routingRecorder("sub", subRuns))
	r, err := New(Config{AppName: "app", Agent: root, SessionService: svc, PluginConfig: PluginConfig{Plugins: plugins}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return r
}

// TestRunner_ClassicPath_AbortSealSkipsAnsweredCall pins the abort guards on
// the classic path: an answer to a call that a seal closed runs the root,
// whether it is the run's message, left out of the routing view because the
// last stored event is the seal (adk-python runners.py:178-179), or a
// plugin's replacement, stored after the seal (agents/_agent_router.py:178-181).
func TestRunner_ClassicPath_AbortSealSkipsAnsweredCall(t *testing.T) {
	t.Parallel()
	events := func() []*session.Event {
		return []*session.Event{
			routingCall("sub", "root.sub@1", "inv_1", "fc_1"),
			routingSeal("sub", "root.sub@1", "inv_1", "fc_1"),
		}
	}
	tests := []struct {
		name    string
		msg     *genai.Content
		plugins func(t *testing.T) []*plugin.Plugin
	}{
		{
			name:    "direct_function_response_after_seal",
			msg:     routingAnswers("fc_1"),
			plugins: func(*testing.T) []*plugin.Plugin { return nil },
		},
		{
			name: "plugin_replaced_function_response_after_seal",
			msg:  genai.NewContentFromText("approve", genai.RoleUser),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replacingPlugin(t, routingAnswers("fc_1"))}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var rootRuns, subRuns []routedRun
			svc := session.InMemoryService()
			r := classicRoutingRunner(t, svc, &rootRuns, &subRuns, tc.plugins(t)...)

			runRouting(t, r, svc, events(), tc.msg)

			if diff := cmp.Diff([]routedRun{{invocationID: "inv_1", branch: ""}}, rootRuns, cmp.AllowUnexported(routedRun{})); diff != "" {
				t.Errorf("root runs mismatch (-want +got):\n%s", diff)
			}
			if len(subRuns) != 0 {
				t.Errorf("sub ran as %+v, want no run", subRuns)
			}
		})
	}
}

// TestRunner_Run_ToolSubBranchPause_NestedAndAlreadyAnsweredInterrupts pins
// how the node path routes an answer to a pause raised inside a tool, on
// resumable and non-resumable runners: through the open interrupts of the
// events before the answer, from the innermost tool caller at the time of
// the pause outwards, then the call's author, then an agent that made a call
// whose ID is a non-numeric run ID of the call event's branch (adk-python
// agents/_agent_router.py:46-91, events/_interrupts.py:58-127).
func TestRunner_Run_ToolSubBranchPause_NestedAndAlreadyAnsweredInterrupts(t *testing.T) {
	t.Parallel()
	const (
		nonTransferable = "non_transferable: non_transferable reply"
		sub1            = "sub_agent1: sub1 reply"
		root            = "root_agent: root reply"
		tool            = "sub_workflow_tool@fc_1.inner_input_node@1"
	)
	tests := []struct {
		name   string
		events func() []*session.Event
		answer string
		want   string
	}{
		{
			// The call that opened the outer tool branch comes after the
			// inner one; the innermost caller still wins.
			name: "nested_sub_branch_innermost_caller_wins",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_inner", "inner_tool"),
					routingNamedCall("sub_agent1", "", "inv_1", "fc_outer", "outer_tool"),
					routingPause("leaf_node", "outer_tool@fc_outer.inner_tool@fc_inner.leaf_node@1", "inv_1", "req_1", "adk_request_input"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			// The tool's call ID is reused after the pause: the owner is
			// the caller known when the pause was raised.
			name: "call_id_reused_after_pause",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_1", "sub_workflow_tool"),
					routingPause("inner_input_node", tool, "inv_1", "req_1", "adk_request_input"),
					routingNamedCall("sub_agent1", "", "inv_2", "fc_1", "other_tool"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			name: "open_interrupt_with_numeric_outer_call_id",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "42", "sub_workflow_tool"),
					routingPause("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			// A regular tool's call pauses through its long-running ID,
			// whatever the call's name.
			name: "long_running_regular_tool",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "42", "sub_workflow_tool"),
					routingPause("inner_node", "sub_workflow_tool@42.inner_node@1", "inv_1", "lro_1", "long_tool"),
				}
			},
			answer: "lro_1",
			want:   nonTransferable,
		},
		{
			// The pausing node's own interim response does not close the
			// interrupt; only the user's response does.
			name: "interim_response_by_node_keeps_interrupt_open",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "42", "sub_workflow_tool"),
					routingPause("inner_node", "sub_workflow_tool@42.inner_node@1", "inv_1", "lro_1", "long_tool"),
					routingEvent("inner_node", "sub_workflow_tool@42.inner_node@1", "inv_1", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "lro_1", Name: "long_tool", Response: map[string]any{"status": "pending"}}}),
				}
			},
			answer: "lro_1",
			want:   nonTransferable,
		},
		{
			// An adk_request_input call stored without long-running IDs is
			// still a pause.
			name: "request_input_without_long_running_ids",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "42", "sub_workflow_tool"),
					routingNamedCall("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			name: "request_credential_without_long_running_ids",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "42", "sub_workflow_tool"),
					routingNamedCall("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "cred_1", "adk_request_credential"),
				}
			},
			answer: "cred_1",
			want:   nonTransferable,
		},
		{
			// A regular call by sub_agent1 reuses the ID of the open
			// interrupt; the interrupt's owner takes the answer.
			name: "open_interrupt_takes_precedence_over_call_author",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_1", "sub_workflow_tool"),
					routingPause("inner_input_node", tool, "inv_1", "req_1", "adk_request_input"),
					routingNamedCall("sub_agent1", "", "inv_2", "req_1", "regular_tool"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			name: "interrupt_answered_by_id_removed_from_index",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_1", "sub_workflow_tool"),
					routingPause("inner_input_node", tool, "inv_1", "req_1", "adk_request_input"),
					routingEvent("user", tool, "inv_2", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "req_1", Name: "adk_request_input"}}),
					routingNamedCall("sub_agent1", "", "inv_3", "req_1", "regular_tool"),
				}
			},
			answer: "req_1",
			want:   sub1,
		},
		{
			name: "interrupt_answered_on_child_sub_branch_removed_from_index",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_1", "sub_workflow_tool"),
					routingPause("inner_input_node", tool, "inv_1", "req_1", "adk_request_input"),
					routingEvent("user", tool+".child@req_1", "inv_2", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "child_1", Name: "child"}}),
					routingNamedCall("sub_agent1", "", "inv_3", "req_1", "regular_tool"),
				}
			},
			answer: "req_1",
			want:   sub1,
		},
		{
			// The run ID of a segment is what follows its last "@".
			name: "multi_at_segment",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_tool", "sub_workflow_tool"),
					routingPause("inner_input_node", "mcp@server@fc_tool.inner_input_node@1", "inv_1", "req_1", "adk_request_input"),
				}
			},
			answer: "req_1",
			want:   nonTransferable,
		},
		{
			// A regular call, not a pause, by a node outside the tree: the
			// agent whose call opened the call's tool branch takes the answer.
			name: "regular_call_on_tool_branch_resolved_by_run_id",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "", "inv_1", "fc_tool", "sub_workflow_tool"),
					routingNamedCall("inner_node", "sub_workflow_tool@fc_tool.inner_node@1", "inv_1", "c1", "inner_tool"),
				}
			},
			answer: "c1",
			want:   nonTransferable,
		},
		{
			// The run-id fallback ignores numeric run IDs, which name node
			// runs, not calls: sub_agent1's call "1" does not take the answer.
			name: "numeric_run_ids_ignored_by_fallback",
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("sub_agent1", "", "inv_1", "1", "some_tool"),
					routingEvent("user", "", "inv_1", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "1", Name: "some_tool"}}),
					routingText("root_agent", "", "inv_2", "root earlier"),
					routingNamedCall("ghost", "node@1.leaf@2", "inv_2", "c9", "t"),
				}
			},
			answer: "c9",
			want:   root,
		},
	}
	for _, tc := range tests {
		for _, resumable := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/resumable=%t", tc.name, resumable), func(t *testing.T) {
				t.Parallel()
				svc := session.InMemoryService()
				r := llmRoutingRunner(t, svc, resumable)

				got := routedReplies(t, r, svc, tc.events(), routingAnswers(tc.answer))

				if diff := cmp.Diff([]string{tc.want}, got); diff != "" {
					t.Errorf("replies mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

// TestRunner_Run_AgentAnswerOnNonResumableRunnerSkipsInterruptIndex pins the
// order of the routing checks: the open interrupts are consulted only once
// the answer is routed to its call's author, which an agent's own answer on
// a non-resumable runner is not (adk-python agents/_agent_router.py:186-193),
// so the scan passes over non_transferable and the root runs.
func TestRunner_Run_AgentAnswerOnNonResumableRunnerSkipsInterruptIndex(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	r := llmRoutingRunner(t, svc, false)
	events := []*session.Event{
		routingPause("non_transferable", "", "inv_1", "c1", "long_tool"),
		routingEvent("non_transferable", "", "inv_1", &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "long_tool"}}),
	}

	got := routedReplies(t, r, svc, events, genai.NewContentFromText("next", genai.RoleUser))

	if diff := cmp.Diff([]string{"root_agent: root reply"}, got); diff != "" {
		t.Errorf("replies mismatch (-want +got):\n%s", diff)
	}
}

// TestRunner_ClassicPath_ToolSubBranchPauseRoutesToOwningSubAgent pins the
// classic path's routing of an answer to a pause raised inside a tool: it
// runs the agent that owns the tool, on its restored branch, whether the
// answer is the run's message or a plugin's stored replacement, and the root
// when the matched call event has no author or the owner is inside a graph
// workflow (adk-python agents/_agent_router.py:46-91, :186).
func TestRunner_ClassicPath_ToolSubBranchPauseRoutesToOwningSubAgent(t *testing.T) {
	t.Parallel()
	owned := func(callID string, pause *session.Event) func() []*session.Event {
		return func() []*session.Event {
			return []*session.Event{
				routingNamedCall("sub", "root.sub@1", "inv_1", callID, "sub_workflow_tool"),
				pause,
			}
		}
	}
	inMemory := func() session.Service { return session.InMemoryService() }
	tests := []struct {
		name         string
		events       func() []*session.Event
		msg          *genai.Content
		plugins      func(t *testing.T) []*plugin.Plugin
		svc          func() session.Service
		wantRootRuns []routedRun
		wantSubRuns  []routedRun
	}{
		{
			name:        "routes_to_owning_sub_agent",
			events:      owned("fc_tool", routingNamedCall("inner_input_node", "sub_workflow_tool@fc_tool.inner_input_node@1", "inv_1", "req_1", "adk_request_input")),
			msg:         routingAnswers("req_1"),
			plugins:     func(*testing.T) []*plugin.Plugin { return nil },
			svc:         inMemory,
			wantSubRuns: []routedRun{{invocationID: "inv_1", branch: "root.sub@1"}},
		},
		{
			name:        "numeric_outer_call_id",
			events:      owned("42", routingPause("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input")),
			msg:         routingAnswers("req_1"),
			plugins:     func(*testing.T) []*plugin.Plugin { return nil },
			svc:         inMemory,
			wantSubRuns: []routedRun{{invocationID: "inv_1", branch: "root.sub@1"}},
		},
		{
			// The replacement is stored before the route is taken again; the
			// routing view holds it once, so it does not close the interrupt
			// it answers.
			name:   "numeric_outer_call_id_plugin_replaced",
			events: owned("42", routingPause("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input")),
			msg:    genai.NewContentFromText("approve", genai.RoleUser),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replacingPlugin(t, routingAnswers("req_1"))}
			},
			svc:         inMemory,
			wantSubRuns: []routedRun{{invocationID: "inv_1", branch: "root.sub@1"}},
		},
		{
			// The stored replacement holds a copy of the plugin's content;
			// it still counts once.
			name:   "numeric_outer_call_id_plugin_replaced_content_copying_service",
			events: owned("42", routingPause("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input")),
			msg:    genai.NewContentFromText("approve", genai.RoleUser),
			plugins: func(t *testing.T) []*plugin.Plugin {
				return []*plugin.Plugin{replacingPlugin(t, routingAnswers("req_1"))}
			},
			svc:         func() session.Service { return contentCopyingService{session.InMemoryService()} },
			wantSubRuns: []routedRun{{invocationID: "inv_1", branch: "root.sub@1"}},
		},
		{
			name:         "empty_call_author_runs_root",
			events:       owned("fc_tool", routingNamedCall("", "sub_workflow_tool@fc_tool.inner_input_node@1", "inv_1", "req_1", "adk_request_input")),
			msg:          routingAnswers("req_1"),
			plugins:      func(*testing.T) []*plugin.Plugin { return nil },
			svc:          inMemory,
			wantRootRuns: []routedRun{{invocationID: "inv_1", branch: ""}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var rootRuns, subRuns []routedRun
			svc := tc.svc()
			r := classicRoutingRunner(t, svc, &rootRuns, &subRuns, tc.plugins(t)...)

			runRouting(t, r, svc, tc.events(), tc.msg)

			if diff := cmp.Diff(tc.wantRootRuns, rootRuns, cmp.AllowUnexported(routedRun{})); diff != "" {
				t.Errorf("root runs mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantSubRuns, subRuns, cmp.AllowUnexported(routedRun{})); diff != "" {
				t.Errorf("sub runs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRunner_ClassicPath_ToolSubBranchPauseInGraphWorkflowRunsRoot pins that
// an owner resolved through the open interrupts is not run directly when it
// is a node agent of a graph workflow: the root runs, so the workflow
// resumes, as for a call the node agent made itself.
func TestRunner_ClassicPath_ToolSubBranchPauseInGraphWorkflowRunsRoot(t *testing.T) {
	t.Parallel()
	var rootRuns, nodeRuns []routedRun
	nodeAgent := routingRecorder("node_agent", &nodeRuns)
	node, err := workflow.NewAgentNode(nodeAgent, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name:      "wf",
		Edges:     workflow.Chain(workflow.Start, node),
		SubAgents: []agent.Agent{nodeAgent},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	svc := session.InMemoryService()
	r, err := New(Config{AppName: "app", Agent: routingRecorder("root", &rootRuns, wf), SessionService: svc})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	runRouting(t, r, svc, []*session.Event{
		routingNamedCall("node_agent", "wf@1.node_agent@1", "inv_1", "42", "sub_workflow_tool"),
		routingPause("inner_input_node", "sub_workflow_tool@42.inner_input_node@1", "inv_1", "req_1", "adk_request_input"),
	}, routingAnswers("req_1"))

	if diff := cmp.Diff([]routedRun{{invocationID: "inv_1", branch: ""}}, rootRuns, cmp.AllowUnexported(routedRun{})); diff != "" {
		t.Errorf("root runs mismatch (-want +got):\n%s", diff)
	}
	if len(nodeRuns) != 0 {
		t.Errorf("node_agent ran as %+v, want no run", nodeRuns)
	}
}

// contentCopyingService is a session.Service that stores a copy of each
// appended event's content, as a service that serializes events does: a
// stored event never holds the caller's *genai.Content.
type contentCopyingService struct{ session.Service }

func (s contentCopyingService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	stored := *ev
	if ev.Content != nil {
		content := *ev.Content
		content.Parts = slices.Clone(ev.Content.Parts)
		stored.Content = &content
	}
	err := s.Service.AppendEvent(ctx, sess, &stored)
	ev.ID = stored.ID
	return err
}

// TestRunner_NodePath_PluginReplacedFunctionResponseResolvesSubBranchInterrupt
// pins the node path's route after a plugin replaced the message with an
// answer to a pause raised inside a tool: the route is taken on the saved
// session alone, as adk-python does (workflow/_node_runner_utils.py:192-203),
// so the replacement counts once whatever the session service stores, the
// interrupt stays open, and its owner runs on its restored branch in the
// invocation of the pause.
func TestRunner_NodePath_PluginReplacedFunctionResponseResolvesSubBranchInterrupt(t *testing.T) {
	t.Parallel()
	ownedPause := func(callID string) func() []*session.Event {
		return func() []*session.Event {
			return []*session.Event{
				routingNamedCall("non_transferable", "root.non_transferable@1", "inv_1", callID, "sub_workflow_tool"),
				routingPause("inner_input_node", "sub_workflow_tool@"+callID+".inner_input_node@1", "inv_1", "req_1", "adk_request_input"),
			}
		}
	}
	inMemory := func() session.Service { return session.InMemoryService() }
	contentCopying := func() session.Service { return contentCopyingService{session.InMemoryService()} }
	tests := []struct {
		name   string
		svc    func() session.Service
		events func() []*session.Event
	}{
		{name: "call_fc_tool", svc: inMemory, events: ownedPause("fc_tool")},
		{name: "call_42", svc: inMemory, events: ownedPause("42")},
		{name: "call_42_content_copying_service", svc: contentCopying, events: ownedPause("42")},
		{
			// The innermost caller at the time of the pause owns it.
			name: "nested_tool_branches_content_copying_service",
			svc:  contentCopying,
			events: func() []*session.Event {
				return []*session.Event{
					routingNamedCall("non_transferable", "root.non_transferable@1", "inv_1", "fc_inner", "inner_tool"),
					routingNamedCall("sub_agent1", "root.sub_agent1@1", "inv_1", "fc_outer", "outer_tool"),
					routingPause("leaf_node", "outer_tool@fc_outer.inner_tool@fc_inner.leaf_node@1", "inv_1", "req_1", "adk_request_input"),
				}
			},
		},
	}
	for _, tc := range tests {
		for _, resumable := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/resumable=%t", tc.name, resumable), func(t *testing.T) {
				t.Parallel()
				svc := tc.svc()
				r := llmRoutingRunner(t, svc, resumable, replacingPlugin(t, routingAnswers("req_1")))

				got := routedEvents(t, r, svc, tc.events(), genai.NewContentFromText("approve", genai.RoleUser))

				want := []string{"non_transferable|root.non_transferable@1|inv_1|non_transferable reply"}
				var gotLines []string
				for _, ev := range got {
					gotLines = append(gotLines, ev.Author+"|"+ev.Branch+"|"+ev.InvocationID+"|"+ev.Content.Parts[0].Text)
				}
				if diff := cmp.Diff(want, gotLines); diff != "" {
					t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}
