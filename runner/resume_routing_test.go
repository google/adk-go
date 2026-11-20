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
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
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
