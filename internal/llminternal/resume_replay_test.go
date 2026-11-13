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

package llminternal

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// replaySession is a session whose events are the slice it holds, so a test
// can append the events a flow yields, as the runner does. The embedded
// Session, when set, supplies everything else.
type replaySession struct {
	session.Session
	events []*session.Event
}

func (s *replaySession) Events() session.Events { return replayEvents(s.events) }

type replayEvents []*session.Event

func (e replayEvents) All() iter.Seq[*session.Event] { return slices.Values(e) }
func (e replayEvents) Len() int                      { return len(e) }
func (e replayEvents) At(i int) *session.Event       { return e[i] }

// replayContext returns an invocation context of invocation "inv-1" for an
// agent named agentName on branch "", whose session holds events in order.
// An event without an ID gets "e<index>", and one without an invocation ID
// gets "inv-1". A resumable context carries a RunConfig with Resumable set.
func replayContext(t *testing.T, resumable bool, agentName string, events ...*session.Event) agent.InvocationContext {
	t.Helper()
	ctx, _ := replayAgentContext(t, resumable, &mockAgent{name: agentName}, events...)
	return ctx
}

// replayAgentContext is replayContext for agent a. It also returns the
// session, whose state comes from an in-memory session so that tools can use
// it.
func replayAgentContext(t *testing.T, resumable bool, a agent.Agent, events ...*session.Event) (agent.InvocationContext, *replaySession) {
	t.Helper()
	for i, ev := range events {
		if ev.ID == "" {
			ev.ID = fmt.Sprintf("e%d", i)
		}
		if ev.InvocationID == "" {
			ev.InvocationID = "inv-1"
		}
	}
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := &replaySession{Session: created.Session, events: events}
	ctx := t.Context()
	if resumable {
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{Resumable: true})
	}
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Session:      sess,
		Agent:        a,
		InvocationID: "inv-1",
	}), sess
}

// runReplayFlow runs f.Run on ctx and appends every non-partial event it
// yields to sess, as the runner stores them, so that a step sees the events
// of the steps before it.
func runReplayFlow(t *testing.T, f *Flow, ctx agent.InvocationContext, sess *replaySession) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if !ev.Partial {
			sess.events = append(sess.events, ev)
		}
		out = append(out, ev)
	}
	return out
}

// replayModel returns a model that answers its i-th call with the text
// texts[i] and counts its calls in calls. A call beyond texts fails the test.
func replayModel(t *testing.T, calls *int, texts ...string) *mockModelForTest {
	return &mockModelForTest{
		name: "scripted",
		generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) {
				*calls++
				if *calls > len(texts) {
					t.Errorf("model called %d times, want at most %d", *calls, len(texts))
					return
				}
				yield(&model.LLMResponse{Content: genai.NewContentFromText(texts[*calls-1], genai.RoleModel)}, nil)
			}
		},
	}
}

// countedTool returns a declared tool named name that counts its runs in runs
// and returns result.
func countedTool(t *testing.T, name string, runs *atomic.Int32, result map[string]any) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name}, func(agent.Context, struct{}) (map[string]any, error) {
		runs.Add(1)
		return result, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", name, err)
	}
	return tl
}

// describeReplayEvents names each event as "<author> call <name>/<id>" or
// "<author> response <name>/<id>" for its first function part, or as
// "<author> text <text>".
func describeReplayEvents(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, describeReplayEvent(ev))
	}
	return out
}

func describeReplayEvent(ev *session.Event) string {
	c := utils.Content(ev)
	if calls := utils.FunctionCalls(c); len(calls) > 0 {
		return fmt.Sprintf("%s call %s/%s", ev.Author, calls[0].Name, calls[0].ID)
	}
	if responses := utils.FunctionResponses(c); len(responses) > 0 {
		return fmt.Sprintf("%s response %s/%s", ev.Author, responses[0].Name, responses[0].ID)
	}
	text := ""
	if c != nil {
		for _, p := range c.Parts {
			text += p.Text
		}
	}
	return ev.Author + " text " + text
}

func fcall(name, id string) *genai.FunctionCall {
	return &genai.FunctionCall{Name: name, ID: id}
}

func fcallArgs(name, id string, args map[string]any) *genai.FunctionCall {
	return &genai.FunctionCall{Name: name, ID: id, Args: args}
}

// callEv returns an event authored author holding calls as its parts.
func callEv(author string, calls ...*genai.FunctionCall) *session.Event {
	parts := make([]*genai.Part, len(calls))
	for i, fc := range calls {
		parts[i] = &genai.Part{FunctionCall: fc}
	}
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleModel, Parts: parts},
	}}
}

// respEv returns an event authored author holding one function response.
func respEv(author, name, id string) *session.Event {
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{Name: name, ID: id, Response: map[string]any{"r": 1}}},
		}},
	}}
}

func textEv(author, text string) *session.Event {
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{
		Content: genai.NewContentFromText(text, genai.RoleModel),
	}}
}

func withLRO(ev *session.Event, ids ...string) *session.Event {
	ev.LongRunningToolIDs = ids
	return ev
}

func withBranch(ev *session.Event, branch string) *session.Event {
	ev.Branch = branch
	return ev
}

func replayTools(names ...string) map[string]tool.Tool {
	tools := map[string]tool.Tool{}
	for _, name := range names {
		tools[name] = &mockFunctionTool{name: name}
	}
	return tools
}

// decision is what a test compares: the action and the calls of the event
// to replay, in order.
type decision struct {
	Action resumeAction
	Calls  []*genai.FunctionCall
}

func summarize(d resumeDecision) decision {
	return decision{Action: d.action, Calls: utils.FunctionCalls(utils.Content(d.event))}
}

var (
	continueStep = decision{Action: resumeActionContinue}
	pauseStep    = decision{Action: resumeActionPause}
)

func replayStep(calls ...*genai.FunctionCall) decision {
	return decision{Action: resumeActionReplayCalls, Calls: calls}
}

func TestDecideStepResume_EmptyAndSingleTextBranchContinue(t *testing.T) {
	otherInvocation := textEv("agent", "earlier")
	otherInvocation.InvocationID = "other_inv"
	tests := []struct {
		name   string
		events []*session.Event
	}{
		{"no events", nil},
		{"only another invocation's event", []*session.Event{otherInvocation}},
		{"one text event", []*session.Event{textEv("agent", "hello")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := replayContext(t, true, "agent", tt.events...)
			got := summarize(decideStepResume(ctx, replayTools("ask")))
			if diff := cmp.Diff(continueStep, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_PartialReplayPreservesNonCallPartsAndOriginalEvent(t *testing.T) {
	ev0 := &session.Event{Author: "agent", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{Text: "Before"},
		{FunctionCall: fcallArgs("ask", "c1", map[string]any{"i": 0})},
		{Text: "Between"},
		{FunctionCall: fcallArgs("ask", "c2", map[string]any{"i": 1})},
		{Text: "After"},
	}}}}
	ev1 := respEv("user", "ask", "")
	ctx := replayContext(t, true, "agent", ev0, ev1)

	d := decideStepResume(ctx, replayTools("ask"))

	if d.action != resumeActionReplayCalls {
		t.Fatalf("decideStepResume().action = %v, want resumeActionReplayCalls", d.action)
	}
	want := []*genai.Part{
		{Text: "Before"},
		{Text: "Between"},
		{FunctionCall: fcallArgs("ask", "c2", map[string]any{"i": 1})},
		{Text: "After"},
	}
	if diff := cmp.Diff(want, d.event.Content.Parts); diff != "" {
		t.Errorf("replay event parts mismatch (-want +got):\n%s", diff)
	}
	if got := len(ev0.Content.Parts); got != 5 {
		t.Errorf("stored call event has %d parts after the decision, want 5", got)
	}
	if got := ev0.Content.Parts[1].FunctionCall.ID; got != "c1" {
		t.Errorf("stored call event part 1 has call ID %q after the decision, want %q", got, "c1")
	}
}

func TestDecideStepResume_SameNameAndIDLessResponseMatching(t *testing.T) {
	askCall := func(i int, id string) *genai.FunctionCall {
		return fcallArgs("ask", id, map[string]any{"index": i})
	}
	tests := []struct {
		name        string
		callIDs     []string
		responseIDs []string
		want        decision
	}{
		{"one ID-less response answers the first call", []string{"c1", "c2"}, []string{""}, replayStep(askCall(1, "c2"))},
		{"two ID-less responses answer both calls", []string{"c1", "c2"}, []string{"", ""}, continueStep},
		{"surplus ID-less responses", []string{"c1", "c2"}, []string{"", "", ""}, continueStep},
		{"explicit ID does not consume an ID-less response", []string{"c1", "c2", "c3"}, []string{"", "c1"}, replayStep(askCall(2, "c3"))},
		{"explicit ID first, then ID-less", []string{"c1", "c2", "c3"}, []string{"c1", ""}, replayStep(askCall(2, "c3"))},
		{"duplicate explicit ID answers one call", []string{"c1", "c2"}, []string{"c1", "c1"}, replayStep(askCall(1, "c2"))},
		{"unknown ID answers nothing", []string{"c1", "c2"}, []string{"", "other"}, replayStep(askCall(1, "c2"))},
		{"ID-less calls, one ID-less response", []string{"", ""}, []string{""}, replayStep(askCall(1, ""))},
		{"mixed ID-less and explicit calls", []string{"", "c2", ""}, []string{"", "c2"}, replayStep(askCall(2, ""))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := make([]*genai.FunctionCall, len(tt.callIDs))
			for i, id := range tt.callIDs {
				calls[i] = askCall(i, id)
			}
			events := []*session.Event{callEv("agent", calls...)}
			for _, id := range tt.responseIDs {
				events = append(events, respEv("user", "ask", id))
			}
			ctx := replayContext(t, true, "agent", events...)
			got := summarize(decideStepResume(ctx, replayTools("ask")))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_SingleCallAnsweredUnderDifferentNameReplays(t *testing.T) {
	tests := []struct {
		name   string
		events func() []*session.Event
	}{
		{"answered under another tool's name", func() []*session.Event {
			return []*session.Event{callEv("agent", fcall("ask", "c1")), respEv("user", "other_tool", "c1")}
		}},
		{"long-running call answered under another name", func() []*session.Event {
			return []*session.Event{withLRO(callEv("agent", fcall("ask", "c1")), "c1"), respEv("user", "other_tool", "c1")}
		}},
		{"agent text between call and answer", func() []*session.Event {
			return []*session.Event{callEv("agent", fcall("ask", "c1")), textEv("agent", "x"), respEv("user", "other", "c1")}
		}},
	}
	for _, tt := range tests {
		for _, resumable := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/resumable=%v", tt.name, resumable), func(t *testing.T) {
				ctx := replayContext(t, resumable, "agent", tt.events()...)
				got := summarize(decideStepResume(ctx, replayTools("ask")))
				if diff := cmp.Diff(replayStep(fcall("ask", "c1")), got); diff != "" {
					t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

func TestDecideStepResume_SubBranchHITLAnswerAndRunIDMatching(t *testing.T) {
	t.Run("input answer on the call's sub-branch", func(t *testing.T) {
		ctx := replayContext(t, true, "agent",
			callEv("agent", fcall("ask", "c1")),
			withBranch(respEv("user", requestInputFunctionCallName, "other"), "wf@r.ask@c1"),
			textEv("agent", "later"),
		)
		got := summarize(decideStepResume(ctx, replayTools("ask")))
		if diff := cmp.Diff(replayStep(fcall("ask", "c1")), got); diff != "" {
			t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("credential answer on the call's sub-branch", func(t *testing.T) {
		ctx := replayContext(t, true, "agent",
			callEv("agent", fcall("ask", "c1")),
			withBranch(respEv("user", toolauth.RequestCredentialFunctionCallName, "other"), "wf@r.ask@c1"),
			textEv("agent", "later"),
		)
		got := summarize(decideStepResume(ctx, replayTools("ask")))
		if diff := cmp.Diff(replayStep(fcall("ask", "c1")), got); diff != "" {
			t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("run ID that only contains the call ID", func(t *testing.T) {
		ctx := replayContext(t, true, "agent",
			callEv("agent", fcall("ask", "abc")),
			withBranch(respEv("user", requestInputFunctionCallName, "other"), "wf@r.ask@abcdef"),
		)
		got := summarize(decideStepResume(ctx, replayTools("ask")))
		if diff := cmp.Diff(pauseStep, got); diff != "" {
			t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("agent-authored answer on the call's sub-branch", func(t *testing.T) {
		// InvocationEvents drops an agent event off the agent's own branch, so
		// this row calls decideResume with the events directly.
		events := []*session.Event{
			callEv("agent", fcall("ask", "c1")),
			withBranch(respEv("agent", "other", "x"), "wf@r.ask@c1"),
		}
		ctx := replayContext(t, true, "agent", events...)
		got := summarize(decideResume(ctx, events, replayTools("ask")))
		if diff := cmp.Diff(pauseStep, got); diff != "" {
			t.Errorf("decideResume() mismatch (-want +got):\n%s", diff)
		}
	})
	for _, resumable := range []bool{true, false} {
		t.Run(fmt.Sprintf("step after the replay/resumable=%v", resumable), func(t *testing.T) {
			ctx := replayContext(t, resumable, "agent",
				callEv("agent", fcall("ask", "c1")),
				withBranch(respEv("user", requestInputFunctionCallName, "other"), "wf@r.ask@c1"),
				respEv("agent", "ask", "c1"),
			)
			got := summarize(decideStepResume(ctx, replayTools("ask")))
			if diff := cmp.Diff(continueStep, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_LongRunningCallNeverReplayedAndBlocksSiblings(t *testing.T) {
	tests := []struct {
		name      string
		resumable bool
		events    []*session.Event
		want      decision
	}{
		{
			name: "unanswered long-running sibling blocks a plain call",
			events: []*session.Event{
				withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2"), fcall("fetch", "c3")), "c1", "c2"),
				respEv("user", "ask", "c1"),
			},
			want: continueStep,
		},
		{
			name:      "unanswered long-running call, resumable",
			resumable: true,
			events: []*session.Event{
				withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")), "c1"),
				respEv("user", "fetch", "c2"),
			},
			want: pauseStep,
		},
		{
			name: "unanswered long-running call, not resumable",
			events: []*session.Event{
				withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")), "c1"),
				respEv("user", "fetch", "c2"),
			},
			want: continueStep,
		},
		{
			name: "plain siblings answered, one long-running call open",
			events: []*session.Event{
				withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2"), fcall("fetch", "c3")), "c1", "c2"),
				respEv("user", "ask", "c1"),
				respEv("user", "fetch", "c3"),
			},
			want: continueStep,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := replayContext(t, tt.resumable, "agent", tt.events...)
			got := summarize(decideStepResume(ctx, replayTools("ask", "fetch")))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_PauseRulesInResumableVsNonResumable(t *testing.T) {
	unansweredCall := func() []*session.Event {
		return []*session.Event{callEv("agent", fcall("ask", "c1")), respEv("user", "unrelated", "zzz")}
	}
	longRunningThenText := func() []*session.Event {
		return []*session.Event{withLRO(callEv("agent", fcall("ask", "c1")), "c1"), textEv("agent", "tail")}
	}
	tests := []struct {
		name      string
		resumable bool
		events    []*session.Event
		want      decision
	}{
		{"unanswered call, resumable", true, unansweredCall(), pauseStep},
		{"unanswered call, not resumable", false, unansweredCall(), continueStep},
		{"long-running call then text, resumable", true, longRunningThenText(), pauseStep},
		{"long-running call then text, not resumable", false, longRunningThenText(), continueStep},
		{"pausing call several events back", true, []*session.Event{
			withLRO(callEv("agent", fcall("other_lro", "c0")), "c0"),
			textEv("agent", "t1"), textEv("agent", "t2"), textEv("agent", "t3"),
		}, pauseStep},
		{"pausing call partly answered", true, []*session.Event{
			withLRO(callEv("agent", fcall("other1", "c1"), fcall("other2", "c2")), "c1", "c2"),
			respEv("user", "other1", "c1"),
			textEv("agent", "tail"),
		}, pauseStep},
		// A pausing event awaits every call ID it holds, not only its
		// long-running ones: fetch/c2 stays open, so it is not replayed.
		{"pausing event's plain call unanswered", true, []*session.Event{
			textEv("user", "hi"),
			withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")), "c1"),
			respEv("user", "ask", "c1"),
		}, pauseStep},
		{"last event pauses", true, []*session.Event{
			textEv("user", "hi"),
			withLRO(callEv("other_agent", fcall("unknown_lro", "c1")), "c1"),
		}, pauseStep},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := replayContext(t, tt.resumable, "agent", tt.events...)
			got := summarize(decideStepResume(ctx, replayTools("ask", "fetch")))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_AgentContentAfterCallPreventsSiblingReplay(t *testing.T) {
	stateOnly := &session.Event{Author: "agent", Actions: session.EventActions{AgentState: map[string]any{"k": "v"}}}
	tests := []struct {
		name   string
		events func() []*session.Event
		want   decision
	}{
		{"agent response after the call", func() []*session.Event {
			return []*session.Event{callEv("agent", fcall("ask", "c1"), fcall("ask", "c2")), respEv("agent", "ask", "")}
		}, continueStep},
		{"agent credential request after the call", func() []*session.Event {
			return []*session.Event{
				callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")),
				withLRO(callEv("agent", fcall(toolauth.RequestCredentialFunctionCallName, "a1")), "a1"),
				respEv("user", toolauth.RequestCredentialFunctionCallName, "a1"),
				respEv("agent", "ask", "c1"),
			}
		}, continueStep},
		{"agent event without content", func() []*session.Event {
			ev := *stateOnly
			return []*session.Event{
				callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")),
				&ev,
				respEv("user", "ask", "c1"),
			}
		}, replayStep(fcall("fetch", "c2"))},
	}
	for _, tt := range tests {
		for _, resumable := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/resumable=%v", tt.name, resumable), func(t *testing.T) {
				ctx := replayContext(t, resumable, "agent", tt.events()...)
				got := summarize(decideStepResume(ctx, replayTools("ask", "fetch")))
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

func TestDecideStepResume_TrailingCallGuards(t *testing.T) {
	partial := callEv("agent", fcall("ask", "c1"))
	partial.Partial = true
	tests := []struct {
		name      string
		resumable bool
		event     *session.Event
		want      decision
	}{
		{"partial trailing call", true, partial, continueStep},
		{"trailing call, not resumable", false, callEv("agent", fcall("ask", "c1")), continueStep},
		{"trailing call by another agent", true, callEv("other_agent", fcall("ask", "c1")), continueStep},
		{"lone long-running call, resumable", true, withLRO(callEv("agent", fcall("ask", "c1")), "c1"), replayStep(fcall("ask", "c1"))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := replayContext(t, tt.resumable, "agent", tt.event)
			got := summarize(decideStepResume(ctx, replayTools("ask")))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecideStepResume_IgnoresForeignAuthorAndUnknownToolCalls(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   decision
	}{
		{"call authored by the user", []*session.Event{callEv("user", fcall("mine", "c1")), textEv("agent", "tail")}, continueStep},
		{"call authored by another agent", []*session.Event{callEv("other_agent", fcall("mine", "c1")), textEv("agent", "tail")}, continueStep},
		{"later call of an unknown tool", []*session.Event{
			callEv("agent", fcall("mine", "c1")),
			callEv("agent", fcall("not_mine", "c2")),
			respEv("user", "not_mine", "c2"),
		}, pauseStep},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := replayContext(t, true, "agent", tt.events...)
			got := summarize(decideStepResume(ctx, replayTools("mine")))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// respsEv returns an event authored author holding responses as its parts.
func respsEv(author string, responses ...*genai.FunctionResponse) *session.Event {
	parts := make([]*genai.Part, len(responses))
	for i, fr := range responses {
		parts[i] = &genai.Part{FunctionResponse: fr}
	}
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleUser, Parts: parts},
	}}
}

func TestDecideStepResume_ResponseToAnotherCallEventIsNotAWrongNameAnswer(t *testing.T) {
	t1c1 := &genai.FunctionResponse{Name: "t1", ID: "c1", Response: map[string]any{"r": 1}}
	t2c2 := &genai.FunctionResponse{Name: "t2", ID: "c2", Response: map[string]any{"r": 1}}
	tests := []struct {
		name   string
		events func() []*session.Event
		want   decision
	}{
		{"one event answers the calls of two call events", func() []*session.Event {
			return []*session.Event{
				textEv("user", "hi"),
				withLRO(callEv("agent", fcall("t1", "c1")), "c1"),
				callEv("agent", fcall("t2", "c2")),
				respsEv("agent", t1c1, t2c2),
			}
		}, continueStep},
		{"one event answers two call events, a sibling call has no response", func() []*session.Event {
			return []*session.Event{
				textEv("user", "hi"),
				withLRO(callEv("agent", fcall("t1", "c1")), "c1"),
				callEv("agent", fcall("t2", "c2"), fcall("t3", "c3")),
				respsEv("agent", t1c1, t2c2),
			}
		}, continueStep},
		{"a response whose ID no call carries still names another tool", func() []*session.Event {
			return []*session.Event{
				textEv("user", "hi"),
				callEv("agent", fcall("t2", "c2"), fcall("t3", "c3")),
				respsEv("agent", &genai.FunctionResponse{Name: "t1", ID: "c1", Response: map[string]any{"r": 1}}, t2c2),
			}
		}, replayStep(fcall("t3", "c3"))},
		{"the answer to a later credential request replays the call", func() []*session.Event {
			return []*session.Event{
				callEv("agent", fcall("t2", "c2")),
				withLRO(callEv("agent", fcall(toolauth.RequestCredentialFunctionCallName, "a1")), "a1"),
				respEv("user", toolauth.RequestCredentialFunctionCallName, "a1"),
			}
		}, replayStep(fcall("t2", "c2"))},
	}
	for _, tt := range tests {
		for _, resumable := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/resumable=%v", tt.name, resumable), func(t *testing.T) {
				ctx := replayContext(t, resumable, "agent", tt.events()...)
				got := summarize(decideStepResume(ctx, replayTools("t1", "t2", "t3")))
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

// wrongNameRow2 is the brief's F2 case: a parallel call whose second call is
// answered under a name that is not its own.
func wrongNameRow2() []*session.Event {
	return []*session.Event{
		callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")),
		respEv("user", "ask", "c1"),
		respEv("user", "other", "c2"),
	}
}

func TestDecideResume_WrongNameAnswerReplaysOnlyUnansweredSibling(t *testing.T) {
	tests := []struct {
		name   string
		events func() []*session.Event
		want   decision
	}{
		{"answered sibling, long-running call answered under another name", func() []*session.Event {
			return []*session.Event{
				withLRO(callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")), "c2"),
				respEv("agent", "ask", "c1"),
				respEv("user", "other", "c2"),
			}
		}, replayStep(fcall("fetch", "c2"))},
		{"answered sibling, call answered under another name", wrongNameRow2, replayStep(fcall("fetch", "c2"))},
		{"every call answered under its own name too", func() []*session.Event {
			return []*session.Event{
				callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")),
				respEv("user", "ask", "c1"),
				respEv("user", "fetch", "c2"),
				respEv("user", "other", "c2"),
			}
		}, continueStep},
		{"wrong-name answer before the last answer", func() []*session.Event {
			return []*session.Event{
				callEv("agent", fcall("ask", "c1"), fcall("fetch", "c2")),
				respEv("user", "other", "c2"),
				respEv("user", "ask", "c1"),
			}
		}, continueStep},
	}
	for _, tt := range tests {
		for _, resumable := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/resumable=%v", tt.name, resumable), func(t *testing.T) {
				ctx := replayContext(t, resumable, "agent", tt.events()...)
				got := summarize(decideStepResume(ctx, replayTools("ask", "fetch")))
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("decideStepResume() mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
	t.Run("flow", func(t *testing.T) {
		var askRuns, fetchRuns atomic.Int32
		modelCalls := 0
		f := &Flow{
			Model: replayModel(t, &modelCalls, "done"),
			Tools: []tool.Tool{
				countedTool(t, "ask", &askRuns, map[string]any{"r": 1}),
				countedTool(t, "fetch", &fetchRuns, map[string]any{"r": 1}),
			},
		}
		ctx, sess := replayAgentContext(t, false, &mockAgent{name: "agent"}, wrongNameRow2()...)

		runReplayFlow(t, f, ctx, sess)

		if got := fetchRuns.Load(); got != 1 {
			t.Errorf("fetch ran %d times, want 1", got)
		}
		if got := askRuns.Load(); got != 0 {
			t.Errorf("ask ran %d times, want 0", got)
		}
	})
}
