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

package workflowinternal_test

import (
	"errors"
	"fmt"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/workflow"
)

// callInvocationID is the invocation every tool call test runs in.
const callInvocationID = "inv"

// historySession serves exactly the seeded events, partial ones included:
// the in-memory service drops a partial event on append.
type historySession struct {
	session.Session
	events []*session.Event
}

func (s *historySession) Events() session.Events { return eventList(s.events) }

type eventList []*session.Event

func (l eventList) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range l {
			if !yield(ev) {
				return
			}
		}
	}
}

func (l eventList) Len() int { return len(l) }

func (l eventList) At(i int) *session.Event { return l[i] }

// toolCall is one RunToolCall made by the agent node of a callRun.
type toolCall struct {
	name string
	id   string
	body workflowinternal.ToolBody
}

type callResult struct {
	out     any
	err     error
	toolCtx agent.Context
}

// callRun describes one activation of a dynamic agent node named
// "root_agent" (so its children's paths start with "root_agent/") that
// makes calls in order.
type callRun struct {
	branch       string
	history      []*session.Event
	resumeInputs map[string]any
	calls        []toolCall
}

// runCalls runs r through the real dynamic node and its sub-scheduler and
// returns each call's result and every event the node emitted.
func runCalls(t *testing.T, r callRun) ([]callResult, []*session.Event) {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "test_app", UserID: "test_user", SessionID: "test_session",
	})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	sess := &historySession{Session: created.Session, events: r.history}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Session:      sess,
		InvocationID: callInvocationID,
		Branch:       r.branch,
	})
	ctx := agent.NewContext(ic)
	if r.resumeInputs != nil {
		ctx = ctx.WithDelta(&agent.CommonContextDelta{ResumeInputs: &r.resumeInputs})
	}

	results := make([]callResult, len(r.calls))
	node := workflow.NewDynamicNode("root_agent",
		func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			for i, c := range r.calls {
				toolCtx := agent.NewToolContext(ctx, c.id, &session.EventActions{}, nil)
				out, err := workflowinternal.RunToolCall(toolCtx, c.name, c.body)
				results[i] = callResult{out: out, err: err, toolCtx: toolCtx}
			}
			return nil, nil
		},
		workflow.NodeConfig{},
	)
	var events []*session.Event
	for ev, err := range node.Run(ctx, nil) {
		if err != nil {
			t.Fatalf("root_agent node: %v", err)
		}
		events = append(events, ev)
	}
	return results, events
}

// message is a model text event as a tool body builds it.
func message(text string) *session.Event {
	return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}}
}

// textEvents returns the events in events whose only text is text.
func textEvents(events []*session.Event, text string) []*session.Event {
	var got []*session.Event
	for _, ev := range events {
		if ev.Content != nil && len(ev.Content.Parts) == 1 && ev.Content.Parts[0].Text == text {
			got = append(got, ev)
		}
	}
	return got
}

// pathOf returns ev's node path, or "" when it has none.
func pathOf(ev *session.Event) string {
	if ev.NodeInfo == nil {
		return ""
	}
	return ev.NodeInfo.Path
}

// countingBody counts its runs in *runs, emits the message "m" and returns "r".
func countingBody(runs *int) workflowinternal.ToolBody {
	return func(_ agent.Context, emit func(*session.Event) error) (any, error) {
		*runs++
		return "r", emit(message("m"))
	}
}

// pausingBody asks for input under interruptID and returns the pause.
func pausingBody(interruptID string) workflowinternal.ToolBody {
	return func(ctx agent.Context, emit func(*session.Event) error) (any, error) {
		_, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: interruptID, Message: "Which region?"})
		return nil, err
	}
}

func responseEvent(invocationID, id, name string) *session.Event {
	return &session.Event{
		InvocationID: invocationID,
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: id, Name: name, Response: map[string]any{"result": "x"},
			}}},
		}},
	}
}

func outputEvent(invocationID, path string, out any) *session.Event {
	return &session.Event{InvocationID: invocationID, NodeInfo: &session.NodeInfo{Path: path}, Output: out}
}

func interruptEvent(invocationID, path, interruptID string) *session.Event {
	return &session.Event{InvocationID: invocationID, NodeInfo: &session.NodeInfo{Path: path}, LongRunningToolIDs: []string{interruptID}}
}

// TestRunToolCall_NodeName fails if the node name's character class loses
// '_', the replacement works per byte (café becoming caf__), or the leading
// '_' for a name that starts with a digit is dropped.
func TestRunToolCall_NodeName(t *testing.T) {
	tests := []struct {
		toolName   string
		wantAuthor string
		wantPath   string
	}{
		{toolName: "analyze", wantAuthor: "analyze", wantPath: "root_agent/analyze@call_1"},
		{toolName: "fetch-data", wantAuthor: "fetch_data", wantPath: "root_agent/fetch_data@call_1"},
		{toolName: "a.b", wantAuthor: "a_b", wantPath: "root_agent/a_b@call_1"},
		{toolName: "9x", wantAuthor: "_9x", wantPath: "root_agent/_9x@call_1"},
		{toolName: "café", wantAuthor: "caf_", wantPath: "root_agent/caf_@call_1"},
		{toolName: "日本", wantAuthor: "__", wantPath: "root_agent/__@call_1"},
		{toolName: "9日", wantAuthor: "_9_", wantPath: "root_agent/_9_@call_1"},
	}
	for _, tc := range tests {
		t.Run(tc.toolName, func(t *testing.T) {
			body := func(_ agent.Context, emit func(*session.Event) error) (any, error) {
				ev := message("m")
				ev.Author = "custom_reporter"
				return nil, emit(ev)
			}
			results, events := runCalls(t, callRun{calls: []toolCall{{name: tc.toolName, id: "call_1", body: body}}})
			if err := results[0].err; err != nil {
				t.Fatalf("RunToolCall() error = %v, want nil", err)
			}
			got := textEvents(events, "m")
			if len(got) != 1 {
				t.Fatalf("got %d events with text %q, want 1", len(got), "m")
			}
			if got[0].Author != tc.wantAuthor {
				t.Errorf("Author = %q, want %q", got[0].Author, tc.wantAuthor)
			}
			if p := pathOf(got[0]); p != tc.wantPath {
				t.Errorf("NodeInfo.Path = %q, want %q", p, tc.wantPath)
			}
		})
	}
}

// TestRunToolCall_Branch fails if the branch uses the sanitized tool name
// or a separator other than '.' after the caller's branch.
func TestRunToolCall_Branch(t *testing.T) {
	tests := []struct {
		callerBranch string
		want         string
	}{
		{callerBranch: "", want: "fetch-data@call_1"},
		{callerBranch: "root.sub", want: "root.sub.fetch-data@call_1"},
	}
	for _, tc := range tests {
		t.Run(tc.want, func(t *testing.T) {
			var runs int
			results, events := runCalls(t, callRun{
				branch: tc.callerBranch,
				calls:  []toolCall{{name: "fetch-data", id: "call_1", body: countingBody(&runs)}},
			})
			if err := results[0].err; err != nil {
				t.Fatalf("RunToolCall() error = %v, want nil", err)
			}
			got := textEvents(events, "m")
			if len(got) != 1 {
				t.Fatalf("got %d events with text %q, want 1", len(got), "m")
			}
			if got[0].Branch != tc.want {
				t.Errorf("Branch = %q, want %q", got[0].Branch, tc.want)
			}
		})
	}
}

// TestRunToolCall_RunID fails if an id that workflow.WithRunID refuses is
// passed through (the call fails), if the all-digits rule is missing, or if
// '%' is not escaped in the escaped form (two ids would share a path).
func TestRunToolCall_RunID(t *testing.T) {
	tests := []struct {
		id       string
		wantPath string
	}{
		{id: "call_1", wantPath: "root_agent/analyze@call_1"},
		{id: "123", wantPath: "root_agent/analyze@~123"},
		{id: "a/b", wantPath: "root_agent/analyze@~a%2Fb"},
		{id: "a@b", wantPath: "root_agent/analyze@~a%40b"},
		{id: "~x", wantPath: "root_agent/analyze@~%7Ex"},
		{id: "100%", wantPath: "root_agent/analyze@100%"},
	}
	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			var runs int
			results, events := runCalls(t, callRun{calls: []toolCall{{name: "analyze", id: tc.id, body: countingBody(&runs)}}})
			if err := results[0].err; err != nil {
				t.Fatalf("RunToolCall() error = %v, want nil", err)
			}
			got := textEvents(events, "m")
			if len(got) != 1 {
				t.Fatalf("got %d events with text %q, want 1", len(got), "m")
			}
			if p := pathOf(got[0]); p != tc.wantPath {
				t.Errorf("NodeInfo.Path = %q, want %q", p, tc.wantPath)
			}
		})
	}

	t.Run("escaped ids do not collide", func(t *testing.T) {
		var runs int
		results, events := runCalls(t, callRun{calls: []toolCall{
			{name: "analyze", id: "a/b", body: countingBody(&runs)},
			{name: "analyze", id: "~a%2Fb", body: countingBody(&runs)},
		}})
		for i, r := range results {
			if r.err != nil {
				t.Fatalf("RunToolCall() of call %d error = %v, want nil", i, r.err)
			}
		}
		var paths []string
		for _, ev := range textEvents(events, "m") {
			paths = append(paths, pathOf(ev))
		}
		want := []string{"root_agent/analyze@~a%2Fb", "root_agent/analyze@~%7Ea%252Fb"}
		if diff := cmp.Diff(want, paths); diff != "" {
			t.Errorf("message paths (-want +got):\n%s", diff)
		}
		if runs != 2 {
			t.Errorf("body runs = %d, want 2", runs)
		}
	})

	t.Run("empty id is generated", func(t *testing.T) {
		var runs int
		results, events := runCalls(t, callRun{calls: []toolCall{{name: "analyze", id: "", body: countingBody(&runs)}}})
		if err := results[0].err; err != nil {
			t.Fatalf("RunToolCall() error = %v, want nil", err)
		}
		id := results[0].toolCtx.FunctionCallID()
		if id == "" {
			t.Fatal("FunctionCallID() is empty")
		}
		got := textEvents(events, "m")
		if len(got) != 1 {
			t.Fatalf("got %d events with text %q, want 1", len(got), "m")
		}
		if p := pathOf(got[0]); p != "root_agent/analyze@"+id {
			t.Errorf("NodeInfo.Path = %q, want %q", p, "root_agent/analyze@"+id)
		}
	})
}

// TestRunToolCall_AnswersFilter fails if the path test lacks the trailing
// '/' (call_10's answer leaks into call_1), if events of another invocation
// count, or if a call sees answers it did not ask for.
func TestRunToolCall_AnswersFilter(t *testing.T) {
	history := []*session.Event{
		interruptEvent(callInvocationID, "root_agent/ask@call_1", "x"),
		interruptEvent(callInvocationID, "root_agent/ask@call_10", "y"),
		interruptEvent(callInvocationID, "root_agent/ask@call_1/inner@1", "n"),
		interruptEvent("other", "root_agent/ask@call_1", "z"),
	}
	seen := map[string]map[string]any{}
	recordAnswers := func(ctx agent.Context, _ func(*session.Event) error) (any, error) {
		got := map[string]any{}
		for _, id := range []string{"x", "y", "n", "z", "w"} {
			if v, ok := ctx.ResumedInput(id); ok {
				got[id] = v
			}
		}
		seen[ctx.FunctionCallID()] = got
		return nil, nil
	}
	results, _ := runCalls(t, callRun{
		history:      history,
		resumeInputs: map[string]any{"x": 1, "y": 2, "n": 3, "z": 4, "w": 5},
		calls: []toolCall{
			{name: "ask", id: "call_1", body: recordAnswers},
			{name: "ask", id: "call_10", body: recordAnswers},
		},
	})
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("RunToolCall() of call %d error = %v, want nil", i, r.err)
		}
	}
	want := map[string]map[string]any{
		"call_1":  {"x": 1, "n": 3},
		"call_10": {"y": 2},
	}
	if diff := cmp.Diff(want, seen); diff != "" {
		t.Errorf("answers seen per call (-want +got):\n%s", diff)
	}
}

// TestRunToolCall_PauseIsTyped fails if a body's request for input is
// returned as a plain error or swallowed.
func TestRunToolCall_PauseIsTyped(t *testing.T) {
	results, events := runCalls(t, callRun{calls: []toolCall{{name: "ask", id: "call_1", body: pausingBody("region")}}})
	err := results[0].err
	if _, ok := errors.AsType[*toolinternal.PausedError](err); !ok {
		t.Fatalf("RunToolCall() error = %v, want a *toolinternal.PausedError", err)
	}
	if !errors.Is(err, workflow.ErrNodeInterrupted) {
		t.Errorf("RunToolCall() error = %v, want errors.Is(err, workflow.ErrNodeInterrupted)", err)
	}
	var requests []string
	for _, ev := range events {
		if ev.RequestedInput != nil {
			requests = append(requests, ev.RequestedInput.InterruptID+" "+pathOf(ev))
		}
	}
	if diff := cmp.Diff([]string{"region root_agent/ask@call_1"}, requests); diff != "" {
		t.Errorf("request-input events (-want +got):\n%s", diff)
	}
}

// TestRunToolCall_MultipleOutputIsTyped fails if a second output returns
// workflow.ErrMultipleOutputs' own text or lacks the typed wrapper that ends
// the run.
func TestRunToolCall_MultipleOutputIsTyped(t *testing.T) {
	body := func(_ agent.Context, emit func(*session.Event) error) (any, error) {
		if err := emit(&session.Event{Output: map[string]any{"rows": 1}}); err != nil {
			return nil, err
		}
		return nil, emit(&session.Event{Output: map[string]any{"rows": 2}})
	}
	results, _ := runCalls(t, callRun{calls: []toolCall{{name: "analyze", id: "call_1", body: body}}})
	err := results[0].err
	if _, ok := errors.AsType[*toolinternal.FailRunError](err); !ok {
		t.Fatalf("RunToolCall() error = %v, want a *toolinternal.FailRunError", err)
	}
	if !errors.Is(err, workflow.ErrMultipleOutputs) {
		t.Errorf("RunToolCall() error = %v, want errors.Is(err, workflow.ErrMultipleOutputs)", err)
	}
	if got, want := err.Error(), "Output already set. A node can produce at most one output."; got != want {
		t.Errorf("RunToolCall() error text = %q, want %q", got, want)
	}
}

// TestRunToolCall_SkipCopiedOnPause fails if the SkipSummarization flag is
// copied to the tool context only after the pause check.
func TestRunToolCall_SkipCopiedOnPause(t *testing.T) {
	body := func(ctx agent.Context, emit func(*session.Event) error) (any, error) {
		ctx.Actions().SkipSummarization = true
		return pausingBody("region")(ctx, emit)
	}
	results, _ := runCalls(t, callRun{calls: []toolCall{{name: "ask", id: "call_1", body: body}}})
	if _, ok := errors.AsType[*toolinternal.PausedError](results[0].err); !ok {
		t.Fatalf("RunToolCall() error = %v, want a *toolinternal.PausedError", results[0].err)
	}
	if !results[0].toolCtx.Actions().SkipSummarization {
		t.Error("toolCtx.Actions().SkipSummarization = false after a pausing body set it, want true")
	}
}

// TestRunToolCall_BodyContextAPIs fails if the body's context is the
// tool-context wrapper, whose ResumedInput, Session and Path are not
// supported.
func TestRunToolCall_BodyContextAPIs(t *testing.T) {
	type observed struct {
		CallID     string
		Path       string
		HasSession bool
		Answer     any
		Answered   bool
	}
	var got observed
	var bodyActions *session.EventActions
	body := func(ctx agent.Context, _ func(*session.Event) error) (any, error) {
		got.CallID = ctx.FunctionCallID()
		got.Path = ctx.Path()
		got.HasSession = ctx.Session() != nil
		got.Answer, got.Answered = ctx.ResumedInput("x")
		bodyActions = ctx.Actions()
		return nil, nil
	}
	results, _ := runCalls(t, callRun{
		history:      []*session.Event{interruptEvent(callInvocationID, "root_agent/analyze@call_1", "x")},
		resumeInputs: map[string]any{"x": map[string]any{"text": "APAC"}},
		calls:        []toolCall{{name: "analyze", id: "call_1", body: body}},
	})
	if err := results[0].err; err != nil {
		t.Fatalf("RunToolCall() error = %v, want nil", err)
	}
	want := observed{
		CallID:     "call_1",
		Path:       "root_agent/analyze@call_1",
		HasSession: true,
		Answer:     map[string]any{"text": "APAC"},
		Answered:   true,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("body context (-want +got):\n%s", diff)
	}
	if bodyActions == nil || bodyActions == results[0].toolCtx.Actions() {
		t.Errorf("body Actions() = %p, want non-nil and different from the tool context's %p", bodyActions, results[0].toolCtx.Actions())
	}
}

// failedRunHistory runs a call of analyze/call_1 whose body emits the output
// "r1" and then fails, and returns the events that run emitted.
func failedRunHistory(t *testing.T) []*session.Event {
	t.Helper()
	body := func(_ agent.Context, emit func(*session.Event) error) (any, error) {
		if err := emit(&session.Event{Output: "r1"}); err != nil {
			return nil, err
		}
		return nil, errors.New("boom")
	}
	results, events := runCalls(t, callRun{calls: []toolCall{{name: "analyze", id: "call_1", body: body}}})
	if results[0].err == nil {
		t.Fatal("failing run: RunToolCall() error = nil, want the body's error")
	}
	var onPath []*session.Event
	for _, ev := range events {
		if pathOf(ev) == "root_agent/analyze@call_1" {
			onPath = append(onPath, ev)
		}
	}
	if len(onPath) < 2 || onPath[0].Output != "r1" || onPath[len(onPath)-1].Output != nil {
		t.Fatalf("failing run: events on the call's path do not hold an output followed by the failure marker: %d events", len(onPath))
	}
	return events
}

// TestRunToolCall_CallSlot fails if the count of answered calls ignores the
// invocation (rows 4, 10), the id (5) or the name (6); if the confirmation
// placeholder is counted (8); if a path the sub-scheduler serves from its
// cache is not detected (9, 12, 14, 15, 16: the body would not run); if the
// order of outcomes on a path or the pause or failure that ends it is
// ignored (11, 13, 17); if a partial event counts (18); if the failure key
// differs from the scheduler's (13); or if the "~<k>" suffix is missing (2, 3).
func TestRunToolCall_CallSlot(t *testing.T) {
	const p = "root_agent/analyze@call_1"
	placeholder := responseEvent(callInvocationID, "call_1", "analyze")
	placeholder.Actions.RequestedToolConfirmations = map[string]toolconfirmation.ToolConfirmation{"call_1": {Hint: "approve"}}
	partialOutput := outputEvent(callInvocationID, p, "r1")
	partialOutput.Partial = true

	tests := []struct {
		row      int
		id       string
		history  func(t *testing.T) []*session.Event
		wantPath string
	}{
		{row: 1, history: nil, wantPath: "root_agent/analyze@call_1"},
		{row: 2, history: seed(responseEvent(callInvocationID, "call_1", "analyze")), wantPath: "root_agent/analyze@~call_1~1"},
		{row: 3, history: seed(
			responseEvent(callInvocationID, "call_1", "analyze"),
			responseEvent(callInvocationID, "call_1", "analyze"),
		), wantPath: "root_agent/analyze@~call_1~2"},
		{row: 4, history: seed(responseEvent("other", "call_1", "analyze")), wantPath: "root_agent/analyze@call_1"},
		{row: 5, history: seed(responseEvent(callInvocationID, "call_2", "analyze")), wantPath: "root_agent/analyze@call_1"},
		{row: 6, history: seed(responseEvent(callInvocationID, "call_1", "adk_request_input")), wantPath: "root_agent/analyze@call_1"},
		{row: 7, id: "123", history: seed(responseEvent(callInvocationID, "123", "analyze")), wantPath: "root_agent/analyze@~123~1"},
		{row: 8, history: seed(placeholder), wantPath: "root_agent/analyze@call_1"},
		{row: 9, history: seed(outputEvent(callInvocationID, p, "r1")), wantPath: "root_agent/analyze@~call_1~1"},
		{row: 10, history: seed(outputEvent("other", p, "r1")), wantPath: "root_agent/analyze@call_1"},
		{row: 11, history: seed(
			outputEvent(callInvocationID, p, "r1"),
			interruptEvent(callInvocationID, p, "region"),
		), wantPath: "root_agent/analyze@call_1"},
		{row: 12, history: seed(
			interruptEvent(callInvocationID, p, "region"),
			outputEvent(callInvocationID, p, "r1"),
		), wantPath: "root_agent/analyze@~call_1~1"},
		{row: 13, history: failedRunHistory, wantPath: "root_agent/analyze@call_1"},
		{row: 14, history: seed(
			outputEvent(callInvocationID, p, "r1"),
			responseEvent(callInvocationID, "call_1", "analyze"),
		), wantPath: "root_agent/analyze@~call_1~1"},
		{row: 15, history: seed(
			outputEvent(callInvocationID, p, "r1"),
			outputEvent(callInvocationID, "root_agent/analyze@~call_1~1", "r1"),
		), wantPath: "root_agent/analyze@~call_1~2"},
		{row: 16, history: seed(
			outputEvent(callInvocationID, p, "r1"),
			responseEvent(callInvocationID, "call_1", "analyze"),
			responseEvent(callInvocationID, "call_1", "analyze"),
		), wantPath: "root_agent/analyze@~call_1~2"},
		{row: 17, history: seed(interruptEvent(callInvocationID, p, "region")), wantPath: "root_agent/analyze@call_1"},
		{row: 18, history: seed(partialOutput), wantPath: "root_agent/analyze@call_1"},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("row_%02d", tc.row), func(t *testing.T) {
			id := tc.id
			if id == "" {
				id = "call_1"
			}
			var history []*session.Event
			if tc.history != nil {
				history = tc.history(t)
			}
			var runs int
			results, got := runCalls(t, callRun{
				history: history,
				calls:   []toolCall{{name: "analyze", id: id, body: countingBody(&runs)}},
			})
			if err := results[0].err; err != nil {
				t.Fatalf("RunToolCall() error = %v, want nil", err)
			}
			if runs != 1 {
				t.Errorf("body runs = %d, want 1", runs)
			}
			var paths []string
			for _, ev := range textEvents(got, "m") {
				paths = append(paths, pathOf(ev))
			}
			if diff := cmp.Diff([]string{tc.wantPath}, paths); diff != "" {
				t.Errorf("message paths (-want +got):\n%s", diff)
			}
		})
	}
}

// seed returns a history function that yields evs.
func seed(evs ...*session.Event) func(*testing.T) []*session.Event {
	return func(*testing.T) []*session.Event { return evs }
}
