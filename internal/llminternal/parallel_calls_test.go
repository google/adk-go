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
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// pcCountingUUIDs installs a UUID provider on ctx that returns
// "<prefix>1", "<prefix>2", ... (or prefix alone when numbered is false) and
// counts its calls.
func pcCountingUUIDs(ctx context.Context, prefix string, numbered bool) (context.Context, *atomic.Int32) {
	var n atomic.Int32
	return platform.WithUUIDProvider(ctx, func() string {
		i := n.Add(1)
		if !numbered {
			return prefix
		}
		return fmt.Sprintf("%s%d", prefix, i)
	}), &n
}

// pcResponseEvent is a function response event of tool name, with delta as its
// state delta.
func pcResponseEvent(id, name string, ts time.Time, delta map[string]any) *session.Event {
	return &session.Event{
		ID:           id,
		InvocationID: "inv",
		Timestamp:    ts,
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: name, Response: map[string]any{"ok": true}}}},
		}},
		Actions: session.EventActions{StateDelta: delta},
	}
}

func pcPartNames(ev *session.Event) []string {
	var names []string
	for _, p := range ev.LLMResponse.Content.Parts {
		names = append(names, p.FunctionResponse.Name)
	}
	return names
}

func TestMergeParallelResponses_NewIDOnlyWhenEventsMerge(t *testing.T) {
	t0 := time.Date(2026, 11, 14, 10, 0, 0, 0, time.UTC)
	t1 := t0.Add(time.Second)
	t2 := t0.Add(2 * time.Second)
	ev0 := func() *session.Event { return pcResponseEvent("ev-0", "t1", t0, nil) }
	ev1 := func() *session.Event { return pcResponseEvent("ev-1", "t2", t1, nil) }
	ev2 := func() *session.Event { return pcResponseEvent("ev-2", "t3", t2, nil) }

	tests := []struct {
		name          string
		events        []*session.Event
		wantNil       bool
		wantID        string
		wantParts     []string
		wantTimestamp time.Time
		wantUUIDCalls int32
	}{
		{name: "two events merge", events: []*session.Event{ev0(), ev1()}, wantID: "merged-id", wantParts: []string{"t1", "t2"}, wantTimestamp: t0, wantUUIDCalls: 1},
		{name: "nil slot between two events", events: []*session.Event{ev0(), nil, ev2()}, wantID: "merged-id", wantParts: []string{"t1", "t3"}, wantTimestamp: t0, wantUUIDCalls: 1},
		{name: "one event after a nil slot", events: []*session.Event{nil, ev1()}, wantID: "ev-1", wantParts: []string{"t2"}, wantTimestamp: t1, wantUUIDCalls: 0},
		{name: "single event", events: []*session.Event{ev0()}, wantID: "ev-0", wantParts: []string{"t1"}, wantTimestamp: t0, wantUUIDCalls: 0},
		{name: "only nil slots", events: []*session.Event{nil, nil}, wantNil: true, wantUUIDCalls: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			uctx, calls := pcCountingUUIDs(t.Context(), "merged-id", false)
			ctx := icontext.NewInvocationContext(uctx, icontext.InvocationContextParams{InvocationID: "inv"})

			got, err := mergeParallelResponses(ctx, tt.events)
			if err != nil {
				t.Fatalf("mergeParallelResponses() error = %v", err)
			}
			if n := calls.Load(); n != tt.wantUUIDCalls {
				t.Errorf("UUID provider calls = %d, want %d", n, tt.wantUUIDCalls)
			}
			if tt.wantNil {
				if got != nil {
					t.Fatalf("mergeParallelResponses() = event %q, want nil", got.ID)
				}
				return
			}
			if got == nil {
				t.Fatal("mergeParallelResponses() = nil, want an event")
			}
			if got.ID != tt.wantID {
				t.Errorf("ID = %q, want %q", got.ID, tt.wantID)
			}
			if got.InvocationID != "inv" {
				t.Errorf("InvocationID = %q, want %q", got.InvocationID, "inv")
			}
			if !got.Timestamp.Equal(tt.wantTimestamp) {
				t.Errorf("Timestamp = %v, want %v", got.Timestamp, tt.wantTimestamp)
			}
			if diff := cmp.Diff(tt.wantParts, pcPartNames(got)); diff != "" {
				t.Errorf("function response names mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergeParallelResponses_LatestWrites(t *testing.T) {
	backing := []any{"a", "b"}
	capped := make([]any, 1, 2)
	capped[0] = "a"
	sessionList := []any{"a"}
	sessionDict := map[string]any{"x": 1}
	conflictDict := map[string]any{"k": 1}
	typedMap := map[string]string{"a": "x"}
	nestedLast := map[string]any{"n": map[string]any{"b": 2}}
	emptyList := []any{}

	tests := []struct {
		name string
		key  string
		// session is the key's session value; noSession leaves the key out.
		session   any
		noSession bool
		// writes[i] is call i's delta value for key; a nil event is a call
		// that gave no response.
		writes    []any
		nilEvents map[int]bool
		want      any
	}{
		{name: "list: earlier call's Set is the session value", key: "items", session: sessionList, writes: []any{sessionList, []any{"b"}}, want: []any{"a"}},
		{name: "dict: earlier call's Set is the session value", key: "doc", session: sessionDict, writes: []any{sessionDict, map[string]any{"y": 2}}, want: map[string]any{"x": 1, "y": 2}},
		{name: "scalar: never re-applied", key: "c", session: 2, writes: []any{2, 1}, want: 1},
		{name: "key absent from the session", key: "items", noSession: true, writes: []any{[]any{"a"}, []any{"b"}}, want: []any{"b"}},
		{name: "equal list built separately is not the same write", key: "items", session: []any{"a"}, writes: []any{[]any{"a"}, []any{"b"}}, want: []any{"b"}},
		{name: "empty list is never the same write", key: "items", session: emptyList, writes: []any{emptyList, []any{"b"}}, want: []any{"b"}},
		{name: "same array, different length", key: "items", session: backing[:1], writes: []any{backing, []any{"z"}}, want: []any{"z"}},
		{name: "same array and length, different capacity", key: "items", session: capped[:1:1], writes: []any{capped[:1], []any{"z"}}, want: []any{"z"}},
		{name: "conflicting dict key: the latest write wins", key: "doc", session: conflictDict, writes: []any{conflictDict, map[string]any{"k": 2}}, want: map[string]any{"k": 1}},
		{name: "typed map is replaced, not deep merged", key: "drafts", session: typedMap, writes: []any{typedMap, map[string]string{"b": "y"}}, want: map[string]string{"a": "x"}},
		{name: "nested dicts of two Sets merge", key: "doc", session: nestedLast, writes: []any{map[string]any{"n": map[string]any{"a": 1}}, nestedLast}, want: map[string]any{"n": map[string]any{"a": 1, "b": 2}}},
		{name: "nil event in the middle", key: "items", session: sessionList, writes: []any{sessionList, nil, []any{"b"}}, nilEvents: map[int]bool{1: true}, want: []any{"a"}},
		{name: "middle call's Set is the session value", key: "items", session: sessionList, writes: []any{[]any{"x"}, sessionList, []any{"z"}}, want: []any{"a"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
			if err != nil {
				t.Fatal(err)
			}
			sess := created.Session
			if !tt.noSession {
				if err := sess.State().Set(tt.key, tt.session); err != nil {
					t.Fatal(err)
				}
			}
			events := make([]*session.Event, len(tt.writes))
			for i, w := range tt.writes {
				if tt.nilEvents[i] {
					continue
				}
				events[i] = pcResponseEvent(fmt.Sprintf("ev-%d", i), fmt.Sprintf("t%d", i), time.Time{}, map[string]any{tt.key: w})
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{InvocationID: "inv", Session: sess})

			got, err := mergeParallelResponses(ctx, events)
			if err != nil {
				t.Fatalf("mergeParallelResponses() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got.Actions.StateDelta[tt.key]); diff != "" {
				t.Errorf("merged state delta[%q] mismatch (-want +got):\n%s", tt.key, diff)
			}
		})
	}
}

func TestMergeParallelResponses_NilSession(t *testing.T) {
	uctx, _ := pcCountingUUIDs(t.Context(), "merged-id", false)
	ctx := icontext.NewInvocationContext(uctx, icontext.InvocationContextParams{InvocationID: "inv"})
	events := []*session.Event{
		pcResponseEvent("ev-0", "t1", time.Time{}, map[string]any{"k": "a"}),
		pcResponseEvent("ev-1", "t2", time.Time{}, map[string]any{"k": "b"}),
	}

	got, err := mergeParallelResponses(ctx, events)
	if err != nil {
		t.Fatalf("mergeParallelResponses() error = %v", err)
	}
	if got.ID != "merged-id" {
		t.Errorf("ID = %q, want %q", got.ID, "merged-id")
	}
	if diff := cmp.Diff(map[string]any{"k": "b"}, got.Actions.StateDelta); diff != "" {
		t.Errorf("merged state delta mismatch (-want +got):\n%s", diff)
	}
}

var errPCBoom = errors.New("boom")

// pcFailingState is a session state whose Get fails with errPCBoom.
type pcFailingState struct{}

func (pcFailingState) Get(string) (any, error)     { return nil, errPCBoom }
func (pcFailingState) Set(string, any) error       { return nil }
func (pcFailingState) All() iter.Seq2[string, any] { return func(func(string, any) bool) {} }

type pcFailingSession struct{}

func (pcFailingSession) ID() string                { return "s" }
func (pcFailingSession) AppName() string           { return "app" }
func (pcFailingSession) UserID() string            { return "user" }
func (pcFailingSession) State() session.State      { return pcFailingState{} }
func (pcFailingSession) Events() session.Events    { return nil }
func (pcFailingSession) LastUpdateTime() time.Time { return time.Time{} }

func TestMergeParallelResponses_StateReadError(t *testing.T) {
	const key = "user_key_17"
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{InvocationID: "inv", Session: pcFailingSession{}})
	events := []*session.Event{
		pcResponseEvent("ev-0", "t1", time.Time{}, map[string]any{key: []any{"a"}}),
		pcResponseEvent("ev-1", "t2", time.Time{}, map[string]any{key: []any{"b"}}),
	}

	_, err := mergeParallelResponses(ctx, events)
	if err == nil {
		t.Fatal("mergeParallelResponses() error = nil, want the state read error")
	}
	if got, want := err.Error(), "read session state for the merge of parallel calls: boom"; got != want {
		t.Errorf("error text = %q, want %q", got, want)
	}
	if !errors.Is(err, errPCBoom) {
		t.Errorf("errors.Is(err, errPCBoom) = false for %v", err)
	}
	if strings.Contains(err.Error(), key) {
		t.Errorf("error text %q contains the state key", err.Error())
	}
}

func TestMergeEventActions_RewindEndOfAgentAgentState(t *testing.T) {
	yes, no := true, false
	tests := []struct {
		name        string
		base, other session.EventActions
		want        session.EventActions
	}{
		{name: "rewind kept when other is empty", base: session.EventActions{RewindBeforeInvocationID: "r1"}, other: session.EventActions{}, want: session.EventActions{RewindBeforeInvocationID: "r1"}},
		{name: "rewind: the later call wins", base: session.EventActions{RewindBeforeInvocationID: "r1"}, other: session.EventActions{RewindBeforeInvocationID: "r2"}, want: session.EventActions{RewindBeforeInvocationID: "r2"}},
		{name: "end of agent kept when other is nil", base: session.EventActions{EndOfAgent: &yes}, other: session.EventActions{}, want: session.EventActions{EndOfAgent: &yes}},
		{name: "end of agent: a later explicit false wins", base: session.EventActions{EndOfAgent: &yes}, other: session.EventActions{EndOfAgent: &no}, want: session.EventActions{EndOfAgent: &no}},
		{name: "end of agent taken from other", base: session.EventActions{}, other: session.EventActions{EndOfAgent: &yes}, want: session.EventActions{EndOfAgent: &yes}},
		{name: "end of agent stays nil", base: session.EventActions{}, other: session.EventActions{}, want: session.EventActions{}},
		{
			name:  "agent state deep merged",
			base:  session.EventActions{AgentState: map[string]any{"a": 1, "n": map[string]any{"x": 1}}},
			other: session.EventActions{AgentState: map[string]any{"b": 2, "n": map[string]any{"y": 2}}},
			want:  session.EventActions{AgentState: map[string]any{"a": 1, "b": 2, "n": map[string]any{"x": 1, "y": 2}}},
		},
		{
			name:  "agent state: the later call wins a shared key",
			base:  session.EventActions{AgentState: map[string]any{"k": 1, "n": map[string]any{"x": 1}}},
			other: session.EventActions{AgentState: map[string]any{"k": 2, "n": map[string]any{"x": 2}}},
			want:  session.EventActions{AgentState: map[string]any{"k": 2, "n": map[string]any{"x": 2}}},
		},
		{name: "agent state taken from other when base is nil", base: session.EventActions{}, other: session.EventActions{AgentState: map[string]any{"b": 2}}, want: session.EventActions{AgentState: map[string]any{"b": 2}}},
		{name: "agent state kept when other is nil", base: session.EventActions{AgentState: map[string]any{"a": 1}}, other: session.EventActions{}, want: session.EventActions{AgentState: map[string]any{"a": 1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base, other := tt.base, tt.other
			got := mergeEventActions(&base, &other)
			if diff := cmp.Diff(tt.want, *got); diff != "" {
				t.Errorf("mergeEventActions() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// pcSessionContext returns an invocation context over a new in-memory session.
func pcSessionContext(t *testing.T, ctx context.Context) (agent.InvocationContext, session.Session) {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		InvocationID: "inv",
		Agent:        &mockAgent{name: "agent_1"},
		Session:      created.Session,
	}), created.Session
}

func pcCalls(calls ...*genai.FunctionCall) *model.LLMResponse {
	parts := make([]*genai.Part, len(calls))
	for i, c := range calls {
		parts[i] = &genai.Part{FunctionCall: c}
	}
	return &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}
}

func TestHandleFunctionCalls_MergedEventGetsNewID(t *testing.T) {
	uctx, _ := pcCountingUUIDs(t.Context(), "id-", true)
	ctx, _ := pcSessionContext(t, uctx)
	tools := map[string]tool.Tool{
		"one": tcValueTool(t, "one", map[string]any{"v": 1}),
		"two": tcValueTool(t, "two", map[string]any{"v": 2}),
	}
	resp := pcCalls(
		&genai.FunctionCall{ID: "c1", Name: "one", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c2", Name: "two", Args: map[string]any{}},
	)

	got, err := (&Flow{}).handleFunctionCalls(ctx, tools, resp, nil, nil)
	if err != nil {
		t.Fatalf("handleFunctionCalls() error = %v", err)
	}
	if got.ID != "id-3" {
		t.Errorf("merged event ID = %q, want %q", got.ID, "id-3")
	}
}

func TestHandleFunctionCalls_WaitCycleEndsAtContextDeadline(t *testing.T) {
	dctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
	defer cancel()
	ctx, sess := pcSessionContext(t, dctx)
	waiter := tcMust(t)(functiontool.New(functiontool.Config{Name: "waiter", Description: "waits for the context"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}))
	setter := tcMust(t)(functiontool.New(functiontool.Config{Name: "setter", Description: "sets k"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			if err := ctx.State().Set("k", 1); err != nil {
				return nil, err
			}
			return map[string]any{"ok": true}, nil
		}))
	tools := map[string]tool.Tool{"waiter": waiter, "setter": setter}
	resp := pcCalls(
		&genai.FunctionCall{ID: "c0", Name: "waiter", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c1", Name: "setter", Args: map[string]any{}},
	)

	type result struct {
		ev  *session.Event
		err error
	}
	done := make(chan result, 1)
	go func() {
		ev, err := (&Flow{}).handleFunctionCalls(ctx, tools, resp, nil, nil)
		done <- result{ev, err}
	}()
	var res result
	select {
	case res = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handleFunctionCalls() did not return within 5s")
	}
	if res.err != nil {
		t.Fatalf("handleFunctionCalls() error = %v", res.err)
	}
	var setterResp map[string]any
	for _, p := range res.ev.LLMResponse.Content.Parts {
		if p.FunctionResponse != nil && p.FunctionResponse.ID == "c1" {
			setterResp = p.FunctionResponse.Response
		}
	}
	if errText, _ := setterResp["error"].(string); !strings.Contains(errText, "context deadline exceeded") {
		t.Errorf("setter response = %v, want an error containing %q", setterResp, "context deadline exceeded")
	}
	if v, err := sess.State().Get("k"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("session state k = %v, %v; want ErrStateKeyNotExist", v, err)
	}
}
