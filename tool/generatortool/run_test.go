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

package generatortool_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// analyzeCall is the model turn every test of this file starts with.
func analyzeCall() *genai.Content {
	return calls(call("call_1", "analyze", map[string]any{"query": "sales"}))
}

// A1. Fails if the author is not replaced (custom_reporter survives), the
// branch or the node path is missing, the output event gets text content, the
// response event comes before the messages, or InvocationID is not the run's
// (unset, or other-inv kept).
func TestRun_EventStreamShape(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (string, error) {
		first := message("Starting sales...")
		first.Author = "custom_reporter"
		first.InvocationID = "other-inv"
		if err := emit(first); err != nil {
			return "", err
		}
		if err := emit(message("Finished sales.")); err != nil {
			return "", err
		}
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	events := h.events()
	want := []evSummary{
		{Author: "root_agent", Path: "root_agent", Call: "analyze"},
		{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Text: "Starting sales..."},
		{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Text: "Finished sales."},
		{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Output: "ok", NoContent: true},
		{Author: "root_agent", Path: "root_agent", Response: map[string]any{"result": "ok"}},
		{Author: "root_agent", Path: "root_agent", Text: "done"},
	}
	if diff := cmp.Diff(want, summarize(events)); diff != "" {
		t.Fatalf("session events mismatch (-want +got):\n%s", diff)
	}
	inv := events[0].InvocationID
	for i, ev := range events[1:4] {
		if ev.InvocationID == "" || ev.InvocationID != inv {
			t.Errorf("tool event %d InvocationID = %q, want the run's %q", i, ev.InvocationID, inv)
		}
	}
}

// A2. Fails if pending context writes are not merged into the next event, if
// the event's own delta wins over a context write of the same key, or if
// writes are left pending after the last event.
func TestRun_PendingStateRidesOnNextEvent(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (string, error) {
		setState(t, ctx, "a", "1")
		setState(t, ctx, "k", "ctx")
		ev := message("m")
		ev.Actions.StateDelta = map[string]any{"b": 2, "k": "ev"}
		if err := emit(ev); err != nil {
			return "", err
		}
		setState(t, ctx, "c", "3")
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	var deltas []map[string]any
	for _, ev := range authoredBy(h.events(), "analyze") {
		deltas = append(deltas, ev.Actions.StateDelta)
	}
	wantDeltas := []map[string]any{{"a": "1", "b": 2, "k": "ctx"}, {"c": "3"}}
	if diff := cmp.Diff(wantDeltas, deltas); diff != "" {
		t.Errorf("tool event state deltas mismatch (-want +got):\n%s", diff)
	}
	gotState := map[string]any{"a": h.state("a"), "b": h.state("b"), "k": h.state("k"), "c": h.state("c")}
	if diff := cmp.Diff(map[string]any{"a": "1", "b": 2, "k": "ctx", "c": "3"}, gotState); diff != "" {
		t.Errorf("session state mismatch (-want +got):\n%s", diff)
	}
}

// A3. Fails if pending deltas are flushed onto a partial event, which is not
// persisted, so the write would be lost.
func TestRun_PartialEventKeepsStatePending(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		setState(t, ctx, "a", "1")
		p := message("p")
		p.Partial = true
		if err := emit(p); err != nil {
			return nil, err
		}
		return nil, emit(message("m"))
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	yielded := h.mustRun(userText("go"))

	got := map[string]map[string]any{}
	for _, ev := range authoredBy(yielded, "analyze") {
		got[textOf(ev)] = ev.Actions.StateDelta
	}
	if diff := cmp.Diff(map[string]map[string]any{"p": nil, "m": {"a": "1"}}, got, cmp.Comparer(sameDelta)); diff != "" {
		t.Errorf("state delta per tool event mismatch (-want +got):\n%s", diff)
	}
}

// sameDelta treats a nil and an empty state delta as equal.
func sameDelta(a, b map[string]any) bool {
	if len(a) == 0 || len(b) == 0 {
		return len(a) == len(b)
	}
	return cmp.Equal(a, b)
}

// A4. Fails if a write after the last emit, with no result, is not sent as a
// delta-only event.
func TestRun_TrailingStateIsDeltaOnlyEvent(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(message("m")); err != nil {
			return nil, err
		}
		setState(t, ctx, "late", "x")
		return nil, nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	events := h.events()
	tool := authoredBy(events, "analyze")
	if len(tool) != 2 {
		t.Fatalf("tool events = %+v, want 2 (message and delta-only)", summarize(tool))
	}
	last := tool[1]
	if diff := cmp.Diff(map[string]any{"late": "x"}, last.Actions.StateDelta); diff != "" {
		t.Errorf("last tool event state delta mismatch (-want +got):\n%s", diff)
	}
	if last.Content != nil || last.Output != nil {
		t.Errorf("last tool event Content = %v, Output = %v, want both nil", last.Content, last.Output)
	}
	if got := h.state("late"); got != "x" {
		t.Errorf("state late = %v, want x", got)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"result": nil}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A37. Fails if a write after the last emit is dropped when the handler then
// returns an error.
func TestRun_TrailingStateSurvivesHandlerError(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(message("m")); err != nil {
			return nil, err
		}
		setState(t, ctx, "late", "x")
		return nil, errors.New("boom")
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	if got := h.state("late"); got != "x" {
		t.Errorf("state late = %v, want x", got)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"error": "boom"}}}
	if diff := cmp.Diff(want, responses(h.events(), "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A38. Fails if an artifact the handler saves does not ride on its next
// non-partial event, or on the trailing delta-only event when it is saved
// after the last emit.
func TestRun_SavedArtifactRidesOnNextEvent(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if _, err := ctx.Artifacts().Save(ctx, "a.txt", genai.NewPartFromText("a")); err != nil {
			return nil, err
		}
		if err := emit(message("m")); err != nil {
			return nil, err
		}
		if _, err := ctx.Artifacts().Save(ctx, "b.txt", genai.NewPartFromText("b")); err != nil {
			return nil, err
		}
		return nil, nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	var deltas []map[string]int64
	for _, ev := range authoredBy(h.events(), "analyze") {
		deltas = append(deltas, ev.Actions.ArtifactDelta)
	}
	want := []map[string]int64{{"a.txt": 1}, {"b.txt": 1}}
	if diff := cmp.Diff(want, deltas); diff != "" {
		t.Errorf("tool event artifact deltas mismatch (-want +got):\n%s", diff)
	}
}

// A5. Fails if the flow adds a text part to the response event of a plain
// generator tool that sets SkipSummarization, or calls the model again.
func TestRun_SkipSummarizationAddsNoText(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, _ func(*session.Event) error) (map[string]any, error) {
		ctx.Actions().SkipSummarization = true
		return map[string]any{"done": "x"}, nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall()}})

	h.mustRun(userText("go"))

	ev := responseEvent(t, h.events())
	if !ev.Actions.SkipSummarization {
		t.Error("response event SkipSummarization = false, want true")
	}
	if got := len(ev.Content.Parts); got != 1 {
		t.Errorf("response event has %d parts, want 1 (only the function response)", got)
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
}

// A6. Fails if SkipSummarization reaches the response event only when the
// handler succeeds.
func TestRun_SkipSummarizationSurvivesHandlerError(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, _ func(*session.Event) error) (any, error) {
		ctx.Actions().SkipSummarization = true
		return nil, errors.New("boom")
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall()}})

	h.mustRun(userText("go"))

	ev := responseEvent(t, h.events())
	if diff := cmp.Diff(map[string]any{"error": "boom"}, ev.Content.Parts[0].FunctionResponse.Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	if !ev.Actions.SkipSummarization {
		t.Error("response event SkipSummarization = false, want true")
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
}

// responseEvent returns the one event that holds a function response.
func responseEvent(t *testing.T, events []*session.Event) *session.Event {
	t.Helper()
	var found []*session.Event
	for _, ev := range events {
		if len(responses([]*session.Event{ev}, "analyze")) > 0 {
			found = append(found, ev)
		}
	}
	if len(found) != 1 {
		t.Fatalf("events with a function response = %d, want 1; events: %+v", len(found), summarize(events))
	}
	return found[0]
}

// A12. Fails if RunToolCall hands the on-tool-error callbacks the workflow's
// NodeRunError wrapper instead of the handler's own error.
func TestRun_OnToolErrorSeesHandlerError(t *testing.T) {
	errBoom := errors.New("boom")
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(message("m")); err != nil {
			return nil, err
		}
		return nil, errBoom
	})
	var recorded error
	h := newHarness(t, harnessConfig{
		tools:     []tool.Tool{analyze},
		responses: []*genai.Content{analyzeCall(), modelText("done")},
		onError: []llmagent.OnToolErrorCallback{func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
			recorded = err
			return map[string]any{"recovered": true}, nil
		}},
	})

	h.mustRun(userText("go"))

	if !errors.Is(recorded, errBoom) || recorded.Error() != "boom" {
		t.Errorf("on-tool-error callback error = %v, want the handler's error %q", recorded, "boom")
	}
	events := h.events()
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"recovered": true}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	wantTool := []evSummary{{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Text: "m"}}
	if diff := cmp.Diff(wantTool, summarize(authoredBy(events, "analyze"))); diff != "" {
		t.Errorf("tool events mismatch (-want +got):\n%s", diff)
	}
}

// A13. Fails if the before-tool or after-tool callbacks are skipped for a
// generator tool, or run in another order.
func TestRun_BeforeAfterCallbacksAround(t *testing.T) {
	var mu sync.Mutex
	var order []string
	record := func(s string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, s)
	}
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (map[string]any, error) {
		record("handler")
		return map[string]any{"rows": 42}, nil
	})
	var beforeTool string
	var beforeArgs, afterResult map[string]any
	h := newHarness(t, harnessConfig{
		tools:     []tool.Tool{analyze},
		responses: []*genai.Content{analyzeCall(), modelText("done")},
		before: []llmagent.BeforeToolCallback{func(_ agent.Context, tl tool.Tool, args map[string]any) (map[string]any, error) {
			record("before")
			beforeTool, beforeArgs = tl.Name(), args
			return nil, nil
		}},
		after: []llmagent.AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
			record("after")
			afterResult = result
			return map[string]any{"rows": 43}, nil
		}},
	})

	h.mustRun(userText("go"))

	if diff := cmp.Diff([]string{"before", "handler", "after"}, order); diff != "" {
		t.Errorf("call order mismatch (-want +got):\n%s", diff)
	}
	if beforeTool != "analyze" {
		t.Errorf("before-tool callback tool = %q, want analyze", beforeTool)
	}
	if diff := cmp.Diff(map[string]any{"query": "sales"}, beforeArgs); diff != "" {
		t.Errorf("before-tool callback args mismatch (-want +got):\n%s", diff)
	}
	// functiontool hands callbacks the result in its JSON form.
	if diff := cmp.Diff(map[string]any{"rows": float64(42)}, afterResult); diff != "" {
		t.Errorf("after-tool callback result mismatch (-want +got):\n%s", diff)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"rows": 43}}}
	if diff := cmp.Diff(want, responses(h.events(), "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A14. Fails if a handler error ends the run instead of being answered to the
// model.
func TestRun_HandlerErrorAnsweredToModel(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
		return nil, errors.New("boom")
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	events := h.events()
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"error": "boom"}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	if got := h.modelRequests(); got != 2 {
		t.Errorf("model requests = %d, want 2", got)
	}
	if got := textOf(events[len(events)-1]); got != "done" {
		t.Errorf("last event text = %q, want done", got)
	}
}

// A15. Fails if a handler panic escapes the runner or poisons later calls.
func TestRun_HandlerPanicRecovered(t *testing.T) {
	var runs atomic.Int32
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (string, error) {
		if runs.Add(1) == 1 {
			panic("x")
		}
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{
		analyzeCall(), modelText("done 1"),
		calls(call("call_2", "analyze", map[string]any{"query": "sales"})), modelText("done 2"),
	}})

	h.mustRun(userText("first"))
	h.mustRun(userText("second"))

	want := []*genai.FunctionResponse{
		{ID: "call_1", Name: "analyze", Response: map[string]any{"error": "Invoking `analyze()` failed with an internal error."}},
		{ID: "call_2", Name: "analyze", Response: map[string]any{"result": "ok"}},
	}
	if diff := cmp.Diff(want, responses(h.events(), "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A16. Fails if the node starts before the mandatory-argument check.
func TestRun_MissingMandatoryArgumentSkipsBody(t *testing.T) {
	var runs atomic.Int32
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (string, error) {
		runs.Add(1)
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{
		calls(call("call_1", "analyze", map[string]any{})), modelText("done"),
	}})

	h.mustRun(userText("go"))

	if got := runs.Load(); got != 0 {
		t.Errorf("handler ran %d times, want 0", got)
	}
	events := h.events()
	if got := authoredBy(events, "analyze"); len(got) != 0 {
		t.Errorf("tool events = %+v, want none", summarize(got))
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{
		"error": "Invoking `analyze()` failed as the following mandatory input parameters are not present:\nquery\nYou could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters.",
	}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A20. Fails if a second validation pass rejects or defaults an optional
// argument the model left out.
func TestRun_OptionalArgumentAbsentIsNil(t *testing.T) {
	type taggedArgs struct {
		Query string  `json:"query"`
		Tag   *string `json:"tag,omitempty"`
	}
	var got taggedArgs
	var runs atomic.Int32
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, args taggedArgs, _ func(*session.Event) error) (string, error) {
		runs.Add(1)
		got = args
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	if runs.Load() != 1 || got.Query != "sales" || got.Tag != nil {
		t.Errorf("handler runs = %d, args = {Query: %q, Tag: %v}, want 1 run with {Query: \"sales\", Tag: nil}", runs.Load(), got.Query, got.Tag)
	}
}

// A24. Fails if emit ignores the run's cancellation and persists an event
// after it.
func TestRun_CancelStopsEmit(t *testing.T) {
	emitted := make(chan struct{})
	secondErr := make(chan error, 1)
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(message("first")); err != nil {
			return nil, err
		}
		close(emitted)
		<-ctx.Done()
		secondErr <- emit(message("second"))
		return nil, ctx.Err()
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go func() {
		<-emitted
		cancel()
	}()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = h.runCtx(ctx, userText("go")) // The run ends with the cancellation.
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("run did not end within 5s of the cancellation")
	}

	select {
	case err := <-secondErr:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("emit after cancel error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("handler did not emit within 5s of the cancellation")
	}
	var texts []string
	for _, ev := range authoredBy(h.events(), "analyze") {
		texts = append(texts, textOf(ev))
	}
	if diff := cmp.Diff([]string{"first"}, texts); diff != "" {
		t.Errorf("tool event texts mismatch (-want +got):\n%s", diff)
	}
}

// A35. Fails if emit still sends after the call has ended: a goroutine the
// handler left behind emits once the handler returned.
func TestRun_EmitAfterHandlerReturnedSendsNothing(t *testing.T) {
	lateErr := make(chan error, 1)
	release := make(chan struct{})
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		go func() {
			<-release
			lateErr <- emit(message("late"))
		}()
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))
	close(release)

	select {
	case err := <-lateErr:
		if err == nil {
			t.Error("emit after the handler returned error = nil, want an error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("late emit did not return within 5s")
	}
	for _, ev := range authoredBy(h.events(), "analyze") {
		if textOf(ev) == "late" {
			t.Error("event \"late\" was persisted after the call ended")
		}
	}
}

// A25. Fails if emit(nil) dereferences nil or sends a blank event.
func TestRun_EmitNilEvent(t *testing.T) {
	var nilErr error
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (string, error) {
		nilErr = emit(nil)
		return "ok", nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	if nilErr == nil {
		t.Error("emit(nil) error = nil, want an error")
	}
	events := h.events()
	wantTool := []evSummary{{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Output: "ok", NoContent: true}}
	if diff := cmp.Diff(wantTool, summarize(authoredBy(events, "analyze"))); diff != "" {
		t.Errorf("tool events mismatch (-want +got):\n%s", diff)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"result": "ok"}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}

// A26. Fails if emit is not safe for concurrent use: a race, or lost or
// reordered events.
func TestRun_EmitConcurrentUse(t *testing.T) {
	const goroutines, perGoroutine = 8, 10
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		var wg sync.WaitGroup
		for g := range goroutines {
			wg.Go(func() {
				for j := range perGoroutine {
					if err := emit(message(fmt.Sprintf("g%d-%d", g, j))); err != nil {
						t.Errorf("emit error = %v", err)
						return
					}
				}
			})
		}
		wg.Wait()
		return nil, nil
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	perG := map[string][]string{}
	seen := map[string]bool{}
	for _, ev := range authoredBy(h.events(), "analyze") {
		text := textOf(ev)
		if ev.Branch != "analyze@call_1" {
			t.Errorf("event %q branch = %q, want analyze@call_1", text, ev.Branch)
		}
		if seen[text] {
			t.Errorf("event %q sent twice", text)
		}
		seen[text] = true
		g, _, _ := strings.Cut(text, "-")
		perG[g] = append(perG[g], text)
	}
	if len(seen) != goroutines*perGoroutine {
		t.Errorf("distinct tool events = %d, want %d", len(seen), goroutines*perGoroutine)
	}
	for g := range goroutines {
		var want []string
		for j := range perGoroutine {
			want = append(want, fmt.Sprintf("g%d-%d", g, j))
		}
		if diff := cmp.Diff(want, perG[fmt.Sprintf("g%d", g)]); diff != "" {
			t.Errorf("events of goroutine %d out of order (-want +got):\n%s", g, diff)
		}
	}
}

// A29. Fails if a call with no output is answered with an empty response or
// sends an event for the missing output.
func TestRun_NoOutputAnswersNilResult(t *testing.T) {
	tests := []struct {
		name     string
		result   any
		wantResp map[string]any
		wantTool []evSummary
	}{
		{name: "nil", result: nil, wantResp: map[string]any{"result": nil}},
		{
			name: "value", result: "x", wantResp: map[string]any{"result": "x"},
			wantTool: []evSummary{{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Output: "x", NoContent: true}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
				return tc.result, nil
			})
			h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

			h.mustRun(userText("go"))

			events := h.events()
			want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: tc.wantResp}}
			if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
				t.Errorf("function responses mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantTool, summarize(authoredBy(events, "analyze"))); diff != "" {
				t.Errorf("tool events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A36. Fails if a nil map, slice or pointer result counts as an output: Func's
// doc says a nil interface, pointer, map or slice means no output.
func TestRun_NilTypedResultIsNoOutput(t *testing.T) {
	type rows struct{ N int }
	tests := []struct {
		name string
		tool func(t *testing.T) tool.Tool
	}{
		{name: "map", tool: func(t *testing.T) tool.Tool {
			return mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (map[string]any, error) {
				return nil, nil
			})
		}},
		{name: "slice", tool: func(t *testing.T) tool.Tool {
			return mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) ([]string, error) {
				return nil, nil
			})
		}},
		{name: "pointer", tool: func(t *testing.T) tool.Tool {
			return mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (*rows, error) {
				return nil, nil
			})
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, harnessConfig{tools: []tool.Tool{tc.tool(t)}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

			h.mustRun(userText("go"))

			if diff := cmp.Diff([]evSummary(nil), summarize(authoredBy(h.events(), "analyze"))); diff != "" {
				t.Errorf("tool events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A31. Fails if a bare workflow.ErrNodeInterrupted with no request for input
// makes the call succeed with a nil result, or is treated as a pause.
func TestRun_BareNodeInterruptedIsHandlerError(t *testing.T) {
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
		return nil, workflow.ErrNodeInterrupted
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	events := h.events()
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{
		"error": `generator tool "analyze": handler returned workflow.ErrNodeInterrupted without requesting input`,
	}}}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	if got := h.modelRequests(); got != 2 {
		t.Errorf("model requests = %d, want 2", got)
	}
	if got := functionCalls(events, workflow.WorkflowInputFunctionCallName); len(got) != 0 {
		t.Errorf("request-input calls = %v, want none", got)
	}
}
