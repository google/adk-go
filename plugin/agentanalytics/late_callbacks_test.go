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

package agentanalytics

import (
	"fmt"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

func TestRunner_AfterRunEventKeepsInvocationSpan(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	emitted := false
	observer, err := baseplugin.New(baseplugin.Config{
		Name: "derived_context",
		AfterRunCallback: func(ctx agent.InvocationContext) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = "derived"
			ev.LLMResponse = model.LLMResponse{Content: textContent("model", "derived context"), TurnComplete: true}
			if _, err := p.OnEventCallback()(ctx, ev); err != nil {
				t.Error(err)
			}
			emitted = true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	root, err := agent.New(agent.Config{
		Name: "plain",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "plain"
				ev.LLMResponse = model.LLMResponse{Content: textContent("model", "answer"), TurnComplete: true}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName: "test-app", Agent: root, SessionService: svc, AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: []*baseplugin.Plugin{p, observer}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "question"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !emitted {
		t.Fatal("derived event callback did not run")
	}
	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return untilInvocationEnd(rows) && len(rowsOfType(rows, EventAgentResponse)) == 2
	})
	requireInvocationCompletedOK(t, rows)
	want := placeOf(onlyRowOfType(t, rows, EventInvocationCompleted))
	var derivedRows int
	for _, row := range rowsOfType(rows, EventAgentResponse) {
		if jsonObjectColumn(t, row, "attributes")["event_author"] == "derived" {
			derivedRows++
			if placeOf(row) != want {
				t.Error("derived event does not retain the completed invocation's span identity")
			}
		}
	}
	if derivedRows != 1 {
		t.Fatalf("got %d derived rows, want 1", derivedRows)
	}
}

func TestTraceManager_LateOperationsKeepTheirSpans(t *testing.T) {
	for _, kind := range []string{"agent", "model", "tool"} {
		t.Run(kind, func(t *testing.T) {
			ctx := t.Context()
			tm := newTraceManager()
			id := identity{invocationID: "completed", agentName: "root"}
			tm.invocationSpan(ctx, id, nil)
			tm.endInvocation(ctx, id)
			var started, ended rowSpan
			switch kind {
			case "agent":
				started = tm.pushAgentSpan(ctx, id)
				ended = tm.popAgentSpan(ctx, id)
			case "model":
				started = tm.pushLLMSpan(ctx, id)
				ended = tm.popLLMSpan(ctx, id, false)
			case "tool":
				started = tm.pushToolSpan(ctx, id, "call")
				ended, _ = tm.popToolSpan(ctx, id, "call", false)
			}
			if started.spanID != ended.spanID || started.parentSpanID != ended.parentSpanID || ended.latency == nil {
				t.Error("late operation lost its span identity or latency")
			}
			if len(trackedInvocations(tm)) != 0 {
				t.Error("late operation recreated active invocation tracking")
			}
		})
	}
}

func TestTraceManager_EndAdoptsTraceBeforeRetainingIdentity(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	id := identity{invocationID: "completed", agentName: "root"}
	tm.invocationSpan(ctx, id, nil)
	traceID, _ := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	spanID, _ := trace.SpanIDFromHex("00f067aa0ba902b7")
	endCtx := trace.ContextWithSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID}))
	ended := tm.endInvocation(endCtx, id)
	if ended.traceID != traceID.String() {
		t.Error("terminal row did not adopt its ambient trace")
	}
	late := tm.eventSpan(ctx, id, "user", "")
	if late.traceID != ended.traceID || late.spanID != ended.spanID {
		t.Error("late callback lost the trace adopted by the terminal row")
	}
}

func TestTraceManager_LateCallbacksDoNotRestartInvocation(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	id := identity{invocationID: "completed", agentName: "root"}
	started := tm.invocationSpan(ctx, id, nil)
	tm.endInvocation(ctx, id)
	activeID := identity{invocationID: "active", agentName: "root"}
	active := tm.invocationSpan(ctx, activeID, nil)
	var wg sync.WaitGroup
	for range 32 {
		wg.Go(func() {
			row := tm.eventSpan(ctx, id, "user", "")
			if row.traceID != started.traceID || row.spanID != started.spanID || row.parentSpanID != started.parentSpanID {
				t.Error("late callback changed invocation span identity")
			}
		})
	}
	wg.Wait()
	if got := trackedInvocations(tm); len(got) != 1 || got[0] != activeID.invocationID {
		t.Error("late callbacks recreated active invocation state or evicted the live run")
	}
	if got := tm.endInvocation(ctx, activeID); got.spanID != active.spanID || got.latency == nil {
		t.Error("late callbacks discarded the live run's span or latency")
	}
}

func TestTraceManager_NewRunReusesInvocationID(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	id := identity{invocationID: "resumed", agentName: "root"}
	first := tm.invocationSpan(ctx, id, nil)
	tm.endInvocation(ctx, id)
	second := tm.invocationSpan(ctx, id, nil)
	if second.spanID == first.spanID {
		t.Error("new run reused the finished invocation's root span")
	}
	if len(trackedInvocations(tm)) != 1 {
		t.Error("new run was not tracked")
	}
	if got := tm.endInvocation(ctx, id); got.spanID != second.spanID || got.latency == nil {
		t.Error("new run lost its root span or latency")
	}
}

func TestTraceManager_StartRestoresEvictedInvocation(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	id := identity{invocationID: "evicted", agentName: "root"}
	first := tm.invocationSpan(ctx, id, nil)
	tm.invocationSpan(ctx, identity{invocationID: "other"}, nil)
	if got := tm.invocationSpan(ctx, id, nil); got.spanID != first.spanID || got.traceID != first.traceID {
		t.Error("start callback discarded the evicted active invocation's identity")
	}
}

func TestTraceManager_CompletedTombstonesStayBounded(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }
	var ids []identity
	var starts []rowSpan
	for range 65 {
		id := identity{invocationID: now.String(), agentName: "root"}
		ids = append(ids, id)
		starts = append(starts, tm.invocationSpan(ctx, id, nil))
		tm.endInvocation(ctx, id)
		now = now.Add(time.Millisecond)
	}
	if len(tm.invocations) != 0 {
		t.Error("completed invocations are still active")
	}
	for i, id := range ids {
		got := tm.eventSpan(ctx, id, "user", "")
		if retained := got.spanID == starts[i].spanID; retained != (i > 0) {
			t.Errorf("completion %d retained = %t, want %t", i, retained, i > 0)
		}
	}
}

func TestTraceManager_LateCallbackRefreshesRetention(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	id := identity{invocationID: "still-receiving", agentName: "root"}
	started := tm.invocationSpan(ctx, id, nil)
	tm.endInvocation(ctx, id)
	for i := range 63 {
		other := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
		tm.invocationSpan(ctx, other, nil)
		tm.endInvocation(ctx, other)
	}
	tm.eventSpan(ctx, id, "user", "")
	other := identity{invocationID: "new-completion", agentName: "root"}
	tm.invocationSpan(ctx, other, nil)
	tm.endInvocation(ctx, other)
	if got := tm.eventSpan(ctx, id, "user", ""); got.spanID != started.spanID {
		t.Error("recent late callback did not refresh completed identity retention")
	}
	if len(trackedInvocations(tm)) != 0 {
		t.Error("late callback recreated active invocation tracking")
	}
}

func TestTraceManager_RepeatedEndPreservesLateCall(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	id := identity{invocationID: "completed", agentName: "root"}
	started := tm.invocationSpan(ctx, id, nil)
	firstEnd := tm.endInvocation(ctx, id)
	toolStart := tm.pushToolSpan(ctx, id, "call")
	secondEnd := tm.endInvocation(ctx, id)
	toolEnd, _ := tm.popToolSpan(ctx, id, "call", false)
	if secondEnd.spanID != started.spanID || firstEnd.spanID != started.spanID {
		t.Error("repeated completion changed the invocation's identity")
	}
	if secondEnd.latency != nil {
		t.Error("repeated completion reported a second invocation latency")
	}
	if toolEnd.spanID != toolStart.spanID || toolEnd.latency == nil {
		t.Error("repeated completion discarded the open late tool call")
	}
	if len(trackedInvocations(tm)) != 0 {
		t.Error("repeated completion recreated active invocation tracking")
	}
}

func TestTraceManager_CompletionsPreserveMidRunEviction(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }
	id := identity{invocationID: "still-running", agentName: "root"}
	started := tm.invocationSpan(ctx, id, nil)
	now = now.Add(tm.ttl + time.Second)
	tm.inserts = sweepInterval - 1
	other := identity{invocationID: "sweep", agentName: "root"}
	tm.invocationSpan(ctx, other, nil)
	tm.endInvocation(ctx, other)
	if _, ok := tm.evicted[id.invocationID]; !ok {
		t.Fatal("fixture did not evict the unfinished invocation")
	}
	// Use the reviewed default-capacity workload, independent of maxTracked.
	for i := range 4096 {
		now = now.Add(time.Millisecond)
		completed := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
		tm.invocationSpan(ctx, completed, nil)
		tm.endInvocation(ctx, completed)
	}
	if _, ok := tm.evicted[id.invocationID]; !ok {
		t.Error("ordinary completions discarded an unfinished invocation's identity")
	}
	ended := tm.endInvocation(ctx, id)
	if ended.traceID != started.traceID || ended.spanID != started.spanID || ended.parentSpanID != started.parentSpanID {
		t.Error("completion lost the mid-run evicted invocation's root span")
	}
	if ended.latency != nil {
		t.Error("completion reported a latency after mid-run eviction")
	}
}

func TestTraceManager_OpenLateCallSurvivesCompletionPressure(t *testing.T) {
	for _, kind := range []string{"agent", "model", "tool"} {
		t.Run(kind, func(t *testing.T) {
			ctx := t.Context()
			tm := newTraceManager()
			tm.maxTracked = 1
			now := time.Unix(1000, 0)
			tm.now = func() time.Time { return now }
			id := identity{invocationID: "late-operation", agentName: "root"}
			tm.invocationSpan(ctx, id, nil)
			tm.endInvocation(ctx, id)
			var started, ended rowSpan
			switch kind {
			case "agent":
				started = tm.pushAgentSpan(ctx, id)
			case "model":
				started = tm.pushLLMSpan(ctx, id)
			case "tool":
				started = tm.pushToolSpan(ctx, id, "call")
			}
			now = now.Add(tm.ttl + time.Second)
			for i := range 65 {
				completed := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
				tm.invocationSpan(ctx, completed, nil)
				tm.endInvocation(ctx, completed)
			}
			switch kind {
			case "agent":
				ended = tm.popAgentSpan(ctx, id)
			case "model":
				ended = tm.popLLMSpan(ctx, id, false)
			case "tool":
				ended, _ = tm.popToolSpan(ctx, id, "call", false)
			}
			if ended.spanID != started.spanID || ended.parentSpanID != started.parentSpanID || ended.latency == nil {
				t.Error("completion pressure split an in-flight late operation")
			}
			if len(trackedInvocations(tm)) != 0 {
				t.Error("late completion recreated active invocation tracking")
			}
		})
	}
}

func TestTraceManager_StartingLateSpanSurvivesCompletionPressure(t *testing.T) {
	for _, kind := range []string{"agent", "model", "tool"} {
		t.Run(kind, func(t *testing.T) {
			ctx := t.Context()
			tm := newTraceManager()
			tm.maxTracked = 1
			id := identity{invocationID: "starting-late-span", agentName: "root"}
			tm.invocationSpan(ctx, id, nil)
			tm.endInvocation(ctx, id)
			ready, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			t.Cleanup(unblock)
			var clockCalls atomic.Int64
			tm.now = func() time.Time {
				if clockCalls.Add(1) == 2 {
					close(ready)
					<-release
				}
				return time.Unix(1000, 0)
			}
			started := make(chan rowSpan, 1)
			go func() {
				switch kind {
				case "agent":
					started <- tm.pushAgentSpan(ctx, id)
				case "model":
					started <- tm.pushLLMSpan(ctx, id)
				case "tool":
					started <- tm.pushToolSpan(ctx, id, "call")
				}
			}()
			select {
			case <-ready:
			case <-time.After(5 * time.Second):
				t.Fatal("late span did not reach its start after invocation lookup")
			}
			for i := range 65 {
				other := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
				tm.invocationSpan(ctx, other, nil)
				tm.endInvocation(ctx, other)
			}
			unblock()
			span := <-started
			var ended rowSpan
			switch kind {
			case "agent":
				ended = tm.popAgentSpan(ctx, id)
			case "model":
				ended = tm.popLLMSpan(ctx, id, false)
			case "tool":
				ended, _ = tm.popToolSpan(ctx, id, "call", false)
			}
			if span.spanID != ended.spanID || ended.latency == nil {
				t.Error("pruning between invocation lookup and span start split the late operation")
			}
			if len(trackedInvocations(tm)) != 0 {
				t.Error("late operation returned to active invocation tracking")
			}
		})
	}
}

func TestTraceManager_ClosedLateStateCanBePruned(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	id := identity{invocationID: "idle-late-state", agentName: "root"}
	first := tm.invocationSpan(ctx, id, nil)
	tm.endInvocation(ctx, id)
	tm.pushToolSpan(ctx, id, "call")
	tm.popToolSpan(ctx, id, "call", false)
	for i := range 64 {
		other := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
		tm.invocationSpan(ctx, other, nil)
		tm.endInvocation(ctx, other)
	}
	if late := tm.eventSpan(ctx, id, "user", ""); late.spanID == first.spanID {
		t.Error("idle initialized late state was retained beyond completed capacity")
	}
}

func TestTraceManager_FullOpenCompletedCacheStaysBounded(t *testing.T) {
	ctx := t.Context()
	tm := newTraceManager()
	tm.maxTracked = 1
	var ids []identity
	var spans []rowSpan
	for i := range 64 {
		id := identity{invocationID: fmt.Sprintf("open-%d", i), agentName: "root"}
		tm.invocationSpan(ctx, id, nil)
		tm.endInvocation(ctx, id)
		ids = append(ids, id)
		spans = append(spans, tm.pushToolSpan(ctx, id, "call"))
	}
	id := identity{invocationID: "new-idle-completion", agentName: "root"}
	started := tm.invocationSpan(ctx, id, nil)
	tm.endInvocation(ctx, id)
	if got := tm.eventSpan(ctx, id, "user", ""); got.spanID == started.spanID {
		t.Error("full cache retained a new idle completion beyond its capacity")
	}
	for i, openID := range ids {
		ended, _ := tm.popToolSpan(ctx, openID, "call", false)
		if ended.spanID != spans[i].spanID || ended.latency == nil {
			t.Errorf("open late operation %d was discarded to retain an idle completion", i)
		}
	}
}
