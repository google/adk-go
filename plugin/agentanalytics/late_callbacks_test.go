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
	"iter"
	"sync"
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
	for _, afterRun := range []bool{false, true} {
		name := "before_end"
		if afterRun {
			name = "after_end"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			emitted := false
			emit := func(ctx agent.InvocationContext) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "user"
				ev.Content = textContent("user", "derived context")
				if _, err := p.OnEventCallback()(ctx, ev); err != nil {
					t.Error(err)
				}
				emitted = true
			}
			observer, err := baseplugin.New(baseplugin.Config{
				Name: "derived_context",
				AfterRunCallback: func(ctx agent.InvocationContext) {
					if afterRun {
						emit(ctx)
					}
				},
				OnEventCallback: func(ctx agent.InvocationContext, _ *session.Event) (*session.Event, error) {
					if !afterRun {
						emit(ctx)
					}
					return nil, nil
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
				break
			}
			if !emitted {
				t.Fatal("derived event callback did not run")
			}
			rows := collectRows(t, requests, func(rows []map[string]any) bool {
				return untilInvocationEnd(rows) && len(rowsOfType(rows, legacyEventType)) == 2
			})
			requireInvocationCompletedOK(t, rows)
			want := placeOf(onlyRowOfType(t, rows, EventInvocationCompleted))
			var derivedRows int
			for _, row := range rowsOfType(rows, legacyEventType) {
				if jsonObjectColumn(t, row, "attributes")["event_author"] == "user" {
					derivedRows++
					if placeOf(row) != want {
						t.Error("derived event does not retain the completed invocation's span identity")
					}
				}
			}
			if derivedRows != 1 {
				t.Fatalf("got %d derived rows, want 1", derivedRows)
			}
		})
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
	for range 65 {
		id := identity{invocationID: now.String(), agentName: "root"}
		tm.invocationSpan(ctx, id, nil)
		tm.endInvocation(ctx, id)
		now = now.Add(time.Millisecond)
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if got := len(tm.evicted); got != 64 {
		t.Errorf("retained %d tombstones, want 64", got)
	}
	if len(tm.invocations) != 0 {
		t.Error("completed invocations are still active")
	}
}
