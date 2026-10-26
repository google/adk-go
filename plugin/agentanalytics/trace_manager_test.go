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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// spanIDPattern matches the 16 lowercase hex digits of an OpenTelemetry
// span ID and of adk-python's uuid4().hex[:16].
var spanIDPattern = regexp.MustCompile(`^[0-9a-f]{16}$`)

// tracePlace holds the trace_id, span_id and parent_span_id columns of a
// row.
type tracePlace struct {
	trace, span, parent string
}

func placeOf(row map[string]any) tracePlace {
	column := func(name string) string {
		s, _ := row[name].(string)
		return s
	}
	return tracePlace{trace: column("trace_id"), span: column("span_id"), parent: column("parent_span_id")}
}

// rowLatency decodes the latency_ms column of row. ok is false if the row
// has no latency_ms column.
func rowLatency(t *testing.T, row map[string]any) (latency map[string]int64, ok bool) {
	t.Helper()
	v, ok := jsonColumnValue(t, row, columnLatencyMS)
	if !ok {
		return nil, false
	}
	obj, isObject := v.(map[string]any)
	if !isObject {
		t.Fatalf("%v row: latency_ms is %T, want a JSON object", row["event_type"], v)
	}
	latency = map[string]int64{}
	for key, raw := range obj {
		n, isNumber := raw.(json.Number)
		if !isNumber {
			t.Fatalf("%v row: latency_ms.%s is %T, want a number", row["event_type"], key, raw)
		}
		ms, err := n.Int64()
		if err != nil {
			t.Fatalf("%v row: latency_ms.%s = %s, want an integer", row["event_type"], key, n)
		}
		latency[key] = ms
	}
	return latency, true
}

// mustLatency returns the decoded latency_ms column of row.
func mustLatency(t *testing.T, row map[string]any) map[string]int64 {
	t.Helper()
	latency, ok := rowLatency(t, row)
	if !ok {
		t.Fatalf("%v row has no latency_ms", row["event_type"])
	}
	return latency
}

// functionCallIDOf returns attributes.adk.function_call_id of row.
func functionCallIDOf(t *testing.T, row map[string]any) string {
	t.Helper()
	v, _ := lookup(jsonObjectColumn(t, row, "attributes"), "adk", "function_call_id")
	s, _ := v.(string)
	return s
}

// withFunctionCall returns a copy of ctx for the tool call id.
func withFunctionCall(ctx *mockInvocationContext, id string) *mockInvocationContext {
	c := *ctx
	c.functionCallID = id
	return &c
}

// TestTraceManager_ConcurrentSiblingToolsRace starts 20 tool calls of an
// agent before any of them ends, as ADK does with the function calls of a
// model response, each of which runs in a goroutine of its own. Every call
// must be a child of the agent, never of a sibling, and must end its own
// span. Under -race, the test also checks the locking.
func TestTraceManager_ConcurrentSiblingToolsRace(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	const calls = 20
	lookupTool := &fakeTool{name: "lookup"}
	ctxs := make([]*mockInvocationContext, calls)
	for i := range ctxs {
		ctxs[i] = withFunctionCall(mCtx, fmt.Sprintf("call-%d", i))
	}
	// concurrently runs f for every call, each in a goroutine of its own,
	// and returns once all of them are done.
	concurrently := func(f func(*mockInvocationContext) error) {
		t.Helper()
		var wg sync.WaitGroup
		errs := make(chan error, calls)
		for _, c := range ctxs {
			wg.Go(func() { errs <- f(c) })
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("tool callback error: %v", err)
			}
		}
	}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.check(t)
	concurrently(func(c *mockInvocationContext) error {
		_, err := p.BeforeToolCallback()(c, lookupTool, map[string]any{"call": c.functionCallID})
		return err
	})
	concurrently(func(c *mockInvocationContext) error {
		_, err := p.AfterToolCallback()(c, lookupTool, nil, map[string]any{"call": c.functionCallID}, nil)
		return err
	})
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentStart := placeOf(onlyRowOfType(t, rows, EventAgentStarting))
	starts := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventToolStarting) {
		starts[functionCallIDOf(t, row)] = placeOf(row)
	}
	if len(starts) != calls {
		t.Fatalf("got TOOL_STARTING rows for %d calls, want %d", len(starts), calls)
	}
	spans := map[string]bool{}
	for id, start := range starts {
		if start.parent != agentStart.span {
			t.Errorf("TOOL_STARTING of %s: parent_span_id = %q, want the agent span %q", id, start.parent, agentStart.span)
		}
		if !spanIDPattern.MatchString(start.span) {
			t.Errorf("TOOL_STARTING of %s: span_id = %q, want 16 hex digits", id, start.span)
		}
		spans[start.span] = true
	}
	if len(spans) != calls {
		t.Errorf("the %d calls have %d distinct spans, want one each", calls, len(spans))
	}
	completes := rowsOfType(rows, EventToolCompleted)
	if len(completes) != calls {
		t.Fatalf("got %d TOOL_COMPLETED rows, want %d", len(completes), calls)
	}
	for _, row := range completes {
		id := functionCallIDOf(t, row)
		if got, want := placeOf(row), starts[id]; got != want {
			t.Errorf("TOOL_COMPLETED of %s is at %+v, want the place of its TOOL_STARTING row %+v", id, got, want)
		}
		mustLatency(t, row)
	}
	agentEnd := onlyRowOfType(t, rows, EventAgentCompleted)
	if got := placeOf(agentEnd); got != agentStart {
		t.Errorf("AGENT_COMPLETED is at %+v, want the place of AGENT_STARTING %+v", got, agentStart)
	}
	mustLatency(t, agentEnd)
}

// TestTraceManager_OnToolErrorThenAfterTool_NoDoublePop replays the
// callbacks of a failed tool call: ADK runs AfterToolCallback after
// OnToolErrorCallback, with the same error. The second callback must find
// the span that the first one ended instead of ending the agent's span.
func TestTraceManager_OnToolErrorThenAfterTool_NoDoublePop(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	toolCtx := withFunctionCall(mCtx, "call-1")
	lookupTool := &fakeTool{name: "lookup"}
	args := map[string]any{"query": "weather"}
	toolErr := errors.New("lookup failed")
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeToolCallback()(toolCtx, lookupTool, args))
	errs.add(p.OnToolErrorCallback()(toolCtx, lookupTool, args, toolErr))
	errs.add(p.AfterToolCallback()(toolCtx, lookupTool, args, nil, toolErr))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	want := []string{EventAgentStarting, EventToolStarting, EventToolError, EventAgentCompleted, EventInvocationCompleted}
	if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	agentStart, toolStart, toolError, agentEnd, invocationEnd := placeOf(rows[0]), placeOf(rows[1]), placeOf(rows[2]), placeOf(rows[3]), placeOf(rows[4])
	if toolStart.parent != agentStart.span {
		t.Errorf("TOOL_STARTING parent_span_id = %q, want the agent span %q", toolStart.parent, agentStart.span)
	}
	if toolError != toolStart {
		t.Errorf("TOOL_ERROR is at %+v, want the place of TOOL_STARTING %+v", toolError, toolStart)
	}
	mustLatency(t, rows[2])
	if agentEnd != agentStart {
		t.Errorf("AGENT_COMPLETED is at %+v, want the place of AGENT_STARTING %+v", agentEnd, agentStart)
	}
	mustLatency(t, rows[3])
	if invocationEnd.span != agentStart.parent {
		t.Errorf("INVOCATION_COMPLETED span_id = %q, want the invocation span %q", invocationEnd.span, agentStart.parent)
	}
	mustLatency(t, rows[4])
}

// TestTraceManager_AmbientOTelTraceIDInherited runs an invocation inside an
// OpenTelemetry span. Every row continues the ambient trace, and the spans
// nest as invocation > agent > model call and tool call, with the
// invocation span under the ambient span.
func TestTraceManager_AmbientOTelTraceIDInherited(t *testing.T) {
	const (
		traceHex = "4bf92f3577b34da6a3ce929d0e0e4736"
		spanHex  = "00f067aa0ba902b7"
	)
	traceID, _ := trace.TraceIDFromHex(traceHex)
	spanID, _ := trace.SpanIDFromHex(spanHex)
	p, requests, mCtx := setupTestPlugin(t)
	mCtx.ctx = trace.ContextWithSpanContext(mCtx.ctx, trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID}))
	toolCtx := withFunctionCall(mCtx, "call-1")
	lookupTool := &fakeTool{name: "lookup"}

	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello")))
	errs.add(p.BeforeRunCallback()(mCtx))
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}))
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "calling lookup")}, nil))
	errs.add(p.BeforeToolCallback()(toolCtx, lookupTool, nil))
	errs.add(p.AfterToolCallback()(toolCtx, lookupTool, nil, map[string]any{"answer": "sunny"}, nil))
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:          "event-1",
		Author:      "test-agent",
		LLMResponse: model.LLMResponse{Content: textContent("model", "It is sunny.")},
	}))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	invocation := placeOf(onlyRowOfType(t, rows, EventInvocationStarting))
	agentSpan := placeOf(onlyRowOfType(t, rows, EventAgentStarting))
	llmSpan := placeOf(onlyRowOfType(t, rows, EventLLMRequest))
	toolSpan := placeOf(onlyRowOfType(t, rows, EventToolStarting))
	spans := map[string]bool{}
	for _, s := range []string{invocation.span, agentSpan.span, llmSpan.span, toolSpan.span} {
		if !spanIDPattern.MatchString(s) || s == spanHex {
			t.Errorf("span_id %q: want 16 hex digits of a span of the invocation", s)
		}
		spans[s] = true
	}
	if len(spans) != 4 {
		t.Errorf("the invocation, agent, model call and tool call have %d distinct spans, want 4", len(spans))
	}

	inInvocation := tracePlace{trace: traceHex, span: invocation.span, parent: spanHex}
	inAgent := tracePlace{trace: traceHex, span: agentSpan.span, parent: invocation.span}
	wantPlaces := map[string]tracePlace{
		EventUserMessageReceived: inInvocation,
		EventInvocationStarting:  inInvocation,
		EventInvocationCompleted: inInvocation,
		EventAgentStarting:       inAgent,
		EventAgentCompleted:      inAgent,
		EventLLMRequest:          {trace: traceHex, span: llmSpan.span, parent: agentSpan.span},
		EventLLMResponse:         {trace: traceHex, span: llmSpan.span, parent: agentSpan.span},
		EventToolStarting:        {trace: traceHex, span: toolSpan.span, parent: agentSpan.span},
		EventToolCompleted:       {trace: traceHex, span: toolSpan.span, parent: agentSpan.span},
	}
	endsSpan := map[string]bool{EventInvocationCompleted: true, EventAgentCompleted: true, EventLLMResponse: true, EventToolCompleted: true}
	sawEvent := false
	for _, row := range rows {
		eventType, _ := row["event_type"].(string)
		attrs := jsonObjectColumn(t, row, "attributes")
		want, ok := wantPlaces[eventType]
		// Rows that describe the session event belong to the agent that
		// wrote it.
		if sourceID, _ := lookup(attrs, "adk", "source_event_id"); sourceID == "event-1" {
			want, ok, sawEvent = inAgent, true, true
		}
		if !ok {
			t.Errorf("unexpected %s row", eventType)
			continue
		}
		if got := placeOf(row); got != want {
			t.Errorf("%s row is at %+v, want %+v", eventType, got, want)
		}
		if _, hasLatency := rowLatency(t, row); hasLatency != endsSpan[eventType] {
			t.Errorf("%s row has latency_ms = %v, want %v", eventType, hasLatency, endsSpan[eventType])
		}
		checkJSONAt(t, attrs, `"test-agent"`, "root_agent_name")
	}
	if !sawEvent {
		t.Errorf("no row describes the session event; event types: %v", eventTypes(rows))
	}
}

// TestTraceManager_StreamingTTFTAndSingleMarshaledLatency streams a model
// response in two chunks. The chunks only time the first token: the final
// response writes the single LLM_RESPONSE row, whose latency_ms column is a
// JSON object, marshaled once, with the time to first token.
func TestTraceManager_StreamingTTFTAndSingleMarshaledLatency(t *testing.T) {
	const gap = 20 * time.Millisecond
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}))
	time.Sleep(gap)
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Partial: true, Content: textContent("model", "Hel")}, nil))
	time.Sleep(gap)
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Partial: true, Content: textContent("model", "lo")}, nil))
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "Hello"), TurnComplete: true}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventLLMRequest, EventLLMResponse, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	request, response := rows[0], rows[1]
	if got, want := placeOf(response), placeOf(request); got != want {
		t.Errorf("LLM_RESPONSE is at %+v, want the place of LLM_REQUEST %+v", got, want)
	}
	raw, ok := response[columnLatencyMS].(string)
	if !ok {
		t.Fatalf("LLM_RESPONSE latency_ms = %v (%T), want JSON text", response[columnLatencyMS], response[columnLatencyMS])
	}
	var decoded any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		t.Fatalf("latency_ms %q is not JSON: %v", raw, err)
	}
	if _, isObject := decoded.(map[string]any); !isObject {
		t.Fatalf("latency_ms = %s, want a JSON object, not JSON text encoded twice", raw)
	}
	latency := mustLatency(t, response)
	if got, want := slices.Sorted(maps.Keys(latency)), []string{"time_to_first_token_ms", "total_ms"}; !slices.Equal(got, want) {
		t.Fatalf("latency_ms keys = %v, want %v", got, want)
	}
	ttft, total := latency["time_to_first_token_ms"], latency["total_ms"]
	if wantMin := gap.Milliseconds(); ttft < wantMin || total-ttft < wantMin {
		t.Errorf("latency_ms = %s, want time_to_first_token_ms >= %d (the first chunk) and total_ms >= time_to_first_token_ms + %d (the last chunk)", raw, wantMin, wantMin)
	}
	checkJSONAt(t, jsonObjectColumn(t, response, "attributes"), `false`, "partial")
	checkJSON(t, "content", jsonObjectColumn(t, response, "content"), `{"response":"text: 'Hello'"}`)
}

// clockedPlugin is an analyticsPlugin whose trace manager reads a clock that
// the test sets, so that latencies are exact. Its batch processor never
// starts: the rows of the plugin's callbacks wait in the processor's queue
// until takeRows takes them.
type clockedPlugin struct {
	*analyticsPlugin
	now time.Time
}

func newClockedPlugin(t *testing.T) *clockedPlugin {
	t.Helper()
	config := applyDefaults(DefaultConfig())
	processor, err := NewBatchProcessor(t.Context(), nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(func() {
		// Close writes the rows that are left, which needs a write client.
		processor.drainQueue()
		processor.Close()
	})
	c := &clockedPlugin{now: time.Unix(1000, 0)}
	traces := newTraceManager()
	traces.now = func() time.Time { return c.now }
	c.analyticsPlugin = &analyticsPlugin{config: config, processor: processor, traces: traces}
	return c
}

// takeRows returns the rows that the callbacks queued since the last call.
func (c *clockedPlugin) takeRows() []map[string]any {
	return c.processor.drainQueue()
}

// firstTokenTimed reports whether the running model call of ctx's agent has a
// time to first token.
func (c *clockedPlugin) firstTokenTimed(t *testing.T, ctx *mockInvocationContext) bool {
	t.Helper()
	id := identityOf(ctx)
	inv := c.traces.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	rec := inv.llmSpans[llmKey(id)]
	if rec == nil {
		t.Fatal("no model call is running")
	}
	return !rec.firstToken.IsZero()
}

// TestAfterModel_FirstTokenIsFirstGeneratedContent streams a model response
// whose first chunk, 10 ms into the call, carries only metadata. That chunk
// says nothing about when the model began to answer, so the time to first
// token is that of the first chunk with generated content, at 800 ms, and not
// that of the metadata.
func TestAfterModel_FirstTokenIsFirstGeneratedContent(t *testing.T) {
	usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}
	tests := []struct {
		name string
		// leading is the chunk that arrives before any generated content.
		leading *model.LLMResponse
	}{
		{"usage only", &model.LLMResponse{Partial: true, UsageMetadata: usage}},
		{"role only", &model.LLMResponse{Partial: true, Content: &genai.Content{Role: "model"}}},
		{"empty parts", &model.LLMResponse{Partial: true, Content: &genai.Content{Role: "model", Parts: []*genai.Part{}}}},
		{"empty part", &model.LLMResponse{Partial: true, Content: &genai.Content{Role: "model", Parts: []*genai.Part{{}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newClockedPlugin(t)
			mCtx := &mockInvocationContext{ctx: t.Context(), agentName: "agent", invocationID: "inv-first-token"}
			start := p.now
			at := func(offset time.Duration) { p.now = start.Add(offset) }

			var errs callbackErrors
			errs.add(p.beforeModel(mCtx, &model.LLMRequest{Model: "gemini-test"}))
			at(10 * time.Millisecond)
			errs.add(p.afterModel(mCtx, tt.leading, nil))
			errs.check(t)
			if p.firstTokenTimed(t, mCtx) {
				t.Fatal("a chunk without generated content timed the first token")
			}

			at(800 * time.Millisecond)
			errs.add(p.afterModel(mCtx, &model.LLMResponse{Partial: true, Content: textContent("model", "Hel")}, nil))
			errs.check(t)
			if !p.firstTokenTimed(t, mCtx) {
				t.Fatal("the first chunk with generated content did not time the first token")
			}

			at(900 * time.Millisecond)
			errs.add(p.afterModel(mCtx, &model.LLMResponse{Partial: true, Content: textContent("model", "lo")}, nil))
			at(1000 * time.Millisecond)
			errs.add(p.afterModel(mCtx, &model.LLMResponse{Content: textContent("model", "Hello"), TurnComplete: true}, nil))
			errs.check(t)

			response := onlyRowOfType(t, p.takeRows(), EventLLMResponse)
			want := map[string]int64{"time_to_first_token_ms": 800, "total_ms": 1000}
			if diff := cmp.Diff(want, mustLatency(t, response)); diff != "" {
				t.Errorf("LLM_RESPONSE latency_ms mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestAfterModel_MetadataOnlyChunksLeaveFirstTokenToFinalResponse streams a
// model response whose chunks all carry only metadata. No chunk times the
// first token, so the call gets a time to first token equal to its total
// latency, as a call that is not streamed does.
func TestAfterModel_MetadataOnlyChunksLeaveFirstTokenToFinalResponse(t *testing.T) {
	p := newClockedPlugin(t)
	mCtx := &mockInvocationContext{ctx: t.Context(), agentName: "agent", invocationID: "inv-metadata-only"}
	start := p.now

	var errs callbackErrors
	errs.add(p.beforeModel(mCtx, &model.LLMRequest{Model: "gemini-test"}))
	p.now = start.Add(10 * time.Millisecond)
	errs.add(p.afterModel(mCtx, &model.LLMResponse{Partial: true, UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}}, nil))
	p.now = start.Add(1000 * time.Millisecond)
	errs.add(p.afterModel(mCtx, &model.LLMResponse{Content: textContent("model", "Hello"), TurnComplete: true}, nil))
	errs.check(t)

	response := onlyRowOfType(t, p.takeRows(), EventLLMResponse)
	want := map[string]int64{"time_to_first_token_ms": 1000, "total_ms": 1000}
	if diff := cmp.Diff(want, mustLatency(t, response)); diff != "" {
		t.Errorf("LLM_RESPONSE latency_ms mismatch (-want +got):\n%s", diff)
	}
}

// TestTraceManager_UnknownToolErrorKeepsAgentSpan replays the call of a tool
// that does not exist, for which ADK runs only OnToolErrorCallback. The
// TOOL_ERROR row gets a span of its own under the agent, and the agent's
// span stays open.
func TestTraceManager_UnknownToolErrorKeepsAgentSpan(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.OnToolErrorCallback()(withFunctionCall(mCtx, "call-1"), &fakeTool{name: "missing"}, nil, errors.New("tool not found")))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	want := []string{EventAgentStarting, EventToolError, EventAgentCompleted, EventInvocationCompleted}
	if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	agentStart, toolError, agentEnd := placeOf(rows[0]), placeOf(rows[1]), placeOf(rows[2])
	if toolError.parent != agentStart.span || toolError.span == agentStart.span || !spanIDPattern.MatchString(toolError.span) {
		t.Errorf("TOOL_ERROR is at %+v, want a span of its own under the agent span %q", toolError, agentStart.span)
	}
	if latency, ok := rowLatency(t, rows[1]); ok {
		t.Errorf("TOOL_ERROR of a call that never started has latency_ms %v", latency)
	}
	if agentEnd != agentStart {
		t.Errorf("AGENT_COMPLETED is at %+v, want the place of AGENT_STARTING %+v", agentEnd, agentStart)
	}
	mustLatency(t, rows[2])
}

func TestTraceManager_ModelCallSpans(t *testing.T) {
	request := &model.LLMRequest{Model: "gemini-test"}
	response := &model.LLMResponse{Content: textContent("model", "hi")}

	t.Run("response without streaming", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeModelCallback()(mCtx, request))
		time.Sleep(5 * time.Millisecond)
		errs.add(p.AfterModelCallback()(mCtx, response, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		latency := mustLatency(t, onlyRowOfType(t, rows, EventLLMResponse))
		// As in adk-python, the whole response arrives with the first token.
		if ttft, ok := latency["time_to_first_token_ms"]; !ok || ttft != latency["total_ms"] {
			t.Errorf("latency_ms = %v, want time_to_first_token_ms equal to total_ms", latency)
		}
	})

	t.Run("model error", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeModelCallback()(mCtx, request))
		errs.add(p.OnModelErrorCallback()(mCtx, request, errors.New("model unavailable")))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationError))
		llmError := onlyRowOfType(t, rows, EventLLMError)
		if got, want := placeOf(llmError), placeOf(onlyRowOfType(t, rows, EventLLMRequest)); got != want {
			t.Errorf("LLM_ERROR is at %+v, want the place of LLM_REQUEST %+v", got, want)
		}
		if latency := mustLatency(t, llmError); !slices.Equal(slices.Collect(maps.Keys(latency)), []string{"total_ms"}) {
			t.Errorf("LLM_ERROR latency_ms = %v, want total_ms only", latency)
		}
	})

	t.Run("response after a model error", func(t *testing.T) {
		// When a plugin turns a model error into a response, ADK runs
		// AfterModelCallback for a call that OnModelErrorCallback ended.
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeModelCallback()(mCtx, request))
		errs.add(p.OnModelErrorCallback()(mCtx, request, errors.New("model unavailable")))
		errs.add(p.AfterModelCallback()(mCtx, response, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		req, llmErr, res := onlyRowOfType(t, rows, EventLLMRequest), onlyRowOfType(t, rows, EventLLMError), onlyRowOfType(t, rows, EventLLMResponse)
		if got, want := placeOf(res), placeOf(req); got != want {
			t.Errorf("LLM_RESPONSE is at %+v, want the place of LLM_REQUEST %+v", got, want)
		}
		if got, want := placeOf(llmErr), placeOf(req); got != want {
			t.Errorf("LLM_ERROR is at %+v, want the place of LLM_REQUEST %+v", got, want)
		}
		mustLatency(t, llmErr)
		if latency, ok := rowLatency(t, res); ok {
			t.Errorf("LLM_RESPONSE of an ended call has latency_ms %v", latency)
		}
	})

	t.Run("request that gets no response", func(t *testing.T) {
		// A BeforeModelCallback that answers a request ends the call
		// without AfterModelCallback. The next request replaces it.
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeModelCallback()(mCtx, request))
		errs.add(p.BeforeModelCallback()(mCtx, request))
		errs.add(p.AfterModelCallback()(mCtx, response, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		reqs := rowsOfType(rows, EventLLMRequest)
		if len(reqs) != 2 {
			t.Fatalf("got %d LLM_REQUEST rows, want 2", len(reqs))
		}
		if got, want := placeOf(onlyRowOfType(t, rows, EventLLMResponse)), placeOf(reqs[1]); got != want {
			t.Errorf("LLM_RESPONSE is at %+v, want the place of the second LLM_REQUEST %+v", got, want)
		}
	})
}

// TestTraceManager_AfterAgentWithoutBeforeAgent ends an agent whose start
// was not seen. The row gets a span of its own under the invocation, whose
// span stays open.
func TestTraceManager_AfterAgentWithoutBeforeAgent(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello")))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	want := []string{EventUserMessageReceived, EventAgentCompleted, EventInvocationCompleted}
	if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	invocation, agentEnd, invocationEnd := placeOf(rows[0]), placeOf(rows[1]), placeOf(rows[2])
	if agentEnd.parent != invocation.span || agentEnd.span == invocation.span {
		t.Errorf("AGENT_COMPLETED is at %+v, want a span of its own under the invocation span %q", agentEnd, invocation.span)
	}
	if latency, ok := rowLatency(t, rows[1]); ok {
		t.Errorf("AGENT_COMPLETED of an agent that never started has latency_ms %v", latency)
	}
	if invocationEnd != invocation {
		t.Errorf("INVOCATION_COMPLETED is at %+v, want the invocation span %+v", invocationEnd, invocation)
	}
	mustLatency(t, rows[2])
}

// TestTraceManager_InvocationsHaveTracesOfTheirOwn runs two invocations
// outside any OpenTelemetry span: each one starts a trace of its own.
func TestTraceManager_InvocationsHaveTracesOfTheirOwn(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	other := *mCtx
	other.invocationID = "inv-2"
	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello")))
	errs.add(p.OnUserMessageCallback()(&other, textContent("user", "hi")))
	errs.check(t)
	p.AfterRunCallback()(mCtx)
	p.AfterRunCallback()(&other)

	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rowsOfType(rows, EventInvocationCompleted)) == 2
	})
	traces := map[string]string{}
	for _, row := range rows {
		invocationID, _ := row["invocation_id"].(string)
		got := placeOf(row)
		if !eventIDPattern.MatchString(got.trace) || !spanIDPattern.MatchString(got.span) || got.parent != "" {
			t.Errorf("%v row of %s is at %+v, want a 32-digit trace and a 16-digit span without parent", row["event_type"], invocationID, got)
		}
		if seen, ok := traces[invocationID]; ok && seen != got.trace {
			t.Errorf("rows of %s have trace_id %q and %q, want one trace", invocationID, seen, got.trace)
		}
		traces[invocationID] = got.trace
	}
	if len(traces) != 2 || traces["inv-id"] == traces["inv-2"] {
		t.Errorf("trace_id by invocation = %v, want a trace for each of the two invocations", traces)
	}
}

// TestTraceManager_LatencyColumnDenied leaves latency_ms out when
// PayloadColumnDenylist names it. The rows keep their spans.
func TestTraceManager_LatencyColumnDenied(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.PayloadColumnDenylist = []string{columnLatencyMS} })
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	for _, row := range rows {
		if v, ok := row[columnLatencyMS]; ok {
			t.Errorf("%v row has latency_ms %v, want none", row["event_type"], v)
		}
	}
	if got, want := placeOf(onlyRowOfType(t, rows, EventAgentCompleted)), placeOf(onlyRowOfType(t, rows, EventAgentStarting)); got != want || !spanIDPattern.MatchString(got.span) {
		t.Errorf("AGENT_COMPLETED is at %+v, want the agent span %+v", got, want)
	}
}

// TestTraceManager_ParallelBranches runs the sub-agents of a parallel agent:
// ADK runs them concurrently, on branches named after the parallel agent,
// and they can end in any order.
func TestTraceManager_ParallelBranches(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	id := func(agentName, branch string) identity {
		return identity{invocationID: "inv", agentName: agentName, branch: branch}
	}
	invocation := tm.invocationSpan(ctx, id("par", ""), nil)
	par := tm.pushAgentSpan(ctx, id("par", ""))
	a := tm.pushAgentSpan(ctx, id("a", "par.a"))
	// ab shares the string prefix "par.a" with branch "par.a", but is a
	// sibling branch under par rather than a sub-branch of a.
	ab := tm.pushAgentSpan(ctx, id("ab", "par.ab"))
	b := tm.pushAgentSpan(ctx, id("b", "par.b"))
	// a1 is a sub-agent of a, such as a step of a sequential agent, which
	// keeps the branch of a.
	a1 := tm.pushAgentSpan(ctx, id("a1", "par.a"))
	bModel := tm.pushLLMSpan(ctx, id("b", "par.b"))
	a1Tool := tm.pushToolSpan(ctx, id("a1", "par.a"), "call-1")
	a1ToolEnd, _ := tm.popToolSpan(ctx, id("a1", "par.a"), "call-1", false)
	a1End := tm.popAgentSpan(ctx, id("a1", "par.a"))
	// a ends while b, which started after it, is still running.
	aEnd := tm.popAgentSpan(ctx, id("a", "par.a"))
	abEnd := tm.popAgentSpan(ctx, id("ab", "par.ab"))
	bModelEnd := tm.popLLMSpan(ctx, id("b", "par.b"), true)
	bTool := tm.pushToolSpan(ctx, id("b", "par.b"), "call-2")
	bEnd := tm.popAgentSpan(ctx, id("b", "par.b"))
	parEnd := tm.popAgentSpan(ctx, id("par", ""))
	invocationEnd := tm.endInvocation(ctx, id("par", ""))

	parents := []struct {
		name          string
		child, parent rowSpan
	}{
		{"par", par, invocation},
		{"a", a, par},
		{"ab", ab, par},
		{"b", b, par},
		{"a1", a1, a},
		{"model call of b", bModel, b},
		{"tool call of a1", a1Tool, a1},
		{"tool call of b", bTool, b},
	}
	for _, tt := range parents {
		if tt.child.parentSpanID != tt.parent.spanID {
			t.Errorf("%s: parent span %q, want %q", tt.name, tt.child.parentSpanID, tt.parent.spanID)
		}
	}
	ends := []struct {
		name       string
		start, end rowSpan
	}{
		{"tool call of a1", a1Tool, a1ToolEnd},
		{"a1", a1, a1End},
		{"a", a, aEnd},
		{"ab", ab, abEnd},
		{"model call of b", bModel, bModelEnd},
		{"b", b, bEnd},
		{"par", par, parEnd},
		{"invocation", invocation, invocationEnd},
	}
	for _, tt := range ends {
		if tt.end.spanID != tt.start.spanID || tt.end.parentSpanID != tt.start.parentSpanID {
			t.Errorf("%s ended at span %q under %q, want %q under %q", tt.name, tt.end.spanID, tt.end.parentSpanID, tt.start.spanID, tt.start.parentSpanID)
		}
		if tt.end.latency == nil {
			t.Errorf("%s ended without latency", tt.name)
		}
		if tt.end.traceID != invocation.traceID {
			t.Errorf("%s: trace %q, want %q", tt.name, tt.end.traceID, invocation.traceID)
		}
	}
}

// spanCounts counts the spans that an invocation tracks.
type spanCounts struct {
	agents, models, tools, erroredTools int
}

// openSpans returns the spans that the invocation invocationID tracks,
// apart from its own: the running agents, model calls and tool calls, and
// the tool calls that OnToolErrorCallback ended.
func openSpans(t *testing.T, tm *traceManager, invocationID string) spanCounts {
	t.Helper()
	tm.mu.Lock()
	inv := tm.invocations[invocationID]
	tm.mu.Unlock()
	if inv == nil {
		t.Fatalf("invocation %q is not tracked", invocationID)
	}
	inv.mu.Lock()
	defer inv.mu.Unlock()
	return spanCounts{
		agents:       len(inv.agentStack) - 1,
		models:       len(inv.llmSpans),
		tools:        len(inv.toolSpans),
		erroredTools: len(inv.completedTools),
	}
}

// TestTraceManager_AgentNesting nests agents on branches: the parallel
// agent fan runs inner, itself a parallel agent, whose sub-agents run on
// branches named after both. worker runs on two of these branches at once.
// After fan, the coordinator runs summary.
func TestTraceManager_AgentNesting(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	id := func(agentName, branch string) identity {
		return identity{invocationID: "inv", agentName: agentName, branch: branch}
	}
	const (
		innerBranch = "fan.inner"
		xBranch     = "fan.inner.inner.x"
		w1Branch    = "fan.inner.inner.w1"
		w2Branch    = "fan.inner.inner.w2"
	)
	lookupKey := toolKey(id("worker", w1Branch), "", "lookup")
	invocation := tm.invocationSpan(ctx, id("coordinator", ""), nil)
	coordinator := tm.pushAgentSpan(ctx, id("coordinator", ""))
	fan := tm.pushAgentSpan(ctx, id("fan", ""))
	inner := tm.pushAgentSpan(ctx, id("inner", innerBranch))
	x := tm.pushAgentSpan(ctx, id("x", xBranch))
	w1 := tm.pushAgentSpan(ctx, id("worker", w1Branch))
	w2 := tm.pushAgentSpan(ctx, id("worker", w2Branch))
	w1Model := tm.pushLLMSpan(ctx, id("worker", w1Branch))
	w2Model := tm.pushLLMSpan(ctx, id("worker", w2Branch))
	w1Tool := tm.pushToolSpan(ctx, id("worker", w1Branch), lookupKey)

	events := []struct {
		name           string
		author, branch string
		want           rowSpan
	}{
		{"event of worker on w1", "worker", w1Branch, w1},
		{"event of worker on w2", "worker", w2Branch, w2},
		// An author that does not run on the event's branch is found by
		// name: its most recent run.
		{"event of worker on another branch", "worker", "fan.elsewhere", w2},
		{"event of x", "x", xBranch, x},
		{"event of the user", "user", "", invocation},
	}
	for _, tt := range events {
		// Event callbacks get the agent that the runner runs.
		got := tm.eventSpan(ctx, id("coordinator", ""), tt.author, tt.branch)
		if got.spanID != tt.want.spanID || got.parentSpanID != tt.want.parentSpanID {
			t.Errorf("%s: span %q under %q, want %q under %q", tt.name, got.spanID, got.parentSpanID, tt.want.spanID, tt.want.parentSpanID)
		}
		if got.latency != nil {
			t.Errorf("%s: latency %v, want none", tt.name, got.latency)
		}
	}

	w1ModelEnd := tm.popLLMSpan(ctx, id("worker", w1Branch), true)
	w1ToolEnd, _ := tm.popToolSpan(ctx, id("worker", w1Branch), lookupKey, false)
	w1End := tm.popAgentSpan(ctx, id("worker", w1Branch))
	w2ModelEnd := tm.popLLMSpan(ctx, id("worker", w2Branch), true)
	xEnd := tm.popAgentSpan(ctx, id("x", xBranch))
	w2End := tm.popAgentSpan(ctx, id("worker", w2Branch))
	innerEnd := tm.popAgentSpan(ctx, id("inner", innerBranch))
	fanEnd := tm.popAgentSpan(ctx, id("fan", ""))
	summary := tm.pushAgentSpan(ctx, id("summary", ""))
	// A model call of the coordinator while summary runs, on the same
	// branch, belongs to the coordinator.
	coordinatorModel := tm.pushLLMSpan(ctx, id("coordinator", ""))
	coordinatorModelEnd := tm.popLLMSpan(ctx, id("coordinator", ""), true)
	summaryEnd := tm.popAgentSpan(ctx, id("summary", ""))
	coordinatorEnd := tm.popAgentSpan(ctx, id("coordinator", ""))
	if got := openSpans(t, tm, "inv"); got != (spanCounts{}) {
		t.Errorf("spans still open after all of them ended: %+v, want none", got)
	}
	invocationEnd := tm.endInvocation(ctx, id("coordinator", ""))

	parents := []struct {
		name          string
		child, parent rowSpan
	}{
		{"coordinator", coordinator, invocation},
		{"fan", fan, coordinator},
		{"inner", inner, fan},
		{"x", x, inner},
		// worker on w1 starts after x, on a sibling branch of x.
		{"worker on w1", w1, inner},
		{"worker on w2", w2, inner},
		{"model call of worker on w1", w1Model, w1},
		{"model call of worker on w2", w2Model, w2},
		{"tool call of worker on w1", w1Tool, w1},
		{"summary", summary, coordinator},
		{"model call of the coordinator", coordinatorModel, coordinator},
	}
	for _, tt := range parents {
		if tt.child.parentSpanID != tt.parent.spanID {
			t.Errorf("%s: parent span %q, want %q", tt.name, tt.child.parentSpanID, tt.parent.spanID)
		}
	}
	ends := []struct {
		name       string
		start, end rowSpan
	}{
		{"model call of worker on w1", w1Model, w1ModelEnd},
		{"tool call of worker on w1", w1Tool, w1ToolEnd},
		{"worker on w1", w1, w1End},
		{"model call of worker on w2", w2Model, w2ModelEnd},
		{"x", x, xEnd},
		{"worker on w2", w2, w2End},
		{"inner", inner, innerEnd},
		{"fan", fan, fanEnd},
		{"model call of the coordinator", coordinatorModel, coordinatorModelEnd},
		{"summary", summary, summaryEnd},
		{"coordinator", coordinator, coordinatorEnd},
		{"invocation", invocation, invocationEnd},
	}
	for _, tt := range ends {
		if tt.end.spanID != tt.start.spanID || tt.end.parentSpanID != tt.start.parentSpanID {
			t.Errorf("%s ended at span %q under %q, want %q under %q", tt.name, tt.end.spanID, tt.end.parentSpanID, tt.start.spanID, tt.start.parentSpanID)
		}
		if tt.end.latency == nil {
			t.Errorf("%s ended without latency", tt.name)
		}
	}
}

// TestTraceManager_FailedToolCallBookkeeping ends failed tool calls as ADK
// does: with OnToolErrorCallback, and then AfterToolCallback for the same
// call.
func TestTraceManager_FailedToolCallBookkeeping(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	id := identity{invocationID: "inv", agentName: "agent"}
	tm.invocationSpan(ctx, id, nil)
	agentSpan := tm.pushAgentSpan(ctx, id)
	// fail runs OnToolErrorCallback and then AfterToolCallback for the call
	// key, and returns the place of the TOOL_ERROR row.
	fail := func(key string) rowSpan {
		t.Helper()
		errorEnd, closedByError := tm.popToolSpan(ctx, id, key, true)
		if closedByError {
			t.Errorf("%s: OnToolErrorCallback found the call ended by an earlier error", key)
		}
		afterEnd, closedByError := tm.popToolSpan(ctx, id, key, false)
		if !closedByError {
			t.Errorf("%s: AfterToolCallback did not find the call ended by OnToolErrorCallback", key)
		}
		if afterEnd.spanID != errorEnd.spanID || afterEnd.parentSpanID != errorEnd.parentSpanID {
			t.Errorf("%s: AfterToolCallback at span %q under %q, want the span of the error %q under %q", key, afterEnd.spanID, afterEnd.parentSpanID, errorEnd.spanID, errorEnd.parentSpanID)
		}
		if afterEnd.latency != nil {
			t.Errorf("%s: AfterToolCallback has latency %v, want none", key, afterEnd.latency)
		}
		return errorEnd
	}

	// The span of a call that fails ends once, with the error.
	start := tm.pushToolSpan(ctx, id, "call-1")
	if got := fail("call-1"); got.spanID != start.spanID || got.parentSpanID != agentSpan.spanID || got.latency == nil {
		t.Errorf("error of a started call at span %q under %q with latency %v, want the span of the call %q under %q with latency", got.spanID, got.parentSpanID, got.latency, start.spanID, agentSpan.spanID)
	}
	// A call whose start was not seen, for example because the
	// BeforeToolCallback of another plugin failed, gets a span of its own.
	if got := fail("call-2"); got.spanID == agentSpan.spanID || got.parentSpanID != agentSpan.spanID || got.latency != nil {
		t.Errorf("error of a call that never started at span %q under %q with latency %v, want a span of its own under %q without latency", got.spanID, got.parentSpanID, got.latency, agentSpan.spanID)
	}
	// Without function call IDs, the calls of a tool by an agent share a
	// key. Consecutive OnToolErrorCallback-only calls (for a tool that never
	// started) each allocate a fresh spanID rather than consuming the
	// previous error's tombstone, and the next started call drops the stale
	// record.
	key := toolKey(id, "", "lookup")
	firstErr, firstClosed := tm.popToolSpan(ctx, id, key, true)
	secondErr, secondClosed := tm.popToolSpan(ctx, id, key, true)
	if firstClosed || secondClosed || firstErr.spanID == secondErr.spanID {
		t.Errorf("consecutive OnToolErrorCallback-only spans = (%q/%t, %q/%t), want distinct spans with closedByError=false",
			firstErr.spanID, firstClosed, secondErr.spanID, secondClosed)
	}
	next := tm.pushToolSpan(ctx, id, key)
	if got, closedByError := tm.popToolSpan(ctx, id, key, false); closedByError || got.spanID != next.spanID || got.latency == nil {
		t.Errorf("end of the next call at span %q (closed by error: %t) with latency %v, want the span of the call %q with latency", got.spanID, closedByError, got.latency, next.spanID)
	}
	if got, want := openSpans(t, tm, "inv"), (spanCounts{agents: 1}); got != want {
		t.Errorf("open spans = %+v, want %+v", got, want)
	}
}

// TestTraceManager_RecoveredToolErrorEmitsCompletedAndAgentResponse replays a
// failed tool call that a subsequent OnToolErrorCallback recovers (err == nil
// in AfterToolCallback). TOOL_STARTING, TOOL_ERROR, and TOOL_COMPLETED share
// the tool call's span under the agent span (with latency_ms reported once on
// TOOL_ERROR), while AGENT_RESPONSE (when in FinalResponseToolNames) is placed
// on the enclosing agent span.
func TestTraceManager_RecoveredToolErrorEmitsCompletedAndAgentResponse(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.FinalResponseToolNames = []string{"final_answer"} })
	toolCtx := withFunctionCall(mCtx, "call-recovered")
	final := &fakeTool{name: "final_answer"}
	args := map[string]any{"answer": "draft"}
	recoveredResult := map[string]any{"answer": "fallback-42"}
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeToolCallback()(toolCtx, final, args))
	errs.add(p.OnToolErrorCallback()(toolCtx, final, args, errors.New("transient error")))
	errs.add(p.AfterToolCallback()(toolCtx, final, args, recoveredResult, nil))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	want := []string{
		EventAgentStarting,
		EventToolStarting,
		EventToolError,
		EventToolCompleted,
		EventAgentResponse,
		EventAgentCompleted,
		EventInvocationCompleted,
	}
	if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	agentStart := placeOf(rows[0])
	toolStart := placeOf(rows[1])
	if toolStart.parent != agentStart.span {
		t.Errorf("TOOL_STARTING parent_span_id = %q, want agent span %q", toolStart.parent, agentStart.span)
	}
	for _, evType := range []string{EventToolError, EventToolCompleted} {
		row := onlyRowOfType(t, rows, evType)
		if got := placeOf(row); got != toolStart {
			t.Errorf("%s is at %+v, want %+v", evType, got, toolStart)
		}
	}
	if got := placeOf(onlyRowOfType(t, rows, EventAgentResponse)); got != agentStart {
		t.Errorf("AGENT_RESPONSE is at %+v, want agent span %+v", got, agentStart)
	}
	mustLatency(t, onlyRowOfType(t, rows, EventToolError))
	for _, evType := range []string{EventToolCompleted, EventAgentResponse} {
		if latency, ok := rowLatency(t, onlyRowOfType(t, rows, evType)); ok {
			t.Errorf("%s after OnToolErrorCallback has latency_ms %v, want none", evType, latency)
		}
	}
	checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolCompleted), "content"),
		`{"tool":"final_answer","result":{"answer":"fallback-42"},"tool_origin":"UNKNOWN"}`)
	if got := placeOf(onlyRowOfType(t, rows, EventAgentCompleted)); got != agentStart {
		t.Errorf("AGENT_COMPLETED is at %+v, want %+v", got, agentStart)
	}
}

// TestTraceManager_FinalResponseToolRow ends a call of a final response
// tool. TOOL_COMPLETED closes the tool call's span with latency_ms, while
// AGENT_RESPONSE is placed on the enclosing agent span without latency_ms.
func TestTraceManager_FinalResponseToolRow(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.FinalResponseToolNames = []string{"final_answer"} })
	mCtx.functionCallID = "call-7"
	final := &fakeTool{name: "final_answer"}
	answer := map[string]any{"answer": "42"}
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeToolCallback()(mCtx, final, answer))
	errs.add(p.AfterToolCallback()(mCtx, final, answer, map[string]any{}, nil))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentStart := placeOf(onlyRowOfType(t, rows, EventAgentStarting))
	toolStart := placeOf(onlyRowOfType(t, rows, EventToolStarting))
	completed, response := onlyRowOfType(t, rows, EventToolCompleted), onlyRowOfType(t, rows, EventAgentResponse)
	if got := placeOf(completed); got != toolStart {
		t.Errorf("TOOL_COMPLETED is at %+v, want the place of TOOL_STARTING %+v", got, toolStart)
	}
	if got := placeOf(response); got != agentStart {
		t.Errorf("AGENT_RESPONSE is at %+v, want the agent span %+v", got, agentStart)
	}
	mustLatency(t, completed)
	if latency, ok := rowLatency(t, response); ok {
		t.Errorf("AGENT_RESPONSE has latency_ms %v, want none", latency)
	}
}

func TestTraceManager_RootAgentName(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	// The runner's callbacks get the agent that the runner runs.
	tm.invocationSpan(ctx, identity{invocationID: "inv-1", agentName: "root"}, nil)
	if got := tm.pushAgentSpan(ctx, identity{invocationID: "inv-1", agentName: "sub"}).rootAgentName; got != "root" {
		t.Errorf("rootAgentName after the runner callbacks = %q, want %q", got, "root")
	}
	// Without them, the first agent to start is the root.
	if got := tm.pushAgentSpan(ctx, identity{invocationID: "inv-2", agentName: "sub"}).rootAgentName; got != "sub" {
		t.Errorf("rootAgentName of the first agent = %q, want %q", got, "sub")
	}
	if got := tm.eventSpan(ctx, identity{invocationID: "inv-2", agentName: "root"}, "sub", "").rootAgentName; got != "sub" {
		t.Errorf("rootAgentName after another agent = %q, want %q", got, "sub")
	}
	// When the first callback carries a dotted branch, the top-level
	// branch prefix names the root agent.
	if got := tm.pushAgentSpan(ctx, identity{invocationID: "inv-3", agentName: "sub", branch: "root.sub"}).rootAgentName; got != "root" {
		t.Errorf("rootAgentName from dotted branch = %q, want %q", got, "root")
	}
}

// trackedInvocations returns the sorted IDs of the invocations that tm
// tracks.
func trackedInvocations(tm *traceManager) []string {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	return slices.Sorted(maps.Keys(tm.invocations))
}

func TestTraceManager_EvictsLeastRecentlyActiveInvocations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	tm.maxTracked = 3
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }
	touch := func(invocationID string) {
		now = now.Add(time.Second)
		tm.invocationSpan(ctx, identity{invocationID: invocationID}, nil)
	}
	touch("inv-1")
	touch("inv-2")
	touch("inv-3")
	touch("inv-1") // inv-2 is now the least recently active.
	touch("inv-4")
	if diff := cmp.Diff([]string{"inv-1", "inv-3", "inv-4"}, trackedInvocations(tm)); diff != "" {
		t.Errorf("tracked invocations mismatch (-want +got):\n%s", diff)
	}
	// The invocation being started is never evicted.
	tm.maxTracked = 1
	touch("inv-5")
	if diff := cmp.Diff([]string{"inv-5"}, trackedInvocations(tm)); diff != "" {
		t.Errorf("tracked invocations mismatch (-want +got):\n%s", diff)
	}
	// Not even when the clock went back, which makes it the least recently
	// active invocation.
	now = now.Add(-time.Hour)
	touch("inv-6")
	if diff := cmp.Diff([]string{"inv-6"}, trackedInvocations(tm)); diff != "" {
		t.Errorf("tracked invocations after the clock went back mismatch (-want +got):\n%s", diff)
	}
}

func TestTraceManager_ExpiresIdleInvocations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	start := time.Unix(1000, 0)
	now := start
	tm.now = func() time.Time { return now }
	start1 := func(invocationID string) { tm.invocationSpan(ctx, identity{invocationID: invocationID}, nil) }
	start1("idle")
	start1("active")
	now = start.Add(90 * time.Minute)
	start1("active") // A callback keeps the invocation active.
	now = start.Add(2 * time.Hour)
	// Expired invocations are dropped every sweepInterval insertions.
	for i := 3; i < sweepInterval; i++ {
		start1(fmt.Sprintf("filler-%d", i))
	}
	if !slices.Contains(trackedInvocations(tm), "idle") {
		t.Fatal("the idle invocation was dropped before a sweep")
	}
	start1("last-filler")
	tracked := trackedInvocations(tm)
	if slices.Contains(tracked, "idle") {
		t.Error("the idle invocation is still tracked after a sweep")
	}
	if !slices.Contains(tracked, "active") {
		t.Error("the active invocation was dropped")
	}
	if got, want := len(tracked), sweepInterval-1; got != want {
		t.Errorf("%d invocations are tracked, want %d", got, want)
	}
}

func TestTraceManager_UntrackedInvocations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	// Callbacks without an invocation ID still get IDs.
	noID := identity{agentName: "solo"}
	spans := []rowSpan{tm.invocationSpan(ctx, noID, nil), tm.pushAgentSpan(ctx, noID), tm.endInvocation(ctx, noID)}
	// So does the end of an invocation that is not tracked, for example
	// because it was evicted.
	spans = append(spans, tm.endInvocation(ctx, identity{invocationID: "never-started"}))
	for i, s := range spans {
		if !eventIDPattern.MatchString(s.traceID) || !spanIDPattern.MatchString(s.spanID) {
			t.Errorf("span %d: trace %q, span %q, want 32 and 16 hex digits", i, s.traceID, s.spanID)
		}
	}
	if latency := spans[2].latency; latency != nil {
		t.Errorf("end of an invocation without ID has latency %v", latency)
	}
	if latency := spans[3].latency; latency != nil {
		t.Errorf("end of an untracked invocation has latency %v", latency)
	}
	// Starting and ending a tracked invocation removes it from tm.invocations.
	trackedID := identity{invocationID: "tracked", agentName: "solo"}
	started := tm.invocationSpan(ctx, trackedID, nil)
	if diff := cmp.Diff([]string{"tracked"}, trackedInvocations(tm)); diff != "" {
		t.Fatalf("tracked invocations while running mismatch (-want +got):\n%s", diff)
	}
	ended := tm.endInvocation(ctx, trackedID)
	if ended.traceID != started.traceID || ended.spanID != started.spanID || ended.latency == nil {
		t.Errorf("endInvocation(tracked) = %+v, want trace %q span %q with latency", ended, started.traceID, started.spanID)
	}
	if got := trackedInvocations(tm); len(got) != 0 {
		t.Errorf("tracked invocations = %v, want none", got)
	}
}

// scriptedLLM is a model.LLM that answers requests with its responses, in
// order.
type scriptedLLM struct {
	mu        sync.Mutex
	responses []*model.LLMResponse
}

func (m *scriptedLLM) Name() string { return "scripted-model" }

func (m *scriptedLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		if len(m.responses) == 0 {
			m.mu.Unlock()
			yield(nil, errors.New("scriptedLLM: no response left"))
			return
		}
		res := m.responses[0]
		m.responses = m.responses[1:]
		m.mu.Unlock()
		yield(res, nil)
	}
}

type zeroOuterConfigAgent struct {
	agent.Agent
	Config llmagent.Config
}

func TestAgentStartingData(t *testing.T) {
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Description: "Helps"})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{
		Name:        "planner",
		Description: "Plans trips",
		Model:       &scriptedLLM{},
		Instruction: "Plan the trip.",
		SubAgents:   []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	provided, err := llmagent.New(llmagent.Config{
		Name:                "provided",
		Model:               &scriptedLLM{},
		Instruction:         "Overridden by the provider.",
		InstructionProvider: func(agent.ReadonlyContext) (string, error) { return "Built at run time.", nil },
	})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	custom, err := agent.New(agent.Config{
		Name:        "custom",
		Description: "Runs code",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error: %v", err)
	}

	tests := []struct {
		name        string
		agent       agent.Agent
		wantContent any
		wantAttrs   map[string]any
	}{
		{
			name:        "LLM agent",
			agent:       planner,
			wantContent: "Plan the trip.",
			wantAttrs:   map[string]any{"agent_description": "Plans trips", "model": "scripted-model", "sub_agents": []string{"helper"}},
		},
		{
			name:        "wrapper with zero outer Config falls through to embedded base agent",
			agent:       &zeroOuterConfigAgent{Agent: planner},
			wantContent: "Plan the trip.",
			wantAttrs:   map[string]any{"agent_description": "Plans trips", "model": "scripted-model", "sub_agents": []string{"helper"}},
		},
		{
			name:      "instruction provider",
			agent:     provided,
			wantAttrs: map[string]any{"model": "scripted-model", "instruction_source": "provider"},
		},
		{
			name:      "LLM agent that inherits its model",
			agent:     helper,
			wantAttrs: map[string]any{"agent_description": "Helps"},
		},
		{
			name:      "custom agent",
			agent:     custom,
			wantAttrs: map[string]any{"agent_description": "Runs code"},
		},
		{name: "no agent", wantAttrs: map[string]any{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content, attrs := agentStartingData(tt.agent)
			if diff := cmp.Diff(tt.wantContent, content); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantAttrs, attrs); diff != "" {
				t.Errorf("attributes mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestTraceManager_AgentNamed looks agents up in the tree of the agent that
// the runner runs, which the first invocation callback that gets it
// records.
func TestTraceManager_AgentNamed(t *testing.T) {
	helper, err := llmagent.New(llmagent.Config{Name: "helper"})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{Name: "planner", SubAgents: []agent.Agent{helper}})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	other, err := llmagent.New(llmagent.Config{Name: "other"})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	ctx := context.Background()
	tm := newTraceManager()
	id := func(agentName string) identity { return identity{invocationID: "inv", agentName: agentName} }
	if got := tm.agentNamed(ctx, id("planner")); got != nil {
		t.Errorf("agentNamed(planner) before the runner's agent is known = %q, want nil", got.Name())
	}
	tm.invocationSpan(ctx, id("planner"), planner)
	tm.invocationSpan(ctx, id("other"), other)
	tests := []struct {
		agentName string
		want      agent.Agent
	}{
		{"planner", planner},
		{"helper", helper},
		// The first agent recorded stays the runner's agent.
		{"other", nil},
		{"", nil},
	}
	for _, tt := range tests {
		if got := tm.agentNamed(ctx, id(tt.agentName)); got != tt.want {
			t.Errorf("agentNamed(%q) = %v, want %v", tt.agentName, got, tt.want)
		}
	}
}

// agentContext is a mockInvocationContext whose Agent method returns a, as
// the contexts of ADK's invocation callbacks do. With a nil, calling Agent
// fails the test: the contexts that ADK gives agent callbacks log a
// warning and return nil.
type agentContext struct {
	*mockInvocationContext
	t *testing.T
	a agent.Agent
}

func (c *agentContext) Agent() agent.Agent {
	if c.a == nil {
		c.t.Errorf("Agent() called on the callback context of agent %q", c.agentName)
	}
	return c.a
}

// TestAgentStarting_LooksUpAgentInRunnerTree starts agents of the tree of
// the agent that the runner runs, and an agent outside it. The agent
// callbacks find the agents by name, without asking their context.
func TestAgentStarting_LooksUpAgentInRunnerTree(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Description: "Helps"})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{
		Name:        "planner",
		Description: "Plans trips",
		Instruction: "Plan the trip.",
		SubAgents:   []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	contextOf := func(agentName string, a agent.Agent) *agentContext {
		c := *mCtx
		c.agentName = agentName
		return &agentContext{mockInvocationContext: &c, t: t, a: a}
	}
	runnerCtx := contextOf("planner", planner)
	var errs callbackErrors
	errs.add(p.BeforeRunCallback()(runnerCtx))
	for _, agentName := range []string{"planner", "helper", "stranger"} {
		errs.add(p.BeforeAgentCallback()(contextOf(agentName, nil)))
		errs.add(p.AfterAgentCallback()(contextOf(agentName, nil)))
	}
	errs.check(t)
	p.AfterRunCallback()(runnerCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	starts := rowsOfType(rows, EventAgentStarting)
	if len(starts) != 3 {
		t.Fatalf("got %d AGENT_STARTING rows, want 3; event types: %v", len(starts), eventTypes(rows))
	}
	tests := []struct {
		agentName   string
		wantContent string
		wantAttrs   map[string]string
	}{
		{
			agentName:   "planner",
			wantContent: `"Plan the trip."`,
			wantAttrs:   map[string]string{"agent_description": `"Plans trips"`, "sub_agents": `["helper"]`},
		},
		{agentName: "helper", wantAttrs: map[string]string{"agent_description": `"Helps"`}},
		{agentName: "stranger"},
	}
	for i, tt := range tests {
		row := starts[i]
		if row["agent"] != tt.agentName {
			t.Errorf("AGENT_STARTING row %d: agent = %v, want %q", i, row["agent"], tt.agentName)
		}
		content, hasContent := jsonColumnValue(t, row, columnContent)
		if hasContent != (tt.wantContent != "") {
			t.Errorf("%s AGENT_STARTING: has content %t (%v), want %t", tt.agentName, hasContent, content, tt.wantContent != "")
		} else if hasContent {
			checkJSON(t, tt.agentName+" AGENT_STARTING content", content, tt.wantContent)
		}
		attrs := jsonObjectColumn(t, row, columnAttributes)
		for _, key := range []string{"agent_description", "sub_agents", "model", "instruction_source"} {
			want, wantSet := tt.wantAttrs[key]
			if _, ok := attrs[key]; ok != wantSet {
				t.Errorf("%s AGENT_STARTING: attributes.%s set = %t, want %t", tt.agentName, key, ok, wantSet)
			} else if wantSet {
				checkJSONAt(t, attrs, want, key)
			}
		}
	}
}

// TestTraceManager_Runner runs an LLM agent with the ADK runner. The model
// calls a tool twice, and ADK runs the two calls concurrently, before the
// model answers.
func TestTraceManager_Runner(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	type lookupArgs struct {
		City string `json:"city"`
	}
	lookupTool, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks up the weather"},
		func(_ agent.Context, args lookupArgs) (map[string]any, error) {
			return map[string]any{"weather": "sunny in " + args.City}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error: %v", err)
	}
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Description: "Helps"})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	llm := &scriptedLLM{responses: []*model.LLMResponse{
		{Content: genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromFunctionCall("lookup", map[string]any{"city": "Paris"}),
			genai.NewPartFromFunctionCall("lookup", map[string]any{"city": "Rome"}),
		}, genai.RoleModel)},
		{Content: textContent("model", "Sunny in both."), TurnComplete: true},
	}}
	planner, err := llmagent.New(llmagent.Config{
		Name:        "planner",
		Description: "Plans trips",
		Model:       llm,
		Instruction: "Plan the trip.",
		Tools:       []tool.Tool{lookupTool},
		SubAgents:   []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             planner,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error: %v", err)
	}
	for _, err := range r.Run(t.Context(), "user-1", "session-1", textContent("user", "What is the weather?"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	gotCounts := map[string]int{}
	for _, eventType := range eventTypes(rows) {
		gotCounts[eventType]++
	}
	wantCounts := map[string]int{
		EventUserMessageReceived: 1,
		EventInvocationStarting:  1,
		EventAgentStarting:       1,
		EventLLMRequest:          2,
		EventLLMResponse:         2,
		EventToolStarting:        2,
		EventToolCompleted:       2,
		EventAgentCompleted:      1,
		EventInvocationCompleted: 1,
		// The final answer session event emits AGENT_RESPONSE.
		EventAgentResponse: 1,
	}
	if diff := cmp.Diff(wantCounts, gotCounts); diff != "" {
		t.Fatalf("event type counts mismatch (-want +got):\n%s\nevent types: %v", diff, eventTypes(rows))
	}

	inv := placeOf(onlyRowOfType(t, rows, EventInvocationStarting))
	if !eventIDPattern.MatchString(inv.trace) || !spanIDPattern.MatchString(inv.span) || inv.parent != "" {
		t.Fatalf("INVOCATION_STARTING place = %+v, want a new trace and a root span", inv)
	}
	for _, eventType := range []string{EventUserMessageReceived, EventInvocationCompleted} {
		if got := placeOf(onlyRowOfType(t, rows, eventType)); got != inv {
			t.Errorf("%s place = %+v, want the invocation span %+v", eventType, got, inv)
		}
	}
	agentPlace := placeOf(onlyRowOfType(t, rows, EventAgentStarting))
	if agentPlace.trace != inv.trace || !spanIDPattern.MatchString(agentPlace.span) || agentPlace.parent != inv.span {
		t.Errorf("AGENT_STARTING place = %+v, want a span under the invocation span %s", agentPlace, inv.span)
	}
	if got := placeOf(onlyRowOfType(t, rows, EventAgentCompleted)); got != agentPlace {
		t.Errorf("AGENT_COMPLETED place = %+v, want the AGENT_STARTING place %+v", got, agentPlace)
	}

	// Every model and tool call has a span of its own under the agent.
	spans := map[string]bool{inv.span: true, agentPlace.span: true}
	checkCallSpan := func(name string, place tracePlace) {
		t.Helper()
		if place.trace != inv.trace || place.parent != agentPlace.span {
			t.Errorf("%s place = %+v, want a span under the agent span %s", name, place, agentPlace.span)
		}
		if spans[place.span] {
			t.Errorf("%s span %q is not a span of its own", name, place.span)
		}
		spans[place.span] = true
	}
	requestRows, responseRows := rowsOfType(rows, EventLLMRequest), rowsOfType(rows, EventLLMResponse)
	for i := range requestRows {
		place := placeOf(requestRows[i])
		checkCallSpan(fmt.Sprintf("LLM_REQUEST %d", i), place)
		if got := placeOf(responseRows[i]); got != place {
			t.Errorf("LLM_RESPONSE %d place = %+v, want the place of its request %+v", i, got, place)
		}
	}
	started := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventToolStarting) {
		callID := functionCallIDOf(t, row)
		started[callID] = placeOf(row)
		checkCallSpan("TOOL_STARTING of call "+callID, placeOf(row))
	}
	if len(started) != 2 {
		t.Errorf("TOOL_STARTING rows have function call IDs %v, want 2 distinct IDs", slices.Collect(maps.Keys(started)))
	}
	for _, row := range rowsOfType(rows, EventToolCompleted) {
		callID := functionCallIDOf(t, row)
		want, ok := started[callID]
		if !ok {
			t.Errorf("TOOL_COMPLETED row of call %q has no TOOL_STARTING row", callID)
			continue
		}
		if got := placeOf(row); got != want {
			t.Errorf("TOOL_COMPLETED place of call %q = %+v, want the TOOL_STARTING place %+v", callID, got, want)
		}
	}

	// The session events of the agent are placed in its span.
	for _, row := range rows {
		if _, ok := lookup(jsonObjectColumn(t, row, columnAttributes), "adk", "source_event_id"); !ok {
			continue
		}
		if got := placeOf(row); got != agentPlace {
			t.Errorf("%v row of a session event: place = %+v, want the agent place %+v", row["event_type"], got, agentPlace)
		}
	}

	agentStarting := onlyRowOfType(t, rows, EventAgentStarting)
	content, _ := jsonColumnValue(t, agentStarting, columnContent)
	checkJSON(t, "AGENT_STARTING content", content, `"Plan the trip."`)
	agentAttrs := jsonObjectColumn(t, agentStarting, columnAttributes)
	checkJSONAt(t, agentAttrs, `"Plans trips"`, "agent_description")
	checkJSONAt(t, agentAttrs, `"scripted-model"`, "model")
	checkJSONAt(t, agentAttrs, `["helper"]`, "sub_agents")

	wantLatencyKeys := map[string][]string{
		EventLLMResponse:         {"time_to_first_token_ms", "total_ms"},
		EventToolCompleted:       {"total_ms"},
		EventAgentCompleted:      {"total_ms"},
		EventInvocationCompleted: {"total_ms"},
	}
	for _, row := range rows {
		eventType, _ := row["event_type"].(string)
		checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), `"planner"`, "root_agent_name")
		latency, ok := rowLatency(t, row)
		if diff := cmp.Diff(wantLatencyKeys[eventType], slices.Sorted(maps.Keys(latency))); diff != "" {
			t.Errorf("%s row (has latency_ms: %t): latency_ms keys mismatch (-want +got):\n%s", eventType, ok, diff)
		}
		for key, ms := range latency {
			if ms < 0 {
				t.Errorf("%s row: latency_ms.%s = %d, want >= 0", eventType, key, ms)
			}
		}
	}
}

// TestTraceManager_SameFunctionCallIDAcrossConcurrentSubAgents verifies that
// two concurrent sub-agents whose model responses assign the same provider
// functionCallID (such as "call_0") do not collide in toolSpans or
// completedTools.
func TestTraceManager_SameFunctionCallIDAcrossConcurrentSubAgents(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	ctxOf := func(agentName, branch, callID string) *mockInvocationContext {
		c := *mCtx
		c.agentName = agentName
		c.branch = branch
		c.functionCallID = callID
		return &c
	}
	parCtx := ctxOf("par", "", "")
	aCtx := ctxOf("a", "par.a", "")
	bCtx := ctxOf("b", "par.b", "")
	aToolCtx := ctxOf("a", "par.a", "call_0")
	bToolCtx := ctxOf("b", "par.b", "call_0")
	lookup := &fakeTool{name: "lookup"}

	var errs callbackErrors
	errs.add(p.BeforeRunCallback()(parCtx))
	errs.add(p.BeforeAgentCallback()(parCtx))
	errs.add(p.BeforeAgentCallback()(aCtx))
	errs.add(p.BeforeAgentCallback()(bCtx))
	errs.add(p.BeforeToolCallback()(aToolCtx, lookup, map[string]any{"q": "a"}))
	errs.add(p.BeforeToolCallback()(bToolCtx, lookup, map[string]any{"q": "b"}))
	errs.add(p.AfterToolCallback()(aToolCtx, lookup, map[string]any{"q": "a"}, map[string]any{"r": "a"}, nil))
	errs.add(p.AfterToolCallback()(bToolCtx, lookup, map[string]any{"q": "b"}, map[string]any{"r": "b"}, nil))
	errs.add(p.AfterAgentCallback()(aCtx))
	errs.add(p.AfterAgentCallback()(bCtx))
	errs.add(p.AfterAgentCallback()(parCtx))
	errs.check(t)
	p.AfterRunCallback()(parCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentSpanByAgent := map[string]string{}
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		agentName, _ := row["agent"].(string)
		agentSpanByAgent[agentName] = placeOf(row).span
	}
	toolStartsByAgent := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventToolStarting) {
		agentName, _ := row["agent"].(string)
		place := placeOf(row)
		if place.parent != agentSpanByAgent[agentName] {
			t.Errorf("TOOL_STARTING of agent %q has parent_span_id %q, want %q", agentName, place.parent, agentSpanByAgent[agentName])
		}
		toolStartsByAgent[agentName] = place
	}
	if toolStartsByAgent["a"].span == toolStartsByAgent["b"].span {
		t.Errorf("concurrent sub-agents with same functionCallID share tool span %q", toolStartsByAgent["a"].span)
	}
	for _, row := range rowsOfType(rows, EventToolCompleted) {
		agentName, _ := row["agent"].(string)
		if got, want := placeOf(row), toolStartsByAgent[agentName]; got != want {
			t.Errorf("TOOL_COMPLETED of agent %q is at %+v, want %+v", agentName, got, want)
		}
		mustLatency(t, row)
	}
}

// TestTraceManager_SameAgentConcurrentNodePaths verifies that two concurrent
// activations of the same agent on the same branch (distinguished by their
// execution node path) correlate their model calls, tool calls, session
// events, and AGENT_COMPLETED rows with their own activation span.
func TestTraceManager_SameAgentConcurrentNodePaths(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	ctxOf := func(agentName, path, callID string) *mockInvocationContext {
		c := *mCtx
		c.agentName = agentName
		c.path = path
		c.functionCallID = callID
		return &c
	}
	rootCtx := ctxOf("wf", "wf@0", "")
	w1Ctx := ctxOf("worker", "wf@0/worker@1", "")
	w2Ctx := ctxOf("worker", "wf@0/worker@2", "")
	w1ToolCtx := ctxOf("worker", "wf@0/worker@1", "call_0")
	w2ToolCtx := ctxOf("worker", "wf@0/worker@2", "call_0")
	lookup := &fakeTool{name: "lookup"}
	req := &model.LLMRequest{Model: "gemini-2.5-pro", Contents: []*genai.Content{textContent("user", "hi")}}
	res := &model.LLMResponse{Content: textContent("model", "ok")}

	ev1 := session.NewEvent(t.Context(), mCtx.invocationID)
	ev1.Author = "worker"
	ev1.NodeInfo = &session.NodeInfo{Path: "wf@0/worker@1"}
	ev1.LLMResponse = model.LLMResponse{Content: textContent("model", "event-1")}

	ev2 := session.NewEvent(t.Context(), mCtx.invocationID)
	ev2.Author = "worker"
	ev2.NodeInfo = &session.NodeInfo{Path: "wf@0/worker@2"}
	ev2.LLMResponse = model.LLMResponse{Content: textContent("model", "event-2")}

	var errs callbackErrors
	errs.add(p.BeforeRunCallback()(rootCtx))
	errs.add(p.BeforeAgentCallback()(rootCtx))
	errs.add(p.BeforeAgentCallback()(w1Ctx))
	errs.add(p.BeforeAgentCallback()(w2Ctx))
	errs.add(p.BeforeModelCallback()(w1Ctx, req))
	errs.add(p.BeforeModelCallback()(w2Ctx, req))
	errs.add(p.AfterModelCallback()(w1Ctx, res, nil))
	errs.add(p.AfterModelCallback()(w2Ctx, res, nil))
	errs.add(p.BeforeToolCallback()(w1ToolCtx, lookup, map[string]any{"id": 1}))
	errs.add(p.BeforeToolCallback()(w2ToolCtx, lookup, map[string]any{"id": 2}))
	errs.add(p.AfterToolCallback()(w1ToolCtx, lookup, map[string]any{"id": 1}, map[string]any{"ok": true}, nil))
	errs.add(p.AfterToolCallback()(w2ToolCtx, lookup, map[string]any{"id": 2}, map[string]any{"ok": true}, nil))
	errs.add(p.OnEventCallback()(rootCtx, ev1))
	errs.add(p.OnEventCallback()(rootCtx, ev2))
	// Activation 1 finishes before activation 2 (FIFO order on the same branch).
	errs.add(p.AfterAgentCallback()(w1Ctx))
	errs.add(p.AfterAgentCallback()(w2Ctx))
	errs.add(p.AfterAgentCallback()(rootCtx))
	errs.check(t)
	p.AfterRunCallback()(rootCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentStarts := rowsOfType(rows, EventAgentStarting)
	if len(agentStarts) != 3 {
		t.Fatalf("got %d AGENT_STARTING rows, want 3", len(agentStarts))
	}
	wfSpan := placeOf(agentStarts[0])
	w1Span := placeOf(agentStarts[1])
	w2Span := placeOf(agentStarts[2])
	if w1Span.parent != wfSpan.span || w2Span.parent != wfSpan.span || w1Span.span == w2Span.span {
		t.Fatalf("worker spans = (%+v, %+v), want two distinct spans parented to wf (%q)", w1Span, w2Span, wfSpan.span)
	}
	llmReqs := rowsOfType(rows, EventLLMRequest)
	llmResps := rowsOfType(rows, EventLLMResponse)
	if placeOf(llmReqs[0]).parent != w1Span.span || placeOf(llmReqs[1]).parent != w2Span.span {
		t.Errorf("LLM_REQUEST parents = (%q, %q), want (%q, %q)", placeOf(llmReqs[0]).parent, placeOf(llmReqs[1]).parent, w1Span.span, w2Span.span)
	}
	if placeOf(llmResps[0]) != placeOf(llmReqs[0]) || placeOf(llmResps[1]) != placeOf(llmReqs[1]) {
		t.Errorf("LLM_RESPONSE places = (%+v, %+v), want (%+v, %+v)", placeOf(llmResps[0]), placeOf(llmResps[1]), placeOf(llmReqs[0]), placeOf(llmReqs[1]))
	}
	toolStarts := rowsOfType(rows, EventToolStarting)
	toolEnds := rowsOfType(rows, EventToolCompleted)
	if placeOf(toolStarts[0]).parent != w1Span.span || placeOf(toolStarts[1]).parent != w2Span.span || placeOf(toolStarts[0]).span == placeOf(toolStarts[1]).span {
		t.Errorf("TOOL_STARTING places = (%+v, %+v), want distinct spans under (%q, %q)", placeOf(toolStarts[0]), placeOf(toolStarts[1]), w1Span.span, w2Span.span)
	}
	if placeOf(toolEnds[0]) != placeOf(toolStarts[0]) || placeOf(toolEnds[1]) != placeOf(toolStarts[1]) {
		t.Errorf("TOOL_COMPLETED places = (%+v, %+v), want (%+v, %+v)", placeOf(toolEnds[0]), placeOf(toolEnds[1]), placeOf(toolStarts[0]), placeOf(toolStarts[1]))
	}
	respRows := rowsOfType(rows, EventAgentResponse)
	if len(respRows) != 2 || placeOf(respRows[0]) != w1Span || placeOf(respRows[1]) != w2Span {
		t.Errorf("AGENT_RESPONSE places = %v, want (%+v, %+v)", respRows, w1Span, w2Span)
	}
	agentEnds := rowsOfType(rows, EventAgentCompleted)
	if placeOf(agentEnds[0]) != w1Span || placeOf(agentEnds[1]) != w2Span || placeOf(agentEnds[2]) != wfSpan {
		t.Errorf("AGENT_COMPLETED places = (%+v, %+v, %+v), want (%+v, %+v, %+v)",
			placeOf(agentEnds[0]), placeOf(agentEnds[1]), placeOf(agentEnds[2]), w1Span, w2Span, wfSpan)
	}
}

// TestTraceManager_SequentialSiblingShortCircuitedInBeforeAgent runs a
// sequential agent whose first sub-agent is short-circuited by a subsequent
// BeforeAgentCallback returning non-nil content (so ADK skips AfterAgentCallback
// for that sub-agent). The second sub-agent's AGENT_STARTING span must still
// be parented to the sequential agent rather than the short-circuited sibling,
// and no orphan agent span may remain when the sequential agent completes.
func TestTraceManager_SequentialSiblingShortCircuitedInBeforeAgent(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	shortcut, err := llmagent.New(llmagent.Config{
		Name: "shortcut",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return textContent("model", "short-circuited"), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(shortcut) error: %v", err)
	}
	next, err := llmagent.New(llmagent.Config{
		Name:  "next",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "done"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(next) error: %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "sequence",
			SubAgents: []agent.Agent{shortcut, next},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New() error: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             seq,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error: %v", err)
	}
	for _, err := range r.Run(t.Context(), "user-1", "session-1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	startByAgent := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		name, _ := row["agent"].(string)
		startByAgent[name] = placeOf(row)
	}
	seqPlace := startByAgent["sequence"]
	shortcutPlace := startByAgent["shortcut"]
	nextPlace := startByAgent["next"]
	if shortcutPlace.parent != seqPlace.span {
		t.Errorf("shortcut AGENT_STARTING parent_span_id = %q, want sequence span %q", shortcutPlace.parent, seqPlace.span)
	}
	if nextPlace.parent != seqPlace.span {
		t.Errorf("next AGENT_STARTING parent_span_id = %q, want sequence span %q (not short-circuited sibling %q)", nextPlace.parent, seqPlace.span, shortcutPlace.span)
	}
	for _, row := range rowsOfType(rows, EventAgentCompleted) {
		name, _ := row["agent"].(string)
		if got, want := placeOf(row), startByAgent[name]; got != want {
			t.Errorf("%s AGENT_COMPLETED place = %+v, want %+v", name, got, want)
		}
	}
}

// TestTraceManager_SubAgentTransferContinuationTurn runs a two-turn session
// where the root agent transfers to a sub-agent on Turn 1, and on Turn 2 ADK's
// Runner resumes directly at the sub-agent before transferring back to the
// root agent. On Turn 2, root_agent_name must still be the configured root
// agent ("planner") and AGENT_STARTING for "planner" must still resolve its
// config from the root agent tree.
func TestTraceManager_SubAgentTransferContinuationTurn(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	transferTo := func(target string) *model.LLMResponse {
		return &model.LLMResponse{
			Content: genai.NewContentFromParts([]*genai.Part{
				genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": target}),
			}, genai.RoleModel),
		}
	}
	helperLLM := &scriptedLLM{responses: []*model.LLMResponse{
		// Turn 1: helper answers after planner transfers to it.
		{Content: textContent("model", "Helper answered turn 1."), TurnComplete: true},
		// Turn 2: Runner starts at helper; helper transfers back to planner.
		transferTo("planner"),
	}}
	plannerLLM := &scriptedLLM{responses: []*model.LLMResponse{
		// Turn 1: planner transfers to helper.
		transferTo("helper"),
		// Turn 2: planner answers after helper transfers back to it.
		{Content: textContent("model", "Planner answered turn 2."), TurnComplete: true},
	}}
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "Helps",
		Model:       helperLLM,
		Instruction: "Help the user.",
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper) error: %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{
		Name:        "planner",
		Description: "Plans trips",
		Model:       plannerLLM,
		Instruction: "Plan the trip.",
		SubAgents:   []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(planner) error: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             planner,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error: %v", err)
	}

	// Turn 1: planner -> helper.
	for _, err := range r.Run(t.Context(), "user-1", "session-1", textContent("user", "Turn 1"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Turn 1 Run() error: %v", err)
		}
	}
	_ = collectRows(t, requests, untilEventType(EventInvocationCompleted))

	// Turn 2: Runner starts at helper (last author of Turn 1), which transfers back to planner.
	for _, err := range r.Run(t.Context(), "user-1", "session-1", textContent("user", "Turn 2"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Turn 2 Run() error: %v", err)
		}
	}
	turn2Rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	for _, row := range turn2Rows {
		checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), `"planner"`, "root_agent_name")
	}
	var plannerStart map[string]any
	for _, row := range rowsOfType(turn2Rows, EventAgentStarting) {
		if row["agent"] == "planner" {
			plannerStart = row
			break
		}
	}
	if plannerStart == nil {
		t.Fatalf("Turn 2 rows have no AGENT_STARTING for planner; event types: %v", eventTypes(turn2Rows))
	}
	content, ok := jsonColumnValue(t, plannerStart, columnContent)
	if !ok {
		t.Fatal("Turn 2 planner AGENT_STARTING has no content")
	}
	checkJSON(t, "Turn 2 planner AGENT_STARTING content", content, `"Plan the trip."`)
	attrs := jsonObjectColumn(t, plannerStart, columnAttributes)
	checkJSONAt(t, attrs, `"Plans trips"`, "agent_description")
	checkJSONAt(t, attrs, `"scripted-model"`, "model")
}

// TestTraceManager_ExactMillisecondLatency verifies that recordFirstToken and
// popLLMSpan compute exact millisecond durations from a controlled clock.
func TestTraceManager_ExactMillisecondLatency(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }
	id := identity{invocationID: "inv-clock", agentName: "agent"}

	tm.invocationSpan(ctx, id, nil)
	tm.pushAgentSpan(ctx, id)
	tm.pushLLMSpan(ctx, id)

	now = now.Add(150 * time.Millisecond)
	tm.recordFirstToken(ctx, id)

	now = now.Add(850 * time.Millisecond)
	llmEnd := tm.popLLMSpan(ctx, id, true)
	wantLLM := map[string]any{
		"time_to_first_token_ms": int64(150),
		"total_ms":               int64(1000),
	}
	if diff := cmp.Diff(wantLLM, llmEnd.latency); diff != "" {
		t.Errorf("LLM latency mismatch (-want +got):\n%s", diff)
	}
}

// TestToolFailure_ShortCircuitedOnToolErrorStillLogsError verifies that when
// an earlier plugin short-circuits OnToolErrorCallback and ADK proceeds to
// AfterToolCallback with a non-nil err, the plugin still emits a TOOL_ERROR
// row at the tool call's span with latency_ms.
func TestToolFailure_ShortCircuitedOnToolErrorStillLogsError(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	toolCtx := withFunctionCall(mCtx, "call-short-circuit")
	lookup := &fakeTool{name: "lookup"}
	args := map[string]any{"city": "Paris"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeToolCallback()(toolCtx, lookup, args))
	// OnToolErrorCallback is skipped because an earlier plugin returned a result/error.
	errs.add(p.AfterToolCallback()(toolCtx, lookup, args, nil, errors.New("tool boom")))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	want := []string{
		EventAgentStarting,
		EventToolStarting,
		EventToolError,
		EventAgentCompleted,
		EventInvocationCompleted,
	}
	if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	toolStart := placeOf(onlyRowOfType(t, rows, EventToolStarting))
	toolErr := onlyRowOfType(t, rows, EventToolError)
	if got := placeOf(toolErr); got != toolStart {
		t.Errorf("TOOL_ERROR is at %+v, want %+v", got, toolStart)
	}
	mustLatency(t, toolErr)
	if got := toolErr["status"]; got != "ERROR" {
		t.Errorf("TOOL_ERROR status = %v, want ERROR", got)
	}
}

// TestTraceManager_InFlightProtectedFromTTLSweepAndPreservesTraceOnCapEviction
// verifies:
//  1. An invocation with an open tool call is not dropped by the idle TTL sweep.
//  2. If maxTracked capacity eviction forces out an in-flight invocation, a
//     warning is logged via logf, subsequent callbacks for the evicted
//     invocation preserve its original trace_id and root_agent_name, and
//     INVOCATION_COMPLETED omits a bogus near-zero latency_ms.
func TestTraceManager_InFlightProtectedFromTTLSweepAndPreservesTraceOnCapEviction(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	start := time.Unix(1000, 0)
	now := start
	tm.now = func() time.Time { return now }
	var warnings []string
	tm.logf = func(format string, args ...any) {
		warnings = append(warnings, fmt.Sprintf(format, args...))
	}

	inFlightID := identity{invocationID: "in-flight", agentName: "root"}
	idleID := identity{invocationID: "idle", agentName: "root"}
	origSpan := tm.invocationSpan(ctx, inFlightID, nil)
	tm.pushAgentSpan(ctx, inFlightID)
	tm.pushToolSpan(ctx, inFlightID, "long-tool")
	tm.invocationSpan(ctx, idleID, nil)

	// Advance past invocationTTL and trigger a sweep: idle is dropped, but
	// in-flight (which still has an open agent and tool span) is retained.
	now = start.Add(2 * time.Hour)
	for i := 3; i <= sweepInterval; i++ {
		id := identity{invocationID: fmt.Sprintf("filler-%d", i)}
		tm.invocationSpan(ctx, id, nil)
		tm.endInvocation(ctx, id)
	}
	tracked := trackedInvocations(tm)
	if slices.Contains(tracked, "idle") {
		t.Error("idle invocation is still tracked after TTL sweep")
	}
	if !slices.Contains(tracked, "in-flight") {
		t.Fatal("in-flight invocation with open spans was dropped by TTL sweep")
	}

	// Now force capacity eviction of in-flight by setting maxTracked = 1 and
	// starting another invocation.
	tm.maxTracked = 1
	now = now.Add(time.Minute)
	tm.invocationSpan(ctx, identity{invocationID: "other", agentName: "other-root"}, nil)
	if len(warnings) != 1 || !strings.Contains(warnings[0], "in-flight") {
		t.Errorf("capacity eviction warnings = %v, want 1 warning naming in-flight", warnings)
	}

	// Subsequent callbacks for in-flight re-create its tracking state using
	// its remembered trace_id and root_agent_name, and omit bogus latency_ms
	// on INVOCATION_COMPLETED.
	now = now.Add(time.Minute)
	toolEnd, _ := tm.popToolSpan(ctx, identity{invocationID: "in-flight", agentName: "sub"}, "long-tool", false)
	if toolEnd.traceID != origSpan.traceID || toolEnd.rootAgentName != "root" {
		t.Errorf("post-eviction popToolSpan = %+v, want traceID %q and rootAgentName %q", toolEnd, origSpan.traceID, "root")
	}
	invEnd := tm.endInvocation(ctx, identity{invocationID: "in-flight", agentName: "sub"})
	if invEnd.traceID != origSpan.traceID || invEnd.spanID != origSpan.spanID || invEnd.rootAgentName != "root" {
		t.Errorf("post-eviction endInvocation = %+v, want traceID %q, spanID %q, rootAgentName %q", invEnd, origSpan.traceID, origSpan.spanID, "root")
	}
	if invEnd.latency != nil {
		t.Errorf("post-eviction endInvocation has latency %v, want nil", invEnd.latency)
	}
}

// TestTraceManager_ConcurrentSubAgentsAndModelCalls stresses per-invocation
// locking (`inv.mu`) with 32 goroutines concurrently running sub-agent spans,
// streaming model calls, tool calls, and event lookups on a single invocation.
func TestTraceManager_ConcurrentSubAgentsAndModelCalls(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	rootID := identity{invocationID: "inv-concurrent", agentName: "root"}
	invSpan := tm.invocationSpan(ctx, rootID, nil)
	rootSpan := tm.pushAgentSpan(ctx, rootID)

	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := range workers {
		go func(i int) {
			defer wg.Done()
			subID := identity{
				invocationID: "inv-concurrent",
				agentName:    fmt.Sprintf("worker-%d", i),
				branch:       fmt.Sprintf("root.worker-%d", i),
				path:         fmt.Sprintf("root@0/worker-%d@%d", i, i+1),
			}
			agentSpan := tm.pushAgentSpan(ctx, subID)
			if agentSpan.traceID != invSpan.traceID || agentSpan.parentSpanID != rootSpan.spanID {
				t.Errorf("worker %d agentSpan = %+v, want trace %q parent %q", i, agentSpan, invSpan.traceID, rootSpan.spanID)
			}
			llmStart := tm.pushLLMSpan(ctx, subID)
			tm.recordFirstToken(ctx, subID)
			llmEnd := tm.popLLMSpan(ctx, subID, true)
			if llmEnd.spanID != llmStart.spanID || llmEnd.parentSpanID != agentSpan.spanID {
				t.Errorf("worker %d llmEnd = %+v, want span %q parent %q", i, llmEnd, llmStart.spanID, agentSpan.spanID)
			}
			key := toolKey(subID, "call_0", "lookup")
			toolStart := tm.pushToolSpan(ctx, subID, key)
			if ev := tm.eventSpan(ctx, rootID, subID.agentName, subID.branch); ev.spanID != agentSpan.spanID {
				t.Errorf("worker %d eventSpan = %+v, want span %q", i, ev, agentSpan.spanID)
			}
			toolEnd, closed := tm.popToolSpan(ctx, subID, key, false)
			if closed || toolEnd.spanID != toolStart.spanID || toolEnd.parentSpanID != agentSpan.spanID {
				t.Errorf("worker %d toolEnd = %+v (closed=%t), want span %q parent %q", i, toolEnd, closed, toolStart.spanID, agentSpan.spanID)
			}
			agentEnd := tm.popAgentSpan(ctx, subID)
			if agentEnd.spanID != agentSpan.spanID {
				t.Errorf("worker %d agentEnd = %+v, want span %q", i, agentEnd, agentSpan.spanID)
			}
		}(i)
	}
	wg.Wait()

	tm.popAgentSpan(ctx, rootID)
	if got := openSpans(t, tm, "inv-concurrent"); got != (spanCounts{}) {
		t.Errorf("openSpans after concurrent workers = %+v, want empty", got)
	}
	tm.endInvocation(ctx, rootID)
	if got := trackedInvocations(tm); len(got) != 0 {
		t.Errorf("trackedInvocations = %v, want empty", got)
	}
}

// TestTraceManager_ConcurrentInvocations stresses traceManager-level locking
// (`tm.mu`) with 32 goroutines concurrently starting, running, and ending
// independent invocations.
func TestTraceManager_ConcurrentInvocations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := range workers {
		go func(i int) {
			defer wg.Done()
			id := identity{
				invocationID: fmt.Sprintf("inv-%d", i),
				agentName:    "planner",
			}
			inv := tm.invocationSpan(ctx, id, nil)
			ag := tm.pushAgentSpan(ctx, id)
			llm := tm.pushLLMSpan(ctx, id)
			tm.recordFirstToken(ctx, id)
			llmEnd := tm.popLLMSpan(ctx, id, true)
			agEnd := tm.popAgentSpan(ctx, id)
			invEnd := tm.endInvocation(ctx, id)
			if ag.parentSpanID != inv.spanID || llm.parentSpanID != ag.spanID || llmEnd.spanID != llm.spanID || agEnd.spanID != ag.spanID || invEnd.spanID != inv.spanID {
				t.Errorf("invocation %d spans inconsistent: inv=%+v ag=%+v llm=%+v", i, inv, ag, llm)
			}
		}(i)
	}
	wg.Wait()
	if got := trackedInvocations(tm); len(got) != 0 {
		t.Errorf("trackedInvocations after all concurrent invocations ended = %v, want empty", got)
	}
}

// TestTraceManager_ConcurrentRunnerTreeAndCapacityEviction stresses concurrent
// invocationSpan calls that pass a configured runner agent tree while capacity
// eviction (maxTracked = 1 and maxTracked = 4) is actively evicting invocations.
// This pins the lock order between tm.mu and inv.mu so rememberAgentTree and
// evictLocked cannot deadlock.
func TestTraceManager_ConcurrentRunnerTreeAndCapacityEviction(t *testing.T) {
	ctx := context.Background()
	worker, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{worker},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}

	for _, capLimit := range []int{1, 4} {
		t.Run(fmt.Sprintf("maxTracked=%d", capLimit), func(t *testing.T) {
			tm := newTraceManager()
			tm.maxTracked = capLimit
			const workers = 24
			var wg sync.WaitGroup
			wg.Add(workers)
			for i := range workers {
				go func(i int) {
					defer wg.Done()
					id := identity{
						invocationID: fmt.Sprintf("inv-evict-%d", i),
						agentName:    "root",
					}
					inv := tm.invocationSpan(ctx, id, root)
					ag := tm.pushAgentSpan(ctx, id)
					llm := tm.pushLLMSpan(ctx, id)
					tm.recordFirstToken(ctx, id)
					_ = tm.popLLMSpan(ctx, id, true)
					_ = tm.popAgentSpan(ctx, id)
					end := tm.endInvocation(ctx, id)
					if inv.traceID == "" || ag.traceID != inv.traceID || llm.traceID != inv.traceID || end.traceID != inv.traceID {
						t.Errorf("invocation %d traceID mismatch: inv=%+v ag=%+v llm=%+v end=%+v", i, inv, ag, llm, end)
					}
					if end.rootAgentName != "root" {
						t.Errorf("invocation %d rootAgentName = %q, want %q", i, end.rootAgentName, "root")
					}
				}(i)
			}
			wg.Wait()
		})
	}
}

// blockingLLM is a model.LLM whose GenerateContent waits until both concurrent
// activations have entered GenerateContent, guaranteeing overlapping live
// execution of two concurrent workflow.RunNode calls.
type blockingLLM struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (m *blockingLLM) Name() string { return "blocking-model" }

func (m *blockingLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.entered <- struct{}{}
		<-m.release
		yield(&model.LLMResponse{
			Content:      textContent("model", "worker done"),
			TurnComplete: true,
		}, nil)
	}
}

// TestTraceManager_RunnerConcurrentWorkflowRunNodeSameAgent runs a real ADK
// Runner whose root agent has a configured SubAgents tree [worker] and fans
// out two concurrent workflow.RunNode activations of worker on the same
// branch. Both activations must get non-empty execution paths through the
// callback context chain, neither sibling may retire the other, and every
// LLM_RESPONSE and AGENT_COMPLETED row must match its starting span with
// non-null latency_ms.
func TestTraceManager_RunnerConcurrentWorkflowRunNodeSameAgent(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	bLLM := &blockingLLM{
		entered: make(chan struct{}, 2),
		release: make(chan struct{}),
	}
	go func() {
		<-bLLM.entered
		<-bLLM.entered
		bLLM.once.Do(func() { close(bLLM.release) })
	}()

	var workerNodes [2]*workflow.AgentNode
	var primaryWorker agent.Agent
	for idx := range 2 {
		w, wErr := llmagent.New(llmagent.Config{
			Name:        "worker",
			Description: "Concurrent worker",
			Model:       bLLM,
			Instruction: "Do the work.",
		})
		if wErr != nil {
			t.Fatalf("llmagent.New(worker %d): %v", idx, wErr)
		}
		if idx == 0 {
			primaryWorker = w
		}
		wn, wnErr := workflow.NewAgentNode(w, workflow.NodeConfig{})
		if wnErr != nil {
			t.Fatalf("workflow.NewAgentNode(worker %d): %v", idx, wnErr)
		}
		workerNodes[idx] = wn
	}

	orch := workflow.NewDynamicNode[any, string]("orch",
		func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			errs := make(chan error, 2)
			var wg sync.WaitGroup
			wg.Add(2)
			for idx := range 2 {
				go func(idx int) {
					defer wg.Done()
					_, runErr := workflow.RunNode[any](ctx, workerNodes[idx], fmt.Sprintf("task-%d", idx))
					errs <- runErr
				}(idx)
			}
			wg.Wait()
			close(errs)
			for runErr := range errs {
				if runErr != nil {
					return "", runErr
				}
			}
			return "all workers done", nil
		},
		workflow.NodeConfig{},
	)
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, orch))
	if err != nil {
		t.Fatalf("workflow.New(wf): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{primaryWorker},
		Run:       wf.Run,
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(t.Context(), "user-1", "session-1", textContent("user", "fan out"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	agentStarts := rowsOfType(rows, EventAgentStarting)
	if len(agentStarts) != 3 {
		t.Fatalf("got %d AGENT_STARTING rows, want 3 (root + 2 workers); event types: %v", len(agentStarts), eventTypes(rows))
	}
	rootSpan := placeOf(agentStarts[0])
	w1Span := placeOf(agentStarts[1])
	w2Span := placeOf(agentStarts[2])
	if w1Span.parent != rootSpan.span || w2Span.parent != rootSpan.span || w1Span.span == w2Span.span {
		t.Fatalf("worker AGENT_STARTING places = (%+v, %+v), want two distinct spans both parented to root span %q", w1Span, w2Span, rootSpan.span)
	}
	workerSpanSet := map[string]bool{w1Span.span: true, w2Span.span: true}

	llmReqs := rowsOfType(rows, EventLLMRequest)
	llmResps := rowsOfType(rows, EventLLMResponse)
	if len(llmReqs) != 2 || len(llmResps) != 2 {
		t.Fatalf("got %d LLM_REQUEST and %d LLM_RESPONSE rows, want 2 of each", len(llmReqs), len(llmResps))
	}
	reqBySpan := map[string]tracePlace{}
	reqParents := map[string]bool{}
	for _, req := range llmReqs {
		pl := placeOf(req)
		if !workerSpanSet[pl.parent] {
			t.Errorf("LLM_REQUEST place %+v is not parented to either worker span (%q, %q)", pl, w1Span.span, w2Span.span)
		}
		reqBySpan[pl.span] = pl
		reqParents[pl.parent] = true
	}
	if len(reqBySpan) != 2 || len(reqParents) != 2 {
		t.Errorf("LLM_REQUEST spans = %v (parents %v), want 2 distinct spans across both worker activations", reqBySpan, reqParents)
	}
	for _, resp := range llmResps {
		pl := placeOf(resp)
		if want, ok := reqBySpan[pl.span]; !ok || pl != want {
			t.Errorf("LLM_RESPONSE place = %+v, want matching LLM_REQUEST in %v", pl, reqBySpan)
		}
		mustLatency(t, resp)
	}

	agentEnds := rowsOfType(rows, EventAgentCompleted)
	if len(agentEnds) != 3 {
		t.Fatalf("got %d AGENT_COMPLETED rows, want 3", len(agentEnds))
	}
	completedSpans := map[string]tracePlace{}
	for _, row := range agentEnds {
		pl := placeOf(row)
		completedSpans[pl.span] = pl
		mustLatency(t, row)
	}
	for _, want := range []tracePlace{rootSpan, w1Span, w2Span} {
		if got, ok := completedSpans[want.span]; !ok || got != want {
			t.Errorf("missing or mismatched AGENT_COMPLETED for %+v; got %v", want, completedSpans)
		}
	}
}

// TestTraceManager_ColdPersistedSessionResumeResolvesConfiguredRoot verifies
// that when a persisted session whose last event was authored by a sub-agent
// ("helper") is resumed on a fresh plugin instance that never observed Turn 1,
// the plugin resolves the configured root agent ("configured_root") from the
// runner's parent map in ctx on the very first callback.
func TestTraceManager_ColdPersistedSessionResumeResolvesConfiguredRoot(t *testing.T) {
	ctx := t.Context()
	sessSvc := session.InMemoryService()
	createResp, err := sessSvc.Create(ctx, &session.CreateRequest{
		AppName:   "test-app",
		UserID:    "user-1",
		SessionID: "persisted-session",
	})
	if err != nil {
		t.Fatalf("sessSvc.Create: %v", err)
	}
	priorEvent := session.NewEvent(ctx, "inv-turn-1")
	priorEvent.Author = "helper"
	priorEvent.LLMResponse = model.LLMResponse{
		Content:      textContent("model", "Helper handled turn 1."),
		TurnComplete: true,
	}
	if err := sessSvc.AppendEvent(ctx, createResp.Session, priorEvent); err != nil {
		t.Fatalf("sessSvc.AppendEvent: %v", err)
	}

	// Brand-new plugin instance that has never seen Turn 1.
	p, requests, _ := setupTestPlugin(t)
	transferToRoot := &model.LLMResponse{
		Content: genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "configured_root"}),
		}, genai.RoleModel),
	}
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "Sub-agent helper",
		Model:       &scriptedLLM{responses: []*model.LLMResponse{transferToRoot}},
		Instruction: "Help the user.",
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	configuredRoot, err := llmagent.New(llmagent.Config{
		Name:        "configured_root",
		Description: "Root coordinator",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: textContent("model", "Root answered on cold resume."), TurnComplete: true},
		}},
		Instruction: "Coordinate everything.",
		SubAgents:   []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(configured_root): %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "test-app",
		Agent:          configuredRoot,
		SessionService: sessSvc,
		PluginConfig:   runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(ctx, "user-1", "persisted-session", textContent("user", "Resume turn 2"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	for _, row := range rows {
		checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), `"configured_root"`, "root_agent_name")
	}
	var rootStart map[string]any
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		if row["agent"] == "configured_root" {
			rootStart = row
			break
		}
	}
	if rootStart == nil {
		t.Fatalf("no AGENT_STARTING row for configured_root; event types: %v", eventTypes(rows))
	}
	content, ok := jsonColumnValue(t, rootStart, columnContent)
	if !ok {
		t.Fatal("configured_root AGENT_STARTING has no content")
	}
	checkJSON(t, "configured_root AGENT_STARTING content", content, `"Coordinate everything."`)
	attrs := jsonObjectColumn(t, rootStart, columnAttributes)
	checkJSONAt(t, attrs, `"Root coordinator"`, "agent_description")
	checkJSONAt(t, attrs, `"scripted-model"`, "model")
}

// TestTraceManager_EvictionPreservesInheritedOTelAndNestedCallbackContext
// verifies that when an invocation started under an ambient OpenTelemetry span
// is evicted due to capacity and later recreated:
//  1. If recreated under a nested callback's child OTel span, it keeps the
//     original trace_id, root span_id, and root parent_span_id rather than
//     reparenting the invocation root to the nested child span.
//  2. If recreated without ambient OTel context, it still keeps the original
//     ambient trace_id, root span_id, and root parent_span_id.
//  3. Even if pushAgentSpan runs on the recreated invocation (setting
//     inv.started = true), endInvocation still omits latency_ms because
//     inv.evicted is true.
func TestTraceManager_EvictionPreservesInheritedOTelAndNestedCallbackContext(t *testing.T) {
	const (
		ambientTraceHex = "4bf92f3577b34da6a3ce929d0e0e4736"
		ambientSpanHex  = "00f067aa0ba902b7"
		nestedTraceHex  = "4bf92f3577b34da6a3ce929d0e0e4736"
		nestedSpanHex   = "1111222233334444"
	)
	ambientTraceID, _ := trace.TraceIDFromHex(ambientTraceHex)
	ambientSpanID, _ := trace.SpanIDFromHex(ambientSpanHex)
	nestedTraceID, _ := trace.TraceIDFromHex(nestedTraceHex)
	nestedSpanID, _ := trace.SpanIDFromHex(nestedSpanHex)

	ambientCtx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: ambientTraceID,
		SpanID:  ambientSpanID,
	}))
	nestedCtx := trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: nestedTraceID,
		SpanID:  nestedSpanID,
	}))

	tm := newTraceManager()
	tm.maxTracked = 1
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }

	// Case 1: Evicted, then recreated under nestedCtx via pushAgentSpan, then
	// ended without ambient OTel context.
	inv1ID := identity{invocationID: "inv-otel-1", agentName: "root"}
	orig1 := tm.invocationSpan(ambientCtx, inv1ID, nil)
	if orig1.traceID != ambientTraceHex || orig1.parentSpanID != ambientSpanHex {
		t.Fatalf("orig1 = %+v, want traceID %q and parentSpanID %q", orig1, ambientTraceHex, ambientSpanHex)
	}
	now = now.Add(time.Second)
	tm.invocationSpan(context.Background(), identity{invocationID: "evictor-1", agentName: "other"}, nil)

	now = now.Add(time.Second)
	subSpan := tm.pushAgentSpan(nestedCtx, identity{invocationID: "inv-otel-1", agentName: "sub"})
	if subSpan.traceID != ambientTraceHex || subSpan.parentSpanID != orig1.spanID || subSpan.rootAgentName != "root" {
		t.Errorf("recreated pushAgentSpan under nestedCtx = %+v, want traceID %q, parentSpanID %q (orig root span), rootAgentName %q",
			subSpan, ambientTraceHex, orig1.spanID, "root")
	}
	_ = tm.popAgentSpan(nestedCtx, identity{invocationID: "inv-otel-1", agentName: "sub"})
	now = now.Add(time.Second)
	end1 := tm.endInvocation(nestedCtx, inv1ID)
	if end1.traceID != ambientTraceHex || end1.spanID != orig1.spanID || end1.parentSpanID != ambientSpanHex || end1.rootAgentName != "root" {
		t.Errorf("post-eviction endInvocation (after pushAgentSpan) = %+v, want traceID %q, spanID %q, parentSpanID %q, rootAgentName %q",
			end1, ambientTraceHex, orig1.spanID, ambientSpanHex, "root")
	}
	if end1.latency != nil {
		t.Errorf("post-eviction endInvocation after pushAgentSpan has latency %v, want nil", end1.latency)
	}

	// Case 2: Evicted, then ended directly with context.Background() (no ambient OTel context).
	inv2ID := identity{invocationID: "inv-otel-2", agentName: "root"}
	orig2 := tm.invocationSpan(ambientCtx, inv2ID, nil)
	now = now.Add(time.Second)
	tm.invocationSpan(context.Background(), identity{invocationID: "evictor-2", agentName: "other"}, nil)
	now = now.Add(time.Second)
	end2 := tm.endInvocation(context.Background(), inv2ID)
	if end2.traceID != ambientTraceHex || end2.spanID != orig2.spanID || end2.parentSpanID != ambientSpanHex || end2.rootAgentName != "root" {
		t.Errorf("post-eviction endInvocation (no ambient OTel) = %+v, want traceID %q, spanID %q, parentSpanID %q, rootAgentName %q",
			end2, ambientTraceHex, orig2.spanID, ambientSpanHex, "root")
	}
	if end2.latency != nil {
		t.Errorf("post-eviction endInvocation has latency %v, want nil", end2.latency)
	}
}

// TestTraceManager_SharedPluginAcrossIndependentRunnersAndApps verifies that
// sharing a single plugin instance across multiple runners with overlapping
// agent names does not leak root_agent_name or AGENT_STARTING metadata:
//  1. A runner with `first_root -> worker` followed by an independent runner
//     rooted at a distinct `worker` agent reports `root_agent_name="worker"`
//     on the second runner, not `"first_root"`.
//  2. Two apps (`planner -> helper` and `support -> helper`) sharing one
//     plugin keep their own root_agent_name and helper instruction even on a
//     continuation turn that starts at `helper`.
func TestTraceManager_SharedPluginAcrossIndependentRunnersAndApps(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)

	// Part 1: first_root -> worker, then standalone worker.
	workerChild, err := llmagent.New(llmagent.Config{
		Name:  "worker",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "child worker"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(workerChild): %v", err)
	}
	firstRoot, err := llmagent.New(llmagent.Config{
		Name:      "first_root",
		Model:     &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "first root done"), TurnComplete: true}}},
		SubAgents: []agent.Agent{workerChild},
	})
	if err != nil {
		t.Fatalf("llmagent.New(firstRoot): %v", err)
	}
	r1, err := runner.New(runner.Config{
		AppName:           "app-1",
		Agent:             firstRoot,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New(r1): %v", err)
	}
	for _, err := range r1.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r1.Run: %v", err)
		}
	}
	_ = collectRows(t, requests, untilEventType(EventInvocationCompleted))

	standaloneWorker, err := llmagent.New(llmagent.Config{
		Name:  "worker",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "standalone worker"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(standaloneWorker): %v", err)
	}
	r2, err := runner.New(runner.Config{
		AppName:           "app-2",
		Agent:             standaloneWorker,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New(r2): %v", err)
	}
	for _, err := range r2.Run(t.Context(), "u1", "s2", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r2.Run: %v", err)
		}
	}
	r2Rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	for _, row := range r2Rows {
		checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), `"worker"`, "root_agent_name")
	}

	// Part 2: Two apps (planner -> helper and support -> helper) sharing p.
	transferTo := func(target string) *model.LLMResponse {
		return &model.LLMResponse{
			Content: genai.NewContentFromParts([]*genai.Part{
				genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": target}),
			}, genai.RoleModel),
		}
	}
	app1Helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Instruction: "Help with planning.",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: textContent("model", "app1 helper turn 1"), TurnComplete: true},
			{Content: textContent("model", "app1 helper turn 2"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(app1Helper): %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{
		Name:      "planner",
		Model:     &scriptedLLM{responses: []*model.LLMResponse{transferTo("helper")}},
		SubAgents: []agent.Agent{app1Helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(planner): %v", err)
	}
	app2Helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Instruction: "Help with support.",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: textContent("model", "app2 helper turn 1"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(app2Helper): %v", err)
	}
	support, err := llmagent.New(llmagent.Config{
		Name:      "support",
		Model:     &scriptedLLM{responses: []*model.LLMResponse{transferTo("helper")}},
		SubAgents: []agent.Agent{app2Helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(support): %v", err)
	}

	app1Runner, err := runner.New(runner.Config{
		AppName:           "app-plan",
		Agent:             planner,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New(app1Runner): %v", err)
	}
	app2Runner, err := runner.New(runner.Config{
		AppName:           "app-support",
		Agent:             support,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New(app2Runner): %v", err)
	}

	// App 1 turn 1 transfers to app1Helper.
	for _, err := range app1Runner.Run(t.Context(), "u1", "s-plan", textContent("user", "plan 1"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("app1Runner turn 1: %v", err)
		}
	}
	_ = collectRows(t, requests, untilEventType(EventInvocationCompleted))

	// App 2 turn 1 runs support -> app2Helper.
	for _, err := range app2Runner.Run(t.Context(), "u1", "s-support", textContent("user", "support 1"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("app2Runner turn 1: %v", err)
		}
	}
	_ = collectRows(t, requests, untilEventType(EventInvocationCompleted))

	// App 1 turn 2 resumes directly at app1Helper; must report root_agent_name="planner"
	// and app1Helper's own instruction.
	for _, err := range app1Runner.Run(t.Context(), "u1", "s-plan", textContent("user", "plan 2"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("app1Runner turn 2: %v", err)
		}
	}
	app1Turn2Rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	for _, row := range app1Turn2Rows {
		checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), `"planner"`, "root_agent_name")
	}
	helperStart := onlyRowOfType(t, app1Turn2Rows, EventAgentStarting)
	content, ok := jsonColumnValue(t, helperStart, columnContent)
	if !ok {
		t.Fatal("app1 turn 2 helper AGENT_STARTING has no content")
	}
	checkJSON(t, "app1 turn 2 helper AGENT_STARTING content", content, `"Help with planning."`)
}

// TestTraceManager_PeerSubAgentTransferPreservesTransferringAgentSpan runs a
// peer transfer chain planner -> helper1 -> helper2 under an llmagent root.
// Because ADK runs a transfer target inside the transferring agent's live run,
// helper1 is still active when helper2 starts: helper2 must nest under helper1,
// and helper1's AGENT_COMPLETED row must close helper1's own span with
// non-null latency_ms.
func TestTraceManager_PeerSubAgentTransferPreservesTransferringAgentSpan(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	transferTo := func(target string) *model.LLMResponse {
		return &model.LLMResponse{
			Content: genai.NewContentFromParts([]*genai.Part{
				genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": target}),
			}, genai.RoleModel),
		}
	}
	helper2, err := llmagent.New(llmagent.Config{
		Name:  "helper2",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "helper2 done"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper2): %v", err)
	}
	helper1, err := llmagent.New(llmagent.Config{
		Name:  "helper1",
		Model: &scriptedLLM{responses: []*model.LLMResponse{transferTo("helper2")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper1): %v", err)
	}
	planner, err := llmagent.New(llmagent.Config{
		Name:      "planner",
		Model:     &scriptedLLM{responses: []*model.LLMResponse{transferTo("helper1")}},
		SubAgents: []agent.Agent{helper1, helper2},
	})
	if err != nil {
		t.Fatalf("llmagent.New(planner): %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             planner,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	startByAgent := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		name, _ := row["agent"].(string)
		startByAgent[name] = placeOf(row)
	}
	if startByAgent["helper1"].parent != startByAgent["planner"].span {
		t.Errorf("helper1 AGENT_STARTING parent = %q, want planner span %q", startByAgent["helper1"].parent, startByAgent["planner"].span)
	}
	if startByAgent["helper2"].parent != startByAgent["helper1"].span {
		t.Errorf("helper2 AGENT_STARTING parent = %q, want helper1 span %q", startByAgent["helper2"].parent, startByAgent["helper1"].span)
	}
	for _, row := range rowsOfType(rows, EventAgentCompleted) {
		name, _ := row["agent"].(string)
		if got, want := placeOf(row), startByAgent[name]; got != want {
			t.Errorf("%s AGENT_COMPLETED place = %+v, want %+v", name, got, want)
		}
		mustLatency(t, row)
	}
}

// TestTraceManager_ShortCircuitedLastChildRetiredWhenParentPops pins the
// descendant-retirement loop in popAgentSpan when the last child of a parent
// agent is short-circuited in BeforeAgentCallback (so no subsequent sibling
// starts before the parent itself finishes).
func TestTraceManager_ShortCircuitedLastChildRetiredWhenParentPops(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	lastChild, err := llmagent.New(llmagent.Config{Name: "shortcut_last"})
	if err != nil {
		t.Fatalf("llmagent.New(lastChild): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "sequence",
			SubAgents: []agent.Agent{lastChild},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	seqID := identity{invocationID: "inv-last-child", agentName: "sequence", path: "sequence@0"}
	childID := identity{invocationID: "inv-last-child", agentName: "shortcut_last", path: "sequence@0"}

	tm.invocationSpan(ctx, seqID, seq)
	tm.pushAgentSpan(ctx, seqID)
	tm.pushAgentSpan(ctx, childID)
	if got := openSpans(t, tm, "inv-last-child"); got.agents != 2 {
		t.Fatalf("open agent spans before parent pop = %d, want 2", got.agents)
	}
	// shortcut_last was short-circuited in BeforeAgentCallback, so only
	// sequence runs popAgentSpan.
	tm.popAgentSpan(ctx, seqID)
	if got := openSpans(t, tm, "inv-last-child"); got != (spanCounts{}) {
		t.Errorf("openSpans after parent pop = %+v, want empty", got)
	}
}

// TestTraceManager_ConcurrentSiblingsAndInterleavedSameAgentActivations pins:
//  1. Two configured sub-agents a and b under a non-LLM parent running
//     concurrently on the same branch without WithUseSubBranch: starting b
//     while a is running must not delete a's open span, both a and b must
//     parent to par, and both must close their own spans with latency.
//  2. Two concurrent activations of p (p@1 and p@2) with child x (p@1/x@1 and
//     p@2/x@2) on the same branch: popping p@1 while p@2 and p@2/x@2 are still
//     open retires only spans whose parentSpanID lineage reaches p@1, leaving
//     p@2 and p@2/x@2 intact to close with latency.
func TestTraceManager_ConcurrentSiblingsAndInterleavedSameAgentActivations(t *testing.T) {
	ctx := context.Background()

	t.Run("configured sub-agents a and b concurrent on same branch", func(t *testing.T) {
		tm := newTraceManager()
		subA, err := llmagent.New(llmagent.Config{Name: "a"})
		if err != nil {
			t.Fatalf("llmagent.New(a): %v", err)
		}
		subB, err := llmagent.New(llmagent.Config{Name: "b"})
		if err != nil {
			t.Fatalf("llmagent.New(b): %v", err)
		}
		par, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      "par",
				SubAgents: []agent.Agent{subA, subB},
			},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(par): %v", err)
		}

		parID := identity{invocationID: "inv-ab", agentName: "par", path: "par@0"}
		aID := identity{invocationID: "inv-ab", agentName: "a", path: "par@0/a@1"}
		bID := identity{invocationID: "inv-ab", agentName: "b", path: "par@0/b@1"}

		tm.invocationSpan(ctx, parID, par)
		parStart := tm.pushAgentSpan(ctx, parID)
		aStart := tm.pushAgentSpan(ctx, aID)
		bStart := tm.pushAgentSpan(ctx, bID)

		if aStart.parentSpanID != parStart.spanID {
			t.Errorf("a parentSpanID = %q, want par spanID %q", aStart.parentSpanID, parStart.spanID)
		}
		if bStart.parentSpanID != parStart.spanID {
			t.Errorf("b parentSpanID = %q, want par spanID %q", bStart.parentSpanID, parStart.spanID)
		}

		aEnd := tm.popAgentSpan(ctx, aID)
		bEnd := tm.popAgentSpan(ctx, bID)
		parEnd := tm.popAgentSpan(ctx, parID)

		if aEnd.spanID != aStart.spanID || aEnd.latency == nil {
			t.Errorf("a popAgentSpan = %+v, want spanID %q with latency", aEnd, aStart.spanID)
		}
		if bEnd.spanID != bStart.spanID || bEnd.latency == nil {
			t.Errorf("b popAgentSpan = %+v, want spanID %q with latency", bEnd, bStart.spanID)
		}
		if parEnd.spanID != parStart.spanID || parEnd.latency == nil {
			t.Errorf("par popAgentSpan = %+v, want spanID %q with latency", parEnd, parStart.spanID)
		}
	})

	t.Run("interleaved p@1/x@1 and p@2/x@2 on same branch", func(t *testing.T) {
		tm := newTraceManager()
		childX, err := llmagent.New(llmagent.Config{Name: "x"})
		if err != nil {
			t.Fatalf("llmagent.New(x): %v", err)
		}
		parentP, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      "p",
				SubAgents: []agent.Agent{childX},
			},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(p): %v", err)
		}

		p1ID := identity{invocationID: "inv-px", agentName: "p", path: "p@1"}
		p2ID := identity{invocationID: "inv-px", agentName: "p", path: "p@2"}
		x1ID := identity{invocationID: "inv-px", agentName: "x", path: "p@1/x@1"}
		x2ID := identity{invocationID: "inv-px", agentName: "x", path: "p@2/x@2"}

		tm.invocationSpan(ctx, p1ID, parentP)
		p1Start := tm.pushAgentSpan(ctx, p1ID)
		p2Start := tm.pushAgentSpan(ctx, p2ID)
		x1Start := tm.pushAgentSpan(ctx, x1ID)
		x2Start := tm.pushAgentSpan(ctx, x2ID)

		if x1Start.parentSpanID != p1Start.spanID {
			t.Errorf("x@1 parentSpanID = %q, want p@1 spanID %q", x1Start.parentSpanID, p1Start.spanID)
		}
		if x2Start.parentSpanID != p2Start.spanID {
			t.Errorf("x@2 parentSpanID = %q, want p@2 spanID %q", x2Start.parentSpanID, p2Start.spanID)
		}

		// Suppose x@1 was short-circuited in BeforeAgentCallback and p@1
		// finishes while p@2 and p@2/x@2 are still open.
		p1End := tm.popAgentSpan(ctx, p1ID)
		if p1End.spanID != p1Start.spanID || p1End.latency == nil {
			t.Errorf("p@1 popAgentSpan = %+v, want spanID %q with latency", p1End, p1Start.spanID)
		}
		// x@1 must have been retired with p@1, while p@2 and x@2 remain open.
		if got := openSpans(t, tm, "inv-px"); got.agents != 2 {
			t.Fatalf("open agent spans after p@1 pop = %d, want 2 (p@2 and x@2)", got.agents)
		}

		x2End := tm.popAgentSpan(ctx, x2ID)
		p2End := tm.popAgentSpan(ctx, p2ID)
		if x2End.spanID != x2Start.spanID || x2End.parentSpanID != p2Start.spanID || x2End.latency == nil {
			t.Errorf("x@2 popAgentSpan = %+v, want spanID %q parent %q with latency", x2End, x2Start.spanID, p2Start.spanID)
		}
		if p2End.spanID != p2Start.spanID || p2End.latency == nil {
			t.Errorf("p@2 popAgentSpan = %+v, want spanID %q with latency", p2End, p2Start.spanID)
		}
		if got := openSpans(t, tm, "inv-px"); got != (spanCounts{}) {
			t.Errorf("openSpans after p@2 pop = %+v, want empty", got)
		}
	})
}

type dummyRouteInfo struct {
	route string
}

type requestCtxWithNilEmbedded struct {
	context.Context
	*dummyRouteInfo
}

func TestContextInspection_NilEmbeddedStructDoesNotPanic(t *testing.T) {
	_ = (&dummyRouteInfo{route: "test"}).route
	ctx := &requestCtxWithNilEmbedded{
		Context:        context.Background(),
		dummyRouteInfo: nil,
	}
	gotPath := contextPath(ctx)
	if gotPath != "" {
		t.Errorf("contextPath(ctx) = %q, want empty string", gotPath)
	}
	gotParents := contextParentMap(ctx)
	if gotParents != nil {
		t.Errorf("contextParentMap(ctx) = %v, want nil", gotParents)
	}
}

func TestTraceManager_UnrecoveredErrorKeying_TwoFailuresOneRecovers(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	id1 := identity{invocationID: "inv-two-errs", agentName: "agentA", path: "pathA"}
	id2 := identity{invocationID: "inv-two-errs", agentName: "agentB", path: "pathB"}
	errA := errors.New("failure A")
	errB := errors.New("failure B")

	tm.pushAgentSpan(ctx, id1)

	keyA := llmKey(id1)
	keyB := llmKey(id2)

	tm.recordError(ctx, id1, keyA, errA)
	tm.recordError(ctx, id2, keyB, errB)

	// Call B recovers.
	tm.clearError(ctx, id2, keyB)

	// Call A is still unrecovered. popUnclosedAgentSpans must retain errA.
	_, err := tm.popUnclosedAgentSpans(ctx, id1)
	if !errors.Is(err, errA) && (err == nil || err.Error() != errA.Error()) {
		t.Fatalf("popUnclosedAgentSpans err = %v, want %v", err, errA)
	}
}

type failingLLM struct {
	err error
}

func (m *failingLLM) Name() string { return "failing-model" }

func (m *failingLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, m.err)
	}
}

func TestRunner_FailedModelCall_EmitsAgentErrorAndInvocationError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)

	modelErr := errors.New("model failure 404")
	bot, err := llmagent.New(llmagent.Config{
		Name:  "failing-bot",
		Model: &failingLLM{err: modelErr},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             bot,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		_ = err
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationError))
	agentErrorRows := rowsOfType(rows, EventAgentError)
	if len(agentErrorRows) == 0 {
		t.Errorf("expected at least one AGENT_ERROR row, got 0")
	} else {
		row := agentErrorRows[0]
		if row["status"] != "ERROR" {
			t.Errorf("AGENT_ERROR status = %v, want ERROR", row["status"])
		}
		if row["error_message"] != modelErr.Error() {
			t.Errorf("AGENT_ERROR error_message = %v, want %q", row["error_message"], modelErr.Error())
		}
	}

	invErrorRows := rowsOfType(rows, EventInvocationError)
	if len(invErrorRows) == 0 {
		t.Fatalf("expected INVOCATION_ERROR row, got 0")
	}
	invRow := invErrorRows[0]
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
	if invRow["error_message"] != modelErr.Error() {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], modelErr.Error())
	}
}

func TestRunner_ToolError_EmitsInvocationCompleted(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)

	errTool, err := functiontool.New(functiontool.Config{
		Name:        "err_tool",
		Description: "tool that returns an error",
	}, func(ctx agent.Context, args struct{}) (struct{}, error) {
		return struct{}{}, errors.New("tool execution failed")
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}

	m := &scriptedLLM{
		responses: []*model.LLMResponse{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "err_tool",
								Args: map[string]any{},
							},
						},
					},
				},
			},
			{
				Content: textContent("model", "I handled the tool error."),
			},
		},
	}
	bot, err := llmagent.New(llmagent.Config{
		Name:  "tool-bot",
		Model: m,
		Tools: []tool.Tool{errTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             bot,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "run tool"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run unexpected error: %v", err)
		}
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	toolErrorRows := rowsOfType(rows, EventToolError)
	if len(toolErrorRows) == 0 {
		t.Errorf("expected TOOL_ERROR row, got 0")
	}
	invCompletedRows := rowsOfType(rows, EventInvocationCompleted)
	if len(invCompletedRows) == 0 {
		t.Fatalf("expected INVOCATION_COMPLETED row, got 0")
	}
	if invCompletedRows[0]["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invCompletedRows[0]["status"])
	}
}

// untilInvocationEnd is a collectRows condition that holds once the
// invocation's terminal row, INVOCATION_COMPLETED or INVOCATION_ERROR, has
// arrived, so that a test asserting one of them fails fast on the other.
func untilInvocationEnd(rows []map[string]any) bool {
	return untilEventType(EventInvocationCompleted)(rows) || untilEventType(EventInvocationError)(rows)
}

// newTestRunner returns a runner for root that logs to the plugins, the
// analytics plugin first.
func newTestRunner(t *testing.T, root agent.Agent, plugins ...*baseplugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:           "test-app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r
}

// agentErrorCounts returns the number of AGENT_ERROR rows per agent.
func agentErrorCounts(rows []map[string]any) map[string]int {
	counts := map[string]int{}
	for _, row := range rowsOfType(rows, EventAgentError) {
		name, _ := row["agent"].(string)
		counts[name]++
	}
	return counts
}

// TestRunner_BeforeRunCancellation_EmitsInvocationError cancels the
// invocation from a BeforeRunCallback, before any agent starts. The runner
// yields the cancellation and runs the after-run callbacks with no agent
// span open; the invocation must still end with INVOCATION_ERROR.
func TestRunner_BeforeRunCancellation_EmitsInvocationError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	// Registered after p, so that INVOCATION_STARTING is logged before the
	// invocation is canceled.
	canceller, err := baseplugin.New(baseplugin.Config{
		Name: "canceller",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			cancel()
			return nil, context.Canceled
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, p, canceller)

	var runErr error
	for _, err := range r.Run(ctx, "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
		}
	}
	if runErr == nil {
		t.Fatal("r.Run yielded no error, want the cancellation of the BeforeRunCallback")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED for a canceled invocation, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
	if invRow["error_message"] != context.Canceled.Error() {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], context.Canceled.Error())
	}
	if n := len(rowsOfType(rows, EventAgentStarting)); n != 0 {
		t.Errorf("got %d AGENT_STARTING rows, want 0: no agent runs after a BeforeRunCallback error", n)
	}
	if got := agentErrorCounts(rows); len(got) != 0 {
		t.Errorf("AGENT_ERROR rows per agent = %v, want none: no agent was running", got)
	}
}

// TestRunner_BeforeModelCallbackError_EmitsAgentErrorAndInvocationError fails
// a model call from the agent's own BeforeModelCallback. ADK yields the error
// without running OnModelErrorCallback, AfterModelCallback or
// AfterAgentCallback, so the plugin learns of the failure only from the agent
// span that is left open with its model call.
func TestRunner_BeforeModelCallbackError_EmitsAgentErrorAndInvocationError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, errors.New("request rejected by before-model callback")
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, p)

	var runErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
		}
	}
	if runErr == nil {
		t.Fatal("r.Run yielded no error, want the before-model callback error")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED after a before-model callback error, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	const wantMsg = `agent "bot" execution failed before completion`
	agentRow := onlyRowOfType(t, rows, EventAgentError)
	if agentRow["agent"] != "bot" {
		t.Errorf("AGENT_ERROR agent = %v, want bot", agentRow["agent"])
	}
	if agentRow["status"] != "ERROR" {
		t.Errorf("AGENT_ERROR status = %v, want ERROR", agentRow["status"])
	}
	if agentRow["error_message"] != wantMsg {
		t.Errorf("AGENT_ERROR error_message = %v, want %q", agentRow["error_message"], wantMsg)
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
	if invRow["error_message"] != wantMsg {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], wantMsg)
	}
	if n := len(rowsOfType(rows, EventAgentCompleted)); n != 0 {
		t.Errorf("got %d AGENT_COMPLETED rows, want 0: the agent did not complete", n)
	}
}

// TestRunner_SequentialSiblingShortcutThenChildFails_EmitsErrorOnlyForFailingChildAndSequence
// runs a sequential agent whose first child is answered by a
// BeforeAgentCallback, which leaves its span open, and whose second child's
// model call fails. The consumer stops at the first error, as a request
// handler that reports the failure to its caller does, so ADK runs no
// AfterAgentCallback for the second child or the sequence either. Only the
// failing child and the sequence it ran under get AGENT_ERROR rows.
func TestRunner_SequentialSiblingShortcutThenChildFails_EmitsErrorOnlyForFailingChildAndSequence(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	child1, err := llmagent.New(llmagent.Config{
		Name:  "child1",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return textContent("model", "answered by callback"), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(child1): %v", err)
	}
	modelErr := errors.New("model failure 500")
	child2, err := llmagent.New(llmagent.Config{Name: "child2", Model: &failingLLM{err: modelErr}})
	if err != nil {
		t.Fatalf("llmagent.New(child2): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{child1, child2}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	r := newTestRunner(t, seq, p)

	var runErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
			break
		}
	}
	if !errors.Is(runErr, modelErr) && (runErr == nil || runErr.Error() != modelErr.Error()) {
		t.Fatalf("r.Run error = %v, want %v", runErr, modelErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED after a model failure, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	var started []string
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		name, _ := row["agent"].(string)
		started = append(started, name)
	}
	if diff := cmp.Diff([]string{"sequence", "child1", "child2"}, started); diff != "" {
		t.Fatalf("AGENT_STARTING agents mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]int{"child2": 1, "sequence": 1}, agentErrorCounts(rows)); diff != "" {
		t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
	}
	for _, row := range rowsOfType(rows, EventAgentError) {
		if row["status"] != "ERROR" {
			t.Errorf("%v AGENT_ERROR status = %v, want ERROR", row["agent"], row["status"])
		}
		if row["error_message"] != modelErr.Error() {
			t.Errorf("%v AGENT_ERROR error_message = %v, want %q", row["agent"], row["error_message"], modelErr.Error())
		}
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
	if invRow["error_message"] != modelErr.Error() {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], modelErr.Error())
	}
}

// TestRunner_CachedBeforeModelCallbackResponse_EmitsInvocationCompletedOK verifies
// that when a BeforeModelCallback answers a model request with a cached response and
// the caller stops iterating after consuming that response, finalization writes
// INVOCATION_COMPLETED with status OK.
func TestRunner_CachedBeforeModelCallbackResponse_EmitsInvocationCompletedOK(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return &model.LLMResponse{Content: textContent("model", "cached response")}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, p)

	for event, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
		if event != nil && event.LLMResponse.Content != nil {
			break
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR after cached response consumed, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	invRow := onlyRowOfType(t, rows, EventInvocationCompleted)
	if invRow["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
	}
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0", n)
	}
}

// TestRunner_CustomAgentOutputEvent_EmitsInvocationCompletedOK verifies that
// a custom agent whose event carries a typed Output and no Content responded.
// The caller stops reading once it has the event, so ADK runs no
// AfterAgentCallback and the agent's span is still open when the invocation
// terminates; finalization must not report the agent as failed.
func TestRunner_CustomAgentOutputEvent_EmitsInvocationCompletedOK(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	customBot, err := agent.New(agent.Config{
		Name: "custom_bot",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
				event.Output = map[string]any{"answer": 42}
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	r := newTestRunner(t, customBot, p)

	for event, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
		if event != nil && event.Output != nil {
			break
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR after the output event was consumed, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	invRow := onlyRowOfType(t, rows, EventInvocationCompleted)
	if invRow["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
	}
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0", n)
	}
}

// TestOnEvent_FinalContentOrOutputMarksAuthorResponded verifies which session
// events show that their author responded, so that its agent span, left open
// when the invocation terminates, is not reported as failed: an event (final or
// streamed partial) with Content or with a typed Output, as the events of a
// custom agent or of a workflow node carry. An event with neither does not.
func TestOnEvent_FinalContentOrOutputMarksAuthorResponded(t *testing.T) {
	tests := []struct {
		name          string
		event         session.Event
		wantResponded bool
	}{
		{name: "content", event: session.Event{LLMResponse: model.LLMResponse{Content: textContent("model", "hi")}}, wantResponded: true},
		{name: "output", event: session.Event{Output: "result"}, wantResponded: true},
		{name: "content and output", event: session.Event{LLMResponse: model.LLMResponse{Content: textContent("model", "hi")}, Output: "result"}, wantResponded: true},
		{name: "partial content", event: session.Event{LLMResponse: model.LLMResponse{Partial: true, Content: textContent("model", "h")}}, wantResponded: true},
		{name: "partial output", event: session.Event{LLMResponse: model.LLMResponse{Partial: true}, Output: "result"}, wantResponded: true},
		{name: "partial without content or output", event: session.Event{LLMResponse: model.LLMResponse{Partial: true}}, wantResponded: false},
		{name: "neither content nor output", event: session.Event{}, wantResponded: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			event := tt.event
			event.Author = mCtx.agentName
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.OnEventCallback()(mCtx, &event))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			if tt.wantResponded {
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED: the author responded; event types: %v", eventTypes(rows))
				}
				onlyRowOfType(t, rows, EventInvocationCompleted)
				if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
					t.Errorf("got %d AGENT_ERROR rows, want 0", n)
				}
				return
			}
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got INVOCATION_COMPLETED, want INVOCATION_ERROR: the author did not respond; event types: %v", eventTypes(rows))
			}
			onlyRowOfType(t, rows, EventInvocationError)
			if got, want := agentErrorCounts(rows), map[string]int{mCtx.agentName: 1}; !maps.Equal(got, want) {
				t.Errorf("AGENT_ERROR rows per agent = %v, want %v", got, want)
			}
		})
	}
}

// TestRunner_CustomRunError verifies that when an agent's custom Run returns an error,
// the error reaches the caller and finalization writes AGENT_ERROR and INVOCATION_ERROR.
func TestRunner_CustomRunError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	runErr := errors.New("custom run failure")
	customBot, err := agent.New(agent.Config{
		Name: "custom_bot",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, runErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	r := newTestRunner(t, customBot, p)

	var gotErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	if !errors.Is(gotErr, runErr) {
		t.Fatalf("r.Run got err %v, want %v", gotErr, runErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	agentErrRow := onlyRowOfType(t, rows, EventAgentError)
	if agentErrRow["status"] != "ERROR" {
		t.Errorf("AGENT_ERROR status = %v, want ERROR", agentErrRow["status"])
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
}

// TestRunner_PriorPluginBeforeModelCallbackError_EmitsAgentErrorAndInvocationError
// verifies that when a plugin registered before the analytics plugin returns an error
// from BeforeModelCallback (so no model span is opened), finalization writes AGENT_ERROR
// and INVOCATION_ERROR.
func TestRunner_PriorPluginBeforeModelCallbackError_EmitsAgentErrorAndInvocationError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	cbErr := errors.New("prior plugin before model failure")
	priorPlugin, err := baseplugin.New(baseplugin.Config{
		Name: "prior_failing_plugin",
		BeforeModelCallback: func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
			return nil, cbErr
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}

	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "hello")}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	r := newTestRunner(t, bot, priorPlugin, p)

	var gotErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	if gotErr == nil {
		t.Fatal("r.Run returned nil error, want prior plugin before model error")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	agentErrRow := onlyRowOfType(t, rows, EventAgentError)
	if agentErrRow["status"] != "ERROR" {
		t.Errorf("AGENT_ERROR status = %v, want ERROR", agentErrRow["status"])
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invRow["status"])
	}
}

// TestRunner_AfterAgentCallbackError verifies that when an agent's AfterAgentCallback
// returns an error, the error reaches the caller while finalization writes INVOCATION_COMPLETED with status OK.
func TestRunner_AfterAgentCallbackError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	cbErr := errors.New("after agent callback rejected")
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "hello")}}},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return nil, cbErr
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, p)

	var gotErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	if gotErr == nil {
		t.Fatal("r.Run returned nil error, want after agent callback error")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	invRow := onlyRowOfType(t, rows, EventInvocationCompleted)
	if invRow["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
	}
}

// TestTraceManager_RepeatedWorker_AttributionToFailingActivationOnly verifies that
// when the same worker runs repeatedly (e.g. @cached at step 1 and @failed at step 2),
// failure attribution selects only the failing activation's lineage.
func TestTraceManager_RepeatedWorker_AttributionToFailingActivationOnly(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	wfID := identity{invocationID: "inv-repeated", agentName: "workflow", path: "wf@0"}
	w1ID := identity{invocationID: "inv-repeated", agentName: "worker", path: "wf@0/worker@cached"}
	w2ID := identity{invocationID: "inv-repeated", agentName: "worker", path: "wf@0/worker@failed"}

	tm.pushAgentSpan(ctx, wfID)
	w1Start := tm.pushAgentSpan(ctx, w1ID)
	w2Start := tm.pushAgentSpan(ctx, w2ID)
	step2Err := errors.New("model failure on step 2")
	tm.recordError(ctx, w2ID, llmKey(w2ID), step2Err)

	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, wfID)
	if !errors.Is(termErr, step2Err) {
		t.Fatalf("popUnclosedAgentSpans termErr = %v, want %v", termErr, step2Err)
	}

	for _, u := range unclosed {
		if u.span.spanID == w1Start.spanID {
			t.Errorf("worker@cached span %q was attributed with error, want only failing activation", w1Start.spanID)
		}
	}
	if len(unclosed) != 2 {
		t.Fatalf("popped unclosed spans count = %d, want 2 (worker@failed and workflow)", len(unclosed))
	}
	if unclosed[0].span.spanID != w2Start.spanID {
		t.Errorf("popped[0] spanID = %q, want worker@failed spanID %q", unclosed[0].span.spanID, w2Start.spanID)
	}
	if unclosed[0].err != step2Err {
		t.Errorf("worker@failed err = %v, want %v", unclosed[0].err, step2Err)
	}
}

// TestTraceManager_TwoFailuresNeitherRecovers_FinalizesBothActivations verifies that
// when two child model calls fail simultaneously without recovery, both failure lineages
// are finalized, each child retains its corresponding error, and shared ancestors are deduplicated.
func TestTraceManager_TwoFailuresNeitherRecovers_FinalizesBothActivations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	parID := identity{invocationID: "inv-two-fail", agentName: "par", path: "par@0"}
	c1ID := identity{invocationID: "inv-two-fail", agentName: "child1", path: "par@0/c1@1"}
	c2ID := identity{invocationID: "inv-two-fail", agentName: "child2", path: "par@0/c2@1"}

	parStart := tm.pushAgentSpan(ctx, parID)
	c1Start := tm.pushAgentSpan(ctx, c1ID)
	c2Start := tm.pushAgentSpan(ctx, c2ID)

	err1 := errors.New("child1 failure 500")
	err2 := errors.New("child2 failure 503")
	tm.recordError(ctx, c1ID, llmKey(c1ID), err1)
	tm.recordError(ctx, c2ID, llmKey(c2ID), err2)

	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, parID)
	if termErr == nil {
		t.Fatal("popUnclosedAgentSpans returned nil termErr, want an error")
	}

	if len(unclosed) != 3 {
		t.Fatalf("len(unclosed) = %d, want 3 (child1, child2, par)", len(unclosed))
	}

	bySpanID := map[string]unclosedAgentSpan{}
	for _, u := range unclosed {
		bySpanID[u.span.spanID] = u
	}

	u1, ok1 := bySpanID[c1Start.spanID]
	if !ok1 {
		t.Errorf("child1 span %q not in unclosed spans", c1Start.spanID)
	} else if u1.err != err1 {
		t.Errorf("child1 unclosed err = %v, want %v", u1.err, err1)
	}

	u2, ok2 := bySpanID[c2Start.spanID]
	if !ok2 {
		t.Errorf("child2 span %q not in unclosed spans", c2Start.spanID)
	} else if u2.err != err2 {
		t.Errorf("child2 unclosed err = %v, want %v", u2.err, err2)
	}

	uPar, okPar := bySpanID[parStart.spanID]
	if !okPar {
		t.Errorf("shared parent span %q not in unclosed spans", parStart.spanID)
	} else if uPar.err == nil {
		t.Errorf("shared parent unclosed err is nil, want non-nil error")
	}
}

// TestTraceManager_FindAgent_DoesNotMatchDifferentNodePath pins that findAgent never matches
// a span whose recorded path is a different non-empty path, preventing late completions of a
// retired activation from ending a live sibling on another path.
func TestTraceManager_FindAgent_DoesNotMatchDifferentNodePath(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	w2ID := identity{invocationID: "inv-diff-path", agentName: "worker", path: "root/b/w2"}
	tm.pushAgentSpan(ctx, w2ID)

	// Attempting to find or pop worker on a different path "root/a/w1" must not match w2.
	w1ID := identity{invocationID: "inv-diff-path", agentName: "worker", path: "root/a/w1"}
	popped := tm.popAgentSpan(ctx, w1ID)

	// Since w1 was not found in agentStack, popAgentSpan should return a fresh orphan span
	// with no latency, and must NOT pop or return w2's span!
	if popped.latency != nil {
		t.Errorf("popAgentSpan(w1) latency = %v, want nil", popped.latency)
	}

	// Verify w2 is still open and tracked on agentStack.
	inv := tm.invocation(ctx, w2ID)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	idx := inv.findAgent("root/b/w2", "worker", "")
	if idx <= 0 {
		t.Fatalf("worker on root/b/w2 was prematurely popped or lost from agentStack")
	}
}

// streamingLLM is a model.LLM that, when ADK asks it to stream, yields its
// chunks as partial responses. It pauses before each chunk and before its
// last response, which aggregates the chunks or, if err is set, is err.
// leading holds partial responses, such as one that only reports token usage,
// that it yields as soon as the stream opens, before the first pause.
type streamingLLM struct {
	leading []*model.LLMResponse
	chunks  []string
	pause   time.Duration
	err     error
}

func (m *streamingLLM) Name() string { return "streaming-model" }

func (m *streamingLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var text strings.Builder
		for _, res := range m.leading {
			if stream && !yield(res, nil) {
				return
			}
		}
		for _, chunk := range m.chunks {
			time.Sleep(m.pause)
			text.WriteString(chunk)
			if stream && !yield(&model.LLMResponse{Content: textContent("model", chunk), Partial: true}, nil) {
				return
			}
		}
		time.Sleep(m.pause)
		if m.err != nil {
			yield(nil, m.err)
			return
		}
		yield(&model.LLMResponse{Content: textContent("model", text.String()), TurnComplete: true}, nil)
	}
}

// TestRunner_SSEStreaming_ModelCallEndKeepsRequestSpanAndLatency streams a
// model call through the runner in SSE mode. ADK passes each streamed chunk
// to OnEventCallback as a partial event before the stream ends. The chunks
// must not end the model call: the row that ends it, LLM_RESPONSE or
// LLM_ERROR, stays on the span of LLM_REQUEST with the call's total latency
// and time to first token. A leading chunk that only reports token usage does
// not time the first token.
func TestRunner_SSEStreaming_ModelCallEndKeepsRequestSpanAndLatency(t *testing.T) {
	const gap = 20 * time.Millisecond
	modelErr := errors.New("stream broken 503")
	tests := []struct {
		name string
		llm  *streamingLLM
		// wantEnd is the type of the row that ends the model call.
		wantEnd         string
		wantInvocation  string
		wantAgentErrors map[string]int
	}{
		{
			name:            "final response",
			llm:             &streamingLLM{chunks: []string{"Hel", "lo"}, pause: gap},
			wantEnd:         EventLLMResponse,
			wantInvocation:  EventInvocationCompleted,
			wantAgentErrors: map[string]int{},
		},
		{
			name: "final response after a leading usage-only chunk",
			llm: &streamingLLM{
				leading: []*model.LLMResponse{{Partial: true, UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}}},
				chunks:  []string{"Hel", "lo"},
				pause:   gap,
			},
			wantEnd:         EventLLMResponse,
			wantInvocation:  EventInvocationCompleted,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "error after a chunk",
			llm:             &streamingLLM{chunks: []string{"Hel"}, pause: gap, err: modelErr},
			wantEnd:         EventLLMError,
			wantInvocation:  EventInvocationError,
			wantAgentErrors: map[string]int{"bot": 1},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			bot, err := llmagent.New(llmagent.Config{Name: "bot", Model: tc.llm})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			r := newTestRunner(t, bot, p)

			partials := 0
			var runErr error
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
				if err != nil {
					runErr = err
					continue
				}
				if ev != nil && ev.Partial {
					partials++
				}
			}
			if partials != len(tc.llm.chunks) {
				t.Fatalf("r.Run yielded %d partial events, want %d: the model call was not streamed", partials, len(tc.llm.chunks))
			}
			if (runErr == nil) != (tc.llm.err == nil) || (runErr != nil && runErr.Error() != tc.llm.err.Error()) {
				t.Fatalf("r.Run error = %v, want %v", runErr, tc.llm.err)
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			invRow := onlyRowOfType(t, rows, tc.wantInvocation)
			if tc.llm.err == nil && invRow["status"] != "OK" {
				t.Errorf("%s status = %v, want OK", tc.wantInvocation, invRow["status"])
			}
			if diff := cmp.Diff(tc.wantAgentErrors, agentErrorCounts(rows)); diff != "" {
				t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
			}
			request := onlyRowOfType(t, rows, EventLLMRequest)
			end := onlyRowOfType(t, rows, tc.wantEnd)
			if got, want := placeOf(end), placeOf(request); got != want {
				t.Errorf("%s is at %+v, want the place of LLM_REQUEST %+v", tc.wantEnd, got, want)
			}
			latency := mustLatency(t, end)
			if got, want := slices.Sorted(maps.Keys(latency)), []string{"time_to_first_token_ms", "total_ms"}; !slices.Equal(got, want) {
				t.Fatalf("%s latency_ms keys = %v, want %v", tc.wantEnd, got, want)
			}
			ttft, total := latency["time_to_first_token_ms"], latency["total_ms"]
			if wantMin := gap.Milliseconds(); ttft < wantMin || total-ttft < wantMin {
				t.Errorf("%s latency_ms = %v, want time_to_first_token_ms >= %d (the first chunk with generated content) and total_ms >= time_to_first_token_ms + %d (the end of the stream)", tc.wantEnd, latency, wantMin, wantMin)
			}
		})
	}
}

// TestRunner_SubAgentCachedBeforeModelCallbackResponse_EmitsInvocationCompletedOK
// answers a sub-agent's model request from the sub-agent's
// BeforeModelCallback, after which ADK runs no AfterModelCallback, and stops
// reading once the cached response arrives. OnEventCallback gets the agent
// that the runner runs, not the sub-agent, so the response event must end the
// model call of its author and mark the author, and the agents it runs under,
// as responded: the invocation completed.
func TestRunner_SubAgentCachedBeforeModelCallbackResponse_EmitsInvocationCompletedOK(t *testing.T) {
	newSub := func(t *testing.T) agent.Agent {
		t.Helper()
		sub, err := llmagent.New(llmagent.Config{
			Name:  "sub",
			Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
			BeforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{Content: textContent("model", "cached response")}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(sub): %v", err)
		}
		return sub
	}
	tests := []struct {
		name    string
		newRoot func(t *testing.T, sub agent.Agent) agent.Agent
	}{
		{
			// The sequential parent writes no response of its own.
			name: "sequential parent",
			newRoot: func(t *testing.T, sub agent.Agent) agent.Agent {
				t.Helper()
				parent, err := sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{Name: "parent", SubAgents: []agent.Agent{sub}},
				})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
				return parent
			},
		},
		{
			name: "transfer from an LLM parent",
			newRoot: func(t *testing.T, sub agent.Agent) agent.Agent {
				t.Helper()
				transfer := &genai.Content{Role: "model", Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{Name: "transfer_to_agent", Args: map[string]any{"agent_name": "sub"}},
				}}}
				parent, err := llmagent.New(llmagent.Config{
					Name:      "parent",
					Model:     &scriptedLLM{responses: []*model.LLMResponse{{Content: transfer}}},
					SubAgents: []agent.Agent{sub},
				})
				if err != nil {
					t.Fatalf("llmagent.New(parent): %v", err)
				}
				return parent
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			r := newTestRunner(t, tc.newRoot(t, newSub(t)), p)

			gotCached := false
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error: %v", err)
				}
				if ev != nil && ev.Author == "sub" && ev.Content != nil {
					gotCached = true
					break
				}
			}
			if !gotCached {
				t.Fatal("r.Run ended without the sub-agent's cached response")
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got INVOCATION_ERROR after the sub-agent's cached response was consumed, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
			}
			invRow := onlyRowOfType(t, rows, EventInvocationCompleted)
			if invRow["status"] != "OK" {
				t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
			}
			if diff := cmp.Diff(map[string]int{}, agentErrorCounts(rows)); diff != "" {
				t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestTraceManager_ErrorOfNormallyEndedActivation_StillFailsInvocation records
// a model error for an activation that then ends normally, as it does when the
// caller keeps reading after the error and ADK runs the agent's
// AfterAgentCallback. The error still fails the invocation, and a sibling that
// responded is not blamed; when an ancestor ends while a descendant still has
// an unrecovered error, the failing descendant and its ancestors stay open so
// popUnclosedAgentSpans emits AGENT_ERROR for them.
func TestTraceManager_ErrorOfNormallyEndedActivation_StillFailsInvocation(t *testing.T) {
	tests := []struct {
		name string
		// failing is the activation whose model call fails. ended is the
		// activation that ends normally: failing itself, or the activation
		// that failing runs under.
		failing, ended identity
		wantUnclosed   []string
	}{
		{
			name:         "failing activation ends",
			failing:      identity{agentName: "branch_a", path: "root/a"},
			ended:        identity{agentName: "branch_a", path: "root/a"},
			wantUnclosed: nil,
		},
		{
			name:         "ancestor of failing activation ends",
			failing:      identity{agentName: "worker", path: "root/a/worker"},
			ended:        identity{agentName: "branch_a", path: "root/a"},
			wantUnclosed: []string{"worker"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			tm := newTraceManager()
			const invocationID = "inv-ended-activation"
			rootID := identity{invocationID: invocationID, agentName: "root", path: "root"}
			endedID, failingID := tc.ended, tc.failing
			endedID.invocationID, failingID.invocationID = invocationID, invocationID
			siblingID := identity{invocationID: invocationID, agentName: "branch_b", path: "root/b"}

			tm.pushAgentSpan(ctx, rootID)
			tm.pushLLMSpan(ctx, rootID)
			tm.popLLMSpan(ctx, rootID, true)
			tm.pushAgentSpan(ctx, endedID)
			if failingID != endedID {
				tm.pushAgentSpan(ctx, failingID)
			}
			tm.pushLLMSpan(ctx, failingID)
			modelErr := errors.New("model failure 500")
			tm.recordError(ctx, failingID, llmKey(failingID), modelErr)
			tm.popLLMSpan(ctx, failingID, false)
			tm.popAgentSpan(ctx, endedID)
			tm.pushAgentSpan(ctx, siblingID)
			tm.pushLLMSpan(ctx, siblingID)
			tm.popLLMSpan(ctx, siblingID, true)

			unclosed, termErr := tm.popUnclosedAgentSpans(ctx, rootID)
			if !errors.Is(termErr, modelErr) {
				t.Errorf("popUnclosedAgentSpans termErr = %v, want %v", termErr, modelErr)
			}
			var got []string
			for _, u := range unclosed {
				got = append(got, u.agentName)
				if !errors.Is(u.err, modelErr) {
					t.Errorf("unclosed %q err = %v, want %v", u.agentName, u.err, modelErr)
				}
			}
			if diff := cmp.Diff(tc.wantUnclosed, got); diff != "" {
				t.Errorf("popUnclosedAgentSpans agents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestTraceManager_ErrorOfUnknownActivation_FailsInvocationWithoutAgentErrors
// records a model error for an activation that has no span, so the error is
// not attributed to any open span. The invocation fails with the error, but
// the open spans are not returned: they did not fail.
func TestTraceManager_ErrorOfUnknownActivation_FailsInvocationWithoutAgentErrors(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invocationID = "inv-unknown-activation"
	rootID := identity{invocationID: invocationID, agentName: "root", path: "root"}
	siblingID := identity{invocationID: invocationID, agentName: "branch_b", path: "root/b"}
	unknownID := identity{invocationID: invocationID, agentName: "unknown", path: "root/unknown"}

	tm.pushAgentSpan(ctx, rootID)
	tm.pushAgentSpan(ctx, siblingID)
	tm.pushLLMSpan(ctx, siblingID)
	tm.popLLMSpan(ctx, siblingID, true)
	modelErr := errors.New("model failure 500")
	tm.recordError(ctx, unknownID, llmKey(unknownID), modelErr)

	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, rootID)
	if !errors.Is(termErr, modelErr) {
		t.Errorf("popUnclosedAgentSpans termErr = %v, want %v", termErr, modelErr)
	}
	for _, u := range unclosed {
		t.Errorf("popUnclosedAgentSpans returned %q with error %v, want no span", u.agentName, u.err)
	}
}

// TestTraceManager_CancellationAttributesEveryOpenSpan terminates a canceled
// invocation without any recorded error. The failing activation is unknown,
// so every open span is returned with the cancellation, including a span
// whose agent already responded.
func TestTraceManager_CancellationAttributesEveryOpenSpan(t *testing.T) {
	tm := newTraceManager()
	const invocationID = "inv-canceled"
	rootID := identity{invocationID: invocationID, agentName: "root", path: "root"}
	childID := identity{invocationID: invocationID, agentName: "child", path: "root/child"}
	rootStart := tm.pushAgentSpan(context.Background(), rootID)
	childStart := tm.pushAgentSpan(context.Background(), childID)
	tm.pushLLMSpan(context.Background(), childID)
	tm.popLLMSpan(context.Background(), childID, true)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, rootID)
	if !errors.Is(termErr, context.Canceled) {
		t.Errorf("popUnclosedAgentSpans termErr = %v, want %v", termErr, context.Canceled)
	}
	var got []string
	for _, u := range unclosed {
		got = append(got, u.span.spanID)
		if !errors.Is(u.err, context.Canceled) {
			t.Errorf("%q err = %v, want %v", u.agentName, u.err, context.Canceled)
		}
	}
	if diff := cmp.Diff([]string{childStart.spanID, rootStart.spanID}, got); diff != "" {
		t.Errorf("popUnclosedAgentSpans span IDs mismatch (-want +got):\n%s", diff)
	}
}

// requireInvocationCompletedOK fails the test unless rows end the invocation
// with INVOCATION_COMPLETED, status OK, and hold neither an INVOCATION_ERROR
// nor an AGENT_ERROR row: the run is reported as completed.
func requireInvocationCompletedOK(t *testing.T, rows []map[string]any) {
	t.Helper()
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	invRow := onlyRowOfType(t, rows, EventInvocationCompleted)
	if invRow["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
	}
	if diff := cmp.Diff(map[string]int{}, agentErrorCounts(rows)); diff != "" {
		t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
	}
}

// TestRunner_AfterToolShortCircuit_EarlyStopDoesNotFail runs an agent
// that calls a tool and then answers. A plugin registered before the analytics
// plugin may return a result from AfterToolCallback, which ends the chain of
// plugin callbacks: the analytics plugin never sees the call complete although
// it opened the call's span. The consumer stops reading at the final answer, so
// ADK runs no AfterAgentCallback either. The function response event shows
// that the call completed, so the successful run must not be reported as
// failed for a tool call that is still open.
func TestRunner_AfterToolShortCircuit_EarlyStopDoesNotFail(t *testing.T) {
	tests := []struct {
		name string
		// earlierResult is what the earlier plugin's AfterToolCallback returns.
		earlierResult     map[string]any
		wantToolCompleted int
	}{
		{
			name:              "earlier plugin replaces the tool result",
			earlierResult:     map[string]any{"status": "replaced"},
			wantToolCompleted: 0,
		},
		{
			name:              "earlier plugin leaves the tool result",
			wantToolCompleted: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			earlier, err := baseplugin.New(baseplugin.Config{
				Name: "earlier_after_tool",
				AfterToolCallback: func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
					return tc.earlierResult, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New: %v", err)
			}
			lookup, err := functiontool.New(functiontool.Config{
				Name:        "lookup",
				Description: "looks up a value",
			}, func(agent.Context, struct{}) (struct{}, error) {
				return struct{}{}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			callLookup := &model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
				genai.NewPartFromFunctionCall("lookup", map[string]any{}),
			}, genai.RoleModel)}
			bot, err := llmagent.New(llmagent.Config{
				Name: "bot",
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					callLookup,
					{Content: textContent("model", "all done")},
				}},
				Tools: []tool.Tool{lookup},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			r := newTestRunner(t, bot, earlier, p)

			var events int
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "look it up"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error after %d events: %v", events, err)
				}
				if ev == nil {
					continue
				}
				events++
				if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "all done" {
					break
				}
			}
			if events != 3 {
				t.Fatalf("consumed %d events, want 3: the function call, its response and the final answer", events)
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolStarting)); n != 1 {
				t.Errorf("got %d TOOL_STARTING rows, want 1", n)
			}
			if n := len(rowsOfType(rows, EventToolCompleted)); n != tc.wantToolCompleted {
				t.Errorf("got %d TOOL_COMPLETED rows, want %d", n, tc.wantToolCompleted)
			}
			requireInvocationCompletedOK(t, rows)
		})
	}
}

// newShortcutSequenceGraph returns the workflow agent "root", whose only node
// is the sequential agent "sequence" of the agent "shortcut", which a
// BeforeAgentCallback answers, and next. The sequence is a graph node, so it is
// not among the SubAgents of root unless registerSequence is set.
func newShortcutSequenceGraph(t *testing.T, next agent.Agent, registerSequence bool) agent.Agent {
	t.Helper()
	shortcut, err := llmagent.New(llmagent.Config{
		Name:  "shortcut",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return textContent("model", "answered by callback"), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(shortcut): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{shortcut, next}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(seq, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	cfg := workflowagent.Config{
		Name:  "root",
		Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	}
	if registerSequence {
		cfg.SubAgents = []agent.Agent{seq}
	}
	root, err := workflowagent.New(cfg)
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	return root
}

// TestRunner_GraphSequentialSiblingAfterShortcut runs a sequential agent as a
// node of a workflow graph. Its first child is answered by a
// BeforeAgentCallback, which leaves the child's span open, and its second
// child must still run under the sequence rather than under that sibling. A
// graph node is not among the SubAgents of the runner's root agent, so the
// hierarchy of the sequence is not known from the configured agent tree.
func TestRunner_GraphSequentialSiblingAfterShortcut(t *testing.T) {
	tests := []struct {
		name             string
		registerSequence bool
	}{
		{name: "sequence only as a graph node"},
		{name: "sequence also in the SubAgents of root", registerSequence: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			next, err := llmagent.New(llmagent.Config{
				Name:  "next",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "next answer")}}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(next): %v", err)
			}
			r := newTestRunner(t, newShortcutSequenceGraph(t, next, tc.registerSequence), p)

			for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error: %v", err)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			starts := rowsOfType(rows, EventAgentStarting)
			if len(starts) != 4 {
				t.Fatalf("got %d AGENT_STARTING rows, want 4; event types: %v", len(starts), eventTypes(rows))
			}
			places := map[string]tracePlace{}
			for _, row := range starts {
				name, _ := row["agent"].(string)
				places[name] = placeOf(row)
			}
			if diff := cmp.Diff([]string{"next", "root", "sequence", "shortcut"}, slices.Sorted(maps.Keys(places))); diff != "" {
				t.Fatalf("AGENT_STARTING agents mismatch (-want +got):\n%s", diff)
			}
			if got, want := places["sequence"].parent, places["root"].span; got != want {
				t.Errorf("sequence parent_span_id = %q, want the span of root %q", got, want)
			}
			if got, want := places["shortcut"].parent, places["sequence"].span; got != want {
				t.Errorf("shortcut parent_span_id = %q, want the span of sequence %q", got, want)
			}
			if got, want := places["next"].parent, places["sequence"].span; got != want {
				t.Errorf("next parent_span_id = %q, want the span of sequence %q, not that of shortcut %q", got, want, places["shortcut"].span)
			}
			requireInvocationCompletedOK(t, rows)
		})
	}
}

// TestRunner_GraphSequentialFailureDoesNotBlameShortcut fails the second child
// of a sequential graph node after the first child was answered by a
// BeforeAgentCallback. The consumer stops at the first error, so ADK runs no
// AfterAgentCallback. Only the failing child and the agents it ran under get
// AGENT_ERROR rows; the sibling that answered did not fail.
func TestRunner_GraphSequentialFailureDoesNotBlameShortcut(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	modelErr := errors.New("model failure 500")
	next, err := llmagent.New(llmagent.Config{Name: "next", Model: &failingLLM{err: modelErr}})
	if err != nil {
		t.Fatalf("llmagent.New(next): %v", err)
	}
	r := newTestRunner(t, newShortcutSequenceGraph(t, next, false), p)

	var runErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
			break
		}
	}
	if !errors.Is(runErr, modelErr) && (runErr == nil || runErr.Error() != modelErr.Error()) {
		t.Fatalf("r.Run error = %v, want %v", runErr, modelErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED after a model failure, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	if diff := cmp.Diff(map[string]int{"next": 1, "sequence": 1, "root": 1}, agentErrorCounts(rows)); diff != "" {
		t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
	}
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if invRow["error_message"] != modelErr.Error() {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], modelErr.Error())
	}
}

// TestRunner_SharedSameAgentRoot shares one analytics plugin between a runner
// whose tree is root -> worker and a runner of the same worker instance as an
// independent root. The parent map of each runner is authoritative, including
// an empty one: the root learned from the other runner's tree must not become
// the root agent of the independent runner's invocation.
func TestRunner_SharedSameAgentRoot(t *testing.T) {
	tests := []struct {
		name      string
		treeFirst bool
	}{
		{name: "tree runner first", treeFirst: true},
		{name: "standalone runner first"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			worker, err := llmagent.New(llmagent.Config{
				Name:  "worker",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "worker answer"), TurnComplete: true}}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			root, err := llmagent.New(llmagent.Config{
				Name:      "root",
				Model:     &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "root answer"), TurnComplete: true}}},
				SubAgents: []agent.Agent{worker},
			})
			if err != nil {
				t.Fatalf("llmagent.New(root): %v", err)
			}
			// run runs r to its end and checks the root agent of every row.
			run := func(r *runner.Runner, wantRoot string) {
				t.Helper()
				for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("r.Run (root %s): %v", wantRoot, err)
					}
				}
				rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
				for _, row := range rows {
					checkJSONAt(t, jsonObjectColumn(t, row, columnAttributes), fmt.Sprintf("%q", wantRoot), "root_agent_name")
				}
			}
			tree := func() { run(newTestRunner(t, root, p), "root") }
			standalone := func() { run(newTestRunner(t, worker, p), "worker") }
			if tc.treeFirst {
				tree()
				standalone()
			} else {
				standalone()
				tree()
			}
		})
	}
}

// TestRunner_CustomParentSwallowsShortCircuitedChild_EmitsInvocationCompletedOK
// runs a custom agent that runs its sub-agent itself, as a sequential agent
// does, and keeps the sub-agent's events to itself. A BeforeAgentCallback
// answers for the sub-agent, which leaves its span open, and its event never
// reaches the runner. The consumer stops at the parent's own final event, so
// ADK runs no AfterAgentCallback. That final event shows that the run
// succeeded; the span left open by the sub-agent must not fail it.
func TestRunner_CustomParentSwallowsShortCircuitedChild_EmitsInvocationCompletedOK(t *testing.T) {
	tests := []struct {
		name string
		// respond sets the final response of the parent on its event, and
		// isFinal reports whether an event carries it.
		respond func(*session.Event)
		isFinal func(*session.Event) bool
	}{
		{
			name:    "content",
			respond: func(ev *session.Event) { ev.Content = textContent("model", "parent answer") },
			isFinal: func(ev *session.Event) bool { return ev.Content != nil },
		},
		{
			name:    "output",
			respond: func(ev *session.Event) { ev.Output = map[string]any{"answer": 42} },
			isFinal: func(ev *session.Event) bool { return ev.Output != nil },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			sub, err := llmagent.New(llmagent.Config{
				Name:  "sub",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						return textContent("model", "answered by callback"), nil
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(sub): %v", err)
			}
			parent, err := agent.New(agent.Config{
				Name:      "parent",
				SubAgents: []agent.Agent{sub},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for _, err := range sub.Run(ctx) {
							_ = err
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						tc.respond(ev)
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}
			r := newTestRunner(t, parent, p)

			var gotFinal bool
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error: %v", err)
				}
				if ev != nil && ev.Author == "parent" && tc.isFinal(ev) {
					gotFinal = true
					break
				}
			}
			if !gotFinal {
				t.Fatal("r.Run ended without the final event of the parent")
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventAgentStarting)); n != 2 {
				t.Errorf("got %d AGENT_STARTING rows, want 2 (parent and sub)", n)
			}
			requireInvocationCompletedOK(t, rows)
		})
	}
}

// TestRunner_SequentialSiblingShortcutThenCustomChildFails_BlamesOnlyFailingChild
// is the counterpart of the test above: the sibling that a BeforeAgentCallback
// answered is settled by nothing, and the second child of the sequence, a
// custom agent that fails without any model call, still gets an AGENT_ERROR row
// together with the sequence.
func TestRunner_SequentialSiblingShortcutThenCustomChildFails_BlamesOnlyFailingChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	child1, err := llmagent.New(llmagent.Config{
		Name:  "child1",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return textContent("model", "answered by callback"), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(child1): %v", err)
	}
	runErr := errors.New("custom child failure")
	child2, err := agent.New(agent.Config{
		Name: "child2",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, runErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child2): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{child1, child2}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	r := newTestRunner(t, seq, p)

	var gotErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	if !errors.Is(gotErr, runErr) {
		t.Fatalf("r.Run error = %v, want %v", gotErr, runErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED after a child failure, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	if diff := cmp.Diff(map[string]int{"child2": 1, "sequence": 1}, agentErrorCounts(rows)); diff != "" {
		t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
	}
	onlyRowOfType(t, rows, EventInvocationError)
}

// TestRunner_PriorPluginAnswersOnlyChild_EmitsInvocationCompletedOK runs a
// sequential agent whose only child is answered by the BeforeAgentCallback of
// a plugin registered before the analytics plugin, so the analytics plugin
// never opens the child's span. The child's answer reaches it as an event of an
// agent that it has no span for. When the consumer stops at that answer, ADK
// runs no AfterAgentCallback for the sequence, whose span is still open; the
// answer shows that the run succeeded.
func TestRunner_PriorPluginAnswersOnlyChild_EmitsInvocationCompletedOK(t *testing.T) {
	tests := []struct {
		name  string
		drain bool
	}{
		{name: "consumer stops at the answer of the child"},
		{name: "consumer drains the run", drain: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			earlier, err := baseplugin.New(baseplugin.Config{
				Name: "earlier_before_agent",
				BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
					if ctx.AgentName() == "child" {
						return textContent("model", "answered by an earlier plugin"), nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New: %v", err)
			}
			child, err := llmagent.New(llmagent.Config{
				Name:  "child",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(child): %v", err)
			}
			seq, err := sequentialagent.New(sequentialagent.Config{
				AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{child}},
			})
			if err != nil {
				t.Fatalf("sequentialagent.New: %v", err)
			}
			r := newTestRunner(t, seq, earlier, p)

			var gotAnswer bool
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error: %v", err)
				}
				if ev != nil && ev.Author == "child" && ev.Content != nil {
					gotAnswer = true
					if !tc.drain {
						break
					}
				}
			}
			if !gotAnswer {
				t.Fatal("r.Run ended without the answer of the child")
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventAgentStarting)); n != 1 {
				t.Errorf("got %d AGENT_STARTING rows, want 1 (the sequence): the earlier plugin answers for the child first", n)
			}
			requireInvocationCompletedOK(t, rows)
		})
	}
}

// TestRunner_RecoveredModelErrorWithSkippedAfterModel_EmitsInvocationCompletedOK
// fails a model call, recovers from the error in the agent's own
// OnModelErrorCallback, and has a plugin registered before the analytics plugin
// return the recovered response from its AfterModelCallback. That ends the
// chain of plugin callbacks, so the analytics plugin sees the error but never
// the response that recovers from it. The consumer stops at the final
// response; it shows that the run succeeded.
func TestRunner_RecoveredModelErrorWithSkippedAfterModel_EmitsInvocationCompletedOK(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	earlier, err := baseplugin.New(baseplugin.Config{
		Name: "earlier_after_model",
		AfterModelCallback: func(_ agent.Context, res *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			return res, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &failingLLM{err: errors.New("model failure 503")},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
			func(agent.Context, *model.LLMRequest, error) (*model.LLMResponse, error) {
				return &model.LLMResponse{Content: textContent("model", "fallback answer")}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, earlier, p)

	var gotAnswer bool
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
		if ev != nil && ev.Content != nil {
			gotAnswer = true
			break
		}
	}
	if !gotAnswer {
		t.Fatal("r.Run ended without the fallback answer")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
		t.Errorf("got %d LLM_ERROR rows, want 1", n)
	}
	if n := len(rowsOfType(rows, EventLLMResponse)); n != 0 {
		t.Errorf("got %d LLM_RESPONSE rows, want 0: the earlier plugin ends the AfterModelCallback chain", n)
	}
	requireInvocationCompletedOK(t, rows)
}

// TestRunner_EventReplacingPluginAfterAnalytics_EmitsInvocationCompletedOK pins
// the registration order that the constructors document: the analytics plugin
// is registered before a plugin whose OnEventCallback returns the event, which
// ends the chain of event callbacks. The analytics plugin sees the event of the
// custom agent, which carries an output and no content, so the run that the
// consumer stops at that event is reported as completed.
func TestRunner_EventReplacingPluginAfterAnalytics_EmitsInvocationCompletedOK(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	replacer, err := baseplugin.New(baseplugin.Config{
		Name: "event_replacer",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			ev.CustomMetadata = map[string]any{"replaced": true}
			return ev, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	customBot, err := agent.New(agent.Config{
		Name: "custom_bot",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.NodeInfo = &session.NodeInfo{Path: "custom_bot"}
				ev.Output = map[string]any{"answer": 42}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	r := newTestRunner(t, customBot, p, replacer)

	var gotReplaced bool
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
		if ev != nil && ev.Output != nil {
			gotReplaced = ev.CustomMetadata["replaced"] == true
			break
		}
	}
	if !gotReplaced {
		t.Fatal("r.Run ended without the event that the second plugin replaced")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventNodeOutput)); n != 1 {
		t.Errorf("got %d %s rows, want 1: the analytics plugin sees the event first", n, EventNodeOutput)
	}
	requireInvocationCompletedOK(t, rows)
}

// runSubAgentFailure runs a custom parent agent that runs a custom sub-agent
// itself, as a sequential agent does. The sub-agent writes an event that only
// changes the state and then fails with childErr, which the parent returns and
// no callback observes. When parentResponds is set, the parent writes an event
// with content after each event of the sub-agent, so it responds while the
// sub-agent's span is open. It returns the rows of the invocation and the first
// error that the runner gave its caller.
func runSubAgentFailure(t *testing.T, childErr error, parentResponds bool) ([]map[string]any, error) {
	t.Helper()
	p, requests, _ := setupTestPlugin(t)
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Actions.StateDelta = map[string]any{"step": 1}
				if yield(ev, nil) {
					yield(nil, childErr)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for ev, err := range child.Run(ctx) {
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(ev, nil) {
						return
					}
					if parentResponds {
						progress := session.NewEvent(ctx, ctx.InvocationID())
						progress.Content = textContent("model", "working on it")
						if !yield(progress, nil) {
							return
						}
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}
	r := newTestRunner(t, parent, p)

	var gotErr error
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			gotErr = err
			break
		}
	}
	return collectRows(t, requests, untilInvocationEnd), gotErr
}

// TestRunner_SubAgentFailsAfterParentProducedContent_EmitsInvocationCompletedOK
// pins a documented limit of the failure detection. A custom parent writes an
// event with content while its sub-agent runs, and the sub-agent then fails
// without any callback observing the error: the runner returns the error to
// its caller. The parent's event settles the activations that still run under
// it, the sub-agent's among them, because the callbacks cannot tell the
// sub-agent apart from one that a callback answered and whose events the
// parent swallowed, as in
// TestRunner_CustomParentSwallowsShortCircuitedChild_EmitsInvocationCompletedOK.
// So the invocation ends with INVOCATION_COMPLETED and no AGENT_ERROR row.
// popUnclosedAgentSpans and afterRun document this: a change that reports
// this failure has to update them and this test together.
func TestRunner_SubAgentFailsAfterParentProducedContent_EmitsInvocationCompletedOK(t *testing.T) {
	childErr := errors.New("child failure")
	rows, gotErr := runSubAgentFailure(t, childErr, true)
	if !errors.Is(gotErr, childErr) {
		t.Fatalf("r.Run error = %v, want %v", gotErr, childErr)
	}
	if n := len(rowsOfType(rows, EventAgentStarting)); n != 2 {
		t.Errorf("got %d AGENT_STARTING rows, want 2 (parent and child)", n)
	}
	requireInvocationCompletedOK(t, rows)
}

// TestRunner_SubAgentFailsWithoutParentContent_EmitsInvocationError is the
// counterpart of the test above: when the parent writes no event with content
// while the sub-agent runs, nothing settles the sub-agent's activation, so its
// failure is reported with an AGENT_ERROR row for the sub-agent and one for the
// parent that runs it, and the invocation ends with INVOCATION_ERROR.
func TestRunner_SubAgentFailsWithoutParentContent_EmitsInvocationError(t *testing.T) {
	childErr := errors.New("child failure")
	rows, gotErr := runSubAgentFailure(t, childErr, false)
	if !errors.Is(gotErr, childErr) {
		t.Fatalf("r.Run error = %v, want %v", gotErr, childErr)
	}
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED after the sub-agent failure, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	if diff := cmp.Diff(map[string]int{"child": 1, "parent": 1}, agentErrorCounts(rows)); diff != "" {
		t.Errorf("AGENT_ERROR rows per agent mismatch (-want +got):\n%s", diff)
	}
	onlyRowOfType(t, rows, EventInvocationError)
}

// TestTraceManager_FunctionResponseRetiresOnlyItsToolCall ends the tool calls
// that the function responses of a final event answer, and no others. An
// earlier plugin's AfterToolCallback that returns a result stops the chain
// before this plugin's, so the response event is all that ends the call. A call
// that has not responded keeps failing the invocation, and so does a call of
// another activation, even one that has the same function call ID.
func TestTraceManager_FunctionResponseRetiresOnlyItsToolCall(t *testing.T) {
	type toolCall struct{ id, name string }
	tests := []struct {
		name string
		// rootCalls and workerCalls are the tool calls that are open under the
		// root agent and under its sub-agent. responses is what
		// functionResponseCalls returns for the final event of the root agent.
		rootCalls, workerCalls []toolCall
		responses              []string
		wantOpen               int
	}{
		{
			name:      "response for every call",
			rootCalls: []toolCall{{"call-1", "lookup"}, {"call-2", "search"}},
			responses: []string{"call-1", "call-2"},
		},
		{
			name:      "response for one of two calls",
			rootCalls: []toolCall{{"call-1", "lookup"}, {"call-2", "search"}},
			responses: []string{"call-1"},
			wantOpen:  1,
		},
		{
			name:      "no function response",
			rootCalls: []toolCall{{"call-1", "lookup"}},
			wantOpen:  1,
		},
		{
			name:      "call without an ID is matched by tool name",
			rootCalls: []toolCall{{"", "lookup"}},
			responses: []string{"lookup"},
		},
		{
			name:        "call of another activation with the same ID",
			rootCalls:   []toolCall{{"call-1", "lookup"}},
			workerCalls: []toolCall{{"call-1", "lookup"}},
			responses:   []string{"call-1"},
			wantOpen:    1,
		},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			tm := newTraceManager()
			invocationID := fmt.Sprintf("inv-function-response-%d", i)
			rootID := identity{invocationID: invocationID, agentName: "root", path: "root"}
			workerID := identity{invocationID: invocationID, agentName: "worker", path: "root/worker"}
			tm.pushAgentSpan(ctx, rootID)
			tm.pushAgentSpan(ctx, workerID)
			for _, c := range tc.rootCalls {
				tm.pushToolSpan(ctx, rootID, toolKey(rootID, c.id, c.name))
			}
			for _, c := range tc.workerCalls {
				tm.pushToolSpan(ctx, workerID, toolKey(workerID, c.id, c.name))
			}

			tm.completeResponse(ctx, rootID, "root", "", tc.responses)

			if got := openSpans(t, tm, invocationID).tools; got != tc.wantOpen {
				t.Errorf("open tool calls = %d, want %d", got, tc.wantOpen)
			}
			tm.mu.Lock()
			inv := tm.invocations[invocationID]
			tm.mu.Unlock()
			if got, want := inv.hasOpenCalls(), tc.wantOpen > 0; got != want {
				t.Errorf("hasOpenCalls() = %v, want %v: the count that protects an invocation from eviction follows the open calls", got, want)
			}
			_, termErr := tm.popUnclosedAgentSpans(ctx, rootID)
			if got, want := termErr != nil, tc.wantOpen > 0; got != want {
				t.Errorf("popUnclosedAgentSpans error = %v, want a failure: %v", termErr, want)
			}
		})
	}
}

// TestTraceManager_FinalEventMarksActivationsAsResponded places the final event
// of an agent. The event settles the activations that it shows have responded,
// so that the invocation does not fail for them when ADK ends it without
// running their AfterAgentCallback: the activation of the author, the
// activations that it runs under and the ones that run under it or, if the
// author has no open span, the activation that it would run under. An event of
// the user or without an author settles nothing, and an event of one sibling
// does not settle another.
func TestTraceManager_FinalEventMarksActivationsAsResponded(t *testing.T) {
	root := identity{agentName: "root", path: "root"}
	sub := identity{agentName: "sub", path: "root/sub"}
	a := identity{agentName: "a", path: "root/a"}
	b := identity{agentName: "b", path: "root/b"}
	tests := []struct {
		name string
		// running holds the open activations in the order they started.
		running []identity
		// author and path are the author and the node path of the event.
		author, path string
		// wantBlamed is the activation that popUnclosedAgentSpans reports
		// as failed, if any.
		wantBlamed string
	}{
		{
			name:    "event of an agent settles a sub-agent that runs under it",
			running: []identity{root, sub},
			author:  "root",
			path:    "root",
		},
		{
			name:    "event of a sub-agent settles the agent that it runs under",
			running: []identity{root, sub},
			author:  "sub",
			path:    "root/sub",
		},
		{
			name:    "event of an agent without an open span settles the activation that it would run under",
			running: []identity{root},
			author:  "late",
			path:    "root/late",
		},
		{
			name:       "event of a sibling leaves the other sibling unsettled",
			running:    []identity{root, a, b},
			author:     "a",
			path:       "root/a",
			wantBlamed: "b",
		},
		{
			name:       "event of the user settles nothing",
			running:    []identity{root},
			author:     userAuthor,
			path:       "root",
			wantBlamed: "root",
		},
		{
			name:       "event without an author settles nothing",
			running:    []identity{root},
			path:       "root",
			wantBlamed: "root",
		},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			tm := newTraceManager()
			invocationID := fmt.Sprintf("inv-final-event-%d", i)
			for _, id := range tc.running {
				id.invocationID = invocationID
				tm.pushAgentSpan(ctx, id)
			}

			// The event callback gets the agent that the runner runs, and the
			// event brings its own node path.
			eventID := identity{invocationID: invocationID, agentName: "root", path: tc.path}
			tm.completeResponse(ctx, eventID, tc.author, "", nil)

			rootID := identity{invocationID: invocationID, agentName: "root", path: "root"}
			unclosed, termErr := tm.popUnclosedAgentSpans(ctx, rootID)
			var gotBlamed string
			if len(unclosed) > 0 {
				gotBlamed = unclosed[0].agentName
			}
			if gotBlamed != tc.wantBlamed {
				t.Errorf("popUnclosedAgentSpans blames %q, want %q", gotBlamed, tc.wantBlamed)
			}
			if got, want := termErr != nil, tc.wantBlamed != ""; got != want {
				t.Errorf("popUnclosedAgentSpans error = %v, want a failure: %v", termErr, want)
			}
		})
	}
}

// TestTraceManager_FinalEventClearsTheErrorsOfItsActivation records model
// errors, then places the final event of an agent. A response from the
// activation that failed shows that it recovered, as when a plugin recovers
// from the error and an earlier plugin's AfterModelCallback answers the call,
// which hides the response from this plugin's AfterModelCallback. The errors of
// other activations stay, including those of activations that run under the
// author: the event does not show that the author handled them.
func TestTraceManager_FinalEventClearsTheErrorsOfItsActivation(t *testing.T) {
	root := identity{agentName: "root", path: "root"}
	a := identity{agentName: "a", path: "root/a"}
	b := identity{agentName: "b", path: "root/b"}
	worker := identity{agentName: "worker", path: "root/a/worker"}
	tests := []struct {
		name string
		// running holds the open activations in the order they started, and
		// failing the ones whose model call failed.
		running, failing []identity
		// author is the author of the final event.
		author identity
		// wantFailed lists the activations that popUnclosedAgentSpans reports
		// as failed, most recent first, and wantErr is its error.
		wantFailed []string
		wantErr    string
	}{
		{
			name:    "error of the activation that responded",
			running: []identity{root, a},
			failing: []identity{a},
			author:  a,
		},
		{
			name:       "error of a sibling activation",
			running:    []identity{root, a, b},
			failing:    []identity{a, b},
			author:     a,
			wantFailed: []string{"b", "root"},
			wantErr:    "model failure of b",
		},
		{
			name:       "error of an activation that runs under the author",
			running:    []identity{root, a, worker},
			failing:    []identity{worker},
			author:     a,
			wantFailed: []string{"worker", "a", "root"},
			wantErr:    "model failure of worker",
		},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			tm := newTraceManager()
			invocationID := fmt.Sprintf("inv-final-event-errors-%d", i)
			in := func(id identity) identity {
				id.invocationID = invocationID
				return id
			}
			for _, id := range tc.running {
				tm.pushAgentSpan(ctx, in(id))
			}
			for _, id := range tc.failing {
				tm.recordError(ctx, in(id), llmKey(in(id)), fmt.Errorf("model failure of %s", id.agentName))
			}

			tm.completeResponse(ctx, in(tc.author), tc.author.agentName, "", nil)

			unclosed, termErr := tm.popUnclosedAgentSpans(ctx, in(root))
			var gotFailed []string
			for _, u := range unclosed {
				gotFailed = append(gotFailed, u.agentName)
			}
			if diff := cmp.Diff(tc.wantFailed, gotFailed); diff != "" {
				t.Errorf("popUnclosedAgentSpans activations mismatch (-want +got):\n%s", diff)
			}
			var gotErr string
			if termErr != nil {
				gotErr = termErr.Error()
			}
			if gotErr != tc.wantErr {
				t.Errorf("popUnclosedAgentSpans error = %q, want %q", gotErr, tc.wantErr)
			}
		})
	}
}

// deltaContext is a context that derives an invocation context with
// WithICDelta, as the callback contexts of ADK do.
type deltaContext struct {
	context.Context
	derive func() agent.InvocationContext
}

func (c deltaContext) WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext {
	return c.derive()
}

// runningInvocation is an invocation context that knows only the agent that
// runs.
type runningInvocation struct {
	agent.InvocationContext
	running agent.Agent
}

func (c runningInvocation) Agent() agent.Agent { return c.running }

// contextRunning returns a callback context of the running agent a.
func contextRunning(a agent.Agent) context.Context {
	return deltaContext{
		Context: context.Background(),
		derive:  func() agent.InvocationContext { return runningInvocation{running: a} },
	}
}

// brokenAgent is an agent whose sub-agents cannot be read.
type brokenAgent struct{ agent.Agent }

func (brokenAgent) Name() string { return "broken" }

func (brokenAgent) SubAgents() []agent.Agent { panic("SubAgents failed") }

// newTestAgent returns a custom agent that runs nothing and has subAgents.
func newTestAgent(t *testing.T, name string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: subAgents,
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error: %v", name, err)
	}
	return a
}

// TestRunningAgentChildren reads the sub-agents of the agent that a callback
// context runs. Anything but the agent that was asked for has no children.
func TestRunningAgentChildren(t *testing.T) {
	a, b := newTestAgent(t, "a"), newTestAgent(t, "b")
	sequence := newTestAgent(t, "sequence", a, b)
	helper, err := llmagent.New(llmagent.Config{Name: "helper"})
	if err != nil {
		t.Fatalf("llmagent.New(helper) error: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{Name: "coordinator", SubAgents: []agent.Agent{helper}})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator) error: %v", err)
	}
	derives := func(derive func() agent.InvocationContext) context.Context {
		return deltaContext{Context: context.Background(), derive: derive}
	}
	tests := []struct {
		name      string
		ctx       context.Context
		agentName string
		want      agentChildren
	}{
		{
			name:      "workflow agent",
			ctx:       contextRunning(sequence),
			agentName: "sequence",
			want:      agentChildren{parent: "sequence", nonLLM: true, names: []string{"a", "b"}},
		},
		{
			name:      "LLM agent",
			ctx:       contextRunning(coordinator),
			agentName: "coordinator",
			want:      agentChildren{parent: "coordinator", names: []string{"helper"}},
		},
		{
			name:      "agent without sub-agents",
			ctx:       contextRunning(a),
			agentName: "a",
			want:      agentChildren{parent: "a", nonLLM: true},
		},
		{name: "another agent runs", ctx: contextRunning(sequence), agentName: "other"},
		{name: "no agent runs", ctx: contextRunning(nil), agentName: "sequence"},
		{name: "no name", ctx: contextRunning(sequence)},
		{name: "sub-agents cannot be read", ctx: contextRunning(brokenAgent{}), agentName: "broken"},
		{
			name:      "context derives nothing",
			ctx:       derives(func() agent.InvocationContext { return nil }),
			agentName: "sequence",
		},
		{
			name:      "derivation panics",
			ctx:       derives(func() agent.InvocationContext { panic("WithICDelta failed") }),
			agentName: "sequence",
		},
		{name: "context without invocation context", ctx: context.Background(), agentName: "sequence"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := runningAgentChildren(tc.ctx, tc.agentName)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(agentChildren{})); diff != "" {
				t.Errorf("runningAgentChildren mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestInvocationTrace_RecordChildrenReplacesTheParent starts a workflow agent
// and an LLM agent that both have a sub-agent named "worker" on separate node
// paths. Recording configured children per activation on spanRecord keeps the
// workflow agent's sub-agents apart from open siblings while letting the LLM
// agent's sub-agents transfer to peers, without either activation overwriting
// the other.
func TestInvocationTrace_RecordChildrenReplacesTheParent(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invocationID = "inv-per-activation-children"

	workerSeq, otherSeq := newTestAgent(t, "worker"), newTestAgent(t, "other")
	sequence := newTestAgent(t, "sequence", workerSeq, otherSeq)

	workerLLM, err := llmagent.New(llmagent.Config{Name: "worker"})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	peerLLM, err := llmagent.New(llmagent.Config{Name: "peer"})
	if err != nil {
		t.Fatalf("llmagent.New(peer): %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		SubAgents: []agent.Agent{peerLLM, workerLLM},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	seqSpan := tm.pushAgentSpan(contextRunning(sequence), identity{invocationID: invocationID, agentName: "sequence", path: "root/sequence@1"})
	coordSpan := tm.pushAgentSpan(contextRunning(coordinator), identity{invocationID: invocationID, agentName: "coordinator", path: "root/coordinator@2"})

	// Under coordinator (an LLM agent), peer starts and stays open when worker
	// starts; worker is parented to peer (peer transfer).
	peerSpan := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "peer", path: "root/coordinator@2"})
	if peerSpan.parentSpanID != coordSpan.spanID {
		t.Errorf("peer parent = %q, want coordinator %q", peerSpan.parentSpanID, coordSpan.spanID)
	}
	coordWorkerSpan := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "worker", path: "root/coordinator@2"})
	if coordWorkerSpan.parentSpanID != peerSpan.spanID {
		t.Errorf("coordinator worker parent = %q, want peer %q", coordWorkerSpan.parentSpanID, peerSpan.spanID)
	}

	// Under sequence (a non-LLM workflow agent), worker starts and stays open
	// when other starts; other skips worker and is parented to sequence.
	seqWorkerSpan := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "worker", path: "root/sequence@1"})
	if seqWorkerSpan.parentSpanID != seqSpan.spanID {
		t.Errorf("sequence worker parent = %q, want sequence %q", seqWorkerSpan.parentSpanID, seqSpan.spanID)
	}
	seqOtherSpan := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "other", path: "root/sequence@1"})
	if seqOtherSpan.parentSpanID != seqSpan.spanID {
		t.Errorf("sequence other parent = %q, want sequence %q rather than sibling %q", seqOtherSpan.parentSpanID, seqSpan.spanID, seqWorkerSpan.spanID)
	}
}

// TestTraceManager_SubAgentsOfRunningAgentStayApartOutsideTheRunnerTree starts
// a workflow agent whose sub-agents are not in the agent tree that the runner
// runs, as when the agent runs as a graph node. Starting the agent records its
// sub-agents, so the second one is parented to the workflow agent rather than
// to the first, whose span is still open because a callback answered for it.
func TestTraceManager_SubAgentsOfRunningAgentStayApartOutsideTheRunnerTree(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invocationID = "inv-running-agent-children"
	id := func(name string) identity {
		return identity{invocationID: invocationID, agentName: name, path: "graph"}
	}
	shortcut, next := newTestAgent(t, "shortcut"), newTestAgent(t, "next")
	sequence := newTestAgent(t, "sequence", shortcut, next)

	sequenceSpan := tm.pushAgentSpan(contextRunning(sequence), id("sequence"))
	shortcutSpan := tm.pushAgentSpan(ctx, id("shortcut"))
	nextSpan := tm.pushAgentSpan(ctx, id("next"))

	if shortcutSpan.parentSpanID != sequenceSpan.spanID {
		t.Errorf("shortcut: parent span %q, want the sequence %q", shortcutSpan.parentSpanID, sequenceSpan.spanID)
	}
	if nextSpan.parentSpanID != sequenceSpan.spanID {
		t.Errorf("next: parent span %q, want the sequence %q rather than its sibling %q", nextSpan.parentSpanID, sequenceSpan.spanID, shortcutSpan.spanID)
	}
}

// TestFunctionResponseCalls lists the function call ID, or else the tool name,
// of each function response of a piece of content.
func TestFunctionResponseCalls(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    []string
	}{
		{name: "no content"},
		{name: "text only", content: textContent("model", "hi")},
		{
			name: "responses among other parts",
			content: &genai.Content{Parts: []*genai.Part{
				nil,
				{Text: "hi"},
				{FunctionCall: &genai.FunctionCall{ID: "call-0", Name: "lookup"}},
				{FunctionResponse: &genai.FunctionResponse{}},
				{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "lookup"}},
				{FunctionResponse: &genai.FunctionResponse{Name: "search"}},
			}},
			want: []string{"call-1", "search"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, functionResponseCalls(tc.content)); diff != "" {
				t.Errorf("functionResponseCalls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestFunctionResponseCallsMatchToolKeys pairs the function response of an
// event with the tool call that it answers: what functionResponseCalls returns
// is what toolKeyCall finds at the end of the key that the tool callbacks
// build for the call.
func TestFunctionResponseCallsMatchToolKeys(t *testing.T) {
	id := identity{agentName: "bot", branch: "root.bot", path: "root/bot"}
	tests := []struct {
		name string
		resp *genai.FunctionResponse
		want string
	}{
		{name: "function call ID", resp: &genai.FunctionResponse{ID: "call-1", Name: "lookup"}, want: "call-1"},
		{name: "tool name without a function call ID", resp: &genai.FunctionResponse{Name: "lookup"}, want: "lookup"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			content := &genai.Content{Parts: []*genai.Part{{FunctionResponse: tc.resp}}}
			got := functionResponseCalls(content)
			if diff := cmp.Diff([]string{tc.want}, got); diff != "" {
				t.Fatalf("functionResponseCalls mismatch (-want +got):\n%s", diff)
			}
			if key := toolKeyCall(toolKey(id, tc.resp.ID, tc.resp.Name)); key != got[0] {
				t.Errorf("toolKeyCall(toolKey(...)) = %q, want %q: a response must name the key of its call", key, got[0])
			}
		})
	}
}

// TestTraceManager_EarlierRunnerAncestryStaysOutOfTheInvocation shares one
// traceManager between a runner of the tree root -> worker -> helper and a later
// runner of an agent of that tree, as a plugin that several runners share does.
// The parent map that a runner attaches to the callback context describes the
// tree of that runner, so it is authoritative even when it is empty: nothing
// that the earlier runner's tree says about the ancestors of the agent is
// recorded for the invocation. Without a parent map, the earlier tree is all
// that is known, and it gives the invocation its root agent.
func TestTraceManager_EarlierRunnerAncestryStaysOutOfTheInvocation(t *testing.T) {
	helper := newTestAgent(t, "helper")
	worker := newTestAgent(t, "worker", helper)
	root := newTestAgent(t, "root", worker)

	type parentsKey struct{}
	tests := []struct {
		name    string
		runner  agent.Agent
		parents map[string]agent.Agent // nil: the context carries no parent map
		// wantRoot is the root agent of the invocation, and wantParents the
		// parent of each agent that the invocation records.
		wantRoot    string
		wantParents map[string]string
	}{
		{
			name:        "parent map of the runner",
			runner:      worker,
			parents:     map[string]agent.Agent{"helper": worker},
			wantRoot:    "worker",
			wantParents: map[string]string{"helper": "worker"},
		},
		{
			name:     "empty parent map",
			runner:   helper,
			parents:  map[string]agent.Agent{},
			wantRoot: "helper",
		},
		{
			name:        "no parent map",
			runner:      worker,
			wantRoot:    "root",
			wantParents: map[string]string{"worker": "root", "helper": "worker"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			// The earlier runner runs root.
			tm.rememberAgentTree(root)
			ctx := context.Background()
			if tc.parents != nil {
				ctx = context.WithValue(ctx, parentsKey{}, tc.parents)
			}
			id := identity{invocationID: "inv-earlier-runner", agentName: tc.runner.Name()}
			if got := tm.invocationSpan(ctx, id, tc.runner).rootAgentName; got != tc.wantRoot {
				t.Errorf("root agent = %q, want %q", got, tc.wantRoot)
			}
			inv := tm.invocation(ctx, id)
			inv.mu.Lock()
			defer inv.mu.Unlock()
			if !maps.Equal(inv.parentAgent, tc.wantParents) {
				t.Errorf("recorded parents = %v, want %v", inv.parentAgent, tc.wantParents)
			}
		})
	}
}

// streamingPartialThenBlockLLM yields one partial streamed chunk with text and
// then yields a final response if the caller continues reading.
type streamingPartialThenBlockLLM struct {
	chunkText string
	finalText string
}

func (m *streamingPartialThenBlockLLM) Name() string { return "streaming-partial-llm" }

func (m *streamingPartialThenBlockLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(&model.LLMResponse{
			Content: textContent("model", m.chunkText),
			Partial: true,
		}, nil) {
			return
		}
		yield(&model.LLMResponse{
			Content:      textContent("model", m.finalText),
			TurnComplete: true,
		}, nil)
	}
}

// blockingUntilContextDoneLLM signals entered when GenerateContent begins and
// blocks until ctx is done before returning.
type blockingUntilContextDoneLLM struct {
	entered chan struct{}
	once    sync.Once
}

func (m *blockingUntilContextDoneLLM) Name() string { return "blocking-until-context-done-llm" }

func (m *blockingUntilContextDoneLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.once.Do(func() { close(m.entered) })
		<-ctx.Done()
	}
}

// waitThenStreamPartialLLM waits for ready to close before yielding a partial
// streamed chunk, and then yields a final response if the caller keeps reading.
type waitThenStreamPartialLLM struct {
	ready     <-chan struct{}
	chunkText string
	finalText string
}

func (m *waitThenStreamPartialLLM) Name() string { return "wait-then-stream-partial-llm" }

func (m *waitThenStreamPartialLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		select {
		case <-m.ready:
		case <-ctx.Done():
			return
		}
		if !yield(&model.LLMResponse{
			Content: textContent("model", m.chunkText),
			Partial: true,
		}, nil) {
			return
		}
		yield(&model.LLMResponse{
			Content:      textContent("model", m.finalText),
			TurnComplete: true,
		}, nil)
	}
}

// TestRunner_StopReadingMidStream_IsNotAnError verifies that when the caller
// runs with StreamingModeSSE and breaks out of the runner.Run iterator after
// receiving a partial streamed chunk, the invocation completes with
// INVOCATION_COMPLETED rather than INVOCATION_ERROR + AGENT_ERROR across
// single-agent, sequential, loop, and parallel roots (including when a parallel
// sibling has an in-flight model call that has not yet delivered a token).
func TestRunner_StopReadingMidStream_IsNotAnError(t *testing.T) {
	newStreamingBot := func(t *testing.T, name string) agent.Agent {
		t.Helper()
		bot, err := llmagent.New(llmagent.Config{
			Name:  name,
			Model: &streamingPartialThenBlockLLM{chunkText: "hello ", finalText: "hello world"},
		})
		if err != nil {
			t.Fatalf("llmagent.New(%s): %v", name, err)
		}
		return bot
	}

	t.Run("single streaming agent", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		r := newTestRunner(t, newStreamingBot(t, "bot"), p)

		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event in StreamingModeSSE")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
	})

	t.Run("sequential root", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		seq, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      "seq_root",
				SubAgents: []agent.Agent{newStreamingBot(t, "bot"), newStreamingBot(t, "step2")},
			},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New: %v", err)
		}
		r := newTestRunner(t, seq, p)

		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event in StreamingModeSSE")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
	})

	t.Run("loop root", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		loop, err := loopagent.New(loopagent.Config{
			MaxIterations: 2,
			AgentConfig: agent.Config{
				Name:      "loop_root",
				SubAgents: []agent.Agent{newStreamingBot(t, "bot")},
			},
		})
		if err != nil {
			t.Fatalf("loopagent.New: %v", err)
		}
		r := newTestRunner(t, loop, p)

		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event in StreamingModeSSE")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
	})

	t.Run("parallel root with sibling model call in flight", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		slowEntered := make(chan struct{})
		slow, err := llmagent.New(llmagent.Config{
			Name:  "slow",
			Model: &blockingUntilContextDoneLLM{entered: slowEntered},
		})
		if err != nil {
			t.Fatalf("llmagent.New(slow): %v", err)
		}
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &waitThenStreamPartialLLM{
				ready:     slowEntered,
				chunkText: "hello ",
				finalText: "hello world",
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		par, err := parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{
				Name:      "par_root",
				SubAgents: []agent.Agent{bot, slow},
			},
		})
		if err != nil {
			t.Fatalf("parallelagent.New: %v", err)
		}
		r := newTestRunner(t, par, p)

		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event in StreamingModeSSE")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
	})
}

// flakyOnFirstPassLLM fails on call 1 and returns a valid final response on
// call 2+.
type flakyOnFirstPassLLM struct {
	mu    sync.Mutex
	calls int
	err   error
	reply string
}

func (m *flakyOnFirstPassLLM) Name() string { return "flaky-on-first-pass-llm" }

func (m *flakyOnFirstPassLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.calls++
		call := m.calls
		m.mu.Unlock()
		if call == 1 {
			yield(nil, m.err)
			return
		}
		yield(&model.LLMResponse{
			Content:      textContent("model", m.reply),
			TurnComplete: true,
		}, nil)
	}
}

// TestRunner_SubAgentModelError_CallerKeepsReading_EmitsInvocationError
// verifies that when a sub-agent's model call fails and the caller continues
// iterating runner.Run until exhaustion (so ADK runs AfterAgentCallback on the
// ancestors or retries the child in a LoopAgent), the invocation still
// terminates with INVOCATION_ERROR.
func TestRunner_SubAgentModelError_CallerKeepsReading_EmitsInvocationError(t *testing.T) {
	t.Run("sequential sub-agent fails after sibling succeeds", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		modelErr := errors.New("subagent model failure 503")

		okChild, err := llmagent.New(llmagent.Config{
			Name:  "ok_child",
			Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "ok"), TurnComplete: true}}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(ok_child): %v", err)
		}
		failChild, err := llmagent.New(llmagent.Config{
			Name:  "fail_child",
			Model: &failingLLM{err: modelErr},
		})
		if err != nil {
			t.Fatalf("llmagent.New(fail_child): %v", err)
		}
		root, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{okChild, failChild},
			},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(root): %v", err)
		}
		r := newTestRunner(t, root, p)

		var errs []error
		for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) == 0 {
			t.Fatal("expected runner.Run to yield an error")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got INVOCATION_COMPLETED after sub-agent model failure when caller kept reading; event types: %v", eventTypes(rows))
		}
		invRow := onlyRowOfType(t, rows, EventInvocationError)
		if invRow["error_message"] != modelErr.Error() {
			t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], modelErr.Error())
		}
		gotErrors := agentErrorCounts(rows)
		if gotErrors["ok_child"] != 0 {
			t.Errorf("ok_child got %d AGENT_ERROR rows, want 0", gotErrors["ok_child"])
		}
	})

	t.Run("loop agent child fails on pass 1 and succeeds on pass 2", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		modelErr := errors.New("pass 1 model failure 503")

		flakyChild, err := llmagent.New(llmagent.Config{
			Name:  "flaky_child",
			Model: &flakyOnFirstPassLLM{err: modelErr, reply: "recovered on pass 2"},
		})
		if err != nil {
			t.Fatalf("llmagent.New(flaky_child): %v", err)
		}
		loop, err := loopagent.New(loopagent.Config{
			MaxIterations: 2,
			AgentConfig: agent.Config{
				Name:      "loop_root",
				SubAgents: []agent.Agent{flakyChild},
			},
		})
		if err != nil {
			t.Fatalf("loopagent.New(loop_root): %v", err)
		}
		r := newTestRunner(t, loop, p)

		var errs []error
		for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) == 0 {
			t.Fatal("expected runner.Run to yield an error on pass 1")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got INVOCATION_COMPLETED after pass 1 model failure in LoopAgent; event types: %v", eventTypes(rows))
		}
		invRow := onlyRowOfType(t, rows, EventInvocationError)
		if invRow["error_message"] != modelErr.Error() {
			t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invRow["error_message"], modelErr.Error())
		}
	})
}

// TestRunner_TaskModeDelegation_EmitsAgentCompletedAndRetiresFirstCoordinatorActivation
// verifies that when a chat-mode coordinator delegates to an llmagent configured
// with llmagent.ModeTask:
//  1. The task agent's finish_task FunctionResponse emits AGENT_COMPLETED for
//     the task agent with its span_id, parent_span_id, and latency.
//  2. When the coordinator re-enters Agent.Run on the same branch and path to
//     continue after the task agent completes, the coordinator's first
//     responded activation is retired so the second coordinator activation is
//     parented to the invocation span rather than to the finished task agent or
//     the first coordinator activation.
//  3. If the coordinator's continuation fails, neither the completed task agent
//     nor the first coordinator activation receives a spurious AGENT_ERROR row.
func TestRunner_TaskModeDelegation_EmitsAgentCompletedAndRetiresFirstCoordinatorActivation(t *testing.T) {
	newTaskWorker := func(t *testing.T) agent.Agent {
		t.Helper()
		worker, err := llmagent.New(llmagent.Config{
			Name:        "task_worker",
			Description: "executes delegated tasks",
			Mode:        llmagent.ModeTask,
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-finish-task",
										Name: "finish_task",
										Args: map[string]any{"result": "delegated task finished"},
									},
								},
							},
						},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(task_worker): %v", err)
		}
		return worker
	}

	delegateCallResponse := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call-task-worker",
						Name: "task_worker",
						Args: map[string]any{"request": "run the task"},
					},
				},
			},
		},
	}

	t.Run("happy path", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		taskWorker := newTaskWorker(t)
		coordinator, err := llmagent.New(llmagent.Config{
			Name:      "coordinator",
			SubAgents: []agent.Agent{taskWorker},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					delegateCallResponse,
					{Content: textContent("model", "all done"), TurnComplete: true},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(coordinator): %v", err)
		}
		r := newTestRunner(t, coordinator, p)

		for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "start"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		invSpan := placeOf(onlyRowOfType(t, rows, EventInvocationStarting)).span

		var coordStarts, workerStarts, coordCompletes, workerCompletes []map[string]any
		for _, row := range rowsOfType(rows, EventAgentStarting) {
			switch row["agent"] {
			case "coordinator":
				coordStarts = append(coordStarts, row)
			case "task_worker":
				workerStarts = append(workerStarts, row)
			}
		}
		for _, row := range rowsOfType(rows, EventAgentCompleted) {
			switch row["agent"] {
			case "coordinator":
				coordCompletes = append(coordCompletes, row)
			case "task_worker":
				workerCompletes = append(workerCompletes, row)
			}
		}

		if len(coordStarts) != 2 {
			t.Fatalf("coordinator AGENT_STARTING count = %d, want 2", len(coordStarts))
		}
		for i, cs := range coordStarts {
			if got := placeOf(cs).parent; got != invSpan {
				t.Errorf("coordinator AGENT_STARTING[%d] parent_span_id = %q, want invocation span %q", i, got, invSpan)
			}
		}
		if len(workerStarts) != 1 {
			t.Fatalf("task_worker AGENT_STARTING count = %d, want 1", len(workerStarts))
		}
		if got, want := placeOf(workerStarts[0]).parent, placeOf(coordStarts[0]).span; got != want {
			t.Errorf("task_worker AGENT_STARTING parent_span_id = %q, want first coordinator span %q", got, want)
		}
		if len(workerCompletes) != 1 {
			t.Fatalf("task_worker AGENT_COMPLETED count = %d, want 1", len(workerCompletes))
		}
		if got, want := placeOf(workerCompletes[0]), placeOf(workerStarts[0]); got != want {
			t.Errorf("task_worker AGENT_COMPLETED place = %+v, want %+v", got, want)
		}
		mustLatency(t, workerCompletes[0])

		if len(coordCompletes) != 1 {
			t.Fatalf("coordinator AGENT_COMPLETED count = %d, want 1", len(coordCompletes))
		}
		if got, want := placeOf(coordCompletes[0]), placeOf(coordStarts[1]); got != want {
			t.Errorf("coordinator AGENT_COMPLETED place = %+v, want second coordinator activation %+v", got, want)
		}
		mustLatency(t, coordCompletes[0])
	})

	t.Run("continuation failure after task completion", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		taskWorker := newTaskWorker(t)
		contErr := errors.New("continuation model failure")
		coordinator, err := llmagent.New(llmagent.Config{
			Name:      "coordinator",
			SubAgents: []agent.Agent{taskWorker},
			Model: &firstCallResponseThenErrLLM{
				first: delegateCallResponse,
				err:   contErr,
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(coordinator): %v", err)
		}
		r := newTestRunner(t, coordinator, p)

		var gotErr error
		for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "start"), agent.RunConfig{}) {
			if err != nil {
				gotErr = err
			}
		}
		if !errors.Is(gotErr, contErr) {
			t.Fatalf("r.Run error = %v, want %v", gotErr, contErr)
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		onlyRowOfType(t, rows, EventInvocationError)
		if diff := cmp.Diff(map[string]int{"coordinator": 1}, agentErrorCounts(rows)); diff != "" {
			t.Errorf("AGENT_ERROR counts mismatch (-want +got):\n%s", diff)
		}
	})
}

// firstCallResponseThenErrLLM returns first on call 1 and yields err on call 2+.
type firstCallResponseThenErrLLM struct {
	mu    sync.Mutex
	calls int
	first *model.LLMResponse
	err   error
}

func (m *firstCallResponseThenErrLLM) Name() string { return "first-call-response-then-err-llm" }

func (m *firstCallResponseThenErrLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.calls++
		call := m.calls
		m.mu.Unlock()
		if call == 1 {
			yield(m.first, nil)
			return
		}
		yield(nil, m.err)
	}
}

// TestProbe1_ConcurrentGraphParentsSameChildNames verifies that when a workflow
// graph runs two sequential workflow agents (flow_a and flow_b) concurrently
// and both have children with the same names ("gate" and "step"), each child's
// AGENT_STARTING and AGENT_COMPLETED rows are parented to its own sequential
// parent (flow_a or flow_b) rather than escaping to the root or invocation
// span.
func TestProbe1_ConcurrentGraphParentsSameChildNames(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)

	gatesReady := make(chan struct{})
	var gateArrivals sync.WaitGroup
	gateArrivals.Add(2)
	go func() {
		gateArrivals.Wait()
		close(gatesReady)
	}()

	newFlow := func(t *testing.T, flowName string) agent.Agent {
		t.Helper()
		gate, err := llmagent.New(llmagent.Config{
			Name:  "gate",
			Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "unused")}}},
			BeforeAgentCallbacks: []agent.BeforeAgentCallback{
				func(ctx agent.Context) (*genai.Content, error) {
					gateArrivals.Done()
					select {
					case <-gatesReady:
					case <-ctx.Done():
						return nil, ctx.Err()
					}
					return textContent("model", flowName+" gate shortcut"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(%s/gate): %v", flowName, err)
		}
		step, err := llmagent.New(llmagent.Config{
			Name:  "step",
			Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", flowName+" step done"), TurnComplete: true}}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(%s/step): %v", flowName, err)
		}
		seq, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      flowName,
				SubAgents: []agent.Agent{gate, step},
			},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(%s): %v", flowName, err)
		}
		return seq
	}

	flowA := newFlow(t, "flow_a")
	flowB := newFlow(t, "flow_b")
	nodeA, err := workflow.NewAgentNode(flowA, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(flow_a): %v", err)
	}
	nodeB, err := workflow.NewAgentNode(flowB, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(flow_b): %v", err)
	}

	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: nodeA},
			{From: workflow.Start, To: nodeB},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(root): %v", err)
	}
	r := newTestRunner(t, root, p)

	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)

	var flowASpan, flowBSpan string
	var gateParents, stepParents []string
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		switch row["agent"] {
		case "flow_a":
			flowASpan = placeOf(row).span
		case "flow_b":
			flowBSpan = placeOf(row).span
		case "gate":
			gateParents = append(gateParents, placeOf(row).parent)
		case "step":
			stepParents = append(stepParents, placeOf(row).parent)
		}
	}
	if flowASpan == "" || flowBSpan == "" {
		t.Fatalf("missing flow spans: flow_a=%q flow_b=%q", flowASpan, flowBSpan)
	}
	wantFlowSpans := []string{flowASpan, flowBSpan}
	slices.Sort(wantFlowSpans)
	slices.Sort(gateParents)
	slices.Sort(stepParents)
	if diff := cmp.Diff(wantFlowSpans, gateParents); diff != "" {
		t.Errorf("gate AGENT_STARTING parent_span_id mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(wantFlowSpans, stepParents); diff != "" {
		t.Errorf("step AGENT_STARTING parent_span_id mismatch (-want +got):\n%s", diff)
	}
}

// TestLogEvent_EventFiltersPreserveSpansLatencyAndInvocationCleanup verifies
// that EventAllowlist and EventDenylist filter row emission in logEvent without
// breaking span parenting, latency calculation, or endInvocation cleanup when
// start events or INVOCATION_COMPLETED are filtered out.
func TestLogEvent_EventFiltersPreserveSpansLatencyAndInvocationCleanup(t *testing.T) {
	t.Run("allowlist keeps only completed rows with intact spans and latency", func(t *testing.T) {
		p, requests, ctx := setupTestPluginWithConfig(t, func(cfg *Config) {
			cfg.EventAllowlist = []string{EventLLMResponse, EventAgentCompleted, EventInvocationCompleted}
		})

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(ctx))
		errs.add(p.BeforeAgentCallback()(ctx))
		errs.add(p.BeforeModelCallback()(ctx, &model.LLMRequest{}))
		errs.add(p.AfterModelCallback()(ctx, &model.LLMResponse{Content: textContent("model", "done")}, nil))
		errs.add(p.AfterAgentCallback()(ctx))
		errs.check(t)
		p.AfterRunCallback()(ctx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		if diff := cmp.Diff([]string{EventLLMResponse, EventAgentCompleted, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
			t.Fatalf("filtered event types mismatch (-want +got):\n%s", diff)
		}
		llmResp := onlyRowOfType(t, rows, EventLLMResponse)
		agentComp := onlyRowOfType(t, rows, EventAgentCompleted)
		invComp := onlyRowOfType(t, rows, EventInvocationCompleted)

		if got, want := placeOf(llmResp).parent, placeOf(agentComp).span; got != want {
			t.Errorf("LLM_RESPONSE parent_span_id = %q, want AGENT_COMPLETED span_id %q", got, want)
		}
		if got, want := placeOf(agentComp).parent, placeOf(invComp).span; got != want {
			t.Errorf("AGENT_COMPLETED parent_span_id = %q, want INVOCATION_COMPLETED span_id %q", got, want)
		}
		mustLatency(t, llmResp)
		mustLatency(t, agentComp)
		mustLatency(t, invComp)
	})

	t.Run("denylist filtering INVOCATION_COMPLETED still cleans up invocation state", func(t *testing.T) {
		p, requests, ctx := setupTestPluginWithConfig(t, func(cfg *Config) {
			cfg.EventDenylist = []string{EventInvocationCompleted}
		})

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(ctx))
		errs.add(p.BeforeAgentCallback()(ctx))
		errs.add(p.AfterAgentCallback()(ctx))
		errs.check(t)
		p.AfterRunCallback()(ctx)

		rows := collectRows(t, requests, untilEventType(EventAgentCompleted))
		if slices.Contains(eventTypes(rows), EventInvocationCompleted) {
			t.Fatalf("denylisted EventInvocationCompleted was emitted: %v", eventTypes(rows))
		}
		firstParent := placeOf(onlyRowOfType(t, rows, EventAgentCompleted)).parent

		// Starting a new run with the same invocation ID after AfterRunCallback
		// must allocate a fresh invocation root span ID, proving endInvocation
		// removed the finished invocation even though INVOCATION_COMPLETED was
		// filtered out.
		errs.add(p.BeforeRunCallback()(ctx))
		errs.add(p.BeforeAgentCallback()(ctx))
		errs.check(t)
		nextRows := collectRows(t, requests, untilEventType(EventAgentStarting))
		secondParent := placeOf(onlyRowOfType(t, nextRows, EventAgentStarting)).parent
		if secondParent == firstParent {
			t.Errorf("subsequent callback reused invocation span_id %q; endInvocation did not clean up invocation state when INVOCATION_COMPLETED was denylisted", firstParent)
		}
	})
}

// TestTraceManager_EvictedTombstoneCapDropsOldestOnly verifies that when
// tm.evicted reaches max(tm.maxTracked, 64), markEvictedLocked drops only the
// oldest tombstone rather than clearing all tombstones.
func TestTraceManager_EvictedTombstoneCapDropsOldestOnly(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	tm.maxTracked = 1
	now := time.Unix(1000, 0)
	tm.now = func() time.Time { return now }

	// Trigger 65 evictions at distinct timestamps; the tombstone cap is max(1, 64) = 64.
	var evictedIDs []string
	for i := range 66 {
		now = now.Add(time.Millisecond)
		id := identity{invocationID: fmt.Sprintf("inv-%03d", i), agentName: "root"}
		tm.pushAgentSpan(ctx, id)
		if i < 65 {
			evictedIDs = append(evictedIDs, id.invocationID)
		}
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()
	if got := len(tm.evicted); got != 64 {
		t.Fatalf("len(tm.evicted) = %d, want 64", got)
	}
	if _, ok := tm.evicted[evictedIDs[0]]; ok {
		t.Errorf("oldest tombstone %q is still present, want it evicted", evictedIDs[0])
	}
	for _, id := range evictedIDs[1:] {
		if _, ok := tm.evicted[id]; !ok {
			t.Errorf("recent tombstone %q was lost when cap was reached", id)
		}
	}
}

// TestTraceManager_AdoptFirstAgentOTelSpanWhenCallerHasNoSpan verifies that
// when OnUserMessageCallback and BeforeRunCallback run without a caller OTel
// span on ctx, and ADK's agent.Run subsequently attaches an OpenTelemetry span
// starting at BeforeAgentCallback, the agent/model/tool rows emit the ambient
// OTel trace_id and record it as adoptedTraceID so subsequent spanless
// callbacks (OnEventCallback, AfterRunCallback) and capacity eviction
// (markEvictedLocked) also use the adopted trace_id.
func TestTraceManager_AdoptFirstAgentOTelSpanWhenCallerHasNoSpan(t *testing.T) {
	const (
		traceHex = "4bf92f3577b34da6a3ce929d0e0e4736"
		spanHex  = "00f067aa0ba902b7"
	)
	traceID, _ := trace.TraceIDFromHex(traceHex)
	spanID, _ := trace.SpanIDFromHex(spanHex)
	sc := trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID})

	t.Run("adopts first ambient span on agent, model, and tool rows and propagates to spanless OnEvent and AfterRun", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		agentCtx := *mCtx
		agentCtx.ctx = trace.ContextWithSpanContext(mCtx.ctx, sc)
		toolCtx := withFunctionCall(&agentCtx, "call-1")
		lookupTool := &fakeTool{name: "lookup"}

		var errs callbackErrors
		// Caller starts with no OTel span on mCtx.
		errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello")))
		errs.add(p.BeforeRunCallback()(mCtx))
		// ADK's agent.Run opens an OTel span on agentCtx.
		errs.add(p.BeforeAgentCallback()(&agentCtx))
		errs.add(p.BeforeModelCallback()(&agentCtx, &model.LLMRequest{Model: "gemini-test"}))
		errs.add(p.AfterModelCallback()(&agentCtx, &model.LLMResponse{Content: textContent("model", "calling lookup")}, nil))
		errs.add(p.BeforeToolCallback()(toolCtx, lookupTool, nil))
		errs.add(p.AfterToolCallback()(toolCtx, lookupTool, nil, map[string]any{"answer": "sunny"}, nil))
		errs.add(p.AfterAgentCallback()(&agentCtx))
		// OnEventCallback and AfterRunCallback run on outer mCtx (no ambient span).
		ev := session.NewEvent(t.Context(), mCtx.invocationID)
		ev.Author = mCtx.agentName
		ev.Actions.StateDelta = map[string]any{"k": "v"}
		errs.add(p.OnEventCallback()(mCtx, ev))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		userPlace := placeOf(onlyRowOfType(t, rows, EventUserMessageReceived))
		invStartPlace := placeOf(onlyRowOfType(t, rows, EventInvocationStarting))
		if userPlace.trace == "" || userPlace.trace == traceHex {
			t.Errorf("USER_MESSAGE_RECEIVED trace_id = %q, want initial generated random trace ID (!= %q)", userPlace.trace, traceHex)
		}
		if invStartPlace.trace != userPlace.trace {
			t.Errorf("INVOCATION_STARTING trace_id = %q, want %q", invStartPlace.trace, userPlace.trace)
		}

		for _, evType := range []string{
			EventAgentStarting,
			EventLLMRequest,
			EventLLMResponse,
			EventToolStarting,
			EventToolCompleted,
			EventAgentCompleted,
			EventStateDelta,
			EventInvocationCompleted,
		} {
			row := onlyRowOfType(t, rows, evType)
			if got := placeOf(row).trace; got != traceHex {
				t.Errorf("%s trace_id = %q, want adopted ambient trace_id %q", evType, got, traceHex)
			}
		}
	})

	t.Run("capacity eviction preserves adopted trace_id for post-eviction AfterRun", func(t *testing.T) {
		tm := newTraceManager()
		tm.maxTracked = 1
		id1 := identity{invocationID: "inv-adopted-evict-1", agentName: "root"}
		id2 := identity{invocationID: "inv-adopted-evict-2", agentName: "root"}

		// Start id1 without an ambient span.
		startSpan := tm.invocationSpan(context.Background(), id1, nil)
		if startSpan.traceID == traceHex {
			t.Fatalf("initial traceID = %q, want random trace ID", startSpan.traceID)
		}
		// BeforeAgentCallback runs with ambient span, adopting traceHex.
		otelCtx := trace.ContextWithSpanContext(context.Background(), sc)
		agentSpan := tm.pushAgentSpan(otelCtx, id1)
		if agentSpan.traceID != traceHex {
			t.Fatalf("pushAgentSpan traceID = %q, want %q", agentSpan.traceID, traceHex)
		}
		tm.popAgentSpan(otelCtx, id1)

		// Starting id2 evicts id1 into tm.evicted via markEvictedLocked.
		tm.invocationSpan(context.Background(), id2, nil)

		// Ending id1 without an ambient span restores the evicted tombstone,
		// which must carry the adopted traceHex via inv.currentTraceID().
		endSpan := tm.endInvocation(context.Background(), id1)
		if endSpan.traceID != traceHex {
			t.Errorf("post-eviction endInvocation traceID = %q, want adopted trace_id %q", endSpan.traceID, traceHex)
		}
	})
}

// transferModel answers its calls in order: transfer to `to` for each name in
// script, then text.
type transferModel struct {
	calls  atomic.Int32
	script []string
}

func (m *transferModel) Name() string { return "transfer-model" }

func (m *transferModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	i := int(m.calls.Add(1)) - 1
	part := genai.NewPartFromText("done")
	if i < len(m.script) {
		part = genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": m.script[i]})
	}
	resp := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}}, TurnComplete: true}
	return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
}

func assertAgentSpansClosed(t *testing.T, rows []map[string]any) {
	t.Helper()
	starts := map[any]map[string]any{}
	ends := map[any]bool{}
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		id := row["span_id"]
		if _, exists := starts[id]; exists {
			t.Error("agent span started twice")
		}
		starts[id] = row
	}
	if len(starts) == 0 {
		t.Fatal("no agent spans started")
	}
	for _, row := range rowsOfType(rows, EventAgentCompleted) {
		id := row["span_id"]
		opened, ok := starts[id]
		if !ok {
			t.Errorf("AGENT_COMPLETED of %v closes no started span", row["agent"])
			continue
		}
		if ends[id] {
			t.Error("agent span completed twice")
		}
		ends[id] = true
		if row["agent"] != opened["agent"] || row["parent_span_id"] != opened["parent_span_id"] {
			t.Error("agent completion changed its owner or parent")
		}
		if row["latency_ms"] == nil {
			t.Errorf("AGENT_COMPLETED of %v has no latency_ms", row["agent"])
		}
	}
	if len(ends) != len(starts) {
		t.Errorf("closed %d of %d agent spans", len(ends), len(starts))
	}
}

func TestRunner_TransferBackToRunningParent_ClosesEverySpan(t *testing.T) {
	for _, tc := range []struct {
		name   string
		first  []string
		second []string
	}{
		{name: "parent", first: []string{"planner"}},
		{name: "peer", first: []string{"second"}, second: []string{"first"}},
		{name: "double bounce", first: []string{"second", "second"}, second: []string{"first", "first"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			first, err := llmagent.New(llmagent.Config{Name: "first", Model: &transferModel{script: tc.first}})
			if err != nil {
				t.Fatal(err)
			}
			second, err := llmagent.New(llmagent.Config{Name: "second", Model: &transferModel{script: tc.second}})
			if err != nil {
				t.Fatal(err)
			}
			planner, err := llmagent.New(llmagent.Config{
				Name: "planner", Model: &transferModel{script: []string{"first"}}, SubAgents: []agent.Agent{first, second},
			})
			if err != nil {
				t.Fatal(err)
			}
			for _, err := range newTestRunner(t, planner, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			assertAgentSpansClosed(t, rows)
		})
	}
}

func TestRunner_ResumedTaskThenTransferBack_ClosesEverySpan(t *testing.T) {
	call := func(name string, args map[string]any) *model.LLMResponse {
		return &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-" + name, Name: name, Args: args}}}}}
	}
	text := func(s string) *model.LLMResponse {
		return &model.LLMResponse{Content: textContent("model", s), TurnComplete: true}
	}
	newAgent := func(c llmagent.Config) agent.Agent {
		a, err := llmagent.New(c)
		if err != nil {
			t.Fatalf("llmagent.New(%s): %v", c.Name, err)
		}
		return a
	}

	for _, tc := range []struct {
		name      string
		setupRoot func() (agent.Agent, []*scriptedLLM)
	}{
		{
			name: "direct parent",
			setupRoot: func() (agent.Agent, []*scriptedLLM) {
				taskModel := &scriptedLLM{responses: []*model.LLMResponse{text("Which color?"), call("finish_task", map[string]any{"result": "blue"})}}
				helperTask := newAgent(llmagent.Config{
					Name: "helper_task", Description: "asks, then finishes", Mode: llmagent.ModeTask, Model: taskModel,
				})
				helperModel := &scriptedLLM{responses: []*model.LLMResponse{
					call("helper_task", map[string]any{"request": "pick a color"}),
					call("transfer_to_agent", map[string]any{"agent_name": "coord"}),
				}}
				helper := newAgent(llmagent.Config{
					Name: "helper", Description: "delegates", SubAgents: []agent.Agent{helperTask}, Model: helperModel,
				})
				// Having a task child makes the runner restart at coord on turn 2.
				coordTask := newAgent(llmagent.Config{Name: "coord_task", Description: "unused", Mode: llmagent.ModeTask, Model: &scriptedLLM{}})
				coordModel := &scriptedLLM{responses: []*model.LLMResponse{
					call("transfer_to_agent", map[string]any{"agent_name": "helper"}), text("waiting for the user"),
					call("transfer_to_agent", map[string]any{"agent_name": "helper"}), text("all done"),
				}}
				coord := newAgent(llmagent.Config{
					Name: "coord", SubAgents: []agent.Agent{coordTask, helper}, Model: coordModel,
				})
				return coord, []*scriptedLLM{taskModel, helperModel, coordModel}
			},
		},
		{
			name: "intermediate chat agent",
			setupRoot: func() (agent.Agent, []*scriptedLLM) {
				taskModel := &scriptedLLM{responses: []*model.LLMResponse{text("Which color?"), call("finish_task", map[string]any{"result": "blue"})}}
				helperTask := newAgent(llmagent.Config{
					Name: "helper_task", Description: "asks, then finishes", Mode: llmagent.ModeTask, Model: taskModel,
				})
				helperModel := &scriptedLLM{responses: []*model.LLMResponse{
					call("helper_task", map[string]any{"request": "pick a color"}),
					call("transfer_to_agent", map[string]any{"agent_name": "mid"}),
				}}
				helper := newAgent(llmagent.Config{
					Name: "helper", Description: "delegates", SubAgents: []agent.Agent{helperTask}, Model: helperModel,
				})
				midModel := &scriptedLLM{responses: []*model.LLMResponse{
					call("transfer_to_agent", map[string]any{"agent_name": "helper"}), text("mid waiting"),
					call("transfer_to_agent", map[string]any{"agent_name": "helper"}), text("mid done"),
				}}
				mid := newAgent(llmagent.Config{
					Name: "mid", Description: "intermediate", SubAgents: []agent.Agent{helper}, Model: midModel,
				})
				coordTask := newAgent(llmagent.Config{Name: "coord_task", Description: "unused", Mode: llmagent.ModeTask, Model: &scriptedLLM{}})
				coordModel := &scriptedLLM{responses: []*model.LLMResponse{
					call("transfer_to_agent", map[string]any{"agent_name": "mid"}),
					call("transfer_to_agent", map[string]any{"agent_name": "mid"}),
				}}
				coord := newAgent(llmagent.Config{
					Name: "coord", SubAgents: []agent.Agent{coordTask, mid}, Model: coordModel,
				})
				return coord, []*scriptedLLM{taskModel, helperModel, midModel, coordModel}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			root, models := tc.setupRoot()
			r := newTestRunner(t, root, p)
			var resumedInvocation string
			for _, msg := range []string{"start", "blue"} {
				for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", msg), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("r.Run: %v", err)
					}
					if ev != nil {
						resumedInvocation = ev.InvocationID
					}
				}
			}
			for i, m := range models {
				if len(m.responses) != 0 {
					t.Fatalf("model %d has %d unconsumed model responses", i, len(m.responses))
				}
			}
			if resumedInvocation == "" {
				t.Fatal("resumed run returned no invocation ID")
			}
			rows := collectRows(t, requests, func(rows []map[string]any) bool {
				return len(rowsOfType(rows, EventInvocationCompleted))+len(rowsOfType(rows, EventInvocationError)) >= 2
			})
			var resumedRows []map[string]any
			for _, row := range rows {
				if row["invocation_id"] == resumedInvocation {
					resumedRows = append(resumedRows, row)
				}
			}
			if len(rowsOfType(resumedRows, EventInvocationCompleted)) != 1 || len(rowsOfType(resumedRows, EventInvocationError)) != 0 {
				t.Fatal("resumed invocation did not complete successfully")
			}
			assertAgentSpansClosed(t, resumedRows)
		})
	}
}

// TestTraceManager_PopTaskAgentSpan_DoesNotMarkNonLLMParent verifies that
// popTaskAgentSpan does NOT mark a non-LLM workflow parent (sequentialagent) as
// taskDelegated when a child ModeTask agent completes via finish_task.
func TestTraceManager_PopTaskAgentSpan_DoesNotMarkNonLLMParent(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	taskModel := &scriptedLLM{responses: []*model.LLMResponse{
		{Content: textContent("model", "done"), TurnComplete: true},
	}}
	stepTask, err := llmagent.New(llmagent.Config{
		Name:        "step_task",
		Description: "executes a step",
		Mode:        llmagent.ModeTask,
		Model:       taskModel,
	})
	if err != nil {
		t.Fatalf("llmagent.New(step_task): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq",
			SubAgents: []agent.Agent{stepTask},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(seq): %v", err)
	}

	seqID := identity{invocationID: "inv-nonllm", agentName: "seq", path: "seq@0"}
	taskID := identity{invocationID: "inv-nonllm", agentName: "step_task", path: "seq@0/step_task@1"}

	tm.invocationSpan(contextRunning(seq), seqID, seq)
	seqStart := tm.pushAgentSpan(contextRunning(seq), seqID)

	// Verify seq was recorded as non-LLM with step_task as a child.
	tm.mu.Lock()
	inv := tm.invocations["inv-nonllm"]
	tm.mu.Unlock()
	inv.mu.Lock()
	var seqRec *spanRecord
	for i := range inv.agentStack {
		if inv.agentStack[i].name == "seq" {
			seqRec = inv.agentStack[i]
		}
	}
	inv.mu.Unlock()
	if seqRec == nil {
		t.Fatalf("seq span not found on the stack")
	}
	if !seqRec.nonLLM {
		t.Fatalf("seq.nonLLM = false, want true (sequentialagent is non-LLM)")
	}
	if !slices.Contains(seqRec.children, "step_task") {
		t.Fatalf("seq.children = %v, want to contain \"step_task\"", seqRec.children)
	}

	// Push the task span (parented to seq because seq has step_task in children).
	taskStart := tm.pushAgentSpan(contextRunning(stepTask), taskID)
	if taskStart.parentSpanID != seqStart.spanID {
		t.Fatalf("task parentSpanID = %q, want seq spanID %q", taskStart.parentSpanID, seqStart.spanID)
	}

	// Pop the task span as popTaskAgentSpan does (finish_task completion).
	tm.popTaskAgentSpan(ctx, taskID)

	// Verify seq was NOT marked taskDelegated (defensive guard).
	inv.mu.Lock()
	for i := range inv.agentStack {
		if inv.agentStack[i].name == "seq" && inv.agentStack[i].taskDelegated {
			inv.mu.Unlock()
			t.Errorf("seq was marked taskDelegated, but a non-LLM workflow agent must never be marked")
			return
		}
	}
	inv.mu.Unlock()
}

// TestTraceManager_PopTaskAgentSpan_MarksDirectLLMParent verifies that
// popTaskAgentSpan still correctly marks a direct LLM parent (nonLLM ==
// false) whose configured SubAgents include the task sub-agent. This is the
// positive control for the !nonLLM guard.
func TestTraceManager_PopTaskAgentSpan_MarksDirectLLMParent(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	taskWorker, err := llmagent.New(llmagent.Config{
		Name:        "task_worker",
		Description: "executes tasks",
		Mode:        llmagent.ModeTask,
	})
	if err != nil {
		t.Fatalf("llmagent.New(task_worker): %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Coordinates",
		SubAgents:   []agent.Agent{taskWorker},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	coordID := identity{invocationID: "inv-llm-parent", agentName: "coordinator", path: "coordinator@0"}
	taskID := identity{invocationID: "inv-llm-parent", agentName: "task_worker", path: "coordinator@0/task_worker@1"}

	tm.invocationSpan(contextRunning(coordinator), coordID, coordinator)
	coordStart := tm.pushAgentSpan(contextRunning(coordinator), coordID)

	// Verify coordinator is an LLM agent with task_worker as a child.
	tm.mu.Lock()
	inv := tm.invocations["inv-llm-parent"]
	tm.mu.Unlock()
	inv.mu.Lock()
	var coordRec *spanRecord
	for i := range inv.agentStack {
		if inv.agentStack[i].name == "coordinator" {
			coordRec = inv.agentStack[i]
		}
	}
	inv.mu.Unlock()
	if coordRec == nil {
		t.Fatalf("coordinator span not found on the stack")
	}
	if coordRec.nonLLM {
		t.Fatalf("coordinator.nonLLM = true, want false (coordinator is an LLM agent)")
	}
	if !slices.Contains(coordRec.children, "task_worker") {
		t.Fatalf("coordinator.children = %v, want to contain \"task_worker\"", coordRec.children)
	}

	taskStart := tm.pushAgentSpan(contextRunning(taskWorker), taskID)
	if taskStart.parentSpanID != coordStart.spanID {
		t.Fatalf("task parentSpanID = %q, want coordinator spanID %q", taskStart.parentSpanID, coordStart.spanID)
	}

	tm.popTaskAgentSpan(ctx, taskID)

	// Verify coordinator WAS marked taskDelegated.
	inv.mu.Lock()
	var marked bool
	for i := range inv.agentStack {
		if inv.agentStack[i].name == "coordinator" && inv.agentStack[i].taskDelegated {
			marked = true
		}
	}
	inv.mu.Unlock()
	if !marked {
		t.Errorf("coordinator was not marked taskDelegated, but a direct LLM parent should be marked")
	}
}

// TestTraceManager_PopTaskAgentSpan_IgnoresUnrelatedAncestor verifies that
// popTaskAgentSpan does NOT mark an ancestor LLM agent (coordinator) as
// taskDelegated when the completing task (task_worker) is not in that
// coordinator's configured SubAgents. This pins the slices.Contains guard.
func TestTraceManager_PopTaskAgentSpan_IgnoresUnrelatedAncestor(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	otherWorker, err := llmagent.New(llmagent.Config{
		Name:        "other_worker",
		Description: "other tasks",
		Mode:        llmagent.ModeTask,
	})
	if err != nil {
		t.Fatalf("llmagent.New(other_worker): %v", err)
	}
	taskWorker, err := llmagent.New(llmagent.Config{
		Name:        "task_worker",
		Description: "executes tasks",
		Mode:        llmagent.ModeTask,
	})
	if err != nil {
		t.Fatalf("llmagent.New(task_worker): %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Coordinates",
		SubAgents:   []agent.Agent{otherWorker}, // note: does NOT contain taskWorker
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	coordID := identity{invocationID: "inv-ancestor", agentName: "coordinator", path: "coordinator@0"}
	taskID := identity{invocationID: "inv-ancestor", agentName: "task_worker", path: "coordinator@0/task_worker@1"}

	tm.invocationSpan(contextRunning(coordinator), coordID, coordinator)
	coordStart := tm.pushAgentSpan(contextRunning(coordinator), coordID)

	taskStart := tm.pushAgentSpan(contextRunning(taskWorker), taskID)
	if taskStart.parentSpanID != coordStart.spanID {
		t.Fatalf("task parentSpanID = %q, want coordinator spanID %q", taskStart.parentSpanID, coordStart.spanID)
	}

	tm.popTaskAgentSpan(ctx, taskID)

	// Verify coordinator was NOT marked taskDelegated because task_worker is not in its children.
	tm.mu.Lock()
	inv := tm.invocations["inv-ancestor"]
	tm.mu.Unlock()
	inv.mu.Lock()
	for i := range inv.agentStack {
		if inv.agentStack[i].name == "coordinator" && inv.agentStack[i].taskDelegated {
			inv.mu.Unlock()
			t.Errorf("coordinator was marked taskDelegated, but task_worker is not in coordinator's SubAgents")
			return
		}
	}
	inv.mu.Unlock()
}
