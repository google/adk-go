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
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

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
	par, _ := tm.pushAgentSpan(ctx, id("par", ""))
	a, _ := tm.pushAgentSpan(ctx, id("a", "par.a"))
	// ab shares the string prefix "par.a" with branch "par.a", but is a
	// sibling branch under par rather than a sub-branch of a.
	ab, _ := tm.pushAgentSpan(ctx, id("ab", "par.ab"))
	b, _ := tm.pushAgentSpan(ctx, id("b", "par.b"))
	// a1 is a sub-agent of a, such as a step of a sequential agent, which
	// keeps the branch of a.
	a1, _ := tm.pushAgentSpan(ctx, id("a1", "par.a"))
	bModel := tm.pushLLMSpan(ctx, id("b", "par.b"))
	a1Tool := tm.pushToolSpan(ctx, id("a1", "par.a"), "call-1")
	a1ToolEnd, _, _ := tm.popToolSpan(ctx, id("a1", "par.a"), "call-1", false)
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
	coordinator, _ := tm.pushAgentSpan(ctx, id("coordinator", ""))
	fan, _ := tm.pushAgentSpan(ctx, id("fan", ""))
	inner, _ := tm.pushAgentSpan(ctx, id("inner", innerBranch))
	x, _ := tm.pushAgentSpan(ctx, id("x", xBranch))
	w1, _ := tm.pushAgentSpan(ctx, id("worker", w1Branch))
	w2, _ := tm.pushAgentSpan(ctx, id("worker", w2Branch))
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
	w1ToolEnd, _, _ := tm.popToolSpan(ctx, id("worker", w1Branch), lookupKey, false)
	w1End := tm.popAgentSpan(ctx, id("worker", w1Branch))
	w2ModelEnd := tm.popLLMSpan(ctx, id("worker", w2Branch), true)
	xEnd := tm.popAgentSpan(ctx, id("x", xBranch))
	w2End := tm.popAgentSpan(ctx, id("worker", w2Branch))
	innerEnd := tm.popAgentSpan(ctx, id("inner", innerBranch))
	fanEnd := tm.popAgentSpan(ctx, id("fan", ""))
	summary, _ := tm.pushAgentSpan(ctx, id("summary", ""))
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
	agentSpan, _ := tm.pushAgentSpan(ctx, id)
	// fail runs OnToolErrorCallback and then AfterToolCallback for the call
	// key, and returns the place of the TOOL_ERROR row.
	fail := func(key string) rowSpan {
		t.Helper()
		errorEnd, _, closedByError := tm.popToolSpan(ctx, id, key, true)
		if closedByError {
			t.Errorf("%s: OnToolErrorCallback found the call ended by an earlier error", key)
		}
		afterEnd, _, closedByError := tm.popToolSpan(ctx, id, key, false)
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
	firstErr, _, firstClosed := tm.popToolSpan(ctx, id, key, true)
	secondErr, _, secondClosed := tm.popToolSpan(ctx, id, key, true)
	if firstClosed || secondClosed || firstErr.spanID == secondErr.spanID {
		t.Errorf("consecutive OnToolErrorCallback-only spans = (%q/%t, %q/%t), want distinct spans with closedByError=false",
			firstErr.spanID, firstClosed, secondErr.spanID, secondClosed)
	}
	next := tm.pushToolSpan(ctx, id, key)
	if got, _, closedByError := tm.popToolSpan(ctx, id, key, false); closedByError || got.spanID != next.spanID || got.latency == nil {
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
	sub1, _ := tm.pushAgentSpan(ctx, identity{invocationID: "inv-1", agentName: "sub"})
	if got := sub1.rootAgentName; got != "root" {
		t.Errorf("rootAgentName after the runner callbacks = %q, want %q", got, "root")
	}
	// Without them, the first agent to start is the root.
	sub2, _ := tm.pushAgentSpan(ctx, identity{invocationID: "inv-2", agentName: "sub"})
	if got := sub2.rootAgentName; got != "sub" {
		t.Errorf("rootAgentName of the first agent = %q, want %q", got, "sub")
	}
	if got := tm.eventSpan(ctx, identity{invocationID: "inv-2", agentName: "root"}, "sub", "").rootAgentName; got != "sub" {
		t.Errorf("rootAgentName after another agent = %q, want %q", got, "sub")
	}
	// When the first callback carries a dotted branch, the top-level
	// branch prefix names the root agent.
	sub3, _ := tm.pushAgentSpan(ctx, identity{invocationID: "inv-3", agentName: "sub", branch: "root.sub"})
	if got := sub3.rootAgentName; got != "root" {
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
	invSpan := tm.invocationSpan(ctx, noID, nil)
	soloSpan, _ := tm.pushAgentSpan(ctx, noID)
	spans := []rowSpan{invSpan, soloSpan, tm.endInvocation(ctx, noID)}
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
	toolEnd, _, _ := tm.popToolSpan(ctx, identity{invocationID: "in-flight", agentName: "sub"}, "long-tool", false)
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
	rootSpan, _ := tm.pushAgentSpan(ctx, rootID)

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
			agentSpan, _ := tm.pushAgentSpan(ctx, subID)
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
			toolEnd, _, closed := tm.popToolSpan(ctx, subID, key, false)
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
			ag, _ := tm.pushAgentSpan(ctx, id)
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
					ag, _ := tm.pushAgentSpan(ctx, id)
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
	subSpan, _ := tm.pushAgentSpan(nestedCtx, identity{invocationID: "inv-otel-1", agentName: "sub"})
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
	// sequence runs popAgentSpan. ADK writes the callback's content as an
	// event of shortcut_last, which shows that it responded.
	tm.completeResponse(ctx, childID, "shortcut_last", "", nil)
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
		parStart, _ := tm.pushAgentSpan(ctx, parID)
		aStart, _ := tm.pushAgentSpan(ctx, aID)
		bStart, _ := tm.pushAgentSpan(ctx, bID)

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
		p1Start, _ := tm.pushAgentSpan(ctx, p1ID)
		p2Start, _ := tm.pushAgentSpan(ctx, p2ID)
		x1Start, _ := tm.pushAgentSpan(ctx, x1ID)
		x2Start, _ := tm.pushAgentSpan(ctx, x2ID)

		if x1Start.parentSpanID != p1Start.spanID {
			t.Errorf("x@1 parentSpanID = %q, want p@1 spanID %q", x1Start.parentSpanID, p1Start.spanID)
		}
		if x2Start.parentSpanID != p2Start.spanID {
			t.Errorf("x@2 parentSpanID = %q, want p@2 spanID %q", x2Start.parentSpanID, p2Start.spanID)
		}

		// Suppose x@1 was short-circuited in BeforeAgentCallback, which writes
		// an event of x@1, and p@1 finishes while p@2 and p@2/x@2 are still
		// open.
		tm.completeResponse(ctx, x1ID, "x", "", nil)
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

// TestOnEvent_ResponseOrPartialEventKeepsAuthorFromFailing verifies which
// session events show that their author responded, so that its agent span,
// left open when the invocation terminates, is not reported as failed: an
// event (final or streamed partial) with Content or with a typed Output, as
// the events of a custom agent or of a workflow node carry. A final event with
// neither does not. A partial event with neither, which ADK emits for a streamed chunk that
// carries only an error code, shows that its author was streaming when the
// caller stopped reading, which is not a failure either.
func TestOnEvent_ResponseOrPartialEventKeepsAuthorFromFailing(t *testing.T) {
	tests := []struct {
		name          string
		event         session.Event
		wantCompleted bool
	}{
		{name: "content", event: session.Event{LLMResponse: model.LLMResponse{Content: textContent("model", "hi")}}, wantCompleted: true},
		{name: "output", event: session.Event{Output: "result"}, wantCompleted: true},
		{name: "content and output", event: session.Event{LLMResponse: model.LLMResponse{Content: textContent("model", "hi")}, Output: "result"}, wantCompleted: true},
		{name: "partial content", event: session.Event{LLMResponse: model.LLMResponse{Partial: true, Content: textContent("model", "h")}}, wantCompleted: true},
		{name: "partial output", event: session.Event{LLMResponse: model.LLMResponse{Partial: true}, Output: "result"}, wantCompleted: true},
		{name: "partial without content or output", event: session.Event{LLMResponse: model.LLMResponse{Partial: true}}, wantCompleted: true},
		{name: "partial error code without content", event: session.Event{LLMResponse: model.LLMResponse{Partial: true, ErrorCode: "SAFETY"}}, wantCompleted: true},
		{name: "neither content nor output", event: session.Event{}, wantCompleted: false},
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
			if tt.wantCompleted {
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
	w1Start, _ := tm.pushAgentSpan(ctx, w1ID)
	w2Start, _ := tm.pushAgentSpan(ctx, w2ID)
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

	parStart, _ := tm.pushAgentSpan(ctx, parID)
	c1Start, _ := tm.pushAgentSpan(ctx, c1ID)
	c2Start, _ := tm.pushAgentSpan(ctx, c2ID)

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
	rootStart, _ := tm.pushAgentSpan(context.Background(), rootID)
	childStart, _ := tm.pushAgentSpan(context.Background(), childID)
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

// requireChildStartedAfterPartialStop calls attempt, which runs an agent
// until the caller stops reading on a partial event and returns the rows of
// the run, until the rows hold an AGENT_STARTING row of an agent named child,
// and requires every run to be reported as completed.
func requireChildStartedAfterPartialStop(t *testing.T, attempt func(t *testing.T) []map[string]any) {
	t.Helper()
	// A parallel agent closes the event's ackChan right before it calls
	// cancelSubAgents when its caller stops reading, so about one run in 30 the
	// branch sees done or ctx.Done() first and returns before it runs child.
	const maxAttempts = 20
	for range maxAttempts {
		rows := attempt(t)
		requireInvocationCompletedOK(t, rows)
		for _, row := range rowsOfType(rows, EventAgentStarting) {
			if row["agent"] == "child" {
				return
			}
		}
	}
	t.Fatalf("child never started after partial stop across %d attempts", maxAttempts)
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

	seqSpan, _ := tm.pushAgentSpan(contextRunning(sequence), identity{invocationID: invocationID, agentName: "sequence", path: "root/sequence@1"})
	coordSpan, _ := tm.pushAgentSpan(contextRunning(coordinator), identity{invocationID: invocationID, agentName: "coordinator", path: "root/coordinator@2"})

	// Under coordinator (an LLM agent), peer starts and stays open when worker
	// starts; worker is parented to peer (peer transfer).
	peerSpan, _ := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "peer", path: "root/coordinator@2"})
	if peerSpan.parentSpanID != coordSpan.spanID {
		t.Errorf("peer parent = %q, want coordinator %q", peerSpan.parentSpanID, coordSpan.spanID)
	}
	coordWorkerSpan, _ := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "worker", path: "root/coordinator@2"})
	if coordWorkerSpan.parentSpanID != peerSpan.spanID {
		t.Errorf("coordinator worker parent = %q, want peer %q", coordWorkerSpan.parentSpanID, peerSpan.spanID)
	}

	// Under sequence (a non-LLM workflow agent), worker starts and stays open
	// when other starts; other skips worker and is parented to sequence.
	seqWorkerSpan, _ := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "worker", path: "root/sequence@1"})
	if seqWorkerSpan.parentSpanID != seqSpan.spanID {
		t.Errorf("sequence worker parent = %q, want sequence %q", seqWorkerSpan.parentSpanID, seqSpan.spanID)
	}
	seqOtherSpan, _ := tm.pushAgentSpan(ctx, identity{invocationID: invocationID, agentName: "other", path: "root/sequence@1"})
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

	sequenceSpan, _ := tm.pushAgentSpan(contextRunning(sequence), id("sequence"))
	shortcutSpan, _ := tm.pushAgentSpan(ctx, id("shortcut"))
	nextSpan, _ := tm.pushAgentSpan(ctx, id("next"))

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

	t.Run("parallel root with sibling model call returning ctx.Err on cancel", func(t *testing.T) {
		requireADKParallelEarlyStopTeardown(t)
		p, requests, _ := setupTestPlugin(t)
		slowEntered := make(chan struct{})
		slow, err := llmagent.New(llmagent.Config{
			Name:  "slow",
			Model: &contextCanceledOnDoneLLM{entered: slowEntered},
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
		for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if runErr != nil {
				t.Fatalf("r.Run error: %v", runErr)
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
		if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
			t.Errorf("got %d LLM_ERROR rows, want 1 (from slow branch context.Canceled)", n)
		}
	})

	t.Run("workflow root with sibling node in flight", func(t *testing.T) {
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
		botNode, err := workflow.NewAgentNode(bot, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(bot): %v", err)
		}
		slowNode, err := workflow.NewAgentNode(slow, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(slow): %v", err)
		}
		wf, err := workflowagent.New(workflowagent.Config{
			Name: "wf_root",
			Edges: []workflow.Edge{
				{From: workflow.Start, To: botNode},
				{From: workflow.Start, To: slowNode},
			},
		})
		if err != nil {
			t.Fatalf("workflowagent.New(wf_root): %v", err)
		}
		r := newTestRunner(t, wf, p)

		var sawPartial bool
		for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if runErr != nil {
				t.Fatalf("r.Run error: %v", runErr)
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

// flakyOnFirstPassLLM fails with err on call 1 and returns text on
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
//     activation, which the delegation interrupted, ends with an
//     AGENT_COMPLETED row and its latency, and the second coordinator
//     activation is parented to the invocation span rather than to the
//     finished task agent or the first coordinator activation.
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

		if len(coordCompletes) != 2 {
			t.Fatalf("coordinator AGENT_COMPLETED count = %d, want 2", len(coordCompletes))
		}
		for i, row := range coordCompletes {
			if got, want := placeOf(row), placeOf(coordStarts[i]); got != want {
				t.Errorf("coordinator AGENT_COMPLETED[%d] place = %+v, want coordinator activation %d %+v", i, got, i, want)
			}
			mustLatency(t, row)
		}
		assertAgentSpansClosed(t, rows)
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

// TestRunner_ConcurrentGraphParentsSameChildNames verifies that when a
// workflow graph runs two sequential workflow agents (flow_a and flow_b)
// concurrently and both have children with the same names ("gate", answered in
// BeforeAgentCallback, and "step"), each child's AGENT_STARTING row and step's
// AGENT_COMPLETED row are parented to its own sequential parent (flow_a or
// flow_b) rather than escaping to the root or invocation span.
func TestRunner_ConcurrentGraphParentsSameChildNames(t *testing.T) {
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

	var rootStart, flowAStart, flowBStart tracePlace
	var gateParents, stepStartParents []string
	stepStartBySpan := map[string]tracePlace{}
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		pl := placeOf(row)
		switch row["agent"] {
		case "root":
			rootStart = pl
		case "flow_a":
			flowAStart = pl
		case "flow_b":
			flowBStart = pl
		case "gate":
			gateParents = append(gateParents, pl.parent)
		case "step":
			stepStartParents = append(stepStartParents, pl.parent)
			stepStartBySpan[pl.span] = pl
		}
	}
	if rootStart.span == "" || flowAStart.span == "" || flowBStart.span == "" {
		t.Fatalf("missing spans: root=%q flow_a=%q flow_b=%q", rootStart.span, flowAStart.span, flowBStart.span)
	}
	wantFlowSpans := []string{flowAStart.span, flowBStart.span}
	slices.Sort(wantFlowSpans)
	slices.Sort(gateParents)
	slices.Sort(stepStartParents)
	if diff := cmp.Diff(wantFlowSpans, gateParents); diff != "" {
		t.Errorf("gate AGENT_STARTING parent_span_id mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(wantFlowSpans, stepStartParents); diff != "" {
		t.Errorf("step AGENT_STARTING parent_span_id mismatch (-want +got):\n%s", diff)
	}

	var stepCompleteParents []string
	var flowAComplete, flowBComplete tracePlace
	for _, row := range rowsOfType(rows, EventAgentCompleted) {
		pl := placeOf(row)
		switch row["agent"] {
		case "flow_a":
			flowAComplete = pl
			mustLatency(t, row)
		case "flow_b":
			flowBComplete = pl
			mustLatency(t, row)
		case "step":
			mustLatency(t, row)
			startPl, ok := stepStartBySpan[pl.span]
			if !ok {
				t.Errorf("step AGENT_COMPLETED span_id %q does not match any step AGENT_STARTING span", pl.span)
				continue
			}
			if pl != startPl {
				t.Errorf("step AGENT_COMPLETED place = %+v, want %+v", pl, startPl)
			}
			stepCompleteParents = append(stepCompleteParents, pl.parent)
		}
	}
	if flowAComplete != flowAStart || flowAStart.parent != rootStart.span {
		t.Errorf("flow_a AGENT_COMPLETED = %+v, start = %+v (want parent %q)", flowAComplete, flowAStart, rootStart.span)
	}
	if flowBComplete != flowBStart || flowBStart.parent != rootStart.span {
		t.Errorf("flow_b AGENT_COMPLETED = %+v, start = %+v (want parent %q)", flowBComplete, flowBStart, rootStart.span)
	}
	slices.Sort(stepCompleteParents)
	if diff := cmp.Diff(wantFlowSpans, stepCompleteParents); diff != "" {
		t.Errorf("step AGENT_COMPLETED parent_span_id mismatch (-want +got):\n%s", diff)
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
		agentSpan, _ := tm.pushAgentSpan(otelCtx, id1)
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
	seqStart, _ := tm.pushAgentSpan(contextRunning(seq), seqID)

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
	taskStart, _ := tm.pushAgentSpan(contextRunning(stepTask), taskID)
	if taskStart.parentSpanID != seqStart.spanID {
		t.Fatalf("task parentSpanID = %q, want seq spanID %q", taskStart.parentSpanID, seqStart.spanID)
	}

	// Pop the task span as popTaskAgentSpan does (finish_task completion).
	tm.popTaskAgentSpan(ctx, taskID, nil)

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
	coordStart, _ := tm.pushAgentSpan(contextRunning(coordinator), coordID)

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

	taskStart, _ := tm.pushAgentSpan(contextRunning(taskWorker), taskID)
	if taskStart.parentSpanID != coordStart.spanID {
		t.Fatalf("task parentSpanID = %q, want coordinator spanID %q", taskStart.parentSpanID, coordStart.spanID)
	}

	tm.popTaskAgentSpan(ctx, taskID, nil)

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
	coordStart, _ := tm.pushAgentSpan(contextRunning(coordinator), coordID)

	taskStart, _ := tm.pushAgentSpan(contextRunning(taskWorker), taskID)
	if taskStart.parentSpanID != coordStart.spanID {
		t.Fatalf("task parentSpanID = %q, want coordinator spanID %q", taskStart.parentSpanID, coordStart.spanID)
	}

	tm.popTaskAgentSpan(ctx, taskID, nil)

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

// singlePartialChunkLLM yields one streamed partial LLMResponse and stops if
// the caller breaks out.
type singlePartialChunkLLM struct {
	chunk *model.LLMResponse
}

func (m *singlePartialChunkLLM) Name() string { return "single-partial-chunk-llm" }

func (m *singlePartialChunkLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(m.chunk, nil)
	}
}

// TestRunner_FirstStreamedChunkRejectedByAgentCallback_EmitsErrorTerminalRows
// verifies that when an agent-level AfterModelCallback rejects the first
// streamed chunk (after plugin AfterModelCallback already ran), the invocation
// ends with AGENT_ERROR and INVOCATION_ERROR rather than INVOCATION_COMPLETED.
func TestRunner_FirstStreamedChunkRejectedByAgentCallback_EmitsErrorTerminalRows(t *testing.T) {
	tests := []struct {
		name  string
		chunk *model.LLMResponse
	}{
		{
			name: "generated content chunk rejected by AfterModelCallback",
			chunk: &model.LLMResponse{
				Partial: true,
				Content: textContent("model", "hello "),
			},
		},
		{
			name: "metadata-only leading chunk rejected by AfterModelCallback",
			chunk: &model.LLMResponse{
				Partial:       true,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			rejectErr := errors.New("policy rejected first chunk")
			bot, err := llmagent.New(llmagent.Config{
				Name:  "bot",
				Model: &singlePartialChunkLLM{chunk: tc.chunk},
				AfterModelCallbacks: []llmagent.AfterModelCallback{
					func(_ agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
						return nil, rejectErr
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			r := newTestRunner(t, bot, p)

			var gotErr error
			var sawPartial bool
			for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{
				StreamingMode: agent.StreamingModeSSE,
			}) {
				if runErr != nil {
					gotErr = runErr
				}
				if ev != nil && ev.Partial {
					sawPartial = true
				}
			}
			if gotErr == nil {
				t.Fatal("expected runner.Run to yield AfterModelCallback error")
			}
			if sawPartial {
				t.Fatal("did not expect a rejected partial chunk to be yielded as a session event")
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			if invErr["status"] != "ERROR" {
				t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invErr["status"])
			}
			if got, want := agentErrorCounts(rows), map[string]int{"bot": 1}; !maps.Equal(got, want) {
				t.Errorf("AGENT_ERROR counts = %v, want %v", got, want)
			}
		})
	}
}

// TestRunner_PartialEventFromCustomRootThenSubAgentError_EmitsErrorTerminalRows
// verifies that a partial event of a custom root does not hide the failure of
// a sub-agent that the root runs after it: the sub-agent's
// BeforeModelCallback fails, which no error callback reports, whether the
// caller stops at the error or reads to the end.
func TestRunner_PartialEventFromCustomRootThenSubAgentError_EmitsErrorTerminalRows(t *testing.T) {
	for _, stopAtError := range []bool{true, false} {
		t.Run(fmt.Sprintf("stop at error %v", stopAtError), func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			subBot, err := llmagent.New(llmagent.Config{
				Name:  "sub_bot",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "unused"), TurnComplete: true}}},
				BeforeModelCallbacks: []llmagent.BeforeModelCallback{
					func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
						return nil, errors.New("sub_bot before-model boom")
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(sub_bot): %v", err)
			}
			customRoot, err := agent.New(agent.Config{
				Name:      "custom_root",
				SubAgents: []agent.Agent{subBot},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						partial := session.NewEvent(ctx, ctx.InvocationID())
						partial.Author = "custom_root"
						partial.Branch = ctx.Branch()
						partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "progress update...")}
						if !yield(partial, nil) {
							return
						}
						for ev, err := range subBot.Run(ctx) {
							if !yield(ev, err) || err != nil {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(custom_root): %v", err)
			}

			r := newTestRunner(t, customRoot, p)
			var sawPartial, sawErr bool
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					sawErr = true
					if stopAtError {
						break
					}
					continue
				}
				if ev != nil && ev.Partial {
					sawPartial = true
				}
			}
			if !sawPartial || !sawErr {
				t.Fatalf("runner.Run yielded partial event %v and error %v, want both", sawPartial, sawErr)
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			onlyRowOfType(t, rows, EventInvocationError)
			if got := agentErrorCounts(rows); got["sub_bot"] != 1 {
				t.Errorf("AGENT_ERROR rows per agent = %v, want one for sub_bot", got)
			}
		})
	}
}

// requireCompletedWithHandledAgentError fails the test unless rows end the
// invocation with INVOCATION_COMPLETED, status OK, and hold one AGENT_ERROR
// row, of agentName, in the span of its first AGENT_STARTING row, with latency
// and an error message that contains wantErr: ADK handled the failure of that
// activation itself.
func requireCompletedWithHandledAgentError(t *testing.T, rows []map[string]any, agentName, wantErr string) {
	t.Helper()
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	if invRow := onlyRowOfType(t, rows, EventInvocationCompleted); invRow["status"] != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", invRow["status"])
	}
	errRow := onlyRowOfType(t, rows, EventAgentError)
	if errRow["agent"] != agentName {
		t.Errorf("AGENT_ERROR agent = %v, want %q", errRow["agent"], agentName)
	}
	var start map[string]any
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		if row["agent"] == agentName {
			start = row
			break
		}
	}
	if start == nil {
		t.Fatalf("no AGENT_STARTING row of %q; event types: %v", agentName, eventTypes(rows))
	}
	if got, want := placeOf(errRow), placeOf(start); got != want {
		t.Errorf("AGENT_ERROR place = %+v, want AGENT_STARTING place %+v", got, want)
	}
	mustLatency(t, errRow)
	if msg, _ := errRow["error_message"].(string); !strings.Contains(msg, wantErr) {
		t.Errorf("AGENT_ERROR error_message = %q, want it to contain %q", msg, wantErr)
	}
}

// TestRunner_InternallyRecoveredErrors_SingleTurnToolAndWorkflowRetry_EmitsCompletedTerminalRows
// verifies that failures handled internally by ADK — a ModeSingleTurn sub-agent
// tool failure converted into a tool error response for its parent, and a
// workflow node retried by RetryConfig — do not cause the enclosing invocation
// to end with INVOCATION_ERROR, while the failed activation ends with an
// AGENT_ERROR row of its own.
func TestRunner_InternallyRecoveredErrors_SingleTurnToolAndWorkflowRetry_EmitsCompletedTerminalRows(t *testing.T) {
	t.Run("single_turn sub-agent tool fails and parent recovers", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)

		helper, err := llmagent.New(llmagent.Config{
			Name:        "helper",
			Description: "Runs a single-turn helper task.",
			Mode:        llmagent.ModeSingleTurn,
			Model:       &failingLLM{err: errors.New("helper model unavailable")},
		})
		if err != nil {
			t.Fatalf("llmagent.New(helper): %v", err)
		}

		coordinator, err := llmagent.New(llmagent.Config{
			Name:      "coordinator",
			Mode:      llmagent.ModeChat,
			SubAgents: []agent.Agent{helper},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-helper-1",
										Name: "helper",
										Args: map[string]any{"request": "try helper"},
									},
								},
							},
						},
					},
					{
						Content:      textContent("model", "answered directly after helper failed"),
						TurnComplete: true,
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(coordinator): %v", err)
		}

		r := newTestRunner(t, coordinator, p)
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "help me"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("r.Run yielded unexpected error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireCompletedWithHandledAgentError(t, rows, "helper", "helper model unavailable")
		if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
			t.Errorf("got %d LLM_ERROR rows, want 1", n)
		}
		if n := len(rowsOfType(rows, EventToolError)); n != 1 {
			t.Errorf("got %d TOOL_ERROR rows, want 1", n)
		}
		agentErrIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
			return r["event_type"] == EventAgentError && r["agent"] == "helper"
		})
		toolErrIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
			return r["event_type"] == EventToolError
		})
		if agentErrIdx < 0 || toolErrIdx < 0 || agentErrIdx >= toolErrIdx {
			t.Errorf("AGENT_ERROR(helper) index = %d, TOOL_ERROR index = %d; want 0 <= AGENT_ERROR < TOOL_ERROR in %v", agentErrIdx, toolErrIdx, eventTypes(rows))
		}
	})

	t.Run("workflow node RetryConfig recovers on retry attempt", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)

		flaky, err := llmagent.New(llmagent.Config{
			Name:  "flaky",
			Mode:  llmagent.ModeSingleTurn,
			Model: &flakyOnFirstPassLLM{err: errors.New("transient 503"), reply: "succeeded on attempt 2"},
		})
		if err != nil {
			t.Fatalf("llmagent.New(flaky): %v", err)
		}

		node, err := workflow.NewAgentNode(flaky, workflow.NodeConfig{
			RetryConfig: &workflow.RetryConfig{
				MaxAttempts:   2,
				InitialDelay:  time.Millisecond,
				MaxDelay:      time.Millisecond,
				BackoffFactor: 1.0,
			},
		})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(flaky): %v", err)
		}

		wfRoot, err := workflowagent.New(workflowagent.Config{
			Name: "wf_root",
			Edges: []workflow.Edge{
				{From: workflow.Start, To: node},
			},
		})
		if err != nil {
			t.Fatalf("workflowagent.New(wf_root): %v", err)
		}

		r := newTestRunner(t, wfRoot, p)
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "run workflow"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("r.Run yielded unexpected error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireCompletedWithHandledAgentError(t, rows, "flaky", "transient 503")
		if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
			t.Errorf("got %d LLM_ERROR rows, want 1 (from failed attempt 1)", n)
		}
		if n := len(rowsOfType(rows, EventLLMResponse)); n != 1 {
			t.Errorf("got %d LLM_RESPONSE rows, want 1 (from successful retry)", n)
		}
	})
}

// TestRunner_SingleTurnToolAnsweredByBeforeAgentCallback_EmitsNoAgentError
// verifies that a ModeSingleTurn sub-agent tool whose BeforeAgentCallback
// answers with cached content, so that it never gets an AfterAgentCallback,
// is not reported with an AGENT_ERROR row once its owner's tool calls end, and
// that the invocation still ends with INVOCATION_COMPLETED, whether the owner
// calls it once or twice in one turn.
func TestRunner_SingleTurnToolAnsweredByBeforeAgentCallback_EmitsNoAgentError(t *testing.T) {
	helperCall := func(id string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "helper", Args: map[string]any{"request": "cached?"}}}
	}
	tests := []struct {
		name  string
		calls []*genai.Part
	}{
		{name: "one call", calls: []*genai.Part{helperCall("call-helper-1")}},
		{name: "two calls in one turn", calls: []*genai.Part{helperCall("call-helper-1"), helperCall("call-helper-2")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.calls) > 1 {
				requireADKRaceFreeSharedSingleTurnTool(t)
			}
			p, requests, _ := setupTestPlugin(t)

			helper, err := llmagent.New(llmagent.Config{
				Name:        "helper",
				Description: "Answers from a cache.",
				Mode:        llmagent.ModeSingleTurn,
				Model:       &scriptedLLM{},
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						return textContent("model", "cached answer"), nil
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(helper): %v", err)
			}

			coordinator, err := llmagent.New(llmagent.Config{
				Name:      "coordinator",
				Mode:      llmagent.ModeChat,
				SubAgents: []agent.Agent{helper},
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					{Content: &genai.Content{Role: "model", Parts: tt.calls}},
					{Content: textContent("model", "done with the cached answer"), TurnComplete: true},
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(coordinator): %v", err)
			}

			r := newTestRunner(t, coordinator, p)
			for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "help me"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("r.Run yielded unexpected error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := agentErrorCounts(rows)["helper"]; n != 0 {
				t.Errorf("got %d AGENT_ERROR rows for helper, want 0; event types: %v", n, eventTypes(rows))
			}
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
			}
			onlyRowOfType(t, rows, EventInvocationCompleted)
		})
	}
}

// TestRunner_RetriedCompositeNodeSkipsCachedChildAgentError verifies that when
// RetryConfig runs a workflow node again that wraps a sequential agent, only
// the failing lineage of the failed attempt ends with an AGENT_ERROR row: the
// child that its BeforeAgentCallback answered with cached content never got an
// AfterAgentCallback, but it did not fail, so it ends without a row and does
// not take the error of the sibling that failed.
func TestRunner_RetriedCompositeNodeSkipsCachedChildAgentError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	cached, err := llmagent.New(llmagent.Config{
		Name:  "cached",
		Model: &scriptedLLM{},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) { return textContent("model", "cached answer"), nil },
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(cached): %v", err)
	}
	flaky, err := llmagent.New(llmagent.Config{
		Name:  "flaky",
		Model: &flakyOnFirstPassLLM{err: errors.New("flaky backend unavailable"), reply: "succeeded on attempt 2"},
	})
	if err != nil {
		t.Fatalf("llmagent.New(flaky): %v", err)
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{cached, flaky}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(pipeline): %v", err)
	}
	node, err := workflow.NewAgentNode(pipeline, workflow.NodeConfig{
		RetryConfig: &workflow.RetryConfig{
			MaxAttempts:   2,
			InitialDelay:  time.Millisecond,
			MaxDelay:      time.Millisecond,
			BackoffFactor: 1.0,
		},
	})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(pipeline): %v", err)
	}
	wfRoot, err := workflowagent.New(workflowagent.Config{
		Name:  "wf_root",
		Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(wf_root): %v", err)
	}

	r := newTestRunner(t, wfRoot, p)
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "run the pipeline"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("r.Run yielded unexpected error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationCompleted)
	if got, want := agentErrorCounts(rows), map[string]int{"flaky": 1, "pipeline": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR rows per agent = %v, want %v; event types: %v", got, want, eventTypes(rows))
	}
	for _, row := range rowsOfType(rows, EventAgentError) {
		if msg, _ := row["error_message"].(string); !strings.Contains(msg, "flaky backend unavailable") {
			t.Errorf("AGENT_ERROR of %v error_message = %q, want it to contain the failure of flaky", row["agent"], msg)
		}
	}
}

// newHelperCoordinator returns a coordinator that calls its single_turn
// sub-agent helper, whose model is helperModel, as a tool once.
func newHelperCoordinator(t *testing.T, helperModel model.LLM) agent.Agent {
	t.Helper()
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "Runs a single-turn helper task.",
		Mode:        llmagent.ModeSingleTurn,
		Model:       helperModel,
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{helper},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "call-helper-1", Name: "helper", Args: map[string]any{"request": "stream something"},
			}}}}},
			{Content: textContent("model", "done after the helper"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}
	return coordinator
}

// runHelperCoordinator runs a coordinator that calls a streaming single_turn
// helper as a tool, and returns the rows of the run. The caller stops reading
// on the first partial event of the helper if stop is set, and reads to the
// end otherwise.
func runHelperCoordinator(t *testing.T, helperModel model.LLM, stop bool) []map[string]any {
	t.Helper()
	p, requests, _ := setupTestPlugin(t)
	var sawPartial bool
	for ev, err := range newTestRunner(t, newHelperCoordinator(t, helperModel), p).Run(t.Context(), "u1", "s1", textContent("user", "help me"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
		if ev != nil && ev.Partial && ev.Author == "helper" {
			sawPartial = true
			if stop {
				break
			}
		}
	}
	if !sawPartial {
		t.Fatal("runner.Run yielded no partial event of helper")
	}
	return collectRows(t, requests, untilInvocationEnd)
}

// TestRunner_StopOnToolRunHelperPartialEvent_EmitsNoAgentError verifies that a
// single_turn helper that its coordinator runs as a tool, and that the run
// cancels because the caller stopped reading on one of its partial events, is
// not reported as failed: the invocation completes, as for any other agent
// that the stop canceled, so no AGENT_ERROR row of the helper contradicts it.
func TestRunner_StopOnToolRunHelperPartialEvent_EmitsNoAgentError(t *testing.T) {
	rows := runHelperCoordinator(t, streamThenWaitLLM{}, true)
	if !slices.ContainsFunc(rowsOfType(rows, EventLLMError), func(row map[string]any) bool { return row["agent"] == "helper" }) {
		t.Fatalf("got no LLM_ERROR row of helper, want the model call that the stop canceled; event types: %v", eventTypes(rows))
	}
	requireInvocationCompletedOK(t, rows)
}

// TestRunner_ToolRunHelperCanceledItselfAfterPartial_EmitsAgentError verifies
// that a single_turn helper whose model cancels a request context of its own
// after a partial chunk still ends with an AGENT_ERROR row when the caller
// reads on: only the cancellation by a stop is no failure, and ADK hands the
// failure to the coordinator's model, so the invocation completes.
func TestRunner_ToolRunHelperCanceledItselfAfterPartial_EmitsAgentError(t *testing.T) {
	rows := runHelperCoordinator(t, &selfCancelingStreamLLM{}, false)
	requireCompletedWithHandledAgentError(t, rows, "helper", "context canceled")
}

// TestRunner_ToolRunHelperCanceledByOuterCtxAfterPartial_EmitsAgentError
// verifies that when the outer run context is canceled while a single_turn
// tool-run helper is streaming a partial event, the helper emits AGENT_ERROR
// and the invocation ends INVOCATION_ERROR (the cancellation is not a consumer
// stop: it is a failure of the run).
func TestRunner_ToolRunHelperCanceledByOuterCtxAfterPartial_EmitsAgentError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	coordinator := newHelperCoordinator(t, streamThenWaitLLM{})
	r := newTestRunner(t, coordinator, p)
	for ev, err := range r.Run(ctx, "u1", "s1", textContent("user", "help me"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			break
		}
		if ev != nil && ev.Partial && ev.Author == "helper" {
			cancel()
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	// The helper must get an AGENT_ERROR row and the run must end
	// INVOCATION_ERROR: canceling the outer run context is a failure of the
	// run, not a consumer stop.
	if got := agentErrorCounts(rows)["helper"]; got != 1 {
		t.Errorf("AGENT_ERROR rows for helper = %d, want 1; event types: %v", got, eventTypes(rows))
	}
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED, want INVOCATION_ERROR; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationError)
}

// TestRunner_TwoHelpersPartialThenHandledError_EmitsCompleted verifies that
// when a coordinator calls two single_turn helpers concurrently, one emits a
// partial event and then hits a model error that ADK handles as a tool error,
// while the other blocks until the consumer stops on the partial event,
// the invocation ends INVOCATION_COMPLETED: the handled helper's model error
// becomes its own AGENT_ERROR row, the other helper is retired quietly, and
// the purged helper's context.Canceled does not become an invocation failure.
// The purge must preserve lastSessionEventPartial and partialLineage so that
// the consumer-stop suppression survives the purge.
func TestRunner_TwoHelpersPartialThenHandledError_EmitsCompleted(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	slowEntered := make(chan struct{})

	// helperModel streams a partial event once slow is inside GenerateContent,
	// then returns a model error.
	helperModel := &streamThenErrorLLM{
		partial: &model.LLMResponse{Content: textContent("model", "partial..."), Partial: true},
		err:     errors.New("helper unavailable"),
	}
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "Runs a single-turn helper task.",
		Mode:        llmagent.ModeSingleTurn,
		Model:       helperModel,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			select {
			case <-slowEntered:
				return nil, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}

	// slowModel signals slowEntered, blocks until ctx.Done, then returns the context error.
	slow, err := llmagent.New(llmagent.Config{
		Name:        "slow",
		Description: "Runs a single-turn slow task.",
		Mode:        llmagent.ModeSingleTurn,
		Model:       &contextCanceledOnDoneLLM{entered: slowEntered},
	})
	if err != nil {
		t.Fatalf("llmagent.New(slow): %v", err)
	}

	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{helper, slow},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "call-helper", Name: "helper", Args: map[string]any{"request": "stream"}},
			}, {
				FunctionCall: &genai.FunctionCall{ID: "call-slow", Name: "slow", Args: map[string]any{"request": "wait"}},
			}}}},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	r := newTestRunner(t, coordinator, p)
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			break
		}
		if ev != nil && ev.Partial && ev.Author == "helper" {
			break // consumer stops on the partial event
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)

	// The handled helper's model error becomes its own AGENT_ERROR row.
	if got := agentErrorCounts(rows)["helper"]; got != 1 {
		t.Errorf("AGENT_ERROR rows for helper = %d, want 1; event types: %v", got, eventTypes(rows))
	}
	// The slow helper is retired quietly: no AGENT_ERROR.
	if got := agentErrorCounts(rows)["slow"]; got != 0 {
		t.Errorf("AGENT_ERROR rows for slow = %d, want 0; event types: %v", got, eventTypes(rows))
	}
	// The invocation completes despite the handled failure.
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationCompleted)
}

// TestRunner_StopOnContentlessPartialHelper_EmitsNoAgentError verifies that a
// single_turn helper that streams one partial chunk with no content, and then
// blocks until ctx.Done, is retired quietly when the caller stops on that
// partial event: the helper never responded, but canceledByStop() exempts it
// because the consumer stop canceled it.
func TestRunner_StopOnContentlessPartialHelper_EmitsNoAgentError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)

	// contentlessPartialLLM streams one partial chunk with no content, then
	// blocks until ctx.Done.
	contentlessModel := &contentlessPartialLLM{}
	helper, err := llmagent.New(llmagent.Config{
		Name:        "helper",
		Description: "Runs a single-turn helper task.",
		Mode:        llmagent.ModeSingleTurn,
		Model:       contentlessModel,
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}

	coordinator, err := llmagent.New(llmagent.Config{
		Name:      "coordinator",
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{helper},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "call-helper", Name: "helper", Args: map[string]any{"request": "stream"}},
			}}}},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	r := newTestRunner(t, coordinator, p)
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			break
		}
		if ev != nil && ev.Partial && ev.Author == "helper" {
			break
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if got := agentErrorCounts(rows)["helper"]; got != 0 {
		t.Errorf("AGENT_ERROR rows for helper = %d, want 0; event types: %v", got, eventTypes(rows))
	}
}

// TestTraceManager_PurgedActivationLatencyFromLastActivity verifies that a
// purged activation's latency and timestamp reflect the latest callback under
// it, not the retry delay that follows it. For each callback that touches an
// agent's lastActivity, the purged activation's latency_ms is the time from
// the activation's start to that callback, and its timestamp is the callback's
// time, not the purge time.
//
// setupFunc runs while the clock is at t0 (the activation's start); touchFunc
// runs after the clock advances to t1, so the test isolates the callback that
// sets lastActivity and fails when that callback's touchAgentActivityLocked
// call is deleted.
func TestTraceManager_PurgedActivationLatencyFromLastActivity(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-activity"
	node := identity{invocationID: invID, agentName: "node", path: "wf@1/node@1"}
	child := identity{invocationID: invID, agentName: "child", path: "wf@1/node@1"}
	t0 := time.Unix(1000, 0)
	t1 := t0.Add(50 * time.Millisecond)
	t2 := t0.Add(2000 * time.Millisecond)
	tests := []struct {
		name      string
		setupFunc func(tm *traceManager)
		touchFunc func(tm *traceManager)
	}{
		{
			name:      "pushAgentSpan",
			setupFunc: func(tm *traceManager) {},
			touchFunc: func(tm *traceManager) {
				tm.pushAgentSpan(ctx, child)
			},
		},
		{
			name: "popAgentSpan",
			setupFunc: func(tm *traceManager) {
				tm.pushAgentSpan(ctx, child)
				tm.pushLLMSpan(ctx, child)
				tm.popLLMSpan(ctx, child, true)
			},
			touchFunc: func(tm *traceManager) {
				tm.popAgentSpan(ctx, child)
			},
		},
		{
			name: "popTaskAgentSpan",
			setupFunc: func(tm *traceManager) {
				tm.pushAgentSpan(ctx, child)
				// Synthetic ctx (context.Background) does not implement
				// WithICDelta (see runningAgentChildren), so pushAgentSpan leaves
				// taskMode=false. Set it so popTaskAgentSpan genuinely
				// pops the child (ok==true).
				inv := tm.invocations[invID]
				inv.mu.Lock()
				inv.agentStack[len(inv.agentStack)-1].taskMode = true
				inv.mu.Unlock()
				tm.pushLLMSpan(ctx, child)
				tm.popLLMSpan(ctx, child, true)
			},
			touchFunc: func(tm *traceManager) {
				tm.popTaskAgentSpan(ctx, child, nil)
			},
		},
		{
			name:      "pushLLMSpan",
			setupFunc: func(tm *traceManager) {},
			touchFunc: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, node)
			},
		},
		{
			name: "recordFirstToken",
			setupFunc: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, node)
			},
			touchFunc: func(tm *traceManager) {
				tm.recordFirstToken(ctx, node)
			},
		},
		{
			name: "recordStreamed without content",
			setupFunc: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, node)
			},
			touchFunc: func(tm *traceManager) {
				tm.recordStreamed(ctx, node, "node", "", false)
			},
		},
		{
			name: "popLLMSpan not responded",
			setupFunc: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, node)
			},
			touchFunc: func(tm *traceManager) {
				tm.popLLMSpan(ctx, node, false)
			},
		},
		{
			name:      "pushToolSpan",
			setupFunc: func(tm *traceManager) {},
			touchFunc: func(tm *traceManager) {
				tm.pushToolSpan(ctx, node, toolKey(node, "call-1", "lookup"))
			},
		},
		{
			name: "popToolSpan",
			setupFunc: func(tm *traceManager) {
				tm.pushToolSpan(ctx, node, toolKey(node, "call-1", "lookup"))
			},
			touchFunc: func(tm *traceManager) {
				tm.popToolSpan(ctx, node, toolKey(node, "call-1", "lookup"), false)
			},
		},
		{
			name:      "recordError",
			setupFunc: func(tm *traceManager) {},
			touchFunc: func(tm *traceManager) {
				tm.recordError(ctx, node, llmKey(node), errors.New("model failed"))
			},
		},
		{
			name: "completeResponse with open span",
			setupFunc: func(tm *traceManager) {
				tm.pushAgentSpan(ctx, child)
				tm.recordError(ctx, node, llmKey(node), errors.New("tool failed"))
			},
			touchFunc: func(tm *traceManager) {
				tm.completeResponse(ctx, child, "child", "", nil)
			},
		},
		{
			name: "completeResponse with no open span",
			setupFunc: func(tm *traceManager) {
				tm.recordError(ctx, node, llmKey(node), errors.New("tool failed"))
			},
			touchFunc: func(tm *traceManager) {
				// completeResponse with an author that has no open agent span
				// touches the activation that the author would have run under.
				tm.completeResponse(ctx, node, "ghost", "", nil)
			},
		},
		{
			name: "eventSpan",
			setupFunc: func(tm *traceManager) {
				tm.recordError(ctx, node, llmKey(node), errors.New("tool failed"))
			},
			touchFunc: func(tm *traceManager) {
				tm.eventSpan(ctx, node, "node", "")
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clock := t0
			tm := newTraceManager()
			tm.now = func() time.Time { return clock }
			tm.invocationSpan(ctx, node, nil)
			tm.pushAgentSpan(ctx, node)
			tt.setupFunc(tm)
			clock = t1
			tt.touchFunc(tm)
			clock = t2
			_, purged := tm.pushAgentSpan(ctx, node)
			var p *unclosedAgentSpan
			for i := range purged {
				if purged[i].agentName == "node" {
					p = &purged[i]
					break
				}
			}
			if p == nil {
				t.Fatalf("pushAgentSpan returned no purged entry for node; entries: %d", len(purged))
			}
			wantMS := t1.Sub(t0).Milliseconds()
			if got := p.span.latency["total_ms"]; got != wantMS {
				t.Errorf("latency total_ms = %v, want %d (start→t1), not %d (start→t2)", got, wantMS, t2.Sub(t0).Milliseconds())
			}
			if p.timestamp != t1 {
				t.Errorf("timestamp = %v, want %v (t1), not %v (t2)", p.timestamp, t1, t2)
			}
		})
	}
}

// TestTraceManager_TouchAgentActivityMonotonic verifies that touchAgentActivity
// never moves lastActivity backward: touching with an earlier clock after a
// later one leaves lastActivity unchanged.
func TestTraceManager_TouchAgentActivityMonotonic(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-mono"
	node := identity{invocationID: invID, agentName: "node", path: "wf@1/node@1"}
	clock := time.Unix(1000, 0)
	tm := newTraceManager()
	tm.now = func() time.Time { return clock }
	tm.invocationSpan(ctx, node, nil)
	tm.pushAgentSpan(ctx, node)
	// Touch at a later time via pushLLMSpan.
	clock = clock.Add(100 * time.Millisecond)
	tm.pushLLMSpan(ctx, node)
	tm.popLLMSpan(ctx, node, true)
	// Touch at an earlier time via recordError.
	clock = clock.Add(-50 * time.Millisecond)
	tm.recordError(ctx, node, llmKey(node), errors.New("model failed"))
	// Purge at an even later time.
	clock = clock.Add(5000 * time.Millisecond)
	_, purged := tm.pushAgentSpan(ctx, node)
	if len(purged) != 1 {
		t.Fatalf("pushAgentSpan purged = %d entries, want 1", len(purged))
	}
	p := purged[0]
	// lastActivity should be the pushLLMSpan time (1000.1s), not the
	// earlier recordError time (1000.05s).
	want := time.Unix(1000, 0).Add(100 * time.Millisecond)
	wantMS := want.Sub(time.Unix(1000, 0)).Milliseconds()
	if got := p.span.latency["total_ms"]; got != wantMS {
		t.Errorf("latency total_ms = %v, want %d (monotonic, not regressed by earlier recordError)", got, wantMS)
	}
	if p.timestamp != want {
		t.Errorf("timestamp = %v, want %v (monotonic)", p.timestamp, want)
	}
}

// TestRunner_PurgedRetryAttemptTimestampPrecedesNextAttempt verifies that a
// retried attempt's AGENT_ERROR row is timestamped before the next attempt's
// AGENT_STARTING row, and its latency reflects its own work, not the retry
// delay. Without node completion events, the attempt's run is not known to be
// over when the next attempt starts, so its row comes when the workflow agent
// ends instead (see endRestartedNodeRuns).
func TestRunner_PurgedRetryAttemptTimestampPrecedesNextAttempt(t *testing.T) {
	requireADKNodeCompletionEvents(t)
	p, requests, _ := setupTestPlugin(t)
	flaky, err := llmagent.New(llmagent.Config{
		Name: "flaky",
		Model: &flakyOnFirstPassLLM{
			err:   errors.New("flaky backend unavailable"),
			reply: "succeeded on attempt 2",
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(flaky): %v", err)
	}
	delay := 25 * time.Millisecond
	node, err := workflow.NewAgentNode(flaky, workflow.NodeConfig{
		RetryConfig: &workflow.RetryConfig{
			MaxAttempts:   2,
			InitialDelay:  delay,
			MaxDelay:      delay,
			BackoffFactor: 1.0,
		},
	})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(flaky): %v", err)
	}
	wfRoot, err := workflowagent.New(workflowagent.Config{
		Name:  "wf_root",
		Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(wf_root): %v", err)
	}
	r := newTestRunner(t, wfRoot, p)
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "run"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("r.Run yielded unexpected error: %v", runErr)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want COMPLETED; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationCompleted)
	// Find the first AGENT_ERROR(flaky), the first and second
	// AGENT_STARTING(flaky), the first LLM_ERROR(flaky), and their row indices.
	var errRow, start1, start2, llmErrRow map[string]any
	errIdx, start2Idx := -1, -1
	startCount := 0
	for i, row := range rows {
		agentName, _ := row["agent"].(string)
		if agentName != "flaky" {
			continue
		}
		switch row["event_type"] {
		case EventAgentError:
			if errRow == nil {
				errRow = row
				errIdx = i
			}
		case EventAgentStarting:
			startCount++
			switch startCount {
			case 1:
				start1 = row
			case 2:
				start2 = row
				start2Idx = i
			}
		case EventLLMError:
			if llmErrRow == nil {
				llmErrRow = row
			}
		}
	}
	if errRow == nil {
		t.Fatal("no AGENT_ERROR(flaky) row found")
	}
	if start1 == nil || start2 == nil {
		t.Fatalf("missing AGENT_STARTING(flaky) rows: start1=%v start2=%v", start1 != nil, start2 != nil)
	}
	errTS, _ := errRow[timestampField].(time.Time)
	start1TS, _ := start1[timestampField].(time.Time)
	start2TS, _ := start2[timestampField].(time.Time)
	if errTS.IsZero() || start1TS.IsZero() || start2TS.IsZero() {
		t.Fatalf("AGENT_ERROR or AGENT_STARTING timestamp is zero: err=%v start1=%v start2=%v", errTS, start1TS, start2TS)
	}
	if !errTS.Before(start2TS) {
		t.Errorf("AGENT_ERROR(flaky) timestamp %v is not before the second AGENT_STARTING(flaky) timestamp %v", errTS, start2TS)
	}
	// The purged AGENT_ERROR(flaky) row must appear before the second
	// AGENT_STARTING(flaky) row in the output, pinning the write order.
	if errIdx < 0 || start2Idx < 0 || errIdx >= start2Idx {
		t.Errorf("AGENT_ERROR(flaky) row index %d is not before second AGENT_STARTING(flaky) index %d", errIdx, start2Idx)
	}
	// The purged AGENT_ERROR timestamp must not be before the LLM_ERROR
	// timestamp of the same attempt, so the agent error does not sort
	// microseconds before the model error that caused it.
	if llmErrRow != nil {
		llmErrTS, _ := llmErrRow[timestampField].(time.Time)
		if llmErrTS.IsZero() {
			t.Fatalf("LLM_ERROR(flaky) timestamp is zero")
		}
		if errTS.Before(llmErrTS) {
			t.Errorf("AGENT_ERROR(flaky) timestamp %v is before LLM_ERROR(flaky) timestamp %v", errTS, llmErrTS)
		}
	}
	// The gap from the error to the next start must include the retry backoff.
	if start2TS.Sub(errTS) < 20*time.Millisecond {
		t.Errorf("start2TS - errTS = %v, want >= 20ms (retry delay is %v)", start2TS.Sub(errTS), delay)
	}
	// The purged attempt's latency runs from start1TS (rec.start) to errTS
	// (lastActivity), excluding the retry backoff regardless of CPU load.
	lat := mustLatency(t, errRow)
	wantTotalMS := errTS.Sub(start1TS).Milliseconds()
	if diff := lat["total_ms"] - wantTotalMS; diff < -1 || diff > 1 {
		t.Errorf("AGENT_ERROR(flaky) latency total_ms = %d, want %d ±1ms (errTS %v - start1TS %v)", lat["total_ms"], wantTotalMS, errTS, start1TS)
	}
}

// TestRunner_AgentCompletedAndEndOfRunErrorTimestampsAreNotBackdated verifies
// that session-event signal rows (STATE_DELTA via eventSpan), AGENT_COMPLETED
// rows (via popAgentSpan), and end-of-run AGENT_ERROR rows (via
// popUnclosedAgentSpans) carry the clock time of their own emission rather than
// being backdated to spanRecord.lastActivity.
func TestRunner_AgentCompletedAndEndOfRunErrorTimestampsAreNotBackdated(t *testing.T) {
	t.Run("state_delta_and_agent_completed", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		const stepDelay = 25 * time.Millisecond
		root, err := agent.New(agent.Config{
			Name: "custom_ok",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					time.Sleep(stepDelay)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Actions.StateDelta = map[string]any{"step": "done"}
					if !yield(ev, nil) {
						return
					}
					time.Sleep(stepDelay)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(custom_ok): %v", err)
		}
		r := newTestRunner(t, root, p)
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("r.Run yielded unexpected error: %v", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		onlyRowOfType(t, rows, EventInvocationCompleted)
		startRow := onlyRowOfType(t, rows, EventAgentStarting)
		deltaRow := onlyRowOfType(t, rows, EventStateDelta)
		completedRow := onlyRowOfType(t, rows, EventAgentCompleted)
		startTS, _ := startRow[timestampField].(time.Time)
		deltaTS, _ := deltaRow[timestampField].(time.Time)
		completedTS, _ := completedRow[timestampField].(time.Time)
		if startTS.IsZero() || deltaTS.IsZero() || completedTS.IsZero() {
			t.Fatalf("zero timestamp in AGENT_STARTING=%v, STATE_DELTA=%v, AGENT_COMPLETED=%v", startTS, deltaTS, completedTS)
		}
		if deltaTS.Sub(startTS) < 20*time.Millisecond {
			t.Errorf("STATE_DELTA timestamp - AGENT_STARTING timestamp = %v, want >= 20ms (not backdated to AGENT_STARTING)", deltaTS.Sub(startTS))
		}
		if completedTS.Sub(deltaTS) < 20*time.Millisecond {
			t.Errorf("AGENT_COMPLETED timestamp - STATE_DELTA timestamp = %v, want >= 20ms (not backdated to lastActivity)", completedTS.Sub(deltaTS))
		}
	})

	t.Run("end_of_run_agent_error", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		const failDelay = 25 * time.Millisecond
		root, err := agent.New(agent.Config{
			Name: "custom_fail",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					time.Sleep(failDelay)
					yield(nil, errors.New("custom agent boom"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(custom_fail): %v", err)
		}
		r := newTestRunner(t, root, p)
		var gotErr error
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				gotErr = runErr
				break
			}
		}
		if gotErr == nil {
			t.Fatal("r.Run yielded nil error, want custom agent boom")
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		onlyRowOfType(t, rows, EventInvocationError)
		startRow := onlyRowOfType(t, rows, EventAgentStarting)
		errRow := onlyRowOfType(t, rows, EventAgentError)
		startTS, _ := startRow[timestampField].(time.Time)
		errTS, _ := errRow[timestampField].(time.Time)
		if startTS.IsZero() || errTS.IsZero() {
			t.Fatalf("zero timestamp in AGENT_STARTING=%v, AGENT_ERROR=%v", startTS, errTS)
		}
		if errTS.Sub(startTS) < 20*time.Millisecond {
			t.Errorf("end-of-run AGENT_ERROR timestamp - AGENT_STARTING timestamp = %v, want >= 20ms (not backdated to lastActivity)", errTS.Sub(startTS))
		}
	})
}

// TestRunner_SequentialRootWithFailedCustomChild_EmitsErrorTerminalRows
// verifies that when a sequential root runs an ok child followed by a custom
// child whose Run yields (nil, err) and no events, draining runner.Run (so
// seq_root's AfterAgentCallback runs) still preserves custom_fail_child as
// failed and ends the invocation with AGENT_ERROR and INVOCATION_ERROR.
func TestRunner_SequentialRootWithFailedCustomChild_EmitsErrorTerminalRows(t *testing.T) {
	requireADKStopOnSubAgentError(t)
	p, requests, _ := setupTestPlugin(t)
	childErr := errors.New("custom child failed")

	okChild, err := llmagent.New(llmagent.Config{
		Name:  "ok_child",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "step 1 ok"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(ok_child): %v", err)
	}

	customFailChild, err := agent.New(agent.Config{
		Name: "custom_fail_child",
		Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, childErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(custom_fail_child): %v", err)
	}

	seqRoot, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq_root",
			SubAgents: []agent.Agent{okChild, customFailChild},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(seq_root): %v", err)
	}

	r := newTestRunner(t, seqRoot, p)
	var errs []error
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			errs = append(errs, runErr)
		}
	}
	if len(errs) != 1 || !errors.Is(errs[0], childErr) {
		t.Fatalf("r.Run errors = %v, want [%v]", errs, childErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
	invErr := onlyRowOfType(t, rows, EventInvocationError)
	if invErr["status"] != "ERROR" {
		t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invErr["status"])
	}
	if got, want := agentErrorCounts(rows), map[string]int{"custom_fail_child": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR counts = %v, want %v", got, want)
	}
}

// contextCanceledOnDoneLLM signals entered when GenerateContent starts, blocks
// until ctx is canceled, and then yields ctx.Err() as a real streaming/API
// model does when parallelagent cancels subAgentsCtx on consumer break.
type contextCanceledOnDoneLLM struct {
	entered chan struct{}
	once    sync.Once
}

func (m *contextCanceledOnDoneLLM) Name() string { return "context-canceled-on-done-llm" }

func (m *contextCanceledOnDoneLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.once.Do(func() { close(m.entered) })
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

// TestRunner_ParallelRootStoppedMidStream_SiblingModelReturnsContextCanceled_EmitsCompletedTerminalRows
// verifies that when a caller breaks out of runner.Run mid-stream on a partial
// event from one branch of a parallelagent and parallelagent's deferred
// subAgentsCtx cancellation causes a sibling branch's model to return
// context.Canceled (triggering onModelError), the invocation still terminates
// with INVOCATION_COMPLETED rather than INVOCATION_ERROR.
func TestRunner_ParallelRootStoppedMidStream_SiblingModelReturnsContextCanceled_EmitsCompletedTerminalRows(t *testing.T) {
	requireADKParallelEarlyStopTeardown(t)
	p, requests, _ := setupTestPlugin(t)
	slowEntered := make(chan struct{})

	slow, err := llmagent.New(llmagent.Config{
		Name:  "slow",
		Model: &contextCanceledOnDoneLLM{entered: slowEntered},
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
		t.Fatalf("parallelagent.New(par_root): %v", err)
	}

	r := newTestRunner(t, par, p)
	var sawPartial bool
	for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	}) {
		if runErr != nil {
			t.Fatalf("r.Run error: %v", runErr)
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
	if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
		t.Errorf("got %d LLM_ERROR rows, want 1 (from slow branch context.Canceled)", n)
	}
}

// TestRunner_ParallelStreamStop_CanceledWorkAfterPartialIsNotAnError verifies
// that a caller that stops reading at a partial event of a parallel branch
// stops mid-stream, not at a failure, when the branch then starts work that
// ends on the branch's cancellation. A parallel agent lets the branch go on
// once the event was yielded, even when the caller stopped reading on it, and
// then cancels its branches.
func TestRunner_ParallelStreamStop_CanceledWorkAfterPartialIsNotAnError(t *testing.T) {
	requireADKParallelEarlyStopTeardown(t)
	for _, tc := range []struct {
		name     string
		newChild func(t *testing.T) agent.Agent
	}{
		{
			name: "sub-agent that fails with the context's error",
			newChild: func(t *testing.T) agent.Agent {
				a, err := agent.New(agent.Config{
					Name: "child",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							<-ctx.Done()
							yield(nil, ctx.Err())
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(child): %v", err)
				}
				return a
			},
		},
		{
			name: "model call that fails with the context's error",
			newChild: func(t *testing.T) agent.Agent {
				a, err := llmagent.New(llmagent.Config{
					Name:  "child",
					Model: &contextCanceledOnDoneLLM{entered: make(chan struct{})},
				})
				if err != nil {
					t.Fatalf("llmagent.New(child): %v", err)
				}
				return a
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requireChildStartedAfterPartialStop(t, func(t *testing.T) []map[string]any {
				p, requests, _ := setupTestPlugin(t)
				child := tc.newChild(t)
				streamer, err := agent.New(agent.Config{
					Name:      "streamer",
					SubAgents: []agent.Agent{child},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							partial := session.NewEvent(ctx, ctx.InvocationID())
							partial.Author = "streamer"
							partial.Branch = ctx.Branch()
							partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "progress update...")}
							if !yield(partial, nil) {
								return
							}
							for ev, err := range child.Run(ctx) {
								if !yield(ev, err) || err != nil {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(streamer): %v", err)
				}
				par, err := parallelagent.New(parallelagent.Config{
					AgentConfig: agent.Config{
						Name:      "par_root",
						SubAgents: []agent.Agent{streamer},
					},
				})
				if err != nil {
					t.Fatalf("parallelagent.New(par_root): %v", err)
				}

				r := newTestRunner(t, par, p)
				var sawPartial bool
				for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
					StreamingMode: agent.StreamingModeSSE,
				}) {
					if runErr != nil {
						t.Fatalf("r.Run error: %v", runErr)
					}
					if ev != nil && ev.Partial {
						sawPartial = true
						break
					}
				}
				if !sawPartial {
					t.Fatal("did not observe a partial event in StreamingModeSSE")
				}
				return collectRows(t, requests, untilInvocationEnd)
			})
		})
	}
}

// newStreamingAuthor returns a custom agent named author that writes one
// partial event, with or without content, and then runs child.
func newStreamingAuthor(t *testing.T, child agent.Agent, withContent bool) agent.Agent {
	t.Helper()
	author, err := agent.New(agent.Config{
		Name:      "author",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				partial := session.NewEvent(ctx, ctx.InvocationID())
				partial.Author = "author"
				partial.Branch = ctx.Branch()
				partial.LLMResponse = model.LLMResponse{Partial: true}
				if withContent {
					partial.Content = textContent("model", "progress...")
				}
				if !yield(partial, nil) {
					return
				}
				for ev, err := range child.Run(ctx) {
					if !yield(ev, err) || err != nil {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(author): %v", err)
	}
	return author
}

// TestRunner_StopOnPartialEventOfAsyncAuthor_IsNotAnError verifies that a
// caller that stops reading on the partial event of an agent that goes on
// without waiting for the caller stops mid-stream, not at a failure: the agent
// then runs its child, which the run cancels, under a parallel agent, whose
// sibling waits for its cancellation, and as a node of a workflow, which does
// not wait for its caller to read a partial event.
func TestRunner_StopOnPartialEventOfAsyncAuthor_IsNotAnError(t *testing.T) {
	newChild := func(t *testing.T) agent.Agent {
		t.Helper()
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					answer := session.NewEvent(ctx, ctx.InvocationID())
					answer.Author = "child"
					answer.Branch = ctx.Branch()
					answer.Content = textContent("model", "child answer")
					yield(answer, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		return child
	}
	newParallel := func(t *testing.T, withContent bool) agent.Agent {
		t.Helper()
		waiter, err := agent.New(agent.Config{
			Name: "waiter",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					<-ctx.Done()
					yield(nil, ctx.Err())
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(waiter): %v", err)
		}
		par, err := parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{
				Name:      "par_root",
				SubAgents: []agent.Agent{newStreamingAuthor(t, newChild(t), withContent), waiter},
			},
		})
		if err != nil {
			t.Fatalf("parallelagent.New(par_root): %v", err)
		}
		return par
	}
	for _, tc := range []struct {
		name    string
		newRoot func(t *testing.T) agent.Agent
	}{
		{name: "parallel agent, partial event with content", newRoot: func(t *testing.T) agent.Agent { return newParallel(t, true) }},
		{name: "parallel agent, partial event without content", newRoot: func(t *testing.T) agent.Agent { return newParallel(t, false) }},
		{name: "workflow node", newRoot: func(t *testing.T) agent.Agent {
			node, err := workflow.NewAgentNode(newStreamingAuthor(t, newChild(t), true), workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode(author): %v", err)
			}
			wf, err := workflowagent.New(workflowagent.Config{
				Name:  "wf_root",
				Edges: []workflow.Edge{{From: workflow.Start, To: node}},
			})
			if err != nil {
				t.Fatalf("workflowagent.New(wf_root): %v", err)
			}
			return wf
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name != "workflow node" {
				requireADKParallelEarlyStopTeardown(t)
			}
			requireChildStartedAfterPartialStop(t, func(t *testing.T) []map[string]any {
				p, requests, _ := setupTestPlugin(t)
				r := newTestRunner(t, tc.newRoot(t), p)
				var sawPartial bool
				for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("r.Run error: %v", err)
					}
					if ev != nil && ev.Partial {
						sawPartial = true
						break
					}
				}
				if !sawPartial {
					t.Fatal("runner.Run yielded no partial event")
				}
				return collectRows(t, requests, untilInvocationEnd)
			})
		})
	}
}

// selfCancelingStreamLLM streams one partial chunk and then cancels a request
// context of its own and fails with its error, as a model client does that
// gives up on a request.
type selfCancelingStreamLLM struct{}

func (m *selfCancelingStreamLLM) Name() string { return "self-canceling-stream-llm" }

func (m *selfCancelingStreamLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		requestCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		if !yield(&model.LLMResponse{Content: textContent("model", "hello "), Partial: true}, nil) {
			return
		}
		cancel()
		yield(nil, requestCtx.Err())
	}
}

// TestRunner_ModelCanceledOwnRequestAfterPartial_EmitsErrorTerminalRows
// verifies that a model call that canceled a request context of its own after
// it streamed a chunk fails the invocation when the caller reads to the end:
// only a cancellation of the work by the run, for a caller that stopped
// reading, is not a failure.
func TestRunner_ModelCanceledOwnRequestAfterPartial_EmitsErrorTerminalRows(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	bot, err := llmagent.New(llmagent.Config{Name: "bot", Model: &selfCancelingStreamLLM{}})
	if err != nil {
		t.Fatalf("llmagent.New(bot): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{bot}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(seq_root): %v", err)
	}

	r := newTestRunner(t, seq, p)
	var sawPartial bool
	var gotErr error
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "stream please"), agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	}) {
		if err != nil {
			gotErr = err
		}
		if ev != nil && ev.Partial {
			sawPartial = true
		}
	}
	if !sawPartial || !errors.Is(gotErr, context.Canceled) {
		t.Fatalf("runner.Run yielded partial event %v and error %v, want a partial event and context.Canceled", sawPartial, gotErr)
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationError)
}

// TestRunner_AgentCanceledSubAgentContextAfterPartial_EmitsErrorTerminalRows
// verifies that a partial event does not hide a sub-agent whose context the
// agent that runs it canceled itself, as on a step timeout of its own, while
// the caller reads on: the sub-agent's model ends the call with
// context.Canceled and the run yields it, whether the custom agent that writes
// the partial event and cancels the sub-agent's context is the root or runs
// under a sequential or a parallel agent.
func TestRunner_AgentCanceledSubAgentContextAfterPartial_EmitsErrorTerminalRows(t *testing.T) {
	for _, wrap := range []string{"root", "sequential", "parallel"} {
		t.Run(wrap, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			entered := make(chan struct{})
			subBot, err := llmagent.New(llmagent.Config{
				Name:  "sub_bot",
				Model: &contextCanceledOnDoneLLM{entered: entered},
			})
			if err != nil {
				t.Fatalf("llmagent.New(sub_bot): %v", err)
			}
			custom, err := agent.New(agent.Config{
				Name:      "custom",
				SubAgents: []agent.Agent{subBot},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						partial := session.NewEvent(ctx, ctx.InvocationID())
						partial.Author = "custom"
						partial.Branch = ctx.Branch()
						partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "progress update...")}
						if !yield(partial, nil) {
							return
						}
						subCtx, cancel := context.WithCancel(ctx)
						defer cancel()
						go func() {
							<-entered
							cancel()
						}()
						for ev, err := range subBot.Run(ctx.WithContext(subCtx)) {
							if !yield(ev, err) || err != nil {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(custom): %v", err)
			}
			root := custom
			switch wrap {
			case "sequential":
				root, err = sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{custom}},
				})
			case "parallel":
				root, err = parallelagent.New(parallelagent.Config{
					AgentConfig: agent.Config{Name: "par_root", SubAgents: []agent.Agent{custom}},
				})
			}
			if err != nil {
				t.Fatalf("New(%s root): %v", wrap, err)
			}

			r := newTestRunner(t, root, p)
			var sawPartial bool
			var gotErr error
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					gotErr = err
				}
				if ev != nil && ev.Partial {
					sawPartial = true
				}
			}
			if !sawPartial || !errors.Is(gotErr, context.Canceled) {
				t.Fatalf("runner.Run yielded partial event %v and error %v, want a partial event and context.Canceled", sawPartial, gotErr)
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			onlyRowOfType(t, rows, EventInvocationError)
			if got := agentErrorCounts(rows); got["sub_bot"] != 1 {
				t.Errorf("AGENT_ERROR rows per agent = %v, want one for sub_bot", got)
			}
		})
	}
}

// streamThenWaitLLM streams one partial chunk, optionally closes acked once
// the partial chunk's yield returned true, and then blocks until its context is
// canceled, as a streaming model does while a parallel agent cancels the branch
// that it runs in because a sibling failed.
type streamThenWaitLLM struct {
	acked chan struct{}
}

func (streamThenWaitLLM) Name() string { return "stream-then-wait-llm" }

func (m streamThenWaitLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(&model.LLMResponse{Content: textContent("model", "streaming..."), Partial: true}, nil) {
			return
		}
		if m.acked != nil {
			close(m.acked)
		}
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

// contentlessPartialLLM streams one partial chunk with no content, then blocks
// until ctx.Done. It pins the canceledByStop exemption for an unresponded
// activation that the consumer stop canceled.
type contentlessPartialLLM struct{}

func (contentlessPartialLLM) Name() string { return "contentless-partial-llm" }

func (contentlessPartialLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(&model.LLMResponse{Partial: true, ErrorCode: "PROBE"}, nil) {
			return
		}
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

// streamThenErrorLLM streams one partial response, then returns an error.
type streamThenErrorLLM struct {
	partial *model.LLMResponse
	err     error
}

func (m *streamThenErrorLLM) Name() string { return "stream-then-error-llm" }

func (m *streamThenErrorLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.partial != nil {
			if !yield(m.partial, nil) {
				return
			}
		}
		yield(nil, m.err)
	}
}

// errSiblingFailed is the failure of the branch that fails while its sibling
// streams.
var errSiblingFailed = errors.New("guardrail rejected the request")

// newGuardedAgent returns an llmagent named failer that waits for seen to be
// closed, and then fails with errSiblingFailed from its BeforeModelCallback,
// which no plugin callback of the analytics plugin observes.
func newGuardedAgent(t *testing.T, seen <-chan struct{}) agent.Agent {
	t.Helper()
	failer, err := llmagent.New(llmagent.Config{
		Name:  "failer",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "unused"), TurnComplete: true}}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
			<-seen
			return nil, errSiblingFailed
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(failer): %v", err)
	}
	return failer
}

// newCustomFailer returns a custom agent named failer that waits for seen to
// be closed, and then yields errSiblingFailed.
func newCustomFailer(t *testing.T, seen <-chan struct{}) agent.Agent {
	t.Helper()
	failer, err := agent.New(agent.Config{
		Name: "failer",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-seen
				yield(nil, errSiblingFailed)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failer): %v", err)
	}
	return failer
}

// newStreamingLLMAgent returns an llmagent named streamer whose model streams
// a chunk, optionally closes acked once the partial chunk's yield returned
// true, and then waits for its context to be canceled.
func newStreamingLLMAgent(t *testing.T, acked chan struct{}) agent.Agent {
	t.Helper()
	streamer, err := llmagent.New(llmagent.Config{Name: "streamer", Model: streamThenWaitLLM{acked: acked}})
	if err != nil {
		t.Fatalf("llmagent.New(streamer): %v", err)
	}
	return streamer
}

// newCustomStreamingAgent returns a custom agent named streamer that yields one
// partial event with content, optionally closes acked once yield returned true,
// and then waits for its context to be canceled without calling any model.
func newCustomStreamingAgent(t *testing.T, acked chan struct{}) agent.Agent {
	t.Helper()
	streamer, err := agent.New(agent.Config{
		Name: "streamer",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				partial := session.NewEvent(ctx, ctx.InvocationID())
				partial.Author = "streamer"
				partial.Branch = ctx.Branch()
				partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}
				if !yield(partial, nil) {
					return
				}
				if acked != nil {
					close(acked)
				}
				<-ctx.Done()
				yield(nil, ctx.Err())
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(streamer): %v", err)
	}
	return streamer
}

// beforeAckStreamLLM streams one partial chunk and closes exited when its
// GenerateContent iterator returns. When an OnEventCallback holds the partial
// event's acknowledgement until a failing sibling cancels the parallel
// errgroup, yield returns false and GenerateContent returns without calling
// yield(nil, ctx.Err()), so no OnModelErrorCallback runs.
type beforeAckStreamLLM struct {
	exited chan struct{}
}

func (*beforeAckStreamLLM) Name() string { return "before-ack-stream-llm" }

func (m *beforeAckStreamLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		defer close(m.exited)
		if !yield(&model.LLMResponse{Content: textContent("model", "streaming..."), Partial: true}, nil) {
			return
		}
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

// newParallelAgent returns a parallel agent named name that runs branches.
func newParallelAgent(t *testing.T, name string, branches ...agent.Agent) agent.Agent {
	t.Helper()
	par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: branches}})
	if err != nil {
		t.Fatalf("parallelagent.New(%s): %v", name, err)
	}
	return par
}

// runUntilBranchFails runs the agent that newRoot returns, whose branch of a
// parallel agent fails once the first partial event of a sibling branch went
// through the plugins and its producer resumed past the event acknowledgement,
// and returns the rows of the run and the last error that the caller received.
// Waiting for both the observer plugin's OnEventCallback (seen) and the
// sibling's post-acknowledgement signal (acked) keeps the post-ack cases
// deterministic both when parallelagent yields directly to runner.Run and when
// workflow.Scheduler queues partial events asynchronously under an LLM root.
// The caller stops reading at the first error if stopAtError is set, and reads
// on until the end otherwise.
func runUntilBranchFails(t *testing.T, newRoot func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent, cfg agent.RunConfig, stopAtError bool) ([]map[string]any, error) {
	t.Helper()
	ctx := t.Context()
	p, requests, _ := setupTestPlugin(t)
	seen := make(chan struct{})
	acked := make(chan struct{})
	ready := make(chan struct{})
	go func() {
		select {
		case <-seen:
		case <-ctx.Done():
			return
		}
		select {
		case <-acked:
			close(ready)
		case <-ctx.Done():
		}
	}()
	var once sync.Once
	observer, err := baseplugin.New(baseplugin.Config{
		Name: "observer",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev != nil && ev.Partial {
				once.Do(func() { close(seen) })
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(observer): %v", err)
	}
	var gotErr error
	for _, err := range newTestRunner(t, newRoot(t, ready, acked), p, observer).Run(ctx, "u1", "s1", textContent("user", "go"), cfg) {
		if err != nil {
			gotErr = err
			if stopAtError {
				break
			}
		}
	}
	if !errors.Is(gotErr, errSiblingFailed) {
		t.Fatalf("runner.Run error = %v, want %v", gotErr, errSiblingFailed)
	}
	return collectRows(t, requests, untilInvocationEnd), gotErr
}

// branchFailureCases are runs in which a branch of a parallel agent fails
// while a sibling branch streams after its partial event was acknowledged: the
// last session event is the sibling's partial event, and the parallel agent's
// errgroup cancels the sibling's work with the failing branch's error as the
// cause.
var branchFailureCases = []struct {
	name              string
	cfg               agent.RunConfig
	wantErrorAgent    string
	wantAnyErrorAgent []string
	noLLMError        bool
	newRoot           func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent
}{
	{
		name:           "parallel root, model callback error",
		cfg:            agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
		wantErrorAgent: "failer",
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			return newParallelAgent(t, "par_root", newStreamingLLMAgent(t, acked), newGuardedAgent(t, ready))
		},
	},
	{
		name:           "parallel root, custom agent error",
		cfg:            agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
		wantErrorAgent: "failer",
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			return newParallelAgent(t, "par_root", newStreamingLLMAgent(t, acked), newCustomFailer(t, ready))
		},
	},
	{
		name:           "parallel agent under a sequential root",
		cfg:            agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
		wantErrorAgent: "failer",
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			par := newParallelAgent(t, "par", newStreamingLLMAgent(t, acked), newGuardedAgent(t, ready))
			seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{par}}})
			if err != nil {
				t.Fatalf("sequentialagent.New(seq_root): %v", err)
			}
			return seq
		},
	},
	{
		name:              "progress event without content, no streaming",
		wantAnyErrorAgent: []string{"failer", "worker"},
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			worker, err := llmagent.New(llmagent.Config{Name: "worker", Model: &contextCanceledOnDoneLLM{entered: acked}})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			return newParallelAgent(t, "par_root", newStreamingAuthor(t, worker, false), newCustomFailer(t, ready))
		},
	},
	{
		name:           "parallel agent that an LLM root transfers to",
		cfg:            agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
		wantErrorAgent: "failer",
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			par := newParallelAgent(t, "par", newStreamingLLMAgent(t, acked), newGuardedAgent(t, ready))
			coord, err := llmagent.New(llmagent.Config{
				Name: "coord",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "call-transfer", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "par"}},
				}}}}}},
				SubAgents: []agent.Agent{par},
			})
			if err != nil {
				t.Fatalf("llmagent.New(coord): %v", err)
			}
			return coord
		},
	},
	{
		name:           "parallel root, custom streaming sibling",
		wantErrorAgent: "failer",
		noLLMError:     true,
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			return newParallelAgent(t, "par_root", newCustomStreamingAgent(t, acked), newCustomFailer(t, ready))
		},
	},
	{
		name:           "LLM root with custom streaming sibling",
		wantErrorAgent: "failer",
		noLLMError:     true,
		newRoot: func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
			par := newParallelAgent(t, "par", newCustomStreamingAgent(t, acked), newGuardedAgent(t, ready))
			coord, err := llmagent.New(llmagent.Config{
				Name: "coord",
				Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "call-transfer", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "par"}},
				}}}}}},
				SubAgents: []agent.Agent{par},
			})
			if err != nil {
				t.Fatalf("llmagent.New(coord): %v", err)
			}
			return coord
		},
	},
}

// assertTerminalErrorRowsContain verifies that INVOCATION_ERROR and every
// AGENT_ERROR row carry wantSubstr in their error_message column.
func assertTerminalErrorRowsContain(t *testing.T, rows []map[string]any, wantSubstr string) {
	t.Helper()
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if msg, _ := invRow["error_message"].(string); !strings.Contains(msg, wantSubstr) {
		t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, wantSubstr)
	}
	for _, r := range rowsOfType(rows, EventAgentError) {
		if msg, _ := r["error_message"].(string); !strings.Contains(msg, wantSubstr) {
			t.Errorf("AGENT_ERROR(%v) error_message = %q, want substring %q", r["agent"], msg, wantSubstr)
		}
	}
}

// TestRunner_ParallelBranchFailsWhileSiblingStreams_EmitsErrorTerminalRows
// verifies that a branch of a parallel agent that fails without any callback
// observing the failure, while a sibling branch streams, fails the invocation
// when the caller reads on: the caller received the error, and ADK ran the
// AfterAgentCallback of the agent that the runner runs, which it does only if
// the caller read every event, so the work that the parallel agent canceled
// does not stand for a caller that stopped reading mid-stream.
func TestRunner_ParallelBranchFailsWhileSiblingStreams_EmitsErrorTerminalRows(t *testing.T) {
	for _, tc := range branchFailureCases {
		t.Run(tc.name, func(t *testing.T) {
			rows, gotErr := runUntilBranchFails(t, tc.newRoot, tc.cfg, false)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got INVOCATION_COMPLETED although the caller received %q; event types: %v", gotErr, eventTypes(rows))
			}
			assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
			errCounts := agentErrorCounts(rows)
			if len(tc.wantAnyErrorAgent) > 0 {
				total := 0
				for _, name := range tc.wantAnyErrorAgent {
					total += errCounts[name]
				}
				if total != 1 {
					t.Errorf("AGENT_ERROR counts = %v, want total 1 across %v; event types: %v", errCounts, tc.wantAnyErrorAgent, eventTypes(rows))
				}
			} else if errCounts[tc.wantErrorAgent] != 1 {
				t.Errorf("AGENT_ERROR counts = %v, want 1 for %q; event types: %v", errCounts, tc.wantErrorAgent, eventTypes(rows))
			}
			if tc.wantErrorAgent == "failer" && errCounts["streamer"] != 0 {
				t.Errorf("AGENT_ERROR counts = %v, want 0 for streamer", errCounts)
			}
			if tc.noLLMError {
				if n := len(rowsOfType(rows, EventLLMError)); n != 0 {
					t.Errorf("got %d LLM_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
				}
			}
		})
	}
}

// TestRunner_ParallelBranchFailsWhileSiblingStreams_StopAtErrorEmitsErrorTerminalRows
// verifies that a caller that stops reading at the failure of the branch
// still gets INVOCATION_ERROR: the parallel agent's errgroup cancels the
// sibling branch with the failing branch's error as the cause, so
// context.Cause(ctx) is not context.Canceled, distinguishing the sibling
// failure from a caller that stopped reading mid-stream.
func TestRunner_ParallelBranchFailsWhileSiblingStreams_StopAtErrorEmitsErrorTerminalRows(t *testing.T) {
	for _, tc := range branchFailureCases {
		t.Run(tc.name, func(t *testing.T) {
			rows, _ := runUntilBranchFails(t, tc.newRoot, tc.cfg, true)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
			}
			assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
			errCounts := agentErrorCounts(rows)
			if len(tc.wantAnyErrorAgent) > 0 {
				total := 0
				for _, name := range tc.wantAnyErrorAgent {
					total += errCounts[name]
				}
				if total != 1 {
					t.Errorf("AGENT_ERROR counts = %v, want total 1 across %v; event types: %v", errCounts, tc.wantAnyErrorAgent, eventTypes(rows))
				}
			} else if errCounts[tc.wantErrorAgent] != 1 {
				t.Errorf("AGENT_ERROR counts = %v, want 1 for %q; event types: %v", errCounts, tc.wantErrorAgent, eventTypes(rows))
			}
			if tc.wantErrorAgent == "failer" && errCounts["streamer"] != 0 {
				t.Errorf("AGENT_ERROR counts = %v, want 0 for streamer", errCounts)
			}
			if tc.noLLMError {
				if n := len(rowsOfType(rows, EventLLMError)); n != 0 {
					t.Errorf("got %d LLM_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
				}
			}
		})
	}
}

// TestRunner_ParallelBranchFailsBeforePartialAck_EmitsErrorTerminalRows
// verifies that when a sibling branch fails while the streaming branch is still
// waiting for parallelagent to acknowledge its partial event (so errgroup
// cancellation makes parallelagent.runSubAgent exit via ctx.Done before closing
// ackChan, yield returns false in GenerateContent, and no OnModelErrorCallback
// runs), openFailureCauseLocked still detects the non-context.Canceled errgroup
// cause and emits AGENT_ERROR(failer) and INVOCATION_ERROR for both draining
// and stopping consumers.
func TestRunner_ParallelBranchFailsBeforePartialAck_EmitsErrorTerminalRows(t *testing.T) {
	cases := []struct {
		name    string
		newRoot func(t *testing.T, seen <-chan struct{}, exited chan struct{}) agent.Agent
	}{
		{
			name: "parallel root, model callback error",
			newRoot: func(t *testing.T, seen <-chan struct{}, exited chan struct{}) agent.Agent {
				streamer, err := llmagent.New(llmagent.Config{Name: "streamer", Model: &beforeAckStreamLLM{exited: exited}})
				if err != nil {
					t.Fatalf("llmagent.New(streamer): %v", err)
				}
				return newParallelAgent(t, "par_root", streamer, newGuardedAgent(t, seen))
			},
		},
		{
			name: "parallel root, custom agent error",
			newRoot: func(t *testing.T, seen <-chan struct{}, exited chan struct{}) agent.Agent {
				streamer, err := llmagent.New(llmagent.Config{Name: "streamer", Model: &beforeAckStreamLLM{exited: exited}})
				if err != nil {
					t.Fatalf("llmagent.New(streamer): %v", err)
				}
				return newParallelAgent(t, "par_root", streamer, newCustomFailer(t, seen))
			},
		},
		{
			name: "parallel agent under a sequential root",
			newRoot: func(t *testing.T, seen <-chan struct{}, exited chan struct{}) agent.Agent {
				streamer, err := llmagent.New(llmagent.Config{Name: "streamer", Model: &beforeAckStreamLLM{exited: exited}})
				if err != nil {
					t.Fatalf("llmagent.New(streamer): %v", err)
				}
				par := newParallelAgent(t, "par", streamer, newGuardedAgent(t, seen))
				seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{par}}})
				if err != nil {
					t.Fatalf("sequentialagent.New(seq_root): %v", err)
				}
				return seq
			},
		},
	}

	for _, stopAtError := range []bool{false, true} {
		mode := "drain"
		if stopAtError {
			mode = "stop_at_error"
		}
		t.Run(mode, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					p, requests, _ := setupTestPlugin(t)
					seen := make(chan struct{})
					exited := make(chan struct{})
					var once sync.Once
					observer, err := baseplugin.New(baseplugin.Config{
						Name: "observer",
						OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
							if ev != nil && ev.Partial {
								once.Do(func() {
									close(seen)
									select {
									case <-exited:
									case <-time.After(5 * time.Second):
										t.Error("timed out waiting for streaming producer to exit before partial ack")
									}
								})
							}
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New(observer): %v", err)
					}

					var gotErr error
					for _, err := range newTestRunner(t, tc.newRoot(t, seen, exited), p, observer).Run(
						t.Context(),
						"u1",
						"s1",
						textContent("user", "go"),
						agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
					) {
						if err != nil {
							gotErr = err
							if stopAtError {
								break
							}
						}
					}
					if !errors.Is(gotErr, errSiblingFailed) {
						t.Fatalf("runner.Run error = %v, want %v", gotErr, errSiblingFailed)
					}

					rows := collectRows(t, requests, untilInvocationEnd)
					if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
						t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
					}
					assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
					if n := len(rowsOfType(rows, EventLLMError)); n != 0 {
						t.Errorf("got %d LLM_ERROR rows, want 0 (canceled before partial ack); event types: %v", n, eventTypes(rows))
					}
					errCounts := agentErrorCounts(rows)
					if errCounts["failer"] != 1 || errCounts["streamer"] != 0 {
						t.Errorf("AGENT_ERROR counts = %v, want failer=1 and streamer=0; event types: %v", errCounts, eventTypes(rows))
					}
				})
			}
		})
	}
}

// waitDoneStreamLLM waits until waitCtx returns a done context (so an earlier
// workflow node's WithAgentTimeout has already expired with
// context.DeadlineExceeded) before yielding a partial chunk and a final
// response.
type waitDoneStreamLLM struct {
	waitCtx func() context.Context
}

func (*waitDoneStreamLLM) Name() string { return "wait-done-stream-llm" }

func (m *waitDoneStreamLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		select {
		case <-m.waitCtx().Done():
		case <-ctx.Done():
			return
		}
		if !yield(&model.LLMResponse{Content: textContent("model", "streaming..."), Partial: true}, nil) {
			return
		}
		yield(&model.LLMResponse{Content: textContent("model", "final answer"), TurnComplete: true}, nil)
	}
}

// TestRunner_WorkflowNodeTimeoutAfterBeforeAgentShortcut_ConsumerStopCompletes
// verifies that an expired NodeConfig.Timeout on an earlier workflow node that
// already completed via BeforeAgentCallback (leaving its responded shortcut
// span open on an uncanceled WithAgentTimeout context) does not turn a later
// consumer stop on a partial or final event, or a drained run, into
// INVOCATION_ERROR "context deadline exceeded".
func TestRunner_WorkflowNodeTimeoutAfterBeforeAgentShortcut_ConsumerStopCompletes(t *testing.T) {
	for _, callerMode := range []string{"stop_on_partial", "stop_on_final", "drain"} {
		t.Run(callerMode, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var cachedCtx context.Context
			cached, err := llmagent.New(llmagent.Config{
				Name:  "cached",
				Model: &scriptedLLM{},
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(cbCtx agent.Context) (*genai.Content, error) {
					cachedCtx = cbCtx
					return textContent("model", "cached answer"), nil
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(cached): %v", err)
			}
			cachedNode, err := workflow.NewAgentNode(cached, workflow.NodeConfig{Timeout: 300 * time.Millisecond})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode(cached): %v", err)
			}
			writer, err := llmagent.New(llmagent.Config{
				Name:  "writer",
				Model: &waitDoneStreamLLM{waitCtx: func() context.Context { return cachedCtx }},
			})
			if err != nil {
				t.Fatalf("llmagent.New(writer): %v", err)
			}
			writerNode, err := workflow.NewAgentNode(writer, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode(writer): %v", err)
			}
			wfRoot, err := workflowagent.New(workflowagent.Config{
				Name: "wf_root",
				Edges: []workflow.Edge{
					{From: workflow.Start, To: cachedNode},
					{From: cachedNode, To: writerNode},
				},
			})
			if err != nil {
				t.Fatalf("workflowagent.New(wf_root): %v", err)
			}

			r := newTestRunner(t, wfRoot, p)
			for ev, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{
				StreamingMode: agent.StreamingModeSSE,
			}) {
				if runErr != nil {
					t.Fatalf("r.Run yielded unexpected error: %v", runErr)
				}
				if ev == nil || ev.Author != "writer" {
					continue
				}
				if callerMode == "stop_on_partial" && ev.Partial {
					break
				}
				if callerMode == "stop_on_final" && !ev.Partial {
					break
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; rows: %v", n, rowsOfType(rows, EventInvocationError))
			}
			onlyRowOfType(t, rows, EventInvocationCompleted)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
			}
		})
	}
}

// TestRunner_WorkflowNodeFailureWhileSiblingStreams_EmitsInvocationError
// verifies that when a workflow node (failer) returns an error while a
// sibling node (streamer) is streaming a partial event, the invocation
// still ends with INVOCATION_ERROR and the failing node (and wf_root) carry
// AGENT_ERROR, even on ADK releases before workflow node completion events
// (GOWORK=off against released ADK v2.0.0–v2.6.0, where
// adk.workflow.node_outcome is not emitted).
//
// On those releases, the workflow scheduler's cancelAll on consumer stop
// cancels every running node with context.Canceled (canceledByStop holds),
// but a node that returns an error is removed from runCancels before
// cancelAll, leaving the failing node's context uncanceled
// (!canceledByStop(failer)), not in partialLineage (!stoppedOn), and
// unresponded (!responded()). The inWorkflowNodeSubtreeLocked check lets
// popUnclosedAgentSpans find the failing node's open call even while
// lastEventPartial is still set without a siblingCause.
func TestRunner_WorkflowNodeFailureWhileSiblingStreams_EmitsInvocationError(t *testing.T) {
	ctx := t.Context()

	makeStreamer := func(t *testing.T, ready <-chan struct{}) agent.Agent {
		t.Helper()
		a, err := agent.New(agent.Config{
			Name: "streamer",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					partial := session.NewEvent(ic, ic.InvocationID())
					partial.Author, partial.Branch = "streamer", ic.Branch()
					partial.LLMResponse = model.LLMResponse{Partial: true}
					partial.Content = textContent("model", "streaming...")
					yield(partial, nil)
					// Keep running until the caller stops or the workflow cancels us.
					<-ic.Done()
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(streamer): %v", err)
		}
		return a
	}
	makeFailer := func(t *testing.T, ready <-chan struct{}) agent.Agent {
		t.Helper()
		a, err := agent.New(agent.Config{
			Name: "failer",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					// Wait for streamer to have emitted its partial event.
					select {
					case <-ready:
					case <-ic.Done():
						return
					}
					yield(nil, errors.New("node boom"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(failer): %v", err)
		}
		return a
	}

	t.Run("2-node workflow", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		ready := make(chan struct{})
		streamer := makeStreamer(t, ready)
		failer := makeFailer(t, ready)
		streamerNode, err := workflow.NewAgentNode(streamer, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(streamer): %v", err)
		}
		failerNode, err := workflow.NewAgentNode(failer, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(failer): %v", err)
		}
		wf, err := workflowagent.New(workflowagent.Config{
			Name: "wf_root",
			Edges: []workflow.Edge{
				{From: workflow.Start, To: streamerNode},
				{From: workflow.Start, To: failerNode},
			},
		})
		if err != nil {
			t.Fatalf("workflowagent.New(wf_root): %v", err)
		}
		r := newTestRunner(t, wf, p)
		var sawPartial bool
		for ev, runErr := range r.Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				break
			}
			if ev != nil && ev.Partial && ev.Author == "streamer" {
				sawPartial = true
				close(ready)
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event from streamer")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		// The invocation should end with INVOCATION_ERROR.
		if n := len(rowsOfType(rows, EventInvocationError)); n != 1 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 1; event types: %v", n, eventTypes(rows))
		}
		// The failer should have an AGENT_ERROR row.
		if n := agentErrorCounts(rows)["failer"]; n != 1 {
			t.Errorf("got %d AGENT_ERROR rows for failer, want 1; event types: %v", n, eventTypes(rows))
		}
	})

	t.Run("with cached shortcut node", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		ready := make(chan struct{})
		streamer := makeStreamer(t, ready)
		failer := makeFailer(t, ready)
		cached, err := llmagent.New(llmagent.Config{
			Name:  "cached",
			Model: &scriptedLLM{},
			BeforeAgentCallbacks: []agent.BeforeAgentCallback{
				func(agent.Context) (*genai.Content, error) { return textContent("model", "cached answer"), nil },
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(cached): %v", err)
		}
		streamerNode, err := workflow.NewAgentNode(streamer, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(streamer): %v", err)
		}
		failerNode, err := workflow.NewAgentNode(failer, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(failer): %v", err)
		}
		cachedNode, err := workflow.NewAgentNode(cached, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(cached): %v", err)
		}
		wf, err := workflowagent.New(workflowagent.Config{
			Name: "wf_root",
			Edges: []workflow.Edge{
				{From: workflow.Start, To: streamerNode},
				{From: workflow.Start, To: failerNode},
				{From: workflow.Start, To: cachedNode},
			},
		})
		if err != nil {
			t.Fatalf("workflowagent.New(wf_root): %v", err)
		}
		r := newTestRunner(t, wf, p)
		var sawPartial bool
		for ev, runErr := range r.Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				break
			}
			if ev != nil && ev.Partial && ev.Author == "streamer" {
				sawPartial = true
				close(ready)
			}
		}
		if !sawPartial {
			t.Fatal("did not observe a partial event from streamer")
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 1 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 1; event types: %v", n, eventTypes(rows))
		}
		if n := agentErrorCounts(rows)["failer"]; n != 1 {
			t.Errorf("got %d AGENT_ERROR rows for failer, want 1; event types: %v", n, eventTypes(rows))
		}
		// The cached shortcut node should NOT be attributed AGENT_ERROR.
		if n := agentErrorCounts(rows)["cached"]; n != 0 {
			t.Errorf("got %d AGENT_ERROR rows for cached, want 0; event types: %v", n, eventTypes(rows))
		}
	})
}

// TestRunner_ParallelThreeBranchesAndStreamingBranchFailure_RecordsCallerError
// verifies that when a parallel agent has three branches ([streamer, failer,
// idle]) or when the streaming branch itself fails after emitting partial
// content ([streamfail, waiter]), both INVOCATION_ERROR and the attributed
// AGENT_ERROR rows carry the failing branch's error text across both draining
// and stopping consumers.
func TestRunner_ParallelThreeBranchesAndStreamingBranchFailure_RecordsCallerError(t *testing.T) {
	for _, stopAtError := range []bool{false, true} {
		mode := "drain"
		if stopAtError {
			mode = "stop_at_error"
		}
		t.Run(mode, func(t *testing.T) {
			t.Run("three_branches", func(t *testing.T) {
				newRoot := func(t *testing.T, ready <-chan struct{}, acked chan struct{}) agent.Agent {
					idle, err := agent.New(agent.Config{
						Name: "idle",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								<-ctx.Done()
								yield(nil, ctx.Err())
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(idle): %v", err)
					}
					return newParallelAgent(t, "par_root", newStreamingLLMAgent(t, acked), newCustomFailer(t, ready), idle)
				}
				rows, _ := runUntilBranchFails(t, newRoot, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}, stopAtError)
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
				}
				assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
				errCounts := agentErrorCounts(rows)
				if errCounts["streamer"] != 0 {
					t.Errorf("AGENT_ERROR counts = %v, want 0 for streamer", errCounts)
				}
				if errCounts["failer"]+errCounts["idle"] != 1 {
					t.Errorf("AGENT_ERROR counts = %v, want total 1 across [failer, idle]", errCounts)
				}
			})

			t.Run("streaming_branch_fails", func(t *testing.T) {
				errStreamBackend := errors.New("stream backend failed")
				p, requests, _ := setupTestPlugin(t)
				waiterEntered := make(chan struct{})
				seen := make(chan struct{})
				var seenOnce sync.Once
				observer, err := baseplugin.New(baseplugin.Config{
					Name: "observer",
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						if ev != nil && ev.Partial {
							seenOnce.Do(func() { close(seen) })
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New(observer): %v", err)
				}
				streamfail, err := agent.New(agent.Config{
					Name: "streamfail",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							<-waiterEntered
							partial := session.NewEvent(ctx, ctx.InvocationID())
							partial.Author = "streamfail"
							partial.Branch = ctx.Branch()
							partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}
							if !yield(partial, nil) {
								return
							}
							<-seen
							yield(nil, errStreamBackend)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(streamfail): %v", err)
				}
				waiter, err := agent.New(agent.Config{
					Name: "waiter",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							close(waiterEntered)
							<-ctx.Done()
							yield(nil, ctx.Err())
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(waiter): %v", err)
				}
				parRoot := newParallelAgent(t, "par_root", streamfail, waiter)
				var gotErr error
				for _, runErr := range newTestRunner(t, parRoot, p, observer).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						gotErr = runErr
						if stopAtError {
							break
						}
					}
				}
				if !errors.Is(gotErr, errStreamBackend) {
					t.Fatalf("runner.Run error = %v, want %v", gotErr, errStreamBackend)
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
				}
				assertTerminalErrorRowsContain(t, rows, errStreamBackend.Error())
				errCounts := agentErrorCounts(rows)
				if errCounts["waiter"] != 1 || errCounts["streamfail"] != 0 {
					t.Errorf("AGENT_ERROR counts = %v, want waiter=1 and streamfail=0", errCounts)
				}
			})
		})
	}
}

// TestRunner_ParallelRespondedBranchFailsWhileSiblingStreams_ConsistentAcrossDrainAndStop
// verifies that when a parallel branch fails after already emitting a response
// while a sibling branch streams, retiredCause preserves the errgroup cause
// across root retirement on drain so both draining and stopping callers report
// INVOCATION_ERROR with the branch's error and zero AGENT_ERROR rows under
// both non-LLM and LLM roots.
func TestRunner_ParallelRespondedBranchFailsWhileSiblingStreams_ConsistentAcrossDrainAndStop(t *testing.T) {
	for _, rootKind := range []string{"parallel_root", "llm_root_transfer"} {
		t.Run(rootKind, func(t *testing.T) {
			for _, stopAtError := range []bool{false, true} {
				mode := "drain"
				if stopAtError {
					mode = "stop_at_error"
				}
				t.Run(mode, func(t *testing.T) {
					ctx := t.Context()
					p, requests, _ := setupTestPlugin(t)
					finalSeen := make(chan struct{})
					partialSeen := make(chan struct{})
					acked := make(chan struct{})
					ready := make(chan struct{})
					go func() {
						select {
						case <-partialSeen:
						case <-ctx.Done():
							return
						}
						select {
						case <-acked:
							close(ready)
						case <-ctx.Done():
						}
					}()
					var finalOnce, partialOnce sync.Once
					observer, err := baseplugin.New(baseplugin.Config{
						Name: "observer",
						OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
							if ev == nil {
								return nil, nil
							}
							if !ev.Partial && ev.Author == "failer" {
								finalOnce.Do(func() { close(finalSeen) })
							}
							if ev.Partial && ev.Author == "streamer" {
								partialOnce.Do(func() { close(partialSeen) })
							}
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New(observer): %v", err)
					}

					streamer, err := agent.New(agent.Config{
						Name: "streamer",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								<-finalSeen
								partial := session.NewEvent(ctx, ctx.InvocationID())
								partial.Author = "streamer"
								partial.Branch = ctx.Branch()
								partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}
								if !yield(partial, nil) {
									return
								}
								close(acked)
								<-ctx.Done()
								yield(nil, ctx.Err())
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(streamer): %v", err)
					}
					failer, err := agent.New(agent.Config{
						Name: "failer",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								finalEv := session.NewEvent(ctx, ctx.InvocationID())
								finalEv.Author = "failer"
								finalEv.Branch = ctx.Branch()
								finalEv.LLMResponse = model.LLMResponse{Content: textContent("model", "initial answer"), TurnComplete: true}
								if !yield(finalEv, nil) {
									return
								}
								<-ready
								yield(nil, errSiblingFailed)
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(failer): %v", err)
					}

					var root agent.Agent
					if rootKind == "parallel_root" {
						root = newParallelAgent(t, "par_root", streamer, failer)
					} else {
						par := newParallelAgent(t, "par", streamer, failer)
						root, err = llmagent.New(llmagent.Config{
							Name: "coord",
							Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "call-transfer", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "par"}},
							}}}}}},
							SubAgents: []agent.Agent{par},
						})
						if err != nil {
							t.Fatalf("llmagent.New(coord): %v", err)
						}
					}

					var gotErr error
					for _, runErr := range newTestRunner(t, root, p, observer).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
						if runErr != nil {
							gotErr = runErr
							if stopAtError {
								break
							}
						}
					}
					if !errors.Is(gotErr, errSiblingFailed) {
						t.Fatalf("runner.Run error = %v, want %v", gotErr, errSiblingFailed)
					}

					rows := collectRows(t, requests, untilInvocationEnd)
					if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
						t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
					}
					assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
					if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
						t.Errorf("got %d AGENT_ERROR rows, want 0 (all branches responded); rows: %v", n, rowsOfType(rows, EventAgentError))
					}
				})
			}
		})
	}
}

// TestTraceManager_RespondedBranchFailureCauseFallback verifies that when a
// branch context ends with a non-context.Canceled,
// non-bare-context.DeadlineExceeded cause after every open activation has
// already responded and no open call or recorded error identifies a failing
// activation, popUnclosedAgentSpans preserves that cause as the terminal error
// without fabricating AGENT_ERROR attribution.
func TestTraceManager_RespondedBranchFailureCauseFallback(t *testing.T) {
	t.Run("custom_cause", func(t *testing.T) {
		ctx := context.Background()
		errBranch := errors.New("branch exploded")
		branchCtx, cancelBranch := context.WithCancelCause(ctx)
		t.Cleanup(func() { cancelBranch(nil) })

		tm := newTraceManager()
		par := identity{invocationID: "inv-cause-fallback-custom", agentName: "par"}
		branch := identity{invocationID: "inv-cause-fallback-custom", agentName: "branch", branch: "par.branch"}
		tm.invocationSpan(ctx, par, nil)
		tm.pushAgentSpan(ctx, par)
		tm.pushAgentSpan(branchCtx, branch)
		tm.recordStreamed(ctx, branch, "branch", branch.branch, true)
		cancelBranch(errBranch)

		unclosed, termErr := tm.popUnclosedAgentSpans(ctx, par)
		if !errors.Is(termErr, errBranch) {
			t.Fatalf("popUnclosedAgentSpans terminal error = %v, want %v", termErr, errBranch)
		}
		if len(unclosed) != 0 {
			t.Errorf("popUnclosedAgentSpans unclosed = %d spans, want 0", len(unclosed))
		}
	})

	t.Run("wrapped_deadline_exceeded", func(t *testing.T) {
		ctx := context.Background()
		errBranch := fmt.Errorf("sub-agent timed out: %w", context.DeadlineExceeded)
		branchCtx, cancelBranch := context.WithCancelCause(ctx)
		t.Cleanup(func() { cancelBranch(nil) })

		tm := newTraceManager()
		par := identity{invocationID: "inv-cause-fallback-deadline", agentName: "par"}
		branch := identity{invocationID: "inv-cause-fallback-deadline", agentName: "branch", branch: "par.branch"}
		tm.invocationSpan(ctx, par, nil)
		tm.pushAgentSpan(ctx, par)
		tm.pushAgentSpan(branchCtx, branch)
		tm.recordStreamed(ctx, branch, "branch", branch.branch, true)
		cancelBranch(errBranch)

		unclosed, termErr := tm.popUnclosedAgentSpans(ctx, par)
		if !errors.Is(termErr, context.DeadlineExceeded) || !strings.Contains(termErr.Error(), "sub-agent timed out") {
			t.Fatalf("popUnclosedAgentSpans terminal error = %v, want wrapped context.DeadlineExceeded containing \"sub-agent timed out\"", termErr)
		}
		if len(unclosed) != 0 {
			t.Errorf("popUnclosedAgentSpans unclosed = %d spans, want 0", len(unclosed))
		}
	})
}

// TestTraceManager_CompleteResponseClearsRetiredCause verifies that when an
// inner parallel agent's responded branch retires with a failure cause onto its
// enclosing agent, a subsequent final response emitted by that enclosing agent
// clears its retiredCause so the invocation ends cleanly.
func TestTraceManager_CompleteResponseClearsRetiredCause(t *testing.T) {
	ctx := context.Background()
	errBranch := errors.New("branch exploded")
	branchCtx, cancelBranch := context.WithCancelCause(ctx)
	t.Cleanup(func() { cancelBranch(nil) })

	tm := newTraceManager()
	const invID = "inv-clear-retired-cause"
	guard := identity{invocationID: invID, agentName: "guard"}
	par := identity{invocationID: invID, agentName: "par", branch: "guard.par"}
	branch := identity{invocationID: invID, agentName: "branch", branch: "guard.par.branch"}
	tm.invocationSpan(ctx, guard, nil)
	tm.pushAgentSpan(ctx, guard)
	tm.pushAgentSpan(ctx, par)
	tm.pushAgentSpan(branchCtx, branch)
	tm.recordStreamed(ctx, branch, "branch", branch.branch, true)
	cancelBranch(errBranch)

	tm.popAgentSpan(ctx, branch)
	tm.popAgentSpan(ctx, par)
	tm.completeResponse(ctx, guard, "guard", "", nil)
	tm.popAgentSpan(ctx, guard)

	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, guard)
	if termErr != nil {
		t.Fatalf("popUnclosedAgentSpans terminal error = %v, want nil", termErr)
	}
	if len(unclosed) != 0 {
		t.Errorf("popUnclosedAgentSpans unclosed = %d spans, want 0", len(unclosed))
	}
}

// TestRunner_CustomAgentHandlesParallelFailure_RecoveryModes pins the recovery
// modes when a custom agent wraps a parallel or sequential sub-agent: reading a
// parallel agent to its end and emitting a final event of the custom agent's
// own clears retiredCause and ends with INVOCATION_COMPLETED only when every
// branch had responded and no canceled sibling recorded a collateral
// context.Canceled error, and reading a sequential agent to its end after its
// child failed silently retires the sequential span so completeResponse's
// open-span walk leaves the nested child open as AGENT_ERROR + INVOCATION_ERROR.
func TestRunner_CustomAgentHandlesParallelFailure_RecoveryModes(t *testing.T) {
	tests := []struct {
		name            string
		failerResponds  bool
		llmStreamer     bool
		readParToEnd    bool
		recovery        string // "none", "fallback", "own_final"
		wantCompleted   bool
		wantAgentErrors map[string]int
	}{
		{
			name:            "reads_par_to_end_returns_without_event",
			failerResponds:  true,
			readParToEnd:    true,
			recovery:        "none",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "reads_par_to_end_fallback_subagent",
			failerResponds:  true,
			readParToEnd:    true,
			recovery:        "fallback",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "reads_par_to_end_guard_own_final_event",
			failerResponds:  true,
			readParToEnd:    true,
			recovery:        "own_final",
			wantCompleted:   true,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "breaks_out_of_par_fallback_subagent",
			failerResponds:  true,
			readParToEnd:    false,
			recovery:        "fallback",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "breaks_out_of_par_guard_own_final_event",
			failerResponds:  true,
			readParToEnd:    false,
			recovery:        "own_final",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{},
		},
		{
			name:            "unresponded_failer_reads_par_to_end_guard_own_final_event",
			failerResponds:  false,
			readParToEnd:    true,
			recovery:        "own_final",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{"failer": 1},
		},
		{
			name:            "llm_streamer_collateral_error_reads_par_to_end_guard_own_final_event",
			failerResponds:  true,
			llmStreamer:     true,
			readParToEnd:    true,
			recovery:        "own_final",
			wantCompleted:   false,
			wantAgentErrors: map[string]int{},
		},
	}

	t.Run("reads_seq_with_silent_child_guard_own_final_event", func(t *testing.T) {
		requireADKStopOnSubAgentError(t)
		ctx := t.Context()
		p, requests, _ := setupTestPlugin(t)
		w, err := agent.New(agent.Config{
			Name: "w",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, errSiblingFailed)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(w): %v", err)
		}
		seq, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{w}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(seq): %v", err)
		}
		guard, err := agent.New(agent.Config{
			Name:      "guard",
			SubAgents: []agent.Agent{seq},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range seq.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "guard"
					ev.Branch = ctx.Branch()
					ev.LLMResponse = model.LLMResponse{Content: textContent("model", "guard recovered"), TurnComplete: true}
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(guard): %v", err)
		}
		for _, runErr := range newTestRunner(t, guard, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run yielded unexpected error: %v", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		assertTerminalErrorRowsContain(t, rows, `agent "w" execution failed before completion`)
		if got, want := agentErrorCounts(rows), map[string]int{"w": 1}; !maps.Equal(got, want) {
			t.Errorf("AGENT_ERROR counts = %v, want %v; rows: %v", got, want, rowsOfType(rows, EventAgentError))
		}
	})

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			p, requests, _ := setupTestPlugin(t)
			finalSeen := make(chan struct{})
			if !tc.failerResponds {
				close(finalSeen)
			}
			partialSeen := make(chan struct{})
			acked := make(chan struct{})
			ready := make(chan struct{})
			go func() {
				select {
				case <-partialSeen:
				case <-ctx.Done():
					return
				}
				select {
				case <-acked:
					close(ready)
				case <-ctx.Done():
				}
			}()
			var finalOnce, partialOnce sync.Once
			observer, err := baseplugin.New(baseplugin.Config{
				Name: "observer",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev == nil {
						return nil, nil
					}
					if tc.failerResponds && !ev.Partial && ev.Author == "failer" {
						finalOnce.Do(func() { close(finalSeen) })
					}
					if ev.Partial && ev.Author == "streamer" {
						partialOnce.Do(func() { close(partialSeen) })
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(observer): %v", err)
			}

			var streamer agent.Agent
			if tc.llmStreamer {
				streamer, err = llmagent.New(llmagent.Config{
					Name: "streamer",
					Model: &funcStreamLLM{fn: func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
						select {
						case <-finalSeen:
						case <-ctx.Done():
							return
						}
						if !yield(&model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}, nil) {
							return
						}
						close(acked)
						<-ctx.Done()
						yield(nil, ctx.Err())
					}},
				})
				if err != nil {
					t.Fatalf("llmagent.New(streamer): %v", err)
				}
			} else {
				streamer, err = agent.New(agent.Config{
					Name: "streamer",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							select {
							case <-finalSeen:
							case <-ctx.Done():
								return
							}
							partial := session.NewEvent(ctx, ctx.InvocationID())
							partial.Author = "streamer"
							partial.Branch = ctx.Branch()
							partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}
							if !yield(partial, nil) {
								return
							}
							close(acked)
							<-ctx.Done()
							yield(nil, ctx.Err())
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(streamer): %v", err)
				}
			}
			failer, err := agent.New(agent.Config{
				Name: "failer",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if tc.failerResponds {
							finalEv := session.NewEvent(ctx, ctx.InvocationID())
							finalEv.Author = "failer"
							finalEv.Branch = ctx.Branch()
							finalEv.LLMResponse = model.LLMResponse{Content: textContent("model", "initial answer"), TurnComplete: true}
							if !yield(finalEv, nil) {
								return
							}
						}
						select {
						case <-ready:
						case <-ctx.Done():
							return
						}
						yield(nil, errSiblingFailed)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(failer): %v", err)
			}
			par := newParallelAgent(t, "par", streamer, failer)

			fallback, err := agent.New(agent.Config{
				Name: "fallback",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "fallback"
						ev.Branch = ctx.Branch()
						ev.LLMResponse = model.LLMResponse{Content: textContent("model", "fallback answer"), TurnComplete: true}
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(fallback): %v", err)
			}

			guard, err := agent.New(agent.Config{
				Name:      "guard",
				SubAgents: []agent.Agent{par, fallback},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								if !tc.readParToEnd {
									break
								}
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						switch tc.recovery {
						case "fallback":
							for ev, runErr := range fallback.Run(ctx) {
								if !yield(ev, runErr) {
									return
								}
							}
						case "own_final":
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "guard"
							ev.Branch = ctx.Branch()
							ev.LLMResponse = model.LLMResponse{Content: textContent("model", "guard recovered"), TurnComplete: true}
							yield(ev, nil)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(guard): %v", err)
			}

			runCfg := agent.RunConfig{}
			if tc.llmStreamer {
				runCfg.StreamingMode = agent.StreamingModeSSE
			}
			for _, runErr := range newTestRunner(t, guard, p, observer).Run(ctx, "u1", "s1", textContent("user", "go"), runCfg) {
				if runErr != nil {
					t.Fatalf("runner.Run yielded unexpected error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if tc.wantCompleted {
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; rows: %v", n, rowsOfType(rows, EventInvocationError))
				}
				onlyRowOfType(t, rows, EventInvocationCompleted)
			} else {
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
				}
				assertTerminalErrorRowsContain(t, rows, errSiblingFailed.Error())
			}
			if got := agentErrorCounts(rows); !maps.Equal(got, tc.wantAgentErrors) {
				t.Errorf("AGENT_ERROR counts = %v, want %v; rows: %v", got, tc.wantAgentErrors, rowsOfType(rows, EventAgentError))
			}
			if tc.llmStreamer {
				if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
					t.Errorf("got %d LLM_ERROR rows, want 1; event types: %v", n, eventTypes(rows))
				}
			}
		})
	}
}

type funcStreamLLM struct {
	fn func(ctx context.Context, yield func(*model.LLMResponse, error) bool)
}

func (*funcStreamLLM) Name() string { return "func-stream-llm" }

func (m *funcStreamLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.fn(ctx, yield)
	}
}

// TestRunner_WorkflowRetryParallelRespondedBranches_PurgesFailedAttemptAndCompletes
// verifies that when a workflow RetryConfig node retries a parallel agent whose
// failed first attempt had every branch respond before failing — both when the
// canceled sibling is a custom agent and when it is an llmagent that records a
// collateral context.Canceled model error — failuresLocked sets aside any
// collateral error and marks only the parallel root as failed with subtreeCause
// so attempt 1 is purged (emitting only AGENT_ERROR(par) with the branch's
// error) and attempt 2 completes the invocation cleanly.
func TestRunner_WorkflowRetryParallelRespondedBranches_PurgesFailedAttemptAndCompletes(t *testing.T) {
	for _, llmStreamer := range []bool{false, true} {
		name := "custom_streamer"
		if llmStreamer {
			name = "llm_streamer_collateral_error"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			p, requests, _ := setupTestPlugin(t)
			finalSeen := make(chan struct{})
			partialSeen := make(chan struct{})
			acked := make(chan struct{})
			ready := make(chan struct{})
			go func() {
				select {
				case <-partialSeen:
				case <-ctx.Done():
					return
				}
				select {
				case <-acked:
					close(ready)
				case <-ctx.Done():
				}
			}()
			var finalOnce, partialOnce sync.Once
			observer, err := baseplugin.New(baseplugin.Config{
				Name: "observer",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev == nil {
						return nil, nil
					}
					if !ev.Partial && ev.Author == "failer" {
						finalOnce.Do(func() { close(finalSeen) })
					}
					if ev.Partial && ev.Author == "streamer" {
						partialOnce.Do(func() { close(partialSeen) })
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(observer): %v", err)
			}

			var streamerAttempts, failerAttempts atomic.Int32
			var streamer agent.Agent
			if llmStreamer {
				streamer, err = llmagent.New(llmagent.Config{
					Name: "streamer",
					Model: &funcStreamLLM{fn: func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
						if streamerAttempts.Add(1) == 1 {
							select {
							case <-finalSeen:
							case <-ctx.Done():
								return
							}
							if !yield(&model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}, nil) {
								return
							}
							close(acked)
							<-ctx.Done()
							yield(nil, ctx.Err())
							return
						}
						yield(&model.LLMResponse{Content: textContent("model", "streamer succeeded on retry"), TurnComplete: true}, nil)
					}},
				})
				if err != nil {
					t.Fatalf("llmagent.New(streamer): %v", err)
				}
			} else {
				streamer, err = agent.New(agent.Config{
					Name: "streamer",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							if streamerAttempts.Add(1) == 1 {
								select {
								case <-finalSeen:
								case <-ctx.Done():
									return
								}
								partial := session.NewEvent(ctx, ctx.InvocationID())
								partial.Author = "streamer"
								partial.Branch = ctx.Branch()
								partial.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "streaming...")}
								if !yield(partial, nil) {
									return
								}
								close(acked)
								<-ctx.Done()
								yield(nil, ctx.Err())
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "streamer"
							ev.Branch = ctx.Branch()
							ev.LLMResponse = model.LLMResponse{Content: textContent("model", "streamer succeeded on retry"), TurnComplete: true}
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(streamer): %v", err)
				}
			}
			failer, err := agent.New(agent.Config{
				Name: "failer",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if failerAttempts.Add(1) == 1 {
							finalEv := session.NewEvent(ctx, ctx.InvocationID())
							finalEv.Author = "failer"
							finalEv.Branch = ctx.Branch()
							finalEv.LLMResponse = model.LLMResponse{Content: textContent("model", "initial answer"), TurnComplete: true}
							if !yield(finalEv, nil) {
								return
							}
							select {
							case <-ready:
							case <-ctx.Done():
								return
							}
							yield(nil, errSiblingFailed)
							return
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "failer"
						ev.Branch = ctx.Branch()
						ev.LLMResponse = model.LLMResponse{Content: textContent("model", "failer succeeded on retry"), TurnComplete: true}
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(failer): %v", err)
			}

			par := newParallelAgent(t, "par", streamer, failer)
			node, err := workflow.NewAgentNode(par, workflow.NodeConfig{
				RetryConfig: &workflow.RetryConfig{
					MaxAttempts:   2,
					InitialDelay:  time.Millisecond,
					MaxDelay:      time.Millisecond,
					BackoffFactor: 1.0,
				},
			})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode(par): %v", err)
			}
			wfRoot, err := workflowagent.New(workflowagent.Config{
				Name:  "wf_root",
				Edges: []workflow.Edge{{From: workflow.Start, To: node}},
			})
			if err != nil {
				t.Fatalf("workflowagent.New(wf_root): %v", err)
			}

			runCfg := agent.RunConfig{}
			if llmStreamer {
				runCfg.StreamingMode = agent.StreamingModeSSE
			}
			for _, runErr := range newTestRunner(t, wfRoot, p, observer).Run(ctx, "u1", "s1", textContent("user", "go"), runCfg) {
				if runErr != nil {
					t.Fatalf("runner.Run yielded unexpected error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; rows: %v", n, rowsOfType(rows, EventInvocationError))
			}
			onlyRowOfType(t, rows, EventInvocationCompleted)
			if got, want := agentErrorCounts(rows), map[string]int{"par": 1}; !maps.Equal(got, want) {
				t.Fatalf("AGENT_ERROR rows per agent = %v, want %v; event types: %v", got, want, eventTypes(rows))
			}
			errRow := onlyRowOfType(t, rows, EventAgentError)
			if msg, _ := errRow["error_message"].(string); !strings.Contains(msg, errSiblingFailed.Error()) {
				t.Errorf("AGENT_ERROR(par) error_message = %q, want it to contain %q", msg, errSiblingFailed.Error())
			}
			if llmStreamer {
				if n := len(rowsOfType(rows, EventLLMError)); n != 1 {
					t.Errorf("got %d LLM_ERROR rows, want 1; event types: %v", n, eventTypes(rows))
				}
			}
		})
	}
}

// TestTraceManager_PopAgentSpanClearsPartialOnlyForDirectWorkflowRoot
// verifies which end of an agent shows that the caller read past the last
// partial event: only that of the workflow or custom agent that the runner
// runs directly. An llmagent that the runner runs is a workflow node that does
// not wait for its caller, and an agent below another agent, such as a branch
// of a parallel agent, can end after the caller stopped reading on a partial
// event, so neither says what the caller did. The sibling that never
// responded fails the invocation only when the caller read on.
func TestTraceManager_PopAgentSpanClearsPartialOnlyForDirectWorkflowRoot(t *testing.T) {
	const invID = "inv-root-end"
	ctx := context.Background()
	llmRoot, err := llmagent.New(llmagent.Config{Name: "root"})
	if err != nil {
		t.Fatalf("llmagent.New(root) error: %v", err)
	}
	author := newTestAgent(t, "author")
	failer := newTestAgent(t, "failer")
	workflowRoot := newTestAgent(t, "root", author, failer)
	rootID := identity{invocationID: invID, agentName: "root"}
	authorID := identity{invocationID: invID, agentName: "author", branch: "root.author"}
	failerID := identity{invocationID: invID, agentName: "failer", branch: "root.failer"}
	for _, tc := range []struct {
		name string
		root agent.Agent
		// ends is the agent that ends before the invocation does.
		ends    identity
		wantErr bool
	}{
		{name: "workflow agent that the runner runs", root: workflowRoot, ends: rootID, wantErr: true},
		{name: "llmagent that the runner runs", root: llmRoot, ends: rootID, wantErr: false},
		{name: "workflow agent below another agent", root: workflowRoot, ends: authorID, wantErr: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.invocationSpan(contextRunning(tc.root), rootID, tc.root)
			tm.pushAgentSpan(contextRunning(tc.root), rootID)
			tm.pushAgentSpan(contextRunning(author), authorID)
			tm.pushAgentSpan(contextRunning(failer), failerID)
			tm.recordStreamed(ctx, authorID, "author", authorID.branch, true)

			tm.popAgentSpan(ctx, tc.ends)

			if _, err := tm.popUnclosedAgentSpans(ctx, rootID); (err != nil) != tc.wantErr {
				t.Errorf("popUnclosedAgentSpans error = %v, want error %v", err, tc.wantErr)
			}
		})
	}
}

// TestRunner_ProgressEventWithoutContent verifies how a run ends after a
// custom agent writes a partial event without content, such as a progress
// event: a caller that stops reading on it stops mid-stream, and a sibling
// that a sequential agent starts next and that fails is reported, because the
// event is not a response of the sequential agent either.
func TestRunner_ProgressEventWithoutContent(t *testing.T) {
	newPinger := func(t *testing.T, name string) agent.Agent {
		t.Helper()
		a, err := agent.New(agent.Config{
			Name: name,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ping := session.NewEvent(ctx, ctx.InvocationID())
					ping.Branch = ctx.Branch()
					ping.Partial = true
					yield(ping, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%s): %v", name, err)
		}
		return a
	}
	failErr := errors.New("failer failed")
	failer, err := agent.New(agent.Config{
		Name: "failer",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, failErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failer): %v", err)
	}

	t.Run("caller stops on the event", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		r := newTestRunner(t, newPinger(t, "worker"), p)
		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatal("runner.Run yielded no partial event")
		}
		requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
	})

	t.Run("sequential sibling fails after the event", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		seq, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{newPinger(t, "pinger"), failer}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(seq_root): %v", err)
		}
		r := newTestRunner(t, seq, p)
		var gotErr error
		for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				gotErr = err
				break
			}
		}
		if !errors.Is(gotErr, failErr) {
			t.Fatalf("runner.Run error = %v, want %v", gotErr, failErr)
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		onlyRowOfType(t, rows, EventInvocationError)
		if got := agentErrorCounts(rows); got["failer"] != 1 {
			t.Errorf("AGENT_ERROR rows per agent = %v, want one for failer", got)
		}
	})
}

// TestTraceManager_PartialEventWithoutContentIsNoResponse verifies that a
// partial event without content does not count as a response of its author,
// so that the author's failure is reported once a session event that is not
// partial, such as one that carries only actions, shows that the caller read
// on.
func TestTraceManager_PartialEventWithoutContentIsNoResponse(t *testing.T) {
	for _, tc := range []struct {
		name        string
		withContent bool
		wantErr     bool
	}{
		{name: "partial event without content", withContent: false, wantErr: true},
		{name: "partial event with content", withContent: true, wantErr: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			tm := newTraceManager()
			worker := identity{invocationID: "inv-ping", agentName: "worker"}
			tm.invocationSpan(ctx, worker, nil)
			tm.pushAgentSpan(ctx, worker)
			tm.recordStreamed(ctx, worker, "worker", "", tc.withContent)
			tm.clearLastPartial(ctx, worker)
			if _, err := tm.popUnclosedAgentSpans(ctx, worker); (err != nil) != tc.wantErr {
				t.Errorf("popUnclosedAgentSpans error = %v, want error %v", err, tc.wantErr)
			}
		})
	}
}

// TestTraceManager_StopOnProgressEventOfParallelBranch verifies that a caller
// that stops reading on a partial event without content of a parallel
// branch, whose author then starts work that the parallel agent cancels,
// stops mid-stream: the parallel agent wrote the event through its branch,
// though its own context is not canceled. The run does fail when a session
// event that is not partial followed, and so does the author of an earlier
// partial event that such a session event followed.
func TestTraceManager_StopOnProgressEventOfParallelBranch(t *testing.T) {
	const invID = "inv-progress"
	par := identity{invocationID: invID, agentName: "par"}
	author := identity{invocationID: invID, agentName: "author", branch: "par.author"}
	child := identity{invocationID: invID, agentName: "child", branch: "par.author"}
	pinger := identity{invocationID: invID, agentName: "pinger", branch: "par.pinger"}
	other := identity{invocationID: invID, agentName: "other", branch: "par.other"}
	for _, tc := range []struct {
		name string
		// before writes session events before the author's partial event.
		before  func(context.Context, *traceManager)
		after   func(context.Context, *traceManager)
		wantErr bool
	}{
		{name: "caller stops on the event", wantErr: false},
		{name: "session event after the event", after: func(ctx context.Context, tm *traceManager) {
			tm.clearLastPartial(ctx, par)
		}, wantErr: true},
		{name: "earlier partial event of another branch", before: func(ctx context.Context, tm *traceManager) {
			tm.recordStreamed(ctx, pinger, "pinger", pinger.branch, false)
			tm.completeResponse(ctx, other, "other", other.branch, nil)
		}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			pingerCtx := context.Background()
			branchCtx, cancel := context.WithCancel(ctx)
			t.Cleanup(cancel)
			tm := newTraceManager()
			tm.invocationSpan(ctx, par, nil)
			tm.pushAgentSpan(ctx, par)
			tm.pushAgentSpan(pingerCtx, pinger)
			tm.pushAgentSpan(branchCtx, other)
			tm.pushAgentSpan(branchCtx, author)
			if tc.before != nil {
				tc.before(ctx, tm)
			}
			tm.recordStreamed(ctx, author, "author", author.branch, false)
			tm.pushAgentSpan(branchCtx, child)
			if tc.after != nil {
				tc.after(ctx, tm)
			}
			cancel()
			if tc.before == nil {
				// The pinger branch ended without failing.
				tm.popAgentSpan(pingerCtx, pinger)
			}
			if _, err := tm.popUnclosedAgentSpans(ctx, par); (err != nil) != tc.wantErr {
				t.Errorf("popUnclosedAgentSpans error = %v, want error %v", err, tc.wantErr)
			}
		})
	}
}

// TestOnEvent_ProgressEventIsNoResponse verifies that a partial event without
// content does not count as a response of its author: once a session event
// that is not partial followed, the author that never responded fails.
func TestOnEvent_ProgressEventIsNoResponse(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	ping := session.NewEvent(t.Context(), mCtx.invocationID)
	ping.Author = mCtx.agentName
	ping.Partial = true
	errs.add(p.OnEventCallback()(mCtx, ping))
	stateOnly := session.NewEvent(t.Context(), mCtx.invocationID)
	stateOnly.Author = mCtx.agentName
	errs.add(p.OnEventCallback()(mCtx, stateOnly))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got INVOCATION_COMPLETED, want INVOCATION_ERROR: the author never responded; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationError)
	if got, want := agentErrorCounts(rows), map[string]int{mCtx.agentName: 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR rows per agent = %v, want %v", got, want)
	}
}

// TestAfterTool_FailedToolRunEndsWithAgentError verifies that a sub-agent that
// a tool call ran and that failed, which ADK ends without AfterAgentCallback,
// ends with an AGENT_ERROR row of its own once its owner has no tool call left
// open, even when the last call to end succeeded, and that the invocation
// still completes.
func TestAfterTool_FailedToolRunEndsWithAgentError(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	with := func(agentName, branch, path, callID string) *mockInvocationContext {
		c := *mCtx
		c.agentName, c.branch, c.path, c.functionCallID = agentName, branch, path, callID
		return &c
	}
	coord := with(mCtx.agentName, "", "", "")
	callA, callB := with(mCtx.agentName, "", "", "call-a"), with(mCtx.agentName, "", "", "call-b")
	helper := with("helper", "helper@1", "/helper@1", "")
	helperTool := &fakeTool{name: "helper"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(coord))
	errs.add(p.BeforeToolCallback()(callA, helperTool, nil))
	errs.add(p.BeforeToolCallback()(callB, helperTool, nil))
	errs.add(p.BeforeAgentCallback()(helper))
	errs.add(p.BeforeModelCallback()(helper, &model.LLMRequest{}))
	errs.add(p.OnModelErrorCallback()(helper, &model.LLMRequest{}, errors.New("helper model failed")))
	errs.add(p.OnToolErrorCallback()(callA, helperTool, nil, errors.New("helper failed")))
	errs.add(p.AfterToolCallback()(callB, helperTool, nil, map[string]any{"answer": "ok"}, nil))
	answer := session.NewEvent(t.Context(), mCtx.invocationID)
	answer.Author = mCtx.agentName
	answer.Content = textContent("model", "done")
	errs.add(p.OnEventCallback()(coord, answer))
	errs.check(t)
	p.AfterRunCallback()(coord)

	requireCompletedWithHandledAgentError(t, collectRows(t, requests, untilInvocationEnd), "helper", "helper model failed")
}

// TestOnToolError_FailedToolRunEndsWithAgentError verifies that a sub-agent
// that the last open tool call ran and that failed ends with an AGENT_ERROR
// row of its own when only OnToolErrorCallback ends the call, as when an
// earlier plugin's AfterToolCallback answers it, and that the invocation still
// completes.
func TestOnToolError_FailedToolRunEndsWithAgentError(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	with := func(agentName, branch, path, callID string) *mockInvocationContext {
		c := *mCtx
		c.agentName, c.branch, c.path, c.functionCallID = agentName, branch, path, callID
		return &c
	}
	coord := with(mCtx.agentName, "", "", "")
	call := with(mCtx.agentName, "", "", "call-a")
	helper := with("helper", "helper@1", "/helper@1", "")
	helperTool := &fakeTool{name: "helper"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(coord))
	errs.add(p.BeforeToolCallback()(call, helperTool, nil))
	errs.add(p.BeforeAgentCallback()(helper))
	errs.add(p.BeforeModelCallback()(helper, &model.LLMRequest{}))
	errs.add(p.OnModelErrorCallback()(helper, &model.LLMRequest{}, errors.New("helper model failed")))
	errs.add(p.OnToolErrorCallback()(call, helperTool, nil, errors.New("helper failed")))
	answer := session.NewEvent(t.Context(), mCtx.invocationID)
	answer.Author = mCtx.agentName
	answer.Content = textContent("model", "done")
	errs.add(p.OnEventCallback()(coord, answer))
	errs.check(t)
	p.AfterRunCallback()(coord)

	requireCompletedWithHandledAgentError(t, collectRows(t, requests, untilInvocationEnd), "helper", "helper model failed")
}

// TestRunner_StopReadingOnPartialEventWithoutContent_IsNotAnError verifies
// that a caller that stops reading on a partial event without content or
// output, such as the progress event of a custom agent, stops mid-stream
// rather than at a failure: alone, and as a sub-agent of a parallel agent
// whose other sub-agent is still running when the caller stops.
func TestRunner_StopReadingOnPartialEventWithoutContent_IsNotAnError(t *testing.T) {
	newPinger := func(t *testing.T) agent.Agent {
		t.Helper()
		a, err := agent.New(agent.Config{
			Name: "pinger",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ping := session.NewEvent(ctx, ctx.InvocationID())
					ping.Branch = ctx.Branch()
					ping.Partial = true
					if !yield(ping, nil) {
						return
					}
					done := session.NewEvent(ctx, ctx.InvocationID())
					done.Branch = ctx.Branch()
					done.Content = textContent("model", "done")
					yield(done, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(pinger): %v", err)
		}
		return a
	}
	newParallel := func(t *testing.T) agent.Agent {
		t.Helper()
		// waiter runs until the parallel agent stops it and then fails with
		// the context's error, without any event.
		waiter, err := agent.New(agent.Config{
			Name: "waiter",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					<-ctx.Done()
					yield(nil, ctx.Err())
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(waiter): %v", err)
		}
		par, err := parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{
				Name:      "par_root",
				SubAgents: []agent.Agent{newPinger(t), waiter},
			},
		})
		if err != nil {
			t.Fatalf("parallelagent.New(par_root): %v", err)
		}
		return par
	}
	tests := []struct {
		name string
		root func(*testing.T) agent.Agent
	}{
		{name: "custom root", root: newPinger},
		{name: "sub-agent of a parallel root", root: newParallel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			r := newTestRunner(t, tt.root(t), p)
			var sawPartial bool
			for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run error: %v", err)
				}
				if ev != nil && ev.Partial {
					sawPartial = true
					break
				}
			}
			if !sawPartial {
				t.Fatal("runner.Run yielded no partial event")
			}

			requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
		})
	}
}

// TestTraceManager_ToolRunsPurgedOnceNoToolCallIsOpen verifies that the
// activations that an agent's tool calls ran and that never ended, as a failed
// single_turn sub-agent does, are purged with their errors only once the agent
// has no tool call left open: until then, an activation may still be running
// for another call.
func TestTraceManager_ToolRunsPurgedOnceNoToolCallIsOpen(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invID = "inv-tool-runs"
	coord := identity{invocationID: invID, agentName: "coordinator"}
	helper := func(runID string) identity {
		return identity{invocationID: invID, agentName: "helper", branch: "helper@" + runID, path: "/helper@" + runID}
	}
	failed, running := helper("1"), helper("2")
	callA, callB := toolKey(coord, "call-a", "helper"), toolKey(coord, "call-b", "helper")

	tm.invocationSpan(ctx, coord, nil)
	tm.pushAgentSpan(ctx, coord)
	tm.pushToolSpan(ctx, coord, callA)
	tm.pushToolSpan(ctx, coord, callB)
	tm.pushAgentSpan(ctx, failed)
	tm.pushAgentSpan(ctx, running)
	tm.pushLLMSpan(ctx, failed)
	tm.recordError(ctx, failed, llmKey(failed), errors.New("helper model failed"))
	tm.popLLMSpan(ctx, failed, false)

	tm.popToolSpan(ctx, coord, callA, true)
	if got := openSpans(t, tm, invID).agents; got != 3 {
		t.Fatalf("open agent spans while call-b is open = %d, want 3", got)
	}
	if end := tm.popAgentSpan(ctx, running); end.latency == nil {
		t.Errorf("AGENT_COMPLETED of the running helper = %+v, want its span with latency", end)
	}

	tm.popToolSpan(ctx, coord, callB, false)
	if got := openSpans(t, tm, invID); got.agents != 1 || got.models != 0 {
		t.Errorf("open spans after the last tool call ended = %+v, want the coordinator alone", got)
	}
	tm.completeResponse(ctx, coord, "coordinator", "", nil)
	if unclosed, err := tm.popUnclosedAgentSpans(ctx, coord); err != nil || len(unclosed) != 0 {
		t.Errorf("popUnclosedAgentSpans = (%v, %v), want no failure", unclosed, err)
	}
}

// TestTraceManager_NodeReentryPurgesOnlyFailedAttempt verifies that an agent
// that runs again on the same node path drops its earlier activation, with its
// calls and errors, only if that activation failed, as an attempt that
// RetryConfig runs again did. A healthy earlier activation, such as the first
// planner of a transfer-back chain within the node, is still running.
func TestTraceManager_NodeReentryPurgesOnlyFailedAttempt(t *testing.T) {
	ctx := context.Background()
	flaky := identity{invocationID: "inv-retry", agentName: "flaky", path: "wf@1/flaky@1"}
	tests := []struct {
		name string
		// attempt runs the first activation of flaky.
		attempt func(*traceManager)
	}{
		{
			name:    "never responded",
			attempt: func(tm *traceManager) {},
		},
		{
			name: "unrecovered error after a partial event",
			attempt: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, flaky)
				tm.recordStreamed(ctx, flaky, "flaky", "", true)
				tm.recordError(ctx, flaky, llmKey(flaky), errors.New("stream broken"))
				tm.popLLMSpan(ctx, flaky, false)
			},
		},
		{
			name: "open tool call",
			attempt: func(tm *traceManager) {
				tm.pushLLMSpan(ctx, flaky)
				tm.popLLMSpan(ctx, flaky, true)
				tm.pushToolSpan(ctx, flaky, toolKey(flaky, "call-1", "lookup"))
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.invocationSpan(ctx, flaky, nil)
			tm.pushAgentSpan(ctx, flaky)
			tt.attempt(tm)

			tm.pushAgentSpan(ctx, flaky)
			if got := openSpans(t, tm, flaky.invocationID); got != (spanCounts{agents: 1}) {
				t.Errorf("open spans after the retry started = %+v, want the retry alone", got)
			}
			// Nothing of the failed attempt, its partial event included,
			// hides that the retry never responded.
			if _, err := tm.popUnclosedAgentSpans(ctx, flaky); err == nil || !strings.Contains(err.Error(), "execution failed") {
				t.Errorf("popUnclosedAgentSpans error = %v, want the failure of the retry", err)
			}
		})
	}

	t.Run("running transfer-back chain", func(t *testing.T) {
		tm := newTraceManager()
		planner := identity{invocationID: "inv-chain", agentName: "planner", path: "wf@1/planner@1"}
		helper := identity{invocationID: "inv-chain", agentName: "helper", path: "wf@1/planner@1"}
		tm.invocationSpan(ctx, planner, nil)
		tm.pushAgentSpan(ctx, planner)
		tm.pushLLMSpan(ctx, planner)
		tm.popLLMSpan(ctx, planner, true)
		tm.pushAgentSpan(ctx, helper)
		tm.pushLLMSpan(ctx, helper)
		tm.popLLMSpan(ctx, helper, true)

		tm.pushAgentSpan(ctx, planner)
		if got := openSpans(t, tm, "inv-chain").agents; got != 3 {
			t.Errorf("open agent spans after planner re-entered = %d, want 3: the first planner and helper still run", got)
		}
	})

	t.Run("failed activation outside a node", func(t *testing.T) {
		// A custom agent that stops reading its sub-agent at an error and runs
		// it again may have yielded the error, so the activation that failed
		// keeps it.
		tm := newTraceManager()
		bot := identity{invocationID: "inv-custom-retry", agentName: "bot"}
		tm.invocationSpan(ctx, bot, nil)
		tm.pushAgentSpan(ctx, bot)
		tm.pushLLMSpan(ctx, bot)
		tm.recordError(ctx, bot, llmKey(bot), errors.New("model failed"))
		tm.popLLMSpan(ctx, bot, false)

		tm.pushAgentSpan(ctx, bot)
		tm.pushLLMSpan(ctx, bot)
		tm.popLLMSpan(ctx, bot, true)
		tm.popAgentSpan(ctx, bot)
		if _, err := tm.popUnclosedAgentSpans(ctx, bot); err == nil {
			t.Error("popUnclosedAgentSpans error = nil, want the error of the first activation")
		}
	})
}

// TestTraceManager_LastPartialClearedByWorkUnderItsAuthor verifies that work
// that starts under the author of the last partial event shows that the caller
// kept reading after the event, so a failure that follows is reported, while
// work of a concurrent sibling of the author, such as another sub-agent of a
// parallel agent, does not.
func TestTraceManager_LastPartialClearedByWorkUnderItsAuthor(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-partial"
	par := identity{invocationID: invID, agentName: "par"}
	author := identity{invocationID: invID, agentName: "author", branch: "par.author"}
	sibling := identity{invocationID: invID, agentName: "sibling", branch: "par.sibling"}
	child := identity{invocationID: invID, agentName: "child", branch: "par.author"}
	tests := []struct {
		name    string
		work    func(*traceManager)
		wantErr bool
	}{
		{name: "sub-agent of the author starts", work: func(tm *traceManager) { tm.pushAgentSpan(ctx, child) }, wantErr: true},
		{name: "author starts a model call", work: func(tm *traceManager) { tm.pushLLMSpan(ctx, author) }, wantErr: true},
		{name: "author starts a tool call", work: func(tm *traceManager) { tm.pushToolSpan(ctx, author, toolKey(author, "call-1", "lookup")) }, wantErr: true},
		{name: "sibling starts a model call", work: func(tm *traceManager) { tm.pushLLMSpan(ctx, sibling) }, wantErr: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.invocationSpan(ctx, par, nil)
			tm.pushAgentSpan(ctx, par)
			tm.pushAgentSpan(ctx, author)
			tm.pushAgentSpan(ctx, sibling)
			tm.recordStreamed(ctx, author, "author", author.branch, true)

			tt.work(tm)

			if _, err := tm.popUnclosedAgentSpans(ctx, par); (err != nil) != tt.wantErr {
				t.Errorf("popUnclosedAgentSpans error = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

// TestTraceManager_CanceledWorkAfterPartialIsNotAFailure verifies that work
// that the author of the last session event, a partial one, starts afterwards
// does not fail the invocation once the context of the work is canceled, as
// when a parallel agent lets the author go on after the caller stopped
// reading and then cancels its sub-agents. The work still fails the
// invocation when its context is live, and when a session event followed the
// partial one.
func TestTraceManager_CanceledWorkAfterPartialIsNotAFailure(t *testing.T) {
	const invID = "inv-canceled-work"
	par := identity{invocationID: invID, agentName: "par"}
	author := identity{invocationID: invID, agentName: "author", branch: "par.author"}
	sibling := identity{invocationID: invID, agentName: "sibling", branch: "par.sibling"}
	child := identity{invocationID: invID, agentName: "child", branch: "par.author"}
	works := []struct {
		name string
		// work starts work with ctx; stop cancels ctx, if the case does,
		// as a parallel agent does once it stops.
		work func(ctx context.Context, stop func(), tm *traceManager)
	}{
		{name: "sub-agent of the author starts", work: func(ctx context.Context, _ func(), tm *traceManager) { tm.pushAgentSpan(ctx, child) }},
		{name: "author starts a model call", work: func(ctx context.Context, _ func(), tm *traceManager) { tm.pushLLMSpan(ctx, author) }},
		{name: "author starts a tool call", work: func(ctx context.Context, _ func(), tm *traceManager) {
			tm.pushToolSpan(ctx, author, toolKey(author, "call-1", "lookup"))
		}},
		{name: "sub-agent of the author fails with the context's error", work: func(ctx context.Context, stop func(), tm *traceManager) {
			tm.pushAgentSpan(ctx, child)
			tm.pushLLMSpan(ctx, child)
			stop()
			tm.recordError(ctx, child, llmKey(child), context.Canceled)
			tm.popLLMSpan(ctx, child, false)
		}},
	}
	cases := []struct {
		name string
		// cancel cancels the context of the sub-agents of par before the
		// invocation ends, as a parallel agent does once it stops.
		cancel bool
		// eventAfter writes a final session event of sibling after the work.
		eventAfter bool
		wantErr    bool
	}{
		{name: "context canceled", cancel: true, wantErr: false},
		{name: "context live", cancel: false, wantErr: true},
		{name: "session event after the partial one", cancel: true, eventAfter: true, wantErr: true},
	}
	for _, w := range works {
		for _, tc := range cases {
			t.Run(w.name+"/"+tc.name, func(t *testing.T) {
				ctx := context.Background()
				branchCtx, cancel := context.WithCancel(ctx)
				t.Cleanup(cancel)
				stop := func() {
					if tc.cancel {
						cancel()
					}
				}
				tm := newTraceManager()
				tm.invocationSpan(ctx, par, nil)
				tm.pushAgentSpan(ctx, par)
				tm.pushAgentSpan(branchCtx, author)
				tm.pushAgentSpan(branchCtx, sibling)
				tm.recordStreamed(ctx, author, "author", author.branch, true)

				w.work(branchCtx, stop, tm)
				if tc.eventAfter {
					tm.completeResponse(ctx, sibling, "sibling", sibling.branch, nil)
				}
				stop()

				if _, err := tm.popUnclosedAgentSpans(ctx, par); (err != nil) != tc.wantErr {
					t.Errorf("popUnclosedAgentSpans error = %v, want error %v", err, tc.wantErr)
				}
			})
		}
	}
}

// TestTraceManager_ToolEndKeepsFailedActivationOutsideToolRun verifies that a
// failed activation under an agent that no tool call ran, such as a transfer
// target on the agent's own node path, stays open with its error when a tool
// call of the agent ends.
func TestTraceManager_ToolEndKeepsFailedActivationOutsideToolRun(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	coord := identity{invocationID: "inv-peer", agentName: "coordinator"}
	peer := identity{invocationID: "inv-peer", agentName: "peer"}
	call := toolKey(coord, "call-1", "lookup")
	tm.invocationSpan(ctx, coord, nil)
	tm.pushAgentSpan(ctx, coord)
	tm.pushLLMSpan(ctx, coord)
	tm.popLLMSpan(ctx, coord, true)
	tm.pushAgentSpan(ctx, peer)
	tm.pushLLMSpan(ctx, peer)
	tm.recordError(ctx, peer, llmKey(peer), errors.New("peer model failed"))
	tm.popLLMSpan(ctx, peer, false)

	tm.pushToolSpan(ctx, coord, call)
	tm.popToolSpan(ctx, coord, call, false)
	if _, err := tm.popUnclosedAgentSpans(ctx, coord); err == nil {
		t.Error("popUnclosedAgentSpans error = nil, want the error of peer")
	}
}

// TestTraceManager_CanceledModelCallAfterStopMidStream verifies that a model
// call that ended with context.Canceled does not fail the invocation when the
// caller stopped reading on a partial event while the invocation's own context
// was live and the context of the error's branch was canceled, as when a
// parallel agent stops its other sub-agents, and that it still fails it, and is
// attributed to its own activation, otherwise, as when a model cancels a
// request context of its own or an agent cancels the context of its
// sub-agent.
func TestTraceManager_CanceledModelCallAfterStopMidStream(t *testing.T) {
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name    string
		ctx     context.Context
		partial bool
		err     error
		// errCtx is the context of the callback that records err, and
		// slowCtx the context that slow's activation starts with.
		errCtx  context.Context
		slowCtx context.Context
		// wantUnclosed is the number of spans that the error is attributed
		// to: slow and par, or none if the invocation did not fail.
		wantUnclosed int
	}{
		{name: "stopped mid-stream", ctx: context.Background(), partial: true, err: context.Canceled, errCtx: canceledCtx, slowCtx: canceledCtx, wantUnclosed: 0},
		{name: "model canceled its own request", ctx: context.Background(), partial: true, err: context.Canceled, errCtx: context.Background(), slowCtx: context.Background(), wantUnclosed: 2},
		{name: "agent canceled its sub-agent's context", ctx: context.Background(), partial: true, err: context.Canceled, errCtx: canceledCtx, slowCtx: context.Background(), wantUnclosed: 2},
		{name: "no partial event", ctx: context.Background(), partial: false, err: context.Canceled, errCtx: canceledCtx, slowCtx: canceledCtx, wantUnclosed: 2},
		{name: "other error after a partial event", ctx: context.Background(), partial: true, err: errors.New("model failed"), errCtx: canceledCtx, slowCtx: canceledCtx, wantUnclosed: 2},
		{name: "invocation canceled", ctx: canceledCtx, partial: true, err: context.Canceled, errCtx: canceledCtx, slowCtx: canceledCtx, wantUnclosed: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := newTraceManager()
			par := identity{invocationID: "inv-cancel", agentName: "par"}
			bot := identity{invocationID: "inv-cancel", agentName: "bot", branch: "par.bot"}
			slow := identity{invocationID: "inv-cancel", agentName: "slow", branch: "par.slow"}
			tm.invocationSpan(tt.ctx, par, nil)
			tm.pushAgentSpan(tt.ctx, par)
			tm.pushAgentSpan(tt.ctx, bot)
			tm.pushAgentSpan(tt.slowCtx, slow)
			// slow responded once, so only its error can fail it.
			tm.pushLLMSpan(tt.ctx, slow)
			tm.popLLMSpan(tt.ctx, slow, true)
			tm.pushLLMSpan(tt.ctx, slow)
			if tt.partial {
				tm.recordStreamed(tt.ctx, bot, "bot", bot.branch, true)
			} else {
				tm.completeResponse(tt.ctx, bot, "bot", bot.branch, nil)
			}
			tm.recordError(tt.errCtx, slow, llmKey(slow), tt.err)
			tm.popLLMSpan(tt.ctx, slow, false)

			unclosed, err := tm.popUnclosedAgentSpans(tt.ctx, par)
			if (err != nil) != (tt.wantUnclosed > 0) || len(unclosed) != tt.wantUnclosed {
				t.Errorf("popUnclosedAgentSpans = (%d spans, %v), want %d spans", len(unclosed), err, tt.wantUnclosed)
			}
		})
	}
}

// TestTraceManager_StopPointNodePathDiffersFromParent verifies that an agent
// activation on a node path that differs from its agent parent's is a stop
// point, so a context.Canceled error recorded for it after a partial event
// stop is attributed to the stop and does not fail the invocation. The
// stopPoint literal in pushAgentSpan includes id.path != parent.path for
// this case; deleting that clause makes this test fail.
func TestTraceManager_StopPointNodePathDiffersFromParent(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-stoppoint-path"
	// agent runs right under the invocation with no branch of its own.
	agent := identity{invocationID: invID, agentName: "agent", branch: ""}
	// helper runs on a distinct node path under agent, in the same branch:
	// only id.path != parent.path makes it a stop point.
	helper := identity{invocationID: invID, agentName: "helper", branch: "", path: "agent/helper@1"}

	// agent's context is not canceled, but helper's own context is.
	agentCtx, agentCancel := context.WithCancel(ctx)
	t.Cleanup(agentCancel)
	helperCtx, helperCancel := context.WithCancel(agentCtx)
	t.Cleanup(helperCancel)

	tm := newTraceManager()
	tm.invocationSpan(ctx, agent, nil)
	tm.pushAgentSpan(agentCtx, agent)
	// agent streams a partial event with content.
	tm.pushLLMSpan(agentCtx, agent)
	tm.recordStreamed(ctx, agent, "agent", agent.branch, true)
	// helper runs on its own node path and then gets canceled.
	tm.pushAgentSpan(helperCtx, helper)
	tm.pushLLMSpan(helperCtx, helper)
	helperCancel()
	tm.recordError(helperCtx, helper, llmKey(helper), context.Canceled)
	tm.popLLMSpan(helperCtx, helper, false)

	if _, err := tm.popUnclosedAgentSpans(ctx, agent); err != nil {
		t.Errorf("popUnclosedAgentSpans error = %v, want nil: helper's context.Canceled should be attributed to the stop, not a failure", err)
	}
}

// TestTraceManager_RecordErrorWrapsNonCollateralCustomCancelCause verifies that
// when a sub-agent on the same branch and node path as its parent (stopPoint ==
// false) has its private sub-context canceled with a custom cause via
// context.WithCancelCause and records context.Canceled on its model call,
// collateralFailureCause reports collateral == false and recordError wraps
// context.Canceled with the custom cause rather than treating the error as
// collateral sibling-failure teardown.
func TestTraceManager_RecordErrorWrapsNonCollateralCustomCancelCause(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-noncollateral-cause"
	root := identity{invocationID: invID, agentName: "root", branch: "", path: ""}
	sub := identity{invocationID: invID, agentName: "sub", branch: "", path: ""}

	errCustom := errors.New("custom sub-agent timeout")
	subCtx, cancelSub := context.WithCancelCause(ctx)
	t.Cleanup(func() { cancelSub(nil) })

	tm := newTraceManager()
	tm.invocationSpan(ctx, root, nil)
	tm.pushAgentSpan(ctx, root)
	tm.pushAgentSpan(subCtx, sub)
	tm.pushLLMSpan(subCtx, sub)
	tm.recordStreamed(ctx, sub, "sub", sub.branch, true)
	cancelSub(errCustom)
	tm.recordError(subCtx, sub, llmKey(sub), context.Canceled)
	tm.popLLMSpan(subCtx, sub, false)

	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, root)
	if !errors.Is(termErr, context.Canceled) || !errors.Is(termErr, errCustom) {
		t.Fatalf("popUnclosedAgentSpans terminal error = %v, want wrapped context.Canceled and %v", termErr, errCustom)
	}
	if !strings.Contains(termErr.Error(), "context canceled") || !strings.Contains(termErr.Error(), errCustom.Error()) {
		t.Errorf("popUnclosedAgentSpans terminal error = %q, want both \"context canceled\" and %q", termErr.Error(), errCustom.Error())
	}
	if len(unclosed) != 2 {
		t.Errorf("popUnclosedAgentSpans unclosed = %d spans, want 2 ([root, sub])", len(unclosed))
	}
}

// TestRunner_ConcurrentWorkflowRetries_OrdersPurgedAgentErrorBeforeNextStarting
// is a concurrency smoke test exercising two concurrent workflow RetryConfig
// nodes that fail their first attempts and re-enter BeforeAgentCallback while
// ContentFormatter pauses on AGENT_ERROR outside inv.mu. When the plugin
// recorded the first attempt's failure (a model error), the node completion
// event lets the second attempt's start purge it, so each node appends its
// prior attempt's AGENT_ERROR row before its next attempt's AGENT_STARTING
// row. When the first attempt failed silently (its agent yielded an error),
// the event does not show that it stopped without responding, as a node can
// write the same metadata while it runs, so it is purged when the workflow
// agent ends instead, and its AGENT_ERROR row follows the second attempt's
// rows. Within-callback ordering is pinned by
// TestRunner_PurgedRetryAttemptTimestampPrecedesNextAttempt, and deterministic
// per-node isolation of purged attempts is pinned by
// TestTraceManager_PushAgentSpanReturnsOnlyOwnNodePurgedAttempt. Like the
// former, it needs node completion events.
func TestRunner_ConcurrentWorkflowRetries_OrdersPurgedAgentErrorBeforeNextStarting(t *testing.T) {
	requireADKNodeCompletionEvents(t)
	for _, recorded := range []bool{true, false} {
		name := "silent failure"
		if recorded {
			name = "recorded failure"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			var firstAgentError atomic.Bool
			p, requests, _ := setupTestPluginWithConfig(t, func(cfg *Config) {
				cfg.ContentFormatter = func(payload any, eventType string) (any, error) {
					if eventType == EventAgentError && firstAgentError.CompareAndSwap(false, true) {
						time.Sleep(25 * time.Millisecond)
					}
					return payload, nil
				}
			})

			errTransient := errors.New("transient attempt-1 failure")
			attempt1Ready := make(chan struct{})
			attempt2Ready := make(chan struct{})
			var attempt1Entered, attempt2Entered atomic.Int32
			var seenByAgent sync.Map

			gate, err := baseplugin.New(baseplugin.Config{
				Name: "retry_gate",
				BeforeAgentCallback: func(cbCtx agent.Context) (*genai.Content, error) {
					name := cbCtx.AgentName()
					if name != "agent_a" && name != "agent_b" {
						return nil, nil
					}
					cntPtr, _ := seenByAgent.LoadOrStore(name, &atomic.Int32{})
					if cntPtr.(*atomic.Int32).Add(1) == 2 {
						if attempt2Entered.Add(1) == 2 {
							close(attempt2Ready)
						}
						select {
						case <-attempt2Ready:
						case <-cbCtx.Done():
						}
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(retry_gate): %v", err)
			}

			makeRetryNode := func(name string) workflow.Node {
				if recorded {
					a, err := llmagent.New(llmagent.Config{Name: name, Model: &gatedFlakyLLM{
						enter: func() {
							if attempt1Entered.Add(1) == 2 {
								close(attempt1Ready)
							}
						},
						ready: attempt1Ready,
						err:   errTransient,
						reply: name + " ok",
					}})
					if err != nil {
						t.Fatalf("llmagent.New(%s): %v", name, err)
					}
					node, err := workflow.NewAgentNode(a, workflow.NodeConfig{
						RetryConfig: &workflow.RetryConfig{
							MaxAttempts:   2,
							InitialDelay:  time.Millisecond,
							MaxDelay:      time.Millisecond,
							BackoffFactor: 1.0,
						},
					})
					if err != nil {
						t.Fatalf("workflow.NewAgentNode(%s): %v", name, err)
					}
					return node
				}
				var attempts atomic.Int32
				a, err := agent.New(agent.Config{
					Name: name,
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							if attempts.Add(1) == 1 {
								if attempt1Entered.Add(1) == 2 {
									close(attempt1Ready)
								}
								select {
								case <-attempt1Ready:
								case <-ctx.Done():
									return
								}
								yield(nil, errTransient)
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = name
							ev.Branch = ctx.Branch()
							ev.LLMResponse = model.LLMResponse{Content: textContent("model", name+" ok"), TurnComplete: true}
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(%s): %v", name, err)
				}
				node, err := workflow.NewAgentNode(a, workflow.NodeConfig{
					RetryConfig: &workflow.RetryConfig{
						MaxAttempts:   2,
						InitialDelay:  time.Millisecond,
						MaxDelay:      time.Millisecond,
						BackoffFactor: 1.0,
					},
				})
				if err != nil {
					t.Fatalf("workflow.NewAgentNode(%s): %v", name, err)
				}
				return node
			}

			nodeA := makeRetryNode("agent_a")
			nodeB := makeRetryNode("agent_b")
			wfA, err := workflowagent.New(workflowagent.Config{
				Name:  "wf_a",
				Edges: []workflow.Edge{{From: workflow.Start, To: nodeA}},
			})
			if err != nil {
				t.Fatalf("workflowagent.New(wf_a): %v", err)
			}
			wfB, err := workflowagent.New(workflowagent.Config{
				Name:  "wf_b",
				Edges: []workflow.Edge{{From: workflow.Start, To: nodeB}},
			})
			if err != nil {
				t.Fatalf("workflowagent.New(wf_b): %v", err)
			}
			parRoot := newParallelAgent(t, "par_root", wfA, wfB)

			for _, runErr := range newTestRunner(t, parRoot, gate, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run yielded unexpected error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			onlyRowOfType(t, rows, EventInvocationCompleted)

			for _, name := range []string{"agent_a", "agent_b"} {
				var agentRows []map[string]any
				for _, r := range rows {
					if r["agent"] == name {
						et, _ := r["event_type"].(string)
						if et == EventAgentStarting || et == EventAgentError || et == EventAgentCompleted {
							agentRows = append(agentRows, r)
						}
					}
				}
				gotTypes := eventTypes(agentRows)
				// A recorded failure purges the first attempt when the second one
				// starts; a silent one only when the workflow agent ends.
				wantTypes := []string{EventAgentStarting, EventAgentError, EventAgentStarting, EventAgentCompleted}
				first, second := [2]int{0, 1}, [2]int{2, 3}
				if !recorded {
					wantTypes = []string{EventAgentStarting, EventAgentStarting, EventAgentCompleted, EventAgentError}
					first, second = [2]int{0, 3}, [2]int{1, 2}
				}
				if !slices.Equal(gotTypes, wantTypes) {
					t.Fatalf("%s lifecycle row order = %v, want %v", name, gotTypes, wantTypes)
				}
				span1Start, _ := agentRows[first[0]]["span_id"].(string)
				span1Err, _ := agentRows[first[1]]["span_id"].(string)
				span2Start, _ := agentRows[second[0]]["span_id"].(string)
				span2Done, _ := agentRows[second[1]]["span_id"].(string)
				if span1Start == "" || span1Start != span1Err {
					t.Errorf("%s attempt-1 span_id mismatch: start=%q error=%q", name, span1Start, span1Err)
				}
				if span2Start == "" || span2Start != span2Done {
					t.Errorf("%s attempt-2 span_id mismatch: start=%q completed=%q", name, span2Start, span2Done)
				}
				if span1Start == span2Start {
					t.Errorf("%s reused attempt-1 span_id %q on attempt 2", name, span1Start)
				}
			}
		})
	}
}

// adkCancelsParallelBranchesOnEarlyStop probes whether parallelagent cancels
// in-flight sub-agent contexts and awaits sub-agent teardown before returning
// when a caller stops reading early (#1186, google.golang.org/adk/v2 >= v2.1.0).
var adkCancelsParallelBranchesOnEarlyStop = sync.OnceValue(func() bool {
	entered := make(chan struct{})
	release := make(chan struct{})
	var canceled atomic.Bool
	waiter, err := agent.New(agent.Config{
		Name: "probe_waiter",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				close(entered)
				select {
				case <-ctx.Done():
					canceled.Store(true)
				case <-release:
				}
			}
		},
	})
	if err != nil {
		return true
	}
	emitter, err := agent.New(agent.Config{
		Name: "probe_emitter",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-entered
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "probe_emitter"
				ev.Branch = ctx.Branch()
				ev.LLMResponse = model.LLMResponse{Partial: true, Content: textContent("model", "chunk")}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		return true
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "probe_par", SubAgents: []agent.Agent{emitter, waiter}},
	})
	if err != nil {
		return true
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "probe", UserID: "u1", SessionID: "s1",
	}); err != nil {
		return true
	}
	r, err := runner.New(runner.Config{AppName: "probe", Agent: par, SessionService: svc})
	if err != nil {
		return true
	}
	for ev, runErr := range r.Run(context.Background(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil || (ev != nil && ev.Partial) {
			break
		}
	}
	got := canceled.Load()
	close(release)
	return got
})

// adkStopsSequentialOnSubAgentError probes whether sequentialagent stops
// iterating a failing sub-agent at its error without running the failed
// sub-agent's AfterAgentCallback (#1684, google.golang.org/adk/v2 >= v2.6.0).
var adkStopsSequentialOnSubAgentError = sync.OnceValue(func() bool {
	var afterRan atomic.Bool
	failer, err := agent.New(agent.Config{
		Name: "probe_failer",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("probe error"))
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				afterRan.Store(true)
				return nil, nil
			},
		},
	})
	if err != nil {
		return true
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "probe_seq", SubAgents: []agent.Agent{failer}},
	})
	if err != nil {
		return true
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "probe", UserID: "u1", SessionID: "s1",
	}); err != nil {
		return true
	}
	r, err := runner.New(runner.Config{AppName: "probe", Agent: seq, SessionService: svc})
	if err != nil {
		return true
	}
	for range r.Run(context.Background(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
	}
	return !afterRan.Load()
})

// adkIsInRepoModule reports whether the test binary is built from the in-repo
// google.golang.org/adk/v2 source (via go.work or a replace directive) rather
// than from a released module version, as under GOWORK=off. It looks at the
// source file the binary records for an ADK function: a released version is
// read from the module cache, whose directory for it ends in "@" and the
// version, and -trimpath keeps that "@" too. debug.ReadBuildInfo cannot tell
// the two apart, because a go test binary records no Deps. Against the in-repo
// module, all capability probes must succeed and a probe failure fails the
// test instead of skipping.
var adkIsInRepoModule = sync.OnceValue(func() bool {
	pc := reflect.ValueOf(agent.New).Pointer()
	fn := runtime.FuncForPC(pc)
	if fn == nil {
		return false
	}
	file, _ := fn.FileLine(pc)
	return file != "" && !strings.Contains(file, "google.golang.org/adk/v2@")
})

func skipOrFailADKProbe(t *testing.T, msg string) {
	t.Helper()
	if adkIsInRepoModule() {
		t.Fatalf("%s (capability probe failed against in-repo google.golang.org/adk/v2)", msg)
	}
	t.Skip(msg)
}

func requireADKParallelEarlyStopTeardown(t *testing.T) {
	t.Helper()
	if !adkCancelsParallelBranchesOnEarlyStop() {
		skipOrFailADKProbe(t, "requires google.golang.org/adk/v2 >= v2.1.0 (#1186 parallelagent early-stop teardown)")
	}
}

func requireADKStopOnSubAgentError(t *testing.T) {
	t.Helper()
	if !adkStopsSequentialOnSubAgentError() {
		skipOrFailADKProbe(t, "requires google.golang.org/adk/v2 > v2.6.0 (#1684 sequentialagent/loopagent stop on sub-agent error)")
	}
}

// adkResolvesAgentModeWithoutMutation probes whether ADK resolves agent mode
// per placement without mutating the shared LlmAgent struct (#1267,
// google.golang.org/adk/v2 >= v2.4.0; earlier releases mutate
// state.IncludeContents in RunLLMAgentAsNode, which races under -race when two
// tool calls invoke the same ModeSingleTurn sub-agent concurrently).
var adkResolvesAgentModeWithoutMutation = sync.OnceValue(func() bool {
	a, err := llmagent.New(llmagent.Config{Name: "probe_mode", Model: &scriptedLLM{}})
	if err != nil {
		return true
	}
	if _, err := workflow.NewAgentNode(a, workflow.NodeConfig{}); err != nil {
		return true
	}
	var got *session.Event
	probe, err := agent.New(agent.Config{
		Name: "probe_runner",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				got = llmagent.PrepareLLMAgentInput(a, ctx, "hi")
			}
		},
	})
	if err != nil {
		return true
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "probe", UserID: "u1", SessionID: "s1",
	}); err != nil {
		return true
	}
	r, err := runner.New(runner.Config{AppName: "probe", Agent: probe, SessionService: svc})
	if err != nil {
		return true
	}
	for range r.Run(context.Background(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
	}
	return got == nil
})

func requireADKRaceFreeSharedSingleTurnTool(t *testing.T) {
	t.Helper()
	if !adkResolvesAgentModeWithoutMutation() {
		skipOrFailADKProbe(t, "requires google.golang.org/adk/v2 >= v2.4.0 (#1267 race-free concurrent ModeSingleTurn RunLLMAgentAsNode)")
	}
}

// adkEmitsNodeCompletionEvents probes whether the workflow scheduler writes a
// node completion event, with the node outcome in its custom metadata, when a
// node run fails (#1572, google.golang.org/adk/v2 later than v2.6.0). Only
// such an event reports to the plugin that the failed attempt of a retried
// node ended before the next attempt starts (see recordNodeCompletion), which
// decides that the attempt is purged then, without its InvocationContext
// being read (see spanRecord.nodeRunEnded). Without it, the failed attempt is
// purged when its workflow agent ends instead (see endRestartedNodeRuns),
// after the next attempt's rows, and the failed attempt of a ParallelAgent
// node is not known to have waited for its branches (see spanRecord.joins).
var adkEmitsNodeCompletionEvents = sync.OnceValue(func() bool {
	var calls atomic.Int32
	flaky := workflow.NewFunctionNode("probe_flaky", func(agent.Context, any) (string, error) {
		if calls.Add(1) == 1 {
			return "", errors.New("probe error")
		}
		return "ok", nil
	}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{
		MaxAttempts:   2,
		InitialDelay:  time.Millisecond,
		MaxDelay:      time.Millisecond,
		BackoffFactor: 1.0,
	}})
	wf, err := workflowagent.New(workflowagent.Config{Name: "probe_wf", Edges: workflow.Chain(workflow.Start, flaky)})
	if err != nil {
		return true
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "probe", UserID: "u1", SessionID: "s1",
	}); err != nil {
		return true
	}
	r, err := runner.New(runner.Config{AppName: "probe", Agent: wf, SessionService: svc})
	if err != nil {
		return true
	}
	var emitted bool
	for ev := range r.Run(context.Background(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if ev == nil {
			continue
		}
		if _, ok := ev.CustomMetadata["adk.workflow.node_outcome"]; ok {
			emitted = true
		}
	}
	return emitted
})

func requireADKNodeCompletionEvents(t *testing.T) {
	t.Helper()
	if !adkEmitsNodeCompletionEvents() {
		skipOrFailADKProbe(t, "requires google.golang.org/adk/v2 workflow node completion events (#1572, after v2.6.0)")
	}
}

// newCacheHitAgent returns an llmagent that its BeforeAgentCallback answers.
// ADK writes the answer as an event, marks the sub-agent's InvocationContext
// as ended, and runs no AfterAgentCallback for it.
func newCacheHitAgent(t *testing.T) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:  "cached",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never sent")}}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) { return textContent("model", "cache hit"), nil },
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(cached): %v", err)
	}
	return a
}

// TestRunner_ParallelWorkerPipelineWithCachedStep_EmitsInvocationCompleted runs
// a workflow.ParallelWorker over a sequential pipeline whose first step is a
// cache hit. ParallelWorker keeps only the items' outputs, so the plugin never
// sees the cache hit's event, including when RetryConfig retries a failed item.
func TestRunner_ParallelWorkerPipelineWithCachedStep_EmitsInvocationCompleted(t *testing.T) {
	t.Run("pipeline_succeeds_on_first_attempt", func(t *testing.T) {
		var logBuf bytes.Buffer
		origOut := log.Writer()
		log.SetOutput(&logBuf)
		t.Cleanup(func() { log.SetOutput(origOut) })

		p, requests, _ := setupTestPlugin(t)
		summarize, err := llmagent.New(llmagent.Config{
			Name: "summarize",
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				{Content: textContent("model", "summary")},
				{Content: textContent("model", "summary")},
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(summarize): %v", err)
		}
		pipeline, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{newCacheHitAgent(t), summarize}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New: %v", err)
		}
		items := workflow.NewFunctionNode("items", func(agent.Context, any) ([]string, error) {
			return []string{"a", "b"}, nil
		}, workflow.NodeConfig{})
		node, err := workflow.NewAgentNode(pipeline, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode: %v", err)
		}
		perItem, err := workflow.NewParallelWorker("per_item", node, 0, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewParallelWorker: %v", err)
		}
		root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, items, perItem)})
		if err != nil {
			t.Fatalf("workflowagent.New: %v", err)
		}
		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}
		requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
		if strings.Contains(logBuf.String(), "not supported for callback context") {
			t.Errorf("unexpected callback context log warning: %q", logBuf.String())
		}
	})

	t.Run("retry_on_parallel_worker_pipeline_with_cached_and_flaky_steps", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		flaky, err := llmagent.New(llmagent.Config{
			Name:  "flaky",
			Model: &flakyOnFirstPassLLM{err: errors.New("attempt 1 flaky 503"), reply: "summary ok"},
		})
		if err != nil {
			t.Fatalf("llmagent.New(flaky): %v", err)
		}
		pipeline, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{newCacheHitAgent(t), flaky}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New: %v", err)
		}
		items := workflow.NewFunctionNode("items", func(agent.Context, any) ([]string, error) {
			return []string{"a", "b"}, nil
		}, workflow.NodeConfig{})
		node, err := workflow.NewAgentNode(pipeline, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode: %v", err)
		}
		perItem, err := workflow.NewParallelWorker("per_item", node, 0, workflow.NodeConfig{
			RetryConfig: &workflow.RetryConfig{
				MaxAttempts:   2,
				InitialDelay:  time.Millisecond,
				MaxDelay:      time.Millisecond,
				BackoffFactor: 1.0,
			},
		})
		if err != nil {
			t.Fatalf("workflow.NewParallelWorker: %v", err)
		}
		root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, items, perItem)})
		if err != nil {
			t.Fatalf("workflowagent.New: %v", err)
		}
		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
		}
		onlyRowOfType(t, rows, EventInvocationCompleted)
		if got, want := agentErrorCounts(rows), map[string]int{"flaky": 1, "pipeline": 1}; !maps.Equal(got, want) {
			t.Errorf("AGENT_ERROR counts = %v, want %v (cached step must not get AGENT_ERROR)", got, want)
		}
	})
}

// TestRunner_CustomParentOverSequenceWithCachedStep_EmitsInvocationCompleted
// runs custom-parent topologies that drain a BeforeAgentCallback-answered
// sub-agent without forwarding its event (both through an intermediate
// sequential agent and directly with or without an event from the custom
// parent).
func TestRunner_CustomParentOverSequenceWithCachedStep_EmitsInvocationCompleted(t *testing.T) {
	t.Run("custom_parent_over_sequence_with_own_answer", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		seq, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{newCacheHitAgent(t)}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New: %v", err)
		}
		parent, err := agent.New(agent.Config{
			Name:      "parent",
			SubAgents: []agent.Agent{seq},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, err := range seq.Run(ctx) {
						if err != nil {
							yield(nil, err)
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Content = textContent("model", "parent answer")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(parent): %v", err)
		}
		for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}
		requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
	})

	t.Run("guard_over_cached_without_own_event", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		cached := newCacheHitAgent(t)
		guard, err := agent.New(agent.Config{
			Name:      "guard",
			SubAgents: []agent.Agent{cached},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, err := range cached.Run(ctx) {
						if err != nil {
							yield(nil, err)
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(guard): %v", err)
		}
		for _, err := range newTestRunner(t, guard, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}
		requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
	})

	t.Run("seq_root_with_silent_guard_over_cached_then_final_llm", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		cached := newCacheHitAgent(t)
		guard, err := agent.New(agent.Config{
			Name:      "guard",
			SubAgents: []agent.Agent{cached},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, err := range cached.Run(ctx) {
						if err != nil {
							yield(nil, err)
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(guard): %v", err)
		}
		finalLLM, err := llmagent.New(llmagent.Config{
			Name:  "final_llm",
			Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "done"), TurnComplete: true}}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(final_llm): %v", err)
		}
		seqRoot, err := sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{Name: "seq_root", SubAgents: []agent.Agent{guard, finalLLM}},
		})
		if err != nil {
			t.Fatalf("sequentialagent.New(seq_root): %v", err)
		}
		for _, err := range newTestRunner(t, seqRoot, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("r.Run error: %v", err)
			}
		}
		requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
	})
}

// TestRunner_WorkflowAgentRunAgainAfterForwardedNodeError_EmitsInvocationError
// verifies that a node error which reached the caller still fails the run when
// the workflow agent runs again and its node starts on the same path, whether
// or not the caller read the first run of the workflow agent to its end.
func TestRunner_WorkflowAgentRunAgainAfterForwardedNodeError_EmitsInvocationError(t *testing.T) {
	for _, tc := range []struct {
		name string
		// stopAtError makes guard stop reading the first run of wf at the
		// node error. That run then gets no AfterAgentCallback, so its span
		// is still open when wf runs again.
		stopAtError bool
	}{
		{name: "guard reads wf to its end", stopAtError: false},
		{name: "guard stops reading wf at the node error", stopAtError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			modelErr := errors.New("pass 1 model failure 503")
			x, err := llmagent.New(llmagent.Config{Name: "x", Model: &flakyOnFirstPassLLM{err: modelErr, reply: "ok on pass 2"}})
			if err != nil {
				t.Fatalf("llmagent.New(x): %v", err)
			}
			node, err := workflow.NewAgentNode(x, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode(x): %v", err)
			}
			wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: []workflow.Edge{{From: workflow.Start, To: node}}})
			if err != nil {
				t.Fatalf("workflowagent.New(wf): %v", err)
			}
			// guard forwards the events and errors of wf, stops at the node
			// error if tc.stopAtError, and then runs wf once more.
			guard, err := agent.New(agent.Config{
				Name:      "guard",
				SubAgents: []agent.Agent{wf},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for range 2 {
							for ev, err := range wf.Run(ctx) {
								if !yield(ev, err) {
									return
								}
								if err != nil && tc.stopAtError {
									break
								}
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(guard): %v", err)
			}
			var errs []error
			for _, runErr := range newTestRunner(t, guard, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					errs = append(errs, runErr)
				}
			}
			if len(errs) != 1 || !errors.Is(errs[0], modelErr) {
				t.Fatalf("r.Run errors = %v, want [%v]", errs, modelErr)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got INVOCATION_COMPLETED although the caller received %v; event types: %v", modelErr, eventTypes(rows))
			}
			if got := onlyRowOfType(t, rows, EventInvocationError)["error_message"]; got != modelErr.Error() {
				t.Errorf("INVOCATION_ERROR error_message = %v, want %q", got, modelErr.Error())
			}
		})
	}
}

// TestTraceManager_PushAgentSpanReturnsOnlyOwnNodePurgedAttempt verifies
// deterministically that when two workflow nodes under the same live parent
// both have failed attempts on agentStack, re-entering node A returns only
// node A's purged attempt and re-entering node B next returns only node B's
// purged attempt.
func TestTraceManager_PushAgentSpanReturnsOnlyOwnNodePurgedAttempt(t *testing.T) {
	tm := newTraceManager()
	ctx := t.Context()
	wfID := identity{invocationID: "inv-purge-own", agentName: "wf"}
	nodeAID := identity{invocationID: "inv-purge-own", agentName: "agent_a", path: "wf@1/node_a@1"}
	nodeBID := identity{invocationID: "inv-purge-own", agentName: "agent_b", path: "wf@1/node_b@1"}
	errA := errors.New("node A attempt 1 boom")
	errB := errors.New("node B attempt 1 boom")

	tm.pushAgentSpan(ctx, wfID)
	spanA1, purgedInitA := tm.pushAgentSpan(ctx, nodeAID)
	if len(purgedInitA) != 0 {
		t.Fatalf("initial pushAgentSpan(nodeA) purged = %v, want empty", purgedInitA)
	}
	tm.recordError(ctx, nodeAID, llmKey(nodeAID), errA)

	spanB1, purgedInitB := tm.pushAgentSpan(ctx, nodeBID)
	if len(purgedInitB) != 0 {
		t.Fatalf("initial pushAgentSpan(nodeB) purged = %v, want empty", purgedInitB)
	}
	tm.recordError(ctx, nodeBID, llmKey(nodeBID), errB)

	spanA2, purgedA := tm.pushAgentSpan(ctx, nodeAID)
	if len(purgedA) != 1 || purgedA[0].agentName != "agent_a" || purgedA[0].span.spanID != spanA1.spanID || !errors.Is(purgedA[0].err, errA) {
		t.Fatalf("retry pushAgentSpan(nodeA) purged = %+v, want [agent_a span=%s err=%v]", purgedA, spanA1.spanID, errA)
	}
	if spanA2.spanID == spanA1.spanID {
		t.Errorf("nodeA attempt 2 reused attempt 1 spanID %q", spanA1.spanID)
	}

	spanB2, purgedB := tm.pushAgentSpan(ctx, nodeBID)
	if len(purgedB) != 1 || purgedB[0].agentName != "agent_b" || purgedB[0].span.spanID != spanB1.spanID || !errors.Is(purgedB[0].err, errB) {
		t.Fatalf("retry pushAgentSpan(nodeB) purged = %+v, want [agent_b span=%s err=%v]", purgedB, spanB1.spanID, errB)
	}
	if spanB2.spanID == spanB1.spanID {
		t.Errorf("nodeB attempt 2 reused attempt 1 spanID %q", spanB1.spanID)
	}
}

// TestTraceManager_CompleteResponsePreservesCollateralError pins the
// !e.collateral guard in completeResponse: a final event of a parallel branch
// that recorded a collateral context.Canceled error must not delete that
// collateral error from unrecoveredErrors, so an enclosing custom agent whose
// own final event clears retiredCause still ends with the sibling branch's
// error.
func TestTraceManager_CompleteResponsePreservesCollateralError(t *testing.T) {
	tm := newTraceManager()
	ctx := t.Context()
	guardID := identity{invocationID: "inv-m2", agentName: "guard"}
	parID := identity{invocationID: "inv-m2", agentName: "par"}
	streamerID := identity{invocationID: "inv-m2", agentName: "streamer", branch: "par.streamer"}
	errSibling := errors.New("sibling branch failed")

	tm.pushAgentSpan(ctx, guardID)
	tm.pushAgentSpan(ctx, parID)
	branchCtx, cancelBranch := context.WithCancelCause(ctx)
	tm.pushAgentSpan(branchCtx, streamerID)

	cancelBranch(errSibling)
	tm.recordError(branchCtx, streamerID, llmKey(streamerID), context.Canceled)

	// A final event from streamer settles streamer, and later guard's own final
	// event clears retiredCause after par retires; only the preserved collateral
	// error retains errSibling for popUnclosedAgentSpans.
	tm.completeResponse(ctx, streamerID, "streamer", "par.streamer", nil)
	tm.popAgentSpan(branchCtx, streamerID)
	tm.popAgentSpan(ctx, parID)
	tm.completeResponse(ctx, guardID, "guard", "", nil)

	_, termErr := tm.popUnclosedAgentSpans(ctx, guardID)
	if !errors.Is(termErr, errSibling) {
		t.Fatalf("popUnclosedAgentSpans terminal error = %v, want %v", termErr, errSibling)
	}
}

// TestTraceManager_FailuresLockedUsesRetiredCauseOnRespondedSpan pins the
// rec.firstRetiredCause use in failuresLocked: when a responded child span
// under a retried composite node carries retiredCause from a retired parallel
// branch, purging the composite node attributes that retiredCause to both the
// responded child and the composite node rather than omitting the child.
func TestTraceManager_FailuresLockedUsesRetiredCauseOnRespondedSpan(t *testing.T) {
	tm := newTraceManager()
	ctx := t.Context()
	wfID := identity{invocationID: "inv-m3", agentName: "wf"}
	outerID := identity{invocationID: "inv-m3", agentName: "outer", path: "wf@1/outer@1"}
	guardID := identity{invocationID: "inv-m3", agentName: "guard", path: "wf@1/outer@1"}
	branchID := identity{invocationID: "inv-m3", agentName: "branch", path: "wf@1/outer@1", branch: "par.branch"}
	errBranch := errors.New("parallel branch failed")

	tm.pushAgentSpan(ctx, wfID)
	tm.pushAgentSpan(ctx, outerID)
	tm.pushAgentSpan(ctx, guardID)
	branchCtx, cancelBranch := context.WithCancelCause(ctx)
	tm.pushAgentSpan(branchCtx, branchID)

	tm.completeResponse(branchCtx, branchID, "branch", "par.branch", nil)
	cancelBranch(errBranch)
	tm.popAgentSpan(branchCtx, branchID)

	_, purged := tm.pushAgentSpan(ctx, outerID)
	if len(purged) != 2 {
		t.Fatalf("purged len = %d (%+v), want 2 ([guard, outer])", len(purged), purged)
	}
	if purged[0].agentName != "guard" || !errors.Is(purged[0].err, errBranch) {
		t.Errorf("purged[0] = {agent:%q err:%v}, want {agent:\"guard\" err:%v}", purged[0].agentName, purged[0].err, errBranch)
	}
	if purged[1].agentName != "outer" || !errors.Is(purged[1].err, errBranch) {
		t.Errorf("purged[1] = {agent:%q err:%v}, want {agent:\"outer\" err:%v}", purged[1].agentName, purged[1].err, errBranch)
	}
}

// TestTraceManager_RetireAgentSkipsCollateralError pins the e.collateral
// exclusion in retireAgentLocked: when a parallel parent retires while a
// responded child branch has a collateral context.Canceled error recorded, the
// child branch is retired with its parent rather than kept open on agentStack.
func TestTraceManager_RetireAgentSkipsCollateralError(t *testing.T) {
	tm := newTraceManager()
	outerCtx, cancelOuter := context.WithCancel(t.Context())
	defer cancelOuter()

	parID := identity{invocationID: "inv-m5", agentName: "par"}
	streamerID := identity{invocationID: "inv-m5", agentName: "streamer", branch: "par.streamer"}
	errSibling := errors.New("sibling branch failed")

	tm.pushAgentSpan(outerCtx, parID)
	branchCtx, cancelBranch := context.WithCancelCause(outerCtx)
	tm.pushAgentSpan(branchCtx, streamerID)
	tm.completeResponse(branchCtx, streamerID, "streamer", "par.streamer", nil)

	cancelBranch(errSibling)
	tm.recordError(branchCtx, streamerID, llmKey(streamerID), context.Canceled)

	tm.popAgentSpan(outerCtx, parID)
	inv := tm.invocation(outerCtx, parID)
	if inv.hasOpenLateSpans() {
		t.Fatal("inv.hasOpenLateSpans() = true after popAgentSpan(par); responded child with collateral error was not retired")
	}

	cancelOuter()
	unclosed, _ := tm.popUnclosedAgentSpans(outerCtx, parID)
	if len(unclosed) != 0 {
		t.Fatalf("popUnclosedAgentSpans unclosed = %+v, want empty", unclosed)
	}
}

// liveTestLLM is a model for live runs whose genai client talks to a fake
// Live server; RunLive calls Client instead of GenerateContent.
type liveTestLLM struct {
	client *genai.Client
}

func (m *liveTestLLM) Name() string { return "live-test-model" }

func (m *liveTestLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("liveTestLLM supports live runs only"))
	}
}

func (m *liveTestLLM) Client() *genai.Client { return m.client }

// newFakeLiveServer starts a loopback WebSocket server that speaks enough of
// the genai Live API protocol for RunLive: it reads the client's setup
// message, sends setupComplete and then messages, and reads until the client
// closes the connection. It returns the server's ws:// URL.
func newFakeLiveServer(t *testing.T, messages ...string) string {
	t.Helper()
	var (
		mu    sync.Mutex
		conns []net.Conn
		wg    sync.WaitGroup
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Header.Get("Sec-WebSocket-Key")
		if key == "" {
			http.Error(w, "not a WebSocket handshake", http.StatusBadRequest)
			return
		}
		// Added while the request is still active, so before ts.Close
		// returns in the cleanup.
		wg.Add(1)
		defer wg.Done()
		conn, rw, err := http.NewResponseController(w).Hijack()
		if err != nil {
			t.Errorf("Hijack: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		mu.Lock()
		conns = append(conns, conn)
		mu.Unlock()
		accept := sha1.Sum([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
		_, _ = rw.WriteString("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
			"Sec-WebSocket-Accept: " + base64.StdEncoding.EncodeToString(accept[:]) + "\r\n\r\n")
		if err := rw.Flush(); err != nil {
			t.Errorf("writing the handshake response: %v", err)
			return
		}
		if _, err := readWSMessage(rw.Reader); err != nil {
			t.Errorf("reading the setup message: %v", err)
			return
		}
		for _, msg := range append([]string{`{"setupComplete":{}}`}, messages...) {
			if err := writeWSText(rw.Writer, msg); err != nil {
				t.Errorf("writing %s: %v", msg, err)
				return
			}
		}
		if err := rw.Flush(); err != nil {
			t.Errorf("flushing the server messages: %v", err)
			return
		}
		for {
			if _, err := readWSMessage(rw.Reader); err != nil {
				return
			}
		}
	}))
	t.Cleanup(func() {
		ts.Close()
		mu.Lock()
		for _, conn := range conns {
			_ = conn.Close()
		}
		mu.Unlock()
		wg.Wait()
	})
	return "ws" + strings.TrimPrefix(ts.URL, "http")
}

// readWSMessage reads a data message from a WebSocket client, unmasking and
// joining its frames and skipping ping and pong frames. It returns io.EOF on
// a close frame.
func readWSMessage(r *bufio.Reader) ([]byte, error) {
	var msg []byte
	for {
		var hdr [2]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return nil, err
		}
		n := uint64(hdr[1] & 0x7f)
		switch n {
		case 126:
			var ext [2]byte
			if _, err := io.ReadFull(r, ext[:]); err != nil {
				return nil, err
			}
			n = uint64(binary.BigEndian.Uint16(ext[:]))
		case 127:
			var ext [8]byte
			if _, err := io.ReadFull(r, ext[:]); err != nil {
				return nil, err
			}
			n = binary.BigEndian.Uint64(ext[:])
		}
		var mask [4]byte
		if hdr[1]&0x80 != 0 {
			if _, err := io.ReadFull(r, mask[:]); err != nil {
				return nil, err
			}
		}
		payload := make([]byte, n)
		if _, err := io.ReadFull(r, payload); err != nil {
			return nil, err
		}
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
		switch hdr[0] & 0x0f {
		case 0x8:
			return nil, io.EOF
		case 0x9, 0xa:
			continue
		}
		msg = append(msg, payload...)
		if hdr[0]&0x80 != 0 {
			return msg, nil
		}
	}
}

// writeWSText writes msg to a WebSocket client as an unmasked text frame.
func writeWSText(w *bufio.Writer, msg string) error {
	hdr := []byte{0x81}
	switch n := len(msg); {
	case n < 126:
		hdr = append(hdr, byte(n))
	case n <= 0xffff:
		hdr = binary.BigEndian.AppendUint16(append(hdr, 126), uint16(n))
	default:
		hdr = binary.BigEndian.AppendUint64(append(hdr, 127), uint64(n))
	}
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	_, err := w.WriteString(msg)
	return err
}

// TestTraceManager_LiveToolCachedChildStopRace stops a live run while a tool
// still runs a sub-agent whose BeforeAgentCallback answered for it. ADK marks
// the sub-agent's InvocationContext as ended on the tool's goroutine without
// synchronizing with AfterRun on the caller's goroutine, so AfterRun must not
// read that state (go test -race reports the data race otherwise). Without
// it, AfterRun cannot tell the open sub-agent apart from one that failed
// silently while the tool runs, so the run is reported with AGENT_ERROR for
// the sub-agent and INVOCATION_ERROR.
func TestTraceManager_LiveToolCachedChildStopRace(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	answered := make(chan struct{})
	release := make(chan struct{})
	toolDone := make(chan struct{})

	child, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(answered)
				return textContent("model", "cache hit"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}
	runChild, err := functiontool.New(functiontool.Config{Name: "run_child", Description: "Runs cached_child"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			// The tool context does not support running an agent; its
			// underlying context does.
			for _, err := range child.Run(ctx.WithICDelta(nil)) {
				if err != nil {
					return nil, err
				}
			}
			<-release
			return map[string]any{"ok": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(run_child): %v", err)
	}
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    newFakeLiveServer(t, `{"toolCall":{"functionCalls":[{"id":"call-1","name":"run_child","args":{}}]}}`),
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "live_root",
		Model: &liveTestLLM{client: client},
		Tools: []tool.Tool{runChild},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				close(toolDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(live_root): %v", err)
	}

	// Holds the event that calls the tool until the sub-agent has started, so
	// that OnEventCallback gets the root agent's event, which has no open span
	// in a live run, while the sub-agent runs on the root agent's branch and
	// node path: the event must not mark the sub-agent as responded.
	var heldTooLong atomic.Bool
	hold, err := baseplugin.New(baseplugin.Config{
		Name: "hold_tool_call_event",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev == nil || ev.Author != "live_root" || ev.Content == nil || !slices.ContainsFunc(ev.Content.Parts, func(p *genai.Part) bool { return p.FunctionCall != nil }) {
				return nil, nil
			}
			select {
			case <-answered:
			case <-time.After(10 * time.Second):
				heldTooLong.Store(true)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(hold_tool_call_event): %v", err)
	}

	liveSess, events, err := newTestRunner(t, root, hold, p).RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("r.RunLive: %v", err)
	}
	// Stop the live run once the BeforeAgentCallback has answered, while the
	// tool's goroutine goes on to end cached_child's InvocationContext.
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		select {
		case <-answered:
		case <-time.After(10 * time.Second):
		}
		_ = liveSess.Close()
	}()
	for _, err := range events {
		if err != nil {
			t.Errorf("live event error: %v", err)
		}
	}
	// The tool call is still open when AfterRun runs: it returns only after
	// the invocation's terminal row.
	rows := collectRows(t, requests, untilInvocationEnd)
	close(release)
	select {
	case <-toolDone:
	case <-time.After(10 * time.Second):
		t.Fatal("run_child did not return")
	}
	<-stopped
	if heldTooLong.Load() {
		t.Error("cached_child did not start while the event that called run_child was held")
	}

	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Errorf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationError)
	if got, want := agentErrorCounts(rows), map[string]int{"cached_child": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR counts = %v, want %v", got, want)
	}
}

// TestTraceManager_CachedChildStopBeforeParentRetires retires a custom parent
// that drained a sub-agent while a sibling branch's tool call (started before
// the sub-agent) is still open, then lets that tool call return and stops
// reading at the sibling branch's answer before the parallel agent ends. The
// sub-agent's event never reaches OnEventCallback, the open tool call keeps
// the sub-agent's span on agentStack when the parent retires, and returning
// from the tool call settles the sub-agent's ended InvocationContext at a real
// completion synchronization point so popUnclosedAgentSpans sees responded()
// via contextEnded (while an unanswered sub-agent still reports an error).
func TestTraceManager_CachedChildStopBeforeParentRetires(t *testing.T) {
	for _, tc := range []struct {
		name      string
		newCached func(t *testing.T) agent.Agent
		wantErr   bool
	}{
		{name: "llm sub-agent", newCached: newCacheHitAgent},
		{name: "custom sub-agent", newCached: func(t *testing.T) agent.Agent {
			t.Helper()
			a, err := agent.New(agent.Config{
				Name: "cached",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) { return textContent("model", "cache hit"), nil },
				},
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(cached): %v", err)
			}
			return a
		}},
		{name: "unanswered custom sub-agent", wantErr: true, newCached: func(t *testing.T) agent.Agent {
			t.Helper()
			a, err := agent.New(agent.Config{
				Name: "cached",
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						yield(nil, errors.New("silent child error"))
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(cached): %v", err)
			}
			return a
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			toolStarted := make(chan struct{})
			parentDone := make(chan struct{})
			cached := tc.newCached(t)
			parent, err := agent.New(agent.Config{
				Name:      "parent",
				SubAgents: []agent.Agent{cached},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						answer := session.NewEvent(ctx, ctx.InvocationID())
						answer.Author = "parent"
						answer.Branch = ctx.Branch()
						answer.Content = textContent("model", "parent answer")
						if !yield(answer, nil) {
							return
						}
						// Wait until the sibling branch's tool call has
						// started so call.seq < cached.seq holds and keeps
						// cached on agentStack when parent retires.
						select {
						case <-toolStarted:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("the wait tool did not start"))
							return
						}
						// The parent drops the sub-agent's event (or swallows
						// its error in the unanswered subtest).
						for _, err := range cached.Run(ctx) {
							if err != nil {
								break
							}
						}
					}
				},
				AfterAgentCallbacks: []agent.AfterAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						close(parentDone)
						return nil, nil
					},
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}
			wait, err := functiontool.New(functiontool.Config{Name: "wait", Description: "Waits for parent to end"},
				func(agent.Context, struct{}) (map[string]any, error) {
					close(toolStarted)
					select {
					case <-parentDone:
						return map[string]any{"ok": true}, nil
					case <-time.After(10 * time.Second):
						return nil, errors.New("parent did not end")
					}
				})
			if err != nil {
				t.Fatalf("functiontool.New(wait): %v", err)
			}
			waiter, err := llmagent.New(llmagent.Config{
				Name: "waiter",
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					{Content: genai.NewContentFromParts([]*genai.Part{genai.NewPartFromFunctionCall("wait", map[string]any{})}, genai.RoleModel)},
					{Content: textContent("model", "done"), TurnComplete: true},
				}},
				Tools: []tool.Tool{wait},
			})
			if err != nil {
				t.Fatalf("llmagent.New(waiter): %v", err)
			}
			par, err := parallelagent.New(parallelagent.Config{
				AgentConfig: agent.Config{
					Name:      "par",
					SubAgents: []agent.Agent{parent, waiter},
				},
			})
			if err != nil {
				t.Fatalf("parallelagent.New(par): %v", err)
			}

			var sawDone bool
			for ev, err := range newTestRunner(t, par, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
				// Stop at the waiter's answer, after popToolSpan settled
				// cached and before the parallel agent retires cached.
				if ev != nil && ev.Author == "waiter" && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "done" {
					sawDone = true
					break
				}
			}
			if !sawDone {
				t.Fatal("the run ended without the waiter's answer")
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if tc.wantErr {
				onlyRowOfType(t, rows, EventInvocationError)
				if got, want := agentErrorCounts(rows), map[string]int{"cached": 1}; !maps.Equal(got, want) {
					t.Errorf("AGENT_ERROR counts = %v, want %v", got, want)
				}
				return
			}
			requireInvocationCompletedOK(t, rows)
		})
	}
}

// TestTraceManager_CachedChildRetiredWhileOtherBranchToolRuns ends a custom
// parent that drained a sub-agent whose BeforeAgentCallback answered while a
// tool call of another branch of a parallel agent is still open. A tool call
// that started before the sub-agent may run it (see launcherOpenLocked), so
// the sub-agent's ended InvocationContext is read when the parent ends, or
// else when the tool call returns, and a caller that stops reading after that
// and before the parallel agent ends gets INVOCATION_COMPLETED.
func TestTraceManager_CachedChildRetiredWhileOtherBranchToolRuns(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	toolStarted := make(chan struct{})
	parentDone := make(chan struct{})
	cached := newCacheHitAgent(t)
	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{cached},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// The parent drops the sub-agent's event.
				for _, err := range cached.Run(ctx) {
					if err != nil {
						yield(nil, err)
						return
					}
				}
				// The parent ends only once the other branch's tool call is
				// open.
				select {
				case <-toolStarted:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("the wait tool did not start"))
				}
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(parentDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}
	wait, err := functiontool.New(functiontool.Config{Name: "wait", Description: "Waits for parent to end"},
		func(agent.Context, struct{}) (map[string]any, error) {
			close(toolStarted)
			select {
			case <-parentDone:
				return map[string]any{"ok": true}, nil
			case <-time.After(10 * time.Second):
				return nil, errors.New("parent did not end")
			}
		})
	if err != nil {
		t.Fatalf("functiontool.New(wait): %v", err)
	}
	waiter, err := llmagent.New(llmagent.Config{
		Name: "waiter",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: genai.NewContentFromParts([]*genai.Part{genai.NewPartFromFunctionCall("wait", map[string]any{})}, genai.RoleModel)},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
		Tools: []tool.Tool{wait},
	})
	if err != nil {
		t.Fatalf("llmagent.New(waiter): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "par",
			SubAgents: []agent.Agent{parent, waiter},
		},
	})
	if err != nil {
		t.Fatalf("parallelagent.New(par): %v", err)
	}

	var sawDone bool
	for ev, err := range newTestRunner(t, par, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
		// Stop at the waiter's answer, before the parallel agent ends.
		if ev != nil && ev.Author == "waiter" && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "done" {
			sawDone = true
			break
		}
	}
	if !sawDone {
		t.Fatal("the run ended without the waiter's answer")
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// TestTraceManager_CustomChildEndsOwnInvocationAfterEvent runs a custom
// sub-agent that writes an event without content and only then ends its own
// InvocationContext, which skips its AfterAgentCallback. Its ended
// InvocationContext is not read when the event arrives (the sub-agent is still
// running on its goroutine), and is settled once its parent ends, reporting
// the run as completed.
func TestTraceManager_CustomChildEndsOwnInvocationAfterEvent(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	gate, err := agent.New(agent.Config{
		Name: "gate",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				state := session.NewEvent(ctx, ctx.InvocationID())
				state.Author = "gate"
				state.Actions.StateDelta = map[string]any{"gate": "closed"}
				if !yield(state, nil) {
					return
				}
				ctx.EndInvocation()
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(gate): %v", err)
	}
	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{gate},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for ev, err := range gate.Run(ctx) {
					if !yield(ev, err) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}
	for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// newSignalingCacheHitAgent returns a custom agent named cached_child whose
// BeforeAgentCallback closes started and answers for it. ADK then ends the
// agent's InvocationContext on the goroutine that runs it, right after the
// callback returns.
func newSignalingCacheHitAgent(t *testing.T, started chan struct{}) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(started)
				return textContent("model", "cache hit"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}
	return a
}

// newBranchA returns the custom agent branch_a, which waits until childStarted
// is closed, writes an event without content if withEvent is set, and ends.
// Its own AfterAgentCallback, which ADK runs after those of the plugins,
// closes done.
func newBranchA(t *testing.T, childStarted, done chan struct{}, withEvent bool) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "branch_a",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				select {
				case <-childStarted:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("cached_child did not start"))
					return
				}
				if withEvent {
					state := session.NewEvent(ctx, ctx.InvocationID())
					state.Author = "branch_a"
					state.Branch = ctx.Branch()
					state.Actions.StateDelta = map[string]any{"seen": true}
					yield(state, nil)
				}
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(done)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("agent.New(branch_a): %v", err)
	}
	return a
}

// TestTraceManager_CrossBranchToolRunsCachedChild runs, from a tool call of
// branch_b of a parallel agent, a sub-agent whose BeforeAgentCallback answers,
// on the branch of branch_a (agent.InvocationContextDelta.Branch), so that it
// runs under branch_a. branch_a writes an event without content and ends while
// the tool call still runs the sub-agent, so neither may read the sub-agent's
// ended InvocationContext (go test -race reports the data race otherwise): it
// is read once the tool call returns, and the run completes.
func TestTraceManager_CrossBranchToolRunsCachedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	childStarted := make(chan struct{})
	branchADone := make(chan struct{})
	cached := newSignalingCacheHitAgent(t, childStarted)
	runChild, err := functiontool.New(functiontool.Config{Name: "run_child", Description: "Runs cached_child on branch_a"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			branch := "par.branch_a"
			// The tool context does not support running an agent; its
			// underlying context does. The sub-agent's event is dropped.
			for _, err := range cached.Run(ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &branch})) {
				if err != nil {
					return nil, err
				}
			}
			select {
			case <-branchADone:
				return map[string]any{"ok": true}, nil
			case <-time.After(10 * time.Second):
				return nil, errors.New("branch_a did not end")
			}
		})
	if err != nil {
		t.Fatalf("functiontool.New(run_child): %v", err)
	}
	branchB, err := llmagent.New(llmagent.Config{
		Name: "branch_b",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: genai.NewContentFromParts([]*genai.Part{genai.NewPartFromFunctionCall("run_child", map[string]any{})}, genai.RoleModel)},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
		Tools: []tool.Tool{runChild},
	})
	if err != nil {
		t.Fatalf("llmagent.New(branch_b): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{newBranchA(t, childStarted, branchADone, true), branchB}},
	})
	if err != nil {
		t.Fatalf("parallelagent.New(par): %v", err)
	}
	for _, err := range newTestRunner(t, par, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// TestTraceManager_CrossBranchDynamicNodeRunsCachedChild runs, from a dynamic
// workflow node of branch_b of a parallel agent, a sub-agent whose
// BeforeAgentCallback answers, on the branch of branch_a
// (workflow.WithOverrideBranch), so that it runs under branch_a. No tool call
// is open. branch_a ends while the sub-agent's event is held, so nothing
// orders the sub-agent's ended InvocationContext before branch_a's end, which
// must not read it (go test -race reports the data race otherwise): it is read
// once branch_b ends, and the run completes.
func TestTraceManager_CrossBranchDynamicNodeRunsCachedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	childStarted := make(chan struct{})
	branchADone := make(chan struct{})
	cachedNode, err := workflow.NewAgentNode(newSignalingCacheHitAgent(t, childStarted), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(cached_child): %v", err)
	}
	dyn := workflow.NewDynamicNode[any, string]("dyn",
		func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			_, err := workflow.RunNode[any](ctx, cachedNode, "go", workflow.WithOverrideBranch("par.branch_a"))
			return "ok", err
		}, workflow.NodeConfig{})
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, dyn))
	if err != nil {
		t.Fatalf("workflow.New(wf): %v", err)
	}
	branchB, err := agent.New(agent.Config{Name: "branch_b", Run: wf.Run})
	if err != nil {
		t.Fatalf("agent.New(branch_b): %v", err)
	}
	var heldTooLong atomic.Bool
	hold, err := baseplugin.New(baseplugin.Config{
		Name: "hold_cached_child_event",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev != nil && ev.Author == "cached_child" {
				select {
				case <-branchADone:
				case <-time.After(10 * time.Second):
					heldTooLong.Store(true)
				}
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(hold_cached_child_event): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{newBranchA(t, childStarted, branchADone, false), branchB}},
	})
	if err != nil {
		t.Fatalf("parallelagent.New(par): %v", err)
	}
	for _, err := range newTestRunner(t, par, hold, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	if heldTooLong.Load() {
		t.Error("branch_a did not end while the event of cached_child was held")
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// endedContext stands in for the InvocationContext of an agent activation:
// endedFunc reads its Ended method, which reports end, as for an activation
// that a BeforeAgentCallback answered, and counts the reads.
type endedContext struct {
	context.Context
	end   bool
	reads int
}

func (c *endedContext) Ended() bool {
	c.reads++
	return c.end
}

// setAgentSpan applies set to the open agent spans named name.
func setAgentSpan(t *testing.T, tm *traceManager, invocationID, name string, set func(*spanRecord)) {
	t.Helper()
	tm.mu.Lock()
	inv := tm.invocations[invocationID]
	tm.mu.Unlock()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	for _, rec := range inv.agentStack {
		if rec.name == name {
			set(rec)
		}
	}
}

// TestTraceManager_SettleWaitsForLauncher reads the ended InvocationContext of
// a sub-agent of branch_a whose BeforeAgentCallback answered. A tool call, a
// custom or workflow agent on another branch, or an enclosing non-LLM agent on
// the same branch and node path above an undeclared sibling parent, that
// started before the sub-agent and is still open may run it (see
// launcherOpenLocked), so the sub-agent is not read when its direct parent
// ends, but once that tool call or enclosing launcher ends. A tool call or
// agent that started after it, an llmagent, an agent whose branch contains the
// sub-agent's, and a direct parent that declared the sub-agent in SubAgents do
// not hold it back.
func TestTraceManager_SettleWaitsForLauncher(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-launcher"
	par := identity{invocationID: invID, agentName: "par"}
	branchA := identity{invocationID: invID, agentName: "branch_a", branch: "par.branch_a"}
	sibling := identity{invocationID: invID, agentName: "sibling", branch: "par.branch_a"}
	branchB := identity{invocationID: invID, agentName: "branch_b", branch: "par.branch_b"}
	cached := identity{invocationID: invID, agentName: "cached", branch: "par.branch_a"}
	call := toolKey(branchB, "call-1", "run_cached")
	nonLLM := func(rec *spanRecord) { rec.nonLLM = true }
	for _, tc := range []struct {
		name string
		// before and after run before and after the sub-agent starts,
		// retire ends the direct parent span (defaults to branchA), and
		// end ends the remaining launcher.
		before, after, end func(tm *traceManager)
		retire             identity
		blocked            bool
	}{
		{
			name:    "tool call started before",
			before:  func(tm *traceManager) { tm.pushToolSpan(ctx, branchB, call) },
			end:     func(tm *traceManager) { tm.popToolSpan(ctx, branchB, call, false) },
			blocked: true,
		},
		{
			name:    "tool call started before, ended by a function response",
			before:  func(tm *traceManager) { tm.pushToolSpan(ctx, branchB, call) },
			end:     func(tm *traceManager) { tm.completeResponse(ctx, par, "branch_b", "par.branch_b", []string{"call-1"}) },
			blocked: true,
		},
		{
			name:  "tool call started after",
			after: func(tm *traceManager) { tm.pushToolSpan(ctx, branchB, call) },
			end:   func(tm *traceManager) { tm.popToolSpan(ctx, branchB, call, false) },
		},
		{
			name:    "custom agent of another branch started before",
			before:  func(tm *traceManager) { setAgentSpan(t, tm, invID, "branch_b", nonLLM) },
			end:     func(tm *traceManager) { tm.popAgentSpan(ctx, branchB) },
			blocked: true,
		},
		{
			name: "llmagent of another branch started before",
			end:  func(tm *traceManager) { tm.popAgentSpan(ctx, branchB) },
		},
		{
			name: "custom agent of another branch started after",
			after: func(tm *traceManager) {
				tm.popAgentSpan(ctx, branchB)
				tm.pushAgentSpan(ctx, branchB)
				setAgentSpan(t, tm, invID, "branch_b", nonLLM)
			},
			end: func(tm *traceManager) { tm.popAgentSpan(ctx, branchB) },
		},
		{
			name: "workflow agent above it",
			before: func(tm *traceManager) {
				setAgentSpan(t, tm, invID, "par", func(rec *spanRecord) {
					rec.nonLLM = true
					rec.isParallel = true
					rec.propagatesChildError = true
					rec.children = []string{"branch_a", "branch_b"}
				})
			},
			end: func(tm *traceManager) { tm.popAgentSpan(ctx, branchB) },
		},
		{
			name:   "custom non-LLM agent above parallel-looking branch without SubAgents",
			before: func(tm *traceManager) { setAgentSpan(t, tm, invID, "par", nonLLM) },
			end: func(tm *traceManager) {
				tm.popAgentSpan(ctx, branchB)
				tm.popAgentSpan(ctx, par)
			},
			blocked: true,
		},
		{
			name: "custom non-LLM agent above parallel-looking branch with SubAgents",
			before: func(tm *traceManager) {
				setAgentSpan(t, tm, invID, "par", func(rec *spanRecord) {
					rec.nonLLM = true
					rec.children = []string{"branch_a", "branch_b"}
				})
			},
			end: func(tm *traceManager) {
				tm.popAgentSpan(ctx, branchB)
				tm.popAgentSpan(ctx, par)
			},
			blocked: true,
		},
		{
			name: "custom parent on same branch without SubAgents declared",
			before: func(tm *traceManager) {
				setAgentSpan(t, tm, invID, "branch_a", nonLLM)
				tm.pushAgentSpan(ctx, sibling)
			},
			retire: sibling,
			end: func(tm *traceManager) {
				tm.popAgentSpan(ctx, branchB)
				tm.popAgentSpan(ctx, branchA)
			},
			blocked: true,
		},
		{
			name: "custom parent on same branch with SubAgents declared on direct parent",
			before: func(tm *traceManager) {
				setAgentSpan(t, tm, invID, "branch_a", nonLLM)
				tm.pushAgentSpan(ctx, sibling)
				setAgentSpan(t, tm, invID, "sibling", func(rec *spanRecord) {
					rec.nonLLM = true
					rec.children = []string{"cached"}
				})
			},
			retire: sibling,
			end: func(tm *traceManager) {
				tm.popAgentSpan(ctx, branchB)
				tm.popAgentSpan(ctx, branchA)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.invocationSpan(ctx, par, nil)
			tm.pushAgentSpan(ctx, par)
			tm.pushAgentSpan(ctx, branchA)
			tm.pushAgentSpan(ctx, branchB)
			if tc.before != nil {
				tc.before(tm)
			}
			src := &endedContext{Context: ctx, end: true}
			tm.pushAgentSpan(src, cached)
			if tc.after != nil {
				tc.after(tm)
			}
			if src.reads != 0 {
				t.Errorf("the sub-agent's ended InvocationContext was read %d times before direct parent ended, want 0", src.reads)
			}
			retire := tc.retire
			if retire.agentName == "" {
				retire = branchA
			}
			tm.popAgentSpan(ctx, retire)
			atEnd := src.reads
			tc.end(tm)
			wantAtEnd := 1
			if tc.blocked {
				wantAtEnd = 0
			}
			if atEnd != wantAtEnd || src.reads != 1 {
				t.Errorf("reads of the sub-agent's ended InvocationContext at direct parent end and once launcher ended = %d, %d; want %d, 1", atEnd, src.reads, wantAtEnd)
			}
			if tc.retire.agentName != "" {
				if got := len(tm.invocations[invID].agentStack); got != 2 {
					t.Errorf("open spans left after enclosing parent retired = %d, want 2 ([root, par])", got)
				}
			}
		})
	}
}

// TestTraceManager_ContentlessEventCancelAndSiblingFailureEndInvocationRace
// verifies under -race that OnEventCallback on a non-partial event without
// content does not read Ended() concurrently with the event's custom-agent
// author calling EndInvocation() when run cancellation or a sibling failure
// unblocks the parallelagent branch while an earlier OnEventCallback holds the
// event.
func TestTraceManager_ContentlessEventCancelAndSiblingFailureEndInvocationRace(t *testing.T) {
	for _, tc := range []struct {
		name          string
		cancelRun     bool
		wantTerminal  string
		wantAgentErrs map[string]int
	}{
		{
			name:          "run canceled while contentless event held",
			cancelRun:     true,
			wantTerminal:  EventInvocationError,
			wantAgentErrs: map[string]int{},
		},
		{
			name:          "sibling fails while contentless event held",
			cancelRun:     false,
			wantTerminal:  EventInvocationError,
			wantAgentErrs: map[string]int{"failer": 1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			runCtx, cancel := context.WithCancel(t.Context())
			defer cancel()

			resumed := make(chan struct{})
			failNow := make(chan struct{})

			ender, err := agent.New(agent.Config{
				Name: "ender",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						defer func() {
							close(resumed)
							ctx.EndInvocation()
						}()
						state := session.NewEvent(ctx, ctx.InvocationID())
						state.Author = "ender"
						state.Branch = ctx.Branch()
						state.Actions.StateDelta = map[string]any{"step": 1}
						yield(state, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(ender): %v", err)
			}

			subAgents := []agent.Agent{ender}
			if !tc.cancelRun {
				failer, err := agent.New(agent.Config{
					Name: "failer",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							select {
							case <-failNow:
								yield(nil, errors.New("boom"))
							case <-ctx.Done():
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(failer): %v", err)
				}
				subAgents = append(subAgents, failer)
			}

			var heldTooLong atomic.Bool
			var triggered sync.Once
			hold, err := baseplugin.New(baseplugin.Config{
				Name: "hold_contentless_event",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil && ev.Author == "ender" && ev.Content == nil && !ev.Partial {
						triggered.Do(func() {
							if tc.cancelRun {
								cancel()
							} else {
								close(failNow)
							}
							select {
							case <-resumed:
							case <-time.After(10 * time.Second):
								heldTooLong.Store(true)
							}
						})
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(hold_contentless_event): %v", err)
			}

			par, err := parallelagent.New(parallelagent.Config{
				AgentConfig: agent.Config{
					Name:      "par",
					SubAgents: subAgents,
				},
			})
			if err != nil {
				t.Fatalf("parallelagent.New(par): %v", err)
			}

			for range newTestRunner(t, par, hold, p).Run(runCtx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			}
			if heldTooLong.Load() {
				t.Fatal("ender did not unblock while its contentless event was held")
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			onlyRowOfType(t, rows, tc.wantTerminal)
			if got := agentErrorCounts(rows); !maps.Equal(got, tc.wantAgentErrs) {
				t.Errorf("AGENT_ERROR counts = %v, want %v", got, tc.wantAgentErrs)
			}
		})
	}
}

// TestTraceManager_ConcurrentChildrenSameBranchCachedEndInvocationRace verifies
// under -race that a custom parent running concurrent sub-agents on its own
// branch and node path (both with and without SubAgents declared) does not read
// a BeforeAgentCallback-answered child's Ended() state concurrently with ADK's
// EndInvocation() write when a sibling sub-agent starts or ends concurrently,
// and still settles and retires the cached child once the parent finishes.
func TestTraceManager_ConcurrentChildrenSameBranchCachedEndInvocationRace(t *testing.T) {
	for _, declareSubAgents := range []bool{true, false} {
		subAgentsLabel := "without SubAgents declared"
		if declareSubAgents {
			subAgentsLabel = "with SubAgents declared"
		}
		for _, siblingEnds := range []bool{false, true} {
			phaseLabel := "sibling starts while cached answers"
			if siblingEnds {
				phaseLabel = "sibling ends while cached answers"
			}
			t.Run(subAgentsLabel+"/"+phaseLabel, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				child1Answered := make(chan struct{})
				siblingStarted := make(chan struct{})

				cachedChild, err := agent.New(agent.Config{
					Name: "cached_child",
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							close(child1Answered)
							return textContent("model", "cache hit"), nil
						},
					},
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(cached_child): %v", err)
				}

				sibling, err := agent.New(agent.Config{
					Name: "sibling",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "sibling"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "sibling answer")
							if !yield(ev, nil) {
								return
							}
							if siblingEnds {
								close(siblingStarted)
								select {
								case <-child1Answered:
								case <-time.After(10 * time.Second):
									yield(nil, errors.New("cached_child did not answer"))
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(sibling): %v", err)
				}

				cfg := agent.Config{
					Name: "parent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							answer := session.NewEvent(ctx, ctx.InvocationID())
							answer.Author = "parent"
							answer.Branch = ctx.Branch()
							answer.Content = textContent("model", "parent answer")
							if !yield(answer, nil) {
								return
							}

							var (
								wg         sync.WaitGroup
								cachedErr  error
								stoppedOut bool
							)
							wg.Add(2)
							go func() {
								defer wg.Done()
								if siblingEnds {
									select {
									case <-siblingStarted:
									case <-time.After(10 * time.Second):
										cachedErr = errors.New("sibling did not start")
										return
									}
								}
								// Drop cached_child's event so only contextEnded
								// settled after parent joins can mark it responded.
								for _, runErr := range cachedChild.Run(ctx) {
									if runErr != nil && cachedErr == nil {
										cachedErr = runErr
									}
								}
							}()
							go func() {
								defer wg.Done()
								if !siblingEnds {
									select {
									case <-child1Answered:
									case <-time.After(10 * time.Second):
										yield(nil, errors.New("cached_child did not answer"))
										return
									}
								}
								for ev, runErr := range sibling.Run(ctx) {
									if !yield(ev, runErr) {
										stoppedOut = true
										return
									}
								}
							}()
							wg.Wait()
							if !stoppedOut && cachedErr != nil {
								yield(nil, cachedErr)
							}
						}
					},
				}
				if declareSubAgents {
					cfg.SubAgents = []agent.Agent{cachedChild, sibling}
				}
				parent, err := agent.New(cfg)
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}

				for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
				}
				requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
			})
		}
	}
}

// TestTraceManager_ToolRunEndsWhileOtherToolCallIsOpen ends the tool call of
// coordinator that ran helper, an agent that runs as a tool, while a tool call
// of another agent that started before helper is still open and may run
// helper's sub-agent (see launcherOpenLocked). helper finished when its tool
// call returned, so its ended InvocationContext is read: it is retired quietly
// if a BeforeAgentCallback answered for it, and purged with an AGENT_ERROR row
// if it failed without a response. Its sub-agent is not known to have failed
// before the other tool call ends, so it gets no row.
func TestTraceManager_ToolRunEndsWhileOtherToolCallIsOpen(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-tool-run-other-call"
	coord := identity{invocationID: invID, agentName: "coordinator"}
	other := identity{invocationID: invID, agentName: "other", branch: "other"}
	helper := identity{invocationID: invID, agentName: "helper", branch: "helper@1", path: "/helper@1"}
	sub := identity{invocationID: invID, agentName: "sub", branch: "helper@1", path: "/helper@1"}
	otherCall, helperCall := toolKey(other, "call-o", "wait"), toolKey(coord, "call-h", "helper")
	for _, tc := range []struct {
		name        string
		helperEnded bool
		withSub     bool
		wantPurged  []string
	}{
		{name: "helper answered", helperEnded: true},
		{name: "helper failed silently", wantPurged: []string{"helper"}},
		{name: "helper failed silently after an answered sub-agent", withSub: true, wantPurged: []string{"helper"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.invocationSpan(ctx, coord, nil)
			tm.pushAgentSpan(ctx, coord)
			tm.pushAgentSpan(ctx, other)
			tm.pushToolSpan(ctx, other, otherCall)
			tm.pushToolSpan(ctx, coord, helperCall)
			tm.pushAgentSpan(&endedContext{Context: ctx, end: tc.helperEnded}, helper)
			if tc.withSub {
				tm.pushAgentSpan(&endedContext{Context: ctx, end: true}, sub)
			}
			_, purged, _ := tm.popToolSpan(ctx, coord, helperCall, false)
			var got []string
			for _, s := range purged {
				got = append(got, s.agentName)
			}
			if !slices.Equal(got, tc.wantPurged) {
				t.Errorf("activations purged with an AGENT_ERROR row when helper's tool call ended = %v, want %v", got, tc.wantPurged)
			}
		})
	}
}

// TestConcurrencyRace_CachedChildOnDescendantBranchUnderUndeclaredSibling
// verifies under -race that a custom parent running a sibling on its own
// branch concurrently with a BeforeAgentCallback-answered child on a
// descendant branch (both with and without SubAgents declared) does not read
// the cached child's Ended() state concurrently with ADK's EndInvocation()
// write when the sibling ends, before the parent joined the cached child: the
// cached child is parented under the still-running sibling, which did not
// launch it. The parent still settles and retires the cached child once it
// finishes.
func TestConcurrencyRace_CachedChildOnDescendantBranchUnderUndeclaredSibling(t *testing.T) {
	for _, declareSubAgents := range []bool{false, true} {
		label := "without SubAgents declared"
		if declareSubAgents {
			label = "with SubAgents declared"
		}
		t.Run(label, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			siblingAnswered := make(chan struct{})
			cachedAnswering := make(chan struct{})

			cachedChild, err := agent.New(agent.Config{
				Name: "cached_child",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						close(cachedAnswering)
						return textContent("model", "cache hit"), nil
					},
				},
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(cached_child): %v", err)
			}

			sibling, err := agent.New(agent.Config{
				Name: "sibling",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "sibling"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "sibling answer")
						if !yield(ev, nil) {
							return
						}
						close(siblingAnswered)
						// End while cached_child answers on its nested branch.
						select {
						case <-cachedAnswering:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("cached_child did not answer"))
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(sibling): %v", err)
			}

			cfg := agent.Config{
				Name: "parent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						answer := session.NewEvent(ctx, ctx.InvocationID())
						answer.Author = "parent"
						answer.Branch = ctx.Branch()
						answer.Content = textContent("model", "parent answer")
						if !yield(answer, nil) {
							return
						}

						var (
							wg         sync.WaitGroup
							cachedErr  error
							stoppedOut bool
						)
						wg.Add(2)
						go func() {
							defer wg.Done()
							select {
							case <-siblingAnswered:
							case <-time.After(10 * time.Second):
								cachedErr = errors.New("sibling did not answer")
								return
							}
							nested := "nested"
							nestedCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &nested})
							// Drop cached_child's event so only contextEnded
							// settled after parent joins can mark it responded.
							for _, runErr := range cachedChild.Run(nestedCtx) {
								if runErr != nil && cachedErr == nil {
									cachedErr = runErr
								}
							}
						}()
						go func() {
							defer wg.Done()
							for ev, runErr := range sibling.Run(ctx) {
								if !yield(ev, runErr) {
									stoppedOut = true
									return
								}
							}
						}()
						wg.Wait()
						if !stoppedOut && cachedErr != nil {
							yield(nil, cachedErr)
						}
					}
				},
			}
			if declareSubAgents {
				cfg.SubAgents = []agent.Agent{cachedChild, sibling}
			}
			parent, err := agent.New(cfg)
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}

			for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
		})
	}
}

// TestConcurrencyRace_ConcurrentSameAgentOnSameBranchAndPath verifies under
// -race that a custom parent, as the root agent and as a workflow node, that
// runs the same sub-agent twice at once on its own branch and node path, one
// run answered by its BeforeAgentCallback and the other answering from Run,
// ties the start and end of each run to its own activation. Neither start
// settles the other, still running, activation as a finished retry, and the
// end of the answering run retires its own activation rather than the cached
// one, whose Ended() state ADK writes concurrently with EndInvocation(). The
// invocation completes, and the AGENT_COMPLETED row of the sub-agent is on the
// answering run's span.
func TestConcurrencyRace_ConcurrentSameAgentOnSameBranchAndPath(t *testing.T) {
	for _, asNode := range []bool{false, true} {
		topology := "custom parent as root"
		if asNode {
			topology = "custom parent as workflow node"
		}
		for _, cachedFirst := range []bool{true, false} {
			order := "answering run starts first"
			if cachedFirst {
				order = "cached run starts first"
			}
			t.Run(topology+"/"+order, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				firstEntered := make(chan struct{})
				cachedAnswered := make(chan struct{})
				var calls atomic.Int32

				child, err := agent.New(agent.Config{
					Name: "child",
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							first := calls.Add(1) == 1
							if first {
								close(firstEntered)
							}
							if first == cachedFirst {
								close(cachedAnswered)
								return textContent("model", "cache hit"), nil
							}
							return nil, nil
						},
					},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							// Answer only once the cached run answered, so
							// that this run ends while the cached activation
							// of the same agent is open.
							select {
							case <-cachedAnswered:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("cached run did not answer"))
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "child"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "child answer")
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(child): %v", err)
				}

				parent, err := agent.New(agent.Config{
					Name: "parent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							answer := session.NewEvent(ctx, ctx.InvocationID())
							answer.Author = "parent"
							answer.Branch = ctx.Branch()
							answer.Content = textContent("model", "parent answer")
							if !yield(answer, nil) {
								return
							}

							var (
								wg         sync.WaitGroup
								runErrs    [2]error
								stoppedOut bool
							)
							// Only the answering run's events are forwarded:
							// the cached run's event is dropped so that only
							// its ended state, settled after parent joins it,
							// marks it responded.
							run := func(i int, forward bool) {
								defer wg.Done()
								if i == 1 {
									select {
									case <-firstEntered:
									case <-time.After(10 * time.Second):
										runErrs[i] = errors.New("first child run did not start")
										return
									}
								}
								for ev, runErr := range child.Run(ctx) {
									if !forward {
										if runErr != nil && runErrs[i] == nil {
											runErrs[i] = runErr
										}
										continue
									}
									if !yield(ev, runErr) {
										stoppedOut = true
										return
									}
								}
							}
							wg.Add(2)
							go run(0, !cachedFirst)
							go run(1, cachedFirst)
							wg.Wait()
							if err := errors.Join(runErrs[0], runErrs[1]); !stoppedOut && err != nil {
								yield(nil, err)
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}
				root := parent
				if asNode {
					node, err := workflow.NewAgentNode(parent, workflow.NodeConfig{})
					if err != nil {
						t.Fatalf("workflow.NewAgentNode: %v", err)
					}
					root, err = workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, node)})
					if err != nil {
						t.Fatalf("workflowagent.New: %v", err)
					}
				}

				for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)

				var started, completed []map[string]any
				for _, row := range rowsOfType(rows, EventAgentStarting) {
					if row["agent"] == "child" {
						started = append(started, row)
					}
				}
				for _, row := range rowsOfType(rows, EventAgentCompleted) {
					if row["agent"] == "child" {
						completed = append(completed, row)
					}
				}
				if len(started) != 2 {
					t.Fatalf("got %d AGENT_STARTING rows of child, want 2; event types: %v", len(started), eventTypes(rows))
				}
				if started[0]["span_id"] == started[1]["span_id"] {
					t.Errorf("both runs of child started on span %v, want distinct spans", started[0]["span_id"])
				}
				// The answering run starts second when the cached run starts
				// first, and it is the only run of child that gets an
				// AfterAgentCallback.
				answering := started[0]
				if cachedFirst {
					answering = started[1]
				}
				if len(completed) != 1 {
					t.Fatalf("got %d AGENT_COMPLETED rows of child, want 1; event types: %v", len(completed), eventTypes(rows))
				}
				if got, want := completed[0]["span_id"], answering["span_id"]; got != want {
					t.Errorf("AGENT_COMPLETED row of child is on span %v, want the answering run's span %v", got, want)
				}
				if got := completed[0]["status"]; got != "OK" {
					t.Errorf("AGENT_COMPLETED row of child has status %v, want OK", got)
				}
			})
		}
	}
}

// TestFailedParallelAgent_RecoveredByRootAfterAgentCallbackReportsCompleted
// runs a guard agent, as the root agent and as a sub-agent of a custom root,
// that swallows the failure of a parallel agent whose branches both answered
// before one of them failed, and recovers with content either from Run or from
// its own AfterAgentCallback. ADK runs the guard's AfterAgentCallback after
// this plugin's, which has retired the guard and passed the failure up to the
// guard's parent by then, so the recovery content must still clear it: the
// runner yields no error, and the invocation completes with the same
// AGENT_ERROR rows as when the guard recovers from Run.
func TestFailedParallelAgent_RecoveredByRootAfterAgentCallbackReportsCompleted(t *testing.T) {
	for _, underRoot := range []bool{false, true} {
		topology := "guard as root"
		if underRoot {
			topology = "guard under custom root"
		}
		var agentErrors [2]map[string]int
		for i, fromCallback := range []bool{false, true} {
			mode := "recovered from Run"
			if fromCallback {
				mode = "recovered from AfterAgentCallback"
			}
			t.Run(topology+"/"+mode, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				answered := make(chan struct{})

				answering, err := agent.New(agent.Config{
					Name: "answering",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							defer close(answered)
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "answering"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "answering branch answer")
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(answering): %v", err)
				}
				failing, err := agent.New(agent.Config{
					Name: "failing",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "failing"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "failing branch answer")
							if !yield(ev, nil) {
								return
							}
							select {
							case <-answered:
							case <-time.After(10 * time.Second):
							}
							yield(nil, errors.New("branch failed after answering"))
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(failing): %v", err)
				}
				par, err := parallelagent.New(parallelagent.Config{
					AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{answering, failing}},
				})
				if err != nil {
					t.Fatalf("parallelagent.New: %v", err)
				}

				guardCfg := agent.Config{
					Name:      "guard",
					SubAgents: []agent.Agent{par},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range par.Run(ctx) {
								if runErr != nil {
									// The guard swallows the branch failure.
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							if fromCallback {
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "guard"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "guard recovered")
							yield(ev, nil)
						}
					},
				}
				if fromCallback {
					guardCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							return textContent("model", "guard recovered"), nil
						},
					}
				}
				guard, err := agent.New(guardCfg)
				if err != nil {
					t.Fatalf("agent.New(guard): %v", err)
				}
				root := guard
				if underRoot {
					root, err = agent.New(agent.Config{
						Name:      "root",
						SubAgents: []agent.Agent{guard},
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								for ev, runErr := range guard.Run(ctx) {
									if !yield(ev, runErr) {
										return
									}
								}
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(root): %v", err)
					}
				}

				var sawRecovery bool
				for ev, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
					if ev != nil && ev.Author == "guard" && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "guard recovered" {
						sawRecovery = true
					}
				}
				if !sawRecovery {
					t.Fatal("runner did not yield the guard's recovery content")
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
				}
				if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
					t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
				}
				agentErrors[i] = agentErrorCounts(rows)
				if counts := agentErrors[i]; counts["guard"] != 0 || counts["root"] != 0 {
					t.Errorf("AGENT_ERROR counts = %v, want none for guard or root, which recovered", counts)
				}
			})
		}
		if agentErrors[0] != nil && agentErrors[1] != nil && !maps.Equal(agentErrors[0], agentErrors[1]) {
			t.Errorf("%s: AGENT_ERROR counts when recovered from AfterAgentCallback = %v, want those when recovered from Run, %v", topology, agentErrors[1], agentErrors[0])
		}
	}
}

// TestFailedAgent_RootFailsAfterContentlessPartialAndCompletedChild runs a
// custom root that yields a contentless partial event, drains an LLM sub-agent
// that completes without forwarding its events, and then fails, at which the
// caller stops reading. The sub-agent's run after the partial event shows that
// the caller read past it, so the root's failure is reported rather than taken
// for the teardown of a caller that stopped on the partial event: the root
// gets an AGENT_ERROR row and the invocation ends with INVOCATION_ERROR.
func TestFailedAgent_RootFailsAfterContentlessPartialAndCompletedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	child, err := llmagent.New(llmagent.Config{
		Name: "child",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: textContent("model", "child done"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(child): %v", err)
	}
	rootErr := errors.New("root failed after child completed")
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				partial := session.NewEvent(ctx, ctx.InvocationID())
				partial.Author = "root"
				partial.Branch = ctx.Branch()
				partial.Partial = true
				if !yield(partial, nil) {
					return
				}
				// Drain child without forwarding its events.
				for _, runErr := range child.Run(ctx) {
					if runErr != nil {
						yield(nil, runErr)
						return
					}
				}
				yield(nil, rootErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	var runErr error
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			runErr = err
			break
		}
	}
	if !errors.Is(runErr, rootErr) {
		t.Fatalf("runner.Run error = %v, want %v", runErr, rootErr)
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	onlyRowOfType(t, rows, EventInvocationError)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Errorf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
	if got, want := agentErrorCounts(rows), map[string]int{"root": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR counts = %v, want %v; event types: %v", got, want, eventTypes(rows))
	}
	var childCompleted int
	for _, row := range rowsOfType(rows, EventAgentCompleted) {
		if row["agent"] == "child" {
			childCompleted++
		}
	}
	if childCompleted != 1 {
		t.Errorf("got %d AGENT_COMPLETED rows of child, want 1; event types: %v", childCompleted, eventTypes(rows))
	}
}

// TestConcurrencyRace_SameOverrideBranchSiblingAndCachedChild verifies under
// -race that when a custom parent runs both a sibling sub-agent and a
// BeforeAgentCallback-short-circuited sub-agent concurrently on the same
// overridden branch, retiring the sibling while the parent still runs the
// short-circuited sub-agent holds back settling the short-circuited sub-agent's
// Ended() state until the parent finishes, both with and without SubAgents
// declared on the parent.
func TestConcurrencyRace_SameOverrideBranchSiblingAndCachedChild(t *testing.T) {
	for _, declareSubAgents := range []bool{false, true} {
		label := "without SubAgents declared"
		if declareSubAgents {
			label = "with SubAgents declared"
		}
		t.Run(label, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			siblingAnswered := make(chan struct{})
			cachedAnswering := make(chan struct{})

			cachedChild, err := agent.New(agent.Config{
				Name: "cached_child",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						close(cachedAnswering)
						return textContent("model", "cache hit"), nil
					},
				},
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(cached_child): %v", err)
			}

			sibling, err := agent.New(agent.Config{
				Name: "sibling",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "sibling"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "sibling answer")
						if !yield(ev, nil) {
							return
						}
						close(siblingAnswered)
						select {
						case <-cachedAnswering:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("cached_child did not answer"))
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(sibling): %v", err)
			}

			cfg := agent.Config{
				Name: "parent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						answer := session.NewEvent(ctx, ctx.InvocationID())
						answer.Author = "parent"
						answer.Branch = ctx.Branch()
						answer.Content = textContent("model", "parent answer")
						if !yield(answer, nil) {
							return
						}

						var (
							wg         sync.WaitGroup
							cachedErr  error
							stoppedOut bool
						)
						nestedS := "nested"
						sibCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &nestedS})
						wg.Add(2)
						go func() {
							defer wg.Done()
							select {
							case <-siblingAnswered:
							case <-time.After(10 * time.Second):
								cachedErr = errors.New("sibling did not answer")
								return
							}
							nested := "nested"
							nestedCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &nested})
							for _, runErr := range cachedChild.Run(nestedCtx) {
								if runErr != nil && cachedErr == nil {
									cachedErr = runErr
								}
							}
						}()
						go func() {
							defer wg.Done()
							for ev, runErr := range sibling.Run(sibCtx) {
								if !yield(ev, runErr) {
									stoppedOut = true
									return
								}
							}
						}()
						wg.Wait()
						if !stoppedOut && cachedErr != nil {
							yield(nil, cachedErr)
						}
					}
				},
			}
			if declareSubAgents {
				cfg.SubAgents = []agent.Agent{cachedChild, sibling}
			}
			parent, err := agent.New(cfg)
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}

			for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
		})
	}
}

// TestConcurrencyRace_NestedDescendantBranchCachedChildUnderOverrideSibling
// verifies under -race that when a custom parent without SubAgents declared
// runs an override-branch sibling ("b1") concurrently with a short-circuited
// sub-agent on a nested descendant branch ("b1.nested"), retiring the sibling
// does not read the short-circuited sub-agent's Ended() state while the parent
// is still running it.
func TestConcurrencyRace_NestedDescendantBranchCachedChildUnderOverrideSibling(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	siblingAnswered := make(chan struct{})
	cachedEntering := make(chan struct{})

	cachedChild, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(cachedEntering)
				return textContent("model", "cache hit"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}

	sibling, err := agent.New(agent.Config{
		Name: "sibling",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "sibling"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "sibling answer")
				if !yield(ev, nil) {
					return
				}
				close(siblingAnswered)
				select {
				case <-cachedEntering:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("cached_child did not enter callback"))
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(sibling): %v", err)
	}

	parent, err := agent.New(agent.Config{
		Name: "parent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				answer := session.NewEvent(ctx, ctx.InvocationID())
				answer.Author = "parent"
				answer.Branch = ctx.Branch()
				answer.Content = textContent("model", "parent answer")
				if !yield(answer, nil) {
					return
				}

				var (
					wg         sync.WaitGroup
					cachedErr  error
					stoppedOut bool
				)
				wg.Add(2)
				go func() {
					defer wg.Done()
					select {
					case <-siblingAnswered:
					case <-time.After(10 * time.Second):
						cachedErr = errors.New("sibling did not answer")
						return
					}
					nested := "b1.nested"
					nestedCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &nested})
					for _, runErr := range cachedChild.Run(nestedCtx) {
						if runErr != nil && cachedErr == nil {
							cachedErr = runErr
						}
					}
				}()
				go func() {
					defer wg.Done()
					b1 := "b1"
					b1Ctx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &b1})
					for ev, runErr := range sibling.Run(b1Ctx) {
						if !yield(ev, runErr) {
							stoppedOut = true
							return
						}
					}
				}()
				wg.Wait()
				if !stoppedOut && cachedErr != nil {
					yield(nil, cachedErr)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}

	for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// TestConcurrencyRace_ConcurrentSameCustomAgent_FailureNotMaskedByPeerResponse
// verifies that when a custom parent runs two concurrent activations of the
// same custom sub-agent on the same branch and node path, one activation
// answering and retiring (whether its response event is delivered before
// retirement or buffered until after retirement, and whether the goroutine
// breaks out on error or pumps child.Run to completion so AfterAgentCallback
// runs on the failing peer too) does not mark the peer activation as responded
// when the peer fails without responding.
func TestConcurrencyRace_ConcurrentSameCustomAgent_FailureNotMaskedByPeerResponse(t *testing.T) {
	testCases := []struct {
		name               string
		bufferUntilRetired bool
	}{
		{name: "response delivered before peer retires", bufferUntilRetired: false},
		{name: "buffered response delivered after peer retires", bufferUntilRetired: true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var entered atomic.Int32
			bothOpen := make(chan struct{})
			run1Done := make(chan struct{})
			run1Delivered := make(chan struct{})
			run2Err := errors.New("second child activation failed before responding")

			child, err := agent.New(agent.Config{
				Name: "child",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						myIdx := entered.Add(1)
						if myIdx == 2 {
							close(bothOpen)
						} else {
							select {
							case <-bothOpen:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("second run never started"))
								return
							}
						}
						if myIdx == 1 {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "child"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "run 1 answer")
							yield(ev, nil)
							return
						}
						select {
						case <-run1Delivered:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("run 1 barrier timed out"))
							return
						}
						yield(nil, run2Err)
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
						type item struct {
							ev  *session.Event
							err error
							ack chan bool
						}
						ch := make(chan item, 8)
						done := make(chan struct{})
						defer close(done)
						var wg sync.WaitGroup
						for idx := range 2 {
							wg.Add(1)
							go func(idx int) {
								defer wg.Done()
								var sawEv bool
								for ev, err := range child.Run(ctx) {
									if ev != nil {
										sawEv = true
										if !tc.bufferUntilRetired {
											ack := make(chan bool, 1)
											ch <- item{ev: ev, err: err, ack: ack}
											select {
											case cont := <-ack:
												if !cont {
													return
												}
											case <-done:
												return
											}
											continue
										}
									}
									ch <- item{ev: ev, err: err}
									if err != nil {
										break
									}
								}
								if sawEv {
									close(run1Done)
								}
							}(idx)
						}
						go func() {
							wg.Wait()
							close(ch)
						}()
						var notified bool
						for it := range ch {
							if tc.bufferUntilRetired && it.ev != nil {
								select {
								case <-run1Done:
								case <-time.After(10 * time.Second):
									yield(nil, errors.New("peer retirement timed out"))
									return
								}
							}
							cont := yield(it.ev, it.err)
							if it.ack != nil {
								it.ack <- cont
							}
							if !cont {
								return
							}
							if it.ev != nil && !notified {
								notified = true
								close(run1Delivered)
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}

			var gotErr error
			for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					gotErr = err
					break
				}
			}
			if !errors.Is(gotErr, run2Err) {
				t.Fatalf("runner.Run error = %v, want %v", gotErr, run2Err)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			onlyRowOfType(t, rows, EventInvocationError)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Errorf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			if got := agentErrorCounts(rows)["child"]; got != 1 {
				t.Errorf("AGENT_ERROR count for child = %d, want 1; counts=%v, event types=%v", got, agentErrorCounts(rows), eventTypes(rows))
			}
			if got := agentErrorCounts(rows)["parent"]; got != 1 {
				t.Errorf("AGENT_ERROR count for parent = %d, want 1; counts=%v, event types=%v", got, agentErrorCounts(rows), eventTypes(rows))
			}
		})
	}
}

// TestConcurrencyRace_ConcurrentSilentCustomAgentRuns_EmitsInvocationCompleted
// verifies under -race that when a custom parent runs two overlapping silent
// activations of the same custom child (side-effect/state-only custom agents
// that emit no session events and return no errors) and joins both before
// returning, all three activations complete cleanly with AGENT_COMPLETED and
// INVOCATION_COMPLETED rather than reporting false AGENT_ERROR or
// INVOCATION_ERROR rows.
func TestConcurrencyRace_ConcurrentSilentCustomAgentRuns_EmitsInvocationCompleted(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	var entered atomic.Int32
	bothOpen := make(chan struct{})

	silentChild, err := agent.New(agent.Config{
		Name: "silent_child",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if entered.Add(1) == 2 {
					close(bothOpen)
				} else {
					select {
					case <-bothOpen:
					case <-time.After(10 * time.Second):
						yield(nil, errors.New("second silent run never started"))
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(silent_child): %v", err)
	}

	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{silentChild},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				errs := make(chan error, 2)
				var wg sync.WaitGroup
				for range 2 {
					wg.Add(1)
					go func() {
						defer wg.Done()
						for _, runErr := range silentChild.Run(ctx) {
							if runErr != nil {
								errs <- runErr
								return
							}
						}
					}()
				}
				wg.Wait()
				close(errs)
				for runErr := range errs {
					if !yield(nil, runErr) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}

	for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentCompleted)); n != 3 {
		t.Errorf("got %d AGENT_COMPLETED rows, want 3; event types: %v", n, eventTypes(rows))
	}
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

type concurrentFailRunKey struct{}

type concurrentBarrierLLM struct {
	name string
	fn   func(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error)
}

func (m *concurrentBarrierLLM) Name() string { return m.name }

func (m *concurrentBarrierLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.fn(ctx, req)
		yield(resp, err)
	}
}

// TestConcurrencyRace_ConcurrentSameLLMAgent_ModelErrorNotClearedByPeerSuccess
// verifies under -race that when a custom parent runs two concurrent
// activations of the same llmagent on the same branch and node path, their
// model spans and unrecoveredErrors entries are isolated per activation so a
// peer activation's succeeding model call — or a trailer event emitted from the
// succeeding peer's own AfterAgentCallback after its activation retired —
// cannot overwrite the failing activation's model span or clear its
// unrecovered error, whether the parent drains the failing activation or stops
// iterating it at the error.
func TestConcurrencyRace_ConcurrentSameLLMAgent_ModelErrorNotClearedByPeerSuccess(t *testing.T) {
	modes := []struct {
		name             string
		breakOnErr       bool
		withTrailer      bool
		holdUntilTrailer bool
		priorToolTurn    bool
	}{
		{name: "drain failing activation"},
		{name: "break on failing activation error", breakOnErr: true},
		{name: "drain failing activation/with_peer_after_agent_trailer", withTrailer: true},
		{name: "hold failing activation open until trailer/with_peer_after_agent_trailer", withTrailer: true, holdUntilTrailer: true},
		{name: "break on failing activation error/with_peer_after_agent_trailer", breakOnErr: true, withTrailer: true},
		{name: "drain failing activation/prior_tool_turn", priorToolTurn: true},
		{name: "break on failing activation error/prior_tool_turn", breakOnErr: true, priorToolTurn: true},
		{name: "drain failing activation/with_peer_after_agent_trailer/prior_tool_turn", withTrailer: true, priorToolTurn: true},
		{name: "hold failing activation open until trailer/with_peer_after_agent_trailer/prior_tool_turn", withTrailer: true, holdUntilTrailer: true, priorToolTurn: true},
		{name: "break on failing activation error/with_peer_after_agent_trailer/prior_tool_turn", breakOnErr: true, withTrailer: true, priorToolTurn: true},
	}
	for _, mode := range modes {
		for _, failingFirst := range []bool{true, false} {
			order := "succeeding run starts first"
			if failingFirst {
				order = "failing run starts first"
			}
			t.Run(mode.name+"/"+order, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				var started, inModel atomic.Int32
				firstStarted := make(chan struct{})
				bothInModel := make(chan struct{})
				failReady := make(chan struct{})
				var failReadyOnce sync.Once
				trailerDelivered := make(chan struct{})
				var trailerOnce sync.Once
				modelErr := errors.New("quota exceeded on failing activation")

				// failCallCount tracks the number of model calls made by the
				// failing activation, so a prior tool turn's function call is
				// returned on the first call and the error on the second.
				var failCallCount atomic.Int32

				mdl := &concurrentBarrierLLM{
					name: "test-model",
					fn: func(ctx context.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
						if inModel.Add(1) == 2 {
							close(bothInModel)
						} else {
							select {
							case <-bothInModel:
							case <-ctx.Done():
								return nil, ctx.Err()
							case <-time.After(10 * time.Second):
								return nil, errors.New("timed out waiting for peer model call")
							}
						}
						isFail := ctx.Value(concurrentFailRunKey{}) != nil
						if mode.priorToolTurn && isFail {
							if failCallCount.Add(1) == 1 {
								// First call of the failing activation: return a
								// function call so the tool turn runs.
								return &model.LLMResponse{
									Content: genai.NewContentFromParts([]*genai.Part{
										genai.NewPartFromFunctionCall("probe", map[string]any{}),
									}, genai.RoleModel),
								}, nil
							}
							// Second call of the failing activation fails.
							return nil, modelErr
						}
						if isFail {
							return nil, modelErr
						}
						select {
						case <-failReady:
						case <-ctx.Done():
							return nil, ctx.Err()
						case <-time.After(10 * time.Second):
							return nil, errors.New("timed out waiting for failing activation")
						}
						return &model.LLMResponse{Content: textContent("model", "peer ok"), TurnComplete: true}, nil
					},
				}
				workerCfg := llmagent.Config{
					Name:  "worker",
					Model: mdl,
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							if started.Add(1) == 1 {
								close(firstStarted)
							}
							return nil, nil
						},
					},
				}
				if mode.priorToolTurn {
					probe, perr := functiontool.New(functiontool.Config{Name: "probe", Description: "Probes before failing"},
						func(agent.Context, struct{}) (map[string]any, error) {
							return map[string]any{"ok": true}, nil
						})
					if perr != nil {
						t.Fatalf("functiontool.New(probe): %v", perr)
					}
					workerCfg.Tools = []tool.Tool{probe}
				}
				if mode.withTrailer {
					workerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
						func(cbCtx agent.Context) (*genai.Content, error) {
							if cbCtx.Value(concurrentFailRunKey{}) != nil {
								return nil, nil
							}
							return textContent("model", "peer trailer"), nil
						},
					}
				}
				worker, err := llmagent.New(workerCfg)
				if err != nil {
					t.Fatalf("llmagent.New(worker): %v", err)
				}
				parent, err := agent.New(agent.Config{
					Name:      "parent",
					SubAgents: []agent.Agent{worker},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							type item struct {
								ev  *session.Event
								err error
								ack chan bool
							}
							ch := make(chan item, 8)
							done := make(chan struct{})
							defer close(done)
							var (
								wg   sync.WaitGroup
								err1 error
							)
							wg.Add(2)
							go func() {
								defer wg.Done()
								defer failReadyOnce.Do(func() { close(failReady) })
								if !failingFirst {
									select {
									case <-firstStarted:
									case <-time.After(10 * time.Second):
										err1 = errors.New("succeeding run never started")
										return
									}
								}
								failCtx := ctx.WithContext(context.WithValue(ctx, concurrentFailRunKey{}, true))
								for _, e := range worker.Run(failCtx) {
									if e != nil && err1 == nil {
										err1 = e
										if mode.breakOnErr {
											break
										}
										if mode.holdUntilTrailer {
											failReadyOnce.Do(func() { close(failReady) })
											select {
											case <-trailerDelivered:
											case <-done:
												return
											case <-time.After(10 * time.Second):
												return
											}
										}
									}
								}
							}()
							go func() {
								defer wg.Done()
								if failingFirst {
									select {
									case <-firstStarted:
									case <-time.After(10 * time.Second):
										ch <- item{err: errors.New("failing run never started")}
										return
									}
								}
								for ev, e := range worker.Run(ctx) {
									if ev != nil {
										ack := make(chan bool, 1)
										ch <- item{ev: ev, err: e, ack: ack}
										select {
										case cont := <-ack:
											if !cont {
												return
											}
										case <-done:
											return
										}
										continue
									}
									ch <- item{ev: ev, err: e}
								}
							}()
							go func() {
								wg.Wait()
								close(ch)
							}()
							for it := range ch {
								cont := yield(it.ev, it.err)
								if it.ack != nil {
									it.ack <- cont
								}
								if it.ev != nil && len(it.ev.Content.Parts) > 0 && it.ev.Content.Parts[0].Text == "peer trailer" {
									trailerOnce.Do(func() { close(trailerDelivered) })
								}
								if !cont {
									return
								}
							}
							if err1 != nil {
								yield(nil, err1)
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}

				var gotErr error
				for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						gotErr = err
						break
					}
				}
				if !errors.Is(gotErr, modelErr) {
					t.Fatalf("runner.Run error = %v, want %v", gotErr, modelErr)
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				invErr := onlyRowOfType(t, rows, EventInvocationError)
				if got := invErr["error_message"]; got != modelErr.Error() {
					t.Errorf("INVOCATION_ERROR error_message = %v, want %q", got, modelErr.Error())
				}
				llmReqs := rowsOfType(rows, EventLLMRequest)
				llmResps := rowsOfType(rows, EventLLMResponse)
				llmErrs := rowsOfType(rows, EventLLMError)
				wantReqs, wantResps, wantErrs := 2, 1, 1
				if mode.priorToolTurn {
					// The failing activation makes two model calls (one returning
					// a function call, one failing), so 3 LLM_REQUEST, 2
					// LLM_RESPONSE (one for the function call, one for the peer)
					// and 1 LLM_ERROR.
					wantReqs, wantResps, wantErrs = 3, 2, 1
				}
				if len(llmReqs) != wantReqs || len(llmResps) != wantResps || len(llmErrs) != wantErrs {
					t.Fatalf("got LLM rows req=%d resp=%d err=%d, want %d/%d/%d; event types: %v", len(llmReqs), len(llmResps), len(llmErrs), wantReqs, wantResps, wantErrs, eventTypes(rows))
				}
				if llmResps[0]["span_id"] == llmErrs[0]["span_id"] {
					t.Errorf("LLM_RESPONSE and LLM_ERROR shared span_id %v, want distinct model spans", llmResps[0]["span_id"])
				}
				wantWorkerErrs := 0
				if mode.breakOnErr {
					wantWorkerErrs = 1
				}
				if got := agentErrorCounts(rows)["worker"]; got != wantWorkerErrs {
					t.Errorf("AGENT_ERROR count for worker = %d, want %d; counts=%v", got, wantWorkerErrs, agentErrorCounts(rows))
				}
				if got := agentErrorCounts(rows)["parent"]; got != wantWorkerErrs {
					t.Errorf("AGENT_ERROR count for parent = %d, want %d; counts=%v", got, wantWorkerErrs, agentErrorCounts(rows))
				}
			})
		}
	}
}

// TestFailedParallelAgent_RecoveredByAfterAgentCallbackWithConcurrentUndeclaredSibling
// verifies that when a custom root without SubAgents declared runs a guard
// sub-agent concurrently with a sibling sub-agent, and the guard swallows a
// parallel sub-agent failure and recovers from its own AfterAgentCallback while
// the sibling is still open at the top of agentStack, completeResponse clears
// the swallowable retiredCause on the ancestor root span so the invocation
// completes with OK.
func TestFailedParallelAgent_RecoveredByAfterAgentCallbackWithConcurrentUndeclaredSibling(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	guardStarted := make(chan struct{})
	siblingOpen := make(chan struct{})
	guardRecovered := make(chan struct{})
	answered := make(chan struct{})

	answering, err := agent.New(agent.Config{
		Name: "answering",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				defer close(answered)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "answering"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "answering branch answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(answering): %v", err)
	}
	failing, err := agent.New(agent.Config{
		Name: "failing",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "failing"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "failing branch answer")
				if !yield(ev, nil) {
					return
				}
				select {
				case <-answered:
				case <-time.After(10 * time.Second):
				}
				yield(nil, errors.New("branch failed after answering"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failing): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{answering, failing}},
	})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}

	guard, err := agent.New(agent.Config{
		Name:      "guard",
		SubAgents: []agent.Agent{par},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				close(guardStarted)
				select {
				case <-siblingOpen:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("sibling never opened"))
					return
				}
				for ev, runErr := range par.Run(ctx) {
					if runErr != nil {
						continue
					}
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				return textContent("model", "guard recovered"), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("agent.New(guard): %v", err)
	}

	sibling, err := agent.New(agent.Config{
		Name: "sibling",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				close(siblingOpen)
				select {
				case <-guardRecovered:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("guard never recovered"))
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "sibling"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "sibling ok")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(sibling): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name: "root",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				type item struct {
					ev  *session.Event
					err error
					ack chan bool
				}
				ch := make(chan item, 16)
				done := make(chan struct{})
				defer close(done)
				send := func(ev *session.Event, err error) bool {
					if ev != nil {
						ack := make(chan bool, 1)
						ch <- item{ev: ev, err: err, ack: ack}
						select {
						case cont := <-ack:
							return cont
						case <-done:
							return false
						}
					}
					ch <- item{ev: ev, err: err}
					return true
				}
				var wg sync.WaitGroup
				wg.Add(2)
				go func() {
					defer wg.Done()
					for ev, err := range guard.Run(ctx) {
						if !send(ev, err) {
							return
						}
					}
				}()
				go func() {
					defer wg.Done()
					select {
					case <-guardStarted:
					case <-time.After(10 * time.Second):
						ch <- item{err: errors.New("guard never started")}
						return
					}
					for ev, err := range sibling.Run(ctx) {
						if !send(ev, err) {
							return
						}
					}
				}()
				go func() {
					wg.Wait()
					close(ch)
				}()
				for it := range ch {
					cont := yield(it.ev, it.err)
					if it.ack != nil {
						it.ack <- cont
					}
					if !cont {
						return
					}
					if it.ev != nil && it.ev.Author == "guard" {
						close(guardRecovered)
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
	}
	if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
		t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
	}
	if counts := agentErrorCounts(rows); counts["guard"] != 0 || counts["sibling"] != 0 || counts["root"] != 0 {
		t.Errorf("AGENT_ERROR counts = %v, want 0 for guard, sibling, and root", counts)
	}
}

// TestRunner_WorkflowAgentWithAfterAgentCallbackDrainedAfterBranchError_EmitsInvocationError
// verifies that built-in workflow agents (such as ParallelAgent directly or
// SequentialAgent wrapping ParallelAgent) that propagate child errors to their
// caller never clear their retiredCauses when drained through their own
// AfterAgentCallback returning trailer content: the invocation still ends with
// INVOCATION_ERROR.
func TestRunner_WorkflowAgentWithAfterAgentCallbackDrainedAfterBranchError_EmitsInvocationError(t *testing.T) {
	for _, wrapInSeq := range []bool{false, true} {
		name := "parallel root with AfterAgentCallback"
		if wrapInSeq {
			name = "sequential root with AfterAgentCallback wrapping parallel"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			answered := make(chan struct{})
			branchErr := errors.New("branch failed after answering")

			answering, err := agent.New(agent.Config{
				Name: "answering",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						defer close(answered)
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "answering"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "answering branch answer")
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(answering): %v", err)
			}
			failing, err := agent.New(agent.Config{
				Name: "failing",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "failing"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "failing branch answer")
						if !yield(ev, nil) {
							return
						}
						select {
						case <-answered:
						case <-time.After(10 * time.Second):
						}
						yield(nil, branchErr)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(failing): %v", err)
			}

			parCfg := agent.Config{
				Name:      "par",
				SubAgents: []agent.Agent{answering, failing},
			}
			if !wrapInSeq {
				parCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						return textContent("model", "par trailer"), nil
					},
				}
			}
			par, err := parallelagent.New(parallelagent.Config{AgentConfig: parCfg})
			if err != nil {
				t.Fatalf("parallelagent.New: %v", err)
			}

			root := par
			if wrapInSeq {
				root, err = sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{
						Name:      "seq_root",
						SubAgents: []agent.Agent{par},
						AfterAgentCallbacks: []agent.AfterAgentCallback{
							func(agent.Context) (*genai.Content, error) {
								return textContent("model", "seq trailer"), nil
							},
						},
					},
				})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
			}

			var sawErr error
			for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil && sawErr == nil {
					sawErr = err
				}
			}
			if !errors.Is(sawErr, branchErr) {
				t.Fatalf("runner.Run error = %v, want %v", sawErr, branchErr)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			if invErr["status"] != "ERROR" {
				t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invErr["status"])
			}
		})
	}
}

// TestRunner_WorkflowRootContentlessPartialStoppedAfterSiblingNodeStarts_EmitsCompleted
// verifies that when a root workflowagent emits a contentless partial event via
// an emitting FunctionNode (on branch="" and root span path="") and another
// workflow node starts sub-agent work under the root before the caller stops
// reading on that partial event, partialLineage is preserved so the
// consumer-stop cancellation under the root workflow settles cleanly with
// INVOCATION_COMPLETED rather than INVOCATION_ERROR.
func TestRunner_WorkflowRootContentlessPartialStoppedAfterSiblingNodeStarts_EmitsCompleted(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	launcherReady := make(chan struct{})
	partialDelivered := make(chan struct{})
	waiterStarted := make(chan struct{})

	emitNode := workflow.NewEmittingFunctionNode[any, any](
		"emitter",
		func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			select {
			case <-launcherReady:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			partial := session.NewEvent(ctx, ctx.InvocationID())
			partial.Author = "wf_root"
			partial.Partial = true
			_ = emit(partial)
			<-ctx.Done()
			return nil, ctx.Err()
		},
		workflow.NodeConfig{},
	)
	waiter, err := agent.New(agent.Config{
		Name: "waiter",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				close(waiterStarted)
				<-ctx.Done()
				yield(nil, ctx.Err())
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(waiter): %v", err)
	}
	launcher, err := agent.New(agent.Config{
		Name:      "launcher",
		SubAgents: []agent.Agent{waiter},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				close(launcherReady)
				select {
				case <-partialDelivered:
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				}
				for ev, runErr := range waiter.Run(ctx) {
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(launcher): %v", err)
	}
	launcherNode, err := workflow.NewAgentNode(launcher, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(launcher): %v", err)
	}
	wfRoot, err := workflowagent.New(workflowagent.Config{
		Name: "wf_root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: emitNode},
			{From: workflow.Start, To: launcherNode},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(wf_root): %v", err)
	}

	var sawPartial bool
	for ev, err := range newTestRunner(t, wfRoot, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
		if ev != nil && ev.Partial {
			close(partialDelivered)
			select {
			case <-waiterStarted:
			case <-time.After(10 * time.Second):
				t.Fatal("waiter did not start before consumer stop")
			}
			sawPartial = true
			break
		}
	}
	if !sawPartial {
		t.Fatal("runner.Run yielded no partial event")
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	var waiterStarts int
	for _, row := range rowsOfType(rows, EventAgentStarting) {
		if row["agent"] == "waiter" {
			waiterStarts++
		}
	}
	if waiterStarts != 1 {
		t.Errorf("got %d AGENT_STARTING rows for waiter, want 1; event types: %v", waiterStarts, eventTypes(rows))
	}
}

// TestConcurrencyRace_ParallelLookingBranchOverrideUnderCustomParent verifies
// under -race that when a custom parent (not a ParallelAgent and without
// SubAgents declared) launches a sibling on a parallel-looking branch
// ("parent.sibling") and a BeforeAgentCallback-answered child on a nested
// branch ("parent.sibling.nested"), retiring the sibling while the parent is
// still open does not bypass launcherOpenLocked or read the child's Ended()
// state concurrently with ADK's EndInvocation().
func TestConcurrencyRace_ParallelLookingBranchOverrideUnderCustomParent(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	siblingReady := make(chan struct{})
	cachedAnswering := make(chan struct{})

	cachedChild, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(cachedAnswering)
				return textContent("model", "cached shortcut"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}

	sibling, err := agent.New(agent.Config{
		Name: "sibling",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "sibling"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "sibling answer")
				if !yield(ev, nil) {
					return
				}
				close(siblingReady)
				select {
				case <-cachedAnswering:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("cached_child did not answer"))
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(sibling): %v", err)
	}

	parent, err := agent.New(agent.Config{
		Name: "parent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				answer := session.NewEvent(ctx, ctx.InvocationID())
				answer.Author = "parent"
				answer.Branch = ctx.Branch()
				answer.Content = textContent("model", "parent answer")
				if !yield(answer, nil) {
					return
				}
				type item struct {
					ev  *session.Event
					err error
					ack chan bool
				}
				ch := make(chan item, 4)
				done := make(chan struct{})
				defer close(done)
				var wg sync.WaitGroup
				wg.Add(2)
				go func() {
					defer wg.Done()
					sibBranch := "parent.sibling"
					sibCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &sibBranch})
					for ev, runErr := range sibling.Run(sibCtx) {
						if ev != nil {
							ack := make(chan bool, 1)
							ch <- item{ev: ev, err: runErr, ack: ack}
							select {
							case cont := <-ack:
								if !cont {
									return
								}
							case <-done:
								return
							}
							continue
						}
						ch <- item{ev: ev, err: runErr}
					}
				}()
				go func() {
					defer wg.Done()
					select {
					case <-siblingReady:
					case <-time.After(10 * time.Second):
						ch <- item{err: errors.New("sibling did not start")}
						return
					}
					nestedBranch := "parent.sibling.nested"
					nestedCtx := ctx.WithICDelta(&agent.InvocationContextDelta{Branch: &nestedBranch})
					// Drop cached_child's event so settlement relies on contextEnded.
					for _, runErr := range cachedChild.Run(nestedCtx) {
						if runErr != nil {
							ch <- item{err: runErr}
							return
						}
					}
				}()
				go func() {
					wg.Wait()
					close(ch)
				}()
				for it := range ch {
					cont := yield(it.ev, it.err)
					if it.ack != nil {
						it.ack <- cont
					}
					if !cont {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}

	for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	requireInvocationCompletedOK(t, collectRows(t, requests, untilInvocationEnd))
}

// TestConcurrencyRace_CustomParentReturnsBeforeUnjoinedChildEndsInvocation
// verifies under -race that when a custom parent spawns a child goroutine whose
// BeforeAgentCallback unblocks the parent and returns non-nil content, and the
// parent returns without joining the child goroutine (both with and without
// SubAgents declared), popAgentSpan on the parent does not read the child's
// unsynchronized Ended() state concurrently with ADK's EndInvocation() write.
//
// It also covers (b) a parent that yields its own answer, runs a sibling child
// to completion forwarding its events, then starts a cached child (answered by
// BeforeAgentCallback) without joining it, and (c) a parent that starts a
// cached child on a goroutine, joins it (waits for it to finish), but does not
// forward its event. All variants must end with INVOCATION_COMPLETED.
func TestConcurrencyRace_CustomParentReturnsBeforeUnjoinedChildEndsInvocation(t *testing.T) {
	for _, declareSubAgents := range []bool{true, false} {
		name := "undeclared_subagent"
		if declareSubAgents {
			name = "declared_subagent"
		}
		t.Run(name, func(t *testing.T) {
			// (a) unjoined child answered by BeforeAgentCallback.
			t.Run("unjoined_cached_child", func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				childInBefore := make(chan struct{})
				childDone := make(chan struct{})

				child, err := agent.New(agent.Config{
					Name: "unjoined_child",
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							close(childInBefore)
							return textContent("model", "shortcut"), nil
						},
					},
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(unjoined_child): %v", err)
				}

				parentCfg := agent.Config{
					Name: "parent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							answer := session.NewEvent(ctx, ctx.InvocationID())
							answer.Author = "parent"
							answer.Branch = ctx.Branch()
							answer.Content = textContent("model", "parent answer")
							if !yield(answer, nil) {
								return
							}
							go func() {
								defer close(childDone)
								for range child.Run(ctx) {
								}
							}()
							select {
							case <-childInBefore:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("unjoined_child never entered BeforeAgentCallback"))
							}
							// Return immediately without waiting for childDone so
							// parent's AfterAgentCallback overlaps child's EndInvocation().
						}
					},
				}
				if declareSubAgents {
					parentCfg.SubAgents = []agent.Agent{child}
				}
				parent, err := agent.New(parentCfg)
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}

				for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				select {
				case <-childDone:
				case <-time.After(10 * time.Second):
					t.Fatal("background child goroutine did not finish")
				}
			})

			// (b) completed_sibling_then_unjoined_cached_child: a parent that
			// yields its own answer, then runs a sibling child to completion
			// (forwarding its events), then starts a cached child (answered by
			// BeforeAgentCallback) without joining it and returns.
			t.Run("completed_sibling_then_unjoined_cached_child", func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				cachedInBefore := make(chan struct{})
				cachedDone := make(chan struct{})

				cachedChild, err := agent.New(agent.Config{
					Name: "cached_child",
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							close(cachedInBefore)
							return textContent("model", "cache hit"), nil
						},
					},
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(cached_child): %v", err)
				}

				sibling, err := agent.New(agent.Config{
					Name: "sibling",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "sibling"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "sibling answer")
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(sibling): %v", err)
				}

				parentCfg := agent.Config{
					Name: "parent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							answer := session.NewEvent(ctx, ctx.InvocationID())
							answer.Author = "parent"
							answer.Branch = ctx.Branch()
							answer.Content = textContent("model", "parent answer")
							if !yield(answer, nil) {
								return
							}
							// Run the sibling to completion, forwarding its events.
							for ev, runErr := range sibling.Run(ctx) {
								if !yield(ev, runErr) {
									return
								}
							}
							// Start the cached child without joining it.
							go func() {
								defer close(cachedDone)
								for range cachedChild.Run(ctx) {
								}
							}()
							select {
							case <-cachedInBefore:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("cached_child never entered BeforeAgentCallback"))
							}
							// Return without waiting for cachedDone so the
							// parent's AfterAgentCallback overlaps the cached
							// child's EndInvocation().
						}
					},
				}
				if declareSubAgents {
					parentCfg.SubAgents = []agent.Agent{sibling, cachedChild}
				}
				parent, err := agent.New(parentCfg)
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}

				for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				select {
				case <-cachedDone:
				case <-time.After(10 * time.Second):
					t.Fatal("background cached child goroutine did not finish")
				}
			})

			// (c) joined_cached_child_without_forwarding: a parent that yields
			// its own answer, then starts a cached child (answered by
			// BeforeAgentCallback) on a goroutine, joins it (waits for it to
			// finish), but does NOT forward its event.
			t.Run("joined_cached_child_without_forwarding", func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				cachedDone := make(chan struct{})

				cachedChild, err := agent.New(agent.Config{
					Name: "cached_child",
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							return textContent("model", "cache hit"), nil
						},
					},
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(cached_child): %v", err)
				}

				parentCfg := agent.Config{
					Name: "parent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							answer := session.NewEvent(ctx, ctx.InvocationID())
							answer.Author = "parent"
							answer.Branch = ctx.Branch()
							answer.Content = textContent("model", "parent answer")
							if !yield(answer, nil) {
								return
							}
							// Start the cached child on a goroutine, drop its
							// event, and wait for it to finish (join).
							go func() {
								defer close(cachedDone)
								for range cachedChild.Run(ctx) {
								}
							}()
							select {
							case <-cachedDone:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("cached_child did not finish"))
							}
						}
					},
				}
				if declareSubAgents {
					parentCfg.SubAgents = []agent.Agent{cachedChild}
				}
				parent, err := agent.New(parentCfg)
				if err != nil {
					t.Fatalf("agent.New(parent): %v", err)
				}

				for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error: %v", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
			})
		})
	}
}

// TestConcurrencyRace_ConcurrentSameAgentWithPromotedSubNodePath verifies under
// -race that when a custom parent promotes a sub-node path via PromoteWithDelta
// (e.g. Path: "custom@1") and runs the same child twice concurrently on that
// path, one run answered by BeforeAgentCallback and the other answering from
// Run, pushAgentSpan does not mistake the concurrent activation for a
// WorkflowAgent RetryConfig retry or read Ended() concurrently with
// EndInvocation().
func TestConcurrencyRace_ConcurrentSameAgentWithPromotedSubNodePath(t *testing.T) {
	for _, cachedFirst := range []bool{true, false} {
		order := "answering_run_starts_first"
		if cachedFirst {
			order = "cached_run_starts_first"
		}
		t.Run(order, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			firstEntered := make(chan struct{})
			cachedAnswered := make(chan struct{})
			var calls atomic.Int32

			child, err := agent.New(agent.Config{
				Name: "child",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						first := calls.Add(1) == 1
						if first {
							close(firstEntered)
						}
						if first == cachedFirst {
							close(cachedAnswered)
							return textContent("model", "cache hit"), nil
						}
						return nil, nil
					},
				},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						select {
						case <-cachedAnswered:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("cached run did not answer"))
							return
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "child"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "child answer")
						yield(ev, nil)
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
						answer := session.NewEvent(ctx, ctx.InvocationID())
						answer.Author = "parent"
						answer.Branch = ctx.Branch()
						answer.Content = textContent("model", "parent answer")
						if !yield(answer, nil) {
							return
						}
						subPath := "custom@1"
						promotedCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath})
						type item struct {
							ev  *session.Event
							err error
							ack chan bool
						}
						ch := make(chan item, 4)
						done := make(chan struct{})
						defer close(done)
						var wg sync.WaitGroup
						wg.Add(2)
						go func() {
							defer wg.Done()
							if !cachedFirst {
								select {
								case <-firstEntered:
								case <-time.After(10 * time.Second):
									ch <- item{err: errors.New("first run never entered")}
									return
								}
							}
							// Drop cached run's event so settlement uses contextEnded.
							for _, runErr := range child.Run(promotedCtx) {
								if runErr != nil {
									ch <- item{err: runErr}
									return
								}
							}
						}()
						go func() {
							defer wg.Done()
							if cachedFirst {
								select {
								case <-firstEntered:
								case <-time.After(10 * time.Second):
									ch <- item{err: errors.New("first run never entered")}
									return
								}
							}
							for ev, runErr := range child.Run(promotedCtx) {
								if ev != nil {
									ack := make(chan bool, 1)
									ch <- item{ev: ev, err: runErr, ack: ack}
									select {
									case cont := <-ack:
										if !cont {
											return
										}
									case <-done:
										return
									}
									continue
								}
								ch <- item{ev: ev, err: runErr}
							}
						}()
						go func() {
							wg.Wait()
							close(ch)
						}()
						for it := range ch {
							cont := yield(it.ev, it.err)
							if it.ack != nil {
								it.ack <- cont
							}
							if !cont {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}

			for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
			}
		})
	}
}

// TestConcurrencyRace_LiveToolConcurrentSameChildWithPromotedPathUnderInvocationRoot
// verifies under -race that a live root agent (RunLive, so its activation
// parent is spanKindInvocation with activation == nil as in a real Live run)
// can run a tool that promotes a context to helper@1 and runs helper twice
// concurrently, each on its own WithICDelta-derived context (distinct
// activation pointers), joined before the tool returns, without
// pushAgentSpan mistaking the concurrent activation for a workflow retry
// or reading Ended() concurrently with EndInvocation().
//
// The first activation is answered by BeforeAgentCallback, which signals the
// second run to start and returns cached content right away, so agent.Run
// calls EndInvocation and skips the model call while the second activation
// starts and runs scriptedLLM. The first run's goroutine then waits until the
// second activation's own BeforeAgentCallback runs, so nothing that it does
// after EndInvocation orders that write before the second activation's
// pushAgentSpan, and reading the first activation's ended state there is
// reported by the race detector. Both helpers run on goroutines of their own,
// so the end of the tool call does not read their state either (see
// markToolRunsDoneLocked), and the invocation must complete with
// INVOCATION_COMPLETED (0 AGENT_ERROR rows).
//
// In RunLive the root agent (live_root) has no span on the agent stack — only
// the invocation root (agentStack[0], activation == nil) — so helper, which is
// not a declared SubAgent, parents directly under agentStack[0]. Both helper
// activations have activation != nil (rec.activation != nil) and distinct
// activation pointers (rec.activation != activation), so isSyntheticRetry
// (activation == nil && rec.activation == nil && rec.parent.activation == nil)
// is false and rec.parent.isWorkflow is false (parent is spanKindInvocation):
// the retry guard in pushAgentSpan does not fire, so the second push neither
// reads nor purges the first. If isSyntheticRetry were reverted to just
// rec.parent.activation == nil, the retry guard WOULD fire (parent has
// activation == nil) and read the first helper's ended state while agent.Run
// writes it.
func TestConcurrencyRace_LiveToolConcurrentSameChildWithPromotedPathUnderInvocationRoot(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	secondEntered := make(chan struct{})
	var calls atomic.Int32

	// helper is an llmagent. The first activation (n == 1) is answered by
	// BeforeAgentCallback: it signals the second run to start and returns
	// cached content, so agent.Run calls EndInvocation and skips the model
	// call. The second activation (n == 2) returns nil, nil so the model call
	// runs (scriptedLLM sets hasResponse via popLLMSpan).
	helper, err := llmagent.New(llmagent.Config{
		Name: "helper",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if calls.Add(1) == 1 {
					close(entered)
					// first activation: return cached content (EndInvocation)
					return genai.NewContentFromText("cached", genai.RoleModel), nil
				}
				close(secondEntered)
				return nil, nil // second activation: run the model
			},
		},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: textContent("model", "ok"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}

	runHelper, err := functiontool.New(functiontool.Config{Name: "run_helper", Description: "Runs helper twice concurrently"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			// Promote the context to helper@1, as a Live tool would.
			subPath := "helper@1"
			promotedCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath})

			// Each run creates its own runCtx via WithICDelta(nil) so each
			// has a distinct activation pointer (rec.activation !=
			// activation), matching two separate WithICDelta calls in a Live
			// tool. The second run starts once the first run's
			// BeforeAgentCallback signals, and the first run waits after
			// each event for wait unless it is nil.
			run := func(wait <-chan struct{}) {
				runCtx := promotedCtx.WithICDelta(nil)
				for _, runErr := range helper.Run(runCtx) {
					if runErr != nil {
						return
					}
					if wait == nil {
						continue
					}
					select {
					case <-wait:
					case <-time.After(10 * time.Second):
					}
				}
			}
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				run(secondEntered)
			}()
			go func() {
				defer wg.Done()
				select {
				case <-entered:
				case <-time.After(10 * time.Second):
					return
				}
				run(nil)
			}()
			wg.Wait()
			return map[string]any{"ok": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(run_helper): %v", err)
	}

	// Use RunLive so the root agent (live_root) has no span: helper parents
	// directly under agentStack[0] (spanKindInvocation, activation == nil),
	// matching real Live runs. The fake server sends a toolCall, then a
	// serverContent with turnComplete to end the turn after the tool returns.
	toolDone := make(chan struct{})
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL: newFakeLiveServer(t,
				`{"toolCall":{"functionCalls":[{"id":"call-1","name":"run_helper","args":{}}]}}`,
				`{"serverContent":{"modelTurn":{"parts":[{"text":"done"}],"role":"model"},"turnComplete":true}}`,
			),
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "live_root",
		Model: &liveTestLLM{client: client},
		Tools: []tool.Tool{runHelper},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				close(toolDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(live_root): %v", err)
	}

	liveSess, events, err := newTestRunner(t, root, p).RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("r.RunLive: %v", err)
	}
	// Close the live session after the tool completes, so the flow shuts
	// down cleanly once both helper goroutines join and the AfterToolCallback
	// fires.
	go func() {
		select {
		case <-toolDone:
		case <-time.After(10 * time.Second):
		}
		_ = liveSess.Close()
	}()
	for _, err := range events {
		if err != nil {
			t.Errorf("live event error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := calls.Load(); n != 2 {
		t.Errorf("helper ran %d times, want 2", n)
	}
}

// TestTraceManager_RetiredPeerTrailerDoesNotClearOpenFailingPeerError is a
// focused deterministic test where a successful peer retires first, leaving a
// failing peer still open on agentStack when the retired peer's
// AfterAgentCallback trailer event arrives. The trailer must not clear the
// failing peer's recorded error: the failing peer overlapped the retired one
// (concurrentPeer), so completeResponse keeps its error even if the trailer
// is taken for its response. TestTraceManager_StaleRetiredPeerTrailerDoesNotClearLaterActivationError
// covers a later activation that overlapped no peer, which only the
// absorption of the trailer (retiredPeerCount) protects. Both single-turn and
// after a prior tool turn variants are covered.
func TestTraceManager_RetiredPeerTrailerDoesNotClearOpenFailingPeerError(t *testing.T) {
	for _, priorToolTurn := range []bool{false, true} {
		name := "single_turn"
		if priorToolTurn {
			name = "after_prior_tool_turn"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var started, inModel atomic.Int32
			firstStarted := make(chan struct{})
			bothInModel := make(chan struct{})
			peerRetired := make(chan struct{})
			failReady := make(chan struct{})
			var failReadyOnce sync.Once
			trailerDelivered := make(chan struct{})
			var trailerOnce sync.Once
			var failCallCount atomic.Int32
			modelErr := errors.New("quota exceeded on failing activation")

			mdl := &concurrentBarrierLLM{
				name: "test-model",
				fn: func(ctx context.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					if inModel.Add(1) == 2 {
						close(bothInModel)
					} else {
						select {
						case <-bothInModel:
						case <-ctx.Done():
							return nil, ctx.Err()
						case <-time.After(10 * time.Second):
							return nil, errors.New("timed out waiting for peer model call")
						}
					}
					isFail := ctx.Value(concurrentFailRunKey{}) != nil
					if priorToolTurn && isFail {
						if failCallCount.Add(1) == 1 {
							return &model.LLMResponse{
								Content: genai.NewContentFromParts([]*genai.Part{
									genai.NewPartFromFunctionCall("probe", map[string]any{}),
								}, genai.RoleModel),
							}, nil
						}
						return nil, modelErr
					}
					if isFail {
						return nil, modelErr
					}
					// Succeeding peer: wait for the failing peer to be ready, then
					// return success so it can retire while the failing peer is
					// still open.
					select {
					case <-failReady:
					case <-ctx.Done():
						return nil, ctx.Err()
					case <-time.After(10 * time.Second):
						return nil, errors.New("timed out waiting for failing activation")
					}
					return &model.LLMResponse{Content: textContent("model", "peer ok"), TurnComplete: true}, nil
				},
			}
			workerCfg := llmagent.Config{
				Name:  "worker",
				Model: mdl,
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						if started.Add(1) == 1 {
							close(firstStarted)
						}
						return nil, nil
					},
				},
				AfterAgentCallbacks: []agent.AfterAgentCallback{
					func(cbCtx agent.Context) (*genai.Content, error) {
						if cbCtx.Value(concurrentFailRunKey{}) != nil {
							return nil, nil
						}
						// Signal that the successful peer retired; its trailer
						// event arrives after retirement.
						close(peerRetired)
						return textContent("model", "peer trailer"), nil
					},
				},
			}
			if priorToolTurn {
				probe, perr := functiontool.New(functiontool.Config{Name: "probe", Description: "Probes before failing"},
					func(agent.Context, struct{}) (map[string]any, error) {
						return map[string]any{"ok": true}, nil
					})
				if perr != nil {
					t.Fatalf("functiontool.New(probe): %v", perr)
				}
				workerCfg.Tools = []tool.Tool{probe}
			}
			worker, err := llmagent.New(workerCfg)
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			parent, err := agent.New(agent.Config{
				Name:      "parent",
				SubAgents: []agent.Agent{worker},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						type item struct {
							ev  *session.Event
							err error
							ack chan bool
						}
						ch := make(chan item, 8)
						done := make(chan struct{})
						defer close(done)
						var (
							wg   sync.WaitGroup
							err1 error
						)
						wg.Add(2)
						// Failing activation.
						go func() {
							defer wg.Done()
							defer failReadyOnce.Do(func() { close(failReady) })
							failCtx := ctx.WithContext(context.WithValue(ctx, concurrentFailRunKey{}, true))
							for _, e := range worker.Run(failCtx) {
								if e != nil && err1 == nil {
									err1 = e
									break
								}
							}
						}()
						// Succeeding activation.
						go func() {
							defer wg.Done()
							for ev, e := range worker.Run(ctx) {
								if ev != nil {
									ack := make(chan bool, 1)
									ch <- item{ev: ev, err: e, ack: ack}
									select {
									case cont := <-ack:
										if !cont {
											return
										}
									case <-done:
										return
									}
									continue
								}
								ch <- item{ev: ev, err: e}
							}
						}()
						go func() {
							wg.Wait()
							close(ch)
						}()
						for it := range ch {
							cont := yield(it.ev, it.err)
							if it.ack != nil {
								it.ack <- cont
							}
							if it.ev != nil && len(it.ev.Content.Parts) > 0 && it.ev.Content.Parts[0].Text == "peer trailer" {
								trailerOnce.Do(func() { close(trailerDelivered) })
							}
							if !cont {
								return
							}
						}
						if err1 != nil {
							yield(nil, err1)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}

			var gotErr error
			for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					gotErr = err
					break
				}
			}
			if !errors.Is(gotErr, modelErr) {
				t.Fatalf("runner.Run error = %v, want %v", gotErr, modelErr)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			if got := invErr["error_message"]; got != modelErr.Error() {
				t.Errorf("INVOCATION_ERROR error_message = %v, want %q", got, modelErr.Error())
			}
			llmErrs := rowsOfType(rows, EventLLMError)
			if len(llmErrs) != 1 {
				t.Fatalf("got %d LLM_ERROR rows, want 1; event types: %v", len(llmErrs), eventTypes(rows))
			}
			// The retired peer's trailer must not have cleared the failing
			// peer's error: the run must end with INVOCATION_ERROR, not
			// INVOCATION_COMPLETED.
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Errorf("got %d INVOCATION_COMPLETED rows, want 0; the retired peer's trailer cleared the failing peer's error", n)
			}
			// Verify the trailer was delivered.
			select {
			case <-trailerDelivered:
			case <-time.After(10 * time.Second):
				t.Fatal("peer trailer was never delivered")
			}
		})
	}
}

// TestConcurrencyRace_FunctionNodeConcurrentCachedChildIsNotRetry verifies
// under -race that a workflow.FunctionNode that runs the same agent twice
// concurrently on its own node path, and joins both runs before it returns,
// does not have the second run taken for a RetryConfig attempt of the first.
// Both activations run under the workflow agent on the same node path with
// distinct activation pointers, but they share the context of the node
// attempt, so the first activation is neither read nor purged when the second
// one starts. The first run's BeforeAgentCallback signals the second run to
// start and returns cached content right away, so agent.Run ends that run's
// InvocationContext while the second run starts, and the first run's
// goroutine then waits until the second activation's own BeforeAgentCallback
// runs, so nothing that it does after EndInvocation orders that write before
// the second activation's pushAgentSpan: reading the first activation's ended
// state there is reported by the race detector, and purging it writes a bogus
// AGENT_ERROR row.
func TestConcurrencyRace_FunctionNodeConcurrentCachedChildIsNotRetry(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	secondEntered := make(chan struct{})
	var calls atomic.Int32
	child, err := agent.New(agent.Config{
		Name: "child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if calls.Add(1) == 1 {
					close(entered)
					return textContent("model", "cache hit"), nil
				}
				close(secondEntered)
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	fan := workflow.NewFunctionNode("fan", func(ctx agent.Context, _ any) (string, error) {
		// runChild runs child and, after each event, waits for wait unless
		// it is nil.
		runChild := func(wait <-chan struct{}) error {
			for _, err := range child.Run(ctx) {
				if err != nil {
					return err
				}
				if wait == nil {
					continue
				}
				select {
				case <-wait:
				case <-time.After(10 * time.Second):
					return errors.New("the second run never entered its BeforeAgentCallback")
				}
			}
			return nil
		}
		errs := make([]error, 2)
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			errs[0] = runChild(secondEntered)
		}()
		go func() {
			defer wg.Done()
			select {
			case <-entered:
			case <-time.After(10 * time.Second):
				errs[1] = errors.New("first run never entered its BeforeAgentCallback")
				return
			}
			errs[1] = runChild(nil)
		}()
		wg.Wait()
		return "fanned out", errors.Join(errs...)
	}, workflow.NodeConfig{})
	root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, fan)})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestConcurrencyRace_FunctionNodeConcurrentChildDistinctCancelScopes is
// TestConcurrencyRace_FunctionNodeConcurrentCachedChildIsNotRetry with a
// separate ctx.WithAgentCancel context for each run of the child: the two
// activations then have different Done channels, as two attempts of a
// workflow node have, but they still run at the same time within one node
// attempt, so the second one must not take the first one for an earlier
// attempt and read or purge it.
func TestConcurrencyRace_FunctionNodeConcurrentChildDistinctCancelScopes(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	secondEntered := make(chan struct{})
	var calls atomic.Int32
	child, err := agent.New(agent.Config{
		Name: "child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if calls.Add(1) == 1 {
					close(entered)
					return textContent("model", "cache hit"), nil
				}
				close(secondEntered)
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	fan := workflow.NewFunctionNode("fan", func(ctx agent.Context, _ any) (string, error) {
		// runChild runs child with its own cancelable context and, after
		// each event, waits for wait unless it is nil.
		runChild := func(wait <-chan struct{}) error {
			childCtx, cancel := ctx.WithAgentCancel()
			defer cancel()
			for _, err := range child.Run(childCtx) {
				if err != nil {
					return err
				}
				if wait == nil {
					continue
				}
				select {
				case <-wait:
				case <-time.After(10 * time.Second):
					return errors.New("the second run never entered its BeforeAgentCallback")
				}
			}
			return nil
		}
		errs := make([]error, 2)
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			errs[0] = runChild(secondEntered)
		}()
		go func() {
			defer wg.Done()
			select {
			case <-entered:
			case <-time.After(10 * time.Second):
				errs[1] = errors.New("first run never entered its BeforeAgentCallback")
				return
			}
			errs[1] = runChild(nil)
		}()
		wg.Wait()
		return "fanned out", errors.Join(errs...)
	}, workflow.NodeConfig{})
	root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, fan)})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestConcurrencyRace_FunctionNodeReturnsBeforeUnjoinedCachedChild verifies
// under -race that a workflow agent that ends does not count as having joined
// an agent that one of its function nodes ran on a goroutine of its own and
// did not wait for: the node returns as soon as the child's
// BeforeAgentCallback has begun, and the child's goroutine then waits until
// the workflow agent's AfterAgentCallback has run, so nothing orders the
// EndInvocation call that agent.Run makes for the cached content before the
// workflow agent's AfterAgentCallback. Reading the child's ended state there
// is reported by the race detector.
func TestConcurrencyRace_FunctionNodeReturnsBeforeUnjoinedCachedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	rootDone := make(chan struct{})
	childDone := make(chan struct{})
	child, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(entered)
				return textContent("model", "cached"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}
	fan := workflow.NewFunctionNode("fan", func(ctx agent.Context, _ any) (string, error) {
		go func() {
			defer close(childDone)
			for range child.Run(ctx) {
				select {
				case <-rootDone:
				case <-time.After(10 * time.Second):
				}
			}
		}()
		select {
		case <-entered:
		case <-time.After(10 * time.Second):
			return "", errors.New("child did not enter its BeforeAgentCallback")
		}
		return "started", nil
	}, workflow.NodeConfig{})
	root, err := workflowagent.New(workflowagent.Config{
		Name:  "root",
		Edges: workflow.Chain(workflow.Start, fan),
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(rootDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	select {
	case <-childDone:
	case <-time.After(10 * time.Second):
		t.Fatal("child did not exit")
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// runUnjoinedLateChild runs the composite agent that build makes over a custom
// sub-agent, launcher, which starts cached_child on a goroutine of its own and
// returns without waiting for it. cached_child starts only once launcher's
// span has ended, so it runs right under the composite agent, on launcher's
// branch, and its BeforeAgentCallback answers for it, so agent.Run ends its
// InvocationContext on that goroutine, which then waits until the composite
// agent's AfterAgentCallback has run. Nothing orders that write before the
// composite agent's end, so reading cached_child's ended state there is
// reported by the race detector.
func runUnjoinedLateChild(t *testing.T, build func(sub agent.Agent, after agent.AfterAgentCallback) (agent.Agent, error)) {
	t.Helper()
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	launcherDone := make(chan struct{})
	parentDone := make(chan struct{})
	childDone := make(chan struct{})
	child, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(entered)
				return textContent("model", "cached"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}
	launcher, err := agent.New(agent.Config{
		Name: "launcher",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				go func() {
					defer close(childDone)
					select {
					case <-launcherDone:
					case <-time.After(10 * time.Second):
						return
					}
					for range child.Run(ctx) {
						select {
						case <-parentDone:
						case <-time.After(10 * time.Second):
						}
					}
				}()
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "launcher"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "launched")
				yield(ev, nil)
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(launcherDone)
				select {
				case <-entered:
				case <-time.After(10 * time.Second):
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("agent.New(launcher): %v", err)
	}
	root, err := build(launcher, func(agent.Context) (*genai.Content, error) {
		close(parentDone)
		return nil, nil
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	select {
	case <-childDone:
	case <-time.After(10 * time.Second):
		t.Fatal("cached_child did not exit")
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestConcurrencyRace_ParallelAgentUnjoinedLateChild verifies under -race
// that a ParallelAgent counts as waiting only for the branch goroutines that
// it started itself: see runUnjoinedLateChild.
func TestConcurrencyRace_ParallelAgentUnjoinedLateChild(t *testing.T) {
	runUnjoinedLateChild(t, func(sub agent.Agent, after agent.AfterAgentCallback) (agent.Agent, error) {
		return parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{
			Name:                "fanout",
			SubAgents:           []agent.Agent{sub},
			AfterAgentCallbacks: []agent.AfterAgentCallback{after},
		}})
	})
}

// TestConcurrencyRace_SequentialAgentUnjoinedLateChild verifies under -race
// that a SequentialAgent counts as waiting only for the sub-agents that ran
// on its own goroutine: see runUnjoinedLateChild.
func TestConcurrencyRace_SequentialAgentUnjoinedLateChild(t *testing.T) {
	runUnjoinedLateChild(t, func(sub agent.Agent, after agent.AfterAgentCallback) (agent.Agent, error) {
		return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
			Name:                "pipeline",
			SubAgents:           []agent.Agent{sub},
			AfterAgentCallbacks: []agent.AfterAgentCallback{after},
		}})
	})
}

// TestConcurrencyRace_RoutedNodeStartedAgainWhileRunning has two join nodes
// route to the same agent node, so the workflow scheduler starts the node a
// second time, on the same node path and branch and under a context of its
// own, while the first run still goes on: the first run ends its
// InvocationContext and then waits until the second run's
// BeforeAgentCallback has begun. The second activation must not take the
// first one for an earlier attempt whose run returned and read its ended
// state or purge it.
func TestConcurrencyRace_RoutedNodeStartedAgainWhileRunning(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	firstStarted := make(chan struct{})
	secondEntered := make(chan struct{})
	// The scheduler keeps the runs of a node by the node's name, so it stops
	// taking events once either run of child completes, and the other run
	// would then wait forever for its event to be processed. yielded[i] is
	// closed once the event of run i+1 was processed; neither run returns
	// before both were.
	yielded := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
	var calls, runs atomic.Int32
	child, err := agent.New(agent.Config{
		Name: "child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if calls.Add(1) == 2 {
					close(secondEntered)
				}
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				n := runs.Add(1)
				if n == 1 {
					close(firstStarted)
					ctx.EndInvocation()
					select {
					case <-secondEntered:
					case <-time.After(10 * time.Second):
					}
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "answer")
				if !yield(ev, nil) || n > 2 {
					return
				}
				close(yielded[n-1])
				select {
				case <-yielded[2-n]:
				case <-time.After(10 * time.Second):
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	x, err := workflow.NewAgentNode(child, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	quick := func(name string) workflow.Node {
		return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) { return name, nil }, workflow.NodeConfig{})
	}
	a, b := quick("a"), quick("b")
	c := workflow.NewFunctionNode("c", func(agent.Context, any) (string, error) {
		select {
		case <-firstStarted:
			return "c", nil
		case <-time.After(10 * time.Second):
			return "", errors.New("the first run of child never started")
		}
	}, workflow.NodeConfig{})
	j1, j2 := workflow.NewJoinNode("j1"), workflow.NewJoinNode("j2")
	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: a},
			{From: workflow.Start, To: b},
			{From: workflow.Start, To: c},
			{From: a, To: j1},
			{From: b, To: j1},
			{From: a, To: j2},
			{From: c, To: j2},
			{From: j1, To: x, Route: workflow.Default},
			{From: j2, To: x, Route: workflow.Default},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Logf("runner.Run error: %v", err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("child started %d times, want 2", got)
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestParseGoroutineTrace covers the header and "created by" lines that
// currentGoroutine reads from runtime.Stack.
func TestParseGoroutineTrace(t *testing.T) {
	const frames = "main.f()\n\t/src/main.go:10 +0x1d\n"
	for _, tc := range []struct {
		name       string
		trace      string
		incomplete bool
		wantID     uint64
		wantOrigin goroutineOrigin
	}{
		{
			name:       "created by",
			trace:      "goroutine 42 [running]:\n" + frames + "created by google.golang.org/adk/v2/workflow.(*scheduler).startNode in goroutine 7\n\t/src/scheduler.go:409 +0x2a5\n",
			wantID:     42,
			wantOrigin: goroutineOrigin{creator: "google.golang.org/adk/v2/workflow.(*scheduler).startNode", parent: 7},
		},
		{
			name:   "main goroutine",
			trace:  "goroutine 1 [running]:\n" + frames,
			wantID: 1,
		},
		{
			name:       "more header fields",
			trace:      "goroutine 42 gp=0xc000102380 m=3 mp=0xc000100008 [running]:\n" + frames + "created by golang.org/x/sync/errgroup.(*Group).Go in goroutine 9\n\t/src/errgroup.go:93 +0x50\n",
			wantID:     42,
			wantOrigin: goroutineOrigin{creator: "golang.org/x/sync/errgroup.(*Group).Go", parent: 9},
		},
		{
			name: "traceback ancestors",
			trace: "goroutine 42 [running]:\n" + frames + "created by main.child in goroutine 9\n\t/src/main.go:20 +0x50\n" +
				"[originating from goroutine 9]:\nmain.child(...)\n\t/src/main.go:19\ncreated by main.parent\n\t/src/main.go:30 +0x60\n",
			wantID:     42,
			wantOrigin: goroutineOrigin{creator: "main.child", parent: 9},
		},
		{
			name:       "truncated",
			trace:      "goroutine 42 [running]:\n" + frames + "created by main.child in goroutine 9\n",
			incomplete: true,
			wantID:     42,
		},
		{
			name:   "created by without goroutine",
			trace:  "goroutine 42 [running]:\n" + frames + "created by main.child\n\t/src/main.go:20 +0x50\n",
			wantID: 42,
		},
		{
			name:   "malformed creator goroutine",
			trace:  "goroutine 42 [running]:\n" + frames + "created by main.child in goroutine x\n",
			wantID: 42,
		},
		{name: "malformed header", trace: "goroutine x [running]:\n" + frames},
		{name: "no header", trace: frames},
		{name: "empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, origin := parseGoroutineTrace([]byte(tc.trace), !tc.incomplete)
			if id != tc.wantID || origin != tc.wantOrigin {
				t.Errorf("parseGoroutineTrace() = %d, %+v, want %d, %+v", id, origin, tc.wantID, tc.wantOrigin)
			}
		})
	}
}

// TestNodeCompletion covers how onEvent recognizes the event that a workflow
// scheduler writes when a node run ends without an output, and the workflow
// agent that wrote it, from the custom metadata keys that ADK uses.
func TestNodeCompletion(t *testing.T) {
	event := func(path string, partial bool, meta map[string]any) *session.Event {
		ev := &session.Event{}
		ev.CustomMetadata = meta
		ev.Partial = partial
		if path != "" {
			ev.NodeInfo = &session.NodeInfo{Path: path}
		}
		return ev
	}
	for _, tc := range []struct {
		name       string
		ev         *session.Event
		wantName   string
		wantFailed bool
		wantOK     bool
	}{
		{
			name: "failed run of a nested workflow's node",
			ev: event("inner@1/flaky@1", false, map[string]any{
				"adk.workflow.base_path.inner": "inner@1", "adk.workflow.base_path.outer": "outer@1", "adk.workflow.node_outcome": "failed",
			}),
			wantName: "inner", wantFailed: true, wantOK: true,
		},
		{
			name:     "cancelled run",
			ev:       event("outer@1/inner@1", false, map[string]any{"adk.workflow.base_path.outer": "outer@1", "adk.workflow.node_outcome": "cancelled"}),
			wantName: "outer", wantFailed: true, wantOK: true,
		},
		{
			name:     "completed run",
			ev:       event("outer@1/inner@1", false, map[string]any{"adk.workflow.base_path.outer": "outer@1", "adk.workflow.node_completed": true}),
			wantName: "outer", wantOK: true,
		},
		{
			name:     "workflow without a node path",
			ev:       event("x", false, map[string]any{"adk.workflow.base_path.wf": "", "adk.workflow.node_completed": true}),
			wantName: "wf", wantOK: true,
		},
		{
			name: "event of a node",
			ev:   event("inner@1/flaky@1", false, map[string]any{"adk.workflow.base_path.inner": "inner@1"}),
		},
		{
			name: "dynamic child of a node",
			ev:   event("wf@1/fn@1/child@1", false, map[string]any{"adk.workflow.base_path.wf": "wf@1", "adk.workflow.node_outcome": "failed"}),
		},
		{
			name: "two workflows at the parent path",
			ev: event("wf@1/x@1", false, map[string]any{
				"adk.workflow.base_path.a": "wf@1", "adk.workflow.base_path.b": "wf@1", "adk.workflow.node_outcome": "failed",
			}),
		},
		{
			name: "no node path",
			ev:   event("", false, map[string]any{"adk.workflow.base_path.wf": "", "adk.workflow.node_outcome": "failed"}),
		},
		{
			name: "partial",
			ev:   event("wf@1/x@1", true, map[string]any{"adk.workflow.base_path.wf": "wf@1", "adk.workflow.node_outcome": "failed"}),
		},
		{name: "nil event"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			name, failed, ok := nodeCompletion(tc.ev)
			if name != tc.wantName || failed != tc.wantFailed || ok != tc.wantOK {
				t.Errorf("nodeCompletion() = %q, %t, %t, want %q, %t, %t", name, failed, ok, tc.wantName, tc.wantFailed, tc.wantOK)
			}
		})
	}
}

// TestTraceManager_RecordNodeCompletionAttributesOnlyUnambiguousRuns checks
// which activation a node completion event marks as over (see
// recordNodeCompletion): only one whose run failed or was canceled, and only
// when exactly one open activation of the workflow agent counts exactly one
// unreported run of the node and its scheduler started exactly one open
// activation for that node. The spans are built directly, with the goroutine
// origins that pushAgentSpan reads from runtime.Stack in a real run.
func TestTraceManager_RecordNodeCompletionAttributesOnlyUnambiguousRuns(t *testing.T) {
	ctx := context.Background()
	const nodePath = "wf@1/x@1"
	key := nodeRunKey{path: nodePath}
	for _, tc := range []struct {
		name string
		// workflows is the number of open activations of the workflow agent
		// wf, each of which counts counted unreported runs of the node x, and
		// nodes the number of open activations of x that the scheduler of the
		// last of them started.
		workflows, counted, nodes int
		failed                    bool
		wantEnded                 bool
		wantCounted               int
	}{
		{name: "one failed run", workflows: 1, counted: 1, nodes: 1, failed: true, wantEnded: true},
		{name: "one completed run", workflows: 1, counted: 1, nodes: 1, wantCounted: 1},
		{name: "run without an open activation", workflows: 1, counted: 1, failed: true},
		{name: "two unreported runs", workflows: 1, counted: 2, nodes: 1, failed: true, wantCounted: 2},
		{name: "two open activations", workflows: 1, counted: 1, nodes: 2, failed: true, wantCounted: 1},
		{name: "two workflow activations", workflows: 2, counted: 1, nodes: 1, failed: true, wantCounted: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			id := identity{invocationID: "inv-node-completion", agentName: "wf", path: "wf@1"}
			tm.invocationSpan(ctx, id, nil)
			inv := tm.invocation(ctx, id)
			inv.mu.Lock()
			var owner *spanRecord
			for k := range tc.workflows {
				owner = &spanRecord{
					spanID: newSpanID(), kind: spanKindAgent, name: "wf", path: "wf@1", parent: inv.agentStack[0],
					isWorkflow: true, goroutineID: uint64(10 + k), nodeRuns: map[nodeRunKey]int{key: tc.counted},
				}
				inv.agentStack = append(inv.agentStack, owner)
			}
			var nodes []*spanRecord
			for k := range tc.nodes {
				node := &spanRecord{
					spanID: newSpanID(), kind: spanKindAgent, name: "x", path: nodePath, parent: owner,
					goroutineID: uint64(20 + k), origin: goroutineOrigin{creator: schedulerCreator, parent: owner.goroutineID},
					attempt: attemptNode,
				}
				inv.agentStack = append(inv.agentStack, node)
				nodes = append(nodes, node)
			}
			inv.mu.Unlock()

			tm.recordNodeCompletion(ctx, id, "wf", nodePath, "", tc.failed)

			inv.mu.Lock()
			defer inv.mu.Unlock()
			for k, node := range nodes {
				if node.nodeRunEnded != tc.wantEnded {
					t.Errorf("nodeRunEnded of activation %d of x = %t, want %t", k, node.nodeRunEnded, tc.wantEnded)
				}
			}
			if got := owner.nodeRuns[key]; got != tc.wantCounted {
				t.Errorf("unreported runs of x = %d, want %d", got, tc.wantCounted)
			}
		})
	}
}

// TestTraceManager_RunsOnThisGoroutineAreOverOnlyAfterTheirFrame checks the
// bound that settleEndedInSubtreeLocked and markToolRunsDoneLocked put on the
// open activations that they take to be over because they started on the
// current goroutine: only one that started there after the activation or tool
// call whose callback runs (a seq above since, or above the call's) ran inside
// that frame, and so returned before the callback ran. One that started there
// earlier, such as the agent that made the tool call, runs below the callback
// and may still go on, and with noFrame none counts. The spans are built
// directly, with the goroutine IDs that pushAgentSpan reads from
// runtime.Stack in a real run; anotherGoroutine stands for a goroutine other
// than the test's.
func TestTraceManager_RunsOnThisGoroutineAreOverOnlyAfterTheirFrame(t *testing.T) {
	ctx := context.Background()
	newInvocation := func(t *testing.T) *invocationTrace {
		t.Helper()
		tm := newTraceManager()
		id := identity{invocationID: "inv-runs-on-this-goroutine", agentName: "custom"}
		tm.invocationSpan(ctx, id, nil)
		return tm.invocation(ctx, id)
	}
	newActivation := func(inv *invocationTrace, name string, parent *spanRecord, goroutineID uint64) *spanRecord {
		rec := &spanRecord{
			spanID: newSpanID(), kind: spanKindAgent, name: name, parent: parent,
			activation: new(int), goroutineID: goroutineID, seq: inv.nextSeqLocked(),
		}
		inv.agentStack = append(inv.agentStack, rec)
		return rec
	}
	t.Run("settleEndedInSubtreeLocked", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			// since returns the since bound for child, the activation that
			// started on the test's goroutine under a custom agent that does
			// not wait for it.
			since    func(child *spanRecord) uint64
			wantDone bool
		}{
			{name: "started after the frame", since: func(child *spanRecord) uint64 { return child.seq - 1 }, wantDone: true},
			{name: "no agent run below the callback", since: func(*spanRecord) uint64 { return 0 }, wantDone: true},
			{name: "started before the frame", since: func(child *spanRecord) uint64 { return child.seq + 1 }},
			{name: "no frame", since: func(*spanRecord) uint64 { return noFrame }},
		} {
			t.Run(tc.name, func(t *testing.T) {
				goid := currentGoroutineID()
				if goid == 0 {
					t.Fatal("currentGoroutineID() returned no ID")
				}
				anotherGoroutine := goid + 1
				inv := newInvocation(t)
				inv.mu.Lock()
				defer inv.mu.Unlock()
				custom := newActivation(inv, "custom", inv.agentStack[0], anotherGoroutine)
				child := newActivation(inv, "child", custom, goid)

				inv.settleEndedInSubtreeLocked(1, false, tc.since(child))

				if child.runDone != tc.wantDone {
					t.Errorf("runDone of an activation with seq %d on this goroutine, since %d = %t, want %t", child.seq, tc.since(child), child.runDone, tc.wantDone)
				}
				if custom.runDone {
					t.Error("runDone of the custom agent on another goroutine = true, want false")
				}
			})
		}
	})
	t.Run("markToolRunsDoneLocked", func(t *testing.T) {
		goid := currentGoroutineID()
		if goid == 0 {
			t.Fatal("currentGoroutineID() returned no ID")
		}
		anotherGoroutine := goid + 1
		inv := newInvocation(t)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		// caller made the tool call on this goroutine, and the tool ran the
		// agent ran on it and started the agent elsewhere on another one.
		caller := newActivation(inv, "caller", inv.agentStack[0], goid)
		call := &spanRecord{spanID: newSpanID(), kind: spanKindTool, parent: caller, seq: inv.nextSeqLocked()}
		ran := newActivation(inv, "ran", caller, goid)
		elsewhere := newActivation(inv, "elsewhere", caller, anotherGoroutine)

		inv.markToolRunsDoneLocked(call)

		for _, c := range []struct {
			rec  *spanRecord
			want bool
		}{{caller, false}, {ran, true}, {elsewhere, false}} {
			if c.rec.runDone != c.want {
				t.Errorf("runDone of %s = %t, want %t", c.rec.name, c.rec.runDone, c.want)
			}
		}
	})
}

// TestCurrentGoroutine checks currentGoroutine against goroutines that the
// test starts: the origin names the function that ran the go statement and
// the goroutine that ran it.
func TestCurrentGoroutine(t *testing.T) {
	parentID, _, _ := currentGoroutine()
	if parentID == 0 {
		t.Fatal("currentGoroutine() returned no ID")
	}
	type result struct {
		id     uint64
		origin goroutineOrigin
	}
	results := make(chan result, 1)
	go func() {
		id, origin, _ := currentGoroutine()
		results <- result{id, origin}
	}()
	got := <-results
	if got.id == 0 || got.id == parentID {
		t.Errorf("currentGoroutine() ID = %d, want a new goroutine's (not %d)", got.id, parentID)
	}
	if !strings.HasSuffix(got.origin.creator, "/agentanalytics.TestCurrentGoroutine") || got.origin.parent != parentID {
		t.Errorf("currentGoroutine() origin = %+v, want TestCurrentGoroutine in goroutine %d", got.origin, parentID)
	}
}

// TestGoroutineCreatorsMatchADK checks that schedulerCreator and
// errgroupCreator name the functions that start the goroutines of the agent
// node of a workflow agent and of the branches of a ParallelAgent in the ADK
// release under test, on the goroutines on which those agents started.
func TestGoroutineCreatorsMatchADK(t *testing.T) {
	type started struct {
		id     uint64
		origin goroutineOrigin
	}
	var mu sync.Mutex
	starts := map[string]started{}
	record := func(name string) agent.BeforeAgentCallback {
		return func(agent.Context) (*genai.Content, error) {
			id, origin, _ := currentGoroutine()
			mu.Lock()
			defer mu.Unlock()
			starts[name] = started{id, origin}
			return nil, nil
		}
	}
	leaf, err := agent.New(agent.Config{
		Name:                 "leaf",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{record("leaf")},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "leaf"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "leaf answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(leaf): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{
		Name:                 "par",
		SubAgents:            []agent.Agent{leaf},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{record("par")},
	}})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(par, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	root, err := workflowagent.New(workflowagent.Config{
		Name:                 "root",
		Edges:                workflow.Chain(workflow.Start, node),
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{record("root")},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if got, want := starts["par"].origin, (goroutineOrigin{creator: schedulerCreator, parent: starts["root"].id}); got != want || want.parent == 0 {
		t.Errorf("origin of the agent node's goroutine = %+v, want %+v", got, want)
	}
	if got, want := starts["leaf"].origin, (goroutineOrigin{creator: errgroupCreator, parent: starts["par"].id}); got != want || want.parent == 0 {
		t.Errorf("origin of the parallel branch's goroutine = %+v, want %+v", got, want)
	}
}

// TestAttemptSiteOfTrace covers the stack traces that attemptSiteOfTrace
// reads: only a goroutine that runs nothing but one agent node attempt, from
// the function that the go statement started up to the BeforeAgentCallback of
// the node's agent, shows an attempt.
func TestAttemptSiteOfTrace(t *testing.T) {
	var (
		callback  = []string{"example.com/plugin.(*recorder).beforeAgent", adkPkgPrefix + "plugin.(*PluginManager).RunBeforeAgentCallback", agentPkgPath + ".runBeforeAgentCallbacks"}
		agentRun  = agentPkgPath + ".(*agent).Run.func1"
		nodeRun   = workflowPkgPath + ".(*AgentNode).Run.func1"
		runNode   = workflowPkgPath + ".runNode"
		runWorker = workflowPkgPath + ".(*ParallelWorker).runWorker"
		runOnce   = workflowPkgPath + ".(*ParallelWorker).runWrappedOnce"
		adapter   = adkPkgPrefix + "agent/llmagent.RunLLMAgentAsNode.func1"
		fnRun     = workflowPkgPath + ".(*FunctionNode).Run.func1"
		userCode  = "example.com/app.handle.func1"
		adkCode   = adkPkgPrefix + "workflow_test.TestFunctionNode.func1"
		startNode = workflowPkgPath + ".(*scheduler).startNode"
		startItem = workflowPkgPath + ".(*ParallelWorker).Run.func1"
	)
	// trace writes frames from the innermost one down as runtime.Stack does;
	// a frame that starts with "..." is written as it is.
	trace := func(createdBy string, frames ...string) []byte {
		var b strings.Builder
		b.WriteString("goroutine 41 [running]:\n")
		for _, f := range frames {
			if strings.HasPrefix(f, "...") {
				b.WriteString(f + "\n")
				continue
			}
			fmt.Fprintf(&b, "%s(0xc000123456, {0x1, 0x2})\n\t/src/file.go:10 +0x2a\n", f)
		}
		if createdBy != "" {
			fmt.Fprintf(&b, "created by %s in goroutine 7\n\t/src/file.go:20 +0x3b\n", createdBy)
		}
		return []byte(b.String())
	}
	// started lists the frames of a BeforeAgentCallback of an agent run
	// above the frames below it.
	started := func(below ...string) []string {
		return slices.Concat(callback, []string{agentRun}, below)
	}
	tests := []struct {
		name     string
		trace    []byte
		complete bool
		want     attemptSite
	}{
		{name: "agent node", trace: trace(startNode, started(nodeRun, runNode)...), complete: true, want: attemptNode},
		{name: "agent node with frames that adapt the agent", trace: trace(startNode, started(adapter, nodeRun, runNode)...), complete: true, want: attemptNode},
		{name: "agent node with go statement wrapper and runtime frames", trace: trace(startNode, started(nodeRun, runNode, startNode+".gowrap1", "runtime.goexit")...), complete: true, want: attemptNode},
		{name: "parallel worker item", trace: trace(startItem, started(nodeRun, runOnce, runWorker)...), complete: true, want: attemptWorker},
		{name: "parallel worker item without runWrappedOnce", trace: trace(startItem, started(nodeRun, runWorker)...), complete: true, want: attemptWorker},
		{name: "incomplete trace", trace: trace(startNode, started(nodeRun, runNode)...), complete: false, want: attemptNone},
		{name: "elided frames", trace: trace(startNode, started("...12 frames elided...", nodeRun, runNode)...), complete: true, want: attemptNone},
		{name: "empty trace", trace: nil, complete: true, want: attemptNone},
		{name: "function node", trace: trace(startNode, started(userCode, fnRun, runNode)...), complete: true, want: attemptNone},
		{name: "function node of a parallel worker item", trace: trace(startItem, started(userCode, fnRun, runOnce, runWorker)...), complete: true, want: attemptNone},
		// Only frames above an AgentNode's Run adapt the agent to a node, so
		// a function node is no attempt even when its function is in a
		// package under ADK's import path.
		{name: "function node whose function is in an ADK package", trace: trace(startNode, started(adkCode, fnRun, runNode)...), complete: true, want: attemptNone},
		{name: "function node of a parallel worker item whose function is in an ADK package", trace: trace(startItem, started(adkCode, fnRun, runOnce, runWorker)...), complete: true, want: attemptNone},
		{name: "agent run inside the run of a node's agent", trace: trace(startNode, started(userCode+"-range1", agentRun, nodeRun, runNode)...), complete: true, want: attemptNone},
		{name: "agent run inside a callback of a node's agent", trace: trace(startNode, started(userCode, agentPkgPath+".runBeforeAgentCallbacks", agentRun, nodeRun, runNode)...), complete: true, want: attemptNone},
		{name: "iter.Pull goroutine", trace: trace("iter.Pull2[...]", started("iter.Pull2[...].func1")...), complete: true, want: attemptNone},
		{name: "not in a BeforeAgentCallback", trace: trace(startNode, userCode, agentRun, nodeRun, runNode), complete: true, want: attemptNone},
		{name: "goroutine of the runner", trace: trace("testing.(*T).Run", started(adkPkgPrefix+"runner.(*Runner).Run.func1", userCode, "testing.tRunner")...), complete: true, want: attemptNone},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := attemptSiteOfTrace(tc.trace, tc.complete); got != tc.want {
				t.Errorf("attemptSiteOfTrace() = %d, want %d; trace:\n%s", got, tc.want, tc.trace)
			}
		})
	}
	if got := attemptSiteOfTrace([]byte("goroutine 41 [running]:\n"+runNode+"\n\t/src/file.go:10\n"), true); got != attemptNone {
		t.Errorf("attemptSiteOfTrace() of a frame without arguments = %d, want %d", got, attemptNone)
	}
}

// TestAttemptSitesMatchADK checks attemptSiteOfTrace against the goroutines
// on which the ADK release under test runs agents: the agent of an agent node
// that the workflow scheduler starts, with or without frames that adapt an
// LLM agent to the node, and the agent node of a workflow.ParallelWorker item
// are attempts, while the agent that the runner runs, a sub-agent of a node's
// agent, an agent that a function node runs once, nested inside its loop over
// the events of another run or through iter.Pull2, and one that the function
// node of a ParallelWorker item runs are not.
func TestAttemptSitesMatchADK(t *testing.T) {
	var mu sync.Mutex
	sites := map[string][]attemptSite{}
	record := []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
		_, _, site := currentGoroutine()
		mu.Lock()
		defer mu.Unlock()
		sites[ctx.AgentName()] = append(sites[ctx.AgentName()], site)
		return nil, nil
	}}
	// newAgent returns a custom agent that runs subAgents one after another
	// and then answers.
	newAgent := func(name string, subAgents ...agent.Agent) agent.Agent {
		a, err := agent.New(agent.Config{
			Name:                 name,
			SubAgents:            subAgents,
			BeforeAgentCallbacks: record,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, sub := range subAgents {
						for _, err := range sub.Run(ctx) {
							if err != nil {
								yield(nil, err)
								return
							}
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = name
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", name+" answer")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%s): %v", name, err)
		}
		return a
	}
	agentNode := func(a agent.Agent) workflow.Node {
		n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(%s): %v", a.Name(), err)
		}
		return n
	}
	drain := func(seq iter.Seq2[*session.Event, error]) error {
		for _, err := range seq {
			if err != nil {
				return err
			}
		}
		return nil
	}
	llm, err := llmagent.New(llmagent.Config{
		Name:                 "llm_node",
		Model:                &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "llm answer")}}},
		BeforeAgentCallbacks: record,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	fnChild, nestedChild, pullChild, itemChild := newAgent("fn_child"), newAgent("nested_child"), newAgent("pull_child"), newAgent("item_fn_child")
	fn := workflow.NewFunctionNode("fn", func(ctx agent.Context, _ any) (string, error) {
		if err := drain(fnChild.Run(ctx)); err != nil {
			return "", err
		}
		for _, err := range nestedChild.Run(ctx) {
			if err != nil {
				return "", err
			}
			if err := drain(nestedChild.Run(ctx)); err != nil {
				return "", err
			}
		}
		next, stop := iter.Pull2(pullChild.Run(ctx))
		defer stop()
		for {
			_, err, ok := next()
			if !ok {
				return "fn", nil
			}
			if err != nil {
				return "", err
			}
		}
	}, workflow.NodeConfig{})
	items := workflow.NewFunctionNode("items", func(agent.Context, any) ([]string, error) {
		return []string{"a", "b"}, nil
	}, workflow.NodeConfig{})
	perItem, err := workflow.NewParallelWorker("per_item", agentNode(newAgent("worker_agent")), 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewParallelWorker(per_item): %v", err)
	}
	itemFn := workflow.NewFunctionNode("item_fn", func(ctx agent.Context, _ any) (string, error) {
		return "item_fn", drain(itemChild.Run(ctx))
	}, workflow.NodeConfig{})
	perItemFn, err := workflow.NewParallelWorker("per_item_fn", itemFn, 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewParallelWorker(per_item_fn): %v", err)
	}
	nodeAgent, llmNode := agentNode(newAgent("node_agent", newAgent("sub_agent"))), agentNode(llm)
	// A workflow may have only one terminal node with an output.
	done := workflow.NewJoinNode("done")
	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: nodeAgent},
			{From: workflow.Start, To: llmNode},
			{From: workflow.Start, To: fn},
			{From: workflow.Start, To: items},
			{From: items, To: perItem},
			{From: items, To: perItemFn},
			{From: nodeAgent, To: done},
			{From: llmNode, To: done},
			{From: fn, To: done},
			{From: perItem, To: done},
			{From: perItemFn, To: done},
		},
		BeforeAgentCallbacks: record,
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	want := map[string][]attemptSite{
		"root":          {attemptNone},
		"node_agent":    {attemptNode},
		"sub_agent":     {attemptNone},
		"llm_node":      {attemptNode},
		"fn_child":      {attemptNone},
		"nested_child":  {attemptNone, attemptNone},
		"pull_child":    {attemptNone},
		"worker_agent":  {attemptWorker, attemptWorker},
		"item_fn_child": {attemptNone, attemptNone},
	}
	if diff := cmp.Diff(want, sites); diff != "" {
		t.Errorf("attempt sites per agent mismatch (-want +got):\n%s", diff)
	}
}

// newOneEventAgent returns a custom agent that answers with one event and
// then, if afterYield is not nil, calls it with its InvocationContext and the
// number of the run, counting from 1.
func newOneEventAgent(t *testing.T, name string, afterYield func(ctx agent.InvocationContext, run int32)) agent.Agent {
	t.Helper()
	var runs atomic.Int32
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				run := runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", name+" answer")
				if yield(ev, nil) && afterYield != nil {
					afterYield(ctx, run)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%s): %v", name, err)
	}
	return a
}

// functionNodeEdges returns the edges of a workflow that runs fn once, right
// after its start or, when underWorker is set, as the node that a
// workflow.ParallelWorker runs for its only item.
func functionNodeEdges(t *testing.T, fn workflow.Node, underWorker bool) []workflow.Edge {
	t.Helper()
	if !underWorker {
		return workflow.Chain(workflow.Start, fn)
	}
	items := workflow.NewFunctionNode("items", func(agent.Context, any) ([]string, error) {
		return []string{"a"}, nil
	}, workflow.NodeConfig{})
	perItem, err := workflow.NewParallelWorker("per_item", fn, 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewParallelWorker: %v", err)
	}
	return workflow.Chain(workflow.Start, items, perItem)
}

// TestRunner_FunctionNodeNestedChildRuns_EmitNoAgentError has a function
// node, on its own and as the node of a workflow.ParallelWorker item, run an
// agent again inside its loop over the events of an earlier run of the
// agent, which stays suspended in a yield meanwhile. Both runs share the
// parent, node path, branch and goroutine, and neither fails: the second
// activation must not take the first for an earlier attempt whose run is
// over, so no AGENT_ERROR row is written and the invocation completes.
func TestRunner_FunctionNodeNestedChildRuns_EmitNoAgentError(t *testing.T) {
	for _, tc := range []struct {
		name        string
		underWorker bool
	}{
		{name: "function_node"},
		{name: "parallel_worker_function_node", underWorker: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			child := newOneEventAgent(t, "child", nil)
			var nestedRuns atomic.Int32
			fn := workflow.NewFunctionNode("fn", func(ctx agent.Context, _ any) (string, error) {
				for _, err := range child.Run(ctx) {
					if err != nil {
						return "", err
					}
					for _, err := range child.Run(ctx) {
						if err != nil {
							return "", err
						}
					}
					nestedRuns.Add(1)
				}
				return "nested", nil
			}, workflow.NodeConfig{})
			root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: functionNodeEdges(t, fn, tc.underWorker)})
			if err != nil {
				t.Fatalf("workflowagent.New: %v", err)
			}
			for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			if got := nestedRuns.Load(); got != 1 {
				t.Fatalf("nested runs of child = %d, want 1", got)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			var starts int
			for _, row := range rowsOfType(rows, EventAgentStarting) {
				if row["agent"] == "child" {
					starts++
				}
			}
			if starts != 2 {
				t.Errorf("got %d AGENT_STARTING rows for child, want 2; event types: %v", starts, eventTypes(rows))
			}
			requireInvocationCompletedOK(t, rows)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
			}
		})
	}
}

// TestConcurrencyRace_FunctionNodeResumesChildAfterNestedRun has a function
// node forward the event of an agent run, run the agent again inside its loop
// over the first run's events, and then resume the first run, which ends its
// own InvocationContext and waits until the agent of another workflow node
// has ended. That agent starts answering only after the nested run, so its
// activation ends, and the plugin settles what has ended, while the first
// run still goes on. The first run answered, so nothing fails, and its
// InvocationContext must not be read before its run returns.
func TestConcurrencyRace_FunctionNodeResumesChildAfterNestedRun(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	nestedDone := make(chan struct{})
	otherEnded := make(chan struct{})
	child := newOneEventAgent(t, "child", func(ctx agent.InvocationContext, run int32) {
		if run != 1 {
			return
		}
		ctx.EndInvocation()
		select {
		case <-otherEnded:
		case <-time.After(10 * time.Second):
		}
	})
	fn := workflow.NewEmittingFunctionNode[any, any]("fn", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		nested := false
		for ev, err := range child.Run(ctx) {
			if err != nil {
				return nil, err
			}
			if err := emit(ev); err != nil {
				return nil, err
			}
			if nested {
				continue
			}
			nested = true
			for _, err := range child.Run(ctx) {
				if err != nil {
					return nil, err
				}
			}
			close(nestedDone)
		}
		return nil, nil
	}, workflow.NodeConfig{})
	var endOther sync.Once
	other, err := agent.New(agent.Config{
		Name: "other",
		// The agent's own AfterAgentCallbacks run after those of the plugins.
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				endOther.Do(func() { close(otherEnded) })
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				select {
				case <-nestedDone:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("the nested run of child never ended"))
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "other"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "other answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(other): %v", err)
	}
	otherNode, err := workflow.NewAgentNode(other, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(other): %v", err)
	}
	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: fn},
			{From: workflow.Start, To: otherNode},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestRunner_FunctionNodeRerunsChildWithoutRetryConfig_KeepsFailure has a
// function node with no RetryConfig recover from the error of a child agent
// by running the agent again: on its own, under the name of the agent, and
// as the node of a workflow.ParallelWorker item. ADK did not handle the
// failure, so the plugin must not purge it as an earlier attempt of the
// node: the child's AGENT_ERROR row stays and the invocation ends with
// INVOCATION_ERROR, as it does when custom recovery handles a failure.
func TestRunner_FunctionNodeRerunsChildWithoutRetryConfig_KeepsFailure(t *testing.T) {
	for _, tc := range []struct {
		name, node  string
		underWorker bool
	}{
		{name: "function_node", node: "fn"},
		{name: "function_node_named_like_child", node: "child"},
		{name: "parallel_worker_function_node", node: "fn", underWorker: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			child, err := llmagent.New(llmagent.Config{
				Name:  "child",
				Model: &flakyOnFirstPassLLM{err: errors.New("child model failure 503"), reply: "child answer"},
				Mode:  llmagent.ModeSingleTurn,
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			var recovered atomic.Bool
			fn := workflow.NewFunctionNode(tc.node, func(ctx agent.Context, _ any) (string, error) {
				for _, err := range child.Run(ctx) {
					if err != nil {
						break
					}
				}
				for _, err := range child.Run(ctx) {
					if err != nil {
						return "", err
					}
				}
				recovered.Store(true)
				return "the function node handled the child error", nil
			}, workflow.NodeConfig{})
			root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: functionNodeEdges(t, fn, tc.underWorker)})
			if err != nil {
				t.Fatalf("workflowagent.New: %v", err)
			}
			for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			if !recovered.Load() {
				t.Fatal("the function node did not recover from the child error")
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Errorf("got %d INVOCATION_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
			}
			onlyRowOfType(t, rows, EventInvocationError)
			if got := agentErrorCounts(rows)["child"]; got != 1 {
				t.Errorf("AGENT_ERROR rows for child = %d, want 1; counts: %v", got, agentErrorCounts(rows))
			}
		})
	}
}

// TestConcurrencyRace_ReportedNodeOutcomeDoesNotEndRunningRun has two join
// nodes route to the same agent node, as in
// TestConcurrencyRace_RoutedNodeStartedAgainWhileRunning, and the first run
// of the node's agent writes the metadata of the completion event of a failed
// node run (see nodeCompletion) on its own answer before the second run
// starts. It then ends its InvocationContext and waits until the second run
// has answered. The report makes the first activation an earlier attempt of
// the node, but does not show that its run is over, so the second activation
// must not read its InvocationContext.
func TestConcurrencyRace_ReportedNodeOutcomeDoesNotEndRunningRun(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	reported := make(chan struct{})
	secondAnswered := make(chan struct{})
	var calls, runs atomic.Int32
	child, err := agent.New(agent.Config{
		Name: "child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				calls.Add(1)
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				n := runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "answer")
				if n == 1 {
					path := contextPath(ctx)
					ev.NodeInfo = &session.NodeInfo{Path: path}
					ev.CustomMetadata = map[string]any{
						workflowBasePathPrefix + "root": path[:max(strings.LastIndexByte(path, '/'), 0)],
						workflowNodeOutcomeKey:          workflowNodeFailedOutcome,
					}
				}
				if !yield(ev, nil) {
					return
				}
				switch n {
				case 1:
					close(reported)
					ctx.EndInvocation()
					select {
					case <-secondAnswered:
					case <-time.After(10 * time.Second):
					}
				case 2:
					close(secondAnswered)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	x, err := workflow.NewAgentNode(child, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	quick := func(name string) workflow.Node {
		return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) { return name, nil }, workflow.NodeConfig{})
	}
	a, b := quick("a"), quick("b")
	c := workflow.NewFunctionNode("c", func(agent.Context, any) (string, error) {
		select {
		case <-reported:
			return "c", nil
		case <-time.After(10 * time.Second):
			return "", errors.New("the first run of child never answered")
		}
	}, workflow.NodeConfig{})
	j1, j2 := workflow.NewJoinNode("j1"), workflow.NewJoinNode("j2")
	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: a},
			{From: workflow.Start, To: b},
			{From: workflow.Start, To: c},
			{From: a, To: j1},
			{From: b, To: j1},
			{From: a, To: j2},
			{From: c, To: j2},
			{From: j1, To: x, Route: workflow.Default},
			{From: j2, To: x, Route: workflow.Default},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Logf("runner.Run error: %v", err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("child started %d times, want 2", got)
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	t.Logf("event types: %v", eventTypes(rows))
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestConcurrencyRace_LiveToolReturnsBeforeUnjoinedCachedChild verifies under
// -race that a live tool that starts a sub-agent on a goroutine of its own and
// returns once the sub-agent's BeforeAgentCallback runs, without waiting for
// the run, does not have the sub-agent's ended state read when the tool call
// ends: the sub-agent did not run on the goroutine of the tool call, so
// nothing shows that its run is over. It is retired without an AGENT_ERROR row
// while agent.Run ends its InvocationContext on the sub-agent's goroutine,
// which then waits until the root agent's AfterToolCallback, which runs after
// this plugin's, so nothing that it does after EndInvocation orders that
// write before the end of the tool call.
func TestConcurrencyRace_LiveToolReturnsBeforeUnjoinedCachedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	toolDone := make(chan struct{})
	helperDone := make(chan struct{})
	helper, err := agent.New(agent.Config{
		Name: "helper",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(entered)
				return genai.NewContentFromText("cached", genai.RoleModel), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(helper): %v", err)
	}
	startHelper, err := functiontool.New(functiontool.Config{Name: "start_helper", Description: "Starts helper without waiting for it"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			subPath := "helper@1"
			runCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath}).WithICDelta(nil)
			go func() {
				defer close(helperDone)
				for range helper.Run(runCtx) {
					select {
					case <-toolDone:
					case <-time.After(10 * time.Second):
					}
				}
			}()
			select {
			case <-entered:
			case <-time.After(10 * time.Second):
				return nil, errors.New("helper never entered its BeforeAgentCallback")
			}
			return map[string]any{"started": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(start_helper): %v", err)
	}

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL: newFakeLiveServer(t,
				`{"toolCall":{"functionCalls":[{"id":"call-1","name":"start_helper","args":{}}]}}`,
				`{"serverContent":{"modelTurn":{"parts":[{"text":"done"}],"role":"model"},"turnComplete":true}}`,
			),
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "live_root",
		Model: &liveTestLLM{client: client},
		Tools: []tool.Tool{startHelper},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				close(toolDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(live_root): %v", err)
	}

	liveSess, events, err := newTestRunner(t, root, p).RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("r.RunLive: %v", err)
	}
	go func() {
		select {
		case <-toolDone:
		case <-time.After(10 * time.Second):
		}
		_ = liveSess.Close()
	}()
	for _, err := range events {
		if err != nil {
			t.Errorf("live event error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	select {
	case <-helperDone:
	case <-time.After(10 * time.Second):
		t.Fatal("helper run never finished")
	}
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestConcurrencyRace_FunctionToolReturnsBeforeUnjoinedCachedChild verifies
// under -race the ordering of
// TestConcurrencyRace_LiveToolReturnsBeforeUnjoinedCachedChild for an ordinary
// function tool that runner.Run calls: the tool starts a sub-agent on a
// goroutine of its own and returns once the sub-agent's BeforeAgentCallback
// runs, without waiting for the run, so the end of the tool call does not read
// the sub-agent's ended state. The sub-agent did not run on the goroutine of
// the tool call, so nothing shows that its run is over, and it is retired
// without an AGENT_ERROR row while agent.Run ends its InvocationContext on the
// sub-agent's goroutine. That goroutine then waits until the root agent's
// AfterToolCallback, which runs after this plugin's, so nothing that it does
// after EndInvocation orders that write before the end of the tool call.
func TestConcurrencyRace_FunctionToolReturnsBeforeUnjoinedCachedChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	entered := make(chan struct{})
	toolDone := make(chan struct{})
	childDone := make(chan struct{})
	child, err := agent.New(agent.Config{
		Name: "cached_child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				close(entered)
				return textContent("model", "cached"), nil
			},
		},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(cached_child): %v", err)
	}
	launch, err := functiontool.New(functiontool.Config{Name: "launch", Description: "Starts cached_child without waiting for it"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			path := "cached_child@1"
			if parentPath := ctx.Path(); parentPath != "" {
				path = parentPath + "/cached_child@1"
			}
			childCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path}).WithICDelta(nil)
			go func() {
				defer close(childDone)
				for range child.Run(childCtx) {
					select {
					case <-toolDone:
					case <-time.After(10 * time.Second):
					}
				}
			}()
			select {
			case <-entered:
			case <-time.After(10 * time.Second):
				return nil, errors.New("cached_child never entered its BeforeAgentCallback")
			}
			return map[string]any{"started": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(launch): %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Tools: []tool.Tool{launch},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "launch", Args: map[string]any{}}},
			}}},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				close(toolDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	select {
	case <-childDone:
	case <-time.After(10 * time.Second):
		t.Fatal("cached_child run never finished")
	}
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
	}
}

// TestRunner_ToolRunCustomParentWithUnjoinedCachedChild_EmitsNoAgentError
// verifies that when a function tool runs a custom agent that starts a
// sub-agent on a goroutine of its own and ends its own InvocationContext
// without waiting for the sub-agent, the end of the tool call reports neither
// the sub-agent, which its BeforeAgentCallback answers, nor the custom agent
// as failed: nothing shows that the sub-agent's run is over, so it is not
// known to have failed by not responding (see failuresLocked), and its ended
// state is not read while agent.Run may still end its InvocationContext. The
// sub-agent is a declared sub-agent of the custom agent, or not declared.
func TestRunner_ToolRunCustomParentWithUnjoinedCachedChild_EmitsNoAgentError(t *testing.T) {
	for _, declared := range []bool{true, false} {
		name := "undeclared_child"
		if declared {
			name = "declared_child"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			entered := make(chan struct{})
			childDone := make(chan struct{})
			cachedChild, err := agent.New(agent.Config{
				Name: "cached_child",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						close(entered)
						return textContent("model", "cache hit"), nil
					},
				},
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(func(*session.Event, error) bool) {}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(cached_child): %v", err)
			}
			var subAgents []agent.Agent
			if declared {
				subAgents = []agent.Agent{cachedChild}
			}
			parent, err := agent.New(agent.Config{
				Name:      "parent",
				SubAgents: subAgents,
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						go func() {
							defer close(childDone)
							for range cachedChild.Run(ctx) {
							}
						}()
						select {
						case <-entered:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("cached_child never entered its BeforeAgentCallback"))
							return
						}
						answer := session.NewEvent(ctx, ctx.InvocationID())
						answer.Author = "parent"
						answer.Branch = ctx.Branch()
						answer.Content = textContent("model", "parent answer")
						if !yield(answer, nil) {
							return
						}
						ctx.EndInvocation()
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(parent): %v", err)
			}
			runParent, err := functiontool.New(functiontool.Config{Name: "run_parent", Description: "Runs parent"},
				func(ctx agent.Context, _ struct{}) (map[string]any, error) {
					subPath := "parent@1"
					for _, err := range parent.Run(agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath})) {
						if err != nil {
							return nil, err
						}
					}
					return map[string]any{"ok": true}, nil
				})
			if err != nil {
				t.Fatalf("functiontool.New(run_parent): %v", err)
			}
			coordinator, err := llmagent.New(llmagent.Config{
				Name:  "coordinator",
				Tools: []tool.Tool{runParent},
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: "call-parent-1", Name: "run_parent", Args: map[string]any{}}},
					}}},
					{Content: textContent("model", "done"), TurnComplete: true},
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(coordinator): %v", err)
			}

			for _, err := range newTestRunner(t, coordinator, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error: %v", err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			select {
			case <-childDone:
			case <-time.After(10 * time.Second):
				t.Fatal("cached_child run never finished")
			}
			requireInvocationCompletedOK(t, rows)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows, want 0; counts: %v", n, agentErrorCounts(rows))
			}
		})
	}
}

// TestTraceManager_StaleRetiredPeerTrailerDoesNotClearLaterActivationError
// verifies that the trailer event of a concurrent peer that already retired,
// which arrives after every peer of its (path, name, branch) retired and a
// later activation of the same agent recorded a model error, is absorbed onto
// the activation that the agent runs under (retiredPeerCount in
// selectResponseOwnerLocked) instead of being taken for the later
// activation's response. The later activation overlapped no peer, so it is no
// concurrentPeer, and completeResponse would clear its error if the trailer
// reached it.
func TestTraceManager_StaleRetiredPeerTrailerDoesNotClearLaterActivationError(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-stale-trailer"
	parent := identity{invocationID: invID, agentName: "parent"}
	worker := identity{invocationID: invID, agentName: "worker"}
	tm := newTraceManager()
	tm.invocationSpan(ctx, parent, nil)
	tm.pushAgentSpan(ctx, parent)

	// Two overlapping runs of worker respond and retire.
	tm.pushAgentSpan(ctx, worker)
	tm.pushAgentSpan(ctx, worker)
	tm.completeResponse(ctx, parent, "worker", "", nil)
	tm.completeResponse(ctx, parent, "worker", "", nil)
	tm.popAgentSpan(ctx, worker)
	tm.popAgentSpan(ctx, worker)

	// A later run of worker, which overlaps no peer, fails its model call.
	modelErr := errors.New("later worker model failed")
	tm.pushAgentSpan(ctx, worker)
	tm.pushLLMSpan(ctx, worker)
	tm.recordError(ctx, worker, llmKey(worker), modelErr)
	tm.popLLMSpan(ctx, worker, false)

	// The trailer of a retired run arrives late.
	tm.completeResponse(ctx, parent, "worker", "", nil)

	if _, err := tm.popUnclosedAgentSpans(ctx, parent); !errors.Is(err, modelErr) {
		t.Errorf("popUnclosedAgentSpans error = %v, want %v: the late trailer must not clear the error of the later activation", err, modelErr)
	}
}

// TestRunner_ParallelToolRunsOfSilentlyFailingAgents_EmitAgentErrors verifies
// that when one model response calls two function tools that each run a
// custom agent that fails before any event, which ADK runs concurrently on
// goroutines of their own, each agent ends with an AGENT_ERROR row of its own
// once the last call ends, and the invocation completes. Each run returned on
// the goroutine of the tool call that ran it before that call returned (see
// markToolRunsDoneLocked), so the end of the last call, on another goroutine,
// still knows that the other agent never responded.
func TestRunner_ParallelToolRunsOfSilentlyFailingAgents_EmitAgentErrors(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	newRunTool := func(name string) tool.Tool {
		t.Helper()
		failing, err := agent.New(agent.Config{
			Name: name,
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("%s failed", name))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%s): %v", name, err)
		}
		runTool, err := functiontool.New(functiontool.Config{Name: "run_" + name, Description: "Runs " + name},
			func(ctx agent.Context, _ struct{}) (map[string]any, error) {
				subPath := name + "@1"
				for _, err := range failing.Run(agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath})) {
					if err != nil {
						return nil, err
					}
				}
				return map[string]any{"ok": true}, nil
			})
		if err != nil {
			t.Fatalf("functiontool.New(run_%s): %v", name, err)
		}
		return runTool
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:  "coordinator",
		Tools: []tool.Tool{newRunTool("failing_a"), newRunTool("failing_b")},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call-a", Name: "run_failing_a", Args: map[string]any{}}},
				{FunctionCall: &genai.FunctionCall{ID: "call-b", Name: "run_failing_b", Args: map[string]any{}}},
			}}},
			{Content: textContent("model", "both helpers failed"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coordinator): %v", err)
	}

	for _, err := range newTestRunner(t, coordinator, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationCompleted)
	if got, want := agentErrorCounts(rows), map[string]int{"failing_a": 1, "failing_b": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR rows per agent = %v, want %v", got, want)
	}
}

// TestRunner_RetriedParallelNodeWithSilentBranchFailure_EmitsBranchAgentError
// verifies that when RetryConfig runs a workflow node again whose parallel
// agent failed because its branch, a custom agent, failed before any event,
// the branch gets an AGENT_ERROR row of its own along with the parallel agent
// when the later attempt starts, and the invocation completes. The branch ran
// on a goroutine that the parallel agent started, and the parallel agent
// waits for its branches before its run returns, so the branch's run is known
// to be over once the workflow reported the end of the node run of the
// parallel agent (see recordNodeCompletion and spanRecord.joins), which ADK
// releases without node completion events do not.
func TestRunner_RetriedParallelNodeWithSilentBranchFailure_EmitsBranchAgentError(t *testing.T) {
	requireADKNodeCompletionEvents(t)
	p, requests, _ := setupTestPlugin(t)
	var attempts atomic.Int32
	silent, err := agent.New(agent.Config{
		Name: "silent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if attempts.Add(1) == 1 {
					yield(nil, errors.New("silent failed before any event"))
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "silent"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "silent answered on retry")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(silent): %v", err)
	}
	node, err := workflow.NewAgentNode(newParallelAgent(t, "par", silent), workflow.NodeConfig{
		RetryConfig: &workflow.RetryConfig{
			MaxAttempts:   2,
			InitialDelay:  time.Millisecond,
			MaxDelay:      time.Millisecond,
			BackoffFactor: 1.0,
		},
	})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(par): %v", err)
	}
	wfRoot, err := workflowagent.New(workflowagent.Config{
		Name:  "wf_root",
		Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(wf_root): %v", err)
	}

	for _, err := range newTestRunner(t, wfRoot, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got INVOCATION_ERROR, want INVOCATION_COMPLETED; event types: %v", eventTypes(rows))
	}
	onlyRowOfType(t, rows, EventInvocationCompleted)
	if got, want := agentErrorCounts(rows), map[string]int{"silent": 1, "par": 1}; !maps.Equal(got, want) {
		t.Errorf("AGENT_ERROR rows per agent = %v, want %v; event types: %v", got, want, eventTypes(rows))
	}
}

// gatedFlakyLLM fails its first request with err once ready is closed, after
// calling enter, and answers later requests with reply.
type gatedFlakyLLM struct {
	calls atomic.Int32
	enter func()
	ready <-chan struct{}
	err   error
	reply string
}

func (m *gatedFlakyLLM) Name() string { return "gated-flaky-model" }

func (m *gatedFlakyLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) == 1 {
			m.enter()
			select {
			case <-m.ready:
			case <-ctx.Done():
				return
			}
			yield(nil, m.err)
			return
		}
		yield(&model.LLMResponse{Content: textContent("model", m.reply), TurnComplete: true}, nil)
	}
}

// agentRowsOfType returns the rows of rows of eventType that agent wrote.
func agentRowsOfType(rows []map[string]any, eventType, agent string) []map[string]any {
	var out []map[string]any
	for _, row := range rowsOfType(rows, eventType) {
		if name, _ := row["agent"].(string); name == agent {
			out = append(out, row)
		}
	}
	return out
}

// TestRunner_ToolReturnsBeforeHelperModelCallEnds_KeepsHelperOpen verifies
// through runner.Run that a function tool that starts a helper on a goroutine
// of its own and returns while the helper's model call is still in flight
// does not end the helper: the tool's return does not show that the helper's
// run is over, so neither its open call nor its own completion is treated as
// a failure. Here the root agent's AfterToolCallback, which ADK runs after
// this plugin's, releases the helper's model and waits for the helper to
// return, so the helper ends before the run does: its LLM_RESPONSE and
// AGENT_COMPLETED rows close the spans of its LLM_REQUEST and AGENT_STARTING
// rows with their latencies, and no AGENT_ERROR row is written.
func TestRunner_ToolReturnsBeforeHelperModelCallEnds_KeepsHelperOpen(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	gate := &blockingLLM{entered: make(chan struct{}, 1), release: make(chan struct{})}
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Model: gate})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	helperDone := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(gate.release) }) }
	t.Cleanup(release)
	launch, err := functiontool.New(functiontool.Config{Name: "launch", Description: "Starts helper without waiting for it"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			path := "helper@1"
			if parentPath := ctx.Path(); parentPath != "" {
				path = parentPath + "/helper@1"
			}
			helperCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path}).WithICDelta(nil)
			go func() {
				defer close(helperDone)
				for range helper.Run(helperCtx) {
				}
			}()
			select {
			case <-gate.entered:
			case <-time.After(10 * time.Second):
				return nil, errors.New("helper never started its model call")
			}
			return map[string]any{"started": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(launch): %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Tools: []tool.Tool{launch},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "launch", Args: map[string]any{}}},
			}}},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				release()
				select {
				case <-helperDone:
				case <-time.After(10 * time.Second):
					return nil, errors.New("helper never returned")
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
	}
	for _, pair := range [][2]string{{EventLLMRequest, EventLLMResponse}, {EventAgentStarting, EventAgentCompleted}} {
		starts, ends := agentRowsOfType(rows, pair[0], "helper"), agentRowsOfType(rows, pair[1], "helper")
		if len(starts) != 1 || len(ends) != 1 {
			t.Errorf("helper rows: %d %s and %d %s, want one of each; event types: %v", len(starts), pair[0], len(ends), pair[1], eventTypes(rows))
			continue
		}
		if got, want := placeOf(ends[0]), placeOf(starts[0]); got != want {
			t.Errorf("helper %s is at %+v, want the place of its %s %+v", pair[1], got, pair[0], want)
		}
		mustLatency(t, ends[0])
	}
}

// TestConcurrencyRace_ForwardedFinishTaskEventDoesNotEndRunningChild verifies
// under -race that a successful finish_task function response that a custom
// agent forwards for a sub-agent that it runs on a goroutine of its own does
// not count as the end of the sub-agent's run: the sub-agent ends its
// InvocationContext after it wrote the event, and the custom agent waits for
// it only later, so OnEventCallback must not read the sub-agent's ended state.
// The sub-agent goes on until the custom agent has forwarded the event, so
// that nothing that its goroutine does after it returns can order its write
// before OnEventCallback runs.
func TestConcurrencyRace_ForwardedFinishTaskEventDoesNotEndRunningChild(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	forwarded := make(chan struct{})
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID:       "call-finish",
					Name:     "finish_task",
					Response: map[string]any{"result": "Task completed."},
				}}}}
				if !yield(ev, nil) {
					return
				}
				ctx.EndInvocation()
				select {
				case <-forwarded:
				case <-time.After(10 * time.Second):
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
				type result struct {
					ev  *session.Event
					err error
				}
				results := make(chan result)
				go func() {
					defer close(results)
					for ev, err := range child.Run(ctx) {
						results <- result{ev: ev, err: err}
					}
				}()
				reading := true
				for res := range results {
					if reading {
						reading = yield(res.ev, res.err)
					}
					select {
					case <-forwarded:
					default:
						close(forwarded)
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}
	for _, err := range newTestRunner(t, parent, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows, want 0; event types: %v", n, eventTypes(rows))
	}
	// The sub-agent does not run in task mode, so its finish_task response
	// does not end its span (see TestRunner_NonTaskModeFinishTaskTool), and ADK
	// runs no AfterAgentCallback for an agent that ended its invocation: its
	// span retires with the custom agent's, without a closing row.
	starts, ends := agentRowsOfType(rows, EventAgentStarting, "child"), agentRowsOfType(rows, EventAgentCompleted, "child")
	if len(starts) != 1 || len(ends) != 0 {
		t.Errorf("child rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want 1 and 0; event types: %v", len(starts), len(ends), eventTypes(rows))
	}
}

// finishTaskLLM answers with a finish_task call, as a task-mode agent's model
// does once its task is done, and closes called on its first request.
type finishTaskLLM struct {
	called chan struct{}
	once   sync.Once
}

func (m *finishTaskLLM) Name() string { return "finish-task-model" }

func (m *finishTaskLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.once.Do(func() { close(m.called) })
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID:   "call-finish-1",
			Name: "finish_task",
			Args: map[string]any{"result": "task finished"},
		}}}}}, nil)
	}
}

// TestConcurrencyRace_FinishTaskEndsTheRunThatMadeTheCall verifies under -race
// that, with two concurrent runs of one task-mode agent on the same branch and
// node path, the finish_task response of the first run ends the first run's
// span, whose finish_task call this plugin saw complete, and does not read the
// InvocationContext of the second run, whose BeforeAgentCallback is answering
// with cached content, and so ending its InvocationContext, at that time.
func TestConcurrencyRace_FinishTaskEndsTheRunThatMadeTheCall(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	modelCalled := make(chan struct{})
	finishCalled := make(chan struct{})
	var finishOnce sync.Once
	var starts atomic.Int32
	task, err := llmagent.New(llmagent.Config{
		Name:  "task",
		Mode:  llmagent.ModeTask,
		Model: &finishTaskLLM{called: modelCalled},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if starts.Add(1) != 2 {
					return nil, nil
				}
				select {
				case <-finishCalled:
				case <-time.After(10 * time.Second):
				}
				return textContent("model", "cached"), nil
			},
		},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(_ agent.Context, t tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
				if t.Name() == "finish_task" {
					finishOnce.Do(func() { close(finishCalled) })
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(task): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name: "root",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				type result struct {
					ev  *session.Event
					err error
				}
				nodeCtx := agent.Promote(ctx)
				results := make(chan result)
				var wg sync.WaitGroup
				runTask := func() {
					defer wg.Done()
					for ev, err := range llmagent.RunLLMAgentAsNode(task, nodeCtx, "do the task") {
						results <- result{ev: ev, err: err}
					}
				}
				wg.Add(2)
				go runTask()
				go func() {
					select {
					case <-modelCalled:
					case <-time.After(10 * time.Second):
					}
					runTask()
				}()
				go func() {
					wg.Wait()
					close(results)
				}()
				reading := true
				for res := range results {
					if reading {
						reading = yield(res.ev, res.err)
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Logf("runner.Run error: %v", err)
		}
	}
	if got := starts.Load(); got != 2 {
		t.Fatalf("task started %d times, want 2", got)
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requests1 := agentRowsOfType(rows, EventLLMRequest, "task")
	if len(requests1) != 1 {
		t.Fatalf("task LLM_REQUEST rows = %d, want 1 (only the first run calls its model); event types: %v", len(requests1), eventTypes(rows))
	}
	first := placeOf(requests1[0]).parent
	var endedFirst bool
	for _, row := range agentRowsOfType(rows, EventAgentCompleted, "task") {
		if placeOf(row).span == first {
			endedFirst = true
		}
	}
	if !endedFirst {
		t.Errorf("no AGENT_COMPLETED row ends span %q of the run that made the finish_task call; event types: %v", first, eventTypes(rows))
	}
}

// TestRunner_ReportedNodeOutcomeBeforeDelayedResponse_EmitsNoAgentError has
// two join nodes route to the same agent node, as in
// TestConcurrencyRace_ReportedNodeOutcomeDoesNotEndRunningRun, and the first
// run of the node's agent writes the metadata of the completion event of a
// failed node run on an event without content, so it has not responded yet,
// and answers only after the second run started. The report shows neither
// that the first run is over nor that it failed, so no AGENT_ERROR row is
// written for either run, which both return without an error.
//
// The workflow keeps one output accumulator per node name, so the second run
// returns without content once the first run's answer has been handled: an
// answer from each run would fail the workflow with ErrMultipleOutputs, and a
// second run that ended first would let the workflow return before the first
// run's answer was handled.
func TestRunner_ReportedNodeOutcomeBeforeDelayedResponse_EmitsNoAgentError(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	reported := make(chan struct{})
	secondStarted := make(chan struct{})
	firstAnswered := make(chan struct{})
	var calls, runs atomic.Int32
	child, err := agent.New(agent.Config{
		Name: "child",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(agent.Context) (*genai.Content, error) {
				if calls.Add(1) == 2 {
					close(secondStarted)
				}
				return nil, nil
			},
		},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if runs.Add(1) != 1 {
					select {
					case <-firstAnswered:
					case <-time.After(10 * time.Second):
					}
					return
				}
				defer close(firstAnswered)
				path := contextPath(ctx)
				report := session.NewEvent(ctx, ctx.InvocationID())
				report.Author = "child"
				report.Branch = ctx.Branch()
				report.NodeInfo = &session.NodeInfo{Path: path}
				report.CustomMetadata = map[string]any{
					workflowBasePathPrefix + "root": path[:max(strings.LastIndexByte(path, '/'), 0)],
					workflowNodeOutcomeKey:          workflowNodeFailedOutcome,
				}
				if !yield(report, nil) {
					return
				}
				close(reported)
				select {
				case <-secondStarted:
				case <-time.After(10 * time.Second):
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	x, err := workflow.NewAgentNode(child, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	quick := func(name string) workflow.Node {
		return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) { return name, nil }, workflow.NodeConfig{})
	}
	a, b := quick("a"), quick("b")
	c := workflow.NewFunctionNode("c", func(agent.Context, any) (string, error) {
		select {
		case <-reported:
			return "c", nil
		case <-time.After(10 * time.Second):
			return "", errors.New("the first run of child never reported")
		}
	}, workflow.NodeConfig{})
	j1, j2 := workflow.NewJoinNode("j1"), workflow.NewJoinNode("j2")
	root, err := workflowagent.New(workflowagent.Config{
		Name: "root",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: a},
			{From: workflow.Start, To: b},
			{From: workflow.Start, To: c},
			{From: a, To: j1},
			{From: b, To: j1},
			{From: a, To: j2},
			{From: c, To: j2},
			{From: j1, To: x, Route: workflow.Default},
			{From: j2, To: x, Route: workflow.Default},
		},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("child started %d times, want 2", got)
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
		t.Errorf("got %d AGENT_ERROR rows for two eventually successful runs, want 0; counts: %v; event types: %v", n, agentErrorCounts(rows), eventTypes(rows))
	}
}

// TestTraceManager_ToolSpanAcrossAfterRun verifies that tool and model calls
// started before AfterRun preserve their open/completed spans on
// completed.late across endInvocation, so completion or error callbacks firing
// after AfterRun reuse the original span_id, parent_span_id, and latency_ms.
func TestTraceManager_ToolSpanAcrossAfterRun(t *testing.T) {
	t.Run("direct pushToolSpan across endInvocation", func(t *testing.T) {
		ctx := context.Background()
		tm := newTraceManager()
		now := time.Unix(2000, 0)
		tm.now = func() time.Time { return now }

		id := identity{invocationID: "inv-live-tool", agentName: "live_agent"}
		invStart := tm.invocationSpan(ctx, id, nil)
		agentStart, _ := tm.pushAgentSpan(ctx, id)
		key := toolKey(id, "call-1", "search")
		toolStart := tm.pushToolSpan(ctx, id, key)

		now = now.Add(25 * time.Millisecond)
		tm.popAgentSpan(ctx, id)
		invEnd := tm.endInvocation(ctx, id)
		if invEnd.spanID != invStart.spanID {
			t.Fatalf("endInvocation spanID = %q, want %q", invEnd.spanID, invStart.spanID)
		}

		now = now.Add(40 * time.Millisecond)
		toolEnd, _, closedByError := tm.popToolSpan(ctx, id, key, false)
		if closedByError {
			t.Errorf("popToolSpan closedByError = true, want false")
		}
		if toolEnd.traceID != toolStart.traceID {
			t.Errorf("popToolSpan traceID = %q, want %q", toolEnd.traceID, toolStart.traceID)
		}
		if toolEnd.spanID != toolStart.spanID {
			t.Errorf("popToolSpan spanID = %q, want original TOOL_STARTING spanID %q", toolEnd.spanID, toolStart.spanID)
		}
		if toolEnd.parentSpanID != agentStart.spanID {
			t.Errorf("popToolSpan parentSpanID = %q, want agent spanID %q", toolEnd.parentSpanID, agentStart.spanID)
		}
		if got, _ := toolEnd.latency["total_ms"].(int64); got != 65 {
			t.Errorf("popToolSpan total_ms = %v, want 65 (latency=%v)", toolEnd.latency["total_ms"], toolEnd.latency)
		}
	})

	t.Run("model error before endInvocation and recovered response after", func(t *testing.T) {
		ctx := context.Background()
		tm := newTraceManager()
		id := identity{invocationID: "inv-live-model", agentName: "live_agent"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		start := tm.pushLLMSpan(ctx, id)
		// OnModelErrorCallback ends the call, and a plugin recovers from the
		// error, so ADK runs AfterModelCallback for it after the run ended.
		tm.popLLMSpan(ctx, id, false)
		tm.popAgentSpan(ctx, id)
		tm.endInvocation(ctx, id)

		end := tm.popLLMSpan(ctx, id, true)
		if end.spanID != start.spanID || end.latency != nil {
			t.Errorf("LLM_RESPONSE place = %+v, want span %q of LLM_REQUEST without latency", end, start.spanID)
		}
	})

	t.Run("made-up span of an unseen call stays behind", func(t *testing.T) {
		ctx := context.Background()
		tm := newTraceManager()
		id := identity{invocationID: "inv-live-unseen", agentName: "live_agent"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		// ADK runs OnToolErrorCallback alone for a call of a tool that does
		// not exist, so the call has no completion callback to wait for.
		tm.popToolSpan(ctx, id, toolKey(id, "call-missing", "missing"), true)
		tm.popAgentSpan(ctx, id)
		tm.endInvocation(ctx, id)

		tm.mu.Lock()
		entry := tm.completed[id.invocationID]
		var late *invocationTrace
		if entry != nil {
			late = entry.late
		}
		tm.mu.Unlock()
		if late != nil {
			t.Error("endInvocation kept late state for a run whose only call this plugin did not see start")
		}
		if entry != nil {
			if len(entry.endedTools) != 0 || len(entry.endedModels) != 0 {
				t.Errorf("endedTools = %d, endedModels = %d, want 0 and 0: a call this plugin did not see start must not wait in the lightweight maps", len(entry.endedTools), len(entry.endedModels))
			}
		}
	})

	t.Run("plugin AfterToolCallback after AfterRunCallback", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		toolCtx := withFunctionCall(mCtx, "call-live-ok")
		lookup := &fakeTool{name: "lookup"}

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, lookup, map[string]any{"q": "go"}))
		ev := session.NewEvent(t.Context(), mCtx.invocationID)
		ev.Author = mCtx.agentName
		ev.Content = textContent("model", "streaming answer")
		ev.Partial = true
		errs.add(p.OnEventCallback()(mCtx, ev))
		errs.check(t)

		// Consumer stops reading in RunLive: AfterRun runs while tool is in flight.
		p.AfterRunCallback()(mCtx)

		errs.add(p.AfterToolCallback()(toolCtx, lookup, map[string]any{"q": "go"}, map[string]any{"ok": true}, nil))
		errs.check(t)

		rows := collectRows(t, requests, untilEventType(EventToolCompleted))
		agentStart := placeOf(onlyRowOfType(t, rows, EventAgentStarting))
		toolStart := placeOf(onlyRowOfType(t, rows, EventToolStarting))
		toolCompRow := onlyRowOfType(t, rows, EventToolCompleted)
		toolComp := placeOf(toolCompRow)

		if toolComp != toolStart {
			t.Errorf("TOOL_COMPLETED place = %+v, want TOOL_STARTING place %+v", toolComp, toolStart)
		}
		if toolComp.parent != agentStart.span {
			t.Errorf("TOOL_COMPLETED parent_span_id = %q, want AGENT_STARTING span_id %q", toolComp.parent, agentStart.span)
		}
		mustLatency(t, toolCompRow)
	})

	t.Run("plugin OnToolErrorCallback and AfterToolCallback after AfterRunCallback", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		toolCtx := withFunctionCall(mCtx, "call-live-err")
		lookup := &fakeTool{name: "lookup"}
		boom := errors.New("live tool failed")

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, lookup, nil))
		ev := session.NewEvent(t.Context(), mCtx.invocationID)
		ev.Author = mCtx.agentName
		ev.Content = textContent("model", "partial")
		ev.Partial = true
		errs.add(p.OnEventCallback()(mCtx, ev))
		errs.check(t)

		p.AfterRunCallback()(mCtx)

		errs.add(p.OnToolErrorCallback()(toolCtx, lookup, nil, boom))
		errs.add(p.AfterToolCallback()(toolCtx, lookup, nil, nil, boom))
		errs.check(t)

		rows := collectRows(t, requests, untilEventType(EventToolError))
		if n := len(rowsOfType(rows, EventToolError)); n != 1 {
			t.Fatalf("got %d TOOL_ERROR rows, want 1", n)
		}
		toolStart := placeOf(onlyRowOfType(t, rows, EventToolStarting))
		toolErrRow := onlyRowOfType(t, rows, EventToolError)
		if got := placeOf(toolErrRow); got != toolStart {
			t.Errorf("TOOL_ERROR place = %+v, want TOOL_STARTING place %+v", got, toolStart)
		}
		mustLatency(t, toolErrRow)
	})

	t.Run("plugin AfterModelCallback after AfterRunCallback", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-live"}))
		errs.check(t)

		// The call has streamed nothing, so it moves to the late state: the
		// model call of an agent that a tool of RunLive runs can end after
		// AfterRun. AfterRun reports the open call; only the late row matters
		// here.
		p.AfterRunCallback()(mCtx)

		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "final")}, nil))
		errs.check(t)

		rows := collectRows(t, requests, untilEventType(EventLLMResponse))
		llmReq := placeOf(onlyRowOfType(t, rows, EventLLMRequest))
		llmRespRow := onlyRowOfType(t, rows, EventLLMResponse)
		if got := placeOf(llmRespRow); got != llmReq {
			t.Errorf("LLM_RESPONSE place = %+v, want LLM_REQUEST place %+v", got, llmReq)
		}
		mustLatency(t, llmRespRow)
	})

	t.Run("plugin OnToolErrorCallback before AfterRunCallback and AfterToolCallback after", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		toolCtx := withFunctionCall(mCtx, "call-split-err")
		lookup := &fakeTool{name: "lookup"}
		boom := errors.New("tool failed before stop")

		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, lookup, nil))
		errs.add(p.OnToolErrorCallback()(toolCtx, lookup, nil, boom))
		ev := session.NewEvent(t.Context(), mCtx.invocationID)
		ev.Author = mCtx.agentName
		ev.Content = textContent("model", "partial")
		ev.Partial = true
		errs.add(p.OnEventCallback()(mCtx, ev))
		errs.check(t)

		p.AfterRunCallback()(mCtx)
		errs.add(p.AfterToolCallback()(toolCtx, lookup, nil, nil, boom))
		errs.check(t)

		// Flush with a sentinel row after AfterToolCallback so any duplicate
		// TOOL_ERROR would be observed before collection stops.
		mCtx2 := &mockInvocationContext{
			ctx:          mCtx.ctx,
			agentName:    "sentinel",
			sessionID:    "sess-sentinel",
			invocationID: "inv-sentinel",
			userID:       "user-sentinel",
		}
		errs.add(p.BeforeRunCallback()(mCtx2))
		errs.check(t)

		rows := collectRows(t, requests, func(rows []map[string]any) bool {
			return len(rowsOfType(rows, EventInvocationStarting)) >= 2
		})
		if n := len(rowsOfType(rows, EventToolError)); n != 1 {
			t.Fatalf("got %d TOOL_ERROR rows, want 1", n)
		}
		toolStart := placeOf(onlyRowOfType(t, rows, EventToolStarting))
		toolErrRow := onlyRowOfType(t, rows, EventToolError)
		if got := placeOf(toolErrRow); got != toolStart {
			t.Errorf("TOOL_ERROR place = %+v, want TOOL_STARTING place %+v", got, toolStart)
		}
	})

	t.Run("handed-over tool call holds back late settle until popToolSpan", func(t *testing.T) {
		ctx := context.Background()
		tm := newTraceManager()
		const invID = "inv-live-tool-seq"
		live := identity{invocationID: invID, agentName: "live_agent"}
		early := identity{invocationID: invID, agentName: "early"}
		call := toolKey(live, "call-1", "run_cached")

		tm.invocationSpan(ctx, live, nil)
		tm.pushAgentSpan(ctx, live)
		tm.pushAgentSpan(ctx, early)
		tm.popAgentSpan(ctx, early)
		tm.pushToolSpan(ctx, live, call)
		tm.popAgentSpan(ctx, live)
		tm.endInvocation(ctx, live)

		branchA := identity{invocationID: invID, agentName: "branch_a", branch: "branch_a"}
		cached := identity{invocationID: invID, agentName: "cached", branch: "branch_a"}
		tm.pushAgentSpan(ctx, branchA)
		src := &endedContext{Context: ctx, end: true}
		tm.pushAgentSpan(src, cached)
		branchARec := lateReviewAgent(t, tm, invID, "branch_a")
		cachedRec := lateReviewAgent(t, tm, invID, "cached")
		tm.popAgentSpan(ctx, branchA)
		atBranchEnd := src.reads
		tm.popToolSpan(ctx, live, call, false)
		if atBranchEnd != 0 || src.reads != 1 {
			t.Errorf("reads of late sub-agent's ended InvocationContext at branch_a end and after popToolSpan = %d, %d; want 0, 1", atBranchEnd, src.reads)
		}
		c := tm.completed[invID]
		// Nothing can end either activation anymore, so neither record
		// keeps a context of its callbacks or the span that it ran under.
		checkLateRecordDropped(t, c.late, branchARec)
		checkLateRecordDropped(t, c.late, cachedRec)
		c.late.mu.Lock()
		defer c.late.mu.Unlock()
		if got := len(c.late.agentStack); got != 1 {
			t.Fatalf("len(late.agentStack) = %d, want 1", got)
		}
		if got := c.late.openAgents.Load(); got {
			t.Fatalf("late.openAgents = %v, want false", got)
		}
	})
}

// TestLateShortcutAgentRetained verifies that a shortcut agent (one whose
// BeforeAgentCallback answered, so that it ends its InvocationContext and
// never gets an AfterAgentCallback) that starts in the late state under a
// late agent leaves the late state once nothing can end it anymore: its
// parent ended while the tool call that AfterRun handed over still held it
// back, and then that call returned. Its record keeps no context of its
// callbacks, and the completed run no longer counts as one with open late
// spans, so that the idle runs that complete after it prune it first.
func TestLateShortcutAgentRetained(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	tm.maxTracked = 1 // Completed runs are kept up to 64 (see rememberCompletedLocked).
	const invID = "inv-late-shortcut"
	live := identity{invocationID: invID, agentName: "live_agent"}
	call := toolKey(live, "call-1", "run_cached")

	tm.invocationSpan(ctx, live, nil)
	tm.pushAgentSpan(ctx, live)
	tm.pushToolSpan(ctx, live, call)
	tm.popAgentSpan(ctx, live)
	tm.endInvocation(ctx, live)

	branchA := identity{invocationID: invID, agentName: "branch_a", branch: "branch_a"}
	cached := identity{invocationID: invID, agentName: "cached", branch: "branch_a"}
	tm.pushAgentSpan(ctx, branchA)
	tm.pushAgentSpan(&endedContext{Context: ctx, end: true}, cached)
	cachedRec := lateReviewAgent(t, tm, invID, "cached")
	tm.popAgentSpan(ctx, branchA)
	if got := lateReviewStack(t, tm, invID); !slices.Equal(got, []string{"cached"}) {
		t.Fatalf("late agents after branch_a ended = %v, want [cached], held back by the open tool call", got)
	}
	tm.popToolSpan(ctx, live, call, false)

	late := lateReviewTrace(t, tm, invID)
	if got := lateReviewStack(t, tm, invID); len(got) != 0 {
		t.Errorf("late agents after the tool call returned = %v, want none", got)
	}
	if late.openAgents.Load() {
		t.Error("late.openAgents = true after the tool call returned, want false")
	}
	checkLateRecordDropped(t, late, cachedRec)
	for i := range 64 {
		idle := identity{invocationID: fmt.Sprintf("idle-%d", i), agentName: "root"}
		tm.invocationSpan(ctx, idle, nil)
		tm.endInvocation(ctx, idle)
	}
	tm.mu.Lock()
	_, kept := tm.completed[invID]
	tm.mu.Unlock()
	if kept {
		t.Error("completed run with nothing open in its late state outlived the 64 idle runs that completed after it, want it pruned first")
	}
}

// TestRunner_NonTaskModeFinishTaskTool verifies that a non-ModeTask LLM agent
// defining a custom tool named "finish_task" that returns
// {"result": "Task completed."} does not trigger popTaskAgentSpan in onEvent
// or emit a premature duplicate AGENT_COMPLETED row.
func TestRunner_NonTaskModeFinishTaskTool(t *testing.T) {
	type emptyArgs struct{}
	newCustomFinishTaskTool := func(t *testing.T) tool.Tool {
		t.Helper()
		ft, err := functiontool.New(
			functiontool.Config{Name: "finish_task", Description: "Custom non-task finish tool"},
			func(_ agent.Context, _ emptyArgs) (map[string]any, error) {
				return map[string]any{"result": "Task completed."}, nil
			},
		)
		if err != nil {
			t.Fatalf("functiontool.New(finish_task): %v", err)
		}
		return ft
	}

	t.Run("chat-mode root agent with custom finish_task tool", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		bot, err := llmagent.New(llmagent.Config{
			Name:  "chat_bot",
			Tools: []tool.Tool{newCustomFinishTaskTool(t)},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-custom-finish",
										Name: "finish_task",
										Args: map[string]any{},
									},
								},
							},
						},
					},
					{Content: textContent("model", "finished normally"), TurnComplete: true},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(chat_bot): %v", err)
		}

		r := newTestRunner(t, bot, p)
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("r.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		if got := len(rowsOfType(rows, EventAgentCompleted)); got != 1 {
			t.Fatalf("got %d AGENT_COMPLETED rows, want 1; event types: %v", got, eventTypes(rows))
		}
		assertAgentSpansClosed(t, rows)
	})

	t.Run("single-turn sub-agent with custom finish_task tool", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		helper, err := llmagent.New(llmagent.Config{
			Name:        "single_turn_helper",
			Description: "Runs in single-turn mode",
			Mode:        llmagent.ModeSingleTurn,
			Tools:       []tool.Tool{newCustomFinishTaskTool(t)},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-helper-finish",
										Name: "finish_task",
										Args: map[string]any{},
									},
								},
							},
						},
					},
					{Content: textContent("model", "helper summary"), TurnComplete: true},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(single_turn_helper): %v", err)
		}
		coord, err := llmagent.New(llmagent.Config{
			Name:      "coord",
			SubAgents: []agent.Agent{helper},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-helper",
										Name: "single_turn_helper",
										Args: map[string]any{"request": "do work"},
									},
								},
							},
						},
					},
					{Content: textContent("model", "coord done"), TurnComplete: true},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(coord): %v", err)
		}

		r := newTestRunner(t, coord, p)
		for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("r.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		assertAgentSpansClosed(t, rows)
	})
}

// TestTraceManager_CompleteResponseEndsOnlyItsAuthorsModelCalls verifies that
// a final response of an agent ends the model calls of its own activation,
// which ADK may end no other way, and leaves open those of a descendant
// activation that the response settles: the descendant's call may still be
// running, and its AfterModelCallback ends it with its span and latency.
func TestTraceManager_CompleteResponseEndsOnlyItsAuthorsModelCalls(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()

	sub, err := llmagent.New(llmagent.Config{Name: "sub"})
	if err != nil {
		t.Fatalf("llmagent.New(sub): %v", err)
	}
	coord, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "coord",
			SubAgents: []agent.Agent{sub},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New(coord): %v", err)
	}

	coordID := identity{invocationID: "inv-settled-llm", agentName: "coord", path: "coord@0"}
	subID := identity{invocationID: "inv-settled-llm", agentName: "sub", path: "coord@0/sub@1"}

	tm.invocationSpan(ctx, coordID, coord)
	tm.pushAgentSpan(ctx, coordID)
	tm.pushLLMSpan(ctx, coordID)
	tm.pushAgentSpan(ctx, subID)
	start := tm.pushLLMSpan(ctx, subID)

	// coord emits a final response settling both coord and sub.
	tm.completeResponse(ctx, coordID, "coord", "", nil)

	if got := openSpans(t, tm, "inv-settled-llm"); got.models != 1 {
		t.Fatalf("open model spans after completeResponse = %d, want 1, the call of sub", got.models)
	}
	if end := tm.popLLMSpan(ctx, subID, true); end.spanID != start.spanID || end.latency == nil {
		t.Errorf("LLM_RESPONSE place of sub = %+v, want the span %q of LLM_REQUEST with latency", end, start.spanID)
	}
	unclosed, termErr := tm.popUnclosedAgentSpans(ctx, coordID)
	if termErr != nil || len(unclosed) != 0 {
		t.Errorf("popUnclosedAgentSpans = (%v, %v), want (nil, nil)", unclosed, termErr)
	}
}

// TestTraceManager_StoppedStreamDoesNotPinCompletedRun verifies that runs
// whose caller stopped reading mid-stream, which end with their streaming
// model call open, leave no late state behind, so that the completed runs
// stay within capacity however many of them stop within the TTL. The partial
// event may carry a node path that the context of the model callbacks lacks,
// as for the agent that runner.Run runs.
func TestTraceManager_StoppedStreamDoesNotPinCompletedRun(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name      string
		eventPath string
	}{
		{name: "event without node path", eventPath: ""},
		{name: "event with node path", eventPath: "bot"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.maxTracked = 1
			limit := max(tm.maxTracked, 64)
			for i := range 5 * limit {
				id := identity{invocationID: fmt.Sprintf("stopped-%d", i), agentName: "bot"}
				tm.invocationSpan(ctx, id, nil)
				tm.pushAgentSpan(ctx, id)
				tm.pushLLMSpan(ctx, id)
				tm.recordFirstToken(ctx, id)
				evID := id
				evID.path = tc.eventPath
				tm.recordStreamed(ctx, evID, "bot", "", true)
				if unclosed, err := tm.popUnclosedAgentSpans(ctx, id); err != nil || len(unclosed) != 0 {
					t.Fatalf("popUnclosedAgentSpans = (%v, %v), want no failure", unclosed, err)
				}
				tm.endInvocation(ctx, id)
			}

			tm.mu.Lock()
			defer tm.mu.Unlock()
			if got := len(tm.completed); got > limit {
				t.Errorf("completed runs = %d, want at most %d", got, limit)
			}
			for id, c := range tm.completed {
				if c.late != nil {
					t.Errorf("completed run %q kept late state for a stream that its caller stopped reading", id)
				}
			}
		})
	}
}

// TestTraceManager_PartialEventStreamsOnlyItsAuthorsCall verifies that a
// partial event whose node path the context of the model callbacks lacks
// marks only the model call of its author as streaming: the open call of
// another agent still moves to the late state, so that a late
// AfterModelCallback ends the call's own span.
func TestTraceManager_PartialEventStreamsOnlyItsAuthorsCall(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	root := identity{invocationID: "two-calls", agentName: "root"}
	streaming := identity{invocationID: "two-calls", agentName: "streaming"}
	waiting := identity{invocationID: "two-calls", agentName: "waiting"}
	tm.invocationSpan(ctx, root, nil)
	tm.pushAgentSpan(ctx, root)
	tm.pushAgentSpan(ctx, streaming)
	tm.pushAgentSpan(ctx, waiting)
	tm.pushLLMSpan(ctx, streaming)
	start := tm.pushLLMSpan(ctx, waiting)
	evID := streaming
	evID.path = streaming.agentName
	tm.recordStreamed(ctx, evID, streaming.agentName, "", true)
	tm.endInvocation(ctx, root)

	tm.mu.Lock()
	c := tm.completed[root.invocationID]
	tm.mu.Unlock()
	if c == nil || c.late == nil {
		t.Fatal("finished run kept no late state for the open call that did not stream")
	}
	c.late.mu.Lock()
	kept := slices.Sorted(maps.Keys(c.late.llmSpans))
	c.late.mu.Unlock()
	if want := []string{llmKey(waiting)}; !slices.Equal(kept, want) {
		t.Errorf("model calls kept for late callbacks = %q, want %q", kept, want)
	}
	if end := tm.popLLMSpan(ctx, waiting, true); end.spanID != start.spanID || end.latency == nil {
		t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST with latency", end, start.spanID)
	}
}

// TestTraceManager_CompletedRunsWithOpenLateCallsStayBounded verifies that
// runs that end with calls still open stay within capacity, as their calls may
// never end, while a run that waits for a late call is kept over ordinary
// completions (up to half of capacity) and a run with a pending span
// reservation is kept over runs waiting for late calls.
func TestTraceManager_CompletedRunsWithOpenLateCallsStayBounded(t *testing.T) {
	ctx := context.Background()
	// endWithOpenTool runs id until its tool call starts and then ends it.
	endWithOpenTool := func(tm *traceManager, id identity) rowSpan {
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		span := tm.pushToolSpan(ctx, id, toolKey(id, "call", "lookup"))
		tm.endInvocation(ctx, id)
		return span
	}
	completed := func(tm *traceManager, id identity) bool {
		tm.mu.Lock()
		defer tm.mu.Unlock()
		return tm.completed[id.invocationID] != nil
	}

	t.Run("open calls beyond capacity", func(t *testing.T) {
		tm := newTraceManager()
		tm.maxTracked = 1
		limit := max(tm.maxTracked, 64)
		var ids []identity
		for i := range limit + 6 {
			id := identity{invocationID: fmt.Sprintf("open-%d", i), agentName: "live"}
			ids = append(ids, id)
			endWithOpenTool(tm, id)
		}
		tm.mu.Lock()
		got := len(tm.completed)
		tm.mu.Unlock()
		if got != limit {
			t.Errorf("completed runs = %d, want %d", got, limit)
		}
		if completed(tm, ids[0]) || !completed(tm, ids[len(ids)-1]) {
			t.Error("capacity pruning did not drop the least recently active run first")
		}
	})

	t.Run("late call outlives ordinary completions", func(t *testing.T) {
		tm := newTraceManager()
		tm.maxTracked = 1
		limit := max(tm.maxTracked, 64)
		live := identity{invocationID: "live", agentName: "live"}
		start := endWithOpenTool(tm, live)
		for i := range limit + 6 {
			other := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
			tm.invocationSpan(ctx, other, nil)
			tm.endInvocation(ctx, other)
		}
		if end, _, _ := tm.popToolSpan(ctx, live, toolKey(live, "call", "lookup"), false); end.spanID != start.spanID || end.latency == nil {
			t.Errorf("TOOL_COMPLETED place = %+v, want the span %q of TOOL_STARTING with latency", end, start.spanID)
		}
	})

	t.Run("late state in which a callback is starting a span", func(t *testing.T) {
		tm := newTraceManager()
		tm.maxTracked = 1
		limit := max(tm.maxTracked, 64)
		id := identity{invocationID: "starting-late-span", agentName: "root"}
		tm.invocationSpan(ctx, id, nil)
		tm.endInvocation(ctx, id)
		ready, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		t.Cleanup(unblock)
		var clockCalls atomic.Int64
		tm.now = func() time.Time {
			// The second reading of the clock is the start of the late span,
			// after its lookup reserved the late state.
			if clockCalls.Add(1) == 2 {
				close(ready)
				<-release
			}
			return time.Unix(1000, 0)
		}
		started := make(chan rowSpan, 1)
		go func() { started <- tm.pushToolSpan(ctx, id, "call") }()
		select {
		case <-ready:
		case <-time.After(5 * time.Second):
			t.Fatal("late span did not reach its start after invocation lookup")
		}
		for i := range limit + 1 {
			endWithOpenTool(tm, identity{invocationID: fmt.Sprintf("open-%d", i), agentName: "live"})
		}
		unblock()
		span := <-started
		if end, _, _ := tm.popToolSpan(ctx, id, "call", false); end.spanID != span.spanID || end.latency == nil {
			t.Errorf("TOOL_COMPLETED place = %+v, want the span %q of TOOL_STARTING with latency", end, span.spanID)
		}
	})
}

// partialThenWaitLLM yields one partial streamed chunk and then waits until its
// context is done, as a model that streams slowly does when its caller stops
// reading, and returns without an error.
type partialThenWaitLLM struct {
	chunkText string
}

func (m *partialThenWaitLLM) Name() string { return "partial-then-wait-llm" }

func (m *partialThenWaitLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(&model.LLMResponse{
			Content: textContent("model", m.chunkText),
			Partial: true,
		}, nil) {
			return
		}
		<-ctx.Done()
	}
}

// newTraceOnlyPlugin returns a plugin whose callbacks keep the trace in tm
// and write no rows: every event type is filtered out, so it needs no
// processor. It registers the callbacks that the production plugin registers.
func newTraceOnlyPlugin(t *testing.T, tm *traceManager) *baseplugin.Plugin {
	t.Helper()
	config := applyDefaults(DefaultConfig())
	config.DisableFlushOnRunEnd = true
	ap := &analyticsPlugin{config: config, allowlist: newSet("NO_SUCH_EVENT"), traces: tm}
	p, err := baseplugin.New(baseplugin.Config{
		Name:                  "trace-only",
		OnUserMessageCallback: ap.onUserMessage,
		BeforeRunCallback:     ap.beforeRun,
		AfterRunCallback:      ap.afterRun,
		OnEventCallback:       ap.onEvent,
		BeforeAgentCallback:   ap.beforeAgent,
		AfterAgentCallback:    ap.afterAgent,
		BeforeModelCallback:   ap.beforeModel,
		AfterModelCallback:    ap.afterModel,
		OnModelErrorCallback:  ap.onModelError,
		BeforeToolCallback:    ap.beforeTool,
		AfterToolCallback:     ap.afterTool,
		OnToolErrorCallback:   ap.onToolError,
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	return p
}

// TestRunner_StoppedStreamsKeepCompletedRunsBounded verifies through
// runner.Run that runs whose caller stops reading at a partial event while the
// model is still streaming, which leave their model call open when
// AfterRunCallback runs, leave no late state behind, so that the completed
// runs stay within the trace manager's capacity.
func TestRunner_StoppedStreamsKeepCompletedRunsBounded(t *testing.T) {
	tm := newTraceManager()
	tm.maxTracked = 1
	limit := max(tm.maxTracked, 64)
	p := newTraceOnlyPlugin(t, tm)
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &partialThenWaitLLM{chunkText: "hello "},
	})
	if err != nil {
		t.Fatalf("llmagent.New(bot): %v", err)
	}
	r := newTestRunner(t, bot, p)

	for i := range limit + 6 {
		var sawPartial bool
		for ev, err := range r.Run(t.Context(), "u1", fmt.Sprintf("s%d", i), textContent("user", "hi"), agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		}) {
			if err != nil {
				t.Fatalf("run %d: r.Run error: %v", i, err)
			}
			if ev != nil && ev.Partial {
				sawPartial = true
				break
			}
		}
		if !sawPartial {
			t.Fatalf("run %d yielded no partial event", i)
		}
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()
	if got := len(tm.completed); got > limit {
		t.Errorf("completed runs = %d, want at most %d", got, limit)
	}
	for id, c := range tm.completed {
		if c.late != nil {
			t.Errorf("completed run %q kept late state for a stream that its caller stopped reading", id)
		}
	}
}

// straddleEndOfRun runs callback on another goroutine, holds it once it has
// looked up the invocation of id, ends the run, and then lets callback go on
// while meanwhile, if any, runs on the test goroutine: a callback reads the
// clock in lookupInvocation and then once more before it locks the trace that
// it found.
func straddleEndOfRun(t *testing.T, tm *traceManager, id identity, callback func(), meanwhile ...func()) {
	t.Helper()
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
		return time.Now()
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		callback()
	}()
	select {
	case <-ready:
	case <-time.After(5 * time.Second):
		t.Fatal("callback did not look up its invocation")
	}
	tm.endInvocation(context.Background(), id)
	unblock()
	for _, f := range meanwhile {
		f()
	}
	<-done
}

// lateOpenCalls returns the number of model and tool calls that the late
// state of the completed run id holds open, and whether it has open spans.
func lateOpenCalls(t *testing.T, tm *traceManager, id identity) (open int, hasOpenSpans bool) {
	t.Helper()
	tm.mu.Lock()
	c := tm.completed[id.invocationID]
	var late *invocationTrace
	if c != nil {
		late = c.late
	}
	tm.mu.Unlock()
	if c == nil {
		t.Fatalf("run %q is not completed", id.invocationID)
	}
	if late == nil {
		return 0, false
	}
	hasOpenSpans = late.hasOpenLateSpans()
	late.mu.Lock()
	defer late.mu.Unlock()
	return len(late.llmSpans) + len(late.toolSpans), hasOpenSpans
}

// TestTraceManager_CallbacksAcrossEndOfRun verifies that a model or tool
// callback that looked the invocation up before AfterRun and acts after
// AfterRun moved the run's calls to the late state acts on the late state,
// where the callbacks that follow look for the call: a call that starts or
// ends leaves no open call behind that would keep the completed run from
// being pruned, a call that an error callback ended is found by the callback
// after it, so that AfterToolCallback writes no second TOOL_ERROR row (it
// writes one only if closedByError is false) and AfterModelCallback reports
// no second latency, and the first token of a call is recorded under the lock
// of the late state, where a late AfterModelCallback may end the call at the
// same time.
func TestTraceManager_CallbacksAcrossEndOfRun(t *testing.T) {
	ctx := context.Background()
	// start runs id with an agent span open.
	start := func(tm *traceManager, id identity) {
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
	}

	t.Run("tool call ends", func(t *testing.T) {
		tm := newTraceManager()
		tm.maxTracked = 1
		id := identity{invocationID: "straddle", agentName: "live"}
		key := toolKey(id, "call", "lookup")
		start(tm, id)
		begin := tm.pushToolSpan(ctx, id, key)
		var end rowSpan
		straddleEndOfRun(t, tm, id, func() { end, _, _ = tm.popToolSpan(ctx, id, key, false) })
		if end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("TOOL_COMPLETED place = %+v, want the span %q of TOOL_STARTING with latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
		// Ordinary completions beyond capacity prune the run first, as it
		// waits for no call.
		for i := range max(tm.maxTracked, 64) {
			other := identity{invocationID: fmt.Sprintf("finished-%d", i), agentName: "root"}
			tm.invocationSpan(ctx, other, nil)
			tm.endInvocation(ctx, other)
		}
		tm.mu.Lock()
		_, kept := tm.completed[id.invocationID]
		tm.mu.Unlock()
		if kept {
			t.Error("capacity pruning kept the run whose call ended across AfterRun over ordinary completions")
		}
	})

	t.Run("tool call fails, then AfterToolCallback", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		key := toolKey(id, "call", "lookup")
		start(tm, id)
		begin := tm.pushToolSpan(ctx, id, key)
		var failed rowSpan
		straddleEndOfRun(t, tm, id, func() { failed, _, _ = tm.popToolSpan(ctx, id, key, true) })
		if failed.spanID != begin.spanID || failed.latency == nil {
			t.Errorf("TOOL_ERROR place = %+v, want the span %q of TOOL_STARTING with latency", failed, begin.spanID)
		}
		if _, _, closedByError := tm.popToolSpan(ctx, id, key, false); !closedByError {
			t.Error("AfterToolCallback did not find the call that OnToolErrorCallback ended, so it writes a second TOOL_ERROR row")
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
	})

	t.Run("call of an unknown tool fails, then AfterToolCallback", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		key := toolKey(id, "call", "missing")
		start(tm, id)
		var failed rowSpan
		straddleEndOfRun(t, tm, id, func() { failed, _, _ = tm.popToolSpan(ctx, id, key, true) })
		if completed, _, closedByError := tm.popToolSpan(ctx, id, key, false); !closedByError || completed.spanID != failed.spanID {
			t.Errorf("AfterToolCallback = (%+v, %v), want the span %q of TOOL_ERROR, closed by the error", completed, closedByError, failed.spanID)
		}
	})

	t.Run("tool call starts, then ends", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		key := toolKey(id, "call", "lookup")
		start(tm, id)
		var begin rowSpan
		straddleEndOfRun(t, tm, id, func() { begin = tm.pushToolSpan(ctx, id, key) })
		if end, _, _ := tm.popToolSpan(ctx, id, key, false); end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("TOOL_COMPLETED place = %+v, want the span %q of TOOL_STARTING with latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
	})

	t.Run("model call ends", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		start(tm, id)
		begin := tm.pushLLMSpan(ctx, id)
		var end rowSpan
		straddleEndOfRun(t, tm, id, func() { end = tm.popLLMSpan(ctx, id, true) })
		if end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST with latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
	})

	t.Run("model call starts, then ends", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		start(tm, id)
		var begin rowSpan
		straddleEndOfRun(t, tm, id, func() { begin = tm.pushLLMSpan(ctx, id) })
		if end := tm.popLLMSpan(ctx, id, true); end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST with latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
	})

	t.Run("agent starts, then ends", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		child := identity{invocationID: "straddle", agentName: "child"}
		start(tm, id)
		var begin rowSpan
		straddleEndOfRun(t, tm, id, func() { begin, _ = tm.pushAgentSpan(ctx, child) })
		if end := tm.popAgentSpan(ctx, child); end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("AGENT_COMPLETED place = %+v, want the span %q of AGENT_STARTING with latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late state after the agent ended: %d open calls (open spans %v), want none", open, hasOpenSpans)
		}
	})

	t.Run("first token arrives while the call ends", func(t *testing.T) {
		// Use a fake clock so start (t0 = 0ms), first-token arrival
		// (t1 = 10ms), run completion (t2 = 50ms) and call end
		// (t3 = 60ms) happen at distinct deterministic timestamps.
		// Without the lockLive re-lookup in recordFirstToken, a first
		// token that straddles endInvocation finds no record on the
		// cleared finished trace and loses time_to_first_token_ms
		// (it falls back to total_ms = 60). With the re-lookup, the
		// token is recorded on the late state while preserving the
		// captured t1 = 10ms arrival timestamp rather than re-reading
		// the clock at t2 = 50ms, so time_to_first_token_ms == 10 and
		// total_ms == 60.
		tm := newTraceManager()
		base := time.Unix(1000, 0)
		t0 := base
		t1 := base.Add(10 * time.Millisecond) // first token
		t2 := base.Add(50 * time.Millisecond) // endInvocation
		t3 := base.Add(60 * time.Millisecond) // popLLMSpan
		var mu sync.Mutex
		clock := t0
		tm.now = func() time.Time {
			mu.Lock()
			defer mu.Unlock()
			return clock
		}
		id := identity{invocationID: "straddle", agentName: "live"}
		start(tm, id)
		begin := tm.pushLLMSpan(ctx, id)

		// Straddle recordFirstToken across endInvocation: block on the
		// second tm.now call (the one in recordFirstToken after the
		// invocation lookup), end the run, then release.
		ready, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		t.Cleanup(unblock)
		var clockCalls atomic.Int64
		realNow := tm.now
		tm.now = func() time.Time {
			if clockCalls.Add(1) == 2 {
				mu.Lock()
				clock = t1 // first token time
				mu.Unlock()
				close(ready)
				<-release
				return t1
			}
			return realNow()
		}
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			tm.recordFirstToken(ctx, id)
		}()
		select {
		case <-ready:
		case <-time.After(5 * time.Second):
			t.Fatal("recordFirstToken did not read clock after looking up its invocation")
		}
		mu.Lock()
		clock = t2 // endInvocation time
		mu.Unlock()
		tm.endInvocation(ctx, id)
		unblock()
		wg.Wait()

		mu.Lock()
		clock = t3 // popLLMSpan time
		mu.Unlock()
		// Restore the plain clock for popLLMSpan.
		tm.now = func() time.Time {
			mu.Lock()
			defer mu.Unlock()
			return clock
		}
		end := tm.popLLMSpan(ctx, id, true)
		if end.spanID != begin.spanID || end.latency == nil {
			t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST with latency", end, begin.spanID)
		}
		if end.latency["time_to_first_token_ms"] != int64(10) || end.latency["total_ms"] != int64(60) {
			t.Errorf("time_to_first_token_ms = %v, total_ms = %v, want 10 and 60", end.latency["time_to_first_token_ms"], end.latency["total_ms"])
		}
	})

	t.Run("model call fails, then AfterModelCallback", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "straddle", agentName: "live"}
		start(tm, id)
		begin := tm.pushLLMSpan(ctx, id)
		var failed rowSpan
		straddleEndOfRun(t, tm, id, func() { failed = tm.popLLMSpan(ctx, id, false) })
		if failed.spanID != begin.spanID || failed.latency == nil {
			t.Errorf("LLM_ERROR place = %+v, want the span %q of LLM_REQUEST with latency", failed, begin.spanID)
		}
		// A plugin recovered from the error, so ADK runs AfterModelCallback.
		if end := tm.popLLMSpan(ctx, id, true); end.spanID != begin.spanID || end.latency != nil {
			t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST without latency", end, begin.spanID)
		}
		if open, hasOpenSpans := lateOpenCalls(t, tm, id); open != 0 || hasOpenSpans {
			t.Errorf("late open calls after the call ended = %d (open spans %v), want 0", open, hasOpenSpans)
		}
	})
}

// TestTraceManager_StreamedEventRacingEndOfRunSharesNoCall verifies, under
// -race, that a recordStreamed that looked the run up before AfterRun and
// takes its lock after it does not write to a model call that moved to the
// late state, which callbacks after AfterRun read under the late state's own
// lock.
func TestTraceManager_StreamedEventRacingEndOfRunSharesNoCall(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	for i := range 200 {
		id := identity{invocationID: fmt.Sprintf("streamed-race-%d", i), agentName: "live"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		tm.pushLLMSpan(ctx, id)
		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Go(func() {
			<-start
			tm.recordStreamed(ctx, id, id.agentName, id.branch, true)
		})
		wg.Go(func() {
			<-start
			tm.endInvocation(ctx, id)
			tm.mu.Lock()
			late := tm.completed[id.invocationID].late
			tm.mu.Unlock()
			if late == nil {
				// recordStreamed ran first, so the streaming call did not move.
				return
			}
			late.mu.Lock()
			late.openCallOwnersLocked(nil)
			late.mu.Unlock()
			tm.popLLMSpan(ctx, id, true)
		})
		close(start)
		wg.Wait()
	}
}

// TestTraceManager_LateStateKeepsNoCallbackContext verifies that the calls
// that stay behind when a run ends keep no context of the run's callbacks:
// not the context of the callback that started a call, not the
// InvocationContext of an activation or a retired cause, which only the
// records of activations get but which a call drops all the same, and not the
// span that it started under, whose record keeps the context of its own
// callback. That holds for the calls that are open at AfterRun and move to the
// late state, for those that an error callback ended, whether they move along
// or wait in the completed entry for a callback after AfterRun, and for the
// span that popToolSpan makes up for a call whose start it did not see.
func TestTraceManager_LateStateKeepsNoCallbackContext(t *testing.T) {
	ctx := context.Background()
	// holdsNoContext fails the test for each call in calls that keeps a
	// context of its callback, and returns the number of calls, so that a test
	// can tell clean calls from none.
	holdsNoContext := func(t *testing.T, where string, calls map[string]*spanRecord) int {
		t.Helper()
		for key, rec := range calls {
			if rec.canceled != nil {
				t.Errorf("%s call %q keeps the context of its callback", where, key)
			}
			if rec.ended != nil || rec.activation != nil {
				t.Errorf("%s call %q keeps the InvocationContext of an activation", where, key)
			}
			if rec.parent != nil {
				t.Errorf("%s call %q keeps the span that it started under", where, key)
			}
			if len(rec.retiredCauses) > 0 || rec.openFailureCause() != nil {
				t.Errorf("%s call %q keeps a failure cause (%v)", where, key, rec.openFailureCause())
			}
		}
		return len(calls)
	}

	t.Run("late state", func(t *testing.T) {
		tm := newTraceManager()
		live := identity{invocationID: "late-context", agentName: "live"}
		failed := identity{invocationID: "late-context", agentName: "failed"}
		tm.invocationSpan(ctx, live, nil)
		inv := tm.invocation(ctx, live)
		// withActivation gives the record of a call what the record of an
		// activation keeps of its run's InvocationContext, so that the test
		// sees the call drop it too.
		withActivation := func(call func() *spanRecord) {
			inv.mu.Lock()
			defer inv.mu.Unlock()
			rec := call()
			rec.ended = func() bool { return false }
			rec.activation = &live
		}
		tm.pushAgentSpan(ctx, live)
		tm.pushLLMSpan(ctx, live)
		tm.pushToolSpan(ctx, live, toolKey(live, "open", "lookup"))
		tm.pushToolSpan(ctx, live, toolKey(live, "failed", "lookup"))
		withActivation(func() *spanRecord { return inv.toolSpans[toolKey(live, "failed", "lookup")] })
		tm.popToolSpan(ctx, live, toolKey(live, "failed", "lookup"), true)
		tm.pushAgentSpan(ctx, failed)
		tm.pushLLMSpan(ctx, failed)
		withActivation(func() *spanRecord { return inv.llmSpans[llmKey(failed)] })
		tm.popLLMSpan(ctx, failed, false)
		withActivation(func() *spanRecord { return inv.llmSpans[llmKey(live)] })
		withActivation(func() *spanRecord { return inv.toolSpans[toolKey(live, "open", "lookup")] })
		inv.mu.Lock()
		inv.toolSpans[toolKey(live, "open", "lookup")].retiredCauses = []retiredCause{{err: errors.New("retired branch failed"), name: "branch"}}
		inv.mu.Unlock()
		tm.endInvocation(ctx, live)

		tm.mu.Lock()
		late := tm.completed[live.invocationID].late
		tm.mu.Unlock()
		if late == nil {
			t.Fatal("the run's open calls did not move to the late state")
		}
		late.mu.Lock()
		defer late.mu.Unlock()
		got := holdsNoContext(t, "open model", late.llmSpans) +
			holdsNoContext(t, "open tool", late.toolSpans) +
			holdsNoContext(t, "ended model", late.completedModels) +
			holdsNoContext(t, "ended tool", late.completedTools)
		if got != 4 {
			t.Errorf("late calls = %d, want 4", got)
		}
	})

	t.Run("ended calls waiting in the completed entry", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "ended-context", agentName: "bot"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		tm.pushToolSpan(ctx, id, toolKey(id, "call", "lookup"))
		tm.popToolSpan(ctx, id, toolKey(id, "call", "lookup"), true)
		tm.pushLLMSpan(ctx, id)
		tm.popLLMSpan(ctx, id, false)
		tm.endInvocation(ctx, id)

		tm.mu.Lock()
		defer tm.mu.Unlock()
		completed := tm.completed[id.invocationID]
		if completed == nil || completed.late != nil {
			t.Fatal("a run that left only ended calls behind did not keep them in its completed entry, without late state")
		}
		got := holdsNoContext(t, "ended model", completed.endedModels) +
			holdsNoContext(t, "ended tool", completed.endedTools)
		if got != 2 {
			t.Errorf("ended calls = %d, want 2", got)
		}
	})

	t.Run("span made up for a call that was not seen to start", func(t *testing.T) {
		tm := newTraceManager()
		id := identity{invocationID: "made-up-context", agentName: "bot"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		// ADK runs OnToolErrorCallback alone for a call of a tool that does
		// not exist.
		tm.popToolSpan(ctx, id, toolKey(id, "call", "missing"), true)

		inv := tm.invocation(ctx, id)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		if got := holdsNoContext(t, "made-up", inv.completedTools); got != 1 {
			t.Errorf("made-up calls = %d, want 1", got)
		}
	})
}

// TestTraceManager_ToolEndBeforeEndOfRunReturnsPurgedActivations verifies
// that an OnToolErrorCallback that ends a tool call before AfterRun gets the
// failed sub-agent activation that the call ran, purged from agentStack,
// straight from popToolSpan, so that the plugin writes its AGENT_ERROR row
// then. AfterRun then keeps the ended call in completed.endedTools, without
// late state, and an AfterToolCallback of the call that comes after AfterRun
// still finds it, with the span of its TOOL_ERROR row, and gets nothing purged
// again, so no second TOOL_ERROR or AGENT_ERROR row is written.
func TestTraceManager_ToolEndBeforeEndOfRunReturnsPurgedActivations(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	coord := identity{invocationID: "purged-late", agentName: "coordinator"}
	helper := identity{invocationID: "purged-late", agentName: "helper", branch: "helper@1", path: "/helper@1"}
	key := toolKey(coord, "call", "helper")
	tm.invocationSpan(ctx, coord, nil)
	tm.pushAgentSpan(ctx, coord)
	tm.pushToolSpan(ctx, coord, key)
	tm.pushAgentSpan(ctx, helper)
	toolErr, purged, closedByError := tm.popToolSpan(ctx, coord, key, true)
	if closedByError {
		t.Fatal("OnToolErrorCallback reported closedByError = true")
	}
	if len(purged) != 1 || purged[0].agentName != "helper" || purged[0].err == nil {
		t.Errorf("activations purged on popToolSpan = %+v, want the helper that the tool call ran, with an error", purged)
	}
	tm.endInvocation(ctx, coord)

	tm.mu.Lock()
	c := tm.completed[coord.invocationID]
	tm.mu.Unlock()
	if c == nil || c.late != nil || len(c.endedTools) != 1 {
		t.Fatalf("completed entry = %+v, want nil late state with 1 lightweight endedTools entry", c)
	}
	end, again, closedByError := tm.popToolSpan(ctx, coord, key, false)
	if !closedByError || end.spanID != toolErr.spanID || len(again) != 0 {
		t.Errorf("late AfterToolCallback = (%+v, %+v, %v), want span %q with no purged entries and closedByError = true", end, again, closedByError, toolErr.spanID)
	}
}

// TestTraceManager_RunsWithOnlyEndedCallsKeepNoLateState verifies that runs
// that end with no call open, but with calls that an error callback ended, as
// every run does whose model error no plugin recovered from, keep no late
// state, and that a callback after AfterRun still finds such a call: the
// AfterModelCallback of a plugin that recovered from the error late, or the
// AfterToolCallback that ADK runs after OnToolErrorCallback.
func TestTraceManager_RunsWithOnlyEndedCallsKeepNoLateState(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const runs = 2000
	var last identity
	var lastBegin, lastToolError rowSpan
	var lastTool string
	for i := range runs {
		id := identity{invocationID: fmt.Sprintf("failed-%d", i), agentName: "bot"}
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		tool := toolKey(id, "call", "lookup")
		tm.pushToolSpan(ctx, id, tool)
		toolError, _, _ := tm.popToolSpan(ctx, id, tool, true)
		begin := tm.pushLLMSpan(ctx, id)
		tm.recordError(ctx, id, llmKey(id), errors.New("model unavailable"))
		tm.popLLMSpan(ctx, id, false)
		_, _ = tm.popUnclosedAgentSpans(ctx, id)
		tm.endInvocation(ctx, id)
		last, lastBegin, lastTool, lastToolError = id, begin, tool, toolError
	}

	tm.mu.Lock()
	var withLateState int
	for _, c := range tm.completed {
		if c.late != nil {
			withLateState++
		}
	}
	tm.mu.Unlock()
	if withLateState != 0 {
		t.Errorf("completed runs with late state = %d of %d, want 0", withLateState, runs)
	}
	// A plugin that recovered from the error late makes ADK run
	// AfterModelCallback after AfterRun.
	if end := tm.popLLMSpan(ctx, last, true); end.spanID != lastBegin.spanID || end.latency != nil {
		t.Errorf("LLM_RESPONSE place = %+v, want the span %q of LLM_REQUEST without latency", end, lastBegin.spanID)
	}
	if end, _, closedByError := tm.popToolSpan(ctx, last, lastTool, false); !closedByError || end.spanID != lastToolError.spanID {
		t.Errorf("AfterToolCallback = (%+v, %v), want the span %q of TOOL_ERROR, closed by the error", end, closedByError, lastToolError.spanID)
	}
}

// TestRunner_FailedModelRunsKeepNoLateState verifies through runner.Run that
// runs whose model call fails, which ADK ends without AfterModelCallback when
// no plugin recovers from the error, keep no late state.
func TestRunner_FailedModelRunsKeepNoLateState(t *testing.T) {
	tm := newTraceManager()
	p := newTraceOnlyPlugin(t, tm)
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &failingLLM{err: errors.New("model unavailable")},
	})
	if err != nil {
		t.Fatalf("llmagent.New(bot): %v", err)
	}
	r := newTestRunner(t, bot, p)
	const runs = 20
	for i := range runs {
		var sawErr bool
		for _, err := range r.Run(t.Context(), "u1", fmt.Sprintf("s%d", i), textContent("user", "hi"), agent.RunConfig{}) {
			sawErr = sawErr || err != nil
		}
		if !sawErr {
			t.Fatalf("run %d: r.Run yielded no error", i)
		}
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()
	if got := len(tm.completed); got != runs {
		t.Fatalf("completed runs = %d, want %d", got, runs)
	}
	for id, c := range tm.completed {
		if c.late != nil {
			t.Errorf("completed run %q keeps late state", id)
		}
	}
}

// TestRunner_ContentEventKeepsInFlightSubAgentModelCall verifies through
// runner.Run that a custom agent's content event, written while the model
// call of its sub-agent is in flight, leaves the call open: the call's
// LLM_REQUEST and LLM_RESPONSE rows share a span, and the response carries
// the call's latency.
func TestRunner_ContentEventKeepsInFlightSubAgentModelCall(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	gate := &blockingLLM{entered: make(chan struct{}), release: make(chan struct{})}
	sub, err := llmagent.New(llmagent.Config{Name: "sub", Model: gate})
	if err != nil {
		t.Fatalf("llmagent.New(sub): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{sub},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// sub runs on a goroutine of its own, and each of its events
				// is acknowledged once yielded, as a parallel agent does.
				type result struct {
					ev  *session.Event
					err error
					ack chan struct{}
				}
				results := make(chan result)
				go func() {
					defer close(results)
					for ev, err := range sub.Run(ctx) {
						ack := make(chan struct{})
						results <- result{ev: ev, err: err, ack: ack}
						<-ack
					}
				}()
				<-gate.entered
				progress := session.NewEvent(ctx, ctx.InvocationID())
				progress.Author = "root"
				progress.Branch = ctx.Branch()
				progress.LLMResponse = model.LLMResponse{Content: textContent("model", "working on it")}
				reading := yield(progress, nil)
				close(gate.release)
				for res := range results {
					if reading {
						reading = yield(res.ev, res.err)
					}
					close(res.ack)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	r := newTestRunner(t, root, p)
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run error: %v", err)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	request := placeOf(onlyRowOfType(t, rows, EventLLMRequest))
	response := onlyRowOfType(t, rows, EventLLMResponse)
	if got := placeOf(response); got != request {
		t.Errorf("LLM_RESPONSE place = %+v, want LLM_REQUEST place %+v", got, request)
	}
	mustLatency(t, response)
}

// TestOnEvent_FinishTaskOfAgentWithoutSpanWritesNoAgentCompleted verifies
// that a successful finish_task function response of an author that has no
// open agent span writes no AGENT_COMPLETED row.
func TestOnEvent_FinishTaskOfAgentWithoutSpanWritesNoAgentCompleted(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	ev := session.NewEvent(t.Context(), mCtx.invocationID)
	ev.Author = "unstarted_helper"
	ev.Content = &genai.Content{
		Role: "user",
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name:     "finish_task",
			Response: map[string]any{"result": "Task completed."},
		}}},
	}
	errs.add(p.OnEventCallback()(mCtx, ev))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventAgentCompleted)); n != 0 {
		t.Errorf("got %d AGENT_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
}

// TestTraceManager_LiveToLatePendingReservationSurvivesCompletionPressure
// verifies that a pushToolSpan or pushLLMSpan callback that reserved pendingSpans
// on a live invocation before AfterRun is not evicted from tm.completed by
// completion pressure (either ordinary completions or completions leaving open
// late calls) before lockLive re-enters lookupInvocation on the late state.
func TestTraceManager_LiveToLatePendingReservationSurvivesCompletionPressure(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name     string
		stuck    bool
		start    func(tm *traceManager, id identity) rowSpan
		complete func(tm *traceManager, id identity) rowSpan
	}{
		{
			name:  "tool_call_ordinary_pressure",
			stuck: false,
			start: func(tm *traceManager, id identity) rowSpan {
				return tm.pushToolSpan(ctx, id, toolKey(id, "call-1", "lookup"))
			},
			complete: func(tm *traceManager, id identity) rowSpan {
				end, _, _ := tm.popToolSpan(ctx, id, toolKey(id, "call-1", "lookup"), false)
				return end
			},
		},
		{
			name:  "tool_call_open_late_pressure",
			stuck: true,
			start: func(tm *traceManager, id identity) rowSpan {
				return tm.pushToolSpan(ctx, id, toolKey(id, "call-1", "lookup"))
			},
			complete: func(tm *traceManager, id identity) rowSpan {
				end, _, _ := tm.popToolSpan(ctx, id, toolKey(id, "call-1", "lookup"), false)
				return end
			},
		},
		{
			name:  "model_call_ordinary_pressure",
			stuck: false,
			start: func(tm *traceManager, id identity) rowSpan {
				return tm.pushLLMSpan(ctx, id)
			},
			complete: func(tm *traceManager, id identity) rowSpan {
				return tm.popLLMSpan(ctx, id, true)
			},
		},
		{
			name:  "model_call_open_late_pressure",
			stuck: true,
			start: func(tm *traceManager, id identity) rowSpan {
				return tm.pushLLMSpan(ctx, id)
			},
			complete: func(tm *traceManager, id identity) rowSpan {
				return tm.popLLMSpan(ctx, id, true)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			tm.maxTracked = 1
			handoffID := identity{invocationID: "inv-handoff", agentName: "live"}
			invStart := tm.invocationSpan(ctx, handoffID, nil)

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
				return time.Now()
			}

			var begin rowSpan
			done := make(chan struct{})
			go func() {
				defer close(done)
				begin = tc.start(tm, handoffID)
			}()

			select {
			case <-ready:
			case <-time.After(5 * time.Second):
				t.Fatal("callback did not pause after lookupInvocation")
			}

			// Move inv-handoff to tm.completed while its live pendingSpans == 1,
			// then flood tm.completed past max(maxTracked, 64) = 64 before the
			// paused callback reaches lockLive().
			tm.endInvocation(ctx, handoffID)
			for i := range 70 {
				other := identity{invocationID: fmt.Sprintf("other-%d", i), agentName: "root"}
				tm.invocationSpan(ctx, other, nil)
				if tc.stuck {
					tm.pushToolSpan(ctx, other, toolKey(other, "stuck", "bg"))
				}
				tm.endInvocation(ctx, other)
			}

			tm.mu.Lock()
			_, stillActive := tm.invocations[handoffID.invocationID]
			completed := tm.completed[handoffID.invocationID]
			tm.mu.Unlock()
			if stillActive {
				t.Fatal("inv-handoff remained in tm.invocations after endInvocation")
			}
			if completed == nil {
				t.Fatal("inv-handoff was pruned from tm.completed while live pendingSpans was held")
			}

			unblock()
			<-done

			if begin.traceID != invStart.traceID || begin.parentSpanID != invStart.spanID {
				t.Fatalf("start span = %+v, want traceID=%q parentSpanID=%q", begin, invStart.traceID, invStart.spanID)
			}

			end := tc.complete(tm, handoffID)
			if end.spanID != begin.spanID || end.latency == nil {
				t.Fatalf("completion span = %+v, want spanID=%q with non-nil latency", end, begin.spanID)
			}

			// Once both livePending and late open calls are 0, subsequent
			// completions prune inv-handoff normally.
			for i := range 65 {
				later := identity{invocationID: fmt.Sprintf("later-%d", i), agentName: "root"}
				tm.invocationSpan(ctx, later, nil)
				tm.endInvocation(ctx, later)
			}
			tm.mu.Lock()
			_, kept := tm.completed[handoffID.invocationID]
			tm.mu.Unlock()
			if kept {
				t.Error("inv-handoff remained in tm.completed after its late call finished and capacity rolled over")
			}
		})
	}
}

// TestTraceManager_PopTaskAgentSpanRejectsNonTaskAgent verifies that
// popTaskAgentSpan returns false without touching the parent's
// lastActivity when the agent is not in task mode. This guards the fix that
// moved the taskMode check before touchAgentActivityLocked: a rejected
// non-task finish_task must not count as a completed descendant.
func TestTraceManager_PopTaskAgentSpanRejectsNonTaskAgent(t *testing.T) {
	ctx := context.Background()
	const invID = "inv-task-reject"
	parent := identity{invocationID: invID, agentName: "parent", path: ""}
	child := identity{invocationID: invID, agentName: "child", path: "parent.child@1"}

	clock := time.Unix(1000, 0)
	t0 := clock
	t1 := t0.Add(100 * time.Millisecond)
	t2 := t0.Add(200 * time.Millisecond)

	tm := newTraceManager()
	tm.now = func() time.Time { return clock }
	tm.invocationSpan(ctx, parent, nil)

	// Push parent and child. Synthetic ctx (context.Background) does not
	// implement WithICDelta (see runningAgentChildren), so pushAgentSpan leaves
	// taskMode=false on child.
	clock = t0
	tm.pushAgentSpan(ctx, parent)
	tm.pushAgentSpan(ctx, child)

	// Record parent agent's lastActivity at t1 via an LLM span.
	clock = t1
	tm.pushLLMSpan(ctx, parent)
	tm.popLLMSpan(ctx, parent, true)

	// Find the parent agent record (agentStack[1], not [0] which is
	// the invocation root).
	inv := tm.invocations[invID]
	inv.mu.Lock()
	parentRec := inv.agentStack[1]
	wantActivity := t1
	if parentRec.lastActivity != wantActivity {
		inv.mu.Unlock()
		t.Fatalf("parent.lastActivity = %v, want %v after LLM pop", parentRec.lastActivity, wantActivity)
	}
	inv.mu.Unlock()

	// Call popTaskAgentSpan on the non-task child at t2. It should return
	// false and NOT advance parent.lastActivity.
	clock = t2
	_, ok := tm.popTaskAgentSpan(ctx, child, nil)
	if ok {
		t.Fatal("popTaskAgentSpan returned ok=true for non-task child, want false")
	}

	// parent.lastActivity must still be t1, not t2.
	inv.mu.Lock()
	if parentRec.lastActivity != wantActivity {
		inv.mu.Unlock()
		t.Fatalf("parent.lastActivity = %v after rejected popTaskAgentSpan, want %v (must not advance)", parentRec.lastActivity, wantActivity)
	}
	inv.mu.Unlock()
}

// TestTraceManager_StraddlingPushDoesNotRetainInvocationTrace runs a callback
// that looks the run up and reserves a span on it before AfterRun, and acts on
// the late state after AfterRun (see lockLive), as pushToolSpan does when it
// straddles the end of the run. Once the callback releases its reservation,
// completedTrace.livePending is a standalone counter at zero, and neither the
// finished run's invocationTrace nor the context of its root callback stays
// reachable: both are garbage collected without any further lookup.
func TestTraceManager_StraddlingPushDoesNotRetainInvocationTrace(t *testing.T) {
	type payload struct{ _ [64]byte }
	type payloadKey struct{}
	ctx := context.Background()
	tm := newTraceManager()
	id := identity{invocationID: "inv-straddle", agentName: "root"}
	rootPayload := func() weak.Pointer[payload] {
		p := &payload{}
		tm.invocationSpan(context.WithValue(ctx, payloadKey{}, p), id, nil)
		return weak.Make(p)
	}()
	var completed *completedTrace
	liveInv := func() weak.Pointer[invocationTrace] {
		inv := tm.lookupInvocation(ctx, id, false, true)
		tm.endInvocation(ctx, id)
		tm.mu.Lock()
		completed = tm.completed[id.invocationID]
		tm.mu.Unlock()
		if completed == nil {
			t.Fatal("completed trace not found after endInvocation")
		}
		if completed.livePending == nil || completed.livePending.Load() != 1 {
			t.Fatal("completed.livePending does not hold the straddling callback's reservation")
		}
		if inv.lockLive() {
			inv.mu.Unlock()
			t.Fatal("lockLive reported the run as live after endInvocation")
		}
		tm.pushToolSpan(ctx, id, "straddle-call")
		inv.pendingSpans.Add(-1)
		return weak.Make(inv)
	}()
	if completed.livePending != nil && completed.livePending.Load() != 0 {
		t.Fatalf("completed.livePending = %d after the straddling callback released its reservation, want 0", completed.livePending.Load())
	}
	runtime.GC()
	if liveInv.Value() != nil {
		t.Error("the finished run's invocationTrace is still reachable after GC")
	}
	if rootPayload.Value() != nil {
		t.Error("the context of the run's root callback is still reachable after GC")
	}
	tm.popToolSpan(ctx, id, "straddle-call", false)
}

// TestRepro_FailedModelCallsPinCompletedCache verifies through runner.Run that
// model calls which ADK ends without any model callback, because an agent's
// BeforeModelCallback returned an error after this plugin's own callback
// started the call, hold no completed-run slot, and that an ordinary run
// that completes after a capacity's worth of them keeps its identity.
func TestRepro_FailedModelCallsPinCompletedCache(t *testing.T) {
	tm := newTraceManager()
	tm.maxTracked = 1
	limit := max(tm.maxTracked, 64)
	config := applyDefaults(DefaultConfig())
	config.DisableFlushOnRunEnd = true
	ap := &analyticsPlugin{config: config, allowlist: newSet("NO_SUCH_EVENT"), traces: tm}
	p, err := baseplugin.New(baseplugin.Config{
		Name:                  "trace-bound",
		OnUserMessageCallback: ap.onUserMessage,
		BeforeRunCallback:     ap.beforeRun,
		AfterRunCallback:      ap.afterRun,
		OnEventCallback:       ap.onEvent,
		BeforeAgentCallback:   ap.beforeAgent,
		AfterAgentCallback:    ap.afterAgent,
		BeforeModelCallback:   ap.beforeModel,
		AfterModelCallback:    ap.afterModel,
		OnModelErrorCallback:  ap.onModelError,
	})
	if err != nil {
		t.Fatal(err)
	}
	guarded, err := llmagent.New(llmagent.Config{
		Name:  "guarded",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "never")}}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, errors.New("blocked by guardrail")
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	rg := newTestRunner(t, guarded, p)
	for i := range limit + 6 {
		for range rg.Run(t.Context(), "u1", fmt.Sprintf("g%d", i), textContent("user", "hi"), agent.RunConfig{}) {
		}
	}
	tm.mu.Lock()
	pinned := 0
	for _, c := range tm.completed {
		if c.late != nil && c.late.hasOpenCalls() {
			pinned++
		}
	}
	tm.mu.Unlock()

	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "ok"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var invID string
	for ev, err := range newTestRunner(t, bot, p).Run(t.Context(), "u1", "ordinary", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev != nil && invID == "" {
			invID = ev.InvocationID
		}
	}
	tm.mu.Lock()
	kept := tm.completed[invID] != nil
	tm.mu.Unlock()
	if pinned > 0 || !kept {
		t.Errorf("runs pinned by a call with no callback = %d, want 0; ordinary run's identity kept = %v, want true", pinned, kept)
	}
}

// newTraceOnlyToolPlugin returns a plugin like newTraceOnlyPlugin that also
// keeps the trace of tool calls.
func newTraceOnlyToolPlugin(t *testing.T, tm *traceManager) *baseplugin.Plugin {
	t.Helper()
	config := applyDefaults(DefaultConfig())
	config.DisableFlushOnRunEnd = true
	ap := &analyticsPlugin{config: config, allowlist: newSet("NO_SUCH_EVENT"), traces: tm}
	p, err := baseplugin.New(baseplugin.Config{
		Name:                  "trace-only-with-tools",
		OnUserMessageCallback: ap.onUserMessage,
		BeforeRunCallback:     ap.beforeRun,
		AfterRunCallback:      ap.afterRun,
		OnEventCallback:       ap.onEvent,
		BeforeAgentCallback:   ap.beforeAgent,
		AfterAgentCallback:    ap.afterAgent,
		BeforeModelCallback:   ap.beforeModel,
		AfterModelCallback:    ap.afterModel,
		OnModelErrorCallback:  ap.onModelError,
		BeforeToolCallback:    ap.beforeTool,
		AfterToolCallback:     ap.afterTool,
		OnToolErrorCallback:   ap.onToolError,
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	return p
}

// toolThenTextLLM asks for a call of the tool toolName until the request ends
// with a function response, and then answers with text.
type toolThenTextLLM struct {
	toolName string
}

func (m *toolThenTextLLM) Name() string { return "tool-then-text-model" }

func (m *toolThenTextLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if n := len(req.Contents); n > 0 && req.Contents[n-1] != nil {
			for _, part := range req.Contents[n-1].Parts {
				if part != nil && part.FunctionResponse != nil {
					yield(&model.LLMResponse{Content: textContent("model", "done"), TurnComplete: true}, nil)
					return
				}
			}
		}
		yield(&model.LLMResponse{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: m.toolName, Args: map[string]any{}}}},
		}}, nil)
	}
}

// transferToLLM asks to transfer to the agent target.
type transferToLLM struct {
	target string
}

func (m *transferToLLM) Name() string { return "transfer-to-model" }

func (m *transferToLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: []*genai.Part{genai.NewPartFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": m.target})},
		}}, nil)
	}
}

// setMaxLLMCalls sets cfg.MaxLLMCalls to n and reports whether the ADK version
// in use has that field, which was added after ADK v2.0.0.
func setMaxLLMCalls(cfg *agent.RunConfig, n int) bool {
	f := reflect.ValueOf(cfg).Elem().FieldByName("MaxLLMCalls")
	if !f.IsValid() || !f.CanSet() || f.Kind() != reflect.Int {
		return false
	}
	f.SetInt(int64(n))
	return true
}

// runOrdinaryBot runs an agent whose model answers at once through a runner
// with plugin p and returns the run's invocation ID.
func runOrdinaryBot(t *testing.T, p *baseplugin.Plugin) string {
	t.Helper()
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "ok"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(bot): %v", err)
	}
	var invID string
	for ev, err := range newTestRunner(t, bot, p).Run(t.Context(), "u1", "ordinary", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("ordinary run: %v", err)
		}
		if ev != nil && invID == "" {
			invID = ev.InvocationID
		}
	}
	return invID
}

// checkLateLookupFindsCompletedRun checks that the run of invID kept its
// identity at its completion, and that a callback after its AfterRun, such as
// OnEventCallback for the compaction summary event of the run, gets the run's
// own trace and invocation span without starting an active invocation.
func checkLateLookupFindsCompletedRun(t *testing.T, tm *traceManager, invID, author string) {
	t.Helper()
	tm.mu.Lock()
	completed := tm.completed[invID]
	var want evictedTrace
	if completed != nil {
		want = completed.identity
	}
	tm.mu.Unlock()
	if invID == "" || completed == nil {
		t.Fatalf("completed run %q lost its identity at its own completion", invID)
	}
	span := tm.eventSpan(context.Background(), identity{invocationID: invID, agentName: author}, author, "")
	tm.mu.Lock()
	_, active := tm.invocations[invID]
	tm.mu.Unlock()
	if active || span.traceID != want.traceID || span.spanID != want.spanID {
		t.Errorf("late event place = (trace %q, span %q), active invocation started = %v; want the completed run's (trace %q, span %q) and none started",
			span.traceID, span.spanID, active, want.traceID, want.spanID)
	}
}

// TestRunner_CallsWithoutCallbacksHoldNoCompletedRuns verifies through
// runner.Run that runs whose model calls ADK leaves without any model
// callback once this plugin's BeforeModelCallback started them (because the
// BeforeModelCallback of an agent, also of one that a sequential, parallel or
// workflow agent, a custom agent or a transfer runs, or of a later plugin
// returns an error, or the model-call budget runs out), as well as runs in
// which a later plugin answers a model or tool call instead, keep no completed
// run waiting for late calls, whether the caller reads each run to its end or
// stops reading at its first error: after a capacity's worth of them, an
// ordinary run keeps its identity at its completion, and a callback after its
// AfterRun finds it.
func TestRunner_CallsWithoutCallbacksHoldNoCompletedRuns(t *testing.T) {
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks something up"},
		func(_ agent.Context, _ struct{}) (struct{}, error) { return struct{}{}, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	refuse := func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
		return nil, errors.New("blocked by guardrail")
	}
	newAgent := func(t *testing.T, cfg llmagent.Config) agent.Agent {
		t.Helper()
		a, err := llmagent.New(cfg)
		if err != nil {
			t.Fatalf("llmagent.New(%s): %v", cfg.Name, err)
		}
		return a
	}
	newGuarded := func(t *testing.T) agent.Agent {
		t.Helper()
		return newAgent(t, llmagent.Config{Name: "guarded", Model: &toolThenTextLLM{toolName: "lookup"}, BeforeModelCallbacks: []llmagent.BeforeModelCallback{refuse}})
	}
	newCustom := func(t *testing.T, run func(ctx agent.InvocationContext, guarded agent.Agent, yield func(*session.Event, error) bool)) agent.Agent {
		t.Helper()
		guarded := newGuarded(t)
		custom, err := agent.New(agent.Config{
			Name:      "custom",
			SubAgents: []agent.Agent{guarded},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { run(ctx, guarded, yield) }
			},
		})
		if err != nil {
			t.Fatalf("agent.New(custom): %v", err)
		}
		return custom
	}
	for _, tc := range []struct {
		name        string
		root        func(t *testing.T) agent.Agent
		later       baseplugin.Config
		maxLLMCalls int
	}{
		{
			name: "agent's BeforeModelCallback returns an error",
			root: newGuarded,
		},
		{
			name: "agent's BeforeModelCallback in a sub-agent of a sequential agent returns an error",
			root: func(t *testing.T) agent.Agent {
				seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{newGuarded(t)}}})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
				return seq
			},
		},
		{
			name: "agent's BeforeModelCallback in a branch of a parallel agent returns an error",
			root: func(t *testing.T) agent.Agent {
				par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "fanout", SubAgents: []agent.Agent{newGuarded(t)}}})
				if err != nil {
					t.Fatalf("parallelagent.New: %v", err)
				}
				return par
			},
		},
		{
			name: "agent's BeforeModelCallback in a node of a workflow agent returns an error",
			root: func(t *testing.T) agent.Agent {
				node, err := workflow.NewAgentNode(newGuarded(t), workflow.NodeConfig{})
				if err != nil {
					t.Fatalf("workflow.NewAgentNode: %v", err)
				}
				wf, err := workflowagent.New(workflowagent.Config{Name: "graph", Edges: []workflow.Edge{{From: workflow.Start, To: node}}})
				if err != nil {
					t.Fatalf("workflowagent.New: %v", err)
				}
				return wf
			},
		},
		{
			name: "agent's BeforeModelCallback in a sub-agent that a custom agent runs in line returns an error",
			root: func(t *testing.T) agent.Agent {
				return newCustom(t, func(ctx agent.InvocationContext, guarded agent.Agent, yield func(*session.Event, error) bool) {
					for ev, err := range guarded.Run(ctx) {
						if !yield(ev, err) {
							return
						}
					}
				})
			},
		},
		{
			name: "agent's BeforeModelCallback in a sub-agent that a custom agent runs on a goroutine and waits for returns an error",
			root: func(t *testing.T) agent.Agent {
				return newCustom(t, func(ctx agent.InvocationContext, guarded agent.Agent, yield func(*session.Event, error) bool) {
					type result struct {
						ev  *session.Event
						err error
					}
					results := make(chan result)
					go func() {
						defer close(results)
						for ev, err := range guarded.Run(ctx) {
							results <- result{ev: ev, err: err}
						}
					}()
					reading := true
					for res := range results {
						if reading {
							reading = yield(res.ev, res.err)
						}
					}
				})
			},
		},
		{
			name: "agent's BeforeModelCallback in an agent that the root agent transfers to returns an error",
			root: func(t *testing.T) agent.Agent {
				return newAgent(t, llmagent.Config{Name: "coordinator", Model: &transferToLLM{target: "guarded"}, SubAgents: []agent.Agent{newGuarded(t)}})
			},
		},
		{
			name: "later plugin's BeforeModelCallback returns an error",
			root: func(t *testing.T) agent.Agent {
				return newAgent(t, llmagent.Config{Name: "guarded", Model: &toolThenTextLLM{toolName: "lookup"}})
			},
			later: baseplugin.Config{Name: "guardrail", BeforeModelCallback: refuse},
		},
		{
			name: "later plugin's BeforeModelCallback returns a response",
			root: func(t *testing.T) agent.Agent {
				return newAgent(t, llmagent.Config{Name: "cached", Model: &toolThenTextLLM{toolName: "lookup"}})
			},
			later: baseplugin.Config{Name: "cache", BeforeModelCallback: func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return &model.LLMResponse{Content: textContent("model", "cached"), TurnComplete: true}, nil
			}},
		},
		{
			name: "model-call budget runs out",
			root: func(t *testing.T) agent.Agent {
				return newAgent(t, llmagent.Config{Name: "budgeted", Model: &toolThenTextLLM{toolName: "lookup"}, Tools: []tool.Tool{lookup}})
			},
			maxLLMCalls: 1,
		},
		{
			name: "later plugin's BeforeToolCallback returns an error",
			root: func(t *testing.T) agent.Agent {
				return newAgent(t, llmagent.Config{Name: "tooled", Model: &toolThenTextLLM{toolName: "lookup"}, Tools: []tool.Tool{lookup}})
			},
			later: baseplugin.Config{Name: "tool-guardrail", BeforeToolCallback: func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return nil, errors.New("tool blocked")
			}},
		},
	} {
		for _, mode := range []struct {
			name        string
			stopAtError bool
		}{
			{name: "caller reads to the end"},
			{name: "caller stops at the first error", stopAtError: true},
		} {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				var cfg agent.RunConfig
				if tc.maxLLMCalls != 0 && !setMaxLLMCalls(&cfg, tc.maxLLMCalls) {
					t.Skip("agent.RunConfig.MaxLLMCalls was added after ADK v2.0.0")
				}
				tm := newTraceManager()
				tm.maxTracked = 1
				limit := max(tm.maxTracked, 64)
				p := newTraceOnlyToolPlugin(t, tm)
				plugins := []*baseplugin.Plugin{p}
				if tc.later.Name != "" {
					later, err := baseplugin.New(tc.later)
					if err != nil {
						t.Fatalf("baseplugin.New(%s): %v", tc.later.Name, err)
					}
					plugins = append(plugins, later)
				}
				r := newTestRunner(t, tc.root(t), plugins...)
				for i := range limit + 6 {
					for _, err := range r.Run(t.Context(), "u1", fmt.Sprintf("s%d", i), textContent("user", "hi"), cfg) {
						if err != nil && mode.stopAtError {
							break
						}
					}
				}
				tm.mu.Lock()
				waiting := 0
				for _, c := range tm.completed {
					if c.hasPendingOrOpenLateSpans() {
						waiting++
					}
				}
				tm.mu.Unlock()
				if waiting != 0 {
					t.Errorf("completed runs waiting for late calls = %d, want 0", waiting)
				}
				checkLateLookupFindsCompletedRun(t, tm, runOrdinaryBot(t, p), "bot")
			})
		}
	}
}

// TestRunner_UnjoinedSubAgentCallOutlivesAfterRun verifies through runner.Run
// that the model call of a sub-agent that a custom agent, or a function node
// of a workflow agent, runs on a goroutine that it does not wait for, which
// may get its callbacks after AfterRun, moves to the late state, where its
// AfterModelCallback ends it without starting an active invocation: neither
// the custom agent nor the workflow agent waits for that goroutine.
func TestRunner_UnjoinedSubAgentCallOutlivesAfterRun(t *testing.T) {
	for _, tc := range []struct {
		name string
		// root returns the agent that the runner runs, which calls start and
		// then returns once wait returns, without waiting for the run that
		// start begins.
		root func(t *testing.T, sub agent.Agent, start func(agent.InvocationContext), wait func()) agent.Agent
	}{
		{
			name: "custom agent",
			root: func(t *testing.T, sub agent.Agent, start func(agent.InvocationContext), wait func()) agent.Agent {
				root, err := agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{sub},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {
							start(ctx)
							wait()
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}
				return root
			},
		},
		{
			name: "function node of a workflow agent",
			root: func(t *testing.T, _ agent.Agent, start func(agent.InvocationContext), wait func()) agent.Agent {
				fan := workflow.NewFunctionNode("fan", func(ctx agent.Context, _ any) (string, error) {
					start(ctx)
					wait()
					return "started", nil
				}, workflow.NodeConfig{})
				root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, fan)})
				if err != nil {
					t.Fatalf("workflowagent.New(root): %v", err)
				}
				return root
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			p := newTraceOnlyPlugin(t, tm)
			gate := &blockingLLM{entered: make(chan struct{}, 1), release: make(chan struct{})}
			sub, err := llmagent.New(llmagent.Config{Name: "sub", Model: gate})
			if err != nil {
				t.Fatalf("llmagent.New(sub): %v", err)
			}
			done := make(chan struct{})
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(gate.release) }) }
			t.Cleanup(func() {
				release()
				select {
				case <-done:
				case <-time.After(10 * time.Second):
					t.Error("the unjoined sub-agent never finished")
				}
			})
			start := func(ctx agent.InvocationContext) {
				go func() {
					defer close(done)
					for range sub.Run(ctx) {
					}
				}()
			}
			// Return once the model call started, without waiting for it.
			wait := func() { <-gate.entered }
			var invID string
			for ev := range newTestRunner(t, tc.root(t, sub, start, wait), p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
				if ev != nil && invID == "" {
					invID = ev.InvocationID
				}
			}
			tm.mu.Lock()
			if invID == "" && tm.completionOrder.Back() != nil {
				invID, _ = tm.completionOrder.Back().Value.(string)
			}
			completed := tm.completed[invID]
			tm.mu.Unlock()
			if completed == nil || completed.late == nil || !completed.late.hasOpenCalls() {
				t.Fatalf("completed run %q handed no open model call to the late state", invID)
			}

			release()
			select {
			case <-done:
			case <-time.After(10 * time.Second):
				t.Fatal("the unjoined sub-agent never finished")
			}
			tm.mu.Lock()
			_, active := tm.invocations[invID]
			tm.mu.Unlock()
			if completed.late.hasOpenCalls() || active {
				t.Errorf("after the unjoined model call ended: call still open in the late state = %v, active invocation started = %v; want false, false",
					completed.late.hasOpenCalls(), active)
			}
		})
	}
}

func TestRepro_ParallelWorkerGuardrailPinsNothing(t *testing.T) {
	// The items run one ModeSingleTurn agent concurrently.
	requireADKRaceFreeSharedSingleTurnTool(t)
	tm := newTraceManager()
	tm.maxTracked = 1
	limit := max(tm.maxTracked, 64)
	p := newTraceOnlyPlugin(t, tm)
	guarded, err := llmagent.New(llmagent.Config{
		Name:  "guarded",
		Model: &scriptedLLM{},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, errors.New("blocked by guardrail")
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	items := workflow.NewFunctionNode("items", func(agent.Context, any) ([]string, error) {
		return []string{"a", "b"}, nil
	}, workflow.NodeConfig{})
	node, err := workflow.NewAgentNode(guarded, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	perItem, err := workflow.NewParallelWorker("per_item", node, 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	root, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, items, perItem)})
	if err != nil {
		t.Fatal(err)
	}
	r := newTestRunner(t, root, p)
	errs := 0
	for i := range limit + 6 {
		for _, err := range r.Run(t.Context(), "u1", fmt.Sprintf("s%d", i), textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				errs++
			}
		}
	}
	tm.mu.Lock()
	pinned := 0
	for _, c := range tm.completed {
		if c.late != nil && c.late.hasOpenCalls() {
			pinned++
		}
	}
	tm.mu.Unlock()
	t.Logf("runs=%d errors=%d pinned=%d", limit+6, errs, pinned)
	if errs == 0 {
		t.Fatal("no run yielded an error: the guardrail never ran")
	}
	bot, err := llmagent.New(llmagent.Config{Name: "bot", Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "ok"), TurnComplete: true}}}})
	if err != nil {
		t.Fatal(err)
	}
	var invID string
	for ev, err := range newTestRunner(t, bot, p).Run(t.Context(), "u1", "ordinary", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev != nil && invID == "" {
			invID = ev.InvocationID
		}
	}
	tm.mu.Lock()
	kept := tm.completed[invID] != nil
	tm.mu.Unlock()
	t.Logf("ordinary run identity kept = %v", kept)
	if pinned > 0 || !kept {
		t.Errorf("runs pinned by calls with no callback = %d, want 0; ordinary identity kept = %v, want true", pinned, kept)
	}
}

type gatedLLM struct {
	entered chan struct{}
	release chan struct{}
}

func (m *gatedLLM) Name() string { return "gated" }

func (m *gatedLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		close(m.entered)
		select {
		case <-m.release:
		case <-time.After(10 * time.Second):
		}
		yield(&model.LLMResponse{Content: textContent("model", "late"), TurnComplete: true}, nil)
	}
}

type waitLLM struct{ wait <-chan struct{} }

func (m *waitLLM) Name() string { return "wait" }

func (m *waitLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		<-m.wait
		yield(&model.LLMResponse{Content: textContent("model", "b"), TurnComplete: true}, nil)
	}
}

// A custom agent starts its sub-agent on a goroutine it does not wait for and
// returns before the sub-agent's model call starts. That call can still get
// its AfterModelCallback after AfterRun, so it should move to the late state.
func TestRepro_UnjoinedCallStartedAfterLauncherReturnedMovesToLate(t *testing.T) {
	tm := newTraceManager()
	p := newTraceOnlyPlugin(t, tm)
	sStarted, cReturned := make(chan struct{}), make(chan struct{})
	sModel := &gatedLLM{entered: make(chan struct{}), release: make(chan struct{})}
	s, err := llmagent.New(llmagent.Config{
		Name:  "s",
		Model: sModel,
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(agent.Context) (*genai.Content, error) {
			close(sStarted)
			<-cReturned
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	sDone := make(chan struct{})
	c, err := agent.New(agent.Config{
		Name:      "c",
		SubAgents: []agent.Agent{s},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				go func() {
					defer close(sDone)
					for range s.Run(ctx) {
					}
				}()
				<-sStarted
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	b, err := llmagent.New(llmagent.Config{Name: "b", Model: &waitLLM{wait: sModel.entered}})
	if err != nil {
		t.Fatal(err)
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{c, b}}})
	if err != nil {
		t.Fatal(err)
	}
	signal, err := baseplugin.New(baseplugin.Config{
		Name: "signal",
		AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			if ctx.AgentName() == "c" {
				close(cReturned)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := newTestRunner(t, pipeline, p, signal)
	var invID string
	for ev, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if ev != nil && invID == "" {
			invID = ev.InvocationID
		}
	}
	tm.mu.Lock()
	cmp := tm.completed[invID]
	moved := 0
	if cmp != nil && cmp.late != nil {
		cmp.late.mu.Lock()
		moved = len(cmp.late.llmSpans)
		cmp.late.mu.Unlock()
	}
	tm.mu.Unlock()
	close(sModel.release)
	<-sDone
	t.Logf("open model calls moved to late state = %d", moved)
	if moved != 1 {
		t.Errorf("s's model call, still running at AfterRun, moved to late state = %d, want 1", moved)
	}
}

// TestRunner_JoinedChildStoppedAtErrorKeepsCompletedIdentities verifies
// through runner.Run that runs that end with a model call that no callback
// ends, and that the callbacks do not show as over, cost the runs that follow
// them neither their completed identity nor their late callbacks. A custom
// agent runs its sub-agent on a goroutine of its own, stops reading it at the
// error that the sub-agent's BeforeModelCallback returns after this plugin's
// started the model call, and waits for the goroutine before it returns: the
// callbacks show neither the end of the sub-agent's run nor that the custom
// agent waited for it, as for a sub-agent whose call does end after AfterRun
// (TestRunner_UnjoinedSubAgentCallOutlivesAfterRun). Such runs go first
// while more of them than half of the capacity remain, so each of the
// ordinary runs that follow keeps its identity, and an event of it after
// AfterRun, as of compaction, finds that identity instead of starting a new
// invocation; after half of the capacity's worth of ordinary runs, such runs
// hold at most half of it.
func TestRunner_JoinedChildStoppedAtErrorKeepsCompletedIdentities(t *testing.T) {
	tm := newTraceManager()
	tm.maxTracked = 1
	limit := max(tm.maxTracked, 64)
	p := newTraceOnlyPlugin(t, tm)
	child, err := llmagent.New(llmagent.Config{
		Name:  "child",
		Model: &scriptedLLM{},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, errors.New("blocked by guardrail")
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(child): %v", err)
	}
	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var childErr error
				done := make(chan struct{})
				go func() {
					defer close(done)
					for _, err := range child.Run(ctx) {
						if err != nil {
							childErr = err
							break
						}
					}
				}()
				<-done
				if childErr != nil {
					yield(nil, childErr)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent): %v", err)
	}
	r := newTestRunner(t, parent, p)
	errs := 0
	for i := range limit + 6 {
		for _, err := range r.Run(t.Context(), "u1", fmt.Sprintf("j%d", i), textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				errs++
			}
		}
	}
	if errs != limit+6 {
		t.Fatalf("runs that yielded the guardrail's error = %d, want %d", errs, limit+6)
	}

	answers := &scriptedLLM{}
	for range limit / 2 {
		answers.responses = append(answers.responses, &model.LLMResponse{Content: textContent("model", "ok"), TurnComplete: true})
	}
	bot, err := llmagent.New(llmagent.Config{Name: "bot", Model: answers})
	if err != nil {
		t.Fatalf("llmagent.New(bot): %v", err)
	}
	rb := newTestRunner(t, bot, p)
	var ordinary []string
	for i := range limit / 2 {
		var invID string
		for ev, err := range rb.Run(t.Context(), "u1", fmt.Sprintf("o%d", i), textContent("user", "hi"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("ordinary run %d: %v", i, err)
			}
			if ev != nil && invID == "" {
				invID = ev.InvocationID
			}
		}
		tm.mu.Lock()
		completed := tm.completed[invID]
		tm.mu.Unlock()
		if completed == nil {
			t.Fatalf("ordinary run %d lost its completed identity at its own completion", i)
		}
		late := tm.invocation(t.Context(), identity{invocationID: invID, agentName: "bot"})
		tm.mu.Lock()
		_, active := tm.invocations[invID]
		tm.mu.Unlock()
		if active || late.rootSpanID != completed.identity.spanID {
			t.Fatalf("event of ordinary run %d after AfterRun: new active invocation = %v, root span = %q, want false and the run's %q",
				i, active, late.rootSpanID, completed.identity.spanID)
		}
		ordinary = append(ordinary, invID)
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()
	waiting := 0
	for _, c := range tm.completed {
		if c.waitsForLateCalls() {
			waiting++
		}
	}
	if waiting > limit/2 {
		t.Errorf("completed runs that wait for late calls = %d, want at most %d", waiting, limit/2)
	}
	for i, invID := range ordinary {
		if tm.completed[invID] == nil {
			t.Errorf("ordinary run %d lost its completed identity to runs that wait for late calls", i)
		}
	}
}

// TestRunner_LiveToolSubAgentCallMovesToLateState verifies through
// runner.RunLive that the open model call of a sub-agent that a tool of a live
// run starts, on a goroutine that the tool does not wait for, moves to the
// late state when the run ends, so that its AfterModelCallback ends the call's
// own span. RunLive runs no agent callbacks for the agent that it runs, so the
// sub-agent's activation runs right under the invocation, where only an
// activation named like the agent that the runner runs is taken for the
// runner's own, whose run is over (see endedWithRun).
func TestRunner_LiveToolSubAgentCallMovesToLateState(t *testing.T) {
	tm := newTraceManager()
	p := newTraceOnlyPlugin(t, tm)
	helperModel := &gatedLLM{entered: make(chan struct{}), release: make(chan struct{})}
	helper, err := llmagent.New(llmagent.Config{Name: "helper", Model: helperModel})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	helperDone := make(chan struct{})
	startHelper, err := functiontool.New(functiontool.Config{Name: "start_helper", Description: "Starts helper without waiting for it"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			subPath := "helper@1"
			runCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath}).WithICDelta(nil)
			go func() {
				defer close(helperDone)
				for range helper.Run(runCtx) {
				}
			}()
			select {
			case <-helperModel.entered:
			case <-time.After(10 * time.Second):
				return nil, errors.New("helper never started its model call")
			}
			return map[string]any{"started": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(start_helper): %v", err)
	}
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL: newFakeLiveServer(t,
				`{"toolCall":{"functionCalls":[{"id":"call-1","name":"start_helper","args":{}}]}}`,
				`{"serverContent":{"modelTurn":{"parts":[{"text":"done"}],"role":"model"},"turnComplete":true}}`,
			),
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	toolDone := make(chan struct{})
	root, err := llmagent.New(llmagent.Config{
		Name:  "live_root",
		Model: &liveTestLLM{client: client},
		Tools: []tool.Tool{startHelper},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				close(toolDone)
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New(live_root): %v", err)
	}
	liveSess, events, err := newTestRunner(t, root, p).RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("r.RunLive: %v", err)
	}
	go func() {
		select {
		case <-toolDone:
		case <-time.After(10 * time.Second):
		}
		_ = liveSess.Close()
	}()
	for _, err := range events {
		if err != nil {
			t.Errorf("live event error: %v", err)
		}
	}
	tm.mu.Lock()
	runs, moved := len(tm.completed), 0
	for _, c := range tm.completed {
		if c.late != nil {
			c.late.mu.Lock()
			moved += len(c.late.llmSpans)
			c.late.mu.Unlock()
		}
	}
	tm.mu.Unlock()
	close(helperModel.release)
	select {
	case <-helperDone:
	case <-time.After(10 * time.Second):
		t.Fatal("helper run never finished")
	}
	if runs != 1 || moved != 1 {
		t.Errorf("completed runs = %d, open model calls moved to the late state = %d, want 1 and 1, the call of helper", runs, moved)
	}
}

// TestRunner_LiveToolReturnsBeforeHelperModelCallEnds_KeepsHelperOpen verifies
// through runner.RunLive, with the BigQuery plugin, that a Live function tool
// that starts a helper on a goroutine of its own and returns while the
// helper's model call is still in flight does not end the helper: the tool's
// return does not show that the helper's run is over, so neither its open call
// nor its own completion is a failure then. The helper's model call is
// released either by the root agent's AfterToolCallback, which also waits for
// the helper to return, so that the run completes with no AGENT_ERROR row and
// the helper's AGENT_COMPLETED row closes the span of its AGENT_STARTING row
// with a latency, or only after the live session closed, so that its callbacks
// run after AfterRun. Either way the helper's LLM_RESPONSE row closes the span
// of its LLM_REQUEST row with a latency, as the tool's TOOL_COMPLETED row and
// the invocation's end row do for their starting rows.
func TestRunner_LiveToolReturnsBeforeHelperModelCallEnds_KeepsHelperOpen(t *testing.T) {
	for _, joined := range []bool{false, true} {
		name := "helper released after the session closed"
		if joined {
			name = "helper released and joined before the session closed"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			gate := &blockingLLM{entered: make(chan struct{}, 1), release: make(chan struct{})}
			helper, err := llmagent.New(llmagent.Config{Name: "helper", Model: gate})
			if err != nil {
				t.Fatalf("llmagent.New(helper): %v", err)
			}
			helperDone := make(chan struct{})
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(gate.release) }) }
			t.Cleanup(release)
			waitHelper := func() error {
				select {
				case <-helperDone:
					return nil
				case <-time.After(10 * time.Second):
					return errors.New("helper never returned")
				}
			}
			startHelper, err := functiontool.New(functiontool.Config{Name: "start_helper", Description: "Starts helper without waiting for it"},
				func(ctx agent.Context, _ struct{}) (map[string]any, error) {
					subPath := "helper@1"
					runCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &subPath}).WithICDelta(nil)
					go func() {
						defer close(helperDone)
						for range helper.Run(runCtx) {
						}
					}()
					select {
					case <-gate.entered:
					case <-time.After(10 * time.Second):
						return nil, errors.New("helper never started its model call")
					}
					return map[string]any{"started": true}, nil
				})
			if err != nil {
				t.Fatalf("functiontool.New(start_helper): %v", err)
			}
			client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
				Backend: genai.BackendGeminiAPI,
				APIKey:  "test-api-key",
				HTTPOptions: genai.HTTPOptions{
					BaseURL: newFakeLiveServer(t,
						`{"toolCall":{"functionCalls":[{"id":"call-1","name":"start_helper","args":{}}]}}`,
						`{"serverContent":{"modelTurn":{"parts":[{"text":"done"}],"role":"model"},"turnComplete":true}}`,
					),
					APIVersion: "v1beta",
				},
			})
			if err != nil {
				t.Fatalf("genai.NewClient: %v", err)
			}
			toolDone := make(chan struct{})
			root, err := llmagent.New(llmagent.Config{
				Name:  "live_root",
				Model: &liveTestLLM{client: client},
				Tools: []tool.Tool{startHelper},
				AfterToolCallbacks: []llmagent.AfterToolCallback{
					func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
						defer close(toolDone)
						if joined {
							release()
							return nil, waitHelper()
						}
						return nil, nil
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(live_root): %v", err)
			}
			liveSess, events, err := newTestRunner(t, root, p).RunLive(t.Context(), "u1", "s1", agent.LiveRunConfig{})
			if err != nil {
				t.Fatalf("r.RunLive: %v", err)
			}
			go func() {
				select {
				case <-toolDone:
				case <-time.After(10 * time.Second):
				}
				_ = liveSess.Close()
			}()
			for _, err := range events {
				if err != nil {
					t.Errorf("live event error: %v", err)
				}
			}
			if !joined {
				release()
				if err := waitHelper(); err != nil {
					t.Fatal(err)
				}
			}
			rows := collectRows(t, requests, func(rows []map[string]any) bool {
				return untilInvocationEnd(rows) && len(agentRowsOfType(rows, EventLLMResponse, "helper")) > 0 &&
					(!joined || len(agentRowsOfType(rows, EventAgentCompleted, "helper")) > 0)
			})
			invocationEnds := rowsOfType(rows, EventInvocationCompleted)
			if joined {
				requireInvocationCompletedOK(t, rows)
			} else {
				// The helper still runs when the session closes, so AfterRun, as on
				// main, reports it with AGENT_ERROR and the run with
				// INVOCATION_ERROR: no callback shows whether it will end. The
				// tool's return did not end it before then.
				invocationEnds = append(invocationEnds, rowsOfType(rows, EventInvocationError)...)
				toolEnd := slices.IndexFunc(rows, func(row map[string]any) bool { return row["event_type"] == EventToolCompleted })
				if i := slices.IndexFunc(rows, func(row map[string]any) bool { return row["event_type"] == EventAgentError }); i >= 0 && i < toolEnd {
					t.Errorf("AGENT_ERROR row before the TOOL_COMPLETED row of the call that started the helper; event types: %v", eventTypes(rows))
				}
			}
			// requireClosedPair checks that the only end row of a kind closes the
			// span of the only start row of that kind with a latency.
			requireClosedPair := func(start, end []map[string]any, what string) {
				t.Helper()
				if len(start) != 1 || len(end) != 1 {
					t.Errorf("%s: %d start rows and %d end rows, want one of each; event types: %v", what, len(start), len(end), eventTypes(rows))
					return
				}
				if got, want := placeOf(end[0]), placeOf(start[0]); got != want {
					t.Errorf("%s: end row at %+v, want the place of its start row %+v", what, got, want)
				}
				mustLatency(t, end[0])
			}
			requireClosedPair(agentRowsOfType(rows, EventLLMRequest, "helper"), agentRowsOfType(rows, EventLLMResponse, "helper"), "helper model call")
			requireClosedPair(rowsOfType(rows, EventToolStarting), rowsOfType(rows, EventToolCompleted), "start_helper tool call")
			requireClosedPair(rowsOfType(rows, EventInvocationStarting), invocationEnds, "invocation")
			if joined {
				requireClosedPair(agentRowsOfType(rows, EventAgentStarting, "helper"), agentRowsOfType(rows, EventAgentCompleted, "helper"), "helper run")
			}
		})
	}
}

// TestTraceManager_ClosedHandoffThenLateCallIsNotPrunedAsWaiting verifies that
// a completed run waits for late calls only while a call that AfterRun handed
// over to its late state is open: once that call ended, a call that a callback
// starts after AfterRun does not make the run wait, so under capacity pressure
// the runs that still wait for handed-over calls are pruned before it, and the
// later call keeps its span and latency.
func TestTraceManager_ClosedHandoffThenLateCallIsNotPrunedAsWaiting(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	tm.maxTracked = 1
	limit := max(tm.maxTracked, 64)
	// endWithOpenTool runs id until its tool call key starts and then ends the
	// run, which hands the call over to the late state.
	endWithOpenTool := func(id identity, key string) {
		tm.invocationSpan(ctx, id, nil)
		tm.pushAgentSpan(ctx, id)
		tm.pushToolSpan(ctx, id, key)
		tm.endInvocation(ctx, id)
	}
	victim := identity{invocationID: "victim", agentName: "live"}
	handed := toolKey(victim, "handed", "lookup")
	endWithOpenTool(victim, handed)
	if end, _, _ := tm.popToolSpan(ctx, victim, handed, false); end.latency == nil {
		t.Fatalf("TOOL_COMPLETED of the handed-over call = %+v, want a latency", end)
	}
	later := toolKey(victim, "later", "lookup")
	start := tm.pushToolSpan(ctx, victim, later)
	for i := range limit - 1 {
		id := identity{invocationID: fmt.Sprintf("waiting-%d", i), agentName: "live"}
		endWithOpenTool(id, toolKey(id, "call", "lookup"))
	}
	idle := identity{invocationID: "idle", agentName: "root"}
	tm.invocationSpan(ctx, idle, nil)
	tm.endInvocation(ctx, idle)
	tm.mu.Lock()
	_, victimKept := tm.completed[victim.invocationID]
	_, firstWaitingKept := tm.completed["waiting-0"]
	tm.mu.Unlock()
	if !victimKept || firstWaitingKept {
		t.Errorf("after capacity was exceeded: run with only a post-AfterRun call kept = %v, least recently active run with a handed-over call kept = %v; want true and false", victimKept, firstWaitingKept)
	}
	if end, _, _ := tm.popToolSpan(ctx, victim, later, false); end.spanID != start.spanID || end.latency == nil {
		t.Errorf("TOOL_COMPLETED place = %+v, want the span %q of TOOL_STARTING with latency", end, start.spanID)
	}
}

// newQueuedRowsPlugin returns a plugin with the callbacks that the production
// plugin registers, whose trace manager is tm and whose rows wait in the queue
// of a batch processor that never starts, until the returned function takes
// them.
func newQueuedRowsPlugin(t *testing.T, tm *traceManager) (*baseplugin.Plugin, func() []map[string]any) {
	t.Helper()
	config := applyDefaults(DefaultConfig())
	config.DisableFlushOnRunEnd = true
	processor, err := NewBatchProcessor(t.Context(), nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(func() {
		// Close writes the rows that are left, which needs a write client.
		processor.drainQueue()
		processor.Close()
	})
	ap := &analyticsPlugin{config: config, processor: processor, traces: tm}
	p, err := baseplugin.New(baseplugin.Config{
		Name:                  "queued-rows",
		OnUserMessageCallback: ap.onUserMessage,
		BeforeRunCallback:     ap.beforeRun,
		AfterRunCallback:      ap.afterRun,
		OnEventCallback:       ap.onEvent,
		BeforeAgentCallback:   ap.beforeAgent,
		AfterAgentCallback:    ap.afterAgent,
		BeforeModelCallback:   ap.beforeModel,
		AfterModelCallback:    ap.afterModel,
		OnModelErrorCallback:  ap.onModelError,
		BeforeToolCallback:    ap.beforeTool,
		AfterToolCallback:     ap.afterTool,
		OnToolErrorCallback:   ap.onToolError,
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	return p, processor.drainQueue
}

// clockReadBy reports whether the traceManager method that reads the clock in
// the caller's stack, the innermost one, is the method name.
func clockReadBy(name string) bool {
	pcs := make([]uintptr, 32)
	frames := runtime.CallersFrames(pcs[:runtime.Callers(2, pcs)])
	for {
		frame, more := frames.Next()
		if _, method, ok := strings.Cut(frame.Function, ".(*traceManager)."); ok {
			return method == name
		}
		if !more {
			return false
		}
	}
}

// TestRunner_AgentStartingAcrossAfterRunStartsInLateState verifies through
// runner.Run that an agent that a custom agent starts on a goroutine that it
// does not wait for, and whose start looked the invocation up before AfterRun
// ended the run but adds its span only after that, starts in the late state,
// where its later callbacks find it: its AGENT_COMPLETED row closes the span of
// its AGENT_STARTING row with a latency, and its model call is under it.
func TestRunner_AgentStartingAcrossAfterRunStartsInLateState(t *testing.T) {
	tm := newTraceManager()
	p, takeRows := newQueuedRowsPlugin(t, tm)
	var childGoroutine atomic.Uint64
	held, release := make(chan struct{}), make(chan struct{})
	var holdOnce, releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	tm.now = func() time.Time {
		// pushAgentSpan reads the clock once it looked the invocation up.
		if g := childGoroutine.Load(); g != 0 && g == currentGoroutineID() && clockReadBy("pushAgentSpan") {
			holdOnce.Do(func() {
				close(held)
				<-release
			})
		}
		return time.Now()
	}
	child, err := llmagent.New(llmagent.Config{
		Name:  "child",
		Model: &scriptedLLM{responses: []*model.LLMResponse{{Content: textContent("model", "late"), TurnComplete: true}}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(child): %v", err)
	}
	childDone := make(chan struct{})
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				go func() {
					defer close(childDone)
					childGoroutine.Store(currentGoroutineID())
					for range child.Run(ctx) {
					}
				}()
				select {
				case <-held:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("the child's start never looked the invocation up"))
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "root"
				ev.Content = textContent("model", "started")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	unblock()
	select {
	case <-childDone:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never returned")
	}
	rows := takeRows()
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 1 {
		t.Fatalf("INVOCATION_COMPLETED rows = %d, want 1; event types: %v", n, eventTypes(rows))
	}
	starts, ends := agentRowsOfType(rows, EventAgentStarting, "child"), agentRowsOfType(rows, EventAgentCompleted, "child")
	if len(starts) != 1 || len(ends) != 1 {
		t.Fatalf("child rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want one of each; event types: %v", len(starts), len(ends), eventTypes(rows))
	}
	if got, want := placeOf(ends[0]), placeOf(starts[0]); got != want {
		t.Errorf("child AGENT_COMPLETED is at %+v, want the place of its AGENT_STARTING %+v", got, want)
	}
	mustLatency(t, ends[0])
	for _, row := range agentRowsOfType(rows, EventLLMRequest, "child") {
		if got, want := placeOf(row).parent, placeOf(starts[0]).span; got != want {
			t.Errorf("child LLM_REQUEST parent = %q, want the child's span %q", got, want)
		}
	}
}

// TestRunner_UnjoinedStoppedStreamLeavesNoLateState verifies through
// runner.Run the stopped stream for which only its partial event leaves the
// open model call behind at AfterRun: a custom agent reads a sub-agent on a
// goroutine that it does not wait for, forwards the sub-agent's partial event
// and stops reading the sub-agent, whose model call then never ends. The
// sub-agent's activation is not known to have finished running (see
// endedWithRun), so without the partial event the call would move to the late
// state and hold the completed run's slot.
func TestRunner_UnjoinedStoppedStreamLeavesNoLateState(t *testing.T) {
	tm := newTraceManager()
	p := newTraceOnlyPlugin(t, tm)
	sub, err := llmagent.New(llmagent.Config{Name: "sub", Model: &partialThenWaitLLM{chunkText: "hello "}})
	if err != nil {
		t.Fatalf("llmagent.New(sub): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{sub},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				partial := make(chan *session.Event, 1)
				go func() {
					defer close(partial)
					for ev, err := range sub.Run(ctx) {
						if err == nil && ev != nil && ev.Partial {
							partial <- ev
							return
						}
					}
				}()
				if ev, ok := <-partial; ok {
					yield(ev, nil)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	var invID string
	var sawPartial bool
	for ev, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	}) {
		if err != nil {
			t.Fatalf("runner.Run error: %v", err)
		}
		if ev != nil {
			invID = ev.InvocationID
			sawPartial = sawPartial || ev.Partial
		}
	}
	if !sawPartial {
		t.Fatal("the run yielded no partial event")
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	c := tm.completed[invID]
	if c == nil {
		t.Fatalf("run %q is not completed", invID)
	}
	if c.late != nil {
		t.Errorf("completed run kept late state (open calls %d) for a stream that its reader stopped reading", c.late.openCalls.Load())
	}
}

// TestRunner_ReportedNodeOutcomeBeforeDelayedResponse_CallInFlight_EmitsNoAgentError
// has the first run of the node's agent in
// TestRunner_ReportedNodeOutcomeBeforeDelayedResponse_EmitsNoAgentError write
// the metadata of the completion event of a failed node run, and then run a
// declared LLM helper whose model call, or whose tool call, stays in flight
// until the second run started. Neither the report nor the open call shows
// that the first run failed or is over, so starting the second run purges
// nothing: both runs and the helper return without an error, and no
// AGENT_ERROR row is written. The helper's call still ends on the span that
// started it.
func TestRunner_ReportedNodeOutcomeBeforeDelayedResponse_CallInFlight_EmitsNoAgentError(t *testing.T) {
	for _, tc := range []struct {
		name     string
		toolCall bool
	}{
		{name: "model call"},
		{name: "tool call", toolCall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			reported := make(chan struct{})
			secondStarted := make(chan struct{})
			firstDone := make(chan struct{})
			var calls, runs atomic.Int32
			helperConfig := llmagent.Config{
				Name:  "helper",
				Model: &blockingLLM{entered: make(chan struct{}, 1), release: secondStarted},
			}
			if tc.toolCall {
				wait, err := functiontool.New(functiontool.Config{Name: "wait", Description: "Waits for the second run of child"},
					func(agent.Context, struct{}) (map[string]any, error) {
						select {
						case <-secondStarted:
							return map[string]any{"waited": true}, nil
						case <-time.After(10 * time.Second):
							return nil, errors.New("the second run of child never started")
						}
					})
				if err != nil {
					t.Fatalf("functiontool.New(wait): %v", err)
				}
				helperConfig.Tools = []tool.Tool{wait}
				helperConfig.Model = &scriptedLLM{responses: []*model.LLMResponse{
					{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: "call-wait", Name: "wait", Args: map[string]any{}}},
					}}},
					{Content: textContent("model", "answer"), TurnComplete: true},
				}}
			}
			helper, err := llmagent.New(helperConfig)
			if err != nil {
				t.Fatalf("llmagent.New(helper): %v", err)
			}
			child, err := agent.New(agent.Config{
				Name:      "child",
				SubAgents: []agent.Agent{helper},
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						if calls.Add(1) == 2 {
							close(secondStarted)
						}
						return nil, nil
					},
				},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if runs.Add(1) != 1 {
							select {
							case <-firstDone:
							case <-time.After(10 * time.Second):
							}
							return
						}
						defer close(firstDone)
						path := contextPath(ctx)
						report := session.NewEvent(ctx, ctx.InvocationID())
						report.Author = "child"
						report.Branch = ctx.Branch()
						report.NodeInfo = &session.NodeInfo{Path: path}
						report.CustomMetadata = map[string]any{
							workflowBasePathPrefix + "root": path[:max(strings.LastIndexByte(path, '/'), 0)],
							workflowNodeOutcomeKey:          workflowNodeFailedOutcome,
						}
						if !yield(report, nil) {
							return
						}
						close(reported)
						for ev, err := range helper.Run(ctx) {
							if !yield(ev, err) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(child): %v", err)
			}
			x, err := workflow.NewAgentNode(child, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("workflow.NewAgentNode: %v", err)
			}
			quick := func(name string) workflow.Node {
				return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) { return name, nil }, workflow.NodeConfig{})
			}
			a, b := quick("a"), quick("b")
			c := workflow.NewFunctionNode("c", func(agent.Context, any) (string, error) {
				select {
				case <-reported:
					return "c", nil
				case <-time.After(10 * time.Second):
					return "", errors.New("the first run of child never reported")
				}
			}, workflow.NodeConfig{})
			j1, j2 := workflow.NewJoinNode("j1"), workflow.NewJoinNode("j2")
			root, err := workflowagent.New(workflowagent.Config{
				Name: "root",
				Edges: []workflow.Edge{
					{From: workflow.Start, To: a},
					{From: workflow.Start, To: b},
					{From: workflow.Start, To: c},
					{From: a, To: j1},
					{From: b, To: j1},
					{From: a, To: j2},
					{From: c, To: j2},
					{From: j1, To: x, Route: workflow.Default},
					{From: j2, To: x, Route: workflow.Default},
				},
			})
			if err != nil {
				t.Fatalf("workflowagent.New: %v", err)
			}
			for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Errorf("runner.Run error: %v", err)
				}
			}
			if got := calls.Load(); got != 2 {
				t.Fatalf("child started %d times, want 2", got)
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows for eventually successful runs, want 0; counts: %v; event types: %v", n, agentErrorCounts(rows), eventTypes(rows))
			}
			pairs := [][2]string{{EventAgentStarting, EventAgentCompleted}, {EventLLMRequest, EventLLMResponse}}
			if tc.toolCall {
				pairs = append(pairs, [2]string{EventToolStarting, EventToolCompleted})
			}
			for _, pair := range pairs {
				starts, ends := agentRowsOfType(rows, pair[0], "helper"), agentRowsOfType(rows, pair[1], "helper")
				if len(starts) == 0 || len(starts) != len(ends) {
					t.Errorf("helper rows: %d %s and %d %s, want as many of each, at least one; event types: %v", len(starts), pair[0], len(ends), pair[1], eventTypes(rows))
					continue
				}
				if got, want := placeOf(ends[0]), placeOf(starts[0]); got != want {
					t.Errorf("helper %s is at %+v, want the place of its %s %+v", pair[1], got, pair[0], want)
				}
			}
		})
	}
}

// TestRunner_FinishTaskWithReusedCallIDEndsTheRunOnTheEventPath runs one
// task-mode agent as a node on two node paths, left/task@1 and then
// right/task@1, through a custom parent that holds back the finish_task
// response of each run and forwards both, in that order, once both runs
// returned. The model gives the finish_task call of each run the same ID, as
// a provider that numbers calls in each request does, so the tool callbacks
// of both runs saw a call with that ID complete. The node path of each
// response names the run that wrote it, so the first response ends the span
// of the left run and the second the span of the right run.
func TestRunner_FinishTaskWithReusedCallIDEndsTheRunOnTheEventPath(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	task, err := llmagent.New(llmagent.Config{
		Name:  "task",
		Mode:  llmagent.ModeTask,
		Model: &finishTaskLLM{called: make(chan struct{})},
	})
	if err != nil {
		t.Fatalf("llmagent.New(task): %v", err)
	}
	paths := []string{"left/task@1", "right/task@1"}
	root, err := agent.New(agent.Config{
		Name: "root",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var held []*session.Event
				for _, path := range paths {
					nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{Path: &path})
					for ev, err := range llmagent.RunLLMAgentAsNode(task, nodeCtx, "do the task") {
						if err != nil {
							yield(nil, err)
							return
						}
						if ev.NodeInfo == nil {
							ev.NodeInfo = &session.NodeInfo{}
						}
						ev.NodeInfo.Path = path
						if hasTaskCompletedResponse(ev.Content) {
							held = append(held, ev)
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
				for _, ev := range held {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("runner.Run error: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	starts, ends := agentRowsOfType(rows, EventAgentStarting, "task"), agentRowsOfType(rows, EventAgentCompleted, "task")
	if len(starts) != len(paths) || len(ends) != len(paths) {
		t.Fatalf("task rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want %d of each; event types: %v", len(starts), len(ends), len(paths), eventTypes(rows))
	}
	for i, path := range paths {
		if got, want := placeOf(ends[i]).span, placeOf(starts[i]).span; got != want {
			t.Errorf("AGENT_COMPLETED row %d, for the finish_task response on %s, ends span %q, want %q, the span of the run on that path", i, path, got, want)
		}
	}
}

// TestFinishedTaskLocked_RequiresOneRunOnTheEventPath verifies that a
// finish_task call ID picks the run that saw the call complete only among the
// runs of the agent on the event's node path, as a provider can reuse a call
// ID in each run, and only when one run there saw it complete: among several,
// the call ID does not tell which one wrote the event.
func TestFinishedTaskLocked_RequiresOneRunOnTheEventPath(t *testing.T) {
	run := func(path string, calls ...string) *spanRecord {
		return &spanRecord{kind: spanKindAgent, name: "task", branch: "b", path: path, finishedTasks: calls}
	}
	for _, tc := range []struct {
		name  string
		path  string
		stack []*spanRecord
		want  int
	}{
		{name: "run on the event path", path: "left/task@1", stack: []*spanRecord{run("left/task@1", "call_0"), run("right/task@1", "call_0")}, want: 1},
		{name: "run on another path only", path: "left/task@1", stack: []*spanRecord{run("right/task@1", "call_0")}, want: -1},
		{name: "two runs on the event path", path: "left/task@1", stack: []*spanRecord{run("left/task@1", "call_0"), run("left/task@1", "call_0")}, want: -1},
		{name: "one of two runs on the event path saw the call", path: "left/task@1", stack: []*spanRecord{run("left/task@1", "call_0"), run("left/task@1", "call_1")}, want: 1},
		{name: "event without a node path", path: "", stack: []*spanRecord{run("left/task@1", "call_0")}, want: 1},
		{name: "run without a node path", path: "left/task@1", stack: []*spanRecord{run("", "call_0")}, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := &invocationTrace{agentStack: append([]*spanRecord{{kind: spanKindAgent, name: "root"}}, tc.stack...)}
			id := identity{agentName: "task", branch: "b", path: tc.path}
			if got := inv.finishedTaskLocked(id, []string{"call_0"}); got != tc.want {
				t.Errorf("finishedTaskLocked = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestFailuresLocked_OpenCallFailsWithoutOpenCallsOnlyARunThatIsOver verifies
// that without openCalls a model call that a run under an earlier node attempt
// still has open shows a failure only if that run is known to be over or a
// report shows that it ended with the attempt (over), as when a recorded
// failure decided the purge of the attempt, and that with openCalls it always
// does.
func TestFailuresLocked_OpenCallFailsWithoutOpenCallsOnlyARunThatIsOver(t *testing.T) {
	for _, tc := range []struct {
		name       string
		recorded   bool
		helperDone bool
		openCalls  bool
		over       []string
		want       []string
	}{
		{name: "nothing recorded", want: nil},
		{name: "nothing recorded with openCalls", openCalls: true, want: []string{"attempt", "helper"}},
		{name: "run known to be over", helperDone: true, want: []string{"attempt", "helper"}},
		{name: "recorded failure, run ended with the attempt", recorded: true, over: []string{"attempt", "helper"}, want: []string{"attempt", "helper"}},
		{name: "recorded failure, run not shown to have ended", recorded: true, over: []string{"attempt"}, want: []string{"attempt"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workflowRec := &spanRecord{spanID: "workflow", kind: spanKindAgent, name: "root"}
			attempt := &spanRecord{spanID: "attempt", parentSpanID: "workflow", parent: workflowRec, kind: spanKindAgent, name: "child"}
			// The helper responded before its open call, so only that call
			// can show a failure for it.
			helper := &spanRecord{spanID: "helper", parentSpanID: "attempt", parent: attempt, kind: spanKindAgent, name: "helper", hasResponse: true, runDone: tc.helperDone}
			inv := &invocationTrace{
				agentStack:        []*spanRecord{{spanID: "invocation", kind: spanKindInvocation}, workflowRec, attempt, helper},
				llmSpans:          map[string]*spanRecord{"helper": {spanID: "call", parentSpanID: "helper", kind: spanKindLLM}},
				unrecoveredErrors: map[string]recordedError{},
			}
			if tc.recorded {
				inv.unrecoveredErrors["attempt"] = recordedError{err: errors.New("model failed"), spanID: "attempt"}
			}
			var over map[string]bool
			for _, id := range tc.over {
				if over == nil {
					over = map[string]bool{}
				}
				over[id] = true
			}
			got := slices.Sorted(maps.Keys(inv.failuresLocked(2, false, over, tc.openCalls)))
			if !slices.Equal(got, tc.want) {
				t.Errorf("failed activations = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestCustomFanOutHandlesCanceledChild verifies that a custom agent that runs
// a sub-agent under a private context.WithCancelCause and handles its failure
// (as errgroup.WithContext does on the first error) ends INVOCATION_COMPLETED
// when the custom agent writes a fallback event of its own. The sub-agent yields
// a response event and then an error, the custom agent cancels the sub-context
// with that error, breaks out, and writes its own fallback answer. The sub-agent's
// span stays open until retireAgentLocked retires it when the custom parent ends;
// without the isParallelBranch guard in contextFailureCause, the sub-agent's
// promoted retired cause would make popUnclosedAgentSpans report
// INVOCATION_ERROR even though runner.Run yields no error and the docs say a
// custom agent outside a parallel branch that writes an event and then returns
// an error ends INVOCATION_COMPLETED.
func TestCustomFanOutHandlesCanceledChild(t *testing.T) {
	ctx := t.Context()

	a, err := agent.New(agent.Config{
		Name: "a",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author, ev.Branch = "a", ctx.Branch()
				ev.Content = textContent("model", "partial answer")
				if yield(ev, nil) {
					yield(nil, errors.New("errA"))
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(a): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name:      "fanout",
		SubAgents: []agent.Agent{a},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				gctx, cancel := context.WithCancelCause(ctx)
				defer cancel(nil)
				for ev, runErr := range a.Run(ctx.WithContext(gctx)) {
					if runErr != nil {
						cancel(runErr) // as errgroup.WithContext does on the first error
						break
					}
					if !yield(ev, nil) {
						return
					}
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author, ev.Branch = "fanout", ctx.Branch()
				ev.Content = textContent("model", "fallback answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(fanout): %v", err)
	}
	p, requests, _ := setupTestPlugin(t)
	for _, runErr := range newTestRunner(t, root, p).Run(ctx, "u1", "s1", textContent("user", "hello"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run yielded an error: %v", runErr)
		}
	}
	if got := len(rowsOfType(collectRows(t, requests, untilInvocationEnd), EventInvocationError)); got != 0 {
		t.Errorf("INVOCATION_ERROR rows = %d, want 0", got)
	}
}

// TestCustomParentPropagatesParallelBranchError verifies #1984: a custom agent
// that runs a ParallelAgent under context.WithCancelCause, cancels with a
// failing branch's error, and yields that error must end with INVOCATION_ERROR
// (and AGENT_ERROR). Climbing past the ParallelAgent to the custom parent used
// to drop the branch cause and leave INVOCATION_COMPLETED.
func TestCustomParentPropagatesParallelBranchError(t *testing.T) {
	ctx := t.Context()

	failer, err := agent.New(agent.Config{
		Name: "failer",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("errBranch"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failer): %v", err)
	}
	ok, err := agent.New(agent.Config{
		Name: "ok",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "ok", ic.Branch()
				ev.Content = textContent("model", "ok")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(ok): %v", err)
	}
	par := newParallelAgent(t, "par", failer, ok)
	root, err := agent.New(agent.Config{
		Name:      "custom",
		SubAgents: []agent.Agent{par},
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				gctx, cancel := context.WithCancelCause(ic)
				defer cancel(nil)
				for ev, runErr := range par.Run(ic.WithContext(gctx)) {
					if runErr != nil {
						cancel(runErr)
						yield(nil, runErr)
						return
					}
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(custom): %v", err)
	}
	p, requests, _ := setupTestPlugin(t)
	var sawRunErr bool
	for _, runErr := range newTestRunner(t, root, p).Run(ctx, "u1", "s1", textContent("user", "hello"), agent.RunConfig{}) {
		if runErr != nil {
			sawRunErr = true
		}
	}
	if !sawRunErr {
		t.Fatal("runner.Run yielded no error, want the branch failure")
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if got := len(rowsOfType(rows, EventInvocationError)); got != 1 {
		t.Errorf("INVOCATION_ERROR rows = %d, want 1; event types: %v", got, eventTypes(rows))
	}
	if got := len(rowsOfType(rows, EventInvocationCompleted)); got != 0 {
		t.Errorf("INVOCATION_COMPLETED rows = %d, want 0; event types: %v", got, eventTypes(rows))
	}
}

// TestCustomFanOutHandlesCanceledChildWithSecondSubAgent is a variant of
// TestCustomFanOutHandlesCanceledChild where the custom parent also runs a
// second sub-agent that succeeds after the first one fails, confirming that the
// custom parent's WithCancelCause cause remains suppressed when a subsequent
// sub-agent runs under the same parent.
func TestCustomFanOutHandlesCanceledChildWithSecondSubAgent(t *testing.T) {
	ctx := t.Context()

	failer, err := agent.New(agent.Config{
		Name: "failer",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "failer", ic.Branch()
				ev.Content = textContent("model", "partial answer")
				if yield(ev, nil) {
					yield(nil, errors.New("errFailer"))
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failer): %v", err)
	}
	succeeder, err := agent.New(agent.Config{
		Name: "succeeder",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "succeeder", ic.Branch()
				ev.LLMResponse = model.LLMResponse{Content: textContent("model", "success answer"), TurnComplete: true}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(succeeder): %v", err)
	}
	root, err := agent.New(agent.Config{
		Name:      "fanout",
		SubAgents: []agent.Agent{failer, succeeder},
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				gctx, cancel := context.WithCancelCause(ic)
				defer cancel(nil)
				for ev, runErr := range failer.Run(ic.WithContext(gctx)) {
					if runErr != nil {
						cancel(runErr) // as errgroup.WithContext does on the first error
						break
					}
					if !yield(ev, nil) {
						return
					}
				}
				for ev, runErr := range succeeder.Run(ic) {
					if runErr != nil {
						t.Fatalf("succeeder.Run yielded error: %v", runErr)
					}
					if !yield(ev, nil) {
						return
					}
				}
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "fanout", ic.Branch()
				ev.Content = textContent("model", "fallback answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(fanout): %v", err)
	}
	p, requests, _ := setupTestPlugin(t)
	for _, runErr := range newTestRunner(t, root, p).Run(ctx, "u1", "s1", textContent("user", "hello"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run yielded an error: %v", runErr)
		}
	}
	if got := len(rowsOfType(collectRows(t, requests, untilInvocationEnd), EventInvocationError)); got != 0 {
		t.Errorf("INVOCATION_ERROR rows = %d, want 0", got)
	}
}

// TestParallelAgentBranchFailureAfterContent_EmitsInvocationError verifies
// that when one branch of a parallelagent fails after emitting a non-partial
// content event while a sibling succeeds, contextFailureCause / retiredCauses
// preserves the failing branch's error on the ParallelAgent branch and the
// invocation ends with EventInvocationError carrying that error.
func TestParallelAgentBranchFailureAfterContent_EmitsInvocationError(t *testing.T) {
	ctx := t.Context()

	failer, err := agent.New(agent.Config{
		Name: "failer",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "failer", ic.Branch()
				ev.Content = textContent("model", "failer content")
				if yield(ev, nil) {
					yield(nil, errors.New("errFailer"))
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(failer): %v", err)
	}
	succeeder, err := agent.New(agent.Config{
		Name: "succeeder",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author, ev.Branch = "succeeder", ic.Branch()
				ev.LLMResponse = model.LLMResponse{Content: textContent("model", "success answer"), TurnComplete: true}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(succeeder): %v", err)
	}
	par, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "par_root",
			SubAgents: []agent.Agent{failer, succeeder},
		},
	})
	if err != nil {
		t.Fatalf("parallelagent.New(par_root): %v", err)
	}
	p, requests, _ := setupTestPlugin(t)
	r := newTestRunner(t, par, p)
	for range r.Run(ctx, "u1", "s1", textContent("user", "hello"), agent.RunConfig{}) {
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	invRow := onlyRowOfType(t, rows, EventInvocationError)
	if msg, _ := invRow["error_message"].(string); !strings.Contains(msg, "errFailer") {
		t.Errorf("INVOCATION_ERROR error_message = %q, want it to contain %q", msg, "errFailer")
	}
	// When the ADK release produces AGENT_ERROR rows for parallel branches, the
	// failing branch should carry errFailer; on older releases without that, the
	// INVOCATION_ERROR already carries it (see assertTerminalErrorRowsContain).
	for _, row := range rowsOfType(rows, EventAgentError) {
		if row["agent"] == "failer" {
			if msg, _ := row["error_message"].(string); !strings.Contains(msg, "errFailer") {
				t.Errorf("AGENT_ERROR(failer) error_message = %q, want it to contain %q", msg, "errFailer")
			}
		}
	}
}

// lateReviewEnded completes the run of invID with nothing open, so that the
// callbacks of the invocation that follow act on its late state.
func lateReviewEnded(tm *traceManager, invID string) {
	live := identity{invocationID: invID, agentName: "live_agent"}
	tm.invocationSpan(context.Background(), live, nil)
	tm.endInvocation(context.Background(), live)
}

// lateReviewTrace returns the late state of the completed run invID.
func lateReviewTrace(t *testing.T, tm *traceManager, invID string) *invocationTrace {
	t.Helper()
	tm.mu.Lock()
	defer tm.mu.Unlock()
	c := tm.completed[invID]
	if c == nil || c.late == nil {
		t.Fatalf("completed run %q has no late state", invID)
	}
	return c.late
}

// lateReviewStack returns the names of the activations on the late agentStack
// of the completed run invID.
func lateReviewStack(t *testing.T, tm *traceManager, invID string) []string {
	t.Helper()
	late := lateReviewTrace(t, tm, invID)
	late.mu.Lock()
	defer late.mu.Unlock()
	var names []string
	for _, rec := range late.agentStack[1:] {
		names = append(names, rec.name)
	}
	return names
}

// lateReviewAgent returns the record of the activation named name on the late
// agentStack of the completed run invID.
func lateReviewAgent(t *testing.T, tm *traceManager, invID, name string) *spanRecord {
	t.Helper()
	late := lateReviewTrace(t, tm, invID)
	late.mu.Lock()
	defer late.mu.Unlock()
	for _, rec := range late.agentStack[1:] {
		if rec.name == name {
			return rec
		}
	}
	t.Fatalf("no activation %q on the late agentStack of %q", name, invID)
	return nil
}

// checkLateRecordDropped fails t if rec, the record of an activation that
// nothing can end anymore, is still on the agentStack of late or keeps a
// context of its callbacks, the span that it ran under or a failure cause
// (see dropRetiredLateContextsLocked).
func checkLateRecordDropped(t *testing.T, late *invocationTrace, rec *spanRecord) {
	t.Helper()
	late.mu.Lock()
	defer late.mu.Unlock()
	if slices.Contains(late.agentStack, rec) {
		t.Errorf("activation %q is still on the late agentStack", rec.name)
	}
	if rec.canceled != nil {
		t.Errorf("retired late activation %q keeps the context of its callback", rec.name)
	}
	if rec.ended != nil || rec.activation != nil {
		t.Errorf("retired late activation %q keeps its InvocationContext", rec.name)
	}
	if rec.parent != nil {
		t.Errorf("retired late activation %q keeps the span that it ran under", rec.name)
	}
	if len(rec.retiredCauses) > 0 {
		t.Errorf("retired late activation %q keeps a failure cause (%v)", rec.name, rec.firstRetiredCause())
	}
}

// TestLateReviewChildOutlivesParent verifies that the AfterAgentCallback of a
// late sub-agent that outlives its parent, because its model call was still
// open when the parent ended, ends the sub-agent's own span, with its latency,
// and no other: a sibling that started after it keeps its span until its own
// AfterAgentCallback.
func TestLateReviewChildOutlivesParent(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name     string
		children []string
	}{
		{name: "sub-agent", children: []string{"x"}},
		{name: "sub-agent and a sibling after it", children: []string{"x", "y"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			const invID = "inv-llm"
			lateReviewEnded(tm, invID)
			p := identity{invocationID: invID, agentName: "p"}
			tm.pushAgentSpan(ctx, p)
			starts := make(map[string]rowSpan)
			for _, name := range tc.children {
				id := identity{invocationID: invID, agentName: name}
				starts[name], _ = tm.pushAgentSpan(ctx, id)
				tm.pushLLMSpan(ctx, id)
			}
			tm.popAgentSpan(ctx, p)
			for _, name := range tc.children {
				id := identity{invocationID: invID, agentName: name}
				tm.popLLMSpan(ctx, id, true)
				var end rowSpan
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Fatalf("AfterAgentCallback of %q panicked: %v", name, r)
						}
					}()
					end = tm.popAgentSpan(ctx, id)
				}()
				if end.spanID != starts[name].spanID {
					t.Errorf("AGENT_COMPLETED span of %q = %q, want the span of its AGENT_STARTING row %q", name, end.spanID, starts[name].spanID)
				}
				if len(end.latency) == 0 {
					t.Errorf("AGENT_COMPLETED row of %q has no latency", name)
				}
			}
			if got := lateReviewStack(t, tm, invID); len(got) != 0 {
				t.Errorf("late agents at the end = %v, want none", got)
			}
		})
	}
}

// TestLateReviewToolRunFailedGrandchild verifies that when a late tool call
// that ran an agent returns, the activations under that agent that failed are
// reported with the failure recorded or propagated for them, and that the
// agent that made the call, which nothing has ended yet, stays open with the
// context of its callbacks.
func TestLateReviewToolRunFailedGrandchild(t *testing.T) {
	ctx := context.Background()
	// checkOwnerOpen fails t unless o, the agent that made the call, is the
	// only late activation left and keeps the context of its callback and
	// the span that it runs under.
	checkOwnerOpen := func(t *testing.T, tm *traceManager, invID string) {
		t.Helper()
		if got := lateReviewStack(t, tm, invID); !slices.Equal(got, []string{"o"}) {
			t.Fatalf("late agents after the tool call returned = %v, want [o]", got)
		}
		o := lateReviewAgent(t, tm, invID, "o")
		late := lateReviewTrace(t, tm, invID)
		late.mu.Lock()
		defer late.mu.Unlock()
		if o.canceled == nil || o.parent == nil {
			t.Errorf("open late agent o keeps the context of its callback = %t and its parent = %t, want both", o.canceled != nil, o.parent != nil)
		}
	}

	t.Run("model error of a grandchild", func(t *testing.T) {
		tm := newTraceManager()
		const invID = "inv-tool-run"
		lateReviewEnded(tm, invID)
		o := identity{invocationID: invID, agentName: "o"}
		g := identity{invocationID: invID, agentName: "g", path: "g"}
		s := identity{invocationID: invID, agentName: "s", path: "g/s"}
		c := identity{invocationID: invID, agentName: "c", path: "g/s/c"}
		call := toolKey(o, "call-1", "g")
		boom := errors.New("boom")
		tm.pushAgentSpan(ctx, o)
		tm.pushToolSpan(ctx, o, call)
		tm.pushAgentSpan(ctx, g)
		tm.pushAgentSpan(ctx, s)
		tm.pushAgentSpan(ctx, c)
		tm.pushLLMSpan(ctx, c)
		tm.recordError(ctx, c, llmKey(c), boom)
		tm.popLLMSpan(ctx, c, false)
		tm.popAgentSpan(ctx, s)
		cRec := lateReviewAgent(t, tm, invID, "c")
		gRec := lateReviewAgent(t, tm, invID, "g")
		_, purged, _ := tm.popToolSpan(ctx, o, call, false)
		errs := make(map[string]error)
		for _, p := range purged {
			errs[p.agentName] = p.err
		}
		if len(purged) != 2 || !errors.Is(errs["c"], boom) || !errors.Is(errs["g"], boom) {
			t.Errorf("AGENT_ERROR rows when the tool call returned = %v, want c and g, each with %v", errs, boom)
		}
		checkOwnerOpen(t, tm, invID)
		late := lateReviewTrace(t, tm, invID)
		checkLateRecordDropped(t, late, cRec)
		checkLateRecordDropped(t, late, gRec)
	})

	t.Run("failure cause of a sub-agent that ended", func(t *testing.T) {
		tm := newTraceManager()
		const invID = "inv-tool-cause"
		lateReviewEnded(tm, invID)
		o := identity{invocationID: invID, agentName: "o"}
		g := identity{invocationID: invID, agentName: "g", path: "g"}
		// In direct traceManager tests, an agent on a branch of its own
		// counts as a parallel branch (see isParallelBranch), whose context
		// failure moves up to the agent above it when it ends.
		c := identity{invocationID: invID, agentName: "c", path: "g/c", branch: "g.c"}
		call := toolKey(o, "call-1", "g")
		siblingErr := errors.New("sibling failed")
		tm.pushAgentSpan(ctx, o)
		tm.pushToolSpan(ctx, o, call)
		tm.pushAgentSpan(ctx, g)
		cctx, cancel := context.WithCancelCause(ctx)
		tm.pushAgentSpan(cctx, c)
		cancel(siblingErr)
		tm.popAgentSpan(cctx, c)
		gRec := lateReviewAgent(t, tm, invID, "g")
		_, purged, _ := tm.popToolSpan(ctx, o, call, false)
		if len(purged) != 1 || purged[0].agentName != "g" || !errors.Is(purged[0].err, siblingErr) {
			var got []string
			for _, p := range purged {
				got = append(got, fmt.Sprintf("%s: %v", p.agentName, p.err))
			}
			t.Errorf("AGENT_ERROR rows when the tool call returned = %q, want only g, with %v", got, siblingErr)
		}
		checkOwnerOpen(t, tm, invID)
		checkLateRecordDropped(t, lateReviewTrace(t, tm, invID), gRec)
	})
}

// TestLateReviewShortcutUnderInvocationSpan verifies that a shortcut sub-agent
// that starts right under the late invocation span, while a tool call that
// AfterRun handed over is open, stays until that call returns, which may have
// run it, and then leaves the late state, as nothing can end it anymore,
// keeping no context of its callbacks.
func TestLateReviewShortcutUnderInvocationSpan(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invID = "inv-root-shortcut"
	live := identity{invocationID: invID, agentName: "live_agent"}
	call := toolKey(live, "call-1", "t")
	tm.invocationSpan(ctx, live, nil)
	tm.pushAgentSpan(ctx, live)
	tm.pushToolSpan(ctx, live, call)
	tm.popAgentSpan(ctx, live)
	tm.endInvocation(ctx, live)
	sub := identity{invocationID: invID, agentName: "sub"}
	tm.pushAgentSpan(&endedContext{Context: ctx, end: true}, sub)
	subRec := lateReviewAgent(t, tm, invID, "sub")
	tm.popToolSpan(ctx, live, call, false)

	late := lateReviewTrace(t, tm, invID)
	if got := lateReviewStack(t, tm, invID); len(got) != 0 {
		t.Errorf("late agents after the tool call returned = %v, want none", got)
	}
	if late.openAgents.Load() {
		t.Error("late.openAgents = true after the tool call returned, want false")
	}
	checkLateRecordDropped(t, late, subRec)
}

// TestLateFailureCauseKeepsActivation verifies that a late activation that
// carries a failure cause stays in the late state, although its run is over,
// it ended its InvocationContext and its parent already ended, until the
// agent above it that is still open ends and takes the cause on: failure
// reporting reads such a cause (see failuresLocked).
func TestLateFailureCauseKeepsActivation(t *testing.T) {
	ctx := context.Background()
	tm := newTraceManager()
	const invID = "inv-late-cause"
	lateReviewEnded(tm, invID)
	o := identity{invocationID: invID, agentName: "o"}
	p := identity{invocationID: invID, agentName: "p", path: "p"}
	x := identity{invocationID: invID, agentName: "x", path: "p/x"}
	// c is a parallel branch, as in TestLateReviewToolRunFailedGrandchild.
	c := identity{invocationID: invID, agentName: "c", path: "p/x/c", branch: "x.c"}
	siblingErr := errors.New("sibling failed")
	tm.pushAgentSpan(ctx, o)
	tm.pushAgentSpan(ctx, p)
	xctx := &endedContext{Context: ctx, end: true}
	tm.pushAgentSpan(xctx, x)
	cctx, cancel := context.WithCancelCause(ctx)
	tm.pushAgentSpan(cctx, c)
	cancel(siblingErr)
	tm.popAgentSpan(cctx, c)
	// x's open model call keeps it open when p ends.
	tm.pushLLMSpan(xctx, x)
	tm.popAgentSpan(ctx, p)
	tm.popLLMSpan(xctx, x, true)
	if got := lateReviewStack(t, tm, invID); !slices.Equal(got, []string{"o", "x"}) {
		t.Fatalf("late agents after x's model call ended = %v, want [o x]", got)
	}
	xRec := lateReviewAgent(t, tm, invID, "x")
	late := lateReviewTrace(t, tm, invID)
	late.mu.Lock()
	settled, cause := xRec.runDone && xRec.contextEnded, xRec.openFailureCause()
	late.mu.Unlock()
	if !settled || !errors.Is(cause, siblingErr) {
		t.Fatalf("x settled = %v with failure cause %v, want true with %v", settled, cause, siblingErr)
	}

	tm.popAgentSpan(ctx, o)
	if got := lateReviewStack(t, tm, invID); len(got) != 0 {
		t.Errorf("late agents after o ended = %v, want none", got)
	}
	checkLateRecordDropped(t, late, xRec)
	late.mu.Lock()
	defer late.mu.Unlock()
	if got := late.agentStack[0].firstRetiredCause(); !errors.Is(got, siblingErr) {
		t.Errorf("failure cause that reached the invocation span = %v, want %v", got, siblingErr)
	}
}
