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

package liveflow

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// ---------------------------------------------------------------------------
// Mocks: LiveConnection / LiveCapableLLM / FunctionTool
// ---------------------------------------------------------------------------

// liveConnPlan describes the next move of a mockLiveConnTel.Receive call.
//   - resp non-nil: return resp, nil
//   - err  non-nil: return nil, err
//   - block       : block until ctx.Done() then return ctx.Err()
type liveConnPlan struct {
	resp  *model.LLMResponse
	err   error
	block bool
}

type mockLiveConnTel struct {
	mu       sync.Mutex
	plan     []liveConnPlan
	idx      int
	sendLog  []*model.LiveRequest
	closedCh chan struct{}
	once     sync.Once
}

func newMockLiveConnTel(plan ...liveConnPlan) *mockLiveConnTel {
	return &mockLiveConnTel{plan: plan, closedCh: make(chan struct{})}
}

func (m *mockLiveConnTel) Send(_ context.Context, req *model.LiveRequest) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sendLog = append(m.sendLog, req)
	return nil
}

func (m *mockLiveConnTel) Receive(ctx context.Context) (*model.LLMResponse, error) {
	m.mu.Lock()
	if m.idx >= len(m.plan) {
		m.mu.Unlock()
		// Plan exhausted: return EOF so the receiver loop unwinds naturally
		// (mirrors runner_live_test.go's mockLiveConnection.Receive).
		return nil, io.EOF
	}
	p := m.plan[m.idx]
	m.idx++
	m.mu.Unlock()
	if p.block {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if p.err != nil {
		return nil, p.err
	}
	return p.resp, nil
}

func (m *mockLiveConnTel) Close() error {
	m.once.Do(func() { close(m.closedCh) })
	return nil
}

// mockLiveLLMTel is a LiveCapableLLM that returns successive connections per
// connect call. Consumed in order; once the slice is exhausted, subsequent
// connects return an error (used by retry-failure tests).
type mockLiveLLMTel struct {
	mu      sync.Mutex
	conns   []model.LiveConnection
	connErr error
}

func newMockLiveLLMTel(conns ...model.LiveConnection) *mockLiveLLMTel {
	return &mockLiveLLMTel{conns: conns}
}

func (m *mockLiveLLMTel) Name() string { return "test-model" }

func (m *mockLiveLLMTel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (m *mockLiveLLMTel) ConnectLive(_ context.Context, _ *model.LLMRequest) (model.LiveConnection, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.connErr != nil {
		return nil, m.connErr
	}
	if len(m.conns) == 0 {
		return nil, errors.New("no more connections available")
	}
	c := m.conns[0]
	m.conns = m.conns[1:]
	return c, nil
}

var _ model.LiveCapableLLM = (*mockLiveLLMTel)(nil)

// telTool implements toolinternal.FunctionTool with a configurable runFn so
// tests can inspect the toolCtx (e.g., capture the parent span ID via
// trace.SpanFromContext) or return a Go error to exercise codes.Error parity.
type telTool struct {
	name        string
	description string
	runFn       func(ctx tool.Context, args map[string]any) (map[string]any, error)
	calls       atomic.Int32
}

func (t *telTool) Name() string        { return t.name }
func (t *telTool) Description() string { return t.description }
func (t *telTool) IsLongRunning() bool { return false }
func (t *telTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.name}
}
func (t *telTool) ProcessRequest(_ tool.Context, _ *model.LLMRequest) error { return nil }

func (t *telTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	t.calls.Add(1)
	if t.runFn == nil {
		return map[string]any{"ok": true}, nil
	}
	argsMap, _ := args.(map[string]any)
	return t.runFn(ctx, argsMap)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// driveLive constructs an agent that wraps LiveFlow.RunLive, drives it via
// agent.Run (which produces the production invoke_agent span), and returns
// the events / errors observed plus the recorded span tree.
//
// Using agent.Run is the closest approximation to runner.RunLive available
// from this package: runner imports llminternal, so we cannot import runner
// here. agent.Run gives us the production invoke_agent → custom Run boundary.
type driveOpts struct {
	tools          []tool.Tool
	coalesceWindow time.Duration
}

func driveLive(t *testing.T, llm *mockLiveLLMTel, opts driveOpts, queueClose bool) (*tracetest.InMemoryExporter, []*session.Event, []error) {
	t.Helper()
	setupTestTracer(t)

	parent := context.Background()

	resp, err := session.InMemoryService().Create(parent, &session.CreateRequest{
		AppName: "test-app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}

	lf := &LiveFlow{
		Model:          llm,
		Tools:          opts.tools,
		CoalesceWindow: opts.coalesceWindow,
	}

	queue := agent.NewLiveRequestQueue(64)
	if queueClose {
		queue.Close()
	}

	connectFn := func(handle string) (model.LiveConnection, error) {
		return llm.ConnectLive(parent, &model.LLMRequest{})
	}

	var ag agent.Agent
	ag, err = agent.New(agent.Config{
		Name:        "test-agent",
		Description: "live-telemetry test agent",
		Run: func(invCtx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return lf.RunLive(invCtx, connectFn, queue)
		},
	})
	if err != nil {
		t.Fatalf("agent create: %v", err)
	}

	invCtx := icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Session:          resp.Session,
		Agent:            ag,
		LiveRequestQueue: queue,
	})

	var events []*session.Event
	var errs []error
	for ev, err := range ag.Run(invCtx) {
		if err != nil {
			errs = append(errs, err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return testExporter, events, errs
}

// findSpansByName returns the spans (in finish order) whose Name matches.
func findSpansByName(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

// findSpansByPrefix returns spans whose Name has the given prefix.
func findSpansByPrefix(spans []sdktrace.ReadOnlySpan, prefix string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if strings.HasPrefix(s.Name(), prefix) {
			out = append(out, s)
		}
	}
	return out
}

// attrMap converts a span's attributes to a key→value-string map.
func attrMap(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	out := make(map[attribute.Key]attribute.Value, len(span.Attributes()))
	for _, kv := range span.Attributes() {
		out[kv.Key] = kv.Value
	}
	return out
}

// requireSpanByName fails the test if no span matches the given name.
func requireSpanByName(t *testing.T, spans []sdktrace.ReadOnlySpan, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	matches := findSpansByName(spans, name)
	if len(matches) == 0 {
		t.Fatalf("expected span %q, recorded %d spans:\n%s", name, len(spans), spanNames(spans))
	}
	if len(matches) > 1 {
		t.Fatalf("expected exactly 1 span %q, got %d", name, len(matches))
	}
	return matches[0]
}

func spanNames(spans []sdktrace.ReadOnlySpan) string {
	var b strings.Builder
	for _, s := range spans {
		fmt.Fprintf(&b, "  - %s\n", s.Name())
	}
	return b.String()
}

// textResp builds a model-content LLMResponse with the given text.
func textResp(textContent string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: genai.NewContentFromText(textContent, "model"),
	}
}

// fcResp builds a single-FunctionCall LLMResponse.
func fcResp(id, name string, args map[string]any) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args}},
			},
		},
	}
}

// multiFCResp builds an LLMResponse carrying multiple FunctionCalls in a
// single message — exercises the merged-tool-calls span path.
func multiFCResp(calls ...*genai.FunctionCall) *model.LLMResponse {
	parts := make([]*genai.Part, 0, len(calls))
	for _, fc := range calls {
		parts = append(parts, &genai.Part{FunctionCall: fc})
	}
	return &model.LLMResponse{
		Content: &genai.Content{Role: "model", Parts: parts},
	}
}

func goAwayResp() *model.LLMResponse {
	return &model.LLMResponse{GoAway: &genai.LiveServerGoAway{}}
}

// ---------------------------------------------------------------------------
// T1: Happy path — single attempt, no tools.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_HappyPath(t *testing.T) {
	conn := newMockLiveConnTel(
		liveConnPlan{resp: textResp("hello")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, events, errs := driveLive(t, llm, driveOpts{}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) == 0 {
		t.Fatalf("expected at least one event")
	}

	spans := exporter.GetSpans().Snapshots()
	invokeAgent := requireSpanByName(t, spans, "invoke_agent test-agent")
	liveSession := requireSpanByName(t, spans, "live_session test-agent")
	generateContent := requireSpanByName(t, spans, "generate_content test-model")

	// Span tree: invoke_agent → live_session → generate_content.
	if liveSession.Parent().SpanID() != invokeAgent.SpanContext().SpanID() {
		t.Errorf("live_session parent = %v, want invoke_agent %v",
			liveSession.Parent().SpanID(), invokeAgent.SpanContext().SpanID())
	}
	if generateContent.Parent().SpanID() != liveSession.SpanContext().SpanID() {
		t.Errorf("generate_content parent = %v, want live_session %v",
			generateContent.Parent().SpanID(), liveSession.SpanContext().SpanID())
	}

	// No execute_tool spans without tool calls.
	if got := len(findSpansByPrefix(spans, "execute_tool")); got != 0 {
		t.Errorf("expected no execute_tool spans, got %d", got)
	}

	// live_session attribute set.
	lsAttrs := attrMap(liveSession)
	if got := lsAttrs[telemetry.GCPVertexAgentOperationKey].AsString(); got != "live_session" {
		t.Errorf("live_session.gcp.vertex.agent.operation = %q, want %q", got, "live_session")
	}
	if got := lsAttrs[semconv.GenAIConversationIDKey].AsString(); got != "sess-1" {
		t.Errorf("live_session.gen_ai.conversation.id = %q, want %q", got, "sess-1")
	}

	// generate_content attribute set (caller-side conversation id + reconnect).
	gcAttrs := attrMap(generateContent)
	if got := gcAttrs[semconv.GenAIConversationIDKey].AsString(); got != "sess-1" {
		t.Errorf("generate_content.gen_ai.conversation.id = %q, want %q", got, "sess-1")
	}
	if got := gcAttrs[telemetry.GCPVertexAgentReconnectAttemptKey].AsInt64(); got != 0 {
		t.Errorf("generate_content.gcp.vertex.agent.reconnect_attempt = %d, want 0", got)
	}

	// Clean iterator close → live_session unset.
	if liveSession.Status().Code != codes.Unset {
		t.Errorf("live_session.Status = %v, want Unset", liveSession.Status().Code)
	}
}

// ---------------------------------------------------------------------------
// T2: Single tool call, capture flag OFF (default) — args & response elided.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_SingleToolCall_PHIDefault(t *testing.T) {
	greet := &telTool{
		name:        "greet",
		description: "greet a person by name",
		runFn: func(_ tool.Context, _ map[string]any) (map[string]any, error) {
			return map[string]any{"message": "hi PHI-RESPONSE"}, nil
		},
	}
	conn := newMockLiveConnTel(
		liveConnPlan{resp: fcResp("fc1", "greet", map[string]any{"name": "PHI-NAME"})},
		liveConnPlan{resp: textResp("done")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{tools: []tool.Tool{greet}}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if greet.calls.Load() != 1 {
		t.Fatalf("expected greet tool called once, got %d", greet.calls.Load())
	}

	spans := exporter.GetSpans().Snapshots()
	executeTool := requireSpanByName(t, spans, "execute_tool greet")
	generateContent := requireSpanByName(t, spans, "generate_content test-model")

	// execute_tool nested under generate_content.
	if executeTool.Parent().SpanID() != generateContent.SpanContext().SpanID() {
		t.Errorf("execute_tool parent = %v, want generate_content %v",
			executeTool.Parent().SpanID(), generateContent.SpanContext().SpanID())
	}

	// No merged span for a single tool call.
	if got := findSpansByName(spans, "execute_tool (merged)"); len(got) != 0 {
		t.Errorf("expected no merged span, got %d", len(got))
	}

	// PHI gating: args/response elided by default.
	etAttrs := attrMap(executeTool)
	args := etAttrs["gcp.vertex.agent.tool_call_args"].AsString()
	resp := etAttrs["gcp.vertex.agent.tool_response"].AsString()
	if args != "<elided>" {
		t.Errorf("tool_call_args = %q, want \"<elided>\"", args)
	}
	if resp != "<elided>" {
		t.Errorf("tool_response = %q, want \"<elided>\"", resp)
	}

	// PHI walk: ensure no attribute value carries the fixture text.
	for _, s := range spans {
		for _, kv := range s.Attributes() {
			val := kv.Value.AsString()
			if strings.Contains(val, "PHI-NAME") || strings.Contains(val, "PHI-RESPONSE") {
				t.Errorf("span %q leaked PHI in attribute %q: %q", s.Name(), kv.Key, val)
			}
		}
	}

	// tool name + call id still recorded (these are not PHI).
	if got := etAttrs[semconv.GenAIToolNameKey].AsString(); got != "greet" {
		t.Errorf("gen_ai.tool.name = %q, want %q", got, "greet")
	}
	if got := etAttrs[semconv.GenAIToolCallIDKey].AsString(); got != "fc1" {
		t.Errorf("gen_ai.tool.call.id = %q, want %q", got, "fc1")
	}
}

// ---------------------------------------------------------------------------
// T2b: Single tool call, capture flag ON — args & response serialized.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_SingleToolCall_PHIEnabled(t *testing.T) {
	telemetry.SetGenAICaptureMessageContent(true)
	t.Cleanup(func() { telemetry.SetGenAICaptureMessageContent(false) })

	greet := &telTool{
		name: "greet",
		runFn: func(_ tool.Context, _ map[string]any) (map[string]any, error) {
			return map[string]any{"message": "hello-CAPTURED"}, nil
		},
	}
	conn := newMockLiveConnTel(
		liveConnPlan{resp: fcResp("fc1", "greet", map[string]any{"name": "Alice-CAPTURED"})},
		liveConnPlan{resp: textResp("done")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{tools: []tool.Tool{greet}}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	spans := exporter.GetSpans().Snapshots()
	et := requireSpanByName(t, spans, "execute_tool greet")
	attrs := attrMap(et)

	if got := attrs["gcp.vertex.agent.tool_call_args"].AsString(); !strings.Contains(got, "Alice-CAPTURED") {
		t.Errorf("expected tool_call_args to contain captured args, got %q", got)
	}
	if got := attrs["gcp.vertex.agent.tool_response"].AsString(); !strings.Contains(got, "hello-CAPTURED") {
		t.Errorf("expected tool_response to contain captured result, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// T3: Multiple tool calls in one flush → merged span + 2 children.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_MergedToolCalls(t *testing.T) {
	t1 := &telTool{name: "t1"}
	t2 := &telTool{name: "t2"}

	conn := newMockLiveConnTel(
		liveConnPlan{resp: multiFCResp(
			&genai.FunctionCall{ID: "id1", Name: "t1", Args: map[string]any{"a": 1}},
			&genai.FunctionCall{ID: "id2", Name: "t2", Args: map[string]any{"b": 2}},
		)},
		liveConnPlan{resp: textResp("done")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{tools: []tool.Tool{t1, t2}}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	spans := exporter.GetSpans().Snapshots()
	merged := requireSpanByName(t, spans, "execute_tool (merged)")
	gc := requireSpanByName(t, spans, "generate_content test-model")
	if merged.Parent().SpanID() != gc.SpanContext().SpanID() {
		t.Errorf("merged span parent = %v, want generate_content %v",
			merged.Parent().SpanID(), gc.SpanContext().SpanID())
	}

	for _, name := range []string{"execute_tool t1", "execute_tool t2"} {
		s := requireSpanByName(t, spans, name)
		if s.Parent().SpanID() != merged.SpanContext().SpanID() {
			t.Errorf("%s parent = %v, want merged %v",
				name, s.Parent().SpanID(), merged.SpanContext().SpanID())
		}
	}

	// Merged response also gated on capture flag (off by default → elided).
	mAttrs := attrMap(merged)
	if got := mAttrs["gcp.vertex.agent.tool_response"].AsString(); got != "<elided>" {
		t.Errorf("merged.tool_response = %q, want \"<elided>\"", got)
	}
}

// ---------------------------------------------------------------------------
// T5: Tool returns Go error → execute_tool span Status = Error.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_ToolReturnsError(t *testing.T) {
	failing := &telTool{
		name: "boom_tool",
		runFn: func(_ tool.Context, _ map[string]any) (map[string]any, error) {
			return nil, errors.New("boom")
		},
	}
	conn := newMockLiveConnTel(
		liveConnPlan{resp: fcResp("fc1", "boom_tool", map[string]any{})},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{tools: []tool.Tool{failing}}, true)
	// The tool's Go error is packed into the FunctionResponse, not yielded as
	// an iterator error — the iterator error slice is expected to stay empty.
	_ = errs

	spans := exporter.GetSpans().Snapshots()
	et := requireSpanByName(t, spans, "execute_tool boom_tool")
	if et.Status().Code != codes.Error {
		t.Errorf("execute_tool.Status = %v, want Error", et.Status().Code)
	}
	if et.Status().Description != "boom" {
		t.Errorf("execute_tool.Status.Description = %q, want %q", et.Status().Description, "boom")
	}
}

// ---------------------------------------------------------------------------
// T6: GoAway reconnect → two sibling generate_content spans.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_ReconnectOnGoAway(t *testing.T) {
	withFastReconnectVars(t, 3, 5*time.Millisecond, 5*time.Millisecond)

	// First connection emits a SessionResumptionUpdate (so the outer loop has
	// a non-empty handle on reconnect, satisfying the resumption gate) and
	// then GoAway. Second connection completes normally.
	conn1 := newMockLiveConnTel(
		liveConnPlan{resp: &model.LLMResponse{
			SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{
				NewHandle: "handle-1",
				Resumable: true,
			},
		}},
		liveConnPlan{resp: goAwayResp()},
	)
	conn2 := newMockLiveConnTel(
		liveConnPlan{resp: textResp("after-reconnect")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn1, conn2)

	exporter, _, errs := driveLive(t, llm, driveOpts{}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	spans := exporter.GetSpans().Snapshots()
	gcSpans := findSpansByName(spans, "generate_content test-model")
	if len(gcSpans) != 2 {
		t.Fatalf("expected 2 generate_content spans, got %d", len(gcSpans))
	}
	ls := requireSpanByName(t, spans, "live_session test-agent")
	for i, s := range gcSpans {
		if s.Parent().SpanID() != ls.SpanContext().SpanID() {
			t.Errorf("generate_content[%d] parent = %v, want live_session %v",
				i, s.Parent().SpanID(), ls.SpanContext().SpanID())
		}
	}

	// reconnect_attempt: 0 and 1.
	seen := make(map[int64]bool)
	for _, s := range gcSpans {
		got := attrMap(s)[telemetry.GCPVertexAgentReconnectAttemptKey].AsInt64()
		seen[got] = true
	}
	if !seen[0] || !seen[1] {
		t.Errorf("expected reconnect_attempt {0,1}, got %v", seen)
	}
}

// ---------------------------------------------------------------------------
// T7: Connect failure exhausts retries → live_session error, no generate_content.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_ConnectFailureExhaustsRetries(t *testing.T) {
	withFastReconnectVars(t, 1, 1*time.Millisecond, 1*time.Millisecond)

	llm := &mockLiveLLMTel{connErr: errors.New("synthetic connect failure")}

	exporter, _, errs := driveLive(t, llm, driveOpts{}, true)
	if len(errs) == 0 {
		t.Fatalf("expected an error from the iterator")
	}
	if !strings.Contains(errs[len(errs)-1].Error(), "failed to connect live after") {
		t.Errorf("expected retry-exhaustion error, got %v", errs[len(errs)-1])
	}

	spans := exporter.GetSpans().Snapshots()
	ls := requireSpanByName(t, spans, "live_session test-agent")
	if ls.Status().Code != codes.Error {
		t.Errorf("live_session.Status = %v, want Error", ls.Status().Code)
	}
	if got := findSpansByName(spans, "generate_content test-model"); len(got) != 0 {
		t.Errorf("expected no generate_content spans, got %d", len(got))
	}
}

// ---------------------------------------------------------------------------
// T9: History send fails → generate_content + live_session error.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_HistorySendFails(t *testing.T) {
	setupTestTracer(t)

	parent := context.Background()
	svc := session.InMemoryService()
	resp, err := svc.Create(parent, &session.CreateRequest{
		AppName: "test-app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	// Append a content event so sendHistory has something to send.
	hev := session.NewEvent("inv-1")
	hev.Content = genai.NewContentFromText("history msg", "user")
	if err := svc.AppendEvent(parent, resp.Session, hev); err != nil {
		t.Fatalf("append history event: %v", err)
	}
	// Re-fetch so the in-memory session sees the appended event.
	got, err := svc.Get(parent, &session.GetRequest{
		AppName: "test-app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("session get: %v", err)
	}

	conn := &sendErrConn{sendErr: errors.New("synthetic send failure")}
	llm := newMockLiveLLMTel(conn)

	queue := agent.NewLiveRequestQueue(8)
	queue.Close()

	connectFn := func(_ string) (model.LiveConnection, error) {
		return llm.ConnectLive(parent, &model.LLMRequest{})
	}

	lf := &LiveFlow{Model: llm}
	ag, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(invCtx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return lf.RunLive(invCtx, connectFn, queue)
		},
	})
	if err != nil {
		t.Fatalf("agent create: %v", err)
	}

	invCtx := icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Session:          got.Session,
		Agent:            ag,
		LiveRequestQueue: queue,
	})

	var iterErrs []error
	for _, err := range ag.Run(invCtx) {
		if err != nil {
			iterErrs = append(iterErrs, err)
		}
	}
	if len(iterErrs) == 0 || !strings.Contains(iterErrs[0].Error(), "history handoff failed") {
		t.Fatalf("expected history handoff failure, got %v", iterErrs)
	}

	spans := testExporter.GetSpans().Snapshots()
	gc := requireSpanByName(t, spans, "generate_content test-model")
	if gc.Status().Code != codes.Error {
		t.Errorf("generate_content.Status = %v, want Error", gc.Status().Code)
	}
	ls := requireSpanByName(t, spans, "live_session test-agent")
	if ls.Status().Code != codes.Error {
		t.Errorf("live_session.Status = %v, want Error", ls.Status().Code)
	}
}

// sendErrConn is a LiveConnection that fails on the first Send.
type sendErrConn struct {
	sendErr error
}

func (c *sendErrConn) Send(_ context.Context, _ *model.LiveRequest) error {
	return c.sendErr
}

func (c *sendErrConn) Receive(ctx context.Context) (*model.LLMResponse, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
func (c *sendErrConn) Close() error { return nil }

// ---------------------------------------------------------------------------
// T13: Tool span context inheritance — trace.SpanFromContext(toolCtx) returns
// the execute_tool span, not generate_content or live_session.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_ToolSpanInheritance(t *testing.T) {
	var seenSpanID atomic.Value

	greet := &telTool{
		name: "greet",
		runFn: func(ctx tool.Context, _ map[string]any) (map[string]any, error) {
			id := trace.SpanFromContext(ctx).SpanContext().SpanID()
			seenSpanID.Store(id)
			return map[string]any{"ok": true}, nil
		},
	}
	conn := newMockLiveConnTel(
		liveConnPlan{resp: fcResp("fc1", "greet", map[string]any{})},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{tools: []tool.Tool{greet}}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	stored, _ := seenSpanID.Load().(trace.SpanID)
	if !stored.IsValid() {
		t.Fatalf("tool did not observe a valid span ID")
	}

	spans := exporter.GetSpans().Snapshots()
	et := requireSpanByName(t, spans, "execute_tool greet")
	if et.SpanContext().SpanID() != stored {
		t.Errorf("tool observed span ID %v, want execute_tool %v",
			stored, et.SpanContext().SpanID())
	}
}

// ---------------------------------------------------------------------------
// PHI scrub: walk every recorded attribute on a happy run with PHI-laden
// content and make sure no attribute carries the fixture string.
// ---------------------------------------------------------------------------

func TestLiveTelemetry_NoPHIInSpans(t *testing.T) {
	conn := newMockLiveConnTel(
		liveConnPlan{resp: textResp("secret patient data")},
		liveConnPlan{resp: &model.LLMResponse{TurnComplete: true}},
	)
	llm := newMockLiveLLMTel(conn)

	exporter, _, errs := driveLive(t, llm, driveOpts{}, true)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	for _, s := range exporter.GetSpans().Snapshots() {
		for _, kv := range s.Attributes() {
			if strings.Contains(kv.Value.AsString(), "secret patient data") {
				t.Errorf("span %q leaked content in attribute %q: %q",
					s.Name(), kv.Key, kv.Value.AsString())
			}
		}
	}
}

// Ensure the local mock plumbing compiles.
var (
	_ model.LiveConnection = (*mockLiveConnTel)(nil)
	_ model.LiveCapableLLM = (*mockLiveLLMTel)(nil)
	_ tool.Tool            = (*telTool)(nil)
)

// testExporter / setupTestTracer mirror the package-level tracer fixture in
// internal/llminternal/base_flow_telemetry_test.go. They are duplicated here
// because the moved telemetry test now lives in package liveflow and can no
// longer reach the private helper across the package boundary.
var (
	testExporter *tracetest.InMemoryExporter
	initTracer   sync.Once
)

func setupTestTracer(t *testing.T) {
	t.Helper()
	initTracer.Do(func() {
		// internal/telemetry initializes the global tracer provider once at startup.
		// Subsequent calls to otel.SetTracerProvider don't update existing tracer providers, so we can override only once.
		testExporter = tracetest.NewInMemoryExporter()
		tp := sdktrace.NewTracerProvider(
			sdktrace.WithSyncer(testExporter),
		)
		otel.SetTracerProvider(tp)
	})
	// Reset the exporter before each test to avoid flakiness.
	testExporter.Reset()
	t.Cleanup(func() {
		testExporter.Reset()
	})
}
