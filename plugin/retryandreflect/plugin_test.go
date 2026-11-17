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

package retryandreflect

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type mockTool struct {
	name string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "" }
func (m *mockTool) IsLongRunning() bool { return false }

type mockContext struct {
	agent.Context
	invocationID string
}

func (m *mockContext) InvocationID() string { return m.invocationID }

func TestNewOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []PluginOption
		wantErr bool
	}{
		{
			name: "defaults",
		},
		{
			name: "custom options",
			opts: []PluginOption{
				WithMaxRetries(5),
				WithErrorIfRetryExceeded(true),
				WithTrackingScope(Global),
			},
		},
		{
			name: "negative max retries",
			opts: []PluginOption{
				WithMaxRetries(-1),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := New(tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if p.Name() != "RetryAndReflectPlugin" {
				t.Errorf("expected plugin name RetryAndReflectPlugin, got %s", p.Name())
			}
			if p.AfterToolCallback() == nil {
				t.Errorf("expected AfterToolCallback to be set")
			}
			if p.OnToolErrorCallback() == nil {
				t.Errorf("expected OnToolErrorCallback to be set")
			}
		})
	}
}

func TestRetryAndReflect_SuccessResets(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "test-tool"}
	args := map[string]any{"arg1": "val1"}
	err := errors.New("some error")

	// Fail twice
	_, _ = r.onToolError(ctx, tl, args, err)
	_, _ = r.onToolError(ctx, tl, args, err)

	if count := r.scopedFailureCounters["inv1"]["test-tool"]; count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}

	// Succeed
	_, _ = r.afterTool(ctx, tl, args, nil, nil)

	if _, ok := r.scopedFailureCounters["inv1"]["test-tool"]; ok {
		t.Errorf("expected failure count to be reset")
	}
}

func TestRetryAndReflect_AfterToolNoResetOnReflection(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "test-tool"}
	args := map[string]any{"arg1": "val1"}
	err := errors.New("some error")

	// Fail once
	res, _ := r.onToolError(ctx, tl, args, err)

	if count := r.scopedFailureCounters["inv1"]["test-tool"]; count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}

	// AfterTool is called with the result of onToolError
	_, _ = r.afterTool(ctx, tl, args, res, nil)

	if count := r.scopedFailureCounters["inv1"]["test-tool"]; count != 1 {
		t.Errorf("expected failure count NOT to be reset, got %d", count)
	}
}

func TestRetryAndReflect_MaxRetries(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            2,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "test-tool"}
	args := map[string]any{"arg1": "val1"}
	err := errors.New("fail")

	// 1st retry
	res, _ := r.onToolError(ctx, tl, args, err)
	if res["retry_count"] != 1 {
		t.Errorf("expected retry_count 1, got %v", res["retry_count"])
	}
	if res["response_type"] != reflectAndRetryResponseType {
		t.Errorf("expected reflectAndRetryResponseType")
	}

	// 2nd retry
	res, _ = r.onToolError(ctx, tl, args, err)
	if res["retry_count"] != 2 {
		t.Errorf("expected retry_count 2, got %v", res["retry_count"])
	}

	// 3rd time - exceed
	res, _ = r.onToolError(ctx, tl, args, err)
	if res["retry_count"] != 2 { // It returns maxRetries in createToolRetryExceedMsg
		t.Errorf("expected retry_count 2 (max), got %v", res["retry_count"])
	}
	if !strings.Contains(res["reflection_guidance"].(string), "exceeded") {
		t.Errorf("expected guidance to mention exceeded, got %v", res["reflection_guidance"])
	}
	if !strings.Contains(res["reflection_guidance"].(string), "failed consecutively 2 times") {
		t.Errorf("expected guidance to mention 'failed consecutively 2 times', got %v", res["reflection_guidance"])
	}
}

func TestRetryAndReflect_ErrorIfRetryExceeded(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            1,
		errorIfRetryExceeded:  true,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "test-tool"}
	err := errors.New("fail")

	// 1st retry
	_, _ = r.onToolError(ctx, tl, nil, err)

	// 2nd time - exceed, should return error
	_, gotErr := r.onToolError(ctx, tl, nil, err)
	if gotErr != err {
		t.Errorf("expected error %v, got %v", err, gotErr)
	}
}

func TestRetryAndReflect_ConfirmationErrorsAreNotCounted(t *testing.T) {
	p, err := New()
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	onToolError := p.OnToolErrorCallback()
	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "pay"}

	res, gotErr := onToolError(ctx, tl, nil, tool.ErrConfirmationRequired)
	if res != nil || gotErr != nil {
		t.Errorf("ErrConfirmationRequired: onToolError() = %v, %v, want nil, nil", res, gotErr)
	}
	res, gotErr = onToolError(ctx, tl, nil, tool.ErrConfirmationRejected)
	if res != nil || gotErr != nil {
		t.Errorf("ErrConfirmationRejected: onToolError() = %v, %v, want nil, nil", res, gotErr)
	}
	res, gotErr = onToolError(ctx, tl, nil, fmt.Errorf("tool pay: %w", tool.ErrConfirmationRequired))
	if res != nil || gotErr != nil {
		t.Errorf("wrapped ErrConfirmationRequired: onToolError() = %v, %v, want nil, nil", res, gotErr)
	}

	res, gotErr = onToolError(ctx, tl, nil, errors.New("card declined"))
	if gotErr != nil {
		t.Fatalf("ordinary error: onToolError() error = %v, want nil", gotErr)
	}
	checkFields(t, res, map[string]any{
		"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
		"retry_count":   1,
	})
}

func TestRetryAndReflect_Scopes(t *testing.T) {
	rInvocation := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	rGlobal := &retryAndReflect{
		maxRetries:            3,
		scope:                 Global,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	ctx1 := &mockContext{invocationID: "inv1"}
	ctx2 := &mockContext{invocationID: "inv2"}
	tl := &mockTool{name: "test-tool"}
	err := errors.New("fail")

	// Invocation scope
	_, _ = rInvocation.onToolError(ctx1, tl, nil, err)
	if rInvocation.scopedFailureCounters["inv1"]["test-tool"] != 1 {
		t.Errorf("expected 1 failure in inv1")
	}
	_, _ = rInvocation.onToolError(ctx2, tl, nil, err)
	if rInvocation.scopedFailureCounters["inv2"]["test-tool"] != 1 {
		t.Errorf("expected 1 failure in inv2")
	}

	// Global scope
	_, _ = rGlobal.onToolError(ctx1, tl, nil, err)
	if rGlobal.scopedFailureCounters[globalScopeKey]["test-tool"] != 1 {
		t.Errorf("expected 1 failure in global scope")
	}
	_, _ = rGlobal.onToolError(ctx2, tl, nil, err)
	if rGlobal.scopedFailureCounters[globalScopeKey]["test-tool"] != 2 {
		t.Errorf("expected 2 failures in global scope")
	}
}

func TestRetryAndReflect_ToolPanic(t *testing.T) {
	panicErr := toolinternal.NewPanicError("test-tool", "secret-user-data-123", []byte("goroutine 28 [running]:\nruntime/debug.Stack()\n"))

	for _, tc := range []struct {
		name string
		err  error
		// calls is how many times the tool fails; with maxRetries 1 the first
		// failure builds the reflection response and the second the
		// retry-limit-exceeded one.
		calls             int
		wantTopOfGuidance string
	}{
		{name: "reflection response", err: panicErr, calls: 1, wantTopOfGuidance: "failed"},
		{name: "retry limit exceeded response", err: panicErr, calls: 2, wantTopOfGuidance: "exceeded"},
		{
			// An error on its way out of a tool is wrapped more than once:
			// agenttool wraps what the sub-agent run returned, and an earlier
			// OnToolError callback is free to wrap it before this plugin sees
			// it.
			name:              "wrapped panic",
			err:               fmt.Errorf("lookup failed: %w", panicErr),
			calls:             1,
			wantTopOfGuidance: "failed",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            1,
				scope:                 Invocation,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			ctx := &mockContext{invocationID: "inv1"}
			tl := &mockTool{name: "test-tool"}

			var res map[string]any
			for range tc.calls {
				res, _ = r.onToolError(ctx, tl, map[string]any{"order_id": "1"}, tc.err)
			}

			if got, want := res["error_details"], "Invoking `test-tool()` failed with an internal error."; got != want {
				t.Errorf("error_details = %v, want %q", got, want)
			}
			if got, want := res["error_type"], "panic"; got != want {
				t.Errorf("error_type = %v, want %q", got, want)
			}
			guidance, _ := res["reflection_guidance"].(string)
			if !strings.Contains(guidance, tc.wantTopOfGuidance) {
				t.Errorf("reflection_guidance does not mention %q: %q", tc.wantTopOfGuidance, guidance)
			}
			// Nothing the plugin answers with may carry the panic value, the
			// stack or the internal error type: the model, the session and REST
			// clients all see this response.
			for key, value := range res {
				text, ok := value.(string)
				if !ok {
					continue
				}
				for _, leaked := range []string{"secret-user-data-123", "goroutine", "PanicError"} {
					if strings.Contains(text, leaked) {
						t.Errorf("response[%q] leaks %q", key, leaked)
					}
				}
			}
		})
	}
}

func TestRetryAndReflect_ReturnedErrorReachesTheModel(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            1,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	res, _ := r.onToolError(&mockContext{invocationID: "inv1"}, &mockTool{name: "test-tool"}, nil, errors.New("order not found"))

	if got, want := res["error_details"], "order not found"; got != want {
		t.Errorf("error_details = %v, want %q", got, want)
	}
	if got, want := res["error_type"], "*errors.errorString"; got != want {
		t.Errorf("error_type = %v, want %q", got, want)
	}
	if guidance, _ := res["reflection_guidance"].(string); !strings.Contains(guidance, "order not found") {
		t.Errorf("reflection_guidance does not carry the tool's error: %q", guidance)
	}
}

// TestToolPanicThroughRunner drives a panicking tool through a full run with
// the plugin installed, because the plugin answers OnToolError with its own
// response and so decides on its own what the model sees.
func TestToolPanicThroughRunner(t *testing.T) {
	panicTool, err := functiontool.New(functiontool.Config{Name: "lookup_order"}, func(agent.Context, struct{}) (struct{}, error) {
		panic("secret-user-data-123")
	})
	if err != nil {
		t.Fatal(err)
	}
	m := &testutil.MockModel{Responses: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "lookup_order", Args: map[string]any{}}}}},
		genai.NewContentFromText("done", "model"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{panicTool}})
	if err != nil {
		t.Fatal(err)
	}
	p, err := New()
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{p}})

	var responses []string
	for ev, err := range r.Run(t, "s1", "hi") {
		if err != nil {
			t.Fatal(err)
		}
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, part := range ev.LLMResponse.Content.Parts {
			if part.FunctionResponse != nil {
				responses = append(responses, fmt.Sprint(part.FunctionResponse.Response))
			}
		}
	}

	if len(responses) == 0 {
		t.Fatal("the run produced no function response")
	}
	// Without this the test would also pass when the plugin stopped answering
	// and the flow's own substitution took over, which is not what it covers.
	if !strings.Contains(responses[0], reflectAndRetryResponseType) {
		t.Errorf("the plugin did not answer the failed call: %.200q", responses[0])
	}
	for _, got := range responses {
		for _, leaked := range []string{"secret-user-data-123", "goroutine", "PanicError"} {
			if strings.Contains(got, leaked) {
				t.Errorf("function response leaks %q: %.200q", leaked, got)
			}
		}
	}
}

// newRunContext returns the context of a run of invocationID over a session
// holding events.
func newRunContext(t *testing.T, invocationID string, events ...*session.Event) agent.InvocationContext {
	t.Helper()
	service := session.InMemoryService()
	resp, err := service.Create(t.Context(), &session.CreateRequest{AppName: "test", UserID: "user", SessionID: "ses1"})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	for _, ev := range events {
		if err := service.AppendEvent(t.Context(), resp.Session, ev); err != nil {
			t.Fatalf("Failed to append event: %v", err)
		}
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Session:      resp.Session,
		InvocationID: invocationID,
	})
}

func longRunningCallEvent(invocationID string, callIDs ...string) *session.Event {
	parts := make([]*genai.Part, 0, len(callIDs))
	for _, id := range callIDs {
		parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "ask_approval"}})
	}
	return &session.Event{
		InvocationID:       invocationID,
		Author:             "agent",
		LongRunningToolIDs: callIDs,
		LLMResponse:        model.LLMResponse{Content: &genai.Content{Role: "model", Parts: parts}},
	}
}

func functionResponseEvent(invocationID, author, callID string, response map[string]any) *session.Event {
	return &session.Event{
		InvocationID: invocationID,
		Author:       author,
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: callID, Name: "ask_approval", Response: response}},
		}}},
	}
}

func TestRetryAndReflect_AfterRunDropsInvocationCounts(t *testing.T) {
	p, err := New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if p.AfterRunCallback() == nil {
		t.Fatal("expected New to register an AfterRunCallback")
	}

	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	tl := &mockTool{name: "test-tool"}
	err = errors.New("some error")
	_, _ = r.onToolError(&mockContext{invocationID: "inv1"}, tl, nil, err)
	_, _ = r.onToolError(&mockContext{invocationID: "inv2"}, tl, nil, err)

	r.afterRun(newRunContext(t, "inv1"))

	if _, ok := r.scopedFailureCounters["inv1"]; ok {
		t.Errorf("expected the counts of the ended invocation inv1 to be dropped")
	}
	if count := r.scopedFailureCounters["inv2"]["test-tool"]; count != 1 {
		t.Errorf("expected the counts of invocation inv2 to be kept, got count %d", count)
	}
}

func TestRetryAndReflect_AfterRunKeepsCountsOfPausedInvocation(t *testing.T) {
	tests := []struct {
		name     string
		events   []*session.Event
		wantKept bool
	}{
		{
			name:     "unanswered long-running call",
			events:   []*session.Event{longRunningCallEvent("inv1", "call-1")},
			wantKept: true,
		},
		{
			name: "long-running call answered by the user",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1"),
				functionResponseEvent("inv1", "user", "call-1", map[string]any{"approved": true}),
			},
			wantKept: false,
		},
		{
			name: "long-running call settled by a failure response of the plugin",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1"),
				functionResponseEvent("inv1", "agent", "call-1", map[string]any{"response_type": reflectAndRetryResponseType}),
			},
			wantKept: false,
		},
		{
			name: "intermediate response of the tool itself",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1"),
				functionResponseEvent("inv1", "agent", "call-1", map[string]any{"status": "pending"}),
			},
			wantKept: true,
		},
		{
			name: "one of two parallel long-running calls answered",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1", "call-2"),
				functionResponseEvent("inv1", "user", "call-1", map[string]any{"approved": true}),
			},
			wantKept: true,
		},
		{
			name: "both parallel long-running calls answered",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1", "call-2"),
				functionResponseEvent("inv1", "user", "call-1", map[string]any{"approved": true}),
				functionResponseEvent("inv1", "user", "call-2", map[string]any{"approved": false}),
			},
			wantKept: false,
		},
		{
			name: "response to an unrelated call ID",
			events: []*session.Event{
				longRunningCallEvent("inv1", "call-1"),
				functionResponseEvent("inv1", "user", "call-unknown", map[string]any{"approved": true}),
			},
			wantKept: true,
		},
		{
			name:     "unanswered long-running call of another invocation",
			events:   []*session.Event{longRunningCallEvent("inv0", "call-0")},
			wantKept: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            3,
				scope:                 Invocation,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			_, _ = r.onToolError(&mockContext{invocationID: "inv1"}, &mockTool{name: "test-tool"}, nil, errors.New("some error"))

			r.afterRun(newRunContext(t, "inv1", tt.events...))

			_, kept := r.scopedFailureCounters["inv1"]
			if kept != tt.wantKept {
				t.Errorf("counts of inv1 kept = %t, want %t", kept, tt.wantKept)
			}
		})
	}
}

func TestRetryAndReflect_AfterRunKeepsGlobalCounts(t *testing.T) {
	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Global,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	_, _ = r.onToolError(&mockContext{invocationID: "inv1"}, &mockTool{name: "test-tool"}, nil, errors.New("some error"))

	r.afterRun(newRunContext(t, "inv1"))

	if count := r.scopedFailureCounters[globalScopeKey]["test-tool"]; count != 1 {
		t.Errorf("expected the global counts to be kept, got count %d", count)
	}
}

func TestRetryAndReflect_AfterRunDropsCountsOfEveryPerInvocationScope(t *testing.T) {
	// WithTrackingScope accepts any value, and every scope but Global keys its
	// counts by invocation ID.
	tests := []struct {
		name  string
		scope TrackingScope
	}{
		{name: "Invocation", scope: Invocation},
		{name: "empty", scope: ""},
		{name: "upper case", scope: "INVOCATION"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            3,
				scope:                 tt.scope,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			_, _ = r.onToolError(&mockContext{invocationID: "inv1"}, &mockTool{name: "test-tool"}, nil, errors.New("some error"))
			if _, ok := r.scopedFailureCounters["inv1"]; !ok {
				t.Fatal("expected the failure to be counted under invocation inv1")
			}

			r.afterRun(newRunContext(t, "inv1"))

			if _, ok := r.scopedFailureCounters["inv1"]; ok {
				t.Errorf("expected the counts of the ended invocation inv1 to be dropped")
			}
		})
	}
}

// eventsCountingSession counts the reads of its events.
type eventsCountingSession struct {
	session.Session
	eventsReads int
}

func (s *eventsCountingSession) Events() session.Events {
	s.eventsReads++
	return s.Session.Events()
}

func TestRetryAndReflect_AfterRunScansEventsOnlyOfInvocationWithCounts(t *testing.T) {
	tests := []struct {
		name             string
		failedInvocation string
		wantEventsReads  int
	}{
		{name: "no failure in the ended invocation", failedInvocation: "inv2", wantEventsReads: 0},
		{name: "failure in the ended invocation", failedInvocation: "inv1", wantEventsReads: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            3,
				scope:                 Invocation,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			_, _ = r.onToolError(&mockContext{invocationID: tt.failedInvocation}, &mockTool{name: "test-tool"}, nil, errors.New("some error"))
			sess := &eventsCountingSession{Session: newRunContext(t, "inv1").Session()}

			r.afterRun(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Session:      sess,
				InvocationID: "inv1",
			}))

			if sess.eventsReads != tt.wantEventsReads {
				t.Errorf("session events read %d times, want %d", sess.eventsReads, tt.wantEventsReads)
			}
		})
	}
}

type noArgs struct{}

// TestRetryAndReflect_RunnerContinuesCountsOfAnsweredRun runs the plugin
// through runner.Run. The run that answers a long-running call reuses the
// invocation ID of the paused run, so it continues that run's retry budget,
// and the counts are dropped once it ends.
func TestRetryAndReflect_RunnerContinuesCountsOfAnsweredRun(t *testing.T) {
	flaky, err := functiontool.New(functiontool.Config{Name: "flaky", Description: "fails"},
		func(agent.Context, noArgs) (map[string]any, error) {
			return nil, errors.New("backend unavailable")
		})
	if err != nil {
		t.Fatalf("Failed to create the flaky tool: %v", err)
	}
	askApproval, err := functiontool.New(functiontool.Config{Name: "ask_approval", Description: "asks the user", IsLongRunning: true},
		func(agent.Context, noArgs) (map[string]any, error) {
			return map[string]any{"status": "pending"}, nil
		})
	if err != nil {
		t.Fatalf("Failed to create the long-running tool: %v", err)
	}
	mockModel := &testutil.MockModel{Responses: []*genai.Content{
		// The paused run: a failure, then a call the user has not answered yet.
		genai.NewContentFromFunctionCall("flaky", map[string]any{}, genai.RoleModel),
		genai.NewContentFromFunctionCall("ask_approval", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("waiting for approval", genai.RoleModel),
		// The run that answers the call: the same tool fails again.
		genai.NewContentFromFunctionCall("flaky", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: mockModel,
		Tools: []tool.Tool{flaky, askApproval},
	})
	if err != nil {
		t.Fatalf("Failed to create the agent: %v", err)
	}
	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	p, err := plugin.New(plugin.Config{
		Name:                "RetryAndReflectPlugin",
		AfterToolCallback:   r.afterTool,
		OnToolErrorCallback: r.onToolError,
		AfterRunCallback:    r.afterRun,
	})
	if err != nil {
		t.Fatalf("Failed to create the plugin: %v", err)
	}
	testRunner := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{p}})
	failureCount := func(invocationID string) (int, bool) {
		r.mu.Lock()
		defer r.mu.Unlock()
		counts, ok := r.scopedFailureCounters[invocationID]
		return counts["flaky"], ok
	}

	paused, err := testutil.CollectEvents(testRunner.Run(t, "ses1", "start"))
	if err != nil {
		t.Fatalf("Paused run failed: %v", err)
	}
	var callID, invocationID string
	for _, ev := range paused {
		if len(ev.LongRunningToolIDs) > 0 {
			callID, invocationID = ev.LongRunningToolIDs[0], ev.InvocationID
		}
	}
	if callID == "" {
		t.Fatal("expected the first run to pause on a long-running call")
	}
	if count, ok := failureCount(invocationID); !ok || count != 1 {
		t.Fatalf("after the paused run: count %d (kept %t), want 1 kept", count, ok)
	}

	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: callID, Name: "ask_approval", Response: map[string]any{"approved": true},
	}}}}
	answered, err := testutil.CollectEvents(testRunner.RunContent(t, "ses1", answer))
	if err != nil {
		t.Fatalf("Answering run failed: %v", err)
	}
	var retryCount any
	for _, ev := range answered {
		if ev.InvocationID != invocationID {
			t.Errorf("event of the answering run has invocation ID %q, want %q", ev.InvocationID, invocationID)
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if fr := part.FunctionResponse; fr != nil && fr.Name == "flaky" {
				retryCount = fr.Response["retry_count"]
			}
		}
	}
	if retryCount != 2 {
		t.Errorf("retry_count of the failure in the answering run = %v, want 2", retryCount)
	}
	if count, ok := failureCount(invocationID); ok {
		t.Errorf("after the answering run: count %d kept, want the counts dropped", count)
	}
}

func TestNew_RegistersOnRunErrorCallback(t *testing.T) {
	p, err := New()
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if p.OnRunErrorCallback() == nil {
		t.Error("New() registered no OnRunErrorCallback, want one")
	}
	if p.AfterRunCallback() == nil {
		t.Error("New() registered no AfterRunCallback, want one")
	}
}

// TestRetryAndReflect_FailedRunDropsInvocationCounts runs the plugin through
// runner.Run on a run that a model failure ends after a tool failure. The
// runner skips the after-run callbacks of such a run, so the counts are
// dropped by the on-run-error callback.
func TestRetryAndReflect_FailedRunDropsInvocationCounts(t *testing.T) {
	flaky, err := functiontool.New(functiontool.Config{Name: "flaky", Description: "fails"},
		func(agent.Context, noArgs) (map[string]any, error) {
			return nil, errors.New("backend unavailable")
		})
	if err != nil {
		t.Fatalf("Failed to create the flaky tool: %v", err)
	}
	// The second model call has no response left and fails.
	mockModel := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("flaky", map[string]any{}, genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: mockModel,
		Tools: []tool.Tool{flaky},
	})
	if err != nil {
		t.Fatalf("Failed to create the agent: %v", err)
	}
	r := &retryAndReflect{
		maxRetries:            3,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	p, err := plugin.New(plugin.Config{
		Name:                "RetryAndReflectPlugin",
		AfterToolCallback:   r.afterTool,
		OnToolErrorCallback: r.onToolError,
		AfterRunCallback:    r.afterRun,
		OnRunErrorCallback:  r.onRunError,
	})
	if err != nil {
		t.Fatalf("Failed to create the plugin: %v", err)
	}
	testRunner := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{p}})
	failureCount := func(invocationID string) (int, bool) {
		r.mu.Lock()
		defer r.mu.Unlock()
		counts, ok := r.scopedFailureCounters[invocationID]
		return counts["flaky"], ok
	}

	var invocationID string
	var runErr error
	for ev, err := range testRunner.Run(t, "ses1", "start") {
		if err != nil {
			runErr = err
			break
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if fr := part.FunctionResponse; fr != nil && fr.Name == "flaky" {
				invocationID = ev.InvocationID
				if count, ok := failureCount(invocationID); !ok || count != 1 {
					t.Errorf("after the tool failure: count %d (kept %t), want 1 kept", count, ok)
				}
			}
		}
	}
	if runErr == nil {
		t.Fatal("run ended without an error, want the model failure")
	}
	if invocationID == "" {
		t.Fatal("run yielded no response of the flaky tool")
	}
	if count, ok := failureCount(invocationID); ok {
		t.Errorf("after the failed run: count %d kept, want the counts dropped", count)
	}
}

// mustAfterTool returns the AfterToolCallback of a plugin built with opts.
func mustAfterTool(t *testing.T, opts ...PluginOption) llmagent.AfterToolCallback {
	t.Helper()
	p, err := New(opts...)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p.AfterToolCallback()
}

// checkFields reports every key of want whose value in got differs.
func checkFields(t *testing.T, got, want map[string]any) {
	t.Helper()
	for key, w := range want {
		if g := got[key]; g != w {
			t.Errorf("response[%q] = %#v, want %#v", key, g, w)
		}
	}
}

// checkGuidance reports each of has that got's reflection_guidance lacks and
// each of lacks that it holds.
func checkGuidance(t *testing.T, got map[string]any, has, lacks []string) {
	t.Helper()
	guidance, _ := got["reflection_guidance"].(string)
	for _, s := range has {
		if !strings.Contains(guidance, s) {
			t.Errorf("reflection_guidance lacks %q:\n%s", s, guidance)
		}
	}
	for _, s := range lacks {
		if strings.Contains(guidance, s) {
			t.Errorf("reflection_guidance holds %q:\n%s", s, guidance)
		}
	}
}

// statusExtractor reports a nil result and a result whose status is "error"
// as failures.
func statusExtractor(_ agent.Context, _ tool.Tool, _, result map[string]any) error {
	if result == nil {
		return errors.New("empty result")
	}
	if result["status"] == "error" {
		return fmt.Errorf("status=error: %v", result["msg"])
	}
	return nil
}

func TestRetryAndReflect_ResultErrorExtractorReflectionAndExceed(t *testing.T) {
	ctx := &mockContext{invocationID: "inv1"}
	search := &mockTool{name: "search"}
	args := map[string]any{"q": "a"}

	t.Run("reflection, exceed and reset", func(t *testing.T) {
		type call struct {
			tool         string
			args, result map[string]any
		}
		var calls []call
		afterTool := mustAfterTool(t, WithMaxRetries(1), WithResultErrorExtractor(func(ctx agent.Context, tl tool.Tool, args, result map[string]any) error {
			calls = append(calls, call{tool: tl.Name(), args: args, result: result})
			return statusExtractor(ctx, tl, args, result)
		}))

		res, err := afterTool(ctx, search, args, map[string]any{"status": "error", "msg": "bad input"}, nil)
		if err != nil {
			t.Fatalf("first failure: error = %v, want nil", err)
		}
		if len(calls) != 1 || calls[0].tool != "search" || calls[0].args["q"] != "a" || calls[0].result["status"] != "error" {
			t.Errorf("extractor calls = %+v, want one call of search with the call's args and result", calls)
		}
		checkFields(t, res, map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_type":    "ToolError",
			"error_details": "status=error: bad input",
			"retry_count":   1,
		})
		checkGuidance(t, res, []string{
			"The call to tool `search` failed.",
			"**Error Details:**\n```\nstatus=error: bad input\n```",
			`"q": "a"`,
			"retry attempt **1 of 1**",
		}, []string{"ToolError:"})

		res, err = afterTool(ctx, search, args, nil, nil)
		if err != nil {
			t.Fatalf("second failure: error = %v, want nil", err)
		}
		if len(calls) != 2 || calls[1].result != nil {
			t.Errorf("extractor calls = %+v, want a second call with a nil result", calls)
		}
		checkFields(t, res, map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_type":    "ToolError",
			"error_details": "empty result",
			"retry_count":   1,
		})
		checkGuidance(t, res, []string{
			"the retry limit has been exceeded",
			"**Last Error:**\n```\nempty result\n```",
			"Do not attempt to use the `search` tool again for this task.",
		}, []string{"ToolError:"})

		res, err = afterTool(ctx, search, args, map[string]any{"status": "ok"}, nil)
		if res != nil || err != nil {
			t.Errorf("success: afterTool() = %v, %v, want nil, nil", res, err)
		}

		// With the count reset, the next failure gets the reflection response
		// again rather than the exceed response.
		res, _ = afterTool(ctx, search, args, map[string]any{"status": "error", "msg": "again"}, nil)
		checkFields(t, res, map[string]any{"error_details": "status=error: again", "retry_count": 1})
		checkGuidance(t, res, []string{"The call to tool `search` failed."}, nil)
	})

	t.Run("max retries 0", func(t *testing.T) {
		afterTool := mustAfterTool(t, WithMaxRetries(0), WithResultErrorExtractor(statusExtractor))

		res, err := afterTool(ctx, search, args, map[string]any{"status": "error", "msg": "bad input"}, nil)
		if err != nil {
			t.Fatalf("error = %v, want nil", err)
		}
		checkFields(t, res, map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_type":    "ToolError",
			"error_details": "status=error: bad input",
			"retry_count":   0,
		})
		checkGuidance(t, res, []string{
			"the retry limit has been exceeded",
			"**Last Error:**\n```\nstatus=error: bad input\n```",
		}, []string{"ToolError:"})
	})

	t.Run("last option nil", func(t *testing.T) {
		afterTool := mustAfterTool(t, WithResultErrorExtractor(statusExtractor), WithResultErrorExtractor(nil))

		res, err := afterTool(ctx, search, args, map[string]any{"status": "error"}, nil)
		if res != nil || err != nil {
			t.Errorf("afterTool() = %v, %v, want nil, nil", res, err)
		}
	})

	t.Run("last option set", func(t *testing.T) {
		afterTool := mustAfterTool(t, WithResultErrorExtractor(nil), WithResultErrorExtractor(func(agent.Context, tool.Tool, map[string]any, map[string]any) error {
			return errors.New("from second")
		}))

		res, _ := afterTool(ctx, search, args, map[string]any{"status": "error"}, nil)
		checkFields(t, res, map[string]any{"error_details": "from second"})
	})
}

func TestRetryAndReflect_ResultErrorExtractorEmptyErrorString(t *testing.T) {
	afterTool := mustAfterTool(t, WithResultErrorExtractor(func(agent.Context, tool.Tool, map[string]any, map[string]any) error {
		return errors.New("")
	}))
	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "search"}

	res, err := afterTool(ctx, tl, nil, map[string]any{}, nil)
	if err != nil {
		t.Fatalf("first failure: error = %v, want nil", err)
	}
	checkFields(t, res, map[string]any{
		"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
		"error_type":    "ToolError",
		"error_details": "",
		"retry_count":   1,
	})

	res, _ = afterTool(ctx, tl, nil, map[string]any{}, nil)
	checkFields(t, res, map[string]any{"retry_count": 2})
}

type customExtractErr struct {
	code int
	msg  string
}

func (e *customExtractErr) Error() string { return e.msg }

func TestRetryAndReflect_ResultErrorExtractorErrorIfRetryExceeded(t *testing.T) {
	wantErr := &customExtractErr{code: 503, msg: "upstream down"}
	extract := WithResultErrorExtractor(func(agent.Context, tool.Tool, map[string]any, map[string]any) error {
		return wantErr
	})
	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "search"}

	t.Run("max retries 1", func(t *testing.T) {
		afterTool := mustAfterTool(t, WithMaxRetries(1), WithErrorIfRetryExceeded(true), extract)

		res, err := afterTool(ctx, tl, nil, map[string]any{}, nil)
		if err != nil {
			t.Fatalf("first failure: error = %v, want nil", err)
		}
		checkFields(t, res, map[string]any{"retry_count": 1})

		res, err = afterTool(ctx, tl, nil, map[string]any{}, nil)
		if res != nil {
			t.Errorf("second failure: response = %v, want nil", res)
		}
		if err == nil || err.Error() != "upstream down" {
			t.Fatalf("second failure: error = %v, want %q", err, "upstream down")
		}
		if !errors.Is(err, wantErr) {
			t.Errorf("errors.Is(%v, wantErr) = false, want true", err)
		}
		var target *customExtractErr
		if !errors.As(err, &target) || target.code != 503 {
			t.Errorf("errors.As(%v) found %+v, want code 503", err, target)
		}
	})

	t.Run("max retries 0", func(t *testing.T) {
		afterTool := mustAfterTool(t, WithMaxRetries(0), WithErrorIfRetryExceeded(true), extract)

		res, err := afterTool(ctx, tl, nil, map[string]any{}, nil)
		if res != nil || !errors.Is(err, wantErr) {
			t.Errorf("afterTool() = %v, %v, want nil and the extractor's error", res, err)
		}
	})
}

func TestRetryAndReflect_ResultErrorExtractorSkipsToolErrorAndReflectionAndHandlesSpecialErrors(t *testing.T) {
	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "search"}

	t.Run("not called for a tool error or a reflection response", func(t *testing.T) {
		calls := 0
		afterTool := mustAfterTool(t, WithResultErrorExtractor(func(agent.Context, tool.Tool, map[string]any, map[string]any) error {
			calls++
			return errors.New("should not run")
		}))

		res, err := afterTool(ctx, tl, nil, nil, errors.New("tool failed"))
		if res != nil || err != nil {
			t.Errorf("tool error: afterTool() = %v, %v, want nil, nil", res, err)
		}
		res, err = afterTool(ctx, tl, nil, map[string]any{"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"}, nil)
		if res != nil || err != nil {
			t.Errorf("reflection response: afterTool() = %v, %v, want nil, nil", res, err)
		}
		if calls != 0 {
			t.Errorf("extractor calls = %d, want 0", calls)
		}
	})

	t.Run("confirmation errors from the extractor count", func(t *testing.T) {
		results := []error{tool.ErrConfirmationRequired, tool.ErrConfirmationRejected}
		afterTool := mustAfterTool(t, WithResultErrorExtractor(func(agent.Context, tool.Tool, map[string]any, map[string]any) error {
			err := results[0]
			results = results[1:]
			return err
		}))

		res, _ := afterTool(ctx, tl, nil, map[string]any{}, nil)
		checkFields(t, res, map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_type":    "ToolError",
			"retry_count":   1,
		})
		res, _ = afterTool(ctx, tl, nil, map[string]any{}, nil)
		checkFields(t, res, map[string]any{"error_type": "ToolError", "retry_count": 2})
	})
}

func TestRetryAndReflect_ResultErrorExtractorConcurrency(t *testing.T) {
	var barrier sync.WaitGroup
	barrier.Add(2)
	afterTool := mustAfterTool(t, WithMaxRetries(10), WithTrackingScope(Global), WithResultErrorExtractor(func(_ agent.Context, _ tool.Tool, _, result map[string]any) error {
		switch result["phase"] {
		case "parallel":
			// Each of the two calls waits here for the other, which both
			// reach only if the plugin holds no lock while it calls the
			// extractor.
			barrier.Done()
			barrier.Wait()
		case "ok":
			return nil
		}
		return errors.New("boom")
	}))
	ctx := &mockContext{invocationID: "inv1"}
	tl := &mockTool{name: "search"}

	counts := make(chan any, 2)
	for range 2 {
		go func() {
			res, _ := afterTool(ctx, tl, nil, map[string]any{"phase": "parallel"}, nil)
			counts <- res["retry_count"]
		}()
	}
	var got []int
	timeout := time.After(30 * time.Second)
	for range 2 {
		select {
		case c := <-counts:
			n, _ := c.(int)
			got = append(got, n)
		case <-timeout:
			t.Fatal("the two parallel calls did not both return; the plugin holds a lock while it calls the extractor")
		}
	}
	slices.Sort(got)
	if want := []int{1, 2}; !slices.Equal(got, want) {
		t.Errorf("retry counts of the parallel calls = %v, want %v", got, want)
	}

	if res, err := afterTool(ctx, tl, nil, map[string]any{"phase": "ok"}, nil); res != nil || err != nil {
		t.Errorf("success: afterTool() = %v, %v, want nil, nil", res, err)
	}
	res, _ := afterTool(ctx, tl, nil, map[string]any{"phase": "after reset"}, nil)
	checkFields(t, res, map[string]any{"retry_count": 1})
}
