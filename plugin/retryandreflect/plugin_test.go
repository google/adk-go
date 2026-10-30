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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
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
