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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
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
