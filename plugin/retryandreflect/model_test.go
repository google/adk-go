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
	"context"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"regexp"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// modelTestContext is the callback context of one agent's model call in one
// invocation. Value forwards to ctx so that platform.NewUUID can read a UUID
// provider from it.
type modelTestContext struct {
	agent.Context
	ctx          context.Context
	invocationID string
	agentName    string
}

func (c *modelTestContext) InvocationID() string { return c.invocationID }
func (c *modelTestContext) AgentName() string    { return c.agentName }
func (c *modelTestContext) Value(key any) any    { return c.ctx.Value(key) }

func newModelTestContext(t *testing.T, invocationID, agentName string) *modelTestContext {
	t.Helper()
	return &modelTestContext{ctx: t.Context(), invocationID: invocationID, agentName: agentName}
}

// modelTestRunContext is the invocation context the run-end callbacks get.
type modelTestRunContext struct {
	agent.InvocationContext
	invocationID string
}

func (c *modelTestRunContext) InvocationID() string { return c.invocationID }

// modelTestLLM is a model.LLM that answers request i with the responses of
// turns[i], in order, and fails once the turns run out. It records the
// function declaration names of every request.
type modelTestLLM struct {
	name  string
	turns [][]*model.LLMResponse

	mu       sync.Mutex
	requests [][]string
}

func (m *modelTestLLM) Name() string { return m.name }

func (m *modelTestLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var declared []string
	if req.Config != nil {
		for _, tl := range req.Config.Tools {
			for _, fd := range tl.FunctionDeclarations {
				declared = append(declared, fd.Name)
			}
		}
	}
	m.mu.Lock()
	i := len(m.requests)
	m.requests = append(m.requests, declared)
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		if i >= len(m.turns) {
			yield(nil, errors.New("script exhausted"))
			return
		}
		for _, r := range m.turns[i] {
			resp := *r
			if !yield(&resp, nil) {
				return
			}
		}
	}
}

func (m *modelTestLLM) requestCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

func (m *modelTestLLM) declarations(i int) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.requests[i]
}

// modelTestRun runs agent "root" on llm with tools and plugins, in that
// order, for one user message, and returns the events up to the first error
// and that error.
func modelTestRun(t *testing.T, llm *modelTestLLM, tools []tool.Tool, plugins []*plugin.Plugin, streaming agent.StreamingMode) ([]*session.Event, error) {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "root", Model: llm, Tools: tools})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: sessions,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() failed: %v", err)
	}
	var events []*session.Event
	msg := genai.NewContentFromText("hi", genai.RoleUser)
	for ev, err := range r.Run(t.Context(), "user", created.Session.ID(), msg, agent.RunConfig{StreamingMode: streaming}) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

func modelTestNewModel(t *testing.T, opts ...ModelOption) *plugin.Plugin {
	t.Helper()
	p, err := NewModel(opts...)
	if err != nil {
		t.Fatalf("NewModel() failed: %v", err)
	}
	return p
}

// modelTestResult holds what an after-model callback returned.
type modelTestResult struct {
	resp *model.LLMResponse
	err  error
}

// modelTestCall runs p's before-model callback for a request to modelName,
// then its after-model callback for resp, as the flow does for one model
// call.
func modelTestCall(t *testing.T, p *plugin.Plugin, ctx agent.Context, modelName string, resp *model.LLMResponse) modelTestResult {
	t.Helper()
	got, err := p.BeforeModelCallback()(ctx, &model.LLMRequest{Model: modelName})
	if got != nil || err != nil {
		t.Fatalf("BeforeModelCallback() = %v, %v; want nil, nil", got, err)
	}
	return modelTestAfter(p, ctx, resp)
}

// modelTestAfter runs only p's after-model callback for resp.
func modelTestAfter(p *plugin.Plugin, ctx agent.Context, resp *model.LLMResponse) modelTestResult {
	got, err := p.AfterModelCallback()(ctx, resp, nil)
	return modelTestResult{resp: got, err: err}
}

// modelTestRetryCount returns the retry_count of the reflection call in got,
// and fails the test when got is not a reflection response.
func modelTestRetryCount(t *testing.T, r modelTestResult) int {
	t.Helper()
	call := modelTestReflectionCall(t, r)
	n, ok := call.Args["retry_count"].(int)
	if !ok {
		t.Fatalf("retry_count = %#v, want an int", call.Args["retry_count"])
	}
	return n
}

func modelTestReflectionCall(t *testing.T, r modelTestResult) *genai.FunctionCall {
	t.Helper()
	got := r.resp
	if r.err != nil {
		t.Fatalf("after-model callback returned error %v, want a reflection response", r.err)
	}
	if got == nil || got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("after-model callback returned %+v, want a reflection response with one function call", got)
	}
	call := got.Content.Parts[0].FunctionCall
	if call.Name != "adk_handle_model_error" {
		t.Fatalf("function call name = %q, want %q", call.Name, "adk_handle_model_error")
	}
	return call
}

func modelTestWantNil(t *testing.T, r modelTestResult) {
	t.Helper()
	if r.resp != nil || r.err != nil {
		t.Fatalf("after-model callback = %+v, %v; want nil, nil", r.resp, r.err)
	}
}

func modelTestWantExhausted(t *testing.T, r modelTestResult, wantText string) {
	t.Helper()
	if r.resp != nil {
		t.Errorf("after-model callback response = %+v, want nil", r.resp)
	}
	err := r.err
	if err == nil {
		t.Fatalf("after-model callback error = nil, want %q", wantText)
	}
	if err.Error() != wantText {
		t.Errorf("after-model callback error = %q, want %q", err.Error(), wantText)
	}
	if !errors.Is(err, ErrModelRetryLimitExceeded) {
		t.Errorf("errors.Is(%v, ErrModelRetryLimitExceeded) = false, want true", err)
	}
}

func modelTestMalformed() *model.LLMResponse {
	return &model.LLMResponse{ErrorCode: "MALFORMED_FUNCTION_CALL", FinishReason: genai.FinishReasonMalformedFunctionCall}
}

func modelTestReservedCall() *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: "adk_handle_model_error", Args: map[string]any{"retry_count": 99}}},
	}}}
}

func modelTestFunctionCalls(events []*session.Event, partial bool) []*genai.FunctionCall {
	var calls []*genai.FunctionCall
	for _, ev := range events {
		if ev.Partial == partial {
			calls = append(calls, ev.FunctionCalls()...)
		}
	}
	return calls
}

func modelTestFunctionResponses(events []*session.Event, name string) []map[string]any {
	var responses []map[string]any
	for _, ev := range events {
		for _, fr := range ev.FunctionResponses() {
			if fr.Name == name {
				responses = append(responses, fr.Response)
			}
		}
	}
	return responses
}

func TestNewModel_DefaultsMatchPython(t *testing.T) {
	t.Run("name", func(t *testing.T) {
		p := modelTestNewModel(t)
		if got, want := p.Name(), "reflect_retry_model_plugin"; got != want {
			t.Errorf("Name() = %q, want %q", got, want)
		}
	})
	t.Run("three retries then an error", func(t *testing.T) {
		p := modelTestNewModel(t)
		ctx := newModelTestContext(t, "I1", "root")
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count of failure 1 = %d, want 1", got)
		}
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 2 {
			t.Errorf("retry_count of failure 2 = %d, want 2", got)
		}
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 3 {
			t.Errorf("retry_count of failure 3 = %d, want 3", got)
		}
		modelTestWantExhausted(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()), "The model has failed consecutively 4 times and the retry limit has been exceeded")
	})
	t.Run("only MALFORMED_FUNCTION_CALL is an error", func(t *testing.T) {
		p := modelTestNewModel(t)
		ctx := newModelTestContext(t, "I1", "root")
		modelTestWantNil(t, modelTestCall(t, p, ctx, "m", &model.LLMResponse{ErrorCode: "SAFETY", FinishReason: genai.FinishReasonSafety}))
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count of MALFORMED_FUNCTION_CALL = %d, want 1", got)
		}
	})
	t.Run("counts per invocation", func(t *testing.T) {
		p := modelTestNewModel(t)
		if got := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count in I1 = %d, want 1", got)
		}
		if got := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I2", "root"), "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count in I2 = %d, want 1", got)
		}
	})
}

func TestNewModel_RejectsInvalidOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    []ModelOption
		wantErr string
	}{
		{
			name:    "negative max retries",
			opts:    []ModelOption{WithModelMaxRetries(-1)},
			wantErr: "retryandreflect: WithModelMaxRetries: must not be negative, got -1",
		},
		{
			name:    "empty scope",
			opts:    []ModelOption{WithModelTrackingScope("")},
			wantErr: `retryandreflect: WithModelTrackingScope: must be "invocation" or "global", got ""`,
		},
		{
			name:    "unknown scope",
			opts:    []ModelOption{WithModelTrackingScope("BOGUS")},
			wantErr: `retryandreflect: WithModelTrackingScope: must be "invocation" or "global", got "BOGUS"`,
		},
		{
			name:    "empty finish reason",
			opts:    []ModelOption{WithModelErrorFinishReasons(genai.FinishReasonMalformedFunctionCall, "")},
			wantErr: "retryandreflect: WithModelErrorFinishReasons: finish reason at index 1 must not be empty",
		},
		{
			name:    "max retries reported before scope",
			opts:    []ModelOption{WithModelMaxRetries(-1), WithModelTrackingScope("BOGUS")},
			wantErr: "retryandreflect: WithModelMaxRetries: must not be negative, got -1",
		},
		{
			name:    "scope reported before finish reasons",
			opts:    []ModelOption{WithModelErrorFinishReasons(""), WithModelTrackingScope("BOGUS")},
			wantErr: `retryandreflect: WithModelTrackingScope: must be "invocation" or "global", got "BOGUS"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewModel(tc.opts...)
			if p != nil {
				t.Errorf("NewModel() plugin = %v, want nil", p)
			}
			if err == nil || err.Error() != tc.wantErr {
				t.Errorf("NewModel() error = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestNewModel_LastOptionWinsAndNilIsSkipped(t *testing.T) {
	t.Run("last max retries wins", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelMaxRetries(5), WithModelMaxRetries(1))
		ctx := newModelTestContext(t, "I1", "root")
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count of failure 1 = %d, want 1", got)
		}
		modelTestWantExhausted(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()), "The model has failed consecutively 2 times and the retry limit has been exceeded")
	})
	t.Run("nil option", func(t *testing.T) {
		p := modelTestNewModel(t, nil)
		if got, want := p.Name(), "reflect_retry_model_plugin"; got != want {
			t.Errorf("Name() = %q, want %q", got, want)
		}
	})
	t.Run("empty name", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelPluginName("x"), WithModelPluginName(""))
		if got := p.Name(); got != "" {
			t.Errorf("Name() = %q, want %q", got, "")
		}
	})
	t.Run("custom name", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelPluginName("x"))
		if got := p.Name(); got != "x" {
			t.Errorf("Name() = %q, want %q", got, "x")
		}
	})
}

func TestModelErrorFinishReasons_ExplicitEmptyVersusDefault(t *testing.T) {
	t.Run("no option keeps the default", func(t *testing.T) {
		p := modelTestNewModel(t)
		if got := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", modelTestMalformed())); got != 1 {
			t.Errorf("retry_count = %d, want 1", got)
		}
	})
	var nilSlice []genai.FinishReason
	tests := []struct {
		name string
		opt  ModelOption
	}{
		{name: "no arguments", opt: WithModelErrorFinishReasons()},
		{name: "empty slice", opt: WithModelErrorFinishReasons([]genai.FinishReason{}...)},
		{name: "nil slice", opt: WithModelErrorFinishReasons(nilSlice...)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t, tc.opt)
			modelTestWantNil(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", modelTestMalformed()))
		})
	}
}

func TestModelErrorFinishReasons_CustomListReplacesDefault(t *testing.T) {
	t.Run("custom list replaces the default", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelErrorFinishReasons(genai.FinishReasonSafety))
		ctx := newModelTestContext(t, "I1", "root")
		call := modelTestReflectionCall(t, modelTestCall(t, p, ctx, "m", &model.LLMResponse{ErrorCode: "SAFETY", FinishReason: genai.FinishReasonSafety}))
		if got := call.Args["finish_reason"]; got != "SAFETY" {
			t.Errorf("finish_reason = %#v, want %q", got, "SAFETY")
		}
		modelTestWantNil(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))
	})
	t.Run("reason the SDK does not list", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelErrorFinishReasons("NEW_REASON"))
		call := modelTestReflectionCall(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", &model.LLMResponse{ErrorCode: "X", FinishReason: "NEW_REASON"}))
		if got := call.Args["finish_reason"]; got != "NEW_REASON" {
			t.Errorf("finish_reason = %#v, want %q", got, "NEW_REASON")
		}
	})
	t.Run("caller's slice is copied", func(t *testing.T) {
		reasons := []genai.FinishReason{genai.FinishReasonSafety}
		p := modelTestNewModel(t, WithModelErrorFinishReasons(reasons...))
		reasons[0] = genai.FinishReasonMalformedFunctionCall
		ctx := newModelTestContext(t, "I1", "root")
		modelTestWantNil(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))
		if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", &model.LLMResponse{ErrorCode: "SAFETY", FinishReason: genai.FinishReasonSafety})); got != 1 {
			t.Errorf("retry_count of SAFETY = %d, want 1", got)
		}
	})
}

func TestBeforeModel_OffersReservedToolWithoutDeclaringIt(t *testing.T) {
	testTool, err := functiontool.New(functiontool.Config{Name: "test_tool", Description: "a test tool"},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"ok": true}, nil })
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	llm := &modelTestLLM{name: "m", turns: [][]*model.LLMResponse{
		{{Content: genai.NewContentFromFunctionCall("test_tool", map[string]any{}, genai.RoleModel)}},
		{{Content: genai.NewContentFromText("done", genai.RoleModel)}},
	}}

	_, err = modelTestRun(t, llm, []tool.Tool{testTool}, []*plugin.Plugin{modelTestNewModel(t)}, agent.StreamingModeNone)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if got := llm.requestCount(); got != 2 {
		t.Fatalf("model requests = %d, want 2", got)
	}
	for i := range 2 {
		if got, want := llm.declarations(i), []string{"test_tool"}; !slices.Equal(got, want) {
			t.Errorf("declarations of request %d = %q, want %q", i, got, want)
		}
	}
}

func TestBeforeModel_CreatesToolMapWhenNil(t *testing.T) {
	p := modelTestNewModel(t)
	req := &model.LLMRequest{Model: "m"}

	got, err := p.BeforeModelCallback()(newModelTestContext(t, "I1", "root"), req)
	if got != nil || err != nil {
		t.Fatalf("BeforeModelCallback() = %v, %v; want nil, nil", got, err)
	}

	reserved, ok := req.Tools["adk_handle_model_error"].(tool.Tool)
	if !ok {
		t.Fatalf("req.Tools[%q] = %#v, want a tool.Tool", "adk_handle_model_error", req.Tools["adk_handle_model_error"])
	}
	if got := reserved.Name(); got != "adk_handle_model_error" {
		t.Errorf("reserved tool Name() = %q, want %q", got, "adk_handle_model_error")
	}
	if req.Config != nil {
		t.Errorf("req.Config = %+v, want nil", req.Config)
	}
}

func TestReservedTool_ReflectionGuidanceText(t *testing.T) {
	llm := &modelTestLLM{name: "m", turns: [][]*model.LLMResponse{
		{modelTestMalformed()},
		{modelTestMalformed()},
		{{Content: genai.NewContentFromText("done", genai.RoleModel)}},
	}}

	events, err := modelTestRun(t, llm, nil, []*plugin.Plugin{modelTestNewModel(t, WithModelMaxRetries(5))}, agent.StreamingModeNone)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	responses := modelTestFunctionResponses(events, "adk_handle_model_error")
	if len(responses) != 2 {
		t.Fatalf("reserved tool responses = %d, want 2", len(responses))
	}
	want := "\nThe call to the model failed.\n\n**Reflection Guidance:**\n- This is retry attempt **2** of **5**\n- Analyze the error and the arguments you provided. Do not repeat the exact same call.\n\nFormulate a new plan based on your analysis and try a corrected or different approach.\n    "
	if got := responses[1]["reflection_guidance"]; got != want {
		t.Errorf("reflection_guidance = %q, want %q", got, want)
	}
}

func TestAfterModel_ErrorCodeWithoutFinishReasonIsNotAnError(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
	}{
		{name: "no finish reason", resp: &model.LLMResponse{ErrorCode: "X"}},
		{name: "unlisted finish reason", resp: &model.LLMResponse{ErrorCode: "OTHER_CODE", FinishReason: genai.FinishReasonOther}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)
			modelTestWantNil(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", tc.resp))
		})
	}
}

func TestAfterModel_ReflectionResponseShape(t *testing.T) {
	p := modelTestNewModel(t)
	ctx := newModelTestContext(t, "I1", "root")
	ctx.ctx = platform.WithUUIDProvider(t.Context(), func() string { return "UUID-1" })

	r := modelTestCall(t, p, ctx, "m", &model.LLMResponse{
		ErrorCode:    "MALFORMED_FUNCTION_CALL",
		ErrorMessage: "boom",
		FinishReason: genai.FinishReasonMalformedFunctionCall,
	})

	call := modelTestReflectionCall(t, r)
	got := r.resp
	if got.ErrorCode != "" || got.ErrorMessage != "" || got.FinishReason != "" {
		t.Errorf("error fields = %q, %q, %q; want all empty", got.ErrorCode, got.ErrorMessage, got.FinishReason)
	}
	if got.Content.Role != "model" {
		t.Errorf("Content.Role = %q, want %q", got.Content.Role, "model")
	}
	if call.ID != "adk_handle_model_error_UUID-1" {
		t.Errorf("call ID = %q, want %q", call.ID, "adk_handle_model_error_UUID-1")
	}
	wantArgs := map[string]any{
		"error_details": "boom",
		"error_type":    "MALFORMED_FUNCTION_CALL",
		"finish_reason": "MALFORMED_FUNCTION_CALL",
		"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
		"retry_count":   1,
	}
	if !reflect.DeepEqual(call.Args, wantArgs) {
		t.Errorf("call Args = %#v, want %#v", call.Args, wantArgs)
	}
	if got, want := got.Content.Parts[0].ThoughtSignature, []byte("skip_thought_signature_validator"); !slices.Equal(got, want) {
		t.Errorf("ThoughtSignature = %q, want %q", got, want)
	}

	// Without an error message, error_details is JSON null, as in adk-python.
	args := modelTestReflectionCall(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())).Args
	if details, ok := args["error_details"]; !ok || details != nil {
		t.Errorf("error_details without a message = %#v (present %t), want untyped nil", details, ok)
	}
}

func TestAfterModel_ReflectionCallIDsAreFreshUUIDs(t *testing.T) {
	p := modelTestNewModel(t)
	ctx := newModelTestContext(t, "I1", "root")
	idPattern := regexp.MustCompile(`^adk_handle_model_error_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

	first := modelTestReflectionCall(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))
	second := modelTestReflectionCall(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))

	if !idPattern.MatchString(first.ID) {
		t.Errorf("first call ID = %q, want a match of %s", first.ID, idPattern)
	}
	if !idPattern.MatchString(second.ID) {
		t.Errorf("second call ID = %q, want a match of %s", second.ID, idPattern)
	}
	if first.ID == second.ID {
		t.Errorf("both call IDs are %q, want different IDs", first.ID)
	}
}

func TestAfterModel_UnconfiguredShapesResetCount(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
	}{
		{name: "finish reason without error code", resp: &model.LLMResponse{FinishReason: genai.FinishReasonMalformedFunctionCall}},
		{name: "unconfigured error", resp: &model.LLMResponse{ErrorCode: "RECITATION", FinishReason: genai.FinishReasonRecitation}},
		{name: "empty response", resp: &model.LLMResponse{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)
			ctx := newModelTestContext(t, "I1", "root")
			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
				t.Errorf("retry_count of failure 1 = %d, want 1", got)
			}
			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 2 {
				t.Errorf("retry_count of failure 2 = %d, want 2", got)
			}

			modelTestWantNil(t, modelTestCall(t, p, ctx, "m", tc.resp))

			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
				t.Errorf("retry_count after %s = %d, want 1", tc.name, got)
			}
		})
	}
}

func TestAfterModel_ModelNameNotRecorded(t *testing.T) {
	tests := []struct {
		name   string
		before *model.LLMRequest // nil: no before-model call
		resp   *model.LLMResponse
	}{
		{name: "error response without record", resp: modelTestMalformed()},
		{name: "success response without record", resp: &model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}},
		{name: "empty model name", before: &model.LLMRequest{Model: ""}, resp: modelTestMalformed()},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)
			ctx := newModelTestContext(t, "I1", "root")
			if tc.before != nil {
				if _, err := p.BeforeModelCallback()(ctx, tc.before); err != nil {
					t.Fatalf("BeforeModelCallback() failed: %v", err)
				}
			}

			r := modelTestAfter(p, ctx, tc.resp)

			if r.resp != nil {
				t.Errorf("after-model callback response = %+v, want nil", r.resp)
			}
			if want := `model of agent "root" not found`; r.err == nil || r.err.Error() != want {
				t.Fatalf("after-model callback error = %v, want %q", r.err, want)
			}
			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
				t.Errorf("retry_count of the next failure = %d, want 1", got)
			}
		})
	}
}

func TestAfterModel_PartialChunkLeavesCountsAlone(t *testing.T) {
	partialReserved := modelTestReservedCall()
	partialReserved.Partial = true
	tests := []struct {
		name    string
		partial *model.LLMResponse
	}{
		{name: "partial text", partial: &model.LLMResponse{Partial: true, Content: genai.NewContentFromText("let me", genai.RoleModel)}},
		{name: "partial error", partial: &model.LLMResponse{Partial: true, ErrorCode: "MALFORMED_FUNCTION_CALL", FinishReason: genai.FinishReasonMalformedFunctionCall}},
		{name: "partial reserved call", partial: partialReserved},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t, WithModelMaxRetries(5))
			ctx := newModelTestContext(t, "I1", "root")
			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
				t.Errorf("retry_count of failure 1 = %d, want 1", got)
			}

			modelTestWantNil(t, modelTestCall(t, p, ctx, "m", tc.partial))
			// An agent with no recorded model: a partial chunk needs no model name.
			modelTestWantNil(t, modelTestAfter(p, newModelTestContext(t, "I1", "unrecorded"), tc.partial))

			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 2 {
				t.Errorf("retry_count after the %s chunk = %d, want 2", tc.name, got)
			}
		})
	}
	t.Run("explicitly non-partial", func(t *testing.T) {
		p := modelTestNewModel(t)
		resp := modelTestMalformed()
		resp.PartialSet = true
		if got := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", resp)); got != 1 {
			t.Errorf("retry_count = %d, want 1", got)
		}
	})
}

func TestModelStreaming_RetryBudgetCountsTurnsNotChunks(t *testing.T) {
	turn := []*model.LLMResponse{
		{Partial: true, Content: genai.NewContentFromText("let me call", genai.RoleModel)},
		{Partial: true, ErrorCode: "MALFORMED_FUNCTION_CALL", ErrorMessage: "bad", FinishReason: genai.FinishReasonMalformedFunctionCall},
		{Content: genai.NewContentFromText("let me call", genai.RoleModel), ErrorCode: "MALFORMED_FUNCTION_CALL", ErrorMessage: "bad", FinishReason: genai.FinishReasonMalformedFunctionCall},
	}
	llm := &modelTestLLM{name: "m", turns: [][]*model.LLMResponse{turn, turn, turn, turn, turn}}

	events, err := modelTestRun(t, llm, nil, []*plugin.Plugin{modelTestNewModel(t, WithModelMaxRetries(2))}, agent.StreamingModeSSE)

	want := "Error in plugin 'reflect_retry_model_plugin' during 'after_model_callback' callback: The model has failed consecutively 3 times and the retry limit has been exceeded"
	if err == nil || err.Error() != want {
		t.Errorf("run error = %v, want %q", err, want)
	}
	if got := llm.requestCount(); got != 3 {
		t.Errorf("model requests = %d, want 3", got)
	}
	var counts []any
	for _, call := range modelTestFunctionCalls(events, false) {
		counts = append(counts, call.Args["retry_count"])
	}
	if want := []any{1, 2}; !reflect.DeepEqual(counts, want) {
		t.Errorf("retry_count of the final reflection calls = %v, want %v", counts, want)
	}
	if calls := modelTestFunctionCalls(events, true); len(calls) != 0 {
		t.Errorf("partial events hold %d function calls, want 0", len(calls))
	}
}

func TestAfterModel_AgentsOnOneModelShareCount(t *testing.T) {
	p := modelTestNewModel(t)

	a := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "a"), "m", modelTestMalformed()))
	b := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "b"), "m", modelTestMalformed()))

	if a != 1 || b != 2 {
		t.Errorf("retry_count of agents a, b = %d, %d; want 1, 2", a, b)
	}
}

func TestAfterModel_UsesModelOfLatestBeforeModelCall(t *testing.T) {
	p := modelTestNewModel(t)
	ctx := newModelTestContext(t, "I1", "root")

	first := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m1", modelTestMalformed()))
	second := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m2", modelTestMalformed()))
	third := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m1", modelTestMalformed()))

	if first != 1 || second != 1 || third != 2 {
		t.Errorf("retry_count on m1, m2, m1 = %d, %d, %d; want 1, 1, 2", first, second, third)
	}
}

func TestAfterModel_ExhaustionErrorCountsRealFailures(t *testing.T) {
	tests := []struct {
		maxRetries int
		failures   int
		wantErr    string // empty: the last failure is still a reflection
	}{
		{maxRetries: 0, failures: 1, wantErr: "The model has failed consecutively 1 times and the retry limit has been exceeded"},
		{maxRetries: 0, failures: 2, wantErr: "The model has failed consecutively 2 times and the retry limit has been exceeded"},
		{maxRetries: 2, failures: 2},
		{maxRetries: 2, failures: 3, wantErr: "The model has failed consecutively 3 times and the retry limit has been exceeded"},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("max %d failure %d", tc.maxRetries, tc.failures), func(t *testing.T) {
			p := modelTestNewModel(t, WithModelMaxRetries(tc.maxRetries))
			ctx := newModelTestContext(t, "I1", "root")
			var r modelTestResult
			for range tc.failures {
				r = modelTestCall(t, p, ctx, "m", modelTestMalformed())
			}
			if tc.wantErr == "" {
				if n := modelTestRetryCount(t, r); n != tc.failures {
					t.Errorf("retry_count = %d, want %d", n, tc.failures)
				}
				return
			}
			modelTestWantExhausted(t, r, tc.wantErr)
		})
	}
}

func TestModelRetryLimit_ErrorReachesCallerWrapped(t *testing.T) {
	llm := &modelTestLLM{name: "m", turns: [][]*model.LLMResponse{{modelTestMalformed()}, {modelTestMalformed()}}}
	p := modelTestNewModel(t, WithModelPluginName("custom_model_retry"), WithModelMaxRetries(1))

	_, err := modelTestRun(t, llm, nil, []*plugin.Plugin{p}, agent.StreamingModeNone)

	want := "Error in plugin 'custom_model_retry' during 'after_model_callback' callback: The model has failed consecutively 2 times and the retry limit has been exceeded"
	if err == nil || err.Error() != want {
		t.Errorf("run error = %v, want %q", err, want)
	}
	if !errors.Is(err, ErrModelRetryLimitExceeded) {
		t.Errorf("errors.Is(%v, ErrModelRetryLimitExceeded) = false, want true", err)
	}
	if got := llm.requestCount(); got != 2 {
		t.Errorf("model requests = %d, want 2", got)
	}
}

func TestAfterModel_ReservedCallWithinBudget(t *testing.T) {
	const reservedDetails = "Model attempted to call reserved tool adk_handle_model_error directly. This tool is reserved for framework use only. Do not call it."
	withOtherCall := modelTestReservedCall()
	withOtherCall.Content.Parts = append([]*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "Paris"}}}}, withOtherCall.Content.Parts...)
	withError := modelTestReservedCall()
	withError.ErrorCode = "MALFORMED_FUNCTION_CALL"
	withError.FinishReason = genai.FinishReasonMalformedFunctionCall
	tests := []struct {
		name string
		resp *model.LLMResponse
	}{
		{name: "reserved call alone", resp: modelTestReservedCall()},
		{name: "reserved call after another call", resp: withOtherCall},
		{name: "reserved call on an error response", resp: withError},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)

			call := modelTestReflectionCall(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", tc.resp))

			wantArgs := map[string]any{
				"error_details": reservedDetails,
				"error_type":    "RESERVED_TOOL_CALL",
				"finish_reason": "OTHER",
				"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
				"retry_count":   1,
			}
			if !reflect.DeepEqual(call.Args, wantArgs) {
				t.Errorf("call Args = %#v, want %#v", call.Args, wantArgs)
			}
		})
	}
}

func TestAfterModel_ReservedCallPastBudget(t *testing.T) {
	t.Run("error option on", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelMaxRetries(0))
		modelTestWantExhausted(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", modelTestReservedCall()), "The model has failed consecutively 1 times and the retry limit has been exceeded")
	})
	t.Run("error option off", func(t *testing.T) {
		p := modelTestNewModel(t, WithModelMaxRetries(0), WithModelErrorIfRetryExceeded(false))
		r := modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m", modelTestReservedCall())
		if r.err != nil {
			t.Fatalf("after-model callback error = %v, want nil", r.err)
		}
		want := &model.LLMResponse{
			ErrorCode:    "RESERVED_TOOL_CALL",
			ErrorMessage: "Model attempted to call reserved tool and retry limit was exceeded.",
		}
		if !reflect.DeepEqual(r.resp, want) {
			t.Errorf("after-model callback response = %+v, want %+v", r.resp, want)
		}
	})
}

func TestAfterModel_ReservedCallSharesBudgetWithModelErrors(t *testing.T) {
	p := modelTestNewModel(t, WithModelMaxRetries(5))
	ctx := newModelTestContext(t, "I1", "root")

	first := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))
	second := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestReservedCall()))
	third := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed()))

	if first != 1 || second != 2 || third != 3 {
		t.Errorf("retry_count of error, reserved call, error = %d, %d, %d; want 1, 2, 3", first, second, third)
	}
}

func TestScope_GlobalStillSeparatesModels(t *testing.T) {
	p := modelTestNewModel(t, WithModelTrackingScope(Global))

	first := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I1", "root"), "m1", modelTestMalformed()))
	second := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I2", "root"), "m2", modelTestMalformed()))
	third := modelTestRetryCount(t, modelTestCall(t, p, newModelTestContext(t, "I3", "root"), "m1", modelTestMalformed()))

	if first != 1 || second != 1 || third != 2 {
		t.Errorf("retry_count of I1/m1, I2/m2, I3/m1 = %d, %d, %d; want 1, 1, 2", first, second, third)
	}
}

func TestRunEnd_DropsModelRecordsOfThatRun(t *testing.T) {
	tests := []struct {
		name   string
		runEnd func(p *plugin.Plugin, ictx agent.InvocationContext)
	}{
		{name: "after run", runEnd: func(p *plugin.Plugin, ictx agent.InvocationContext) { p.AfterRunCallback()(ictx) }},
		{name: "on run error", runEnd: func(p *plugin.Plugin, ictx agent.InvocationContext) {
			p.OnRunErrorCallback()(ictx, errors.New("run failed"))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)
			ctx1 := newModelTestContext(t, "I1", "root")
			ctx2 := newModelTestContext(t, "I2", "root")
			for _, ctx := range []agent.Context{ctx1, ctx2} {
				if _, err := p.BeforeModelCallback()(ctx, &model.LLMRequest{Model: "m"}); err != nil {
					t.Fatalf("BeforeModelCallback() failed: %v", err)
				}
			}

			tc.runEnd(p, &modelTestRunContext{invocationID: "I1"})

			r1 := modelTestAfter(p, ctx1, modelTestMalformed())
			if want := `model of agent "root" not found`; r1.err == nil || r1.err.Error() != want {
				t.Errorf("after-model callback in I1 error = %v, want %q", r1.err, want)
			}
			if n := modelTestRetryCount(t, modelTestAfter(p, ctx2, modelTestMalformed())); n != 1 {
				t.Errorf("retry_count in I2 = %d, want 1", n)
			}
		})
	}
}

func TestRunEnd_KeepsCountsOnBothHooks(t *testing.T) {
	tests := []struct {
		name   string
		runEnd func(p *plugin.Plugin, ictx agent.InvocationContext)
	}{
		{name: "after run", runEnd: func(p *plugin.Plugin, ictx agent.InvocationContext) { p.AfterRunCallback()(ictx) }},
		{name: "on run error", runEnd: func(p *plugin.Plugin, ictx agent.InvocationContext) {
			p.OnRunErrorCallback()(ictx, errors.New("run failed"))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := modelTestNewModel(t)
			ctx := newModelTestContext(t, "I1", "root")
			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 1 {
				t.Errorf("retry_count before the run end = %d, want 1", got)
			}

			tc.runEnd(p, &modelTestRunContext{invocationID: "I1"})

			if got := modelTestRetryCount(t, modelTestCall(t, p, ctx, "m", modelTestMalformed())); got != 2 {
				t.Errorf("retry_count after the run end = %d, want 2", got)
			}
		})
	}
}

func TestAfterModel_ConcurrentFailuresLoseNoIncrement(t *testing.T) {
	const goroutines = 64
	p := modelTestNewModel(t, WithModelMaxRetries(1000))
	ctx := newModelTestContext(t, "I1", "root")
	if _, err := p.BeforeModelCallback()(ctx, &model.LLMRequest{Model: "m"}); err != nil {
		t.Fatalf("BeforeModelCallback() failed: %v", err)
	}

	results := make([]modelTestResult, goroutines)
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Go(func() {
			results[i] = modelTestAfter(p, ctx, modelTestMalformed())
		})
	}
	wg.Wait()

	var counts []int
	for _, r := range results {
		counts = append(counts, modelTestRetryCount(t, r))
	}
	slices.Sort(counts)
	var want []int
	for i := 1; i <= goroutines; i++ {
		want = append(want, i)
	}
	if !slices.Equal(counts, want) {
		t.Errorf("sorted retry counts = %v, want 1 to %d each once", counts, goroutines)
	}
}

func TestBeforeModel_ConcurrentRecordsAreRaceFree(t *testing.T) {
	const goroutines = 32
	p := modelTestNewModel(t)

	results := make([]modelTestResult, goroutines)
	var wg sync.WaitGroup
	for i := range goroutines {
		ctx := newModelTestContext(t, "I1", fmt.Sprintf("a%d", i))
		wg.Go(func() {
			if _, err := p.BeforeModelCallback()(ctx, &model.LLMRequest{Model: fmt.Sprintf("m%d", i)}); err != nil {
				results[i] = modelTestResult{err: err}
				return
			}
			results[i] = modelTestAfter(p, ctx, modelTestMalformed())
		})
	}
	wg.Wait()

	for i, r := range results {
		if got := modelTestRetryCount(t, r); got != 1 {
			t.Errorf("retry_count of agent a%d = %d, want 1", i, got)
		}
	}
}

func TestAfterModel_PastBudgetPassThroughStopsLaterCallbacks(t *testing.T) {
	var counted atomic.Int32
	counter, err := plugin.New(plugin.Config{
		Name: "counter",
		AfterModelCallback: func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
			counted.Add(1)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() failed: %v", err)
	}
	llm := &modelTestLLM{name: "m", turns: [][]*model.LLMResponse{{{
		ErrorCode:    "MALFORMED_FUNCTION_CALL",
		ErrorMessage: "boom",
		FinishReason: genai.FinishReasonMalformedFunctionCall,
	}}}}
	retry := modelTestNewModel(t, WithModelMaxRetries(0), WithModelErrorIfRetryExceeded(false))

	events, err := modelTestRun(t, llm, nil, []*plugin.Plugin{retry, counter}, agent.StreamingModeNone)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if got := counted.Load(); got != 0 {
		t.Errorf("after-model calls of the later plugin = %d, want 0", got)
	}
	if got := llm.requestCount(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
	if len(events) == 0 {
		t.Fatal("the run yielded no event")
	}
	last := events[len(events)-1]
	if last.ErrorCode != "MALFORMED_FUNCTION_CALL" || last.ErrorMessage != "boom" || last.FinishReason != genai.FinishReasonMalformedFunctionCall {
		t.Errorf("last event error fields = %q, %q, %q; want %q, %q, %q", last.ErrorCode, last.ErrorMessage, last.FinishReason,
			"MALFORMED_FUNCTION_CALL", "boom", genai.FinishReasonMalformedFunctionCall)
	}
}

func TestAfterModel_SuccessResetsOnlyItsOwnModel(t *testing.T) {
	p := modelTestNewModel(t)
	agentA := newModelTestContext(t, "I1", "a")
	agentB := newModelTestContext(t, "I1", "b")

	first := modelTestRetryCount(t, modelTestCall(t, p, agentA, "m1", modelTestMalformed()))
	modelTestWantNil(t, modelTestCall(t, p, agentB, "m2", &model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}))
	second := modelTestRetryCount(t, modelTestCall(t, p, agentA, "m1", modelTestMalformed()))

	if first != 1 || second != 2 {
		t.Errorf("retry_count of m1 before and after a success on m2 = %d, %d; want 1, 2", first, second)
	}
}
