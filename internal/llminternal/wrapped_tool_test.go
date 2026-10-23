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

package llminternal

import (
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type transparentTool struct{ tool.Tool }

func (w *transparentTool) Unwrap() tool.Tool { return w.Tool }

type callableTool struct {
	name string
	run  func(agent.Context, any) (map[string]any, error)
}

func (c *callableTool) Name() string        { return c.name }
func (c *callableTool) Description() string { return "test tool" }
func (c *callableTool) IsLongRunning() bool { return false }
func (c *callableTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: c.name, Description: c.Description()}
}

func (c *callableTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, c)
}

func (c *callableTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if c.run != nil {
		return c.run(ctx, args)
	}
	return map[string]any{"result": "completed"}, nil
}

type streamingCallableTool struct{ tool.Tool }

func (s *streamingCallableTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: s.Name()}
}

type declarationOnlyTool struct{ tool.Tool }

func (d *declarationOnlyTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: d.Name()}
}

func (s *streamingCallableTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, s)
}

func (d *declarationOnlyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, d)
}

func (s *streamingCallableTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for _, chunk := range []string{"first", "second"} {
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

type processingWrapper struct {
	transparentTool
	process func(agent.Context, *model.LLMRequest) error
}

func (w *processingWrapper) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return w.process(ctx, req)
}

type overridingFunctionTool struct {
	toolinternal.FunctionTool
	run func(agent.Context, any) (map[string]any, error)
}

func (w *overridingFunctionTool) Unwrap() tool.Tool { return w.FunctionTool }
func (w *overridingFunctionTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return w.run(ctx, args)
}

type deferringWrapper struct {
	transparentTool
	deferResponse bool
}

func (w *deferringWrapper) DefersResponse() bool { return w.deferResponse }

type displayingWrapper struct {
	transparentTool
	display bool
}

func (w *displayingWrapper) DisplayResultOnSkipSummarization() bool { return w.display }

var (
	_ toolinternal.FunctionTool             = (*callableTool)(nil)
	_ toolinternal.StreamingFunctionTool    = (*streamingCallableTool)(nil)
	_ tool.Wrapper                          = (*transparentTool)(nil)
	_ tool.RequestProcessor                 = (*processingWrapper)(nil)
	_ tool.ResponseDeferrer                 = (*deferringWrapper)(nil)
	_ tool.SkipSummarizationResultDisplayer = (*displayingWrapper)(nil)
)

func TestToolPreprocess_RequiresRequestProcessor(t *testing.T) {
	function := &callableTool{name: "callable"}
	streaming := &streamingCallableTool{Tool: function}
	for _, inner := range []tool.Tool{
		struct{ toolinternal.FunctionTool }{function},
		struct {
			toolinternal.StreamingFunctionTool
		}{streaming},
	} {
		for _, outer := range []tool.Tool{inner, &transparentTool{Tool: inner}} {
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			req := &model.LLMRequest{}
			if err := toolPreprocess(ctx, req, []tool.Tool{outer}); err == nil {
				t.Error("tool without RequestProcessor was accepted")
			}
			if len(req.Tools) != 0 || req.Config != nil {
				t.Error("tool without RequestProcessor was automatically registered")
			}
		}
	}
}

func TestToolPreprocess_WrappedProcessor(t *testing.T) {
	processErr := errors.New("processor failed")
	for _, tc := range []struct {
		name     string
		register bool
		err      error
	}{
		{name: "registers_tool", register: true},
		{name: "request_only"},
		{name: "returns_error", err: processErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inner := &callableTool{name: "processed"}
			calls := 0
			processor := &processingWrapper{
				transparentTool: transparentTool{Tool: inner},
				process: func(_ agent.Context, req *model.LLMRequest) error {
					calls++
					req.Model = "processor-model"
					if tc.err != nil {
						return tc.err
					}
					if tc.register {
						return toolutils.PackTool(req, inner)
					}
					return nil
				},
			}
			outer := &transparentTool{Tool: &transparentTool{Tool: processor}}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			req := &model.LLMRequest{}
			if err := toolPreprocess(ctx, req, []tool.Tool{outer}); !errors.Is(err, tc.err) {
				t.Error("processor error was not preserved")
			}
			if calls != 1 || req.Model != "processor-model" {
				t.Error("wrapped processor must run exactly once and preserve request changes")
			}
			if !tc.register {
				if len(req.Tools) != 0 || req.Config != nil {
					t.Error("processor that did not register a tool acquired a function declaration")
				}
				return
			}
			if req.Tools[outer.Name()] != outer {
				t.Error("inner registration bypassed the outer wrapper")
			}
			if req.Config == nil || len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
				t.Fatal("processor declaration was lost or duplicated")
			}
			if err := toolPreprocess(ctx, req, []tool.Tool{outer}); err == nil {
				t.Error("processor duplicate-registration error was discarded")
			}
		})
	}
}

func TestToolPreprocess_OuterProcessorWins(t *testing.T) {
	inner := &processingWrapper{
		transparentTool: transparentTool{Tool: &callableTool{name: "processed"}},
		process: func(agent.Context, *model.LLMRequest) error {
			t.Error("inner processor ran despite outer override")
			return nil
		},
	}
	outerCalls := 0
	outer := &processingWrapper{
		transparentTool: transparentTool{Tool: inner},
		process: func(agent.Context, *model.LLMRequest) error {
			outerCalls++
			return nil
		},
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	if err := toolPreprocess(ctx, &model.LLMRequest{}, []tool.Tool{outer}); err != nil {
		t.Fatal(err)
	}
	if outerCalls != 1 {
		t.Errorf("outer processor ran %d times, want 1", outerCalls)
	}
}

func TestToolPreprocess_WrappedProcessorPreservesRegistration(t *testing.T) {
	for _, existing := range []bool{false, true} {
		name := "nil_registration"
		if existing {
			name = "existing_registration"
		}
		t.Run(name, func(t *testing.T) {
			inner := &callableTool{name: "processed"}
			req := &model.LLMRequest{Tools: make(map[string]any)}
			var want any
			if existing {
				want = inner
				req.Tools[inner.Name()] = inner
			}
			processor := &processingWrapper{
				transparentTool: transparentTool{Tool: inner},
				process: func(_ agent.Context, req *model.LLMRequest) error {
					if !existing {
						req.Tools[inner.Name()] = nil
					}
					return nil
				},
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			if err := toolPreprocess(ctx, req, []tool.Tool{&transparentTool{Tool: processor}}); err != nil {
				t.Fatal(err)
			}
			if req.Tools[inner.Name()] != want {
				t.Error("processor replaced a registration it did not create")
			}
		})
	}
}

func TestToolPreprocess_MetadataOnly(t *testing.T) {
	outer := &transparentTool{Tool: &fakeTool{name: "metadata_only"}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	if err := toolPreprocess(ctx, &model.LLMRequest{}, []tool.Tool{outer}); err == nil {
		t.Error("tool without a declaration or request processor was accepted")
	}
}

func TestAppendTools_WrappedDeclaration(t *testing.T) {
	outer := &transparentTool{Tool: &transparentTool{Tool: &callableTool{name: "appended"}}}
	req := &model.LLMRequest{}
	if err := appendTools(req, outer); err != nil {
		t.Fatal(err)
	}
	if req.Tools[outer.Name()] != outer {
		t.Error("appended tool lost its outer wrapper")
	}
	if req.Config == nil || len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatal("appended wrapper lost its function declaration")
	}
	if req.Config.Tools[0].FunctionDeclarations[0].Name != outer.Name() {
		t.Error("appended declaration does not match the tool name")
	}
}

func callWrappedTool(t *testing.T, flow *Flow, wrapped tool.Tool, live agent.LiveSession) *session.Event {
	t.Helper()
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "wrapped-invocation",
		Agent:        &mockAgent{name: "test-agent"},
	})
	req := &model.LLMRequest{}
	if err := toolPreprocess(ctx, req, []tool.Tool{wrapped}); err != nil {
		t.Fatal(err)
	}
	registered, ok := req.Tools[wrapped.Name()].(tool.Tool)
	if !ok {
		t.Fatal("preprocessing did not register a tool")
	}
	response := &model.LLMResponse{Content: &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "wrapped-call", Name: wrapped.Name(), Args: map[string]any{},
		}}},
	}}
	event, err := flow.handleFunctionCalls(ctx, map[string]tool.Tool{wrapped.Name(): registered}, response, nil, live)
	if err != nil {
		t.Fatal(err)
	}
	return event
}

func TestHandleFunctionCalls_WrappedFunction(t *testing.T) {
	for _, fail := range []bool{false, true} {
		name := "success"
		if fail {
			name = "error"
		}
		t.Run(name, func(t *testing.T) {
			runErr := errors.New("tool failed")
			inner := &callableTool{name: "wrapped", run: func(agent.Context, any) (map[string]any, error) {
				if fail {
					return nil, runErr
				}
				return map[string]any{"result": "completed"}, nil
			}}
			outer := &transparentTool{Tool: &transparentTool{Tool: inner}}
			var before, after, onError int
			flow := &Flow{
				BeforeToolCallbacks: []BeforeToolCallback{func(_ agent.Context, got tool.Tool, _ map[string]any) (map[string]any, error) {
					before++
					if got != outer {
						t.Error("before callback received a different tool")
					}
					return nil, nil
				}},
				AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, got tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
					after++
					if got != outer {
						t.Error("after callback received a different tool")
					}
					return nil, nil
				}},
				OnToolErrorCallbacks: []OnToolErrorCallback{func(_ agent.Context, got tool.Tool, _ map[string]any, err error) (map[string]any, error) {
					onError++
					if got != outer || !errors.Is(err, runErr) {
						t.Error("error callback lost tool identity or original error")
					}
					return map[string]any{"result": "recovered"}, nil
				}},
			}
			event := callWrappedTool(t, flow, outer, nil)
			if event == nil || event.Content == nil || len(event.Content.Parts) != 1 || event.Content.Parts[0].FunctionResponse == nil {
				t.Fatal("wrapped call did not produce a function response")
			}
			want := "completed"
			wantErrors := 0
			if fail {
				want, wantErrors = "recovered", 1
			}
			if event.Content.Parts[0].FunctionResponse.Response["result"] != want {
				t.Error("wrapped call produced an unexpected result")
			}
			if before != 1 || after != 1 || onError != wantErrors {
				t.Errorf("callback counts = (%d, %d, %d), want (1, 1, %d)", before, after, onError, wantErrors)
			}
		})
	}
}

func TestHandleFunctionCalls_WrappedFunctionOverride(t *testing.T) {
	inner := &callableTool{name: "overridden", run: func(agent.Context, any) (map[string]any, error) {
		t.Error("inner handler ran despite outer override")
		return nil, nil
	}}
	overrideCalls := 0
	override := &overridingFunctionTool{FunctionTool: inner, run: func(agent.Context, any) (map[string]any, error) {
		overrideCalls++
		return map[string]any{"result": "override"}, nil
	}}
	event := callWrappedTool(t, &Flow{}, &transparentTool{Tool: override}, nil)
	if overrideCalls != 1 || event == nil {
		t.Error("outer function override was not called exactly once")
	}
}

func TestHandleFunctionCalls_WrappedInvalidContract(t *testing.T) {
	inner := &declarationOnlyTool{Tool: &callableTool{name: "invalid"}}
	outer := &transparentTool{Tool: inner}
	calls := 0
	flow := &Flow{OnToolErrorCallbacks: []OnToolErrorCallback{
		func(_ agent.Context, got tool.Tool, _ map[string]any, err error) (map[string]any, error) {
			calls++
			if got != outer {
				t.Error("invalid-contract callback lost the registered tool identity")
			}
			if err == nil || !strings.Contains(err.Error(), "tool.FunctionTool") || !strings.Contains(err.Error(), "tool.StreamingFunctionTool") {
				t.Error("invalid-contract diagnostic did not identify the callable interfaces")
			}
			return map[string]any{"result": "handled"}, nil
		},
	}}
	event := callWrappedTool(t, flow, outer, nil)
	if calls != 1 || event == nil || event.Content == nil || len(event.Content.Parts) != 1 || event.Content.Parts[0].FunctionResponse == nil {
		t.Fatal("invalid-contract error was not handled by the callback")
	}
	if event.Content.Parts[0].FunctionResponse.Response["result"] != "handled" {
		t.Error("invalid-contract callback response was discarded")
	}
}

func TestHandleFunctionCalls_WrappedStreaming(t *testing.T) {
	inner := &streamingCallableTool{Tool: &callableTool{name: "stream"}}
	outer := &transparentTool{Tool: &transparentTool{Tool: inner}}
	t.Run("aggregate", func(t *testing.T) {
		event := callWrappedTool(t, &Flow{}, outer, nil)
		if event == nil || event.Content == nil || len(event.Content.Parts) != 1 || event.Content.Parts[0].FunctionResponse == nil {
			t.Fatal("wrapped stream did not produce a function response")
		}
		if event.Content.Parts[0].FunctionResponse.Response["result"] != "firstsecond" {
			t.Error("wrapped streaming results were not aggregated")
		}
	})
	t.Run("live", func(t *testing.T) {
		chunks := make(chan agent.LiveRequest, 2)
		live := &mockLiveSession{sendFunc: func(req agent.LiveRequest) error {
			chunks <- req
			return nil
		}}
		event := callWrappedTool(t, &Flow{}, outer, live)
		if event == nil || event.Content == nil || len(event.Content.Parts) != 1 || event.Content.Parts[0].FunctionResponse == nil {
			t.Fatal("wrapped live stream did not produce a pending response")
		}
		if _, ok := event.Content.Parts[0].FunctionResponse.Response["status"]; !ok {
			t.Error("wrapped live stream did not report pending status")
		}
		for _, want := range []string{"Function stream returned: first", "Function stream returned: second"} {
			select {
			case req := <-chunks:
				if req.Content == nil || len(req.Content.Parts) != 1 || req.Content.Parts[0].Text != want {
					t.Error("wrapped live stream delivered an unexpected chunk")
				}
			case <-time.After(time.Second):
				t.Fatal("wrapped live stream did not deliver its chunks")
			}
		}
	})
}

func TestHandleFunctionCalls_WrappedResponseCapabilities(t *testing.T) {
	t.Run("deferral", func(t *testing.T) {
		inner := &callableTool{name: "deferred", run: func(agent.Context, any) (map[string]any, error) {
			return nil, nil
		}}
		deferrer := &deferringWrapper{transparentTool: transparentTool{Tool: inner}, deferResponse: true}
		if event := callWrappedTool(t, &Flow{}, &transparentTool{Tool: deferrer}, nil); event != nil {
			t.Error("wrapped deferred tool emitted a response")
		}
		override := &deferringWrapper{transparentTool: transparentTool{Tool: deferrer}, deferResponse: false}
		if event := callWrappedTool(t, &Flow{}, &transparentTool{Tool: override}, nil); event == nil {
			t.Error("outer deferral override did not emit a response")
		}
	})
	t.Run("skip_summarization", func(t *testing.T) {
		inner := &callableTool{name: "displayed", run: func(ctx agent.Context, _ any) (map[string]any, error) {
			ctx.Actions().SkipSummarization = true
			return map[string]any{"result": "visible result"}, nil
		}}
		displayer := &displayingWrapper{transparentTool: transparentTool{Tool: inner}, display: true}
		for _, display := range []bool{true, false} {
			var capability tool.Tool = displayer
			if !display {
				capability = &displayingWrapper{transparentTool: transparentTool{Tool: displayer}, display: false}
			}
			event := callWrappedTool(t, &Flow{}, &transparentTool{Tool: capability}, nil)
			if event == nil || event.Content == nil || !event.Actions.SkipSummarization {
				t.Fatal("wrapped tool lost its skip-summarization response")
			}
			wantParts := 1
			if display {
				wantParts = 2
			}
			if len(event.Content.Parts) != wantParts {
				t.Errorf("response parts = %d, want %d", len(event.Content.Parts), wantParts)
			} else if display && event.Content.Parts[1].Text != "visible result" {
				t.Error("wrapped tool result was not displayed")
			}
		}
	})
}

// An ordinary request processor may register a separate callable adapter.
type adapterProcessor struct {
	tool.Tool
	adapter toolutils.Tool
}

func (p *adapterProcessor) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, p.adapter)
}

func TestToolPreprocess_PreservesCallableAdapter(t *testing.T) {
	function := &callableTool{name: "adapter"}
	streaming := &streamingCallableTool{Tool: function}
	for _, tc := range []struct {
		name    string
		adapter tool.Tool
		result  string
	}{
		{"function", struct{ toolinternal.FunctionTool }{function}, "completed"},
		{"streaming", struct {
			toolinternal.StreamingFunctionTool
		}{streaming}, "firstsecond"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			processor := &adapterProcessor{Tool: tc.adapter, adapter: tc.adapter.(toolutils.Tool)}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			req := &model.LLMRequest{}
			if err := toolPreprocess(ctx, req, []tool.Tool{processor}); err != nil {
				t.Fatal("process adapter registration")
			}
			if req.Tools[tc.adapter.Name()] != tc.adapter {
				t.Fatal("request processor's callable adapter was replaced")
			}
			event := callWrappedTool(t, &Flow{}, processor, nil)
			if event == nil || event.Content == nil || len(event.Content.Parts) != 1 || event.Content.Parts[0].FunctionResponse == nil {
				t.Fatal("registered adapter did not produce a function response")
			}
			if event.Content.Parts[0].FunctionResponse.Response["result"] != tc.result {
				t.Error("execution-only adapter did not execute")
			}
		})
	}
}
