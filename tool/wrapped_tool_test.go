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

package tool_test

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// wrapper retains the decorator from the issue reproduction. Unwrap opts it
// into capability discovery without changing its existing forwarding methods.
type wrapper struct {
	tool.Tool
	ran *bool
	// reregister mirrors what tool.WithConfirmation does: after the inner tool
	// packs itself into req.Tools, overwrite the entry with the wrapper.
	reregister bool
}

var _ tool.Wrapper = (*wrapper)(nil)

func (w *wrapper) Unwrap() tool.Tool { return w.Tool }

func (w *wrapper) Declaration() *genai.FunctionDeclaration {
	if d, ok := w.Tool.(interface {
		Declaration() *genai.FunctionDeclaration
	}); ok {
		return d.Declaration()
	}
	return nil
}

func (w *wrapper) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	rp, ok := w.Tool.(interface {
		ProcessRequest(agent.Context, *model.LLMRequest) error
	})
	if !ok {
		return nil
	}
	_, existedBefore := req.Tools[w.Name()]
	if err := rp.ProcessRequest(ctx, req); err != nil {
		return err
	}
	if w.reregister && !existedBefore && req.Tools != nil && req.Tools[w.Name()] != nil {
		req.Tools[w.Name()] = w
	}
	return nil
}

func (w *wrapper) Run(ctx agent.Context, args any) (map[string]any, error) {
	*w.ran = true
	if r, ok := w.Tool.(interface {
		Run(agent.Context, any) (map[string]any, error)
	}); ok {
		return r.Run(ctx, args)
	}
	return map[string]any{"error": "inner tool is not runnable"}, nil
}

// A wrapper registered on the agent is the tool the run loop calls.
func TestWrapperIsDispatched(t *testing.T) {
	var wrapperRan, innerRan bool

	inner, err := functiontool.New[noArgs, noRes](functiontool.Config{
		Name: "probe", Description: "probe",
	}, func(ctx agent.Context, _ noArgs) (noRes, error) {
		innerRan = true
		return noRes{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}

	runAgent(t, &wrapper{Tool: inner, ran: &wrapperRan})

	t.Logf("inner tool ran = %v, wrapper Run called = %v", innerRan, wrapperRan)
	if !wrapperRan {
		t.Fatal("wrapper was registered on the agent but its Run was never called")
	}
	if !innerRan {
		t.Fatal("wrapper did not forward the call to the inner tool")
	}
}

// Wrapping a streaming tool keeps it streaming.
func TestWrapperKeepsStreamingTool(t *testing.T) {
	var wrapperRan, handlerRan bool

	inner, err := functiontool.NewStreaming[noArgs](functiontool.Config{
		Name: "probe", Description: "probe",
	}, func(ctx agent.Context, _ noArgs) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			handlerRan = true
			yield("chunk", nil)
		}
	})
	if err != nil {
		t.Fatalf("NewStreaming: %v", err)
	}

	resp := runAgent(t, &wrapper{Tool: inner, ran: &wrapperRan, reregister: true})

	t.Logf("streaming handler ran = %v, wrapper Run called = %v", handlerRan, wrapperRan)
	if !handlerRan {
		t.Fatal("wrapping a streaming tool stopped its handler from running")
	}
	if wrapperRan {
		t.Fatal("streaming call used the synchronous wrapper Run method")
	}
	if resp["result"] != "chunk" {
		t.Fatal("function response did not contain the streaming result")
	}
}

// TestWrapperOverridesStreamingTool is separate from the two issue reproductions
// so their original setup remains easy to compare.
func TestWrapperOverridesStreamingTool(t *testing.T) {
	innerRuns := 0
	inner, err := functiontool.NewStreaming[noArgs](functiontool.Config{
		Name: "probe", Description: "probe",
	}, func(agent.Context, noArgs) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			innerRuns++
			for _, chunk := range []string{"first", "second"} {
				if !yield(chunk, nil) {
					return
				}
			}
		}
	})
	if err != nil {
		t.Fatalf("functiontool.NewStreaming: %v", err)
	}
	capability, ok := tool.As[tool.StreamingFunctionTool](inner)
	if !ok {
		t.Fatal("tool has no streaming capability")
	}
	streaming := &streamingRunnerWrapper{StreamingFunctionTool: capability}
	var wrapperRan bool
	outer := transparentTool{Tool: &wrapper{Tool: streaming, ran: &wrapperRan}}
	resp := runAgent(t, outer)
	if innerRuns != 1 || streaming.runs != 1 || wrapperRan {
		t.Fatal("streaming call did not execute the override and inner handler exactly once")
	}
	if resp["result"] != "wrapped:firstwrapped:second" {
		t.Fatal("function response did not contain the overridden streaming result")
	}
}

type streamingRunnerWrapper struct {
	tool.StreamingFunctionTool
	runs int
}

var _ tool.Wrapper = (*streamingRunnerWrapper)(nil)

func (w *streamingRunnerWrapper) Unwrap() tool.Tool { return w.StreamingFunctionTool }

func (w *streamingRunnerWrapper) RunStream(ctx agent.Context, args any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		w.runs++
		for chunk, err := range w.StreamingFunctionTool.RunStream(ctx, args) {
			if !yield("wrapped:"+chunk, err) {
				return
			}
		}
	}
}

// --- harness ---

type (
	noArgs struct{}
	noRes  struct{}
)

type callOnceLLM struct{ n int }

func (f *callOnceLLM) Name() string { return "fake" }

func (f *callOnceLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	f.n++
	first := f.n == 1
	return func(yield func(*model.LLMResponse, error) bool) {
		if first {
			yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{Name: "probe", Args: map[string]any{}}},
			}}}, nil)
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
	}
}

func runAgent(t *testing.T, tl tool.Tool) map[string]any {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name: "probe", Model: &callOnceLLM{}, Instruction: "x", Tools: []tool.Tool{tl},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "probe", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()

	var resp map[string]any
	for e, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("runner returned an unexpected error")
		}
		if e == nil || e.LLMResponse.Content == nil {
			continue
		}
		for _, p := range e.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				resp = p.FunctionResponse.Response
			}
		}
	}
	if resp == nil {
		t.Fatal("runner did not produce a function response")
	}
	return resp
}
