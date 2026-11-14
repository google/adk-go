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
	"context"
	"iter"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// packingTool is a function tool that registers itself through
// toolutils.PackTool, the way real tools do. A nil decl makes it a tool
// without a declaration. A non-empty instruction is appended to the system
// instruction before packing. runs counts the calls of Run.
type packingTool struct {
	name        string
	decl        *genai.FunctionDeclaration
	instruction string
	runs        atomic.Int32
}

func (p *packingTool) Name() string                            { return p.name }
func (p *packingTool) Description() string                     { return "packing tool" }
func (p *packingTool) IsLongRunning() bool                     { return false }
func (p *packingTool) Declaration() *genai.FunctionDeclaration { return p.decl }

func (p *packingTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if p.instruction != "" {
		utils.AppendInstructions(req, p.instruction)
	}
	return toolutils.PackTool(req, p)
}

func (p *packingTool) Run(agent.Context, any) (map[string]any, error) {
	p.runs.Add(1)
	return map[string]any{"ran": p.name}, nil
}

var _ tool.Tool = (*packingTool)(nil)

// recordedRequest is what callThenAnswerModel saw in one request.
type recordedRequest struct {
	toolKeys          []string
	declarationNames  []string
	systemInstruction string
}

// callThenAnswerModel answers its first request with a call of the function
// named call (none when call is empty) and every later request with the text
// "done". It records each request it receives.
type callThenAnswerModel struct {
	call     string
	requests []recordedRequest
}

func (m *callThenAnswerModel) Name() string { return "call-then-answer" }

func (m *callThenAnswerModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		first := len(m.requests) == 0
		m.requests = append(m.requests, record(req))
		if first && m.call != "" {
			yield(&model.LLMResponse{Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: m.call, Args: map[string]any{}}}},
			}}, nil)
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
	}
}

func record(req *model.LLMRequest) recordedRequest {
	r := recordedRequest{toolKeys: slices.Sorted(func(yield func(string) bool) {
		for k := range req.Tools {
			if !yield(k) {
				return
			}
		}
	})}
	if req.Config == nil {
		return r
	}
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			r.declarationNames = append(r.declarationNames, d.Name)
		}
	}
	if si := req.Config.SystemInstruction; si != nil {
		for _, p := range si.Parts {
			r.systemInstruction += p.Text
		}
	}
	return r
}

// runToFirstFunctionResponse runs f to the end in request/response mode and
// returns the response map of the first function response it yields.
func runToFirstFunctionResponse(t *testing.T, f *Flow) map[string]any {
	t.Helper()
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}},
	)
	var first map[string]any
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		if first != nil || ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				first = p.FunctionResponse.Response
				break
			}
		}
	}
	return first
}

func TestRun_UndeclaredToolIsNotCallable(t *testing.T) {
	noDecl := &packingTool{name: "no_decl"}
	f := &Flow{
		Model: &callThenAnswerModel{call: "no_decl"},
		Tools: []tool.Tool{noDecl},
	}

	resp := runToFirstFunctionResponse(t, f)

	if got := noDecl.runs.Load(); got != 0 {
		t.Errorf("no_decl Run called %d times, want 0", got)
	}
	errText, _ := resp["error"].(string)
	if want := "Invoking `no_decl()` failed as no tool with that name is available. The tools you can call are: none."; !strings.HasPrefix(errText, want) {
		t.Errorf("function response error = %q, want prefix %q", errText, want)
	}
}

func TestRun_UndeclaredToolStillProcessesRequest(t *testing.T) {
	m := &callThenAnswerModel{}
	f := &Flow{
		Model: m,
		Tools: []tool.Tool{
			&packingTool{name: "no_decl", instruction: "undeclared tool instruction"},
			&packingTool{name: "decl_tool", decl: &genai.FunctionDeclaration{Name: "decl_tool"}},
		},
	}

	runToFirstFunctionResponse(t, f)

	if len(m.requests) == 0 {
		t.Fatal("model received no request")
	}
	want := recordedRequest{
		toolKeys:          []string{"decl_tool"},
		declarationNames:  []string{"decl_tool"},
		systemInstruction: "undeclared tool instruction",
	}
	if diff := cmp.Diff(want, m.requests[0], cmp.AllowUnexported(recordedRequest{})); diff != "" {
		t.Errorf("first request mismatch (-want +got):\n%s", diff)
	}
}

func TestRun_ToolPackedByCallbackWithoutDeclarationIsNotCallable(t *testing.T) {
	cbTool := &packingTool{name: "cb_tool"}
	f := &Flow{
		Model: &callThenAnswerModel{call: "cb_tool"},
		BeforeModelCallbacks: []BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, toolutils.PackTool(req, cbTool)
			},
		},
	}

	resp := runToFirstFunctionResponse(t, f)

	if got := cbTool.runs.Load(); got != 0 {
		t.Errorf("cb_tool Run called %d times, want 0", got)
	}
	errText, _ := resp["error"].(string)
	if want := "Invoking `cb_tool()` failed as no tool with that name is available. The tools you can call are: none."; !strings.HasPrefix(errText, want) {
		t.Errorf("function response error = %q, want prefix %q", errText, want)
	}
}

func TestPreprocess_DropsOnlyEntriesThatReportNoDeclaration(t *testing.T) {
	seed := func(_ agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
		req.Tools = map[string]any{
			"declared": &packingTool{name: "declared", decl: &genai.FunctionDeclaration{Name: "declared"}},
			"opaque":   "an entry with no Declaration method",
		}
		return func(func(*session.Event, error) bool) {}
	}
	llmAgent := &mockLLMAgent{s: &State{Toolsets: []tool.Toolset{&mockRequestProcessorToolset{
		name: "packing_toolset",
		process: func(_ agent.Context, req *model.LLMRequest) error {
			return toolutils.PackTool(req, &packingTool{name: "undeclared"})
		},
	}}}}
	f := &Flow{RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{seed}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: llmAgent})
	req := &model.LLMRequest{}

	for _, err := range f.preprocess(ctx, req) {
		if err != nil {
			t.Fatalf("preprocess yielded error: %v", err)
		}
	}

	if diff := cmp.Diff([]string{"declared", "opaque"}, record(req).toolKeys); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
}
