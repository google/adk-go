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
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// packingFunctionTool is a function tool whose request hook packs itself into
// the request, as functiontool's does.
type packingFunctionTool struct {
	name string
	decl *genai.FunctionDeclaration
	long bool
	run  func(ctx agent.Context, args any) (map[string]any, error)
}

func (f *packingFunctionTool) Name() string        { return f.name }
func (f *packingFunctionTool) Description() string { return "" }
func (f *packingFunctionTool) IsLongRunning() bool { return f.long }
func (f *packingFunctionTool) Declaration() *genai.FunctionDeclaration {
	return f.decl
}

func (f *packingFunctionTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return f.run(ctx, args)
}

func (f *packingFunctionTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, f)
}

// newFunctionTool returns a packingFunctionTool named name with a declaration
// of the same name.
func newFunctionTool(name string) *packingFunctionTool {
	return &packingFunctionTool{name: name, decl: &genai.FunctionDeclaration{Name: name}}
}

// letterStreamTool is a streaming tool that yields "a" and "b".
type letterStreamTool struct{}

func (letterStreamTool) Name() string        { return "stream" }
func (letterStreamTool) Description() string { return "Streams letters" }
func (letterStreamTool) IsLongRunning() bool { return false }
func (letterStreamTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "stream", Description: "Streams letters"}
}

func (letterStreamTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if !yield("a", nil) {
			return
		}
		yield("b", nil)
	}
}

// nameOnlyTool has neither a Declaration nor a ProcessRequest method.
type nameOnlyTool struct{ name string }

func (b nameOnlyTool) Name() string        { return b.name }
func (b nameOnlyTool) Description() string { return "" }
func (b nameOnlyTool) IsLongRunning() bool { return false }

// declOnlyTool has a declaration and a request hook that packs it, and
// neither Run nor RunStream.
type declOnlyTool struct{}

func (declOnlyTool) Name() string        { return "decl_only" }
func (declOnlyTool) Description() string { return "" }
func (declOnlyTool) IsLongRunning() bool { return false }
func (declOnlyTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "decl_only"}
}

func (d declOnlyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, d)
}

// responseFlags answers the two optional response methods of the flow.
type responseFlags struct{ defers, display bool }

func (f responseFlags) DefersResponse() bool                   { return f.defers }
func (f responseFlags) DisplayResultOnSkipSummarization() bool { return f.display }

type flaggedFunctionTool struct {
	*packingFunctionTool
	responseFlags
}

type flaggedStreamTool struct {
	letterStreamTool
	responseFlags
}

type flaggedNameOnlyTool struct {
	nameOnlyTool
	responseFlags
}

// agentFunctionTool is a function tool that wraps an agent, as agenttool's
// tool does.
type agentFunctionTool struct {
	*packingFunctionTool
	agent agent.Agent
}

func (a *agentFunctionTool) WrappedAgent() agent.Agent { return a.agent }

// countingToolset returns lists[i] on its i-th call, or the last list after
// that, and records the contexts it was given.
type countingToolset struct {
	lists [][]tool.Tool
	calls int
	ctxs  []agent.ReadonlyContext
}

func (c *countingToolset) Name() string { return "counting" }
func (c *countingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	c.ctxs = append(c.ctxs, ctx)
	i := min(c.calls, len(c.lists)-1)
	c.calls++
	return c.lists[i], nil
}

// specialToolset has a method no wrapper has.
type specialToolset struct{ testToolset }

func (s *specialToolset) Special() string { return "special" }

// failingToolset fails to list with err, or with ctx.Err() when err is nil.
type failingToolset struct{ err error }

func (f *failingToolset) Name() string { return "failing" }
func (f *failingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	if f.err != nil {
		return []tool.Tool{newFunctionTool("partial")}, f.err
	}
	return []tool.Tool{newFunctionTool("partial")}, ctx.Err()
}

// hookToolset has a request hook that sets req.Model to "hooked" and counts
// its calls, or returns err.
type hookToolset struct {
	testToolset
	err    error
	calls  int
	gotReq *model.LLMRequest
	gotCtx agent.Context
}

func (h *hookToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	h.calls++
	h.gotReq = req
	h.gotCtx = ctx
	if h.err != nil {
		return h.err
	}
	req.Model = "hooked"
	return nil
}

// listTools returns ts's tools, failing the test on an error.
func listTools(t *testing.T, ts tool.Toolset) []tool.Tool {
	t.Helper()
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	return tools
}

// onlyTool returns the single tool of ts.
func onlyTool(t *testing.T, ts tool.Toolset) tool.Tool {
	t.Helper()
	tools := listTools(t, ts)
	if len(tools) != 1 {
		t.Fatalf("Tools() returned %d tools, want 1", len(tools))
	}
	return tools[0]
}

func toolNames(tools []tool.Tool) []string {
	names := []string{}
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	return names
}

// processRequest runs tl's request hook on req, as the flow does.
func processRequest(t *testing.T, tl tool.Tool, req *model.LLMRequest) error {
	t.Helper()
	rp, ok := tl.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatalf("%T does not implement RequestProcessor", tl)
	}
	return rp.ProcessRequest(nil, req)
}

func requestToolKeys(req *model.LLMRequest) []string {
	keys := []string{}
	for k := range req.Tools {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

func requestDeclarationNames(req *model.LLMRequest) []string {
	names := []string{}
	if req.Config == nil {
		return names
	}
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

func TestWithToolNamePrefix_ListsAgainOnEveryCall(t *testing.T) {
	x, y := newFunctionTool("x"), newFunctionTool("y")
	inner := &countingToolset{lists: [][]tool.Tool{{x}, {x, y}}}
	ts := tool.WithToolNamePrefix(inner, "p")
	ctx := &testContext{Context: t.Context()}

	var got [][]string
	for range 3 {
		tools, err := ts.Tools(ctx)
		if err != nil {
			t.Fatalf("Tools() error = %v", err)
		}
		got = append(got, toolNames(tools))
	}

	want := [][]string{{"p_x"}, {"p_x", "p_y"}, {"p_x", "p_y"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("listings mismatch (-want +got):\n%s", diff)
	}
	if inner.calls != 3 {
		t.Errorf("inner Tools calls = %d, want 3", inner.calls)
	}
	if inner.ctxs[0] != ctx {
		t.Errorf("inner Tools got context %v, want the caller's %v", inner.ctxs[0], ctx)
	}
	if diff := cmp.Diff([]string{"x", "y"}, []string{x.Name(), y.Name()}); diff != "" {
		t.Errorf("original names changed (-want +got):\n%s", diff)
	}
}

func TestWithToolNamePrefix_StreamingStandIn(t *testing.T) {
	tests := []struct {
		name       string
		original   tool.Tool
		wantName   string
		wantDesc   string
		wantChunks []string
	}{
		{
			name:       "streaming tool",
			original:   letterStreamTool{},
			wantName:   "p_stream",
			wantDesc:   "Streams letters",
			wantChunks: []string{"a", "b"},
		},
		{
			name:       "tool with Run and RunStream",
			original:   &dualTool{},
			wantName:   "p_dualTool",
			wantChunks: []string{"chunk"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"))

			st, ok := tl.(toolinternal.StreamingFunctionTool)
			if !ok {
				t.Fatalf("stand-in %T is not a StreamingFunctionTool", tl)
			}
			if _, ok := tl.(toolinternal.FunctionTool); ok {
				t.Errorf("stand-in %T is also a FunctionTool, so the flow could reach Run", tl)
			}
			if got := tl.Name(); got != tt.wantName {
				t.Errorf("Name() = %q, want %q", got, tt.wantName)
			}
			decl := st.Declaration()
			if decl.Name != tt.wantName || decl.Description != tt.wantDesc {
				t.Errorf("Declaration() = {Name: %q, Description: %q}, want {%q, %q}", decl.Name, decl.Description, tt.wantName, tt.wantDesc)
			}
			var chunks []string
			for chunk, err := range st.RunStream(&testContext{Context: t.Context()}, map[string]any{}) {
				if err != nil {
					t.Fatalf("RunStream() error = %v", err)
				}
				chunks = append(chunks, chunk)
			}
			if diff := cmp.Diff(tt.wantChunks, chunks); diff != "" {
				t.Errorf("RunStream() chunks mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestWithToolNamePrefix_StreamingStandInPassesItsContext(t *testing.T) {
	original := &ctxStreamTool{}
	tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{original}}, "p"))
	ctx := &testContext{Context: t.Context()}

	for _, err := range tl.(toolinternal.StreamingFunctionTool).RunStream(ctx, map[string]any{}) {
		if err != nil {
			t.Fatalf("RunStream() error = %v", err)
		}
	}

	if original.gotCtx != ctx {
		t.Errorf("original's RunStream got context %v, want the stand-in's %v", original.gotCtx, ctx)
	}
}

// ctxStreamTool is letterStreamTool that records the context RunStream gets.
type ctxStreamTool struct {
	letterStreamTool
	gotCtx agent.Context
}

func (c *ctxStreamTool) RunStream(ctx agent.Context, args any) iter.Seq2[string, error] {
	c.gotCtx = ctx
	return c.letterStreamTool.RunStream(ctx, args)
}

func TestWithToolNamePrefix_RequestOnlyStandIn(t *testing.T) {
	tests := []struct {
		name             string
		original         tool.Tool
		model            string
		wantName         string
		wantDeclaration  bool
		wantErr          string
		wantToolKeys     []string
		wantDeclNames    []string
		wantGoogleSearch int
		wantRegistered   bool
	}{
		{
			name:             "built-in on a Gemini model",
			original:         geminitool.GoogleSearch{},
			model:            "gemini-2.5-flash",
			wantName:         "p_google_search",
			wantToolKeys:     []string{},
			wantDeclNames:    []string{},
			wantGoogleSearch: 1,
		},
		{
			name:     "built-in error passes through",
			original: geminitool.GoogleSearch{},
			model:    "gpt-4",
			wantName: "p_google_search",
			wantErr:  "cannot process: Google search tool is not supported for model gpt-4",
		},
		{
			name:     "tool without a request hook",
			original: nameOnlyTool{name: "bare"},
			wantName: "p_bare",
			wantErr:  `tool "p_bare" does not implement RequestProcessor() method`,
		},
		{
			name:            "declared tool without Run",
			original:        declOnlyTool{},
			wantName:        "p_decl_only",
			wantDeclaration: true,
			wantToolKeys:    []string{"p_decl_only"},
			wantDeclNames:   []string{"p_decl_only"},
			wantRegistered:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"))
			if got := tl.Name(); got != tt.wantName {
				t.Errorf("Name() = %q, want %q", got, tt.wantName)
			}
			_, hasDecl := tl.(interface {
				Declaration() *genai.FunctionDeclaration
			})
			if hasDecl != tt.wantDeclaration {
				t.Errorf("stand-in has a Declaration method = %v, want %v", hasDecl, tt.wantDeclaration)
			}
			if _, ok := tl.(toolinternal.FunctionTool); ok {
				t.Errorf("stand-in %T is a FunctionTool, want no Run", tl)
			}

			req := &model.LLMRequest{Model: tt.model}
			err := processRequest(t, tl, req)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("ProcessRequest() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ProcessRequest() error = %v", err)
			}
			if diff := cmp.Diff(tt.wantToolKeys, requestToolKeys(req)); diff != "" {
				t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
			}
			if got := req.Tools[tt.wantName]; tt.wantRegistered && got != tl {
				t.Errorf("req.Tools[%q] = %T(%p), want the listed stand-in %T(%p)", tt.wantName, got, got, tl, tl)
			}
			if diff := cmp.Diff(tt.wantDeclNames, requestDeclarationNames(req)); diff != "" {
				t.Errorf("declarations mismatch (-want +got):\n%s", diff)
			}
			searches := 0
			for _, gt := range req.Config.Tools {
				if gt.GoogleSearch != nil {
					searches++
				}
			}
			if searches != tt.wantGoogleSearch {
				t.Errorf("GoogleSearch tools in request = %d, want %d", searches, tt.wantGoogleSearch)
			}
		})
	}
}

func TestWithToolNamePrefix_NilDeclaration(t *testing.T) {
	original := &packingFunctionTool{name: "nodecl"}
	tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{original}}, "p"))

	ft, ok := tl.(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("stand-in %T is not a FunctionTool", tl)
	}
	if decl := ft.Declaration(); decl != nil {
		t.Errorf("Declaration() = %+v, want nil", decl)
	}
	req := &model.LLMRequest{}
	if err := processRequest(t, tl, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
	if diff := cmp.Diff([]string{"p_nodecl"}, requestToolKeys(req)); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
	if n := len(req.Config.Tools); n != 0 {
		t.Errorf("len(req.Config.Tools) = %d, want 0", n)
	}
}

func TestWithToolNamePrefix_DeclarationIsACopy(t *testing.T) {
	stored := &genai.FunctionDeclaration{
		Name:        "t",
		Description: "Finds t",
		Parameters: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{"q": {Type: genai.TypeString}},
		},
	}
	original := &packingFunctionTool{name: "t", decl: stored}
	ts := tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{original}}, "p")
	tl := onlyTool(t, ts)
	_ = onlyTool(t, ts)

	decl := tl.(toolinternal.FunctionTool).Declaration()
	if decl.Name != "p_t" || decl.Description != "Finds t" {
		t.Errorf("stand-in Declaration() = {Name: %q, Description: %q}, want {%q, %q}", decl.Name, decl.Description, "p_t", "Finds t")
	}
	if decl.Parameters == nil || decl.Parameters.Properties["q"] == nil || decl.Parameters.Properties["q"].Type != genai.TypeString {
		t.Errorf("stand-in Declaration().Parameters = %+v, want the original's with q of type STRING", decl.Parameters)
	}
	req := &model.LLMRequest{}
	if err := processRequest(t, tl, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
	if diff := cmp.Diff([]string{"p_t"}, requestDeclarationNames(req)); diff != "" {
		t.Errorf("declarations mismatch (-want +got):\n%s", diff)
	}
	if stored.Name != "t" {
		t.Errorf("stored declaration Name = %q, want %q", stored.Name, "t")
	}
	if got := original.Declaration().Name; got != "t" {
		t.Errorf("original Declaration().Name = %q, want %q", got, "t")
	}
	if got := original.Name(); got != "t" {
		t.Errorf("original Name() = %q, want %q", got, "t")
	}
}

func TestWithToolNamePrefix_DuplicatePrefixedName(t *testing.T) {
	first := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{newFunctionTool("b_c")}}, "a"))
	second := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{newFunctionTool("c")}}, "a_b"))
	req := &model.LLMRequest{}
	if err := processRequest(t, first, req); err != nil {
		t.Fatalf("first ProcessRequest() error = %v", err)
	}

	err := processRequest(t, second, req)

	if err == nil || !strings.Contains(err.Error(), `duplicate tool: "a_b_c"`) {
		t.Errorf("second ProcessRequest() error = %v, want one containing %q", err, `duplicate tool: "a_b_c"`)
	}
	if diff := cmp.Diff([]string{"a_b_c"}, requestToolKeys(req)); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
	if req.Tools["a_b_c"] != first {
		t.Errorf("req.Tools[%q] = %v, want the first stand-in", "a_b_c", req.Tools["a_b_c"])
	}
	if diff := cmp.Diff([]string{"a_b_c"}, requestDeclarationNames(req)); diff != "" {
		t.Errorf("declarations mismatch (-want +got):\n%s", diff)
	}
}

// TestWithToolNamePrefix_EarlierDeclarationKeepsItsName checks that only the
// declaration the original's hook appends is renamed, so an agent that offers
// x both as its own tool and through a prefix gives the model x and p_x.
func TestWithToolNamePrefix_EarlierDeclarationKeepsItsName(t *testing.T) {
	tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{newFunctionTool("x")}}, "p"))
	req := &model.LLMRequest{}
	if err := toolutils.PackTool(req, newFunctionTool("x")); err != nil {
		t.Fatalf("PackTool() error = %v", err)
	}

	if err := processRequest(t, tl, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}

	if diff := cmp.Diff([]string{"x", "p_x"}, requestDeclarationNames(req)); diff != "" {
		t.Errorf("declarations mismatch (-want +got):\n%s", diff)
	}
}

func TestWithToolNamePrefix_ForwardsOptionalMethods(t *testing.T) {
	longFn := newFunctionTool("long")
	longFn.long = true
	tests := []struct {
		name        string
		original    tool.Tool
		wantDefers  bool
		wantDisplay bool
		wantLong    bool
	}{
		{
			name:       "function tool that defers and runs long",
			original:   &flaggedFunctionTool{longFn, responseFlags{defers: true}},
			wantDefers: true,
			wantLong:   true,
		},
		{
			name:        "streaming tool that displays",
			original:    &flaggedStreamTool{responseFlags: responseFlags{display: true}},
			wantDisplay: true,
		},
		{
			name:        "function tool with both flags",
			original:    &flaggedFunctionTool{newFunctionTool("both"), responseFlags{defers: true, display: true}},
			wantDefers:  true,
			wantDisplay: true,
		},
		{
			name:        "tool without a declaration that displays",
			original:    &flaggedNameOnlyTool{nameOnlyTool{name: "shown"}, responseFlags{display: true}},
			wantDisplay: true,
		},
		{
			name:     "tool with none of the methods",
			original: newFunctionTool("plain"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"))

			d, ok := tl.(toolinternal.ResponseDeferrer)
			gotDefers := ok && d.DefersResponse()
			s, ok := tl.(toolinternal.SkipSummarizationResultDisplayer)
			gotDisplay := ok && s.DisplayResultOnSkipSummarization()
			if gotDefers != tt.wantDefers || gotDisplay != tt.wantDisplay || tl.IsLongRunning() != tt.wantLong {
				t.Errorf("(DefersResponse, DisplayResultOnSkipSummarization, IsLongRunning) = (%v, %v, %v), want (%v, %v, %v)",
					gotDefers, gotDisplay, tl.IsLongRunning(), tt.wantDefers, tt.wantDisplay, tt.wantLong)
			}
			if _, ok := tl.(toolinternal.AgentWrapper); ok {
				t.Errorf("stand-in %T implements AgentWrapper, but its original does not", tl)
			}
		})
	}
}

// scheduling answers the flow's ResponseScheduling question.
type scheduling genai.FunctionResponseScheduling

func (s scheduling) ResponseScheduling() genai.FunctionResponseScheduling {
	return genai.FunctionResponseScheduling(s)
}

// rawFunctionTool is a function tool with a scheduling and a RunRaw method
// that returns raw.
type rawFunctionTool struct {
	*packingFunctionTool
	scheduling
	raw any
}

func (r *rawFunctionTool) RunRaw(agent.Context, any) (any, error) { return r.raw, nil }

type scheduledStreamTool struct {
	letterStreamTool
	scheduling
}

type scheduledAgentFunctionTool struct {
	*rawFunctionTool
	agent agent.Agent
}

func (a *scheduledAgentFunctionTool) WrappedAgent() agent.Agent { return a.agent }

func TestWithToolNamePrefix_ForwardsSchedulingAndRawResults(t *testing.T) {
	child, err := agent.New(agent.Config{Name: "child"})
	if err != nil {
		t.Fatal(err)
	}
	part := genai.NewPartFromBytes([]byte("png"), "image/png")
	plain := newFunctionTool("plain")
	plain.run = func(agent.Context, any) (map[string]any, error) { return map[string]any{"ran": "Run"}, nil }
	tests := []struct {
		name           string
		original       tool.Tool
		wantScheduling genai.FunctionResponseScheduling
		wantRaw        any
	}{
		{
			name:           "function tool with RunRaw",
			original:       &rawFunctionTool{newFunctionTool("raw"), scheduling(genai.FunctionResponseSchedulingSilent), part},
			wantScheduling: genai.FunctionResponseSchedulingSilent,
			wantRaw:        part,
		},
		{
			name:           "agent function tool with RunRaw",
			original:       &scheduledAgentFunctionTool{&rawFunctionTool{newFunctionTool("agent_raw"), scheduling(genai.FunctionResponseSchedulingWhenIdle), part}, child},
			wantScheduling: genai.FunctionResponseSchedulingWhenIdle,
			wantRaw:        part,
		},
		{
			name:     "function tool without RunRaw runs Run",
			original: plain,
			wantRaw:  map[string]any{"ran": "Run"},
		},
		{
			name:           "streaming tool",
			original:       &scheduledStreamTool{scheduling: scheduling(genai.FunctionResponseSchedulingInterrupt)},
			wantScheduling: genai.FunctionResponseSchedulingInterrupt,
		},
		{
			name:     "tool without a declaration or a scheduling",
			original: nameOnlyTool{name: "bare"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"))

			s, ok := tl.(tool.ResponseScheduler)
			if !ok {
				t.Fatalf("stand-in %T does not implement tool.ResponseScheduler", tl)
			}
			if got := s.ResponseScheduling(); got != tt.wantScheduling {
				t.Errorf("ResponseScheduling() = %q, want %q", got, tt.wantScheduling)
			}
			r, ok := tl.(interface {
				RunRaw(agent.Context, any) (any, error)
			})
			if tt.wantRaw == nil {
				if ok {
					t.Errorf("stand-in %T has RunRaw, but its original runs neither Run nor RunRaw", tl)
				}
				return
			}
			if !ok {
				t.Fatalf("stand-in %T has no RunRaw", tl)
			}
			got, err := r.RunRaw(nil, map[string]any{})
			if err != nil {
				t.Fatalf("RunRaw() error = %v", err)
			}
			if diff := cmp.Diff(tt.wantRaw, got); diff != "" {
				t.Errorf("RunRaw() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// agentStreamTool is a streaming tool that wraps an agent.
type agentStreamTool struct {
	letterStreamTool
	agent agent.Agent
}

func (a *agentStreamTool) WrappedAgent() agent.Agent { return a.agent }

func (a *agentStreamTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, a)
}

// agentDeclOnlyTool is a declared tool without Run that wraps an agent.
type agentDeclOnlyTool struct {
	declOnlyTool
	agent agent.Agent
}

func (a *agentDeclOnlyTool) WrappedAgent() agent.Agent { return a.agent }

func (a *agentDeclOnlyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, a)
}

// agentNameOnlyTool is a tool without a declaration that wraps an agent; its
// request hook registers it in req.Tools.
type agentNameOnlyTool struct {
	nameOnlyTool
	agent agent.Agent
}

func (a *agentNameOnlyTool) WrappedAgent() agent.Agent { return a.agent }

func (a *agentNameOnlyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = map[string]any{}
	}
	req.Tools[a.Name()] = a
	return nil
}

func TestWithToolNamePrefix_AgentToolStandIn(t *testing.T) {
	child, err := agent.New(agent.Config{Name: "child"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	tests := []struct {
		name          string
		original      tool.Tool
		wantKey       string
		wantFunction  bool
		wantStreaming bool
	}{
		{
			name:         "function tool",
			original:     &agentFunctionTool{packingFunctionTool: newFunctionTool("helper"), agent: child},
			wantKey:      "p_helper",
			wantFunction: true,
		},
		{
			name:          "streaming tool",
			original:      &agentStreamTool{agent: child},
			wantKey:       "p_stream",
			wantStreaming: true,
		},
		{
			name:     "declared tool without Run",
			original: &agentDeclOnlyTool{agent: child},
			wantKey:  "p_decl_only",
		},
		{
			name:     "tool without a declaration",
			original: &agentNameOnlyTool{nameOnlyTool: nameOnlyTool{name: "bare"}, agent: child},
			wantKey:  "p_bare",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"))

			aw, ok := tl.(toolinternal.AgentWrapper)
			if !ok {
				t.Fatalf("stand-in %T does not implement AgentWrapper", tl)
			}
			if aw.WrappedAgent() != child {
				t.Errorf("WrappedAgent() = %v, want the original's agent %v", aw.WrappedAgent(), child)
			}
			_, isFunction := tl.(toolinternal.FunctionTool)
			_, isStreaming := tl.(toolinternal.StreamingFunctionTool)
			if isFunction != tt.wantFunction || isStreaming != tt.wantStreaming {
				t.Errorf("stand-in %T is (FunctionTool, StreamingFunctionTool) = (%v, %v), want (%v, %v)",
					tl, isFunction, isStreaming, tt.wantFunction, tt.wantStreaming)
			}
			req := &model.LLMRequest{}
			if err := processRequest(t, tl, req); err != nil {
				t.Fatalf("ProcessRequest() error = %v", err)
			}
			if diff := cmp.Diff([]string{tt.wantKey}, requestToolKeys(req)); diff != "" {
				t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
			}
			if req.Tools[tt.wantKey] != tl {
				t.Errorf("req.Tools[%q] = %T, want the listed stand-in %T", tt.wantKey, req.Tools[tt.wantKey], tl)
			}
		})
	}
}

func TestWithToolNamePrefix_RunDelegatesToTheOriginal(t *testing.T) {
	ctx := &testContext{Context: t.Context()}
	calls := 0
	echo := newFunctionTool("echo")
	echo.run = func(gotCtx agent.Context, args any) (map[string]any, error) {
		if gotCtx != ctx {
			return nil, errors.New("wrong context")
		}
		calls++
		return map[string]any{"echo": args.(map[string]any)["q"], "n": calls}, nil
	}
	ts := tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{echo}}, "p")
	first := onlyTool(t, ts).(toolinternal.FunctionTool)
	second := onlyTool(t, ts).(toolinternal.FunctionTool)

	got1, err1 := first.Run(ctx, map[string]any{"q": "a"})
	got2, err2 := second.Run(ctx, map[string]any{"q": "b"})

	if err1 != nil || err2 != nil {
		t.Fatalf("Run() errors = %v, %v", err1, err2)
	}
	if diff := cmp.Diff(map[string]any{"echo": "a", "n": 1}, got1); diff != "" {
		t.Errorf("first Run() mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"echo": "b", "n": 2}, got2); diff != "" {
		t.Errorf("second Run() mismatch (-want +got):\n%s", diff)
	}

	errBoom := errors.New("boom")
	failing := newFunctionTool("failing")
	failing.run = func(agent.Context, any) (map[string]any, error) { return nil, errBoom }
	stand := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{failing}}, "p")).(toolinternal.FunctionTool)
	if _, err := stand.Run(ctx, map[string]any{}); !errors.Is(err, errBoom) || err.Error() != "boom" {
		t.Errorf("Run() error = %v, want %v unwrapped", err, errBoom)
	}
}

func TestWithToolNamePrefix_EmptyPrefixReturnsTheToolset(t *testing.T) {
	ts := &specialToolset{}

	got, ok := tool.WithToolNamePrefix(ts, "").(*specialToolset)

	if !ok || got != ts {
		t.Fatalf("WithToolNamePrefix(ts, \"\") = %v, want ts itself %v", got, ts)
	}
	if got.Special() != "special" {
		t.Errorf("Special() = %q, want %q", got.Special(), "special")
	}
}

func TestWithToolNamePrefix_KeepsTheToolsetName(t *testing.T) {
	ts := tool.WithToolNamePrefix(&countingToolset{}, "p")

	if got := ts.Name(); got != "counting" {
		t.Errorf("Name() = %q, want the wrapped toolset's %q", got, "counting")
	}
}

func TestWithToolNamePrefix_NilToolsetPanics(t *testing.T) {
	for _, prefix := range []string{"p", ""} {
		t.Run("prefix "+prefix, func(t *testing.T) {
			defer func() {
				if r := recover(); r != "toolset must not be nil" {
					t.Errorf("recovered %v, want %q", r, "toolset must not be nil")
				}
			}()
			tool.WithToolNamePrefix(nil, prefix)
		})
	}
}

func TestWithToolNamePrefix_ListingErrorPassesThrough(t *testing.T) {
	errBoom := errors.New("boom")
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	tests := []struct {
		name    string
		inner   tool.Toolset
		ctx     agent.ReadonlyContext
		wantErr error
		wantMsg string
	}{
		{name: "listing fails", inner: &failingToolset{err: errBoom}, ctx: &testContext{Context: t.Context()}, wantErr: errBoom, wantMsg: "boom"},
		{name: "context cancelled", inner: &failingToolset{}, ctx: &testContext{Context: cancelled}, wantErr: context.Canceled, wantMsg: "context canceled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tools, err := tool.WithToolNamePrefix(tt.inner, "p").Tools(tt.ctx)

			if tools != nil {
				t.Errorf("Tools() tools = %v, want nil", toolNames(tools))
			}
			if err != tt.wantErr || err.Error() != tt.wantMsg {
				t.Errorf("Tools() error = %v, want %v itself with text %q", err, tt.wantErr, tt.wantMsg)
			}
		})
	}
}

func TestWithToolNamePrefix_CloseAndUnwrap(t *testing.T) {
	errClose := errors.New("close failed")
	t.Run("closes inner", func(t *testing.T) {
		inner := &closingToolset{}
		if err := closeToolset(t, tool.WithToolNamePrefix(inner, "p")); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
		if inner.closes != 1 {
			t.Errorf("inner closes = %d, want 1", inner.closes)
		}
	})
	t.Run("returns inner error", func(t *testing.T) {
		inner := &closingToolset{err: errClose}
		if err := closeToolset(t, tool.WithToolNamePrefix(inner, "p")); !errors.Is(err, errClose) {
			t.Errorf("Close() = %v, want %v", err, errClose)
		}
	})
	t.Run("inner is not a closer", func(t *testing.T) {
		if err := closeToolset(t, tool.WithToolNamePrefix(&testToolset{}, "p")); err != nil {
			t.Errorf("Close() = %v, want nil", err)
		}
	})
	t.Run("unwraps to inner", func(t *testing.T) {
		inner := &closingToolset{}
		if got := unwrapToolset(t, tool.WithToolNamePrefix(inner, "p")); got != inner {
			t.Errorf("Unwrap() = %v, want the wrapped toolset %v", got, inner)
		}
	})
}

func TestWithToolNamePrefix_ComposesWithFilter(t *testing.T) {
	inner := &testToolset{tools: []tool.Tool{newFunctionTool("search"), newFunctionTool("fetch")}}
	tests := []struct {
		name string
		ts   tool.Toolset
		want []string
	}{
		{
			name: "prefix over filter matches original names",
			ts:   tool.WithToolNamePrefix(tool.FilterToolset(inner, tool.AllowedToolsPredicate([]string{"search"})), "docs"),
			want: []string{"docs_search"},
		},
		{
			name: "filter over prefix matches prefixed names",
			ts:   tool.FilterToolset(tool.WithToolNamePrefix(inner, "docs"), tool.AllowedToolsPredicate([]string{"docs_search"})),
			want: []string{"docs_search"},
		},
		{
			name: "filter over prefix does not match original names",
			ts:   tool.FilterToolset(tool.WithToolNamePrefix(inner, "docs"), tool.AllowedToolsPredicate([]string{"search"})),
			want: []string{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, toolNames(listTools(t, tt.ts))); diff != "" {
				t.Errorf("Tools() names mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestWithConfirmation_ProviderSeesTheNameOfTheOuterWrapper(t *testing.T) {
	type call struct {
		Name string
		Args any
	}
	tests := []struct {
		name string
		wrap func(tool.Toolset, tool.ConfirmationProvider) tool.Toolset
		want call
	}{
		{
			name: "confirmation over prefix",
			wrap: func(ts tool.Toolset, p tool.ConfirmationProvider) tool.Toolset {
				return tool.WithConfirmation(tool.WithToolNamePrefix(ts, "docs"), false, p)
			},
			want: call{"docs_search", map[string]any{"q": "go"}},
		},
		{
			name: "prefix over confirmation",
			wrap: func(ts tool.Toolset, p tool.ConfirmationProvider) tool.Toolset {
				return tool.WithToolNamePrefix(tool.WithConfirmation(ts, false, p), "docs")
			},
			want: call{"search", map[string]any{"q": "go"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []call
			provider := func(name string, args any) bool {
				calls = append(calls, call{name, args})
				return false
			}
			runs := 0
			search := newFunctionTool("search")
			search.run = func(agent.Context, any) (map[string]any, error) {
				runs++
				return map[string]any{}, nil
			}
			tl := onlyTool(t, tt.wrap(&testToolset{tools: []tool.Tool{search}}, provider))

			if _, err := tl.(toolinternal.FunctionTool).Run(&testContext{Context: t.Context()}, map[string]any{"q": "go"}); err != nil {
				t.Fatalf("Run() error = %v", err)
			}

			if diff := cmp.Diff([]call{tt.want}, calls); diff != "" {
				t.Errorf("provider calls mismatch (-want +got):\n%s", diff)
			}
			if runs != 1 {
				t.Errorf("original runs = %d, want 1", runs)
			}
		})
	}
}

func TestToolsetWrappers_ForwardRequestHook(t *testing.T) {
	wrappers := []struct {
		name string
		wrap func(tool.Toolset) tool.Toolset
	}{
		{
			name: "FilterToolset",
			wrap: func(ts tool.Toolset) tool.Toolset { return tool.FilterToolset(ts, tool.AllowedToolsPredicate(nil)) },
		},
		{
			name: "WithConfirmation",
			wrap: func(ts tool.Toolset) tool.Toolset { return tool.WithConfirmation(ts, false, nil) },
		},
		{
			name: "WithToolNamePrefix",
			wrap: func(ts tool.Toolset) tool.Toolset { return tool.WithToolNamePrefix(ts, "p") },
		},
		{
			name: "FilterToolset over WithToolNamePrefix",
			wrap: func(ts tool.Toolset) tool.Toolset {
				return tool.FilterToolset(tool.WithToolNamePrefix(ts, "p"), tool.AllowedToolsPredicate(nil))
			},
		},
	}
	for _, w := range wrappers {
		t.Run(w.name, func(t *testing.T) {
			t.Run("runs the inner hook", func(t *testing.T) {
				inner := &hookToolset{}
				req := &model.LLMRequest{}
				if err := toolsetProcessor(t, w.wrap(inner)).ProcessRequest(nil, req); err != nil {
					t.Fatalf("ProcessRequest() error = %v", err)
				}
				if req.Model != "hooked" || inner.calls != 1 || inner.gotReq != req {
					t.Errorf("after ProcessRequest: req.Model = %q, hook calls = %d, same request = %v; want %q, 1, true",
						req.Model, inner.calls, inner.gotReq == req, "hooked")
				}
			})
			t.Run("passes its context to the inner hook", func(t *testing.T) {
				inner := &hookToolset{}
				ctx := &testContext{Context: t.Context()}
				if err := toolsetProcessor(t, w.wrap(inner)).ProcessRequest(ctx, &model.LLMRequest{}); err != nil {
					t.Fatalf("ProcessRequest() error = %v", err)
				}
				if inner.gotCtx != ctx {
					t.Errorf("inner hook got context %v, want the wrapper's %v", inner.gotCtx, ctx)
				}
			})
			t.Run("returns the inner error", func(t *testing.T) {
				errHook := errors.New("hook failed")
				err := toolsetProcessor(t, w.wrap(&hookToolset{err: errHook})).ProcessRequest(nil, &model.LLMRequest{})
				if !errors.Is(err, errHook) || err.Error() != "hook failed" {
					t.Errorf("ProcessRequest() error = %v, want %v unwrapped", err, errHook)
				}
			})
			t.Run("inner without a hook", func(t *testing.T) {
				req := &model.LLMRequest{}
				if err := toolsetProcessor(t, w.wrap(&testToolset{})).ProcessRequest(nil, req); err != nil {
					t.Errorf("ProcessRequest() error = %v, want nil", err)
				}
				if diff := cmp.Diff(&model.LLMRequest{}, req); diff != "" {
					t.Errorf("request changed (-want +got):\n%s", diff)
				}
			})
		})
	}
}

// toolsetProcessor returns ts as the flow sees a toolset's request hook.
func toolsetProcessor(t *testing.T, ts tool.Toolset) toolinternal.RequestProcessor {
	t.Helper()
	rp, ok := ts.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatalf("%T does not implement RequestProcessor, so the flow skips the wrapped toolset's hook", ts)
	}
	return rp
}

func TestWithToolNamePrefix_ConcurrentUse(t *testing.T) {
	ts := tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{newFunctionTool("x"), newFunctionTool("y")}}, "p")
	const workers = 16
	type result struct {
		names []string
		decls []string
		err   error
	}
	results := make([]result, workers)

	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			tools, err := ts.Tools(nil)
			if err != nil {
				results[i] = result{err: err}
				return
			}
			req := &model.LLMRequest{}
			err = tools[0].(toolinternal.RequestProcessor).ProcessRequest(nil, req)
			results[i] = result{names: toolNames(tools), decls: requestDeclarationNames(req), err: err}
		})
	}
	wg.Wait()

	want := result{names: []string{"p_x", "p_y"}, decls: []string{"p_x"}}
	for i, got := range results {
		if diff := cmp.Diff(want, got, cmp.AllowUnexported(result{}), cmp.Comparer(func(a, b error) bool { return a == b })); diff != "" {
			t.Errorf("worker %d mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestWithToolNamePrefix_RunsOriginalProcessRequestExtras(t *testing.T) {
	x := &extrasTool{packingFunctionTool: newFunctionTool("x")}
	tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{x}}, "p"))
	req := &model.LLMRequest{}

	if err := processRequest(t, tl, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}

	if diff := cmp.Diff([]string{"p_x"}, requestToolKeys(req)); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
	if req.Tools["p_x"] != tl {
		t.Errorf("req.Tools[%q] = %T, want the listed stand-in", "p_x", req.Tools["p_x"])
	}
	if diff := cmp.Diff([]string{"p_x"}, requestDeclarationNames(req)); diff != "" {
		t.Errorf("declarations mismatch (-want +got):\n%s", diff)
	}
	si := req.Config.SystemInstruction
	if si == nil || len(si.Parts) != 1 || strings.Count(si.Parts[0].Text, "EXTRA") != 1 {
		t.Errorf("SystemInstruction = %+v, want one part holding EXTRA once", si)
	}
}

func TestWithToolNamePrefix_StandInPassesItsContextToTheOriginalsHook(t *testing.T) {
	original := &ctxRecordingTool{packingFunctionTool: newFunctionTool("x")}
	tl := onlyTool(t, tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{original}}, "p"))
	ctx := &testContext{Context: t.Context()}

	if err := tl.(toolinternal.RequestProcessor).ProcessRequest(ctx, &model.LLMRequest{}); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}

	if original.gotCtx != ctx {
		t.Errorf("original's hook got context %v, want the stand-in's %v", original.gotCtx, ctx)
	}
}

// ctxRecordingTool records the context its request hook gets, then packs
// itself.
type ctxRecordingTool struct {
	*packingFunctionTool
	gotCtx agent.Context
}

func (c *ctxRecordingTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	c.gotCtx = ctx
	return toolutils.PackTool(req, c)
}

// extrasTool packs itself and then adds an instruction, as loadmemorytool does.
type extrasTool struct{ *packingFunctionTool }

func (e *extrasTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if err := toolutils.PackTool(req, e); err != nil {
		return err
	}
	utils.AppendInstructions(req, "EXTRA")
	return nil
}

func TestWithConfirmation_OverPrefixedToolset_KeepsDeferral(t *testing.T) {
	tests := []struct {
		name     string
		original tool.Tool
		want     responseFlags
	}{
		{
			name:     "function tool that defers",
			original: &flaggedFunctionTool{newFunctionTool("deferring"), responseFlags{defers: true}},
			want:     responseFlags{defers: true},
		},
		{
			name:     "function tool that displays",
			original: &flaggedFunctionTool{newFunctionTool("showing"), responseFlags{display: true}},
			want:     responseFlags{display: true},
		},
		{
			name:     "streaming tool that defers",
			original: &flaggedStreamTool{responseFlags: responseFlags{defers: true}},
			want:     responseFlags{defers: true},
		},
		{
			name:     "streaming tool that displays",
			original: &flaggedStreamTool{responseFlags: responseFlags{display: true}},
			want:     responseFlags{display: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := tool.WithConfirmation(tool.WithToolNamePrefix(&testToolset{tools: []tool.Tool{tt.original}}, "p"), false, nil)
			tl := onlyTool(t, ts)

			d, ok := tl.(toolinternal.ResponseDeferrer)
			gotDefers := ok && d.DefersResponse()
			s, ok := tl.(toolinternal.SkipSummarizationResultDisplayer)
			gotDisplay := ok && s.DisplayResultOnSkipSummarization()
			if got := (responseFlags{defers: gotDefers, display: gotDisplay}); got != tt.want {
				t.Errorf("(DefersResponse, DisplayResultOnSkipSummarization) = (%v, %v), want (%v, %v)",
					got.defers, got.display, tt.want.defers, tt.want.display)
			}
		})
	}
}
