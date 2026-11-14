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
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/plugininternal/plugincontext"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/tool/toolutils"
)

var (
	tcPNG  = genai.NewPartFromBytes([]byte("png"), "image/png")
	tcJPEG = genai.NewPartFromBytes([]byte("jpeg"), "image/jpeg")
)

func tcPNGPart() *genai.FunctionResponsePart {
	return &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte("png"), MIMEType: "image/png"}}
}

func tcJPEGPart() *genai.FunctionResponsePart {
	return &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte("jpeg"), MIMEType: "image/jpeg"}}
}

// The not-found texts for a call of missing_fn, with list naming the tools.
func tcShortText(name, list string) string {
	return "Invoking `" + name + "()` failed as no tool with that name is available. The tools you can call are: " + list + ". You could retry, but it is IMPORTANT that you only call a tool from that list."
}

func tcLongText(name, list string) string {
	return "tool '" + name + "' not found.\nAvailable tools: " + list + "\n\nPossible causes:\n  1. LLM hallucinated the function name - review agent instruction clarity\n  2. Tool not registered - verify agent.tools list\n  3. Name mismatch - check for typos\n\nSuggested fixes:\n  - Review agent instruction to ensure tool usage is clear\n  - Verify tool is included in agent.tools list\n  - Check for typos in function name"
}

// tcCallsModel answers its first request with one model turn holding calls,
// and every later request with the text "done".
type tcCallsModel struct {
	calls []*genai.FunctionCall
	mu    sync.Mutex
	n     int
}

func tcModel(name string) *tcCallsModel {
	return &tcCallsModel{calls: []*genai.FunctionCall{{ID: "call-1", Name: name, Args: map[string]any{}}}}
}

func (m *tcCallsModel) Name() string { return "calls-model" }

func (m *tcCallsModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		first := m.n == 0
		m.n++
		m.mu.Unlock()
		if !first {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		parts := make([]*genai.Part, len(m.calls))
		for i, c := range m.calls {
			parts[i] = &genai.Part{FunctionCall: c}
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}, nil)
	}
}

// tcPlugins is a plugin manager whose tool callbacks are the given functions;
// a nil function answers nothing.
type tcPlugins struct {
	before func(agent.Context, tool.Tool, map[string]any) (map[string]any, error)
	after  func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error)
	onErr  func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error)
}

func (*tcPlugins) RunBeforeModelCallback(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
	return nil, nil
}

func (*tcPlugins) RunAfterModelCallback(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
	return nil, nil
}

func (*tcPlugins) RunOnModelErrorCallback(agent.Context, *model.LLMRequest, error) (*model.LLMResponse, error) {
	return nil, nil
}

func (p *tcPlugins) RunBeforeToolCallback(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if p.before == nil {
		return nil, nil
	}
	return p.before(ctx, t, args)
}

func (p *tcPlugins) RunAfterToolCallback(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if p.after == nil {
		return nil, nil
	}
	return p.after(ctx, t, args, result, err)
}

func (p *tcPlugins) RunOnToolErrorCallback(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
	if p.onErr == nil {
		return nil, nil
	}
	return p.onErr(ctx, t, args, err)
}

// tcRunIn runs f to its end in request/response mode over base, with plugins
// installed when not nil, and returns the events it yielded.
func tcRunIn(t *testing.T, base context.Context, f *Flow, plugins *tcPlugins) []*session.Event {
	t.Helper()
	base = runconfig.ToContext(base, &runconfig.RunConfig{})
	if plugins != nil {
		base = context.WithValue(base, plugincontext.PluginManagerCtxKey, pluginManager(plugins))
	}
	ctx := icontext.NewInvocationContext(base, icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}})
	var events []*session.Event
	for ev, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return events
}

func tcRun(t *testing.T, f *Flow, plugins *tcPlugins) []*session.Event {
	t.Helper()
	return tcRunIn(t, t.Context(), f, plugins)
}

// tcResponseEvent returns the one event of events that holds function
// responses.
func tcResponseEvent(t *testing.T, events []*session.Event) *session.Event {
	t.Helper()
	var found []*session.Event
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				found = append(found, ev)
				break
			}
		}
	}
	if len(found) != 1 {
		t.Fatalf("got %d function response events, want 1", len(found))
	}
	return found[0]
}

// tcOnlyResponse returns the function response of the one function response
// event of events.
func tcOnlyResponse(t *testing.T, events []*session.Event) *genai.FunctionResponse {
	t.Helper()
	return tcResponseEvent(t, events).LLMResponse.Content.Parts[0].FunctionResponse
}

// tcHandle runs calls through f.handleFunctionCalls with tools and returns the
// merged response event, nil when no call produced one.
func tcHandle(t *testing.T, f *Flow, tools map[string]tool.Tool, confirmations map[string]*toolconfirmation.ToolConfirmation, calls ...*genai.FunctionCall) *session.Event {
	t.Helper()
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}})
	parts := make([]*genai.Part, len(calls))
	for i, c := range calls {
		parts[i] = &genai.Part{FunctionCall: c}
	}
	ev, err := f.handleFunctionCalls(ctx, tools, &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}, confirmations, nil)
	if err != nil {
		t.Fatalf("handleFunctionCalls() error = %v", err)
	}
	return ev
}

func tcCall(name string) *genai.FunctionCall {
	return &genai.FunctionCall{ID: "call-1", Name: name, Args: map[string]any{}}
}

// tcMust returns a function that returns the tool it is given, failing t on
// an error.
func tcMust(t *testing.T) func(tool.Tool, error) tool.Tool {
	return func(tl tool.Tool, err error) tool.Tool {
		t.Helper()
		if err != nil {
			t.Fatalf("building tool: %v", err)
		}
		return tl
	}
}

// tcValueTool is a function tool named name whose handler returns v.
func tcValueTool(t *testing.T, name string, v any) tool.Tool {
	t.Helper()
	return tcMust(t)(functiontool.New(functiontool.Config{Name: name, Description: "returns a value"},
		func(agent.Context, struct{}) (any, error) { return v, nil }))
}

// tcToolsetOf lists tools on every call.
type tcToolsetOf []tool.Tool

func (tcToolsetOf) Name() string                                       { return "tc_toolset" }
func (l tcToolsetOf) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return l, nil }

func tcToolset(tools ...tool.Tool) tool.Toolset { return tcToolsetOf(tools) }

// tcWrap returns the one tool ts lists.
func tcWrap(t *testing.T, ts tool.Toolset) tool.Tool {
	t.Helper()
	tools, err := ts.Tools(nil)
	if err != nil || len(tools) != 1 {
		t.Fatalf("Tools() = %v, %v; want one tool", tools, err)
	}
	return tools[0]
}

// tcDeclared is a declared tool that packs itself and implements no Run, so
// the flow cannot run it.
type tcDeclared struct{ name string }

func (d *tcDeclared) Name() string        { return d.name }
func (d *tcDeclared) Description() string { return "declared" }
func (d *tcDeclared) IsLongRunning() bool { return false }
func (d *tcDeclared) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: d.name}
}

func (d *tcDeclared) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, d)
}

// tcDisplayer is a function tool that skips summarization and asks for its
// result to be shown.
type tcDisplayer struct{ tcDeclared }

func (*tcDisplayer) DisplayResultOnSkipSummarization() bool { return true }

func (d *tcDisplayer) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, d)
}

func (*tcDisplayer) Run(ctx agent.Context, _ any) (map[string]any, error) {
	ctx.Actions().SkipSummarization = true
	return map[string]any{"chart": tcPNG, "summary": "up 3%"}, nil
}

// tcDeferrer defers its response and returns raw from RunRaw.
type tcDeferrer struct {
	tcDeclared
	raw any
}

func (*tcDeferrer) DefersResponse() bool { return true }

func (d *tcDeferrer) Run(agent.Context, any) (map[string]any, error) {
	m, _ := d.raw.(map[string]any)
	return m, nil
}

func (d *tcDeferrer) RunRaw(agent.Context, any) (any, error) { return d.raw, nil }

// tcScheduled is a function tool outside functiontool with a scheduling.
type tcScheduled struct {
	tcDeclared
	scheduling genai.FunctionResponseScheduling
}

func (*tcScheduled) Run(agent.Context, any) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

func (s *tcScheduled) ResponseScheduling() genai.FunctionResponseScheduling { return s.scheduling }

func tcBuild(v any) *genai.FunctionResponse {
	return buildFunctionResponse(nil, "call-1", "media_tool", normalizeRawResult(v), "").FunctionResponse
}

func TestMediaExtraction_EdgeParts(t *testing.T) {
	both := genai.Part{
		InlineData: &genai.Blob{Data: []byte("inline"), MIMEType: "image/png", DisplayName: "drop-me.png"},
		FileData:   &genai.FileData{FileURI: "gs://b/f", MIMEType: "image/jpeg", DisplayName: "drop-file.jpg"},
	}
	inline := []*genai.FunctionResponsePart{{InlineData: &genai.FunctionResponseBlob{Data: []byte("inline"), MIMEType: "image/png"}}}
	text := genai.NewPartFromText("hi")
	nilData := &genai.Part{InlineData: &genai.Blob{Data: nil, MIMEType: "image/png"}}
	noMIME := &genai.Part{InlineData: &genai.Blob{Data: []byte("x"), MIMEType: ""}}
	noURI := &genai.Part{FileData: &genai.FileData{FileURI: "", MIMEType: "image/png"}}
	tests := []struct {
		name         string
		v            any
		wantResponse map[string]any
		wantParts    []*genai.FunctionResponsePart
	}{
		{
			name:         "empty inline data is media",
			v:            &genai.Part{InlineData: &genai.Blob{Data: []byte{}, MIMEType: "image/png"}},
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{{InlineData: &genai.FunctionResponseBlob{Data: []byte{}, MIMEType: "image/png"}}},
		},
		{name: "inline data wins over file data", v: &both, wantResponse: map[string]any{}, wantParts: inline},
		{name: "part value", v: both, wantResponse: map[string]any{}, wantParts: inline},
		{
			name:         "file data drops display name",
			v:            &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/f", MIMEType: "image/jpeg", DisplayName: "chart.png"}},
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{{FileData: &genai.FunctionResponseFileData{FileURI: "gs://b/f", MIMEType: "image/jpeg"}}},
		},
		{name: "text part", v: text, wantResponse: map[string]any{"result": text}},
		{name: "inline data without bytes", v: nilData, wantResponse: map[string]any{"result": nilData}},
		{name: "inline data without MIME type", v: noMIME, wantResponse: map[string]any{"result": noMIME}},
		{name: "file data without URI", v: noURI, wantResponse: map[string]any{"result": noURI}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tcBuild(tc.v)

			want := &genai.FunctionResponse{ID: "call-1", Name: "media_tool", Response: tc.wantResponse, Parts: tc.wantParts}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("buildFunctionResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMediaExtraction_ContainerEdges(t *testing.T) {
	pz := genai.NewPartFromBytes([]byte("z"), "image/png")
	pa := genai.NewPartFromBytes([]byte("a"), "image/png")
	pm := genai.NewPartFromBytes([]byte("m"), "image/png")
	inline := func(data string) *genai.FunctionResponsePart {
		return &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte(data), MIMEType: "image/png"}}
	}
	type chartHolder struct{ Chart *genai.Part }
	tests := []struct {
		name         string
		v            any
		wantResponse map[string]any
		wantParts    []*genai.FunctionResponsePart
	}{
		{
			name:         "map keys in sorted order",
			v:            map[string]any{"z_img": pz, "a_img": pa, "m_img": pm},
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{inline("a"), inline("m"), inline("z")},
		},
		{
			name:         "slice of media empties to a map",
			v:            []*genai.Part{tcPNG, tcJPEG},
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart(), tcJPEGPart()},
		},
		{
			name:         "array of media",
			v:            [2]*genai.Part{tcPNG, tcJPEG},
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart(), tcJPEGPart()},
		},
		{name: "bytes are a value", v: []byte("raw-bytes"), wantResponse: map[string]any{"result": []byte("raw-bytes")}},
		{name: "raw JSON is a value", v: json.RawMessage(`{"a":1}`), wantResponse: map[string]any{"result": json.RawMessage(`{"a":1}`)}},
		{
			name:         "inner container without media is kept",
			v:            map[string]any{"images": []any{}, "ok": true},
			wantResponse: map[string]any{"images": []any{}, "ok": true},
		},
		{
			name:         "inner container keeps its plain values",
			v:            map[string]any{"items": []any{tcPNG, "keep"}, "ok": true},
			wantResponse: map[string]any{"items": []any{"keep"}, "ok": true},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart()},
		},
		{
			name:         "struct is not searched",
			v:            chartHolder{Chart: tcPNG},
			wantResponse: map[string]any{"result": chartHolder{Chart: tcPNG}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tcBuild(tc.v)

			want := &genai.FunctionResponse{ID: "call-1", Name: "media_tool", Response: tc.wantResponse, Parts: tc.wantParts}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("buildFunctionResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMediaExtraction_ListOfMapsWrapperIdentityAndInPlaceCallbackMutation(t *testing.T) {
	tests := []struct {
		name         string
		result       any
		after        AfterToolCallback
		wantResponse map[string]any
		wantParts    []*genai.FunctionResponsePart
	}{
		{
			name:   "callback returns nil",
			result: []any{map[string]any{"image": tcPNG, "label": "one"}},
			after: func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				return nil, nil
			},
			wantResponse: map[string]any{"result": []any{map[string]any{"label": "one"}}},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart()},
		},
		{
			name:   "callback returns the map it received",
			result: []any{map[string]any{"image": tcPNG, "label": "one"}},
			after: func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				return result, nil
			},
			wantResponse: map[string]any{"result": []any{map[string]any{"label": "one"}}},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart()},
		},
		{
			name:   "callback adds a key in place",
			result: "raw",
			after: func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				result["extra"] = "added"
				return nil, nil
			},
			wantResponse: map[string]any{"result": "raw", "extra": "added"},
		},
		{
			name:   "callback replaces the result in place",
			result: "old",
			after: func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				result["result"] = map[string]any{"nested": map[string]any{"img": tcPNG}}
				return result, nil
			},
			wantResponse: map[string]any{"result": map[string]any{"nested": map[string]any{"img": tcPNG}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{
				Model:              tcModel("media_tool"),
				Tools:              []tool.Tool{tcValueTool(t, "media_tool", tc.result)},
				AfterToolCallbacks: []AfterToolCallback{tc.after},
			}

			fr := tcOnlyResponse(t, tcRun(t, f, nil))

			if diff := cmp.Diff(tc.wantResponse, fr.Response); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantParts, fr.Parts); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMediaExtraction_FromCallbacksAndSkipSummarizationDisplay(t *testing.T) {
	answer := func() map[string]any { return map[string]any{"note": "from-cb", "chart": tcPNG} }
	failing := func(t *testing.T) tool.Tool {
		return tcMust(t)(functiontool.New(functiontool.Config{Name: "media_tool", Description: "fails"},
			func(agent.Context, struct{}) (any, error) { return nil, errors.New("boom") }))
	}
	for _, tc := range []struct {
		name    string
		tool    func(*testing.T) tool.Tool
		flow    Flow
		plugins *tcPlugins
	}{
		{
			name:    "plugin before-tool callback",
			tool:    func(t *testing.T) tool.Tool { return tcValueTool(t, "media_tool", map[string]any{"plain": true}) },
			plugins: &tcPlugins{before: func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) { return answer(), nil }},
		},
		{
			name: "agent before-tool callback",
			tool: func(t *testing.T) tool.Tool { return tcValueTool(t, "media_tool", map[string]any{"plain": true}) },
			flow: Flow{BeforeToolCallbacks: []BeforeToolCallback{func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return answer(), nil
			}}},
		},
		{
			name: "agent on-tool-error callback",
			tool: failing,
			flow: Flow{OnToolErrorCallbacks: []OnToolErrorCallback{func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return answer(), nil
			}}},
		},
		{
			name: "agent after-tool callback",
			tool: func(t *testing.T) tool.Tool { return tcValueTool(t, "media_tool", map[string]any{"plain": true}) },
			flow: Flow{AfterToolCallbacks: []AfterToolCallback{func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				return answer(), nil
			}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.flow
			f.Model = tcModel("media_tool")
			f.Tools = []tool.Tool{tc.tool(t)}

			fr := tcOnlyResponse(t, tcRun(t, &f, tc.plugins))

			if diff := cmp.Diff(map[string]any{"note": "from-cb"}, fr.Response); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]*genai.FunctionResponsePart{tcPNGPart()}, fr.Parts); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("after-tool callback sees the part", func(t *testing.T) {
		var seen string
		f := &Flow{
			Model: tcModel("media_tool"),
			Tools: []tool.Tool{tcValueTool(t, "media_tool", map[string]any{"chart": tcPNG, "summary": "up 3%"})},
			AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				seen = fmt.Sprintf("%T", result["chart"])
				return nil, nil
			}},
		}

		tcRun(t, f, nil)

		if seen != "*genai.Part" {
			t.Errorf("after-tool callback saw result[\"chart\"] of type %s, want *genai.Part", seen)
		}
	})

	t.Run("skip-summarization text shows the result before extraction", func(t *testing.T) {
		f := &Flow{Model: tcModel("display_tool"), Tools: []tool.Tool{&tcDisplayer{tcDeclared{name: "display_tool"}}}}

		ev := tcResponseEvent(t, tcRun(t, f, nil))

		parts := ev.LLMResponse.Content.Parts
		if len(parts) != 2 {
			t.Fatalf("got %d parts, want 2", len(parts))
		}
		if !strings.Contains(parts[1].Text, `"chart":`) || !strings.Contains(parts[1].Text, `"summary":"up 3%"`) {
			t.Errorf("text part = %q, want it to hold the chart and the summary", parts[1].Text)
		}
		if diff := cmp.Diff(map[string]any{"summary": "up 3%"}, parts[0].FunctionResponse.Response); diff != "" {
			t.Errorf("Response mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]*genai.FunctionResponsePart{tcPNGPart()}, parts[0].FunctionResponse.Parts); diff != "" {
			t.Errorf("Parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("skip summarization without a displayer adds no text", func(t *testing.T) {
		plain := tcMust(t)(functiontool.New(functiontool.Config{Name: "plain_tool", Description: "skips"},
			func(ctx agent.Context, _ struct{}) (map[string]any, error) {
				ctx.Actions().SkipSummarization = true
				return map[string]any{"chart": tcPNG, "summary": "up 3%"}, nil
			}))
		f := &Flow{Model: tcModel("plain_tool"), Tools: []tool.Tool{plain}}

		ev := tcResponseEvent(t, tcRun(t, f, nil))

		if got := len(ev.LLMResponse.Content.Parts); got != 1 {
			t.Errorf("got %d parts, want 1", got)
		}
	})

	t.Run("part two containers deep reaches callbacks and stays", func(t *testing.T) {
		var seen string
		f := &Flow{
			Model: tcModel("media_tool"),
			Tools: []tool.Tool{tcValueTool(t, "media_tool", map[string]any{"report": map[string]any{"charts": map[string]any{"first": tcPNG}}})},
			AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				seen = fmt.Sprintf("%T", result["report"].(map[string]any)["charts"].(map[string]any)["first"])
				return nil, nil
			}},
		}

		fr := tcOnlyResponse(t, tcRun(t, f, nil))

		if seen != "*genai.Part" {
			t.Errorf("after-tool callback saw report.charts.first of type %s, want *genai.Part", seen)
		}
		if got := fr.Response["report"].(map[string]any)["charts"].(map[string]any)["first"]; got != tcPNG {
			t.Errorf("Response report.charts.first = %v, want the handler's part", got)
		}
		if fr.Parts != nil {
			t.Errorf("Parts = %v, want nil", fr.Parts)
		}
	})
}

func TestFunctionTool_MissingMandatoryArgs_CallbackRoutingAndFieldOrder(t *testing.T) {
	type mArgs struct {
		ZFirst    string `json:"z_first"`
		ASecond   int    `json:"a_second"`
		Optional  string `json:"optional,omitempty"`
		Coalesced int    `json:"coalesced,omitzero"`
	}
	type afterCall struct {
		who    string
		result map[string]any
		err    error
	}
	ran := false
	myTool := tcMust(t)(functiontool.New(functiontool.Config{Name: "my_tool", Description: "M.", RequireConfirmation: true},
		func(agent.Context, mArgs) (map[string]any, error) {
			ran = true
			return map[string]any{"ok": true}, nil
		}))
	var onErrors []string
	var afters []afterCall
	plugins := &tcPlugins{
		onErr: func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
			onErrors = append(onErrors, "plugin")
			return nil, nil
		},
		after: func(_ agent.Context, _ tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
			afters = append(afters, afterCall{"plugin", result, err})
			return nil, nil
		},
	}
	f := &Flow{
		Model: &tcCallsModel{calls: []*genai.FunctionCall{{ID: "call-1", Name: "my_tool", Args: map[string]any{"unexpected": 1}}}},
		Tools: []tool.Tool{myTool},
		OnToolErrorCallbacks: []OnToolErrorCallback{func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
			onErrors = append(onErrors, "agent")
			return nil, nil
		}},
		AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
			afters = append(afters, afterCall{"agent", result, err})
			return nil, nil
		}},
	}

	ev := tcResponseEvent(t, tcRun(t, f, plugins))

	want := map[string]any{"error": "Invoking `my_tool()` failed as the following mandatory input parameters are not present:\nz_first\na_second\nYou could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."}
	if ran {
		t.Error("handler ran, want it not to")
	}
	if n := len(ev.Actions.RequestedToolConfirmations); n != 0 {
		t.Errorf("event requests %d confirmations, want none", n)
	}
	if len(onErrors) != 0 {
		t.Errorf("on-tool-error callbacks ran: %v, want none", onErrors)
	}
	wantAfters := []afterCall{{"plugin", want, nil}, {"agent", want, nil}}
	if diff := cmp.Diff(wantAfters, afters, cmp.AllowUnexported(afterCall{})); diff != "" {
		t.Errorf("after-tool calls mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, ev.LLMResponse.Content.Parts[0].FunctionResponse.Response); diff != "" {
		t.Errorf("Response mismatch (-want +got):\n%s", diff)
	}
}

type tcStruct struct{ N int }

func TestHandleFunctionCalls_LongRunningFalsyResultsAndNilResultComposition(t *testing.T) {
	longRunning := func(t *testing.T, v any) tool.Tool {
		return tcMust(t)(functiontool.New(functiontool.Config{Name: "media_tool", Description: "long", IsLongRunning: true},
			func(agent.Context, struct{}) (any, error) { return v, nil }))
	}
	type row struct {
		name          string
		tool          func(*testing.T) tool.Tool
		confirmations map[string]*toolconfirmation.ToolConfirmation
		after         AfterToolCallback
		wantEvent     bool
		wantResponse  map[string]any
		wantParts     []*genai.FunctionResponsePart
	}
	rows := []row{
		{
			name:         "ordinary tool returns nil",
			tool:         func(t *testing.T) tool.Tool { return tcValueTool(t, "media_tool", nil) },
			wantEvent:    true,
			wantResponse: map[string]any{"result": nil},
		},
		{
			name: "ordinary tool returns a nil pointer",
			tool: func(t *testing.T) tool.Tool {
				return tcMust(t)(functiontool.New(functiontool.Config{Name: "media_tool", Description: "nil"},
					func(agent.Context, struct{}) (*tcStruct, error) { return nil, nil }))
			},
			wantEvent:    true,
			wantResponse: map[string]any{"result": nil},
		},
		{
			name: "ordinary tool returns a nil map",
			tool: func(t *testing.T) tool.Tool {
				return tcMust(t)(functiontool.New(functiontool.Config{Name: "media_tool", Description: "nil"},
					func(agent.Context, struct{}) (map[string]any, error) { return nil, nil }))
			},
			wantEvent:    true,
			wantResponse: map[string]any{"result": nil},
		},
		{
			name: "confirmed tool returns nil",
			tool: func(t *testing.T) tool.Tool {
				return tcWrap(t, tool.WithConfirmation(tcToolset(tcValueTool(t, "media_tool", nil)), true, nil))
			},
			confirmations: map[string]*toolconfirmation.ToolConfirmation{"call-1": {Confirmed: true}},
			wantEvent:     true,
			wantResponse:  map[string]any{"result": nil},
		},
		{
			name:         "long-running tool returns a one-key map",
			tool:         func(t *testing.T) tool.Tool { return longRunning(t, map[string]any{"result": nil}) },
			wantEvent:    true,
			wantResponse: map[string]any{"result": nil},
		},
		{
			name: "after-tool callback answers a long-running nil",
			tool: func(t *testing.T) tool.Tool { return longRunning(t, nil) },
			after: func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				return map[string]any{"result": nil}, nil
			},
			wantEvent:    true,
			wantResponse: map[string]any{"result": nil},
		},
		{
			name:         "long-running tool returns media",
			tool:         func(t *testing.T) tool.Tool { return longRunning(t, tcPNG) },
			wantEvent:    true,
			wantResponse: map[string]any{},
			wantParts:    []*genai.FunctionResponsePart{tcPNGPart()},
		},
		{
			name: "after-tool callback empties a pending result in place",
			tool: func(t *testing.T) tool.Tool { return longRunning(t, "pending") },
			after: func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
				result["result"] = ""
				return nil, nil
			},
			wantEvent:    true,
			wantResponse: map[string]any{"result": ""},
		},
	}
	falsy := []struct {
		name string
		v    any
	}{
		{"nil", nil},
		{"nil pointer", (*tcStruct)(nil)},
		{"nil map", map[string]any(nil)},
		{"empty map", map[string]any{}},
		{"empty string", ""},
		{"zero", 0},
		{"false", false},
		{"empty list", []any{}},
	}
	for _, fv := range falsy {
		rows = append(rows,
			row{name: "long-running tool returns " + fv.name, tool: func(t *testing.T) tool.Tool { return longRunning(t, fv.v) }},
			row{name: "deferring tool returns " + fv.name, tool: func(*testing.T) tool.Tool {
				return &tcDeferrer{tcDeclared: tcDeclared{name: "media_tool"}, raw: fv.v}
			}},
		)
	}
	for _, tc := range rows {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{}
			if tc.after != nil {
				f.AfterToolCallbacks = []AfterToolCallback{tc.after}
			}

			ev := tcHandle(t, f, map[string]tool.Tool{"media_tool": tc.tool(t)}, tc.confirmations, tcCall("media_tool"))

			if !tc.wantEvent {
				if ev != nil {
					t.Errorf("got a function response event %v, want none", ev.LLMResponse.Content.Parts[0].FunctionResponse.Response)
				}
				return
			}
			if ev == nil {
				t.Fatal("got no function response event, want one")
			}
			fr := ev.LLMResponse.Content.Parts[0].FunctionResponse
			if diff := cmp.Diff(tc.wantResponse, fr.Response); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantParts, fr.Parts); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// tcRecorder records the order and the arguments of the tool callbacks of
// one run.
type tcRecorder struct {
	mu    sync.Mutex
	order []string
	tools []string
	args  []map[string]any
	errs  []error
}

func (r *tcRecorder) record(who string, t tool.Tool, args map[string]any, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, who)
	r.tools = append(r.tools, t.Name()+"|"+t.Description())
	r.args = append(r.args, args)
	if err != nil {
		r.errs = append(r.errs, err)
	}
}

// tcRecordingFlow returns a flow whose model calls call, with tools, and
// whose plugin and agent tool callbacks record into r. A non-nil answer
// function of a kind makes the agent callback of that kind answer with it.
func tcRecordingFlow(call *genai.FunctionCall, tools []tool.Tool, r *tcRecorder, before func(map[string]any) map[string]any, onErr func(error) (map[string]any, error), after func(map[string]any) map[string]any) (*Flow, *tcPlugins) {
	plugins := &tcPlugins{
		before: func(_ agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			r.record("plugin_before", t, args, nil)
			return nil, nil
		},
		onErr: func(_ agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
			r.record("plugin_on_error", t, args, err)
			return nil, nil
		},
		after: func(_ agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
			r.record("plugin_after", t, result, err)
			return nil, nil
		},
	}
	f := &Flow{
		Model: &tcCallsModel{calls: []*genai.FunctionCall{call}},
		Tools: tools,
		BeforeToolCallbacks: []BeforeToolCallback{func(_ agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			r.record("agent_before", t, args, nil)
			if before == nil {
				return nil, nil
			}
			return before(args), nil
		}},
		OnToolErrorCallbacks: []OnToolErrorCallback{func(_ agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
			r.record("agent_on_error", t, args, err)
			if onErr == nil {
				return nil, nil
			}
			return onErr(err)
		}},
		AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
			r.record("agent_after", t, result, err)
			if after == nil {
				return nil, nil
			}
			return after(result), nil
		}},
	}
	return f, plugins
}

func TestHandleFunctionCalls_UnknownToolFullLifecycleAndAsymmetry(t *testing.T) {
	declared := func(name string) tool.Tool {
		return &packingTool{name: name, decl: &genai.FunctionDeclaration{Name: name}}
	}
	for _, variant := range []struct {
		name  string
		tools func() []tool.Tool
		list  string
	}{
		{name: "unknown name", tools: func() []tool.Tool { return []tool.Tool{declared("tool_z"), declared("tool_a")} }, list: "tool_z, tool_a"},
		{
			name: "registered tool that cannot run",
			tools: func() []tool.Tool {
				return []tool.Tool{declared("tool_z"), declared("tool_a"), &tcDeclared{name: "missing_fn"}}
			},
			list: "tool_z, tool_a, missing_fn",
		},
	} {
		t.Run(variant.name, func(t *testing.T) {
			placeholder := "missing_fn|Tool not found"

			t.Run("no callback answers", func(t *testing.T) {
				setupTestTracer(t)
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(tcCall("missing_fn"), variant.tools(), r, nil, nil, nil)

				fr := tcOnlyResponse(t, tcRun(t, f, plugins))

				if diff := cmp.Diff([]string{"plugin_before", "agent_before", "plugin_on_error", "agent_on_error"}, r.order); diff != "" {
					t.Errorf("callback order mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]string{placeholder, placeholder, placeholder, placeholder}, r.tools); diff != "" {
					t.Errorf("callback tools mismatch (-want +got):\n%s", diff)
				}
				for _, err := range r.errs {
					if got := fmt.Sprintf("%T", err); got != "*errors.errorString" {
						t.Errorf("on-tool-error error type = %s, want *errors.errorString", got)
					}
					if diff := cmp.Diff(tcLongText("missing_fn", variant.list), err.Error()); diff != "" {
						t.Errorf("on-tool-error text mismatch (-want +got):\n%s", diff)
					}
				}
				want := map[string]any{"error": tcShortText("missing_fn", variant.list)}
				if diff := cmp.Diff(want, fr.Response); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
				var spanErr string
				found := false
				for _, s := range testExporter.GetSpans() {
					if s.Name == "execute_tool missing_fn" {
						found, spanErr = true, s.Status.Description
					}
				}
				if !found || spanErr != tcShortText("missing_fn", variant.list) {
					t.Errorf("span execute_tool missing_fn found = %t with error %q, want the short text", found, spanErr)
				}
			})

			t.Run("before-tool callback answers", func(t *testing.T) {
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(tcCall("missing_fn"), variant.tools(), r,
					func(map[string]any) map[string]any { return map[string]any{} }, nil,
					func(map[string]any) map[string]any { return map[string]any{"replaced": true} })

				fr := tcOnlyResponse(t, tcRun(t, f, plugins))

				if diff := cmp.Diff([]string{"plugin_before", "agent_before", "plugin_after", "agent_after"}, r.order); diff != "" {
					t.Errorf("callback order mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]string{placeholder, placeholder, placeholder, placeholder}, r.tools); diff != "" {
					t.Errorf("callback tools mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff([]map[string]any{{}, {}, {}, {}}, r.args); diff != "" {
					t.Errorf("callback arguments and results mismatch (-want +got):\n%s", diff)
				}
				if len(r.errs) != 0 {
					t.Errorf("callbacks received errors %v, want none", r.errs)
				}
				if diff := cmp.Diff(map[string]any{"replaced": true}, fr.Response); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
			})

			t.Run("on-tool-error callback answers", func(t *testing.T) {
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(tcCall("missing_fn"), variant.tools(), r, nil,
					func(error) (map[string]any, error) { return map[string]any{}, nil }, nil)

				fr := tcOnlyResponse(t, tcRun(t, f, plugins))

				if diff := cmp.Diff([]string{"plugin_before", "agent_before", "plugin_on_error", "agent_on_error"}, r.order); diff != "" {
					t.Errorf("callback order mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(map[string]any{}, fr.Response); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
			})

			t.Run("on-tool-error callback returns the lookup error wrapped", func(t *testing.T) {
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(tcCall("missing_fn"), variant.tools(), r, nil,
					func(err error) (map[string]any, error) { return nil, fmt.Errorf("seen: %w", err) }, nil)

				fr := tcOnlyResponse(t, tcRun(t, f, plugins))

				if diff := cmp.Diff([]string{"plugin_before", "agent_before", "plugin_on_error", "agent_on_error"}, r.order); diff != "" {
					t.Errorf("callback order mismatch (-want +got):\n%s", diff)
				}
				want := map[string]any{"error": tcShortText("missing_fn", variant.list)}
				if diff := cmp.Diff(want, fr.Response); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
			})

			t.Run("on-tool-error callback returns another error", func(t *testing.T) {
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(tcCall("missing_fn"), variant.tools(), r, nil,
					func(error) (map[string]any, error) { return nil, errors.New("boom") }, nil)

				fr := tcOnlyResponse(t, tcRun(t, f, plugins))

				if diff := cmp.Diff([]string{"plugin_before", "agent_before", "plugin_on_error", "agent_on_error"}, r.order); diff != "" {
					t.Errorf("callback order mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(map[string]any{"error": "boom"}, fr.Response); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
			})

			t.Run("callbacks get an empty map for a call without arguments", func(t *testing.T) {
				r := &tcRecorder{}
				f, plugins := tcRecordingFlow(&genai.FunctionCall{ID: "call-1", Name: "missing_fn"}, variant.tools(), r, nil, nil, nil)

				tcRun(t, f, plugins)

				for i, args := range r.args {
					if args == nil || len(args) != 0 {
						t.Errorf("%s received arguments %#v, want an empty non-nil map", r.order[i], args)
					}
				}
			})

			t.Run("callbacks get a deep copy of the arguments", func(t *testing.T) {
				r := &tcRecorder{}
				call := &genai.FunctionCall{ID: "call-1", Name: "missing_fn", Args: map[string]any{"nested": map[string]any{"k": "v"}}}
				f, plugins := tcRecordingFlow(call, variant.tools(), r, func(args map[string]any) map[string]any {
					args["nested"].(map[string]any)["k"] = "mutated"
					return nil
				}, nil, nil)

				events := tcRun(t, f, plugins)

				var modelArgs map[string]any
				for _, ev := range events {
					if ev.LLMResponse.Content != nil && ev.LLMResponse.Content.Parts[0].FunctionCall != nil {
						modelArgs = ev.LLMResponse.Content.Parts[0].FunctionCall.Args
					}
				}
				if got := modelArgs["nested"].(map[string]any)["k"]; got != "v" {
					t.Errorf("model's call argument nested.k = %v, want v", got)
				}
				if len(r.order) != 4 {
					t.Fatalf("callbacks ran %v, want two before-tool and two on-tool-error callbacks", r.order)
				}
				if got := r.args[3]["nested"].(map[string]any)["k"]; r.order[3] != "agent_on_error" || got != "mutated" {
					t.Errorf("%s received nested.k = %v, want agent_on_error with mutated", r.order[3], got)
				}
			})
		})
	}
}

func TestHandleFunctionCalls_RegistryOrderAcrossCallers(t *testing.T) {
	declared := func(names ...string) []tool.Tool {
		tools := make([]tool.Tool, len(names))
		for i, n := range names {
			tools[i] = &packingTool{name: n, decl: &genai.FunctionDeclaration{Name: n}}
		}
		return tools
	}
	errorOf := func(fr *genai.FunctionResponse) string {
		s, _ := fr.Response["error"].(string)
		return s
	}

	t.Run("request processors, tools, then a callback's declaration", func(t *testing.T) {
		transfer := &packingTool{name: "transfer_to_agent", decl: &genai.FunctionDeclaration{Name: "transfer_to_agent"}}
		cbDeclared := &packingTool{name: "cb_declared", decl: &genai.FunctionDeclaration{Name: "cb_declared"}}
		f := &Flow{
			Model: tcModel("missing_fn"),
			RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
				func(_ agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if err := toolutils.PackTool(req, transfer); err != nil {
							yield(nil, err)
						}
					}
				},
			},
			Tools: declared("tool_z", "tool_a", "tool_m"),
			BeforeModelCallbacks: []BeforeModelCallback{func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, toolutils.PackTool(req, cbDeclared)
			}},
		}

		fr := tcOnlyResponse(t, tcRun(t, f, nil))

		if diff := cmp.Diff(tcShortText("missing_fn", "transfer_to_agent, tool_z, tool_a, tool_m, cb_declared"), errorOf(fr)); diff != "" {
			t.Errorf("error text mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("25 tools are all listed", func(t *testing.T) {
		names := make([]string, 25)
		for i := range names {
			names[i] = fmt.Sprintf("t%02d", i)
		}
		f := &Flow{Model: tcModel("missing_fn"), Tools: declared(names...)}

		fr := tcOnlyResponse(t, tcRun(t, f, nil))

		if diff := cmp.Diff(tcShortText("missing_fn", strings.Join(names, ", ")), errorOf(fr)); diff != "" {
			t.Errorf("error text mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("live flow", func(t *testing.T) {
		baseline, _ := runLiveStacks()
		defer assertNoRunLiveLeak(t, baseline)
		const toolCall = `{"toolCall":{"functionCalls":[{"id":"m_1","name":"missing_fn","args":{}}]}}`
		client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
				t.Errorf("writing toolCall failed: %v", err)
				return
			}
			blockUntilClientCloses(conn)
		})
		var befores sync.Map
		f := &Flow{
			Model:             &fakeLiveModel{client: client},
			RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
			Tools:             declared("decl_z", "decl_a", "decl_m", "decl_b"),
			BeforeToolCallbacks: []BeforeToolCallback{func(_ agent.Context, t tool.Tool, _ map[string]any) (map[string]any, error) {
				befores.Store(t.Name(), true)
				return nil, nil
			}},
		}
		ctx, cancel := newLiveInvocationContext(t)
		defer cancel()
		sess, seq, err := f.RunLive(ctx)
		if err != nil {
			t.Fatalf("RunLive failed: %v", err)
		}
		defer func() { _ = sess.Close() }()
		next, stop := iter.Pull2(seq)
		defer stop()

		var fr *genai.FunctionResponse
		for fr == nil {
			ev, err, ok := next()
			if !ok {
				t.Fatal("iterator ended before the function response")
			}
			if err != nil {
				t.Fatalf("RunLive yielded an error: %v", err)
			}
			if ev == nil || ev.LLMResponse.Content == nil {
				continue
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.FunctionResponse != nil {
					fr = p.FunctionResponse
				}
			}
		}

		if _, ok := befores.Load("missing_fn"); !ok {
			t.Error("before-tool callback did not run for missing_fn")
		}
		if diff := cmp.Diff(tcShortText("missing_fn", "decl_z, decl_a, decl_m, decl_b"), errorOf(fr)); diff != "" {
			t.Errorf("error text mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("callers that pass the agent's tools", func(t *testing.T) {
		// authPreprocessor and RequestConfirmationRequestProcessor build their
		// tool map from f.Tools and call handleFunctionCalls with it.
		f := &Flow{Tools: declared("tool_z", "tool_a", "tool_m", "tool_b")}
		tools := map[string]tool.Tool{}
		for _, tl := range f.Tools {
			tools[tl.Name()] = tl
		}

		ev := tcHandle(t, f, tools, nil, tcCall("missing_fn"))

		if diff := cmp.Diff(tcShortText("missing_fn", "tool_z, tool_a, tool_m, tool_b"), errorOf(ev.LLMResponse.Content.Parts[0].FunctionResponse)); diff != "" {
			t.Errorf("error text mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestResponseScheduling_AllSourcesWrappersParallelAndNonLiveIsolation(t *testing.T) {
	silentNew := func(t *testing.T, name string, result any, err error) tool.Tool {
		return tcMust(t)(functiontool.New(functiontool.Config{Name: name, Description: "silent", ResponseScheduling: genai.FunctionResponseSchedulingSilent},
			func(agent.Context, struct{}) (any, error) { return result, err }))
	}
	schedulingOf := func(t *testing.T, ev *session.Event) []genai.FunctionResponseScheduling {
		t.Helper()
		if ev == nil {
			t.Fatal("got no function response event")
		}
		var got []genai.FunctionResponseScheduling
		for _, p := range ev.LLMResponse.Content.Parts {
			got = append(got, p.FunctionResponse.Scheduling)
		}
		return got
	}

	t.Run("every tool kind, in parallel", func(t *testing.T) {
		whenIdle := tcMust(t)(functiontool.New(functiontool.Config{Name: "when_idle_tool", Description: "w", ResponseScheduling: genai.FunctionResponseSchedulingWhenIdle},
			func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"ok": true}, nil }))
		interrupt := tcMust(t)(functiontool.NewStreaming(functiontool.Config{Name: "interrupt_stream", Description: "i", ResponseScheduling: genai.FunctionResponseSchedulingInterrupt},
			func(agent.Context, struct{}) iter.Seq2[string, error] {
				return func(yield func(string, error) bool) { yield("chunk", nil) }
			}))
		custom := &tcScheduled{tcDeclared: tcDeclared{name: "custom_silent"}, scheduling: genai.FunctionResponseSchedulingSilent}
		tools := map[string]tool.Tool{"when_idle_tool": whenIdle, "interrupt_stream": interrupt, "custom_silent": custom, "default_tool": tcValueTool(t, "default_tool", "x")}
		calls := []*genai.FunctionCall{
			{ID: "c1", Name: "when_idle_tool", Args: map[string]any{}},
			{ID: "c2", Name: "interrupt_stream", Args: map[string]any{}},
			{ID: "c3", Name: "custom_silent", Args: map[string]any{}},
			{ID: "c4", Name: "default_tool", Args: map[string]any{}},
			{ID: "c5", Name: "unknown_tool", Args: map[string]any{}},
		}

		ev := tcHandle(t, &Flow{}, tools, nil, calls...)

		want := []genai.FunctionResponseScheduling{genai.FunctionResponseSchedulingWhenIdle, genai.FunctionResponseSchedulingInterrupt, genai.FunctionResponseSchedulingSilent, "", ""}
		if diff := cmp.Diff(want, schedulingOf(t, ev)); diff != "" {
			t.Errorf("Scheduling mismatch (-want +got):\n%s", diff)
		}
	})

	for _, tc := range []struct {
		name string
		tool func(*testing.T) tool.Tool
		flow Flow
	}{
		{name: "handler result", tool: func(t *testing.T) tool.Tool { return silentNew(t, "silent_tool", map[string]any{"ok": true}, nil) }},
		{
			name: "before-tool callback answer",
			tool: func(t *testing.T) tool.Tool { return silentNew(t, "silent_tool", map[string]any{"ok": true}, nil) },
			flow: Flow{BeforeToolCallbacks: []BeforeToolCallback{func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
				return map[string]any{"from": "before"}, nil
			}}},
		},
		{
			name: "on-tool-error callback answer",
			tool: func(t *testing.T) tool.Tool { return silentNew(t, "silent_tool", nil, errors.New("boom")) },
			flow: Flow{OnToolErrorCallbacks: []OnToolErrorCallback{func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return map[string]any{}, nil
			}}},
		},
		{
			name: "after-tool callback answer",
			tool: func(t *testing.T) tool.Tool { return silentNew(t, "silent_tool", map[string]any{"ok": true}, nil) },
			flow: Flow{AfterToolCallbacks: []AfterToolCallback{func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				return map[string]any{"from": "after"}, nil
			}}},
		},
		{
			name: "confirmation gate of a function tool",
			tool: func(t *testing.T) tool.Tool {
				return tcWrap(t, tool.WithConfirmation(tcToolset(silentNew(t, "silent_tool", map[string]any{"ok": true}, nil)), true, nil))
			},
		},
		{
			name: "confirmation gate of a streaming tool",
			tool: func(t *testing.T) tool.Tool {
				stream := tcMust(t)(functiontool.NewStreaming(functiontool.Config{Name: "silent_tool", Description: "s", ResponseScheduling: genai.FunctionResponseSchedulingSilent},
					func(agent.Context, struct{}) iter.Seq2[string, error] {
						return func(yield func(string, error) bool) { yield("chunk", nil) }
					}))
				return tcWrap(t, tool.WithConfirmation(tcToolset(stream), true, nil))
			},
		},
		{
			name: "result holding a scheduling key",
			tool: func(t *testing.T) tool.Tool {
				return silentNew(t, "silent_tool", map[string]any{"scheduling": "INTERRUPT", "ok": true}, nil)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.flow

			ev := tcHandle(t, &f, map[string]tool.Tool{"silent_tool": tc.tool(t)}, nil, tcCall("silent_tool"))

			if diff := cmp.Diff([]genai.FunctionResponseScheduling{genai.FunctionResponseSchedulingSilent}, schedulingOf(t, ev)); diff != "" {
				t.Errorf("Scheduling mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("request declaration of a function tool", func(t *testing.T) {
		f := &Flow{Model: tcModel("silent_tool"), Tools: []tool.Tool{silentNew(t, "silent_tool", map[string]any{"ok": true}, nil)}}

		ev := tcResponseEvent(t, tcRunIn(t, feature.WithLegacyFunctionDeclarationSchema(t.Context(), true), f, nil))

		if diff := cmp.Diff([]genai.FunctionResponseScheduling{genai.FunctionResponseSchedulingSilent}, schedulingOf(t, ev)); diff != "" {
			t.Errorf("Scheduling mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("prefixed tool keeps scheduling and media", func(t *testing.T) {
		prefixed := tcWrap(t, tool.WithToolNamePrefix(tcToolset(silentNew(t, "silent_tool", tcPNG, nil)), "p"))

		ev := tcHandle(t, &Flow{}, map[string]tool.Tool{prefixed.Name(): prefixed}, nil, tcCall(prefixed.Name()))

		if diff := cmp.Diff([]genai.FunctionResponseScheduling{genai.FunctionResponseSchedulingSilent}, schedulingOf(t, ev)); diff != "" {
			t.Errorf("Scheduling mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]*genai.FunctionResponsePart{tcPNGPart()}, ev.LLMResponse.Content.Parts[0].FunctionResponse.Parts); diff != "" {
			t.Errorf("Parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("override wins over the tool's scheduling", func(t *testing.T) {
		silent := silentNew(t, "silent_tool", nil, nil)
		res := toolResult{m: map[string]any{"ok": true}}

		withOverride := buildFunctionResponse(silent, "call-1", "silent_tool", res, genai.FunctionResponseSchedulingInterrupt).FunctionResponse.Scheduling
		without := buildFunctionResponse(silent, "call-1", "silent_tool", res, "").FunctionResponse.Scheduling

		if withOverride != genai.FunctionResponseSchedulingInterrupt || without != genai.FunctionResponseSchedulingSilent {
			t.Errorf("Scheduling = %q with an override and %q without, want INTERRUPT and SILENT", withOverride, without)
		}
	})
}

func TestHandleFunctionCalls_NamelessCallUsesUnnamedPlaceholder(t *testing.T) {
	declared := func(name string) tool.Tool {
		return &packingTool{name: name, decl: &genai.FunctionDeclaration{Name: name}}
	}
	setupTestTracer(t)
	r := &tcRecorder{}
	f, _ := tcRecordingFlow(&genai.FunctionCall{ID: "call-1"}, []tool.Tool{declared("tool_b"), declared("tool_a")}, r, nil, nil, nil)

	fr := tcOnlyResponse(t, tcRun(t, f, nil))

	if diff := cmp.Diff([]string{"<unnamed>|Tool not found", "<unnamed>|Tool not found"}, r.tools); diff != "" {
		t.Errorf("callback tools mismatch (-want +got):\n%s", diff)
	}
	if len(r.errs) != 1 || r.errs[0].Error() != tcLongText("<unnamed>", "tool_b, tool_a") {
		t.Errorf("on-tool-error errors = %v, want one with the long text for <unnamed>", r.errs)
	}
	want := &genai.FunctionResponse{ID: "call-1", Name: "<unnamed>", Response: map[string]any{"error": tcShortText("<unnamed>", "tool_b, tool_a")}}
	if diff := cmp.Diff(want, fr); diff != "" {
		t.Errorf("FunctionResponse mismatch (-want +got):\n%s", diff)
	}
	var spans []string
	for _, s := range testExporter.GetSpans() {
		if strings.HasPrefix(s.Name, "execute_tool") {
			spans = append(spans, s.Name)
		}
	}
	if diff := cmp.Diff([]string{"execute_tool <unnamed>"}, spans); diff != "" {
		t.Errorf("execute_tool spans mismatch (-want +got):\n%s", diff)
	}
}

// TestHandleFunctionCalls_ConfirmationGateStopsTheHandler checks that a
// function tool under tool.WithConfirmation, run by the flow, does not run its
// handler until the call is confirmed. It fails if the wrapper's RunRaw, the
// method the flow calls, skips the confirmation gate.
func TestHandleFunctionCalls_ConfirmationGateStopsTheHandler(t *testing.T) {
	for _, tc := range []struct {
		name          string
		confirmations map[string]*toolconfirmation.ToolConfirmation
		wantResponse  map[string]any
	}{
		{
			name:         "no confirmation yet",
			wantResponse: map[string]any{"error": `error tool "guarded" requires confirmation, please approve or reject`},
		},
		{
			name:          "rejected",
			confirmations: map[string]*toolconfirmation.ToolConfirmation{"call-1": {Confirmed: false}},
			wantResponse:  map[string]any{"error": `error tool "guarded" call is rejected`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ran := false
			inner := tcMust(t)(functiontool.New(functiontool.Config{Name: "guarded", Description: "guarded"},
				func(agent.Context, struct{}) (any, error) {
					ran = true
					return "done", nil
				}))
			guarded := tcWrap(t, tool.WithConfirmation(tcToolset(inner), true, nil))

			ev := tcHandle(t, &Flow{}, map[string]tool.Tool{"guarded": guarded}, tc.confirmations, tcCall("guarded"))

			if ran {
				t.Error("handler ran, want it not to run before the call is confirmed")
			}
			if diff := cmp.Diff(tc.wantResponse, ev.LLMResponse.Content.Parts[0].FunctionResponse.Response); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// tcBackendRecorder records the backend its request hook sees.
type tcBackendRecorder struct {
	tcDeclared
	seen genai.Backend
}

func (r *tcBackendRecorder) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	r.seen = googlellm.BackendFromContext(ctx)
	return toolutils.PackTool(req, r)
}

func TestPreprocess_ToolRequestHooksSeeTheModelBackend(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model model.LLM
		want  genai.Backend
	}{
		{name: "Vertex AI", model: &capableLLM{name: "m", backend: genai.BackendVertexAI}, want: genai.BackendVertexAI},
		{name: "Gemini API", model: &capableLLM{name: "m", backend: genai.BackendGeminiAPI}, want: genai.BackendGeminiAPI},
		{name: "model without a backend", model: tcModel("x"), want: genai.BackendUnspecified},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &tcBackendRecorder{tcDeclared: tcDeclared{name: "recorder"}}
			f := &Flow{Model: tc.model, Tools: []tool.Tool{rec}}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "agent_1"}})

			for _, err := range f.preprocess(ctx, &model.LLMRequest{}) {
				if err != nil {
					t.Fatalf("preprocess() error = %v", err)
				}
			}

			if rec.seen != tc.want {
				t.Errorf("ProcessRequest saw backend %v, want %v", rec.seen, tc.want)
			}
		})
	}
}

// tcBackendModel is a model of a backend that records the requests it gets
// and answers each with the text "done".
type tcBackendModel struct {
	backend genai.Backend
	mu      sync.Mutex
	reqs    []*model.LLMRequest
}

func (m *tcBackendModel) Name() string                       { return "backend-model" }
func (m *tcBackendModel) GetGoogleLLMVariant() genai.Backend { return m.backend }

func (m *tcBackendModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.reqs = append(m.reqs, req)
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
	}
}

// TestRun_ToolDeclarationsFollowTheResolvedModelsBackend checks that an agent
// without a model of its own declares its tools for the backend of the model
// it resolves from its parent. It fails if the flow takes the backend from
// Flow.Model, which is nil for such an agent: the parameterless tool would
// then keep the closed-object ParametersJsonSchema that Vertex AI omits.
func TestRun_ToolDeclarationsFollowTheResolvedModelsBackend(t *testing.T) {
	vertex := &tcBackendModel{backend: genai.BackendVertexAI}
	leaf := newLinkedLLMAgent(t, "leaf", nil)
	newLinkedLLMAgent(t, "parent", vertex, leaf)
	ping := tcMust(t)(functiontool.New(functiontool.Config{Name: "ping", Description: "Ping."},
		func(agent.Context, struct{}) (string, error) { return "pong", nil }))
	f := &Flow{Tools: []tool.Tool{ping}}
	base := runconfig.ToContext(t.Context(), &runconfig.RunConfig{})
	ctx := icontext.NewInvocationContext(base, icontext.InvocationContextParams{InvocationID: "inv_1", Agent: leaf})

	for _, err := range f.Run(ctx) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
	}

	if len(vertex.reqs) != 1 {
		t.Fatalf("parent's model got %d requests, want 1", len(vertex.reqs))
	}
	got := vertex.reqs[0].Config.Tools[0].FunctionDeclarations[0]
	want := &genai.FunctionDeclaration{Name: "ping", Description: "Ping.", ResponseJsonSchema: &jsonschema.Schema{Type: "string"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("declaration mismatch (-want +got):\n%s", diff)
	}
}
