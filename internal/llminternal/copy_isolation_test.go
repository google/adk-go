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
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// The tests below run the in-tree steps that edit the request copy in place,
// found by searching for writes through req.Contents and req.Config, and check
// that the stored event or the agent's config is left as it was. Steps outside
// this package are in copy_isolation_external_test.go.

// isolationHistory returns a history holding the parts the req.Contents
// mutators below edit.
func isolationHistory() []*session.Event {
	return []*session.Event{
		{ID: "u", Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{Text: "look"},
			{}, // dropped by the empty-part filter
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}, DisplayName: "img.png"}},
			{FileData: &genai.FileData{FileURI: "gs://b/f.pdf", DisplayName: "f.pdf"}},
		}}}},
		{ID: "c", Author: "agent", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "adk-1", Name: "f", Args: map[string]any{"a": "b"}}},
		}}}},
		{ID: "r", Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "adk-1", Name: "f", Response: map[string]any{"ok": true}}},
		}}}},
	}
}

func buildIsolationContents(t *testing.T, events []*session.Event) []*genai.Content {
	t.Helper()
	contents, err := buildContentsDefault("agent", "", "", events, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(contents) != 3 {
		t.Fatalf("built %d contents, want 3", len(contents))
	}
	return contents
}

// TestBuildContentsLeavesHistoryAlone covers the edits buildContents makes to
// its copy: the empty-part filter compacts Parts in place, and
// RemoveClientFunctionCallID clears the adk- call and response IDs.
func TestBuildContentsLeavesHistoryAlone(t *testing.T) {
	events := isolationHistory()
	contents := buildIsolationContents(t, events)

	if got := len(contents[0].Parts); got != 3 {
		t.Errorf("first content has %d parts, want 3 after dropping the empty one", got)
	}
	if id := contents[1].Parts[0].FunctionCall.ID; id != "" {
		t.Errorf("request function call ID = %q, want it cleared", id)
	}
	if id := contents[2].Parts[0].FunctionResponse.ID; id != "" {
		t.Errorf("request function response ID = %q, want it cleared", id)
	}
	if diff := cmp.Diff(isolationHistory(), events); diff != "" {
		t.Errorf("building contents changed the history (-want +got):\n%s", diff)
	}
}

type geminiAPIModel struct{ mockLLM }

func (geminiAPIModel) GetGoogleLLMVariant() genai.Backend { return genai.BackendGeminiAPI }

// TestRemoveDisplayNameLeavesHistoryAlone covers removeDisplayNameIfExists,
// which clears InlineData.DisplayName and FileData.DisplayName in place.
func TestRemoveDisplayNameLeavesHistoryAlone(t *testing.T) {
	events := isolationHistory()
	req := &model.LLMRequest{Contents: buildIsolationContents(t, events)}
	base, err := agent.New(agent.Config{Name: "agent"})
	if err != nil {
		t.Fatal(err)
	}
	a := &mockLLMAgent{Agent: base, s: &State{Model: &geminiAPIModel{mockLLM{name: "gemini-2.5-flash"}}}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	for _, err := range removeDisplayNameIfExists(ctx, req, &Flow{}) {
		if err != nil {
			t.Fatal(err)
		}
	}

	parts := req.Contents[0].Parts
	if parts[1].InlineData.DisplayName != "" || parts[2].FileData.DisplayName != "" {
		t.Fatalf("removeDisplayNameIfExists kept the display names %q, %q", parts[1].InlineData.DisplayName, parts[2].FileData.DisplayName)
	}
	if diff := cmp.Diff(isolationHistory(), events); diff != "" {
		t.Errorf("removing display names changed the history (-want +got):\n%s", diff)
	}
}

// TestRequestCopyBytesAreOwn checks that writing into the request's inline
// bytes and thought signature, which a callback may do, leaves the history
// alone.
func TestRequestCopyBytesAreOwn(t *testing.T) {
	events := isolationHistory()
	events[1].LLMResponse.Content.Parts[0].ThoughtSignature = []byte("sig")
	contents := buildIsolationContents(t, events)
	contents[0].Parts[1].InlineData.Data[0] = 9
	contents[1].Parts[0].ThoughtSignature[0] = 'X'

	want := isolationHistory()
	want[1].LLMResponse.Content.Parts[0].ThoughtSignature = []byte("sig")
	if diff := cmp.Diff(want, events); diff != "" {
		t.Errorf("writing request bytes changed the history (-want +got):\n%s", diff)
	}
}

// isolationConfig returns an agent config holding the fields the req.Config
// mutators below edit. FunctionDeclarations has spare capacity, so an append
// through a shared backing array would write into the agent's.
func isolationConfig() *genai.GenerateContentConfig {
	decls := make([]*genai.FunctionDeclaration, 1, 4)
	decls[0] = &genai.FunctionDeclaration{Name: "existing"}
	return &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "agent instruction"}}},
		Tools:             []*genai.Tool{{FunctionDeclarations: decls}},
		HTTPOptions:       &genai.HTTPOptions{Headers: http.Header{"X-Existing": {"1"}}},
	}
}

func checkConfigUnchanged(t *testing.T, cfg *genai.GenerateContentConfig) {
	t.Helper()
	if diff := cmp.Diff(isolationConfig(), cfg); diff != "" {
		t.Errorf("the agent's config changed (-want +got):\n%s", diff)
	}
	decls := cfg.Tools[0].FunctionDeclarations
	if spare := decls[:cap(decls)][len(decls)]; spare != nil {
		t.Errorf("the agent's FunctionDeclarations got %v written past its length", spare)
	}
}

// TestAppendInstructionsLeavesAgentConfigAlone covers utils.AppendInstructions,
// which appends to the text of the last SystemInstruction part in place.
func TestAppendInstructionsLeavesAgentConfigAlone(t *testing.T) {
	cfg := isolationConfig()
	req := &model.LLMRequest{Config: newConfigRequester(t, cfg)()}
	utils.AppendInstructions(req, "more")
	if got := req.Config.SystemInstruction.Parts[0].Text; got != "agent instruction\n\nmore" {
		t.Fatalf("request instruction = %q", got)
	}
	checkConfigUnchanged(t, cfg)
}

// TestAppendToolsLeavesAgentConfigAlone covers appendTools, which appends to
// the FunctionDeclarations of the first function tool in req.Config.Tools.
func TestAppendToolsLeavesAgentConfigAlone(t *testing.T) {
	cfg := isolationConfig()
	req := &model.LLMRequest{Config: newConfigRequester(t, cfg)()}
	if err := appendTools(req, &mockTool{name: "added"}); err != nil {
		t.Fatal(err)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 2 {
		t.Fatalf("request has %d declarations, want 2", got)
	}
	checkConfigUnchanged(t, cfg)
}

// TestRequestHeadersLeaveAgentConfigAlone covers the Gemini model's
// GenerateContent, which sets tracking headers on req.Config.HTTPOptions.Headers
// in place.
func TestRequestHeadersLeaveAgentConfigAlone(t *testing.T) {
	cfg := isolationConfig()
	req := &model.LLMRequest{Config: newConfigRequester(t, cfg)()}
	req.Config.HTTPOptions.Headers.Set("x-goog-api-client", "adk")
	req.Config.HTTPOptions.Headers["X-Existing"][0] = "2"
	checkConfigUnchanged(t, cfg)
}

// TestOutputSchemaLeavesAgentConfigAlone covers basicRequestProcessor's own
// edit, ResponseSchema and ResponseMIMEType.
func TestOutputSchemaLeavesAgentConfigAlone(t *testing.T) {
	cfg := isolationConfig()
	cfg.Tools = nil
	base, err := agent.New(agent.Config{Name: "agent"})
	if err != nil {
		t.Fatal(err)
	}
	schema := &genai.Schema{Type: genai.TypeObject}
	a := &mockLLMAgent{Agent: base, s: &State{Model: &mockLLM{name: "m"}, GenerateContentConfig: cfg, OutputSchema: schema}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	req := &model.LLMRequest{}
	for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if req.Config.ResponseSchema == nil {
		t.Fatal("basicRequestProcessor did not set ResponseSchema")
	}
	if cfg.ResponseSchema != nil || cfg.ResponseMIMEType != "" {
		t.Errorf("the agent's config got ResponseSchema %v, ResponseMIMEType %q", cfg.ResponseSchema, cfg.ResponseMIMEType)
	}
}
