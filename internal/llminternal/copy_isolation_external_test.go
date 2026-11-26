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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin/functioncallmodifier"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// Mutators of req.Config outside this package, run on the copy
// basicRequestProcessor makes (CloneGenerateContentConfig).

// toolConfig returns an agent config whose Tools and FunctionDeclarations
// have spare capacity, so an append through a shared backing array would
// write into the agent's.
func toolConfig() *genai.GenerateContentConfig {
	decls := make([]*genai.FunctionDeclaration, 1, 4)
	decls[0] = &genai.FunctionDeclaration{Name: "existing", Description: "d", Parameters: &genai.Schema{Type: genai.TypeObject}}
	tools := make([]*genai.Tool, 1, 4)
	tools[0] = &genai.Tool{FunctionDeclarations: decls}
	return &genai.GenerateContentConfig{Tools: tools}
}

func checkToolConfigUnchanged(t *testing.T, cfg *genai.GenerateContentConfig) {
	t.Helper()
	if diff := cmp.Diff(toolConfig(), cfg); diff != "" {
		t.Errorf("the agent's config changed (-want +got):\n%s", diff)
	}
	decls := cfg.Tools[0].FunctionDeclarations
	if spare := decls[:cap(decls)][len(decls)]; spare != nil {
		t.Errorf("the agent's FunctionDeclarations got %v written past its length", spare)
	}
	if spare := cfg.Tools[:cap(cfg.Tools)][len(cfg.Tools)]; spare != nil {
		t.Errorf("the agent's Tools got %v written past its length", spare)
	}
}

// TestGeminiToolLeavesAgentConfigAlone covers the geminitool tools, which
// append to req.Config.Tools.
func TestGeminiToolLeavesAgentConfigAlone(t *testing.T) {
	cfg := toolConfig()
	req := &model.LLMRequest{Model: "gemini-2.5-flash", Config: llminternal.CloneGenerateContentConfig(cfg)}
	if err := (geminitool.GoogleSearch{}).ProcessRequest(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := len(req.Config.Tools); got != 2 {
		t.Fatalf("request has %d tools, want 2", got)
	}
	checkToolConfigUnchanged(t, cfg)
}

type packedTool struct{}

func (packedTool) Name() string { return "packed" }
func (packedTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "packed"}
}

// TestPackToolLeavesAgentConfigAlone covers toolutils.PackTool, which appends
// to the FunctionDeclarations of the first function tool in req.Config.Tools.
func TestPackToolLeavesAgentConfigAlone(t *testing.T) {
	cfg := toolConfig()
	req := &model.LLMRequest{Config: llminternal.CloneGenerateContentConfig(cfg)}
	if err := toolutils.PackTool(req, packedTool{}); err != nil {
		t.Fatal(err)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 2 {
		t.Fatalf("request has %d declarations, want 2", got)
	}
	checkToolConfigUnchanged(t, cfg)
}

// TestFunctionCallModifierLeavesAgentConfigAlone covers the
// functioncallmodifier plugin, which replaces entries of
// req.Config.Tools[i].FunctionDeclarations in place.
func TestFunctionCallModifierLeavesAgentConfigAlone(t *testing.T) {
	cfg := toolConfig()
	req := &model.LLMRequest{
		Config: llminternal.CloneGenerateContentConfig(cfg),
		Tools:  map[string]any{"existing": nil},
	}
	p := functioncallmodifier.MustNewPlugin(functioncallmodifier.FunctionCallModifierConfig{
		Predicate:           func(string) bool { return true },
		Args:                map[string]*genai.Schema{"extra": {Type: genai.TypeString}},
		OverrideDescription: func(d string) string { return d + " (modified)" },
	})
	if _, err := p.BeforeModelCallback()(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := req.Config.Tools[0].FunctionDeclarations[0].Description; got != "d (modified)" {
		t.Fatalf("request declaration description = %q", got)
	}
	checkToolConfigUnchanged(t, cfg)
}
