// Copyright 2025 Google LLC
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

package completions

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func toolReq(cfg *genai.GenerateContentConfig) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config:   cfg,
	}
}

var weatherTool = &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{
	Name:        "get_weather",
	Description: "look up the weather",
	Parameters: &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
	},
}}}

// TestConvertTools_NestedFunctionShape pins the extra "function" level Chat
// Completions requires and the flat Responses shape does not have.
func TestConvertTools_NestedFunctionShape(t *testing.T) {
	wire := chatWire(t, toolReq(&genai.GenerateContentConfig{Tools: []*genai.Tool{weatherTool}}))
	tools, ok := wire["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want one", wire["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("type = %v, want function", tool["type"])
	}
	fn, ok := tool["function"].(map[string]any)
	if !ok {
		t.Fatalf("no nested function object: %#v", tool)
	}
	if fn["name"] != "get_weather" || fn["description"] != "look up the weather" {
		t.Errorf("function = %#v", fn)
	}
	// The flat form would have put these on the tool itself.
	if _, ok := tool["name"]; ok {
		t.Error("name on the tool; that is the Responses shape")
	}
	params := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("parameters type = %v, want lowercase object", params["type"])
	}
}

func TestConvertTools_NoParametersDefaultsToEmptyObject(t *testing.T) {
	wire := chatWire(t, toolReq(&genai.GenerateContentConfig{Tools: []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "ping"}},
	}}}))
	fn := wire["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("parameters = %#v, want an empty object schema", params)
	}
}

func TestConvertTools_RejectsNonFunctionTools(t *testing.T) {
	_, err := buildParams("m", toolReq(&genai.GenerateContentConfig{
		Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
	}))
	if err == nil || !strings.Contains(err.Error(), "non-function tools") {
		t.Fatalf("err = %v, want a non-function-tool error", err)
	}
}

func TestConvertToolChoice_Modes(t *testing.T) {
	tests := []struct {
		name   string
		mode   genai.FunctionCallingConfigMode
		want   any
		absent bool
	}{
		{name: "auto leaves it to the provider", mode: genai.FunctionCallingConfigModeAuto, absent: true},
		{name: "none", mode: genai.FunctionCallingConfigModeNone, want: "none"},
		{name: "any becomes required", mode: genai.FunctionCallingConfigModeAny, want: "required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := chatWire(t, toolReq(&genai.GenerateContentConfig{
				Tools:      []*genai.Tool{weatherTool},
				ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: tt.mode}},
			}))
			got, present := wire["tool_choice"]
			if tt.absent {
				if present {
					t.Fatalf("tool_choice = %v, want it absent", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("tool_choice = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertToolChoice_UnsupportedMode(t *testing.T) {
	_, err := buildParams("m", toolReq(&genai.GenerateContentConfig{
		Tools:      []*genai.Tool{weatherTool},
		ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: "TELEPATHY"}},
	}))
	if err == nil || !strings.Contains(err.Error(), "TELEPATHY") {
		t.Fatalf("err = %v, want one naming the mode", err)
	}
}

// TestConvertToolChoice_AllowedTools pins the nesting again: an allowed
// tool names its function one level deeper than the Responses form.
func TestConvertToolChoice_AllowedTools(t *testing.T) {
	for _, tt := range []struct {
		name     string
		mode     genai.FunctionCallingConfigMode
		names    []string
		wantMode string
	}{
		{name: "auto with one name", mode: genai.FunctionCallingConfigModeAuto, names: []string{"get_weather"}, wantMode: "auto"},
		{name: "any with several names", mode: genai.FunctionCallingConfigModeAny, names: []string{"get_weather", "get_time"}, wantMode: "required"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			wire := chatWire(t, toolReq(&genai.GenerateContentConfig{
				Tools: []*genai.Tool{weatherTool},
				ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 tt.mode,
					AllowedFunctionNames: tt.names,
				}},
			}))
			choice, ok := wire["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("tool_choice = %#v, want an allowed-tools object", wire["tool_choice"])
			}
			if choice["type"] != "allowed_tools" {
				t.Fatalf("type = %v, want allowed_tools", choice["type"])
			}
			allowed := choice["allowed_tools"].(map[string]any)
			if allowed["mode"] != tt.wantMode {
				t.Errorf("mode = %v, want %v", allowed["mode"], tt.wantMode)
			}
			entries := allowed["tools"].([]any)
			if len(entries) != len(tt.names) {
				t.Fatalf("allowed tools = %#v, want %d", entries, len(tt.names))
			}
			for i, raw := range entries {
				fn, ok := raw.(map[string]any)["function"].(map[string]any)
				if !ok || fn["name"] != tt.names[i] {
					t.Errorf("allowed tool %d = %#v, want %q nested under function", i, raw, tt.names[i])
				}
			}
		})
	}
}

// TestConvertToolChoice_AnyWithOneNameNamesTheFunction pins the older
// named-function form for the one case it can express, which compatible
// providers accept where many reject allowed_tools.
func TestConvertToolChoice_AnyWithOneNameNamesTheFunction(t *testing.T) {
	wire := chatWire(t, toolReq(&genai.GenerateContentConfig{
		Tools: []*genai.Tool{weatherTool},
		ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 genai.FunctionCallingConfigModeAny,
			AllowedFunctionNames: []string{"", "get_weather"},
		}},
	}))
	choice, ok := wire["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %#v, want a named-function object", wire["tool_choice"])
	}
	if choice["type"] != "function" {
		t.Errorf("type = %v, want function", choice["type"])
	}
	if fn, ok := choice["function"].(map[string]any); !ok || fn["name"] != "get_weather" {
		t.Errorf("tool_choice = %#v, want get_weather named under function", choice)
	}
}

// TestBuildParams_ToolChoiceNeedsTools pins that tool_choice travels only
// with tools: the API rejects one sent without them, "none" included, which a
// turn whose toolset came up empty would otherwise hit.
func TestBuildParams_ToolChoiceNeedsTools(t *testing.T) {
	for _, mode := range []genai.FunctionCallingConfigMode{
		genai.FunctionCallingConfigModeNone,
		genai.FunctionCallingConfigModeAny,
	} {
		t.Run(string(mode), func(t *testing.T) {
			wire := chatWire(t, toolReq(&genai.GenerateContentConfig{
				ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: mode}},
			}))
			if got, ok := wire["tool_choice"]; ok {
				t.Errorf("tool_choice = %v, want it absent without tools", got)
			}
		})
	}
}

func TestConvertToolChoice_AllBlankNamesFallsBack(t *testing.T) {
	wire := chatWire(t, toolReq(&genai.GenerateContentConfig{
		Tools: []*genai.Tool{weatherTool},
		ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 genai.FunctionCallingConfigModeAny,
			AllowedFunctionNames: []string{""},
		}},
	}))
	// Nothing nameable was allowed, so no choice is sent rather than an empty
	// allowed-tools object the API would reject.
	if got, ok := wire["tool_choice"]; ok {
		t.Errorf("tool_choice = %v, want it absent", got)
	}
}
