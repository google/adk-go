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

package utils_test

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/platform"
)

func TestGenerateFunctionCallIDUsesProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "fixed" })

	got := utils.GenerateFunctionCallID(ctx)

	// The generated ID must carry the "adk-" prefix that RemoveClientFunctionCallID
	// relies on, and must incorporate the value from the installed provider.
	if !strings.HasPrefix(got, "adk-") {
		t.Errorf("GenerateFunctionCallID() = %q, want \"adk-\" prefix", got)
	}
	if !strings.HasSuffix(got, "fixed") {
		t.Errorf("GenerateFunctionCallID() = %q, want it to use the provider value %q", got, "fixed")
	}
}

func TestGenerateFunctionCallIDDefaultIsUnique(t *testing.T) {
	first := utils.GenerateFunctionCallID(t.Context())
	second := utils.GenerateFunctionCallID(t.Context())

	if first == second {
		t.Errorf("GenerateFunctionCallID() returned %q twice; want unique values", first)
	}
}

func TestPopulateClientFunctionCallIDUsesProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "generated" })

	content := &genai.Content{
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "needs_id"}},
			{FunctionCall: &genai.FunctionCall{ID: "keep", Name: "has_id"}},
		},
	}

	utils.PopulateClientFunctionCallID(ctx, content)

	if got := content.Parts[0].FunctionCall.ID; got != "adk-generated" {
		t.Errorf("empty function call ID = %q, want %q", got, "adk-generated")
	}
	if got := content.Parts[1].FunctionCall.ID; got != "keep" {
		t.Errorf("preset function call ID = %q, want it left untouched (%q)", got, "keep")
	}
}

// A *genai.Part element can be nil, so the accessors must skip nil entries
// instead of dereferencing them.
func TestPartAccessorsSkipNilParts(t *testing.T) {
	call := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "call"}}
	resp := &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "resp"}}
	text := &genai.Part{Text: "hi"}

	for _, tc := range []struct {
		name  string
		parts func(keep *genai.Part) []*genai.Part
		want  int
	}{
		{"leading", func(keep *genai.Part) []*genai.Part { return []*genai.Part{nil, keep} }, 1},
		{"middle", func(keep *genai.Part) []*genai.Part { return []*genai.Part{keep, nil, keep} }, 2},
		{"trailing", func(keep *genai.Part) []*genai.Part { return []*genai.Part{keep, nil} }, 1},
		{"only", func(keep *genai.Part) []*genai.Part { return []*genai.Part{nil} }, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := utils.FunctionCalls(&genai.Content{Parts: tc.parts(call)}); len(got) != tc.want {
				t.Errorf("FunctionCalls() returned %d calls, want %d", len(got), tc.want)
			}
			if got := utils.FunctionResponses(&genai.Content{Parts: tc.parts(resp)}); len(got) != tc.want {
				t.Errorf("FunctionResponses() returned %d responses, want %d", len(got), tc.want)
			}
			if got := utils.TextParts(&genai.Content{Parts: tc.parts(text)}); len(got) != tc.want {
				t.Errorf("TextParts() returned %d texts, want %d", len(got), tc.want)
			}
		})
	}
}
