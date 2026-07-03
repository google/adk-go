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
	"context"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/platform"
)

func TestGenerateFunctionCallIDUsesProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(context.Background(), func() string { return "fixed" })

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
	first := utils.GenerateFunctionCallID(context.Background())
	second := utils.GenerateFunctionCallID(context.Background())

	if first == second {
		t.Errorf("GenerateFunctionCallID() returned %q twice; want unique values", first)
	}
}

func TestPopulateClientFunctionCallIDUsesProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(context.Background(), func() string { return "generated" })

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
