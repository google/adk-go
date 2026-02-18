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

package openai

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/responses"
)

func TestConvertResponse_Text(t *testing.T) {
	resp := &responses.Response{
		ID:    "resp-1",
		Model: "gpt-test",
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "hello"},
				},
			},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  5,
			OutputTokens: 2,
			TotalTokens:  7,
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	if got.Candidates == nil || got.Candidates[0].Content.Parts[0].Text != "hello" {
		t.Fatalf("unexpected candidate contents: %+v", got.Candidates)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.PromptTokenCount != 5 {
		t.Fatalf("usage metadata missing: %+v", got.UsageMetadata)
	}
}

func TestConvertResponse_Refusal(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "refusal", Refusal: "nope"},
				},
			},
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	part := got.Candidates[0].Content.Parts[0]
	if diff := cmp.Diff("nope", part.Text); diff != "" {
		t.Fatalf("refusal mismatch (-want +got):\n%s", diff)
	}
}

func TestConvertResponse_NoOutput(t *testing.T) {
	_, err := convertResponse(&responses.Response{})
	if err == nil {
		t.Fatalf("expected error for empty output")
	}
}
