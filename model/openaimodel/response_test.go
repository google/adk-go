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

package openaimodel

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
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

func TestConvertResponse_Logprobs(t *testing.T) {
	tests := []struct {
		name       string
		logprobs   []responses.ResponseOutputTextLogprob
		wantResult *genai.LogprobsResult
	}{
		{
			name:       "empty config",
			logprobs:   nil,
			wantResult: nil,
		},
		{
			name: "fully specified",
			logprobs: []responses.ResponseOutputTextLogprob{
				{
					Token:   "hel",
					Logprob: -0.1,
					TopLogprobs: []responses.ResponseOutputTextLogprobTopLogprob{
						{Token: "hel", Logprob: -0.1},
						{Token: "hi", Logprob: -2.3},
					},
				},
				{
					Token:   "lo",
					Logprob: -0.2,
				},
			},
			wantResult: &genai.LogprobsResult{
				ChosenCandidates: []*genai.LogprobsResultCandidate{
					{Token: "hel", LogProbability: -0.1},
					{Token: "lo", LogProbability: -0.2},
				},
				TopCandidates: []*genai.LogprobsResultTopCandidates{
					{
						Candidates: []*genai.LogprobsResultCandidate{
							{Token: "hel", LogProbability: -0.1},
							{Token: "hi", LogProbability: -2.3},
						},
					},
					{
						Candidates: nil,
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := &responses.Response{
				ID:    "resp-1",
				Model: "gpt-test",
				Output: []responses.ResponseOutputItemUnion{
					{
						Type: "message",
						Content: []responses.ResponseOutputMessageContentUnion{
							{
								Type:     "output_text",
								Text:     "hello",
								Logprobs: tc.logprobs,
							},
						},
					},
				},
			}
			got, err := convertResponse(resp)
			if err != nil {
				t.Fatalf("convertResponse() err = %v", err)
			}
			if got.Candidates == nil {
				t.Fatalf("expected candidates")
			}
			cand := got.Candidates[0]
			if diff := cmp.Diff(tc.wantResult, cand.LogprobsResult); diff != "" {
				t.Errorf("LogprobsResult mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertResponse_IncompleteDetails(t *testing.T) {
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
		IncompleteDetails: responses.ResponseIncompleteDetails{
			Reason: "max_output_tokens",
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	if got.PromptFeedback != nil {
		t.Errorf("expected PromptFeedback to be nil, got: %+v", got.PromptFeedback)
	}
	if got.Candidates == nil {
		t.Fatalf("expected candidates")
	}
	if got.Candidates[0].FinishReason != genai.FinishReasonMaxTokens {
		t.Errorf("expected FinishReasonMaxTokens, got: %v", got.Candidates[0].FinishReason)
	}
}

func TestConvertFunctionCall(t *testing.T) {
	tests := []struct {
		name     string
		call     responses.ResponseOutputItemUnion
		wantErr  bool
		wantID   string
		wantName string
		wantArg  string
	}{
		{
			name: "valid",
			call: responses.ResponseOutputItemUnion{
				CallID:    "call-1",
				Name:      "test_fn",
				Arguments: `{"arg":"val"}`,
			},
			wantID:   "call-1",
			wantName: "test_fn",
			wantArg:  "val",
		},
		{
			name: "bad json",
			call: responses.ResponseOutputItemUnion{
				CallID:    "call-1",
				Name:      "test_fn",
				Arguments: `{bad`,
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertFunctionCall(tc.call)
			if (err != nil) != tc.wantErr {
				t.Fatalf("convertFunctionCall() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if got == nil {
				t.Fatalf("expected result, got nil")
			}
			if got.FunctionCall.ID != tc.wantID || got.FunctionCall.Name != tc.wantName {
				t.Fatalf("unexpected fields: %+v", got)
			}
			if got.FunctionCall.Args["arg"] != tc.wantArg {
				t.Fatalf("unexpected args: %+v", got.FunctionCall.Args)
			}
		})
	}
}

func TestFinishReason(t *testing.T) {
	tests := []struct {
		reason string
		want   genai.FinishReason
	}{
		{"stop", genai.FinishReasonOther},
		{"max_output_tokens", genai.FinishReasonMaxTokens},
		{"content_filter", genai.FinishReasonSafety},
		{"", genai.FinishReasonStop},
		{"other", genai.FinishReasonOther},
	}
	for _, tc := range tests {
		t.Run(tc.reason, func(t *testing.T) {
			resp := &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: tc.reason,
				},
			}
			got := finishReason(resp)
			if got != tc.want {
				t.Errorf("finishReason(%q) = %v, want %v", tc.reason, got, tc.want)
			}
		})
	}

	t.Run("nil response", func(t *testing.T) {
		if got := finishReason(nil); got != genai.FinishReasonUnspecified {
			t.Errorf("finishReason(nil) = %v, want Unspecified", got)
		}
	})
}

func TestConvertOutputItems(t *testing.T) {
	tests := []struct {
		name    string
		items   []responses.ResponseOutputItemUnion
		want    []*genai.Part
		wantErr error
	}{
		{
			name: "valid items",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "output_text", Text: "text1"},
						{Type: "refusal", Refusal: "nope"},
					},
				},
				{
					Type:      "function_call",
					CallID:    "call-1",
					Name:      "fn",
					Arguments: `{}`,
				},
				{
					Type: "reasoning",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Text: "thought1"},
					},
					Summary: []responses.ResponseReasoningItemSummary{
						{Text: "summary1"},
					},
				},
			},
			want: []*genai.Part{
				{Text: "text1"},
				{Text: "nope"},
				{
					FunctionCall: &genai.FunctionCall{
						Name: "fn",
						ID:   "call-1",
						Args: map[string]any{},
					},
				},
				{Text: "thought1", Thought: true},
				{Text: "summary1", Thought: true},
			},
		},
		{
			name:    "empty items",
			items:   nil,
			wantErr: ErrNoOutputItems,
		},
		{
			name: "invalid type",
			items: []responses.ResponseOutputItemUnion{
				{Type: "invalid"},
			},
			wantErr: ErrUnsupportedOutputItemType,
		},
		{
			name: "invalid message content type",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "invalid"},
					},
				},
			},
			wantErr: ErrUnsupportedMessageContentType,
		},
		{
			name: "empty message content",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "output_text", Text: ""}, // Empty text is skipped
					},
				},
			},
			wantErr: ErrNoTextOrToolContent,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parts, err := convertOutputItems(tc.items)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("convertOutputItems() error = %v, wantErr %v", err, tc.wantErr)
			}
			if len(parts) != len(tc.want) {
				t.Fatalf("expected %d parts, got %d", len(tc.want), len(parts))
			}
			if diff := cmp.Diff(tc.want, parts); diff != "" {
				t.Errorf("convertOutputItems() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPromptFeedback(t *testing.T) {
	tests := []struct {
		name string
		resp *responses.Response
		want *genai.GenerateContentResponsePromptFeedback
	}{
		{
			name: "content filter",
			resp: &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: "content_filter",
				},
			},
			want: &genai.GenerateContentResponsePromptFeedback{
				BlockReason:        genai.BlockedReasonSafety,
				BlockReasonMessage: "content_filter",
			},
		},
		{
			name: "max_output_tokens",
			resp: &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: "max_output_tokens",
				},
			},
			want: nil,
		},
		{
			name: "nil response",
			resp: nil,
			want: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := promptFeedback(tc.resp)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("promptFeedback() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertLogprobs(t *testing.T) {
	tests := []struct {
		name  string
		items []responses.ResponseOutputItemUnion
		want  *genai.LogprobsResult
	}{
		{
			name:  "empty items",
			items: nil,
			want:  nil,
		},
		{
			name: "fully specified",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{
							Type: "output_text",
							Logprobs: []responses.ResponseOutputTextLogprob{
								{
									Token:   "hel",
									Logprob: -0.1,
									TopLogprobs: []responses.ResponseOutputTextLogprobTopLogprob{
										{Token: "hel", Logprob: -0.1},
										{Token: "hi", Logprob: -2.3},
									},
								},
								{
									Token:   "lo",
									Logprob: -0.2,
								},
							},
						},
					},
				},
			},
			want: &genai.LogprobsResult{
				ChosenCandidates: []*genai.LogprobsResultCandidate{
					{Token: "hel", LogProbability: -0.1},
					{Token: "lo", LogProbability: -0.2},
				},
				TopCandidates: []*genai.LogprobsResultTopCandidates{
					{
						Candidates: []*genai.LogprobsResultCandidate{
							{Token: "hel", LogProbability: -0.1},
							{Token: "hi", LogProbability: -2.3},
						},
					},
					{
						Candidates: nil,
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := convertLogprobs(tc.items)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("convertLogprobs() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
