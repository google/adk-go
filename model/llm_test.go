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

package model_test

import (
	"reflect"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/converters"
	"google.golang.org/adk/v2/model"
)

const (
	FinishReasonStop       genai.FinishReason = "STOP"
	FinishReasonSafety     genai.FinishReason = "SAFETY"
	FinishReasonRecitation genai.FinishReason = "RECITATION"
)

const (
	BlockedReasonSafety genai.BlockedReason = "SAFETY"
)

func TestCreateResponse(t *testing.T) {
	// Pre-defined complex structs for reuse
	emptyLogprobs := &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{},
		TopCandidates:    []*genai.LogprobsResultTopCandidates{},
	}
	concreteLogprobs := &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{
			{Token: "The", LogProbability: -0.1, TokenID: 123},
			{Token: " capital", LogProbability: -0.5, TokenID: 456},
			{Token: " of", LogProbability: -0.2, TokenID: 789},
		},
		TopCandidates: []*genai.LogprobsResultTopCandidates{
			{Candidates: []*genai.LogprobsResultCandidate{{Token: "The"}, {Token: "A"}, {Token: "This"}}},
			{Candidates: []*genai.LogprobsResultCandidate{{Token: " capital"}, {Token: " city"}, {Token: " main"}}},
		},
	}
	partialLogprobs := &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{
			{Token: "Hello", LogProbability: -0.05, TokenID: 111},
			{Token: " world", LogProbability: -0.8, TokenID: 222},
		},
		TopCandidates: []*genai.LogprobsResultTopCandidates{},
	}
	citationMeta := &genai.CitationMetadata{
		Citations: []*genai.Citation{{StartIndex: 0, EndIndex: 10, URI: "https://example.com"}},
	}

	testCases := []struct {
		name  string
		input genai.GenerateContentResponse
		want  model.LLMResponse
	}{
		{
			name: "CreateWithLogprobs",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:        &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
					FinishReason:   FinishReasonStop,
					AvgLogprobs:    -0.75,
					LogprobsResult: emptyLogprobs,
				}},
			},
			want: model.LLMResponse{
				Content:        &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
				FinishReason:   FinishReasonStop,
				AvgLogprobs:    -0.75,
				LogprobsResult: emptyLogprobs,
			},
		},
		{
			name: "CreateWithoutLogprobs",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:      &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
					FinishReason: FinishReasonStop,
				}},
			},
			want: model.LLMResponse{
				Content:      &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
				FinishReason: FinishReasonStop,
			},
		},
		{
			name: "CreateErrorCaseWithLogprobs",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					FinishReason:  FinishReasonSafety,
					FinishMessage: "Safety filter triggered",
					AvgLogprobs:   -2.1,
				}},
			},
			want: model.LLMResponse{
				ErrorCode:    string(FinishReasonSafety),
				ErrorMessage: "Safety filter triggered",
				AvgLogprobs:  -2.1,
				FinishReason: FinishReasonSafety,
			},
		},
		{
			name: "CreateNoCandidates",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{},
				PromptFeedback: &genai.GenerateContentResponsePromptFeedback{
					BlockReason:        BlockedReasonSafety,
					BlockReasonMessage: "Prompt blocked for safety",
				},
			},
			want: model.LLMResponse{
				ErrorCode:    string(BlockedReasonSafety),
				ErrorMessage: "Prompt blocked for safety",
			},
		},
		{
			name: "CreateWithConcreteLogprobsResult",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:        &genai.Content{Parts: []*genai.Part{{Text: "The capital of France is Paris."}}},
					FinishReason:   FinishReasonStop,
					AvgLogprobs:    -0.27,
					LogprobsResult: concreteLogprobs,
				}},
			},
			want: model.LLMResponse{
				Content:        &genai.Content{Parts: []*genai.Part{{Text: "The capital of France is Paris."}}},
				FinishReason:   FinishReasonStop,
				AvgLogprobs:    -0.27,
				LogprobsResult: concreteLogprobs,
			},
		},
		{
			name: "CreateWithPartial*genai.LogprobsResult",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:        &genai.Content{Parts: []*genai.Part{{Text: "Hello world"}}},
					FinishReason:   FinishReasonStop,
					AvgLogprobs:    -0.425,
					LogprobsResult: partialLogprobs,
				}},
			},
			want: model.LLMResponse{
				Content:        &genai.Content{Parts: []*genai.Part{{Text: "Hello world"}}},
				FinishReason:   FinishReasonStop,
				AvgLogprobs:    -0.425,
				LogprobsResult: partialLogprobs,
			},
		},
		{
			name: "CreateWithCitationMetadata",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:          &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
					FinishReason:     FinishReasonStop,
					CitationMetadata: citationMeta,
				}},
			},
			want: model.LLMResponse{
				Content:          &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
				FinishReason:     FinishReasonStop,
				CitationMetadata: citationMeta,
			},
		},
		{
			name: "CreateWithoutCitationMetadata",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:      &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
					FinishReason: FinishReasonStop,
				}},
			},
			want: model.LLMResponse{
				Content:      &genai.Content{Parts: []*genai.Part{{Text: "Response text"}}},
				FinishReason: FinishReasonStop,
			},
		},
		{
			name: "CreateErrorCaseWithCitationMetadata",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					FinishReason:     FinishReasonRecitation,
					FinishMessage:    "Response blocked due to recitation triggered",
					CitationMetadata: citationMeta,
				}},
			},
			want: model.LLMResponse{
				ErrorCode:        string(FinishReasonRecitation),
				ErrorMessage:     "Response blocked due to recitation triggered",
				CitationMetadata: citationMeta,
				FinishReason:     FinishReasonRecitation,
			},
		},
		{
			name: "CreateEmptyContentWithStopReason",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:      &genai.Content{Parts: []*genai.Part{}},
					FinishReason: FinishReasonStop,
				}},
			},
			want: model.LLMResponse{
				Content:      &genai.Content{Parts: []*genai.Part{}},
				FinishReason: FinishReasonStop,
			},
		},
		{
			name: "CreateErrorCaseDropsGroundingMetadata",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					Content:           &genai.Content{Role: "model", Parts: []*genai.Part{}},
					FinishReason:      FinishReasonSafety,
					FinishMessage:     "Safety filter triggered",
					GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q1"}},
					LogprobsResult:    emptyLogprobs,
				}},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5, TotalTokenCount: 5},
				ModelVersion:  "gemini-2.5-flash",
			},
			want: model.LLMResponse{
				ErrorCode:      string(FinishReasonSafety),
				ErrorMessage:   "Safety filter triggered",
				FinishReason:   FinishReasonSafety,
				LogprobsResult: emptyLogprobs,
				UsageMetadata:  &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5, TotalTokenCount: 5},
				ModelVersion:   "gemini-2.5-flash",
			},
		},
		{
			name: "CreateSuccessWithGroundingAndSecondCandidateIgnored",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{
					{
						Content:           &genai.Content{Parts: []*genai.Part{{Text: "first"}}},
						FinishReason:      FinishReasonStop,
						GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q1"}},
					},
					{
						Content:           &genai.Content{Parts: []*genai.Part{{Text: "second"}}},
						FinishReason:      FinishReasonSafety,
						GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q2"}},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 3, CandidatesTokenCount: 1, TotalTokenCount: 4},
				ModelVersion:  "gemini-2.5-flash",
			},
			want: model.LLMResponse{
				Content:           &genai.Content{Parts: []*genai.Part{{Text: "first"}}},
				FinishReason:      FinishReasonStop,
				GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: []string{"q1"}},
				UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 3, CandidatesTokenCount: 1, TotalTokenCount: 4},
				ModelVersion:      "gemini-2.5-flash",
			},
		},
		{
			name: "CreateNoCandidatesWithUsageAndModelVersion",
			input: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{},
				PromptFeedback: &genai.GenerateContentResponsePromptFeedback{
					BlockReason:        BlockedReasonSafety,
					BlockReasonMessage: "Prompt blocked",
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 4, TotalTokenCount: 4},
				ModelVersion:  "gemini-2.5-flash",
			},
			want: model.LLMResponse{
				ErrorCode:     string(BlockedReasonSafety),
				ErrorMessage:  "Prompt blocked",
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 4, TotalTokenCount: 4},
				ModelVersion:  "gemini-2.5-flash",
			},
		},
		{
			name: "CreateEmptyCandidatesWithoutPromptFeedback",
			input: genai.GenerateContentResponse{
				Candidates:    []*genai.Candidate{},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, TotalTokenCount: 10},
				ModelVersion:  "gemini-2.5-flash",
			},
			want: model.LLMResponse{
				Content:       &genai.Content{Role: "model", Parts: []*genai.Part{}},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, TotalTokenCount: 10},
				ModelVersion:  "gemini-2.5-flash",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := converters.Genai2LLMResponse(&tc.input)

			if tc.want.AvgLogprobs != got.AvgLogprobs {
				t.Errorf("AvgLogprobs mismatch: want %f, got %f", tc.want.AvgLogprobs, got.AvgLogprobs)
			}

			if got.ErrorCode != tc.want.ErrorCode {
				t.Errorf("ErrorCode mismatch: want %v, got %v", tc.want.ErrorCode, got.ErrorCode)
			}

			if got.ErrorMessage != tc.want.ErrorMessage {
				t.Errorf("ErrorMessage mismatch: want '%s', got '%s'", tc.want.ErrorMessage, got.ErrorMessage)
			}

			if got.FinishReason != tc.want.FinishReason {
				t.Errorf("FinishReason mismatch: want %s, got %s", tc.want.FinishReason, got.FinishReason)
			}

			// Use DeepEqual for complex nested structs
			if !reflect.DeepEqual(got.Content, tc.want.Content) {
				t.Errorf("Content mismatch: want %+v, got %+v", tc.want.Content, got.Content)
			}

			if !reflect.DeepEqual(got.LogprobsResult, tc.want.LogprobsResult) {
				t.Errorf("*genai.LogprobsResult mismatch: want %+v, got %+v", tc.want.LogprobsResult, got.LogprobsResult)
			}

			if !reflect.DeepEqual(got.CitationMetadata, tc.want.CitationMetadata) {
				t.Errorf("CitationMetadata mismatch: want %+v, got %+v", tc.want.CitationMetadata, got.CitationMetadata)
			}

			if !reflect.DeepEqual(got.GroundingMetadata, tc.want.GroundingMetadata) {
				t.Errorf("GroundingMetadata mismatch: want %+v, got %+v", tc.want.GroundingMetadata, got.GroundingMetadata)
			}

			if !reflect.DeepEqual(got.UsageMetadata, tc.want.UsageMetadata) {
				t.Errorf("UsageMetadata mismatch: want %+v, got %+v", tc.want.UsageMetadata, got.UsageMetadata)
			}

			if got.ModelVersion != tc.want.ModelVersion {
				t.Errorf("ModelVersion mismatch: want %q, got %q", tc.want.ModelVersion, got.ModelVersion)
			}
		})
	}
}
