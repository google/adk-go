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

package anthropicmodel

import (
	"math"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

// TestExtractUsageMetadata_SaturatesAtMaxInt32 pins that every count in the
// usage metadata saturates at math.MaxInt32 and is never negative.
func TestExtractUsageMetadata_SaturatesAtMaxInt32(t *testing.T) {
	tests := []struct {
		name  string
		usage anthropic.Usage
		want  *genai.GenerateContentResponseUsageMetadata
	}{
		{
			name:  "counts that fit are unchanged",
			usage: anthropic.Usage{InputTokens: 100, CacheReadInputTokens: 20, CacheCreationInputTokens: 30, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        150,
				CandidatesTokenCount:    10,
				TotalTokenCount:         160,
				CachedContentTokenCount: 20,
			},
		},
		{
			name:  "input one above MaxInt32",
			usage: anthropic.Usage{InputTokens: math.MaxInt32 + 1, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     math.MaxInt32,
				CandidatesTokenCount: 10,
				TotalTokenCount:      math.MaxInt32,
			},
		},
		{
			name:  "input cache read and cache creation of 2 to the 30 each",
			usage: anthropic.Usage{InputTokens: 1 << 30, CacheReadInputTokens: 1 << 30, CacheCreationInputTokens: 1 << 30, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        math.MaxInt32,
				CandidatesTokenCount:    10,
				TotalTokenCount:         math.MaxInt32,
				CachedContentTokenCount: 1 << 30,
			},
		},
		{
			name:  "input of 2 to the 32",
			usage: anthropic.Usage{InputTokens: 1 << 32, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     math.MaxInt32,
				CandidatesTokenCount: 10,
				TotalTokenCount:      math.MaxInt32,
			},
		},
		{
			name:  "output of 2 to the 31",
			usage: anthropic.Usage{InputTokens: 100, OutputTokens: 1 << 31},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     100,
				CandidatesTokenCount: math.MaxInt32,
				TotalTokenCount:      math.MaxInt32,
			},
		},
		{
			name:  "cache read of 2 to the 31",
			usage: anthropic.Usage{CacheReadInputTokens: 1 << 31, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        math.MaxInt32,
				CandidatesTokenCount:    10,
				TotalTokenCount:         math.MaxInt32,
				CachedContentTokenCount: math.MaxInt32,
			},
		},
		{
			name: "output of 2 to the 32 with thinking of 2 to the 31",
			usage: anthropic.Usage{
				InputTokens:         100,
				OutputTokens:        1 << 32,
				OutputTokensDetails: anthropic.OutputTokensDetails{ThinkingTokens: 1 << 31},
			},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     100,
				CandidatesTokenCount: math.MaxInt32,
				TotalTokenCount:      math.MaxInt32,
				ThoughtsTokenCount:   math.MaxInt32,
			},
		},
		{
			name:  "prompt fits but prompt plus output does not",
			usage: anthropic.Usage{InputTokens: math.MaxInt32, OutputTokens: 10},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     math.MaxInt32,
				CandidatesTokenCount: 10,
				TotalTokenCount:      math.MaxInt32,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := extractUsageMetadata(tc.usage)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("extractUsageMetadata() mismatch (-want +got); counts must saturate at math.MaxInt32 and never be negative:\n%s", diff)
			}
		})
	}
}
