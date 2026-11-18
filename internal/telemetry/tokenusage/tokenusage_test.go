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

package tokenusage

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
)

// Cases from https://github.com/google/adk-python/blob/16898f30c9a5c6b2ae1ecf099424afff8fd28e77/tests/unittests/telemetry/test_token_usage.py
// that apply to genai Go, where an unreported count is zero.
func TestFromMetadata(t *testing.T) {
	text := func(n int32) []*genai.ModalityTokenCount {
		return []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: n}}
	}
	image := func(n int32) []*genai.ModalityTokenCount {
		return []*genai.ModalityTokenCount{{Modality: genai.MediaModalityImage, TokenCount: n}}
	}
	for _, tc := range []struct {
		name string
		in   genai.GenerateContentResponseUsageMetadata
		want Usage
	}{
		{"input all present", genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, ToolUsePromptTokenCount: 5}, Usage{Input: 15}},
		{"input only prompt", genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}, Usage{Input: 10}},
		{"input only tool", genai.GenerateContentResponseUsageMetadata{ToolUsePromptTokenCount: 5}, Usage{Input: 5}},
		{"output all present", genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 20, ThoughtsTokenCount: 8}, Usage{Output: 28, Reasoning: 8}},
		{"output only candidates", genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: 20}, Usage{Output: 20}},
		{"output only thoughts", genai.GenerateContentResponseUsageMetadata{ThoughtsTokenCount: 8}, Usage{Output: 8, Reasoning: 8}},
		{"nothing", genai.GenerateContentResponseUsageMetadata{}, Usage{}},
		{
			"every count",
			genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 100, ToolUsePromptTokenCount: 20, CandidatesTokenCount: 30, ThoughtsTokenCount: 15, CachedContentTokenCount: 60, TotalTokenCount: 9999},
			Usage{Input: 120, Output: 45, CacheRead: 60, Reasoning: 15},
		},
		{
			// The input details cover both prompt and tool-use tokens.
			"details",
			genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: 20, PromptTokensDetails: text(20),
				ToolUsePromptTokenCount: 5, ToolUsePromptTokensDetails: image(5),
				CandidatesTokenCount: 30, CandidatesTokensDetails: text(30),
				CachedContentTokenCount: 8, CacheTokensDetails: text(8),
			},
			Usage{
				Input: 25, Output: 30, CacheRead: 8,
				InputDetails:     append(text(20), image(5)...),
				OutputDetails:    text(30),
				CacheReadDetails: text(8),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, fromMetadata(&tc.in)); diff != "" {
				t.Errorf("fromMetadata() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFromResponses(t *testing.T) {
	usage := func(prompt, candidates int32) *genai.GenerateContentResponseUsageMetadata {
		return &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: prompt, CandidatesTokenCount: candidates}
	}
	for _, tc := range []struct {
		name      string
		responses []*model.LLMResponse
		want      Usage
	}{
		{
			"keeps usage a trailing response omits",
			[]*model.LLMResponse{{UsageMetadata: usage(10, 4)}, {Partial: true}},
			Usage{Input: 10, Output: 4},
		},
		{
			"takes the newest report",
			[]*model.LLMResponse{{UsageMetadata: usage(10, 1)}, {UsageMetadata: usage(10, 4)}},
			Usage{Input: 10, Output: 4},
		},
		{
			"cache writes reported with the newest report",
			[]*model.LLMResponse{
				{UsageMetadata: usage(10, 1), CustomMetadata: map[string]any{anthropicmodel.CacheCreationInputTokensMetadataKey: int64(5)}},
				{UsageMetadata: usage(10, 4), CustomMetadata: map[string]any{anthropicmodel.CacheCreationInputTokensMetadataKey: int64(6)}},
			},
			Usage{Input: 10, Output: 4, CacheWrite: 6},
		},
		{
			"cache writes of an older report",
			[]*model.LLMResponse{
				{UsageMetadata: usage(10, 1), CustomMetadata: map[string]any{anthropicmodel.CacheCreationInputTokensMetadataKey: int64(5)}},
				{UsageMetadata: usage(10, 4)},
			},
			Usage{Input: 10, Output: 4},
		},
		{"no responses", nil, Usage{}},
		{"no report", []*model.LLMResponse{{}}, Usage{}},
		{
			"report counting no tokens",
			[]*model.LLMResponse{{UsageMetadata: &genai.GenerateContentResponseUsageMetadata{TotalTokenCount: 7}}},
			Usage{},
		},
		{
			// The newest report decides, even when it counts nothing and an
			// older one did, as in ADK Python. Its other counts are dropped.
			"newest report counting no tokens",
			[]*model.LLMResponse{{UsageMetadata: usage(10, 4)}, {UsageMetadata: &genai.GenerateContentResponseUsageMetadata{CachedContentTokenCount: 3}}},
			Usage{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, FromResponses(tc.responses)); diff != "" {
				t.Errorf("FromResponses() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
