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

// Package tokenusage works out the tokens one model call spent, the way the
// client inference metrics report them.
package tokenusage

import (
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/modelmetadata"
	"google.golang.org/adk/v2/model"
)

// Usage is what one model call spent, split the way the client inference
// metrics report it.
type Usage struct {
	Input      int64 // includes tool-use prompt tokens
	Output     int64 // includes reasoning tokens
	CacheRead  int64
	CacheWrite int64
	Reasoning  int64

	// Per-modality breakdowns of Input, Output and CacheRead, if the model
	// gives them.
	InputDetails     []*genai.ModalityTokenCount
	OutputDetails    []*genai.ModalityTokenCount
	CacheReadDetails []*genai.ModalityTokenCount
}

func fromMetadata(u *genai.GenerateContentResponseUsageMetadata) Usage {
	return Usage{
		Input:            int64(u.PromptTokenCount) + int64(u.ToolUsePromptTokenCount),
		Output:           int64(u.CandidatesTokenCount) + int64(u.ThoughtsTokenCount),
		CacheRead:        int64(u.CachedContentTokenCount),
		Reasoning:        int64(u.ThoughtsTokenCount),
		InputDetails:     slices.Concat(u.PromptTokensDetails, u.ToolUsePromptTokensDetails),
		OutputDetails:    u.CandidatesTokensDetails,
		CacheReadDetails: u.CacheTokensDetails,
	}
}

// FromResponses returns what one model call spent: its newest usage
// report, as each report covers the whole call so far. It returns the zero
// value when no response reports usage, or when the newest report counts
// neither input nor output tokens.
func FromResponses(responses []*model.LLMResponse) Usage {
	for _, r := range slices.Backward(responses) {
		if r.UsageMetadata == nil {
			continue
		}
		if u := fromMetadata(r.UsageMetadata); u.Input != 0 || u.Output != 0 {
			u.CacheWrite, _ = r.CustomMetadata[modelmetadata.AnthropicCacheCreationInputTokens].(int64)
			return u
		}
		return Usage{}
	}
	return Usage{}
}
