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

package telemetry

import (
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// tokenUsage is what one model call spent, split the way the client
// inference metrics report it.
type tokenUsage struct {
	input      int64 // includes tool-use prompt tokens
	output     int64 // includes reasoning tokens
	cacheRead  int64
	cacheWrite int64
	reasoning  int64

	// Per-modality breakdowns of input, output and cacheRead, if the model
	// gives them.
	inputDetails     []*genai.ModalityTokenCount
	outputDetails    []*genai.ModalityTokenCount
	cacheReadDetails []*genai.ModalityTokenCount
}

func usageFromMetadata(u *genai.GenerateContentResponseUsageMetadata) tokenUsage {
	return tokenUsage{
		input:            int64(u.PromptTokenCount) + int64(u.ToolUsePromptTokenCount),
		output:           int64(u.CandidatesTokenCount) + int64(u.ThoughtsTokenCount),
		cacheRead:        int64(u.CachedContentTokenCount),
		reasoning:        int64(u.ThoughtsTokenCount),
		inputDetails:     slices.Concat(u.PromptTokensDetails, u.ToolUsePromptTokensDetails),
		outputDetails:    u.CandidatesTokensDetails,
		cacheReadDetails: u.CacheTokensDetails,
	}
}

// cacheWriteKey is the custom metadata key under which the Anthropic model
// reports the tokens a response wrote to the prompt cache. Duplicated so that
// telemetry does not import the Anthropic SDK.
const cacheWriteKey = "anthropic_cache_creation_input_tokens"

// usageFromResponses returns what one model call spent: its newest usage
// report, as each report covers the whole call so far. It returns the zero
// value when no response reports usage, or when the newest report counts
// neither input nor output tokens.
func usageFromResponses(responses []*model.LLMResponse) tokenUsage {
	for _, r := range slices.Backward(responses) {
		if r.UsageMetadata == nil {
			continue
		}
		if u := usageFromMetadata(r.UsageMetadata); u.input != 0 || u.output != 0 {
			u.cacheWrite, _ = r.CustomMetadata[cacheWriteKey].(int64)
			return u
		}
		return tokenUsage{}
	}
	return tokenUsage{}
}
