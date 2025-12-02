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

package converters

import (
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func Genai2LLMResponse(res *genai.GenerateContentResponse) *model.LLMResponse {
	usageMetadata := res.UsageMetadata
	if len(res.Candidates) > 0 && res.Candidates[0] != nil {
		candidate := res.Candidates[0]
		if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			return &model.LLMResponse{
				Content:           candidate.Content,
				GroundingMetadata: candidate.GroundingMetadata,
				FinishReason:      candidate.FinishReason,
				CitationMetadata:  candidate.CitationMetadata,
				AvgLogprobs:       candidate.AvgLogprobs,
				LogprobsResult:    candidate.LogprobsResult,
				UsageMetadata:     usageMetadata,
			}
		}
		return &model.LLMResponse{
			ErrorCode:         string(candidate.FinishReason),
			ErrorMessage:      candidate.FinishMessage,
			GroundingMetadata: candidate.GroundingMetadata,
			FinishReason:      candidate.FinishReason,
			CitationMetadata:  candidate.CitationMetadata,
			AvgLogprobs:       candidate.AvgLogprobs,
			LogprobsResult:    candidate.LogprobsResult,
			UsageMetadata:     usageMetadata,
		}

	}
	if res.PromptFeedback != nil {
		return &model.LLMResponse{
			ErrorCode:     string(res.PromptFeedback.BlockReason),
			ErrorMessage:  res.PromptFeedback.BlockReasonMessage,
			UsageMetadata: usageMetadata,
		}
	}
	return &model.LLMResponse{
		ErrorCode:     "UNKNOWN_ERROR",
		ErrorMessage:  "Unknown error.",
		UsageMetadata: usageMetadata,
	}
}
