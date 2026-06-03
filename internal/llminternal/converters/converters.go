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
		if (candidate.Content != nil && len(candidate.Content.Parts) > 0) || candidate.FinishReason == genai.FinishReasonStop {
			return &model.LLMResponse{
				Content:           candidate.Content,
				GroundingMetadata: candidate.GroundingMetadata,
				FinishReason:      candidate.FinishReason,
				CitationMetadata:  candidate.CitationMetadata,
				AvgLogprobs:       candidate.AvgLogprobs,
				LogprobsResult:    candidate.LogprobsResult,
				UsageMetadata:     usageMetadata,
				ModelVersion:      res.ModelVersion,
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
			ModelVersion:      res.ModelVersion,
		}
	}
	if res.PromptFeedback != nil {
		return &model.LLMResponse{
			ErrorCode:     string(res.PromptFeedback.BlockReason),
			ErrorMessage:  res.PromptFeedback.BlockReasonMessage,
			UsageMetadata: usageMetadata,
			ModelVersion:  res.ModelVersion,
		}
	}
	// no candidates, no prompt feedback.
	// Sometimes gemini-3* invoked via aiplatform (VertexAI) sends empty entries at the beginning.
	// sample stream of SSE chunks (first 3):
	// data: {"candidates": [{"content": {"role": "model","parts": [{"text": ""}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-3.1-flash-lite","createTime": "2026-05-28T09:40:03.380865Z","responseId": "REDACTED"}
	//
	// data: {"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-3.1-flash-lite","createTime": "2026-05-28T09:40:03.380865Z","responseId": "REDACTED"}
	//
	// data: {"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-3.1-flash-lite","createTime": "2026-05-28T09:40:03.380865Z","responseId": "REDACTED"}
	// we should treat them as valid, empty responses and let the downstream to process usageMetadata
	return &model.LLMResponse{
		Content:       &genai.Content{Parts: []*genai.Part{}, Role: "model"},
		UsageMetadata: usageMetadata,
		ModelVersion:  res.ModelVersion,
	}
}
