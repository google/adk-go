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

package llm

import (
	"context"
	"iter"

	"google.golang.org/genai"
)

// Model represents LLM.
type Model interface {
	Name() string
	Generate(ctx context.Context, req *Request) (*Response, error)
	GenerateStream(ctx context.Context, req *Request) iter.Seq2[*Response, error]
}

// Request is the raw LLM request.
type Request struct {
	Contents       []*genai.Content
	GenerateConfig *genai.GenerateContentConfig

	// TODO: this field should be removed
	// any is of the Tool type. Used temporarily to migrate code and avoid cycle dependency.
	Tools map[string]any
}

// Response is the raw LLM response.
// It provides the first candidate response from the model if available.
type Response struct {
	Content           *genai.Content
	CitationMetadata  *genai.CitationMetadata
	GroundingMetadata *genai.GroundingMetadata
	UsageMetadata     *genai.GenerateContentResponseUsageMetadata
	LogprobsResult    *genai.LogprobsResult
	// Partial indicates whether the content is part of a unfinished content stream.
	// Only used for streaming mode and when the content is plain text.
	Partial bool
	// Indicates whether the response from the model is complete.
	// Only used for streaming mode.
	TurnComplete bool
	// Flag indicating that LLM was interrupted when generating the content.
	// Usually it is due to user interruption during a bidi streaming.
	Interrupted  bool
	ErrorCode    string
	ErrorMessage string
	FinishReason genai.FinishReason
	AvgLogprobs  float64
}

func CreateResponse(res *genai.GenerateContentResponse) *Response {
	usageMetadata := res.UsageMetadata
	if len(res.Candidates) > 0 && res.Candidates[0] != nil {
		candidate := res.Candidates[0]
		if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			return &Response{
				Content:           candidate.Content,
				GroundingMetadata: candidate.GroundingMetadata,
				FinishReason:      candidate.FinishReason,
				CitationMetadata:  candidate.CitationMetadata,
				AvgLogprobs:       candidate.AvgLogprobs,
				LogprobsResult:    candidate.LogprobsResult,
				UsageMetadata:     usageMetadata,
			}
		}
		return &Response{
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
		return &Response{
			ErrorCode:     string(res.PromptFeedback.BlockReason),
			ErrorMessage:  res.PromptFeedback.BlockReasonMessage,
			UsageMetadata: usageMetadata,
		}
	}
	return &Response{
		ErrorCode:     "UNKNOWN_ERROR",
		ErrorMessage:  "Unkown error.",
		UsageMetadata: usageMetadata,
	}
}
