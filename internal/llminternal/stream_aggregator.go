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

package llminternal

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

// streamingResponseAggregator aggregates partial streaming responses.
// It aggregates content from partial responses, and generates LlmResponses for
// individual (partial) model responses, as well as for aggregated content.
type streamingResponseAggregator struct {
	text        string
	thoughtText string
	response    *llm.Response
	role        string
}

// NewStreamingResponseAggregator creates a new, initialized streamingResponseAggregator.
func NewStreamingResponseAggregator() *streamingResponseAggregator {
	return &streamingResponseAggregator{}
}

// ProcessResponse transforms the GenerateContentResponse into an llm.Response and yields that result,
// also yielding an aggregated response if the GenerateContentResponse has zero parts or is audio data
func (s *streamingResponseAggregator) ProcessResponse(ctx context.Context, genResp *genai.GenerateContentResponse) iter.Seq2[*llm.Response, error] {
	return func(yield func(*llm.Response, error) bool) {
		if len(genResp.Candidates) == 0 {
			// shouldn't happen?
			yield(nil, fmt.Errorf("empty response"))
			return
		}
		candidate := genResp.Candidates[0]
		complete := candidate.FinishReason != ""
		resp := &llm.Response{
			Content:           candidate.Content,
			GroundingMetadata: candidate.GroundingMetadata,
			UsageMetadata:     genResp.UsageMetadata,
			Partial:           false,
			TurnComplete:      complete,
			Interrupted:       false, // no interruptions in unary
		}
		if aggrResp := s.aggregateResponse(resp); aggrResp != nil {
			if !yield(aggrResp, nil) {
				return // Consumer stopped
			}
		}
		if !yield(resp, nil) {
			return // Consumer stopped
		}
	}
}

// aggregateResponse processes a single model response,
// returning an aggregated response if the next event has zero parts or is audio data
func (s *streamingResponseAggregator) aggregateResponse(llmResponse *llm.Response) *llm.Response {
	s.response = llmResponse

	var part0 *genai.Part
	if llmResponse.Content != nil && len(llmResponse.Content.Parts) > 0 {
		part0 = llmResponse.Content.Parts[0]
		s.role = llmResponse.Content.Role
	}

	// If part is text append it
	if part0 != nil && part0.Text != "" {
		if part0.Thought {
			s.thoughtText += part0.Text
		} else {
			s.text += part0.Text
		}
		llmResponse.Partial = true
	} else
	// If there is aggregated text and there is no content or parts return aggregated response
	if (s.thoughtText != "" || s.text != "") &&
		(llmResponse.Content == nil ||
			len(llmResponse.Content.Parts) == 0 ||
			// don't yield the merged text event when receiving audio data
			(len(llmResponse.Content.Parts) > 0 && llmResponse.Content.Parts[0].InlineData == nil)) {
		return s.Close()
	}

	return nil
}

// Close generates an aggregated response at the end, if needed,
// this should be called after all the model responses are processed.
func (s *streamingResponseAggregator) Close() *llm.Response {
	if (s.text != "" || s.thoughtText != "") && s.response != nil {
		var parts []*genai.Part
		if s.thoughtText != "" {
			parts = append(parts, &genai.Part{Text: s.thoughtText, Thought: true})
		}
		if s.text != "" {
			parts = append(parts, &genai.Part{Text: s.text, Thought: false})
		}

		response := &llm.Response{
			Content:           &genai.Content{Parts: parts, Role: s.role},
			ErrorCode:         s.response.ErrorCode,
			ErrorMessage:      s.response.ErrorMessage,
			UsageMetadata:     s.response.UsageMetadata,
			GroundingMetadata: s.response.GroundingMetadata,
		}
		s.Clear()
		return response
	}
	s.Clear()
	return nil
}

func (s *streamingResponseAggregator) Clear() {
	s.response = nil
	s.text = ""
	s.thoughtText = ""
	s.role = ""
}
