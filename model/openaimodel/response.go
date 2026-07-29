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
	"encoding/json"
	"fmt"
	"math"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// convertResponse takes an OpenAI API response and transforms it into our
// generic genai.GenerateContentResponse format.
func convertResponse(resp *responses.Response) (*genai.GenerateContentResponse, error) {
	if resp == nil {
		return nil, ErrEmptyResponse
	}
	candidate, err := buildCandidate(resp)
	if err != nil {
		return nil, err
	}
	return &genai.GenerateContentResponse{
		Candidates:     []*genai.Candidate{candidate},
		ModelVersion:   string(resp.Model),
		ResponseID:     resp.ID,
		UsageMetadata:  convertUsage(resp.Usage),
		PromptFeedback: promptFeedback(resp),
	}, nil
}

func buildCandidate(resp *responses.Response) (*genai.Candidate, error) {
	parts, err := convertOutputItems(resp.Output)
	if err != nil {
		return nil, err
	}
	return &genai.Candidate{
		Content: &genai.Content{
			Role:  string(genai.RoleModel),
			Parts: parts,
		},
		FinishReason:   finishReason(resp),
		LogprobsResult: convertLogprobs(resp.Output),
	}, nil
}

// convertOutputItems processes a slice of OpenAI ResponseOutputItemUnion and
// converts them into a slice of our generic genai.Part. We handle different
// types of output items, such as messages (text, refusal), function calls,
// and reasoning (thoughts and summaries), extracting the relevant information
// for each.
func convertOutputItems(items []responses.ResponseOutputItemUnion) ([]*genai.Part, error) {
	if len(items) == 0 {
		return nil, ErrNoOutputItems
	}
	var parts []*genai.Part
	for _, item := range items {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					if content.Text != "" {
						parts = append(parts, &genai.Part{Text: content.Text})
					}
				case "refusal":
					parts = append(parts, &genai.Part{Text: content.Refusal})
				default:
					return nil, fmt.Errorf("%w: %q", ErrUnsupportedMessageContentType, content.Type)
				}
			}
		case "function_call":
			part, err := convertFunctionCall(item)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case "reasoning":
			for _, chunk := range item.Content {
				if chunk.Text != "" {
					parts = append(parts, &genai.Part{Text: chunk.Text, Thought: true})
				}
				// We also check for summary content within reasoning items.
			}
			for _, summary := range item.Summary {
				if summary.Text != "" {
					parts = append(parts, &genai.Part{Text: summary.Text, Thought: true})
				}
			}
		default:
			return nil, fmt.Errorf("%w: %q", ErrUnsupportedOutputItemType, item.Type)
		}
	}
	if len(parts) == 0 {
		return nil, ErrNoTextOrToolContent
	}
	return parts, nil
}

func convertFunctionCall(item responses.ResponseOutputItemUnion) (*genai.Part, error) {
	args := map[string]any{}
	if item.Arguments != "" {
		if err := json.Unmarshal([]byte(item.Arguments), &args); err != nil {
			return nil, fmt.Errorf("openai: parse function call args: %w", err)
		}
	}
	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			Name: item.Name,
			ID:   item.CallID,
			Args: args,
		},
	}, nil
}

func finishReason(resp *responses.Response) genai.FinishReason {
	if resp == nil {
		return genai.FinishReasonUnspecified
	}
	switch resp.IncompleteDetails.Reason {
	case "max_output_tokens":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "":
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonOther
	}
}

func safeInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(v)
}

func convertUsage(usage responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        safeInt32(usage.InputTokens),
		CandidatesTokenCount:    safeInt32(usage.OutputTokens),
		TotalTokenCount:         safeInt32(usage.TotalTokens),
		CachedContentTokenCount: safeInt32(usage.InputTokensDetails.CachedTokens),
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.InputTokens)},
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.OutputTokens)},
		},
		ThoughtsTokenCount: safeInt32(usage.OutputTokensDetails.ReasoningTokens),
	}
}

func promptFeedback(resp *responses.Response) *genai.GenerateContentResponsePromptFeedback {
	if resp == nil || resp.IncompleteDetails.Reason != "content_filter" {
		return nil
	}
	return &genai.GenerateContentResponsePromptFeedback{
		BlockReason:        genai.BlockedReasonSafety,
		BlockReasonMessage: resp.IncompleteDetails.Reason,
	}
}

func convertLogprobs(items []responses.ResponseOutputItemUnion) *genai.LogprobsResult {
	if len(items) == 0 {
		return nil
	}
	var res *genai.LogprobsResult
	for _, item := range items {
		if item.Type == "message" {
			for _, content := range item.Content {
				if content.Type == "output_text" && len(content.Logprobs) > 0 {
					if res == nil {
						res = &genai.LogprobsResult{}
					}
					for _, lp := range content.Logprobs {
						res.ChosenCandidates = append(res.ChosenCandidates, &genai.LogprobsResultCandidate{
							Token:          lp.Token,
							LogProbability: float32(lp.Logprob),
						})
						var topCands []*genai.LogprobsResultCandidate
						for _, tlp := range lp.TopLogprobs {
							topCands = append(topCands, &genai.LogprobsResultCandidate{
								Token:          tlp.Token,
								LogProbability: float32(tlp.Logprob),
							})
						}
						res.TopCandidates = append(res.TopCandidates, &genai.LogprobsResultTopCandidates{
							Candidates: topCands,
						})
					}
				}
			}
		}
	}
	return res
}
