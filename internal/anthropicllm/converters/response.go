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
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// MessageToLLMResponse converts an Anthropic Message to a model.LLMResponse.
func MessageToLLMResponse(msg *anthropic.Message) (*model.LLMResponse, error) {
	if msg == nil {
		return nil, fmt.Errorf("nil message received")
	}

	content := &genai.Content{
		Role:  "model",
		Parts: make([]*genai.Part, 0, len(msg.Content)),
	}

	var allCitations []*genai.Citation
	for _, block := range msg.Content {
		part, err := ContentBlockToGenaiPart(block)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content block: %w", err)
		}
		if part != nil {
			content.Parts = append(content.Parts, part)
		}
		// Collect citations from text blocks
		if textBlock, ok := block.AsAny().(anthropic.TextBlock); ok {
			if citations := textCitationsToSlice(textBlock.Citations); len(citations) > 0 {
				allCitations = append(allCitations, citations...)
			}
		}
	}

	resp := &model.LLMResponse{
		Content:       content,
		UsageMetadata: UsageToMetadata(msg.Usage),
		FinishReason:  StopReasonToFinishReason(msg.StopReason),
	}

	if len(allCitations) > 0 {
		resp.CitationMetadata = &genai.CitationMetadata{Citations: allCitations}
	}

	return resp, nil
}

// ContentBlockToGenaiPart converts an Anthropic ContentBlockUnion to a genai.Part.
func ContentBlockToGenaiPart(block anthropic.ContentBlockUnion) (*genai.Part, error) {
	switch variant := block.AsAny().(type) {
	case anthropic.TextBlock:
		return &genai.Part{Text: variant.Text}, nil

	case anthropic.ThinkingBlock:
		// Map thinking blocks to genai.Part with Thought=true
		signature, _ := base64.StdEncoding.DecodeString(variant.Signature)
		return &genai.Part{
			Text:             variant.Thinking,
			Thought:          true,
			ThoughtSignature: signature,
		}, nil

	case anthropic.RedactedThinkingBlock:
		// Redacted thinking - we can't see the content but preserve the marker
		return &genai.Part{
			Text:    "[thinking redacted]",
			Thought: true,
		}, nil

	case anthropic.ToolUseBlock:
		// Convert to FunctionCall
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is json.RawMessage, unmarshal it
			if err := json.Unmarshal(variant.Input, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: variant.Name,
				Args: args,
			},
		}, nil

	case anthropic.ServerToolUseBlock:
		// Server-side tool use (web search, etc.)
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is an any type, convert through JSON
			inputBytes, err := json.Marshal(variant.Input)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal server tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
			if err := json.Unmarshal(inputBytes, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal server tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: string(variant.Name),
				Args: args,
			},
		}, nil

	case anthropic.WebSearchToolResultBlock:
		// Web search results from Anthropic's built-in web search tool
		return webSearchResultToFunctionResponse(variant), nil

	default:
		// Unknown block type - skip
		return nil, nil
	}
}

// webSearchResultToFunctionResponse converts a WebSearchToolResultBlock to a FunctionResponse Part.
func webSearchResultToFunctionResponse(block anthropic.WebSearchToolResultBlock) *genai.Part {
	response := make(map[string]any)

	// Check if it's an error or results
	if results := block.Content.AsWebSearchResultBlockArray(); len(results) > 0 {
		searchResults := make([]map[string]any, 0, len(results))
		for _, result := range results {
			searchResults = append(searchResults, map[string]any{
				"title":   result.Title,
				"url":     result.URL,
				"pageAge": result.PageAge,
			})
		}
		response["results"] = searchResults
	} else if errBlock := block.Content.AsResponseWebSearchToolResultError(); errBlock.ErrorCode != "" {
		response["error"] = string(errBlock.ErrorCode)
	}

	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       block.ToolUseID,
			Name:     "web_search",
			Response: response,
		},
	}
}

// textCitationsToSlice converts Anthropic text citations to a slice of genai.Citation.
func textCitationsToSlice(citations []anthropic.TextCitationUnion) []*genai.Citation {
	if len(citations) == 0 {
		return nil
	}

	result := make([]*genai.Citation, 0, len(citations))
	for _, c := range citations {
		citation := &genai.Citation{
			Title: c.DocumentTitle,
		}

		// Map based on citation type
		switch c.Type {
		case "char_location":
			citation.StartIndex = int32(c.StartCharIndex)
			citation.EndIndex = int32(c.EndCharIndex)
		case "web_search_result_location":
			citation.Title = c.Title
			citation.URI = c.URL
		case "search_result_location":
			citation.Title = c.Title
		}

		result = append(result, citation)
	}

	return result
}

// UsageToMetadata converts Anthropic Usage to genai UsageMetadata.
func UsageToMetadata(usage anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.InputTokens + usage.OutputTokens),
	}
}

// StopReasonToFinishReason maps Anthropic StopReason to genai FinishReason.
func StopReasonToFinishReason(sr anthropic.StopReason) genai.FinishReason {
	switch sr {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonUnspecified
	}
}

// StreamDeltaToPartialResponse converts a streaming content block delta to a partial LLMResponse.
// Used for streaming text updates.
func StreamDeltaToPartialResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: text},
			},
		},
		Partial: true,
	}
}

// StreamThinkingDeltaToPartialResponse converts a streaming thinking delta to a partial LLMResponse.
func StreamThinkingDeltaToPartialResponse(thinking string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:    thinking,
					Thought: true,
				},
			},
		},
		Partial: true,
	}
}
