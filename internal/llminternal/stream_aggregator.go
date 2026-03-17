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
	"maps"
	"reflect"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/llminternal/converters"
	"google.golang.org/adk/model"
)

// streamingResponseAggregator aggregates partial streaming responses.
// It aggregates content from partial responses, and generates LlmResponses for
// individual (partial) model responses, as well as for aggregated content.
type streamingResponseAggregator struct {
	usageMetadata     *genai.GenerateContentResponseUsageMetadata
	groundingMetadata *genai.GroundingMetadata
	citationMetadata  *genai.CitationMetadata
	response          *model.LLMResponse

	sequence             []*genai.Part
	currentTextBuffer    string
	currentTextIsThought bool
	finishReason         genai.FinishReason

	currentFunctionName             string
	currentFunctionID               string
	currentFunctionArgs             map[string]any
	currentFunctionThoughtSignature []byte
}

// NewStreamingResponseAggregator creates a new, initialized streamingResponseAggregator.
func NewStreamingResponseAggregator() *streamingResponseAggregator {
	return &streamingResponseAggregator{}
}

// ProcessResponse transforms the GenerateContentResponse into an model.Response and yields that result,
// also yielding an aggregated response if the GenerateContentResponse has zero parts or is audio data
func (s *streamingResponseAggregator) ProcessResponse(ctx context.Context, genResp *genai.GenerateContentResponse) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(genResp.Candidates) == 0 {
			// shouldn't happen?
			yield(nil, fmt.Errorf("empty response"))
			return
		}
		candidate := genResp.Candidates[0]
		resp := converters.Genai2LLMResponse(genResp)
		resp.TurnComplete = candidate.FinishReason != ""
		// Aggregate the response and check if an intermediate event to yield was created
		if aggrResp := s.aggregateResponse(resp); aggrResp != nil {
			if !yield(aggrResp, nil) {
				return // Consumer stopped
			}
		}
		// Yield the processed response
		if !yield(resp, nil) {
			return // Consumer stopped
		}
	}
}

func (s *streamingResponseAggregator) aggregateResponse(llmResponse *model.LLMResponse) *model.LLMResponse {
	s.response = llmResponse
	s.usageMetadata = llmResponse.UsageMetadata
	if llmResponse.GroundingMetadata != nil {
		s.groundingMetadata = llmResponse.GroundingMetadata
	}
	if llmResponse.CitationMetadata != nil {
		s.citationMetadata = llmResponse.CitationMetadata
	}

	if llmResponse.FinishReason != "" {
		s.finishReason = llmResponse.FinishReason
	}
	llmResponse.Partial = true

	if llmResponse.Content == nil {
		return nil
	}

	for _, part := range llmResponse.Content.Parts {
		// gemini 3 in streaming returns a last response with an empty part. We will filter it out.
		if reflect.ValueOf(*part).IsZero() {
			continue
		}
		if part.Text != "" {
			if s.currentTextBuffer != "" && part.Thought != s.currentTextIsThought {
				s.flushTextBufferToSequence()
			}
			if s.currentTextBuffer == "" {
				s.currentTextIsThought = part.Thought
			}
			s.currentTextBuffer += part.Text
		} else if part.FunctionCall != nil {
			// Process function call (handles both streaming Args and non-streaming Args
			s.processFunctionCallPart(part)
		} else {
			// Other non-text parts (bytes, etc.)
			// Flush any buffered text first, then add the non-text part
			s.flushTextBufferToSequence()
			s.sequence = append(s.sequence, part)
		}
	}
	return nil
}

func (s *streamingResponseAggregator) processFunctionCallPart(part *genai.Part) {
	if part.FunctionCall == nil {
		return
	}
	if part.FunctionCall.PartialArgs != nil || (part.FunctionCall.WillContinue != nil && *part.FunctionCall.WillContinue) {
		if len(part.ThoughtSignature) > 0 && s.currentFunctionThoughtSignature == nil {
			s.currentFunctionThoughtSignature = part.ThoughtSignature
		}
		s.processStreamingFunctionCallPart(part)
	} else {
		if part.FunctionCall.Name != "" {
			s.flushTextBufferToSequence()
			s.sequence = append(s.sequence, part)
		}
	}
}

// Process a streaming function call with partialArgs.
func (s *streamingResponseAggregator) processStreamingFunctionCallPart(part *genai.Part) {
	if part.FunctionCall.Name != "" {
		s.currentFunctionName = part.FunctionCall.Name
	}
	if part.FunctionCall.ID != "" {
		s.currentFunctionID = part.FunctionCall.ID
	}
	for _, arg := range part.FunctionCall.PartialArgs {
		jsonPath := arg.JsonPath
		if jsonPath == "" {
			continue
		}
		value, ok := s.getValueFromPartialArg(arg, jsonPath)
		if !ok {
			continue
		}
		s.setValueByJSONPath(jsonPath, value)
	}
	if part.FunctionCall.WillContinue != nil && *part.FunctionCall.WillContinue {
		return
	}
	s.flushTextBufferToSequence()
	s.flushFunctionCallToSequence()
}

func (s *streamingResponseAggregator) getValueFromPartialArg(partialArg *genai.PartialArg, jsonPath string) (any, bool) {
	var value any
	var hasValue bool

	if partialArg.StringValue != "" {
		stringChunk := partialArg.StringValue
		hasValue = true

		// Clean up the JSONPath prefix
		pathWithoutPrefix := jsonPath
		if strings.HasPrefix(jsonPath, "$.") {
			pathWithoutPrefix = jsonPath[2:]
		}
		pathParts := strings.Split(pathWithoutPrefix, ".")

		// Try to get existing value by traversing the map
		var existingValue any = s.currentFunctionArgs
		for _, part := range pathParts {
			if m, ok := existingValue.(map[string]any); ok {
				if val, exists := m[part]; exists {
					existingValue = val
					continue
				}
			}
			// If we can't find the path or it's not a map, reset existingValue
			existingValue = nil
			break
		}

		// Append to existing string or set new value
		if str, ok := existingValue.(string); ok {
			value = str + stringChunk
		} else {
			value = stringChunk
		}

	} else if partialArg.NumberValue != nil {
		value = *partialArg.NumberValue
		hasValue = true
	} else if partialArg.BoolValue != nil {
		value = *partialArg.BoolValue
		hasValue = true
	} else if partialArg.NULLValue != "" {
		value = nil
		hasValue = true
	}

	return value, hasValue
}

func (s *streamingResponseAggregator) setValueByJSONPath(jsonPath string, value any) {
	// Initialize the map if it hasn't been already
	if s.currentFunctionArgs == nil {
		s.currentFunctionArgs = make(map[string]any)
	}

	// Remove leading "$." from jsonPath
	path := jsonPath
	if strings.HasPrefix(jsonPath, "$.") {
		path = jsonPath[2:]
	}

	// Split path into components
	pathParts := strings.Split(path, ".")
	if len(pathParts) == 0 || (len(pathParts) == 1 && pathParts[0] == "") {
		return // Handle empty path case
	}

	// Navigate to the correct location
	current := s.currentFunctionArgs

	// Iterate through all parts except the last one
	for _, part := range pathParts[:len(pathParts)-1] {
		next, exists := current[part]

		// If the path doesn't exist, or the existing value isn't a map,
		// create a new map at this node.
		nextMap, ok := next.(map[string]any)
		if !exists || !ok {
			nextMap = make(map[string]any)
			current[part] = nextMap
		}

		current = nextMap
	}

	// Set the final value at the last key
	lastKey := pathParts[len(pathParts)-1]
	current[lastKey] = value
}

func (s *streamingResponseAggregator) flushTextBufferToSequence() {
	// Check if buffer has content (strings.Builder.Len() is efficient)
	if s.currentTextBuffer != "" {
		s.sequence = append(s.sequence, &genai.Part{
			Text:    s.currentTextBuffer,
			Thought: s.currentTextIsThought,
		})
		// Reset the buffer and the state
		s.currentTextBuffer = ""
		s.currentTextIsThought = false
	}
}

func (s *streamingResponseAggregator) flushFunctionCallToSequence() {
	if s.currentFunctionName != "" {
		fc := &genai.FunctionCall{
			Name: s.currentFunctionName,
			Args: maps.Clone(s.currentFunctionArgs),
			ID:   s.currentFunctionID,
		}

		fcPart := &genai.Part{
			FunctionCall: fc,
		}
		if s.currentFunctionThoughtSignature != nil {
			fcPart.ThoughtSignature = s.currentFunctionThoughtSignature
		}

		s.sequence = append(s.sequence, fcPart)

		s.currentFunctionName = ""
		s.currentFunctionID = ""
		s.currentFunctionThoughtSignature = nil
		s.currentFunctionArgs = make(map[string]any)
	}
}

// Close generates an aggregated response at the end, if needed,
// this should be called after all the model responses are processed.
func (s *streamingResponseAggregator) Close() *model.LLMResponse {
	if s.response != nil {
		s.flushTextBufferToSequence()
		s.flushFunctionCallToSequence()
		errorCode := ""
		errorMessage := ""
		if s.finishReason != genai.FinishReasonStop {
			if s.response.ErrorCode != "" {
				errorCode = s.response.ErrorCode
			} else {
				errorCode = "error"
			}
			if s.response.ErrorMessage != "" {
				errorMessage = s.response.ErrorMessage
			} else {
				errorMessage = "error"
			}
		}

		return &model.LLMResponse{
			Content: &genai.Content{
				Parts: s.sequence,
				Role:  genai.RoleModel,
			},
			UsageMetadata:     s.usageMetadata,
			GroundingMetadata: s.groundingMetadata,
			CitationMetadata:  s.citationMetadata,
			ErrorCode:         errorCode,
			ErrorMessage:      errorMessage,
		}
	}
	return nil
}
