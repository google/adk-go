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
	"reflect"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/llminternal/converters"
	"google.golang.org/adk/model"
)

// streamingResponseAggregator aggregates partial streaming responses.
// It aggregates content from partial responses, and generates LlmResponses for
// individual (partial) model responses, as well as for aggregated content.
type streamingResponseAggregator struct {
	response *model.LLMResponse
	role     string

	textParts            []*genai.Part
	currentTextBuffer    string
	currentTextIsThought *bool

	currentFunctionCallName string
	currentFunctionCallID   string
	currentFunctionCallArgs map[string]any
	currentThoughtSignature []byte
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

// aggregateResponse processes a single model response,
// returning an aggregated response if the next event has zero parts or is audio data
func (s *streamingResponseAggregator) aggregateResponse(llmResponse *model.LLMResponse) *model.LLMResponse {
	s.response = llmResponse

	if llmResponse.Content != nil {
		s.role = llmResponse.Content.Role
	}

	if llmResponse.Content == nil || len(llmResponse.Content.Parts) == 0 {
		if s.hasPendingTextParts() {
			return s.createAggregateResponse()
		}
		return nil
	}

	parts := llmResponse.Content.Parts
	sawNonEmptyText := false
	sawFunctionCall := false
	sawInlineData := false

	for _, part := range parts {
		if part == nil {
			continue
		}

		if part.FunctionCall != nil {
			sawFunctionCall = true
			s.flushTextBuffer()
			s.handleFunctionCall(part, llmResponse)
			continue
		}

		if part.Text != "" || len(part.ThoughtSignature) > 0 {
			if part.Text != "" {
				sawNonEmptyText = true
			}
			s.handleTextPart(part)
			llmResponse.Partial = true
			continue
		}

		if reflect.ValueOf(*part).IsZero() {
			llmResponse.Partial = true
			continue
		}

		sawInlineData = true
		s.flushTextBuffer()
	}

	if s.hasPendingTextParts() && (sawInlineData || (!sawNonEmptyText && !sawFunctionCall)) {
		return s.createAggregateResponse()
	}

	return nil
}

// Close generates an aggregated response at the end, if needed,
// this should be called after all the model responses are processed.
func (s *streamingResponseAggregator) Close() *model.LLMResponse {
	if resp := s.createAggregateResponse(); resp != nil {
		return resp
	}
	if resp := s.createPendingFunctionCallResponse(); resp != nil {
		return resp
	}
	s.clearTextBuffers()
	return nil
}

func (s *streamingResponseAggregator) createAggregateResponse() *model.LLMResponse {
	s.flushTextBuffer()
	if len(s.textParts) == 0 || s.response == nil {
		return nil
	}

	parts := make([]*genai.Part, len(s.textParts))
	copy(parts, s.textParts)

	response := &model.LLMResponse{
		Content:           &genai.Content{Parts: parts, Role: s.role},
		ErrorCode:         s.response.ErrorCode,
		ErrorMessage:      s.response.ErrorMessage,
		UsageMetadata:     s.response.UsageMetadata,
		GroundingMetadata: s.response.GroundingMetadata,
		FinishReason:      s.response.FinishReason,
	}
	s.clearTextBuffers()
	return response
}

func (s *streamingResponseAggregator) clearTextBuffers() {
	s.response = nil
	s.textParts = nil
	s.currentTextBuffer = ""
	s.currentTextIsThought = nil
	s.role = ""
}

func (s *streamingResponseAggregator) handleTextPart(part *genai.Part) {
	if len(part.ThoughtSignature) > 0 {
		s.flushTextBuffer()
		s.textParts = append(s.textParts, cloneTextPart(part))
		return
	}

	if part.Text == "" {
		return
	}

	if s.currentTextIsThought == nil || *s.currentTextIsThought != part.Thought {
		s.flushTextBuffer()
		val := part.Thought
		s.currentTextIsThought = &val
	}
	s.currentTextBuffer += part.Text
}

func (s *streamingResponseAggregator) flushTextBuffer() {
	if s.currentTextBuffer == "" {
		return
	}
	thought := false
	if s.currentTextIsThought != nil {
		thought = *s.currentTextIsThought
	}
	s.textParts = append(s.textParts, &genai.Part{Text: s.currentTextBuffer, Thought: thought})
	s.currentTextBuffer = ""
	s.currentTextIsThought = nil
}

func (s *streamingResponseAggregator) hasPendingTextParts() bool {
	return s.currentTextBuffer != "" || len(s.textParts) > 0
}

func (s *streamingResponseAggregator) handleFunctionCall(part *genai.Part, llmResponse *model.LLMResponse) {
	fc := part.FunctionCall
	if len(fc.PartialArgs) == 0 {
		if !s.hasPendingFunctionCall() {
			s.resetFunctionCallState()
		}
		return
	}

	if fc.Name != "" {
		s.currentFunctionCallName = fc.Name
	}
	if fc.ID != "" {
		s.currentFunctionCallID = fc.ID
	}
	if len(part.ThoughtSignature) > 0 && len(s.currentThoughtSignature) == 0 {
		s.currentThoughtSignature = append([]byte(nil), part.ThoughtSignature...)
	}
	if s.currentFunctionCallArgs == nil {
		s.currentFunctionCallArgs = make(map[string]any)
	}

	for _, partialArg := range fc.PartialArgs {
		value, ok := convertPartialArgValue(partialArg)
		if !ok {
			continue
		}
		pathTokens, err := parseJSONPath(partialArg.JsonPath)
		if err != nil {
			continue
		}
		if strVal, isString := value.(string); isString {
			if existing, ok := getValueAtPath(s.currentFunctionCallArgs, pathTokens); ok {
				if existingStr, ok := existing.(string); ok {
					value = existingStr + strVal
				}
			}
		}
		updated := setValueAtPath(s.currentFunctionCallArgs, pathTokens, value)
		if root, ok := updated.(map[string]any); ok {
			s.currentFunctionCallArgs = root
		}
	}

	if fcWillContinue(fc) {
		llmResponse.Partial = true
		return
	}

	if finalPart := s.buildFunctionCallPart(); finalPart != nil {
		if llmResponse.Content == nil {
			llmResponse.Content = &genai.Content{Role: s.role}
		}
		llmResponse.Content.Parts = []*genai.Part{finalPart}
		llmResponse.Partial = false
	}
	s.resetFunctionCallState()
}

func (s *streamingResponseAggregator) buildFunctionCallPart() *genai.Part {
	if s.currentFunctionCallName == "" && len(s.currentFunctionCallArgs) == 0 {
		return nil
	}
	args := cloneValue(s.currentFunctionCallArgs).(map[string]any)
	part := genai.NewPartFromFunctionCall(s.currentFunctionCallName, args)
	if part.FunctionCall != nil {
		part.FunctionCall.ID = s.currentFunctionCallID
	}
	if len(s.currentThoughtSignature) > 0 {
		part.ThoughtSignature = append([]byte(nil), s.currentThoughtSignature...)
	}
	return part
}

func (s *streamingResponseAggregator) resetFunctionCallState() {
	s.currentFunctionCallArgs = nil
	s.currentFunctionCallID = ""
	s.currentFunctionCallName = ""
	s.currentThoughtSignature = nil
}

func (s *streamingResponseAggregator) hasPendingFunctionCall() bool {
	return len(s.currentFunctionCallArgs) > 0 || s.currentFunctionCallName != ""
}

func (s *streamingResponseAggregator) createPendingFunctionCallResponse() *model.LLMResponse {
	if !s.hasPendingFunctionCall() || s.response == nil {
		return nil
	}
	part := s.buildFunctionCallPart()
	if part == nil {
		return nil
	}
	response := &model.LLMResponse{
		Content:           &genai.Content{Parts: []*genai.Part{part}, Role: s.role},
		ErrorCode:         s.response.ErrorCode,
		ErrorMessage:      s.response.ErrorMessage,
		UsageMetadata:     s.response.UsageMetadata,
		GroundingMetadata: s.response.GroundingMetadata,
		FinishReason:      s.response.FinishReason,
	}
	s.resetFunctionCallState()
	s.clearTextBuffers()
	return response
}

func cloneTextPart(part *genai.Part) *genai.Part {
	if part == nil {
		return nil
	}
	out := &genai.Part{
		Text:    part.Text,
		Thought: part.Thought,
	}
	if len(part.ThoughtSignature) > 0 {
		out.ThoughtSignature = append([]byte(nil), part.ThoughtSignature...)
	}
	return out
}

type jsonPathToken struct {
	key     string
	index   int
	isIndex bool
}

func parseJSONPath(path string) ([]jsonPathToken, error) {
	if path == "" {
		return nil, fmt.Errorf("json path cannot be empty")
	}
	if !strings.HasPrefix(path, "$") {
		return nil, fmt.Errorf("json path must start with '$'")
	}
	var tokens []jsonPathToken
	i := 1
	for i < len(path) {
		switch path[i] {
		case '.':
			i++
			start := i
			for i < len(path) && path[i] != '.' && path[i] != '[' {
				i++
			}
			if start == i {
				return nil, fmt.Errorf("invalid json path %q", path)
			}
			tokens = append(tokens, jsonPathToken{key: path[start:i]})
		case '[':
			i++
			if i >= len(path) {
				return nil, fmt.Errorf("unterminated '[' in json path %q", path)
			}
			if path[i] == '\'' || path[i] == '"' {
				quote := path[i]
				i++
				start := i
				for i < len(path) && path[i] != quote {
					i++
				}
				if i >= len(path) {
					return nil, fmt.Errorf("unterminated quoted key in json path %q", path)
				}
				key := path[start:i]
				i++
				if i >= len(path) || path[i] != ']' {
					return nil, fmt.Errorf("missing closing bracket in json path %q", path)
				}
				tokens = append(tokens, jsonPathToken{key: key})
				i++
				continue
			}
			start := i
			for i < len(path) && path[i] != ']' {
				i++
			}
			if i >= len(path) {
				return nil, fmt.Errorf("unterminated array index in json path %q", path)
			}
			idx, err := strconv.Atoi(path[start:i])
			if err != nil {
				return nil, fmt.Errorf("invalid array index in json path %q: %w", path, err)
			}
			tokens = append(tokens, jsonPathToken{index: idx, isIndex: true})
			i++
		default:
			return nil, fmt.Errorf("unexpected character %q in json path %q", path[i], path)
		}
	}
	return tokens, nil
}

func convertPartialArgValue(arg *genai.PartialArg) (any, bool) {
	if arg == nil {
		return nil, false
	}
	switch {
	case arg.StringValue != "":
		return arg.StringValue, true
	case arg.NumberValue != nil:
		return *arg.NumberValue, true
	case arg.BoolValue != nil:
		return *arg.BoolValue, true
	case arg.NULLValue != "":
		return nil, true
	default:
		return nil, false
	}
}

func getValueAtPath(current any, tokens []jsonPathToken) (any, bool) {
	if len(tokens) == 0 {
		return current, true
	}
	head := tokens[0]
	rest := tokens[1:]
	if head.isIndex {
		slice, ok := current.([]any)
		if !ok || slice == nil {
			return nil, false
		}
		if head.index < 0 || head.index >= len(slice) {
			return nil, false
		}
		return getValueAtPath(slice[head.index], rest)
	}
	obj, ok := current.(map[string]any)
	if !ok || obj == nil {
		return nil, false
	}
	val, ok := obj[head.key]
	if !ok {
		return nil, false
	}
	return getValueAtPath(val, rest)
}

func setValueAtPath(current any, tokens []jsonPathToken, value any) any {
	if len(tokens) == 0 {
		return value
	}
	head := tokens[0]
	rest := tokens[1:]
	if head.isIndex {
		var slice []any
		if existing, ok := current.([]any); ok && existing != nil {
			slice = existing
		}
		if head.index < 0 {
			head.index = 0
		}
		if len(slice) <= head.index {
			newSlice := make([]any, head.index+1)
			copy(newSlice, slice)
			slice = newSlice
		}
		slice[head.index] = setValueAtPath(slice[head.index], rest, value)
		return slice
	}
	var obj map[string]any
	if existing, ok := current.(map[string]any); ok && existing != nil {
		obj = existing
	} else {
		obj = make(map[string]any)
	}
	obj[head.key] = setValueAtPath(obj[head.key], rest, value)
	return obj
}

func cloneValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(val))
		for k, v := range val {
			out[k] = cloneValue(v)
		}
		return out
	case []any:
		out := make([]any, len(val))
		for i, v := range val {
			out[i] = cloneValue(v)
		}
		return out
	default:
		return v
	}
}

func fcWillContinue(fc *genai.FunctionCall) bool {
	if fc == nil || fc.WillContinue == nil {
		return false
	}
	return *fc.WillContinue
}
