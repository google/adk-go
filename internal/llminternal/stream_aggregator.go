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
	"iter"
	"maps"
	"reflect"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/converters"
	"google.golang.org/adk/v2/model"
)

// streamingResponseAggregator aggregates partial streaming responses.
// It aggregates content from partial responses, and generates LlmResponses for
// individual (partial) model responses, as well as for aggregated content.
type streamingResponseAggregator struct {
	usageMetadata     *genai.GenerateContentResponseUsageMetadata
	groundingMetadata *genai.GroundingMetadata
	citationMetadata  *genai.CitationMetadata
	response          *model.LLMResponse

	currentThoughtSignature []byte

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
		resp := converters.Genai2LLMResponse(genResp)
		if len(genResp.Candidates) > 0 {
			candidate := genResp.Candidates[0]
			resp.TurnComplete = candidate.FinishReason != ""
		}
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
	if llmResponse.UsageMetadata != nil {
		s.usageMetadata = llmResponse.UsageMetadata
	}
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
		if len(part.ThoughtSignature) > 0 {
			s.currentThoughtSignature = part.ThoughtSignature
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
			if part.ThoughtSignature == nil && s.currentThoughtSignature != nil {
				part.ThoughtSignature = s.currentThoughtSignature
			}
			s.currentThoughtSignature = nil
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
		segments, ok := parseJSONPath(arg.JsonPath)
		if !ok {
			continue
		}
		value, ok := s.getValueFromPartialArg(arg, segments)
		if !ok {
			continue
		}
		s.setValueByPath(segments, value)
	}
	if part.FunctionCall.WillContinue != nil && *part.FunctionCall.WillContinue {
		return
	}
	s.flushTextBufferToSequence()
	s.flushFunctionCallToSequence()
}

func (s *streamingResponseAggregator) getValueFromPartialArg(partialArg *genai.PartialArg, segments []pathSegment) (any, bool) {
	var value any
	var hasValue bool

	if partialArg.StringValue != "" {
		stringChunk := partialArg.StringValue
		hasValue = true

		// A string is streamed in chunks that all carry the same path, so
		// append this one to whatever of it has already been assembled.
		existingValue, _ := valueByPath(s.currentFunctionArgs, segments)

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

// pathSegment is one step of a JSON Path: an object member name, or an array
// index when isIndex is set.
type pathSegment struct {
	name    string
	index   int
	isIndex bool
}

// maxPathIndex is the largest array index a path may address.
// parseBracketSegment checks it after every digit, and that check is what keeps
// the index from overflowing int: without it "$.a[9999999999999999999]" wraps
// to a negative index and setInto, which trusts the index it is given, panics.
// It also bounds the slice a single index can grow; no function call argument
// list comes close to it.
const maxPathIndex = 1 << 16

// parseJSONPath splits the RFC 9535 JSON Path that addresses a streamed
// argument (https://datatracker.ietf.org/doc/html/rfc9535) into its segments.
// Only the selectors a normalized path is built from are accepted — "$.name",
// "$['name']" and "$[0]" — because that is all the model produces; anything
// else, including wildcards, slices and \u escapes in a quoted name, reports
// false so that the chunk is dropped rather than written to an invented key.
func parseJSONPath(jsonPath string) ([]pathSegment, bool) {
	rest := strings.TrimPrefix(jsonPath, "$")
	var segments []pathSegment
	for rest != "" {
		if rest[0] == '[' {
			segment, remainder, ok := parseBracketSegment(rest)
			if !ok {
				return nil, false
			}
			segments, rest = append(segments, segment), remainder
			continue
		}
		if rest[0] == '.' {
			rest = rest[1:]
		} else if len(segments) > 0 {
			return nil, false
		}
		name, remainder := splitMemberName(rest)
		if name == "" {
			return nil, false
		}
		segments, rest = append(segments, pathSegment{name: name}), remainder
	}
	if len(segments) == 0 {
		return nil, false
	}
	return segments, true
}

// splitMemberName takes the shorthand member name at the front of a path.
func splitMemberName(path string) (name, rest string) {
	if i := strings.IndexAny(path, ".["); i >= 0 {
		return path[:i], path[i:]
	}
	return path, ""
}

// parseBracketSegment takes the bracketed selector at the front of a path,
// which is either an array index or a quoted member name.
func parseBracketSegment(path string) (pathSegment, string, bool) {
	body := path[1:]
	if body != "" && (body[0] == '\'' || body[0] == '"') {
		name, rest, ok := parseQuotedName(body)
		if !ok || rest == "" || rest[0] != ']' {
			return pathSegment{}, "", false
		}
		return pathSegment{name: name}, rest[1:], true
	}
	end := strings.IndexByte(body, ']')
	if end <= 0 {
		return pathSegment{}, "", false
	}
	index := 0
	for i := range end {
		digit := body[i]
		if digit < '0' || digit > '9' {
			return pathSegment{}, "", false
		}
		index = index*10 + int(digit-'0')
		if index > maxPathIndex {
			return pathSegment{}, "", false
		}
	}
	return pathSegment{index: index, isIndex: true}, body[end+1:], true
}

// parseQuotedName reads a quoted member name, undoing the single-character
// escapes RFC 9535 defines for one.
func parseQuotedName(path string) (name, rest string, ok bool) {
	quote := path[0]
	var parsed strings.Builder
	for i := 1; i < len(path); i++ {
		switch char := path[i]; char {
		case quote:
			return parsed.String(), path[i+1:], true
		case '\\':
			i++
			if i == len(path) {
				return "", "", false
			}
			escaped, ok := unescape(path[i])
			if !ok {
				return "", "", false
			}
			parsed.WriteByte(escaped)
		default:
			parsed.WriteByte(char)
		}
	}
	return "", "", false
}

func unescape(char byte) (byte, bool) {
	switch char {
	case '\'', '"', '\\', '/':
		return char, true
	case 'b':
		return '\b', true
	case 'f':
		return '\f', true
	case 'n':
		return '\n', true
	case 'r':
		return '\r', true
	case 't':
		return '\t', true
	}
	return 0, false
}

// setValueByPath writes value at the location the segments address, creating
// the maps and slices on the way to it.
func (s *streamingResponseAggregator) setValueByPath(segments []pathSegment, value any) {
	// Initialize the map if it hasn't been already
	if s.currentFunctionArgs == nil {
		s.currentFunctionArgs = make(map[string]any)
	}
	// Function call arguments are a JSON object, so a path that starts at an
	// array index addresses nothing and setInto would return a slice.
	if args, ok := setInto(s.currentFunctionArgs, segments, value).(map[string]any); ok {
		s.currentFunctionArgs = args
	}
}

// setInto returns container with value written at the location segments
// address within it, which is a new container when the one that is there
// cannot hold that segment. A slice grows to fit an index, so an argument
// whose elements arrive out of order keeps the positions the model chose.
func setInto(container any, segments []pathSegment, value any) any {
	segment := segments[0]
	if segment.isIndex {
		elements, _ := container.([]any)
		for len(elements) <= segment.index {
			elements = append(elements, nil)
		}
		elements[segment.index] = descend(elements[segment.index], segments, value)
		return elements
	}
	members, ok := container.(map[string]any)
	if !ok {
		members = make(map[string]any)
	}
	members[segment.name] = descend(members[segment.name], segments, value)
	return members
}

func descend(existing any, segments []pathSegment, value any) any {
	if len(segments) == 1 {
		return value
	}
	return setInto(existing, segments[1:], value)
}

// valueByPath reads the value the segments address, reporting whether the
// whole path exists.
func valueByPath(args map[string]any, segments []pathSegment) (any, bool) {
	var current any = args
	for _, segment := range segments {
		if segment.isIndex {
			elements, ok := current.([]any)
			if !ok || segment.index >= len(elements) {
				return nil, false
			}
			current = elements[segment.index]
			continue
		}
		members, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		if current, ok = members[segment.name]; !ok {
			return nil, false
		}
	}
	return current, true
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
			errorCode = s.response.ErrorCode
			errorMessage = s.response.ErrorMessage
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
			FinishReason:      s.finishReason,
		}
	}
	return nil
}
