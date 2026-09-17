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
	"math"
	"reflect"
	"regexp"
	"strconv"
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
		parsedPath := parseJSONPath(arg.JsonPath)
		if len(parsedPath) == 0 {
			continue
		}
		value, ok := s.getValueFromPartialArg(arg, parsedPath)
		if !ok {
			continue
		}
		s.setValueByJSONPath(parsedPath, value)
	}
	if part.FunctionCall.WillContinue != nil && *part.FunctionCall.WillContinue {
		return
	}
	s.flushTextBufferToSequence()
	s.flushFunctionCallToSequence()
}

func (s *streamingResponseAggregator) getValueFromPartialArg(partialArg *genai.PartialArg, parsedPath []any) (any, bool) {
	var value any
	var hasValue bool

	if partialArg.StringValue != "" {
		hasValue = true
		value = partialArg.StringValue

		// A string argument streams across several parts at one path, so a new
		// chunk appends to whatever the path already holds.
		if existingValue, found := valueByJSONPath(s.currentFunctionArgs, parsedPath); found {
			if str, ok := existingValue.(string); ok {
				value = str + partialArg.StringValue
			}
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

func (s *streamingResponseAggregator) setValueByJSONPath(parsedPath []any, value any) {
	// Initialize the map if it hasn't been already
	if s.currentFunctionArgs == nil {
		s.currentFunctionArgs = make(map[string]any)
	}
	if args, ok := setByJSONPath(s.currentFunctionArgs, parsedPath, value).(map[string]any); ok {
		s.currentFunctionArgs = args
	}
}

// maxJSONPathArrayIndex bounds the index a streamed JSON path may address. The
// path language permits sparse arrays, and padding one with nils is how a
// "$.items[9999999]" would otherwise allocate a slice of that length.
const maxJSONPathArrayIndex = 10000

// jsonPathTokenRE tokenizes the RFC 9535 subset partial arguments carry: a
// [n] array index, a ['name'] or ["name"] quoted member, or a dotted or bare
// member name.
var jsonPathTokenRE = regexp.MustCompile(
	`\[\s*(\d+)\s*\]` +
		`|\['((?:[^'\\]|\\.)*)'\]` +
		`|\["((?:[^"\\]|\\.)*)"\]` +
		`|\.?((?:[^.\[\]\\]|\\.)+)`)

// jsonPathEscapeRE finds the escapes a quoted member name may carry: a \uXXXX
// code point or a single escaped character.
var jsonPathEscapeRE = regexp.MustCompile(`\\u[0-9a-fA-F]{4}|\\.`)

func unescapeJSONPathString(s string) string {
	return jsonPathEscapeRE.ReplaceAllStringFunc(s, func(esc string) string {
		if esc[1] == 'u' && len(esc) == 6 {
			cp, _ := strconv.ParseUint(esc[2:], 16, 32)
			return string(rune(cp))
		}
		switch esc[1] {
		case 'n':
			return "\n"
		case 'r':
			return "\r"
		case 't':
			return "\t"
		case 'b':
			return "\b"
		case 'f':
			return "\f"
		}
		return esc[1:]
	})
}

// parseJSONPath parses a JSON path (RFC 9535) such as "$.foo.bar[0].data" into
// its components: a string for each member name, an int for each array index.
// The leading "$" is optional. A path that yields no components returns nil.
func parseJSONPath(jsonPath string) []any {
	path := jsonPath
	if rest, found := strings.CutPrefix(path, "$."); found {
		path = rest
	} else if rest, found := strings.CutPrefix(path, "$"); found {
		path = rest
	}
	var result []any
	for _, m := range jsonPathTokenRE.FindAllStringSubmatchIndex(path, -1) {
		// Index, not submatch text: a quoted member can match empty ($['']) and
		// a non-participating group reports -1 offsets rather than "".
		switch {
		case m[2] >= 0:
			idx, err := strconv.Atoi(path[m[2]:m[3]])
			if err != nil {
				// More digits than an int holds is past maxJSONPathArrayIndex
				// either way; keep the component and let the setter reject it.
				idx = math.MaxInt
			}
			result = append(result, idx)
		case m[4] >= 0:
			result = append(result, unescapeJSONPathString(path[m[4]:m[5]]))
		case m[6] >= 0:
			result = append(result, unescapeJSONPathString(path[m[6]:m[7]]))
		case m[8] >= 0:
			result = append(result, unescapeJSONPathString(path[m[8]:m[9]]))
		}
	}
	return result
}

// valueByJSONPath reads the value parsedPath addresses inside target, which is
// a map[string]any or []any, reporting false when any component is absent or
// lands on a value of the other kind.
func valueByJSONPath(target any, parsedPath []any) (any, bool) {
	current := target
	for _, part := range parsedPath {
		switch key := part.(type) {
		case string:
			m, ok := current.(map[string]any)
			if !ok {
				return nil, false
			}
			v, exists := m[key]
			if !exists {
				return nil, false
			}
			current = v
		case int:
			s, ok := current.([]any)
			if !ok || key < 0 || key >= len(s) {
				return nil, false
			}
			current = s[key]
		default:
			return nil, false
		}
	}
	return current, true
}

// newJSONPathContainer returns the container next addresses into: a slice for
// an array index, a map for a member name.
func newJSONPathContainer(next any) any {
	if _, isIndex := next.(int); isIndex {
		return []any{}
	}
	return map[string]any{}
}

// setByJSONPath writes value at parsedPath inside target, which must be a
// map[string]any or []any, creating intermediate containers shaped by the
// component that follows. A string component addresses a map; an int addresses
// a slice, padded with nils up to it. A component that lands on a value of the
// other kind, a negative index, or one past maxJSONPathArrayIndex leaves
// target untouched, so a path conflicting with what streamed earlier is
// dropped rather than misshaped. It returns the container, which may be a new
// slice: growing one reallocates it.
func setByJSONPath(target any, parsedPath []any, value any) any {
	if len(parsedPath) == 0 {
		return target
	}
	switch key := parsedPath[0].(type) {
	case string:
		m, ok := target.(map[string]any)
		if !ok {
			return target
		}
		if len(parsedPath) == 1 {
			m[key] = value
			return m
		}
		child, exists := m[key]
		if !exists {
			child = newJSONPathContainer(parsedPath[1])
			m[key] = child
		}
		m[key] = setByJSONPath(child, parsedPath[1:], value)
		return m
	case int:
		s, ok := target.([]any)
		if !ok || key < 0 || key > maxJSONPathArrayIndex {
			return target
		}
		for len(s) <= key {
			s = append(s, nil)
		}
		if len(parsedPath) == 1 {
			s[key] = value
			return s
		}
		if s[key] == nil {
			s[key] = newJSONPathContainer(parsedPath[1])
		}
		s[key] = setByJSONPath(s[key], parsedPath[1:], value)
		return s
	default:
		return target
	}
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
