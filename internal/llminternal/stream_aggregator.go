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

	currentFunctionCalls         map[string]*functionCallState
	activeFunctionCallOrder      []string
	activeFunctionCallKeysByName map[string][]string
	lastFunctionCallKey          string
	unnamedSequence              int
	unnamedCursor                int
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
	if fc == nil {
		return
	}

	if !isStreamingFunctionCall(fc) {
		if !s.hasPendingFunctionCall() || fc.Name != "" || fc.ID != "" || len(fc.PartialArgs) > 0 {
			return
		}
		// Empty functionCall chunk can mark the end of streamed args.
	}

	state := s.ensureFunctionCallState(fc)

	if fc.Name != "" {
		state.name = fc.Name
	}
	if fc.ID != "" {
		state.id = fc.ID
	}
	if len(part.ThoughtSignature) > 0 && len(state.thoughtSignature) == 0 {
		state.thoughtSignature = append([]byte(nil), part.ThoughtSignature...)
	}
	if state.args == nil {
		state.args = make(map[string]any)
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
			if existing, ok := getValueAtPath(state.args, pathTokens); ok {
				if existingStr, ok := existing.(string); ok {
					value = existingStr + strVal
				}
			}
		}
		updated := setValueAtPath(state.args, pathTokens, value)
		if root, ok := updated.(map[string]any); ok {
			state.args = root
		}
	}

	if fcWillContinue(fc) {
		llmResponse.Partial = true
		return
	}

	if !state.hasData() {
		return
	}

	if finalPart := s.buildFunctionCallPart(state); finalPart != nil {
		if llmResponse.Content == nil {
			llmResponse.Content = &genai.Content{Role: s.role}
		}
		llmResponse.Content.Parts = []*genai.Part{finalPart}
		llmResponse.Partial = false
	}
	s.clearFunctionCallState(state.key)
}

func (s *streamingResponseAggregator) buildFunctionCallPart(state *functionCallState) *genai.Part {
	if state == nil || !state.hasData() {
		return nil
	}
	args := cloneValue(state.args).(map[string]any)
	part := genai.NewPartFromFunctionCall(state.name, args)
	if part.FunctionCall != nil {
		part.FunctionCall.ID = state.id
	}
	if len(state.thoughtSignature) > 0 {
		part.ThoughtSignature = append([]byte(nil), state.thoughtSignature...)
	}
	return part
}

func (s *streamingResponseAggregator) clearFunctionCallState(key string) {
	if key == "" || s.currentFunctionCalls == nil {
		return
	}
	delete(s.currentFunctionCalls, key)
	s.removeActiveFunctionCallKey(key)
}

func (s *streamingResponseAggregator) hasPendingFunctionCall() bool {
	return s.currentFunctionCalls != nil && len(s.currentFunctionCalls) > 0
}

func (s *streamingResponseAggregator) createPendingFunctionCallResponse() *model.LLMResponse {
	if !s.hasPendingFunctionCall() || s.response == nil {
		return nil
	}
	parts := s.buildPendingFunctionCallParts()
	if len(parts) == 0 {
		return nil
	}
	response := &model.LLMResponse{
		Content:           &genai.Content{Parts: parts, Role: s.role},
		ErrorCode:         s.response.ErrorCode,
		ErrorMessage:      s.response.ErrorMessage,
		UsageMetadata:     s.response.UsageMetadata,
		GroundingMetadata: s.response.GroundingMetadata,
		FinishReason:      s.response.FinishReason,
	}
	s.clearAllFunctionCallState()
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

type functionCallState struct {
	key              string
	name             string
	id               string
	args             map[string]any
	thoughtSignature []byte
}

func (s *functionCallState) hasData() bool {
	return s.name != "" || len(s.args) > 0
}

func (s *streamingResponseAggregator) ensureFunctionCallState(fc *genai.FunctionCall) *functionCallState {
	key := s.resolveFunctionCallKey(fc)
	if s.currentFunctionCalls == nil {
		s.currentFunctionCalls = make(map[string]*functionCallState)
	}
	state, ok := s.currentFunctionCalls[key]
	if !ok {
		state = &functionCallState{key: key}
		s.currentFunctionCalls[key] = state
		s.trackActiveFunctionCall(fc, key)
	}
	s.lastFunctionCallKey = key
	return state
}

func (s *streamingResponseAggregator) resolveFunctionCallKey(fc *genai.FunctionCall) string {
	if fc == nil {
		return "__default__"
	}
	if fc.ID != "" {
		return fc.ID
	}
	if fc.Name != "" {
		if s.shouldStartNewUnnamedCall(fc) {
			return s.newSyntheticKey(fc.Name)
		}
		if key := s.singleActiveKeyForName(fc.Name); key != "" {
			return key
		}
		if key := s.mostRecentKeyForName(fc.Name); key != "" {
			return key
		}
		return s.newSyntheticKey(fc.Name)
	}
	if key := s.nextUnnamedFunctionCallKey(); key != "" {
		return key
	}
	return "__default__"
}

func isStreamingFunctionCall(fc *genai.FunctionCall) bool {
	if fc == nil {
		return false
	}
	return len(fc.PartialArgs) > 0 || fc.WillContinue != nil
}

func (s *streamingResponseAggregator) trackActiveFunctionCall(fc *genai.FunctionCall, key string) {
	if key == "" {
		return
	}
	s.activeFunctionCallOrder = append(s.activeFunctionCallOrder, key)
	name := ""
	if fc != nil {
		name = fc.Name
	}
	if name == "" {
		return
	}
	if s.activeFunctionCallKeysByName == nil {
		s.activeFunctionCallKeysByName = make(map[string][]string)
	}
	s.activeFunctionCallKeysByName[name] = append(s.activeFunctionCallKeysByName[name], key)
}

func (s *streamingResponseAggregator) removeActiveFunctionCallKey(key string) {
	if key == "" {
		return
	}
	if len(s.activeFunctionCallOrder) > 0 {
		out := s.activeFunctionCallOrder[:0]
		for _, k := range s.activeFunctionCallOrder {
			if k != key {
				out = append(out, k)
			}
		}
		s.activeFunctionCallOrder = out
		if s.unnamedCursor >= len(s.activeFunctionCallOrder) {
			s.unnamedCursor = 0
		}
	}
	if len(s.activeFunctionCallKeysByName) == 0 {
		return
	}
	for name, keys := range s.activeFunctionCallKeysByName {
		out := keys[:0]
		for _, k := range keys {
			if k != key {
				out = append(out, k)
			}
		}
		if len(out) == 0 {
			delete(s.activeFunctionCallKeysByName, name)
		} else {
			s.activeFunctionCallKeysByName[name] = out
		}
	}
}

func (s *streamingResponseAggregator) clearAllFunctionCallState() {
	s.currentFunctionCalls = nil
	s.activeFunctionCallOrder = nil
	s.activeFunctionCallKeysByName = nil
	s.lastFunctionCallKey = ""
	s.unnamedSequence = 0
	s.unnamedCursor = 0
}

func (s *streamingResponseAggregator) singleActiveKeyForName(name string) string {
	if name == "" || len(s.activeFunctionCallKeysByName) == 0 {
		return ""
	}
	keys := s.activeFunctionCallKeysByName[name]
	if len(keys) == 1 {
		return keys[0]
	}
	return ""
}

func (s *streamingResponseAggregator) mostRecentKeyForName(name string) string {
	if name == "" || len(s.activeFunctionCallKeysByName) == 0 {
		return ""
	}
	keys := s.activeFunctionCallKeysByName[name]
	if len(keys) == 0 {
		return ""
	}
	return keys[len(keys)-1]
}

func (s *streamingResponseAggregator) shouldStartNewUnnamedCall(fc *genai.FunctionCall) bool {
	if fc == nil || fc.Name == "" {
		return false
	}
	keys := s.activeFunctionCallKeysByName[fc.Name]
	return len(keys) == 0
}

func (s *streamingResponseAggregator) newSyntheticKey(name string) string {
	s.unnamedSequence++
	if name == "" {
		return fmt.Sprintf("__call_%d__", s.unnamedSequence)
	}
	return fmt.Sprintf("%s#%d", name, s.unnamedSequence)
}

func (s *streamingResponseAggregator) nextUnnamedFunctionCallKey() string {
	if len(s.activeFunctionCallOrder) == 0 {
		return ""
	}
	if s.unnamedCursor >= len(s.activeFunctionCallOrder) {
		s.unnamedCursor = 0
	}
	key := s.activeFunctionCallOrder[s.unnamedCursor]
	s.unnamedCursor++
	return key
}

func (s *streamingResponseAggregator) buildPendingFunctionCallParts() []*genai.Part {
	if s.currentFunctionCalls == nil {
		return nil
	}
	var parts []*genai.Part
	for _, key := range s.activeFunctionCallOrder {
		state := s.currentFunctionCalls[key]
		if state == nil || !state.hasData() {
			continue
		}
		if part := s.buildFunctionCallPart(state); part != nil {
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		for _, state := range s.currentFunctionCalls {
			if state == nil || !state.hasData() {
				continue
			}
			if part := s.buildFunctionCallPart(state); part != nil {
				parts = append(parts, part)
			}
		}
	}
	return parts
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
