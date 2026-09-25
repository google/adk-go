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

	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

// convertChatCompletion converts a Chat Completions response into the generic
// genai response. A streamed turn is finalized through here too, so the two
// paths cannot disagree about tool calls, finish reason or usage.
func convertChatCompletion(resp *openai.ChatCompletion) (*genai.GenerateContentResponse, error) {
	if resp == nil {
		return nil, ErrEmptyResponse
	}
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoices
	}
	choice := resp.Choices[0]
	parts, err := convertChatMessage(choice.Message)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, ErrNoTextOrToolContent
	}
	out := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:        &genai.Content{Role: string(genai.RoleModel), Parts: parts},
			FinishReason:   chatFinishReason(choice.FinishReason),
			LogprobsResult: convertChatLogprobs(choice.Logprobs),
		}},
		ModelVersion: resp.Model,
		ResponseID:   resp.ID,
	}
	if resp.JSON.Usage.Valid() {
		// A provider that omits usage has not said the turn cost nothing, which
		// zeros would; streaming leaves it unset in the same case.
		out.UsageMetadata = convertChatUsage(resp.Usage)
	}
	return out, nil
}

// convertChatMessage converts an assistant message into genai parts. A refusal
// becomes text, matching what the Responses path does with a refusal content
// block, so one model reports a refusal the same way on either endpoint.
func convertChatMessage(msg openai.ChatCompletionMessage) ([]*genai.Part, error) {
	var parts []*genai.Part
	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}
	if msg.Refusal != "" {
		parts = append(parts, &genai.Part{Text: msg.Refusal})
	}
	for _, call := range msg.ToolCalls {
		fn := call.Function
		args, err := chatToolCallArgs(fn.Arguments)
		if err != nil {
			// Name the offending call: conversion aborts on the first error, so
			// nothing else identifies it.
			return nil, fmt.Errorf("%w (name %q, call_id %q)", err, fn.Name, call.ID)
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{Name: fn.Name, ID: call.ID, Args: args},
		})
	}
	return parts, nil
}

// chatToolCallArgs decodes a tool call's arguments, reading an empty payload as
// a call that takes none.
func chatToolCallArgs(raw string) (map[string]any, error) {
	if raw == "" {
		return map[string]any{}, nil
	}
	args := map[string]any{}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFunctionCallArgs, err)
	}
	if args == nil {
		// The payload was JSON null: the call takes no arguments.
		return map[string]any{}, nil
	}
	return args, nil
}

// chatFinishReason maps a Chat Completions finish_reason onto the genai finish
// reason. "tool_calls" is a clean stop, as it is in adk-python's LiteLLM path:
// the model finished its turn by calling a tool.
func chatFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop", "tool_calls", "function_call":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "":
		// A provider that states nothing says nothing about why the turn ended,
		// and reading that silence as a clean stop would have a caller that
		// retries on anything but STOP accept a partial answer as final.
		return genai.FinishReasonUnspecified
	default:
		return genai.FinishReasonOther
	}
}

// convertChatUsage converts Chat Completions token accounting into genai usage
// metadata, whose field names differ from this endpoint's on every count.
func convertChatUsage(usage openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        safeInt32(usage.PromptTokens),
		CandidatesTokenCount:    safeInt32(usage.CompletionTokens),
		TotalTokenCount:         safeInt32(usage.TotalTokens),
		CachedContentTokenCount: safeInt32(usage.PromptTokensDetails.CachedTokens),
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.PromptTokens)},
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.CompletionTokens)},
		},
		ThoughtsTokenCount: safeInt32(usage.CompletionTokensDetails.ReasoningTokens),
	}
}

// convertChatLogprobs converts a choice's content log probabilities into the
// genai shape. Refusal logprobs are left out, because the tokens they describe
// are not the ones a caller reads back as the answer.
func convertChatLogprobs(logprobs openai.ChatCompletionChoiceLogprobs) *genai.LogprobsResult {
	if len(logprobs.Content) == 0 {
		return nil
	}
	res := &genai.LogprobsResult{}
	for _, lp := range logprobs.Content {
		res.ChosenCandidates = append(res.ChosenCandidates, &genai.LogprobsResultCandidate{
			Token:          lp.Token,
			LogProbability: float32(lp.Logprob),
		})
		var top []*genai.LogprobsResultCandidate
		for _, tlp := range lp.TopLogprobs {
			top = append(top, &genai.LogprobsResultCandidate{
				Token:          tlp.Token,
				LogProbability: float32(tlp.Logprob),
			})
		}
		res.TopCandidates = append(res.TopCandidates, &genai.LogprobsResultTopCandidates{Candidates: top})
	}
	return res
}
