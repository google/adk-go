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
	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

// convertChatCompletion converts a Chat Completions response into the generic
// genai response, and is what a streamed turn is finalized with too.
func convertChatCompletion(resp *openai.ChatCompletion) (*genai.GenerateContentResponse, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// convertChatMessage converts one assistant message into genai parts, text,
// refusal and tool calls alike.
func convertChatMessage(msg openai.ChatCompletionMessage) ([]*genai.Part, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// chatFinishReason maps a Chat Completions finish_reason onto the genai finish
// reason, reading an unrecognized one as OTHER.
func chatFinishReason(reason string) genai.FinishReason { //nolint:unused // scaffold; wired when the conversion code lands
	return genai.FinishReasonUnspecified
}

// convertChatUsage converts Chat Completions token accounting into genai usage
// metadata.
func convertChatUsage(usage openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata { //nolint:unused // scaffold; wired when the conversion code lands
	return nil
}

// convertChatLogprobs converts a choice's token log probabilities into the
// genai shape.
func convertChatLogprobs(logprobs openai.ChatCompletionChoiceLogprobs) *genai.LogprobsResult { //nolint:unused // scaffold; wired when the conversion code lands
	return nil
}
