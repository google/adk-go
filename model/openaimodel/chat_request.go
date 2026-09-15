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

	"google.golang.org/adk/v2/model"
)

// buildChatParams converts a generic LLMRequest into Chat Completions request
// params.
func buildChatParams(modelName string, req *model.LLMRequest) (openai.ChatCompletionNewParams, error) {
	return openai.ChatCompletionNewParams{}, errNotImplemented
}

// convertChatMessages converts ADK contents into a Chat Completions message
// list, pairing each tool result with the call it answers and leaving prior-turn
// thoughts out.
func convertChatMessages(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// chatRole maps a genai content role onto the Chat Completions message role,
// keeping the system and developer roles the Responses path already
// distinguishes.
func chatRole(role genai.Role) (string, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return "", errNotImplemented
}

// applyChatGenerationConfig translates the generation config onto Chat
// Completions params, including the stop sequences, penalties and seed the
// Responses API cannot express.
func applyChatGenerationConfig(params *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) error { //nolint:unused // scaffold; wired when the conversion code lands
	return errNotImplemented
}

// applyChatThinkingConfig maps genai's thinking config onto reasoning_effort,
// which carries no summary and so cannot honour IncludeThoughts.
func applyChatThinkingConfig(params *openai.ChatCompletionNewParams, cfg *genai.ThinkingConfig) error { //nolint:unused // scaffold; wired when the conversion code lands
	return errNotImplemented
}
