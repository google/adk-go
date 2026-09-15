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

// convertChatTools converts function declarations into Chat Completions tools,
// whose declaration nests under a "function" object the Responses shape does
// not have.
func convertChatTools(cfg *genai.GenerateContentConfig) ([]openai.ChatCompletionToolUnionParam, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// convertChatFunctionDeclaration converts one function declaration into a Chat
// Completions function tool.
func convertChatFunctionDeclaration(fn *genai.FunctionDeclaration) (*openai.ChatCompletionFunctionToolParam, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// convertChatToolChoice translates a tool config into the Chat Completions
// tool_choice, which names a function one level deeper than Responses does.
func convertChatToolChoice(toolCfg *genai.ToolConfig) (*openai.ChatCompletionToolChoiceOptionUnionParam, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}
