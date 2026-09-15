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

// chatStreamTranslator turns Chat Completions chunks into genai responses. Tool
// calls are not emitted as they stream: the protocol has no event marking one
// complete, so they reach the caller only on the final response the
// accumulator's snapshot builds.
type chatStreamTranslator struct { //nolint:unused // scaffold; wired when the conversion code lands
	acc openai.ChatCompletionAccumulator
}

// newChatStreamTranslator returns a translator for one streamed turn.
func newChatStreamTranslator() *chatStreamTranslator { //nolint:unused // scaffold; wired when the conversion code lands
	return &chatStreamTranslator{}
}

// process folds one chunk into the accumulated turn and returns the partial it
// contributes, or nil for a chunk a caller sees nothing of.
func (t *chatStreamTranslator) process(chunk openai.ChatCompletionChunk) (*genai.GenerateContentResponse, error) { //nolint:unused // scaffold; wired when the conversion code lands
	return nil, errNotImplemented
}

// completion is the whole turn as the blocking path would have received it,
// which is what the final streamed response is built from.
func (t *chatStreamTranslator) completion() *openai.ChatCompletion { //nolint:unused // scaffold; wired when the conversion code lands
	return nil
}
