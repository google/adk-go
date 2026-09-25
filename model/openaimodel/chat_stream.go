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

// chatStreamTranslator turns Chat Completions chunks into genai responses while
// accumulating the whole turn.
//
// Tool calls are not emitted as they stream. The protocol has no event marking
// one complete — arguments arrive as fragments and the only signal they have
// stopped is the stream ending — so they reach the caller on the final response,
// built from the accumulated snapshot by the same converter the blocking path
// uses.
type chatStreamTranslator struct {
	acc openai.ChatCompletionAccumulator
	// usage is the latest usage a chunk reported, or nil if none did. It is
	// kept apart from the accumulator, which sums every report: right only for
	// a provider sending usage once, while several resend the running total on
	// every chunk. The latest report is the turn's total either way, which is
	// also how adk-python's LiteLLM path reads it.
	usage *openai.CompletionUsage
}

// newChatStreamTranslator returns a translator for one streamed turn.
func newChatStreamTranslator() *chatStreamTranslator {
	return &chatStreamTranslator{}
}

// process folds one chunk into the accumulated turn and reports the partial it
// contributes, or nil for a chunk a caller sees nothing of.
func (t *chatStreamTranslator) process(chunk openai.ChatCompletionChunk) *genai.GenerateContentResponse {
	// A chunk the accumulator rejects — a choice index beyond its bounds, or a
	// tool-call index that would grow it too far — leaves the snapshot
	// untouched. The delta is still yielded, so text a caller could read does
	// not vanish because the snapshot could not hold it.
	t.acc.AddChunk(chunk)
	if chunk.JSON.Usage.Valid() {
		usage := chunk.Usage
		t.usage = &usage
	}

	if len(chunk.Choices) == 0 {
		// The usage-only chunk that closes a stream requesting usage.
		return nil
	}
	delta := chunk.Choices[0].Delta
	switch {
	case delta.Content != "":
		return singlePartResponse(&genai.Part{Text: delta.Content})
	case delta.Refusal != "":
		// Blocking reports a refusal as text, so streaming does the same.
		return singlePartResponse(&genai.Part{Text: delta.Refusal})
	}
	return nil
}

// completion is the whole turn as the blocking path would have received it.
func (t *chatStreamTranslator) completion() *openai.ChatCompletion {
	return &t.acc.ChatCompletion
}
