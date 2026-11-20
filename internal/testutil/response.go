// Copyright 2026 Google LLC
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

package testutil

import (
	"fmt"
	"iter"
	"strings"

	"google.golang.org/adk/v2/model"
)

// TextResponse holds the concatenated text from a response stream,
// separated into partial and final parts.
type TextResponse struct {
	// PartialText is the full text concatenated from all partial (streaming) responses.
	PartialText string
	// FinalText is the full text concatenated from all final (non-partial) responses.
	FinalText string
}

// ReadTextResponse reads a model response stream, concatenating the text of
// the first part of each partial response into PartialText and of each final
// response into FinalText. It returns an error for a response with no content.
// When the stream yields an error, it returns that error with the text read so
// far.
func ReadTextResponse(s iter.Seq2[*model.LLMResponse, error]) (TextResponse, error) {
	var partialBuilder, finalBuilder strings.Builder
	var result TextResponse

	for resp, err := range s {
		if err != nil {
			// Return what we have so far, along with the error.
			result.PartialText = partialBuilder.String()
			result.FinalText = finalBuilder.String()
			return result, err
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return result, fmt.Errorf("encountered a response with no content (partial=%t)", resp.Partial)
		}

		text := resp.Content.Parts[0].Text
		if resp.Partial {
			partialBuilder.WriteString(text)
		} else {
			finalBuilder.WriteString(text)
		}
	}

	result.PartialText = partialBuilder.String()
	result.FinalText = finalBuilder.String()
	return result, nil
}
