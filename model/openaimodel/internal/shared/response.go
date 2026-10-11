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

package shared

import (
	"errors"
	"iter"
	"math"
	"reflect"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// SafeInt32 narrows a token count to the int32 genai carries, saturating at
// math.MaxInt32 rather than wrapping.
func SafeInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(v)
}

// SingleErrorSequence is a response sequence that yields err and ends.
func SingleErrorSequence(err error) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, err)
	}
}

// IsEmptyOutput reports whether a conversion failed for want of anything to
// convert, as against something unusable.
func IsEmptyOutput(err error) bool {
	return errors.Is(err, ErrNoOutputItems) || errors.Is(err, ErrNoTextOrToolContent) ||
		errors.Is(err, ErrNoChoices)
}

// CarriesContent reports whether a response holds anything a caller can read.
// An aggregated turn can arrive empty: a streamed function call with no name is
// dropped, and deltas may contribute no part at all.
func CarriesContent(resp *model.LLMResponse) bool {
	return resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0
}

// PartsWithoutCalls splits a turn's parts into those that are not function
// calls, the calls themselves, both in the order they streamed, and the index
// among the kept parts where the first call sat — where the calls the turn ends
// up reporting belong. A turn holding no call reports the index past the last
// part.
func PartsWithoutCalls(content *genai.Content) ([]*genai.Part, []*genai.FunctionCall, int) {
	if content == nil {
		return nil, nil, 0
	}
	kept := make([]*genai.Part, 0, len(content.Parts))
	var calls []*genai.FunctionCall
	at := -1
	for _, part := range content.Parts {
		if part.FunctionCall != nil {
			if at < 0 {
				at = len(kept)
			}
			calls = append(calls, part.FunctionCall)
			continue
		}
		kept = append(kept, part)
	}
	if at < 0 {
		at = len(kept)
	}
	return kept, calls, at
}

// CompletedContentSupersedes reports whether a terminal snapshot can safely
// replace content assembled from stream deltas. Reasoning alone is not a usable
// replacement, and the snapshot must retain all visible text and function calls
// that callers already received from the stream.
// When replacement is allowed, reasoning follows the completed response to match
// the blocking path. Streamed thoughts absent from that snapshot are not added
// back; they have already been delivered as partial responses.
func CompletedContentSupersedes(aggregate, completed *genai.Content) bool {
	if completed == nil {
		return false
	}

	var aggregateText, completedText strings.Builder
	var aggregateCalls, completedCalls []*genai.FunctionCall
	usable := false
	for _, part := range aggregate.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" && !part.Thought {
			aggregateText.WriteString(part.Text)
		}
		if part.FunctionCall != nil {
			aggregateCalls = append(aggregateCalls, part.FunctionCall)
		}
	}
	for _, part := range completed.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" && !part.Thought {
			completedText.WriteString(part.Text)
			usable = true
		}
		if part.FunctionCall != nil {
			completedCalls = append(completedCalls, part.FunctionCall)
			usable = true
		}
	}
	if !usable || !strings.Contains(completedText.String(), aggregateText.String()) {
		return false
	}

	matched := make([]bool, len(completedCalls))
	for _, aggregateCall := range aggregateCalls {
		found := false
		for i, completedCall := range completedCalls {
			if !matched[i] && reflect.DeepEqual(aggregateCall, completedCall) {
				matched[i] = true
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// AnswerText is the response's text as a caller reads it, thoughts excluded:
// what logprobs have to describe.
func AnswerText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var text strings.Builder
	for _, part := range content.Parts {
		if part.Thought {
			continue
		}
		text.WriteString(part.Text)
	}
	return text.String()
}

// SinglePartResponse wraps one streamed part as a genai response.
//
// The candidate deliberately carries no finish reason: the aggregator treats any
// non-empty one as terminal, and genai.FinishReasonUnspecified is the non-empty
// string "FINISH_REASON_UNSPECIFIED", so setting it marks every delta the end of
// the turn. generateStream reports the real reason on the final response.
func SinglePartResponse(part *genai.Part) *genai.GenerateContentResponse {
	if part == nil {
		return nil
	}
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role:  string(genai.RoleModel),
					Parts: []*genai.Part{part},
				},
			},
		},
	}
}
