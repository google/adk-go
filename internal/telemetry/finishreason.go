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

package telemetry

import (
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// Members of the semantic conventions' FinishReason enum.
const (
	finishStop          = "stop"
	finishLength        = "length"
	finishContentFilter = "content_filter"
	finishToolCall      = "tool_call"
	finishError         = "error"
)

// hasFunctionCall reports whether content carries a tool call. It reads the
// genai parts directly rather than converting them, since this runs on every
// span.
func hasFunctionCall(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p != nil && p.FunctionCall != nil {
			return true
		}
	}
	return false
}

// schemaFinishReason maps a response onto the semantic conventions'
// finish_reason.
//
// A non-nil error, a response carrying an error code, or one cut short by an
// interruption did not complete whatever its finish reason claims, so it
// reports "error".
// Gemini reports STOP for a turn that is a tool call, which the schema
// distinguishes from a plain stop, so a turn containing tool call parts
// reports "tool_call". genai enum values are protobuf wire names in
// SCREAMING_SNAKE_CASE and are never emitted verbatim. A value this mapping
// does not know is lowercased, which the schema allows since finish_reason
// accepts any string.
func schemaFinishReason(resp *model.LLMResponse, err error) string {
	if resp == nil || err != nil || resp.ErrorCode != "" || resp.Interrupted {
		return finishError
	}
	switch resp.FinishReason {
	case "", genai.FinishReasonUnspecified, genai.FinishReasonStop:
		if hasFunctionCall(resp.Content) {
			return finishToolCall
		}
		return finishStop
	case genai.FinishReasonMaxTokens:
		return finishLength
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return finishContentFilter
	case genai.FinishReasonOther,
		genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall,
		genai.FinishReason("TOO_MANY_TOOL_CALLS"), // no constant in the genai version v1 pins
		genai.FinishReasonNoImage,
		genai.FinishReasonImageOther:
		return finishError
	default:
		return strings.ToLower(string(resp.FinishReason))
	}
}
