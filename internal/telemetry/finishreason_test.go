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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// genaiTooManyToolCalls has no constant in the genai version v1 pins.
const genaiTooManyToolCalls = genai.FinishReason("TOO_MANY_TOOL_CALLS")

// schemaFinishReasons are the members of the semantic conventions'
// FinishReason enum.
var schemaFinishReasons = map[string]bool{
	"stop": true, "length": true, "content_filter": true,
	"tool_call": true, "compaction": true, "error": true,
}

func TestSchemaFinishReason(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
		want string
		err  error
	}{
		{name: "stop", resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop}, want: "stop"},
		{name: "stop with a tool call", resp: &model.LLMResponse{
			Content:      &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
			FinishReason: genai.FinishReasonStop,
		}, want: "tool_call"},
		{name: "unset", resp: &model.LLMResponse{}, want: "stop"},
		{name: "unset with a tool call", resp: &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
		}, want: "tool_call"},
		{name: "unspecified", resp: &model.LLMResponse{FinishReason: genai.FinishReasonUnspecified}, want: "stop"},
		{name: "max tokens", resp: &model.LLMResponse{FinishReason: genai.FinishReasonMaxTokens}, want: "length"},
		{name: "safety", resp: &model.LLMResponse{FinishReason: genai.FinishReasonSafety}, want: "content_filter"},
		{name: "recitation", resp: &model.LLMResponse{FinishReason: genai.FinishReasonRecitation}, want: "content_filter"},
		{name: "blocklist", resp: &model.LLMResponse{FinishReason: genai.FinishReasonBlocklist}, want: "content_filter"},
		{name: "spii", resp: &model.LLMResponse{FinishReason: genai.FinishReasonSPII}, want: "content_filter"},
		{name: "malformed function call", resp: &model.LLMResponse{FinishReason: genai.FinishReasonMalformedFunctionCall}, want: "error"},
		{name: "too many tool calls", resp: &model.LLMResponse{FinishReason: genaiTooManyToolCalls}, want: "error"},
		{name: "other", resp: &model.LLMResponse{FinishReason: genai.FinishReasonOther}, want: "error"},
		{
			name: "error code beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, ErrorCode: "429"},
			want: "error",
		},
		{
			name: "interruption beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, Interrupted: true},
			want: "error",
		},
		{
			name: "error code beats a tool call",
			resp: &model.LLMResponse{
				Content:   &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
				ErrorCode: "500",
			},
			want: "error",
		},
		{
			name: "error beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop},
			err:  errTest,
			want: "error",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := schemaFinishReason(tc.resp, tc.err); got != tc.want {
				t.Errorf("schemaFinishReason() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestSchemaFinishReason_EveryGenaiValue(t *testing.T) {
	all := []genai.FinishReason{
		genai.FinishReasonUnspecified, genai.FinishReasonStop, genai.FinishReasonMaxTokens,
		genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonLanguage,
		genai.FinishReasonOther, genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII, genai.FinishReasonMalformedFunctionCall, genai.FinishReasonImageSafety,
		genai.FinishReasonUnexpectedToolCall, genaiTooManyToolCalls,
		genai.FinishReasonImageProhibitedContent, genai.FinishReasonNoImage,
		genai.FinishReasonImageRecitation, genai.FinishReasonImageOther,
	}
	for _, fr := range all {
		got := schemaFinishReason(&model.LLMResponse{FinishReason: fr}, nil)
		if !schemaFinishReasons[got] {
			t.Errorf("genai %s maps to %q, which the schema does not define", fr, got)
		}
	}
	// An unknown future value is lowercased rather than passed through.
	if got := schemaFinishReason(&model.LLMResponse{FinishReason: "BRAND_NEW_REASON"}, nil); got != "brand_new_reason" {
		t.Errorf("unknown finish reason = %q, want brand_new_reason", got)
	}
}
