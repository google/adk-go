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
	"math"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// The usage fields are int32, so adding two of them in int32 can wrap before
// the sum is widened to the int an OpenTelemetry attribute takes. The sum must
// be taken in a wider type.

func TestGenerateContent_UsageSumsDoNotOverflow(t *testing.T) {
	tests := []struct {
		name  string
		usage *genai.GenerateContentResponseUsageMetadata
		key   attribute.Key
		want  string
	}{
		{
			name:  "input is prompt plus tool-use prompt",
			usage: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: math.MaxInt32, ToolUsePromptTokenCount: 5},
			key:   semconv.GenAIUsageInputTokensKey,
			want:  "2147483652",
		},
		{
			name:  "output is candidates plus thoughts",
			usage: &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: math.MaxInt32, ThoughtsTokenCount: 5},
			key:   semconv.GenAIUsageOutputTokensKey,
			want:  "2147483652",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exporter := setupTestTracer(t)

			_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{ModelName: "test-model"})
			traceGenerateContentResult(span, generateContentResult{
				Response: &model.LLMResponse{UsageMetadata: tc.usage},
			})
			span.End()

			spans := exporter.GetSpans()
			if len(spans) != 1 {
				t.Fatalf("got %d spans, want 1", len(spans))
			}
			got := attributesToMap(spans[0].Attributes)[tc.key]
			if got != tc.want {
				t.Errorf("attribute %q = %q, want %q: the sum must be computed in a type wider than int32 and must not be negative", tc.key, got, tc.want)
			}
		})
	}
}

func TestTraceCompactionResult_UsageSumDoesNotOverflow(t *testing.T) {
	exporter := setupTestTracer(t)

	_, span := StartCompactEventsSpan(t.Context(), StartCompactEventsSpanParams{Trigger: CompactionTriggerSlidingWindow})
	TraceCompactionResult(span, TraceCompactionResultParams{
		Declined: true,
		Usage:    &genai.GenerateContentResponseUsageMetadata{CandidatesTokenCount: math.MaxInt32, ThoughtsTokenCount: 5},
	})
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	const want = "2147483652"
	if got := attributesToMap(spans[0].Attributes)[genAICompactionOutputTokens]; got != want {
		t.Errorf("attribute %q = %q, want %q: the candidates plus thoughts sum must be computed in a type wider than int32 and must not be dropped or negative", genAICompactionOutputTokens, got, want)
	}
}
