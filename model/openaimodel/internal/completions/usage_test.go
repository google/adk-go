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

package completions

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

// TestModel_ReasoningTokensAreNotCountedAsCandidates covers the accounting for
// reasoning models. Chat Completions reports completion_tokens as everything
// the model generated, reasoning included, and reasoning_tokens as the share of
// it spent thinking. genai keeps the two apart: CandidatesTokenCount is the
// visible output only, ThoughtsTokenCount is the thinking, and the two add up
// to the output. Copying completion_tokens into CandidatesTokenCount counts the
// reasoning twice, and a consumer that sums the two fields, as the telemetry
// package does for gen_ai.usage.output_tokens, reports completion_tokens plus
// reasoning_tokens.
func TestModel_ReasoningTokensAreNotCountedAsCandidates(t *testing.T) {
	tests := []struct {
		name           string
		reasoning      int
		wantCandidates int32
		wantThoughts   int32
	}{
		{name: "some reasoning", reasoning: 1000, wantCandidates: 3000, wantThoughts: 1000},
		{name: "no reasoning", reasoning: 0, wantCandidates: 4000, wantThoughts: 0},
		{name: "reasoning is the whole output", reasoning: 4000, wantCandidates: 0, wantThoughts: 4000},
	}
	for _, tc := range tests {
		// prompt 1000 + completion 4000 = total 5000, reasoning included in the 4000.
		// One line, because the test helper writes each frame after a single "data: " prefix.
		usage := fmt.Sprintf(`{"prompt_tokens":1000,"completion_tokens":4000,"total_tokens":5000,"completion_tokens_details":{"reasoning_tokens":%d}}`, tc.reasoning)
		for _, stream := range []bool{false, true} {
			mode := "blocking"
			handler := serveJSON(`{"id":"c","model":"m",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"sunny"}}],
				"usage":` + usage + `}`)
			if stream {
				mode = "stream"
				handler = serveSSE(
					`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"sunny"}}]}`,
					`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
					`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[],"usage":`+usage+`}`,
				)
			}
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				got, err := ask(t, newTestRig(t, handler).model(t), stream)
				if err != nil {
					t.Fatalf("GenerateContent() err = %v", err)
				}
				if len(got) == 0 || got[len(got)-1].UsageMetadata == nil {
					t.Fatalf("responses = %d, want a final one carrying usage metadata", len(got))
				}
				usage := got[len(got)-1].UsageMetadata
				// The text entry of CandidatesTokensDetails is the visible output too,
				// so it follows CandidatesTokenCount. -1 marks a missing entry.
				textDetail := int32(-1)
				for _, d := range usage.CandidatesTokensDetails {
					if d.Modality == genai.MediaModalityText {
						textDetail = d.TokenCount
					}
				}
				type counts struct{ Prompt, Candidates, CandidatesTextDetail, Thoughts, Total int32 }
				want := counts{Prompt: 1000, Candidates: tc.wantCandidates, CandidatesTextDetail: tc.wantCandidates, Thoughts: tc.wantThoughts, Total: 5000}
				gotCounts := counts{
					Prompt:               usage.PromptTokenCount,
					Candidates:           usage.CandidatesTokenCount,
					CandidatesTextDetail: textDetail,
					Thoughts:             usage.ThoughtsTokenCount,
					Total:                usage.TotalTokenCount,
				}
				if diff := cmp.Diff(want, gotCounts); diff != "" {
					t.Errorf("usage counts (-want +got): CandidatesTokenCount and its text detail must exclude the reasoning tokens that completion_tokens includes:\n%s", diff)
				}
				if out := usage.CandidatesTokenCount + usage.ThoughtsTokenCount; out != 4000 {
					t.Errorf("CandidatesTokenCount + ThoughtsTokenCount = %d, want completion_tokens (4000): the output must not be counted twice", out)
				}
			})
		}
	}
}
