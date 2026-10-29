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

package llminternal

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// BenchmarkStreamAggregator feeds one streamed answer of n text chunks
// through the aggregator alone, the part of a streamed model call that is
// paid before any event exists. Bytes allocated grow with the square of the
// chunk count when the text is accumulated by concatenation.
func BenchmarkStreamAggregator(b *testing.B) {
	for _, chunks := range []int{200, 2000} {
		b.Run(fmt.Sprintf("chunks=%d", chunks), func(b *testing.B) {
			chunk := strings.Repeat("x", 50)
			responses := make([]*genai.GenerateContentResponse, chunks)
			for i := range responses {
				var finishReason genai.FinishReason
				if i == chunks-1 {
					finishReason = genai.FinishReasonStop
				}
				responses[i] = &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
					Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: chunk}}},
					FinishReason: finishReason,
				}}}
			}
			b.ReportAllocs()
			for b.Loop() {
				aggregator := NewStreamingResponseAggregator()
				for _, resp := range responses {
					for _, err := range aggregator.ProcessResponse(b.Context(), resp) {
						if err != nil {
							b.Fatal(err)
						}
					}
				}
				if got := aggregator.Close(); len(got.Content.Parts[0].Text) != chunks*len(chunk) {
					b.Fatalf("aggregated %d bytes, want %d", len(got.Content.Parts[0].Text), chunks*len(chunk))
				}
			}
		})
	}
}
