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

package responses

import (
	"fmt"
	"testing"

	"google.golang.org/adk/v2/model"
)

// usageBody builds a completed Responses body that carries the given usage
// counts and a one-word answer.
func usageBody(input, output, total, reasoning int64) string {
	return fmt.Sprintf(`{"id":"resp_1","model":"stream-model","status":"completed",`+
		`"usage":{"input_tokens":%d,"output_tokens":%d,"total_tokens":%d,`+
		`"output_tokens_details":{"reasoning_tokens":%d}},`+
		`"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`,
		input, output, total, reasoning)
}

// TestModel_ImpossibleUsageIsUnknown pins that usage which cannot be true is
// reported as unknown rather than stored as is, in blocking and streaming mode.
// A total below input plus output, or reasoning above output, is a provider
// bug: passing the numbers on makes readers that sum the fields disagree with
// readers that use the total. Unknown (nil) is what readers already handle. The
// turn itself still succeeds.
func TestModel_ImpossibleUsageIsUnknown(t *testing.T) {
	tests := []struct {
		name                            string
		input, output, total, reasoning int64
		wantUsage                       bool
		why                             string
	}{
		{
			name:  "consistent counts are kept",
			input: 100, output: 50, total: 150, reasoning: 0,
			wantUsage: true,
		},
		{
			name:  "consistent counts with reasoning are kept",
			input: 100, output: 50, total: 150, reasoning: 20,
			wantUsage: true,
		},
		{
			name:  "total below input plus output",
			input: 100, output: 50, total: 7, reasoning: 0,
			why: "total_tokens 7 is below input_tokens 100 + output_tokens 50",
		},
		{
			name:  "reasoning above output",
			input: 100, output: 10, total: 110, reasoning: 50,
			why: "reasoning_tokens 50 is above output_tokens 10",
		},
	}
	for _, tt := range tests {
		body := usageBody(tt.input, tt.output, tt.total, tt.reasoning)
		for _, stream := range []bool{false, true} {
			name := tt.name + "/blocking"
			if stream {
				name = tt.name + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				var (
					got []*model.LLMResponse
					err error
				)
				if stream {
					got, err = runStream(t, evCreated, evDelta1, evDelta2,
						`{"type":"response.completed","response":`+body+`}`)
				} else {
					got, err = runBlocking(t, body)
				}
				if err != nil {
					t.Fatalf("GenerateContent() err = %v, want the turn to succeed", err)
				}
				if len(got) == 0 {
					t.Fatal("GenerateContent() returned no responses")
				}
				final := got[len(got)-1]
				if text := allText(final.Content); text != "hello" {
					t.Errorf("final text = %q, want %q", text, "hello")
				}
				switch {
				case tt.wantUsage && final.UsageMetadata == nil:
					t.Error("UsageMetadata = nil, want the provider's usage kept: the counts are consistent")
				case !tt.wantUsage && final.UsageMetadata != nil:
					u := final.UsageMetadata
					t.Errorf("UsageMetadata = {Prompt: %d, Candidates: %d, Thoughts: %d, Total: %d}, "+
						"want nil: usage that cannot be true is unknown (%s)",
						u.PromptTokenCount, u.CandidatesTokenCount, u.ThoughtsTokenCount, u.TotalTokenCount, tt.why)
				}
			})
		}
	}
}
