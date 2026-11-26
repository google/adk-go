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
)

// usageFields builds the usage member of a Chat Completions body or final
// stream chunk. A negative total omits total_tokens from the JSON.
func usageFields(prompt, completion, total, reasoning int64) string {
	totalField := ""
	if total >= 0 {
		totalField = fmt.Sprintf(`"total_tokens":%d,`, total)
	}
	return fmt.Sprintf(`"usage":{"prompt_tokens":%d,"completion_tokens":%d,%s`+
		`"completion_tokens_details":{"reasoning_tokens":%d}}`,
		prompt, completion, totalField, reasoning)
}

// TestModel_ImpossibleUsageIsUnknown pins that usage which cannot be true is
// reported as unknown rather than stored as is, in blocking and streaming mode.
// The test treats total_tokens as counting at least the prompt and completion
// tokens, so a total below their sum contradicts its own parts. reasoning_tokens is a subset of
// completion_tokens, so reasoning above completion is a contradiction in the
// same way. Unknown (nil) says that the provider's numbers cannot be trusted.
// The turn itself still succeeds.
//
// Some OpenAI-compatible providers may instead count reasoning outside
// completion_tokens and add it to total_tokens. With reasoning up to
// completion that shape is kept by the rule above. Whether to accept it with
// reasoning above completion is left open, and no row pins it. The
// reasoning-above-completion rows below have a total of prompt plus
// completion, or no total, so they do not fit that shape either.
//
// A total above the sum, reasoning equal to completion, and an omitted total
// are not contradictions. An omitted total decodes to 0 and means not
// reported, so it must not be treated as a total below the sum. Those rows keep
// the provider's counts unchanged. The test does not assert
// CandidatesTokenCount, which depends on how reasoning maps onto candidates.
func TestModel_ImpossibleUsageIsUnknown(t *testing.T) {
	tests := []struct {
		name                                 string
		prompt, completion, total, reasoning int64
		wantUsage                            bool
		why                                  string
	}{
		{
			name:   "consistent counts are kept",
			prompt: 100, completion: 50, total: 150, reasoning: 0,
			wantUsage: true,
		},
		{
			name:   "consistent counts with reasoning are kept",
			prompt: 100, completion: 50, total: 150, reasoning: 20,
			wantUsage: true,
		},
		{
			name:   "total above prompt plus completion is kept",
			prompt: 100, completion: 50, total: 180, reasoning: 20,
			wantUsage: true,
		},
		{
			name:   "reasoning equal to completion is kept",
			prompt: 100, completion: 50, total: 150, reasoning: 50,
			wantUsage: true,
		},
		{
			name:   "omitted total is kept",
			prompt: 100, completion: 50, total: -1, reasoning: 20,
			wantUsage: true,
		},
		{
			name:   "total below prompt plus completion",
			prompt: 100, completion: 50, total: 120, reasoning: 0,
			why: "total_tokens 120 is below prompt_tokens 100 + completion_tokens 50",
		},
		{
			name:   "reasoning above completion",
			prompt: 100, completion: 10, total: 110, reasoning: 50,
			why: "reasoning_tokens 50 is above completion_tokens 10",
		},
		{
			name:   "reasoning above completion with total omitted",
			prompt: 100, completion: 10, total: -1, reasoning: 50,
			why: "reasoning_tokens 50 is above completion_tokens 10, with total_tokens omitted",
		},
	}
	for _, tt := range tests {
		usage := usageFields(tt.prompt, tt.completion, tt.total, tt.reasoning)
		blocking := `{"id":"c","model":"m","object":"chat.completion",` +
			`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],` +
			usage + `}`
		stream := []string{
			`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}`,
			`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` + usage + `}`,
		}
		for _, streaming := range []bool{false, true} {
			name := tt.name + "/blocking"
			handler := serveJSON(blocking)
			if streaming {
				name = tt.name + "/stream"
				handler = serveSSE(stream...)
			}
			t.Run(name, func(t *testing.T) {
				rig := newTestRig(t, handler)
				got, err := ask(t, rig.model(t), streaming)
				if err != nil {
					t.Fatalf("GenerateContent() err = %v, want the turn to succeed", err)
				}
				if len(got) == 0 {
					t.Fatal("GenerateContent() returned no responses")
				}
				final := got[len(got)-1]
				if text := responseText(final); text != "hello" {
					t.Errorf("final text = %q, want %q", text, "hello")
				}
				switch {
				case tt.wantUsage && final.UsageMetadata == nil:
					t.Error("UsageMetadata = nil, want the provider's usage kept: the counts are not contradictory")
				case tt.wantUsage:
					u := final.UsageMetadata
					wantTotal := max(tt.total, 0) // an omitted total decodes to 0
					if u.PromptTokenCount != int32(tt.prompt) || u.TotalTokenCount != int32(wantTotal) ||
						u.ThoughtsTokenCount != int32(tt.reasoning) {
						t.Errorf("UsageMetadata = {Prompt: %d, Thoughts: %d, Total: %d}, want {Prompt: %d, Thoughts: %d, Total: %d}: kept counts are unchanged",
							u.PromptTokenCount, u.ThoughtsTokenCount, u.TotalTokenCount, tt.prompt, tt.reasoning, wantTotal)
					}
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
