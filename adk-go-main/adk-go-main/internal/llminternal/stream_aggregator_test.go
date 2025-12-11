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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
)

type streamAggregatorTest struct {
	name                 string
	initialResponses     []*genai.Content
	numberOfStreamCalls  int
	streamResponsesCount int
	want                 []*genai.Content
	wantPartial          []bool
}

func TestStreamAggregator(t *testing.T) {
	ctx := t.Context()
	testCases := []streamAggregatorTest{
		{
			name: "two streams of 2 responses each",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  2,
			streamResponsesCount: 2,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				// Results from second GenerateStream call
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false,
				// Results from second GenerateStream call
				true, true, false,
			},
		},
		{
			name: "two streams of 3 and 2 responses each",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response5", "model"),
			},
			numberOfStreamCalls:  2,
			streamResponsesCount: 3,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response1response2response3", "model"),
				// Results from second GenerateStream call
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response5", "model"),
				genai.NewContentFromText("response4response5", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, true, false,
				// Results from second GenerateStream call
				true, true, false,
			},
		},
		{
			name: "stream with intermediate response should reset",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				nil, // force reset with empty context
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 5,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				nil, // proxy still send the nil
				// Results from second GenerateStream call
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false, false,
				true, true, false,
			},
		},
		{
			name: "stream with audio should reset",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 5,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false, false,
				true, true, false,
			},
		},
		{
			name: "audio stream should not generate any aggregated",
			initialResponses: []*genai.Content{
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 3,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				false, false, false,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			responsesCopy := make([]*genai.Content, len(tc.initialResponses))
			copy(responsesCopy, tc.initialResponses)

			mockModel := &testutil.MockModel{
				Responses:            responsesCopy,
				StreamResponsesCount: tc.streamResponsesCount,
			}

			count := 0
			callCount := 0
			for callCount < tc.numberOfStreamCalls {
				for got, err := range mockModel.GenerateStream(ctx, &model.LLMRequest{}) {
					if err != nil {
						t.Fatalf("found error while iterating stream")
					}
					if count >= len(tc.want) {
						t.Fatalf("stream generated more values than the expected %d", len(tc.want))
					}
					if diff := cmp.Diff(tc.want[count], got.Content); diff != "" {
						t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.Content, tc.want[count], diff)
					}
					if got.Partial != tc.wantPartial[count] {
						t.Errorf("Model.GenerateStream() = %v, want Partial value %v\n", got.Partial, tc.wantPartial[count])
					}
					count++
				}
				callCount++
			}
			if count != len(tc.want) {
				t.Errorf("unexpected stream length, expected %d got %d", len(tc.want), count)
			}
		})
	}
}
