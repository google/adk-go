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

package adka2a

import (
	"fmt"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func benchmark5KBFunctionResponsePart() *genai.Part {
	items := make([]any, 20)
	for i := range items {
		items[i] = map[string]any{
			"id":      fmt.Sprintf("item-%02d", i),
			"title":   fmt.Sprintf("Document title %02d", i),
			"snippet": strings.Repeat("structured response payload content ", 6),
			"score":   float64(i) * 0.05,
		}
	}
	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:         "call-bench-1",
			Name:       "search_documents",
			Scheduling: genai.FunctionResponseSchedulingWhenIdle,
			Response: map[string]any{
				"status": "ok",
				"items":  items,
			},
		},
	}
}

// BenchmarkToA2APartsFunctionResponse measures converting a ~5 KB GenAI
// FunctionResponse part into an A2A DataPart.
func BenchmarkToA2APartsFunctionResponse(b *testing.B) {
	parts := []*genai.Part{benchmark5KBFunctionResponsePart()}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := ToA2AParts(parts, nil)
		if err != nil {
			b.Fatalf("ToA2AParts failed: %v", err)
		}
		if len(out) != 1 {
			b.Fatalf("len(out) = %d, want 1", len(out))
		}
	}
}

// BenchmarkToGenAIPartsFunctionResponse measures converting a ~5 KB A2A
// FunctionResponse DataPart into a GenAI FunctionResponse part.
func BenchmarkToGenAIPartsFunctionResponse(b *testing.B) {
	a2aParts, err := ToA2AParts([]*genai.Part{benchmark5KBFunctionResponsePart()}, nil)
	if err != nil {
		b.Fatalf("ToA2AParts setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := ToGenAIParts(a2aParts)
		if err != nil {
			b.Fatalf("ToGenAIParts failed: %v", err)
		}
		if len(out) != 1 {
			b.Fatalf("len(out) = %d, want 1", len(out))
		}
	}
}

// BenchmarkUsageMetadataRoundTrip measures converting UsageMetadata to A2A
// event metadata and back.
func BenchmarkUsageMetadataRoundTrip(b *testing.B) {
	srcEvent := &session.Event{
		LLMResponse: model.LLMResponse{
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        512,
				CandidatesTokenCount:    128,
				ThoughtsTokenCount:      64,
				ToolUsePromptTokenCount: 32,
				TotalTokenCount:         736,
				CacheTokensDetails: []*genai.ModalityTokenCount{
					{Modality: genai.MediaModalityText, TokenCount: 256},
				},
			},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		meta, err := toEventMeta(invocationMeta{}, srcEvent)
		if err != nil {
			b.Fatalf("toEventMeta failed: %v", err)
		}
		var dst session.Event
		if err := processA2AMeta(&a2a.TaskStatusUpdateEvent{Metadata: meta}, &dst); err != nil {
			b.Fatalf("processA2AMeta failed: %v", err)
		}
	}
}
