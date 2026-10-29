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
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// BenchmarkBuildContentsDefault is prompt assembly on its own: the per-turn
// pass over the whole history that every LLM call pays.
func BenchmarkBuildContentsDefault(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			events := benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0))
			user := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := buildContentsDefault("agent", "", "", events, false, user); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkBuildContentsInlineData is prompt assembly over a short history
// that holds one image, the shape of a multimodal chat. The image stays in
// the history until compaction removes it, so this is paid on every later
// model call.
func BenchmarkBuildContentsInlineData(b *testing.B) {
	for _, kb := range []int{100, 1024} {
		b.Run(fmt.Sprintf("image=%dKB", kb), func(b *testing.B) {
			events := benchutil.BenchEvents(10, "agent", time.Unix(1_700_000_000, 0))
			events[0] = &session.Event{
				ID: "img", InvocationID: "inv-img", Author: "user", Timestamp: events[0].Timestamp,
				LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
					{Text: "what is this?"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, kb<<10)}},
				}}},
			}
			user := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := buildContentsDefault("agent", "", "", events, false, user); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
