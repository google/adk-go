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
	"context"
	"fmt"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/benchutil"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
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
		b.Run(fmt.Sprintf("well-formed/events=%d", n), func(b *testing.B) {
			events := benchutil.BenchEventsWellFormed(n, "agent", time.Unix(1_700_000_000, 0))
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

type benchNoopSummarizer struct{}

func (benchNoopSummarizer) SummarizeEvents(_ context.Context, _ []*session.Event) (compaction.SummarizeResult, error) {
	return compaction.SummarizeResult{}, nil
}

type benchPackingTool struct {
	name string
}

func (t *benchPackingTool) Name() string        { return t.name }
func (t *benchPackingTool) Description() string { return "benchmark tool" }
func (t *benchPackingTool) IsLongRunning() bool { return false }
func (t *benchPackingTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.name, Description: "benchmark tool"}
}
func (t *benchPackingTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func newBenchLLMAgent(b *testing.B, s *State) *mockLLMAgent {
	b.Helper()
	base, err := agent.New(agent.Config{Name: "agent"})
	if err != nil {
		b.Fatal(err)
	}
	return &mockLLMAgent{Agent: base, s: s}
}

// BenchmarkCompactionRequestProcessor measures the per-model-call cost of
// CompactionRequestProcessor when TokenThreshold > 0 and the prompt is still
// under the threshold.
func BenchmarkCompactionRequestProcessor(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := session.InMemoryService()
			events := benchutil.BenchEventsWellFormed(n, "agent", time.Unix(1_700_000_000, 0))
			sess, err := benchutil.SeedSessionEvents(b.Context(), svc, "app", "user", "sess", events)
			if err != nil {
				b.Fatal(err)
			}
			cfg := &compaction.Config{
				TokenThreshold:     1_000_000,
				EventRetentionSize: 10,
				Summarizer:         benchNoopSummarizer{},
			}
			ctx := compactionctx.ToContext(b.Context(), compactionctx.New(cfg, svc))
			ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Agent:   newBenchLLMAgent(b, &State{}),
				Session: sess,
			})
			req := &model.LLMRequest{}
			f := &Flow{}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				for _, err := range CompactionRequestProcessor(ictx, req, f) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// BenchmarkRequestConfirmationRequestProcessor measures the per-model-call cost
// of RequestConfirmationRequestProcessor when the latest user turn carries no
// confirmation response.
func BenchmarkRequestConfirmationRequestProcessor(b *testing.B) {
	tools := make([]tool.Tool, 20)
	for i := range tools {
		tools[i] = &benchPackingTool{name: fmt.Sprintf("tool_%02d", i)}
	}
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := session.InMemoryService()
			events := benchutil.BenchEventsWellFormed(n, "agent", time.Unix(1_700_000_000, 0))
			sess, err := benchutil.SeedSessionEvents(b.Context(), svc, "app", "user", "sess", events)
			if err != nil {
				b.Fatal(err)
			}
			ictx := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{
				Agent:   newBenchLLMAgent(b, &State{Tools: tools}),
				Session: sess,
			})
			req := &model.LLMRequest{}
			f := &Flow{Tools: tools}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				for _, err := range RequestConfirmationRequestProcessor(ictx, req, f) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// BenchmarkToolPreprocess measures packing 20 tools into an LLMRequest via
// toolPreprocess + toolutils.PackTool.
func BenchmarkToolPreprocess(b *testing.B) {
	tools := make([]tool.Tool, 20)
	for i := range tools {
		tools[i] = &benchPackingTool{name: fmt.Sprintf("tool_%02d", i)}
	}
	ictx := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{
		Agent: newBenchLLMAgent(b, &State{Tools: tools}),
	})
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var req model.LLMRequest
		if err := toolPreprocess(ictx, &req, tools); err != nil {
			b.Fatal(err)
		}
	}
}
