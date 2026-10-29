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

package runner_test

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// streamModel answers every request with one text turn delivered as `chunks`
// text chunks of chunkBytes bytes each, fed through the real streaming
// aggregator the way model/gemini does, so the partial events reach the
// runner exactly as a streamed Gemini answer would.
type streamModel struct {
	chunks     int
	chunkBytes int
}

func (m *streamModel) Name() string { return "stream" }

func (m *streamModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	chunk := strings.Repeat("x", m.chunkBytes)
	return func(yield func(*model.LLMResponse, error) bool) {
		aggregator := llminternal.NewStreamingResponseAggregator()
		for i := range m.chunks {
			var finishReason genai.FinishReason
			if i == m.chunks-1 {
				finishReason = genai.FinishReasonStop
			}
			resp := &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
				Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: chunk}}},
				FinishReason: finishReason,
			}}}
			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

type benchToolArgs struct {
	Query string `json:"query"`
}

type benchToolResult struct {
	Answer string `json:"answer"`
}

// benchTools returns n function tools, the declarations every model call
// packs into the request and the map runOneStep rebuilds per response.
func benchTools(b *testing.B, n int) []tool.Tool {
	b.Helper()
	tools := make([]tool.Tool, 0, n)
	for i := range n {
		t, err := functiontool.New(functiontool.Config{
			Name:        fmt.Sprintf("tool_%02d", i),
			Description: "a tool that is declared but never called",
		}, func(agent.Context, benchToolArgs) (benchToolResult, error) {
			return benchToolResult{}, nil
		})
		if err != nil {
			b.Fatal(err)
		}
		tools = append(tools, t)
	}
	return tools
}

// BenchmarkRunTurnStreaming is one SSE turn over a 12-event history: the
// model streams its answer in chunks and every chunk is a partial event that
// runOneStep finalizes and the runner forwards. Allocations and bytes scale
// with the chunk count (and, with +=, with the square of it), so the
// per-chunk cost is the difference between the two chunk counts.
func BenchmarkRunTurnStreaming(b *testing.B) {
	for _, tc := range []struct{ chunks, tools int }{
		{200, 0},
		{200, 20},
		{2000, 20},
	} {
		b.Run(fmt.Sprintf("chunks=%d/tools=%d", tc.chunks, tc.tools), func(b *testing.B) {
			m := &streamModel{chunks: tc.chunks, chunkBytes: 50}
			msg := genai.NewContentFromText("hello", genai.RoleUser)
			var events int64
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				svc := session.InMemoryService()
				if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 12); err != nil {
					b.Fatal(err)
				}
				a, err := llmagent.New(llmagent.Config{Name: "agent", Model: m, Instruction: "You are terse.", Tools: benchTools(b, tc.tools)})
				if err != nil {
					b.Fatal(err)
				}
				r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
				if err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				for ev, err := range r.Run(b.Context(), "user", "s", msg, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
					if err != nil {
						b.Fatal(err)
					}
					if ev.Partial {
						events++
					}
				}
			}
			b.ReportMetric(float64(events)/float64(b.N), "partial-events/turn")
		})
	}
}
