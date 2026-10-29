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

package method

import (
	"context"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"os"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// benchChunkText is one streamed chunk, ~40 bytes.
const benchChunkText = "The quick brown fox jumps over the lazy dog. "

// BenchmarkStreamQueryHandler measures one async_stream_query request whose
// model streams 100 chunks, exercising the handler loop once per partial event
// plus the final event.
func BenchmarkStreamQueryHandler(b *testing.B) {
	const chunks = 100
	a, err := llmagent.New(llmagent.Config{Name: "chunky", Model: &chunkingModel{n: chunks}, Instruction: "You are terse."})
	if err != nil {
		b.Fatal(err)
	}
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		b.Fatal(err)
	}
	defer devnull.Close()
	prev := log.Writer()
	defer log.SetOutput(prev)

	for _, sink := range []struct {
		name string
		w    io.Writer
	}{
		{"log=discard", io.Discard},
		{"log=devnull", devnull},
	} {
		b.Run(fmt.Sprintf("events=%d/%s", chunks+1, sink.name), func(b *testing.B) {
			log.SetOutput(sink.w)
			config := &launcher.Config{AgentLoader: agent.NewSingleLoader(a), SessionService: session.InMemoryService()}
			h := NewStreamQueryHandler(config, "app", "async_stream_query", "")
			rw := &discardFlusher{h: http.Header{}}
			b.ReportAllocs()
			for i := 0; b.Loop(); i++ {
				payload := fmt.Sprintf(`{"class_method":"async_stream_query","input":{"message":"hi","session_id":"s-%d","user_id":"u"}}`, i)
				if err := h.streamJSONL(b.Context(), rw, []byte(payload)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

type chunkingModel struct{ n int }

func (m *chunkingModel) Name() string { return "chunking" }

func (m *chunkingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for range m.n {
			if !yield(&model.LLMResponse{
				Content: genai.NewContentFromText(benchChunkText, genai.RoleModel),
				Partial: true,
			}, nil) {
				return
			}
		}
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(strings.Repeat(benchChunkText, m.n), genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}, nil)
	}
}

type discardFlusher struct{ h http.Header }

func (d *discardFlusher) Header() http.Header         { return d.h }
func (d *discardFlusher) Write(p []byte) (int, error) { return len(p), nil }
func (d *discardFlusher) WriteHeader(int)             {}
func (d *discardFlusher) Flush()                      {}
