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

package runner_test

import (
	"context"
	"fmt"
	"net/http"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

func toolAgent(tb testing.TB, m model.LLM) agent.Agent {
	tb.Helper()
	tools, err := benchutil.Tools()
	if err != nil {
		tb.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: m, Instruction: benchutil.ToolInstruction, Tools: tools})
	if err != nil {
		tb.Fatal(err)
	}
	return a
}

var toolTurnSizes = []int{0, 100, 1000}

// BenchmarkToolTurn is one realistic turn through runner.Run on the
// in-memory service: an LlmAgent with three function tools gets a question,
// the model asks for a tool, the tool runs, and the model answers, so the
// turn makes two model calls over a history of n prior events and persists
// four events. Every iteration gets a freshly seeded session outside the
// timer, so the history is exactly n events long.
func BenchmarkToolTurn(b *testing.B) {
	for _, n := range toolTurnSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			benchToolTurn(b, n, runner.Config{})
		})
	}
}

func benchToolTurn(b *testing.B, n int, cfg runner.Config) {
	m := &benchutil.ToolCallModel{}
	a := toolAgent(b, m)
	history := benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0))
	msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
	events := 0
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		svc := session.InMemoryService()
		if _, err := benchutil.SeedSessionEvents(b.Context(), svc, "app", "user", "s", benchutil.ToolState(), history); err != nil {
			b.Fatal(err)
		}
		cfg := cfg
		cfg.AppName, cfg.Agent, cfg.SessionService = "app", a, svc
		r, err := runner.New(cfg)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		for _, err := range r.Run(b.Context(), "user", "s", msg, agent.RunConfig{}) {
			if err != nil {
				b.Fatal(err)
			}
			events++
		}
	}
	b.StopTimer()
	if got, want := m.Calls.Load(), 2*int64(b.N); got != want {
		b.Fatalf("model calls = %d, want %d", got, want)
	}
	b.ReportMetric(float64(events)/float64(b.N), "events/turn")
	b.ReportMetric(float64(m.LastContents.Load()), "contents-in-prompt")
}

// presetSessions seeds count sessions holding the same n history events into
// one in-memory service, so a benchmark can run each turn on a fresh session
// without stopping the timer.
func presetSessions(b *testing.B, count, n int) session.Service {
	b.Helper()
	svc := session.InMemoryService()
	history := benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0))
	for i := range count {
		if _, err := benchutil.SeedSessionEvents(b.Context(), svc, "app", "user", fmt.Sprintf("s%d", i), benchutil.ToolState(), history); err != nil {
			b.Fatal(err)
		}
	}
	return svc
}

// BenchmarkToolTurnSteady is BenchmarkToolTurn with every session seeded up
// front, so the loop runs back to back turns with no timer stops, each of
// which would read memory statistics with the world stopped. It is the
// sequential baseline for BenchmarkToolTurnParallel.
func BenchmarkToolTurnSteady(b *testing.B) {
	for _, n := range toolTurnSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			m := &benchutil.ToolCallModel{}
			svc := presetSessions(b, b.N, n)
			r, err := runner.New(runner.Config{AppName: "app", Agent: toolAgent(b, m), SessionService: svc})
			if err != nil {
				b.Fatal(err)
			}
			msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
			b.ReportAllocs()
			runtime.GC()
			b.ResetTimer()
			start := benchutil.SampleRuntime()
			for i := 0; i < b.N; i++ {
				for _, err := range r.Run(b.Context(), "user", fmt.Sprintf("s%d", i), msg, agent.RunConfig{}) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			benchutil.ReportRuntime(b, start)
			b.StopTimer()
			if got, want := m.Calls.Load(), 2*int64(b.N); got != want {
				b.Fatalf("model calls = %d, want %d", got, want)
			}
		})
	}
}

// BenchmarkToolTurnParallel runs BenchmarkToolTurnSteady's turns from
// GOMAXPROCS goroutines at once, each turn on its own session of one shared
// service and runner. ns/op is wall time per turn across all goroutines, so
// the ratio to BenchmarkToolTurnSteady is the speedup the cores deliver,
// which a shared bottleneck holds below the core count.
func BenchmarkToolTurnParallel(b *testing.B) {
	for _, n := range toolTurnSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			m := &benchutil.ToolCallModel{}
			svc := presetSessions(b, b.N, n)
			r, err := runner.New(runner.Config{AppName: "app", Agent: toolAgent(b, m), SessionService: svc})
			if err != nil {
				b.Fatal(err)
			}
			msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
			var next atomic.Int64
			b.ReportAllocs()
			runtime.GC()
			b.ResetTimer()
			start := benchutil.SampleRuntime()
			b.RunParallel(func(pb *testing.PB) {
				// A context per goroutine, as a server has one per request.
				// Turns sharing b.Context() would all lock its child list
				// and its Done channel, contention a server whose turns
				// derive from their own request contexts does not have.
				ctx, cancel := context.WithCancel(b.Context())
				defer cancel()
				for pb.Next() {
					id := fmt.Sprintf("s%d", next.Add(1)-1)
					for _, err := range r.Run(ctx, "user", id, msg, agent.RunConfig{}) {
						if err != nil {
							b.Error(err)
							return
						}
					}
				}
			})
			benchutil.ReportRuntime(b, start)
			b.StopTimer()
			if got, want := m.Calls.Load(), 2*int64(b.N); got != want {
				b.Fatalf("model calls = %d, want %d", got, want)
			}
		})
	}
}

// BenchmarkToolTurnGemini is BenchmarkToolTurnSteady through the real Gemini
// model over a transport that answers without a connection, so each turn
// also pays the client side of its two model calls: building and encoding
// the request with the whole history, decoding the response. Comparing it
// with BenchmarkToolTurnSteady shows the share of a served turn that is the
// model client rather than ADK.
func BenchmarkToolTurnGemini(b *testing.B) {
	for _, n := range toolTurnSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			m, err := gemini.NewModel(b.Context(), "gemini-2.5-flash", &genai.ClientConfig{
				APIKey:     "bench-key",
				Backend:    genai.BackendGeminiAPI,
				HTTPClient: &http.Client{Transport: &benchutil.GeminiTransport{Bodies: benchutil.GeminiToolTurn}},
			})
			if err != nil {
				b.Fatal(err)
			}
			svc := presetSessions(b, b.N, n)
			r, err := runner.New(runner.Config{AppName: "app", Agent: toolAgent(b, m), SessionService: svc})
			if err != nil {
				b.Fatal(err)
			}
			msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
			b.ReportAllocs()
			runtime.GC()
			b.ResetTimer()
			start := benchutil.SampleRuntime()
			for i := 0; i < b.N; i++ {
				events := 0
				for _, err := range r.Run(b.Context(), "user", fmt.Sprintf("s%d", i), msg, agent.RunConfig{}) {
					if err != nil {
						b.Fatal(err)
					}
					events++
				}
				if events != 3 {
					b.Fatalf("turn %d yielded %d events, want 3 (call, response, answer)", i, events)
				}
			}
			benchutil.ReportRuntime(b, start)
		})
	}
}
