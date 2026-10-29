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

package api

import (
	"context"
	"iter"
	"testing"

	"github.com/gorilla/mux"
	lognoop "go.opentelemetry.io/otel/log/noop"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/internal/benchutil"
	itel "google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/telemetry"
)

type benchEchoModel struct{}

func (benchEchoModel) Name() string { return "echo" }

func (benchEchoModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(benchutil.BenchText, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}, nil)
	}
}

// BenchmarkRunTurnDefaultAPILauncher runs a 100-event turn with the telemetry
// providers that apiLauncher.SetupSubrouters configures when -include_debug_api
// is left at its default (false).
func BenchmarkRunTurnDefaultAPILauncher(b *testing.B) {
	for _, tc := range []struct {
		name    string
		capture string
	}{
		{name: "capture=off", capture: ""},
		{name: "capture=event_only", capture: "EVENT_ONLY"},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.Cleanup(itel.ApplyEnv)
			b.Setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", tc.capture)
			itel.ApplyEnv()

			a, err := llmagent.New(llmagent.Config{Name: "agent", Model: benchEchoModel{}, Instruction: "You are terse."})
			if err != nil {
				b.Fatal(err)
			}
			l := NewLauncher()
			if _, err := l.Parse(nil); err != nil {
				b.Fatal(err)
			}
			cfg := &launcher.Config{
				AgentLoader:    agent.NewSingleLoader(a),
				SessionService: session.InMemoryService(),
			}
			if err := l.SetupSubrouters(mux.NewRouter().StrictSlash(true), cfg); err != nil {
				b.Fatal(err)
			}

			spans := &benchutil.CountingSpanExporter{}
			if len(cfg.TelemetryOptions) > 0 {
				providers, err := telemetry.New(b.Context(), cfg.TelemetryOptions...)
				if err != nil {
					b.Fatal(err)
				}
				b.Cleanup(func() { _ = providers.Shutdown(context.Background()) })
				if providers.TracerProvider != nil {
					providers.TracerProvider.RegisterSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))
					itel.OverrideTracerForTesting(b, providers.TracerProvider)
				} else {
					itel.OverrideTracerForTesting(b, tracenoop.NewTracerProvider())
				}
				if providers.LoggerProvider != nil {
					itel.OverrideLoggerForTesting(b, providers.LoggerProvider)
				} else {
					itel.OverrideLoggerForTesting(b, lognoop.NewLoggerProvider())
				}
			} else {
				itel.OverrideTracerForTesting(b, tracenoop.NewTracerProvider())
				itel.OverrideLoggerForTesting(b, lognoop.NewLoggerProvider())
			}

			msg := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				svc := session.InMemoryService()
				if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 100); err != nil {
					b.Fatal(err)
				}
				r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
				if err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				for _, err := range r.Run(b.Context(), "user", "s", msg, agent.RunConfig{}) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(spans.Spans.Load())/float64(b.N), "spans-recorded/turn")
		})
	}
}
