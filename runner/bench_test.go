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
	"testing"

	"github.com/glebarez/sqlite"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

// echoModel answers every request with one fixed text turn and records how
// many contents the last request carried, so a benchmark can confirm the
// whole history reached the model.
type echoModel struct {
	lastContents int
}

func (m *echoModel) Name() string { return "echo" }

func (m *echoModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastContents = len(req.Contents)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(benchutil.BenchText, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}, nil)
	}
}

func benchAgent(b *testing.B, m model.LLM) agent.Agent {
	b.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: m, Instruction: "You are terse."})
	if err != nil {
		b.Fatal(err)
	}
	return a
}

func sqliteService(b *testing.B) session.Service {
	b.Helper()
	db, err := gorm.Open(sqlite.Open(b.TempDir()+"/bench.db"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		b.Fatal(err)
	}
	svc, err := database.NewSessionServiceFromDB(db)
	if err != nil {
		b.Fatal(err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		b.Fatal(err)
	}
	return svc
}

// benchRunTurn drives one full turn: load the session, assemble the prompt
// from n prior events, one model call, persist the user and model events.
// Every iteration gets a freshly seeded session, outside the timer, so the
// history size is exactly n rather than growing with the iteration count.
// It reports session-service traffic per turn alongside time and allocations.
func benchRunTurn(b *testing.B, backend string, newService func(*testing.B) session.Service) {
	for _, n := range []int{0, 100, 1000} {
		b.Run(fmt.Sprintf("%s/events=%d", backend, n), func(b *testing.B) {
			benchRunTurnN(b, n, newService)
		})
	}
}

func benchRunTurnN(b *testing.B, n int, newService func(*testing.B) session.Service) {
	m := &echoModel{}
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	var gets, appends, loaded int64
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		svc := &benchutil.CountingSessionService{Service: newService(b)}
		if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
			b.Fatal(err)
		}
		r, err := runner.New(runner.Config{AppName: "app", Agent: benchAgent(b, m), SessionService: svc})
		if err != nil {
			b.Fatal(err)
		}
		svc.Reset()
		b.StartTimer()
		for _, err := range r.Run(b.Context(), "user", "s", msg, agent.RunConfig{}) {
			if err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		gets += svc.Gets.Load()
		appends += svc.Appends.Load()
		loaded += svc.EventsLoaded.Load()
		b.StartTimer()
	}
	b.ReportMetric(float64(gets)/float64(b.N), "gets/turn")
	b.ReportMetric(float64(appends)/float64(b.N), "appends/turn")
	b.ReportMetric(float64(loaded)/float64(b.N), "events-loaded/turn")
	b.ReportMetric(float64(m.lastContents), "contents-in-prompt")
}

func BenchmarkRunTurn(b *testing.B) {
	benchRunTurn(b, "inmemory", func(*testing.B) session.Service { return session.InMemoryService() })
	benchRunTurn(b, "sqlite", sqliteService)
}

// BenchmarkRunTurnCapture is BenchmarkRunTurn/inmemory/events=1000 under
// message-content capture: off; EVENT_ONLY with no LoggerProvider, which is
// what `adkgo deploy agentengine` configures when no OTLP endpoint is set;
// and EVENT_ONLY and SPAN_AND_EVENT consumed by SDK providers over counting
// exporters. It reports the spans and log records those providers received.
func BenchmarkRunTurnCapture(b *testing.B) {
	for _, tc := range []struct {
		name           string
		capture        string
		tracer, logger bool
	}{
		{name: "off"},
		{name: "event/no-provider", capture: "EVENT_ONLY"},
		{name: "event/sdk-logger", capture: "EVENT_ONLY", logger: true},
		{name: "span_and_event/sdk-both", capture: "SPAN_AND_EVENT", tracer: true, logger: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			// Runs after b.Setenv restores the environment.
			b.Cleanup(telemetry.ApplyEnv)
			b.Setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", tc.capture)
			telemetry.ApplyEnv()
			spans := &benchutil.CountingSpanExporter{}
			records := &benchutil.CountingLogExporter{}
			if tc.tracer {
				telemetry.OverrideTracerForTesting(b, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(spans))))
			}
			if tc.logger {
				telemetry.OverrideLoggerForTesting(b, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(records))))
			}
			benchRunTurnN(b, 1000, func(*testing.B) session.Service { return session.InMemoryService() })
			b.ReportMetric(float64(spans.Spans.Load())/float64(b.N), "spans/turn")
			b.ReportMetric(float64(records.Records.Load())/float64(b.N), "records/turn")
		})
	}
}

// BenchmarkRunnerNew measures constructing a Runner, which the REST server
// does once per request rather than once per process.
func BenchmarkRunnerNew(b *testing.B) {
	m := &echoModel{}
	a := benchAgent(b, m)
	svc := session.InMemoryService()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc}); err != nil {
			b.Fatal(err)
		}
	}
}
