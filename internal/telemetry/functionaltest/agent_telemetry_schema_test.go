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

package functionaltest_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/internal/telemetry/telemetrytest"
	"google.golang.org/adk/internal/telemetry/telemetrytestcase"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// captureMessageContentEnvVar is the OpenTelemetry-spec env var
// that controls whether ADK emits full message content in log
// records or elides it for privacy.
const captureMessageContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

// TestTelemetrySchema_AgentWithTool runs the canonical
// "llmagent with one FunctionTool" scenario end-to-end and asserts
// the emitted span+log tree matches the expected shape exactly.
// Parametrized over the
// OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT env var:
// the "elided" subtest expects the production default (no message
// content in log bodies); the "capture_content" subtest expects
// the full decoded JSON content. Hermetic: no network calls, no
// GCP, no live model.
//
// Mirrors test_telemetry_schema in
// adk-python/tests/unittests/telemetry/test_functional.py.
func TestTelemetrySchema_AgentWithTool(t *testing.T) {
	tests := []struct {
		name           string
		captureContent bool
		want           *telemetrytest.SpanDigest
	}{
		{
			name:           "elided",
			captureContent: false,
			want:           telemetrytestcase.AgentWithToolCase,
		},
		{
			name:           "capture_content",
			captureContent: true,
			want:           telemetrytestcase.AgentWithToolCaptureContentCase,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.captureContent {
				t.Setenv(captureMessageContentEnvVar, "true")
			} else {
				t.Setenv(captureMessageContentEnvVar, "")
			}
			telemetry.ApplyEnv()

			// Install in-memory tracer + logger so the test sees
			// every span/log without depending on global OTel state.
			spanExp := tracetest.NewInMemoryExporter()
			telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))

			logExp := telemetrytest.NewInMemoryLogExporter()
			telemetry.OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(
				sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExp)),
			))

			rootAgent := newAgentWithToolScenario(t)
			if err := telemetrytest.RunScenario(t, rootAgent, "hello"); err != nil {
				t.Fatalf("RunScenario: %v", err)
			}

			got := telemetrytest.BuildDigests(t, spanExp.GetSpans(), logExp.Records())
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("telemetry mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// newAgentWithToolScenario constructs the canonical llmagent +
// 1 FunctionTool + 2 LLM turns scenario. The first turn returns a
// function call to the tool; the second turn returns the final
// text response. The expected telemetry shape for this scenario
// is declared in telemetrytestcase.AgentWithToolCase (and its
// capture-content variant).
func newAgentWithToolScenario(t *testing.T) agent.Agent {
	t.Helper()
	mockModel := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "some_tool",
						Args: map[string]any{"arg1": "val1"},
					},
				}},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "text response"}},
			},
		},
	}

	type Args struct {
		Arg1 string `json:"arg1"`
	}
	sampleTool, err := functiontool.New(functiontool.Config{
		Name:        "some_tool",
		Description: "A sample tool.",
	}, func(_ agent.Context, in Args) (string, error) {
		return "processed " + in.Arg1, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "some_root_agent",
		Description: "A sample root agent.",
		Model:       mockModel,
		Instruction: "you are helpful",
		Tools:       []tool.Tool{sampleTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return rootAgent
}
