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

package telemetry

import (
	"context"
	"iter"
	"testing"

	sdklog "go.opentelemetry.io/otel/sdk/log"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"

	"google.golang.org/adk/v2/model"
)

// TestSchemaSwitchAfterFirstUse checks that the cached tracer and logger
// follow a schema change made after they were first used, unlike
// TestSchemaURL, where the env is set before the provider is installed.
func TestSchemaSwitchAfterFirstUse(t *testing.T) {
	spans := setupTestTracer(t)
	logs := &inMemoryExporter{}
	OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(logs))))
	generate := func() {
		call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
			return func(yield func(*model.LLMResponse, error) bool) { yield(&model.LLMResponse{}, nil) }
		}
		params := GenerateContentParams{ModelName: "m", Request: &model.LLMRequest{}}
		for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
		}
	}

	setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: "otel_semconv_1_44"})
	generate()
	setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: otelSemconv136})
	generate()

	got := spans.GetSpans()
	if len(got) != 2 {
		t.Fatalf("got %d spans, want 2", len(got))
	}
	if url := got[1].InstrumentationScope.SchemaURL; url != semconv.SchemaURL {
		t.Errorf("span after the switch has schema URL %q, want %q", url, semconv.SchemaURL)
	}
	if len(logs.records) == 0 {
		t.Fatal("no log records emitted")
	}
	// The 1.44 schema emits one operation-details record per call and the
	// 1.36 schema its message records, so the last record is from the second
	// call.
	if url := logs.records[len(logs.records)-1].InstrumentationScope().SchemaURL; url != semconv.SchemaURL {
		t.Errorf("log record after the switch has schema URL %q, want %q", url, semconv.SchemaURL)
	}
}
