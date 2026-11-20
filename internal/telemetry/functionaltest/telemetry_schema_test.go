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
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcaseimpl"
	"google.golang.org/adk/v2/internal/telemetry/telemetrytest"
)

var update = flag.Bool("update", false, "re-record testdata/<test_id>.json from this run")

const (
	captureContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
	// adkTelemetrySchemaVersionOptIn selects the telemetry format. Every case is
	// recorded under each value, including before ADK reads the variable, so
	// the change that starts reading it shows which telemetry changed and
	// that the legacy format did not.
	adkTelemetrySchemaVersionOptIn = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"
)

func goldenPath(c testcasedata.Case) string {
	return filepath.Join("testdata", c.ID()+".json")
}

// TestTelemetrySchema drives each case end to end and holds the spans and log
// records it emits to the recorded golden.
func TestTelemetrySchema(t *testing.T) {
	for _, tc := range testcaseimpl.Cases() {
		t.Run(tc.ID(), func(t *testing.T) {
			got := record(t, tc)
			checkInferenceErrorTypes(t, tc, got)
			if *update {
				writeGolden(t, goldenPath(tc), got)
				return
			}
			want := readGolden(t, goldenPath(tc))
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("telemetry differs from %s (-want +got):\n%s\nIf the change is intended, re-record with `go test ./internal/telemetry/functionaltest -update`.", goldenPath(tc), diff)
			}
		})
	}
}

// TestGoldensHaveCases fails on a golden no case records any more, so a
// renamed or dropped case cannot leave a stale file behind.
func TestGoldensHaveCases(t *testing.T) {
	want := map[string]bool{}
	for _, tc := range testcaseimpl.Cases() {
		want[goldenPath(tc)] = true
	}
	files, err := filepath.Glob(filepath.Join("testdata", "*", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if !want[f] {
			t.Errorf("%s is recorded by no case; delete it", f)
		}
	}
}

// checkInferenceErrorTypes fails the test when a recording's generate_content
// spans and its client inference metrics disagree about failure. Under either
// schema, the spans with ERROR status must match the metric points with an
// error type: some of each, or none of either. Under the default schema, where
// spans carry an error type too, the two must also report the same set of
// error types, counting no error type as one. A metric data point aggregates
// calls, so the check compares sets rather than calls. It is a narrower form
// of adk-python's tests/unittests/telemetry/test_semconv_conformance.py, which
// checks every attribute a span and its metrics share.
func checkInferenceErrorTypes(t *testing.T, tc testcasedata.Case, d *telemetrytest.Digest) {
	t.Helper()
	spanFailed, spanTypes := map[bool]bool{}, map[any]bool{}
	var walk func(*telemetrytest.SpanDigest)
	walk = func(s *telemetrytest.SpanDigest) {
		if s.Attributes["gen_ai.operation.name"] == "generate_content" {
			spanFailed[s.Status == "ERROR"] = true
			spanTypes[s.Attributes["error.type"]] = true
		}
		for _, c := range s.Children {
			walk(c)
		}
	}
	walk(d.RootSpan)
	legacy := tc.SchemaVersion == "otel_semconv_1_36"
	for _, name := range []string{
		"gen_ai.client.inference.duration",
		"gen_ai.client.inference.operation.input_tokens",
		"gen_ai.client.inference.operation.output_tokens",
	} {
		pointFailed, pointTypes := map[bool]bool{}, map[any]bool{}
		for _, p := range d.MetricPoints[name] {
			errorType, ok := p.Attributes["error.type"]
			pointFailed[ok] = true
			pointTypes[errorType] = true
		}
		if diff := cmp.Diff(spanFailed, pointFailed); diff != "" {
			t.Errorf("failures of generate_content spans and %s differ (-spans +metric):\n%s", name, diff)
		}
		if diff := cmp.Diff(spanTypes, pointTypes); !legacy && diff != "" {
			t.Errorf("error types of generate_content spans and %s differ (-spans +metric):\n%s", name, diff)
		}
	}
}

// record runs tc and returns its digest, in the form a golden stores it.
func record(t *testing.T, tc testcasedata.Case) *telemetrytest.Digest {
	t.Helper()
	// Registered before t.Setenv, so it runs after the environment is
	// restored and leaves the package state as the next test expects.
	t.Cleanup(telemetry.ApplyEnv)
	t.Setenv(captureContentEnvVar, string(tc.Capture))
	t.Setenv(adkTelemetrySchemaVersionOptIn, string(tc.SchemaVersion))
	telemetry.ApplyEnv()

	spanExp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))
	logExp := telemetrytest.NewInMemoryLogExporter()
	telemetry.OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExp))))
	metricReader := sdkmetric.NewManualReader()
	telemetry.OverrideMeterForTesting(t, sdkmetric.NewMeterProvider(sdkmetric.WithReader(metricReader)))
	// The mock model names no provider, so spans, events and metrics guess one
	// from these.
	t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")

	err := testcaseimpl.Run(t, tc.Scenario)
	if gotErr := err != nil; gotErr != tc.Scenario.WantErr() {
		t.Fatalf("run error = %v, want error: %t", err, tc.Scenario.WantErr())
	}

	var metrics metricdata.ResourceMetrics
	if err := metricReader.Collect(t.Context(), &metrics); err != nil {
		t.Fatalf("collect metrics: %v", err)
	}
	d, err := telemetrytest.BuildDigest(spanExp.GetSpans(), logExp.Records(), metrics)
	if err != nil {
		t.Fatalf("build digest: %v", err)
	}
	// Round-tripped through JSON so the recording holds the same types a
	// golden read back from disk does.
	return roundTrip(t, d)
}

func roundTrip(t *testing.T, d *telemetrytest.Digest) *telemetrytest.Digest {
	t.Helper()
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal digest: %v", err)
	}
	var out telemetrytest.Digest
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal digest: %v", err)
	}
	return &out
}

func readGolden(t *testing.T, path string) *telemetrytest.Digest {
	t.Helper()
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing golden %s; record it with `go test ./internal/telemetry/functionaltest -update`", path)
	}
	if err != nil {
		t.Fatal(err)
	}
	var d telemetrytest.Digest
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return &d
}

func writeGolden(t *testing.T, path string, d *telemetrytest.Digest) {
	t.Helper()
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(d); err != nil {
		t.Fatalf("marshal digest: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
