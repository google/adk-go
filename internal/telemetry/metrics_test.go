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
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"google.golang.org/adk/v2/internal/version"
)

func TestInstruments(t *testing.T) {
	reader := setupTestMeter(t)
	ctx := t.Context()
	inst.inferenceDuration.Record(ctx, 1)
	inst.inferenceInputTokens.Record(ctx, 1)
	inst.inferenceOutputTokens.Record(ctx, 1)
	inst.usageInputTokens.Add(ctx, 1)
	inst.usageOutputTokens.Add(ctx, 1)
	inst.usageCacheReadInputTokens.Add(ctx, 1)
	inst.usageCacheWriteInputTokens.Add(ctx, 1)
	inst.usageReasoningOutputTokens.Add(ctx, 1)
	inst.timeToFirstChunk.Record(ctx, 1)
	inst.timePerOutputChunk.Record(ctx, 1)

	type instrument struct {
		Kind, Unit, Description string
		Bounds                  []float64
	}
	seconds := []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	tokens := []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
	// Descriptions from https://github.com/open-telemetry/semantic-conventions-genai/blob/95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8/docs/gen-ai/client-inference.md
	// and gen-ai-token-metrics.md.
	want := map[string]instrument{
		"gen_ai.client.inference.duration":                       {"float64 histogram", "s", "GenAI client inference operation duration.", seconds},
		"gen_ai.client.inference.operation.input_tokens":         {"int64 histogram", "{token}", "The number of input (prompt) tokens used per inference operation.", tokens},
		"gen_ai.client.inference.operation.output_tokens":        {"int64 histogram", "{token}", "The number of output (completion) tokens used per inference operation.", tokens},
		"gen_ai.client.inference.usage.input_tokens":             {"int64 counter", "{token}", "The number of input (prompt) tokens used, including cached tokens.", nil},
		"gen_ai.client.inference.usage.output_tokens":            {"int64 counter", "{token}", "The number of output (completion) tokens used, including reasoning tokens.", nil},
		"gen_ai.client.inference.usage.cache_read.input_tokens":  {"int64 counter", "{token}", "The number of input tokens served from a provider-managed cache.", nil},
		"gen_ai.client.inference.usage.cache_write.input_tokens": {"int64 counter", "{token}", "The number of input tokens written to a provider-managed cache.", nil},
		"gen_ai.client.inference.usage.reasoning.output_tokens":  {"int64 counter", "{token}", "The number of output tokens used for reasoning (e.g. chain-of-thought, extended thinking).", nil},
		"gen_ai.client.inference.time_to_first_chunk":            {"float64 histogram", "s", "Time from the start of a logical inference operation to the first chunk in its streaming response.", seconds},
		"gen_ai.client.inference.time_per_output_chunk":          {"float64 histogram", "s", "Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk.", seconds},
	}

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(ctx, &rm); err != nil {
		t.Fatal(err)
	}
	if len(rm.ScopeMetrics) != 1 {
		t.Fatalf("got %d scopes, want 1", len(rm.ScopeMetrics))
	}
	scope := rm.ScopeMetrics[0].Scope
	if scope.Name != "gcp.vertex.agent" || scope.Version != version.Version || scope.SchemaURL != "" {
		t.Errorf("scope = %+v, want gcp.vertex.agent at version %s without a schema URL", scope, version.Version)
	}
	got := map[string]instrument{}
	for _, m := range rm.ScopeMetrics[0].Metrics {
		i := instrument{Unit: m.Unit, Description: m.Description}
		switch d := m.Data.(type) {
		case metricdata.Histogram[float64]:
			i.Kind, i.Bounds = "float64 histogram", d.DataPoints[0].Bounds
		case metricdata.Histogram[int64]:
			i.Kind, i.Bounds = "int64 histogram", d.DataPoints[0].Bounds
		case metricdata.Sum[int64]:
			if d.IsMonotonic {
				i.Kind = "int64 counter"
			}
		}
		got[m.Name] = i
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("instruments mismatch (-want +got):\n%s", diff)
	}
}

// setupTestMeter makes metrics record into a fresh provider until the test
// ends, and returns the reader that collects them.
func setupTestMeter(t *testing.T) *sdkmetric.ManualReader {
	t.Helper()
	reader := sdkmetric.NewManualReader()
	OverrideMeterForTesting(t, sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader)))
	return reader
}

// metricPoint is one collected data point: a counter's total, or a
// histogram's sum and count.
type metricPoint struct {
	Attrs map[attribute.Key]string
	Value float64
	Count uint64
}

// collectMetrics returns every data point the reader holds, by metric name.
func collectMetrics(t *testing.T, reader *sdkmetric.ManualReader) map[string][]metricPoint {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.WithoutCancel(t.Context()), &rm); err != nil {
		t.Fatal(err)
	}
	out := map[string][]metricPoint{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			switch d := m.Data.(type) {
			case metricdata.Histogram[float64]:
				for _, p := range d.DataPoints {
					out[m.Name] = append(out[m.Name], metricPoint{attributesToMap(p.Attributes.ToSlice()), p.Sum, p.Count})
				}
			case metricdata.Histogram[int64]:
				for _, p := range d.DataPoints {
					out[m.Name] = append(out[m.Name], metricPoint{attributesToMap(p.Attributes.ToSlice()), float64(p.Sum), p.Count})
				}
			case metricdata.Sum[int64]:
				for _, p := range d.DataPoints {
					out[m.Name] = append(out[m.Name], metricPoint{attributesToMap(p.Attributes.ToSlice()), float64(p.Value), 0})
				}
			default:
				t.Fatalf("%s: unexpected data type %T", m.Name, m.Data)
			}
		}
	}
	return out
}
