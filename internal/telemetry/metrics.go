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
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"

	"google.golang.org/adk/v2/internal/version"
)

var (
	// inferenceDurationBuckets are in seconds. The chunk timings use them too.
	inferenceDurationBuckets = []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	// inferenceTokenBuckets are in tokens.
	inferenceTokenBuckets = []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
)

// The global provider at package initialization. Instruments it creates
// forward to whichever provider the application installs later, so this works
// before telemetry setup runs.
var meterProvider = otel.GetMeterProvider()

// inst holds every instrument ADK records into.
var inst = newInstruments(meterProvider)

// instruments has one field per metric.
type instruments struct {
	inferenceDuration          metric.Float64Histogram
	inferenceInputTokens       metric.Int64Histogram
	inferenceOutputTokens      metric.Int64Histogram
	usageInputTokens           metric.Int64Counter
	usageOutputTokens          metric.Int64Counter
	usageCacheReadInputTokens  metric.Int64Counter
	usageCacheWriteInputTokens metric.Int64Counter
	usageReasoningOutputTokens metric.Int64Counter
	timeToFirstChunk           metric.Float64Histogram
	timePerOutputChunk         metric.Float64Histogram
}

// newInstruments creates every instrument on one meter of the given provider.
func newInstruments(mp metric.MeterProvider) instruments {
	m := mp.Meter(systemName, metric.WithInstrumentationVersion(version.Version))
	return instruments{
		inferenceDuration: created(m.Float64Histogram("gen_ai.client.inference.duration",
			metric.WithUnit("s"),
			metric.WithDescription("GenAI client inference operation duration."),
			metric.WithExplicitBucketBoundaries(inferenceDurationBuckets...))),
		inferenceInputTokens: created(m.Int64Histogram("gen_ai.client.inference.operation.input_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of input (prompt) tokens used per inference operation."),
			metric.WithExplicitBucketBoundaries(inferenceTokenBuckets...))),
		inferenceOutputTokens: created(m.Int64Histogram("gen_ai.client.inference.operation.output_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of output (completion) tokens used per inference operation."),
			metric.WithExplicitBucketBoundaries(inferenceTokenBuckets...))),
		usageInputTokens: created(m.Int64Counter("gen_ai.client.inference.usage.input_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of input (prompt) tokens used, including cached tokens."))),
		usageOutputTokens: created(m.Int64Counter("gen_ai.client.inference.usage.output_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of output (completion) tokens used, including reasoning tokens."))),
		usageCacheReadInputTokens: created(m.Int64Counter("gen_ai.client.inference.usage.cache_read.input_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of input tokens served from a provider-managed cache."))),
		usageCacheWriteInputTokens: created(m.Int64Counter("gen_ai.client.inference.usage.cache_write.input_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of input tokens written to a provider-managed cache."))),
		usageReasoningOutputTokens: created(m.Int64Counter("gen_ai.client.inference.usage.reasoning.output_tokens",
			metric.WithUnit("{token}"),
			metric.WithDescription("The number of output tokens used for reasoning (e.g. chain-of-thought, extended thinking)."))),
		timeToFirstChunk: created(m.Float64Histogram("gen_ai.client.inference.time_to_first_chunk",
			metric.WithUnit("s"),
			metric.WithDescription("Time from the start of a logical inference operation to the first chunk in its streaming response."),
			metric.WithExplicitBucketBoundaries(inferenceDurationBuckets...))),
		timePerOutputChunk: created(m.Float64Histogram("gen_ai.client.inference.time_per_output_chunk",
			metric.WithUnit("s"),
			metric.WithDescription("Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk."),
			metric.WithExplicitBucketBoundaries(inferenceDurationBuckets...))),
	}
}

// created reports a failure to create an instrument to the OpenTelemetry
// error handler, and keeps the instrument the meter returned with the error,
// so that a metric that fails to register never fails the agent. The
// OpenTelemetry SDK returns a usable instrument alongside such an error.
func created[T any](instrument T, err error) T {
	if err != nil {
		otel.Handle(err)
	}
	return instrument
}

// OverrideMeterForTesting makes metrics record into the given provider until
// the test ends.
func OverrideMeterForTesting(t interface{ Cleanup(func()) }, mp metric.MeterProvider) {
	originalProvider, originalInstruments := meterProvider, inst
	meterProvider, inst = mp, newInstruments(mp)
	t.Cleanup(func() { meterProvider, inst = originalProvider, originalInstruments })
}
