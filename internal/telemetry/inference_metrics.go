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
	"cmp"
	"context"
	"maps"
	"slices"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

var genAITokenModality = attribute.Key("gen_ai.token.modality")

// clientInference collects what the client inference metrics of one model call record.
type clientInference struct {
	params   GenerateContentParams
	provider attribute.KeyValue
	// start is when the call started, on the wall clock the span records.
	start time.Time
	// responses holds every response the call yielded, in order, including
	// those yielded after the span ended.
	responses []*model.LLMResponse
	// err is the first error the call yielded that is not a cancellation.
	err error
	// lastChunk is when the previous timed chunk arrived, or the zero time
	// before the first one.
	lastChunk time.Time
	// sawPartial reports whether any response so far was marked partial.
	sawPartial bool
}

func newClientInference(params GenerateContentParams, start time.Time) *clientInference {
	return &clientInference{
		params:   params,
		provider: providerName(params.ModelName, params.Backend),
		start:    start,
	}
}

// observe notes one response or error the model yielded, and records the
// wait for a streamed chunk. The caller reports whether the inference span was
// still open when the value arrived, since a chunk that arrives after the span
// ended is not timed.
func (c *clientInference) observe(ctx context.Context, resp *model.LLMResponse, err error, spanOpen bool) {
	if c.err == nil {
		c.err = inferenceError(err)
	}
	if resp == nil {
		return
	}
	c.responses = append(c.responses, resp)
	// ADK's stream aggregator marks each chunk partial, then yields the
	// answer it assembled from them unmarked. Nothing waited for that
	// aggregate, so it is not timed. A model that never marks a response
	// partial has every response timed. Chunks are not checked for new output,
	// as in adk-python, though the conventions exclude events that carry none.
	aggregate := c.sawPartial && !resp.Partial
	c.sawPartial = c.sawPartial || resp.Partial
	if !c.params.Stream || !spanOpen || err != nil || aggregate {
		return
	}
	now := time.Now()
	// The response model comes only from the chunk itself: a chunk timing
	// describes that chunk.
	attrs := metric.WithAttributes(c.attributes(resp.ModelVersion)...)
	if c.lastChunk.IsZero() {
		inst.timeToFirstChunk.Record(ctx, now.Sub(c.start).Seconds(), attrs)
	} else {
		inst.timePerOutputChunk.Record(ctx, now.Sub(c.lastChunk).Seconds(), attrs)
	}
	c.lastChunk = now
}

// record records the duration and token usage of the whole call, given the
// time its span ended. Every call records each of them once, failed calls
// included; usage the call did not report counts as zero tokens.
func (c *clientInference) record(ctx context.Context, end time.Time) {
	var responseModel string
	if len(c.responses) > 0 {
		responseModel = cmp.Or(c.responses[len(c.responses)-1].ModelVersion, c.params.ModelName)
	}
	attrs := c.attributes(responseModel)
	histogramAttrs := attrs
	if c.err != nil {
		histogramAttrs = append(slices.Clip(attrs), errorType(c.err))
	}
	histogram := metric.WithAttributes(histogramAttrs...)
	// From wall-clock readings, without the monotonic one, so the metric equals
	// the duration exporters compute from the span's timestamps even when the
	// system clock is adjusted mid-call.
	inst.inferenceDuration.Record(ctx, end.Round(0).Sub(c.start.Round(0)).Seconds(), histogram)

	usage := usageFromResponses(c.responses)
	inst.inferenceInputTokens.Record(ctx, usage.input, histogram)
	inst.inferenceOutputTokens.Record(ctx, usage.output, histogram)
	addByModality(ctx, inst.usageInputTokens, usage.input, usage.inputDetails, attrs)
	addByModality(ctx, inst.usageOutputTokens, usage.output, usage.outputDetails, attrs)
	addByModality(ctx, inst.usageCacheReadInputTokens, usage.cacheRead, usage.cacheReadDetails, attrs)
	// The model reports cache writes and reasoning tokens without a modality
	// breakdown.
	addByModality(ctx, inst.usageCacheWriteInputTokens, usage.cacheWrite, nil, attrs)
	addByModality(ctx, inst.usageReasoningOutputTokens, usage.reasoning, nil, attrs)
}

// attributes are the attributes every client inference metric shares. An
// empty request or response model is left out.
func (c *clientInference) attributes(responseModel string) []attribute.KeyValue {
	attrs := []attribute.KeyValue{semconv.GenAIOperationNameGenerateContent, c.provider}
	if c.params.ModelName != "" {
		attrs = append(attrs, semconv.GenAIRequestModel(c.params.ModelName))
	}
	if responseModel != "" {
		attrs = append(attrs, semconv.GenAIResponseModel(responseModel))
	}
	return attrs
}

// addByModality adds a token count to a counter once per modality, split by
// the model's per-modality breakdown. A split with nothing in it, such as zero
// tokens, adds zero under the unknown modality, so every call adds a point.
// This matches adk-python, though the conventions require strictly positive
// points.
func addByModality(ctx context.Context, counter metric.Int64Counter, tokens int64, details []*genai.ModalityTokenCount, attrs []attribute.KeyValue) {
	split := splitByModality(tokens, details)
	if len(split) == 0 {
		split = map[string]int64{"unknown": 0}
	}
	for modality, n := range split {
		counter.Add(ctx, n, metric.WithAttributes(append(slices.Clip(attrs), genAITokenModality.String(modality))...))
	}
}

// splitByModality splits a token count across the modalities of a
// per-modality breakdown. Modalities are lowercased, and an unspecified
// modality counts as text. Whatever the breakdown leaves unaccounted for, such
// as reasoning tokens, goes to unknown, so the values sum to the count unless
// the breakdown exceeds it.
func splitByModality(total int64, details []*genai.ModalityTokenCount) map[string]int64 {
	split := map[string]int64{}
	var accounted int64
	for _, d := range details {
		if d == nil || d.Modality == "" {
			continue
		}
		modality := strings.ToLower(string(d.Modality))
		if d.Modality == genai.MediaModalityUnspecified {
			modality = "text"
		}
		split[modality] += int64(d.TokenCount)
		accounted += int64(d.TokenCount)
	}
	if unknown := total - accounted; unknown > 0 {
		split["unknown"] = unknown
	}
	maps.DeleteFunc(split, func(_ string, n int64) bool { return n == 0 })
	return split
}
