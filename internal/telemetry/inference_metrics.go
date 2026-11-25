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
	"slices"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/telemetry/tokenusage"
	"google.golang.org/adk/v2/model"
)

var genAITokenModality = attribute.Key("gen_ai.token.modality")

// clientInference collects what the client inference metrics of one model call record.
type clientInference struct {
	params   GenerateContentParams
	provider attribute.KeyValue
	// start is when the call started, which the first chunk is timed from.
	start time.Time
	// responses holds every response the call yielded, in order, including
	// those yielded after the span ended.
	responses []*model.LLMResponse
	// err is the first error the call yielded that is not a cancellation.
	err error
	// lastChunk is when the previous timed chunk arrived, or the zero time
	// before the first one.
	lastChunk  time.Time
	sawPartial bool
}

func newClientInference(params GenerateContentParams, start time.Time) *clientInference {
	return &clientInference{
		params:   params,
		provider: providerName(params.Provider, params.ModelName, params.Backend),
		start:    start,
	}
}

// observe notes one response the model yielded, and records the wait for a
// streamed chunk. A value that carries an error is not timed. The caller
// reports whether the inference span was still open when the value arrived,
// since a chunk that arrives after the span ended is not timed.
func (c *clientInference) observe(ctx context.Context, resp *model.LLMResponse, err error, spanOpen bool) {
	if resp == nil {
		return
	}
	c.responses = append(c.responses, resp)
	// ADK's stream aggregator marks each chunk partial, then yields the
	// answer it assembled from them unmarked. Nothing waited for that
	// aggregate, so it is not timed. A model that never marks a response
	// partial has every response timed. A response without output, such as
	// one carrying only usage or a finish reason, is not a chunk.
	aggregate := c.sawPartial && !resp.Partial
	c.sawPartial = c.sawPartial || resp.Partial
	if !c.params.Stream || !spanOpen || err != nil || aggregate || !carriesOutput(resp) {
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

func carriesOutput(resp *model.LLMResponse) bool {
	if resp.Content == nil {
		return false
	}
	for _, p := range resp.Content.Parts {
		switch {
		case p == nil:
		case p.Text != "",
			p.InlineData != nil && len(p.InlineData.Data) > 0,
			p.FunctionCall != nil && (p.FunctionCall.Name != "" || len(p.FunctionCall.Args) > 0 || len(p.FunctionCall.PartialArgs) > 0),
			p.ToolCall != nil && (p.ToolCall.ToolType != "" || len(p.ToolCall.Args) > 0):
			return true
		}
	}
	return false
}

// record records the duration and token usage of the whole call, given how
// long it took. Every call records the duration and token histograms once,
// failed calls included, with usage the call did not report as zero tokens.
// The usage counters get only the tokens the call reported.
func (c *clientInference) record(ctx context.Context, elapsed time.Duration) {
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
	inst.inferenceDuration.Record(ctx, elapsed.Seconds(), histogram)

	usage := tokenusage.FromResponses(c.responses)
	inst.inferenceInputTokens.Record(ctx, usage.Input, histogram)
	inst.inferenceOutputTokens.Record(ctx, usage.Output, histogram)
	addByModality(ctx, inst.usageInputTokens, usage.Input, usage.InputDetails, attrs)
	addByModality(ctx, inst.usageOutputTokens, usage.Output, usage.OutputDetails, attrs)
	addByModality(ctx, inst.usageCacheReadInputTokens, usage.CacheRead, usage.CacheReadDetails, attrs)
	// The model reports cache writes and reasoning tokens without a modality
	// breakdown.
	addByModality(ctx, inst.usageCacheWriteInputTokens, usage.CacheWrite, nil, attrs)
	addByModality(ctx, inst.usageReasoningOutputTokens, usage.Reasoning, nil, attrs)
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
// the model's per-modality breakdown. A modality with no tokens adds nothing,
// as the conventions require strictly positive measurements.
func addByModality(ctx context.Context, counter metric.Int64Counter, tokens int64, details []*genai.ModalityTokenCount, attrs []attribute.KeyValue) {
	split := splitByModality(tokens, details)
	for _, c := range []struct {
		modality string
		tokens   int64
	}{
		{"text", split.text},
		{"image", split.image},
		{"video", split.video},
		{"audio", split.audio},
		{"document", split.document},
		{"unknown", split.unknown},
	} {
		if c.tokens > 0 {
			counter.Add(ctx, c.tokens, metric.WithAttributes(append(slices.Clip(attrs), genAITokenModality.String(c.modality))...))
		}
	}
}

// modalitySplit is a token count split across the modalities the genai SDK
// defines, plus unknown.
type modalitySplit struct {
	text, image, video, audio, document, unknown int64
}

// splitByModality splits a token count across the modalities of a
// per-modality breakdown. An unspecified or unrecognized modality counts as
// unknown. Whatever the breakdown leaves unaccounted for, such as reasoning
// tokens, goes to unknown too, so the split sums to the count unless the
// breakdown exceeds it.
func splitByModality(total int64, details []*genai.ModalityTokenCount) modalitySplit {
	var split modalitySplit
	var accounted int64
	for _, d := range details {
		if d == nil || d.Modality == "" {
			continue
		}
		n := int64(d.TokenCount)
		switch genai.MediaModality(strings.ToUpper(string(d.Modality))) {
		case genai.MediaModalityText:
			split.text += n
		case genai.MediaModalityImage:
			split.image += n
		case genai.MediaModalityVideo:
			split.video += n
		case genai.MediaModalityAudio:
			split.audio += n
		case genai.MediaModalityDocument:
			split.document += n
		default:
			split.unknown += n
		}
		accounted += n
	}
	if unknown := total - accounted; unknown > 0 {
		split.unknown += unknown
	}
	return split
}
