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
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/session"
)

const compactEventsName = "compact_events"

// timestampLayout renders compaction range bounds. Event timestamps are
// time.Time, and RFC 3339 with nanoseconds preserves their full precision while
// staying readable in a trace viewer.
const timestampLayout = time.RFC3339Nano

// Compaction trigger names. Each becomes the suffix of the span name, so a
// trace distinguishes the two strategies at a glance.
const (
	CompactionTriggerSlidingWindow  = "sliding_window"
	CompactionTriggerTokenThreshold = "token_threshold"
)

var (
	genAICompactionTrigger        = attribute.Key("gen_ai.compaction.trigger")
	genAICompactionSummarizerType = attribute.Key("gen_ai.compaction.summarizer_type")
	genAICompactionEventCount     = attribute.Key("gen_ai.compaction.event_count")
	genAICompactionTokenThreshold = attribute.Key("gen_ai.compaction.token_threshold")
	genAICompactionEventRetention = attribute.Key("gen_ai.compaction.event_retention_size")
	genAICompactionInterval       = attribute.Key("gen_ai.compaction.compaction_interval")
	genAICompactionOverlapSize    = attribute.Key("gen_ai.compaction.overlap_size")
	genAICompactionResultEventID  = attribute.Key("gen_ai.compaction.result_event_id")
	genAICompactionStartTimestamp = attribute.Key("gen_ai.compaction.start_timestamp")
	genAICompactionEndTimestamp   = attribute.Key("gen_ai.compaction.end_timestamp")
)

// StartCompactEventsSpanParams contains parameters for [StartCompactEventsSpan].
//
// The configuration values are passed as plain ints rather than a
// compaction.Config so this package does not import session/compaction, which
// imports this one.
type StartCompactEventsSpanParams struct {
	// Trigger names the strategy that fired, e.g. [CompactionTriggerSlidingWindow].
	Trigger string
	// SessionID is the session whose history is being compacted.
	SessionID string
	// SummarizerType is the concrete Go type of the summarizer in use.
	SummarizerType string
	// EventCount is how many events were selected for summarization.
	EventCount int

	// The configured thresholds. Zero means the corresponding strategy is
	// disabled, and the attribute is omitted.
	CompactionInterval int
	OverlapSize        int
	TokenThreshold     int
	EventRetentionSize int
}

// StartCompactEventsSpan starts a span covering one context-compaction
// summarization, named "compact_events <trigger>".
//
// The span name and the gen_ai.compaction.* attribute keys are part of ADK's
// cross-language telemetry contract, so a dashboard or query written against
// one ADK implementation works against the others. Renaming them here breaks
// that, so treat them as fixed.
//
// The span wraps only the summarizer call, so its presence in a trace means
// compaction really ran. A run that evaluated a trigger and declined produces
// no span, which keeps the signal meaningful.
func StartCompactEventsSpan(ctx context.Context, params StartCompactEventsSpanParams) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(compactEventsName),
		semconv.GenAIConversationID(params.SessionID),
		genAICompactionTrigger.String(params.Trigger),
		genAICompactionSummarizerType.String(params.SummarizerType),
		genAICompactionEventCount.Int(params.EventCount),
	}
	// Omit a threshold that is not configured, so a span shows only the
	// strategy that actually fired.
	if params.CompactionInterval > 0 {
		attrs = append(attrs,
			genAICompactionInterval.Int(params.CompactionInterval),
			genAICompactionOverlapSize.Int(params.OverlapSize))
	}
	if params.TokenThreshold > 0 {
		attrs = append(attrs,
			genAICompactionTokenThreshold.Int(params.TokenThreshold),
			genAICompactionEventRetention.Int(params.EventRetentionSize))
	}
	return tracer.Start(ctx, fmt.Sprintf("%s %s", compactEventsName, params.Trigger), trace.WithAttributes(attrs...))
}

// TraceCompactionResultParams contains parameters for [TraceCompactionResult].
type TraceCompactionResultParams struct {
	// ResultEvent is the compaction event produced, or nil when the summarizer
	// declined. Its identity fields must already be stamped.
	ResultEvent *session.Event
	// Error is the summarization failure, if any.
	Error error
}

// TraceCompactionResult records the outcome of a compaction on span.
//
// A nil ResultEvent with a nil Error is a summarizer that declined; the span is
// left successful with no result attributes, which distinguishes "ran and
// produced nothing" from "ran and failed".
func TraceCompactionResult(span trace.Span, params TraceCompactionResultParams) {
	recordErrorAndStatus(span, params.Error)

	ev := params.ResultEvent
	if ev == nil || ev.Actions.Compaction == nil {
		return
	}
	span.SetAttributes(
		genAICompactionResultEventID.String(ev.ID),
		genAICompactionStartTimestamp.String(ev.Actions.Compaction.StartTimestamp.Format(timestampLayout)),
		genAICompactionEndTimestamp.String(ev.Actions.Compaction.EndTimestamp.Format(timestampLayout)),
	)
}
