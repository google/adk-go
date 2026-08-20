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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

const compactEventsName = "compact_events"

// epochSeconds renders a compaction range bound the way the reference
// implementation does.
//
// adk-python models these bounds as float seconds since the epoch and puts that
// float straight on the span, so a consumer joining traces across the two
// implementations has to see the same type under the same key. An RFC 3339
// string would also carry the host's zone offset onto the wire and, with
// fractional zeros stripped, would not even sort in time order.
func epochSeconds(t time.Time) float64 {
	return float64(t.UnixNano()) / float64(time.Second)
}

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
	genAICompactionDeclined       = attribute.Key("gen_ai.compaction.declined")
	genAICompactionResultEventID  = attribute.Key("gen_ai.compaction.result_event_id")
	genAICompactionStartTimestamp = attribute.Key("gen_ai.compaction.start_timestamp")
	genAICompactionEndTimestamp   = attribute.Key("gen_ai.compaction.end_timestamp")
	genAICompactionInvocationID   = attribute.Key("gcp.vertex.agent.invocation_id")
	genAICompactionInputTokens    = attribute.Key("gen_ai.usage.input_tokens")
	genAICompactionOutputTokens   = attribute.Key("gen_ai.usage.output_tokens")
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
	// InvocationID is the turn that triggered the compaction, or "" when it is
	// not known. The span is not a child of the turn's span, so without this
	// there is no way to ask which turn a compaction belonged to.
	InvocationID string
	// SummarizerType is the bare type name of the summarizer in use.
	SummarizerType string
	// Backend is the Google backend the summarizer's model talks to, used to
	// label the span with gen_ai.system. BackendUnspecified omits the attribute
	// rather than guessing.
	Backend genai.Backend
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
// The span name and the gen_ai.compaction.* attribute keys match adk-python,
// which was read from source rather than assumed: the ten keys, the span name
// and the operation name are identical there. Nothing enforces that agreement,
// so treat them as fixed and change them only alongside the other
// implementations. adk-kotlin has no compaction telemetry at all today, so
// "cross-language" here means two implementations, not all of them.
//
// The span wraps the summarizer call rather than the whole compaction. What
// precedes it is an in-memory scan and window selection, microseconds against a
// model call, and starting the span earlier would emit one for every evaluation
// that declines. A span therefore means compaction really ran, which is the
// more useful signal.
func StartCompactEventsSpan(ctx context.Context, params StartCompactEventsSpanParams) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{
		semconv.GenAIOperationNameKey.String(compactEventsName),
		semconv.GenAIConversationID(params.SessionID),
		genAICompactionTrigger.String(params.Trigger),
		genAICompactionSummarizerType.String(params.SummarizerType),
		genAICompactionEventCount.Int(params.EventCount),
	}
	if params.InvocationID != "" {
		attrs = append(attrs, genAICompactionInvocationID.String(params.InvocationID))
	}
	// gen_ai.system names the system that produced the summary, from the one
	// definition this repo has for it, so a future change to that mapping
	// reaches compaction too.
	//
	// Two known divergences from adk-python, both repo-wide rather than
	// compaction's. The values come from this repo's semconv version, which
	// prefixes them "gcp.", where adk-python is on an older generation and
	// emits the bare "gemini" and "vertex_ai". And the attribute is omitted for
	// a provider this mapping does not know, where adk-python always emits one:
	// naming a provider we cannot identify would be worse than saying nothing.
	if sys, ok := GenAISystemAttr(params.Backend); ok {
		attrs = append(attrs, sys)
	}
	// Omit a threshold that is not configured, so a span carries only the
	// knobs in play. Both strategies may be configured at once, so this says
	// nothing about which one produced this span; Trigger is what names that.
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
	// DiscardReason, when set, says why a summary that was produced never
	// reached the session. It is not an error: the turn was fine and the
	// summary was simply not worth keeping.
	DiscardReason string
}

// TraceCompactionResult records the outcome of a compaction on span.
//
// A nil ResultEvent with a nil Error is a summarizer that declined; the span is
// left successful with no result attributes, which distinguishes "ran and
// produced nothing" from "ran and failed".
func TraceCompactionResult(span trace.Span, params TraceCompactionResultParams) {
	recordErrorAndStatus(span, params.Error)
	if params.DiscardReason != "" {
		// Produced but not kept. Recorded on the same key as a decline, because
		// to anything reading the trace the outcome is the same: compaction was
		// wanted, a model call was spent, and the prompt did not shrink.
		span.SetAttributes(genAICompactionDeclined.String(params.DiscardReason))
	}
	if params.Error != nil {
		// A failed compaction has no result to describe. A summarizer may
		// return an event alongside an error, and the caller discards it, so
		// recording its identity here would leave one span that is at once an
		// error and a success, naming an event that was never appended.
		return
	}

	ev := params.ResultEvent
	if ev == nil || ev.Actions.Compaction == nil {
		return
	}
	// The summarizer's own token usage. Compaction spends a model call to save
	// tokens later, and without this the span cannot say what it spent, so
	// nobody can tell a compaction that paid for itself from one that did not.
	if u := ev.LLMResponse.UsageMetadata; u != nil {
		if u.PromptTokenCount > 0 {
			span.SetAttributes(genAICompactionInputTokens.Int(int(u.PromptTokenCount)))
		}
		// Candidates plus thoughts, matching TraceGenerateContentResult in this
		// package and the semconv note it cites. Counting candidates alone made
		// two spans in one trace mean different things by the same key, and
		// under-reported what a thinking model charged for the summary.
		if out := u.CandidatesTokenCount + u.ThoughtsTokenCount; out > 0 {
			span.SetAttributes(genAICompactionOutputTokens.Int(int(out)))
		}
	}
	attrs := []attribute.KeyValue{genAICompactionResultEventID.String(ev.ID)}
	// A zero time means "no bound recorded", not a real instant. Sent as epoch
	// seconds it reports the year 1754, which turned three seconds of history
	// into a range 271 years wide on a span that otherwise says the compaction
	// succeeded. The reference implementation omits the key instead, and an
	// absent attribute is the one form a consumer can recognise as missing.
	if ts := ev.Actions.Compaction.StartTimestamp; !ts.IsZero() {
		attrs = append(attrs, genAICompactionStartTimestamp.Float64(epochSeconds(ts)))
	}
	if ts := ev.Actions.Compaction.EndTimestamp; !ts.IsZero() {
		attrs = append(attrs, genAICompactionEndTimestamp.Float64(epochSeconds(ts)))
	}
	span.SetAttributes(attrs...)
}

// TraceCompactionDeclined records a compaction that fired but could not run.
//
// The span carries the same attributes as one that did run, plus the reason, so
// "the threshold is crossed and nothing can be done about it" is visible rather
// than looking exactly like an idle session. The attribute has no counterpart in
// the reference implementation, which emits nothing for this state at all.
func TraceCompactionDeclined(ctx context.Context, params StartCompactEventsSpanParams, reason string) {
	_, span := StartCompactEventsSpan(ctx, params)
	span.SetAttributes(genAICompactionDeclined.String(reason))
	span.End()
}
