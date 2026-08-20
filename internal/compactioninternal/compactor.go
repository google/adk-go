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

package compactioninternal

import (
	"context"
	"fmt"
	"reflect"
	"slices"

	"go.opentelemetry.io/otel/codes"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// HasSlidingWindow reports whether sliding-window compaction is enabled.
//
// These live here rather than as methods on compaction.Config because nothing
// outside the framework needs to ask: the runner and the request processor are
// the only callers, and keeping them off the public type leaves users with just
// the fields they set.
func HasSlidingWindow(cfg *compaction.Config) bool {
	return cfg != nil && cfg.CompactionInterval > 0
}

// HasTailRetention reports whether tail-retention compaction is enabled.
func HasTailRetention(cfg *compaction.Config) bool {
	return cfg != nil && cfg.TokenThreshold > 0
}

// SlidingWindow summarizes a window of completed invocations once enough of
// them have accumulated, and returns the resulting compaction event, ready for
// the caller to append to the session.
//
// It returns a nil event, and no error, whenever there is nothing to do: fewer
// than cfg.CompactionInterval invocations since the last compaction, a window
// with no self-contained prefix, or a summarizer that declined to produce a
// summary. Callers treat all three the same way, by leaving history untouched.
//
// The runner calls this after an invocation finishes and all of its events have
// been persisted; compacting mid-invocation is the tail-retention strategy's
// job.
//
// The returned [Finish] must be called exactly once with what became of the
// summary, which is what closes its span. It is never nil.
func SlidingWindow(ctx context.Context, cfg *compaction.Config, sess session.Session, invocationID string) (*session.Event, Finish, error) {
	noop := func(error, string) {}
	if !HasSlidingWindow(cfg) {
		return nil, noop, nil
	}
	if cfg.Summarizer == nil {
		return nil, noop, fmt.Errorf("no Summarizer configured")
	}
	if sess == nil {
		return nil, noop, nil
	}

	events := collect(sess)
	window := selectSlidingWindow(events, cfg.CompactionInterval, cfg.OverlapSize)
	if len(window) == 0 {
		return nil, noop, nil
	}

	summary, finish, err := summarizeTraced(ctx, cfg, sess, invocationID, telemetry.CompactionTriggerSlidingWindow, window)
	if err != nil {
		return nil, noop, fmt.Errorf("sliding-window summarization failed: %w", err)
	}
	return summary, finish, nil
}

// Finish reports what became of a summary and closes its span.
//
// A summarization is not over when the summarizer returns. The caller still has
// to decide whether to keep the result, and it can throw it away for half a
// dozen reasons: a cancelled turn, a failed re-read, a competing compaction, a
// plugin rejecting it, or a failed append. Ending the span at the summarizer
// left every one of those reporting success, with a result_event_id naming an
// event that exists in no session.
//
// Exactly one call, and the summary is not stored until it is made.
type Finish func(err error, discardReason string)

// summarizeTraced runs the configured summarizer inside a compact_events span,
// validates what comes back, and stamps it.
//
// The span stays open until the returned [Finish] is called, so it reports what
// actually happened to the summary rather than what the summarizer returned.
// Its presence in a trace still means compaction really ran: a trigger that was
// evaluated and declined produces a decline span instead.
func summarizeTraced(ctx context.Context, cfg *compaction.Config, sess session.Session, invocationID, trigger string, window []*session.Event) (*session.Event, Finish, error) {
	sessionID := ""
	if sess != nil {
		sessionID = sess.ID()
	}
	// The turn that triggered this compaction. The span is not a child of that
	// turn's span, so this attribute is the only way to ask which turn a
	// compaction belonged to.
	//
	// The caller passes it, because the caller knows. Reading the newest event
	// in the session instead was a guess that went wrong exactly when it
	// mattered: with two invocations in flight on one session, both compactions
	// read the same newest event, so at least one named a turn that did not
	// cause it. The fallback remains for a caller that has no ID to give.
	if invocationID == "" {
		invocationID = latestInvocationID(sess)
	}
	ctx, span := telemetry.StartCompactEventsSpan(ctx, spanParams(cfg, sessionID, invocationID, trigger, len(window)))

	var summary *session.Event
	var finished bool
	finish := func(err error, discardReason string) {
		if finished {
			return
		}
		finished = true
		stored := summary
		if err != nil || discardReason != "" {
			// Nothing reached the session, so naming a result would point at an
			// event no session holds.
			stored = nil
		}
		telemetry.TraceCompactionResult(span, telemetry.TraceCompactionResultParams{
			ResultEvent:   stored,
			Error:         err,
			DiscardReason: discardReason,
		})
		span.End()
	}

	// A Summarizer is third-party code and may panic. The OTel SDK records an
	// exception event on the way out but leaves the status Unset, which reads
	// as success, so a panicking summarizer would look like a healthy one that
	// happened to produce nothing. Mark it, record it as an exception so an
	// alert keyed on exception.type sees it, and let the panic continue.
	defer func() {
		if r := recover(); r != nil {
			err := fmt.Errorf("summarizer panicked: %v", r)
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
			span.End()
			finished = true
			panic(r)
		}
	}()

	content, usage, err := cfg.Summarizer.SummarizeEvents(ctx, snapshotForSummarizer(window))

	// The framework builds the event, so a summarizer contributes the summary
	// and nothing else. Everything that decides what happens to history -- the
	// covered range, the authorship, the actions -- is derived here from the
	// window that was handed over.
	switch {
	case err != nil:
	case content == nil:
		// A decline. Usage may still have been reported, and the span records
		// it, so a summarizer that spent a call and got nothing usable back is
		// distinguishable from one that never tried.
	default:
		summary, err = newSummaryEvent(window, collect(sess), content, usage)
	}
	// Stamped only once the result is known to be usable, so a discarded
	// summary never spends a UUID or hands telemetry the identity of something
	// that did not reach the session.
	if err != nil {
		summary = nil
	} else {
		summary = stamp(ctx, summary)
	}
	if err != nil {
		finish(err, "")
		return nil, nil, err
	}
	if summary == nil {
		// A decline. Nothing further can happen to it, so close the span here.
		finish(nil, "")
		return nil, func(error, string) {}, nil
	}
	return summary, finish, nil
}

// stamp fills in the identity fields a [Summarizer] leaves blank, so the
// returned event is ready to append.
//
// The invocation ID is deliberately fresh rather than borrowed from the covered
// turns: sliding-window selection counts invocations, and reusing a covered one
// would skew the next window. Both the ID and the timestamp come from
// [platform], so a test that installs providers keeps deterministic output.
func stamp(ctx context.Context, ev *session.Event) *session.Event {
	if ev == nil {
		return nil
	}
	if ev.ID == "" {
		ev.ID = platform.NewUUID(ctx)
	}
	if ev.InvocationID == "" {
		ev.InvocationID = "e-" + platform.NewUUID(ctx)
	}
	if ev.Timestamp.IsZero() {
		ev.Timestamp = platform.Now(ctx)
	}
	return ev
}

// collect materializes a session's events into a slice.
func collect(sess session.Session) []*session.Event {
	all := sess.Events()
	if all == nil {
		return nil
	}
	events := make([]*session.Event, 0, all.Len())
	for ev := range all.All() {
		events = append(events, ev)
	}
	return events
}

// snapshotForSummarizer returns copies of the events to hand to third-party
// code.
//
// The interface says the events passed in are never modified, and nothing
// enforced it: the slice was copied but the events were not, so a Summarizer
// received the session's live pointers. Narrowing the return type stopped it
// declaring an authorship or a covered range, and left it able to impose both
// by writing to its input, because the record is derived from those same
// objects after the call. Rewriting the stored conversation, moving timestamps
// to dictate the range, and clearing Branch to escape an isolation scope were
// all reachable, and so was planting a compaction record on a live event.
//
// The snapshot is built field by field from what a summarizer is for, rather
// than by copying the event and severing the pointers afterwards. Copying and
// severing was the first approach and it does not hold: session.Event and
// genai.Part between them reach sixteen pointers, maps and slices, a struct
// copy shares every one, and each field added upstream is silently shared until
// somebody notices. Naming the fields inverts that, so a new field is absent
// from the summarizer's view until it is deliberately added.
//
// What a summarizer needs is the conversation: who spoke, when, and what was
// said, including the name and arguments of a tool call, because a transcript
// renders those. What it does not need is the framework's own bookkeeping. The
// compaction record is the sharpest case: it is a live pointer into stored
// history, it decides what every future prompt drops, and a summarizer writing
// through it put an unpaired function call into a real model prompt. Only
// whether an event is a summary, and the range it stood for, survive into the
// copy, both as scalars. The text of a previous summary is still readable,
// because the seed carries it as ordinary content.
func snapshotForSummarizer(events []*session.Event) []*session.Event {
	out := make([]*session.Event, 0, len(events))
	for _, ev := range events {
		if ev == nil {
			out = append(out, nil)
			continue
		}
		clone := &session.Event{
			ID:             ev.ID,
			Timestamp:      ev.Timestamp,
			InvocationID:   ev.InvocationID,
			Branch:         ev.Branch,
			IsolationScope: ev.IsolationScope,
			Author:         ev.Author,
		}
		if c := ev.LLMResponse.Content; c != nil {
			clone.LLMResponse.Content = copyContent(c)
		}
		if rng := ev.Actions.Compaction; rng != nil {
			// Scalars only. Enough to tell a summary apart from a turn and to
			// see what it spanned, carrying no pointer back into the store.
			clone.Actions.Compaction = &session.EventCompaction{
				StartTimestamp: rng.StartTimestamp,
				EndTimestamp:   rng.EndTimestamp,
			}
		}
		out = append(out, clone)
	}
	return out
}

// copyContent deep-copies the parts of a content, including the members a
// transcript reads through: a tool call's name and arguments, a tool response's
// payload, and inline data. Copying the Part struct alone leaves all three
// shared with the store.
func copyContent(c *genai.Content) *genai.Content {
	content := *c
	content.Parts = slices.Clone(c.Parts)
	for i, p := range content.Parts {
		if p == nil {
			continue
		}
		part := *p
		if fc := p.FunctionCall; fc != nil {
			call := *fc
			call.Args = copyAny(fc.Args).(map[string]any)
			part.FunctionCall = &call
		}
		if fr := p.FunctionResponse; fr != nil {
			resp := *fr
			resp.Response = copyAny(fr.Response).(map[string]any)
			part.FunctionResponse = &resp
		}
		if b := p.InlineData; b != nil {
			blob := *b
			blob.Data = slices.Clone(b.Data)
			part.InlineData = &blob
		}
		content.Parts[i] = &part
	}
	return &content
}

// copyAny deep-copies the decoded-JSON shapes a tool payload is made of. A
// shallow map clone protects the top level and leaves a nested map shared,
// which is the same hole one level down.
func copyAny(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if t == nil {
			return map[string]any(nil)
		}
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = copyAny(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = copyAny(val)
		}
		return out
	default:
		return v
	}
}

// summarizerTypeName is the bare type name of a Summarizer, without package
// qualifier or pointer marker.
//
// The reference implementation puts type(summarizer).__name__ on this span, so
// "LLMSummarizer" is what a consumer joining traces across implementations
// expects to match against. Sprintf("%T") would emit
// "*compaction.LLMSummarizer", which names a Go type rather than a summarizer.
func summarizerTypeName(s compaction.Summarizer) string {
	if s == nil {
		return ""
	}
	t := reflect.TypeOf(s)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if name := t.Name(); name != "" {
		return name
	}
	return t.String()
}

// summarizerBackend reports which Google backend a Summarizer's model talks to.
//
// It is an optional interface rather than a field, matching how the rest of the
// framework distinguishes Vertex AI from the Gemini API: a third-party
// Summarizer that has no model, or does not care to say, simply leaves the
// span's gen_ai.system unset rather than being forced to invent one.
func summarizerBackend(s compaction.Summarizer) genai.Backend {
	if v, ok := s.(interface{ GetGoogleLLMVariant() genai.Backend }); ok {
		return v.GetGoogleLLMVariant()
	}
	return genai.BackendUnspecified
}

// latestInvocationID returns the invocation of the newest event in sess.
func latestInvocationID(sess session.Session) string {
	events := collect(sess)
	for i := len(events) - 1; i >= 0; i-- {
		if id := events[i].InvocationID; id != "" {
			return id
		}
	}
	return ""
}

// spanParams builds the attribute set shared by every compaction span.
func spanParams(cfg *compaction.Config, sessionID, invocationID, trigger string, eventCount int) telemetry.StartCompactEventsSpanParams {
	return telemetry.StartCompactEventsSpanParams{
		Trigger:            trigger,
		SessionID:          sessionID,
		InvocationID:       invocationID,
		SummarizerType:     summarizerTypeName(cfg.Summarizer),
		Backend:            summarizerBackend(cfg.Summarizer),
		EventCount:         eventCount,
		CompactionInterval: cfg.CompactionInterval,
		OverlapSize:        cfg.OverlapSize,
		TokenThreshold:     cfg.TokenThreshold,
		EventRetentionSize: cfg.EventRetentionSize,
	}
}
