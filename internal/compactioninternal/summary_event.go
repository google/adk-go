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
	"fmt"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// newSummaryEvent builds the event that carries a summary: it names the events
// the summary replaces, derives the bounding box over them, and applies the
// authorship a stored summary needs.
//
// The returned event carries no ID, invocation ID or timestamp. Those are
// assigned when it is appended, and the invocation ID is deliberately fresh
// rather than one belonging to a covered turn, because sliding-window selection
// counts invocations.
//
// Only prose parts of summary survive into the stored event. Whatever a
// summarizer returns is replayed into later prompts as though the framework had
// produced it, so a function call it invented or was tricked into emitting
// cannot ride along, and a thought is not something the model chose to say.
//
// events must be non-empty and hold no nil element, and summary must be
// non-nil and hold prose. usage may be nil. Bad input is an error rather than
// a silently broken event, because a compaction that stands for nothing still
// costs a model call and still leaves the prompt as large as it was.
func newSummaryEvent(events, all []*session.Event, summary *genai.Content, usage *genai.GenerateContentResponseUsageMetadata) (*session.Event, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("cannot summarize an empty event list")
	}
	// An empty summary is rejected, not just a nil one. Recording a compaction
	// whose content says nothing deletes the covered turns from every future
	// prompt and puts nothing in their place, which is worse than not
	// compacting at all.
	if !hasProse(summary) {
		return nil, fmt.Errorf("summary content is empty, so compacting would delete the covered events and replace them with nothing")
	}
	// The window arrives from window selection rather than from a literal, and
	// the snapshot handed to a summarizer preserves nil elements, so a nil here
	// is an input to reject rather than a panic to hand back from the middle of
	// a turn whose tools have already run.
	for i, ev := range events {
		if ev == nil {
			return nil, fmt.Errorf("events[%d] is nil", i)
		}
	}
	// The bounding box over the window, taken as a true minimum and maximum
	// rather than as its first and last element.
	//
	// A stored event list is in append order, and a timestamp is stamped when
	// an event is created, so two invocations in flight on one session leave
	// the list non-monotonic with a single clock and no skew. Requiring the
	// window to be sorted rejected exactly those sessions, and because nothing
	// was then recorded the same window was re-selected and re-rejected on
	// every later turn: two overlapping invocations were enough to stop a
	// session compacting for good.
	//
	// Widening the box to the true span is safe now that the covered set names
	// its events. It could not be done while coverage was the interval itself,
	// because stretching the interval past the window's own endpoints would
	// swallow events that were never summarized.
	start, end := events[0].Timestamp, events[0].Timestamp
	for _, ev := range events[1:] {
		if ev.Timestamp.Before(start) {
			start = ev.Timestamp
		}
		if ev.Timestamp.After(end) {
			end = ev.Timestamp
		}
	}

	// Only prose survives into the stored summary. Whatever the summarizer
	// returns is injected into later prompts verbatim, so a non-text part
	// reaches the model as if the framework had produced it. A hallucinated or
	// maliciously supplied FunctionCall would arrive unpaired, and a model may
	// act on it. A summary is prose by definition, so anything else is dropped.
	//
	// A surviving part is copied whole rather than rebuilt from its text. A
	// text part can carry metadata that belongs with it and that the model
	// expects back, a thought signature above all, and rebuilding would drop
	// that silently.
	content := genai.Content{Role: "model"}
	for _, p := range summary.Parts {
		if !utils.IsProsePart(p) {
			continue
		}
		part := *p
		content.Parts = append(content.Parts, &part)
	}
	if len(content.Parts) == 0 {
		return nil, fmt.Errorf("summary content holds no prose, so compacting would delete the covered events and replace them with nothing")
	}

	// The summary inherits the branch and isolation scope of what it covers.
	// Without them it carries Branch "" and IsolationScope "", which every
	// branch filter admits and which makes it visible outside the scope its
	// source events belonged to, leaking scoped content across the boundary the
	// filters exist to enforce.
	branch, scope := events[0].Branch, events[0].IsolationScope

	// The holes: events inside the range that this summary does not stand in
	// for, because window selection filtered them out. Everything else in the
	// range is covered, so the common case, a window with no holes in it,
	// records nothing here at all.
	//
	// Referred to by invocation and timestamp, which survive every backend,
	// rather than by event ID, which the Vertex AI service replaces on read.
	//
	// Being imprecise about a reference is not symmetric, and not safe in both
	// directions. A reference matching two events of one invocation that share
	// a timestamp leaves an extra event raw beside a summary of it, which is
	// visible and recoverable. A reference matching nothing does not fall back
	// to anything: coverage is the range minus the exclusions, so a hole that
	// fails to match stops being a hole, and the event it named is dropped in
	// favour of a summary that never described it. Over-naming is the direction
	// to prefer, and under-naming is the one that loses conversation.
	//
	// Window membership is therefore decided by identity rather than by the
	// same key. The window holds the very pointers the session holds, so this
	// is exact, where the key is not: an event outside the window colliding
	// with one inside it used to be read as summarized and recorded as no hole
	// at all, which is the under-naming case above. The synthetic seed is the
	// one window element absent from the session, and it matches nothing here,
	// which is correct because it stands for events rather than being one.
	summarized := make(map[*session.Event]struct{}, len(events))
	for _, ev := range events {
		summarized[ev] = struct{}{}
	}

	// A window rolling up an earlier summary carries it as its first element,
	// so everything that summary stood for is inside the new range while being
	// absent from the window. Those events are represented, transitively, and
	// recording them as holes is what makes a rolling summary fail to replace
	// the one it was built from: the new record then leaves out events the old
	// one covered, so it cannot subsume it, and every pass adds another summary
	// to the prompt instead of superseding the last. The exclusion list grows
	// with the session on top of that, since each pass inherits the previous
	// pass's holes.
	var rolled []*session.EventCompaction
	for _, ev := range events {
		if hasCompaction(ev) {
			rolled = append(rolled, ev.Actions.Compaction)
		}
	}
	covered := func(ev *session.Event) bool {
		for _, rng := range rolled {
			if inRange(ev, rng) && !excludes(rng, ev) {
				return true
			}
		}
		return false
	}

	var excluded []session.EventRef
	seen := make(map[string]struct{})
	for _, ev := range all {
		if ev == nil || hasCompaction(ev) {
			continue
		}
		if ev.Timestamp.Before(start) || ev.Timestamp.After(end) {
			continue
		}
		if _, ok := summarized[ev]; ok {
			continue
		}
		k := refKey(ev)
		if _, ok := seen[k]; ok {
			continue
		}
		if covered(ev) {
			continue
		}
		seen[k] = struct{}{}
		excluded = append(excluded, session.EventRef{InvocationID: ev.InvocationID, Timestamp: ev.Timestamp})
	}

	return &session.Event{
		// Authored as "user" because a summary is injected context rather than
		// something the agent said. It is re-authored as "model" when
		// materialized into a prompt, so the model reads it as prior context.
		Author:         "user",
		Branch:         branch,
		IsolationScope: scope,
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   start,
				EndTimestamp:     end,
				CompactedContent: &content,
				ExcludedEvents:   excluded,
			},
		},
		LLMResponse: model.LLMResponse{UsageMetadata: usage},
	}, nil
}

// hasProse reports whether c carries at least one prose part.
func hasProse(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if utils.IsProsePart(p) {
			return true
		}
	}
	return false
}

// SanitizeSummary strips anything from a compaction record that must not reach
// a prompt, and reports whether the record is still usable.
//
// The framework builds a summary event and filters its content, but a plugin
// can replace that event wholesale on its way to the session, and the
// replacement went to storage unexamined. A plugin returning content with a
// text part and a FunctionCall got that unpaired call into a real model prompt,
// which is the exact thing the filter on the summarizer path exists to stop.
//
// Reports false when nothing usable survives, which the caller treats as a
// summary not worth storing rather than as an error: the plugin was within its
// rights to redact everything.
func SanitizeSummary(ev *session.Event) bool {
	if ev == nil || ev.Actions.Compaction == nil {
		return false
	}
	c := ev.Actions.Compaction.CompactedContent
	if c == nil {
		return false
	}
	kept := make([]*genai.Part, 0, len(c.Parts))
	for _, p := range c.Parts {
		if utils.IsProsePart(p) {
			part := *p
			kept = append(kept, &part)
		}
	}
	if len(kept) == 0 {
		return false
	}
	content := *c
	content.Parts = kept
	ev.Actions.Compaction.CompactedContent = &content
	return true
}

// refKey is the comparable form of an event's reference.
func refKey(ev *session.Event) string {
	if ev == nil {
		return ""
	}
	return ev.InvocationID + "@" + ev.Timestamp.UTC().Format(time.RFC3339Nano)
}
