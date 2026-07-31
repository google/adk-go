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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TokenCounter estimates the prompt token count implied by events.
//
// It is consulted only when no event carries an observed prompt token count,
// for instance before the first model response of a session. Returning zero
// means the count could not be determined, which suppresses compaction.
type TokenCounter func(events []*session.Event) int

// TailRetention summarizes everything but the most recent events once the
// prompt has grown past cfg.TokenThreshold, and returns the resulting
// compaction event, ready for the caller to append to the session.
//
// It returns a nil event, and no error, whenever there is nothing to do: the
// threshold is not reached, too few events exist beyond the retained tail, the
// window has no self-contained prefix, or the summarizer declined.
//
// Unlike [SlidingWindow] this runs *inside* an invocation, before a model call,
// which is what lets it react to a single long turn rather than waiting for the
// turn to end. Callers must run it before assembling contents so the fresh
// summary is reflected in the request.
func TailRetention(ctx context.Context, cfg *compaction.Config, sess session.Session, estimate TokenCounter) (*session.Event, error) {
	if !HasTailRetention(cfg) {
		return nil, nil
	}
	if cfg.Summarizer == nil {
		return nil, fmt.Errorf("no Summarizer configured")
	}
	if sess == nil {
		return nil, nil
	}

	events := collect(sess)
	tokens, ok := promptTokenCount(events, estimate)
	if !ok || tokens < cfg.TokenThreshold {
		return nil, nil
	}

	window := selectTailRetentionWindow(events, cfg.EventRetentionSize)
	if len(window) == 0 {
		return nil, nil
	}

	summary, err := summarizeTraced(ctx, cfg, sess, telemetry.CompactionTriggerTokenThreshold, window)
	if err != nil {
		return nil, fmt.Errorf("tail-retention summarization failed: %w", err)
	}
	return summary, nil
}

// charsPerToken is the crude characters-to-tokens ratio used when no model has
// reported a real prompt token count yet.
const charsPerToken = 4

// promptTokenCount returns the most recently observed prompt token count in
// events, falling back to estimate when no event reports one.
//
// The observed count is preferred because it is what the model actually
// charged for the last call, which accounts for the system instruction, tool
// declarations and non-text parts that a character count cannot see. The
// estimate only matters before the first model response of a session.
//
// The second result is false when no count could be determined, which callers
// treat as "do not compact yet".
func promptTokenCount(events []*session.Event, estimate TokenCounter) (int, bool) {
	for i := len(events) - 1; i >= 0; i-- {
		if usage := events[i].UsageMetadata; usage != nil && usage.PromptTokenCount > 0 {
			return int(usage.PromptTokenCount), true
		}
	}
	if estimate == nil {
		return 0, false
	}
	if tokens := estimate(events); tokens > 0 {
		return tokens, true
	}
	return 0, false
}

// EstimateTokensFromContents returns a crude token estimate for contents, by
// counting text characters and dividing by [charsPerToken].
//
// It exists so callers that already build prompt contents can reuse the same
// approximation the other ADK implementations use, rather than inventing their
// own.
//
// It counts only text parts, so it under-counts a prompt dominated by inline
// data, and it sees nothing outside contents -- notably not the system
// instruction or tool declarations, which for an agent with many tools or a
// large skills catalogue can dominate. It is therefore a floor, not an
// estimate, and is consulted only until the first model response reports a real
// prompt token count.
func EstimateTokensFromContents(contents []*genai.Content) int {
	textChars := 0
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil {
				textChars += len(part.Text)
			}
		}
	}
	if textChars <= 0 {
		return 0
	}
	return textChars / charsPerToken
}

// selectTailRetentionWindow returns the events a tail-retention compaction
// should summarize, or nil when there is nothing to compact.
//
// It takes every event since the last compaction except the most recent
// retentionSize, which stay raw so the model keeps immediate continuity, and
// trims the result with longestSelfContainedPrefix.
//
// When an earlier compaction exists its summary is prepended to the window, so
// the new summary covers and supersedes it. That keeps history as one rolling
// summary plus a raw tail, rather than an ever-growing chain of summaries.
func selectTailRetentionWindow(events []*session.Event, retentionSize int) []*session.Event {
	if retentionSize < 0 {
		return nil
	}

	latest := LatestCompactionEvent(events)
	var candidates []*session.Event
	for _, ev := range events {
		if hasCompaction(ev) {
			continue
		}
		// Events already covered by the previous summary must not be
		// summarized again; only what came after it is a candidate.
		if latest != nil && !ev.Timestamp.After(latest.Actions.Compaction.EndTimestamp) {
			continue
		}
		candidates = append(candidates, ev)
	}
	if len(candidates) <= retentionSize {
		return nil
	}

	// firstRetained is where the raw tail begins; everything before it is
	// eligible for summarization.
	firstRetained := len(candidates)
	if retentionSize > 0 {
		firstRetained -= retentionSize
		// Move the cut back past any same-timestamp group. Compaction coverage
		// is inclusive of EndTimestamp, so a retained event sharing a timestamp
		// with the last summarized one would be dropped from the prompt despite
		// never having been summarized.
		boundary := candidates[firstRetained].Timestamp
		for firstRetained > 0 && !candidates[firstRetained-1].Timestamp.Before(boundary) {
			firstRetained--
		}
	}

	window := longestSelfContainedPrefix(candidates[:firstRetained])
	if len(window) == 0 {
		return nil
	}

	if latest == nil {
		return window
	}

	// Seed the window with the previous summary, timestamped at the start of
	// the range it covered. The new compaction therefore spans a strictly wider
	// range, which subsumes the old one at prompt-build time.
	prev := latest.Actions.Compaction
	seed := &session.Event{
		Author:      "model",
		Timestamp:   prev.StartTimestamp,
		Branch:      latest.Branch,
		LLMResponse: model.LLMResponse{Content: prev.CompactedContent},
	}
	return append([]*session.Event{seed}, window...)
}
