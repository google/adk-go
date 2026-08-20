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

// Package compaction summarizes older session events so an agent's prompt stays
// small as its conversation grows.
//
// A compaction never modifies or deletes history. Summarizing a range of events
// appends one new [session.Event] carrying a [session.EventCompaction] that
// records the covered timestamp range and the summary content. When the next
// prompt is built, the raw events inside that range are dropped and the summary
// is materialized in their place.
//
// # What each strategy achieves
//
// Sliding window replaces each group of invocations with one summary, but
// summaries are never themselves re-summarized. Prompt size therefore still
// grows with conversation length, at a reduced constant factor rather than
// being bounded.
//
// Tail retention is what bounds it: each new summary is seeded with the
// previous one, so history stays as a single rolling summary plus a raw tail.
// An agent that needs a genuine ceiling on prompt size should enable it.
//
// # Enable one strategy, not both
//
// The two do not compose, despite firing at different points: tail retention
// runs mid-invocation before a model call, the sliding window once an
// invocation has completed.
//
// They share a candidate rule. Tail retention summarizes the events that no
// compaction already covers, and the sliding window covers everything it
// reaches, every CompactionInterval invocations. What is left uncovered never
// exceeds EventRetentionSize, so tail retention finds nothing to do and never
// fires. It cannot fall back to consolidating the summaries either, because
// those are compaction events and no strategy re-summarizes one.
//
// So adding the sliding window to a bounded configuration unbounds it. Measured
// over 160 turns with a 520-character summary: tail retention alone held the
// prompt flat at about 550 characters, and the two together grew it linearly to
// 41,000. adk-python starves its own token-threshold strategy the same way, so
// this is a property of the shared design rather than of this implementation.
//
// Enable tail retention for a ceiling, or the sliding window for a
// constant-factor reduction. Enabling both gives the sliding window's behaviour
// at the cost of both.
//
// Compaction is enabled per runner. See the EventsCompactionConfig field on
// runner.Config:
//
//	r, err := runner.New(runner.Config{
//		AppName:        "my-app",
//		Agent:          rootAgent,
//		SessionService: session.InMemoryService(),
//		EventsCompactionConfig: &compaction.Config{
//			CompactionInterval: 3,
//			OverlapSize:        1,
//		},
//	})
package compaction

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// ErrCompaction marks an error as a compaction failure rather than a failure of
// the turn itself.
//
// Compaction is bookkeeping: the events of the turn are already persisted
// before it runs, so a failure costs a smaller prompt later, not the user's
// answer. It still surfaces, because a summarizer that never succeeds is worth
// knowing about, but a caller that would rather log it than fail the turn can
// tell the two apart:
//
//	for event, err := range r.Run(...) {
//		if errors.Is(err, compaction.ErrCompaction) {
//			log.Printf("compaction failed: %v", err)
//			continue
//		}
//		...
//	}
var ErrCompaction = errors.New("context compaction failed")

// Config configures context compaction for an application.
//
// Two strategies are available, and at least one must be enabled. A Config that
// enables neither is rejected by [Config.Validate], because it would cost a
// configuration step and do nothing; leave the whole Config nil to disable
// compaction:
//
//   - Sliding window (CompactionInterval, OverlapSize) runs after an invocation
//     completes and summarizes whole invocations at a time.
//   - Tail retention (TokenThreshold, EventRetentionSize) runs inside an
//     invocation before a model call and summarizes everything but the most
//     recent events once the prompt grows past a token budget.
//
// Choose one. They are not independent: the sliding window consumes the events
// tail retention would otherwise summarize, so enabling both leaves tail
// retention permanently idle and the prompt unbounded. See the package
// documentation for the measurements. Validate accepts the combination, because
// it is well-formed and because rejecting it would break configurations that
// already exist, but it is the sliding window alone that you get.
type Config struct {
	// CompactionInterval is the number of new user-initiated invocations that,
	// once fully represented in the session's events, triggers a sliding-window
	// compaction. Zero, the default, disables sliding-window compaction.
	//
	// It also bounds the window: one compaction covers at most this many new
	// invocations, so enabling compaction on a session that already has a long
	// history drains the backlog a window at a time rather than summarizing all
	// of it in one call.
	CompactionInterval int

	// OverlapSize is how many already-compacted invocations to pull back into
	// the next sliding window, creating an overlap between consecutive
	// summaries for continuity. Only meaningful alongside CompactionInterval.
	//
	// The overlap is repeated, not shared: an invocation pulled back in is
	// described by both summaries, so the model sees it twice and the prompt
	// carries roughly OverlapSize invocations of extra text per summary. That
	// is the cost of the continuity, and it cannot be trimmed away afterwards,
	// because by then the repetition lives inside summary prose rather than in
	// the ranges. Leave it at zero unless summaries are visibly losing the
	// thread between windows.
	OverlapSize int

	// TokenThreshold is the prompt token count at which intra-invocation
	// tail-retention compaction fires before a model call. Zero, the default,
	// disables tail-retention compaction.
	TokenThreshold int

	// EventRetentionSize is how many of the most recent events are kept raw
	// when tail-retention compaction fires. Everything older is summarized.
	// Only meaningful alongside TokenThreshold, and required with it: at zero
	// the window would extend to the newest event, which includes the question
	// the model is about to answer, so the turn in progress would be summarized
	// out of its own prompt.
	EventRetentionSize int

	// Summarizer produces the summary content. When nil, the runner supplies an
	// [LLMSummarizer] backed by the root agent's model, which therefore has to
	// be an LLM agent.
	Summarizer Summarizer
}

// hasSlidingWindow reports whether sliding-window compaction is enabled.
func (c *Config) hasSlidingWindow() bool {
	return c != nil && c.CompactionInterval > 0
}

// hasTailRetention reports whether tail-retention compaction is enabled.
func (c *Config) hasTailRetention() bool {
	return c != nil && c.TokenThreshold > 0
}

// Validate reports whether the configuration is usable.
//
// A nil Config is valid and means compaction is disabled. A non-nil Config with
// no strategy enabled is not: allocating one and setting nothing is a mistake
// worth reporting rather than silently doing nothing, and nil already expresses
// "disabled".
func (c *Config) Validate() error {
	if c == nil {
		return nil
	}
	if c.CompactionInterval < 0 {
		return fmt.Errorf("CompactionInterval must not be negative, got %d", c.CompactionInterval)
	}
	if c.OverlapSize < 0 {
		return fmt.Errorf("OverlapSize must not be negative, got %d", c.OverlapSize)
	}
	if c.TokenThreshold < 0 {
		return fmt.Errorf("TokenThreshold must not be negative, got %d", c.TokenThreshold)
	}
	if c.EventRetentionSize < 0 {
		return fmt.Errorf("EventRetentionSize must not be negative, got %d", c.EventRetentionSize)
	}
	if c.OverlapSize > 0 && c.CompactionInterval == 0 {
		return fmt.Errorf("OverlapSize is set to %d but CompactionInterval is 0, so sliding-window compaction never runs", c.OverlapSize)
	}
	if c.TokenThreshold > 0 && c.EventRetentionSize == 0 {
		return fmt.Errorf("TokenThreshold is set to %d but EventRetentionSize is 0, so a compaction would summarize the whole conversation including the turn being answered", c.TokenThreshold)
	}
	if c.EventRetentionSize > 0 && c.TokenThreshold == 0 {
		return fmt.Errorf("EventRetentionSize is set to %d but TokenThreshold is 0, so tail-retention compaction never runs", c.EventRetentionSize)
	}
	if !c.hasSlidingWindow() && !c.hasTailRetention() {
		return fmt.Errorf("no compaction strategy is enabled, set CompactionInterval or TokenThreshold (or leave the whole config nil to disable compaction)")
	}
	return nil
}

// Summarizer condenses a range of events into a single piece of content.
//
// Implement it to control which parts of an event reach the summary and how the
// summary is produced. [LLMSummarizer] is the default implementation.
//
// An implementation returns only the summary. The framework builds the event
// that carries it, derives the range it covers from the events it handed over,
// and appends it. That division is deliberate: a summarizer that returned a
// whole event could also set the authorship, the state delta, an agent
// transfer, and the range of history to delete, none of which is summarizing.
type Summarizer interface {
	// SummarizeEvents summarizes events into one piece of content, with the
	// token usage the summary cost when that is known. The events passed in are
	// never modified.
	//
	// Returning no content and no error is a decline: this range was not
	// summarized, the caller leaves history alone and carries on. Returning an
	// error is a failure, which is reported and traced. Reporting a failure as
	// a decline makes a summarizer that never succeeds look identical to an
	// idle one while the prompt keeps growing on every turn.
	//
	// Usage may be reported alongside a decline, for a summarizer that spent a
	// model call and got nothing usable back. It is nil when unknown.
	//
	// ctx must be honoured. An implementation that ignores it holds the turn
	// open for as long as it runs, and cancelling the caller's context does not
	// cut it short, because the run does not return until this call does.
	// Post-invocation compaction is driven from a deferred call, so this
	// outlasts even a consumer that has stopped reading events. The framework
	// bounds the summarizer it installs by default and cannot bound one it is
	// handed, so an implementation that calls a model should carry its own
	// deadline.
	SummarizeEvents(ctx context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error)
}
