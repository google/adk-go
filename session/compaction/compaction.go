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
// prompt is built, [Apply] drops the raw events inside that range and
// materializes the summary in their place.
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
	"fmt"

	"google.golang.org/adk/v2/session"
)

// Config configures context compaction for an application.
//
// Two independent strategies are available, and enabling neither disables
// compaction entirely:
//
//   - Sliding window (CompactionInterval, OverlapSize) runs after an invocation
//     completes and summarizes whole invocations at a time.
//   - Tail retention (TokenThreshold, EventRetentionSize) runs inside an
//     invocation before a model call and summarizes everything but the most
//     recent events once the prompt grows past a token budget.
type Config struct {
	// CompactionInterval is the number of new user-initiated invocations that,
	// once fully represented in the session's events, triggers a sliding-window
	// compaction. Zero, the default, disables sliding-window compaction.
	CompactionInterval int

	// OverlapSize is how many already-compacted invocations to pull back into
	// the next sliding window, creating an overlap between consecutive
	// summaries for continuity. Only meaningful alongside CompactionInterval.
	OverlapSize int

	// TokenThreshold is the prompt token count at which intra-invocation
	// tail-retention compaction fires before a model call. Zero, the default,
	// disables tail-retention compaction.
	TokenThreshold int

	// EventRetentionSize is how many of the most recent events are kept raw
	// when tail-retention compaction fires; everything older is summarized.
	// Only meaningful alongside TokenThreshold.
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
	if c.EventRetentionSize > 0 && c.TokenThreshold == 0 {
		return fmt.Errorf("EventRetentionSize is set to %d but TokenThreshold is 0, so tail-retention compaction never runs", c.EventRetentionSize)
	}
	if !c.hasSlidingWindow() && !c.hasTailRetention() {
		return fmt.Errorf("no compaction strategy is enabled, set CompactionInterval or TokenThreshold (or leave the whole config nil to disable compaction)")
	}
	return nil
}

// Summarizer compacts a range of events into a single summary event.
//
// Implement it to control which parts of an event reach the summary and how the
// summary is produced; [LLMSummarizer] is the default implementation.
type Summarizer interface {
	// SummarizeEvents summarizes events into one new event carrying the result
	// on its Actions.Compaction field. It returns a nil event when no summary
	// was produced, which callers treat as "skip this compaction" rather than
	// as an error. The events passed in are never modified.
	SummarizeEvents(ctx context.Context, events []*session.Event) (*session.Event, error)
}

// IsCompactionEvent reports whether ev carries a context-compaction summary
// that can actually be shown to a model: it declares a compaction, and that
// compaction has content.
//
// Use it to count stored summaries, or to decide what to materialize into a
// prompt. Note that it answers "is there a usable summary here", not "is this
// event bookkeeping rather than conversation" — an event whose compaction has
// no content is still bookkeeping, and this returns false for it. Only
// [session.EventActions.Compaction] being non-nil answers the second question.
func IsCompactionEvent(ev *session.Event) bool {
	return ev != nil && ev.Actions.Compaction != nil && ev.Actions.Compaction.CompactedContent != nil
}
