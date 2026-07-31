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

	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// HasSlidingWindow reports whether sliding-window compaction is enabled.
//
// This lives here rather than as a method on compaction.Config because nothing
// outside the framework needs to ask, and keeping it off the public type leaves
// users with just the fields they set.
func HasSlidingWindow(cfg *compaction.Config) bool {
	return cfg != nil && cfg.CompactionInterval > 0
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
func SlidingWindow(ctx context.Context, cfg *compaction.Config, sess session.Session) (*session.Event, error) {
	if !HasSlidingWindow(cfg) {
		return nil, nil
	}
	if cfg.Summarizer == nil {
		return nil, fmt.Errorf("no Summarizer configured")
	}
	if sess == nil {
		return nil, nil
	}

	events := collect(sess)
	window := selectSlidingWindow(events, cfg.CompactionInterval, cfg.OverlapSize)
	if len(window) == 0 {
		return nil, nil
	}

	summary, err := cfg.Summarizer.SummarizeEvents(ctx, window)
	if err != nil {
		return nil, fmt.Errorf("sliding-window summarization failed: %w", err)
	}
	return stamp(ctx, summary), nil
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
