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

// Package app provides the App container — the top-level grouping of an
// agentic application that pairs a root agent (or, in v2 Phase 2+, a root
// workflow node) with shared plugins and runtime configuration.
//
// Mirrors adk-python's google.adk.apps.App. App is consumed by the runner;
// see runner.Config.App.
package app

import (
	"context"
	"errors"
	"fmt"
	"unicode"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
)

// App is the top-level container for an agentic application.
//
// It carries either a root agent or (post Phase 2) a root workflow node, plus
// the application-wide plugins and runtime configurations (event compaction,
// context caching, resumability).
//
// At least RootAgent must be set in Phase 1. Once the workflow package
// lands, RootNode will be added as an alternative.
type App struct {
	// Name is the application identifier. Must be a valid Go identifier and
	// must not be the reserved word "user".
	Name string

	// RootAgent is the entry-point agent. Required in Phase 1. (Phase 2 adds
	// the alternative RootNode field for workflow-rooted apps.)
	RootAgent agent.Agent

	// Plugins are application-wide and run for every invocation in this app.
	Plugins []*plugin.Plugin

	// EventsCompactionConfig optionally enables session-event compaction so
	// long conversations don't overflow the model context window. See the
	// EventsCompactionConfig docs.
	EventsCompactionConfig *EventsCompactionConfig

	// ContextCacheConfig optionally configures prompt caching for LLM calls
	// originating from agents in this app.
	ContextCacheConfig *ContextCacheConfig

	// ResumabilityConfig optionally enables resumable invocations across this
	// app's agents.
	ResumabilityConfig *ResumabilityConfig
}

// ResumabilityConfig configures whether agents in the app support pausing on
// long-running calls and resuming from session events.
//
// Resume is best-effort: tool calls must be idempotent because at-least-once
// semantics apply, and any temporary in-memory state is lost.
type ResumabilityConfig struct {
	// IsResumable enables resume for all agents in the app.
	IsResumable bool
}

// ContextCacheConfig is a placeholder shape for prompt-cache configuration.
// Concrete fields land alongside the LLM-side cache plumbing.
type ContextCacheConfig struct {
	// TTL is the cache time-to-live; zero means provider default.
	TTL int64
	// CachedContentNameOverride lets callers pin a server-side cache name.
	CachedContentNameOverride string
	// MinTokensToCache hints when the runner should bother caching.
	MinTokensToCache int
}

// EventsCompactionConfig configures runner-side event compaction. Two
// triggers are supported and the runner applies them after each invocation:
//
//  1. Sliding window: every CompactionInterval new user invocations, the
//     runner asks the Summarizer to summarize the events spanning from
//     OverlapSize invocations before the new block through the latest one.
//
//  2. Token-threshold: if TokenThreshold is set and the most recently
//     observed prompt token count meets or exceeds it, the runner compacts
//     everything older than the last EventRetentionSize raw events. Takes
//     precedence over sliding window for the same invocation.
//
// Mirrors apps/app.py:EventsCompactionConfig.
type EventsCompactionConfig struct {
	// Summarizer produces the synthesized summary event. Required at runtime.
	// (The interface is defined in this package; see EventsSummarizer.)
	Summarizer EventsSummarizer

	// CompactionInterval is the number of new user-initiated invocations
	// that, once fully represented in the session, trigger a compaction.
	// Required (>0).
	CompactionInterval int

	// OverlapSize is the number of preceding invocations to overlap with the
	// previous compaction range so context isn't lost across summaries.
	// Required (>=0).
	OverlapSize int

	// TokenThreshold, when set, enables the token-based trigger. Must be set
	// together with EventRetentionSize.
	TokenThreshold *int

	// EventRetentionSize, when set, controls how many of the most recent raw
	// events are kept un-compacted on a token-based compaction. Must be set
	// together with TokenThreshold.
	EventRetentionSize *int
}

// EventsSummarizer is the runner-facing contract for producing a compacted
// summary event from a range of session events.
//
// MaybeSummarize returns nil (without error) if the implementation decides
// no summary is warranted (e.g. trivial range), in which case the runner
// skips the compaction event for this turn. The returned Event must have
// Actions.Compaction populated with the start/end timestamps and synthesized
// content; the runner appends it to the session as-is.
type EventsSummarizer interface {
	MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error)
}

// New constructs and validates an App. Returns an error if the name is not a
// valid identifier, RootAgent is nil, or any of the configs are inconsistent.
func New(a App) (*App, error) {
	if err := validateAppName(a.Name); err != nil {
		return nil, err
	}
	if a.RootAgent == nil {
		return nil, errors.New("app: RootAgent must be provided")
	}
	if a.EventsCompactionConfig != nil {
		if err := a.EventsCompactionConfig.validate(); err != nil {
			return nil, fmt.Errorf("app: invalid EventsCompactionConfig: %w", err)
		}
	}
	out := a
	return &out, nil
}

// validate enforces the runtime invariants documented on each field.
func (c *EventsCompactionConfig) validate() error {
	if c.CompactionInterval <= 0 {
		return errors.New("CompactionInterval must be > 0")
	}
	if c.OverlapSize < 0 {
		return errors.New("OverlapSize must be >= 0")
	}
	if (c.TokenThreshold == nil) != (c.EventRetentionSize == nil) {
		return errors.New("TokenThreshold and EventRetentionSize must be set together")
	}
	if c.TokenThreshold != nil && *c.TokenThreshold <= 0 {
		return errors.New("TokenThreshold must be > 0 when set")
	}
	if c.EventRetentionSize != nil && *c.EventRetentionSize < 0 {
		return errors.New("EventRetentionSize must be >= 0 when set")
	}
	return nil
}

// validateAppName rejects names that are not valid Go identifiers (or the
// reserved "user" name).
func validateAppName(name string) error {
	if name == "" {
		return errors.New("app: Name must not be empty")
	}
	if name == "user" {
		return errors.New("app: Name cannot be 'user'; reserved for end-user input")
	}
	for i, r := range name {
		if i == 0 {
			if !unicode.IsLetter(r) && r != '_' {
				return fmt.Errorf("app: Name %q must start with a letter or underscore", name)
			}
			continue
		}
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
			return fmt.Errorf("app: Name %q must contain only letters, digits, and underscores", name)
		}
	}
	return nil
}
