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

// Package compaction implements runner-side event compaction. It mirrors
// adk-python's apps/compaction.py.
//
// Two triggers are supported:
//
//   - Token-threshold: when the most recently observed prompt token count
//     reaches EventsCompactionConfig.TokenThreshold, compact everything older
//     than the last EventRetentionSize raw events.
//
//   - Sliding window: every CompactionInterval new user-initiated invocations,
//     compact events spanning from OverlapSize invocations before the new
//     block through the last invocation.
//
// MaybeRun is the main entry point. It is invoked by the runner after each
// invocation finishes. Token-threshold takes precedence over sliding window
// for the same turn.
package compaction

import (
	"context"
	"errors"
	"time"

	"google.golang.org/adk/session"
)

// Summarizer is the contract MaybeRun calls into. The app package defines
// EventsSummarizer with the same shape; passing one as the other is a
// straight assignment because Go interfaces are structural.
type Summarizer interface {
	MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error)
}

// MaybeRunInput carries the runtime data MaybeRun needs. Fields mirror
// app.EventsCompactionConfig but are accepted as primitives so this
// internal package doesn't need to import app (which would create an import
// cycle: app → plugin → llmagent → llminternal → compaction).
type MaybeRunInput struct {
	Summarizer         Summarizer
	CompactionInterval int
	OverlapSize        int
	TokenThreshold     *int
	EventRetentionSize *int

	Session        session.Session
	SessionService session.Service
	AppName        string
	UserID         string
	SessionID      string
	// CurrentBranch is the agent branch active when the compaction fires;
	// used by the prompt-token estimator.
	CurrentBranch string
	// AgentName is forwarded to the summarizer when present.
	AgentName string
}

// MaybeRun executes one round of post-invocation compaction. It returns
// (true, nil) when a compaction event was appended to the session, (false,
// nil) when no compaction was warranted, or (false, err) on a service error.
//
// The token-threshold trigger is attempted first when fully configured; if
// it doesn't fire, the sliding-window trigger is attempted.
func MaybeRun(ctx context.Context, in MaybeRunInput) (bool, error) {
	if in.Session == nil || in.SessionService == nil {
		return false, nil
	}
	if !in.hasAnyTrigger() {
		return false, nil
	}
	if in.Summarizer == nil {
		return false, errors.New("compaction: Summarizer is nil")
	}

	events := collectEvents(in.Session)
	if len(events) == 0 {
		return false, nil
	}

	if in.hasTokenThresholdConfig() {
		ok, err := runTokenThreshold(ctx, in, events)
		if err != nil {
			return false, err
		}
		if ok {
			return true, nil
		}
	}
	if in.hasSlidingWindowConfig() {
		ok, err := runSlidingWindow(ctx, in, events)
		if err != nil {
			return false, err
		}
		return ok, nil
	}
	return false, nil
}

func (in MaybeRunInput) hasAnyTrigger() bool {
	return in.hasTokenThresholdConfig() || in.hasSlidingWindowConfig()
}

func (in MaybeRunInput) hasTokenThresholdConfig() bool {
	return in.TokenThreshold != nil && in.EventRetentionSize != nil
}

func (in MaybeRunInput) hasSlidingWindowConfig() bool {
	return in.CompactionInterval > 0
}

// collectEvents pulls events out of the session into a slice for indexed
// scanning. Cheap; avoids re-iterating the iterator.
func collectEvents(sess session.Session) []*session.Event {
	if sess == nil {
		return nil
	}
	evs := sess.Events()
	if evs == nil {
		return nil
	}
	out := make([]*session.Event, 0, evs.Len())
	for e := range evs.All() {
		out = append(out, e)
	}
	return out
}

// validCompaction is a row in the working set used during compaction-event
// scanning. It captures the event's index along with the start/end of its
// compaction range for easy range-comparison.
type validCompaction struct {
	index int
	start time.Time
	end   time.Time
	event *session.Event
}

// validCompactions returns all events whose Actions.Compaction has both
// timestamps and a non-nil compacted content. Mirrors _valid_compactions.
func validCompactions(events []*session.Event) []validCompaction {
	var out []validCompaction
	for i, ev := range events {
		c := ev.Actions.Compaction
		if c == nil || c.CompactedContent == nil {
			continue
		}
		if c.StartTimestamp.IsZero() || c.EndTimestamp.IsZero() {
			continue
		}
		out = append(out, validCompaction{i, c.StartTimestamp, c.EndTimestamp, ev})
	}
	return out
}

// isCompactionSubsumed reports whether row's range is fully contained by
// another compaction in the set. Mirrors _is_compaction_subsumed: ties on
// identical ranges treat the earlier-indexed event as subsumed.
func isCompactionSubsumed(row validCompaction, all []validCompaction) bool {
	for _, other := range all {
		if other.index == row.index {
			continue
		}
		if !other.start.After(row.start) && !other.end.Before(row.end) {
			if other.start.Before(row.start) ||
				other.end.After(row.end) ||
				other.index > row.index {
				return true
			}
		}
	}
	return false
}

// latestCompactionEvent returns the latest non-subsumed compaction event in
// stream order. Mirrors _latest_compaction_event.
func latestCompactionEvent(events []*session.Event) *session.Event {
	rows := validCompactions(events)
	latestIdx := -1
	var latest *session.Event
	for _, r := range rows {
		if isCompactionSubsumed(r, rows) {
			continue
		}
		if r.index > latestIdx {
			latestIdx = r.index
			latest = r.event
		}
	}
	return latest
}

// latestCompactionEnd returns the EndTimestamp of the most recent
// non-subsumed compaction, or the zero time when none exists.
func latestCompactionEnd(events []*session.Event) time.Time {
	ev := latestCompactionEvent(events)
	if ev == nil || ev.Actions.Compaction == nil {
		return time.Time{}
	}
	return ev.Actions.Compaction.EndTimestamp
}

// latestPromptTokenCount returns the most recently observed prompt token
// count from session events. Falls back to a character-based estimate when
// no usage metadata is available, matching Python's behavior at chars/4.
//
// Returns (count, true) when a value is available.
func latestPromptTokenCount(events []*session.Event) (int, bool) {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.UsageMetadata != nil && ev.UsageMetadata.PromptTokenCount > 0 {
			return int(ev.UsageMetadata.PromptTokenCount), true
		}
	}
	// Character-count fallback: count text chars across non-compaction events.
	chars := 0
	for _, ev := range events {
		if ev.Actions.Compaction != nil {
			continue
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			chars += len(p.Text)
		}
	}
	if chars <= 0 {
		return 0, false
	}
	return chars / 4, true
}

// eventFunctionCallIDs returns the set of function call IDs in ev.
func eventFunctionCallIDs(ev *session.Event) map[string]struct{} {
	out := map[string]struct{}{}
	if ev.Content == nil {
		return out
	}
	for _, p := range ev.Content.Parts {
		if p == nil || p.FunctionCall == nil {
			continue
		}
		if id := p.FunctionCall.ID; id != "" {
			out[id] = struct{}{}
		}
	}
	return out
}

// eventFunctionResponseIDs returns the set of function response IDs in ev.
func eventFunctionResponseIDs(ev *session.Event) map[string]struct{} {
	out := map[string]struct{}{}
	if ev.Content == nil {
		return out
	}
	for _, p := range ev.Content.Parts {
		if p == nil || p.FunctionResponse == nil {
			continue
		}
		if id := p.FunctionResponse.ID; id != "" {
			out[id] = struct{}{}
		}
	}
	return out
}

// pendingFunctionCallIDs returns the set of call IDs that have no matching
// response in events.
func pendingFunctionCallIDs(events []*session.Event) map[string]struct{} {
	calls := map[string]struct{}{}
	resps := map[string]struct{}{}
	for _, ev := range events {
		for id := range eventFunctionCallIDs(ev) {
			calls[id] = struct{}{}
		}
		for id := range eventFunctionResponseIDs(ev) {
			resps[id] = struct{}{}
		}
	}
	for id := range resps {
		delete(calls, id)
	}
	return calls
}

// truncateBeforePendingCall returns the leading events from input that
// don't reference any pending call ID.
func truncateBeforePendingCall(input []*session.Event, pending map[string]struct{}) []*session.Event {
	for i, ev := range input {
		callIDs := eventFunctionCallIDs(ev)
		if len(callIDs) == 0 {
			continue
		}
		for id := range callIDs {
			if _, has := pending[id]; has {
				return input[:i]
			}
		}
	}
	return input
}

// safeTokenCompactionSplitIndex picks an index <= initialSplit such that the
// retained tail (input[index:]) does not contain function responses whose
// matching calls are in the prefix (input[:index]).
//
// Iterates backwards once, maintaining unmatched-response IDs.
func safeTokenCompactionSplitIndex(input []*session.Event, retentionSize int) int {
	initial := len(input) - retentionSize
	if initial <= 0 {
		return 0
	}
	unmatched := map[string]struct{}{}
	best := 0
	for i := len(input) - 1; i >= 0; i-- {
		ev := input[i]
		for id := range eventFunctionResponseIDs(ev) {
			unmatched[id] = struct{}{}
		}
		for id := range eventFunctionCallIDs(ev) {
			delete(unmatched, id)
		}
		if len(unmatched) == 0 && i <= initial {
			best = i
			break
		}
	}
	return best
}

// eventsToCompactForTokenThreshold collects token-threshold compaction
// candidates with rolling-summary seed.
func eventsToCompactForTokenThreshold(events []*session.Event, retentionSize int) []*session.Event {
	latestEv := latestCompactionEvent(events)
	lastEnd := latestCompactionEnd(events)

	var candidates []*session.Event
	for _, ev := range events {
		if ev.Actions.Compaction != nil {
			continue
		}
		if !ev.Timestamp.After(lastEnd) {
			continue
		}
		candidates = append(candidates, ev)
	}
	if len(candidates) <= retentionSize {
		return nil
	}

	var toCompact []*session.Event
	if retentionSize == 0 {
		toCompact = candidates
	} else {
		split := safeTokenCompactionSplitIndex(candidates, retentionSize)
		toCompact = candidates[:split]
	}
	pending := pendingFunctionCallIDs(events)
	toCompact = truncateBeforePendingCall(toCompact, pending)
	if len(toCompact) == 0 {
		return nil
	}

	// If a previous compaction exists, prepend its summary as a synthetic
	// seed so the next summary supersedes it.
	if latestEv != nil && latestEv.Actions.Compaction != nil &&
		!latestEv.Actions.Compaction.StartTimestamp.IsZero() &&
		latestEv.Actions.Compaction.CompactedContent != nil {
		seed := session.NewEvent(latestEv.InvocationID + ":seed")
		seed.Author = "model"
		seed.Branch = latestEv.Branch
		seed.Timestamp = latestEv.Actions.Compaction.StartTimestamp
		seed.Content = latestEv.Actions.Compaction.CompactedContent
		toCompact = append([]*session.Event{seed}, toCompact...)
	}
	return toCompact
}

// runTokenThreshold returns true if the token-threshold trigger fired and a
// compaction event was appended.
func runTokenThreshold(ctx context.Context, in MaybeRunInput, events []*session.Event) (bool, error) {
	tokens, ok := latestPromptTokenCount(events)
	if !ok || tokens < *in.TokenThreshold {
		return false, nil
	}
	toCompact := eventsToCompactForTokenThreshold(events, *in.EventRetentionSize)
	if len(toCompact) == 0 {
		return false, nil
	}
	return summarizeAndAppend(ctx, in, toCompact)
}

// runSlidingWindow returns true if the sliding-window trigger fired and a
// compaction event was appended.
func runSlidingWindow(ctx context.Context, in MaybeRunInput, events []*session.Event) (bool, error) {
	lastEnd := latestCompactionEnd(events)

	type invInfo struct {
		latest time.Time
	}
	infos := map[string]*invInfo{}
	var order []string
	for _, ev := range events {
		if ev.Actions.Compaction != nil || ev.InvocationID == "" {
			continue
		}
		if info, ok := infos[ev.InvocationID]; ok {
			if ev.Timestamp.After(info.latest) {
				info.latest = ev.Timestamp
			}
		} else {
			infos[ev.InvocationID] = &invInfo{latest: ev.Timestamp}
			order = append(order, ev.InvocationID)
		}
	}

	var newIDs []string
	for _, id := range order {
		if infos[id].latest.After(lastEnd) {
			newIDs = append(newIDs, id)
		}
	}
	if len(newIDs) < in.CompactionInterval {
		return false, nil
	}

	endID := newIDs[len(newIDs)-1]
	firstNewID := newIDs[0]
	firstNewIdx := indexOf(order, firstNewID)
	startIdx := firstNewIdx - in.OverlapSize
	if startIdx < 0 {
		startIdx = 0
	}
	startID := order[startIdx]

	// Find first event with InvocationID == startID and last event with
	// InvocationID == endID.
	firstEventIdx := -1
	for i, ev := range events {
		if ev.InvocationID == startID {
			firstEventIdx = i
			break
		}
	}
	lastEventIdx := -1
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].InvocationID == endID {
			lastEventIdx = i
			break
		}
	}
	if firstEventIdx < 0 || lastEventIdx < firstEventIdx {
		return false, nil
	}

	span := events[firstEventIdx : lastEventIdx+1]
	var toCompact []*session.Event
	for _, ev := range span {
		if ev.Actions.Compaction == nil {
			toCompact = append(toCompact, ev)
		}
	}
	pending := pendingFunctionCallIDs(events)
	toCompact = truncateBeforePendingCall(toCompact, pending)
	if len(toCompact) == 0 {
		return false, nil
	}
	return summarizeAndAppend(ctx, in, toCompact)
}

// summarizeAndAppend calls the configured summarizer and appends the result
// to the session.
func summarizeAndAppend(ctx context.Context, in MaybeRunInput, toCompact []*session.Event) (bool, error) {
	out, err := in.Summarizer.MaybeSummarize(ctx, toCompact)
	if err != nil {
		return false, err
	}
	if out == nil {
		return false, nil
	}
	if err := in.SessionService.AppendEvent(ctx, in.Session, out); err != nil {
		return false, err
	}
	return true, nil
}

func indexOf(slice []string, v string) int {
	for i, s := range slice {
		if s == v {
			return i
		}
	}
	return -1
}
