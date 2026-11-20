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

// Package eventmetadata manages reserved ADK keys in session.Event.CustomMetadata
// and merges run-level custom metadata into events.
//
// ADK records its own bookkeeping in an event's CustomMetadata and reads it
// back from session history. Callers must not be able to forge those entries
// through agent.RunConfig.CustomMetadata, so the runner filters the caller's
// map with WithoutInternal and merges the result into events with
// MergeIntoEvent. This mirrors adk-python's events/_internal_metadata.py and
// runners.py _apply_run_config_custom_metadata.
package eventmetadata

import (
	"maps"
	"strings"

	"google.golang.org/adk/v2/session"
)

const (
	// InternalPrefix is the custom_metadata key prefix reserved for metadata
	// that only ADK code may write, matching adk-python's INTERNAL_METADATA_PREFIX.
	InternalPrefix = "__adk_internal_"

	// BeforeRunReplyKey marks persisted BeforeRunCallback replies that bypassed
	// agent execution.
	BeforeRunReplyKey = "adk:before_run_reply"

	// WorkflowPrefix is the custom_metadata key prefix used by workflow
	// persistence to record node outcomes, completion, delegated outputs,
	// failed child paths, and base paths in session history.
	WorkflowPrefix = "adk.workflow."
)

// IsInternalKey reports whether key is reserved for ADK's own event metadata
// and therefore must be stripped from caller-supplied RunConfig.CustomMetadata.
//
// A key is internal iff it starts with InternalPrefix ("__adk_internal_"),
// equals BeforeRunReplyKey ("adk:before_run_reply"), or starts with
// WorkflowPrefix ("adk.workflow."). Keys starting with "a2a:" are not
// internal: callers may set them, as in adk-python.
//
// IsInternalKey is a pure function and safe for concurrent use.
func IsInternalKey(key string) bool {
	return strings.HasPrefix(key, InternalPrefix) ||
		key == BeforeRunReplyKey ||
		strings.HasPrefix(key, WorkflowPrefix)
}

// WithoutInternal returns a new map containing the entries of metadata whose
// keys satisfy !IsInternalKey(k), or nil if metadata is empty or all keys are
// internal. The input map is never mutated, and the result never shares
// storage with it. Values are copied as they are, not deep-copied.
//
// WithoutInternal is safe for concurrent use provided the caller does not
// concurrently mutate metadata.
func WithoutInternal(metadata map[string]any) map[string]any {
	var kept map[string]any
	for k, v := range metadata {
		if IsInternalKey(k) {
			continue
		}
		if kept == nil {
			kept = make(map[string]any, len(metadata))
		}
		kept[k] = v
	}
	return kept
}

// WithInternalFrom returns a new map combining replacement with any reserved
// ADK-internal entries (satisfying [IsInternalKey]) from original.
//
// Preconditions:
//   - replacement and original may be nil or non-nil maps; neither may be
//     mutated concurrently during the call.
//
// Postconditions:
//   - If len(replacement) == 0 and original has no keys satisfying
//     IsInternalKey(k), returns nil without allocating.
//   - Otherwise, returns a newly allocated map[string]any containing every
//     entry of replacement overlaid with every entry (k, v) of original where
//     IsInternalKey(k) is true (original's internal keys win on conflict).
//   - Neither replacement nor original is mutated, and the returned map never
//     shares storage with original or replacement.
//
// WithInternalFrom is a pure function and safe for concurrent use.
func WithInternalFrom(replacement, original map[string]any) map[string]any {
	var out map[string]any
	if len(replacement) > 0 {
		out = maps.Clone(replacement)
	}
	for k, v := range original {
		if !IsInternalKey(k) {
			continue
		}
		if out == nil {
			out = make(map[string]any)
		}
		out[k] = v
	}
	return out
}

// MergeIntoEvent merges runMetadata (stripping any internal keys via
// IsInternalKey) into event.CustomMetadata.
//
// Postconditions:
//   - If event == nil or runMetadata has no non-internal keys,
//     MergeIntoEvent is a no-op and does not allocate or modify
//     event.CustomMetadata.
//   - Otherwise, event.CustomMetadata is replaced with a newly allocated
//     map[string]any containing every non-internal entry of runMetadata
//     overlaid with every entry of event's prior CustomMetadata (event's own
//     keys win on conflict, and event's own internal keys are preserved).
//   - Neither runMetadata nor the prior event.CustomMetadata map is
//     mutated in place.
//
// MergeIntoEvent writes event.CustomMetadata, so the caller must own event.
func MergeIntoEvent(event *session.Event, runMetadata map[string]any) {
	if event == nil {
		return
	}
	merged := WithoutInternal(runMetadata)
	if merged == nil {
		return
	}
	// merged is a fresh map, so copying the event's entries over it changes
	// neither runMetadata nor the map the event held before.
	maps.Copy(merged, event.CustomMetadata)
	event.CustomMetadata = merged
}
