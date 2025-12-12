// Copyright 2025 Google LLC
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

package session

import "google.golang.org/genai"

// EventCompaction represents a compacted summary of a range of events in the session.
// It is stored as part of EventActions to preserve the conversation history
// in a compressed form for long-running sessions.
//
// The compacted events are replaced by this summary in the LLM context,
// while remaining in the database for full audit trail purposes.
type EventCompaction struct {
	// StartTimestamp is the timestamp (in seconds) of the first event in the compacted range.
	// This allows filtering to identify which events are included in this compaction.
	StartTimestamp float64

	// EndTimestamp is the timestamp (in seconds) of the last event in the compacted range.
	// Combined with StartTimestamp, this defines the time window of compacted events.
	EndTimestamp float64

	// CompactedContent is the LLM-generated summary of the compacted events.
	// It contains the key information from the original events compressed using an LLM
	// to maintain context while reducing token count by 60-80%.
	CompactedContent *genai.Content
}
