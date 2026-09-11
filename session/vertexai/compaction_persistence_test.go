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

package vertexai

import (
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// A context-compaction summary carries its content only on
// Actions.Compaction. LLMResponse.Content is nil, and there is no state or
// artifact delta, so the structured actions column stays empty too. raw_event
// is therefore the only slot that can hold it, and eventNeedsRawEvent is what
// decides whether raw_event gets written at all.
//
// When it did not list Compaction the summary reached no slot: the backend
// stored an effectively empty event, and on reload the session held neither the
// summary nor any record that compaction had run, so the same range was
// summarized and billed again on every later trigger.
//
// These run offline because the replay-based suite needs a recording that only
// a live Agent Engine project can produce.
func TestEventNeedsRawEventForCompaction(t *testing.T) {
	t.Parallel()

	compactionEvent := func() *session.Event {
		return &session.Event{
			ID:           "summary",
			Author:       "user",
			InvocationID: "inv-compaction",
			Actions: session.EventActions{
				Compaction: &session.EventCompaction{
					StartTimestamp:   time.Unix(1, 0).UTC(),
					EndTimestamp:     time.Unix(5, 0).UTC(),
					CompactedContent: genai.NewContentFromText("summary of earlier turns", "model"),
				},
			},
		}
	}

	if !eventNeedsRawEvent(compactionEvent()) {
		t.Error("eventNeedsRawEvent() = false for a compaction summary, so it would be written nowhere and lost on reload")
	}

	// An ordinary event must still stay on the legacy wire format, so existing
	// replay recordings remain valid.
	plain := &session.Event{ID: "plain", Author: "user", InvocationID: "inv1"}
	if eventNeedsRawEvent(plain) {
		t.Error("eventNeedsRawEvent() = true for a plain event, which would change the wire format for unrelated events")
	}
}

func TestCompactionRoundTripsThroughRawEvent(t *testing.T) {
	t.Parallel()

	start := time.Unix(1, 0).UTC()
	end := time.Unix(5, 0).UTC()
	want := &session.Event{
		ID:           "summary",
		Author:       "user",
		InvocationID: "inv-compaction",
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   start,
				EndTimestamp:     end,
				CompactedContent: genai.NewContentFromText("summary of earlier turns", "model"),
			},
		},
	}

	raw, err := eventToRawEvent(want)
	if err != nil {
		t.Fatalf("eventToRawEvent() error = %v", err)
	}
	got, err := eventFromRawEvent(raw)
	if err != nil {
		t.Fatalf("eventFromRawEvent() error = %v", err)
	}

	c := got.Actions.Compaction
	if c == nil {
		t.Fatal("Actions.Compaction did not survive the raw_event round trip")
	}
	if !c.StartTimestamp.Equal(start) || !c.EndTimestamp.Equal(end) {
		t.Errorf("range = [%v, %v], want [%v, %v]", c.StartTimestamp, c.EndTimestamp, start, end)
	}
	if c.CompactedContent == nil || len(c.CompactedContent.Parts) == 0 {
		t.Fatal("compacted content did not survive the round trip")
	}
	if got, want := c.CompactedContent.Parts[0].Text, "summary of earlier turns"; got != want {
		t.Errorf("compacted content = %q, want %q", got, want)
	}
}
