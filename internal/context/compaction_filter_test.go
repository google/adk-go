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

package context

import (
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// MockEventsForFilter implements session.Events for testing the filter.
type MockEventsForFilter struct {
	events []*session.Event
}

func (m *MockEventsForFilter) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, e := range m.events {
			if !yield(e) {
				return
			}
		}
	}
}

func (m *MockEventsForFilter) Len() int {
	return len(m.events)
}

func (m *MockEventsForFilter) At(i int) *session.Event {
	return m.events[i]
}

func TestFilterEventsForLLM_NoCompactions(t *testing.T) {
	// Test with events that have no compactions - all should be included
	events := &MockEventsForFilter{
		events: []*session.Event{
			createFilterTestEvent("user", 100),
			createFilterTestEvent("model", 101),
			createFilterTestEvent("user", 102),
			createFilterTestEvent("model", 103),
		},
	}

	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(events) {
		filtered = append(filtered, event)
	}

	if len(filtered) != 4 {
		t.Errorf("expected 4 events, got %d", len(filtered))
	}
}

func TestFilterEventsForLLM_WithCompactions(t *testing.T) {
	// Create events with a compaction in the middle
	events := []*session.Event{
		createFilterTestEvent("user", 100),
		createFilterTestEvent("model", 101),
		createFilterTestEvent("user", 102),
		createFilterTestEvent("model", 103),
		// Compaction event covering 100-103
		createCompactionTestEvent(100, 103, 104),
		createFilterTestEvent("user", 105),
		createFilterTestEvent("model", 106),
	}

	mockEvents := &MockEventsForFilter{events: events}

	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(mockEvents) {
		filtered = append(filtered, event)
	}

	// Should have:
	// - 1 compaction event (at time 104)
	// - 2 events after compaction (at times 105, 106)
	// Total: 3 events
	if len(filtered) != 3 {
		t.Errorf("expected 3 filtered events, got %d", len(filtered))
	}

	// Verify the compaction event is included
	hasCompaction := false
	for _, event := range filtered {
		if event.Actions.Compaction != nil {
			hasCompaction = true
			break
		}
	}
	if !hasCompaction {
		t.Errorf("expected compaction event to be included")
	}

	// Verify compacted events are excluded
	for _, event := range filtered {
		if event.Actions.Compaction == nil && event.Timestamp.Unix() >= 100 && event.Timestamp.Unix() <= 103 {
			t.Errorf("event at time %d should have been excluded (within compaction range)", event.Timestamp.Unix())
		}
	}
}

func TestFilterEventsForLLM_MultipleCompactions(t *testing.T) {
	// Test with multiple compactions covering different ranges
	events := []*session.Event{
		createFilterTestEvent("user", 100),
		createFilterTestEvent("model", 101),
		// First compaction covers 100-101
		createCompactionTestEvent(100, 101, 102),
		createFilterTestEvent("user", 103),
		createFilterTestEvent("model", 104),
		// Second compaction covers 103-104
		createCompactionTestEvent(103, 104, 105),
		createFilterTestEvent("user", 106),
		createFilterTestEvent("model", 107),
	}

	mockEvents := &MockEventsForFilter{events: events}

	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(mockEvents) {
		filtered = append(filtered, event)
	}

	// Should have:
	// - 2 compaction events (at 102 and 105)
	// - 2 events after all compactions (at 106, 107)
	// Total: 4 events
	if len(filtered) != 4 {
		t.Errorf("expected 4 filtered events, got %d", len(filtered))
	}

	// Verify we have exactly 2 compaction events
	compactionCount := 0
	for _, event := range filtered {
		if event.Actions.Compaction != nil {
			compactionCount++
		}
	}
	if compactionCount != 2 {
		t.Errorf("expected 2 compaction events, got %d", compactionCount)
	}
}

func TestFilterEventsForLLM_OverlappingCompactions(t *testing.T) {
	// Test with overlapping compaction ranges (edge case)
	events := []*session.Event{
		createFilterTestEvent("user", 100),
		createFilterTestEvent("model", 101),
		createFilterTestEvent("user", 102),
		createFilterTestEvent("model", 103),
		// Compaction covering 100-102
		createCompactionTestEvent(100, 102, 104),
		createFilterTestEvent("user", 105),
		// Compaction covering 101-105 (overlaps with previous)
		createCompactionTestEvent(101, 105, 106),
		createFilterTestEvent("user", 107),
	}

	mockEvents := &MockEventsForFilter{events: events}

	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(mockEvents) {
		filtered = append(filtered, event)
	}

	// Should have:
	// - 2 compaction events (at 104 and 106)
	// - 1 event after all compactions (at 107)
	// Events 100-105 are all within at least one compaction range
	// Total: 3 events
	if len(filtered) != 3 {
		t.Errorf("expected 3 filtered events, got %d", len(filtered))
	}

	// Verify we have exactly 2 compaction events
	compactionCount := 0
	for _, event := range filtered {
		if event.Actions.Compaction != nil {
			compactionCount++
		}
	}
	if compactionCount != 2 {
		t.Errorf("expected 2 compaction events, got %d", compactionCount)
	}
}

func TestFilterEventsForLLM_EmptyEvents(t *testing.T) {
	events := &MockEventsForFilter{events: []*session.Event{}}

	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(events) {
		filtered = append(filtered, event)
	}

	if len(filtered) != 0 {
		t.Errorf("expected 0 events for empty input, got %d", len(filtered))
	}
}

func TestFilterEventsForLLM_NilEvents(t *testing.T) {
	filtered := make([]*session.Event, 0)
	for event := range FilterEventsForLLM(nil) {
		filtered = append(filtered, event)
	}

	if len(filtered) != 0 {
		t.Errorf("expected 0 events for nil input, got %d", len(filtered))
	}
}

func TestCompactionRange_IsWithinRange(t *testing.T) {
	tests := []struct {
		name      string
		eventTime float64
		start     float64
		end       float64
		want      bool
	}{
		{
			name:      "event before range",
			eventTime: 99,
			start:     100,
			end:       105,
			want:      false,
		},
		{
			name:      "event at start boundary",
			eventTime: 100,
			start:     100,
			end:       105,
			want:      true,
		},
		{
			name:      "event in middle",
			eventTime: 102,
			start:     100,
			end:       105,
			want:      true,
		},
		{
			name:      "event at end boundary",
			eventTime: 105,
			start:     100,
			end:       105,
			want:      true,
		},
		{
			name:      "event after range",
			eventTime: 106,
			start:     100,
			end:       105,
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := createFilterTestEvent("user", int64(tt.eventTime))
			ranges := []*CompactionRange{
				{
					StartTimestamp: tt.start,
					EndTimestamp:   tt.end,
				},
			}

			result := isEventWithinCompactionRange(event, ranges)
			if result != tt.want {
				t.Errorf("isEventWithinCompactionRange() = %v, want %v", result, tt.want)
			}
		})
	}
}

// Helper functions

func createFilterTestEvent(author string, unixSeconds int64) *session.Event {
	return &session.Event{
		ID:           "test-id",
		InvocationID: "test-inv",
		Timestamp:    time.Unix(unixSeconds, 0),
		Author:       author,
		Actions:      session.EventActions{StateDelta: make(map[string]any)},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: author,
				Parts: []*genai.Part{{
					Text: "Test content",
				}},
			},
		},
	}
}

func createCompactionTestEvent(startTime, endTime, eventTime int64) *session.Event {
	return &session.Event{
		ID:           "compaction-id",
		InvocationID: "test-inv",
		Timestamp:    time.Unix(eventTime, 0),
		Author:       "system",
		Actions: session.EventActions{
			StateDelta: make(map[string]any),
			Compaction: &session.EventCompaction{
				StartTimestamp: float64(startTime),
				EndTimestamp:   float64(endTime),
				CompactedContent: &genai.Content{
					Role: "assistant",
					Parts: []*genai.Part{{
						Text: "Compacted summary",
					}},
				},
			},
		},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "assistant",
				Parts: []*genai.Part{{
					Text: "Session history compacted",
				}},
			},
		},
	}
}
