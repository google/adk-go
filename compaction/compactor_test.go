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

package compaction

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// MockLLM implements model.LLM for testing.
type MockLLM struct {
	name     string
	response *genai.Content
}

func (m *MockLLM) Name() string {
	return m.name
}

func (m *MockLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.response == nil {
			m.response = &genai.Content{
				Role: "assistant",
				Parts: []*genai.Part{{
					Text: "Summary of events",
				}},
			}
		}
		yield(&model.LLMResponse{Content: m.response}, nil)
	}
}

// MockEvents implements session.Events for testing.
type MockEvents struct {
	events []*session.Event
}

func (m *MockEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, e := range m.events {
			if !yield(e) {
				return
			}
		}
	}
}

func (m *MockEvents) Len() int {
	return len(m.events)
}

func (m *MockEvents) At(i int) *session.Event {
	return m.events[i]
}

// MockSession implements session.Session for testing.
type MockSession struct {
	id     string
	events session.Events
}

func (m *MockSession) ID() string      { return m.id }
func (m *MockSession) AppName() string { return "test-app" }
func (m *MockSession) UserID() string  { return "test-user" }
func (m *MockSession) State() session.State {
	return nil // Not used in tests
}
func (m *MockSession) Events() session.Events {
	return m.events
}
func (m *MockSession) LastUpdateTime() time.Time {
	return time.Now()
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    Config
		wantError bool
	}{
		{
			name:      "disabled config is valid",
			config:    Config{Enabled: false},
			wantError: false,
		},
		{
			name: "valid enabled config",
			config: Config{
				Enabled:            true,
				CompactionInterval: 5,
				OverlapSize:        2,
			},
			wantError: false,
		},
		{
			name: "zero compaction interval",
			config: Config{
				Enabled:            true,
				CompactionInterval: 0,
				OverlapSize:        2,
			},
			wantError: true,
		},
		{
			name: "zero overlap size",
			config: Config{
				Enabled:            true,
				CompactionInterval: 5,
				OverlapSize:        0,
			},
			wantError: true,
		},
		{
			name: "overlap size >= compaction interval",
			config: Config{
				Enabled:            true,
				CompactionInterval: 5,
				OverlapSize:        5,
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestCompactorNew(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true

	tests := []struct {
		name      string
		config    Config
		llm       model.LLM
		wantError bool
	}{
		{
			name:      "invalid config",
			config:    Config{Enabled: true, CompactionInterval: 0},
			llm:       &MockLLM{name: "test"},
			wantError: true,
		},
		{
			name:      "nil llm",
			config:    config,
			llm:       nil,
			wantError: true,
		},
		{
			name:      "valid compactor",
			config:    config,
			llm:       &MockLLM{name: "test"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.config, tt.llm)
			if (err != nil) != tt.wantError {
				t.Errorf("New() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestShouldCompact(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 3,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	tests := []struct {
		name              string
		events            []*session.Event
		wantShouldCompact bool
	}{
		{
			name:              "empty events",
			events:            []*session.Event{},
			wantShouldCompact: false,
		},
		{
			name: "fewer invocations than threshold",
			events: []*session.Event{
				createTestEvent("inv1", "user", 100),
				createTestEvent("inv1", "model", 101),
				createTestEvent("inv2", "user", 102),
			},
			wantShouldCompact: false,
		},
		{
			name: "exactly threshold invocations",
			events: []*session.Event{
				createTestEvent("inv1", "user", 100),
				createTestEvent("inv2", "user", 101),
				createTestEvent("inv3", "user", 102),
			},
			wantShouldCompact: true,
		},
		{
			name: "more than threshold invocations",
			events: []*session.Event{
				createTestEvent("inv1", "user", 100),
				createTestEvent("inv2", "user", 101),
				createTestEvent("inv3", "user", 102),
				createTestEvent("inv4", "user", 103),
			},
			wantShouldCompact: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldCompact, _ := compactor.shouldCompact(tt.events)
			if shouldCompact != tt.wantShouldCompact {
				t.Errorf("shouldCompact() = %v, want %v", shouldCompact, tt.wantShouldCompact)
			}
		})
	}
}

func TestSelectEventsToCompact(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 3,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv1", "model", 101),
		createTestEvent("inv2", "user", 102),
		createTestEvent("inv2", "model", 103),
		createTestEvent("inv3", "user", 104),
		createTestEvent("inv3", "model", 105),
	}

	selected, startTime, endTime := compactor.selectEventsToCompact(events, 0)

	// Should compact inv1-inv2 (3 invocations - 1 overlap = 2 to compact)
	if len(selected) == 0 {
		t.Errorf("expected events to be selected, got none")
	}

	// Start and end times should be valid
	if startTime >= endTime {
		t.Errorf("startTime (%v) should be < endTime (%v)", startTime, endTime)
	}

	// Verify the compacted range doesn't include events from inv3 (the overlap invocation)
	for _, evt := range selected {
		if evt.InvocationID == "inv3" {
			t.Errorf("inv3 should not be in compacted range (it's the overlap)")
		}
	}
}

func TestMaybeCompact(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 2,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv2", "user", 101),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: events},
	}

	ctx := context.Background()
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "test-invocation")

	if err != nil {
		t.Errorf("MaybeCompact() error = %v", err)
	}

	if compactionEvent == nil {
		t.Errorf("expected compaction event, got nil")
	}

	if compactionEvent.Actions.Compaction == nil {
		t.Errorf("expected compaction event to have Actions.Compaction set")
	}

	if compactionEvent.Author != "system" {
		t.Errorf("expected author to be 'system', got %q", compactionEvent.Author)
	}
}

func TestEventCompactionStructure(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 2,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	mockContent := &genai.Content{
		Role: "assistant",
		Parts: []*genai.Part{{
			Text: "Test summary",
		}},
	}
	compactor.llm.(*MockLLM).response = mockContent

	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv2", "user", 101),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: events},
	}

	ctx := context.Background()
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "test-invocation")

	if err != nil {
		t.Errorf("MaybeCompact() error = %v", err)
	}

	if compactionEvent != nil {
		ec := compactionEvent.Actions.Compaction
		if ec == nil {
			t.Fatal("expected compaction event to have Actions.Compaction set")
		}

		if ec.StartTimestamp == 0 || ec.EndTimestamp == 0 {
			t.Errorf("timestamps should be non-zero: start=%v, end=%v", ec.StartTimestamp, ec.EndTimestamp)
		}

		if ec.StartTimestamp > ec.EndTimestamp {
			t.Errorf("start time should be <= end time: %v > %v", ec.StartTimestamp, ec.EndTimestamp)
		}

		if ec.CompactedContent == nil {
			t.Errorf("expected CompactedContent to be set")
		}
	}
}

// Helper function to create test events
func createTestEvent(invID, author string, unixSeconds int64) *session.Event {
	event := &session.Event{
		ID:           "test-id",
		InvocationID: invID,
		Timestamp:    time.Unix(unixSeconds, 0),
		Author:       author,
		Actions:      session.EventActions{StateDelta: make(map[string]any)},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: author,
				Parts: []*genai.Part{{
					Text: "Test event content",
				}},
			},
		},
	}
	return event
}
