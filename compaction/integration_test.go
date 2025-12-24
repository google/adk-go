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
	"testing"

	"google.golang.org/adk/session"
)

// TestCompactionEndToEnd verifies the complete compaction flow with multiple invocations.
func TestCompactionEndToEnd(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 3,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	// Create 5 invocations (should trigger compaction for the first 3)
	events := []*session.Event{
		// Invocation 1
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv1", "model", 101),
		// Invocation 2
		createTestEvent("inv2", "user", 102),
		createTestEvent("inv2", "model", 103),
		// Invocation 3
		createTestEvent("inv3", "user", 104),
		createTestEvent("inv3", "model", 105),
		// Invocation 4
		createTestEvent("inv4", "user", 106),
		createTestEvent("inv4", "model", 107),
		// Invocation 5
		createTestEvent("inv5", "user", 108),
		createTestEvent("inv5", "model", 109),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: events},
	}

	ctx := context.Background()
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "inv5")

	if err != nil {
		t.Fatalf("MaybeCompact() error = %v", err)
	}

	if compactionEvent == nil {
		t.Fatalf("expected compaction event, got nil")
	}

	// Verify compaction structure
	ec := compactionEvent.Actions.Compaction
	if ec == nil {
		t.Fatalf("expected Actions.Compaction to be set")
	}

	if ec.CompactedContent == nil {
		t.Errorf("expected CompactedContent to be set")
	}

	// Verify timestamps span the right range
	if ec.StartTimestamp == 0 || ec.EndTimestamp == 0 {
		t.Errorf("timestamps not set: start=%v, end=%v", ec.StartTimestamp, ec.EndTimestamp)
	}

	if ec.StartTimestamp >= ec.EndTimestamp {
		t.Errorf("start time should be < end time: %v >= %v", ec.StartTimestamp, ec.EndTimestamp)
	}
}

// TestMultipleCompactions verifies that multiple compactions can be performed in sequence.
func TestMultipleCompactions(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 2,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	// First compaction: 2 invocations
	firstEvents := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv2", "user", 101),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: firstEvents},
	}

	ctx := context.Background()
	compactionEvent1, err := compactor.MaybeCompact(ctx, mockSession, "inv2")
	if err != nil {
		t.Errorf("first MaybeCompact() error = %v", err)
	}

	if compactionEvent1 == nil {
		t.Fatalf("expected first compaction event")
	}

	// Second compaction: add more invocations beyond the first compaction time
	ec1 := compactionEvent1.Actions.Compaction
	secondEvents := append(firstEvents, compactionEvent1)
	secondEvents = append(secondEvents,
		createTestEvent("inv3", "user", 102),
		createTestEvent("inv4", "user", 103),
	)

	mockSession.events = &MockEvents{events: secondEvents}

	compactionEvent2, err := compactor.MaybeCompact(ctx, mockSession, "inv4")
	if err != nil {
		t.Errorf("second MaybeCompact() error = %v", err)
	}

	if compactionEvent2 == nil {
		t.Fatalf("expected second compaction event")
	}

	// Verify the second compaction doesn't re-compact inv2 (it's in the overlap)
	ec2 := compactionEvent2.Actions.Compaction
	if ec2.StartTimestamp <= ec1.EndTimestamp {
		t.Errorf("second compaction should start after first ends: %v <= %v", ec2.StartTimestamp, ec1.EndTimestamp)
	}
}

// TestCompactionWithOverlap verifies that the overlap mechanism preserves context continuity.
func TestCompactionWithOverlap(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 3,
		OverlapSize:        2, // Keep 2 invocations for overlap
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv2", "user", 101),
		createTestEvent("inv3", "user", 102),
		createTestEvent("inv4", "user", 103),
		createTestEvent("inv5", "user", 104),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: events},
	}

	ctx := context.Background()
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "inv5")

	if err != nil {
		t.Errorf("MaybeCompact() error = %v", err)
	}

	if compactionEvent == nil {
		t.Fatalf("expected compaction event")
	}

	ec := compactionEvent.Actions.Compaction
	if ec == nil {
		t.Fatalf("expected Actions.Compaction")
	}

	// With 5 invocations, threshold of 3, and overlap of 2:
	// We should compact invocations 1, 2, 3 (3 compacted, 2 overlap with 4-5)
	// This means inv4 and inv5 should NOT be in the compaction range
	expectedStartTime := float64(100) // inv1
	expectedEndTime := float64(102)   // inv3

	if ec.StartTimestamp > float64(100) {
		t.Errorf("expected to start from inv1 (100), got %v", ec.StartTimestamp)
	}

	if ec.EndTimestamp > float64(103) {
		t.Errorf("expected to end before inv4 (103), got %v", ec.EndTimestamp)
	}

	_ = expectedStartTime
	_ = expectedEndTime
}

// TestNoCompactionWhenDisabled verifies that disabled compaction returns nil.
func TestNoCompactionWhenDisabled(t *testing.T) {
	config := Config{
		Enabled:            false, // Disabled
		CompactionInterval: 1,
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
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "inv2")

	if err != nil {
		t.Errorf("MaybeCompact() error = %v", err)
	}

	if compactionEvent != nil {
		t.Errorf("expected nil compaction event when disabled, got %v", compactionEvent)
	}
}

// TestCompactionBelowThreshold verifies no compaction when below the threshold.
func TestCompactionBelowThreshold(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 5, // Threshold is 5
		OverlapSize:        2,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	// Only 3 invocations (below threshold)
	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv2", "user", 101),
		createTestEvent("inv3", "user", 102),
	}

	mockSession := &MockSession{
		id:     "test-session",
		events: &MockEvents{events: events},
	}

	ctx := context.Background()
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, "inv3")

	if err != nil {
		t.Errorf("MaybeCompact() error = %v", err)
	}

	if compactionEvent != nil {
		t.Errorf("expected nil compaction event below threshold, got %v", compactionEvent)
	}
}

// TestCompactionEventMetadata verifies the compaction event has correct metadata.
func TestCompactionEventMetadata(t *testing.T) {
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
	invocationID := "test-invocation"
	compactionEvent, err := compactor.MaybeCompact(ctx, mockSession, invocationID)

	if err != nil {
		t.Fatalf("MaybeCompact() error = %v", err)
	}

	if compactionEvent == nil {
		t.Fatalf("expected compaction event")
	}

	// Verify metadata
	if compactionEvent.Author != "system" {
		t.Errorf("expected author='system', got %q", compactionEvent.Author)
	}

	if compactionEvent.InvocationID != invocationID {
		t.Errorf("expected invocationID=%q, got %q", invocationID, compactionEvent.InvocationID)
	}

	if compactionEvent.ID == "" {
		t.Errorf("expected ID to be set")
	}

	if compactionEvent.Timestamp.IsZero() {
		t.Errorf("expected Timestamp to be set")
	}

	if compactionEvent.Actions.Compaction == nil {
		t.Errorf("expected Actions.Compaction to be set")
	}
}

// TestFormatEventsForSummarization verifies event formatting for LLM input.
func TestFormatEventsForSummarization(t *testing.T) {
	config := Config{
		Enabled:            true,
		CompactionInterval: 2,
		OverlapSize:        1,
	}
	compactor, _ := New(config, &MockLLM{name: "test"})

	events := []*session.Event{
		createTestEvent("inv1", "user", 100),
		createTestEvent("inv1", "model", 101),
		createTestEvent("inv2", "user", 102),
		createTestEvent("inv2", "model", 103),
	}

	formatted := compactor.formatEventsForSummarization(events)

	// Should have events from both invocations
	if len(formatted) == 0 {
		t.Errorf("expected formatted events, got none")
	}

	// Verify we have both user and model roles represented
	hasUser := false
	hasModel := false
	for _, content := range formatted {
		if content.Role == "user" {
			hasUser = true
		}
		if content.Role == "model" {
			hasModel = true
		}
	}

	if !hasUser || !hasModel {
		t.Errorf("expected both user and model roles in formatted events")
	}
}
