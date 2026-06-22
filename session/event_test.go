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

package session_test

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"

	"google.golang.org/adk/platform"
	"google.golang.org/adk/session"
)

// TestNewEventDefaults covers the deprecated NewEvent, which must keep its
// original signature and use the wall clock and a random UUID.
func TestNewEventDefaults(t *testing.T) {
	before := time.Now()
	ev := session.NewEvent(t.Context(), "inv-1")
	after := time.Now()

	if ev.InvocationID != "inv-1" {
		t.Errorf("InvocationID = %q, want %q", ev.InvocationID, "inv-1")
	}
	if _, err := uuid.Parse(ev.ID); err != nil {
		t.Errorf("ID = %q, not a valid UUID: %v", ev.ID, err)
	}
	if ev.Timestamp.Before(before) || ev.Timestamp.After(after) {
		t.Errorf("Timestamp = %v, want within [%v, %v]", ev.Timestamp, before, after)
	}
}

func TestNewEventUsesProviders(t *testing.T) {
	fixedTime := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(context.Background(), func() time.Time { return fixedTime })
	ctx = platform.WithUUIDProvider(ctx, func() string { return "fixed-event-id" })

	ev := session.NewEvent(ctx, "inv-1")

	if ev.ID != "fixed-event-id" {
		t.Errorf("ID = %q, want %q", ev.ID, "fixed-event-id")
	}
	if !ev.Timestamp.Equal(fixedTime) {
		t.Errorf("Timestamp = %v, want %v", ev.Timestamp, fixedTime)
	}
	if ev.InvocationID != "inv-1" {
		t.Errorf("InvocationID = %q, want %q", ev.InvocationID, "inv-1")
	}
}

// TestNewEventDeterministicReplay verifies that replaying the same sequence of
// provider values produces identical events. This is the property a workflow
// engine relies on to make event creation replay-safe.
func TestNewEventDeterministicReplay(t *testing.T) {
	newCtx := func() context.Context {
		var ids int
		ctx := platform.WithUUIDProvider(context.Background(), func() string {
			ids++
			return "event-" + string(rune('0'+ids))
		})
		var times int
		return platform.WithTimeProvider(ctx, func() time.Time {
			times++
			return time.Date(2024, time.January, 1, 0, 0, times, 0, time.UTC)
		})
	}

	run := func(ctx context.Context) []*session.Event {
		return []*session.Event{
			session.NewEvent(ctx, "inv"),
			session.NewEvent(ctx, "inv"),
		}
	}

	first := run(newCtx())
	second := run(newCtx())

	for i := range first {
		if first[i].ID != second[i].ID {
			t.Errorf("event %d ID: first run %q, second run %q", i, first[i].ID, second[i].ID)
		}
		if !first[i].Timestamp.Equal(second[i].Timestamp) {
			t.Errorf("event %d Timestamp: first run %v, second run %v", i, first[i].Timestamp, second[i].Timestamp)
		}
	}
}
