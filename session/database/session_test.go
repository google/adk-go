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

package database

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"
)

func TestLocalSessionEventsSnapshot(t *testing.T) {
	base := time.Date(2026, time.September, 4, 0, 0, 0, 0, time.UTC)
	initial := make([]*session.Event, 3, 4)
	initial[0] = &session.Event{ID: "first", Timestamp: base}
	initial[1] = &session.Event{ID: "second", Timestamp: base.Add(2 * time.Second)}
	initial[2] = &session.Event{ID: "third", Timestamp: base.Add(3 * time.Second)}
	s := &localSession{events: initial}
	snapshot := s.Events()

	// The live slice has spare capacity, so an in-place out-of-order insertion
	// shifts its existing slots. A snapshot returned before that insertion must
	// still describe the history that existed when the caller requested it.
	if err := s.appendEvent(&session.Event{
		ID: "between", Timestamp: base.Add(time.Second),
	}); err != nil {
		t.Fatal(err)
	}
	if got := snapshot.Len(); got != 3 {
		t.Fatalf("snapshot.Len() = %d, want 3", got)
	}
	var snapshotIDs []string
	for event := range snapshot.All() {
		snapshotIDs = append(snapshotIDs, event.ID)
	}
	if got, want := fmt.Sprint(snapshotIDs), "[first second third]"; got != want {
		t.Errorf("snapshot IDs = %s, want %s", got, want)
	}
	var liveIDs []string
	for event := range s.Events().All() {
		liveIDs = append(liveIDs, event.ID)
	}
	if got, want := fmt.Sprint(liveIDs), "[first between second third]"; got != want {
		t.Errorf("live IDs = %s, want %s", got, want)
	}
}

func TestLocalSessionEventsConcurrentAppend(t *testing.T) {
	const count = 1000
	s := &localSession{}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := range count {
			if err := s.appendEvent(&session.Event{ID: fmt.Sprint(i)}); err != nil {
				t.Errorf("appendEvent: %v", err)
				return
			}
			runtime.Gosched()
		}
	}()
	close(start)
	for range count {
		snapshot := s.Events()
		n := 0
		for event := range snapshot.All() {
			if event == nil || event.ID != fmt.Sprint(n) {
				t.Errorf("snapshot event %d = %v, want ID %q", n, event, fmt.Sprint(n))
				break
			}
			n++
		}
		if n != snapshot.Len() {
			t.Errorf("snapshot iteration count = %d, want %d", n, snapshot.Len())
		}
		runtime.Gosched()
	}
	wg.Wait()
	if got := s.Events().Len(); got != count {
		t.Errorf("final event count = %d, want %d", got, count)
	}
}
