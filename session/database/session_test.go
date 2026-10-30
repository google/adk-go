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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"
)

func TestLocalSessionEventsSnapshot(t *testing.T) {
	first := &session.Event{ID: "first"}
	s := &localSession{events: []*session.Event{first}}
	snapshot := s.Events()

	// Replacing a slot must not change a previously returned history snapshot.
	s.mu.Lock()
	s.events[0] = &session.Event{ID: "replacement"}
	s.mu.Unlock()
	if err := s.appendEvent(&session.Event{ID: "second"}); err != nil {
		t.Fatal(err)
	}
	if got := snapshot.Len(); got != 1 {
		t.Fatalf("snapshot.Len() = %d, want 1", got)
	}
	if got := snapshot.At(0); got != first {
		t.Errorf("snapshot.At(0) = %v, want original event %v", got, first)
	}
	for event := range snapshot.All() {
		if event != first {
			t.Errorf("snapshot.All() yielded %v, want original event %v", event, first)
		}
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

// TestAppendEventConcurrentAppenders runs two appenders on one session object,
// which tail-retention compaction does when it stores its summary while
// sub-agents are still producing events. Neither may be refused as stale. That
// happens if updatedAt is set only after the commit: in between, the session
// holds an older update time than storage, and the sibling's check reads it.
func TestAppendEventConcurrentAppenders(t *testing.T) {
	s := emptyService(t)
	ctx := t.Context()
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session

	const perWriter = 200
	var appended, stale atomic.Int32
	var wg sync.WaitGroup
	for w := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range perWriter {
				err := s.AppendEvent(ctx, sess, &session.Event{
					ID:           fmt.Sprintf("w%d-e%d", w, i),
					Author:       "user",
					InvocationID: "inv1",
					Timestamp:    time.Now(),
				})
				switch {
				case err == nil:
					appended.Add(1)
				case strings.Contains(err.Error(), "stale session"):
					// Matched on text: the package has no sentinel for it.
					stale.Add(1)
				default:
					// SQLite refusing one of the two concurrent transactions
					// with a lock error, which is not under test.
				}
			}
		}()
	}
	wg.Wait()
	if appended.Load() == 0 {
		t.Fatal("no append committed, so nothing exercised the updatedAt write")
	}
	if n := stale.Load(); n > 0 {
		t.Errorf("%d appends refused as stale by a sibling append on the same session", n)
	}
}

// TestAppendEventStaleCheckReadsUnderLock covers the stale-session check's read
// of updatedAt. SQLite never lets the two transactions in
// TestAppendEventConcurrentAppenders overlap — it aborts one with a lock error
// — so a sibling's commit never writes updatedAt during that read there. A
// backend that runs transactions concurrently does, so this stands in for that
// commit by setting updatedAt from another goroutine. Run under -race.
func TestAppendEventStaleCheckReadsUnderLock(t *testing.T) {
	s := emptyService(t)
	ctx := t.Context()
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session.(*localSession)

	// Ahead of every append, so a write landing late never leaves the session
	// behind storage and the check never refuses an append as stale.
	ahead := time.Now().Add(time.Hour)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-done:
				return
			default:
			}
			sess.mu.Lock()
			sess.updatedAt = ahead
			sess.mu.Unlock()
		}
	}()

	for i := range 50 {
		if err := s.AppendEvent(ctx, sess, &session.Event{
			ID:           fmt.Sprintf("e%d", i),
			Author:       "user",
			InvocationID: "inv1",
			Timestamp:    time.Now(),
		}); err != nil {
			t.Errorf("AppendEvent() error = %v", err)
			break
		}
	}
	close(done)
	wg.Wait()
}
