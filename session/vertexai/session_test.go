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

package vertexai

import (
	"fmt"
	"runtime"
	"sync"
	"testing"

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
