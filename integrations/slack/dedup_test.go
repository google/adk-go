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

package slack

import (
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func seen(t *testing.T, d Deduper, key string) bool {
	t.Helper()
	ok, err := d.Seen(t.Context(), key)
	if err != nil {
		t.Fatalf("Seen(%q) error: %v", key, err)
	}
	return ok
}

func TestLRUDeduperSeen(t *testing.T) {
	d := newLRUDeduper(10, time.Hour)
	if seen(t, d, "a") {
		t.Error("first Seen(a) = true, want false")
	}
	if !seen(t, d, "a") {
		t.Error("second Seen(a) = false, want true")
	}
	if seen(t, d, "b") {
		t.Error("first Seen(b) = true, want false")
	}
}

func TestLRUDeduperTTL(t *testing.T) {
	d := newLRUDeduper(10, 20*time.Millisecond)
	seen(t, d, "a")
	time.Sleep(50 * time.Millisecond)
	if seen(t, d, "a") {
		t.Error("Seen(a) after ttl = true, want false")
	}
}

func TestLRUDeduperCapacity(t *testing.T) {
	d := newLRUDeduper(2, time.Hour)
	for _, k := range []string{"a", "b", "c"} {
		seen(t, d, k)
	}
	if seen(t, d, "a") {
		t.Error("Seen(a) after eviction = true, want false")
	}
	// Re-adding a evicted b, the oldest, so c is still recorded.
	if !seen(t, d, "c") {
		t.Error("Seen(c) = false, want true")
	}
}

// Exactly one of many concurrent first sightings of a key may report it new,
// or a redelivered event is answered twice.
func TestLRUDeduperConcurrent(t *testing.T) {
	const keys, workers = 2000, 8
	d := newLRUDeduper(keys, time.Hour)
	var fresh atomic.Int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range workers {
		wg.Go(func() {
			<-start
			for k := range keys {
				if ok, _ := d.Seen(t.Context(), strconv.Itoa(k)); !ok {
					fresh.Add(1)
				}
			}
		})
	}
	close(start)
	wg.Wait()
	if got := fresh.Load(); got != keys {
		t.Errorf("%d first sightings reported new, want %d", got, keys)
	}
}
