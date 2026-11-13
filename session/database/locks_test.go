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

package database

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// updateMax records the highest value counter has reached.
func updateMax(max *atomic.Int32, n int32) {
	for {
		seen := max.Load()
		if n <= seen || max.CompareAndSwap(seen, n) {
			return
		}
	}
}

// The lock for a session is dropped once nobody holds or waits for it, so a
// long-lived service does not accumulate one lock per session it has seen.
func TestKeyedMutex_DropsTheEntryWhenItIsUnused(t *testing.T) {
	var k keyedMutex[string]

	unlock, err := k.lock(t.Context(), "a")
	if err != nil {
		t.Fatalf("lock: %v", err)
	}
	if got := len(k.locks); got != 1 {
		t.Errorf("locks while one is held = %d; want 1", got)
	}
	unlock()
	if got := len(k.locks); got != 0 {
		t.Errorf("locks after the last user left = %d; want 0", got)
	}

	// The key can be locked again afterwards.
	unlock, err = k.lock(t.Context(), "a")
	if err != nil {
		t.Fatalf("lock after the entry was dropped: %v", err)
	}
	unlock()
}

// Two operations on the same key do not overlap, and an operation on another
// key is not held up by them.
func TestKeyedMutex_SerializesOneKeyAtATime(t *testing.T) {
	var k keyedMutex[string]

	var globalInFlight, globalMax atomic.Int32
	var sameInFlight, sameMax atomic.Int32
	release := make(chan struct{})
	firstInside := make(chan struct{})
	otherInside := make(chan struct{})
	var holdFirst atomic.Bool

	hold := func(sameKey bool) {
		n := globalInFlight.Add(1)
		defer globalInFlight.Add(-1)
		updateMax(&globalMax, n)
		if sameKey {
			same := sameInFlight.Add(1)
			defer sameInFlight.Add(-1)
			updateMax(&sameMax, same)
			if holdFirst.CompareAndSwap(false, true) {
				close(firstInside)
				<-release
			}
			return
		}
		close(otherInside)
		<-release
	}

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		unlock, err := k.lock(t.Context(), "a")
		if err != nil {
			t.Errorf("lock: %v", err)
			return
		}
		defer unlock()
		hold(true)
	}()
	go func() {
		defer wg.Done()
		unlock, err := k.lock(t.Context(), "a")
		if err != nil {
			t.Errorf("lock: %v", err)
			return
		}
		defer unlock()
		hold(true)
	}()
	go func() {
		defer wg.Done()
		unlock, err := k.lock(t.Context(), "b")
		if err != nil {
			t.Errorf("lock: %v", err)
			return
		}
		defer unlock()
		hold(false)
	}()

	for _, inside := range []chan struct{}{firstInside, otherInside} {
		select {
		case <-inside:
		case <-time.After(10 * time.Second):
			t.Fatal("an operation never reached the locked section")
		}
	}
	close(release)
	wg.Wait()

	if got := sameMax.Load(); got != 1 {
		t.Errorf("operations on one key at the same time = %d; want 1", got)
	}
	if got := globalMax.Load(); got != 2 {
		t.Errorf("operations on two keys at the same time = %d; want 2", got)
	}
}

// A waiter whose request is canceled leaves the queue instead of blocking
// until the holder finishes its transaction. Without this, a client that has
// already given up keeps a connection and a goroutine until an append it no
// longer cares about commits.
func TestKeyedMutex_CanceledWaiterDoesNotWaitForTheHolder(t *testing.T) {
	var k keyedMutex[string]

	unlock, err := k.lock(t.Context(), "a")
	if err != nil {
		t.Fatalf("lock: %v", err)
	}
	defer unlock()

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	waiterDone := make(chan error, 1)
	go func() {
		_, err := k.lock(ctx, "a")
		waiterDone <- err
	}()

	cancel()

	select {
	case err := <-waiterDone:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("canceled waiter error = %v; want %v", err, context.Canceled)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the canceled waiter is still waiting for the holder")
	}

	// Only the holder's reference is left, so the entry goes when it releases.
	if got := len(k.locks); got != 1 {
		t.Errorf("locks after the waiter canceled = %d; want 1", got)
	}
}

// A request that is canceled before it takes a free lock does not take it:
// the select between the free slot and the canceled context would otherwise
// pick the slot at random.
func TestKeyedMutex_CanceledRequestDoesNotTakeAFreeLock(t *testing.T) {
	var k keyedMutex[string]

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := k.lock(ctx, "a"); !errors.Is(err, context.Canceled) {
		t.Errorf("lock error = %v; want %v", err, context.Canceled)
	}
	if got := len(k.locks); got != 0 {
		t.Errorf("locks after a canceled request = %d; want 0", got)
	}
}
