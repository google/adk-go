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

package main

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// arrivalBarrier returns a hook that blocks until workers callers have arrived,
// or until timeout, whichever comes first, and a counter of how many entered.
//
// The asymmetry is the whole mechanism. If the caller's critical section is
// atomic, only one caller can ever be inside the hook, arrivals never reach
// workers, and that one caller waits out the timeout and proceeds alone. If the
// section has been split, every caller gets in, arrivals reach workers, and all
// of them are released together -- which is not a race the test hopes to catch
// but an interleaving it forces.
func arrivalBarrier(workers int, timeout time.Duration) (hook func(), entered *atomic.Int64) {
	var (
		mu      sync.Mutex
		arrived int
		release = make(chan struct{})
		count   atomic.Int64
	)
	return func() {
		count.Add(1)
		mu.Lock()
		arrived++
		if arrived == workers {
			close(release)
		}
		mu.Unlock()
		select {
		case <-release:
		case <-time.After(timeout):
		}
	}, &count
}

// The claim's read-modify-write must be one critical section, so that of
// several concurrent callers for the same field exactly one can reserve it.
// ADK executes a turn's tool calls concurrently -- handleFunctionCalls hands one
// task per call to platform.RunTasks, a goroutine per task -- so this is what
// stops an attacker-steered model writing a type twice.
//
// Killing mutation: split the hold in claimType or claimLabel -- read the map
// under the lock, unlock, then re-lock to write.
//
// Measured against exactly that mutation, in fresh processes each time:
//   - the pre-existing concurrency tests (N goroutines released together,
//     asserting one winner): killed it 0 times in 10.
//   - this test: killed it 10 times in 10 for claimType and 10 times in 10 for
//     claimLabel, with 0 spurious failures in 20 full-suite runs against
//     correct code (-race -shuffle=on).
//
// The difference is not the worker count. It is that those tests race the
// window between the read and the write, which is two instructions wide, while
// this one holds every caller inside the section until they have all arrived.
// That needs an injection point inside the section, which is what claimBarrier
// is for and the only reason it exists.
//
// TestDoChangeTypeConcurrentSingleWrite covers the same property one layer up,
// through the real tool functions and a live server. It establishes the outcome
// under real contention. This establishes the mechanism.
func TestAClaimsCriticalSectionAdmitsOneCallerAtATime(t *testing.T) {
	for _, tc := range []struct {
		field string
		open  need
		claim func(*Client) (bool, bool)
	}{
		{"type", need{typ: true}, func(c *Client) (bool, bool) { return c.claimType(7) }},
		{"label", need{label: true}, func(c *Client) (bool, bool) { return c.claimLabel(7) }},
	} {
		t.Run(tc.field, func(t *testing.T) {
			// Eight is enough: the barrier, not the worker count, is what makes
			// the interleaving deterministic, and more workers queueing through
			// one release channel buys nothing.
			const workers = 8
			hook, entered := arrivalBarrier(workers, 500*time.Millisecond)
			orig := claimBarrier
			claimBarrier = hook
			t.Cleanup(func() { claimBarrier = orig })

			c := &Client{
				cfg:        testConfig(),
				log:        discardLogger(),
				authorized: make(map[int]need),
				attempts:   make(map[attemptKey]int),
			}
			c.authorize(7, tc.open)

			var (
				wg    sync.WaitGroup
				wins  atomic.Int64
				start = make(chan struct{})
			)
			for range workers {
				wg.Add(1)
				go func() {
					defer wg.Done()
					<-start
					if claimed, _ := tc.claim(c); claimed {
						wins.Add(1)
					}
				}()
			}
			close(start)
			wg.Wait()

			if got := wins.Load(); got != 1 {
				t.Errorf("%d concurrent callers reserved the %s field %d times, want 1", workers, tc.field, got)
			}
			// The stronger of the two, and the one that fails on a split hold
			// even when the winners happen to serialize: more than one caller
			// inside the section at once means it is not a critical section.
			if got := entered.Load(); got != 1 {
				t.Errorf("%d callers entered the %s claim's critical section, want 1: the read-modify-write is not atomic",
					got, tc.field)
			}
		})
	}
}
