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
	"sync"
)

// keyedMutex serializes work per key.
//
// One operation per key runs at a time, so two operations on the same key run
// one after the other while operations on different keys run concurrently. The
// entry for a key is created on demand and dropped once the last holder or
// waiter releases it, so the map does not grow with the number of keys a
// service has seen.
//
// This mirrors adk-python's reference-counted per-session locks.
type keyedMutex[K comparable] struct {
	mu    sync.Mutex
	locks map[K]*keyedMutexEntry
}

// keyedMutexEntry is the lock for one key, with the number of users that
// hold or wait for it.
type keyedMutexEntry struct {
	// held is a semaphore with one slot: a send takes the lock and a receive
	// releases it. A channel is used instead of sync.Mutex so that a waiter
	// can give up when its context is canceled.
	held chan struct{}
	// refs is the number of holders and waiters. It is guarded by keyedMutex.mu.
	refs int
}

// lock locks the entry for key and returns the function that releases it.
// The returned function must be called exactly once, normally with defer.
//
// Waiting honors ctx: a call whose context is canceled or whose deadline
// expires while it is queued returns ctx.Err() without taking the lock and
// drops its reference to the entry, so it neither blocks on a holder nor keeps
// an unused entry alive. The request that held the lock is the only one that
// has to finish.
func (k *keyedMutex[K]) lock(ctx context.Context, key K) (func(), error) {
	k.mu.Lock()
	if k.locks == nil {
		k.locks = make(map[K]*keyedMutexEntry)
	}
	entry, ok := k.locks[key]
	if !ok {
		entry = &keyedMutexEntry{held: make(chan struct{}, 1)}
		k.locks[key] = entry
	}
	// Count the user before releasing k.mu, so that the entry is not
	// dropped while somebody is about to wait for it.
	entry.refs++
	k.mu.Unlock()

	// A request that is already over must not take the lock even when it is
	// free: the select below could otherwise pick the free slot over the
	// canceled context.
	if err := ctx.Err(); err != nil {
		k.unref(key, entry)
		return nil, err
	}

	select {
	case entry.held <- struct{}{}:
	case <-ctx.Done():
		k.unref(key, entry)
		return nil, ctx.Err()
	}

	return func() {
		<-entry.held
		k.unref(key, entry)
	}, nil
}

// unref drops one user of entry and removes the entry when none is left.
func (k *keyedMutex[K]) unref(key K, entry *keyedMutexEntry) {
	k.mu.Lock()
	entry.refs--
	if entry.refs == 0 {
		// The last user is gone, so the entry can go with it. A later lock
		// for the same key creates a fresh one.
		delete(k.locks, key)
	}
	k.mu.Unlock()
}

// sessionKey identifies a session. Appends to different sessions are not
// serialized against each other.
type sessionKey struct {
	appName   string
	userID    string
	sessionID string
}
