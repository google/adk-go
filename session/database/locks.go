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

import "sync"

// keyedMutex serializes work per key.
//
// One mutex is held per key at a time, so two operations on the same key run
// one after the other while operations on different keys run concurrently. The
// mutex for a key is created on demand and dropped once the last holder
// releases it, so the map does not grow with the number of keys a service has
// seen.
//
// This mirrors adk-python's reference-counted per-session locks.
type keyedMutex[K comparable] struct {
	mu    sync.Mutex
	locks map[K]*keyedMutexEntry
}

// keyedMutexEntry is the mutex for one key, with the number of users that
// hold or wait for it.
type keyedMutexEntry struct {
	mu   sync.Mutex
	refs int
}

// lock locks the entry for key and returns the function that releases it.
// The returned function must be called exactly once, normally with defer.
func (k *keyedMutex[K]) lock(key K) func() {
	k.mu.Lock()
	if k.locks == nil {
		k.locks = make(map[K]*keyedMutexEntry)
	}
	entry, ok := k.locks[key]
	if !ok {
		entry = &keyedMutexEntry{}
		k.locks[key] = entry
	}
	// Count the user before releasing k.mu, so that the entry is not
	// dropped while somebody is about to wait for it.
	entry.refs++
	k.mu.Unlock()

	entry.mu.Lock()

	return func() {
		entry.mu.Unlock()

		k.mu.Lock()
		entry.refs--
		if entry.refs == 0 {
			// The last user is gone, so the entry can go with it. A
			// later lock for the same key creates a fresh one.
			delete(k.locks, key)
		}
		k.mu.Unlock()
	}
}

// sessionKey identifies a session. Appends to different sessions are not
// serialized against each other.
type sessionKey struct {
	appName   string
	userID    string
	sessionID string
}
