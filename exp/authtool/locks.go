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

package authtool

import (
	"context"
	"sync"
)

// credentialLockKey names one user's credential: adk-python's lock key
// (app name, user ID, credential key), credential_manager.py:117-121.
type credentialLockKey struct {
	appName, userID, credentialKey string
}

// credentialLock is held by whoever has put a value in held. refs counts the
// calls holding or waiting for it, so the last one removes it from the table.
type credentialLock struct {
	held chan struct{}
	refs int
}

// The lock table is process-wide, as adk-python's class-level
// _credential_locks is, so that every manager of a config shares one lock per
// user's credential. credentialLocksMu guards credentialLocks and every refs.
var (
	credentialLocksMu sync.Mutex
	credentialLocks   = map[credentialLockKey]*credentialLock{}
)

// lockCredential waits until no other call holds the lock for k, then takes
// it and returns the function that releases it. It returns ctx's error,
// holding nothing, when ctx ends first. An entry stays in the table only
// while a call holds or waits for it.
func lockCredential(ctx context.Context, k credentialLockKey) (unlock func(), err error) {
	credentialLocksMu.Lock()
	l := credentialLocks[k]
	if l == nil {
		l = &credentialLock{held: make(chan struct{}, 1)}
		credentialLocks[k] = l
	}
	l.refs++
	credentialLocksMu.Unlock()

	release := func() {
		credentialLocksMu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(credentialLocks, k)
		}
		credentialLocksMu.Unlock()
	}
	select {
	case l.held <- struct{}{}:
		return func() {
			<-l.held
			release()
		}, nil
	case <-ctx.Done():
		release()
		return nil, ctx.Err()
	}
}
