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

package session_test

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/session"
	"google.golang.org/adk/session/session_test"
)

func Test_inMemoryService(t *testing.T) {
	opts := session_test.SuiteOptions{SupportsUserProvidedSessionID: true} // InMemory supports custom IDs
	session_test.RunServiceTests(t, opts, func(t *testing.T) session.Service {
		return session.InMemoryService()
	})
}

func Test_inMemoryService_CreateConcurrentAccess(t *testing.T) {
	s := session.InMemoryService()
	const goroutines = 16
	const attempts = 32

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(goroutines)

	req := &session.CreateRequest{
		AppName:   "race-app",
		UserID:    "race-user",
		SessionID: "race-session",
	}

	var successCount atomic.Int32
	var errorCount atomic.Int32

	for range goroutines {
		go func() {
			defer wg.Done()
			<-start
			for range attempts {
				_, err := s.Create(t.Context(), req)
				if err == nil {
					successCount.Add(1)
				} else if strings.Contains(err.Error(), "already exists") {
					errorCount.Add(1)
				}
			}
		}()
	}

	close(start)
	wg.Wait()

	if successCount.Load() != 1 {
		t.Errorf("expected 1 successful creation, but got %d", successCount.Load())
	}

	expectedErrors := int32(goroutines*attempts - 1)
	if errorCount.Load() != expectedErrors {
		t.Errorf("expected %d 'already exists' errors, but got %d", expectedErrors, errorCount.Load())
	}
}

// TestInMemoryState_All_ConcurrentSet is a regression test for the concurrent map
// panic described in https://github.com/google/adk-go/issues/561.
//
// state.All() previously unlocked the mutex between each iteration step, allowing
// concurrent Set() calls to mutate the map while it was being ranged over,
// triggering a fatal "concurrent map iteration and map write" panic.
//
// The fix snapshots the map under the lock (maps.Clone) before iterating.
// This test races All() against concurrent Set() calls to verify no panic occurs.
// Run with: go test -race ./session/...
func TestInMemoryState_All_ConcurrentSet(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"k0": 0},
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	sess := resp.Session

	const writers = 8
	const iterations = 200

	var wg sync.WaitGroup
	start := make(chan struct{})

	// Concurrent writers: append events that update state.
	wg.Add(writers)
	for w := range writers {
		go func(w int) {
			defer wg.Done()
			<-start
			for i := range iterations {
				ev := &session.Event{
					Timestamp: time.Now(),
					Actions: session.EventActions{
						StateDelta: map[string]any{
							fmt.Sprintf("w%d_k%d", w, i): i,
						},
					},
				}
				if err := svc.AppendEvent(ctx, sess, ev); err != nil {
					// Stale session errors are expected under concurrent writes; ignore.
					_ = err
				}
			}
		}(w)
	}

	// Concurrent reader: iterate all state entries.
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for range iterations {
			for k, v := range sess.State().All() {
				_ = k
				_ = v
			}
		}
	}()

	close(start)
	wg.Wait()
}

func TestInMemorySession_AppendEvent_Deadlock(t *testing.T) {
	ctx := t.Context()
	service := session.InMemoryService()

	// Create a session
	createReq := &session.CreateRequest{
		AppName: "testapp",
		UserID:  "testuser",
	}
	createResp, err := service.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	sess := createResp.Session

	// Event with StateDelta to trigger updateSessionState
	event := &session.Event{
		ID:        "event1",
		Timestamp: time.Now(),
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"test_key": "test_value",
			},
		},
	}

	// This call should hang if the deadlock is present
	err = service.AppendEvent(ctx, sess, event)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	// If it doesn't hang, the test passes (meaning no deadlock)
	t.Log("AppendEvent did not deadlock")
}
