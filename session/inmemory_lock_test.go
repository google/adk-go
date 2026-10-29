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

package session

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestInMemoryService_AppendEvent_LockScope pins which appends exclude each
// other through the service lock. An append that changes only session state
// must not wait for another holder of the service read lock, so appends to
// different sessions run in parallel. One that changes app or user state
// must wait, because that state is shared between sessions.
func TestInMemoryService_AppendEvent_LockScope(t *testing.T) {
	for _, tc := range []struct {
		name      string
		delta     map[string]any
		wantBlock bool
	}{
		{"no delta", nil, false},
		{"session delta", map[string]any{"k": 1}, false},
		{"temp delta", map[string]any{"temp:k": 1}, false},
		{"app delta", map[string]any{"app:k": 1}, true},
		{"user delta", map[string]any{"user:k": 1}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := InMemoryService().(*inMemoryService)
			resp, err := svc.Create(t.Context(), &CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
			if err != nil {
				t.Fatal(err)
			}
			// Stands in for a concurrent append to another session, or a Get.
			svc.mu.RLock()
			done := make(chan error, 1)
			go func() {
				done <- svc.AppendEvent(t.Context(), resp.Session, &Event{ID: "e", Timestamp: time.Now(), Actions: EventActions{StateDelta: tc.delta}})
			}()
			select {
			case err := <-done:
				svc.mu.RUnlock()
				if tc.wantBlock {
					t.Fatalf("AppendEvent returned (err %v) while the service read lock was held, want it to wait", err)
				}
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(200 * time.Millisecond):
				svc.mu.RUnlock()
				if err := <-done; err != nil {
					t.Fatal(err)
				}
				if !tc.wantBlock {
					t.Fatal("AppendEvent waited for the service read lock, want it to run alongside other readers")
				}
			}
		})
	}
}

// TestInMemoryService_GetSeesWholeAppends checks that a Get running alongside
// appends to the same session sees each append whole: its event, its session
// state and its app state together, or none of them. Every event sets
// "count" and "app:count" to the number of events after it.
func TestInMemoryService_GetSeesWholeAppends(t *testing.T) {
	svc := InMemoryService()
	const sessions, appends = 4, 300
	var wg sync.WaitGroup
	for i := range sessions {
		id := fmt.Sprintf("s%d", i)
		resp, err := svc.Create(t.Context(), &CreateRequest{AppName: "app", UserID: "user", SessionID: id})
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(2)
		go func() {
			defer wg.Done()
			for n := 1; n <= appends; n++ {
				delta := map[string]any{"count": n}
				if i == 0 {
					delta["app:count"] = n
				}
				if err := svc.AppendEvent(t.Context(), resp.Session, &Event{Timestamp: time.Now(), Actions: EventActions{StateDelta: delta}}); err != nil {
					t.Error(err)
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			for range appends {
				got, err := svc.Get(t.Context(), &GetRequest{AppName: "app", UserID: "user", SessionID: id})
				if err != nil {
					t.Error(err)
					return
				}
				n := got.Session.Events().Len()
				count, _ := got.Session.State().Get("count")
				if n > 0 && count != n {
					t.Errorf("session %s: Get saw %d events and count %v, want them equal", id, n, count)
					return
				}
				if i == 0 && n > 0 {
					if appCount, _ := got.Session.State().Get("app:count"); appCount != n {
						t.Errorf("session %s: Get saw %d events and app:count %v, want them equal", id, n, appCount)
						return
					}
				}
				// List reads every stored session's state; -race checks it
				// takes the same lock as the appends writing it.
				if _, err := svc.List(t.Context(), &ListRequest{AppName: "app", UserID: "user"}); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
