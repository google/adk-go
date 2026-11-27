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

package testutil

import (
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

func TestRecordingSessionService(t *testing.T) {
	ctx := t.Context()
	svc := &RecordingSessionService{Service: session.InMemoryService()}
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "s1",
	}); err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	req1 := session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1", ExcludeEvents: true}
	req2 := session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"}

	for _, req := range []session.GetRequest{req1, req2} {
		r := req
		resp, err := svc.Get(ctx, &r)
		if err != nil {
			t.Fatalf("Get(%+v) failed: %v", req, err)
		}
		if resp.Session.ID() != "s1" {
			t.Fatalf("Session.ID() = %q, want %q", resp.Session.ID(), "s1")
		}
	}

	got := svc.Gets()
	want := []session.GetRequest{req1, req2}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("Gets() mismatch (-want +got):\n%s", diff)
	}

	// Mutating the returned slice must not affect recorded requests.
	got[0].SessionID = "mutated"
	if svc.Gets()[0].SessionID != "s1" {
		t.Fatalf("Gets() returned aliased slice: %+v", svc.Gets())
	}
}

func TestRecordingSessionService_Concurrent(t *testing.T) {
	ctx := t.Context()
	svc := &RecordingSessionService{Service: session.InMemoryService()}
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "s1",
	}); err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	const goroutines = 8
	var wg sync.WaitGroup
	for range goroutines {
		wg.Go(func() {
			_, _ = svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
			_ = svc.Gets()
		})
	}
	wg.Wait()

	if got := len(svc.Gets()); got != goroutines {
		t.Fatalf("len(Gets()) = %d, want %d", got, goroutines)
	}
}
