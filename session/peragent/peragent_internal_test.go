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

package peragent

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/session"
)

// failingService is a routed service whose Close fails with err. No public
// input makes closing a SQLite database fail.
type failingService struct {
	session.Service
	err error
}

func (f failingService) Close() error { return f.err }

func TestClose_ClosesEveryServiceWhenOneFails(t *testing.T) {
	ctx := t.Context()
	s, err := New(Config{AgentsRoot: t.TempDir()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := s.Create(ctx, &session.CreateRequest{AppName: "app_a", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ps := s.(*service)
	var appA route
	for _, r := range ps.routes {
		appA = r
	}
	errA, errB := errors.New("close fail_a"), errors.New("close fail_b")
	ps.routes["fail_a"] = route{svc: failingService{err: errA}}
	ps.routes["fail_b"] = route{svc: failingService{err: errB}}

	err = ps.Close()

	if !errors.Is(err, errA) || !errors.Is(err, errB) {
		t.Errorf("Close() error = %v, want both %v and %v", err, errA, errB)
	}
	if len(ps.routes) != 0 {
		t.Errorf("routing table after Close() holds %d entries, want 0", len(ps.routes))
	}
	if _, err := appA.svc.Get(ctx, &session.GetRequest{AppName: "app_a", UserID: "u1", SessionID: "s1"}); err == nil {
		t.Error("Get() on app_a's database after Close() succeeded, want an error from the closed pool")
	}
}
