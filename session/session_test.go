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

package session

import (
	"context"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
)

func TestInMemorySessionService(t *testing.T) {
	ctx := context.Background()
	service := NewInMemorySessionService()

	appName := "test-app"
	userID := "test-user"
	sessionID := "test-session"

	// Test Create
	createReq := &adk.SessionCreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}
	session, err := service.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	wantSession := &adk.Session{
		AppName: appName,
		UserID:  userID,
		ID:      sessionID,
	}
	if diff := cmp.Diff(wantSession, session); diff != "" {
		t.Errorf("Create() mismatch (-want +got):\n%s", diff)
	}

	// Test Create again with same ID
	_, err = service.Create(ctx, createReq)
	if err == nil {
		t.Errorf("Create() with existing ID should have failed")
	}

	// Test Get
	getReq := &adk.SessionGetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}
	gotSession, err := service.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if diff := cmp.Diff(wantSession, gotSession); diff != "" {
		t.Errorf("Get() mismatch (-want +got):\n%s", diff)
	}

	// Test Get non-existent
	_, err = service.Get(ctx, &adk.SessionGetRequest{AppName: "dne", UserID: "dne", SessionID: "dne"})
	if err == nil {
		t.Errorf("Get() with non-existent session should have failed")
	}

	// Test AppendEvent
	event := adk.NewEvent("test-invocation")
	appendReq := &adk.SessionAppendEventRequest{
		Session: session,
		Event:   event,
	}
	if err := service.AppendEvent(ctx, appendReq); err != nil {
		t.Errorf("AppendEvent() failed: %v", err)
	}

	// Check events after AppendEvent.
	gotSession, err = service.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() after AppendEvent failed: %v", err)
	}
	if len(gotSession.Events) != 1 {
		t.Fatalf("Get() after AppendEvent: expected 1 event, got %d", len(gotSession.Events))
	}
	if diff := cmp.Diff(event, gotSession.Events[0]); diff != "" {
		t.Errorf("Get() after AppendEvent mismatch (-want +got):\n%s", diff)
	}

	// Test Delete
	deleteReq := &adk.SessionDeleteRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}
	if err := service.Delete(ctx, deleteReq); err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}

	// Test Get after Delete
	if _, err = service.Get(ctx, getReq); err == nil {
		t.Errorf("Get() after Delete() should have failed")
	}

	// Test Delete non-existent
	if err := service.Delete(ctx, deleteReq); err != nil {
		t.Errorf("Delete() on non-existent session failed: %v", err)
	}
}
