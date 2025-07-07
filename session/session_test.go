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
	"reflect"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
)

func TestInMemorySessionService_Basic(t *testing.T) {
	ctx := context.Background()
	service := &InMemorySessionService{}

	// Create a session
	createReq := &adk.SessionCreateRequest{
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	sess, err := service.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	if sess.ID != "test-session" {
		t.Errorf("Create() returned session with ID %q, want %q", sess.ID, "test-session")
	}

	// Get the session
	getReq := &adk.SessionGetRequest{
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	gotSess, err := service.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if gotSess.ID != "test-session" {
		t.Errorf("Get() returned session with ID %q, want %q", gotSess.ID, "test-session")
	}

	// Try to create a session that already exists
	_, err = service.Create(ctx, createReq)
	if err == nil {
		t.Errorf("Create() with existing session ID succeeded, want error")
	}

	// Delete the session
	deleteReq := &adk.SessionDeleteRequest{
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	if err := service.Delete(ctx, deleteReq); err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}

	// Try to get the deleted session
	_, err = service.Get(ctx, getReq)
	if err == nil {
		t.Errorf("Get() after Delete() succeeded, want error")
	}

	// Try to delete a non-existent session (should error. See TODO in Delete)
	if err := service.Delete(ctx, deleteReq); err == nil {
		t.Error("Delete() on non-existent session succeeded, want error")
	}
}

func TestInMemorySessionService_AppendEvent(t *testing.T) {
	ctx := context.Background()
	service := &InMemorySessionService{}

	// Create a session
	createReq := &adk.SessionCreateRequest{
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	sess, err := service.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	// Append an event
	event1 := &adk.Event{
		ID:           "e-12345",
		InvocationID: "inv-12345",
		Author:       "user",
		Branch:       "foo.bar",
	}
	if err := service.AppendEvent(ctx, sess, event1); err != nil {
		t.Fatalf("AppendEvent() failed: %v", err)
	}
	if diff := cmp.Diff([]*adk.Event{event1}, sess.Events); diff != "" {
		t.Errorf("AppendEvent() did not update the session object's Events (-want +got):\n%s", diff)
	}

	// Get the session and check events
	getReq := &adk.SessionGetRequest{
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	gotSess, err := service.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if diff := cmp.Diff([]*adk.Event{event1}, gotSess.Events); diff != "" {
		t.Errorf("Get() returned events mismatch (-want +got):\n%s", diff)
	}

	// Append another event
	event2 := &adk.Event{ID: "e-7890"}
	if err := service.AppendEvent(ctx, sess, event2); err != nil {
		t.Fatalf("AppendEvent() failed: %v", err)
	}
	if diff := cmp.Diff([]*adk.Event{event1, event2}, sess.Events); diff != "" {
		t.Errorf("AppendEvent() did not update the session object's Events (-want +got):\n%s", diff)
	}

	// Get the session and check events.
	gotSess, err = service.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if diff := cmp.Diff(gotSess.Events, []*adk.Event{event1, event2}); diff != "" {
		t.Errorf("Get() returned events mismatch (-got +want):\n%s", diff)
	}
}

func TestInMemorySessionService_AppendEvent_nonexistant(t *testing.T) {
	ctx := context.Background()
	service := &InMemorySessionService{}

	// Append an event
	event1 := &adk.Event{
		ID:           "e-12345",
		InvocationID: "inv-12345",
		Author:       "user",
		Branch:       "foo.bar",
	}

	// Append to non-existent session
	sess := &adk.Session{
		ID:      "non-existent",
		AppName: "foo",
		UserID:  "bar",
	}
	if err := service.AppendEvent(ctx, sess, event1); err == nil {
		t.Errorf("AppendEvent() to non-existent session succeeded, want error")
	}
	if sess.ID != "non-existent" || sess.AppName != "foo" || sess.UserID != "bar" || len(sess.Events) != 0 {
		t.Errorf("AppendEvent(non-existent session) unexpectedly modified the session object: %v", sess)
	}
	got, err := service.List(ctx, &adk.SessionListRequest{AppName: "foo", UserID: "bar"})
	if err != nil || len(got) != 0 {
		t.Fatalf("List() = (%v, %v), want ([], nil)", got, err)
	}
}

func TestInMemorySessionService_List(t *testing.T) {
	ctx := context.Background()
	service := &InMemorySessionService{}

	// Setup: create sessions for different users and apps
	sessionsToCreate := []adk.SessionCreateRequest{
		{AppName: "app1", UserID: "user1", SessionID: "s1"},
		{AppName: "app1", UserID: "user1", SessionID: "s2"},
		{AppName: "app1", UserID: "user2", SessionID: "s3"},
		{AppName: "app2", UserID: "user1", SessionID: "s4"},
	}
	for _, req := range sessionsToCreate {
		if _, err := service.Create(ctx, &req); err != nil {
			t.Fatalf("Setup: Create() failed for %+v: %v", req, err)
		}
	}

	// Test cases
	testCases := []struct {
		name           string
		appName        string
		userID         string
		wantSessionIDs []string
	}{
		{
			name:           "List for user1/app1",
			appName:        "app1",
			userID:         "user1",
			wantSessionIDs: []string{"s1", "s2"},
		},
		{
			name:           "List for user2/app1",
			appName:        "app1",
			userID:         "user2",
			wantSessionIDs: []string{"s3"},
		},
		{
			name:           "List for user1/app2",
			appName:        "app2",
			userID:         "user1",
			wantSessionIDs: []string{"s4"},
		},
		{
			name:           "List for non-existent user",
			appName:        "app1",
			userID:         "user3",
			wantSessionIDs: nil,
		},
		{
			name:           "List for non-existent app",
			appName:        "app3",
			userID:         "user1",
			wantSessionIDs: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			listReq := &adk.SessionListRequest{
				AppName: tc.appName,
				UserID:  tc.userID,
			}
			sessions, err := service.List(ctx, listReq)
			if err != nil {
				t.Fatalf("List() failed: %v", err)
			}

			var gotSessionIDs []string
			if sessions != nil {
				gotSessionIDs = make([]string, len(sessions))
				for i, s := range sessions {
					gotSessionIDs[i] = s.ID
				}
			}

			if got, want := gotSessionIDs, tc.wantSessionIDs; !reflect.DeepEqual(got, want) {
				t.Errorf("List() = %v, want %v", got, want)
			}
		})
	}
}
