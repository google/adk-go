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

package sessioninternal_test

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/session"
)

func createMutableSession(ctx context.Context, t *testing.T, sessionID string, initialData map[string]any) (*sessioninternal.MutableSession, session.Service) {
	t.Helper()
	service := session.InMemoryService()
	req := &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: sessionID,
		State:     initialData,
	}
	createResp, err := service.Create(ctx, req)
	if err != nil {
		t.Fatalf("Failed to create session %q: %v", sessionID, err)
	}

	return sessioninternal.NewMutableSession(service, createResp.Session), service
}

func TestMutableSession_SetGet(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name    string
		key     string
		value   any
		initial map[string]any
	}{
		{name: "string", key: "myKey", value: "myValue"},
		{name: "int", key: "count", value: 123},
		{name: "bool", key: "enabled", value: true},
		{name: "overwrite", key: "myKey", value: "newValue", initial: map[string]any{"myKey": "oldValue"}},
		{name: "new_key", key: "newKey", value: 456, initial: map[string]any{"existing": "val"}},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sessionID := fmt.Sprintf("testSetGet-%d", i)
			ms, _ := createMutableSession(ctx, t, sessionID, tc.initial)

			if err := ms.Set(tc.key, tc.value); err != nil {
				t.Fatalf("Set(%q, %v) failed unexpectedly: %v", tc.key, tc.value, err)
			}

			got, err := ms.Get(tc.key)
			if err != nil {
				t.Fatalf("Get(%q) failed unexpectedly: %v", tc.key, err)
			}
			if diff := cmp.Diff(tc.value, got); diff != "" {
				t.Errorf("Get(%q) returned diff (-want +got):\n%s", tc.key, diff)
			}
		})
	}
}

func TestMutableSession_All(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name    string
		initial map[string]any
	}{
		{
			name: "multiple",
			initial: map[string]any{
				"key1": "value1",
				"key2": 100,
				"key3": true,
			},
		},
		{
			name:    "empty",
			initial: nil,
		},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sessionID := fmt.Sprintf("testAll-%d", i)
			ms, _ := createMutableSession(ctx, t, sessionID, tc.initial)

			gotMap := make(map[string]any)
			for k, v := range ms.All() {
				gotMap[k] = v
			}

			wantMap := tc.initial
			if wantMap == nil {
				wantMap = map[string]any{}
			}

			if diff := cmp.Diff(wantMap, gotMap); diff != "" {
				t.Errorf("All() returned diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMutableSession_PassthroughMethods(t *testing.T) {
	ctx := context.Background()
	appName, userID, sessionID := "testApp", "testUser", "testPassthrough"

	service := session.InMemoryService()
	createReq := &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}
	createResp, err := service.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	ms := sessioninternal.NewMutableSession(service, createResp.Session)

	if got := ms.ID(); !reflect.DeepEqual(got, sessionID) {
		t.Errorf("ID() = %v, want %v", got, sessionID)
	}

	if got := ms.AppName(); !reflect.DeepEqual(got, appName) {
		t.Errorf("AppName() = %v, want %v", got, appName)
	}

	if got := ms.UserID(); !reflect.DeepEqual(got, userID) {
		t.Errorf("UserID() = %v, want %v", got, userID)
	}

	wantUpdatedTime := createResp.Session.LastUpdateTime()
	if got := ms.LastUpdateTime(); !got.Equal(wantUpdatedTime) || got.IsZero() {
		t.Errorf("LastUpdateTime() = %v, want %v (non-zero)", got, wantUpdatedTime)
	}

	if ms.Events() == nil {
		t.Errorf("Events() = nil, want non-nil")
	}

	state := ms.State()
	if state == nil {
		t.Fatalf("State() returned nil")
	}
	if _, ok := state.(*sessioninternal.MutableSession); !ok {
		t.Errorf("State() did not return *mutableSession as expected")
	}
}
