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

package firestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "cloud.google.com/go/firestore/apiv1/firestorepb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/sessiontestsuite"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func newTestService(t *testing.T, cfg Config) (*firestoreService, *fakeFirestoreServer) {
	t.Helper()
	client, srv := newTestFirestoreClient(t)
	svc, err := NewSessionService(client, cfg)
	if err != nil {
		t.Fatalf("NewSessionService() failed: %v", err)
	}
	return svc.(*firestoreService), srv
}

func TestFirestoreService_ConformanceSuite(t *testing.T) {
	opts := sessiontestsuite.SuiteOptions{
		SupportsUserProvidedSessionID: true,
	}
	sessiontestsuite.RunServiceTests(t, opts, func(t *testing.T) session.Service {
		svc, _ := newTestService(t, Config{})
		return svc
	})
}

func TestNewSessionService_ConfigAndValidation(t *testing.T) {
	t.Run("nil client returns error", func(t *testing.T) {
		if _, err := NewSessionService(nil, Config{}); err == nil {
			t.Fatal("NewSessionService(nil, Config{}) = nil error, want error")
		}
	})

	t.Run("defaults match adk-python", func(t *testing.T) {
		t.Setenv("ADK_FIRESTORE_ROOT_COLLECTION", "")
		svc, _ := newTestService(t, Config{})
		if svc.rootCollection != DefaultRootCollection {
			t.Errorf("rootCollection = %q, want %q", svc.rootCollection, DefaultRootCollection)
		}
		if svc.appStateCollection != DefaultAppStateCollection {
			t.Errorf("appStateCollection = %q, want %q", svc.appStateCollection, DefaultAppStateCollection)
		}
		if svc.userStateCollection != DefaultUserStateCollection {
			t.Errorf("userStateCollection = %q, want %q", svc.userStateCollection, DefaultUserStateCollection)
		}
	})

	t.Run("ADK_FIRESTORE_ROOT_COLLECTION env var is used when Config.RootCollection is empty", func(t *testing.T) {
		t.Setenv("ADK_FIRESTORE_ROOT_COLLECTION", "env-root-coll")
		svc, _ := newTestService(t, Config{})
		if svc.rootCollection != "env-root-coll" {
			t.Errorf("rootCollection = %q, want %q", svc.rootCollection, "env-root-coll")
		}
	})

	t.Run("explicit Config overrides env var and defaults", func(t *testing.T) {
		t.Setenv("ADK_FIRESTORE_ROOT_COLLECTION", "env-root-coll")
		svc, _ := newTestService(t, Config{
			RootCollection:      "custom-root",
			AppStateCollection:  "custom-app",
			UserStateCollection: "custom-user",
		})
		if svc.rootCollection != "custom-root" {
			t.Errorf("rootCollection = %q, want %q", svc.rootCollection, "custom-root")
		}
		if svc.appStateCollection != "custom-app" {
			t.Errorf("appStateCollection = %q, want %q", svc.appStateCollection, "custom-app")
		}
		if svc.userStateCollection != "custom-user" {
			t.Errorf("userStateCollection = %q, want %q", svc.userStateCollection, "custom-user")
		}
	})

	t.Run("invalid collection names are rejected", func(t *testing.T) {
		client, _ := newTestFirestoreClient(t)
		for _, tc := range []struct {
			name string
			cfg  Config
		}{
			{"slash in root", Config{RootCollection: "a/b"}},
			{"dot in app", Config{AppStateCollection: "."}},
			{"reserved in user", Config{UserStateCollection: "__reserved__"}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := NewSessionService(client, tc.cfg); err == nil {
					t.Errorf("NewSessionService(%+v) = nil error, want error", tc.cfg)
				}
			})
		}
	})
}

func TestFirestoreService_CreateUsesPlatformProviders(t *testing.T) {
	fixedTime := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixedTime })
	ctx = platform.WithUUIDProvider(ctx, func() string { return "fixed-session-id" })

	svc, srv := newTestService(t, Config{})
	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"sk": "sv", "app:ak": "av", "user:uk": "uv", "temp:tk": "tv"},
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got := resp.Session.ID(); got != "fixed-session-id" {
		t.Errorf("Session.ID() = %q, want %q", got, "fixed-session-id")
	}
	if got := resp.Session.LastUpdateTime(); !got.Equal(fixedTime) {
		t.Errorf("LastUpdateTime() = %v, want %v", got, fixedTime)
	}

	// Verify Firestore document layout matches adk-python:
	// session state is stored as a JSON string with only session-scoped keys.
	doc, ok := srv.getDoc("adk-session/app/users/user/sessions/fixed-session-id")
	if !ok {
		t.Fatal("session document was not created at expected path")
	}
	if got := doc.GetFields()["revision"].GetIntegerValue(); got != 0 {
		t.Errorf("stored revision = %d, want 0", got)
	}
	stateJSON := doc.GetFields()["state"].GetStringValue()
	var persistedSessionState map[string]any
	if err := json.Unmarshal([]byte(stateJSON), &persistedSessionState); err != nil {
		t.Fatalf("session state is not valid JSON string (%q): %v", stateJSON, err)
	}
	if diff := cmp.Diff(map[string]any{"sk": "sv"}, persistedSessionState); diff != "" {
		t.Errorf("persisted session state mismatch (-want +got):\n%s", diff)
	}

	// Get and List must return the exact same LastUpdateTime as the in-memory handle.
	gotResp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "fixed-session-id",
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := gotResp.Session.LastUpdateTime(); !got.Equal(fixedTime) {
		t.Errorf("Get().Session.LastUpdateTime() = %v, want %v", got, fixedTime)
	}

	eventTime := fixedTime.Add(5*time.Minute + 123456789*time.Nanosecond)
	wantTruncatedEventTime := eventTime.Truncate(time.Microsecond)
	if err := svc.AppendEvent(ctx, resp.Session, &session.Event{
		ID:           "ev-clock",
		Author:       "user",
		InvocationID: "inv1",
		Timestamp:    eventTime,
	}); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	if got := resp.Session.LastUpdateTime(); !got.Equal(wantTruncatedEventTime) {
		t.Errorf("in-memory LastUpdateTime() after AppendEvent = %v, want microsecond-truncated %v", got, wantTruncatedEventTime)
	}
	gotAfterAppend, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "fixed-session-id",
	})
	if err != nil {
		t.Fatalf("Get() after AppendEvent error = %v", err)
	}
	if got := gotAfterAppend.Session.LastUpdateTime(); !got.Equal(wantTruncatedEventTime) {
		t.Errorf("Get().Session.LastUpdateTime() after AppendEvent = %v, want %v", got, wantTruncatedEventTime)
	}
}

func TestFirestoreService_CreateReplacesNestedAppAndUserState(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "test_app",
		UserID:  "test_user",
		State: map[string]any{
			"app:cfg":  map[string]any{"name": "alice", "role": "admin"},
			"user:cfg": map[string]any{"name": "alice", "role": "admin"},
		},
	}); err != nil {
		t.Fatalf("first Create() error = %v", err)
	}

	s2, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "test_app",
		UserID:  "test_user",
		State: map[string]any{
			"app:cfg":  map[string]any{"name": "bob"},
			"user:cfg": map[string]any{"name": "bob"},
		},
	})
	if err != nil {
		t.Fatalf("second Create() error = %v", err)
	}

	for _, key := range []string{"app:cfg", "user:cfg"} {
		val, err := s2.Session.State().Get(key)
		if err != nil {
			t.Fatalf("State().Get(%q) error = %v", key, err)
		}
		if diff := cmp.Diff(map[string]any{"name": "bob"}, val); diff != "" {
			t.Errorf("State()[%q] mismatch (-want +got):\n%s", key, diff)
		}
	}

	// Also verify the underlying app_states and user_states documents in Firestore
	// do not retain the old nested "role" key.
	for _, docPath := range []string{"app_states/test_app", "user_states/test_app/users/test_user"} {
		doc, ok := srv.getDoc(docPath)
		if !ok {
			t.Fatalf("expected document %s to exist", docPath)
		}
		cfgFields := doc.GetFields()["cfg"].GetMapValue().GetFields()
		if _, hasRole := cfgFields["role"]; hasRole {
			t.Errorf("document %s retained stale nested key 'role': %v", docPath, cfgFields)
		}
	}
}

func TestFirestoreService_NonJSONStateErrorSurfaces(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user-bad",
		SessionID: "s-bad",
		State:     map[string]any{"bad": math.NaN()},
	}); err == nil {
		t.Fatal("Create with NaN state = nil error, want error")
	}

	if _, ok := srv.getDoc("adk-session/app/users/user-bad/sessions/s-bad"); ok {
		t.Error("session document should not be persisted when state marshaling fails")
	}
}

func TestFirestoreService_PathBreakingIdentityRejected(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	badIDs := []string{
		"",
		"alice/sessions/s",
		"a/b/c/d",
		".",
		"..",
		"__reserved__",
		"____",
		strings.Repeat("x", maxIdentityComponentBytes+1),
	}

	for _, bad := range badIDs {
		if _, err := svc.Create(ctx, &session.CreateRequest{AppName: bad, UserID: "alice"}); err == nil {
			t.Errorf("Create(AppName=%q) = nil error, want error", bad)
		}
		if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: bad}); err == nil {
			t.Errorf("Create(UserID=%q) = nil error, want error", bad)
		}
		if bad != "" {
			if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "alice", SessionID: bad}); err == nil {
				t.Errorf("Create(SessionID=%q) = nil error, want error", bad)
			}
		}
		if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: bad, SessionID: "s"}); err == nil {
			t.Errorf("Get(UserID=%q) = nil error, want error", bad)
		}
		if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "alice", SessionID: bad}); err == nil {
			t.Errorf("Get(SessionID=%q) = nil error, want error", bad)
		}
		if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "app", UserID: "alice", SessionID: bad}); err == nil {
			t.Errorf("Delete(SessionID=%q) = nil error, want error", bad)
		}
		if bad != "" {
			if _, err := svc.List(ctx, &session.ListRequest{AppName: "app", UserID: bad}); err == nil {
				t.Errorf("List(UserID=%q) = nil error, want error", bad)
			}
		}
		if _, err := svc.List(ctx, &session.ListRequest{AppName: bad}); err == nil {
			t.Errorf("List(AppName=%q) = nil error, want error", bad)
		}
	}

	// Normal identities including email addresses with dots or single underscores must succeed.
	for _, okID := range []string{"alice", "alice@example.com", "user-123_abc", "a.b.c", "_single_"} {
		resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: okID})
		if err != nil {
			t.Errorf("Create(UserID=%q) unexpected error: %v", okID, err)
			continue
		}
		if resp.Session.UserID() != okID {
			t.Errorf("Session.UserID() = %q, want %q", resp.Session.UserID(), okID)
		}
	}
}

func TestFirestoreService_AppendEvent_OptimisticConcurrencyAndRevision(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// Load a second independent handle on the same session (both at revision 0).
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	staleHandle := got.Session

	// Append through the first handle -> revision becomes 1.
	ev1 := &session.Event{ID: "e1", Author: "agent", InvocationID: "inv1", Timestamp: time.Date(2026, 7, 17, 11, 0, 0, 0, time.UTC)}
	if err := svc.AppendEvent(ctx, created.Session, ev1); err != nil {
		t.Fatalf("AppendEvent(ev1) error = %v", err)
	}

	doc, ok := srv.getDoc("adk-session/app/users/user/sessions/s1")
	if !ok {
		t.Fatal("session document missing")
	}
	if gotRev := doc.GetFields()["revision"].GetIntegerValue(); gotRev != 1 {
		t.Errorf("stored revision after 1st append = %d, want 1", gotRev)
	}

	// Appending through staleHandle (still at revision 0) must fail with ErrStaleSession.
	ev2 := &session.Event{ID: "e2", Author: "agent", InvocationID: "inv2"}
	err = svc.AppendEvent(ctx, staleHandle, ev2)
	if err == nil {
		t.Fatal("AppendEvent through stale handle = nil error, want ErrStaleSession")
	}
	if !errors.Is(err, ErrStaleSession) {
		t.Errorf("AppendEvent through stale handle error = %v, want wrapping ErrStaleSession", err)
	}

	// Appending a second event through the fresh handle (at revision 1) succeeds -> revision 2.
	ev3 := &session.Event{ID: "e3", Author: "agent", InvocationID: "inv3"}
	if err := svc.AppendEvent(ctx, created.Session, ev3); err != nil {
		t.Fatalf("AppendEvent(ev3) error = %v", err)
	}
	doc, _ = srv.getDoc("adk-session/app/users/user/sessions/s1")
	if gotRev := doc.GetFields()["revision"].GetIntegerValue(); gotRev != 2 {
		t.Errorf("stored revision after 2nd append = %d, want 2", gotRev)
	}
}

func TestFirestoreService_DeletingStatusHidesSessionAndRecoversOnRetry(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// Simulate an interrupted Delete that marked the session DELETING before removing it.
	doc, _ := srv.getDoc("adk-session/app/users/user/sessions/s1")
	doc.Fields["status"] = &pb.Value{ValueType: &pb.Value_StringValue{StringValue: "DELETING"}}
	srv.putDoc("adk-session/app/users/user/sessions/s1", doc.Fields)

	err = svc.AppendEvent(ctx, created.Session, &session.Event{ID: "e1", Author: "user", InvocationID: "inv1"})
	if err == nil {
		t.Fatal("AppendEvent on DELETING session = nil error, want error")
	}
	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent on DELETING session error = %v, want wrapping session.ErrNotFound", err)
	}
	if !strings.Contains(err.Error(), "being deleted") {
		t.Errorf("AppendEvent on DELETING session error = %q, want mention of 'being deleted'", err.Error())
	}

	// Get must treat a DELETING session as ErrNotFound, and List must omit it.
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get on DELETING session error = %v, want wrapping session.ErrNotFound", err)
	}
	listResp, err := svc.List(ctx, &session.ListRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(listResp.Sessions) != 0 {
		t.Errorf("List() returned %d sessions, want 0 while session is DELETING", len(listResp.Sessions))
	}

	// Calling Delete again on a session stuck in DELETING must cleanly finish removing it.
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "app", UserID: "user", SessionID: "s1"}); err != nil {
		t.Fatalf("Delete() recovery on DELETING session error = %v", err)
	}
	if _, ok := srv.getDoc("adk-session/app/users/user/sessions/s1"); ok {
		t.Error("session document still exists after recovery Delete()")
	}
}

func TestFirestoreService_AppendEvent_ConcurrentSameSessionHandle(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s-conc"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ev := &session.Event{
				ID:           fmt.Sprintf("ev-%d", idx),
				Author:       "agent",
				InvocationID: "inv1",
				Timestamp:    time.Date(2026, 1, 1, 0, 0, idx, 0, time.UTC),
				Actions: session.EventActions{
					StateDelta: map[string]any{fmt.Sprintf("k%d", idx): idx},
				},
			}
			if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent AppendEvent failed: %v", err)
	}

	// Verify lock table cleaned up after all goroutines finished.
	svc.locksMu.Lock()
	remainingLocks := len(svc.sessionLocks)
	svc.locksMu.Unlock()
	if remainingLocks != 0 {
		t.Errorf("sessionLocks count after completion = %d, want 0", remainingLocks)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s-conc"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got.Session.Events().Len() != workers {
		t.Errorf("Events().Len() = %d, want %d", got.Session.Events().Len(), workers)
	}
}

func TestFirestoreService_AppendEvent_PreservesInputEventTempState(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "testapp",
		UserID:    "testuser",
		SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	event := &session.Event{
		ID:        "event1",
		Author:    "user",
		Timestamp: time.Date(2026, 5, 1, 10, 0, 0, 0, time.UTC),
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"temp:k1": "v1",
				"sk":      "v2",
			},
		},
	}

	if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	// 1. Caller's input event map must not have been mutated in place.
	if _, exists := event.Actions.StateDelta["temp:k1"]; !exists {
		t.Errorf("expected temp:k1 to be preserved on caller's input event, got %v", event.Actions.StateDelta)
	}

	// 2. In-memory session State() has temp:k1 for the current turn, while in-memory Events() stripped it.
	if val, err := created.Session.State().Get("temp:k1"); err != nil || val != "v1" {
		t.Errorf("in-memory session State().Get(temp:k1) = (%v, %v), want (v1, nil)", val, err)
	}
	storedInMem := created.Session.Events().At(0)
	if _, exists := storedInMem.Actions.StateDelta["temp:k1"]; exists {
		t.Errorf("temp:k1 leaked into in-memory session event: %v", storedInMem.Actions.StateDelta)
	}

	// 3. Persisted session state JSON in Firestore must not contain temp:k1.
	sessDoc, ok := srv.getDoc("adk-session/testapp/users/testuser/sessions/s1")
	if !ok {
		t.Fatal("session doc not found")
	}
	if strings.Contains(sessDoc.GetFields()["state"].GetStringValue(), "temp:k1") {
		t.Errorf("temp:k1 leaked into persisted Firestore session state: %s", sessDoc.GetFields()["state"].GetStringValue())
	}
}

func TestFirestoreService_AppendEvent_WorkflowAndTranscriptionFieldsRoundTrip(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	event := &session.Event{
		ID:             "wf_event",
		Author:         "agent",
		InvocationID:   "inv-wf",
		Timestamp:      time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC),
		NodeInfo:       &session.NodeInfo{Path: "ask_name", MessageAsOutput: true, OutputFor: []string{"root"}},
		RequestedInput: &session.RequestInput{InterruptID: "ask_name", Message: "What is your name?"},
		Routes:         []string{"route_a"},
		IsolationScope: "task-1",
		LLMResponse: model.LLMResponse{
			InputTranscription:  &genai.Transcription{Text: "hello", Finished: true},
			OutputTranscription: &genai.Transcription{Text: "hi there"},
		},
	}
	if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got.Session.Events().Len() != 1 {
		t.Fatalf("Events().Len() = %d, want 1", got.Session.Events().Len())
	}
	ev := got.Session.Events().At(0)
	if diff := cmp.Diff(event.NodeInfo, ev.NodeInfo); diff != "" {
		t.Errorf("NodeInfo mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(event.RequestedInput, ev.RequestedInput); diff != "" {
		t.Errorf("RequestedInput mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(event.Routes, ev.Routes); diff != "" {
		t.Errorf("Routes mismatch (-want +got):\n%s", diff)
	}
	if ev.IsolationScope != "task-1" {
		t.Errorf("IsolationScope = %q, want %q", ev.IsolationScope, "task-1")
	}
	if diff := cmp.Diff(event.InputTranscription, ev.InputTranscription); diff != "" {
		t.Errorf("InputTranscription mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(event.OutputTranscription, ev.OutputTranscription); diff != "" {
		t.Errorf("OutputTranscription mismatch (-want +got):\n%s", diff)
	}
}

func TestFirestoreService_CrossRuntimePythonSessionAndEventRead(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	// Simulate a session document and an event document written by adk-python:
	// - session state as a native Firestore map (or JSON string)
	// - event_data with float epoch timestamp and empty nodeInfo: {}
	sessMap := &pb.MapValue{
		Fields: map[string]*pb.Value{
			"py_key": {ValueType: &pb.Value_StringValue{StringValue: "py_val"}},
		},
	}
	srv.putDoc("adk-session/app/users/user/sessions/py-sess", map[string]*pb.Value{
		"id":         {ValueType: &pb.Value_StringValue{StringValue: "py-sess"}},
		"appName":    {ValueType: &pb.Value_StringValue{StringValue: "app"}},
		"userId":     {ValueType: &pb.Value_StringValue{StringValue: "user"}},
		"state":      {ValueType: &pb.Value_MapValue{MapValue: sessMap}},
		"updateTime": {ValueType: &pb.Value_DoubleValue{DoubleValue: 1733040000.0}},
		"revision":   {ValueType: &pb.Value_IntegerValue{IntegerValue: 3}},
	})

	pyEventMap := &pb.MapValue{
		Fields: map[string]*pb.Value{
			"id":           {ValueType: &pb.Value_StringValue{StringValue: "py-evt-1"}},
			"invocationId": {ValueType: &pb.Value_StringValue{StringValue: "py-inv-1"}},
			"author":       {ValueType: &pb.Value_StringValue{StringValue: "user"}},
			"timestamp":    {ValueType: &pb.Value_DoubleValue{DoubleValue: 1733040000.5}},
			"nodeInfo":     {ValueType: &pb.Value_MapValue{MapValue: &pb.MapValue{}}},
			"content": {
				ValueType: &pb.Value_MapValue{
					MapValue: &pb.MapValue{
						Fields: map[string]*pb.Value{
							"role": {ValueType: &pb.Value_StringValue{StringValue: "user"}},
							"parts": {
								ValueType: &pb.Value_ArrayValue{
									ArrayValue: &pb.ArrayValue{
										Values: []*pb.Value{
											{
												ValueType: &pb.Value_MapValue{
													MapValue: &pb.MapValue{
														Fields: map[string]*pb.Value{
															"text": {ValueType: &pb.Value_StringValue{StringValue: "from python"}},
														},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	srv.putDoc("adk-session/app/users/user/sessions/py-sess/events/py-evt-1", map[string]*pb.Value{
		"event_data": {ValueType: &pb.Value_MapValue{MapValue: pyEventMap}},
		"timestamp":  {ValueType: &pb.Value_TimestampValue{TimestampValue: timestamppb.New(time.UnixMilli(1733040000500).UTC())}},
		"appName":    {ValueType: &pb.Value_StringValue{StringValue: "app"}},
		"userId":     {ValueType: &pb.Value_StringValue{StringValue: "user"}},
	})

	// Event 2: a camelCase Event.model_dump(mode="json", by_alias=True) payload,
	// the Vertex AI shape (reused from session/vertexai/raw_event_crossruntime_test.go)
	// rather than what adk-python's Firestore service writes, which
	// TestFirestoreService_Get_ReadsAdkPythonEventData covers. It verifies that
	// camelCase input still decodes: requestedAuthConfigs, functionCall, groundingMetadata,
	// float-formatted integer tokens in artifactDelta and usageMetadata,
	// nodeInfo, output, and float epoch timestamp decode cleanly.
	const pythonDumpedEventJSON = `{
	  "actions": {"artifactDelta": {"artifact": 1.0}, "escalate": true,
	              "requestedAuthConfigs": {}, "requestedToolConfirmations": {},
	              "skipSummarization": true, "stateDelta": {"state": "updated"},
	              "transferToAgent": "test-agent"},
	  "author": "user",
	  "branch": "test-branch",
	  "content": {"parts": [{"text": "Hello, raw_event!"},
	                        {"functionCall": {"args": {"arg1": "value1"},
	                                          "id": "test-id", "name": "test-function"}}],
	              "role": "user"},
	  "customMetadata": {"custom-key": "custom-value"},
	  "errorCode": "400",
	  "errorMessage": "test error",
	  "groundingMetadata": {"retrievalQueries": ["query2"], "webSearchQueries": ["query1"]},
	  "id": "py-evt-2",
	  "interrupted": true,
	  "invocationId": "test-invocation-id",
	  "isolationScope": "scope-1",
	  "longRunningToolIds": ["test-tool-id-1", "test-tool-id-2"],
	  "nodeInfo": {"messageAsOutput": true, "outputFor": ["wf/A@1/B@1", "wf/A@1"],
	               "path": "wf/A@1/B@1"},
	  "output": {"count": 3.0, "result": "ok"},
	  "partial": false,
	  "timestamp": 1733040001.0,
	  "turnComplete": true,
	  "usageMetadata": {"candidatesTokenCount": 50.0, "promptTokenCount": 100.0,
	                    "totalTokenCount": 200.0}
	}`
	srv.putDoc("adk-session/app/users/user/sessions/py-sess/events/py-evt-2", map[string]*pb.Value{
		"event_data": {ValueType: &pb.Value_StringValue{StringValue: pythonDumpedEventJSON}},
		"timestamp":  {ValueType: &pb.Value_TimestampValue{TimestampValue: timestamppb.New(time.Unix(1733040001, 0).UTC())}},
		"appName":    {ValueType: &pb.Value_StringValue{StringValue: "app"}},
		"userId":     {ValueType: &pb.Value_StringValue{StringValue: "user"}},
	})

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "py-sess"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if v, err := got.Session.State().Get("py_key"); err != nil || v != "py_val" {
		t.Errorf("State().Get(py_key) = (%v, %v), want (py_val, nil)", v, err)
	}
	if got.Session.Events().Len() != 2 {
		t.Fatalf("Events().Len() = %d, want 2", got.Session.Events().Len())
	}
	ev := got.Session.Events().At(0)
	if ev.ID != "py-evt-1" || ev.InvocationID != "py-inv-1" {
		t.Errorf("decoded event ID/InvocationID = (%q, %q), want (py-evt-1, py-inv-1)", ev.ID, ev.InvocationID)
	}
	if wantTS := time.UnixMilli(1733040000500).UTC(); !ev.Timestamp.UTC().Equal(wantTS) {
		t.Errorf("decoded event Timestamp = %v, want %v", ev.Timestamp.UTC(), wantTS)
	}
	if ev.NodeInfo != nil {
		t.Errorf("decoded empty Python nodeInfo = %#v, want nil", ev.NodeInfo)
	}

	ev2 := got.Session.Events().At(1)
	if ev2.ID != "py-evt-2" || ev2.Actions.TransferToAgent != "test-agent" {
		t.Errorf("decoded py-evt-2 = (%q, %q), want (py-evt-2, test-agent)", ev2.ID, ev2.Actions.TransferToAgent)
	}
	if ev2.UsageMetadata == nil || ev2.UsageMetadata.TotalTokenCount != 200 {
		t.Errorf("decoded py-evt-2 UsageMetadata = %+v, want TotalTokenCount=200", ev2.UsageMetadata)
	}
	if ev2.NodeInfo == nil || ev2.NodeInfo.Path != "wf/A@1/B@1" || !ev2.NodeInfo.MessageAsOutput {
		t.Errorf("decoded py-evt-2 NodeInfo = %+v, want Path=wf/A@1/B@1, MessageAsOutput=true", ev2.NodeInfo)
	}
}

func TestFirestoreService_Delete_BatchingOver500Events(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "big-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// Seed 505 event documents directly into the fake server.
	for i := range 505 {
		docPath := fmt.Sprintf("adk-session/app/users/user/sessions/big-sess/events/ev-%04d", i)
		srv.putDoc(docPath, map[string]*pb.Value{
			"appName": {ValueType: &pb.Value_StringValue{StringValue: "app"}},
			"userId":  {ValueType: &pb.Value_StringValue{StringValue: "user"}},
		})
	}

	srv.mu.Lock()
	srv.batchCommits = 0
	srv.mu.Unlock()

	if err := svc.Delete(ctx, &session.DeleteRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: created.Session.ID(),
	}); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	// 505 event deletes in batches of 500 = 2 batch commits + 1 session delete commit = 3 non-tx commits.
	srv.mu.Lock()
	batchCommits := srv.batchCommits
	srv.mu.Unlock()
	if batchCommits < 2 {
		t.Errorf("batchCommits = %d, want >= 2 for 505 events", batchCommits)
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "big-sess"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get after Delete error = %v, want wrapping session.ErrNotFound", err)
	}
}

func TestFirestoreService_EventsSharingTimestampOrderIsStable(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	ts := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// Use non-monotonic UUID-like IDs so lexicographical document-ID order
	// differs from insertion order, proving that the stored event revision
	// preserves exact insertion order on timestamp ties.
	nonMonotonicIDs := []string{
		"f47ac10b-58cc-4372-a567-0e02b2c3d479",
		"123e4567-e89b-12d3-a456-426614174000",
		"9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
		"018f1234-5678-7abc-def0-123456789abc",
		"e02fd0e4-00fd-090A-ca30-0d00a0038ba0",
		"550e8400-e29b-41d4-a716-446655440000",
	}
	var want []string
	for i, id := range nonMonotonicIDs {
		ev := &session.Event{
			ID:           id,
			Author:       "author",
			InvocationID: "inv1",
			Timestamp:    ts,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText(fmt.Sprintf("event %d", i), genai.RoleUser),
			},
		}
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
		want = append(want, ev.ID)
	}

	read := func() []string {
		got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
		if err != nil {
			t.Fatalf("Get() error = %v", err)
		}
		var ids []string
		for ev := range got.Session.Events().All() {
			ids = append(ids, ev.ID)
		}
		return ids
	}

	first := read()
	for range 5 {
		if diff := cmp.Diff(first, read()); diff != "" {
			t.Fatalf("order of tied events changed between reads (-first +later):\n%s", diff)
		}
	}
	if diff := cmp.Diff(want, first); diff != "" {
		t.Errorf("tied events order mismatch (-want +got):\n%s", diff)
	}
}

func TestFirestoreService_AppendEvent_RejectsOversizedInlinePayload(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	oversized := make([]byte, maxFirestoreDocumentBytes+1024)
	ev := &session.Event{
		ID:           "huge-event",
		Author:       "user",
		InvocationID: "inv1",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromBytes(oversized, "application/octet-stream", "user"),
		},
	}

	err = svc.AppendEvent(ctx, created.Session, ev)
	if err == nil {
		t.Fatal("AppendEvent with >1 MiB payload = nil error, want error")
	}
	if !strings.Contains(err.Error(), "artifact") {
		t.Errorf("oversized event error = %q, want guidance referencing artifact service", err.Error())
	}
}

func TestFirestoreService_ValidationAndStorageErrors(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	if _, err := svc.Create(ctx, nil); err == nil {
		t.Error("Create(nil) = nil, want error")
	}
	if _, err := svc.Get(ctx, nil); err == nil {
		t.Error("Get(nil) = nil, want error")
	}
	if _, err := svc.List(ctx, nil); err == nil {
		t.Error("List(nil) = nil, want error")
	}
	if err := svc.Delete(ctx, nil); err == nil {
		t.Error("Delete(nil) = nil, want error")
	}
	if err := svc.AppendEvent(ctx, nil, &session.Event{}); err == nil {
		t.Error("AppendEvent(nil session) = nil, want error")
	}

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if err := svc.AppendEvent(ctx, created.Session, nil); err == nil {
		t.Error("AppendEvent(nil event) = nil, want error")
	}
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{ID: "bad/event", Author: "user"}); err == nil {
		t.Error("AppendEvent(path-breaking event ID) = nil, want error")
	}
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		ID:     "nan-state",
		Author: "user",
		Actions: session.EventActions{
			StateDelta: map[string]any{"bad": math.NaN()},
		},
	}); err == nil {
		t.Error("AppendEvent(NaN state delta) = nil, want error")
	}

	// Empty document data in Firestore on Get should report ErrNotFound.
	srv.putDoc("adk-session/app/users/user/sessions/empty-doc", map[string]*pb.Value{})
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "empty-doc"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(empty doc) error = %v, want wrapping session.ErrNotFound", err)
	}

	// Backend read error on Get must not wrap session.ErrNotFound.
	srv.mu.Lock()
	srv.failNextRead = status.Error(codes.PermissionDenied, "permission denied")
	srv.mu.Unlock()
	_, err = svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err == nil {
		t.Fatal("Get() during PermissionDenied = nil, want error")
	}
	if errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get() during PermissionDenied wrapped session.ErrNotFound: %v", err)
	}

	// Backend query error on List.
	srv.mu.Lock()
	srv.failNextQuery = status.Error(codes.PermissionDenied, "query permission denied")
	srv.mu.Unlock()
	if _, err := svc.List(ctx, &session.ListRequest{AppName: "app", UserID: "user"}); err == nil {
		t.Fatal("List() during PermissionDenied = nil, want error")
	}

	// Backend write error on AppendEvent must not mutate in-memory session state or events.
	srv.mu.Lock()
	srv.failNextWrite = status.Error(codes.PermissionDenied, "write permission denied")
	srv.mu.Unlock()
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		ID:     "failed-event",
		Author: "user",
		Actions: session.EventActions{
			StateDelta: map[string]any{"should_not_persist": "yes"},
		},
	}); err == nil {
		t.Fatal("AppendEvent() during PermissionDenied = nil, want error")
	}
	if created.Session.Events().Len() != 0 {
		t.Errorf("in-memory Events().Len() after failed AppendEvent = %d, want 0", created.Session.Events().Len())
	}
	if _, err := created.Session.State().Get("should_not_persist"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("in-memory State().Get(should_not_persist) after failed AppendEvent = %v, want ErrStateKeyNotExist", err)
	}

	// Backend write error on Delete.
	srv.mu.Lock()
	srv.failNextQuery = status.Error(codes.PermissionDenied, "delete query denied")
	srv.mu.Unlock()
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "app", UserID: "user", SessionID: "s1"}); err == nil {
		t.Fatal("Delete() during PermissionDenied = nil, want error")
	}

	// Sort order in List: sorted by (LastUpdateTime, UserID, ID).
	fixedTime := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	sortCtx := platform.WithTimeProvider(ctx, func() time.Time { return fixedTime })
	_, _ = svc.Create(sortCtx, &session.CreateRequest{AppName: "sortApp", UserID: "u2", SessionID: "b"})
	_, _ = svc.Create(sortCtx, &session.CreateRequest{AppName: "sortApp", UserID: "u1", SessionID: "b"})
	_, _ = svc.Create(sortCtx, &session.CreateRequest{AppName: "sortApp", UserID: "u1", SessionID: "a"})

	listResp, err := svc.List(ctx, &session.ListRequest{AppName: "sortApp"})
	if err != nil {
		t.Fatalf("List(sortApp) error = %v", err)
	}
	var gotOrder []string
	for _, s := range listResp.Sessions {
		gotOrder = append(gotOrder, s.UserID()+"/"+s.ID())
	}
	wantOrder := []string{"u1/a", "u1/b", "u2/b"}
	if diff := cmp.Diff(wantOrder, gotOrder); diff != "" {
		t.Errorf("List(sortApp) order mismatch (-want +got):\n%s", diff)
	}
}

func TestFirestoreService_Get_NumRecentEventsReturnsMostRecentTail(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "tail-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	for i := 1; i <= 5; i++ {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%d", i),
			Author:       "user",
			InvocationID: "inv1",
			Timestamp:    base.Add(time.Duration(i) * time.Minute),
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText(fmt.Sprintf("msg %d", i), genai.RoleUser),
			},
		}
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("AppendEvent(%d) error = %v", i, err)
		}
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:         "app",
		UserID:          "user",
		SessionID:       "tail-sess",
		NumRecentEvents: 3,
	})
	if err != nil {
		t.Fatalf("Get(NumRecentEvents=3) error = %v", err)
	}

	var gotIDs []string
	for ev := range got.Session.Events().All() {
		gotIDs = append(gotIDs, ev.ID)
	}
	wantIDs := []string{"ev-3", "ev-4", "ev-5"}
	if diff := cmp.Diff(wantIDs, gotIDs); diff != "" {
		t.Errorf("Get(NumRecentEvents=3) event IDs mismatch (-want +got):\n%s", diff)
	}

	// Combine After and NumRecentEvents: events >= ev-2 are [ev-2, ev-3, ev-4, ev-5],
	// and the last 2 of those are [ev-4, ev-5].
	gotCombined, err := svc.Get(ctx, &session.GetRequest{
		AppName:         "app",
		UserID:          "user",
		SessionID:       "tail-sess",
		After:           base.Add(2 * time.Minute),
		NumRecentEvents: 2,
	})
	if err != nil {
		t.Fatalf("Get(After+NumRecentEvents) error = %v", err)
	}
	var gotCombinedIDs []string
	for ev := range gotCombined.Session.Events().All() {
		gotCombinedIDs = append(gotCombinedIDs, ev.ID)
	}
	wantCombinedIDs := []string{"ev-4", "ev-5"}
	if diff := cmp.Diff(wantCombinedIDs, gotCombinedIDs); diff != "" {
		t.Errorf("Get(After+NumRecentEvents) event IDs mismatch (-want +got):\n%s", diff)
	}

	// When events share equal timestamps and have non-monotonic document IDs,
	// NumRecentEvents must return the most recent events by insertion order
	// (revision), not by Firestore __name__ lexicographical ordering.
	equalTimeSess, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "equal-ts-tail-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	equalTS := base.Add(10 * time.Minute)
	equalIDs := []string{"z-first", "a-second", "y-third", "b-fourth"}
	for _, id := range equalIDs {
		ev := &session.Event{
			ID:           id,
			Author:       "user",
			InvocationID: "inv1",
			Timestamp:    equalTS,
		}
		if err := svc.AppendEvent(ctx, equalTimeSess.Session, ev); err != nil {
			t.Fatalf("AppendEvent(%s) error = %v", id, err)
		}
	}
	gotEqualTail, err := svc.Get(ctx, &session.GetRequest{
		AppName:         "app",
		UserID:          "user",
		SessionID:       "equal-ts-tail-sess",
		NumRecentEvents: 2,
	})
	if err != nil {
		t.Fatalf("Get(NumRecentEvents=2) error = %v", err)
	}
	var gotEqualTailIDs []string
	for ev := range gotEqualTail.Session.Events().All() {
		gotEqualTailIDs = append(gotEqualTailIDs, ev.ID)
	}
	wantEqualTailIDs := []string{"y-third", "b-fourth"}
	if diff := cmp.Diff(wantEqualTailIDs, gotEqualTailIDs); diff != "" {
		t.Errorf("Get(NumRecentEvents=2 on equal timestamps) mismatch (-want +got):\n%s", diff)
	}
}

func TestFirestoreService_AppendEvent_ZeroTimestampIsPopulatedAndQueryable(t *testing.T) {
	createTime := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	eventTime := createTime.Add(30 * time.Second)
	now := createTime
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return now })

	svc, _ := newTestService(t, Config{})
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "zero-ts-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	now = eventTime
	ev := &session.Event{
		ID:           "zero-ts-ev",
		Author:       "user",
		InvocationID: "inv1",
	}
	if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	if !ev.Timestamp.Equal(eventTime) {
		t.Errorf("ev.Timestamp after AppendEvent = %v, want %v", ev.Timestamp, eventTime)
	}
	if !created.Session.LastUpdateTime().Equal(eventTime) {
		t.Errorf("Session.LastUpdateTime() = %v, want %v", created.Session.LastUpdateTime(), eventTime)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "zero-ts-sess",
		After:     createTime.Add(10 * time.Second),
	})
	if err != nil {
		t.Fatalf("Get(After) error = %v", err)
	}
	if got.Session.Events().Len() != 1 {
		t.Fatalf("Get(After) events len = %d, want 1", got.Session.Events().Len())
	}
	if !got.Session.Events().At(0).Timestamp.Equal(eventTime) {
		t.Errorf("stored event Timestamp = %v, want %v", got.Session.Events().At(0).Timestamp, eventTime)
	}
}

func TestFirestoreService_TransactionConflictRetryAndDeleteError(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "occ-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// 1. Simulate a concurrent write to app_states/app between tx.Get and Commit
	// on the first attempt of AppendEvent. RunTransaction must abort with
	// codes.Aborted, retry, and succeed while preserving both app state keys.
	var injectedConflict atomic.Bool
	srv.mu.Lock()
	srv.onBeforeCommit = func(_ string) {
		if injectedConflict.CompareAndSwap(false, true) {
			srv.putDoc("app_states/app", map[string]*pb.Value{
				"concurrent_key": {ValueType: &pb.Value_StringValue{StringValue: "from_other_replica"}},
			})
		}
	}
	srv.mu.Unlock()

	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		ID:           "ev-occ-1",
		Author:       "user",
		InvocationID: "inv1",
		Actions: session.EventActions{
			StateDelta: map[string]any{"app:our_key": "from_this_replica"},
		},
	}); err != nil {
		t.Fatalf("AppendEvent() after transient OCC conflict error = %v", err)
	}
	srv.mu.Lock()
	srv.onBeforeCommit = nil
	aborts := srv.txAborts
	srv.mu.Unlock()
	if aborts < 1 {
		t.Errorf("txAborts = %d, want >= 1", aborts)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "occ-sess",
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if v, err := got.Session.State().Get("app:concurrent_key"); err != nil || v != "from_other_replica" {
		t.Errorf("State().Get(app:concurrent_key) = (%v, %v), want (from_other_replica, nil)", v, err)
	}
	if v, err := got.Session.State().Get("app:our_key"); err != nil || v != "from_this_replica" {
		t.Errorf("State().Get(app:our_key) = (%v, %v), want (from_this_replica, nil)", v, err)
	}

	// 2. If the DELETING-status transaction in Delete fails with a non-NotFound
	// error (e.g. PermissionDenied on read), Delete must return the error rather
	// than swallowing it.
	srv.mu.Lock()
	srv.failNextRead = status.Error(codes.PermissionDenied, "cannot read session for delete")
	srv.mu.Unlock()
	if err := svc.Delete(ctx, &session.DeleteRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "occ-sess",
	}); err == nil {
		t.Fatal("Delete() when DELETING transaction fails = nil, want error")
	}
}

func TestFirestoreService_WithSessionLockRespectsContextCancellation(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})

	holdEntered := make(chan struct{})
	releaseHold := make(chan struct{})
	holderDone := make(chan error, 1)

	go func() {
		holderDone <- svc.withSessionLock(ctx, "app", "user", "locked-sess", func() error {
			close(holdEntered)
			<-releaseHold
			return nil
		})
	}()

	<-holdEntered

	waitCtx, cancel := context.WithCancel(ctx)
	cancel()

	err := svc.withSessionLock(waitCtx, "app", "user", "locked-sess", func() error {
		t.Error("fn executed despite canceled context")
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("withSessionLock(canceled) error = %v, want %v", err, context.Canceled)
	}

	close(releaseHold)
	if err := <-holderDone; err != nil {
		t.Fatalf("holder withSessionLock error = %v", err)
	}

	// Lock map must be empty after both holder and canceled waiter finish.
	svc.locksMu.Lock()
	remainingLocks := len(svc.sessionLocks)
	svc.locksMu.Unlock()
	if remainingLocks != 0 {
		t.Errorf("sessionLocks count = %d, want 0", remainingLocks)
	}
}

func TestFirestoreService_TransactionMaxAttemptsExhaustion(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{
		TransactionMaxAttempts: 1,
	})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "max-attempts-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// With TransactionMaxAttempts: 1, a single concurrent modification of
	// app_states/app between tx.Get and Commit must fail without retrying
	// (whereas the default 5 attempts would retry once and succeed).
	var conflictOnce atomic.Bool
	srv.mu.Lock()
	srv.onBeforeCommit = func(_ string) {
		if conflictOnce.CompareAndSwap(false, true) {
			srv.putDoc("app_states/app", map[string]*pb.Value{
				"other": {ValueType: &pb.Value_StringValue{StringValue: "conflict"}},
			})
		}
	}
	srv.mu.Unlock()
	defer func() {
		srv.mu.Lock()
		srv.onBeforeCommit = nil
		srv.mu.Unlock()
	}()

	err = svc.AppendEvent(ctx, created.Session, &session.Event{
		ID:           "ev-exhaust",
		Author:       "user",
		InvocationID: "inv1",
		Actions: session.EventActions{
			StateDelta: map[string]any{"app:k": "v"},
		},
	})
	if err == nil {
		t.Fatal("AppendEvent() with TransactionMaxAttempts=1 and persistent conflict = nil, want error")
	}
}

func TestFirestoreService_ListAllUsersFiltersCrossRootSessions(t *testing.T) {
	ctx := t.Context()
	client, srv := newTestFirestoreClient(t)

	svc1, err := NewSessionService(client, Config{RootCollection: "root-one"})
	if err != nil {
		t.Fatalf("NewSessionService(root-one) failed: %v", err)
	}
	svc2, err := NewSessionService(client, Config{RootCollection: "root-two"})
	if err != nil {
		t.Fatalf("NewSessionService(root-two) failed: %v", err)
	}

	// Create session under root-one.
	s1, err := svc1.Create(ctx, &session.CreateRequest{
		AppName:   "shared-app",
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("svc1.Create() error = %v", err)
	}
	// Create session under root-two with the same AppName.
	s2, err := svc2.Create(ctx, &session.CreateRequest{
		AppName:   "shared-app",
		UserID:    "user-2",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("svc2.Create() error = %v", err)
	}

	// Also simulate an arbitrary sessions collection document not under root-one.
	srv.putDoc("other-root/shared-app/users/user-3/sessions/sess-3", map[string]*pb.Value{
		"id":      {ValueType: &pb.Value_StringValue{StringValue: "sess-3"}},
		"appName": {ValueType: &pb.Value_StringValue{StringValue: "shared-app"}},
		"userId":  {ValueType: &pb.Value_StringValue{StringValue: "user-3"}},
	})
	// A collection named like root-one but nested under another document.
	srv.putDoc("tenant/t1/root-one/shared-app/users/user-4/sessions/sess-4", map[string]*pb.Value{
		"id":      {ValueType: &pb.Value_StringValue{StringValue: "sess-4"}},
		"appName": {ValueType: &pb.Value_StringValue{StringValue: "shared-app"}},
		"userId":  {ValueType: &pb.Value_StringValue{StringValue: "user-4"}},
	})
	// An appName field that disagrees with the app in the document path.
	srv.putDoc("root-one/other-app/users/user-5/sessions/sess-5", map[string]*pb.Value{
		"id":      {ValueType: &pb.Value_StringValue{StringValue: "sess-5"}},
		"appName": {ValueType: &pb.Value_StringValue{StringValue: "shared-app"}},
		"userId":  {ValueType: &pb.Value_StringValue{StringValue: "user-5"}},
	})
	// A sessions collection under a document that is not in "users".
	srv.putDoc("root-one/shared-app/admins/user-6/sessions/sess-6", map[string]*pb.Value{
		"id":      {ValueType: &pb.Value_StringValue{StringValue: "sess-6"}},
		"appName": {ValueType: &pb.Value_StringValue{StringValue: "shared-app"}},
		"userId":  {ValueType: &pb.Value_StringValue{StringValue: "user-6"}},
	})

	// Listing all users under svc1 must only return sess-1.
	listResp1, err := svc1.List(ctx, &session.ListRequest{AppName: "shared-app"})
	if err != nil {
		t.Fatalf("svc1.List() error = %v", err)
	}
	if len(listResp1.Sessions) != 1 || listResp1.Sessions[0].ID() != s1.Session.ID() {
		t.Errorf("svc1.List() returned %d sessions (IDs: %v), want exactly [sess-1]",
			len(listResp1.Sessions), sessionIDs(listResp1.Sessions))
	}

	// Listing all users under svc2 must only return sess-2.
	listResp2, err := svc2.List(ctx, &session.ListRequest{AppName: "shared-app"})
	if err != nil {
		t.Fatalf("svc2.List() error = %v", err)
	}
	if len(listResp2.Sessions) != 1 || listResp2.Sessions[0].ID() != s2.Session.ID() {
		t.Errorf("svc2.List() returned %d sessions (IDs: %v), want exactly [sess-2]",
			len(listResp2.Sessions), sessionIDs(listResp2.Sessions))
	}
}

func sessionIDs(sessions []session.Session) []string {
	ids := make([]string, len(sessions))
	for i, s := range sessions {
		ids[i] = s.ID()
	}
	return ids
}

func TestFirestoreService_AppendEvent_DuplicateEventIDRejected(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "dup-id-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	ev1 := &session.Event{
		ID:           "duplicate-id",
		Author:       "user",
		InvocationID: "inv1",
	}
	if err := svc.AppendEvent(ctx, created.Session, ev1); err != nil {
		t.Fatalf("AppendEvent(ev1) error = %v", err)
	}

	// Appending an event with the exact same ID must fail with an AlreadyExists
	// error rather than silently overwriting the stored event.
	ev2 := &session.Event{
		ID:           "duplicate-id",
		Author:       "user",
		InvocationID: "inv2",
	}
	err = svc.AppendEvent(ctx, created.Session, ev2)
	if status.Code(err) != codes.AlreadyExists {
		t.Fatalf("AppendEvent(duplicate event ID) error = %v, want AlreadyExists status", err)
	}

	// The failed append must leave the stored event and revision untouched.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "dup-id-sess"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if n := got.Session.Events().Len(); n != 1 {
		t.Fatalf("Events().Len() = %d, want 1", n)
	}
	if inv := got.Session.Events().At(0).InvocationID; inv != "inv1" {
		t.Errorf("stored event InvocationID = %q, want inv1 (overwritten by the duplicate)", inv)
	}
	doc, ok := srv.getDoc("adk-session/app/users/user/sessions/dup-id-sess")
	if !ok {
		t.Fatal("session document not found")
	}
	if rev := doc.GetFields()["revision"].GetIntegerValue(); rev != 1 {
		t.Errorf("session revision = %d, want 1 after the rejected append", rev)
	}
}

func TestFirestoreService_AppendEvent_EncodesPythonCompatibleEventData(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "compat-sess",
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	eventTS := time.Date(2026, 11, 5, 12, 34, 56, 789000, time.UTC)
	compactionStart := time.Date(2026, 3, 4, 5, 6, 7, 123456789, time.UTC)
	ev := &session.Event{
		ID:           "compat-ev",
		Author:       "user",
		InvocationID: "inv-compat",
		Timestamp:    eventTS,
		Branch:       "branch-main",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"user_key_camelCase": "val1",
				"user_key_snake":     "val2",
			},
			ArtifactDelta: map[string]int64{
				"file_name.txt": 1,
			},
			TransferToAgent: "helper-agent",
			Compaction: &session.EventCompaction{
				StartTimestamp:   compactionStart,
				EndTimestamp:     compactionStart.Add(5 * time.Second),
				CompactedContent: genai.NewContentFromText("summary", "model"),
				ExcludedEvents:   []session.EventRef{{InvocationID: "inv-1", Timestamp: compactionStart}},
			},
		},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call-1",
							Name: "my_tool",
							Args: map[string]any{
								"camelCaseParam": "argVal",
							},
						},
					},
				},
			},
			OutputTranscription: &genai.Transcription{Text: "transcription text"},
		},
	}

	if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	// Inspect the raw document in fake Firestore.
	doc, ok := srv.getDoc("adk-session/app/users/user/sessions/compat-sess/events/compat-ev")
	if !ok {
		t.Fatal("event document not found in fake Firestore")
	}

	// With no nested array, event_data must be the native map released
	// adk-python stores with event.model_dump(exclude_none=True, mode="json"):
	// snake_case schema keys, user keys verbatim, integers as integers, float
	// epoch seconds, and a compaction holding only the fields Python's
	// extra='forbid' EventCompaction accepts. This payload was checked with
	// Event.model_validate in google-adk 2.11.0 and 2.13.0.
	if _, ok := doc.GetFields()["event_data"].GetValueType().(*pb.Value_MapValue); !ok {
		t.Fatalf("event_data = %v, want a map", doc.GetFields()["event_data"])
	}
	artifactDelta := doc.GetFields()["event_data"].GetMapValue().GetFields()["actions"].GetMapValue().GetFields()["artifact_delta"]
	if _, ok := artifactDelta.GetMapValue().GetFields()["file_name.txt"].GetValueType().(*pb.Value_IntegerValue); !ok {
		t.Errorf("artifact_delta = %v, want an integer version", artifactDelta)
	}
	snap, err := svc.client.Doc("adk-session/app/users/user/sessions/compat-sess/events/compat-ev").Get(ctx)
	if err != nil {
		t.Fatalf("reading event document: %v", err)
	}
	rawEvent, err := json.Marshal(snap.Data()["event_data"])
	if err != nil {
		t.Fatal(err)
	}
	const wantEventData = `{
	  "actions": {
	    "artifact_delta": {"file_name.txt": 1},
	    "compaction": {
	      "compacted_content": {"parts": [{"text": "summary"}], "role": "model"},
	      "end_timestamp": 1772600772.123456,
	      "start_timestamp": 1772600767.123456
	    },
	    "state_delta": {"user_key_camelCase": "val1", "user_key_snake": "val2"},
	    "transfer_to_agent": "helper-agent"
	  },
	  "adk_go_compaction": {
	    "end_timestamp": "2026-03-04T05:06:12.123456789Z",
	    "excluded_events": [{"invocation_id": "inv-1", "timestamp": "2026-03-04T05:06:07.123456789Z"}],
	    "start_timestamp": "2026-03-04T05:06:07.123456789Z"
	  },
	  "author": "user",
	  "branch": "branch-main",
	  "content": {
	    "parts": [{"function_call": {"args": {"camelCaseParam": "argVal"}, "id": "call-1", "name": "my_tool"}}],
	    "role": "user"
	  },
	  "id": "compat-ev",
	  "invocation_id": "inv-compat",
	  "output_transcription": {"text": "transcription text"},
	  "timestamp": 1793882096.000789
	}`
	var got, want map[string]any
	if err := json.Unmarshal([]byte(rawEvent), &got); err != nil {
		t.Fatalf("event_data is not JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(wantEventData), &want); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("event_data mismatch (-want +got):\n%s", diff)
	}

	// Get must decode back into Go session.Event perfectly.
	gotResp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "compat-sess",
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if gotResp.Session.Events().Len() != 1 {
		t.Fatalf("Events().Len() = %d, want 1", gotResp.Session.Events().Len())
	}
	decodedEv := gotResp.Session.Events().At(0)
	if decodedEv.InvocationID != "inv-compat" {
		t.Errorf("decoded InvocationID = %q, want %q", decodedEv.InvocationID, "inv-compat")
	}
	if decodedEv.Actions.TransferToAgent != "helper-agent" {
		t.Errorf("decoded TransferToAgent = %q, want %q", decodedEv.Actions.TransferToAgent, "helper-agent")
	}
	if decodedEv.Actions.StateDelta["user_key_camelCase"] != "val1" {
		t.Errorf("decoded StateDelta[user_key_camelCase] = %v, want val1", decodedEv.Actions.StateDelta["user_key_camelCase"])
	}
	if decodedEv.Actions.ArtifactDelta["file_name.txt"] != 1 {
		t.Errorf("decoded ArtifactDelta[file_name.txt] = %v, want 1", decodedEv.Actions.ArtifactDelta["file_name.txt"])
	}
	if decodedEv.OutputTranscription == nil || decodedEv.OutputTranscription.Text != "transcription text" {
		t.Errorf("decoded OutputTranscription = %+v, want 'transcription text'", decodedEv.OutputTranscription)
	}
	if diff := cmp.Diff(ev.Actions.Compaction, decodedEv.Actions.Compaction); diff != "" {
		t.Errorf("decoded Compaction mismatch (-want +got):\n%s", diff)
	}
}

func TestFirestoreService_Get_ReadsAdkPythonEventData(t *testing.T) {
	// Both payloads were produced by google-adk 2.11.0 from one Event.
	// adk-python's main branch stores event.model_dump_json(exclude_none=True)
	// (firestore_session_service.py:701); releases up to 2.11.0 stored
	// event.model_dump(exclude_none=True, mode="json") as a native map. In the
	// map case the function response is swapped for one without a nested
	// array, which Firestore rejects, and with a wrapper-like key that
	// adk-python reads verbatim.
	const pythonJSON = `{"content":{"parts":[{"function_call":{"id":"call-1","args":{"camelArg":1},"name":"t"},"thought_signature":"-_-_","part_metadata":{"snake_key":2}},{"function_response":{"id":"call-1","name":"t","response":{"rows":[["a",1]]}}},{"inline_data":{"data":"-_-_AQ==","mime_type":"image/png"}}],"role":"model"},"custom_metadata":{"myKey":"v"},"invocation_id":"py-inv-3","author":"agent","actions":{"state_delta":{"camelKey":"v","user:count":2},"artifact_delta":{"f.txt":1},"transfer_to_agent":"helper","requested_auth_configs":{},"requested_tool_confirmations":{},"compaction":{"start_timestamp":1733040000.5,"end_timestamp":1733040001.75,"compacted_content":{"parts":[{"text":"summary"}],"role":"model"}}},"node_info":{"path":""},"long_running_tool_ids":["lr-1"],"branch":"b","id":"py-evt-3","timestamp":1733040002.25}`
	const pythonMapJSON = `{"actions": {"artifact_delta": {"f.txt": 1}, "compaction": {"compacted_content": {"parts": [{"text": "summary"}], "role": "model"}, "end_timestamp": 1733040001.75, "start_timestamp": 1733040000.5}, "requested_auth_configs": {}, "requested_tool_confirmations": {}, "state_delta": {"camelKey": "v", "user:count": 2}, "transfer_to_agent": "helper"}, "author": "agent", "branch": "b", "content": {"parts": [{"function_call": {"args": {"camelArg": 1}, "id": "call-1", "name": "t"}, "part_metadata": {"snake_key": 2}, "thought_signature": "-_-_"}, {"function_response": {"id": "call-1", "name": "t", "response": {"rows": [{"_adk_list": ["a"]}]}}}, {"inline_data": {"data": "-_-_AQ==", "mime_type": "image/png"}}], "role": "model"}, "custom_metadata": {"myKey": "v"}, "id": "py-evt-3", "invocation_id": "py-inv-3", "long_running_tool_ids": ["lr-1"], "node_info": {"path": ""}, "timestamp": 1733040002.25}`
	var pythonMap map[string]any
	if err := json.Unmarshal([]byte(pythonMapJSON), &pythonMap); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name         string
		eventData    any
		wantResponse map[string]any
	}{
		{name: "json_string", eventData: pythonJSON, wantResponse: map[string]any{"rows": []any{[]any{"a", float64(1)}}}},
		{name: "native_map", eventData: pythonMap, wantResponse: map[string]any{"rows": []any{map[string]any{"_adk_list": []any{"a"}}}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			client, _ := newTestFirestoreClient(t)
			svc, err := NewSessionService(client, Config{})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "py-sess"}); err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			evTS := time.UnixMilli(1733040002250).UTC()
			if _, err := client.Doc("adk-session/app/users/user/sessions/py-sess/events/py-evt-3").Set(ctx, map[string]any{
				"event_data": tc.eventData,
				"timestamp":  evTS,
				"appName":    "app",
				"userId":     "user",
			}); err != nil {
				t.Fatalf("writing Python event: %v", err)
			}

			got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "py-sess"})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			if n := got.Session.Events().Len(); n != 1 {
				t.Fatalf("Events().Len() = %d, want 1", n)
			}
			ev := got.Session.Events().At(0)
			if ev.ID != "py-evt-3" || ev.InvocationID != "py-inv-3" || ev.Branch != "b" || !ev.Timestamp.Equal(evTS) {
				t.Errorf("event = (%q, %q, %q, %v), want (py-evt-3, py-inv-3, b, %v)", ev.ID, ev.InvocationID, ev.Branch, ev.Timestamp, evTS)
			}
			if diff := cmp.Diff([]string{"lr-1"}, ev.LongRunningToolIDs); diff != "" {
				t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"camelKey": "v", "user:count": float64(2)}, ev.Actions.StateDelta); diff != "" {
				t.Errorf("StateDelta mismatch (-want +got):\n%s", diff)
			}
			if ev.Actions.ArtifactDelta["f.txt"] != 1 || ev.Actions.TransferToAgent != "helper" {
				t.Errorf("actions = (%v, %q), want (f.txt:1, helper)", ev.Actions.ArtifactDelta, ev.Actions.TransferToAgent)
			}
			c := ev.Actions.Compaction
			if c == nil {
				t.Fatal("Actions.Compaction = nil, want the Python compaction")
			}
			if !c.StartTimestamp.Equal(time.UnixMilli(1733040000500)) || !c.EndTimestamp.Equal(time.UnixMilli(1733040001750)) {
				t.Errorf("compaction range = [%v, %v], want [1733040000.5, 1733040001.75]", c.StartTimestamp, c.EndTimestamp)
			}
			parts := ev.Content.Parts
			if len(parts) != 3 {
				t.Fatalf("len(Content.Parts) = %d, want 3", len(parts))
			}
			if fc := parts[0].FunctionCall; fc == nil || fc.Name != "t" || fc.Args["camelArg"] != float64(1) {
				t.Errorf("FunctionCall = %+v, want t(camelArg=1)", fc)
			}
			if fr := parts[1].FunctionResponse; fr == nil || !cmp.Equal(tc.wantResponse, fr.Response) {
				t.Errorf("FunctionResponse = %+v, want response %v", fr, tc.wantResponse)
			}
			// pydantic writes bytes as URL-safe base64 ("-_-_"), which Go's
			// []byte decoding rejects unless it is normalized.
			if got := string(parts[0].ThoughtSignature); got != "\xfb\xff\xbf" {
				t.Errorf("ThoughtSignature = %q, want %q", got, "\xfb\xff\xbf")
			}
			if diff := cmp.Diff(map[string]any{"snake_key": float64(2)}, parts[0].PartMetadata); diff != "" {
				t.Errorf("PartMetadata mismatch (-want +got):\n%s", diff)
			}
			if blob := parts[2].InlineData; blob == nil || blob.MIMEType != "image/png" || string(blob.Data) != "\xfb\xff\xbf\x01" {
				t.Errorf("InlineData = %+v, want image/png [fb ff bf 01]", blob)
			}
			if ev.CustomMetadata["myKey"] != "v" {
				t.Errorf("CustomMetadata = %v, want myKey:v", ev.CustomMetadata)
			}
		})
	}
}

func TestFirestoreMapWrappers_MatchAdkPython(t *testing.T) {
	// Each encoded value is what adk-python's _encode_firestore_map
	// (firestore_session_service.py:84-122) produces for the same input.
	tests := []struct {
		name, value, encoded string
	}{
		{"matrix", `[["a","b"],["c",1]]`, `[{"_adk_list":["a","b"]},{"_adk_list":["c",1]}]`},
		{"list_in_dict_in_list", `[{"k":[[1]]}]`, `[{"k":[{"_adk_list":[1]}]}]`},
		{"deep", `[[[1]]]`, `[{"_adk_list":[{"_adk_list":[1]}]}]`},
		{"reserved_list_key_outside_list", `{"_adk_list":[1,2]}`, `{"_adk_list":[1,2]}`},
		{"reserved_list_key_in_list", `[{"_adk_list":[1,2]}]`, `[{"_adk_dict":{"_adk_list":[1,2]}}]`},
		{"reserved_dict_key_in_list", `[{"_adk_dict":{"a":1}}]`, `[{"_adk_dict":{"_adk_dict":{"a":1}}}]`},
		{"two_reserved_keys_in_list", `[{"_adk_list":1,"_adk_dict":2}]`, `[{"_adk_list":1,"_adk_dict":2}]`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var value, wantEncoded any
			if err := json.Unmarshal([]byte(tc.value), &value); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.encoded), &wantEncoded); err != nil {
				t.Fatal(err)
			}
			encoded := encodeFirestoreMap(map[string]any{"v": value})
			if diff := cmp.Diff(wantEncoded, encoded["v"]); diff != "" {
				t.Errorf("encodeFirestoreMap mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"v": value}, decodeFirestoreMap(map[string]any{"v": wantEncoded})); diff != "" {
				t.Errorf("decodeFirestoreMap mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFirestoreService_NestedArraysAndEscapedDicts(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})

	// 1. App state and User state with nested 2D arrays (e.g. matrix: [["a", "b"], ["c", "d"]]).
	matrix := []any{
		[]any{"a", "b"},
		[]any{"c", "d"},
	}
	// A map that contains only reserved wrapper keys "_adk_list" or "_adk_dict".
	trickyDict := map[string]any{
		"_adk_list": "not_actually_a_list",
	}

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "nest-app",
		UserID:  "nest-user",
		State: map[string]any{
			"app:matrix":     matrix,
			"user:matrix":    matrix,
			"user:escaped":   trickyDict,
			"session_matrix": matrix,
		},
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// Verify app_states doc in fake Firestore used _adk_list wrapper so no raw nested array was written.
	appDoc, ok := srv.getDoc("app_states/nest-app")
	if !ok {
		t.Fatal("app_states doc not found")
	}
	matrixVal := appDoc.GetFields()["matrix"]
	if matrixVal == nil || matrixVal.GetArrayValue() == nil {
		t.Fatalf("matrix field is not an ArrayValue: %v", matrixVal)
	}
	firstElem := matrixVal.GetArrayValue().GetValues()[0]
	// First element must be a MapValue with "_adk_list" key.
	if firstElem.GetMapValue() == nil || firstElem.GetMapValue().GetFields()["_adk_list"] == nil {
		t.Errorf("first element of matrix was not encoded with _adk_list: %v", firstElem)
	}

	// 2. Append event with nested arrays in function_response and state_delta.
	ev := &session.Event{
		ID:           "nest-ev",
		Author:       "agent",
		InvocationID: "inv1",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"app:nested_delta": matrix,
				"user:nested_del":  matrix,
				"delta_matrix":     matrix,
			},
		},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "call-1",
							Name: "matrix_tool",
							Response: map[string]any{
								"rows": matrix,
							},
						},
					},
				},
			},
		},
	}

	if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
		t.Fatalf("AppendEvent() with nested arrays error = %v", err)
	}
	// Firestore rejects these arrays in a map, so this event is stored as the
	// JSON string adk-python's main branch writes for every event.
	evDoc, ok := srv.getDoc("adk-session/nest-app/users/nest-user/sessions/" + created.Session.ID() + "/events/nest-ev")
	if !ok {
		t.Fatal("event document not found")
	}
	if _, ok := evDoc.GetFields()["event_data"].GetValueType().(*pb.Value_StringValue); !ok {
		t.Errorf("event_data with nested arrays = %v, want a JSON string", evDoc.GetFields()["event_data"])
	}

	// 3. Read back via Get and List to ensure exact round-trip.
	gotResp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "nest-app",
		UserID:    "nest-user",
		SessionID: created.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	val, err := gotResp.Session.State().Get("app:matrix")
	if err != nil {
		t.Fatalf("State().Get(app:matrix) error = %v", err)
	}
	if diff := cmp.Diff(matrix, val); diff != "" {
		t.Errorf("app:matrix round-trip mismatch (-want +got):\n%s", diff)
	}

	val, err = gotResp.Session.State().Get("user:escaped")
	if err != nil {
		t.Fatalf("State().Get(user:escaped) error = %v", err)
	}
	if diff := cmp.Diff(trickyDict, val); diff != "" {
		t.Errorf("user:escaped round-trip mismatch (-want +got):\n%s", diff)
	}

	if gotResp.Session.Events().Len() != 1 {
		t.Fatalf("Events().Len() = %d, want 1", gotResp.Session.Events().Len())
	}
	decodedFR := gotResp.Session.Events().At(0).Content.Parts[0].FunctionResponse
	if diff := cmp.Diff(matrix, decodedFR.Response["rows"]); diff != "" {
		t.Errorf("function_response rows mismatch (-want +got):\n%s", diff)
	}
}

func TestEventData_KeepsUserKeysVerbatim(t *testing.T) {
	// Keys inside user data must survive both the snake_case encoding and the
	// round trip back to Go. A snake_case user key would come back camelCase,
	// and a camelCase one would be stored snake_case, if either were treated
	// as schema.
	user := func() map[string]any { return map[string]any{"snake_key": "s", "camelKey": "c"} }
	ev := &session.Event{
		ID:        "keys-ev",
		Author:    "agent",
		Timestamp: time.Date(2026, 1, 2, 3, 4, 5, 6000, time.UTC),
		Output:    user(),
		RequestedInput: &session.RequestInput{
			InterruptID: "int-1",
			Payload:     user(),
		},
		Actions: session.EventActions{
			StateDelta: user(),
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call_1": {Hint: "h", Payload: user()},
			},
		},
		LLMResponse: model.LLMResponse{
			CustomMetadata: user(),
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "f", Args: user()}, PartMetadata: user()},
				{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "f", Response: user()}},
				{ToolCall: &genai.ToolCall{ID: "tc_1", Args: user()}},
				{ToolResponse: &genai.ToolResponse{ID: "tc_1", Response: user()}},
			}},
		},
	}

	raw, err := encodeEventData(ev)
	if err != nil {
		t.Fatalf("encodeEventData() error = %v", err)
	}
	rawJSON, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	const userMaps = 10
	if got := strings.Count(string(rawJSON), `"camelKey":"c"`); got != userMaps {
		t.Errorf("event_data keeps camelKey verbatim in %d places, want %d:\n%s", got, userMaps, rawJSON)
	}
	if got := strings.Count(string(rawJSON), `"snake_key":"s"`); got != userMaps {
		t.Errorf("event_data keeps snake_key verbatim in %d places, want %d:\n%s", got, userMaps, rawJSON)
	}

	got, err := decodeEventData(raw)
	if err != nil {
		t.Fatalf("decodeEventData() error = %v", err)
	}
	if diff := cmp.Diff(ev, got); diff != "" {
		t.Errorf("event round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDecodeEventData_KeepsResponseSchemaKeywords(t *testing.T) {
	// JSON Schema keywords are camelCase and must not be rewritten when a
	// snake_case adk-python event is decoded.
	const raw = `{"id":"e","author":"a","timestamp":1.5,"requested_input":{"interrupt_id":"i","response_schema":{"type":"object","additionalProperties":false,"properties":{"snake_field":{"type":"string"}}}}}`
	ev, err := decodeEventData(raw)
	if err != nil {
		t.Fatalf("decodeEventData() error = %v", err)
	}
	s := ev.RequestedInput.ResponseSchema
	if s == nil || s.AdditionalProperties == nil {
		t.Fatalf("ResponseSchema = %+v, want additionalProperties kept", s)
	}
	if _, ok := s.Properties["snake_field"]; !ok {
		t.Errorf("ResponseSchema.Properties = %v, want snake_field kept", s.Properties)
	}
}

func TestFirestoreService_AppendEvent_KeepsStoredStateTypes(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})
	stamp := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	// App state as adk-python writes it, with native Firestore types.
	srv.putDoc("app_states/app", map[string]*pb.Value{
		"count": {ValueType: &pb.Value_IntegerValue{IntegerValue: 2}},
		"when":  {ValueType: &pb.Value_TimestampValue{TimestampValue: timestamppb.New(stamp)}},
		"blob":  {ValueType: &pb.Value_BytesValue{BytesValue: []byte{1, 2}}},
	})
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		ID:      "e1",
		Author:  "agent",
		Actions: session.EventActions{StateDelta: map[string]any{"app:goCount": 3, "app:ratio": 0.5}},
	}); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	doc, ok := srv.getDoc("app_states/app")
	if !ok {
		t.Fatal("app state document not found")
	}
	f := doc.GetFields()
	if _, ok := f["count"].GetValueType().(*pb.Value_IntegerValue); !ok {
		t.Errorf("count = %v, want an integer value kept", f["count"])
	}
	if got := f["when"].GetTimestampValue(); got == nil || !got.AsTime().Equal(stamp) {
		t.Errorf("when = %v, want timestamp %v kept", f["when"], stamp)
	}
	if got := f["blob"].GetBytesValue(); string(got) != "\x01\x02" {
		t.Errorf("blob = %v, want bytes kept", f["blob"])
	}
	if got, ok := f["goCount"].GetValueType().(*pb.Value_IntegerValue); !ok || got.IntegerValue != 3 {
		t.Errorf("goCount = %v, want integer 3", f["goCount"])
	}
	if got := f["ratio"].GetDoubleValue(); got != 0.5 {
		t.Errorf("ratio = %v, want double 0.5", f["ratio"])
	}
}

func TestFirestoreService_RejectsOversizedSharedState(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})
	// Sized from Firestore's 1 MiB document limit, not from the package's cap.
	big := strings.Repeat("x", 1<<20)
	for _, key := range []string{"app:big", "user:big"} {
		_, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", State: map[string]any{key: big}})
		if err == nil || !strings.Contains(err.Error(), "serialized state") {
			t.Errorf("Create(%s of 1 MiB) error = %v, want state payload limit error", key, err)
		}
	}

	// Each event fits, but together they push the shared document past 1 MiB.
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	half := strings.Repeat("x", 600<<10)
	for _, prefix := range []string{"app:", "user:"} {
		if err := svc.AppendEvent(ctx, created.Session, &session.Event{
			ID: prefix + "1", Author: "agent",
			Actions: session.EventActions{StateDelta: map[string]any{prefix + "a": half}},
		}); err != nil {
			t.Fatalf("AppendEvent(%sa of 600 KiB) error = %v", prefix, err)
		}
		err := svc.AppendEvent(ctx, created.Session, &session.Event{
			ID: prefix + "2", Author: "agent",
			Actions: session.EventActions{StateDelta: map[string]any{prefix + "b": half}},
		})
		if err == nil || !strings.Contains(err.Error(), "serialized state") {
			t.Errorf("AppendEvent(%sb past 1 MiB) error = %v, want state payload limit error", prefix, err)
		}
	}
}

func TestDecodeEventData_NormalizesURLSafeBase64(t *testing.T) {
	// pydantic writes every genai bytes field as URL-safe base64; "-_-_" is
	// 0xfb 0xff 0xbf, which the standard alphabet spells "+/+/".
	const raw = `{"id":"e","author":"a","timestamp":1.5,
	  "content":{"parts":[
	    {"text":"t","thought_signature":"-_-_"},
	    {"inline_data":{"mime_type":"image/png","data":"-_-_"}},
	    {"function_response":{"name":"f","response":{},"parts":[{"inline_data":{"mime_type":"image/png","data":"-_-_"}}]}}]},
	  "actions":{"compaction":{"start_timestamp":1,"end_timestamp":2,
	    "compacted_content":{"parts":[{"text":"s","thought_signature":"-_-_"}]}}},
	  "grounding_metadata":{"search_entry_point":{"rendered_content":"<b>","sdk_blob":"-_-_"}}}`
	ev, err := decodeEventData(raw)
	if err != nil {
		t.Fatalf("decodeEventData() error = %v", err)
	}
	const want = "\xfb\xff\xbf"
	parts := ev.Content.Parts
	got := map[string]string{
		"thought_signature":         string(parts[0].ThoughtSignature),
		"inline_data":               string(parts[1].InlineData.Data),
		"function_response_part":    string(parts[2].FunctionResponse.Parts[0].InlineData.Data),
		"compacted_content_thought": string(ev.Actions.Compaction.CompactedContent.Parts[0].ThoughtSignature),
		"search_entry_point_blob":   string(ev.GroundingMetadata.SearchEntryPoint.SDKBlob),
	}
	for field, v := range got {
		if v != want {
			t.Errorf("%s = %q, want %q", field, v, want)
		}
	}
}

func TestFirestoreService_Get_OrdersTiedTimestampsByRevision(t *testing.T) {
	ctx := t.Context()
	svc, srv := newTestService(t, Config{})
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ts := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	// One timestamp, revisions mixed with an adk-python event that has none.
	// Comparing revisions only when both are set orders these in a cycle.
	for _, e := range []struct {
		id       string
		revision int64
	}{{"z-rev1", 1}, {"m-norev", 0}, {"a-rev2", 2}} {
		fields := map[string]*pb.Value{
			"event_data": {ValueType: &pb.Value_StringValue{StringValue: fmt.Sprintf(`{"id":%q,"author":"a","timestamp":%v}`, e.id, float64(ts.Unix()))}},
			"timestamp":  {ValueType: &pb.Value_TimestampValue{TimestampValue: timestamppb.New(ts)}},
		}
		if e.revision > 0 {
			fields["revision"] = &pb.Value{ValueType: &pb.Value_IntegerValue{IntegerValue: e.revision}}
		}
		srv.putDoc("adk-session/app/users/user/sessions/s/events/"+e.id, fields)
	}

	for _, n := range []int{0, 2} {
		got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: n})
		if err != nil {
			t.Fatalf("Get(NumRecentEvents=%d) error = %v", n, err)
		}
		var ids []string
		for ev := range got.Session.Events().All() {
			ids = append(ids, ev.ID)
		}
		want := []string{"m-norev", "z-rev1", "a-rev2"}
		if n > 0 {
			want = want[len(want)-n:]
		}
		if diff := cmp.Diff(want, ids); diff != "" {
			t.Errorf("Get(NumRecentEvents=%d) event IDs mismatch (-want +got):\n%s", n, diff)
		}
	}
}

func TestFirestoreService_Get_AfterIsTruncatedToMicroseconds(t *testing.T) {
	ctx := t.Context()
	svc, _ := newTestService(t, Config{})
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ts := time.Date(2026, 1, 2, 3, 4, 5, 123456000, time.UTC)
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{ID: "e1", Author: "a", Timestamp: ts}); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	// The stored timestamp has microsecond resolution, so a cursor carrying
	// the event's own sub-microsecond remainder must still include it.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", After: ts.Add(789 * time.Nanosecond)})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if n := got.Session.Events().Len(); n != 1 {
		t.Errorf("Get(After=ts+789ns) returned %d events, want 1", n)
	}
}
