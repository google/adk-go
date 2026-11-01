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

	// Event 2: cross-runtime adk-python Event.model_dump(mode="json", by_alias=True)
	// payload (reused from session/vertexai/raw_event_crossruntime_test.go),
	// verifying that requestedAuthConfigs, functionCall, groundingMetadata,
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
