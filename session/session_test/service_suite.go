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

package session_test

import (
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// ExpectedSession represents a snapshot of a session's public state for test comparisons.
type ExpectedSession struct {
	AppName   string
	UserID    string
	SessionID string
	State     map[string]any
	Events    []*session.Event
}

// Snapshot extracts values from ANY session.Session implementation.
func Snapshot(s session.Session) ExpectedSession {
	if s == nil {
		return ExpectedSession{}
	}

	state := make(map[string]any)
	if s.State() != nil {
		for k, v := range s.State().All() {
			state[k] = v
		}
	}

	var events []*session.Event
	if s.Events() != nil {
		for e := range s.Events().All() {
			events = append(events, e)
		}
	}

	return ExpectedSession{
		AppName:   s.AppName(),
		UserID:    s.UserID(),
		SessionID: s.ID(),
		State:     state,
		Events:    events,
	}
}

// SuiteOptions holds configuration for adaptive test runs.
type SuiteOptions struct {
	SupportsUserProvidedSessionID bool
	ProvidesServerAssignedEventID bool
	AppName                       string
}

// RunServiceTests runs a battery of standard tests against a Session.Service.
func RunServiceTests(t *testing.T, opts SuiteOptions, setup func(t *testing.T) session.Service) {
	testAppName := "testApp"
	if opts.AppName != "" {
		testAppName = opts.AppName
	}
	t.Run("Create", func(t *testing.T) {
		t.Run("full_key", func(t *testing.T) {
			if !opts.SupportsUserProvidedSessionID {
				t.Skip("Skipping full key test: requires user provided session ID support")
			}
			s := setup(t)
			req := &session.CreateRequest{
				AppName:   testAppName,
				UserID:    "testUserID",
				SessionID: "testSessionID",
				State: map[string]any{
					"k": float64(5),
				},
			}

			got, err := s.Create(t.Context(), req)
			if opts.SupportsUserProvidedSessionID {
				if err != nil {
					t.Fatalf("Create() error = %v, wantErr %v", err, false)
				}
				if got.Session.AppName() != req.AppName {
					t.Errorf("AppName got: %v, want: %v", got.Session.AppName(), req.AppName)
				}
				if got.Session.UserID() != req.UserID {
					t.Errorf("UserID got: %v, want: %v", got.Session.UserID(), req.UserID)
				}
				if got.Session.ID() != req.SessionID {
					t.Errorf("SessionID got: %v, want: %v", got.Session.ID(), req.SessionID)
				}
			} else {
				if err == nil {
					t.Fatalf("Expected error for user-provided SessionID")
				}
			}
		})

		t.Run("generated_session_id", func(t *testing.T) {
			s := setup(t)
			req := &session.CreateRequest{
				AppName: testAppName,
				UserID:  "testUserID",
				State:   map[string]any{"k": float64(5)},
			}

			got, err := s.Create(t.Context(), req)
			if err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			if got.Session.ID() == "" {
				t.Errorf("Expected generated SessionID, got empty")
			}
		})

		t.Run("when_already_exists,_it_fails", func(t *testing.T) {
			s := setup(t)
			req := &session.CreateRequest{
				AppName: testAppName,
				UserID:  "testUserID",
			}

			got1, err := s.Create(t.Context(), req)
			if err != nil {
				t.Fatalf("First Create() failed: %v", err)
			}

			req2 := &session.CreateRequest{
				AppName:   req.AppName,
				UserID:    req.UserID,
				SessionID: got1.Session.ID(),
			}

			_, err = s.Create(t.Context(), req2)
			if opts.SupportsUserProvidedSessionID {
				if err == nil {
					t.Errorf("Expected failure when creating duplicate session")
				}
			} else {
				if err == nil {
					t.Errorf("Expected failure (unsupported or duplicate)")
				}
			}
		})
	})

	t.Run("Get", func(t *testing.T) {
		t.Run("ok", func(t *testing.T) {
			s := setup(t)
			req := &session.CreateRequest{
				AppName: testAppName,
				UserID:  "testUserID",
				State:   map[string]any{"k1": "v1"},
			}
			created, err := s.Create(t.Context(), req)
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			got, err := s.Get(t.Context(), &session.GetRequest{
				AppName:   req.AppName,
				UserID:    req.UserID,
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}

			snap := Snapshot(got.Session)
			if snap.AppName != req.AppName {
				t.Errorf("Get AppName = %v, want %v", snap.AppName, req.AppName)
			}
			if snap.UserID != req.UserID {
				t.Errorf("Get UserID = %v, want %v", snap.UserID, req.UserID)
			}
			if snap.State["k1"] != "v1" {
				t.Errorf("Get State[k1] = %v, want v1", snap.State["k1"])
			}
		})

		t.Run("error_when_not_found", func(t *testing.T) {
			s := setup(t)
			_, err := s.Get(t.Context(), &session.GetRequest{
				AppName:   "nonExistent",
				UserID:    "user",
				SessionID: "s1",
			})
			if err == nil {
				t.Errorf("Expected error for non-existent session")
			}
		})

		t.Run("get_session_respects_user_id", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			c1, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Create user1 failed: %v", err)
			}

			err = s.AppendEvent(ctx, c1.Session, &session.Event{
				ID:           "event1",
				Author:       "user",
				InvocationID: "inv1",
			})
			if err != nil {
				t.Fatalf("AppendEvent failed: %v", err)
			}

			_, err = s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user2",
				SessionID: c1.Session.ID(),
			})
			if err == nil {
				t.Errorf("Expected error or not found when getting session with wrong UserID")
			}
		})

		t.Run("with_config_filters", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()
			req := &session.CreateRequest{AppName: testAppName, UserID: "user1"}
			if opts.SupportsUserProvidedSessionID {
				req.SessionID = "s1"
			}
			created, err := s.Create(ctx, req)
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			numTestEvents := 5
			var timestamps []time.Time
			baseTime := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

			for i := 1; i <= numTestEvents; i++ {
				ts := baseTime.Add(time.Duration(i) * time.Second)
				timestamps = append(timestamps, ts)
				event := &session.Event{
					ID:           strconv.Itoa(i),
					Author:       "user",
					InvocationID: "inv1",
					Timestamp:    ts,
				}
				if err := s.AppendEvent(ctx, created.Session, event); err != nil {
					t.Fatalf("AppendEvent failed: %v", err)
				}
			}

			gotNormal, err := s.Get(ctx, &session.GetRequest{AppName: testAppName, UserID: "user1", SessionID: created.Session.ID()})
			if err != nil {
				t.Fatalf("Get error: %v", err)
			}
			if gotNormal.Session.Events().Len() != 5 {
				t.Errorf("Get no filter events len = %d, want 5", gotNormal.Session.Events().Len())
			}

			gotLimit, err := s.Get(ctx, &session.GetRequest{AppName: testAppName, UserID: "user1", SessionID: created.Session.ID(), NumRecentEvents: 3})
			if err != nil {
				t.Fatalf("Get error: %v", err)
			}
			if gotLimit.Session.Events().Len() != 3 {
				t.Errorf("Get NumRecentEvents len = %d, want 3", gotLimit.Session.Events().Len())
			}

			gotAfter, err := s.Get(ctx, &session.GetRequest{AppName: testAppName, UserID: "user1", SessionID: created.Session.ID(), After: timestamps[1]})
			if err != nil {
				t.Fatalf("Get error: %v", err)
			}
			if gotAfter.Session.Events().Len() != 4 {
				t.Errorf("Get After filter events len = %d, want 4", gotAfter.Session.Events().Len())
			}

			gotCombined, err := s.Get(ctx, &session.GetRequest{AppName: testAppName, UserID: "user1", SessionID: created.Session.ID(), NumRecentEvents: 2, After: timestamps[0]})
			if err != nil {
				t.Fatalf("Get error: %v", err)
			}
			if gotCombined.Session.Events().Len() != 2 {
				t.Errorf("Get Combined filters len = %d, want 2", gotCombined.Session.Events().Len())
			}
		})
	})

	t.Run("List", func(t *testing.T) {
		s := setup(t)
		ctx := t.Context()

		_, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
		if err != nil {
			t.Fatalf("Create user1 session1 failed: %v", err)
		}
		_, err = s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
		if err != nil {
			t.Fatalf("Create user1 session2 failed: %v", err)
		}

		_, err = s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user2"})
		if err != nil {
			t.Fatalf("Create user2 failed: %v", err)
		}

		got1, err := s.List(ctx, &session.ListRequest{AppName: testAppName, UserID: "user1"})
		if err != nil {
			t.Fatalf("List user1 failed: %v", err)
		}
		if len(got1.Sessions) != 2 {
			t.Errorf("List user1 len = %d, want 2", len(got1.Sessions))
		}

		got2, err := s.List(ctx, &session.ListRequest{AppName: testAppName, UserID: "user2"})
		if err != nil {
			t.Fatalf("List user2 failed: %v", err)
		}
		if len(got2.Sessions) != 1 {
			t.Errorf("List user2 len = %d, want 1", len(got2.Sessions))
		}

		got3, err := s.List(ctx, &session.ListRequest{AppName: testAppName, UserID: "nonExistent"})
		if err != nil {
			t.Fatalf("List nonExistent failed: %v", err)
		}
		if len(got3.Sessions) != 0 {
			t.Errorf("List nonExistent len = %d, want 0", len(got3.Sessions))
		}

		gotAll, err := s.List(ctx, &session.ListRequest{AppName: testAppName})
		if err != nil {
			t.Fatalf("List all users failed: %v", err)
		}
		if len(gotAll.Sessions) != 3 {
			t.Errorf("List all users len = %d, want 3", len(gotAll.Sessions))
		}
	})

	t.Run("Delete", func(t *testing.T) {
		s := setup(t)
		ctx := t.Context()

		created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
		if err != nil {
			t.Fatalf("Setup: Create failed: %v", err)
		}

		err = s.Delete(ctx, &session.DeleteRequest{
			AppName:   testAppName,
			UserID:    "user1",
			SessionID: created.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Delete() error = %v", err)
		}

		_, err = s.Get(ctx, &session.GetRequest{
			AppName:   testAppName,
			UserID:    "user1",
			SessionID: created.Session.ID(),
		})
		if err == nil {
			t.Errorf("Expected error when getting deleted session")
		}

		if opts.SupportsUserProvidedSessionID {
			err = s.Delete(ctx, &session.DeleteRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: "nonExistent",
			})
			if err != nil {
				t.Errorf("Delete() non-existent error = %v, want nil", err)
			}
		}
	})

	t.Run("AppendEvent", func(t *testing.T) {
		t.Run("when_session_not_found_should_fail", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			m := &mockSession{appName: testAppName, userID: "user1", id: "nonExistent"}
			event := &session.Event{ID: "event1", Author: "user", InvocationID: "inv1"}

			err := s.AppendEvent(ctx, m, event)
			if err == nil {
				t.Errorf("AppendEvent() expected error for non-existent session, got nil")
			}
		})

		t.Run("ok", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			event := &session.Event{
				ID:           "new_event1",
				Author:       "user",
				InvocationID: "inv1",
			}
			err = s.AppendEvent(ctx, created.Session, event)
			if err != nil {
				t.Fatalf("AppendEvent() error = %v", err)
			}

			got, err := s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}

			snap := Snapshot(got.Session)
			if len(snap.Events) != 1 {
				t.Errorf("Expected 1 event, got %d", len(snap.Events))
			} else {
				if !opts.ProvidesServerAssignedEventID && snap.Events[0].ID != event.ID {
					t.Errorf("Event ID mismatch: got %v, want %v", snap.Events[0].ID, event.ID)
				}
			}
		})

		t.Run("partial_events_are_not_persisted", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			event := &session.Event{
				ID:           "partial_event",
				Author:       "user",
				InvocationID: "inv1",
				LLMResponse: model.LLMResponse{
					Partial: true,
				},
			}
			err = s.AppendEvent(ctx, created.Session, event)
			if err != nil {
				t.Fatalf("AppendEvent() error = %v", err)
			}

			got, err := s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}

			snap := Snapshot(got.Session)
			if len(snap.Events) != 0 {
				t.Errorf("Expected 0 events (Partial=true should be skipped), got %d", len(snap.Events))
			}
		})

		t.Run("with_bytes_content", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			event := &session.Event{
				ID:           "event_with_bytes",
				Author:       "user",
				InvocationID: "inv1",
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromBytes([]byte("test_image_data"), "image/png", "user"),
				},
			}
			err = s.AppendEvent(ctx, created.Session, event)
			if err != nil {
				t.Fatalf("AppendEvent() error = %v", err)
			}

			got, err := s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}

			snap := Snapshot(got.Session)
			if len(snap.Events) != 1 {
				t.Fatalf("Expected 1 event, got %d", len(snap.Events))
			}
			gotEvent := snap.Events[0]

			if diff := cmp.Diff(event.LLMResponse.Content, gotEvent.LLMResponse.Content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}
		})

		t.Run("with_existing_events", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			baseTime := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
			event1 := &session.Event{ID: "event1", Author: "user", InvocationID: "inv1", Timestamp: baseTime.Add(1 * time.Second)}
			if err := s.AppendEvent(ctx, created.Session, event1); err != nil {
				t.Fatalf("AppendEvent(1) failed: %v", err)
			}

			event2 := &session.Event{ID: "event2", Author: "user", InvocationID: "inv2", Timestamp: baseTime.Add(2 * time.Second)}
			if err := s.AppendEvent(ctx, created.Session, event2); err != nil {
				t.Fatalf("AppendEvent(2) failed: %v", err)
			}

			got, err := s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() failed: %v", err)
			}

			snap := Snapshot(got.Session)
			if len(snap.Events) != 2 {
				t.Fatalf("Expected 2 events, got %d", len(snap.Events))
			}
			expectedOrder := []string{"inv1", "inv2"}
			for i := range snap.Events {
				if snap.Events[i].InvocationID != expectedOrder[i] {
					t.Errorf("Expected event order for %v, got events with InvocationIDs", expectedOrder)
					break
				}
			}
		})

		t.Run("with_all_fields", func(t *testing.T) {
			s := setup(t)
			ctx := t.Context()

			created, err := s.Create(ctx, &session.CreateRequest{AppName: testAppName, UserID: "user1"})
			if err != nil {
				t.Fatalf("Setup: Create failed: %v", err)
			}

			event := &session.Event{
				ID:                 "event_complete",
				Author:             "user",
				InvocationID:       "inv1",
				LongRunningToolIDs: []string{"tool123"},
				Actions:            session.EventActions{StateDelta: map[string]any{"user:k2": "v2"}},
				LLMResponse: model.LLMResponse{
					Content:      genai.NewContentFromText("test_text", "user"),
					TurnComplete: true,
					Partial:      false,
					ErrorCode:    "error_code",
					ErrorMessage: "error_message",
					Interrupted:  true,
					GroundingMetadata: &genai.GroundingMetadata{
						WebSearchQueries: []string{"query1"},
					},
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     1,
						CandidatesTokenCount: 1,
						TotalTokenCount:      2,
					},
					CitationMetadata: &genai.CitationMetadata{
						Citations: []*genai.Citation{{Title: "test", URI: "google.com"}},
					},
					CustomMetadata: map[string]any{
						"custom_key": "custom_value",
					},
				},
			}

			err = s.AppendEvent(ctx, created.Session, event)
			if err != nil {
				t.Fatalf("AppendEvent() error = %v", err)
			}

			got, err := s.Get(ctx, &session.GetRequest{
				AppName:   testAppName,
				UserID:    "user1",
				SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}

			snap := Snapshot(got.Session)
			if len(snap.Events) != 1 {
				t.Fatalf("Expected 1 event, got %d", len(snap.Events))
			}
			gotEvent := snap.Events[0]

			cmpOpts := []cmp.Option{
				cmp.AllowUnexported(session.Event{}),
			}
			if opts.ProvidesServerAssignedEventID {
				cmpOpts = append(cmpOpts, cmpopts.IgnoreFields(session.Event{}, "ID"))
				cmpOpts = append(cmpOpts, cmpopts.IgnoreFields(model.LLMResponse{}, "CitationMetadata", "UsageMetadata"))
			}

			if diff := cmp.Diff(event, gotEvent, cmpOpts...); diff != "" {
				t.Errorf("Event mismatch (-want +got):\n%s", diff)
			}
		})
	})

	t.Run("StateManagement", func(t *testing.T) {
		ctx := t.Context()
		appName := testAppName

		t.Run("app_state_is_shared", func(t *testing.T) {
			s := setup(t)
			s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"app:k1": "v1"}})
			if err := s.AppendEvent(ctx, s1.Session, &session.Event{
				ID:           "event1",
				Author:       "user",
				InvocationID: "inv1",
				Actions:      session.EventActions{StateDelta: map[string]any{"app:k2": "v2"}},
			}); err != nil {
				t.Fatalf("AppendEvent failed: %v", err)
			}

			s2, err := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2"})
			if err != nil {
				t.Fatalf("Create user2 failed: %v", err)
			}

			snap := Snapshot(s2.Session)
			if snap.State["app:k1"] != "v1" || snap.State["app:k2"] != "v2" {
				t.Errorf("App state not shared, got: %v", snap.State)
			}
		})

		t.Run("user_state_is_user_specific", func(t *testing.T) {
			s := setup(t)
			s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"user:k1": "v1"}})
			if err := s.AppendEvent(ctx, s1.Session, &session.Event{
				ID:           "event1",
				Author:       "user",
				InvocationID: "inv1",
				Actions:      session.EventActions{StateDelta: map[string]any{"user:k2": "v2"}},
			}); err != nil {
				t.Fatalf("AppendEvent failed: %v", err)
			}

			s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
			snap1b := Snapshot(s1b.Session)
			if snap1b.State["user:k1"] != "v1" || snap1b.State["user:k2"] != "v2" {
				t.Errorf("User state missing for same user, got: %v", snap1b.State)
			}

			s2, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2"})
			snap2 := Snapshot(s2.Session)
			if _, exists := snap2.State["user:k1"]; exists {
				t.Errorf("User state leaked to user2, got: %v", snap2.State)
			}
		})

		t.Run("session_state_is_not_shared", func(t *testing.T) {
			s := setup(t)
			s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"sk1": "v1"}})
			if err := s.AppendEvent(ctx, s1.Session, &session.Event{
				ID:           "event1",
				Author:       "user",
				InvocationID: "inv1",
				Actions:      session.EventActions{StateDelta: map[string]any{"sk2": "v2"}},
			}); err != nil {
				t.Fatalf("AppendEvent failed: %v", err)
			}

			s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
			snapS1b := Snapshot(s1b.Session)
			if _, exists := snapS1b.State["sk1"]; exists {
				t.Errorf("Session state leaked between sessions, got: %v", snapS1b.State)
			}
		})

		t.Run("temp_state_is_not_persisted", func(t *testing.T) {
			s := setup(t)
			s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
			_ = s.AppendEvent(ctx, s1.Session, &session.Event{
				ID:           "event1",
				Author:       "user",
				InvocationID: "inv1",
				Actions:      session.EventActions{StateDelta: map[string]any{"temp:k1": "v1", "sk": "v2"}},
			})

			got, _ := s.Get(ctx, &session.GetRequest{AppName: appName, UserID: "u1", SessionID: s1.Session.ID()})
			snap := Snapshot(got.Session)
			if _, exists := snap.State["temp:k1"]; exists {
				t.Errorf("Temp state leaked to persist step, got: %v", snap.State)
			}
			if snap.State["sk"] != "v2" {
				t.Errorf("Standard state update missing: got %v", snap.State)
			}
		})
	})
}

type mockSession struct {
	appName string
	userID  string
	id      string
}

func (m *mockSession) ID() string                { return m.id }
func (m *mockSession) AppName() string           { return m.appName }
func (m *mockSession) UserID() string            { return m.userID }
func (m *mockSession) State() session.State      { return nil }
func (m *mockSession) Events() session.Events    { return nil }
func (m *mockSession) LastUpdateTime() time.Time { return time.Now() }
