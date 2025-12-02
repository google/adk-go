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

package database

import (
	"maps"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func Test_databaseService_Create(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T) *databaseService
		req     *session.CreateRequest
		want    session.Session
		wantErr bool
	}{
		{
			name:  "full key",
			setup: emptyService,
			req: &session.CreateRequest{
				AppName:   "testApp",
				UserID:    "testUserID",
				SessionID: "testSessionID",
				State: map[string]any{
					"k": 5,
				},
			},
		},
		{
			name:  "generated session id",
			setup: emptyService,
			req: &session.CreateRequest{
				AppName: "testApp",
				UserID:  "testUserID",
				State: map[string]any{
					"k": 5,
				},
			},
		},
		{
			name:  "when already exists, it fails", // this differs from inmemmory impl
			setup: serviceDbWithData,
			req: &session.CreateRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
				State: map[string]any{
					"k": 10,
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.setup(t)

			got, err := s.Create(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("databaseService.Create() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			if got.Session.AppName() != tt.req.AppName {
				t.Errorf("AppName got: %v, want: %v", got.Session.AppName(), tt.wantErr)
			}

			if got.Session.UserID() != tt.req.UserID {
				t.Errorf("UserID got: %v, want: %v", got.Session.UserID(), tt.wantErr)
			}

			if tt.req.SessionID != "" {
				if got.Session.ID() != tt.req.SessionID {
					t.Errorf("SessionID got: %v, want: %v", got.Session.ID(), tt.wantErr)
				}
			} else {
				if got.Session.ID() == "" {
					t.Errorf("SessionID was not generated on empty user input.")
				}
			}

			gotState := maps.Collect(got.Session.State().All())
			wantState := tt.req.State

			if diff := cmp.Diff(wantState, gotState); diff != "" {
				t.Errorf("Create State mismatch: (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_databaseService_Delete(t *testing.T) {
	tests := []struct {
		name    string
		req     *session.DeleteRequest
		setup   func(t *testing.T) *databaseService
		wantErr bool
	}{
		{
			name:  "delete ok",
			setup: serviceDbWithData,
			req: &session.DeleteRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
			},
		},
		{
			name:  "no error when not found",
			setup: serviceDbWithData,
			req: &session.DeleteRequest{
				AppName:   "appTest",
				UserID:    "user1",
				SessionID: "session1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.setup(t)
			if err := s.Delete(t.Context(), tt.req); (err != nil) != tt.wantErr {
				t.Errorf("databaseService.Delete() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_databaseService_Get(t *testing.T) {
	// This setup function is required for a test case.
	// It creates the specific scenario from 'test_get_session_respects_user_id'.
	setupGetRespectsUserID := func(t *testing.T) *databaseService {
		t.Helper()
		s := serviceDbWithData(t) // Starts with the standard data

		// u1 creates s1 and adds an event.
		// 'serviceDbWithData' already created
		// (app1, user1, session1)
		// (app1, user2, session1)
		// We just need to add an event to it.
		session1, err := s.Get(t.Context(), &session.GetRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "session1",
		})
		if err != nil {
			t.Fatalf("setupGetRespectsUserID failed to get session1: %v", err)
		}

		// Update 'updatedAt' to pass stale validation on append
		session1.Session.(*localSession).updatedAt = time.Now()

		err = s.AppendEvent(t.Context(), session1.Session.(*localSession), &session.Event{
			ID:     "event_for_user1",
			Author: "user",
			LLMResponse: model.LLMResponse{
				Partial: false,
			},
		})
		if err != nil {
			t.Fatalf("setupGetRespectsUserID failed to append event: %v", err)
		}
		return s
	}

	setupGetWithConfig := func(t *testing.T) *databaseService {
		t.Helper()
		s := emptyService(t)
		ctx := t.Context()
		numTestEvents := 5
		created, err := s.Create(ctx, &session.CreateRequest{
			AppName:   "my_app",
			UserID:    "user",
			SessionID: "s1",
		})
		if err != nil {
			t.Fatalf("setupGetWithConfig failed to create session: %v", err)
		}

		for i := 1; i <= numTestEvents; i++ {
			created.Session.(*localSession).updatedAt = time.Now()
			event := &session.Event{
				ID:          strconv.Itoa(i),
				Author:      "user",
				Timestamp:   time.Time{}.Add(time.Duration(i)),
				LLMResponse: model.LLMResponse{},
			}
			if err := s.AppendEvent(ctx, created.Session.(*localSession), event); err != nil {
				t.Fatalf("setupGetWithConfig failed to append event %d: %v", i, err)
			}
		}
		return s
	}

	tests := []struct {
		name         string
		req          *session.GetRequest
		setup        func(t *testing.T) *databaseService
		wantResponse *session.GetResponse
		wantEvents   []*session.Event
		wantErr      bool
	}{
		{
			name:  "ok",
			setup: serviceDbWithData,
			req: &session.GetRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
			},
			wantResponse: &session.GetResponse{
				Session: &localSession{
					appName:   "app1",
					userID:    "user1",
					sessionID: "session1",
					state: map[string]any{
						"k1": "v1",
					},
					events: []*session.Event{},
				},
			},
		},
		{
			name:  "error when not found",
			setup: serviceDbWithData,
			req: &session.GetRequest{
				AppName:   "testApp",
				UserID:    "user1",
				SessionID: "session1",
			},
			wantErr: true,
		},
		{
			name:  "get session respects user id",
			setup: setupGetRespectsUserID,
			req: &session.GetRequest{
				AppName:   "app1",
				UserID:    "user2",
				SessionID: "session1",
			},
			wantResponse: &session.GetResponse{
				Session: &localSession{
					appName:   "app1",
					userID:    "user2",
					sessionID: "session1",
					// This is user2's session, which should have its own state
					state: map[string]any{
						"k1": "v2",
					},
					// Critically, it should NOT have the event from user1's session
					events: []*session.Event{},
				},
			},
			wantErr: false,
		},
		{
			name:  "with config_no config returns all events",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: "my_app", UserID: "user", SessionID: "s1",
			},
			wantEvents: []*session.Event{
				{ID: "1", Author: "user", Timestamp: time.Time{}.Add(1), LLMResponse: model.LLMResponse{}},
				{ID: "2", Author: "user", Timestamp: time.Time{}.Add(2), LLMResponse: model.LLMResponse{}},
				{ID: "3", Author: "user", Timestamp: time.Time{}.Add(3), LLMResponse: model.LLMResponse{}},
				{ID: "4", Author: "user", Timestamp: time.Time{}.Add(4), LLMResponse: model.LLMResponse{}},
				{ID: "5", Author: "user", Timestamp: time.Time{}.Add(5), LLMResponse: model.LLMResponse{}},
			},
		},
		{
			name:  "with config_num recent events",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: "my_app", UserID: "user", SessionID: "s1",
				NumRecentEvents: 3,
			},
			wantEvents: []*session.Event{
				{ID: "3", Author: "user", Timestamp: time.Time{}.Add(3), LLMResponse: model.LLMResponse{}},
				{ID: "4", Author: "user", Timestamp: time.Time{}.Add(4), LLMResponse: model.LLMResponse{}},
				{ID: "5", Author: "user", Timestamp: time.Time{}.Add(5), LLMResponse: model.LLMResponse{}},
			},
		},
		{
			name:  "with config_after timestamp",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: "my_app", UserID: "user", SessionID: "s1",
				After: time.Time{}.Add(4),
			},
			wantEvents: []*session.Event{
				{ID: "4", Author: "user", Timestamp: time.Time{}.Add(4), LLMResponse: model.LLMResponse{}},
				{ID: "5", Author: "user", Timestamp: time.Time{}.Add(5), LLMResponse: model.LLMResponse{}},
			},
		},
		{
			name:  "with config_combined filters",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: "my_app", UserID: "user", SessionID: "s1",
				NumRecentEvents: 3,
				After:           time.Time{}.Add(4),
			},
			wantEvents: []*session.Event{
				{ID: "4", Author: "user", Timestamp: time.Time{}.Add(4), LLMResponse: model.LLMResponse{}},
				{ID: "5", Author: "user", Timestamp: time.Time{}.Add(5), LLMResponse: model.LLMResponse{}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.setup(t)

			got, err := s.Get(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("databaseService.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			if tt.wantResponse != nil {
				if diff := cmp.Diff(tt.wantResponse, got,
					cmp.AllowUnexported(localSession{}),
					cmpopts.IgnoreFields(localSession{}, "mu", "updatedAt")); diff != "" {
					t.Errorf("Get session mismatch: (-want +got):\n%s", diff)
				}
			}

			if tt.wantEvents != nil {
				opts := []cmp.Option{
					cmpopts.SortSlices(func(a, b *session.Event) bool { return a.Timestamp.Before(b.Timestamp) }),
				}
				if diff := cmp.Diff(events(tt.wantEvents), got.Session.Events(), opts...); diff != "" {
					t.Errorf("Get session events mismatch: (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func Test_databaseService_List(t *testing.T) {
	tests := []struct {
		name         string
		req          *session.ListRequest
		setup        func(t *testing.T) *databaseService
		wantResponse *session.ListResponse
		wantErr      bool
	}{
		{
			name:  "list for user1",
			setup: serviceDbWithData,
			req: &session.ListRequest{
				AppName: "app1",
				UserID:  "user1",
			},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{
						appName:   "app1",
						userID:    "user1",
						sessionID: "session1",
						state: map[string]any{
							"k1": "v1",
						},
					},
					&localSession{
						appName:   "app1",
						userID:    "user1",
						sessionID: "session2",
						state: map[string]any{
							"k1": "v2",
						},
					},
				},
			},
		},
		{
			name:  "empty list for non-existent user",
			setup: serviceDbWithData,
			req: &session.ListRequest{
				AppName: "app1",
				UserID:  "custom_user",
			},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{},
			},
		},
		{
			name:  "list for user2",
			setup: serviceDbWithData,
			req: &session.ListRequest{
				AppName: "app1",
				UserID:  "user2",
			},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{
						appName:   "app1",
						userID:    "user2",
						sessionID: "session1",
						state: map[string]any{
							"k1": "v2",
						},
					},
				},
			},
		},
		{
			name:  "list all users for app",
			setup: serviceDbWithData,
			req:   &session.ListRequest{AppName: "app1", UserID: ""},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{appName: "app1", userID: "user1", sessionID: "session1", state: map[string]any{"k1": "v1"}},
					&localSession{appName: "app1", userID: "user1", sessionID: "session2", state: map[string]any{"k1": "v2"}},
					&localSession{appName: "app1", userID: "user2", sessionID: "session1", state: map[string]any{"k1": "v2"}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.setup(t)
			got, err := s.List(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("databaseService.List() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				// Sort slices for stable comparison
				opts := []cmp.Option{
					cmp.AllowUnexported(localSession{}),
					cmpopts.IgnoreFields(localSession{}, "mu", "updatedAt"),
					cmpopts.SortSlices(func(a, b session.Session) bool {
						return a.ID() < b.ID()
					}),
				}
				if diff := cmp.Diff(tt.wantResponse, got, opts...); diff != "" {
					t.Errorf("databaseService.List() = %v (-want +got):\n%s", got, diff)
				}
			}
		})
	}
}

func Test_databaseService_AppendEvent(t *testing.T) {
	tests := []struct {
		name              string
		setup             func(t *testing.T) *databaseService
		session           *localSession
		event             *session.Event
		wantStoredSession *localSession // State of the session after Get
		wantEventCount    int           // Expected event count in storage
		wantErr           bool
	}{
		{
			name:  "append event to the session and overwrite in storage",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID: "new_event1",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
			},
			wantStoredSession: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID: "new_event1",
						LLMResponse: model.LLMResponse{
							Partial: false,
						},
					},
				},
				state: map[string]any{
					"k1": "v1",
				},
			},
			wantEventCount: 1,
		},
		{
			name:  "append event to the session with events and overwrite in storage",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app2",
				userID:    "user2",
				sessionID: "session2",
			},
			event: &session.Event{
				ID: "new_event1",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
			},
			wantStoredSession: &localSession{
				appName:   "app2",
				userID:    "user2",
				sessionID: "session2",
				events: []*session.Event{
					{
						ID: "existing_event1",
						LLMResponse: model.LLMResponse{
							Partial: false,
						},
					},
					{
						ID: "new_event1",
						LLMResponse: model.LLMResponse{
							Partial: false,
						},
					},
				},
				state: map[string]any{
					"k2": "v2",
				},
			},
			wantEventCount: 2,
		},
		{
			name:  "append event when session not found should fail",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "custom_session",
			},
			event: &session.Event{
				ID: "new_event2",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
			},
			wantErr: true,
		},
		{
			name:  "append event with bytes content",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:     "event_with_bytes",
				Author: "user",
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromBytes([]byte("test_image_data"), "image/png", "user"),
					GroundingMetadata: &genai.GroundingMetadata{
						SearchEntryPoint: &genai.SearchEntryPoint{
							SDKBlob: []byte("test_sdk_blob"),
						},
					},
				},
			},
			wantStoredSession: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID:     "event_with_bytes",
						Author: "user",
						LLMResponse: model.LLMResponse{
							Content: genai.NewContentFromBytes([]byte("test_image_data"), "image/png", "user"),
							GroundingMetadata: &genai.GroundingMetadata{
								SearchEntryPoint: &genai.SearchEntryPoint{
									SDKBlob: []byte("test_sdk_blob"),
								},
							},
						},
					},
				},
				state: map[string]any{
					"k1": "v1",
				},
			},
			wantEventCount: 1,
		},
		{
			name:  "append event with all fields",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:                 "event_complete",
				Author:             "user",
				LongRunningToolIDs: []string{"tool123"},
				Actions:            session.EventActions{StateDelta: map[string]any{"k2": "v2"}},
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
			},
			wantStoredSession: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID:                 "event_complete",
						Author:             "user",
						LongRunningToolIDs: []string{"tool123"},
						Actions:            session.EventActions{StateDelta: map[string]any{"k2": "v2"}},
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
					},
				},
				state: map[string]any{
					"k1": "v1",
					"k2": "v2",
				},
			},
			wantEventCount: 1,
		},
		{
			name:  "partial events are not persisted",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:     "partial_event",
				Author: "user",
				LLMResponse: model.LLMResponse{
					Partial: true, // This is the key field
				},
			},
			wantStoredSession: &localSession{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
				events:    []*session.Event{}, // No event should be stored
				state: map[string]any{
					"k1": "v1",
				},
			},
			wantEventCount: 0, // Expect 0 events
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()

			s := tt.setup(t)

			tt.session.updatedAt = time.Now() // set updatedAt value to pass stale validation
			err := s.AppendEvent(ctx, tt.session, tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("databaseService.AppendEvent() error = %v, wantErr %v", err, tt.wantErr)
			}

			if err != nil {
				return
			}

			resp, err := s.Get(ctx, &session.GetRequest{
				AppName:   tt.session.AppName(),
				UserID:    tt.session.UserID(),
				SessionID: tt.session.ID(),
			})
			if err != nil {
				t.Fatalf("databaseService.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check event count first
			if resp.Session.Events().Len() != tt.wantEventCount {
				t.Errorf("AppendEvent returned %d events, want %d", resp.Session.Events().Len(), tt.wantEventCount)
			}

			// Define comparison options
			opts := []cmp.Option{
				cmp.AllowUnexported(localSession{}),
				cmpopts.IgnoreFields(localSession{}, "mu", "updatedAt"),
				cmpopts.IgnoreFields(session.Event{}, "Timestamp"),
				// Add sorters if event order is not guaranteed
				cmpopts.SortSlices(func(a, b *session.Event) bool {
					return a.ID < b.ID
				}),
			}

			if diff := cmp.Diff(tt.wantStoredSession, resp.Session, opts...); diff != "" {
				t.Errorf("AppendEvent session mismatch: (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_databaseService_StateManagement(t *testing.T) {
	ctx := t.Context()
	appName := "my_app"

	t.Run("app_state_is_shared", func(t *testing.T) {
		s := emptyService(t)
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1", State: map[string]any{"app:k1": "v1"}})
		s1.Session.(*localSession).updatedAt = time.Now()
		err := s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:          "event1",
			Actions:     session.EventActions{StateDelta: map[string]any{"app:k2": "v2"}},
			LLMResponse: model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s2, err := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2", SessionID: "s2"})
		if err != nil {
			t.Fatalf("Failed to create session for user 2: %v", err)
		}

		wantState := map[string]any{"app:k1": "v1", "app:k2": "v2"}
		gotState := maps.Collect(s2.Session.State().All())
		if diff := cmp.Diff(wantState, gotState); diff != "" {
			t.Errorf("User 2 state mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("user_state_is_user_specific", func(t *testing.T) {
		s := emptyService(t)
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1", State: map[string]any{"user:k1": "v1"}})
		s1.Session.(*localSession).updatedAt = time.Now()
		err := s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:          "event1",
			Actions:     session.EventActions{StateDelta: map[string]any{"user:k2": "v2"}},
			LLMResponse: model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1b"})
		wantStateU1 := map[string]any{"user:k1": "v1", "user:k2": "v2"}
		gotStateU1 := maps.Collect(s1b.Session.State().All())
		if diff := cmp.Diff(wantStateU1, gotStateU1); diff != "" {
			t.Errorf("User 1 second session state mismatch (-want +got):\n%s", diff)
		}

		s2, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2", SessionID: "s2"})
		gotStateU2 := maps.Collect(s2.Session.State().All())
		if len(gotStateU2) != 0 {
			t.Errorf("User 2 should have empty state, but got: %v", gotStateU2)
		}
	})

	t.Run("session_state_is_not_shared", func(t *testing.T) {
		s := emptyService(t)
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1", State: map[string]any{"sk1": "v1"}})
		s1.Session.(*localSession).updatedAt = time.Now()
		err := s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:          "event1",
			Actions:     session.EventActions{StateDelta: map[string]any{"sk2": "v2"}},
			LLMResponse: model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1_got, _ := s.Get(ctx, &session.GetRequest{AppName: appName, UserID: "u1", SessionID: "s1"})
		wantState := map[string]any{"sk1": "v1", "sk2": "v2"}
		gotState := maps.Collect(s1_got.Session.State().All())
		if diff := cmp.Diff(wantState, gotState); diff != "" {
			t.Errorf("Refetched s1 state mismatch (-want +got):\n%s", diff)
		}

		s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1b"})
		gotStateS1b := maps.Collect(s1b.Session.State().All())
		if len(gotStateS1b) != 0 {
			t.Errorf("Session s1b should have empty state, but got: %v", gotStateS1b)
		}
	})

	t.Run("temp_state_is_not_persisted", func(t *testing.T) {
		s := emptyService(t)
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", SessionID: "s1"})
		s1.Session.(*localSession).updatedAt = time.Now()
		event := &session.Event{
			ID:          "event1",
			Actions:     session.EventActions{StateDelta: map[string]any{"temp:k1": "v1", "sk": "v2"}},
			LLMResponse: model.LLMResponse{},
		}
		err := s.AppendEvent(ctx, s1.Session.(*localSession), event)
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1_got, _ := s.Get(ctx, &session.GetRequest{AppName: appName, UserID: "u1", SessionID: "s1"})
		wantState := map[string]any{"sk": "v2"}
		gotState := maps.Collect(s1_got.Session.State().All())
		if diff := cmp.Diff(wantState, gotState); diff != "" {
			t.Errorf("Persisted state mismatch (-want +got):\n%s", diff)
		}

		storedEvents := s1_got.Session.Events()
		if storedEvents.Len() != 1 {
			t.Fatalf("Expected 1 stored event, got %d", storedEvents.Len())
		}
		storedDelta := storedEvents.At(0).Actions.StateDelta
		if _, exists := storedDelta["temp:k1"]; exists {
			t.Errorf("temp:k1 key was found in the stored event's state delta")
		}
		if storedDelta["sk"] != "v2" {
			t.Errorf("Expected 'sk' key in stored event, but was missing or wrong value")
		}
	})
}

func serviceDbWithData(t *testing.T) *databaseService {
	t.Helper()

	service := emptyService(t)

	for _, storedSession := range []*localSession{
		{
			appName:   "app1",
			userID:    "user1",
			sessionID: "session1",
			state: map[string]any{
				"k1": "v1",
			},
		},
		{
			appName:   "app1",
			userID:    "user2",
			sessionID: "session1",
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			appName:   "app1",
			userID:    "user1",
			sessionID: "session2",
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			appName:   "app2",
			userID:    "user2",
			sessionID: "session2",
			state: map[string]any{
				"k2": "v2",
			},
			events: []*session.Event{
				{
					ID: "existing_event1",
					LLMResponse: model.LLMResponse{
						Partial: false,
					},
				},
			},
		},
	} {
		// TODO: Consider changing to SQL insert
		resp, err := service.Create(t.Context(), &session.CreateRequest{
			AppName:   storedSession.appName,
			UserID:    storedSession.userID,
			SessionID: storedSession.sessionID,
			State:     storedSession.state,
		})
		if err != nil {
			t.Fatalf("Failed to create sample sessions on db initialization: %v", err)
		}

		for _, ev := range storedSession.events {
			err = service.AppendEvent(t.Context(), resp.Session, ev)
			if err != nil {
				t.Fatalf("Failed to append event to session on db initialization: %v", err)
			}
		}
	}

	return service
}

func emptyService(t *testing.T) *databaseService {
	t.Helper()
	gormConfig := &gorm.Config{
		PrepareStmt: true,
	}

	service, err := NewSessionService(sqlite.Open("file::memory:?cache=shared"), gormConfig)
	if err != nil {
		t.Fatalf("Failed to create session service: %v", err)
	}
	dbservice, ok := service.(*databaseService)
	if !ok {
		t.Fatalf("invalid session service type")
	}

	err = AutoMigrate(service)
	if err != nil {
		t.Fatalf("Failed to AutoMigrate db: %v", err)
	}

	t.Cleanup(func() {
		t.Log("CLEANUP: Deleting all rows from tables...")

		// Define models in Child-to-Parent order
		modelsToDelete := []any{
			&storageEvent{}, // Child-most
			&storageSession{},
			&storageUserState{},
			&storageAppState{}, // Parent-most
		}

		for _, model := range modelsToDelete {
			// GORM statement parser to get table names
			stmt := &gorm.Statement{DB: dbservice.db}
			// Parse the model to get its table name
			if err := stmt.Parse(model); err != nil {
				t.Errorf("Failed to parse model schema for cleanup: %v", err)
				continue
			}
			tableName := stmt.Table

			// Exec with "WHERE true" instead of gorm.Delete()
			// satisfies Spanner's requirement for a WHERE clause.
			if err := dbservice.db.Exec(`DELETE FROM ` + tableName + ` WHERE true`).Error; err != nil {
				t.Errorf("Failed to delete from table %s: %v", tableName, err)
			}
		}
		sqlDB, err := dbservice.db.DB()
		if err != nil {
			t.Errorf("Failed to get underlying *sql.DB: %v", err)
			return
		}
		if err := sqlDB.Close(); err != nil {
			t.Errorf("Failed to close database connection: %v", err)
		}
	})

	if err != nil {
		t.Fatalf("Failed to open GORM connection: %v", err)
	}
	return dbservice
}
