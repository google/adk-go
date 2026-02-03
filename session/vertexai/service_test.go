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

package vertexai

import (
	"context"
	"maps"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/rpcreplay"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

const (
	ProjectID = "adk-go-test"
	Location  = "us-central1"
	EngineId  = "5576569044451983360"
	EngineId2 = "8602987994044956672"
	UserID    = "test-user"
)

func Test_vertexaiService_Create(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(t *testing.T, name string) (session.Service, map[string]string)
		req        *session.CreateRequest
		want       session.Session
		wantErr    bool
		errMessage string
	}{
		{
			name:  "full key",
			setup: emptyService,
			req: &session.CreateRequest{
				AppName:   EngineId,
				UserID:    "testUserID",
				SessionID: "testSessionID",
				State: map[string]any{
					"k": 5,
				},
			},
			wantErr:    true,
			errMessage: "user-provided Session id is not supported for VertexAISessionService: \"testSessionID\"",
		},
		{
			name:  "generated session id",
			setup: emptyService,
			req: &session.CreateRequest{
				AppName: EngineId,
				UserID:  "testUserID",
				State: map[string]any{
					// TODO had to parse to float64, sending int was modified by vertex or by vertex client, int should work
					"k": float64(5),
				},
			},
		},
		{
			name:  "when already exists, it fails",
			setup: serviceDbWithData,
			req: &session.CreateRequest{
				AppName:   EngineId,
				UserID:    "user1",
				SessionID: "session1",
				State: map[string]any{
					"k": 10,
				},
			},
			wantErr:    true,
			errMessage: "user-provided Session id is not supported for VertexAISessionService: \"session1\"",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := tt.setup(t, tt.name)

			got, err := s.Create(t.Context(), tt.req)
			if err != nil {
				if tt.wantErr && err.Error() == tt.errMessage {
					return
				}
				t.Fatalf("vertexAiService.Create() error = %v, wantErr %v", err, tt.errMessage)
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

func Test_vertexaiService_Get(t *testing.T) {
	// This setup function is required for a test case.
	// It creates the specific scenario from 'test_get_session_respects_user_id'.
	setupGetRespectsUserID := func(t *testing.T, name string) (session.Service, map[string]string) {
		t.Helper()
		s, l := serviceDbWithData(t, name) // Starts with the standard data

		// u1 creates s1 and adds an event.
		// 'serviceDbWithData' already created
		// (app1, user1, session1)
		// (app1, user2, session1)
		// We just need to add an event to it.
		session1, err := s.Get(t.Context(), &session.GetRequest{
			AppName:   EngineId,
			UserID:    "user1",
			SessionID: l[EngineId+"user1session1"],
		})
		if err != nil {
			t.Fatalf("setupGetRespectsUserID failed to get session1: %v", err)
		}

		// Update 'updatedAt' to pass stale validation on append
		session1.Session.(*localSession).updatedAt = time.Now()

		err = s.AppendEvent(t.Context(), session1.Session.(*localSession), &session.Event{
			ID:           "event_for_user1",
			InvocationID: "test",
			Author:       "user",
			LLMResponse: model.LLMResponse{
				Partial: false,
			},
		})
		if err != nil {
			t.Fatalf("setupGetRespectsUserID failed to append event: %v", err)
		}
		return s, l
	}

	setupGetWithConfig := func(t *testing.T, name string) (session.Service, map[string]string) {
		t.Helper()
		s, l := emptyService(t, name)
		ctx := t.Context()
		numTestEvents := 5
		created, err := s.Create(ctx, &session.CreateRequest{
			AppName: EngineId2,
			UserID:  "user",
		})
		if err != nil {
			t.Fatalf("setupGetWithConfig failed to create session: %v", err)
		}

		l[created.Session.AppName()+created.Session.UserID()+"s1"] = created.Session.ID()

		for i := 1; i <= numTestEvents; i++ {
			created.Session.(*localSession).updatedAt = time.Now()
			event := &session.Event{
				ID:           strconv.Itoa(i),
				InvocationID: "test",
				Author:       "user",
				Timestamp:    time.Time{}.Add(time.Duration(i) * time.Second),
				LLMResponse:  model.LLMResponse{},
			}
			if err := s.AppendEvent(ctx, created.Session.(*localSession), event); err != nil {
				t.Fatalf("setupGetWithConfig failed to append event %d: %v", i, err)
			}
		}
		return s, l
	}

	tests := []struct {
		name         string
		req          *session.GetRequest
		setup        func(t *testing.T, name string) (session.Service, map[string]string)
		wantResponse *session.GetResponse
		wantEvents   []*session.Event
		wantErr      bool
	}{
		{
			name:  "ok",
			setup: serviceDbWithData,
			req: &session.GetRequest{
				AppName:   EngineId,
				UserID:    "user1",
				SessionID: "session1",
			},
			wantResponse: &session.GetResponse{
				Session: &localSession{
					appName:   EngineId,
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
				AppName:   EngineId,
				UserID:    "user1",
				SessionID: "session4",
			},
			wantErr: true,
		},
		{
			name:  "get session respects user id",
			setup: setupGetRespectsUserID,
			req: &session.GetRequest{
				AppName:   EngineId,
				UserID:    "user2",
				SessionID: "session1",
			},
			wantResponse: &session.GetResponse{
				Session: &localSession{
					appName:   EngineId,
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
				AppName: EngineId2, UserID: "user", SessionID: "s1",
			},
			wantEvents: []*session.Event{
				{
					ID: "1", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(1 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "2", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(2 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "3", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(3 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "4", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(4 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "5", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(5 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
			},
		},
		{
			name:  "with config_num recent events",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: EngineId2, UserID: "user", SessionID: "s1",
				NumRecentEvents: 3,
			},
			wantEvents: []*session.Event{
				{
					ID: "3", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(3 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "4", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(4 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "5", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(5 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
			},
		},
		{
			name:  "with config_after timestamp",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: EngineId2, UserID: "user", SessionID: "s1",
				After: time.Time{}.Add(4 * time.Second),
			},
			wantErr: false,
			wantEvents: []*session.Event{
				{
					ID: "4", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(4 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "5", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(5 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
			},
		},
		{
			name:  "with config_combined filters",
			setup: setupGetWithConfig,
			req: &session.GetRequest{
				AppName: EngineId2, UserID: "user", SessionID: "s1",
				NumRecentEvents: 3,
				After:           time.Time{}.Add(4 * time.Second),
			},
			wantErr: false,
			wantEvents: []*session.Event{
				{
					ID: "4", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(4 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
				{
					ID: "5", Author: "user", InvocationID: "test", Timestamp: time.Time{}.Add(5 * time.Second),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{},
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, l := tt.setup(t, tt.name)
			tt.req.SessionID = l[tt.req.AppName+tt.req.UserID+tt.req.SessionID]
			got, err := s.Get(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("vertexAiService.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			if tt.wantResponse != nil {
				if diff := cmp.Diff(tt.wantResponse, got,
					cmp.AllowUnexported(localSession{}),
					cmpopts.IgnoreFields(localSession{}, "mu", "updatedAt", "sessionID")); diff != "" {
					t.Errorf("Get session mismatch: (-want +got):\n%s", diff)
				}
			}

			if tt.wantEvents != nil {
				opts := []cmp.Option{
					cmpopts.SortSlices(func(a, b *session.Event) bool { return a.Timestamp.Before(b.Timestamp) }),
					cmpopts.IgnoreFields(session.Event{}, "ID"),
				}
				if diff := cmp.Diff(events(tt.wantEvents), got.Session.Events(), opts...); diff != "" {
					t.Errorf("Get session events mismatch: (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func Test_vertexaiService_List(t *testing.T) {
	tests := []struct {
		name         string
		req          *session.ListRequest
		setup        func(t *testing.T, name string) (session.Service, map[string]string)
		wantResponse *session.ListResponse
		wantErr      bool
	}{
		{
			name:  "list for user1",
			setup: serviceDbWithData,
			req: &session.ListRequest{
				AppName: EngineId,
				UserID:  "user1",
			},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{
						appName:   EngineId,
						userID:    "user1",
						sessionID: "session1",
						state: map[string]any{
							"k1": "v1",
						},
					},
					&localSession{
						appName:   EngineId,
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
				AppName: EngineId,
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
				AppName: EngineId,
				UserID:  "user2",
			},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{
						appName:   EngineId,
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
			req:   &session.ListRequest{AppName: EngineId, UserID: ""},
			wantResponse: &session.ListResponse{
				Sessions: []session.Session{
					&localSession{appName: EngineId, userID: "user1", sessionID: "session1", state: map[string]any{"k1": "v1"}},
					&localSession{appName: EngineId, userID: "user1", sessionID: "session2", state: map[string]any{"k1": "v2"}},
					&localSession{appName: EngineId, userID: "user2", sessionID: "session1", state: map[string]any{"k1": "v2"}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, l := tt.setup(t, tt.name)
			got, err := s.List(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.List() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			for _, s1 := range tt.wantResponse.Sessions {
				ls := s1.(*localSession)
				ls.sessionID = l[ls.appName+ls.userID+ls.sessionID]
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
					t.Errorf("vertexAiService.List() = %v (-want +got):\n%s", got, diff)
				}
			}
		})
	}
}

func Test_vertexaiService_AppendEvent(t *testing.T) {
	tests := []struct {
		name              string
		setup             func(t *testing.T, name string) (session.Service, map[string]string)
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:           "new_event1",
				Author:       "test",
				InvocationID: "test",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
			},
			wantStoredSession: &localSession{
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID:           "new_event1",
						Author:       "test",
						InvocationID: "test",
						LLMResponse: model.LLMResponse{
							Content: &genai.Content{},
							Partial: false,
						},
						Actions: session.EventActions{
							StateDelta: map[string]any{},
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
			name:    "missing session id",
			setup:   emptyService,
			session: &localSession{appName: EngineId, userID: UserID},
			event:   &session.Event{},
			wantErr: true,
		},
		{
			name:  "nil event",
			setup: emptyService,
			session: &localSession{
				appName:   EngineId2,
				userID:    "user2",
				sessionID: "session2",
			},
			event:   nil,
			wantErr: true,
		},
		{
			name:  "missing author",
			setup: emptyService,
			session: &localSession{
				appName:   EngineId2,
				userID:    "user2",
				sessionID: "session2",
			},
			event: &session.Event{
				Timestamp:    time.Now(),
				InvocationID: uuid.NewString(),
			},
			wantErr: true,
		},
		{
			name:  "missing invocation id",
			setup: emptyService,
			session: &localSession{
				appName:   EngineId2,
				userID:    "user2",
				sessionID: "session2",
			},
			event: &session.Event{
				Timestamp: time.Now(),
				Author:    UserID,
			},
			wantErr: true,
		},
		{
			name:  "append event to the session with events and overwrite in storage",
			setup: serviceDbWithData,
			session: &localSession{
				appName:   EngineId2,
				userID:    "user2",
				sessionID: "session2",
			},
			event: &session.Event{
				ID:           "new_event1",
				Author:       "test",
				InvocationID: "test",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
			},
			wantStoredSession: &localSession{
				appName:   EngineId2,
				userID:    "user2",
				sessionID: "session2",
				events: []*session.Event{
					{
						ID:           "existing_event1",
						Author:       "test",
						InvocationID: "test",
						LLMResponse: model.LLMResponse{
							Content: &genai.Content{},
							Partial: false,
						},
						Actions: session.EventActions{
							StateDelta: map[string]any{},
						},
					},
					{
						ID:           "new_event1",
						Author:       "test",
						InvocationID: "test",
						LLMResponse: model.LLMResponse{
							Content: &genai.Content{},
							Partial: false,
						},
						Actions: session.EventActions{
							StateDelta: map[string]any{},
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "custom_session",
			},
			event: &session.Event{
				ID:           "new_event2",
				Author:       "test",
				InvocationID: "test",
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:           "event_with_bytes",
				Author:       "user",
				InvocationID: "test",
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID:           "event_with_bytes",
						Author:       "user",
						InvocationID: "test",
						LLMResponse: model.LLMResponse{
							Content: genai.NewContentFromBytes([]byte("test_image_data"), "image/png", "user"),
							GroundingMetadata: &genai.GroundingMetadata{
								SearchEntryPoint: &genai.SearchEntryPoint{
									SDKBlob: []byte("test_sdk_blob"),
								},
							},
						},
						Actions: session.EventActions{
							StateDelta: map[string]any{},
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:                 "event_complete",
				Author:             "user",
				InvocationID:       "test",
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
				events: []*session.Event{
					{
						ID:                 "event_complete",
						Author:             "user",
						InvocationID:       "test",
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
				appName:   EngineId,
				userID:    "user1",
				sessionID: "session1",
			},
			event: &session.Event{
				ID:           "partial_event",
				Author:       "user",
				InvocationID: "test",
				LLMResponse: model.LLMResponse{
					Partial: true, // This is the key field
				},
			},
			wantStoredSession: &localSession{
				appName:   EngineId,
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

			s, l := tt.setup(t, tt.name)

			tt.session.sessionID = l[tt.session.appName+tt.session.userID+tt.session.sessionID]
			if tt.wantStoredSession != nil {
				tt.wantStoredSession.sessionID = tt.session.sessionID
			}
			tt.session.updatedAt = time.Now() // set updatedAt value to pass stale validation
			err := s.AppendEvent(ctx, tt.session, tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.AppendEvent() error = %v, wantErr %v", err, tt.wantErr)
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
				t.Fatalf("vertexAiService.Get() error = %v, wantErr %v", err, tt.wantErr)
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
				cmpopts.IgnoreFields(session.Event{}, "Timestamp", "ID"),
				cmpopts.IgnoreFields(model.LLMResponse{}, "CitationMetadata", "UsageMetadata"),
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

func Test_vertexaiService_StateManagement(t *testing.T) {
	ctx := t.Context()
	appName := EngineId

	t.Run("app_state_is_shared", func(t *testing.T) {
		s, _ := emptyService(t, "app_state_is_shared")
		s1, err := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"app:k1": "v1"}})
		if err != nil {
			t.Fatalf("Failed to create session for user 1: %v", err)
		}
		s1.Session.(*localSession).updatedAt = time.Now()
		err = s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:           "event1",
			Author:       "test",
			InvocationID: "test",
			Actions:      session.EventActions{StateDelta: map[string]any{"app:k2": "v2"}},
			LLMResponse:  model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s2, err := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2"})
		if err != nil {
			t.Fatalf("Failed to create session for user 2: %v", err)
		}

		wantState := map[string]any{"app:k1": "v1", "app:k2": "v2"}
		gotState := maps.Collect(s2.Session.State().All())
		if diff := cmp.Diff(wantState, gotState); diff != "" {
			t.Errorf("User 2 state mismatch (-want +got):\n%s", diff)
		}

		t.Cleanup(func() {
			err := s.AppendEvent(ctx, s2.Session, &session.Event{
				ID:           "clean_up_event",
				Author:       "test",
				InvocationID: "test",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
				Actions: session.EventActions{
					StateDelta: map[string]any{
						"app:k1": nil,
						"app:k2": nil,
					},
				},
			})
			if err != nil {
				t.Fatalf("Failed to appendEvent on cleanup: %v", err)
			}
		})
	})

	t.Run("user_state_is_user_specific", func(t *testing.T) {
		s, _ := emptyService(t, "user_state_is_user_specific")
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"user:k1": "v1"}})
		s1.Session.(*localSession).updatedAt = time.Now()
		err := s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:           "event1",
			Author:       "test",
			InvocationID: "test",
			Actions:      session.EventActions{StateDelta: map[string]any{"user:k2": "v2"}},
			LLMResponse:  model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
		wantStateU1 := map[string]any{"user:k1": "v1", "user:k2": "v2"}
		gotStateU1 := maps.Collect(s1b.Session.State().All())
		if diff := cmp.Diff(wantStateU1, gotStateU1); diff != "" {
			t.Errorf("User 1 second session state mismatch (-want +got):\n%s", diff)
		}

		s2, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u2"})
		gotStateU2 := maps.Collect(s2.Session.State().All())
		if len(gotStateU2) != 0 {
			t.Errorf("User 2 should have empty state, but got: %v", gotStateU2)
		}

		t.Cleanup(func() {
			err := s.AppendEvent(ctx, s1b.Session, &session.Event{
				ID:           "clean_up_event",
				Author:       "test",
				InvocationID: "test",
				LLMResponse: model.LLMResponse{
					Partial: false,
				},
				Actions: session.EventActions{
					StateDelta: map[string]any{
						"user:k1": nil,
						"user:k2": nil,
					},
				},
			})
			if err != nil {
				t.Fatalf("Failed to appendEvent on cleanup: %v", err)
			}
		})
	})

	t.Run("session_state_is_not_shared", func(t *testing.T) {
		s, _ := emptyService(t, "session_state_is_not_shared")
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1", State: map[string]any{"sk1": "v1"}})
		s1.Session.(*localSession).updatedAt = time.Now()
		err := s.AppendEvent(ctx, s1.Session.(*localSession), &session.Event{
			ID:           "event1",
			Author:       "test",
			InvocationID: "test",
			Actions:      session.EventActions{StateDelta: map[string]any{"sk2": "v2"}},
			LLMResponse:  model.LLMResponse{},
		})
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1_got, _ := s.Get(ctx, &session.GetRequest{AppName: appName, UserID: "u1", SessionID: s1.Session.ID()})
		wantState := map[string]any{"sk1": "v1", "sk2": "v2"}
		gotState := maps.Collect(s1_got.Session.State().All())
		if diff := cmp.Diff(wantState, gotState); diff != "" {
			t.Errorf("Refetched s1 state mismatch (-want +got):\n%s", diff)
		}

		s1b, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
		gotStateS1b := maps.Collect(s1b.Session.State().All())
		if len(gotStateS1b) != 0 {
			t.Errorf("Session s1b should have empty state, but got: %v", gotStateS1b)
		}
	})

	t.Run("temp_state_is_not_persisted", func(t *testing.T) {
		s, _ := emptyService(t, "temp_state_is_not_persisted")
		s1, _ := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: "u1"})
		s1.Session.(*localSession).updatedAt = time.Now()
		event := &session.Event{
			ID:           "event1",
			Author:       "test",
			InvocationID: "test",
			Actions:      session.EventActions{StateDelta: map[string]any{"temp:k1": "v1", "sk": "v2"}},
			LLMResponse:  model.LLMResponse{},
		}
		err := s.AppendEvent(ctx, s1.Session.(*localSession), event)
		if err != nil {
			t.Fatalf("Failed to appendEvent: %v", err)
		}

		s1_got, _ := s.Get(ctx, &session.GetRequest{AppName: appName, UserID: s1.Session.UserID(), SessionID: s1.Session.ID()})
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
		if storedDelta["sk"] != "v2" {
			t.Errorf("Expected 'sk' key in stored event, but was missing or wrong value")
		}
	})
}

func emptyService(t *testing.T, name string) (session.Service, map[string]string) {
	t.Helper()
	replayFile := sanitizeFilename(name)
	opts, teardown, err := setupReplay(t, replayFile)
	if err != nil {
		t.Fatalf("Failed to setup replay: %v", err)
	}

	v, err := NewSessionService(t.Context(), VertexAIServiceConfig{
		Location:  Location,
		ProjectID: ProjectID,
	}, opts...)
	if err != nil {
		t.Fatalf("%s", err)
	}

	t.Cleanup(func() {
		t.Log("CLEANUP")
		deleteAll(t, v)
		defer teardown()
	})

	return v, make(map[string]string, 0)
}

func deleteAll(t *testing.T, v session.Service) {
	deleteAllFromApp(t, v, EngineId)
	deleteAllFromApp(t, v, EngineId2)
}

func deleteAllFromApp(t *testing.T, v session.Service, app string) {
	cleanupCtx := context.Background()
	sessionsResp, err := v.List(cleanupCtx, &session.ListRequest{
		AppName: app,
	})
	if err != nil {
		t.Errorf("error listing session for delete all: %s", err)
	}

	for _, s := range sessionsResp.Sessions {
		err := v.Delete(cleanupCtx, &session.DeleteRequest{
			AppName:   s.AppName(),
			UserID:    s.UserID(),
			SessionID: s.ID(),
		})
		if err != nil {
			t.Errorf("error deleting session for delete all: %s", err)
		}
	}
}

func serviceDbWithData(t *testing.T, name string) (session.Service, map[string]string) {
	t.Helper()

	service, _ := emptyService(t, name)
	ids := make(map[string]string, 4)

	for _, storedSession := range []*localSession{
		{
			appName:   EngineId,
			userID:    "user1",
			sessionID: "session1",
			state: map[string]any{
				"k1": "v1",
			},
		},
		{
			appName:   EngineId,
			userID:    "user2",
			sessionID: "session1",
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			appName:   EngineId,
			userID:    "user1",
			sessionID: "session2",
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			appName:   EngineId2,
			userID:    "user2",
			sessionID: "session2",
			state: map[string]any{
				"k2": "v2",
			},
			events: []*session.Event{
				{
					Author:       "test",
					InvocationID: "test",
					ID:           "existing_event1",
					LLMResponse: model.LLMResponse{
						Partial: false,
					},
				},
			},
		},
	} {
		resp, err := service.Create(t.Context(), &session.CreateRequest{
			AppName: storedSession.appName,
			UserID:  storedSession.userID,
			State:   storedSession.state,
		})
		if err != nil {
			t.Fatalf("Failed to create sample sessions on db initialization: %v", err)
		}

		ids[resp.Session.AppName()+resp.Session.UserID()+storedSession.sessionID] = resp.Session.ID()

		for _, ev := range storedSession.events {
			err = service.AppendEvent(t.Context(), resp.Session, ev)
			if err != nil {
				t.Fatalf("Failed to append event to session on db initialization: %v", err)
			}
		}
	}

	return service, ids
}

// setupReplay determines if we are recording real traffic or replaying from a file.
// returns: client options, a teardown function, and an error.
func setupReplay(t *testing.T, filename string) ([]option.ClientOption, func(), error) {
	filePath := filepath.Join("testdata", filename)

	var grpcOpts []grpc.DialOption
	var teardown func() error

	// 1. Determine mode (Record vs Replay)
	if os.Getenv("UPDATE_REPLAYS") == "true" {
		t.Logf("Recording payload to %s", filePath)
		_ = os.MkdirAll("testdata", 0o755)

		rec, err := rpcreplay.NewRecorder(filePath, nil)
		if err != nil {
			return nil, nil, err
		}
		grpcOpts = rec.DialOptions()
		teardown = rec.Close
	} else {
		t.Logf("Replaying from %s", filePath)
		rep, err := rpcreplay.NewReplayer(filePath)
		if err != nil {
			return nil, nil, err
		}
		grpcOpts = rep.DialOptions()
		teardown = rep.Close
	}

	// 2. CONVERSION STEP: Convert []grpc.DialOption -> []option.ClientOption
	var clientOpts []option.ClientOption
	for _, opt := range grpcOpts {
		clientOpts = append(clientOpts, option.WithGRPCDialOption(opt))
		if os.Getenv("UPDATE_REPLAYS") != "true" {
			clientOpts = append(clientOpts, option.WithoutAuthentication())
		}
	}

	// 3. Return the SAFE client options
	return clientOpts, func() {
		if err := teardown(); err != nil {
			t.Errorf("Failed to close replayer/recorder: %v", err)
		}
	}, nil
}

func sanitizeFilename(name string) string {
	// Replace spaces and special chars with underscores
	safe := strings.ReplaceAll(name, " ", "_")
	safe = strings.ReplaceAll(safe, ",", "_")
	safe = strings.ReplaceAll(safe, "/", "-")
	return safe + ".replay"
}
