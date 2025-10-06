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
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func Test_inMemoryService_Create(t *testing.T) {
	tests := []struct {
		name            string
		inMemoryService *inMemoryService
		req             *CreateRequest
		want            Session
		wantErr         bool
	}{
		{
			name:            "full key",
			inMemoryService: &inMemoryService{},
			req: &CreateRequest{
				AppName:   "testApp",
				UserID:    "testUserID",
				SessionID: "testSessionID",
				State: map[string]any{
					"k": 5,
				},
			},
		},
		{
			name:            "generated session id",
			inMemoryService: &inMemoryService{},
			req: &CreateRequest{
				AppName: "testApp",
				UserID:  "testUserID",
				State: map[string]any{
					"k": 5,
				},
			},
		},
		{
			name:            "when already exists, it's overwritten", // to be consistent with python/java
			inMemoryService: serviceWithData(t),
			req: &CreateRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
				State: map[string]any{
					"k": 10,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			s := tt.inMemoryService

			got, err := s.Create(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("inMemoryService.Create() error = %v, wantErr %v", err, tt.wantErr)
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

func Test_inMemoryService_Get(t *testing.T) {
	tests := []struct {
		name            string
		req             *GetRequest
		inMemoryService *inMemoryService
		wantResponse    *GetResponse
		wantErr         bool
	}{
		{
			name:            "ok",
			inMemoryService: serviceWithData(t),
			req: &GetRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
			},
			wantResponse: &GetResponse{
				Session: &session{
					id: id{
						appName:   "app1",
						userID:    "user1",
						sessionID: "session1",
					},
					state: map[string]any{
						"k1": "v1",
					},
				},
			},
		},
		{
			name:            "error when not found",
			inMemoryService: serviceWithData(t),
			req: &GetRequest{
				AppName:   "testApp",
				UserID:    "user1",
				SessionID: "session1",
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			s := tt.inMemoryService

			got, err := s.Get(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("inMemoryService.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			if diff := cmp.Diff(tt.wantResponse, got,
				cmp.AllowUnexported(session{}),
				cmp.AllowUnexported(id{}),
				cmpopts.IgnoreFields(session{}, "mu")); diff != "" {
				t.Errorf("Create session mismatch: (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_inMemoryService_List(t *testing.T) {
	tests := []struct {
		name            string
		req             *ListRequest
		inMemoryService *inMemoryService
		wantResponse    *ListResponse
		wantErr         bool
	}{
		{
			name:            "ok",
			inMemoryService: serviceWithData(t),
			req: &ListRequest{
				AppName: "app1",
				UserID:  "user1",
			},
			wantResponse: &ListResponse{
				Sessions: []Session{
					&session{
						id: id{
							appName:   "app1",
							userID:    "user1",
							sessionID: "session1",
						},
						state: map[string]any{
							"k1": "v1",
						},
					},
					&session{
						id: id{
							appName:   "app1",
							userID:    "user1",
							sessionID: "session2",
						},
						state: map[string]any{
							"k1": "v2",
						},
					},
				},
			},
		},
		{
			name:            "empty list for non-existent user",
			inMemoryService: serviceWithData(t),
			req: &ListRequest{
				AppName: "app1",
				UserID:  "custom_user",
			},
			wantResponse: &ListResponse{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			s := tt.inMemoryService
			got, err := s.List(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("inMemoryService.List() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				if diff := cmp.Diff(tt.wantResponse, got,
					cmp.AllowUnexported(session{}),
					cmp.AllowUnexported(id{}),
					cmpopts.IgnoreFields(session{}, "mu")); diff != "" {
					t.Errorf("inMemoryService.List() = %v (-want +got):\n%s", got, diff)
				}
			}
		})
	}
}

func Test_inMemoryService_Delete(t *testing.T) {
	tests := []struct {
		name            string
		req             *DeleteRequest
		inMemoryService *inMemoryService
		wantErr         bool
	}{
		{
			name:            "delete ok",
			inMemoryService: serviceWithData(t),
			req: &DeleteRequest{
				AppName:   "app1",
				UserID:    "user1",
				SessionID: "session1",
			},
		},
		{
			name:            "no error when not found",
			inMemoryService: serviceWithData(t),
			req: &DeleteRequest{
				AppName:   "appTest",
				UserID:    "user1",
				SessionID: "session1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			s := tt.inMemoryService
			if err := s.Delete(t.Context(), tt.req); (err != nil) != tt.wantErr {
				t.Errorf("inMemoryService.Delete() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_inMemoryService_AppendEvent(t *testing.T) {
	tests := []struct {
		name              string
		inMemoryService   *inMemoryService
		session           *session
		event             *Event
		wantStoredSession *session
		wantErr           bool
	}{
		{
			name:            "append event to the session and overwrite in storage",
			inMemoryService: serviceWithData(t),
			session: &session{
				id: id{
					appName:   "app1",
					userID:    "user1",
					sessionID: "session1",
				},
			},
			event: &Event{
				ID: "new_event",
			},
			wantStoredSession: &session{
				id: id{
					appName:   "app1",
					userID:    "user1",
					sessionID: "session1",
				},
				events: []*Event{
					{
						ID: "new_event",
					},
				},
			},
		},
		{
			name:            "append event when session not found",
			inMemoryService: serviceWithData(t),
			session: &session{
				id: id{
					appName:   "app1",
					userID:    "user1",
					sessionID: "custom_session",
				},
			},
			event: &Event{
				ID: "new_event",
			},
			wantStoredSession: &session{
				id: id{
					appName:   "app1",
					userID:    "user1",
					sessionID: "custom_session",
				},
				events: []*Event{
					{
						ID: "new_event",
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ctx := t.Context()

			s := tt.inMemoryService

			err := s.AppendEvent(ctx, tt.session, tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("inMemoryService.AppendEvent() error = %v, wantErr %v", err, tt.wantErr)
			}

			if err != nil {
				return
			}

			resp, err := s.Get(ctx, &GetRequest{
				AppName:   tt.session.AppName(),
				UserID:    tt.session.UserID(),
				SessionID: tt.session.ID(),
			})
			if err != nil {
				t.Fatalf("inMemoryService.Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if diff := cmp.Diff(tt.wantStoredSession, resp.Session,
				cmp.AllowUnexported(session{}),
				cmp.AllowUnexported(id{}),
				cmpopts.IgnoreFields(session{}, "mu")); diff != "" {
				t.Errorf("Create session mismatch: (-want +got):\n%s", diff)
			}
		})
	}
}

func serviceWithData(t *testing.T) *inMemoryService {
	t.Helper()

	service := &inMemoryService{}

	for _, storedSession := range []*session{
		{
			id: id{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session1",
			},
			state: map[string]any{
				"k1": "v1",
			},
		},
		{
			id: id{
				appName:   "app1",
				userID:    "user2",
				sessionID: "session1",
			},
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			id: id{
				appName:   "app1",
				userID:    "user1",
				sessionID: "session2",
			},
			state: map[string]any{
				"k1": "v2",
			},
		},
		{
			id: id{
				appName:   "app2",
				userID:    "user2",
				sessionID: "session2",
			},
			state: map[string]any{
				"k2": "v2",
			},
		},
	} {
		service.sessions.Set(storedSession.id.Encode(), storedSession)
	}

	return service
}

// TODO: test concurrency
