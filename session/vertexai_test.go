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
	"time"

	"github.com/google/uuid"
)

const (
	ProjectID = "adk-go-samples-sandbox-390724"
	Location  = "us-central1"
	EngineId  = "5577659759986737152"
	AppName   = "test-app"
	UserID    = "test-user"
	SessionId = "8864667638287040512"
)

func TestNewVertexSessionSerice(t *testing.T) {
	tests := []struct {
		name    string
		wantErr bool
	}{
		{
			name:    "with project and location",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := newVertexAiSessionService(Location, ProjectID, EngineId)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewVertexSession() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got == nil {
				t.Errorf("NewVertexSession() got nil, want session")
			}
		})
	}
}

func Test_vertexAiService_Create(t *testing.T) {
	s, err := newVertexAiSessionService(Location, ProjectID, EngineId)
	if err != nil {
		t.Fatalf("newVertexAiSessionService() error = %v", err)
	}
	tests := []struct {
		name    string
		req     *CreateRequest
		wantErr bool
	}{
		{
			name: "create session ok",
			req: &CreateRequest{
				AppName: AppName,
				UserID:  UserID,
			},
			wantErr: false,
		},
		{
			name:    "missing required fields",
			req:     &CreateRequest{},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := s.Create(context.Background(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.Create() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && (resp.Session == nil || resp.Session.ID() == "") {
				t.Errorf("vertexAiService.Create() did not return a valid session")
			}
		})
	}
}

func Test_vertexAiService_Get(t *testing.T) {
	s, err := newVertexAiSessionService(Location, ProjectID, EngineId)
	if err != nil {
		t.Fatalf("newVertexAiSessionService() error = %v", err)
	}

	tests := []struct {
		name    string
		req     *GetRequest
		wantErr bool
	}{
		{
			name: "get session ok",
			req: &GetRequest{
				AppName:   AppName,
				UserID:    UserID,
				SessionID: SessionId,
			},
			wantErr: false,
		},
		{
			name:    "missing required fields",
			req:     &GetRequest{},
			wantErr: true,
		},
		{
			name: "session not found",
			req: &GetRequest{
				AppName:   AppName,
				UserID:    UserID,
				SessionID: "invalid",
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Get(context.Background(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.Get() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_vertexAiService_List(t *testing.T) {
	tests := []struct {
		name    string
		req     *ListRequest
		wantErr bool
	}{
		{
			name: "list sessions",
			req: &ListRequest{
				AppName: AppName,
				UserID:  UserID,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := newVertexAiSessionService(Location, ProjectID, EngineId)
			if err != nil {
				t.Fatalf("newVertexAiSessionService() error = %v", err)
			}
			sessions, err := s.List(context.Background(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.List() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if sessions == nil || len(sessions.Sessions) == 0 {
				t.Errorf("vertexAiService.List() returned no sessions")
			}
		})
	}
}

func Test_vertexAiService_Delete(t *testing.T) {
	s, err := newVertexAiSessionService(Location, ProjectID, EngineId)
	if err != nil {
		t.Fatalf("newVertexAiSessionService() error = %v", err)
	}

	createReq := &CreateRequest{
		AppName: AppName,
		UserID:  UserID,
	}
	createResp, err := s.Create(context.Background(), createReq)
	if err != nil {
		t.Fatalf("vertexAiService.Create() for Delete test failed: %v", err)
	}
	sessionID := createResp.Session.ID()

	tests := []struct {
		name    string
		req     *DeleteRequest
		wantErr bool
	}{
		{
			name: "delete session ok",
			req: &DeleteRequest{
				AppName:   AppName,
				UserID:    UserID,
				SessionID: sessionID,
			},
			wantErr: false,
		},
		{
			name:    "missing required fields",
			req:     &DeleteRequest{},
			wantErr: true,
		},
		{
			name: "session not found",
			req: &DeleteRequest{
				AppName:   AppName,
				UserID:    UserID,
				SessionID: "invalid-session-id",
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.Delete(context.Background(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.Delete() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_vertexAiService_AppendEvent(t *testing.T) {
	s, err := newVertexAiSessionService(Location, ProjectID, EngineId)
	if err != nil {
		t.Fatalf("newVertexAiSessionService() error = %v", err)
	}

	// First, create a session to have something to append to.
	createReq := &CreateRequest{
		AppName: AppName,
		UserID:  UserID,
	}
	createResp, err := s.Create(context.Background(), createReq)
	if err != nil {
		t.Fatalf("vertexAiService.Create() for AppendEvent test failed: %v", err)
	}
	sessionToAppend := createResp.Session

	tests := []struct {
		name    string
		session Session
		event   *Event
		wantErr bool
	}{
		{
			name:    "append event ok",
			session: sessionToAppend,
			event: &Event{
				Timestamp:    time.Now(),
				InvocationID: uuid.NewString(),
				Author:       UserID,
			},
			wantErr: false,
		},
		{
			name:    "missing session id",
			session: &session{id: id{appName: AppName, userID: UserID}},
			event:   &Event{},
			wantErr: true,
		},
		{
			name:    "nil event",
			session: sessionToAppend,
			event:   nil,
			wantErr: true,
		},
		{
			name:    "missing author",
			session: sessionToAppend,
			event: &Event{
				Timestamp:    time.Now(),
				InvocationID: uuid.NewString(),
			},
			wantErr: true,
		},
		{
			name:    "missing invocation id",
			session: sessionToAppend,
			event: &Event{
				Timestamp: time.Now(),
				Author:    UserID,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.AppendEvent(context.Background(), tt.session, tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("vertexAiService.AppendEvent() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
