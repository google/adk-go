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
	"fmt"
)

// VertexAiSessionService
type vertexAiService struct {
	client *vertexAiClient
}

func newVertexAiSessionService(location string, projectID string, reasoningEngine string) (Service, error) {
	client, err := newVertexAiClient(location, projectID, reasoningEngine)
	if err != nil {
		return nil, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}

	return &vertexAiService{client: client}, nil
}

func (s *vertexAiService) Create(ctx context.Context, req *CreateRequest) (*CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}
	session, err := s.client.createSession(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return &CreateResponse{Session: session}, nil
}

func (s *vertexAiService) Get(ctx context.Context, req *GetRequest) (*GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
	session, err := s.client.getSession(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}
	return &GetResponse{Session: session}, nil
}

func (s *vertexAiService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}
	sessions, err := s.client.listSessions(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to request sessions list: %w", err)
	}
	return &ListResponse{Sessions: sessions}, nil
}

func (s *vertexAiService) Delete(ctx context.Context, req *DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
	err := s.client.deleteSession(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

func (s *vertexAiService) AppendEvent(ctx context.Context, session Session, event *Event) error {
	if session.ID() == "" || event == nil {
		return fmt.Errorf("session_id and event are required, got session_id: %q, event_id: %t", session.ID(), event == nil)
	}
	err := s.client.appendEvent(ctx, session.ID(), event)
	if err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	return nil
}
