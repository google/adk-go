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
	"fmt"

	"golang.org/x/sync/errgroup"
	"google.golang.org/api/option"

	"google.golang.org/adk/session"
)

// VertexAiSessionService
type vertexAiService struct {
	client *vertexAiClient
}

type VertexAIServiceConfig struct {
	Location        string
	ProjectID       string
	ReasoningEngine string
}

// NewSessionService returns VertextAiSessionService implementation.
func NewSessionService(ctx context.Context, cfg VertexAIServiceConfig, opts ...option.ClientOption) (session.Service, error) {
	client, err := newVertexAiClient(ctx, cfg.Location, cfg.ProjectID, cfg.ReasoningEngine, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}

	return &vertexAiService{client: client}, nil
}

func (s *vertexAiService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}
	if req.SessionID != "" {
		return nil, fmt.Errorf("user-provided Session id is not supported for VertexAISessionService: %q", req.SessionID)
	}
	sess, err := s.client.createSession(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return &session.CreateResponse{Session: sess}, nil
}

func (s *vertexAiService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}

	// gCtx will be canceled if either function returns an error
	g, gCtx := errgroup.WithContext(ctx)

	var (
		sess   *localSession
		events []*session.Event
	)

	g.Go(func() error {
		var err error
		sess, err = s.client.getSession(gCtx, req)
		if err != nil {
			return fmt.Errorf("failed to get session: %w", err)
		}
		return nil
	})

	g.Go(func() error {
		var err error
		events, err = s.client.listSessionEvents(gCtx, req.AppName, req.SessionID, req.After, req.NumRecentEvents)
		if err != nil {
			return fmt.Errorf("failed to list session events: %w", err)
		}
		return nil
	})

	if err := g.Wait(); err != nil {
		return nil, err
	}
	sess.events = events
	return &session.GetResponse{Session: sess}, nil
}

func (s *vertexAiService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	sessions, err := s.client.listSessions(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to request sessions list: %w", err)
	}
	return &session.ListResponse{Sessions: sessions}, nil
}

func (s *vertexAiService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return fmt.Errorf("app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
	err := s.client.deleteSession(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

func (s *vertexAiService) AppendEvent(ctx context.Context, sess session.Session, event *session.Event) error {
	if sess.ID() == "" || event == nil {
		return fmt.Errorf("session_id and event are required, got session_id: %q, event_id: %t", sess.ID(), event == nil)
	}
	err := s.client.appendEvent(ctx, sess.AppName(), sess.ID(), event)
	if err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	sessInt, ok := sess.(*localSession)
	if !ok {
		return fmt.Errorf("AppendEvent for Vertex AI service only supports sessions created by it, got %T", sess)
	}
	err = sessInt.appendEvent(event)
	if err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	return nil
}
