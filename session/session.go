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
	"slices"
	"sync"

	"github.com/google/adk-go"
)

// InMemorySessionService is an in-memory implementation of adk.SessionService.
// It is primarily for testing and demonstration purposes.
type InMemorySessionService struct {
	mu sync.Mutex
	// TODO: use ordered key instead of nested maps?
	// appID -> userID -> sessionID -> Session
	sessions map[string]map[string]map[string]*session
	// TODO: user_state
	// TODO: app_state
}

type session struct {
	AppName string
	UserID  string
	ID      string

	events []*adk.Event
}

func (s *session) AppendEvent(ctx context.Context, event *adk.Event) {
	s.events = append(s.events, event)
}

// AppendEvent implements adk.SessionService.
func (s *InMemorySessionService) AppendEvent(ctx context.Context, req *adk.SessionAppendEventRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	session := s.getSession(req.Session.AppName, req.Session.UserID, req.Session.ID, false)
	session.events = append(session.events, req.Event)
	return nil
}

func (s *InMemorySessionService) getSession(appID, userID, sessionID string, create bool) *session {
	byAppID, ok := s.sessions[appID]
	if !ok {
		if !create {
			return nil
		}
		byAppID = make(map[string]map[string]*session)
		s.sessions[appID] = byAppID
	}
	byUserID, ok := byAppID[userID]
	if !ok {
		if !create {
			return nil
		}
		byUserID = make(map[string]*session)
		byAppID[userID] = byUserID
	}
	bySessionID, ok := byUserID[sessionID]
	if !ok {
		if !create {
			return nil
		}
		bySessionID = &session{
			ID:      sessionID,
			AppName: appID,
			UserID:  userID,
		}
		byUserID[sessionID] = bySessionID
	} else {
		if create {
			return nil
		}
	}
	return bySessionID
}

// Create implements adk.SessionService.
func (s *InMemorySessionService) Create(ctx context.Context, req *adk.SessionCreateRequest) (*adk.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	session := s.getSession(req.AppName, req.UserID, req.SessionID, true)
	if session == nil {
		return nil, fmt.Errorf("session already exists")
	}
	return &adk.Session{
		AppName: session.AppName,
		UserID:  session.UserID,
		ID:      session.ID,
	}, nil
}

// Delete implements adk.SessionService.
func (s *InMemorySessionService) Delete(ctx context.Context, req *adk.SessionDeleteRequest) error {
	// TODO: should we return err if session doesn't exist?
	s.mu.Lock()
	defer s.mu.Unlock()
	byAppID, ok := s.sessions[req.AppName]
	if !ok {
		return nil
	}
	byUserID, ok := byAppID[req.UserID]
	if !ok {
		return nil
	}
	delete(byUserID, req.SessionID)
	if len(byUserID) == 0 {
		delete(byAppID, req.UserID)
	}
	if len(byAppID) == 0 {
		delete(s.sessions, req.AppName)
	}
	return nil
}

// Get implements adk.SessionService.
func (s *InMemorySessionService) Get(ctx context.Context, req *adk.SessionGetRequest) (*adk.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	session := s.getSession(req.AppName, req.UserID, req.SessionID, false)
	if session == nil {
		return nil, fmt.Errorf("session not found")
	}
	return &adk.Session{
		AppName: session.AppName,
		UserID:  session.UserID,
		ID:      session.ID,
		Events:  slices.Clone(session.events),
	}, nil
}

// List implements adk.SessionService.
func (s *InMemorySessionService) List(ctx context.Context, req *adk.SessionListRequest) ([]*adk.Session, error) {
	panic("unimplemented")
}

// NewInMemorySessionService creates a new InMemorySessionService.
func NewInMemorySessionService() *InMemorySessionService {
	return &InMemorySessionService{
		sessions: make(map[string]map[string]map[string]*session),
	}
}

var _ adk.SessionService = (*InMemorySessionService)(nil)
