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

package sessionservice

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/adk/session"
	"rsc.io/omap"
	"rsc.io/ordered"
)

// inMemoryService is an in-memory implementation of sessionService.Service.
// Thread-safe.
type inMemoryService struct {
	mu       sync.RWMutex
	sessions omap.Map[string, *storedSession] // session.ID) -> storedSession
}

func (s *inMemoryService) Create(ctx context.Context, req *CreateRequest) (StoredSession, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.NewString()
	}

	key := sessionKey{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: sessionID,
	}

	encodedKey := key.Encode()

	val := &storedSession{
		id:        session.ID(key),
		state:     req.State,
		updatedAt: time.Now(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.sessions.Set(encodedKey, val)

	return val, nil
}

func (s *inMemoryService) Get(ctx context.Context, req *GetRequest) (StoredSession, error) {
	appName, userID, sessionID := req.ID.AppName, req.ID.UserID, req.ID.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	res, ok := s.sessions.Get(sessionKey(req.ID).Encode())
	if !ok {
		return nil, fmt.Errorf("session %+v not found", req.ID)
	}

	// TODO: handle req.NumRecentEvents and req.After
	return res, nil
}

// List returns a list of sessions.
func (s *inMemoryService) List(ctx context.Context, req *ListRequest) ([]StoredSession, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	lo := sessionKey{AppName: req.AppName, UserID: req.UserID}.Encode()
	hi := sessionKey{AppName: req.AppName, UserID: req.UserID + "\x00"}.Encode()

	var res []StoredSession
	for k, storedSession := range s.sessions.Scan(lo, hi) {
		var key sessionKey
		if err := key.Decode(k); err != nil {
			return nil, fmt.Errorf("failed to decode key: %w", err)
		}

		if key.AppName != req.AppName && key.UserID != req.UserID {
			break
		}

		res = append(res, storedSession)
	}
	return res, nil
}

func (s *inMemoryService) Delete(ctx context.Context, req *DeleteRequest) error {
	appName, userID, sessionID := req.ID.AppName, req.ID.UserID, req.ID.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.sessions.Delete(sessionKey(req.ID).Encode())
	return nil
}

func (s *inMemoryService) AppendEvent(ctx context.Context, session StoredSession, event *session.Event) error {
	if session == nil || event == nil {
		return fmt.Errorf("session or event are nil")
	}

	// TODO: no-op if event is partial.
	// TODO: process event actions and state delta.

	storedSession, ok := session.(*storedSession)
	if !ok {
		return fmt.Errorf("unexpected session type %T", session)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	storedSession.appendEvent(event)

	s.sessions.Set(sessionKey(session.ID()).Encode(), storedSession)

	return nil
}

type sessionKey session.ID

func (sk sessionKey) Encode() string {
	return string(ordered.Encode(sk.AppName, sk.UserID, sk.SessionID))
}

func (sk *sessionKey) Decode(key string) error {
	return ordered.Decode([]byte(key), &sk.AppName, &sk.UserID, &sk.SessionID)
}

type storedSession struct {
	id session.ID

	// guards all mutable fields
	mu        sync.RWMutex
	events    []*session.Event
	state     map[string]any
	updatedAt time.Time
}

func (s *storedSession) ID() session.ID {
	return s.id
}

func (s *storedSession) State() session.ReadOnlyState {
	return &state{
		mu:    &s.mu,
		state: s.state,
	}
}

func (s *storedSession) Events() session.Events {
	return events(s.events)
}

func (s *storedSession) Updated() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.updatedAt
}

func (s *storedSession) appendEvent(event *session.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.events = append(s.events, event)
	s.updatedAt = event.Time
}

type events []*session.Event

func (e events) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e {
			if !yield(event) {
				return
			}
		}
	}
}

func (e events) Len() int {
	return len(e)
}

func (e events) At(i int) *session.Event {
	if i >= 0 && i < len(e) {
		return e[i]
	}
	return nil
}

type state struct {
	mu    *sync.RWMutex
	state map[string]any
}

func (s *state) Get(key string) any {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.state[key]
}

func (s *state) All() iter.Seq2[string, any] {
	return func(yield func(key string, val any) bool) {
		s.mu.RLock()

		for k, v := range s.state {
			s.mu.RUnlock()
			if !yield(k, v) {
				return
			}
			s.mu.RLock()
		}

		s.mu.RUnlock()
	}
}

var _ Service = (*inMemoryService)(nil)
