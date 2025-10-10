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
	"iter"
	"sync"
	"time"

	"github.com/google/uuid"
	"rsc.io/omap"
	"rsc.io/ordered"
)

// inMemoryService is an in-memory implementation of sessionService.Service.
// Thread-safe.
type inMemoryService struct {
	mu       sync.RWMutex
	sessions omap.Map[string, *session] // session.ID) -> storedSession
}

func (s *inMemoryService) Create(ctx context.Context, req *CreateRequest) (*CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.NewString()
	}

	key := id{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: sessionID,
	}

	encodedKey := key.Encode()

	stateMap := req.State
	if stateMap == nil {
		stateMap = make(map[string]any)
	}
	val := &session{
		id:        key,
		state:     stateMap,
		updatedAt: time.Now(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.sessions.Set(encodedKey, val)

	return &CreateResponse{
		Session: val,
	}, nil
}

func (s *inMemoryService) Get(ctx context.Context, req *GetRequest) (*GetResponse, error) {
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	id := id{
		appName:   appName,
		userID:    userID,
		sessionID: sessionID,
	}

	res, ok := s.sessions.Get(id.Encode())
	if !ok {
		return nil, fmt.Errorf("session %+v not found", req.SessionID)
	}

	// TODO: handle req.NumRecentEvents and req.After
	return &GetResponse{
		Session: res,
	}, nil
}

func (s *inMemoryService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	lo := id{appName: req.AppName, userID: req.UserID}.Encode()
	hi := id{appName: req.AppName, userID: req.UserID + "\x00"}.Encode()

	var res []Session
	for k, storedSession := range s.sessions.Scan(lo, hi) {
		var key id
		if err := key.Decode(k); err != nil {
			return nil, fmt.Errorf("failed to decode key: %w", err)
		}

		if key.appName != req.AppName && key.userID != req.UserID {
			break
		}

		res = append(res, storedSession)
	}
	return &ListResponse{
		Sessions: res,
	}, nil
}

func (s *inMemoryService) Delete(ctx context.Context, req *DeleteRequest) error {
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	id := id{
		appName:   appName,
		userID:    userID,
		sessionID: sessionID,
	}

	s.sessions.Delete(id.Encode())
	return nil
}

func (s *inMemoryService) AppendEvent(ctx context.Context, curSession Session, event *Event) error {
	if curSession == nil || event == nil {
		return fmt.Errorf("session or event are nil")
	}

	// TODO: no-op if event is partial.
	// TODO: process event actions and state delta.

	sess, ok := curSession.(*session)
	if !ok {
		return fmt.Errorf("unexpected session type %T", sess)
	}

	if event.Actions.StateDelta != nil {
		state := sess.State()
		for key, value := range event.Actions.StateDelta {
			err := state.Set(key, value)
			if err != nil {
				return fmt.Errorf("fail to set state on appendEvent: %w", err)
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	sess.appendEvent(event)

	s.sessions.Set(sess.id.Encode(), sess)

	return nil
}

func (id id) Encode() string {
	return string(ordered.Encode(id.appName, id.userID, id.sessionID))
}

func (id *id) Decode(key string) error {
	return ordered.Decode([]byte(key), &id.appName, &id.userID, &id.sessionID)
}

type id struct {
	appName   string
	userID    string
	sessionID string
}

type session struct {
	id id

	// guards all mutable fields
	mu        sync.RWMutex
	events    []*Event
	state     map[string]any
	updatedAt time.Time
}

func (s *session) ID() string {
	return s.id.sessionID
}

func (s *session) AppName() string {
	return s.id.appName
}

func (s *session) UserID() string {
	return s.id.userID
}

func (s *session) State() State {
	return &state{
		mu:    &s.mu,
		state: s.state,
	}
}

func (s *session) Events() Events {
	return events(s.events)
}

func (s *session) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.updatedAt
}

func (s *session) appendEvent(event *Event) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.events = append(s.events, event)
	s.updatedAt = event.Timestamp
}

type events []*Event

func (e events) All() iter.Seq[*Event] {
	return func(yield func(*Event) bool) {
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

func (e events) At(i int) *Event {
	if i >= 0 && i < len(e) {
		return e[i]
	}
	return nil
}

type state struct {
	mu    *sync.RWMutex
	state map[string]any
}

func (s *state) Get(key string) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	val, ok := s.state[key]
	if !ok {
		return nil, ErrStateKeyNotExist
	}

	return val, nil
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

func (s *state) Set(key string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.state[key] = value
	return nil
}

var _ Service = (*inMemoryService)(nil)
