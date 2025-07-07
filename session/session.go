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
	"slices"
	"sync"

	"github.com/google/adk-go"
	"rsc.io/omap"
	"rsc.io/ordered"
)

// InMemorySessionService is an in-memory implementation of adk.SessionService.
// It is primarily for testing and demonstration purposes.
type InMemorySessionService struct {
	mu sync.RWMutex
	// ordered(appName, userID, sessionID) -> session
	sessions omap.Map[string, *session]
}

type sessionKey struct {
	AppName string
	UserID  string
	ID      string
}

func (sk sessionKey) Encode() string {
	return string(ordered.Encode(sk.AppName, sk.UserID, sk.ID))
}

func (sk *sessionKey) Decode(key string) error {
	return ordered.Decode([]byte(key), &sk.AppName, &sk.UserID, &sk.ID)
}

type session struct {
	mu     sync.Mutex
	events []*adk.Event
}

func (s *session) AppendEvent(ctx context.Context, event *adk.Event) {
	s.mu.Lock()
	s.events = append(s.events, event)
	s.mu.Unlock()
}

func (s *session) Events() []*adk.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.events)
}

// scan returns an iterator over all key-value pairs
// in the range begin ≤ key ≤ end.
// TODO: add a concurrent tests.
func (s *InMemorySessionService) scan(lo, hi string) iter.Seq2[sessionKey, *session] {
	return func(yield func(key sessionKey, val *session) bool) {
		s.mu.RLock()
		locked := true
		defer func() {
			if locked {
				s.mu.RUnlock()
			}
		}()
		for k, val := range s.sessions.Scan(lo, hi) {
			var key sessionKey
			if err := key.Decode(k); err != nil {
				println("decode error: %v", err)
				continue
			}

			s.mu.RUnlock()
			locked = false
			if !yield(key, val) {
				return
			}
			s.mu.RLock()
			locked = true
		}
	}
}

// get looks up the session with reader lock.
func (s *InMemorySessionService) get(appName, userID, sessionID string) (*session, bool) {
	key := sessionKey{
		AppName: appName,
		UserID:  userID,
		ID:      sessionID,
	}.Encode()
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions.Get(key)
}

// AppendEvent implements adk.SessionService.
func (s *InMemorySessionService) AppendEvent(ctx context.Context, session *adk.Session, event *adk.Event) error {
	// TODO: no-op if event is partial.
	// TODO: process event actions and state delta.
	sess, ok := s.get(session.AppName, session.UserID, session.ID)
	if !ok {
		return fmt.Errorf("session not found")
	}
	sess.AppendEvent(ctx, event)
	session.Events = append(session.Events, event)
	return nil
}

// Create implements adk.SessionService.
func (s *InMemorySessionService) Create(ctx context.Context, req *adk.SessionCreateRequest) (*adk.Session, error) {
	// TODO: handle req.State.

	key := sessionKey{
		AppName: req.AppName,
		UserID:  req.UserID,
		ID:      req.SessionID,
	}.Encode()

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions.Get(key); ok {
		return nil, fmt.Errorf("session already exists")
	}
	s.sessions.Set(key, &session{})

	return &adk.Session{
		ID:      req.SessionID,
		AppName: req.AppName,
		UserID:  req.UserID,
		// Events: nil
	}, nil
}

// Delete implements adk.SessionService.
func (s *InMemorySessionService) Delete(ctx context.Context, req *adk.SessionDeleteRequest) error {
	// TODO: should we return err if session doesn't exist? This may be difficult or expensive
	// for certain implementations.
	key := sessionKey{
		AppName: req.AppName,
		UserID:  req.UserID,
		ID:      req.SessionID,
	}.Encode()

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions.Get(key); !ok {
		return fmt.Errorf("session %v/%v/%v does not exist", req.AppName, req.UserID, req.SessionID)
	}
	s.sessions.Delete(key)
	return nil
}

// Get implements adk.SessionService.
func (s *InMemorySessionService) Get(ctx context.Context, req *adk.SessionGetRequest) (*adk.Session, error) {
	sess, ok := s.get(req.AppName, req.UserID, req.SessionID)
	if !ok {
		return nil, fmt.Errorf("session %v/%v/%v not found", req.AppName, req.UserID, req.SessionID)
	}
	return &adk.Session{
		AppName: req.AppName,
		UserID:  req.UserID,
		ID:      req.SessionID,
		Events:  sess.Events(),
	}, nil
}

// List implements adk.SessionService.
func (s *InMemorySessionService) List(ctx context.Context, req *adk.SessionListRequest) ([]*adk.Session, error) {
	lo := sessionKey{AppName: req.AppName, UserID: req.UserID}.Encode()
	hi := sessionKey{AppName: req.AppName, UserID: req.UserID + "\x00"}.Encode()

	var ret []*adk.Session
	for key, sess := range s.scan(lo, hi) {
		if key.AppName != req.AppName && key.UserID != req.UserID {
			break
		}
		ret = append(ret, &adk.Session{
			AppName: key.AppName,
			UserID:  key.UserID,
			ID:      key.ID,
			Events:  sess.Events(),
		})
	}
	return ret, nil
}

var _ adk.SessionService = (*InMemorySessionService)(nil)
