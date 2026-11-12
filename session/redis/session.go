// Copyright 2026 Google LLC
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

package redis

import (
	"iter"
	"maps"
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/session"
)

// storedSession is the JSON document kept at a session key. State holds only
// session-scoped keys: app: and user: keys live in their own documents and
// temp: keys are never stored. LastUpdateTime keeps full time.Time precision,
// so a session read back reports the same time as the one Create or
// AppendEvent updated.
type storedSession struct {
	ID             string           `json:"id"`
	AppName        string           `json:"app_name"`
	UserID         string           `json:"user_id"`
	State          map[string]any   `json:"state"`
	Events         []*session.Event `json:"events"`
	LastUpdateTime time.Time        `json:"last_update_time"`
}

// localSession is the session.Session the service returns. Its state holds
// the merged app:, user: and session-scoped keys, plus any temp: keys applied
// to it.
type localSession struct {
	appName   string
	userID    string
	sessionID string

	// mu guards events, state and updatedAt.
	mu        sync.RWMutex
	events    []*session.Event
	state     map[string]any
	updatedAt time.Time
}

func (s *localSession) ID() string      { return s.sessionID }
func (s *localSession) AppName() string { return s.appName }
func (s *localSession) UserID() string  { return s.userID }

func (s *localSession) State() session.State {
	return &state{mu: &s.mu, state: s.state}
}

func (s *localSession) Events() session.Events {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return events(slices.Clone(s.events))
}

func (s *localSession) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.updatedAt
}

// CommitAppend has the method set of adk-go's internal MutableSession, so the
// in-memory and database services, and this one, update a localSession the
// same way after a stored append. trimmedEvent must be a *session.Event and is
// appended as is. rawStateDelta is applied with its temp: keys, which stay
// readable on s for the rest of the invocation.
func (s *localSession) CommitAppend(trimmedEvent any, rawStateDelta map[string]any, updatedAt time.Time) {
	ev := trimmedEvent.(*session.Event)

	s.mu.Lock()
	defer s.mu.Unlock()
	maps.Copy(s.state, rawStateDelta)
	s.events = append(s.events, ev)
	s.updatedAt = updatedAt
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

func (e events) Len() int { return len(e) }

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

func (s *state) Get(key string) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.state[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

func (s *state) All() iter.Seq2[string, any] {
	s.mu.RLock()
	stateCopy := maps.Clone(s.state)
	s.mu.RUnlock()
	return maps.All(stateCopy)
}

func (s *state) Set(key string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[key] = value
	return nil
}

var (
	_ session.Session = (*localSession)(nil)
	_ session.Events  = events(nil)
	_ session.State   = (*state)(nil)
)
