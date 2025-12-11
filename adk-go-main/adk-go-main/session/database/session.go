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

package database

import (
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/session"
)

// TODO localSession is identical to session.session. Move to sessioninternal
type localSession struct {
	appName   string
	userID    string
	sessionID string

	// guards all mutable fields
	mu        sync.RWMutex
	events    []*session.Event
	state     map[string]any
	updatedAt time.Time
}

func (s *localSession) ID() string {
	return s.sessionID
}

func (s *localSession) AppName() string {
	return s.appName
}

func (s *localSession) UserID() string {
	return s.userID
}

func (s *localSession) State() session.State {
	return &state{
		mu:    &s.mu,
		state: s.state,
	}
}

func (s *localSession) Events() session.Events {
	return events(s.events)
}

func (s *localSession) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.updatedAt
}

func (s *localSession) appendEvent(event *session.Event) error {
	if event.Partial {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	processedEvent := trimTempDeltaState(event)
	if err := updateSessionState(s, processedEvent); err != nil {
		return fmt.Errorf("failed to update localSession state: %w", err)
	}

	s.events = append(s.events, event)
	s.updatedAt = event.Timestamp
	return nil
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

// TrimTempDeltaState removes temporary state delta keys from the event.
func trimTempDeltaState(event *session.Event) *session.Event {
	if len(event.Actions.StateDelta) == 0 {
		return event
	}

	// Iterate over the map and build a new one with the keys we want to keep.
	filteredStateDelta := make(map[string]any)
	for key, value := range event.Actions.StateDelta {
		if !strings.HasPrefix(key, session.KeyPrefixTemp) {
			filteredStateDelta[key] = value
		}
	}

	// Replace the old map with the newly filtered one.
	event.Actions.StateDelta = filteredStateDelta

	return event
}

// updateSessionState updates the session state based on the event state delta.
func updateSessionState(sess *localSession, event *session.Event) error {
	if event.Actions.StateDelta == nil {
		return nil // Nothing to do
	}

	// Ensure the session state map is initialized
	if sess.state == nil {
		sess.state = make(map[string]any)
	}

	for key, value := range event.Actions.StateDelta {
		if strings.HasPrefix(key, session.KeyPrefixTemp) {
			continue
		}
		sess.state[key] = value
	}

	return nil
}

var (
	_ session.Session = (*localSession)(nil)
	_ session.Events  = (*events)(nil)
	_ session.State   = (*state)(nil)
)
