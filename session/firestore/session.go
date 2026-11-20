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

package firestore

import (
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/session"
)

type localSession struct {
	appName   string
	userID    string
	sessionID string

	mu        sync.RWMutex
	events    []*session.Event
	state     map[string]any
	updatedAt time.Time
	revision  int64
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
	s.mu.RLock()
	defer s.mu.RUnlock()

	return events(slices.Clone(s.events))
}

func (s *localSession) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.updatedAt
}

func (s *localSession) currentRevision() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.revision
}

func (s *localSession) commitAppend(stateDelta map[string]any, trimmedEvent *session.Event, newRevision int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == nil {
		s.state = make(map[string]any)
	}
	if stateDelta != nil {
		maps.Copy(s.state, stateDelta)
	}

	s.events = append(s.events, trimmedEvent)
	s.updatedAt = trimmedEvent.Timestamp
	s.revision = newRevision
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
	s.mu.RLock()
	stateCopy := maps.Clone(s.state)
	s.mu.RUnlock()

	return func(yield func(key string, val any) bool) {
		for k, v := range stateCopy {
			if !yield(k, v) {
				return
			}
		}
	}
}

func (s *state) Set(key string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == nil {
		return fmt.Errorf("session state map is nil")
	}
	s.state[key] = value
	return nil
}

// trimTempDeltaState returns an event with temporary state delta keys removed,
// without mutating the caller's Event or its StateDelta map.
func trimTempDeltaState(event *session.Event) *session.Event {
	if len(event.Actions.StateDelta) == 0 {
		return event
	}

	filteredStateDelta := make(map[string]any, len(event.Actions.StateDelta))
	for key, value := range event.Actions.StateDelta {
		if !strings.HasPrefix(key, session.KeyPrefixTemp) {
			filteredStateDelta[key] = value
		}
	}

	if len(filteredStateDelta) == len(event.Actions.StateDelta) {
		return event
	}

	eventCopy := *event
	eventCopy.Actions.StateDelta = filteredStateDelta
	return &eventCopy
}

// extractStateDeltas splits a state delta map into app, user, and session
// maps based on key prefixes, dropping temporary keys.
func extractStateDeltas(delta map[string]any) (appDelta, userDelta, sessionDelta map[string]any) {
	appDelta = make(map[string]any)
	userDelta = make(map[string]any)
	sessionDelta = make(map[string]any)

	for key, value := range delta {
		if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixApp); found {
			appDelta[cleanKey] = value
		} else if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixUser); found {
			userDelta[cleanKey] = value
		} else if !strings.HasPrefix(key, session.KeyPrefixTemp) {
			sessionDelta[key] = value
		}
	}
	return appDelta, userDelta, sessionDelta
}

// mergeStates combines app, user, and session state maps into a single map,
// restoring the "app:" and "user:" prefixes.
func mergeStates(appState, userState, sessionState map[string]any) map[string]any {
	totalSize := len(appState) + len(userState) + len(sessionState)
	merged := make(map[string]any, totalSize)

	maps.Copy(merged, sessionState)
	for key, value := range appState {
		merged[session.KeyPrefixApp+key] = value
	}
	for key, value := range userState {
		merged[session.KeyPrefixUser+key] = value
	}
	return merged
}

var (
	_ session.Session = (*localSession)(nil)
	_ session.Events  = (*events)(nil)
	_ session.State   = (*state)(nil)
)
