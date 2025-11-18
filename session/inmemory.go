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
	"maps"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"rsc.io/omap"
	"rsc.io/ordered"

	"google.golang.org/adk/internal/sessionutils"
)

type stateMap map[string]any

// inMemoryService is an in-memory implementation of sessionService.Service.
// Thread-safe.
type inMemoryService struct {
	mu        sync.RWMutex
	sessions  omap.Map[string, *session] // session.ID) -> storedSession
	userState map[string]map[string]stateMap
	appState  map[string]stateMap
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
	_, ok := s.sessions.Get(encodedKey)
	if ok {
		return nil, fmt.Errorf("session %s already exists", req.SessionID)
	}

	state := req.State
	if state == nil {
		state = make(stateMap)
	}
	val := &session{
		id:        key,
		state:     state,
		updatedAt: time.Now(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.sessions.Set(encodedKey, val)
	appDelta, userDelta, _ := sessionutils.ExtractStateDeltas(req.State)
	appState := s.updateAppState(appDelta, req.AppName)
	userState := s.updateUserState(userDelta, req.AppName, req.UserID)
	val.state = sessionutils.MergeStates(appState, userState, state)

	copiedSession := copySessionWithoutStateAndEvents(val)
	copiedSession.state = maps.Clone(val.state)
	copiedSession.events = slices.Clone(val.events)

	return &CreateResponse{
		Session: copiedSession,
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

	copiedSession := copySessionWithoutStateAndEvents(res)
	copiedSession.state = s.mergeStates(res.state, appName, userID)

	filteredEvents := res.events
	if req.NumRecentEvents > 0 {
		start := max(len(filteredEvents)-req.NumRecentEvents, 0)
		// create a new slice header pointing to the same array
		filteredEvents = filteredEvents[start:]
	}
	// apply timestamp filter, assuming list is sorted
	if !req.After.IsZero() && len(filteredEvents) > 0 {
		firstIndexToKeep := sort.Search(len(filteredEvents), func(i int) bool {
			// Find the first event that is not before the timestamp
			return !filteredEvents[i].Timestamp.Before(req.After)
		})
		filteredEvents = filteredEvents[firstIndexToKeep:]
	}

	copiedSession.events = make([]*Event, 0, len(filteredEvents))
	copiedSession.events = append(copiedSession.events, filteredEvents...)

	return &GetResponse{
		Session: copiedSession,
	}, nil
}

func (s *inMemoryService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	appName, userID := req.AppName, req.UserID
	if appName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", appName)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	lo := id{appName: appName, userID: userID}.Encode()

	var hi string
	if userID == "" {
		hi = id{appName: appName + "\x00"}.Encode()
	} else {
		hi = id{appName: appName, userID: userID + "\x00"}.Encode()
	}

	sessions := make([]Session, 0)
	for k, storedSession := range s.sessions.Scan(lo, hi) {
		var key id
		if err := key.Decode(k); err != nil {
			return nil, fmt.Errorf("failed to decode key: %w", err)
		}

		if key.appName != appName && key.userID != userID {
			break
		}
		copiedSession := copySessionWithoutStateAndEvents(storedSession)
		copiedSession.state = s.mergeStates(storedSession.state, appName, storedSession.UserID())
		sessions = append(sessions, copiedSession)
	}
	return &ListResponse{
		Sessions: sessions,
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
	if curSession == nil {
		return fmt.Errorf("session is nil")
	}
	if event == nil {
		return fmt.Errorf("event is nil")
	}
	if event.Partial {
		return nil
	}

	sess, ok := curSession.(*session)
	if !ok {
		return fmt.Errorf("unexpected session type %T", sess)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	stored_session, ok := s.sessions.Get(sess.id.Encode())
	if !ok {
		return fmt.Errorf("session not found, cannot apply event")
	}

	// update the in-memory session
	if err := sess.appendEvent(event); err != nil {
		return fmt.Errorf("fail to set state on appendEvent: %w", err)
	}

	// update the in-memory session service
	stored_session.events = append(stored_session.events, event)
	stored_session.updatedAt = event.Timestamp
	if len(event.Actions.StateDelta) > 0 {
		appDelta, userDelta, sessionDelta := sessionutils.ExtractStateDeltas(event.Actions.StateDelta)
		s.updateAppState(appDelta, curSession.AppName())
		s.updateUserState(userDelta, curSession.AppName(), curSession.UserID())
		maps.Copy(stored_session.state, sessionDelta)
	}
	return nil
}

func (s *inMemoryService) updateAppState(appDelta stateMap, appName string) stateMap {
	innerMap, ok := s.appState[appName]
	if !ok {
		innerMap = make(stateMap)
		s.appState[appName] = innerMap
	}
	maps.Copy(innerMap, appDelta)
	return innerMap
}

func (s *inMemoryService) updateUserState(userDelta stateMap, appName, userID string) stateMap {
	innerUsersMap, ok := s.userState[appName]
	if !ok {
		innerUsersMap = make(map[string]stateMap)
		s.userState[appName] = innerUsersMap
	}
	innerMap, ok := innerUsersMap[userID]
	if !ok {
		innerMap = make(stateMap)
		innerUsersMap[userID] = innerMap
	}
	maps.Copy(innerMap, userDelta)
	return innerMap
}

func (s *inMemoryService) mergeStates(state stateMap, appName, userID string) stateMap {
	appState := s.appState[appName]
	var userState stateMap
	userStateMap, ok := s.userState[appName]
	if ok {
		userState = userStateMap[userID]
	}
	return sessionutils.MergeStates(appState, userState, state)
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

func (s *session) appendEvent(event *Event) error {
	if event.Partial {
		return nil
	}

	processedEvent := trimTempDeltaState(event)
	if err := updateSessionState(s, processedEvent); err != nil {
		return fmt.Errorf("error on appendEvent: %w", err)
	}

	s.events = append(s.events, event)
	s.updatedAt = event.Timestamp
	return nil
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

// trimTempDeltaState removes temporary state delta keys from the event.
func trimTempDeltaState(event *Event) *Event {
	if len(event.Actions.StateDelta) == 0 {
		return event
	}

	// Iterate over the map and build a new one with the keys we want to keep.
	filteredStateDelta := make(map[string]any)
	for key, value := range event.Actions.StateDelta {
		if !strings.HasPrefix(key, KeyPrefixTemp) {
			filteredStateDelta[key] = value
		}
	}

	// Replace the old map with the newly filtered one.
	event.Actions.StateDelta = filteredStateDelta

	return event
}

// updateSessionState updates the session state based on the event state delta.
func updateSessionState(session *session, event *Event) error {
	if event.Actions.StateDelta == nil {
		return nil // Nothing to do
	}

	// ensure the session state map is initialized
	if session.state == nil {
		session.state = make(map[string]any)
	}

	state := session.State()
	for key, value := range event.Actions.StateDelta {
		if strings.HasPrefix(key, KeyPrefixTemp) {
			continue
		}
		err := state.Set(key, value)
		if err != nil {
			return fmt.Errorf("error on updateSessionState state: %w", err)
		}
	}
	return nil
}

func copySessionWithoutStateAndEvents(sess *session) *session {
	return &session{
		id: id{
			appName:   sess.id.appName,
			userID:    sess.id.userID,
			sessionID: sess.id.sessionID,
		},
		updatedAt: sess.updatedAt,
	}
}

var _ Service = (*inMemoryService)(nil)
