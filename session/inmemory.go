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
	"cmp"
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"rsc.io/omap"
	"rsc.io/ordered"

	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/platform"
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
		sessionID = platform.NewUUID(ctx)
	}

	key := id{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: sessionID,
	}

	encodedKey := key.Encode()
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.sessions.Get(encodedKey); ok {
		return nil, fmt.Errorf("session %s already exists", req.SessionID)
	}

	// The stored session keeps only session-scoped keys: app: and user: keys
	// go to the shared scoped state and temp: keys are not stored at all, as
	// in adk-python (in_memory_session_service.py, _create_session_impl).
	appDelta, userDelta, sessionState := sessionutils.ExtractStateDeltas(req.State)
	val := &session{
		id:        key,
		state:     sessionState,
		updatedAt: platform.Now(ctx),
	}

	s.sessions.Set(encodedKey, val)
	s.updateAppState(appDelta, req.AppName)
	s.updateUserState(userDelta, req.AppName, req.UserID)

	copiedSession := copySessionWithoutStateAndEvents(val)
	copiedSession.state = s.mergeStates(val.state, req.AppName, req.UserID)
	copiedSession.events = slices.Clone(val.events)

	return &CreateResponse{
		Session: copiedSession,
	}, nil
}

// GetUserState implements [UserStateReader]. It returns a deep copy of the
// user state stored for (req.AppName, req.UserID), as adk-python's
// get_user_state does (in_memory_session_service.py).
func (s *inMemoryService) GetUserState(ctx context.Context, req *GetUserStateRequest) (*GetUserStateResponse, error) {
	if req == nil {
		return nil, errors.New("session: get user state request is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Copy under the lock: Create and AppendEvent write the stored user state
	// under s.mu.
	return &GetUserStateResponse{State: sessionutils.CloneStateMap(s.userState[req.AppName][req.UserID])}, nil
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
		return nil, fmt.Errorf("%w: %q", ErrNotFound, req.SessionID)
	}

	copiedSession := copySessionWithoutStateAndEvents(res)
	copiedSession.state = s.mergeStates(res.state, appName, userID)

	filteredEvents := res.events
	if req.NumRecentEvents > 0 {
		start := max(len(filteredEvents)-req.NumRecentEvents, 0)
		// create a new slice header pointing to the same array
		filteredEvents = filteredEvents[start:]
	}

	copiedSession.events = make([]*Event, 0, len(filteredEvents))
	for _, event := range filteredEvents {
		// Test every event rather than searching for a cut point: events are
		// kept in append order, which need not be timestamp order, so a
		// binary search over it could keep events from before req.After and
		// drop ones at or after it. With NumRecentEvents unset, this returns
		// every event at or after req.After, as the database and Vertex AI
		// backends do. With it set, the trim above runs first, in append
		// order, as in adk-python, while those backends filter first and
		// then take the newest N. adk-python's in-memory filter scans back
		// from the newest event to the first one before After, so it also
		// drops a qualifying event appended ahead of that one.
		if !req.After.IsZero() && event.Timestamp.Before(req.After) {
			continue
		}
		copiedSession.events = append(copiedSession.events, event)
	}

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
	slices.SortStableFunc(sessions, compareByLastUpdate)
	return &ListResponse{
		Sessions: sessions,
	}, nil
}

// compareByLastUpdate orders sessions by last update time ascending, ties by
// user ID then session ID, as [Service.List] documents.
func compareByLastUpdate(a, b Session) int {
	return cmp.Or(
		a.LastUpdateTime().Compare(b.LastUpdateTime()),
		strings.Compare(a.UserID(), b.UserID()),
		strings.Compare(a.ID(), b.ID()),
	)
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
	// A backend that persists events could not read this request back, so
	// this one refuses it too.
	if err := sessionutils.CheckRequestedAuthConfigs(event.Actions.RequestedAuthConfigs); err != nil {
		return fmt.Errorf("cannot append event: %w", err)
	}
	// Give the event an identity if it arrived without one, the same way a
	// missing session ID is filled in on Create. [NewEvent] assigns one, but an
	// event built as a struct literal by an agent or a tool never goes through
	// it, and anything that identifies events by ID cannot tell two ID-less
	// events apart.
	if event.ID == "" {
		event.ID = platform.NewUUID(ctx)
	}

	// The session is found by its identity, whatever the concrete type of
	// curSession: a copy made with CloneSession or a value the caller built
	// names a stored session as well as the value Create returned.
	key := id{appName: curSession.AppName(), userID: curSession.UserID(), sessionID: curSession.ID()}

	s.mu.Lock()
	defer s.mu.Unlock()

	stored_session, ok := s.sessions.Get(key.Encode())
	if !ok {
		return fmt.Errorf("%w: %q, cannot apply event", ErrNotFound, key.sessionID)
	}

	trimmedEvent := trimTempDeltaState(event)
	// The canonical record gets a map of its own: trimTempDeltaState returns
	// the caller's map when it strips nothing.
	trimmedDelta := maps.Clone(trimTempDeltaState(event).Actions.StateDelta)

	// Only StateDelta comes from the trim above; every other field below is
	// read off event. Compaction, ArtifactDelta, RequestedToolConfirmations,
	// RequestedAuthConfigs, SetModelResponse, EndOfAgent, AgentState and
	// RenderUIWidgets are cloned so that the caller's event shares none of
	// them with the canonical record.
	eventCopy := &Event{
		ID:             event.ID,
		InvocationID:   event.InvocationID,
		Timestamp:      event.Timestamp,
		Author:         event.Author,
		Branch:         event.Branch,
		IsolationScope: event.IsolationScope,
		Actions: EventActions{
			StateDelta:                 trimmedDelta,
			ArtifactDelta:              maps.Clone(event.Actions.ArtifactDelta),
			RequestedToolConfirmations: maps.Clone(event.Actions.RequestedToolConfirmations),
			RequestedAuthConfigs:       maps.Clone(event.Actions.RequestedAuthConfigs),
			TransferToAgent:            event.Actions.TransferToAgent,
			TransferReason:             event.Actions.TransferReason,
			Escalate:                   event.Actions.Escalate,
			SetModelResponse:           sessionutils.CloneValue(event.Actions.SetModelResponse),
			RewindBeforeInvocationID:   event.Actions.RewindBeforeInvocationID,
			EndOfAgent:                 sessionutils.CloneValue(event.Actions.EndOfAgent),
			AgentState:                 sessionutils.CloneValue(event.Actions.AgentState),
			SkipSummarization:          event.Actions.SkipSummarization,
			Compaction:                 event.Actions.Compaction.clone(),
			RenderUIWidgets:            sessionutils.CloneValue(event.Actions.RenderUIWidgets),
		},
		LongRunningToolIDs: slices.Clone(event.LongRunningToolIDs),
		Routes:             slices.Clone(event.Routes),
		RequestedInput:     event.RequestedInput,
		LLMResponse:        event.LLMResponse,
		Output:             event.Output,
		NodeInfo:           event.NodeInfo,
	}

	// An equal re-delivery of a stored event is ignored, so a caller that
	// retries AppendEvent neither stores the event twice nor applies its
	// state delta twice. Distinct events that share an ID are all kept, as in
	// adk-python (in_memory_session_service.py, append_event).
	if slices.ContainsFunc(stored_session.events, func(e *Event) bool {
		return e.ID == eventCopy.ID && eventsEqual(e, eventCopy)
	}) {
		return nil
	}

	// update the in-memory session service
	stored_session.events = append(stored_session.events, eventCopy)
	stored_session.updatedAt = event.Timestamp
	if len(event.Actions.StateDelta) > 0 {
		appDelta, userDelta, sessionDelta := sessionutils.ExtractStateDeltas(event.Actions.StateDelta)
		s.updateAppState(appDelta, curSession.AppName())
		s.updateUserState(userDelta, curSession.AppName(), curSession.UserID())
		maps.Copy(stored_session.state, sessionDelta)
	}

	// curSession changes only now that the stored session has, and under
	// s.mu, so concurrent appends through one value reach it in the order
	// they reached storage.
	if err := commitToHeldSession(curSession, trimmedEvent, event.Actions.StateDelta, event.Timestamp); err != nil {
		return fmt.Errorf("event %q is stored in session %q, but the session value was not updated: %w", event.ID, key.sessionID, err)
	}
	return nil
}

// commitToHeldSession updates the session value a caller passed to
// AppendEvent after the append has been stored. A value the service or
// CloneSession made takes the trimmed event, the whole state delta with its
// temp: keys, and the update time. A value the caller built can only take the
// state delta, through its State, and the first error its State.Set returns is
// returned.
func commitToHeldSession(curSession Session, trimmedEvent *Event, stateDelta map[string]any, updatedAt time.Time) error {
	if ms, ok := curSession.(sessionutils.MutableSession); ok {
		ms.CommitAppend(trimmedEvent, stateDelta, updatedAt)
		return nil
	}
	st := curSession.State()
	if st == nil {
		return nil
	}
	for k, v := range stateDelta {
		if err := st.Set(k, v); err != nil {
			return err
		}
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
	s.mu.RLock()
	defer s.mu.RUnlock()
	return events(s.events)
}

func (s *session) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.updatedAt
}

// CommitAppend implements [sessionutils.MutableSession]. trimmedEvent must be
// a *Event, and it is appended as is: s holds the caller's event, or its copy
// without temp: keys, as adk-python's session holds the event object the
// runner yields, so a change the caller makes to that event later shows in s.
// rawStateDelta is applied to s's state with its temp: keys, so that later
// readers of s within the invocation see them; storage never holds them.
func (s *session) CommitAppend(trimmedEvent any, rawStateDelta map[string]any, updatedAt time.Time) {
	ev := trimmedEvent.(*Event)

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(rawStateDelta) > 0 {
		if s.state == nil {
			s.state = make(map[string]any)
		}
		maps.Copy(s.state, rawStateDelta)
	}
	s.events = append(s.events, ev)
	s.updatedAt = updatedAt
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
	s.mu.RLock()
	// Create a copy of the state to iterate over it without holding the lock.
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

	s.state[key] = value
	return nil
}

// eventsEqual reports whether a and b are the same event field by field, as
// adk-python's model equality compares two events. Timestamps compare as
// instants, whatever their location or monotonic reading. An empty map or
// slice equals a nil one, because adk-python defaults those fields to empty
// values where a Go struct literal leaves them nil.
func eventsEqual(a, b *Event) bool {
	if !a.Timestamp.Equal(b.Timestamp) {
		return false
	}
	ac, bc := *a, *b
	ac.Timestamp, bc.Timestamp = time.Time{}, time.Time{}
	nilEmptyCollections(&ac)
	nilEmptyCollections(&bc)
	return reflect.DeepEqual(&ac, &bc)
}

// nilEmptyCollections sets every empty map or slice field of e that a default
// can leave either nil or empty to nil. e must be a copy the caller owns.
func nilEmptyCollections(e *Event) {
	e.Actions.StateDelta = nilIfEmptyMap(e.Actions.StateDelta)
	e.Actions.ArtifactDelta = nilIfEmptyMap(e.Actions.ArtifactDelta)
	e.Actions.RequestedToolConfirmations = nilIfEmptyMap(e.Actions.RequestedToolConfirmations)
	e.Actions.RequestedAuthConfigs = nilIfEmptyMap(e.Actions.RequestedAuthConfigs)
	e.Actions.RenderUIWidgets = nilIfEmptySlice(e.Actions.RenderUIWidgets)
	e.CustomMetadata = nilIfEmptyMap(e.CustomMetadata)
	e.LongRunningToolIDs = nilIfEmptySlice(e.LongRunningToolIDs)
	e.Routes = nilIfEmptySlice(e.Routes)
}

func nilIfEmptyMap[M ~map[K]V, K comparable, V any](m M) M {
	if len(m) == 0 {
		return nil
	}
	return m
}

func nilIfEmptySlice[S ~[]E, E any](s S) S {
	if len(s) == 0 {
		return nil
	}
	return s
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

	// If no keys were filtered out, return the original event without copying.
	if len(filteredStateDelta) == len(event.Actions.StateDelta) {
		return event
	}

	// Create a copy of the event to avoid mutating the original.
	eventCopy := *event
	eventCopy.Actions.StateDelta = filteredStateDelta

	return &eventCopy
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

var (
	_ Service                     = (*inMemoryService)(nil)
	_ UserStateReader             = (*inMemoryService)(nil)
	_ sessionutils.MutableSession = (*session)(nil)
)
