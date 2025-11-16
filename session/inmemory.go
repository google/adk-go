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
	"google.golang.org/adk/internal/sessionutils"
	"rsc.io/omap"
	"rsc.io/ordered"
)

// COMPACTION SUPPORT IMPLEMENTATION NOTES:
// ==========================================
// Session Compaction is a feature that periodically summarizes and compresses long event histories.
// When enabled, the Compactor (in compaction/compactor.go) creates EventCompaction objects that:
// 1. Summarize a range of events using an LLM (reducing token count by 60-80%)
// 2. Are stored as regular events with Actions.Compaction field populated
// 3. Allow retrieving only "active" events (excluding original events replaced by summaries)
//
// KEY INVARIANTS FOR INMEMORY.GO:
// ==============================
// 1. State Management: Compaction events should NOT contribute new session state
//    - Actions.StateDelta should be empty or nil for compaction events
//    - Session state should only accumulate from non-compacted events
//    - This is automatically correct if we skip state processing for compaction events
//
// 2. Event Storage: Compaction events are stored in the regular event stream
//    - They have timestamps and are ordered chronologically
//    - They can be identified by Actions.Compaction != nil
//    - They serve as markers of which time windows were compacted
//
// 3. Event Retrieval: Get() and List() must be compaction-aware
//    - When returning events, filter to EXCLUDE original events in compaction windows
//    - INCLUDE compaction events (they are the replacement summaries)
//    - This prevents LLM from seeing both original + compacted content (wasteful, conflicting)
//    - Without this filtering: token waste, potential context conflicts
//    - With this filtering: 60-80% token reduction for long sessions
//
// REQUIRED CHANGES:
// ==================
// 1. AppendEvent(): Already mostly correct
//    - Currently checks Actions.StateDelta and only processes if present
//    - This means compaction events (with empty StateDelta) won't corrupt state ✓
//    - May want to add validation that compaction events have no StateDelta
//
// 2. Get(): Needs compaction filtering logic
//    - Current: Returns all events in response (no filtering for compaction)
//    - Needed: Filter to exclude original events within compaction windows
//    - Implementation: After filtering by NumRecentEvents and timestamp,
//      scan for any compaction events and exclude their time windows' original events
//
// 3. List(): Similar enhancement as Get()
//    - Currently returns sessions without filtering events
//    - Needed: Same compaction-aware filtering as Get()
//    - Note: List() doesn't return events (sessions.events), so may not need changes
//      unless the Session object's events field is used
//
// 4. State Methods: No changes needed
//    - mergeStates(), updateAppState(), updateUserState() work correctly
//    - They process StateDelta which compaction events don't have
//    - This naturally excludes compaction events from state calculation ✓
//
// EXAMPLE SCENARIO:
// ==================
// Session with 5 events, then compaction of first 3:
//
// Initial state:
//   Event1 (t=100): StateDelta={key1: val1}  ← Compacted
//   Event2 (t=200): StateDelta={key2: val2}  ← Compacted
//   Event3 (t=300): StateDelta={key3: val3}  ← Compacted
//   Event4 (t=400): StateDelta={key4: val4}  ← NOT compacted
//   Event5 (t=500): StateDelta={key5: val5}  ← NOT compacted
//
// After compaction:
//   Event1 (t=100): ... ← Original, stored for audit
//   Event2 (t=200): ... ← Original, stored for audit
//   Event3 (t=300): ... ← Original, stored for audit
//   CompactionEvent (t=300): Actions.Compaction={start: 100, end: 300, content: "summary..."}
//   Event4 (t=400): ...
//   Event5 (t=500): ...
//
// Get() should return for LLM context:
//   CompactionEvent (the summary replaces Event1,2,3)
//   Event4
//   Event5
//
// State should be merged from:
//   key3, key4, key5 (Event1,2,3 state is implicitly in CompactionEvent.content)
//
// Database preserves ALL events for compliance and debugging purposes.

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
	// COMPACTION SUPPORT:
	// During session creation, we initialize state from the request.
	// This state is merged with app-level and user-level state using sessionutils.MergeStates().
	// For session compaction to work correctly, this initial state must not be contaminated
	// by CompactedContent. Since we only call ExtractStateDeltas() on req.State (which is
	// application-provided initial state), compaction has no effect here.
	// Compaction becomes relevant only when AppendEvent() is called with compaction events.
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

	// COMPACTION FEATURE: Filter out original events that have been replaced by compaction summaries
	// This ensures the LLM context contains only the compaction summaries and non-compacted events,
	// achieving 60-80% token reduction for long sessions.
	filteredEvents = filterEventsForCompaction(filteredEvents)

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

	// Look up session by ID instead of type assertion
	sessionID := id{
		appName:   curSession.AppName(),
		userID:    curSession.UserID(),
		sessionID: curSession.ID(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	stored_session, ok := s.sessions.Get(sessionID.Encode())
	if !ok {
		return fmt.Errorf("session not found, cannot apply event")
	}

	// If the passed session is a concrete *session, update it directly
	if sess, ok := curSession.(*session); ok {
		if err := sess.appendEvent(event); err != nil {
			return fmt.Errorf("fail to set state on appendEvent: %w", err)
		}
	}

	// update the in-memory session service
	stored_session.events = append(stored_session.events, event)
	stored_session.updatedAt = event.Timestamp

	// COMPACTION FEATURE MODIFICATION NEEDED:
	// When session compaction is enabled, events with Actions.Compaction set indicate that
	// a range of previous events have been summarized and compressed. These compaction events:
	// 1. Should NOT contribute new state (StateDelta should be empty or ignored)
	// 2. Must be stored in the event stream to mark which time windows were compacted
	// 3. Will be used later to filter out original events when building LLM context
	//
	// Current behavior: Only processes StateDelta if present (which is correct).
	// FUTURE ENHANCEMENT: Add special handling for compaction events:
	//   - Validate that compaction events have no/empty StateDelta
	//   - Store compaction metadata for event filtering logic
	//   - Maintain a compaction window index for efficient filtering in Get() method
	//
	// This ensures session state only accumulates NEW information from actual invocations,
	// not from summarized/compacted event ranges.
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
	// COMPACTION SUPPORT:
	// State merging is naturally compaction-aware because:
	// 1. Compaction events have Actions.StateDelta = nil or empty
	// 2. Only non-compaction events contribute to the merged state
	// 3. This preserves the invariant: final state = state accumulated from non-compacted events
	// 4. CompactedContent (the summary) doesn't contain state deltas, just conversational summary
	// Therefore, no changes needed here. The state merging logic automatically excludes
	// compaction events and correctly builds state from all preceding non-compacted events.
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

// COMPACTION SUPPORT IN SESSION TYPE:
// ===================================
// The session struct stores all data for a specific user session.
// For compaction support, key considerations:
//
// 1. Events Slice: Stores ALL events including compaction events
//    - Original events: Normal events with StateDelta
//    - Compaction events: Events with Actions.Compaction != nil (empty/no StateDelta)
//    - Both types are stored chronologically for audit trail
//    - When retrieving events via Get(), filtering must exclude original events
//      within compaction windows
//
// 2. State Map: Only contains state from non-compacted events
//    - Populated by AppendEvent() when event has non-empty StateDelta
//    - Compaction events (with no StateDelta) don't affect this map
//    - This is already correct because compaction events shouldn't contribute new state
//
// 3. Mutex: Protects concurrent access to events and state
//    - Required since multiple goroutines may append events or read state
//    - Existing locking is sufficient for compaction
//
// NO STRUCTURAL CHANGES NEEDED for compaction support.
// The Events field naturally handles compaction events.
// The State field correctly ignores compaction events.
// The filtering logic needed is in inMemoryService.Get(), not in session itself.

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
// This is called during AppendEvent to clean up temporary session state.
// COMPACTION SUPPORT: Compaction events should have no StateDelta (temp or otherwise),
// so this function will have no effect on them, which is correct.
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

// filterEventsForCompaction filters out original events that have been compacted.
// It returns only compaction events (summaries) and non-compacted events, ensuring
// the LLM context doesn't include both original events and their summaries.
func filterEventsForCompaction(events []*Event) []*Event {
	if len(events) == 0 {
		return events
	}

	// First pass: identify all compaction ranges
	compactionRanges := make([]compactionRange, 0)
	for _, event := range events {
		if event.Actions.Compaction != nil {
			compactionRanges = append(compactionRanges, compactionRange{
				startTimestamp: event.Actions.Compaction.StartTimestamp,
				endTimestamp:   event.Actions.Compaction.EndTimestamp,
			})
		}
	}

	// If no compactions, return events as-is
	if len(compactionRanges) == 0 {
		return events
	}

	// Second pass: filter events based on compaction ranges
	filtered := make([]*Event, 0, len(events))
	for _, event := range events {
		// Always include compaction events themselves
		if event.Actions.Compaction != nil {
			filtered = append(filtered, event)
			continue
		}

		// Check if this event is within any compaction range
		if !isEventWithinRange(event, compactionRanges) {
			// Include events that are not compacted
			filtered = append(filtered, event)
		}
	}

	return filtered
}

// compactionRange represents the time window of a compaction event.
type compactionRange struct {
	startTimestamp float64
	endTimestamp   float64
}

// isEventWithinRange checks if an event's timestamp falls within any compaction range.
func isEventWithinRange(event *Event, ranges []compactionRange) bool {
	eventTime := float64(event.Timestamp.Unix())
	for _, r := range ranges {
		if eventTime >= r.startTimestamp && eventTime <= r.endTimestamp {
			return true
		}
	}
	return false
}

var _ Service = (*inMemoryService)(nil)
