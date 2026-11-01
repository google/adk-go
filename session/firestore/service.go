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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"cloud.google.com/go/firestore"
	"golang.org/x/sync/errgroup"
	"google.golang.org/api/iterator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// Default Firestore collection names, matching adk-python's
// FirestoreSessionService so sessions can be shared across runtimes.
const (
	DefaultRootCollection      = "adk-session"
	DefaultSessionsCollection  = "sessions"
	DefaultEventsCollection    = "events"
	DefaultAppStateCollection  = "app_states"
	DefaultUserStateCollection = "user_states"

	rootCollectionEnvVar      = "ADK_FIRESTORE_ROOT_COLLECTION"
	sessionStatusDeleting     = "DELETING"
	maxFirestoreDocumentBytes = 1 << 20                           // 1 MiB Firestore document limit
	maxFirestorePayloadBytes  = maxFirestoreDocumentBytes - 65536 // 960 KiB payload cap (64 KiB headroom for document metadata)
	maxIdentityComponentBytes = 1500                              // Firestore maximum document ID byte length
)

// ErrStaleSession reports that the session document in Firestore has been
// modified since the caller's [session.Session] handle was loaded or created.
// Callers must reload the session with [session.Service.Get] before appending
// further events.
var ErrStaleSession = errors.New("stale session: the session has been modified in storage since it was loaded")

// Config configures the Firestore [session.Service] constructed by
// [NewSessionService].
type Config struct {
	// RootCollection is the top-level collection under which per-app user and
	// session subcollections are stored. If empty, NewSessionService reads the
	// ADK_FIRESTORE_ROOT_COLLECTION environment variable and falls back to
	// DefaultRootCollection ("adk-session").
	RootCollection string

	// AppStateCollection is the top-level collection storing app-scoped ("app:")
	// state documents. If empty, it defaults to DefaultAppStateCollection ("app_states").
	AppStateCollection string

	// UserStateCollection is the top-level collection storing user-scoped ("user:")
	// state documents. If empty, it defaults to DefaultUserStateCollection ("user_states").
	UserStateCollection string

	// TransactionMaxAttempts is the maximum number of attempts for each
	// Firestore transaction in Create, AppendEvent, and Delete. If zero or
	// negative, the Firestore client default (firestore.DefaultTransactionMaxAttempts,
	// 5) is used.
	TransactionMaxAttempts int
}

type sessionLockKey struct {
	appName   string
	userID    string
	sessionID string
}

type refCountedLock struct {
	ch   chan struct{}
	refs int
}

type firestoreService struct {
	client                 *firestore.Client
	rootCollection         string
	sessionsCollection     string
	eventsCollection       string
	appStateCollection     string
	userStateCollection    string
	transactionMaxAttempts int

	locksMu      sync.Mutex
	sessionLocks map[sessionLockKey]*refCountedLock
}

// NewSessionService returns a [session.Service] backed by the provided
// Firestore client. The returned service is safe for concurrent use by
// multiple goroutines.
//
// It returns an error if client is nil or if any collection name in cfg is
// invalid as a Firestore path segment. The caller retains ownership of client
// and is responsible for closing it when no longer needed.
func NewSessionService(client *firestore.Client, cfg Config) (session.Service, error) {
	if client == nil {
		return nil, errors.New("firestore: client must not be nil")
	}

	rootColl := cfg.RootCollection
	if rootColl == "" {
		rootColl = os.Getenv(rootCollectionEnvVar)
	}
	if rootColl == "" {
		rootColl = DefaultRootCollection
	}

	appColl := cfg.AppStateCollection
	if appColl == "" {
		appColl = DefaultAppStateCollection
	}

	userColl := cfg.UserStateCollection
	if userColl == "" {
		userColl = DefaultUserStateCollection
	}

	if err := validateIdentityComponent("root_collection", rootColl); err != nil {
		return nil, err
	}
	if err := validateIdentityComponent("app_state_collection", appColl); err != nil {
		return nil, err
	}
	if err := validateIdentityComponent("user_state_collection", userColl); err != nil {
		return nil, err
	}

	return &firestoreService{
		client:                 client,
		rootCollection:         rootColl,
		sessionsCollection:     DefaultSessionsCollection,
		eventsCollection:       DefaultEventsCollection,
		appStateCollection:     appColl,
		userStateCollection:    userColl,
		transactionMaxAttempts: cfg.TransactionMaxAttempts,
		sessionLocks:           make(map[sessionLockKey]*refCountedLock),
	}, nil
}

func (s *firestoreService) runTransaction(ctx context.Context, fn func(context.Context, *firestore.Transaction) error) error {
	if s.transactionMaxAttempts > 0 {
		return s.client.RunTransaction(ctx, fn, firestore.MaxAttempts(s.transactionMaxAttempts))
	}
	return s.client.RunTransaction(ctx, fn)
}

// withSessionLock serializes AppendEvent calls for the same (appName, userID, sessionID)
// within this process, matching adk-python's _with_session_lock while respecting
// context cancellation while waiting for the lock.
func (s *firestoreService) withSessionLock(ctx context.Context, appName, userID, sessionID string, fn func() error) error {
	key := sessionLockKey{appName: appName, userID: userID, sessionID: sessionID}

	s.locksMu.Lock()
	entry, ok := s.sessionLocks[key]
	if !ok {
		entry = &refCountedLock{ch: make(chan struct{}, 1)}
		entry.ch <- struct{}{}
		s.sessionLocks[key] = entry
	}
	entry.refs++
	s.locksMu.Unlock()

	releaseRef := func() {
		s.locksMu.Lock()
		entry.refs--
		if entry.refs <= 0 {
			delete(s.sessionLocks, key)
		}
		s.locksMu.Unlock()
	}

	select {
	case <-entry.ch:
	case <-ctx.Done():
		releaseRef()
		return ctx.Err()
	}

	defer func() {
		entry.ch <- struct{}{}
		releaseRef()
	}()

	return fn()
}

func (s *firestoreService) sessionsRef(appName, userID string) (*firestore.CollectionRef, error) {
	if err := validateIdentityComponent("app_name", appName); err != nil {
		return nil, err
	}
	if err := validateIdentityComponent("user_id", userID); err != nil {
		return nil, err
	}
	return s.client.Collection(s.rootCollection).
		Doc(appName).
		Collection("users").
		Doc(userID).
		Collection(s.sessionsCollection), nil
}

func (s *firestoreService) sessionDocRef(appName, userID, sessionID string) (*firestore.DocumentRef, error) {
	if err := validateIdentityComponent("session_id", sessionID); err != nil {
		return nil, err
	}
	coll, err := s.sessionsRef(appName, userID)
	if err != nil {
		return nil, err
	}
	return coll.Doc(sessionID), nil
}

func (s *firestoreService) appStateDocRef(appName string) *firestore.DocumentRef {
	return s.client.Collection(s.appStateCollection).Doc(appName)
}

func (s *firestoreService) userStateDocRef(appName, userID string) *firestore.DocumentRef {
	return s.client.Collection(s.userStateCollection).Doc(appName).Collection("users").Doc(userID)
}

// validateIdentityComponent rejects an identifier that is empty, alters the
// Firestore document path with "/", is "." or "..", or matches the
// Firestore-reserved "__.*__" pattern, matching adk-python's
// _validate_identity_component.
func validateIdentityComponent(name, value string) error {
	if value == "" {
		return fmt.Errorf("firestore: %s must be a non-empty string", name)
	}
	if len(value) > maxIdentityComponentBytes {
		return fmt.Errorf("firestore: %s length (%d bytes) exceeds Firestore %d-byte limit", name, len(value), maxIdentityComponentBytes)
	}
	if strings.Contains(value, "/") {
		return fmt.Errorf("firestore: %s must not contain '/': %q would alter the Firestore document path", name, value)
	}
	if value == "." || value == ".." {
		return fmt.Errorf("firestore: %s must not be '.' or '..', got %q", name, value)
	}
	if strings.HasPrefix(value, "__") && strings.HasSuffix(value, "__") {
		return fmt.Errorf("firestore: %s must not match the Firestore-reserved __.*__ pattern, got %q", name, value)
	}
	return nil
}

// Create creates a new session document in Firestore and applies any initial
// "app:" or "user:" scoped state within a single transaction.
func (s *firestoreService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req == nil {
		return nil, errors.New("firestore: create request must not be nil")
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = platform.NewUUID(ctx)
	}

	sessionRef, err := s.sessionDocRef(req.AppName, req.UserID, sessionID)
	if err != nil {
		return nil, err
	}

	appDelta, userDelta, sessionState := extractStateDeltas(req.State)
	sessionStateJSON, err := marshalSessionState(sessionState)
	if err != nil {
		return nil, fmt.Errorf("firestore: marshaling initial session state: %w", err)
	}

	appRef := s.appStateDocRef(req.AppName)
	userRef := s.userStateDocRef(req.AppName, req.UserID)

	// Stamp createTime and updateTime from platform.Now(ctx).UTC() truncated to
	// microsecond precision (Cloud Firestore's native timestamp resolution)
	// rather than firestore.ServerTimestamp (adk-python
	// firestore_session_service.py:255-256) so the returned
	// Session.LastUpdateTime() matches what Get and List read back, consistent
	// with session.InMemoryService and session/database.
	now := platform.Now(ctx).UTC().Truncate(time.Microsecond)
	sessionData := map[string]any{
		"id":         sessionID,
		"appName":    req.AppName,
		"userId":     req.UserID,
		"state":      sessionStateJSON,
		"createTime": now,
		"updateTime": now,
		"revision":   int64(0),
	}

	var currentApp, currentUser map[string]any
	err = s.runTransaction(ctx, func(_ context.Context, tx *firestore.Transaction) error {
		snap, err := tx.Get(sessionRef)
		if err != nil && status.Code(err) != codes.NotFound {
			return fmt.Errorf("reading session %q: %w", sessionID, err)
		}
		if snap != nil && snap.Exists() {
			return fmt.Errorf("session %q already exists", sessionID)
		}

		appSnap, err := tx.Get(appRef)
		if err != nil && status.Code(err) != codes.NotFound {
			return fmt.Errorf("reading app state for %q: %w", req.AppName, err)
		}
		userSnap, err := tx.Get(userRef)
		if err != nil && status.Code(err) != codes.NotFound {
			return fmt.Errorf("reading user state for %q: %w", req.UserID, err)
		}

		currentApp = snapshotMapOrEmpty(appSnap)
		currentUser = snapshotMapOrEmpty(userSnap)

		// Documents are written whole without firestore.MergeAll: a merge write
		// deep-merges nested maps and would keep keys that the new value dropped
		// (matching adk-python firestore_session_service.py:279-288).
		if len(appDelta) > 0 {
			maps.Copy(currentApp, appDelta)
			if err := tx.Set(appRef, currentApp); err != nil {
				return fmt.Errorf("writing app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			maps.Copy(currentUser, userDelta)
			if err := tx.Set(userRef, currentUser); err != nil {
				return fmt.Errorf("writing user state: %w", err)
			}
		}

		if err := tx.Set(sessionRef, sessionData); err != nil {
			return fmt.Errorf("writing session %q: %w", sessionID, err)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("firestore: create session: %w", err)
	}

	mergedState := mergeStates(currentApp, currentUser, sessionState)
	return &session.CreateResponse{
		Session: &localSession{
			appName:   req.AppName,
			userID:    req.UserID,
			sessionID: sessionID,
			state:     mergedState,
			events:    make([]*session.Event, 0),
			updatedAt: now,
			revision:  0,
		},
	}, nil
}

// Get retrieves a session and its events from Firestore, applying optional
// After and NumRecentEvents filters. If the session does not exist, Get returns
// an error wrapping [session.ErrNotFound].
func (s *firestoreService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req == nil {
		return nil, errors.New("firestore: get request must not be nil")
	}

	sessionRef, err := s.sessionDocRef(req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return nil, err
	}

	doc, err := sessionRef.Get(ctx)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return nil, fmt.Errorf("%w: %q: %w", session.ErrNotFound, req.SessionID, err)
		}
		return nil, fmt.Errorf("firestore: fetching session %q: %w", req.SessionID, err)
	}
	if !doc.Exists() {
		return nil, fmt.Errorf("%w: %q", session.ErrNotFound, req.SessionID)
	}

	data := doc.Data()
	if len(data) == 0 {
		return nil, fmt.Errorf("%w: %q", session.ErrNotFound, req.SessionID)
	}
	if statusStr, _ := data["status"].(string); statusStr == sessionStatusDeleting {
		return nil, fmt.Errorf("%w: %q is being deleted", session.ErrNotFound, req.SessionID)
	}

	appRef := s.appStateDocRef(req.AppName)
	userRef := s.userStateDocRef(req.AppName, req.UserID)

	eventsRef := sessionRef.Collection(s.eventsCollection)
	query := eventsRef.OrderBy("timestamp", firestore.Asc)
	if !req.After.IsZero() {
		// Stored event timestamps are UTC at microsecond precision; converting
		// req.After to UTC and truncating to microseconds matches Firestore's
		// timestamp resolution.
		query = query.Where("timestamp", ">=", req.After.UTC().Truncate(time.Microsecond))
	}
	// In Go's session.GetRequest, NumRecentEvents == 0 means the filter is not
	// applied (session/service.go:110-112), whereas adk-python uses
	// Optional[int] = None (firestore_session_service.py:347).
	if req.NumRecentEvents > 0 {
		query = query.LimitToLast(req.NumRecentEvents)
	}

	var (
		eventDocs []*firestore.DocumentSnapshot
		appDoc    *firestore.DocumentSnapshot
		userDoc   *firestore.DocumentSnapshot
	)

	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		docs, qErr := query.Documents(gctx).GetAll()
		if qErr != nil {
			return fmt.Errorf("fetching events for session %q: %w", req.SessionID, qErr)
		}
		eventDocs = docs
		return nil
	})
	g.Go(func() error {
		snap, aErr := appRef.Get(gctx)
		if aErr != nil && status.Code(aErr) != codes.NotFound {
			return fmt.Errorf("fetching app state for %q: %w", req.AppName, aErr)
		}
		appDoc = snap
		return nil
	})
	g.Go(func() error {
		snap, uErr := userRef.Get(gctx)
		if uErr != nil && status.Code(uErr) != codes.NotFound {
			return fmt.Errorf("fetching user state for %q: %w", req.UserID, uErr)
		}
		userDoc = snap
		return nil
	})
	if err := g.Wait(); err != nil {
		return nil, fmt.Errorf("firestore: get session: %w", err)
	}

	type indexedEvent struct {
		ev       *session.Event
		revision int64
		docID    string
	}
	indexedEvents := make([]indexedEvent, 0, len(eventDocs))
	for _, evDoc := range eventDocs {
		evData := evDoc.Data()
		rawEvent, ok := evData["event_data"]
		if !ok || rawEvent == nil {
			continue
		}
		ev, err := decodeEventData(rawEvent)
		if err != nil {
			return nil, fmt.Errorf("firestore: decoding event %q in session %q: %w", evDoc.Ref.ID, req.SessionID, err)
		}
		indexedEvents = append(indexedEvents, indexedEvent{
			ev:       ev,
			revision: parseRevision(evData["revision"]),
			docID:    evDoc.Ref.ID,
		})
	}
	// Tie-break events that share the same timestamp by their session revision
	// (when present) and then by document ID, preserving insertion order even
	// when event IDs are random UUIDs.
	slices.SortStableFunc(indexedEvents, func(a, b indexedEvent) int {
		if c := a.ev.Timestamp.Compare(b.ev.Timestamp); c != 0 {
			return c
		}
		if a.revision > 0 && b.revision > 0 && a.revision != b.revision {
			return cmp.Compare(a.revision, b.revision)
		}
		return cmp.Compare(a.docID, b.docID)
	})
	loadedEvents := make([]*session.Event, 0, len(indexedEvents))
	for _, item := range indexedEvents {
		loadedEvents = append(loadedEvents, item.ev)
	}

	sessionState, err := decodeSessionState(data["state"])
	if err != nil {
		return nil, fmt.Errorf("firestore: decoding state for session %q: %w", req.SessionID, err)
	}
	appState := snapshotMapOrEmpty(appDoc)
	userState := snapshotMapOrEmpty(userDoc)
	mergedState := mergeStates(appState, userState, sessionState)

	return &session.GetResponse{
		Session: &localSession{
			appName:   req.AppName,
			userID:    req.UserID,
			sessionID: req.SessionID,
			state:     mergedState,
			events:    loadedEvents,
			updatedAt: parseUpdateTime(data["updateTime"]),
			revision:  parseRevision(data["revision"]),
		},
	}, nil
}

// List returns all sessions matching req.AppName and, when non-empty,
// req.UserID.
//
// In Go's [session.ListRequest], an empty UserID represents listing sessions
// across all users of the application (the equivalent of user_id=None in
// adk-python's list_sessions, firestore_session_service.py:410-414), executed
// via a collection-group query on the sessions collection filtered by appName.
func (s *firestoreService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req == nil {
		return nil, errors.New("firestore: list request must not be nil")
	}
	if err := validateIdentityComponent("app_name", req.AppName); err != nil {
		return nil, err
	}

	var query firestore.Query
	if req.UserID != "" {
		coll, err := s.sessionsRef(req.AppName, req.UserID)
		if err != nil {
			return nil, err
		}
		query = coll.Where("appName", "==", req.AppName)
	} else {
		query = s.client.CollectionGroup(s.sessionsCollection).Where("appName", "==", req.AppName)
	}

	docs, err := query.Documents(ctx).GetAll()
	if err != nil {
		return nil, fmt.Errorf("firestore: listing sessions for app %q: %w", req.AppName, err)
	}

	var sessionDocs []map[string]any
	for _, doc := range docs {
		d := doc.Data()
		if len(d) == 0 {
			continue
		}
		if statusStr, _ := d["status"].(string); statusStr == sessionStatusDeleting {
			continue
		}
		sessionDocs = append(sessionDocs, d)
	}

	appSnap, err := s.appStateDocRef(req.AppName).Get(ctx)
	if err != nil && status.Code(err) != codes.NotFound {
		return nil, fmt.Errorf("firestore: fetching app state for %q: %w", req.AppName, err)
	}
	appState := snapshotMapOrEmpty(appSnap)

	userStatesMap := make(map[string]map[string]any)
	if req.UserID != "" {
		userSnap, err := s.userStateDocRef(req.AppName, req.UserID).Get(ctx)
		if err != nil && status.Code(err) != codes.NotFound {
			return nil, fmt.Errorf("firestore: fetching user state for %q: %w", req.UserID, err)
		}
		if userSnap != nil && userSnap.Exists() {
			userStatesMap[req.UserID] = snapshotMapOrEmpty(userSnap)
		}
	} else {
		uniqueUsers := make(map[string]struct{})
		for _, sd := range sessionDocs {
			if uid, ok := sd["userId"].(string); ok && uid != "" {
				uniqueUsers[uid] = struct{}{}
			}
		}
		if len(uniqueUsers) > 0 {
			sortedUIDs := make([]string, 0, len(uniqueUsers))
			for uid := range uniqueUsers {
				sortedUIDs = append(sortedUIDs, uid)
			}
			slices.Sort(sortedUIDs)

			refs := make([]*firestore.DocumentRef, 0, len(sortedUIDs))
			for _, uid := range sortedUIDs {
				refs = append(refs, s.userStateDocRef(req.AppName, uid))
			}
			userSnaps, err := s.client.GetAll(ctx, refs)
			if err != nil {
				return nil, fmt.Errorf("firestore: batch fetching user states for app %q: %w", req.AppName, err)
			}
			for _, uSnap := range userSnaps {
				if uSnap != nil && uSnap.Exists() {
					userStatesMap[uSnap.Ref.ID] = snapshotMapOrEmpty(uSnap)
				}
			}
		}
	}

	sessions := make([]session.Session, 0, len(sessionDocs))
	for _, sd := range sessionDocs {
		sid, _ := sd["id"].(string)
		appName, _ := sd["appName"].(string)
		uid, _ := sd["userId"].(string)

		sState, err := decodeSessionState(sd["state"])
		if err != nil {
			return nil, fmt.Errorf("firestore: decoding state for session %q: %w", sid, err)
		}
		uState := userStatesMap[uid]
		merged := mergeStates(appState, uState, sState)

		sessions = append(sessions, &localSession{
			appName:   appName,
			userID:    uid,
			sessionID: sid,
			state:     merged,
			events:    make([]*session.Event, 0),
			updatedAt: parseUpdateTime(sd["updateTime"]),
			revision:  parseRevision(sd["revision"]),
		})
	}

	// Sort by (LastUpdateTime, UserID, ID), matching adk-python
	// firestore_session_service.py:477.
	slices.SortFunc(sessions, func(a, b session.Session) int {
		if c := a.LastUpdateTime().Compare(b.LastUpdateTime()); c != 0 {
			return c
		}
		if c := cmp.Compare(a.UserID(), b.UserID()); c != 0 {
			return c
		}
		return cmp.Compare(a.ID(), b.ID())
	})

	return &session.ListResponse{
		Sessions: sessions,
	}, nil
}

// Delete removes a session and all of its event documents from Firestore.
// Deleting a session that does not exist is a no-op and returns nil.
func (s *firestoreService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req == nil {
		return errors.New("firestore: delete request must not be nil")
	}

	sessionRef, err := s.sessionDocRef(req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return err
	}

	// Mark the session as DELETING in a transaction first so concurrent
	// AppendEvent calls observe the status and abort cleanly (matching
	// adk-python firestore_session_service.py:503-515). Unlike adk-python,
	// which logs and swallows errors here, ADK Go returns non-NotFound errors
	// to the caller instead of proceeding with an unmarked session.
	if err := s.runTransaction(ctx, func(_ context.Context, tx *firestore.Transaction) error {
		snap, txErr := tx.Get(sessionRef)
		if txErr != nil {
			if status.Code(txErr) == codes.NotFound {
				return nil
			}
			return txErr
		}
		if snap.Exists() {
			return tx.Update(sessionRef, []firestore.Update{
				{Path: "status", Value: sessionStatusDeleting},
			})
		}
		return nil
	}); err != nil && status.Code(err) != codes.NotFound {
		return fmt.Errorf("firestore: marking session %q as deleting: %w", req.SessionID, err)
	}

	eventsRef := sessionRef.Collection(s.eventsCollection)
	it := eventsRef.Documents(ctx)
	defer it.Stop()

	bw := s.client.BulkWriter(ctx)
	var jobs []*firestore.BulkWriterJob
	for {
		doc, iterErr := it.Next()
		if errors.Is(iterErr, iterator.Done) {
			break
		}
		if iterErr != nil {
			bw.End()
			return fmt.Errorf("firestore: streaming events for deletion of session %q: %w", req.SessionID, iterErr)
		}
		job, bErr := bw.Delete(doc.Ref)
		if bErr != nil {
			bw.End()
			return fmt.Errorf("firestore: deleting event batch for session %q: %w", req.SessionID, bErr)
		}
		jobs = append(jobs, job)
	}
	bw.End()
	for _, job := range jobs {
		if _, jErr := job.Results(); jErr != nil {
			return fmt.Errorf("firestore: deleting event batch for session %q: %w", req.SessionID, jErr)
		}
	}

	if _, err := sessionRef.Delete(ctx); err != nil && status.Code(err) != codes.NotFound {
		return fmt.Errorf("firestore: deleting session document %q: %w", req.SessionID, err)
	}
	return nil
}

// AppendEvent appends an event to curSession and persists the event and state
// deltas in a single Firestore transaction.
func (s *firestoreService) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	if curSession == nil {
		return errors.New("firestore: session is nil")
	}
	if event == nil {
		return errors.New("firestore: event is nil")
	}

	sess, ok := curSession.(*localSession)
	if !ok {
		return fmt.Errorf("firestore: unexpected session type %T", curSession)
	}

	sessionRef, err := s.sessionDocRef(sess.AppName(), sess.UserID(), sess.ID())
	if err != nil {
		return err
	}

	if event.Partial {
		return nil
	}

	if event.ID == "" {
		event.ID = platform.NewUUID(ctx)
	}
	if err := validateIdentityComponent("event_id", event.ID); err != nil {
		return err
	}
	// Events constructed as struct literals without session.NewEvent arrive
	// with a zero Timestamp; assign platform.Now(ctx).UTC() so they do not sort
	// to year 0001 or rewind the session's LastUpdateTime. Truncate to
	// microsecond precision to match Cloud Firestore's Timestamp resolution and
	// session/database (session/database/service.go:385).
	if event.Timestamp.IsZero() {
		event.Timestamp = platform.Now(ctx).UTC()
	}
	event.Timestamp = event.Timestamp.Truncate(time.Microsecond)

	trimmedEvent := trimTempDeltaState(event)
	eventData, err := encodeEventData(trimmedEvent)
	if err != nil {
		return err
	}

	appUpdates, userUpdates, sessionUpdates := extractStateDeltas(event.Actions.StateDelta)
	appRef := s.appStateDocRef(sess.AppName())
	userRef := s.userStateDocRef(sess.AppName(), sess.UserID())
	eventRef := sessionRef.Collection(s.eventsCollection).Doc(event.ID)

	return s.withSessionLock(ctx, sess.AppName(), sess.UserID(), sess.ID(), func() error {
		expectedRev := sess.currentRevision()

		var newRevision int64
		err = s.runTransaction(ctx, func(_ context.Context, tx *firestore.Transaction) error {
			sessionSnap, txErr := tx.Get(sessionRef)
			if txErr != nil {
				if status.Code(txErr) == codes.NotFound {
					return fmt.Errorf("%w: %q: %w", session.ErrNotFound, sess.ID(), txErr)
				}
				return fmt.Errorf("reading session %q: %w", sess.ID(), txErr)
			}
			if !sessionSnap.Exists() {
				return fmt.Errorf("%w: %q", session.ErrNotFound, sess.ID())
			}

			sessionDoc := sessionSnap.Data()
			if statusStr, _ := sessionDoc["status"].(string); statusStr == sessionStatusDeleting {
				return fmt.Errorf("session %q is currently being deleted", sess.ID())
			}

			currentRevision := parseRevision(sessionDoc["revision"])
			if currentRevision != expectedRev {
				return fmt.Errorf("%w: session %q expected revision %d, stored revision is %d",
					ErrStaleSession, sess.ID(), expectedRev, currentRevision)
			}

			// Merge session-scoped deltas onto the stored session state inside the
			// transaction, matching session/database (service.go:489-491) rather than
			// adk-python's in-memory snapshot (firestore_session_service.py:567-576)
			// so uncommitted in-memory State().Set mutations do not leak to storage.
			currentSessionState, stateErr := decodeSessionState(sessionDoc["state"])
			if stateErr != nil {
				return fmt.Errorf("decoding stored state for session %q: %w", sess.ID(), stateErr)
			}
			maps.Copy(currentSessionState, sessionUpdates)
			sessionStateJSON, stateErr := marshalSessionState(currentSessionState)
			if stateErr != nil {
				return fmt.Errorf("marshaling session state: %w", stateErr)
			}

			var appSnap, userSnap *firestore.DocumentSnapshot
			if len(appUpdates) > 0 {
				appSnap, txErr = tx.Get(appRef)
				if txErr != nil && status.Code(txErr) != codes.NotFound {
					return fmt.Errorf("reading app state: %w", txErr)
				}
			}
			if len(userUpdates) > 0 {
				userSnap, txErr = tx.Get(userRef)
				if txErr != nil && status.Code(txErr) != codes.NotFound {
					return fmt.Errorf("reading user state: %w", txErr)
				}
			}

			if len(appUpdates) > 0 {
				currentApp := snapshotMapOrEmpty(appSnap)
				maps.Copy(currentApp, appUpdates)
				if err := tx.Set(appRef, currentApp); err != nil {
					return fmt.Errorf("writing app state: %w", err)
				}
			}
			if len(userUpdates) > 0 {
				currentUser := snapshotMapOrEmpty(userSnap)
				maps.Copy(currentUser, userUpdates)
				if err := tx.Set(userRef, currentUser); err != nil {
					return fmt.Errorf("writing user state: %w", err)
				}
			}

			newRevision = currentRevision + 1
			eventTimeUTC := event.Timestamp.UTC()
			if err := tx.Update(sessionRef, []firestore.Update{
				{Path: "state", Value: sessionStateJSON},
				{Path: "updateTime", Value: eventTimeUTC},
				{Path: "revision", Value: newRevision},
			}); err != nil {
				return fmt.Errorf("updating session %q: %w", sess.ID(), err)
			}

			if err := tx.Set(eventRef, map[string]any{
				"event_data":   eventData,
				"timestamp":    eventTimeUTC,
				"invocationId": event.InvocationID,
				"appName":      sess.AppName(),
				"userId":       sess.UserID(),
				"revision":     newRevision,
			}); err != nil {
				return fmt.Errorf("writing event %q: %w", event.ID, err)
			}

			return nil
		})
		if err != nil {
			return fmt.Errorf("firestore: append event: %w", err)
		}

		sess.commitAppend(event.Actions.StateDelta, trimmedEvent, newRevision)
		return nil
	})
}

func snapshotMapOrEmpty(snap *firestore.DocumentSnapshot) map[string]any {
	if snap == nil || !snap.Exists() {
		return make(map[string]any)
	}
	data := snap.Data()
	if data == nil {
		return make(map[string]any)
	}
	return data
}

func marshalSessionState(state map[string]any) (string, error) {
	if state == nil {
		state = make(map[string]any)
	}
	b, err := json.Marshal(state)
	if err != nil {
		return "", err
	}
	if len(b) > maxFirestorePayloadBytes {
		return "", fmt.Errorf("serialized session state (%d bytes) exceeds Firestore %d-byte payload limit", len(b), maxFirestorePayloadBytes)
	}
	return string(b), nil
}

func decodeSessionState(raw any) (map[string]any, error) {
	if raw == nil {
		return make(map[string]any), nil
	}
	switch v := raw.(type) {
	case string:
		if v == "" {
			return make(map[string]any), nil
		}
		var out map[string]any
		if err := json.Unmarshal([]byte(v), &out); err != nil {
			return nil, err
		}
		if out == nil {
			return make(map[string]any), nil
		}
		return out, nil
	case map[string]any:
		return maps.Clone(v), nil
	default:
		return nil, fmt.Errorf("unexpected session state type %T", raw)
	}
}

func encodeEventData(event *session.Event) (map[string]any, error) {
	b, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("firestore: marshaling event %q: %w", event.ID, err)
	}
	if len(b) > maxFirestorePayloadBytes {
		return nil, fmt.Errorf(
			"firestore: event %q serialized size (%d bytes) exceeds the %d-byte Firestore payload limit; store large binary payloads in artifact.Service and reference them via genai.FileData URIs",
			event.ID, len(b), maxFirestorePayloadBytes,
		)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("firestore: decoding marshaled event %q: %w", event.ID, err)
	}
	return out, nil
}

func decodeEventData(raw any) (*session.Event, error) {
	ev := &session.Event{}
	switch v := raw.(type) {
	case string:
		if err := json.Unmarshal([]byte(v), ev); err == nil {
			return ev, nil
		}
		// Fall back through map[string]any so float-formatted whole numbers
		// (such as 1.0 in artifactDelta or usageMetadata from cross-runtime
		// dumps) normalize to integer JSON tokens before struct decoding.
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil, err
		}
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, ev); err != nil {
			return nil, err
		}
		return ev, nil
	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, ev); err != nil {
			return nil, err
		}
		return ev, nil
	default:
		return nil, fmt.Errorf("unexpected event_data type %T", raw)
	}
}

func parseUpdateTime(raw any) time.Time {
	switch v := raw.(type) {
	case time.Time:
		return v
	case float64:
		micros := math.Round(v * 1e6)
		return time.UnixMicro(int64(micros)).UTC()
	case int64:
		return time.Unix(v, 0).UTC()
	case string:
		if t, err := time.Parse(time.RFC3339Nano, v); err == nil {
			return t
		}
	}
	return time.Time{}
}

func parseRevision(raw any) int64 {
	switch v := raw.(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case float64:
		return int64(v)
	}
	return 0
}

var _ session.Service = (*firestoreService)(nil)
