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
	"bytes"
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

	fs "cloud.google.com/go/firestore"
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
	nestedListKey             = "_adk_list"
	escapedDictKey            = "_adk_dict"
	goCompactionKey           = "adk_go_compaction"
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
	client                 *fs.Client
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
func NewSessionService(client *fs.Client, cfg Config) (session.Service, error) {
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

func (s *firestoreService) runTransaction(ctx context.Context, fn func(context.Context, *fs.Transaction) error) error {
	if s.transactionMaxAttempts > 0 {
		return s.client.RunTransaction(ctx, fn, fs.MaxAttempts(s.transactionMaxAttempts))
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

func (s *firestoreService) sessionsRef(appName, userID string) (*fs.CollectionRef, error) {
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

func (s *firestoreService) sessionDocRef(appName, userID, sessionID string) (*fs.DocumentRef, error) {
	if err := validateIdentityComponent("session_id", sessionID); err != nil {
		return nil, err
	}
	coll, err := s.sessionsRef(appName, userID)
	if err != nil {
		return nil, err
	}
	return coll.Doc(sessionID), nil
}

func (s *firestoreService) appStateDocRef(appName string) *fs.DocumentRef {
	return s.client.Collection(s.appStateCollection).Doc(appName)
}

func (s *firestoreService) userStateDocRef(appName, userID string) *fs.DocumentRef {
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
	// "__" and "___" are not reserved, as in adk-python
	// (firestore_session_service.py:254).
	if len(value) >= 4 && strings.HasPrefix(value, "__") && strings.HasSuffix(value, "__") {
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
	// firestore_session_service.py:273) so the returned
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
	err = s.runTransaction(ctx, func(_ context.Context, tx *fs.Transaction) error {
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
		// (matching adk-python firestore_session_service.py:324-332).
		if len(appDelta) > 0 {
			encodedApp, encErr := encodeFirestoreState(currentApp, appDelta)
			if encErr != nil {
				return fmt.Errorf("encoding app state: %w", encErr)
			}
			if err := tx.Set(appRef, encodedApp); err != nil {
				return fmt.Errorf("writing app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			encodedUser, encErr := encodeFirestoreState(currentUser, userDelta)
			if encErr != nil {
				return fmt.Errorf("encoding user state: %w", encErr)
			}
			if err := tx.Set(userRef, encodedUser); err != nil {
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

	maps.Copy(integersAsFloat64(currentApp), appDelta)
	maps.Copy(integersAsFloat64(currentUser), userDelta)
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
	query := eventsRef.OrderBy("timestamp", fs.Asc)
	if !req.After.IsZero() {
		// Stored event timestamps are UTC at microsecond precision; converting
		// req.After to UTC and truncating to microseconds matches Firestore's
		// timestamp resolution.
		query = query.Where("timestamp", ">=", req.After.UTC().Truncate(time.Microsecond))
	}

	var (
		eventDocs []*fs.DocumentSnapshot
		appDoc    *fs.DocumentSnapshot
		userDoc   *fs.DocumentSnapshot
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
	// when event IDs are random UUIDs. Apply NumRecentEvents after sorting so
	// timestamp ties are resolved by revision rather than Firestore's __name__
	// ordering, without requiring a composite index on (timestamp, revision) or
	// dropping adk-python events that omit the revision field.
	slices.SortStableFunc(indexedEvents, func(a, b indexedEvent) int {
		if c := a.ev.Timestamp.Compare(b.ev.Timestamp); c != 0 {
			return c
		}
		// adk-python events carry no revision and sort as revision 0. Comparing
		// revisions only when both are set would not be transitive.
		if c := cmp.Compare(a.revision, b.revision); c != 0 {
			return c
		}
		return cmp.Compare(a.docID, b.docID)
	})
	loadedEvents := make([]*session.Event, 0, len(indexedEvents))
	for _, item := range indexedEvents {
		loadedEvents = append(loadedEvents, item.ev)
	}
	if req.NumRecentEvents > 0 && len(loadedEvents) > req.NumRecentEvents {
		loadedEvents = loadedEvents[len(loadedEvents)-req.NumRecentEvents:]
	}

	sessionState, err := decodeSessionState(data["state"], false)
	if err != nil {
		return nil, fmt.Errorf("firestore: decoding state for session %q: %w", req.SessionID, err)
	}
	appState := integersAsFloat64(snapshotMapOrEmpty(appDoc))
	userState := integersAsFloat64(snapshotMapOrEmpty(userDoc))
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
// adk-python's list_sessions, firestore_session_service.py:449-463), executed
// via a collection-group query on the sessions collection filtered by appName.
func (s *firestoreService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req == nil {
		return nil, errors.New("firestore: list request must not be nil")
	}
	if err := validateIdentityComponent("app_name", req.AppName); err != nil {
		return nil, err
	}

	var query fs.Query
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
		if req.UserID == "" && !belongsToRootCollection(doc.Ref, s.rootCollection, req.AppName, s.sessionsCollection) {
			continue
		}
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
	appState := integersAsFloat64(snapshotMapOrEmpty(appSnap))

	userStatesMap := make(map[string]map[string]any)
	if req.UserID != "" {
		userSnap, err := s.userStateDocRef(req.AppName, req.UserID).Get(ctx)
		if err != nil && status.Code(err) != codes.NotFound {
			return nil, fmt.Errorf("firestore: fetching user state for %q: %w", req.UserID, err)
		}
		if userSnap != nil && userSnap.Exists() {
			userStatesMap[req.UserID] = integersAsFloat64(snapshotMapOrEmpty(userSnap))
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

			refs := make([]*fs.DocumentRef, 0, len(sortedUIDs))
			for _, uid := range sortedUIDs {
				refs = append(refs, s.userStateDocRef(req.AppName, uid))
			}
			userSnaps, err := s.client.GetAll(ctx, refs)
			if err != nil {
				return nil, fmt.Errorf("firestore: batch fetching user states for app %q: %w", req.AppName, err)
			}
			for _, uSnap := range userSnaps {
				if uSnap != nil && uSnap.Exists() {
					userStatesMap[uSnap.Ref.ID] = integersAsFloat64(snapshotMapOrEmpty(uSnap))
				}
			}
		}
	}

	sessions := make([]session.Session, 0, len(sessionDocs))
	for _, sd := range sessionDocs {
		sid, _ := sd["id"].(string)
		appName, _ := sd["appName"].(string)
		uid, _ := sd["userId"].(string)

		sState, err := decodeSessionState(sd["state"], false)
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
	// firestore_session_service.py:531.
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
	// adk-python firestore_session_service.py:557-563). Unlike adk-python,
	// which swallows errors here (lines 565-569), ADK Go returns non-NotFound errors
	// to the caller instead of proceeding with an unmarked session.
	if err := s.runTransaction(ctx, func(_ context.Context, tx *fs.Transaction) error {
		snap, txErr := tx.Get(sessionRef)
		if txErr != nil {
			if status.Code(txErr) == codes.NotFound {
				return nil
			}
			return txErr
		}
		if snap.Exists() {
			return tx.Update(sessionRef, []fs.Update{
				{Path: "status", Value: sessionStatusDeleting},
			})
		}
		return nil
	}); err != nil && status.Code(err) != codes.NotFound {
		return fmt.Errorf("firestore: marking session %q as deleting: %w", req.SessionID, err)
	}

	eventsRef := sessionRef.Collection(s.eventsCollection)
	// Select with no fields returns document names only.
	it := eventsRef.Select().Documents(ctx)
	defer it.Stop()

	bw := s.client.BulkWriter(ctx)
	var jobs []*fs.BulkWriterJob
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
	// session/database, which truncates the same way.
	if event.Timestamp.IsZero() {
		event.Timestamp = platform.Now(ctx).UTC()
	}
	event.Timestamp = event.Timestamp.Truncate(time.Microsecond)

	trimmedEvent := trimTempDeltaState(event)
	eventData, err := encodeEventData(trimmedEvent)
	if err != nil {
		return err
	}
	// Refuse an event that would not read back, such as one whose requested
	// auth config encodes but does not decode, as the in-memory, database and
	// Vertex AI services do. Stored, it would make every Get of the session
	// fail.
	if _, err := decodeEventData(eventData); err != nil {
		return fmt.Errorf("firestore: event %q would not read back from storage: %w", event.ID, err)
	}

	appUpdates, userUpdates, sessionUpdates := extractStateDeltas(event.Actions.StateDelta)
	appRef := s.appStateDocRef(sess.AppName())
	userRef := s.userStateDocRef(sess.AppName(), sess.UserID())
	eventRef := sessionRef.Collection(s.eventsCollection).Doc(event.ID)

	return s.withSessionLock(ctx, sess.AppName(), sess.UserID(), sess.ID(), func() error {
		expectedRev := sess.currentRevision()

		var newRevision int64
		appendTx := func(eventData any) error {
			return s.runTransaction(ctx, func(_ context.Context, tx *fs.Transaction) error {
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
					return fmt.Errorf("%w: session %q is being deleted", session.ErrNotFound, sess.ID())
				}

				currentRevision := parseRevision(sessionDoc["revision"])
				if currentRevision != expectedRev {
					return fmt.Errorf("%w: session %q expected revision %d, stored revision is %d",
						ErrStaleSession, sess.ID(), expectedRev, currentRevision)
				}

				// Merge session-scoped deltas onto the stored session state inside the
				// transaction, matching session/database rather than
				// adk-python's in-memory snapshot (firestore_session_service.py:675-681)
				// so uncommitted in-memory State().Set mutations do not leak to storage.
				// Numbers stay as written so the rewrite keeps the keys this event
				// does not touch exact, such as an integer above 2^53 that
				// adk-python stored.
				currentSessionState, stateErr := decodeSessionState(sessionDoc["state"], true)
				if stateErr != nil {
					return fmt.Errorf("decoding stored state for session %q: %w", sess.ID(), stateErr)
				}
				maps.Copy(currentSessionState, sessionUpdates)
				sessionStateJSON, stateErr := marshalSessionState(currentSessionState)
				if stateErr != nil {
					return fmt.Errorf("marshaling session state: %w", stateErr)
				}

				var appSnap, userSnap *fs.DocumentSnapshot
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
					encodedApp, encErr := encodeFirestoreState(currentApp, appUpdates)
					if encErr != nil {
						return fmt.Errorf("encoding app state: %w", encErr)
					}
					if err := tx.Set(appRef, encodedApp); err != nil {
						return fmt.Errorf("writing app state: %w", err)
					}
				}
				if len(userUpdates) > 0 {
					currentUser := snapshotMapOrEmpty(userSnap)
					encodedUser, encErr := encodeFirestoreState(currentUser, userUpdates)
					if encErr != nil {
						return fmt.Errorf("encoding user state: %w", encErr)
					}
					if err := tx.Set(userRef, encodedUser); err != nil {
						return fmt.Errorf("writing user state: %w", err)
					}
				}

				newRevision = currentRevision + 1
				eventTimeUTC := event.Timestamp.UTC()
				if err := tx.Update(sessionRef, []fs.Update{
					{Path: "state", Value: sessionStateJSON},
					{Path: "updateTime", Value: eventTimeUTC},
					{Path: "revision", Value: newRevision},
				}); err != nil {
					return fmt.Errorf("updating session %q: %w", sess.ID(), err)
				}

				if err := tx.Create(eventRef, map[string]any{
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
		}
		err := appendTx(eventData)
		// measureFirestoreValue does not model every Firestore limit: an
		// index entry's size depends on the document's name, and databases
		// differ on some shapes. A commit applies nothing when it fails, so
		// an event Firestore rejects as a map is appended as a string. When
		// the string is too large or would not read back, Firestore's error
		// stands.
		if m, isMap := eventData.(map[string]any); isMap && status.Code(err) == codes.InvalidArgument {
			if asString, strErr := eventDataString(event.ID, m); strErr == nil {
				if _, strErr = decodeEventData(asString); strErr == nil {
					err = appendTx(asString)
				}
			}
		}
		if err != nil {
			return fmt.Errorf("firestore: append event: %w", err)
		}

		sess.commitAppend(event.Actions.StateDelta, trimmedEvent, newRevision)
		return nil
	})
}

// belongsToRootCollection reports whether ref is a session document at
// rootCollection/appName/users/{userID}/sessions/{sessionID}, the only path
// [firestoreService.Get] reads. A collection-group query matches every
// sessions collection in the database, including those of services with
// another RootCollection, which adk-python's list_sessions returns too
// (firestore_session_service.py:463); ADK Go drops them so List only returns
// sessions Get can load.
func belongsToRootCollection(ref *fs.DocumentRef, rootCollection, appName, sessionsCollection string) bool {
	if ref == nil || ref.Parent == nil || ref.Parent.ID != sessionsCollection {
		return false
	}
	userDoc := ref.Parent.Parent
	if userDoc == nil || userDoc.Parent == nil || userDoc.Parent.ID != "users" {
		return false
	}
	appDoc := userDoc.Parent.Parent
	if appDoc == nil || appDoc.ID != appName || appDoc.Parent == nil || appDoc.Parent.ID != rootCollection {
		return false
	}
	return appDoc.Parent.Parent == nil
}

func snapshotMapOrEmpty(snap *fs.DocumentSnapshot) map[string]any {
	if snap == nil || !snap.Exists() {
		return make(map[string]any)
	}
	data := snap.Data()
	if data == nil {
		return make(map[string]any)
	}
	return decodeFirestoreMap(data)
}

// encodeFirestoreState returns the app or user state document that applies
// delta to stored, the decoded current document. Only delta goes through
// encoding/json, which turns Go values into the maps and slices
// encodeFirestoreMap wraps, and integers stay int64. Stored values keep their
// native Firestore types (integers, timestamps, bytes), which adk-python
// writes natively so they keep their rich types
// (firestore_session_service.py:604-606).
func encodeFirestoreState(stored, delta map[string]any) (map[string]any, error) {
	b, err := json.Marshal(delta)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var normalized map[string]any
	if err := dec.Decode(&normalized); err != nil {
		return nil, err
	}
	merged := maps.Clone(stored)
	if merged == nil {
		merged = make(map[string]any, len(normalized))
	}
	for k, v := range normalized {
		merged[k] = jsonNumbersToGo(v)
	}
	encoded := encodeFirestoreMap(merged)
	// A document Firestore would reject for another reason is left for it to
	// report, as adk-python does.
	if m := measureFirestoreDocument(encoded); m.ok && m.size > maxFirestorePayloadBytes {
		return nil, fmt.Errorf("state document (%d bytes) exceeds Firestore %d-byte payload limit", m.size, maxFirestorePayloadBytes)
	}
	return encoded, nil
}

// integersAsFloat64 replaces each int64 in the decoded state map m with a
// float64, in place, and returns m. Firestore keeps integers and doubles
// apart, but session-scoped state and the other session services return every
// number as a float64, so a caller reads an app or user state number the same
// way whichever runtime wrote it and whether or not it is whole.
func integersAsFloat64(m map[string]any) map[string]any {
	for k, v := range m {
		m[k] = integerAsFloat64(v)
	}
	return m
}

func integerAsFloat64(v any) any {
	switch x := v.(type) {
	case int64:
		return float64(x)
	case map[string]any:
		return integersAsFloat64(x)
	case []any:
		for i, e := range x {
			x[i] = integerAsFloat64(e)
		}
		return x
	default:
		return v
	}
}

// jsonNumbersToGo replaces each json.Number in v with an int64 when it is an
// integer and a float64 otherwise.
func jsonNumbersToGo(v any) any {
	switch x := v.(type) {
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return i
		}
		f, _ := x.Float64()
		return f
	case map[string]any:
		for k, e := range x {
			x[k] = jsonNumbersToGo(e)
		}
		return x
	case []any:
		for i, e := range x {
			x[i] = jsonNumbersToGo(e)
		}
		return x
	default:
		return v
	}
}

// encodeFirestoreMap and decodeFirestoreMap wrap an array that sits directly
// inside another array in {"_adk_list": [...]}, and escape a one-key map in an
// array whose key is "_adk_list" or "_adk_dict" in {"_adk_dict": ...}, so app
// and user state can hold nested arrays, which Firestore rejects. Only values
// inside an array are wrapped or unwrapped, matching adk-python's
// _encode_firestore_value / _decode_firestore_value
// (firestore_session_service.py:84-122) so either runtime reads the other's
// state documents unchanged.
func encodeFirestoreMap(data map[string]any) map[string]any {
	out := make(map[string]any, len(data))
	for k, v := range data {
		out[k] = encodeFirestoreValue(v, false)
	}
	return out
}

func encodeFirestoreValue(value any, inList bool) any {
	switch v := value.(type) {
	case []any:
		items := make([]any, len(v))
		for i, item := range v {
			items[i] = encodeFirestoreValue(item, true)
		}
		if inList {
			return map[string]any{nestedListKey: items}
		}
		return items
	case map[string]any:
		encoded := encodeFirestoreMap(v)
		if inList && len(encoded) == 1 {
			_, isList := encoded[nestedListKey]
			_, isDict := encoded[escapedDictKey]
			if isList || isDict {
				return map[string]any{escapedDictKey: encoded}
			}
		}
		return encoded
	default:
		return value
	}
}

func decodeFirestoreMap(data map[string]any) map[string]any {
	out := make(map[string]any, len(data))
	for k, v := range data {
		out[k] = decodeFirestoreValue(v, false)
	}
	return out
}

func decodeFirestoreValue(value any, inList bool) any {
	switch v := value.(type) {
	case []any:
		return decodeFirestoreList(v)
	case map[string]any:
		if inList && len(v) == 1 {
			if inner, ok := v[nestedListKey].([]any); ok {
				return decodeFirestoreList(inner)
			}
			if inner, ok := v[escapedDictKey].(map[string]any); ok {
				return decodeFirestoreMap(inner)
			}
		}
		return decodeFirestoreMap(v)
	default:
		return value
	}
}

func decodeFirestoreList(items []any) []any {
	out := make([]any, len(items))
	for i, item := range items {
		out[i] = decodeFirestoreValue(item, true)
	}
	return out
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

// decodeSessionState decodes a session document's state field, which is a
// JSON object string or, from older adk-python, a map. With exactNumbers a
// number keeps the form that marshals back as it was written, a json.Number
// from a string and an int64 or float64 from a map, and otherwise it is a
// float64.
func decodeSessionState(raw any, exactNumbers bool) (map[string]any, error) {
	if raw == nil {
		return make(map[string]any), nil
	}
	switch v := raw.(type) {
	case string:
		if v == "" {
			return make(map[string]any), nil
		}
		dec := json.NewDecoder(strings.NewReader(v))
		if exactNumbers {
			dec.UseNumber()
		}
		var out map[string]any
		if err := dec.Decode(&out); err != nil {
			return nil, err
		}
		if out == nil {
			return make(map[string]any), nil
		}
		return out, nil
	case map[string]any:
		if exactNumbers {
			return maps.Clone(v), nil
		}
		return integersAsFloat64(maps.Clone(v)), nil
	default:
		return nil, fmt.Errorf("unexpected session state type %T", raw)
	}
}

// encodeEventData serializes event with snake_case schema keys and float
// epoch-second timestamps, the shape adk-python's Event dumps. It returns a
// native map, as adk-python releases through 2.13.0 store and read
// (event.model_dump(exclude_none=True, mode="json")), unless Firestore would
// reject the map, or might, as for an array directly inside another array
// (see firestoreMeasure);
// that event is returned as the JSON string adk-python's main branch stores
// for every event (event.model_dump_json(exclude_none=True),
// firestore_session_service.py:701) and reads alongside maps (line 418).
//
// Writing a map whenever Firestore accepts one diverges from adk-python main
// on purpose: released adk-python cannot read a string, so this keeps Go
// sessions readable by it.
// TODO(#2057): store every event as a string once a released adk-python
// reads one.
func encodeEventData(event *session.Event) (any, error) {
	b, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("firestore: marshaling event %q: %w", event.ID, err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out map[string]any
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("firestore: decoding marshaled event %q: %w", event.ID, err)
	}
	// Integers stay int64 so a map stores them as Firestore integers, as
	// adk-python does.
	for k, v := range out {
		out[k] = jsonNumbersToGo(v)
	}
	out = convertEventMapKeyCase(out, camelToSnake)
	out["timestamp"] = timeToEpochSeconds(event.Timestamp)
	if c := event.Actions.Compaction; c != nil {
		if err := encodeCompactionForPython(out, c); err != nil {
			return nil, fmt.Errorf("firestore: encoding compaction of event %q: %w", event.ID, err)
		}
	}
	if m := measureFirestoreValue(out); m.ok && m.size <= maxFirestorePayloadBytes && m.values <= maxFirestoreIndexedValues {
		return out, nil
	}
	return eventDataString(event.ID, out)
}

// eventDataString returns the event_data map of the event with the given ID
// as the JSON string adk-python's main branch stores.
func eventDataString(eventID string, eventData map[string]any) (any, error) {
	b, err := json.Marshal(eventData)
	if err != nil {
		return nil, fmt.Errorf("firestore: marshaling event %q: %w", eventID, err)
	}
	if len(b) > maxFirestorePayloadBytes {
		return nil, fmt.Errorf(
			"firestore: event %q serialized size (%d bytes) exceeds the %d-byte Firestore payload limit; store large binary payloads in artifact.Service and reference them via genai.FileData URIs",
			eventID, len(b), maxFirestorePayloadBytes,
		)
	}
	return string(b), nil
}

const (
	// maxFirestoreValueDepth is how many maps and arrays Firestore lets a
	// field nest, the field's own value included.
	maxFirestoreValueDepth = 20
	// maxFirestoreFieldPathBytes bounds the field names along one path. It
	// is below Firestore's 1,500-byte limit, so a path near the limit is
	// treated as too long rather than risked.
	maxFirestoreFieldPathBytes = 1024
	// maxFirestoreIndexedValues bounds the values of one field that
	// Firestore indexes. It writes two index entries for each by default and
	// rejects a document with more than 40,000; this leaves room for the
	// document's other fields.
	maxFirestoreIndexedValues = 18000
)

// firestoreMeasure is what measureFirestoreValue learns about a value.
type firestoreMeasure struct {
	// ok is false when Firestore does not store the value as a field: it
	// holds an array directly inside an array, a map key that is empty or
	// reserved (__ at both ends of a longer name), or it nests or names
	// fields beyond Firestore's limits. Some databases accept an array
	// inside an array, but the Firestore emulator and others do not.
	ok bool
	// size is the value's storage size by Firestore's rules: a string is
	// its UTF-8 length plus one, a number 8 bytes, a boolean or null 1 byte,
	// a map or array the sum of its entries, a map key its length plus one.
	size int
	// values bounds what Firestore indexes: the value itself, every value
	// of a map at any depth, and each element of an array. Firestore counts
	// equal elements of an array once and does not index the fields of a map
	// inside an array, so nothing under an event's content parts counts.
	// Measured on a real database: 19,000 distinct strings in a map or an
	// array are accepted and 21,000 rejected, as are 9,900 and 10,100
	// one-key maps or one-element arrays in a map; an array of 19,000
	// distinct three-key maps is accepted.
	values int
	// arrays is how many arrays enclose the value being measured.
	arrays int
}

// measureFirestoreValue measures v as the value of a document field.
func measureFirestoreValue(v any) firestoreMeasure {
	m := firestoreMeasure{ok: true}
	m.add(v, 1, 0, false)
	return m
}

// measureFirestoreDocument measures fields as the fields of a document.
func measureFirestoreDocument(fields map[string]any) firestoreMeasure {
	m := firestoreMeasure{ok: true}
	m.add(fields, 0, 0, false)
	return m
}

// add measures v, which sits depth levels deep, under field names totalling
// pathBytes, and directly inside an array when inList is set.
func (m *firestoreMeasure) add(v any, depth, pathBytes int, inList bool) {
	if m.arrays == 0 {
		m.values++
	}
	switch x := v.(type) {
	case map[string]any:
		if depth > maxFirestoreValueDepth {
			m.ok = false
			return
		}
		for k, e := range x {
			keyPath := pathBytes + len(k) + 1
			reserved := len(k) > 4 && strings.HasPrefix(k, "__") && strings.HasSuffix(k, "__")
			if k == "" || reserved || keyPath > maxFirestoreFieldPathBytes {
				m.ok = false
				return
			}
			m.size += len(k) + 1
			m.add(e, depth+1, keyPath, false)
		}
	case []any:
		if inList || depth > maxFirestoreValueDepth {
			m.ok = false
			return
		}
		if m.arrays == 0 {
			m.values += len(x)
		}
		m.arrays++
		for _, e := range x {
			m.add(e, depth+1, pathBytes, true)
		}
		m.arrays--
	case string:
		m.size += len(x) + 1
	case []byte:
		m.size += len(x)
	case nil, bool:
		m.size++
	default:
		// Numbers and timestamps.
		m.size += 8
	}
}

// encodeCompactionForPython rewrites the snake_case compaction in eventMap so
// adk-python's EventCompaction, which is extra='forbid' and types both
// timestamps as float seconds (events/event_actions.py:90-106), can validate
// it. Float seconds keep only microseconds, and ExcludedEvents has no Python
// counterpart, yet compaction matches both exactly. So the RFC 3339 range
// and ExcludedEvents also go under a top-level key that adk-python's Event
// ignores (extra='ignore', events/event.py:108), and decodeEventData prefers
// them.
func encodeCompactionForPython(eventMap map[string]any, c *session.EventCompaction) error {
	actions, ok := eventMap["actions"].(map[string]any)
	if !ok {
		return fmt.Errorf("unexpected actions type %T", eventMap["actions"])
	}
	comp, ok := actions["compaction"].(map[string]any)
	if !ok {
		return fmt.Errorf("unexpected compaction type %T", actions["compaction"])
	}
	exact := map[string]any{
		"start_timestamp": comp["start_timestamp"],
		"end_timestamp":   comp["end_timestamp"],
	}
	if refs, ok := comp["excluded_events"]; ok {
		delete(comp, "excluded_events")
		exact["excluded_events"] = refs
	}
	eventMap[goCompactionKey] = exact
	comp["start_timestamp"] = timeToEpochSeconds(c.StartTimestamp)
	comp["end_timestamp"] = timeToEpochSeconds(c.EndTimestamp)
	return nil
}

func decodeEventData(raw any) (*session.Event, error) {
	var m map[string]any
	switch v := raw.(type) {
	case string:
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil, err
		}
	case map[string]any:
		// adk-python before the JSON-string format stored a native map
		// without nested-array wrappers, and its reader validates the map
		// as is (firestore_session_service.py:422), so this does not unwrap.
		m = v
	default:
		return nil, fmt.Errorf("unexpected event_data type %T", raw)
	}

	// Normalize snake_case structural keys from adk-python's
	// event.model_dump(exclude_none=True, mode="json") to Go's camelCase json
	// tags while keeping user-defined keys in state_delta, function_call.args,
	// function_response.response, custom_metadata, and output untouched.
	m = convertEventMapKeyCase(m, snakeToCamel)
	restoreGoCompaction(m)
	normalizeCompactionTimestampsForGo(m)
	normalizeContentBytes(m["content"])
	if actions, ok := m["actions"].(map[string]any); ok {
		if comp, ok := actions["compaction"].(map[string]any); ok {
			normalizeContentBytes(comp["compactedContent"])
		}
	}
	// The only bytes field of an event outside its content.
	if gm, ok := m["groundingMetadata"].(map[string]any); ok {
		if sep, ok := gm["searchEntryPoint"].(map[string]any); ok {
			toStdBase64(sep, "sdkBlob")
		}
	}

	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	ev := &session.Event{}
	if err := json.Unmarshal(b, ev); err != nil {
		return nil, err
	}
	return ev, nil
}

// restoreGoCompaction moves the fields encodeCompactionForPython stored at the
// top level of a camelCase event map back into actions.compaction.
func restoreGoCompaction(eventMap map[string]any) {
	key := snakeToCamel(goCompactionKey)
	exact, ok := eventMap[key].(map[string]any)
	if !ok {
		return
	}
	actions, _ := eventMap["actions"].(map[string]any)
	if comp, ok := actions["compaction"].(map[string]any); ok {
		maps.Copy(comp, exact)
	}
}

func timeToEpochSeconds(t time.Time) float64 {
	return float64(t.UTC().UnixMicro()) / 1e6
}

func normalizeCompactionTimestampsForGo(eventMap map[string]any) {
	actionsMap, ok := eventMap["actions"].(map[string]any)
	if !ok {
		return
	}
	compMap, ok := actionsMap["compaction"].(map[string]any)
	if !ok {
		return
	}
	for _, k := range []string{"startTimestamp", "endTimestamp"} {
		if secs, ok := compMap[k].(float64); ok {
			compMap[k] = epochSecondsToRFC3339(secs)
		}
	}
}

// urlSafeToStdBase64 maps the URL-safe base64 alphabet, which pydantic uses
// for genai bytes fields, to the standard one encoding/json expects for []byte.
var urlSafeToStdBase64 = strings.NewReplacer("-", "+", "_", "/")

// normalizeContentBytes rewrites the bytes fields of a camelCase genai.Content
// map (thoughtSignature, inlineData.data and the inline data of function
// response parts) from URL-safe to standard base64, so content written by
// adk-python decodes into Go's []byte fields.
func normalizeContentBytes(content any) {
	c, _ := content.(map[string]any)
	parts, _ := c["parts"].([]any)
	for _, p := range parts {
		part, ok := p.(map[string]any)
		if !ok {
			continue
		}
		toStdBase64(part, "thoughtSignature")
		if blob, ok := part["inlineData"].(map[string]any); ok {
			toStdBase64(blob, "data")
		}
		fr, _ := part["functionResponse"].(map[string]any)
		frParts, _ := fr["parts"].([]any)
		for _, fp := range frParts {
			fpMap, _ := fp.(map[string]any)
			if blob, ok := fpMap["inlineData"].(map[string]any); ok {
				toStdBase64(blob, "data")
			}
		}
	}
}

func toStdBase64(m map[string]any, key string) {
	s, ok := m[key].(string)
	if !ok {
		return
	}
	m[key] = urlSafeToStdBase64.Replace(s)
}

func epochSecondsToRFC3339(secs float64) string {
	micros := math.Round(secs * 1e6)
	return time.UnixMicro(int64(micros)).UTC().Format(time.RFC3339Nano)
}

func convertEventMapKeyCase(m map[string]any, keyFn func(string) string) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		nk := keyFn(k)
		switch nk {
		case "customMetadata", "custom_metadata", "output":
			out[nk] = v
		case "actions":
			if am, ok := v.(map[string]any); ok {
				out[nk] = convertActionsMapKeyCase(am, keyFn)
			} else {
				out[nk] = v
			}
		case "requestedInput", "requested_input":
			if rm, ok := v.(map[string]any); ok {
				out[nk] = convertRequestedInputMapKeyCase(rm, keyFn)
			} else {
				out[nk] = v
			}
		case "content":
			if cm, ok := v.(map[string]any); ok {
				out[nk] = convertContentMapKeyCase(cm, keyFn)
			} else {
				out[nk] = v
			}
		default:
			out[nk] = convertSchemaValueKeyCase(v, keyFn)
		}
	}
	return out
}

func convertActionsMapKeyCase(m map[string]any, keyFn func(string) string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		nk := keyFn(k)
		switch nk {
		case "stateDelta", "state_delta",
			"artifactDelta", "artifact_delta",
			"requestedAuthConfigs", "requested_auth_configs",
			"requestedToolConfirmations", "requested_tool_confirmations":
			out[nk] = v
		case "compaction":
			if cm, ok := v.(map[string]any); ok {
				out[nk] = convertCompactionMapKeyCase(cm, keyFn)
			} else {
				out[nk] = v
			}
		default:
			out[nk] = convertSchemaValueKeyCase(v, keyFn)
		}
	}
	return out
}

func convertCompactionMapKeyCase(m map[string]any, keyFn func(string) string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		nk := keyFn(k)
		if nk == "compactedContent" || nk == "compacted_content" {
			if cm, ok := v.(map[string]any); ok {
				out[nk] = convertContentMapKeyCase(cm, keyFn)
				continue
			}
		}
		out[nk] = convertSchemaValueKeyCase(v, keyFn)
	}
	return out
}

func convertRequestedInputMapKeyCase(m map[string]any, keyFn func(string) string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		nk := keyFn(k)
		switch nk {
		case "payload", "responseSchema", "response_schema":
			out[nk] = v
		default:
			out[nk] = convertSchemaValueKeyCase(v, keyFn)
		}
	}
	return out
}

func convertContentMapKeyCase(m map[string]any, keyFn func(string) string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		nk := keyFn(k)
		switch nk {
		case "partMetadata", "part_metadata":
			out[nk] = v
		case "functionCall", "function_call", "toolCall", "tool_call":
			out[nk] = convertKeepingUserField(v, "args", keyFn, convertSchemaValueKeyCase)
		case "functionResponse", "function_response", "toolResponse", "tool_response":
			out[nk] = convertKeepingUserField(v, "response", keyFn, convertContentValueKeyCase)
		default:
			out[nk] = convertContentValueKeyCase(v, keyFn)
		}
	}
	return out
}

// convertKeepingUserField converts the keys of v, when it is a map, with
// convert, except for the value under userField, which holds user data and is
// kept verbatim.
func convertKeepingUserField(v any, userField string, keyFn func(string) string, convert func(any, func(string) string) any) any {
	m, ok := v.(map[string]any)
	if !ok {
		return v
	}
	out := make(map[string]any, len(m))
	for k, val := range m {
		nk := keyFn(k)
		if nk == userField {
			out[nk] = val
		} else {
			out[nk] = convert(val, keyFn)
		}
	}
	return out
}

func convertContentValueKeyCase(v any, keyFn func(string) string) any {
	switch val := v.(type) {
	case map[string]any:
		return convertContentMapKeyCase(val, keyFn)
	case []any:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = convertContentValueKeyCase(item, keyFn)
		}
		return out
	default:
		return v
	}
}

func convertSchemaValueKeyCase(v any, keyFn func(string) string) any {
	switch val := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(val))
		for k, item := range val {
			out[keyFn(k)] = convertSchemaValueKeyCase(item, keyFn)
		}
		return out
	case []any:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = convertSchemaValueKeyCase(item, keyFn)
		}
		return out
	default:
		return v
	}
}

func camelToSnake(s string) string {
	var b strings.Builder
	for i, r := range s {
		if r >= 'A' && r <= 'Z' {
			if i > 0 {
				b.WriteByte('_')
			}
			b.WriteRune(r + ('a' - 'A'))
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func snakeToCamel(s string) string {
	if !strings.Contains(s, "_") {
		return s
	}
	var b strings.Builder
	upperNext := false
	for i, r := range s {
		if r == '_' {
			if i > 0 {
				upperNext = true
			} else {
				b.WriteRune(r)
			}
			continue
		}
		if upperNext && r >= 'a' && r <= 'z' {
			b.WriteRune(r - ('a' - 'A'))
		} else {
			b.WriteRune(r)
		}
		upperNext = false
	}
	return b.String()
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
