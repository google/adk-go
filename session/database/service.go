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

// Package database provides a session.Service backed by a relational
// database (for example PostgreSQL, Spanner, or SQLite) using GORM.
//
// The service never creates or alters its tables. Call [AutoMigrate] after
// constructing it, on every startup: a release of this package may add
// columns, and writes to that table fail until they exist. Applications that
// manage the schema themselves instead of calling AutoMigrate must add those
// columns before deploying the release that introduces them.
package database

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// databaseService is an database implementation of sessionService.Service.
type databaseService struct {
	db *gorm.DB
	// ownsDB records that the service opened db's connection pool itself,
	// so Close closes it.
	ownsDB bool
	// appendOrderTiebreak makes Get order events that share a timestamp by
	// append order instead of by event ID. Only [New] sets it.
	appendOrderTiebreak bool

	sessionLocksMu sync.Mutex
	// sessionLocks holds the lock of every session an AppendEvent call holds
	// or waits for. Guarded by sessionLocksMu; nil until the first append.
	sessionLocks map[sessionLockKey]*sessionLock
}

// sessionLockKey identifies a session in databaseService.sessionLocks.
type sessionLockKey struct {
	appName, userID, sessionID string
}

// sessionLock serializes appends to one session. refs counts the calls that
// hold or wait for gate, and is guarded by databaseService.sessionLocksMu.
type sessionLock struct {
	gate chan struct{}
	refs int
}

// Config configures a database [session.Service] created with [New].
//
// The zero Config is not usable: New returns an error when Dialector is nil.
type Config struct {
	// Dialector specifies the database connection. It is required.
	Dialector gorm.Dialector
	// AppendOrderTiebreak configures the service to break timestamp ties
	// by the order events were appended, rather than by event ID.
	AppendOrderTiebreak bool
	// GORMOptions configures the underlying GORM database.
	GORMOptions []gorm.Option
}

// New creates a [session.Service] that stores sessions in the relational
// database cfg.Dialector connects to (e.g., PostgreSQL, Spanner, SQLite),
// opened with gorm.Open(cfg.Dialector, cfg.GORMOptions...).
//
// By default, the service orders events with equal timestamps ascending by
// event ID, matching adk-python's DatabaseSessionService; this is also the
// order of every service created with [NewSessionService] or
// [NewSessionServiceFromDB]. When cfg.AppendOrderTiebreak is true, events
// sharing a timestamp are returned in the order they were appended, matching
// adk-python's SqliteSessionService.
//
// The returned service implements [io.Closer]. Close closes the connection
// pool when cfg.Dialector opened it, as sqlite.Open(dsn) does, and leaves
// open a connection the module's SQLite dialector carries
// (sqlite.Dialector{Conn: db} or &sqlite.Dialector{Conn: db}). A caller who
// passes an open connection through any other dialector and wants it kept
// open builds the service with [NewSessionServiceFromDB].
//
// It returns an error if cfg.Dialector is nil or if [gorm.Open] fails. The
// service does not create its tables. See [AutoMigrate].
func New(cfg Config) (session.Service, error) {
	// gorm.Open accepts a nil dialector without an error.
	if cfg.Dialector == nil {
		return nil, errors.New("dialector must not be nil")
	}
	db, err := gorm.Open(cfg.Dialector, cfg.GORMOptions...)
	if err != nil {
		return nil, fmt.Errorf("error creating database session service: %w", err)
	}
	return &databaseService{db: db, ownsDB: opensOwnConnection(cfg.Dialector), appendOrderTiebreak: cfg.AppendOrderTiebreak}, nil
}

var (
	_ session.UserStateReader = (*databaseService)(nil)
	_ io.Closer               = (*databaseService)(nil)
)

// opensOwnConnection reports whether gorm.Open(dialector) opens a connection
// pool of its own rather than using one the caller opened. Only the module's
// SQLite dialector, as a value or a pointer, is recognized as carrying the
// caller's connection (its Conn field); a caller who passes an open
// connection through another dialector uses [NewSessionServiceFromDB] to keep
// it open.
func opensOwnConnection(dialector gorm.Dialector) bool {
	switch d := dialector.(type) {
	case *sqlite.Dialector:
		return d.Conn == nil
	case sqlite.Dialector:
		return d.Conn == nil
	default:
		return true
	}
}

// Close closes the underlying [*sql.DB] connection pool when s opened it,
// and otherwise leaves the caller's connection open and returns nil. A
// connection passed in a sqlite.Dialector or *sqlite.Dialector stays open.
// It may be called more than once and concurrently. Calls on the service
// after Close fail with the closed pool's error.
func (s *databaseService) Close() error {
	if !s.ownsDB {
		return nil
	}
	sqlDB, err := s.db.DB()
	if err != nil {
		return fmt.Errorf("close database session service: %w", err)
	}
	return sqlDB.Close()
}

// NewSessionService creates a new [session.Service] implementation that uses a
// relational database (e.g., PostgreSQL, Spanner, SQLite) via the GORM library.
//
// It requires a [gorm.Dialector] to specify the database connection and
// accepts optional [gorm.Option] values for further GORM configuration.
//
// The returned service implements [io.Closer]. Close closes the connection
// pool when dialector opened it, as sqlite.Open(dsn) does, and leaves open a
// connection the module's SQLite dialector carries
// (sqlite.Dialector{Conn: db} or &sqlite.Dialector{Conn: db}). A caller who
// passes an open connection through any other dialector and wants it kept
// open builds the service with [NewSessionServiceFromDB].
//
// It returns the new [session.Service] or an error if the database connection
// [gorm.Open] fails. The service does not create its tables. See [AutoMigrate].
func NewSessionService(dialector gorm.Dialector, opts ...gorm.Option) (session.Service, error) {
	db, err := gorm.Open(dialector, opts...)
	if err != nil {
		return nil, fmt.Errorf("error creating database session service: %w", err)
	}
	return &databaseService{db: db, ownsDB: opensOwnConnection(dialector)}, nil
}

// NewSessionServiceFromDB creates a new [session.Service] implementation using
// an existing [*gorm.DB] connection. This is useful when the application
// already manages a database connection and wants to share it across multiple
// services.
//
// The service always orders events with equal timestamps ascending by event
// ID; [New] is the only constructor that offers append order.
//
// The returned service implements [io.Closer]; its Close leaves db open.
//
// It returns an error if db is nil. The service does not create its tables.
// See [AutoMigrate].
func NewSessionServiceFromDB(db *gorm.DB) (session.Service, error) {
	if db == nil {
		return nil, fmt.Errorf("db must not be nil")
	}
	return &databaseService{db: db}, nil
}

// AutoMigrate runs the GORM auto-migration tool to ensure the database schema
// matches the internal storage models (e.g., storageSession, storageEvent).
// It creates missing tables and columns, alters existing columns whose type,
// size or nullability differs from the models, and never drops a column. It
// can be called repeatedly and is meant to run on every startup.
//
// NOTE: This function relies on a type assertion to the concrete *databaseService
// implementation. It will return an error if the provided session.Service is
// a different implementation.
func AutoMigrate(service session.Service) error {
	dbservice, ok := service.(*databaseService)
	if !ok {
		return fmt.Errorf("invalid session service type")
	}
	err := dbservice.db.AutoMigrate(&storageSession{}, &storageEvent{}, &storageAppState{}, &storageUserState{})
	if err != nil {
		return fmt.Errorf("auto migrate failed: %w", err)
	}
	return nil
}

// Create generates a session and inserts it to the db, implements session.Service
//
// req.AppName is required. An empty req.UserID is the user "", as in
// adk-python. An empty req.SessionID gets a generated ID; any other value is
// stored verbatim.
func (s *databaseService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required")
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = platform.NewUUID(ctx)
	}

	stateMap := req.State
	if stateMap == nil {
		stateMap = make(map[string]any)
	}
	val := &localSession{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: sessionID,
		state:     stateMap,
		updatedAt: platform.Now(ctx),
	}
	createdSession, err := createStorageSession(ctx, val)
	if err != nil {
		return nil, err
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		appDelta, userDelta, sessionState := extractStateDeltas(req.State)

		var err error
		// Lock the shared rows this session's state changes, app before user,
		// for the same reasons and in the same order as applyEvent.
		appTx := tx
		if len(appDelta) > 0 {
			if appTx, err = lockStateRow(tx, &storageAppState{AppName: req.AppName, State: map[string]any{}, UpdateTime: createdSession.UpdateTime}); err != nil {
				return fmt.Errorf("failed to lock app state: %w", err)
			}
		}
		storageApp, err := fetchStorageAppState(appTx, req.AppName)
		if err != nil {
			return fmt.Errorf("error on create session: %w", err)
		}
		userTx := tx
		if len(userDelta) > 0 {
			if userTx, err = lockStateRow(tx, &storageUserState{AppName: req.AppName, UserID: req.UserID, State: map[string]any{}, UpdateTime: createdSession.UpdateTime}); err != nil {
				return fmt.Errorf("failed to lock user state: %w", err)
			}
		}
		storageUser, err := fetchStorageUserState(userTx, req.AppName, req.UserID)
		if err != nil {
			return fmt.Errorf("error on create session: %w", err)
		}

		// apply state delta
		if len(appDelta) > 0 {
			maps.Copy(storageApp.State, appDelta)
			// Maintain UpdateTime explicitly: an unset time.Time serializes to
			// a zero datetime that MySQL rejects under strict mode.
			storageApp.UpdateTime = createdSession.UpdateTime
			if err := tx.Save(&storageApp).Error; err != nil {
				return fmt.Errorf("failed to save app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			maps.Copy(storageUser.State, userDelta)
			storageUser.UpdateTime = createdSession.UpdateTime
			if err := updateUserState(tx, storageUser); err != nil {
				return fmt.Errorf("failed to save user state: %w", err)
			}
		}
		createdSession.State = sessionState

		if err := tx.Create(createdSession).Error; err != nil {
			return fmt.Errorf("error creating session on database: %w", err)
		}

		val.state = mergeStates(storageApp.State, storageUser.State, sessionState)
		val.updatedAt = createdSession.UpdateTime
		return nil
	})
	if err != nil {
		return nil, err
	}

	return &session.CreateResponse{
		Session: val,
	}, nil
}

// Get retrieves a single session from the database using its composite primary key.
//
// req.AppName is required. An empty req.UserID is the user "", and an empty
// req.SessionID is looked up like any other, so Get returns
// [session.ErrNotFound] for it, as in adk-python. A negative
// req.NumRecentEvents is an error. With req.OmitEvents, Get does not read
// the session's events and returns none.
func (s *databaseService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", appName)
	}
	if err := sessionutils.CheckNumRecentEvents(req.NumRecentEvents); err != nil {
		return nil, err
	}

	// String conditions, not a struct: GORM drops a zero-valued struct field
	// from the WHERE clause, and "" is a valid user and session ID here.
	var foundSession storageSession
	err := s.db.WithContext(ctx).
		Where("app_name = ? AND user_id = ? AND id = ?", appName, userID, sessionID).
		First(&foundSession).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("%w: %q: %w", session.ErrNotFound, sessionID, err)
		}
		return nil, fmt.Errorf("database error while fetching session: %w", err)
	}

	// Fetch events
	eventQuery := s.db.WithContext(ctx).
		Model(&storageEvent{}).
		Where("app_name = ?", appName).
		Where("user_id = ?", userID).
		Where("session_id = ?", sessionID)

	// Apply conditional filters from the request
	if !req.After.IsZero() {
		eventQuery = eventQuery.Where("timestamp >= ?", req.After)
	}

	// Order by timestamp DESC to get the most recent events when limiting, with
	// id as a tiebreak so the order is total.
	//
	// Without the tiebreak, events sharing a timestamp come back in whatever
	// order the engine happens to yield and the reversal below then flips them:
	// SQLite's sort is stable, so a tie is returned in reverse insertion order.
	// Timestamps are truncated to microseconds on write, so ties are ordinary
	// rather than exotic. Compaction reads this order to choose what a summary
	// stands for, and a pair that swaps between two reads means a record can
	// cover an event nothing summarized, which is that event gone from every
	// later prompt.
	//
	// The id is arbitrary as an ordering, but it is stable, and stable is what
	// callers need. adk-python's DatabaseSessionService orders the same way,
	// by timestamp then id.
	//
	// With appendOrderTiebreak, ties go by seq, the append order, as
	// adk-python's SqliteSessionService orders by rowid. Rows written before
	// seq existed all hold 0, so among themselves they still go by id.
	orderClause := "timestamp DESC, id DESC"
	if s.appendOrderTiebreak {
		orderClause = "timestamp DESC, seq DESC, id DESC"
	}
	eventQuery = eventQuery.Order(orderClause)

	if req.NumRecentEvents > 0 {
		eventQuery = eventQuery.Limit(req.NumRecentEvents)
	}

	var storageEvents []storageEvent
	// adk-python skips the events query for num_recent_events == 0
	// (database_session_service.py:753).
	if !req.OmitEvents {
		if err := eventQuery.Find(&storageEvents).Error; err != nil {
			// This is a system failure, not a "not found"
			return nil, fmt.Errorf("database error while fetching events: %w", err)
		}
	}

	// fetch app and user states
	storageApp, err := fetchStorageAppState(s.db.WithContext(ctx), appName)
	if err != nil {
		return nil, fmt.Errorf("error on get session: %w", err)
	}
	storageUser, err := fetchStorageUserState(s.db.WithContext(ctx), appName, userID)
	if err != nil {
		return nil, fmt.Errorf("error on get session: %w", err)
	}

	responseSession, err := createSessionFromStorageSession(&foundSession)
	responseSession.state = mergeStates(storageApp.State, storageUser.State, responseSession.state)
	if err != nil {
		return nil, fmt.Errorf("failed to map storage object: %w", err)
	}

	// We fetched in DESC order to get the most recent ones (due to LIMIT).
	// Now we reverse them to be in chronological ASC order for the response.
	// Convert storage events to response events
	responseEvents := make([]*session.Event, 0, len(storageEvents))
	for i := len(storageEvents) - 1; i >= 0; i-- {
		evt, err := createEventFromStorageEvent(&storageEvents[i])
		if err != nil {
			return nil, fmt.Errorf("failed to map storage event: %w", err)
		}
		responseEvents = append(responseEvents, evt)
	}
	responseSession.events = responseEvents

	return &session.GetResponse{
		Session: responseSession,
	}, nil
}

// List retrieves sessions from the database using its appName and optional UserID
func (s *databaseService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	appName, userID := req.AppName, req.UserID
	if appName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}

	oneUser := userID != "" || req.FilterEmptyUserID
	var foundSessions []storageSession
	listQuery := s.db.WithContext(ctx).Where("app_name = ?", appName)
	if oneUser {
		listQuery = listQuery.Where("user_id = ?", userID)
	}

	err := listQuery.Find(&foundSessions).Error
	if err != nil {
		// Specifically check if the error is "record not found".
		if errors.Is(err, gorm.ErrRecordNotFound) {
			// This is not a system failure. The record simply doesn't exist.
			return &session.ListResponse{
				Sessions: make([]session.Session, 0),
			}, nil
		}
		// For any other error (e.g., connection lost), return it as a system error.
		return nil, fmt.Errorf("database error while fetching session: %w", err)
	}

	storageApp, err := fetchStorageAppState(s.db.WithContext(ctx), appName)
	if err != nil {
		return nil, fmt.Errorf("error on list sessions: %w", err)
	}

	var userStates map[string]*storageUserState
	if oneUser {
		userState, err := fetchStorageUserState(s.db.WithContext(ctx), appName, userID)
		if err != nil {
			return nil, fmt.Errorf("error on list sessions: %w", err)
		}
		userStates = map[string]*storageUserState{userID: userState}
	} else {
		userStates, err = fetchAllAppStorageUserState(s.db.WithContext(ctx), appName)
		if err != nil {
			return nil, fmt.Errorf("error on list sessions: %w", err)
		}
	}

	// Sort here rather than with ORDER BY. SQLite stores update_time as RFC 3339
	// text with the value's own offset, so it would compare text, not instants,
	// for rows written in different zones.
	slices.SortStableFunc(foundSessions, func(a, b storageSession) int {
		return cmp.Or(
			a.UpdateTime.Compare(b.UpdateTime),
			strings.Compare(a.UserID, b.UserID),
			strings.Compare(a.ID, b.ID),
		)
	})

	// Create response sessions, transform the storageSessions into
	responseSessions := make([]session.Session, 0, len(foundSessions))
	for _, storage := range foundSessions {
		sess, err := createSessionFromStorageSession(&storage)
		if err != nil {
			// If we encounter a single mapping error, we fail the whole request.
			return nil, fmt.Errorf("failed to map storage object for session %s: %w", storage.ID, err)
		}

		userState, ok := userStates[sess.UserID()]
		if !ok {
			userState = &storageUserState{AppName: appName, UserID: userID, State: make(map[string]any)}
		}
		sess.state = mergeStates(storageApp.State, userState.State, sess.state)
		responseSessions = append(responseSessions, sess)
	}

	return &session.ListResponse{
		Sessions: responseSessions,
	}, nil
}

// Delete, deletes a session given a specific id returning error on failure, implements session.Service
//
// req.AppName is required. An empty req.UserID is the user "", and an empty
// req.SessionID is looked up like any other, so Delete deletes nothing for it
// and returns nil, as in adk-python.
func (s *databaseService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" {
		return fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Delete the events explicitly instead of relying on the ON DELETE
		// CASCADE foreign key: SQLite enforces foreign keys only when the
		// connection enables them, and without it the session row goes and its
		// events stay behind.
		//
		// String conditions, not structs: GORM drops a zero-valued struct
		// field from the WHERE clause, and "" is a valid user and session ID.
		if err := tx.Where("app_name = ? AND user_id = ? AND session_id = ?", req.AppName, req.UserID, req.SessionID).
			Delete(&storageEvent{}).Error; err != nil {
			return fmt.Errorf("database error during session events deletion: %w", err)
		}

		result := tx.Where("app_name = ? AND user_id = ? AND id = ?", req.AppName, req.UserID, req.SessionID).
			Delete(&storageSession{})

		if result.Error != nil {
			return fmt.Errorf("database error during session deletion: %w", result.Error)
		}

		return nil // Returning nil commits the transaction
	})
}

func (s *databaseService) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	if curSession == nil {
		return fmt.Errorf("session is nil")
	}
	if event == nil {
		return fmt.Errorf("event is nil")
	}
	// ignore partial events
	if event.Partial {
		return nil
	}
	// A stored request that does not decode would fail every later read of
	// the session. Checked before the local session takes the event.
	if err := sessionutils.CheckRequestedAuthConfigs(event.Actions.RequestedAuthConfigs); err != nil {
		return fmt.Errorf("cannot append event: %w", err)
	}
	// Give the event an identity if it arrived without one, matching the
	// in-memory service. An event built as a struct literal by an agent or a
	// tool never passes through session.NewEvent, and anything that identifies
	// events by ID cannot tell two ID-less events apart.
	if event.ID == "" {
		event.ID = platform.NewUUID(ctx)
	}

	// Truncate timestamp to microsecond precision to match database precision and prevent rounding errors.
	event.Timestamp = event.Timestamp.Truncate(time.Microsecond)

	// Neither the stored event nor the session's copy keeps temp: keys.
	trimmedEvent := trimTempDeltaState(event)

	unlock, err := s.lockSession(ctx, curSession.AppName(), curSession.UserID(), curSession.ID())
	if err != nil {
		return err
	}
	defer unlock()
	if err := s.applyEvent(ctx, curSession, trimmedEvent); err != nil {
		return err
	}
	// Only a committed event reaches curSession, so a rejected append leaves
	// it as it was. adk-python applies temp: keys to the session before it
	// checks the session exists or is current (database_session_service.py:908).
	return commitToSession(curSession, trimmedEvent, event.Actions.StateDelta, event.Timestamp)
}

// lockSession waits until no other AppendEvent call of s holds the session
// (appName, userID, sessionID) or ctx ends, and returns the function that
// releases it.
//
// The database locks the session row too, but SQLite ignores FOR UPDATE: two
// connections that read the session and then both write fail with "database
// is locked" instead of the later one seeing the earlier commit and returning
// session.ErrStaleSession. adk-python serializes appends per session in the
// same way (database_session_service.py:499-530).
func (s *databaseService) lockSession(ctx context.Context, appName, userID, sessionID string) (func(), error) {
	key := sessionLockKey{appName: appName, userID: userID, sessionID: sessionID}
	s.sessionLocksMu.Lock()
	if s.sessionLocks == nil {
		s.sessionLocks = make(map[sessionLockKey]*sessionLock)
	}
	l := s.sessionLocks[key]
	if l == nil {
		l = &sessionLock{gate: make(chan struct{}, 1)}
		s.sessionLocks[key] = l
	}
	l.refs++
	s.sessionLocksMu.Unlock()

	dropRef := func() {
		s.sessionLocksMu.Lock()
		defer s.sessionLocksMu.Unlock()
		// Drop the entry with its last user, so the map holds only the
		// sessions being appended to.
		l.refs--
		if l.refs == 0 {
			delete(s.sessionLocks, key)
		}
	}

	select {
	case l.gate <- struct{}{}:
		return func() {
			<-l.gate
			dropRef()
		}, nil
	case <-ctx.Done():
		dropRef()
		return nil, ctx.Err()
	}
}

// commitToSession updates curSession with an event applyEvent committed:
// rawStateDelta, temp: keys included, goes into its state, and trimmedEvent
// and updatedAt into its events and update time. A Session the caller
// implemented has no setters for its events or update time, so only its
// state, when it has one, is updated.
func commitToSession(curSession session.Session, trimmedEvent *session.Event, rawStateDelta map[string]any, updatedAt time.Time) error {
	if ms, ok := curSession.(sessionutils.MutableSession); ok {
		ms.CommitAppend(trimmedEvent, rawStateDelta, updatedAt)
		return nil
	}
	st := curSession.State()
	if st == nil {
		return nil
	}
	for k, v := range rawStateDelta {
		if err := st.Set(k, v); err != nil {
			return fmt.Errorf("event %q is stored, but setting the session's state failed: %w", trimmedEvent.ID, err)
		}
	}
	return nil
}

// applyEvent fetches the session, validates it, applies state changes from an
// event, and saves the event atomically. It does not modify curSession.
func (s *databaseService) applyEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	// Wrap database operations in a single transaction.
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Fetch and lock the session row, so that a concurrent append to the
		// same session waits for this one and then sees its update time and
		// sequence number. adk-python locks the row the same way
		// (database_session_service.py:935-936). SQLite ignores the clause;
		// lockSession serializes appends there.
		var storageSess storageSession
		err := tx.Clauses(clause.Locking{Strength: clause.LockingStrengthUpdate}).
			Where("app_name = ? AND user_id = ? AND id = ?", curSession.AppName(), curSession.UserID(), curSession.ID()).
			First(&storageSess).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return fmt.Errorf("%w: %q, cannot apply event: %w", session.ErrNotFound, curSession.ID(), err)
			}
			return fmt.Errorf("failed to get session: %w", err)
		}

		// Ensure the session object is not stale.
		// We use UnixMicro() for microsecond-level precision, matching the Python code.
		storageUpdateTime := storageSess.UpdateTime.UnixMicro()
		sessionUpdateTime := curSession.LastUpdateTime().UnixMicro()
		if storageUpdateTime > sessionUpdateTime {
			return fmt.Errorf(
				"%w: stale session error: last update time from request (%s) is older than in database (%s)",
				session.ErrStaleSession,
				time.UnixMicro(sessionUpdateTime).Format(time.RFC3339Nano),
				time.UnixMicro(storageUpdateTime).Format(time.RFC3339Nano),
			)
		}

		appDelta, userDelta, sessionDelta := extractStateDeltas(event.Actions.StateDelta)

		// App and user state are shared with other sessions and written back
		// whole, so lock their rows before reading them when this event changes
		// them. Otherwise a concurrent append from another session reads the
		// same row and the later write drops the earlier one's keys. This
		// matches adk-python, which locks these rows FOR UPDATE.
		//
		// Lock and read the app row before touching the user row. Create
		// writes app then user, and inserting the user row first would hold
		// it while waiting for an app row lock that Create holds, so the two
		// could deadlock.
		appTx := tx
		if len(appDelta) > 0 {
			if appTx, err = lockStateRow(tx, &storageAppState{AppName: curSession.AppName(), State: map[string]any{}, UpdateTime: event.Timestamp}); err != nil {
				return fmt.Errorf("failed to lock app state: %w", err)
			}
		}
		storageApp, err := fetchStorageAppState(appTx, curSession.AppName())
		if err != nil {
			return err
		}

		userTx := tx
		if len(userDelta) > 0 {
			if userTx, err = lockStateRow(tx, &storageUserState{AppName: curSession.AppName(), UserID: curSession.UserID(), State: map[string]any{}, UpdateTime: event.Timestamp}); err != nil {
				return fmt.Errorf("failed to lock user state: %w", err)
			}
		}
		storageUser, err := fetchStorageUserState(userTx, curSession.AppName(), curSession.UserID())
		if err != nil {
			return err
		}

		// Merge state deltas and update the storage objects.
		// Save is safe for the app row, whose key is never empty; the user
		// and session rows go through updateUserState and updateSession.
		if len(appDelta) > 0 {
			maps.Copy(storageApp.State, appDelta)
			// Maintain UpdateTime explicitly (see Create): an unset time.Time
			// serializes to a zero datetime that MySQL rejects under strict mode.
			storageApp.UpdateTime = event.Timestamp
			if err := tx.Save(&storageApp).Error; err != nil {
				return fmt.Errorf("failed to save app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			maps.Copy(storageUser.State, userDelta)
			storageUser.UpdateTime = event.Timestamp
			if err := updateUserState(tx, storageUser); err != nil {
				return fmt.Errorf("failed to save user state: %w", err)
			}
		}
		if len(sessionDelta) > 0 {
			maps.Copy(storageSess.State, sessionDelta)
			// The session state update will be saved along with the event timestamp update.
		}

		// Create the new event record in the database.
		storageEv, err := createStorageEvent(curSession, event)
		if err != nil {
			return fmt.Errorf("failed to map event to storage model: %w", err)
		}
		// Number the event after the session's last one. The session row lock
		// (and lockSession on SQLite) keeps two appends from reading the same
		// maximum. COALESCE covers a session with no events yet.
		var maxSeq int64
		if err := tx.Model(&storageEvent{}).
			Where("app_name = ? AND user_id = ? AND session_id = ?", curSession.AppName(), curSession.UserID(), curSession.ID()).
			Select("COALESCE(MAX(seq), 0)").
			Scan(&maxSeq).Error; err != nil {
			return fmt.Errorf("failed to query event sequence: %w", err)
		}
		storageEv.Seq = maxSeq + 1
		if err := tx.Create(storageEv).Error; err != nil {
			return fmt.Errorf("failed to save event: %w", err)
		}

		storageSess.UpdateTime = event.Timestamp
		if err := updateSession(tx, &storageSess); err != nil {
			return fmt.Errorf("failed to save session state: %w", err)
		}

		return nil // Returning nil commits the transaction.
	})

	return err
}

// lockStateRow makes sure the state row exists and returns tx set to read it
// FOR UPDATE. A missing row is inserted empty first, since FOR UPDATE cannot
// lock a row that does not exist yet, and a concurrent insert of the same row
// is ignored. SQLite ignores the locking clause; it serializes writers anyway.
func lockStateRow(tx *gorm.DB, emptyRow any) (*gorm.DB, error) {
	if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(emptyRow).Error; err != nil {
		return nil, err
	}
	return tx.Clauses(clause.Locking{Strength: clause.LockingStrengthUpdate}), nil
}

// updateSession writes sess.State and sess.UpdateTime to the existing row
// with sess's key. Save would not do: it leaves a zero-valued key field, such
// as the user "", out of its WHERE clause and then inserts a duplicate row.
// RowsAffected is not checked, because MySQL counts changed rows, not matched
// ones; the caller has read the row inside the same transaction.
func updateSession(tx *gorm.DB, sess *storageSession) error {
	return tx.Model(&storageSession{}).
		Where("app_name = ? AND user_id = ? AND id = ?", sess.AppName, sess.UserID, sess.ID).
		Select("State", "UpdateTime").
		Updates(sess).Error
}

// updateUserState writes st.State and st.UpdateTime to the existing row with
// st's key, for the same reasons as updateSession. The caller has made sure
// the row exists with lockStateRow.
func updateUserState(tx *gorm.DB, st *storageUserState) error {
	return tx.Model(&storageUserState{}).
		Where("app_name = ? AND user_id = ?", st.AppName, st.UserID).
		Select("State", "UpdateTime").
		Updates(st).Error
}

func fetchStorageAppState(tx *gorm.DB, appName string) (*storageAppState, error) {
	var storageApp storageAppState
	if err := tx.First(&storageApp, "app_name = ?", appName).Error; err != nil {
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("failed to fetch app state: %w", err)
		}
		// If not found, initialize a new object to be created later.
		storageApp = storageAppState{AppName: appName, State: make(map[string]any)}
	}
	return &storageApp, nil
}

func fetchStorageUserState(tx *gorm.DB, appName, userID string) (*storageUserState, error) {
	var storageUser storageUserState
	if err := tx.First(&storageUser, "app_name = ? AND user_id = ?", appName, userID).Error; err != nil {
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("failed to fetch user state: %w", err)
		}
		// If not found, initialize a new object.
		storageUser = storageUserState{AppName: appName, UserID: userID, State: make(map[string]any)}
	}
	return &storageUser, nil
}

func fetchAllAppStorageUserState(tx *gorm.DB, appName string) (map[string]*storageUserState, error) {
	var storageUserStates []storageUserState

	if err := tx.Find(&storageUserStates, "app_name = ?", appName).Error; err != nil {
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("failed to fetch user states: %w", err)
		}
		return make(map[string]*storageUserState), nil
	}
	statesByUserId := make(map[string]*storageUserState, len(storageUserStates))
	for _, storageUserState := range storageUserStates {
		statesByUserId[storageUserState.UserID] = &storageUserState
	}
	return statesByUserId, nil
}

// extractStateDeltas splits a single state delta map into three separate maps
// for app, user, and session states based on key prefixes.
// Temporary keys (starting with session.KeyPrefixTemp) are ignored.
func extractStateDeltas(delta map[string]any) (
	appStateDelta, userStateDelta, sessionStateDelta map[string]any,
) {
	// Initialize the maps to be returned.
	appStateDelta = make(map[string]any)
	userStateDelta = make(map[string]any)
	sessionStateDelta = make(map[string]any)

	if delta == nil {
		return appStateDelta, userStateDelta, sessionStateDelta
	}

	for key, value := range delta {
		if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixApp); found {
			appStateDelta[cleanKey] = value
		} else if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixUser); found {
			userStateDelta[cleanKey] = value
		} else if !strings.HasPrefix(key, session.KeyPrefixTemp) {
			// This key belongs to the session state, as long as it's not temporary.
			sessionStateDelta[key] = value
		}
	}
	return appStateDelta, userStateDelta, sessionStateDelta
}

// mergeStates combines app, user, and session state maps into a single map
// for client-side responses, adding the appropriate prefixes back.
func mergeStates(appState, userState, sessionState map[string]any) map[string]any {
	// Pre-allocate map capacity for efficiency.
	totalSize := len(appState) + len(userState) + len(sessionState)
	mergedState := make(map[string]any, totalSize)

	// In Go, we create a new map and copy key-value pairs. This is equivalent
	// to the goal of Python's copy.deepcopy() in this context, which is to
	// avoid modifying the original sessionState map.
	maps.Copy(mergedState, sessionState)

	for key, value := range appState {
		mergedState[session.KeyPrefixApp+key] = value
	}

	for key, value := range userState {
		mergedState[session.KeyPrefixUser+key] = value
	}

	return mergedState
}

// GetUserState returns the user-scoped state stored for exactly
// (req.AppName, req.UserID), with keys stripped of [session.KeyPrefixUser],
// and an empty map when no row exists. Empty ids are matched literally. Each
// call decodes the stored JSON afresh, so the returned map shares nothing with
// the database or with earlier results. A nil req, a failed query or a
// canceled ctx returns a nil response and an error, which wraps ctx.Err() in
// the last case. It implements [session.UserStateReader].
func (s *databaseService) GetUserState(ctx context.Context, req *session.GetUserStateRequest) (*session.GetUserStateResponse, error) {
	if req == nil {
		return nil, errors.New("get user state request is nil")
	}
	storageUser, err := fetchStorageUserState(s.db.WithContext(ctx), req.AppName, req.UserID)
	if err != nil {
		return nil, err
	}
	state := storageUser.State
	if state == nil {
		// A state column holding JSON null decodes to a nil map.
		state = make(map[string]any)
	}
	return &session.GetUserStateResponse{State: state}, nil
}
