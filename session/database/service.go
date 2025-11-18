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
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"google.golang.org/adk/session"
)

// databaseService is an database implementation of sessionService.Service.
type databaseService struct {
	db *gorm.DB
}

// NewSessionService creates a new [session.Service] implementation that uses a
// relational database (e.g., PostgreSQL, Spanner, SQLite) via the GORM library.
//
// It requires a [gorm.Dialector] to specify the database connection and
// accepts optional [gorm.Option] values for further GORM configuration.
//
// It returns the new [session.Service] or an error if the database connection
// [gorm.Open] fails.
func NewSessionService(dialector gorm.Dialector, opts ...gorm.Option) (session.Service, error) {
	db, err := gorm.Open(dialector, opts...)
	if err != nil {
		return nil, fmt.Errorf("error creating database session service: %w", err)
	}
	return &databaseService{db: db}, nil
}

// AutoMigrate runs the GORM auto-migration tool to ensure the database schema
// matches the internal storage models (e.g., storageSession, storageEvent).
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
func (s *databaseService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required")
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.NewString()
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
		updatedAt: time.Now(),
	}
	createdSession, err := createStorageSession(val)
	if err != nil {
		return nil, err
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		storageApp, err := fetchStorageAppState(tx, req.AppName)
		if err != nil {
			return fmt.Errorf("error on create session: %w", err)
		}
		storageUser, err := fetchStorageUserState(tx, req.AppName, req.UserID)
		if err != nil {
			return fmt.Errorf("error on create session: %w", err)
		}

		appDelta, userDelta, sessionState := extractStateDeltas(req.State)

		// apply state delta
		if len(appDelta) > 0 {
			maps.Copy(storageApp.State, appDelta)
			if err := tx.Save(&storageApp).Error; err != nil {
				return fmt.Errorf("failed to save app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			maps.Copy(storageUser.State, userDelta)
			if err := tx.Save(&storageUser).Error; err != nil {
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
func (s *databaseService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	// Ensure all parts of the composite key are provided.
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	var foundSession storageSession
	err := s.db.WithContext(ctx).
		Where(&storageSession{
			AppName: appName,
			UserID:  userID,
			ID:      sessionID,
		}).
		First(&foundSession).Error
	if err != nil {
		// For any error including ErrRecordNotFound, return it as a system error.
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

	// Order by timestamp DESC to get the most recent events when limiting
	eventQuery = eventQuery.Order("timestamp DESC")

	if req.NumRecentEvents > 0 {
		eventQuery = eventQuery.Limit(req.NumRecentEvents)
	}

	var storageEvents []storageEvent
	if err := eventQuery.Find(&storageEvents).Error; err != nil {
		// This is a system failure, not a "not found"
		return nil, fmt.Errorf("database error while fetching events: %w", err)
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

	var foundSessions []storageSession
	listQuery := s.db.WithContext(ctx).
		Where(&storageSession{
			AppName: appName,
		})

	if userID != "" {
		listQuery = listQuery.Where(&storageSession{
			UserID: userID,
		})
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
	if userID != "" {
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

	// Create response sessions, transform the storageSessions into
	responseSessions := make([]session.Session, 0, len(foundSessions))
	for _, storage := range foundSessions {
		s := storage
		sess, err := createSessionFromStorageSession(&s)
		if err != nil {
			// If we encounter a single mapping error, we fail the whole request.
			return nil, fmt.Errorf("failed to map storage object for session %s: %w", s.ID, err)
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
func (s *databaseService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", appName, userID, sessionID)
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		target := &storageSession{}

		result := tx.Where(&storageSession{
			AppName: req.AppName,
			UserID:  req.UserID,
			ID:      req.SessionID,
		}).Delete(target)

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

	// Trim temp state before persisting
	event = trimTempDeltaState(event)

	sess, ok := curSession.(*localSession)
	if !ok {
		return fmt.Errorf("unexpected session type %T", sess)
	}

	// applyChanges and persist them
	err := s.applyEvent(ctx, sess, event)
	if err != nil {
		return err
	}

	// append it to session
	return sess.appendEvent(event)
}

// applyEvent fetches the session, validates it, applies state changes from an
// event, and saves the event atomically.
func (s *databaseService) applyEvent(ctx context.Context, session *localSession, event *session.Event) error {
	// Wrap database operations in a single transaction.
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Fetch the session object from storage.
		var storageSess storageSession
		err := tx.Where(&storageSession{AppName: session.AppName(), UserID: session.UserID(), ID: session.ID()}).
			First(&storageSess).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return fmt.Errorf("session not found, cannot apply event")
			}
			return fmt.Errorf("failed to get session: %w", err)
		}

		// Ensure the session object is not stale.
		// We use UnixNano() for microsecond-level precision, matching the Python code.
		storageUpdateTime := storageSess.UpdateTime.UnixNano()
		sessionUpdateTime := session.updatedAt.UnixNano()
		if storageUpdateTime > sessionUpdateTime {
			return fmt.Errorf(
				"stale session error: last update time from request (%s) is older than in database (%s)",
				time.Unix(0, sessionUpdateTime).Format(time.RFC3339Nano),
				time.Unix(0, storageUpdateTime).Format(time.RFC3339Nano),
			)
		}

		// Fetch App and User states.
		storageApp, err := fetchStorageAppState(tx, session.AppName())
		if err != nil {
			return err
		}
		storageUser, err := fetchStorageUserState(tx, session.AppName(), session.UserID())
		if err != nil {
			return err
		}

		appDelta, userDelta, sessionDelta := extractStateDeltas(event.Actions.StateDelta)

		// Merge state deltas and update the storage objects.
		// GORM's .Save() method will correctly perform an INSERT or UPDATE.
		if len(appDelta) > 0 {
			maps.Copy(storageApp.State, appDelta)
			if err := tx.Save(&storageApp).Error; err != nil {
				return fmt.Errorf("failed to save app state: %w", err)
			}
		}
		if len(userDelta) > 0 {
			maps.Copy(storageUser.State, userDelta)
			if err := tx.Save(&storageUser).Error; err != nil {
				return fmt.Errorf("failed to save user state: %w", err)
			}
		}
		if len(sessionDelta) > 0 {
			maps.Copy(storageSess.State, sessionDelta)
			// The session state update will be saved along with the event timestamp update.
		}

		// Create the new event record in the database.
		storageEv, err := createStorageEvent(session, event)
		if err != nil {
			return fmt.Errorf("failed to map event to storage model: %w", err)
		}
		if err := tx.Create(storageEv).Error; err != nil {
			return fmt.Errorf("failed to save event: %w", err)
		}

		storageSess.UpdateTime = event.Timestamp
		// Save the session to update its state and UpdateTime.
		if err := tx.Save(&storageSess).Error; err != nil {
			return fmt.Errorf("failed to save session state: %w", err)
		}

		session.updatedAt = storageSess.UpdateTime

		return nil // Returning nil commits the transaction.
	})

	return err
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
// Temporary keys (starting with TempStatePrefix) are ignored.
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
