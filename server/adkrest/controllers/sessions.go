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

package controllers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

// TODO: Confirm error handling and target semantic for REST API.

// SessionsAPIController is the controller for the Sessions API.
type SessionsAPIController struct {
	service session.Service
}

// NewSessionsAPIController creates a new SessionsAPIController.
func NewSessionsAPIController(service session.Service) *SessionsAPIController {
	return &SessionsAPIController{service: service}
}

// CreateSessionHandler is an HTTP handler for the create session API.
func (c *SessionsAPIController) CreateSessionHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	createSessionRequest := models.CreateSessionRequest{}
	if req.Body != nil {
		err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
		if err != nil && !errors.Is(err, io.EOF) {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
	}
	respSession, err := c.createSession(req.Context(), sessionID, createSessionRequest)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(respSession, http.StatusOK, rw)
}

func (c *SessionsAPIController) createSession(ctx context.Context, sessionID models.SessionID, createSessionRequest models.CreateSessionRequest) (models.Session, error) {
	session, err := c.service.Create(ctx, &session.CreateRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
		State:     createSessionRequest.State,
	})
	if err != nil {
		return models.Session{}, err
	}
	for _, event := range createSessionRequest.Events {
		err = c.service.AppendEvent(ctx, session.Session, models.ToSessionEvent(event))
		if err != nil {
			return models.Session{}, err
		}
	}
	return models.FromSession(session.Session)
}

// DeleteSessionHandler handles deleting a specific session.
func (c *SessionsAPIController) DeleteSessionHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}

	err = c.service.Delete(req.Context(), &session.DeleteRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
	})
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(nil, http.StatusOK, rw)
}

// GetSessionHandler retrieves a specific session by its ID.
func (c *SessionsAPIController) GetSessionHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	storedSession, err := c.service.Get(req.Context(), &session.GetRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
	})
	if err != nil {
		writeSessionServiceError(rw, err)
		return
	}
	session, err := models.FromSession(storedSession.Session)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(session, http.StatusOK, rw)
}

// UpdateSessionHandler applies a state delta to an existing session and returns
// the updated session.
//
// The ADK web UI PATCHes this route to rename a session — the new name is
// written to state as __session_metadata__.displayName — and to edit session
// state by hand. The delta is applied through [session.Service] by appending an
// event carrying it in Actions.StateDelta, the same path an agent turn takes,
// so every backend persists and scopes the change identically.
func (c *SessionsAPIController) UpdateSessionHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}

	updateRequest := models.UpdateSessionRequest{}
	if req.Body != nil {
		err := json.NewDecoder(req.Body).Decode(&updateRequest)
		if err != nil && !errors.Is(err, io.EOF) {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
	}

	getRequest := &session.GetRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
	}
	storedSession, err := c.service.Get(req.Context(), getRequest)
	if err != nil {
		writeSessionServiceError(rw, err)
		return
	}

	// Reject scoped keys before they reach the service. A state delta is
	// split by prefix on the way to storage: "app:" lands in app-wide state
	// shared by every user, and "user:" in state shared by every session that
	// user owns. Passing a client's delta through unfiltered would make this
	// the one REST route that writes outside the session it names — a caller
	// could PATCH their own session and change what every other user of the
	// app sees.
	//
	// What the client echoes back is not a write, so drop that first. See
	// dropEchoedScopedKeys.
	stateDelta := dropEchoedScopedKeys(updateRequest.StateDelta, storedSession.Session.State())
	if scoped := scopedStateKeys(stateDelta); len(scoped) > 0 {
		http.Error(rw, fmt.Sprintf(
			"stateDelta may only contain session-scoped keys; remove %v (the %q, %q and %q prefixes write outside this session)",
			scoped, session.KeyPrefixApp, session.KeyPrefixUser, session.KeyPrefixTemp),
			http.StatusBadRequest)
		return
	}

	// An empty or absent delta is a no-op: return the session as it stands
	// rather than record an event that changes nothing.
	if len(stateDelta) > 0 {
		event := session.NewEvent(req.Context(), platform.NewUUID(req.Context()))
		event.Author = "user"
		event.Actions.StateDelta = stateDelta
		if err := c.service.AppendEvent(req.Context(), storedSession.Session, event); err != nil {
			writeSessionServiceError(rw, err)
			return
		}
		// Re-read: app- and user-scoped keys are merged back in only on read,
		// so this is what a follow-up GET would show.
		storedSession, err = c.service.Get(req.Context(), getRequest)
		if err != nil {
			writeSessionServiceError(rw, err)
			return
		}
	}

	respSession, err := models.FromSession(storedSession.Session)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(respSession, http.StatusOK, rw)
}

// writeSessionServiceError answers a [session.Service] failure. A session the
// service cannot find means the client asked for something that is not there,
// which is a 404; anything else is the server's fault.
func writeSessionServiceError(rw http.ResponseWriter, err error) {
	if errors.Is(err, session.ErrNotFound) {
		http.Error(rw, err.Error(), http.StatusNotFound)
		return
	}
	http.Error(rw, err.Error(), http.StatusInternalServerError)
}

// ListSessionsHandler handles listing all sessions for a given app and user.
func (c *SessionsAPIController) ListSessionsHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	// Not `var sessions []models.Session`: a nil slice encodes as JSON null,
	// and clients expect an empty list for a user with no sessions.
	sessions := []models.Session{}
	resp, err := c.service.List(req.Context(), &session.ListRequest{
		AppName: sessionID.AppName,
		UserID:  sessionID.UserID,
	})
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, session := range resp.Sessions {
		respSession, err := models.FromSession(session)
		if err != nil {
			http.Error(rw, err.Error(), http.StatusInternalServerError)
			return
		}
		sessions = append(sessions, respSession)
	}
	EncodeJSONResponse(sessions, http.StatusOK, rw)
}

// scopedStateKeys returns the keys of delta that are not session-scoped, in a
// stable order. See [SessionsAPIController.UpdateSessionHandler].
func scopedStateKeys(delta map[string]any) []string {
	var scoped []string
	for key := range delta {
		if isScopedStateKey(key) {
			scoped = append(scoped, key)
		}
	}
	slices.Sort(scoped)
	return scoped
}

// isScopedStateKey reports whether key writes outside the session that names
// it: "app:" into state shared by every user of the app, "user:" into state
// shared by every session that user owns, "temp:" into per-invocation state.
func isScopedStateKey(key string) bool {
	return strings.HasPrefix(key, session.KeyPrefixApp) ||
		strings.HasPrefix(key, session.KeyPrefixUser) ||
		strings.HasPrefix(key, session.KeyPrefixTemp)
}

// dropEchoedScopedKeys returns delta without the scoped keys whose submitted
// value already matches what the session holds.
//
// The web UI's session list renames a session by posting back the whole state
// map it read, with one metadata key changed. A read merges app- and
// user-scoped state in, so that payload carries scoped keys the user never
// touched. Refusing it breaks rename for every agent that uses scoped state,
// and the UI discards the error, so the name silently does not change.
//
// A key echoed back unchanged would write the value that is already there, so
// dropping it leaves the session exactly as accepting it would. That is the
// whole exception: a scoped key carrying a different value is still a write
// outside this session, and is still refused.
//
// Values are compared as JSON rather than with reflect.DeepEqual, because the
// submitted value has been through a JSON round-trip and the stored one has
// not. An agent that wrote int64(5) gets float64(5) back from the client, and
// those are not equal in Go.
func dropEchoedScopedKeys(delta map[string]any, stored session.ReadonlyState) map[string]any {
	kept := make(map[string]any, len(delta))
	for key, value := range delta {
		if isScopedStateKey(key) && matchesStoredValue(stored, key, value) {
			continue
		}
		kept[key] = value
	}
	return kept
}

// matchesStoredValue reports whether stored already holds value under key. A
// key the session does not have, or a value that will not marshal, counts as a
// change: the caller then treats it as a write and refuses it.
func matchesStoredValue(stored session.ReadonlyState, key string, value any) bool {
	if stored == nil {
		return false
	}
	current, err := stored.Get(key)
	if err != nil {
		return false
	}
	encodedCurrent, err := json.Marshal(current)
	if err != nil {
		return false
	}
	encodedValue, err := json.Marshal(value)
	if err != nil {
		return false
	}
	return bytes.Equal(encodedCurrent, encodedValue)
}
