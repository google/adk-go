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
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

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

	// An empty or absent delta is a no-op: return the session as it stands
	// rather than record an event that changes nothing.
	if len(updateRequest.StateDelta) > 0 {
		event := session.NewEvent(req.Context(), platform.NewUUID(req.Context()))
		event.Author = "user"
		event.Actions.StateDelta = updateRequest.StateDelta
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
