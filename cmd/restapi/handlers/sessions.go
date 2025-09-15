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

package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/restapi/models"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

// SessionsAPIController is the controller for the Sessions API.
type SessionsAPIController struct {
	service sessionservice.Service
}

// NewSessionsAPIController creates a new SessionsAPIController.
func NewSessionsAPIController(service sessionservice.Service) *SessionsAPIController {
	return &SessionsAPIController{service: service}
}

// DeleteSession handles deleting a specific session.
func (c *SessionsAPIController) CreateSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	createSessionRequest := models.CreateSessionRequest{}
	// No state and no events, fails to decode req.Body failing with "EOF"
	if req.ContentLength > 0 {
		err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
		if err != nil {
			return NewStatusError(err, http.StatusBadRequest)
		}
	}
	session, err := c.service.Create(req.Context(), &sessionservice.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     createSessionRequest.State,
	})
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	for _, event := range createSessionRequest.Events {
		err = c.service.AppendEvent(req.Context(), session.Session, models.ToSessionEvent(event))
		if err != nil {
			return NewStatusError(err, http.StatusInternalServerError)
		}
	}
	respSession, err := models.FromSession(session.Session)
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	json.NewEncoder(rw).Encode(respSession)
	return nil
}

func (c *SessionsAPIController) DeleteSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	err := c.service.Delete(req.Context(), &sessionservice.DeleteRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	return nil
}

// GetSession retrieves a specific session by its ID.
func (c *SessionsAPIController) GetSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	session, err := c.service.Get(req.Context(), &sessionservice.GetRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	respSession, err := models.FromSession(session.Session)
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	json.NewEncoder(rw).Encode(respSession)
	return nil
}

// ListSessions handles listing all sessions for a given app and user.
func (c *SessionsAPIController) ListSessions(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	resp, err := c.service.List(req.Context(), &sessionservice.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return NewStatusError(err, http.StatusInternalServerError)
	}
	var sessions []models.Session
	for _, session := range resp.Sessions {
		respSession, err := models.FromSession(session)
		if err != nil {
			return NewStatusError(err, http.StatusInternalServerError)
		}
		sessions = append(sessions, respSession)
	}
	json.NewEncoder(rw).Encode(sessions)
	return nil
}
