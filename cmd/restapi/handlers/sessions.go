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
	"google.golang.org/adk/cmd/restapi/errors"
	"google.golang.org/adk/cmd/restapi/models"
	"google.golang.org/adk/cmd/restapi/utils"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

type SessionsApiController struct {
	service sessionservice.Service
}

func NewSessionsApiController(service sessionservice.Service) *SessionsApiController {
	return &SessionsApiController{service: service}
}

func (c *SessionsApiController) CreateSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	createSessionRequest := models.CreateSessionRequest{}
	// No state and no events, fails to decode req.Body failing with "EOF"
	if req.ContentLength > 0 {
		err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
		if err != nil {
			return errors.NewStatusError(err, http.StatusBadRequest)
		}
	}
	session, err := c.service.Create(req.Context(), &sessionservice.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     createSessionRequest.State,
	})
	if err != nil {
		return errors.NewStatusError(fmt.Errorf("create session: %w", err), http.StatusInternalServerError)
	}
	resp, err := models.FromSession(session.Session)
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	err = json.NewEncoder(rw).Encode(resp)
	if err != nil {
		return errors.NewStatusError(fmt.Errorf("encode response: %w", err), http.StatusInternalServerError)
	}
	for _, event := range createSessionRequest.Events {
		err = c.service.AppendEvent(req.Context(), session.Session, models.ToSessionEvent(event))
		if err != nil {
			return errors.NewStatusError(err, http.StatusInternalServerError)
		}
	}
	respSession, err := models.FromSession(session.Session)
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	utils.EncodeJSONResponse(respSession, http.StatusOK, rw)
	return nil
}

func (c *SessionsApiController) DeleteSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return errors.NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	err := c.service.Delete(req.Context(), &sessionservice.DeleteRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	rw.WriteHeader(http.StatusOK)
	return nil
}

// GetSession handles receiving a sesion from the system.
func (c *SessionsApiController) GetSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return errors.NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	session, err := c.service.Get(req.Context(), &sessionservice.GetRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	respSession, err := models.FromSession(session.Session)
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	utils.EncodeJSONResponse(respSession, http.StatusOK, rw)
	return nil
}

func (c *SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	resp, err := c.service.List(req.Context(), &sessionservice.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	sessions := []models.Session{}
	for _, session := range resp.Sessions {
		respSession, err := models.FromSession(session)
		if err != nil {
			return errors.NewStatusError(err, http.StatusInternalServerError)
		}
		sessions = append(sessions, respSession)
	}
	utils.EncodeJSONResponse(sessions, http.StatusOK, rw)
	return nil
}
