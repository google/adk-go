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
	"net/http"

	"google.golang.org/adk/sessionservice"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

// SessionsAPIController is the controller for the Sessions API.
type SessionsAPIController struct {
	service sessionservice.Service
}

// New creates a new SessionsAPIController.
func New(service sessionservice.Service) *SessionsAPIController {
	return &SessionsAPIController{service: service}
}

// CreateSession creates a new session and appends events if provided.
func (c *SessionsAPIController) CreateSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// DeleteSession handles deleting a specific session.
func (*SessionsAPIController) DeleteSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// GetSession retrieves a specific session by its ID.
func (*SessionsAPIController) GetSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// ListSessions handles listing all sessions for a given app and user.
func (*SessionsAPIController) ListSessions(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
