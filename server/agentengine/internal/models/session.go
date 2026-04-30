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

package models

import (
	"google.golang.org/adk/session"
)

// CreateSessionRequest represents an request aligned with aiplatform standard
type CreateSessionRequest struct {
	ClassMethod string             `json:"class_method"`
	Input       CreateSessionInput `json:"input"`
}

// CreateSessionInput contains input parameters
type CreateSessionInput struct {
	UserID    string         `json:"user_id"`
	SessionID string         `json:"session_id,omitempty"`
	State     map[string]any `json:"state,omitempty"`
}

// CreateSessionResponse contains response
type CreateSessionResponse struct {
	Output SessionData `json:"output"`
}

// SessionData represents data about the session
type SessionData struct {
	UserID         string          `json:"user_id"`
	LastUpdateTime float64         `json:"last_update_time"`
	AppName        string          `json:"app_name"`
	ID             string          `json:"id"`
	State          map[string]any  `json:"state"`
	Events         []session.Event `json:"events"`
}

// ListSessionRequest represents an request aligned with aiplatform standard
type ListSessionRequest struct {
	ClassMethod string           `json:"class_method"`
	Input       ListSessionInput `json:"input"`
}

// ListSessionInput contains input parameters
type ListSessionInput struct {
	UserID string `json:"user_id"`
}

// ListSessionResponse contains response
type ListSessionResponse struct {
	Output Sessions `json:"output"`
}

// Sessions contains list of sessions
type Sessions struct {
	Sessions []SessionData `json:"sessions"`
}

// GetSessionRequest represents an request aligned with aiplatform standard
type GetSessionRequest struct {
	ClassMethod string          `json:"class_method"`
	Input       GetSessionInput `json:"input"`
}

// GetSessionInput contains input parameters
type GetSessionInput struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
}

// GetSessionResponse contains response
type GetSessionResponse struct {
	Output SessionData `json:"output"`
}

// DeleteSessionRequest represents an request aligned with aiplatform standard
type DeleteSessionRequest struct {
	ClassMethod string             `json:"class_method"`
	Input       DeleteSessionInput `json:"input"`
}

// DeleteSessionInput contains input parameters
type DeleteSessionInput struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
}

// DeleteSessionResponse contains response
type DeleteSessionResponse struct {
	Output string `json:"output"`
}

func FromSession(sess session.Session) SessionData {
	stateMap := make(map[string]any)
	for k, v := range sess.State().All() {
		stateMap[k] = v
	}

	evs := []session.Event{}
	for ev := range sess.Events().All() {
		evs = append(evs, *ev)
	}

	return SessionData{
		UserID:         sess.UserID(),
		LastUpdateTime: float64(sess.LastUpdateTime().UnixNano()) / 1e9, // converts nanosec to sec
		AppName:        sess.AppName(),
		ID:             sess.ID(),
		State:          stateMap,
		Events:         evs,
	}
}
