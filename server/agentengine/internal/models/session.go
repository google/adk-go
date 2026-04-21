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

type CreateSessionRequest struct {
	ClassMethod string             `json:"class_method"`
	Input       CreateSessionInput `json:"input"`
}

type CreateSessionInput struct {
	UserID string         `json:"user_id"`
	State  map[string]any `json:"state,omitempty"`
}

type CreateSessionResponse struct {
	Output SessionData `json:"output"`
}

type SessionData struct {
	UserID         string         `json:"user_id"`
	LastUpdateTime float64        `json:"last_update_time"`
	AppName        string         `json:"app_name"`
	ID             string         `json:"id"`
	State          map[string]any `json:"state"`
	Events         session.Events `json:"events"`
}

type ListSessionRequest struct {
	ClassMethod string           `json:"class_method"`
	Input       ListSessionInput `json:"input"`
}

type ListSessionInput struct {
	UserID string `json:"user_id"`
}

// data: {"output": {"sessions": [{"lastUpdateTime": 1775728518.103596,"id": "1121075662136803328","appName": "app","userId": "u_12345","state": {},"events": []}]}}
type ListSessionResponse struct {
	Data ListSessionData `json:"data"`
}

type ListSessionData struct {
	Output Sessions `json:"output"`
}

type Sessions struct {
	Sessions []SessionData `json:"sessions"`
}

type GetSessionRequest struct {
	ClassMethod string          `json:"class_method"`
	Input       GetSessionInput `json:"input"`
}

type GetSessionInput struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
}

type GetSessionResponse struct {
	Output SessionData `json:"output"`
}

type DeleteSessionRequest struct {
	ClassMethod string             `json:"class_method"`
	Input       DeleteSessionInput `json:"input"`
}

type DeleteSessionInput struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
}

type DeleteSessionResponse struct {
	Output SessionData `json:"output"`
}
