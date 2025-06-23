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

package adk

import (
	"context"
	"time"
)

// SessionService abstracts the session storage.
type SessionService interface {
	Create(ctx context.Context, req *SessionCreateRequest) (*Session, error)
	Get(ctx context.Context, req *SessionGetRequest) (*Session, error)
	List(ctx context.Context, req *SessionListRequest) ([]*Session, error)
	Delete(ctx context.Context, req *SessionDeleteRequest) error
	AppendEvent(ctx context.Context, req *SessionAppendEventRequest) error
}

// Session represents a series of interaction between a user and agents.
type Session struct {
	ID      string // Session ID
	AppName string
	UserID  string

	Events []*Event
}

// SessionCreateRequest is the request for SessionService's Create.
type SessionCreateRequest struct {
	// Required.
	AppName, UserID string

	// If unset, the service will assign a new session ID.
	SessionID string
	// State is an optional field to configure the initial state of the session.
	State map[string]any
}

// SessionGetRequest is the request for SessionService's Get.
type SessionGetRequest struct {
	// Required.
	AppName, UserID, SessionID string

	// Optional fields.
	NumRecentEvents int
	After           time.Time
}

// SessionListRequest is the request for SessionService's List.
type SessionListRequest struct {
	// App name and user id. Required.
	AppName, UserID string
}

// SessionDeleteRequest is the request for SessionService's Delete.
type SessionDeleteRequest struct {
	// Identifies a unique session object. Required.
	AppName, UserID, SessionID string
}

// SessionAppendEventRequest is the request for SesssionService's AppendEvent.
type SessionAppendEventRequest struct {
	// Required.
	Session *Session // TODO: why not just AppName/UserID/SessionID?
	Event   *Event
}
