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
	// Create creates and returns a new session.
	// If the session already exists, it returns an error.
	Create(ctx context.Context, req *SessionCreateRequest) (*Session, error)
	// Get returns the requested session.
	// It returns an error if the requested session does not exist.
	Get(ctx context.Context, req *SessionGetRequest) (*Session, error)
	// List lists the requested sessions.
	// It returns an empty list if no session matches.
	List(ctx context.Context, req *SessionListRequest) ([]*Session, error)
	// Delete deletes the requested session.
	// It reports an error if the requested session does not exist.
	Delete(ctx context.Context, req *SessionDeleteRequest) error

	// AppendEvent appends the event to the session object.
	// The change is reflected both in the session storage and
	// the provided Session object's Events field.
	// If the event is marked as partial, it is a no-op.
	AppendEvent(ctx context.Context, session *Session, ev *Event) error
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
