package adk

import (
	"context"
	"time"
)

// SessionService abstracts the session storage.
type SessionService interface {
	Create(ctx context.Context, appName, userID string, opts *SessionCreateOption) (*Session, error)
	Get(ctx context.Context, appName, userID, sessionID string, opts *SessionGetOption) (*Session, error)
	List(ctx context.Context, appName, userID string, opts *SessionListOption) ([]*Session, error)
	Delete(ctx context.Context, appName, userID, sessionID string, opts *SessionDeleteOption) error
	AppendEvent(ctx context.Context, sessionID string, event *Event) error
}

// Session represents a series of interaction between a user and agents.
type Session struct {
	ID      string // Session ID
	AppName string
	UserID  string

	// backing storage (e.g. in-memory, vertex ai session service, ...)
	store SessionService
}

// AppendEvent appends the event to the session.
func (s *Session) AppendEvent(ctx context.Context, event *Event) error {
	// This corresponds to python SessionService.append_event.
	return s.store.AppendEvent(ctx, s.ID, event)
}

// SessionCreateOption is the option for SessionService's Create.
type SessionCreateOption struct {
	// If unset, the service will assign a new session ID.
	SessionID string
	// State is an optional field to configure the initial state of the session.
	State map[string]any
}

// SessionGetOption is the option for SessionService's Get.
type SessionGetOption struct {
	NumRecentEvents int
	After           time.Time
}

// SessionListOption is the option for SessionService's List.
type SessionListOption struct{}

// SessionDeleteOption is the option for SessionService's Delete.
type SessionDeleteOption struct{}
