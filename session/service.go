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

package session

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrNotFound reports that the requested session does not exist.
//
// Every [Service] implementation wraps it from [Service.Get] and from
// [Service.AppendEvent] when the session named in the request is absent, so
// that a caller can tell a missing session from a storage failure:
//
//	if errors.Is(err, session.ErrNotFound) { … }
//
// No other method reports it, because no other method treats a missing session
// as a failure: [Service.Delete] is a no-op on one, and [Service.List] returns
// an empty result. A session that belongs to another user counts as missing:
// Get reports it as ErrNotFound and Delete is a no-op on it.
//
// It does not cover every error Get and AppendEvent can return. A request that
// fails validation and a backend that is simply unreachable stay ordinary
// errors, so errors.Is never reads "not valid" or "not working" as "not there".
//
// The REST layer relies on this to answer 404 rather than 500. Wrap it first,
// as fmt.Errorf("%w: …: %w", session.ErrNotFound, err).
var ErrNotFound = errors.New("session not found")

// ErrStaleSession reports that the session passed to [Service.AppendEvent] is
// stale because a newer update has been committed to storage since the session
// value was loaded or copied.
//
// Persistent [Service] implementations that enforce optimistic concurrency
// (such as session/database) wrap ErrStaleSession from [Service.AppendEvent]
// when the stored session's update time is newer than the caller's
// [Session.LastUpdateTime]:
//
//	if errors.Is(err, session.ErrStaleSession) { … }
//
// When AppendEvent returns ErrStaleSession, neither storage nor the caller's
// [Session] value is modified.
var ErrStaleSession = errors.New("stale session")

// Service is a session storage service.
//
// It provides a set of methods for managing sessions and events.
type Service interface {
	Create(context.Context, *CreateRequest) (*CreateResponse, error)
	Get(context.Context, *GetRequest) (*GetResponse, error)
	// List returns the sessions that match the request, ordered by last update
	// time ascending, so the last session is the most recently active one.
	// Sessions with equal update times are ordered by user ID, then by session
	// ID.
	List(context.Context, *ListRequest) (*ListResponse, error)
	Delete(context.Context, *DeleteRequest) error
	// AppendEvent is used to append an event to a session, and remove temporary state keys from the event.
	//
	// The session is identified by its AppName, UserID and ID, whatever the
	// concrete type of the [Session] value: a value returned by Create or Get,
	// a copy made with [CloneSession], or one the caller built. When storage
	// holds no such session, AppendEvent returns an error wrapping
	// [ErrNotFound]. A persistent implementation that detects a newer stored
	// update than the value's [Session.LastUpdateTime] returns an error
	// wrapping [ErrStaleSession]. On any error, neither storage nor the value
	// is modified, with one exception: when the State of a value the caller
	// built fails to Set a key, the event stays stored and AppendEvent returns
	// that error wrapped. On success, the value's state receives the event's
	// state delta and, for values returned by the service or by
	// [CloneSession], its events and last update time include the event.
	//
	// Two further obligations, both checked by the shared conformance suite in
	// session/sessiontestsuite, so an implementation that misses either goes
	// red there rather than failing quietly in production:
	//
	// An event arriving with no ID must end up with one. Events built as struct
	// literals by an agent or a tool never pass through [NewEvent] and arrive
	// unnamed, and a stored event that cannot be named cannot be referred to by
	// anything that identifies events by ID.
	//
	// Where the implementation assigns the ID itself, it must be on the
	// caller's event, in place. Where the server assigns it, the ID appears on
	// the event when it is read back instead, and the conformance suite waives
	// the in-place half for those: session/vertexai is one, and it does not
	// assign in place. What holds everywhere is that the stored event has an
	// ID, which the suite checks on the read either way.
	//
	// [EventActions.Compaction] must survive the round trip. A context
	// compaction summary carries its content only there: LLMResponse.Content is
	// nil and there is no state or artifact delta, so a backend that decides
	// what to persist by looking at content or deltas drops it without
	// complaint. The session then comes back with no summary and no record that
	// compaction ran, and the same range is summarized and billed again on
	// every later turn.
	AppendEvent(context.Context, Session, *Event) error
}

// InMemoryService returns an in-memory implementation of the session service.
func InMemoryService() Service {
	return &inMemoryService{
		appState:  make(map[string]stateMap),
		userState: make(map[string]map[string]stateMap),
	}
}

// CreateRequest represents a request to create a session.
type CreateRequest struct {
	AppName string
	UserID  string
	// SessionID is the client-provided ID of the session to create.
	// Optional: if not set, it will be autogenerated.
	SessionID string
	// State is the initial state of the session.
	State map[string]any
}

// CreateResponse represents a response for newly created session.
type CreateResponse struct {
	Session Session
}

// GetRequest represents a request to get a session.
type GetRequest struct {
	AppName   string
	UserID    string
	SessionID string

	// NumRecentEvents returns at most NumRecentEvents most recent events.
	// Optional: if zero, the filter is not applied.
	NumRecentEvents int
	// After returns events with timestamp >= the given time.
	// Optional: if zero, the filter is not applied.
	After time.Time
}

// GetResponse represents a response from [Service.Get].
type GetResponse struct {
	Session Session
}

// ListRequest represents a request to list sessions.
type ListRequest struct {
	AppName string
	UserID  string
}

// ListResponse represents a response from [Service.List].
type ListResponse struct {
	// Sessions is ordered by last update time ascending, ties broken by user
	// ID and then session ID. The last element is the most recently active.
	Sessions []Session
}

// DeleteRequest represents a request to delete a session.
type DeleteRequest struct {
	AppName   string
	UserID    string
	SessionID string
}

// UserStateReader is an optional interface implemented by [Service]
// implementations that can read user-scoped state for an (AppName, UserID)
// pair without an active session.
//
// Implementations must be safe for concurrent use by multiple goroutines.
type UserStateReader interface {
	// GetUserState returns a detached snapshot of the user-scoped state for
	// (req.AppName, req.UserID), with keys stripped of [KeyPrefixUser].
	GetUserState(ctx context.Context, req *GetUserStateRequest) (*GetUserStateResponse, error)
}

// GetUserStateRequest represents a request to read user-scoped state.
type GetUserStateRequest struct {
	AppName string
	UserID  string
}

// GetUserStateResponse represents a response from [GetUserState] and
// [UserStateReader.GetUserState].
type GetUserStateResponse struct {
	// State holds the user-scoped key-value pairs for (AppName, UserID)
	// without [KeyPrefixUser]. On a nil error, State is always non-nil
	// (an empty map when no user state is stored) and shares no mutable
	// memory with the service.
	State map[string]any
}

// GetUserState reads the user-scoped state for (req.AppName, req.UserID) from svc
// without requiring a session.
//
// If svc does not implement [UserStateReader], GetUserState returns an error
// wrapping [errors.ErrUnsupported]. Callers that need a fallback on unsupported
// backends can check errors.Is(err, errors.ErrUnsupported) and enumerate
// sessions via [Service.List] and [Service.Get].
func GetUserState(ctx context.Context, svc Service, req *GetUserStateRequest) (*GetUserStateResponse, error) {
	if svc == nil {
		return nil, errors.New("session: service is nil")
	}
	if req == nil {
		return nil, errors.New("session: get user state request is nil")
	}
	r, ok := svc.(UserStateReader)
	if !ok {
		return nil, fmt.Errorf("session: %T does not support GetUserState; list sessions and get each session to read merged state: %w", svc, errors.ErrUnsupported)
	}
	return r.GetUserState(ctx, req)
}
