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
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/server/authz"
	"google.golang.org/adk/v2/session"
)

// TODO: Confirm error handling and target semantic for REST API.

// SessionsAPIController is the controller for the Sessions API.
type SessionsAPIController struct {
	service    session.Service
	authorizer authz.Authorizer
}

// NewSessionsAPIController creates a new SessionsAPIController.
func NewSessionsAPIController(service session.Service) *SessionsAPIController {
	return &SessionsAPIController{service: service, authorizer: authz.NewNoop()}
}

// WithAuthorizer sets the authorizer for the Controller. Provided in order not to change NewSessionsAPIController.
func (c *SessionsAPIController) WithAuthorizer(authorizer authz.Authorizer) {
	c.authorizer = authorizer
}

// CreateSessionHandler is an HTTP handler for the create session API.
func (c *SessionsAPIController) CreateSessionHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(params)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), sessionID.UserID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
	}

	createSessionRequest := models.CreateSessionRequest{}
	if req.Body != nil {
		err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
		if err != nil && !errors.Is(err, io.EOF) {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
	}
	// Convert every event before creating anything, so an event the request
	// cannot carry is a 400 with no session left behind, rather than a 500
	// after the session is already stored.
	events := make([]*session.Event, 0, len(createSessionRequest.Events))
	for i, event := range createSessionRequest.Events {
		sessionEvent, err := models.ToSessionEvent(event)
		if err != nil {
			http.Error(rw, fmt.Sprintf("events[%d]: %v", i, err), http.StatusBadRequest)
			return
		}
		events = append(events, sessionEvent)
	}
	respSession, err := c.createSession(req.Context(), sessionID, createSessionRequest.State, events)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(respSession, http.StatusOK, rw)
}

func (c *SessionsAPIController) createSession(ctx context.Context, sessionID models.SessionID, state map[string]any, events []*session.Event) (models.Session, error) {
	session, err := c.service.Create(ctx, &session.CreateRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
		State:     state,
	})
	if err != nil {
		return models.Session{}, err
	}
	for _, event := range events {
		err = c.service.AppendEvent(ctx, session.Session, event)
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
	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), sessionID.UserID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
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

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), sessionID.UserID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
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
	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), sessionID.UserID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
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

	// Reject scoped keys before they reach the service. A state delta is
	// split by prefix on the way to storage: "app:" lands in app-wide state
	// shared by every user, and "user:" in state shared by every session that
	// user owns. Passing a client's delta through unfiltered would make this
	// the one REST route that writes outside the session it names — a caller
	// could PATCH their own session and change what every other user of the
	// app sees.
	//
	// What the client echoes back is not a write, so drop that first. See
	// dropEchoedKeys.
	stateDelta := dropEchoedKeys(updateRequest.StateDelta, storedSession.Session.State())
	if scoped := scopedStateKeys(stateDelta); len(scoped) > 0 {
		http.Error(rw, fmt.Sprintf(
			"stateDelta may only contain session-scoped keys; remove %v (the %q, %q and %q prefixes write outside this session)",
			scoped, session.KeyPrefixApp, session.KeyPrefixUser, session.KeyPrefixTemp),
			http.StatusBadRequest)
		return
	}

	// An empty or absent delta is a no-op: return the session as it stands
	// rather than record an event that changes nothing.
	if len(stateDelta) > 0 {
		event := session.NewEvent(req.Context(), platform.NewUUID(req.Context()))
		event.Author = "user"
		event.Actions.StateDelta = stateDelta
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

	if c.authorizer != nil {
		if err := c.authorizer.CanActAsUser(req.Context(), sessionID.UserID); err != nil {
			authz.WriteHTTPStatusForAuthError(rw, err)
			return
		}
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

// scopedStateKeys returns the keys of delta that are not session-scoped, in a
// stable order. See [SessionsAPIController.UpdateSessionHandler].
func scopedStateKeys(delta map[string]any) []string {
	var scoped []string
	for key := range delta {
		if isScopedStateKey(key) {
			scoped = append(scoped, key)
		}
	}
	slices.Sort(scoped)
	return scoped
}

// isScopedStateKey reports whether key writes outside the session that names
// it: "app:" into state shared by every user of the app, "user:" into state
// shared by every session that user owns, "temp:" into per-invocation state.
func isScopedStateKey(key string) bool {
	return strings.HasPrefix(key, session.KeyPrefixApp) ||
		strings.HasPrefix(key, session.KeyPrefixUser) ||
		strings.HasPrefix(key, session.KeyPrefixTemp)
}

// dropEchoedKeys returns delta without the keys a client sent back as a read
// returned them: scoped keys, and keys whose value a read returns with
// credential secrets stripped.
//
// The web UI's session list renames a session by posting back the whole state
// map it read, with one metadata key changed. A read merges app- and
// user-scoped state in and strips credential secrets, so that payload carries
// scoped keys the user never touched and credentials without their secrets.
// Refusing the scoped keys breaks rename for every agent that uses scoped
// state, and writing the redacted credentials back overwrites the stored
// secrets.
//
// Dropping such a key leaves the session as it stands. A scoped key carrying a
// different value is still a write outside this session, and is still
// refused. Every other key is written as submitted, except that a value the
// client sent back without the credential secrets a read stripped gets those
// omitted secrets restored.
func dropEchoedKeys(delta map[string]any, stored session.ReadonlyState) map[string]any {
	read := models.RedactedState(stored)
	kept := make(map[string]any, len(delta))
	for key, value := range delta {
		readValue, ok := read[key]
		if ok && sameJSON(readValue, value) && (isScopedStateKey(key) || strippedOnRead(stored, key, readValue)) {
			continue
		}
		// A value that a read returned with credential secrets stripped is not
		// an echo: the client edited something in it and sent it back without
		// the secrets it never saw. Restore those before writing it.
		if ok && !isScopedStateKey(key) && strippedOnRead(stored, key, readValue) {
			if storedValue, err := stored.Get(key); err == nil {
				if merged, restored, err := models.MergeRedactedValue(storedValue, value); err == nil && restored {
					value = merged
				}
			}
		}
		kept[key] = value
	}
	return kept
}

// strippedOnRead reports whether readValue, the value a read returns for key,
// lacks credential secrets that the stored value holds.
func strippedOnRead(stored session.ReadonlyState, key string, readValue json.RawMessage) bool {
	value, err := stored.Get(key)
	return err == nil && !sameJSON(readValue, value)
}

// sameJSON reports whether encoded and value are the same JSON value once both
// are decoded the way a request body is. A number then compares as the float64
// a client sends back: an echo matches a stored 1.0, 1e3 or 9007199254740993,
// which a read writes as it was stored, and an agent's int64(5) matches the
// float64(5) the client returns. A value that will not encode or decode counts
// as different, so the caller treats it as a write.
func sameJSON(encoded json.RawMessage, value any) bool {
	b, err := json.Marshal(value)
	if err != nil {
		return false
	}
	var want, got any
	if json.Unmarshal(encoded, &want) != nil || json.Unmarshal(b, &got) != nil {
		return false
	}
	return reflect.DeepEqual(want, got)
}
