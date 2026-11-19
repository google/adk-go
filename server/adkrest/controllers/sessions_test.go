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

package controllers_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

func TestGetSession(t *testing.T) {
	id := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	tc := []struct {
		name           string
		storedSessions map[fakes.SessionKey]fakes.TestSession
		sessionID      fakes.SessionKey
		wantSession    models.Session
		wantErr        error
		wantStatus     int
	}{
		{
			name: "session exists",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: id,
			wantSession: models.Session{
				ID:        "testSession",
				AppName:   "testApp",
				UserID:    "testUser",
				UpdatedAt: time.Now().Unix(),
				Events:    []models.Event{},
				State: map[string]any{
					"foo": "bar",
				},
			},
			wantStatus: http.StatusOK,
		},
		{
			// The fake wraps session.ErrNotFound for a missing session, as the
			// real services do, so the handler answers 404. A service failure
			// that is not a missing session stays a 500: see
			// TestGetSessionServiceFailureIsInternalError.
			name:           "missing session is not found",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{},
			sessionID:      id,
			wantErr:        fmt.Errorf("session not found: %q", id.SessionID),
			wantStatus:     http.StatusNotFound,
		},
		{
			name: "user ID is missing in input",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: fakes.SessionKey{
				AppName:   "testApp",
				SessionID: "testSession",
			},
			wantErr:    fmt.Errorf("user_id parameter is required"),
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "session ID is missing",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id: fakes.SessionKey{
						AppName: "testApp",
						UserID:  "testUser",
					},
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID:  id,
			wantErr:    fmt.Errorf("session_id is empty in received session"),
			wantStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := fakes.FakeSessionService{Sessions: tt.storedSessions}
			apiController := controllers.NewSessionsAPIController(&sessionService)
			req, err := http.NewRequest(http.MethodGet, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: "testUser"}))
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			apiController.GetSessionHandler(rr, req)

			if status := rr.Code; status != tt.wantStatus {
				t.Fatalf("handler returned wrong status code: got %v want %v", status, tt.wantStatus)
			}
			if tt.wantErr != nil {
				respErr := strings.Trim(rr.Body.String(), "\n")
				if tt.wantErr.Error() != respErr {
					t.Errorf("CreateSession() mismatch (-want +got):\n%v, %v", tt.wantErr.Error(), respErr)
				}
				return
			}
			var gotSession models.Session
			err = json.NewDecoder(rr.Body).Decode(&gotSession)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSession, gotSession, EquateApproxInt(int64(time.Second))); diff != "" {
				t.Errorf("GetSession() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCreateSession(t *testing.T) {
	id := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	tc := []struct {
		name             string
		storedSessions   map[fakes.SessionKey]fakes.TestSession
		sessionID        fakes.SessionKey
		createRequestObj models.CreateSessionRequest
		wantSession      models.Session
		wantErr          error
		wantStatus       int
	}{
		{
			name: "session exists",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID:  id,
			wantErr:    fmt.Errorf("session already exists"),
			wantStatus: http.StatusInternalServerError,
		},
		{
			name:           "successful create operation",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{},
			sessionID:      id,
			createRequestObj: models.CreateSessionRequest{
				State: map[string]any{
					"foo": "bar",
				},
				Events: []models.Event{
					{
						ID:     "eventID",
						Author: "testUser",
					},
				},
			},
			wantSession: models.Session{
				ID:      "testSession",
				AppName: "testApp",
				UserID:  "testUser",
				State: map[string]any{
					"foo": "bar",
				},
				Events: []models.Event{
					{
						ID:     "eventID",
						Author: "testUser",
					},
				},
			},
			wantStatus: http.StatusOK,
		},
		{
			name:           "user id is missing",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{},
			sessionID: fakes.SessionKey{
				AppName:   "testApp",
				SessionID: "testSession",
			},
			createRequestObj: models.CreateSessionRequest{},
			wantStatus:       http.StatusBadRequest,
			wantErr:          fmt.Errorf("user_id parameter is required"),
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := fakes.FakeSessionService{Sessions: tt.storedSessions}
			apiController := controllers.NewSessionsAPIController(&sessionService)
			reqBytes, err := json.Marshal(tt.createRequestObj)
			if err != nil {
				t.Fatalf("marshal request: %v", err)
			}
			req, err := http.NewRequest(http.MethodPost, "/apps/testApp/users/testUser/sessions/testSession", bytes.NewBuffer(reqBytes))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: "testUser"}))

			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			apiController.CreateSessionHandler(rr, req)

			if status := rr.Code; status != tt.wantStatus {
				t.Errorf("handler returned wrong status code: got %v want %v", status, tt.wantStatus)
			}
			if tt.wantErr != nil {
				respErr := strings.Trim(rr.Body.String(), "\n")
				if tt.wantErr.Error() != respErr {
					t.Errorf("CreateSession() mismatch (-want +got):\n%v, %v", tt.wantErr.Error(), respErr)
				}
				return
			}
			var gotSession models.Session
			err = json.NewDecoder(rr.Body).Decode(&gotSession)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSession, gotSession, EquateApproxInt(int64(time.Second)),
				cmpopts.IgnoreFields(models.Session{}, "UpdatedAt")); diff != "" {
				t.Errorf("CreateSession() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCreateSessionDropsNilParts(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	sessionService := session.InMemoryService()
	apiController := controllers.NewSessionsAPIController(sessionService)
	rr := httptest.NewRecorder()

	// The event without content exercises the nil check in the handler.
	body := strings.NewReader(`{"events": [
		{"author": "other_agent"},
		{"author": "other_agent", "content": {"role": "model", "parts": [null, {"text": "It is sunny in Paris."}, null]}}
	]}`)
	apiController.CreateSessionHandler(rr, newSessionRequest(t, http.MethodPost, id, body, id.UserID))

	if rr.Code != http.StatusOK {
		t.Fatalf("CreateSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	stored, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   id.AppName,
		UserID:    id.UserID,
		SessionID: id.SessionID,
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	events := stored.Session.Events()
	if events.Len() != 2 {
		t.Fatalf("session has %d events, want 2", events.Len())
	}
	parts, err := json.Marshal(events.At(1).Content.Parts)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if want := `[{"text":"It is sunny in Paris."}]`; string(parts) != want {
		t.Errorf("stored parts = %s, want %s", parts, want)
	}
}

func TestDeleteSession(t *testing.T) {
	id := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	tc := []struct {
		name           string
		storedSessions map[fakes.SessionKey]fakes.TestSession
		sessionID      fakes.SessionKey
		wantStatus     int
	}{
		{
			name: "session exists",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID:  id,
			wantStatus: http.StatusOK,
		},
		{
			name:           "session does not exist",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{},
			sessionID:      id,
			wantStatus:     http.StatusInternalServerError,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := fakes.FakeSessionService{Sessions: tt.storedSessions}
			apiController := controllers.NewSessionsAPIController(&sessionService)
			req, err := http.NewRequest(http.MethodDelete, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: "testUser"}))

			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			apiController.DeleteSessionHandler(rr, req)
			if status := rr.Code; status != tt.wantStatus {
				t.Fatalf("handler returned wrong status code: got %v want %v", status, tt.wantStatus)
			}
			if _, ok := sessionService.Sessions[tt.sessionID]; ok {
				t.Errorf("session was not deleted")
			}
		})
	}
}

func TestListSessions(t *testing.T) {
	id := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}
	newSessionID := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "newSession",
	}
	oldSessionID := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "oldSession",
	}

	tc := []struct {
		name           string
		storedSessions map[fakes.SessionKey]fakes.TestSession
		wantSessions   []models.Session
		wantStatus     int
	}{
		{
			name: "session exists",
			storedSessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{"foo": "bar"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
				newSessionID: {
					Id:            newSessionID,
					SessionState:  fakes.TestState{"xyz": "abc"},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
				oldSessionID: {
					Id:            oldSessionID,
					SessionState:  fakes.TestState{},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			wantSessions: []models.Session{
				{
					ID:        "testSession",
					AppName:   "testApp",
					UserID:    "testUser",
					UpdatedAt: time.Now().Unix(),
					Events:    []models.Event{},
					State: map[string]any{
						"foo": "bar",
					},
				},
				{
					ID:        "newSession",
					AppName:   "testApp",
					UserID:    "testUser",
					UpdatedAt: time.Now().Unix(),
					Events:    []models.Event{},
					State: map[string]any{
						"xyz": "abc",
					},
				},
				{
					ID:        "oldSession",
					AppName:   "testApp",
					UserID:    "testUser",
					State:     map[string]any{},
					UpdatedAt: time.Now().Unix(),
					Events:    []models.Event{},
				},
			},
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := fakes.FakeSessionService{Sessions: tt.storedSessions}
			apiController := controllers.NewSessionsAPIController(&sessionService)
			req, err := http.NewRequest(http.MethodDelete, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: "testUser"}))
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, map[string]string{
				"app_name": "testApp",
				"user_id":  "testUser",
			})

			rr := httptest.NewRecorder()

			apiController.ListSessionsHandler(rr, req)
			if status := rr.Code; status != tt.wantStatus {
				t.Fatalf("handler returned wrong status code: got %v want %v", status, tt.wantStatus)
			}
			got := []models.Session{}
			err = json.NewDecoder(rr.Body).Decode(&got)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSessions, got, EquateApproxInt(int64(time.Second)), cmpopts.SortSlices(func(a, b models.Session) bool {
				return a.ID < b.ID
			})); diff != "" {
				t.Errorf("ListSessions() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGetSessionMissingSessionIsNotFound covers the 404 path. A service that
// wraps session.ErrNotFound is telling the handler the session is absent, not
// that the server broke, so the client must get a 404.
func TestGetSessionMissingSessionIsNotFound(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "missingSession"}
	apiController := controllers.NewSessionsAPIController(session.InMemoryService())
	rr := httptest.NewRecorder()

	apiController.GetSessionHandler(rr, newSessionRequest(t, http.MethodGet, id, nil, id.UserID))

	if rr.Code != http.StatusNotFound {
		t.Errorf("GetSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

// getFailingService fails Get with a plain error that does not wrap
// session.ErrNotFound.
type getFailingService struct {
	session.Service
}

func (getFailingService) Get(context.Context, *session.GetRequest) (*session.GetResponse, error) {
	return nil, errors.New("connection refused")
}

// TestGetSessionServiceFailureIsInternalError is the other side of the 404
// mapping: a Get failure that is not a missing session is the server's fault.
func TestGetSessionServiceFailureIsInternalError(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	apiController := controllers.NewSessionsAPIController(getFailingService{})
	rr := httptest.NewRecorder()

	apiController.GetSessionHandler(rr, newSessionRequest(t, http.MethodGet, id, nil, id.UserID))

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("GetSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusInternalServerError, rr.Body.String())
	}
}

// TestGetSessionExistingSessionIsOK is the counterpart: a session that is there
// is still a 200 after the not-found mapping.
func TestGetSessionExistingSessionIsOK(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	apiController := controllers.NewSessionsAPIController(newServiceWithSession(t, id, map[string]any{"foo": "bar"}))
	rr := httptest.NewRecorder()

	apiController.GetSessionHandler(rr, newSessionRequest(t, http.MethodGet, id, nil, id.UserID))

	if rr.Code != http.StatusOK {
		t.Fatalf("GetSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	var got models.Session
	if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.ID != id.SessionID {
		t.Errorf("GetSessionHandler() session ID = %q, want %q", got.ID, id.SessionID)
	}
}

// TestListSessionsWithoutSessionsEncodesEmptyArray asserts the raw bytes.
// Decoding accepts both null and [], so a test that decodes cannot tell them
// apart — but the web UI iterates the response and a null breaks it.
func TestListSessionsWithoutSessionsEncodesEmptyArray(t *testing.T) {
	sessionService := fakes.FakeSessionService{Sessions: map[fakes.SessionKey]fakes.TestSession{}}
	apiController := controllers.NewSessionsAPIController(&sessionService)
	req, err := http.NewRequest(http.MethodGet, "/apps/testApp/users/testUser/sessions", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req = req.WithContext(authn.WithCaller(req.Context(), &authn.Caller{UserID: "testUser"}))
	req = mux.SetURLVars(req, map[string]string{"app_name": "testApp", "user_id": "testUser"})
	rr := httptest.NewRecorder()

	apiController.ListSessionsHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("handler returned wrong status code: got %v want %v", rr.Code, http.StatusOK)
	}
	if got := strings.TrimRight(rr.Body.String(), "\n"); got != "[]" {
		t.Errorf("ListSessionsHandler() raw body = %q, want %q", got, "[]")
	}
}

func TestUpdateSession(t *testing.T) {
	id := fakes.SessionKey{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	tc := []struct {
		name          string
		createSession bool
		sessionID     fakes.SessionKey
		body          string
		wantStatus    int
		wantState     map[string]any
	}{
		{
			name:          "state delta is applied",
			createSession: true,
			sessionID:     id,
			body:          `{"stateDelta":{"foo":"baz","count":2}}`,
			wantStatus:    http.StatusOK,
			wantState:     map[string]any{"foo": "baz", "count": float64(2)},
		},
		{
			// The rename path: the UI stores the new name in session state.
			name:          "session metadata round-trips",
			createSession: true,
			sessionID:     id,
			body:          `{"stateDelta":{"__session_metadata__":{"displayName":"My renamed chat"}}}`,
			wantStatus:    http.StatusOK,
			wantState: map[string]any{
				"foo":                  "bar",
				"__session_metadata__": map[string]any{"displayName": "My renamed chat"},
			},
		},
		{
			name:          "empty delta leaves the session unchanged",
			createSession: true,
			sessionID:     id,
			body:          `{}`,
			wantStatus:    http.StatusOK,
			wantState:     map[string]any{"foo": "bar"},
		},
		{
			name:          "absent delta leaves the session unchanged",
			createSession: true,
			sessionID:     id,
			body:          `{"stateDelta":{}}`,
			wantStatus:    http.StatusOK,
			wantState:     map[string]any{"foo": "bar"},
		},
		{
			name:          "session does not exist",
			createSession: false,
			sessionID:     id,
			body:          `{"stateDelta":{"foo":"baz"}}`,
			wantStatus:    http.StatusNotFound,
		},
		{
			name:          "malformed json",
			createSession: true,
			sessionID:     id,
			body:          `{"stateDelta":`,
			wantStatus:    http.StatusBadRequest,
		},
		{
			name:          "user ID is missing in input",
			createSession: true,
			sessionID:     fakes.SessionKey{AppName: "testApp", SessionID: "testSession"},
			body:          `{"stateDelta":{"foo":"baz"}}`,
			wantStatus:    http.StatusBadRequest,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := session.InMemoryService()
			if tt.createSession {
				sessionService = newServiceWithSession(t, id, map[string]any{"foo": "bar"})
			}
			apiController := controllers.NewSessionsAPIController(sessionService)
			rr := httptest.NewRecorder()

			apiController.UpdateSessionHandler(rr, newSessionRequest(t, http.MethodPatch, tt.sessionID, strings.NewReader(tt.body), id.UserID))

			if rr.Code != tt.wantStatus {
				t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", rr.Code, tt.wantStatus, rr.Body.String())
			}
			if tt.wantStatus != http.StatusOK {
				return
			}

			var got models.Session
			if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantState, map[string]any(got.State)); diff != "" {
				t.Errorf("UpdateSession() response state mismatch (-want +got):\n%s", diff)
			}

			// A follow-up GET must show the same state: the delta has to reach
			// the store, not just the response body.
			getRR := httptest.NewRecorder()
			apiController.GetSessionHandler(getRR, newSessionRequest(t, http.MethodGet, tt.sessionID, nil, id.UserID))
			if getRR.Code != http.StatusOK {
				t.Fatalf("GetSessionHandler() status = %d, want %d; body: %s", getRR.Code, http.StatusOK, getRR.Body.String())
			}
			var reread models.Session
			if err := json.NewDecoder(getRR.Body).Decode(&reread); err != nil {
				t.Fatalf("decode get response: %v", err)
			}
			if diff := cmp.Diff(tt.wantState, map[string]any(reread.State)); diff != "" {
				t.Errorf("state after re-read mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestUpdateSessionAppliesDeltaThroughAnEvent guards the mechanism, not just
// the result: the delta must be applied through session.Service as an event, so
// that every backend records it the way an agent turn would.
func TestUpdateSessionAppliesDeltaThroughAnEvent(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	sessionService := newServiceWithSession(t, id, map[string]any{"foo": "bar"})
	apiController := controllers.NewSessionsAPIController(sessionService)
	rr := httptest.NewRecorder()

	body := strings.NewReader(`{"stateDelta":{"foo":"baz"}}`)
	apiController.UpdateSessionHandler(rr, newSessionRequest(t, http.MethodPatch, id, body, id.UserID))

	if rr.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	stored, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   id.AppName,
		UserID:    id.UserID,
		SessionID: id.SessionID,
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	events := stored.Session.Events()
	if events.Len() != 1 {
		t.Fatalf("session has %d events, want 1 carrying the delta", events.Len())
	}
	want := map[string]any{"foo": "baz"}
	if diff := cmp.Diff(want, events.At(0).Actions.StateDelta); diff != "" {
		t.Errorf("appended event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

// TestUpdateSessionWritesAnUnchangedPlainKey pins the edge of the echo
// exception: a session-scoped key holding no credential is written even when
// its value matches the stored one, so the PATCH records an event carrying it.
func TestUpdateSessionWritesAnUnchangedPlainKey(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	sessionService := newServiceWithSession(t, id, map[string]any{"theme": "dark"})
	apiController := controllers.NewSessionsAPIController(sessionService)
	rr := httptest.NewRecorder()

	body := strings.NewReader(`{"stateDelta":{"theme":"dark"}}`)
	apiController.UpdateSessionHandler(rr, newSessionRequest(t, http.MethodPatch, id, body, id.UserID))

	if rr.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	stored, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   id.AppName,
		UserID:    id.UserID,
		SessionID: id.SessionID,
	})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	events := stored.Session.Events()
	if events.Len() != 1 {
		t.Fatalf("session has %d events, want 1 carrying the unchanged key", events.Len())
	}
	if diff := cmp.Diff(map[string]any{"theme": "dark"}, events.At(0).Actions.StateDelta); diff != "" {
		t.Errorf("appended event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

// TestUpdateSessionResponseMatchesGetSession pins the response shape: the UI
// reuses the PATCH response as if it came from GET.
func TestUpdateSessionResponseMatchesGetSession(t *testing.T) {
	id := fakes.SessionKey{AppName: "testApp", UserID: "testUser", SessionID: "testSession"}
	apiController := controllers.NewSessionsAPIController(newServiceWithSession(t, id, map[string]any{"foo": "bar"}))

	patchRR := httptest.NewRecorder()
	body := strings.NewReader(`{"stateDelta":{"foo":"baz"}}`)
	apiController.UpdateSessionHandler(patchRR, newSessionRequest(t, http.MethodPatch, id, body, id.UserID))
	if patchRR.Code != http.StatusOK {
		t.Fatalf("UpdateSessionHandler() status = %d, want %d; body: %s", patchRR.Code, http.StatusOK, patchRR.Body.String())
	}
	var patched models.Session
	if err := json.NewDecoder(patchRR.Body).Decode(&patched); err != nil {
		t.Fatalf("decode patch response: %v", err)
	}

	getRR := httptest.NewRecorder()
	apiController.GetSessionHandler(getRR, newSessionRequest(t, http.MethodGet, id, nil, id.UserID))
	if getRR.Code != http.StatusOK {
		t.Fatalf("GetSessionHandler() status = %d, want %d; body: %s", getRR.Code, http.StatusOK, getRR.Body.String())
	}
	var fetched models.Session
	if err := json.NewDecoder(getRR.Body).Decode(&fetched); err != nil {
		t.Fatalf("decode get response: %v", err)
	}

	if diff := cmp.Diff(fetched, patched); diff != "" {
		t.Errorf("PATCH response differs from GET response (-get +patch):\n%s", diff)
	}
}

// newServiceWithSession returns an in-memory session service holding one
// session. Unlike fakes.FakeSessionService it reports a missing session with
// session.ErrNotFound and applies state deltas, which is what the not-found and
// PATCH paths need.
func newServiceWithSession(t *testing.T, id fakes.SessionKey, state map[string]any) session.Service {
	t.Helper()
	service := session.InMemoryService()
	if _, err := service.Create(t.Context(), &session.CreateRequest{
		AppName:   id.AppName,
		UserID:    id.UserID,
		SessionID: id.SessionID,
		State:     state,
	}); err != nil {
		t.Fatalf("create session: %v", err)
	}
	return service
}

func newSessionRequest(t *testing.T, method string, id fakes.SessionKey, body io.Reader, authenticatedUserID string) *http.Request {
	t.Helper()
	url := fmt.Sprintf("/apps/%s/users/%s/sessions/%s", id.AppName, id.UserID, id.SessionID)
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	if authenticatedUserID != "" {
		req = req.WithContext(authn.WithCaller(req.Context(), &authn.Caller{UserID: authenticatedUserID}))
	}

	return mux.SetURLVars(req, sessionVars(id))
}

func sessionVars(sessionID fakes.SessionKey) map[string]string {
	return map[string]string{
		"app_name":   sessionID.AppName,
		"user_id":    sessionID.UserID,
		"session_id": sessionID.SessionID,
	}
}

// EquateApproxInt returns a cmp.Comparer option that determines integer values
// to be equal if they are within a certain absolute margin.
func EquateApproxInt(margin int64) cmp.Option {
	return cmp.Comparer(func(x, y int64) bool {
		diff := x - y
		if diff < 0 {
			diff = -diff
		}

		return diff <= margin
	})
}

// TestUpdateSessionRejectsScopedStateKeys pins the tenancy boundary.
//
// A state delta is split by prefix on the way to storage: "app:" lands in
// app-wide state shared by every user, "user:" in state shared by every session
// that user owns. Passing a client's delta through unfiltered made this the one
// REST route that writes outside the session it names, so a caller could PATCH
// their own session and change what every other user of the app sees.
func TestUpdateSessionRejectsScopedStateKeys(t *testing.T) {
	for _, tt := range []struct {
		name  string
		delta map[string]any
		want  int
	}{
		{"app-scoped key", map[string]any{"app:shared": "x"}, http.StatusBadRequest},
		{"user-scoped key", map[string]any{"user:shared": "x"}, http.StatusBadRequest},
		{"temp-scoped key", map[string]any{"temp:scratch": "x"}, http.StatusBadRequest},
		{"scoped mixed with a session key", map[string]any{"mine": "ok", "app:shared": "x"}, http.StatusBadRequest},
		{"session-scoped key", map[string]any{"mine": "ok"}, http.StatusOK},
		{"the UI's rename key", map[string]any{"__session_metadata__": map[string]any{"displayName": "n"}}, http.StatusOK},
	} {
		t.Run(tt.name, func(t *testing.T) {
			service := session.InMemoryService()
			created, err := service.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
			if err != nil {
				t.Fatalf("Create() failed: %v", err)
			}
			controller := controllers.NewSessionsAPIController(service)

			body, err := json.Marshal(map[string]any{"stateDelta": tt.delta})
			if err != nil {
				t.Fatalf("Marshal() failed: %v", err)
			}
			ctx := t.Context()
			ctx = authn.WithCaller(ctx, &authn.Caller{UserID: "user"})
			request := httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body))
			request = mux.SetURLVars(request, map[string]string{
				"app_name": "app", "user_id": "user", "session_id": created.Session.ID(),
			})
			recorder := httptest.NewRecorder()
			controller.UpdateSessionHandler(recorder, request)

			if recorder.Code != tt.want {
				t.Errorf("status = %d, want %d; body: %s", recorder.Code, tt.want, recorder.Body.String())
			}
		})
	}
}

// TestUpdateSessionAcceptsEchoedScopedStateKeys covers the payload the bundled
// web UI actually sends when you rename a session from the session list.
//
// It posts back the whole state map it read, with one metadata key changed, and
// a read merges app- and user-scoped state in. So the delta carries scoped keys
// the user never touched. Refusing it breaks rename for every agent that uses
// scoped state, and the UI subscribes with `error: () => {}`, so the name just
// silently does not change.
//
// The tenancy boundary is unaffected: a scoped key carrying a value different
// from the stored one is still a write outside this session, and still a 400.
func TestUpdateSessionAcceptsEchoedScopedStateKeys(t *testing.T) {
	// What an agent using scoped state leaves behind, and what a read hands the
	// client. storedCount is an int: the client can only echo it back as a JSON
	// number, so comparing the two in Go without a JSON round-trip says the
	// value changed when it did not.
	stored := map[string]any{
		"plain":       "kept",
		"app:config":  map[string]any{"theme": "dark"},
		"user:locale": "en",
		"app:count":   7,
	}

	for _, tt := range []struct {
		name       string
		delta      map[string]any
		wantStatus int
		// wantRejected are the keys the error must name. Checked only for 400.
		wantRejected []string
	}{
		{
			name: "the UI's rename payload echoes every scoped key back",
			delta: map[string]any{
				"plain":                "kept",
				"app:config":           map[string]any{"theme": "dark"},
				"user:locale":          "en",
				"app:count":            7,
				"__session_metadata__": map[string]any{"displayName": "renamed"},
			},
			wantStatus: http.StatusOK,
		},
		{
			name:       "an echoed int survives the client's JSON round-trip",
			delta:      map[string]any{"app:count": 7.0},
			wantStatus: http.StatusOK,
		},
		{
			name:       "changing an echoed scoped key is still refused",
			delta:      map[string]any{"app:config": map[string]any{"theme": "light"}},
			wantStatus: http.StatusBadRequest,
			// The nested map differs only in its value, so a comparison that
			// looked at keys alone would let this through.
			wantRejected: []string{"app:config"},
		},
		{
			name: "only the changed key is named, not the echoed ones",
			delta: map[string]any{
				"app:config":  map[string]any{"theme": "dark"},
				"user:locale": "fr",
			},
			wantStatus:   http.StatusBadRequest,
			wantRejected: []string{"user:locale"},
		},
		{
			name:         "a scoped key the session does not hold is a write",
			delta:        map[string]any{"app:brandnew": "x"},
			wantStatus:   http.StatusBadRequest,
			wantRejected: []string{"app:brandnew"},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			service := session.InMemoryService()
			created, err := service.Create(t.Context(), &session.CreateRequest{
				AppName: "app", UserID: "user", State: stored,
			})
			if err != nil {
				t.Fatalf("Create() failed: %v", err)
			}
			controller := controllers.NewSessionsAPIController(service)

			body, err := json.Marshal(map[string]any{"stateDelta": tt.delta})
			if err != nil {
				t.Fatalf("Marshal() failed: %v", err)
			}
			ctx := authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"})
			request := httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body))
			request = mux.SetURLVars(request, map[string]string{
				"app_name": "app", "user_id": "user", "session_id": created.Session.ID(),
			})
			recorder := httptest.NewRecorder()
			controller.UpdateSessionHandler(recorder, request)

			if recorder.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d; body: %s", recorder.Code, tt.wantStatus, recorder.Body.String())
			}

			if tt.wantStatus != http.StatusOK {
				for _, key := range tt.wantRejected {
					if !strings.Contains(recorder.Body.String(), key) {
						t.Errorf("error body %q does not name the rejected key %q", recorder.Body.String(), key)
					}
				}
				// An echoed key must not be blamed for someone else's write.
				for _, key := range []string{"plain", "__session_metadata__"} {
					if strings.Contains(recorder.Body.String(), key) {
						t.Errorf("error body %q names %q, which the client did not change", recorder.Body.String(), key)
					}
				}
				return
			}

			var got models.Session
			if err := json.NewDecoder(recorder.Body).Decode(&got); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			// The scoped keys must come back untouched, not dropped from the
			// session: filtering them out of the delta must not delete them.
			for key, want := range map[string]any{
				"app:config":  map[string]any{"theme": "dark"},
				"user:locale": "en",
				"app:count":   float64(7),
				"plain":       "kept",
			} {
				if diff := cmp.Diff(want, got.State[key]); diff != "" {
					t.Errorf("state[%q] mismatch (-want +got):\n%s", key, diff)
				}
			}
		})
	}
}

// TestUpdateSessionRenameAppliesWithScopedStateInPlay is the end-to-end shape of
// the bug: rename a session belonging to an agent that uses scoped state, then
// read it back and check the new name is there.
func TestUpdateSessionRenameAppliesWithScopedStateInPlay(t *testing.T) {
	service := session.InMemoryService()
	created, err := service.Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "user",
		State: map[string]any{"app:config": "shared", "user:locale": "en"},
	})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	controller := controllers.NewSessionsAPIController(service)

	// Exactly what the bundle posts: the state map it read, plus the new name.
	body, err := json.Marshal(map[string]any{"stateDelta": map[string]any{
		"app:config":           "shared",
		"user:locale":          "en",
		"__session_metadata__": map[string]any{"displayName": "My renamed chat"},
	}})
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	ctx := authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"})
	request := httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body))
	request = mux.SetURLVars(request, map[string]string{
		"app_name": "app", "user_id": "user", "session_id": created.Session.ID(),
	})
	recorder := httptest.NewRecorder()
	controller.UpdateSessionHandler(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("rename status = %d, want %d; body: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}

	reread, err := service.Get(t.Context(), &session.GetRequest{
		AppName: "app", UserID: "user", SessionID: created.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	metadata, err := reread.Session.State().Get("__session_metadata__")
	if err != nil {
		t.Fatalf("the rename did not persist: %v", err)
	}
	want := map[string]any{"displayName": "My renamed chat"}
	if diff := cmp.Diff(want, metadata); diff != "" {
		t.Errorf("__session_metadata__ mismatch (-want +got):\n%s", diff)
	}
}

// TestUpdateSessionEchoedRedactedCredentials covers the GET-then-PATCH rename
// flow when session, user:, or app: state holds credentials whose secrets GET
// strips. Echoing the redacted state back must neither overwrite a
// session-scoped credential's secrets nor reject a scoped credential as a
// write, while a credential value that differs from the redacted stored one is
// still treated as a write.
func TestUpdateSessionEchoedRedactedCredentials(t *testing.T) {
	sessionCred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "app",
			AccessToken:  "session-at",
			RefreshToken: "session-rt",
		},
	}
	appCred := toolauth.Credential{
		AuthType: toolauth.CredentialTypeOAuth2,
		OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "app",
			AccessToken:  "app-at",
			RefreshToken: "app-rt",
		},
	}
	userCred := map[string]any{
		"authType": "oauth2",
		"oauth2": map[string]any{
			"clientId":     "app",
			"accessToken":  "user-at",
			"refreshToken": "user-rt",
		},
	}

	renameFromGET := func(t *testing.T, service session.Service, sessionID string) *httptest.ResponseRecorder {
		t.Helper()
		controller := controllers.NewSessionsAPIController(service)
		ctx := authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"})
		vars := map[string]string{
			"app_name":   "app",
			"user_id":    "user",
			"session_id": sessionID,
		}

		getReq := mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodGet, "/", nil), vars)
		getRec := httptest.NewRecorder()
		controller.GetSessionHandler(getRec, getReq)
		if getRec.Code != http.StatusOK {
			t.Fatalf("GetSessionHandler() status = %d, want %d", getRec.Code, http.StatusOK)
		}
		var fetched struct {
			State map[string]any `json:"state"`
		}
		if err := json.Unmarshal(getRec.Body.Bytes(), &fetched); err != nil {
			t.Fatalf("Unmarshal(GET) failed: %v", err)
		}

		fetched.State["__session_metadata__"] = map[string]any{"displayName": "My renamed chat"}
		body, err := json.Marshal(map[string]any{"stateDelta": fetched.State})
		if err != nil {
			t.Fatalf("Marshal(PATCH) failed: %v", err)
		}
		patchReq := mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(body)), vars)
		patchRec := httptest.NewRecorder()
		controller.UpdateSessionHandler(patchRec, patchReq)
		return patchRec
	}

	t.Run("session-scoped credential keeps its stored secrets", func(t *testing.T) {
		service := session.InMemoryService()
		created, err := service.Create(t.Context(), &session.CreateRequest{
			AppName: "app",
			UserID:  "user",
			State:   map[string]any{"drive": sessionCred},
		})
		if err != nil {
			t.Fatalf("Create() failed: %v", err)
		}

		patchRec := renameFromGET(t, service, created.Session.ID())
		if patchRec.Code != http.StatusOK {
			t.Fatalf("UpdateSessionHandler() status = %d, want %d", patchRec.Code, http.StatusOK)
		}

		reread, err := service.Get(t.Context(), &session.GetRequest{
			AppName: "app", UserID: "user", SessionID: created.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Get() failed: %v", err)
		}
		metadata, err := reread.Session.State().Get("__session_metadata__")
		if err != nil {
			t.Fatalf("State().Get(__session_metadata__) failed: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"displayName": "My renamed chat"}, metadata); diff != "" {
			t.Errorf("__session_metadata__ mismatch (-want +got):\n%s", diff)
		}
		gotDrive, err := reread.Session.State().Get("drive")
		if err != nil {
			t.Fatalf("State().Get(drive) failed: %v", err)
		}
		if diff := cmp.Diff(sessionCred, gotDrive); diff != "" {
			t.Errorf("stored session credential after rename mismatch (-want +got):\n%s", diff)
		}
	})

	// A read writes these numbers as they were stored, and the client sends
	// back the float64 it decoded them to.
	numbers := []struct {
		name  string
		value any
	}{
		{name: "a trailing zero", value: json.Number("1.0")},
		{name: "an exponent", value: json.Number("1e3")},
		{name: "an integer past float64 precision", value: int64(9007199254740993)},
	}
	for _, n := range numbers {
		t.Run("session credential holding "+n.name+" keeps its stored secrets", func(t *testing.T) {
			cred := map[string]any{
				"authType": "oauth2",
				"oauth2": map[string]any{
					"clientId":     "app",
					"accessToken":  "session-at",
					"refreshToken": "session-rt",
					"ratio":        n.value,
				},
			}
			service := session.InMemoryService()
			created, err := service.Create(t.Context(), &session.CreateRequest{
				AppName: "app",
				UserID:  "user",
				State:   map[string]any{"drive": cred},
			})
			if err != nil {
				t.Fatalf("Create() failed: %v", err)
			}

			patchRec := renameFromGET(t, service, created.Session.ID())
			if patchRec.Code != http.StatusOK {
				t.Fatalf("UpdateSessionHandler() status = %d, want %d", patchRec.Code, http.StatusOK)
			}

			reread, err := service.Get(t.Context(), &session.GetRequest{
				AppName: "app", UserID: "user", SessionID: created.Session.ID(),
			})
			if err != nil {
				t.Fatalf("Get() failed: %v", err)
			}
			gotDrive, err := reread.Session.State().Get("drive")
			if err != nil {
				t.Fatalf("State().Get(drive) failed: %v", err)
			}
			if diff := cmp.Diff(cred, gotDrive); diff != "" {
				t.Errorf("stored session credential after rename mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("user and app scoped credentials do not turn the rename into a 400", func(t *testing.T) {
		service := session.InMemoryService()
		created, err := service.Create(t.Context(), &session.CreateRequest{
			AppName: "app",
			UserID:  "user",
			State: map[string]any{
				"app:drive":  appCred,
				"user:drive": userCred,
			},
		})
		if err != nil {
			t.Fatalf("Create() failed: %v", err)
		}

		patchRec := renameFromGET(t, service, created.Session.ID())
		if patchRec.Code != http.StatusOK {
			t.Fatalf("UpdateSessionHandler() status = %d, want %d", patchRec.Code, http.StatusOK)
		}

		reread, err := service.Get(t.Context(), &session.GetRequest{
			AppName: "app", UserID: "user", SessionID: created.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Get() failed: %v", err)
		}
		metadata, err := reread.Session.State().Get("__session_metadata__")
		if err != nil {
			t.Fatalf("State().Get(__session_metadata__) failed: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"displayName": "My renamed chat"}, metadata); diff != "" {
			t.Errorf("__session_metadata__ mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("credential differing from the redacted stored value is a write", func(t *testing.T) {
		service := session.InMemoryService()
		created, err := service.Create(t.Context(), &session.CreateRequest{
			AppName: "app",
			UserID:  "user",
			State: map[string]any{
				"drive":      sessionCred,
				"user:drive": userCred,
			},
		})
		if err != nil {
			t.Fatalf("Create() failed: %v", err)
		}
		controller := controllers.NewSessionsAPIController(service)
		ctx := authn.WithCaller(t.Context(), &authn.Caller{UserID: "user"})
		vars := map[string]string{
			"app_name":   "app",
			"user_id":    "user",
			"session_id": created.Session.ID(),
		}

		changedScopedBody, err := json.Marshal(map[string]any{"stateDelta": map[string]any{
			"user:drive": map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "other-app"},
			},
		}})
		if err != nil {
			t.Fatalf("Marshal(changed scoped PATCH) failed: %v", err)
		}
		changedScopedReq := mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(changedScopedBody)), vars)
		changedScopedRec := httptest.NewRecorder()
		controller.UpdateSessionHandler(changedScopedRec, changedScopedReq)
		if changedScopedRec.Code != http.StatusBadRequest {
			t.Errorf("UpdateSessionHandler() changed scoped credential status = %d, want %d", changedScopedRec.Code, http.StatusBadRequest)
		}

		newDrive := map[string]any{
			"authType": "oauth2",
			"oauth2": map[string]any{
				"clientId":                "app",
				"accessToken":             "new-session-at",
				"tokenEndpointAuthMethod": "client_secret_basic",
			},
		}
		changedSessionBody, err := json.Marshal(map[string]any{"stateDelta": map[string]any{
			"drive": newDrive,
		}})
		if err != nil {
			t.Fatalf("Marshal(changed session PATCH) failed: %v", err)
		}
		changedSessionReq := mux.SetURLVars(httptest.NewRequestWithContext(ctx, http.MethodPatch, "/", bytes.NewReader(changedSessionBody)), vars)
		changedSessionRec := httptest.NewRecorder()
		controller.UpdateSessionHandler(changedSessionRec, changedSessionReq)
		if changedSessionRec.Code != http.StatusOK {
			t.Fatalf("UpdateSessionHandler() changed session credential status = %d, want %d", changedSessionRec.Code, http.StatusOK)
		}
		afterWrite, err := service.Get(t.Context(), &session.GetRequest{
			AppName: "app", UserID: "user", SessionID: created.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Get() after write failed: %v", err)
		}
		gotUpdatedDrive, err := afterWrite.Session.State().Get("drive")
		if err != nil {
			t.Fatalf("State().Get(drive) after write failed: %v", err)
		}
		if diff := cmp.Diff(newDrive, gotUpdatedDrive); diff != "" {
			t.Errorf("stored session credential after write mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("root authType state keeps session secrets and accepts echoed scoped keys", func(t *testing.T) {
		oauth := map[string]any{"clientId": "app", "accessToken": "at", "refreshToken": "rt"}
		profile := map[string]any{"name": "ann", "token": "page-2"}
		sessionService := session.InMemoryService()
		sessionCreated, err := sessionService.Create(t.Context(), &session.CreateRequest{
			AppName: "app",
			UserID:  "user",
			State: map[string]any{
				"authType": "oauth2",
				"oauth2":   oauth,
				"profile":  profile,
			},
		})
		if err != nil {
			t.Fatalf("Create() failed: %v", err)
		}

		patchRec := renameFromGET(t, sessionService, sessionCreated.Session.ID())
		if patchRec.Code != http.StatusOK {
			t.Fatalf("UpdateSessionHandler() status = %d, want %d", patchRec.Code, http.StatusOK)
		}
		reread, err := sessionService.Get(t.Context(), &session.GetRequest{
			AppName: "app", UserID: "user", SessionID: sessionCreated.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Get() failed: %v", err)
		}
		gotOAuth, err := reread.Session.State().Get("oauth2")
		if err != nil {
			t.Fatalf("State().Get(oauth2) failed: %v", err)
		}
		if diff := cmp.Diff(oauth, gotOAuth); diff != "" {
			t.Errorf("stored oauth2 after rename mismatch (-want +got):\n%s", diff)
		}
		gotProfile, err := reread.Session.State().Get("profile")
		if err != nil {
			t.Fatalf("State().Get(profile) failed: %v", err)
		}
		if diff := cmp.Diff(profile, gotProfile); diff != "" {
			t.Errorf("stored profile after rename mismatch (-want +got):\n%s", diff)
		}

		scopedService := session.InMemoryService()
		scopedCreated, err := scopedService.Create(t.Context(), &session.CreateRequest{
			AppName: "app",
			UserID:  "user",
			State: map[string]any{
				"authType":     "oauth2",
				"user:profile": map[string]any{"name": "ann", "token": "t"},
				"app:config":   map[string]any{"name": "cfg", "apiKey": "k"},
			},
		})
		if err != nil {
			t.Fatalf("Create() scoped failed: %v", err)
		}

		scopedPatchRec := renameFromGET(t, scopedService, scopedCreated.Session.ID())
		if scopedPatchRec.Code != http.StatusOK {
			t.Fatalf("UpdateSessionHandler() scoped status = %d, want %d", scopedPatchRec.Code, http.StatusOK)
		}
		scopedReread, err := scopedService.Get(t.Context(), &session.GetRequest{
			AppName: "app", UserID: "user", SessionID: scopedCreated.Session.ID(),
		})
		if err != nil {
			t.Fatalf("Get() scoped failed: %v", err)
		}
		metadata, err := scopedReread.Session.State().Get("__session_metadata__")
		if err != nil {
			t.Fatalf("State().Get(__session_metadata__) failed: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"displayName": "My renamed chat"}, metadata); diff != "" {
			t.Errorf("__session_metadata__ mismatch (-want +got):\n%s", diff)
		}
	})
}
