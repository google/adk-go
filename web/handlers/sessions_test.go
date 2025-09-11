package handlers_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/gorilla/mux"
	"google.golang.org/adk/session"
	"google.golang.org/adk/web/handlers"
	"google.golang.org/adk/web/models"
	"google.golang.org/adk/web/utils"
)

func TestGetSession(t *testing.T) {
	tc := []struct {
		name           string
		storedSessions map[session.ID]utils.TestSession
		sessionID      session.ID
		wantSession    models.Session
		wantErr        error
	}{
		{
			name: "session exists",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", "testSession"),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: sessionID("testApp", "testUser", "testSession"),
			wantSession: models.Session{
				ID:        "testSession",
				AppName:   "testApp",
				UserID:    "testUser",
				UpdatedAt: time.Now(),
				Events:    []models.Event{},
				State: map[string]any{
					"foo": "bar",
				},
			},
		},
		{
			name:           "session does not exist",
			storedSessions: map[session.ID]utils.TestSession{},
			sessionID:      sessionID("testApp", "testUser", "testSession"),
			wantErr:        fmt.Errorf("not found"),
		},
		{
			name: "session ID is missing in input",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", "testSession"),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: sessionID("testApp", "testUser", ""),
			wantErr:   fmt.Errorf("session_id parameter is required"),
		},
		{
			name: "session ID is missing",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", ""),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: sessionID("testApp", "testUser", "testSession"),
			wantErr:   fmt.Errorf("session_id is empty in received session"),
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := utils.FakeSessionService{Sessions: tt.storedSessions}
			apiController := handlers.NewSessionsApiController(&sessionService)
			req, err := http.NewRequest(http.MethodGet, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			err = apiController.GetSession(rr, req)
			if tt.wantErr == nil && err != nil {
				t.Fatalf("get session: %v", err)
			} else if tt.wantErr != nil {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if err.Error() != tt.wantErr.Error() {
					t.Fatalf("expected error %q, got %q", tt.wantErr, err)
				}
				return
			}
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
			var gotSession models.Session
			err = json.NewDecoder(rr.Body).Decode(&gotSession)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSession, gotSession, cmpopts.EquateApproxTime(time.Second)); diff != "" {
				t.Errorf("GetSession() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCreateSession(t *testing.T) {
	tc := []struct {
		name             string
		storedSessions   map[session.ID]utils.TestSession
		sessionID        session.ID
		createRequestObj models.CreateSessionRequest
		wantSession      models.Session
		wantErr          error
	}{
		{
			name: "session exists",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", "testSession"),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: sessionID("testApp", "testUser", "testSession"),
			wantErr:   fmt.Errorf("session already exists"),
		},
		{
			name:           "successful create operation",
			storedSessions: map[session.ID]utils.TestSession{},
			sessionID:      sessionID("testApp", "testUser", "testSession"),
			createRequestObj: models.CreateSessionRequest{
				State: map[string]any{
					"foo": "bar",
				},
				Events: []models.Event{
					{
						ID:     "eventID",
						Time:   time.Now().Add(5 * time.Minute),
						Author: "testUser",
					},
				},
			},
			wantSession: models.Session{
				ID:        "testSession",
				AppName:   "testApp",
				UserID:    "testUser",
				UpdatedAt: time.Now().Add(5 * time.Minute),
				State: map[string]any{
					"foo": "bar",
				},
				Events: []models.Event{
					{
						ID:     "eventID",
						Author: "testUser",
						Time:   time.Now().Add(5 * time.Minute),
					},
				},
			},
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := utils.FakeSessionService{Sessions: tt.storedSessions}
			apiController := handlers.NewSessionsApiController(&sessionService)
			reqBytes, err := json.Marshal(tt.createRequestObj)
			if err != nil {
				t.Fatalf("marshal request: %v", err)
			}
			req, err := http.NewRequest(http.MethodPost, "/apps/testApp/users/testUser/sessions/testSession", bytes.NewBuffer(reqBytes))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			err = apiController.CreateSession(rr, req)
			if tt.wantErr == nil && err != nil {
				t.Fatalf("create session: %v", err)
			} else if tt.wantErr != nil {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if err.Error() != tt.wantErr.Error() {
					t.Fatalf("expected error %q, got %q", tt.wantErr, err)
				}
				return
			}
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
			var gotSession models.Session
			err = json.NewDecoder(rr.Body).Decode(&gotSession)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSession, gotSession, cmpopts.EquateApproxTime(time.Second)); diff != "" {
				t.Errorf("GetSession() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDeleteSession(t *testing.T) {
	tc := []struct {
		name           string
		storedSessions map[session.ID]utils.TestSession
		sessionID      session.ID
		wantErr        error
	}{
		{
			name: "session exists",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", "testSession"),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			sessionID: sessionID("testApp", "testUser", "testSession"),
		},
		{
			name:           "session does not exist",
			storedSessions: map[session.ID]utils.TestSession{},
			sessionID:      sessionID("testApp", "testUser", "testSession"),
			wantErr:        fmt.Errorf("not found"),
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := utils.FakeSessionService{Sessions: tt.storedSessions}
			apiController := handlers.NewSessionsApiController(&sessionService)
			req, err := http.NewRequest(http.MethodDelete, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, sessionVars(tt.sessionID))
			rr := httptest.NewRecorder()

			err = apiController.DeleteSession(rr, req)
			if tt.wantErr == nil && err != nil {
				t.Fatalf("get session: %v", err)
			} else if tt.wantErr != nil {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if err.Error() != tt.wantErr.Error() {
					t.Fatalf("expected error %q, got %q", tt.wantErr, err)
				}
				return
			}
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
			if _, ok := sessionService.Sessions[tt.sessionID]; ok {
				t.Errorf("session was not deleted")
			}
		})
	}
}

func TestListSessions(t *testing.T) {
	tc := []struct {
		name           string
		storedSessions map[session.ID]utils.TestSession
		wantSessions   []models.Session
	}{
		{
			name: "session exists",
			storedSessions: map[session.ID]utils.TestSession{
				sessionID("testApp", "testUser", "testSession"): {
					Id:            sessionID("testApp", "testUser", "testSession"),
					SessionState:  utils.TestState{"foo": "bar"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
				sessionID("testApp", "testUser", "newSession"): {
					Id:            sessionID("testApp", "testUser", "newSession"),
					SessionState:  utils.TestState{"xyz": "abc"},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
				sessionID("testApp", "testUser", "oldSession"): {
					Id:            sessionID("testApp", "testUser", "oldSession"),
					SessionState:  utils.TestState{},
					SessionEvents: utils.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
			wantSessions: []models.Session{
				{
					ID:        "testSession",
					AppName:   "testApp",
					UserID:    "testUser",
					UpdatedAt: time.Now(),
					Events:    []models.Event{},
					State: map[string]any{
						"foo": "bar",
					},
				},
				{
					ID:        "newSession",
					AppName:   "testApp",
					UserID:    "testUser",
					UpdatedAt: time.Now(),
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
					UpdatedAt: time.Now(),
					Events:    []models.Event{},
				},
			},
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			sessionService := utils.FakeSessionService{Sessions: tt.storedSessions}
			apiController := handlers.NewSessionsApiController(&sessionService)
			req, err := http.NewRequest(http.MethodDelete, "/apps/testApp/users/testUser/sessions/testSession", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			// Manually set the URL variables on the request using mux.SetURLVars.
			req = mux.SetURLVars(req, map[string]string{
				"app_name": "testApp",
				"user_id":  "testUser",
			})
			rr := httptest.NewRecorder()

			err = apiController.ListSessions(rr, req)
			if err != nil {
				t.Fatalf("get session: %v", err)
			}
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
			got := []models.Session{}
			err = json.NewDecoder(rr.Body).Decode(&got)
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if diff := cmp.Diff(tt.wantSessions, got, cmpopts.EquateApproxTime(time.Second), cmpopts.SortSlices(func(a, b models.Session) bool {
				return a.ID < b.ID
			})); diff != "" {
				t.Errorf("ListSessions() mismatch (-want +got):\n%s", diff)
			}
		})
	}

}

func sessionID(appName, userID, sessionID string) session.ID {
	return session.ID{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}
}

func sessionVars(sessionID session.ID) map[string]string {
	return map[string]string{
		"app_name":   sessionID.AppName,
		"user_id":    sessionID.UserID,
		"session_id": sessionID.SessionID,
	}
}
