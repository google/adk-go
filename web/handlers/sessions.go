package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/web/models"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

type SessionsApiController struct {
	service sessionservice.Service
}

func NewSessionsApiController(service sessionservice.Service) *SessionsApiController {
	return &SessionsApiController{service: service}
}

func (c *SessionsApiController) CreateSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return StatusError{error: fmt.Errorf("app_name parameter is required"), Code: http.StatusBadRequest}
	}
	userID := params["user_id"]
	if userID == "" {
		return StatusError{error: fmt.Errorf("user_id parameter is required"), Code: http.StatusBadRequest}
	}
	sessionID := params["session_id"]
	var createSessionRequest models.CreateSessionRequest
	err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
	if err != nil {
		return StatusError{error: err, Code: http.StatusBadRequest}
	}
	fmt.Printf("CreateSessionRequest: %v", createSessionRequest)
	session, err := c.service.Create(req.Context(), &sessionservice.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     createSessionRequest.State,
	})
	if err != nil {
		return StatusError{error: err, Code: http.StatusInternalServerError}
	}
	json.NewEncoder(rw).Encode(models.FromSession(session.Session))
	return nil
}

func (c *SessionsApiController) DeleteSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return StatusError{error: fmt.Errorf("app_name parameter is required"), Code: http.StatusBadRequest}
	}
	userID := params["user_id"]
	if userID == "" {
		return StatusError{error: fmt.Errorf("user_id parameter is required"), Code: http.StatusBadRequest}
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return StatusError{error: fmt.Errorf("session_id parameter is required"), Code: http.StatusBadRequest}
	}
	err := c.service.Delete(req.Context(), &sessionservice.DeleteRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return StatusError{error: err, Code: http.StatusInternalServerError}
	}
	return nil
}

// GetSession handles receiving a sesion from the system.
func (c *SessionsApiController) GetSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return StatusError{error: fmt.Errorf("app_name parameter is required"), Code: http.StatusBadRequest}
	}
	userID := params["user_id"]
	if userID == "" {
		return StatusError{error: fmt.Errorf("user_id parameter is required"), Code: http.StatusBadRequest}
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return StatusError{error: fmt.Errorf("session_id parameter is required"), Code: http.StatusBadRequest}
	}
	session, err := c.service.Get(req.Context(), &sessionservice.GetRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return StatusError{error: err, Code: http.StatusInternalServerError}
	}
	json.NewEncoder(rw).Encode(models.FromSession(session.Session))
	return nil
}

func (c *SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return StatusError{error: fmt.Errorf("app_name parameter is required"), Code: http.StatusBadRequest}
	}
	userID := params["user_id"]
	if userID == "" {
		return StatusError{error: fmt.Errorf("user_id parameter is required"), Code: http.StatusBadRequest}
	}
	resp, err := c.service.List(req.Context(), &sessionservice.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return StatusError{error: err, Code: http.StatusInternalServerError}
	}
	sessions := []models.Session{}
	for _, session := range resp.Sessions {
		sessions = append(sessions, models.FromSession(session))
	}
	json.NewEncoder(rw).Encode(sessions)
	return nil
}
