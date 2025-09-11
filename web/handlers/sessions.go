package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/web/errors"
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
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	createSessionRequest := models.CreateSessionRequest{}
	if req.ContentLength > 0 {
		err := json.NewDecoder(req.Body).Decode(&createSessionRequest)
		if err != nil {
			return errors.NewStatusError(fmt.Errorf("decode request: %w", err), http.StatusBadRequest)
		}
	}
	fmt.Printf("CreateSessionRequest: %v", createSessionRequest)
	session, err := c.service.Create(req.Context(), &sessionservice.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     createSessionRequest.State,
	})
	if err != nil {
		return errors.NewStatusError(fmt.Errorf("create session: %w", err), http.StatusInternalServerError)
	}
	resp := models.FromSession(session.Session)
	err = json.NewEncoder(rw).Encode(resp)
	if err != nil {
		return errors.NewStatusError(fmt.Errorf("encode response: %w", err), http.StatusInternalServerError)
	}
	return nil
}

func (c *SessionsApiController) DeleteSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return errors.NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	err := c.service.Delete(req.Context(), &sessionservice.DeleteRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	return nil
}

// GetSession handles receiving a sesion from the system.
func (c *SessionsApiController) GetSession(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	sessionID := params["session_id"]
	if sessionID == "" {
		return errors.NewStatusError(fmt.Errorf("session_id parameter is required"), http.StatusBadRequest)
	}
	session, err := c.service.Get(req.Context(), &sessionservice.GetRequest{
		ID: session.ID{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		},
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	json.NewEncoder(rw).Encode(models.FromSession(session.Session))
	return nil
}

func (c *SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) error {
	params := mux.Vars(req)
	appName := params["app_name"]
	if appName == "" {
		return errors.NewStatusError(fmt.Errorf("app_name parameter is required"), http.StatusBadRequest)
	}
	userID := params["user_id"]
	if userID == "" {
		return errors.NewStatusError(fmt.Errorf("user_id parameter is required"), http.StatusBadRequest)
	}
	resp, err := c.service.List(req.Context(), &sessionservice.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return errors.NewStatusError(err, http.StatusInternalServerError)
	}
	sessions := []models.Session{}
	for _, session := range resp.Sessions {
		sessions = append(sessions, models.FromSession(session))
	}
	json.NewEncoder(rw).Encode(sessions)
	return nil
}
