package models

import (
	"maps"
	"time"

	"google.golang.org/adk/sessionservice"
)

type Session struct {
	ID        string         `json:"id"`
	AppName   string         `json:"app_name"`
	UserID    string         `json:"user_id"`
	UpdatedAt time.Time      `json:"updated_at"`
	Events    []Event        `json:"events"`
	State     map[string]any `json:"state"`
}

type CreateSessionRequest struct {
	State  map[string]any `json:"state"`
	Events []Event        `json:"events"`
}

func FromSession(session sessionservice.StoredSession) Session {
	id := session.ID()
	state := map[string]any{}
	maps.Insert(state, session.State().All())
	return Session{
		ID:        id.SessionID,
		AppName:   id.AppName,
		UserID:    id.UserID,
		UpdatedAt: session.Updated(),
		Events:    []Event{},
		State:     state,
	}
}
