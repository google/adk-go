package models

import (
	"fmt"
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

func FromSession(session sessionservice.StoredSession) (Session, error) {
	id := session.ID()
	state := map[string]any{}
	maps.Insert(state, session.State().All())
	events := []Event{}
	for event := range session.Events().All() {
		events = append(events, FromSessionEvent(*event))
	}
	mappedSession := Session{
		ID:        id.SessionID,
		AppName:   id.AppName,
		UserID:    id.UserID,
		UpdatedAt: session.Updated(),
		Events:    events,
		State:     state,
	}
	return mappedSession, mappedSession.Validate()
}

func (s Session) Validate() error {
	if s.AppName == "" {
		return fmt.Errorf("app_name is empty in received session")
	}
	if s.UserID == "" {
		return fmt.Errorf("user_id is empty in received session")
	}
	if s.ID == "" {
		return fmt.Errorf("session_id is empty in received session")
	}
	if s.UpdatedAt.IsZero() {
		return fmt.Errorf("updated_at is empty")
	}
	if s.State == nil {
		return fmt.Errorf("state is nil")
	}
	if s.Events == nil {
		return fmt.Errorf("events is nil")
	}
	return nil
}
