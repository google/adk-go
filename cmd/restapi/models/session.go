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
