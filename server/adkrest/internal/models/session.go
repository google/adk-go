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
	"encoding/json"
	"fmt"
	"maps"

	"github.com/mitchellh/mapstructure"

	"google.golang.org/adk/v2/session"
)

// Session represents an agent's session.
type Session struct {
	ID        string      `json:"id"`
	AppName   string      `json:"appName"`
	UserID    string      `json:"userId"`
	UpdatedAt int64       `json:"lastUpdateTime"`
	Events    []Event     `json:"events"`
	State     stateValues `json:"state"`
}

type CreateSessionRequest struct {
	State  map[string]any `json:"state"`
	Events []Event        `json:"events"`
}

// UpdateSessionRequest is the body of a PATCH session request.
//
// StateDelta holds the state keys to add or overwrite; keys it does not mention
// are left as they are. An empty or absent delta is a no-op.
type UpdateSessionRequest struct {
	StateDelta map[string]any `json:"stateDelta"`
}

type SessionID struct {
	ID      string `mapstructure:"session_id,optional"`
	AppName string `mapstructure:"app_name,required"`
	UserID  string `mapstructure:"user_id,required"`
}

func SessionIDFromHTTPParameters(vars map[string]string) (SessionID, error) {
	var sessionID SessionID
	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		WeaklyTypedInput: true,
		Result:           &sessionID,
	})
	if err != nil {
		return sessionID, err
	}
	err = decoder.Decode(vars)
	if err != nil {
		return sessionID, err
	}
	if sessionID.AppName == "" {
		return sessionID, fmt.Errorf("app_name parameter is required")
	}
	if sessionID.UserID == "" {
		return sessionID, fmt.Errorf("user_id parameter is required")
	}
	return sessionID, nil
}

func FromSession(session session.Session) (Session, error) {
	state := stateValues{}
	maps.Insert(state, session.State().All())
	events := []Event{}
	for event := range session.Events().All() {
		events = append(events, FromSessionEvent(*event))
	}
	mappedSession := Session{
		ID:        session.ID(),
		AppName:   session.AppName(),
		UserID:    session.UserID(),
		UpdatedAt: session.LastUpdateTime().Unix(),
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
	// UpdatedAt is not checked. It started as a check that the time.Time was
	// set, and became UpdatedAt == 0 when the field turned into epoch seconds.
	// That form never catches an unset time, whose Unix() is -62135596800, but
	// it rejects every last update in the first second of 1970. An event
	// timestamp such as 0.5 puts a session there, which then failed every
	// GET, and every list of its user's sessions, with a 500.
	if s.State == nil {
		return fmt.Errorf("state is nil")
	}
	if s.Events == nil {
		return fmt.Errorf("events is nil")
	}
	return nil
}

// RedactedState encodes stored with the credential redaction [Session] applies
// to State, returning one raw JSON value per key, or nil if stored is nil or
// cannot be encoded.
func RedactedState(stored session.ReadonlyState) map[string]json.RawMessage {
	if stored == nil {
		return nil
	}
	state := stateValues{}
	maps.Insert(state, stored.All())
	b, err := json.Marshal(state)
	if err != nil {
		return nil
	}
	var redacted map[string]json.RawMessage
	if err := json.Unmarshal(b, &redacted); err != nil {
		return nil
	}
	return redacted
}
