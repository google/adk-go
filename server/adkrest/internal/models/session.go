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

	"github.com/mitchellh/mapstructure"

	"google.golang.org/adk/session"
)

// State delta directive constants
const (
	// stateUpdateKey is the special key used in state delta directives
	// to indicate a patch operation (e.g., delete).
	stateUpdateKey = "$adk_state_update"

	// stateUpdateDelete is the directive value indicating a key should be deleted.
	stateUpdateDelete = "delete"
)

// Session represents an agent's session.
type Session struct {
	ID        string         `json:"id"`
	AppName   string         `json:"appName"`
	UserID    string         `json:"userId"`
	UpdatedAt int64          `json:"lastUpdateTime"`
	Events    []Event        `json:"events"`
	State     map[string]any `json:"state"`
}

type CreateSessionRequest struct {
	State  map[string]any `json:"state"`
	Events []Event        `json:"events"`
}

type PatchSessionStateDeltaRequest struct {
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
	state := map[string]any{}
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
	if s.UpdatedAt == 0 {
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

// NormalizeStateDelta processes state delta directives and converts them
// into a normalized representation suitable for the service layer.
// Delete directives ({"$adk_state_update": "delete"}) are converted to nil values.
// Returns a new map with normalized values.
func NormalizeStateDelta(stateDelta map[string]any) (map[string]any, error) {
	normalized := make(map[string]any, len(stateDelta))
	for key, value := range stateDelta {
		// Check if value is a directive (map with special key)
		directive, isDirective := value.(map[string]any)
		if isDirective {
			// Check if this map contains a state update directive
			updateValue, hasDirective := directive[stateUpdateKey]
			if hasDirective {
				normalizedValue, err := processDirective(key, updateValue)
				if err != nil {
					return nil, err
				}
				normalized[key] = normalizedValue
				continue
			}
			// else: it's a normal map value, fall through and set it as-is
		}

		// Normal value (including normal maps): keep it directly.
		normalized[key] = value
	}

	return normalized, nil
}

// processDirective handles a state update directive and returns the normalized value.
func processDirective(key string, updateValue any) (any, error) {
	updateStr, ok := updateValue.(string)
	if !ok {
		return nil, fmt.Errorf(
			"invalid directive value type for key %q: expected string, got %T",
			key,
			updateValue,
		)
	}

	switch updateStr {
	case stateUpdateDelete:
		// Delete directive: return nil to indicate deletion
		return nil, nil
	default:
		return nil, fmt.Errorf("unknown state update directive %q for key %q", updateStr, key)
	}
}
