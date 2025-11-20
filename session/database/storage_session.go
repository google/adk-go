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

package database

import (
	"encoding/json"
	"fmt"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// storageSession corresponds to the 'sessions' table.
type storageSession struct {
	AppName    string `gorm:"primaryKey;"`
	UserID     string `gorm:"primaryKey;"`
	ID         string `gorm:"primaryKey;"`
	State      stateMap
	CreateTime time.Time
	UpdateTime time.Time

	// Has-Many relationship: A session has many events.
	Events []storageEvent `gorm:"foreignKey:AppName,UserID,SessionID;references:AppName,UserID,ID"`
}

// TableName explicitly sets the table name for the storageSession struct.
func (storageSession) TableName() string {
	return "sessions"
}

// Helper to map from internal struct to GORM struct
func createStorageSession(s *localSession) (*storageSession, error) {
	return &storageSession{
		UserID:     s.userID,
		AppName:    s.appName,
		ID:         s.sessionID,
		State:      s.state,
		CreateTime: time.Now(),
		UpdateTime: time.Now(),
	}, nil
}

// Helper to map from GORM struct to internal struct
func createSessionFromStorageSession(storage *storageSession) (*localSession, error) {
	return &localSession{
		appName:   storage.AppName,
		userID:    storage.UserID,
		sessionID: storage.ID,
		state:     storage.State,
		updatedAt: storage.UpdateTime,
	}, nil
}

// storageEvent corresponds to the 'events' table.
type storageEvent struct {
	ID        string `gorm:"primaryKey;"`
	AppName   string `gorm:"primaryKey;"`
	UserID    string `gorm:"primaryKey;"`
	SessionID string `gorm:"primaryKey;"`

	InvocationID string
	Author       string
	// In Python, this is a pickled object. In Go, the raw bytes are the closest
	// equivalent. Unpickling would require a custom library or service.
	Actions                []byte
	LongRunningToolIDsJSON dynamicJSON
	Branch                 *string
	Timestamp              time.Time

	// Fields from llm_response
	Content           dynamicJSON
	GroundingMetadata dynamicJSON
	CustomMetadata    dynamicJSON
	UsageMetadata     dynamicJSON
	CitationMetadata  dynamicJSON

	Partial      *bool
	TurnComplete *bool
	ErrorCode    *string
	ErrorMessage *string
	Interrupted  *bool

	// Belongs-To relationship: An event belongs to a session.
	Session storageSession `gorm:"foreignKey:AppName,UserID,SessionID;references:AppName,UserID,ID"`
}

// TableName explicitly sets the table name for the storageEvent struct.
func (storageEvent) TableName() string {
	return "events"
}

// createStorageEvent translates the application-level Session and Event models
// into a GORM-compatible storageEvent struct, ready for database insertion.
func createStorageEvent(session session.Session, event *session.Event) (*storageEvent, error) {
	// Initialize the base storageEvent with direct field mappings.
	storageEv := &storageEvent{
		ID:           event.ID,
		InvocationID: event.InvocationID,
		Author:       event.Author,
		SessionID:    session.ID(),
		AppName:      session.AppName(),
		UserID:       session.UserID(),
		Timestamp:    event.Timestamp,
	}

	// --- Handle complex or nullable fields ---
	// Serialize the entire Actions struct into a JSON byte slice.
	actionsJSON, err := json.Marshal(event.Actions)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal event actions: %w", err)
	}
	storageEv.Actions = actionsJSON

	// Serialize the list of tool IDs into a JSON string
	if len(event.LongRunningToolIDs) > 0 {
		toolIDsJSON, err := json.Marshal(event.LongRunningToolIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal long running tool IDs: %w", err)
		}
		storageEv.LongRunningToolIDsJSON = toolIDsJSON
	}

	// Handle optional fields by taking the address of the value.
	// An empty string from the event becomes a nil pointer in storage.
	if event.Branch != "" {
		storageEv.Branch = &event.Branch
	}
	if event.ErrorCode != "" {
		storageEv.ErrorCode = &event.ErrorCode
	}
	if event.ErrorMessage != "" {
		storageEv.ErrorMessage = &event.ErrorMessage
	}

	// For booleans, we can assign pointers directly.
	storageEv.Partial = &event.Partial
	storageEv.TurnComplete = &event.TurnComplete
	storageEv.Interrupted = &event.Interrupted

	// --- Handle JSON content fields ---
	if event.Content != nil {
		storageEv.Content, err = json.Marshal(event.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal content: %w", err)
		}
	}
	if event.GroundingMetadata != nil {
		storageEv.GroundingMetadata, err = json.Marshal(event.GroundingMetadata)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal grounding metadata: %w", err)
		}
	}
	if len(event.CustomMetadata) > 0 {
		storageEv.CustomMetadata, err = json.Marshal(event.CustomMetadata)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal custom metadata: %w", err)
		}
	}
	if event.UsageMetadata != nil {
		storageEv.UsageMetadata, err = json.Marshal(event.UsageMetadata)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal usage metadata: %w", err)
		}
	}
	if event.CitationMetadata != nil {
		storageEv.CitationMetadata, err = json.Marshal(event.CitationMetadata)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal citation metadata: %w", err)
		}
	}

	return storageEv, nil
}

// derefOrZero safely dereferences a pointer, returning the zero value
// of its type if the pointer is nil.
func derefOrZero[T any](p *T) T {
	if p == nil {
		var zero T
		return zero
	}
	return *p
}

// createEventFromStorageEvent translates a GORM storageEvent back into an
// application-level Event model.
func createEventFromStorageEvent(se *storageEvent) (*session.Event, error) {
	var actions session.EventActions
	if len(se.Actions) > 0 {
		if err := json.Unmarshal(se.Actions, &actions); err != nil {
			return nil, fmt.Errorf("failed to unmarshal actions: %w", err)
		}
	}

	var content *genai.Content
	if len(se.Content) > 0 {
		if err := json.Unmarshal(se.Content, &content); err != nil {
			return nil, fmt.Errorf("failed to unmarshal content: %w", err)
		}
	}

	var groundingMetadata *genai.GroundingMetadata
	if len(se.GroundingMetadata) > 0 {
		if err := json.Unmarshal(se.GroundingMetadata, &groundingMetadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal grounding metadata: %w", err)
		}
	}

	var customMetadata map[string]any
	if len(se.CustomMetadata) > 0 {
		if err := json.Unmarshal(se.CustomMetadata, &customMetadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal custom metadata: %w", err)
		}
	}

	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if len(se.UsageMetadata) > 0 {
		if err := json.Unmarshal(se.UsageMetadata, &usageMetadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal usage metadata: %w", err)
		}
	}

	var citationMetadata *genai.CitationMetadata
	if len(se.CitationMetadata) > 0 {
		if err := json.Unmarshal(se.CitationMetadata, &citationMetadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal citation metadata: %w", err)
		}
	}

	// --- Handle JSON-encoded *string field ---
	var toolIDs []string
	if se.LongRunningToolIDsJSON != nil {
		if err := json.Unmarshal([]byte(se.LongRunningToolIDsJSON), &toolIDs); err != nil {
			return nil, fmt.Errorf("failed to unmarshal long running tool IDs: %w", err)
		}
	}

	// --- Handle simple pointer fields (dereference or use zero value) ---
	// Use the helper to safely get the value or its zero-value default
	branch := derefOrZero(se.Branch)
	errorCode := derefOrZero(se.ErrorCode)
	errorMessage := derefOrZero(se.ErrorMessage)
	partial := derefOrZero(se.Partial)
	turnComplete := derefOrZero(se.TurnComplete)
	interrupted := derefOrZero(se.Interrupted)

	// --- Assemble the final Event struct ---
	event := &session.Event{
		ID:                 se.ID,
		InvocationID:       se.InvocationID,
		Author:             se.Author,
		Timestamp:          se.Timestamp,
		Actions:            actions,
		LongRunningToolIDs: toolIDs,
		Branch:             branch,
		LLMResponse: model.LLMResponse{
			Content:           content,
			GroundingMetadata: groundingMetadata,
			CustomMetadata:    customMetadata,
			UsageMetadata:     usageMetadata,
			CitationMetadata:  citationMetadata,
			ErrorCode:         errorCode,
			ErrorMessage:      errorMessage,
			Partial:           partial,
			TurnComplete:      turnComplete,
			Interrupted:       interrupted,
		},
	}

	return event, nil
}

// AppState corresponds to the 'app_states' table.
type storageAppState struct {
	AppName    string `gorm:"primaryKey;"`
	State      stateMap
	UpdateTime time.Time
}

// TableName explicitly sets the table name for the AppState struct.
func (storageAppState) TableName() string {
	return "app_states"
}

// UserState corresponds to the 'user_states' table.
type storageUserState struct {
	AppName    string `gorm:"primaryKey;"`
	UserID     string `gorm:"primaryKey;"`
	State      stateMap
	UpdateTime time.Time
}

// TableName explicitly sets the table name for the UserState struct.
func (storageUserState) TableName() string {
	return "user_states"
}
