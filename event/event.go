// Package event provides ADK events.
package event

import (
	"time"

	"github.com/google/uuid"
)

// NewEvent creates a new event.
func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Time:         time.Now(),
	}
}

// Event represents an ADK event.
type Event struct {
	// TODO: model.Response
	ID                 string
	InvocationID       string
	LongRunningToolIDs []string
	Time               time.Time
	Actions            []*Action
	Author             string
	Branch             string
}

// Action is an event action.
type Action struct {
	// TODO(jbd): Implement.
}

// State represents event state.
type State map[string]any
