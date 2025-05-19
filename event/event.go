package event

import (
	"time"

	"github.com/google/uuid"
)

func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Time:         time.Now(),
	}
}

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

type Action struct {
	// TODO(jbd): Implement.
}

type State map[string]any
