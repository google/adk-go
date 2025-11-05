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

package session

import (
	"errors"
	"iter"
	"time"

	"github.com/google/uuid"
	"google.golang.org/adk/model"
)

// Session represents a series of interactions between a user and agents.
type Session interface {
	// ID returns the unique identifier of the session.
	ID() string
	// AppName returns name of the app.
	AppName() string
	// UserID returns the id of the user.
	UserID() string

	// State returns the state of the session.
	State() State
	// Events return the events of the session, e.g. user input, model response, function call/response, etc.
	Events() Events
	// LastUpdateTime returns the time of the last update.
	LastUpdateTime() time.Time
}

// State defines a standard interface for a key-value store.
// It provides basic methods for accessing, modifying, and iterating over
// key-value pairs.
type State interface {
	// Get retrieves the value associated with a given key.
	// It return a ErrStateKeyNotExist error if the key does not exist.
	Get(string) (any, error)
	// Set assigns the given value to the given key, overwriting any
	// existing value. It returns an error if the underlying storage
	// operation fails.
	Set(string, any) error
	// All returns an iterator (iter.Seq2) that yields all key-value pairs
	// currently in the state. The order of iteration is not guaranteed.
	All() iter.Seq2[string, any]
}

// ReadonlyState defines a standard interface for a key-value store.
// It provides basic methods for accessing, and iterating over
// key-value pairs.
type ReadonlyState interface {
	// Get retrieves the value associated with a given key.
	// It return a ErrStateKeyNotExist error if the key does not exist.
	Get(string) (any, error)
	// All returns an iterator (iter.Seq2) that yields all key-value pairs
	// currently in the state. The order of iteration is not guaranteed.
	All() iter.Seq2[string, any]
}

// Events defines a standard interface for an Event list.
// It provides methods for iterating over the sequence and accessing
// individual events by their index.
type Events interface {
	// All returns an iterator (iter.Seq) that yields all events
	// in the sequence, preserving their order.
	All() iter.Seq[*Event]
	// Len returns the total number of events in the sequence.
	Len() int
	// At returns the event at the specified index i.
	At(i int) *Event
}

// TODO: Clarify what fields should be set when Event is created/processed.
// TODO: Verify if we can hide Event completely; how Agents work with events.
// TODO: Potentially expose as user-visible event or layer.

// Event represents an interaction in a conversation between agents and users.
// It is used to store the content of the conversation, as well as
// the actions taken by the agents like function calls, etc.
type Event struct {
	model.LLMResponse

	// Set by storage
	ID        string
	Timestamp time.Time

	// Set by agent.Context implementation.
	InvocationID string
	// The branch of the event.
	//
	// The format is like agent_1.agent_2.agent_3, where agent_1 is
	// the parent of agent_2, and agent_2 is the parent of agent_3.
	//
	// Branch is used when multiple sub-agent shouldn't see their peer agents'
	// conversation history.
	Branch string
	// Author is the name of the event's author
	Author string

	// The actions taken by the agent.
	Actions EventActions
	// Set of IDs of the long running function calls.
	// Agent client will know from this field about which function call is long running.
	// Only valid for function call event.
	LongRunningToolIDs []string
}

// NewEvent creates a new event defining now as the timestamp.
func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Timestamp:    time.Now(),
		Actions:      EventActions{StateDelta: make(map[string]any)},
	}
}

// TODO: Set by clients?

// EventActions represents the actions attached to an event.
type EventActions struct {
	// Set by agent.Context implementation.
	StateDelta map[string]any

	// Indicates that the event is updating an artifact. key is the filename,
	// value is the version.
	ArtifactDelta map[string]int64

	// If true, it won't call model to summarize function response.
	// Only valid for function response event.
	SkipSummarization bool
	// If set, the event transfers to the specified agent.
	TransferToAgent string
	// The agent is escalating to a higher level agent.
	Escalate bool
}

// Prefixes for defining session's state scopes
const (
	KeyPrefixApp  string = "app:"
	KeyPrefixTemp string = "temp:"
	KeyPrefixUser string = "user:"
)

// ErrStateKeyNotExist defines the error thrown by State Get when key does not exist
var ErrStateKeyNotExist = errors.New("state key does not exist")
