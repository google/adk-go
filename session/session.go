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
	"iter"
	"time"

	"github.com/google/uuid"
	"google.golang.org/adk/model"
)

type Session interface {
	ID() ID
	State() State
	Events() Events
	Updated() time.Time
}

type ID struct {
	AppName   string
	UserID    string
	SessionID string
}

type State interface {
	Get(string) (any, error)
	Set(string, any) error
	All() iter.Seq2[string, any]
}

// TODO: It is provided for use by SessionService, and perhaps it should move there.
type ReadOnlyState interface {
	Get(string) (any, error)
	All() iter.Seq2[string, any]
}

type Events interface {
	All() iter.Seq[*Event]
	Len() int
	At(i int) *Event
}

// TODO: Clarify what fields should be set when Event is created/processed.
// TODO: Verify if we can hide Event completely; how Agents work with events.
// TODO: Potentially expose as user-visible event or layer.
//
// Event represents an even in a conversation between agents and users.
// It is used to store the content of the conversation, as well as
// the actions taken by the agents like function calls, etc.
type Event struct {
	// Set by storage
	ID   string
	Time time.Time

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
	Author string

	Partial bool
	// The actions taken by the agent.
	Actions Actions
	// Set of IDs of the long running function calls.
	// Agent client will know from this field about which function call is long running.
	// Only valid for function call event.
	LongRunningToolIDs []string
	LLMResponse        *model.LLMResponse
}

// NewEvent creates a new event.
func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Time:         time.Now(),
	}
}

// Actions represents the actions attached to an event.
type Actions struct {
	// Set by agent.Context implementation.
	StateDelta map[string]any

	// TODO: Set by clients?
	//
	// If true, it won't call model to summarize function response.
	// Only valid for function response event.
	SkipSummarization bool
	// If set, the event transfers to the specified agent.
	TransferToAgent string
	// The agent is escalating to a higher level agent.
	Escalate bool
}
