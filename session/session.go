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
	"google.golang.org/adk/llm"
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
	Get(string) any
	Set(string, any)
	All() iter.Seq2[string, any]
}

// TODO: It is provided for use by SessionService, and perhaps it should move there.
type ReadOnlyState interface {
	Get(string) any
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
type Event struct {
	// Set by storage
	ID   string
	Time time.Time

	// Set by agent.Context implementation.
	InvocationID string
	Branch       string
	Author       string

	Partial            bool
	Actions            Actions
	LongRunningToolIDs []string
	LLMResponse        *llm.Response
}

// NewEvent creates a new event.
func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Time:         time.Now(),
	}
}

func (e *Event) Clone() *Event { return nil }

type Actions struct {
	// Set by agent.Context implementation.
	StateDelta map[string]any

	// Set by clients?
	SkipSummarization bool
	TransferToAgent   string
	Escalate          bool
}
