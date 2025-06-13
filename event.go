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

package adk

import (
	"iter"
	"time"

	"github.com/google/uuid"
)

type EventStream = iter.Seq2[*Event, error]

// NewEvent creates a new event.
func NewEvent(invocationID string) *Event {
	return &Event{
		ID:           uuid.NewString(),
		InvocationID: invocationID,
		Time:         time.Now(),
	}
}

// Event represents an even in a conversation between agents and users.
// It is used to sore the content of the conversation, as well as
// the actions taken by the agents like function calls, etc.
type Event struct {
	// The followings are set by the session.
	ID   string
	Time time.Time

	// The invocation ID of the event.
	InvocationID string

	// Set of IDs of the long running function calls.
	// Agent client will know from this field about which function call is long running.
	// Only valid for function call event.
	LongRunningToolIDs []string
	// User or the name of the agent, indicating who appended the event to the session.
	Author string
	// The branch of the event.
	//
	// The format is like agent_1.gent_2.agent_3, where agent_1 is
	// the parent of agent_2, and agent_2 is the parent of agent_3.
	//
	// Branch is used when multiple sub-agent shouldn't see their peer agents'
	// conversation history.
	Branch string

	// The actions taken by the agent.
	Actions []*EventAction

	LLMResponse *LLMResponse
}

// EventAction is an event action.
type EventAction struct {
	// If true, it won't call model to summarize function response.
	// Only valid for function response event.
	SkipSummarization bool
	// If set, the event transfers to the specified agent.
	TransferToAgent string
	// The agent is escalating to a higher level agent.
	Escalate bool

	StateDelta map[string]any
	//ArtifactDelta map[string]any
	//RequestedAuthConfigs map[string]*AuthConfig
}

// EventState represents event state.
type EventState map[string]any
