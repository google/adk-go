// Copyright 2026 Google LLC
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

package workflow

import (
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/adk/session"
)

// runStateSessionKeyPrefix is the prefix used by RunStateSessionKey
// for namespacing workflow RunStates inside session.State.
const runStateSessionKeyPrefix = "adk.workflow.runstate."

// RunStateSessionKey returns the session.State key under which a
// workflow's RunState is persisted between invocations. Namespaced
// by workflow name so multiple workflows in the same session do
// not collide.
func RunStateSessionKey(workflowName string) string {
	return runStateSessionKeyPrefix + workflowName
}

// LoadRunState reads and decodes the workflow's persisted RunState
// from the given session. Returns (nil, nil) when no state has
// been stored yet, so callers can distinguish "nothing to resume"
// from "load failed". An empty workflowName disables persistence
// and always returns (nil, nil).
func LoadRunState(sess session.Session, workflowName string) (*RunState, error) {
	if sess == nil || workflowName == "" {
		return nil, nil
	}
	state := sess.State()
	if state == nil {
		return nil, nil
	}
	raw, err := state.Get(RunStateSessionKey(workflowName))
	if err != nil {
		if errors.Is(err, session.ErrStateKeyNotExist) {
			return nil, nil
		}
		return nil, err
	}

	// raw is JSON-encoded []byte (or its base64-string form when
	// the session backend round-trips StateDelta through JSON).
	decode := func(b []byte) (*RunState, error) {
		var state RunState
		if err := json.Unmarshal(b, &state); err != nil {
			return nil, fmt.Errorf("workflow: decode run state: %w", err)
		}
		return &state, nil
	}
	switch v := raw.(type) {
	case []byte:
		return decode(v)
	case string:
		return decode([]byte(v))
	default:
		return nil, fmt.Errorf("workflow: run state has unexpected type %T (want []byte or string)", raw)
	}
}

// NewRunStateEvent builds a session.Event whose Actions.StateDelta
// carries the workflow's serialised RunState. Workflow.Run and
// Workflow.Resume yield this event before returning so the
// surrounding event-append pipeline can persist the state
// alongside every other delta-based update.
//
// Persistence backends apply state mutations only when they
// observe them on Event.Actions.StateDelta during the append
// path; a direct session.State().Set updates the per-invocation
// copy but is not propagated to storage. Returning nil for an
// empty workflowName lets callers use NewRunStateEvent
// unconditionally and skip the yield when persistence is not
// desired.
func NewRunStateEvent(invocationID, workflowName string, state *RunState) (*session.Event, error) {
	if workflowName == "" || state == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("workflow: encode run state: %w", err)
	}
	ev := session.NewEvent(invocationID)
	ev.Actions.StateDelta[RunStateSessionKey(workflowName)] = bytes
	return ev, nil
}
