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

// ReconstructRunState rebuilds the paused RunState for this workflow
// by scanning session history, instead of loading a persisted
// RunState blob. This mirrors adk-python, which does NOT persist a
// run-state event: it reconstructs paused state from the session's
// events on resume (workflow/utils/_rehydration_utils.py).
//
// For each node in the graph, it finds long-running tool call IDs the
// node raised (Event.LongRunningToolIDs, attributed by event author ==
// node name) that have no matching FunctionResponse later in history.
// A node with such open interrupts is reconstructed as NodeWaiting
// with those IDs in NodeState.Interrupts, so Resume can match a
// human's FunctionResponse back to it.
//
// Returns (nil, nil) when no node has an open interrupt (nothing to
// resume). Anonymous workflows (empty name) still reconstruct: unlike
// the persisted path, rehydration needs no name.
func (w *Workflow) ReconstructRunState(sess session.Session) (*RunState, error) {
	if sess == nil {
		return nil, nil
	}
	nodesByName := buildNodesByName(w.graph)
	events := sess.Events()

	// open maps nodeName -> set of unresolved long-running IDs. A
	// FunctionResponse anywhere in history (without re-raising the ID)
	// resolves it. Authorship attributes a raised ID to its node.
	open := map[string]map[string]struct{}{}
	answered := map[string]struct{}{}
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil {
			continue
		}
		for _, id := range ev.LongRunningToolIDs {
			if id == "" {
				continue
			}
			if _, ok := nodesByName[ev.Author]; !ok {
				continue
			}
			if open[ev.Author] == nil {
				open[ev.Author] = map[string]struct{}{}
			}
			open[ev.Author][id] = struct{}{}
		}
		// An interrupt is resolved only by a FunctionResponse the USER
		// supplied (the human/external reply on a later turn). The
		// long-running tool's own initial "pending" response is
		// authored by the agent/node, not the user, so it does NOT
		// resolve the interrupt. Mirrors adk-python rehydration, which
		// gates resolution on event.author == 'user'.
		if ev.Author != "user" || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil || p.FunctionResponse == nil || p.FunctionResponse.ID == "" {
				continue
			}
			answered[p.FunctionResponse.ID] = struct{}{}
		}
	}

	var state *RunState
	for nodeName, ids := range open {
		var interrupts []string
		for id := range ids {
			if _, done := answered[id]; done {
				continue
			}
			interrupts = append(interrupts, id)
		}
		if len(interrupts) == 0 {
			continue
		}
		if state == nil {
			state = NewRunState()
		}
		ns := state.EnsureNode(nodeName)
		ns.Status = NodeWaiting
		ns.Interrupts = interrupts
	}
	return state, nil
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
