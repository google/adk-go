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
	"fmt"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// callerActionsMu serializes RunNode's writes to the actions of the context
// it was called with: concurrent calls on one context share them.
var callerActionsMu sync.Mutex

// newFallbackSubScheduler returns the scheduler RunNode uses for one call on
// a context that carries none. Its children derive from the unwrapped
// context, and their deltas merge into its actions, because a tool or
// callback context answers Session and IsolationScope with a log line and a
// zero value, and a callback context answers Actions that way too.
func newFallbackSubScheduler(ctx agent.Context) agent.DynamicSubScheduler {
	base := ctx.WithDelta(&agent.CommonContextDelta{})
	sink := &droppedEvents{actions: base.Actions()}
	return newStandaloneSubScheduler(base, ctx.Path(), sink.deliver)
}

// isSessionless reports whether ctx is not a tool context and carries no
// session: no runner drives such a context, so RunNode has nowhere to run a
// child for it.
func isSessionless(ctx agent.Context) bool {
	return ctx.WithDelta(&agent.CommonContextDelta{}).Session() == nil && ctx.FunctionCallID() == ""
}

// droppedEvents is the emit target of a fallback scheduler. No channel leads
// from the context to the user, so an event gets no further than the actions
// of the event the context itself produces.
type droppedEvents struct {
	actions *session.EventActions
}

// deliver merges the deltas of a non-partial event into d.actions, when set,
// and refuses an event that asks for input. It runs on the child's goroutine.
func (d *droppedEvents) deliver(ev *session.Event) error {
	if ev.Partial {
		return nil
	}
	if d.actions != nil {
		callerActionsMu.Lock()
		mergeDelta(d.actions, &ev.Actions)
		callerActionsMu.Unlock()
	}
	if ev.RequestedInput != nil {
		return fmt.Errorf("cannot request input outside a workflow node: %w", ErrInvalidRunNodeContext)
	}
	return nil
}
