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
	"github.com/google/uuid"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// NewRequestInputEvent constructs a session.Event that asks the
// surrounding workflow to pause and surface a human-input prompt.
// A node yields the returned event and then exits its iter.Seq2;
// the scheduler observes Event.RequestedInput and transitions the
// node to NodeWaiting.
//
// If req.InterruptID is empty, a UUID is generated so the
// downstream contract (a non-empty correlation key on
// NodeState.PendingRequest) is always satisfied. Pass an explicit
// InterruptID when the prompt has a stable, author-defined intent
// that the UI or a multi-prompt node needs to recognise.
//
// Example:
//
//	func (n *MyNode) Run(ctx agent.InvocationContext, in any) iter.Seq2[*session.Event, error] {
//	    return func(yield func(*session.Event, error) bool) {
//	        yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
//	            InterruptID: "human_review",
//	            Message:     "Please review this draft.",
//	            Payload:     draft,
//	        }), nil)
//	    }
//	}
func NewRequestInputEvent(ctx agent.InvocationContext, req session.RequestInput) *session.Event {
	if req.InterruptID == "" {
		req.InterruptID = uuid.NewString()
	}
	ev := session.NewEvent(ctx.InvocationID())
	ev.RequestedInput = &req
	return ev
}
