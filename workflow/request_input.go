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
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// WorkflowInputFunctionCallName is the FunctionCall name carried on
// a request event so the agent runtime's generic FunctionResponse-
// by-ID dispatch can route the user's follow-up reply back to the
// workflow agent that issued the request. Mirrors
// toolconfirmation.FunctionCallName.
const WorkflowInputFunctionCallName = "adk_request_workflow_input"

// NewRequestInputEvent constructs a session.Event that asks the
// surrounding workflow to pause and surface a human-input prompt.
// A node yields the returned event and then exits its iter.Seq2;
// the scheduler observes Event.RequestedInput and transitions the
// node to NodeWaiting.
//
// The event also carries a synthesised FunctionCall part with name
// WorkflowInputFunctionCallName and ID equal to req.InterruptID,
// plus req.InterruptID in LongRunningToolIDs. This shape makes
// Event.IsFinalResponse() return true so the surrounding agent
// loop terminates after the request is yielded, and lets the
// generic FunctionResponse-by-ID dispatch route the human's reply
// back to the workflow agent that issued the request.
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

	// Synthesise the FunctionCall part so generic
	// FunctionResponse-by-ID dispatch can route the human's reply
	// back to this agent on the follow-up turn. Args mirror the
	// RequestInput fields so a client that does not parse the
	// dedicated RequestedInput field can still display the prompt
	// from the args alone.
	args := map[string]any{
		"interruptId": req.InterruptID,
		"message":     req.Message,
		"payload":     req.Payload,
	}
	if req.ResponseSchema != nil {
		args["responseSchema"] = req.ResponseSchema
	}
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					ID:   req.InterruptID,
					Name: WorkflowInputFunctionCallName,
					Args: args,
				},
			}},
		},
	}
	ev.LongRunningToolIDs = []string{req.InterruptID}

	if a := ctx.Agent(); a != nil {
		ev.Author = a.Name()
	}
	ev.Branch = ctx.Branch()

	return ev
}
