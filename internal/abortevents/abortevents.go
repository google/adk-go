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

// Package abortevents builds the events that seal an aborted invocation and
// recognizes them.
package abortevents

import (
	"context"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

const (
	// ErrorCode is the error code of every event that seals an aborted
	// invocation.
	ErrorCode = "INVOCATION_ABORTED"
	// Message is the error message of every event that seals an aborted
	// invocation, and the "error" value of each function response it holds.
	Message = "Invocation was aborted by client."
)

// IsAbortEvent reports whether ev seals an aborted invocation, that is,
// whether its error code is [ErrorCode]. It reports false for nil.
func IsAbortEvent(ev *session.Event) bool {
	return ev != nil && ev.ErrorCode == ErrorCode
}

// BuildAbortEvents returns the events that seal the aborted invocation invID.
//
// events are the session's events in order; only those whose InvocationID is
// invID are read, and none is modified. A function call is open when it has a
// non-empty ID and no function response with that ID in the invocation;
// long-running calls are no exception. BuildAbortEvents returns:
//
//   - no events when the invocation's last event is the final reply of a task
//     agent (an event with an isolation scope, not authored by "user", that is
//     a final response), because that agent is paused and waits for the user;
//   - otherwise one event per (author, branch, isolation scope) of the open
//     calls, in the order each group first appears; an event's branch is the
//     call event's branch, or defaultBranch when that is empty; its content
//     has role "user" and one function response {"error": [Message]} per
//     call, in call order;
//   - otherwise, when no call is open, one event with no content authored by
//     rootAgentName on defaultBranch.
//
// Every returned event has error code [ErrorCode], error message [Message],
// and the ID, timestamp and empty action maps that [session.NewEvent] gives it
// under ctx. The result is nil exactly when it holds no events.
func BuildAbortEvents(ctx context.Context, events []*session.Event, invID, rootAgentName, defaultBranch string) []*session.Event {
	var invEvents []*session.Event
	for _, ev := range events {
		if ev.InvocationID == invID {
			invEvents = append(invEvents, ev)
		}
	}
	if len(invEvents) > 0 && isPausedTaskReply(invEvents[len(invEvents)-1]) {
		return nil
	}

	answered := map[string]bool{}
	for _, ev := range invEvents {
		for _, fr := range utils.FunctionResponses(ev.Content) {
			answered[fr.ID] = true
		}
	}

	// order keeps the groups in first-seen order; map iteration would not.
	var order []groupKey
	groups := map[groupKey][]*genai.FunctionCall{}
	for _, ev := range invEvents {
		for _, fc := range utils.FunctionCalls(ev.Content) {
			if fc.ID == "" || answered[fc.ID] {
				continue
			}
			branch := ev.Branch
			if branch == "" {
				branch = defaultBranch
			}
			key := groupKey{author: ev.Author, branch: branch, isolationScope: ev.IsolationScope}
			if _, ok := groups[key]; !ok {
				order = append(order, key)
			}
			groups[key] = append(groups[key], fc)
		}
	}

	if len(order) == 0 {
		return []*session.Event{newAbortEvent(ctx, invID, groupKey{author: rootAgentName, branch: defaultBranch}, nil)}
	}
	out := make([]*session.Event, 0, len(order))
	for _, key := range order {
		out = append(out, newAbortEvent(ctx, invID, key, groups[key]))
	}
	return out
}

type groupKey struct {
	author, branch, isolationScope string
}

// isPausedTaskReply reports whether ev is a task agent's reply that pauses
// the task until the user answers; such an invocation is not sealed.
func isPausedTaskReply(ev *session.Event) bool {
	return ev.IsolationScope != "" && ev.Author != "user" && ev.IsFinalResponse()
}

// newAbortEvent returns the sealing event for key, answering calls; with no
// calls it has no content.
func newAbortEvent(ctx context.Context, invID string, key groupKey, calls []*genai.FunctionCall) *session.Event {
	ev := session.NewEvent(ctx, invID)
	ev.Author = key.author
	ev.Branch = key.branch
	ev.IsolationScope = key.isolationScope
	ev.ErrorCode = ErrorCode
	ev.ErrorMessage = Message
	if len(calls) == 0 {
		return ev
	}
	content := &genai.Content{Role: genai.RoleUser}
	for _, fc := range calls {
		content.Parts = append(content.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       fc.ID,
			Name:     fc.Name,
			Response: map[string]any{"error": Message},
		}})
	}
	ev.Content = content
	return ev
}
