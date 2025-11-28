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

package remoteagent

import (
	"fmt"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

type userFunctionCall struct {
	event     *session.Event
	taskID    a2a.TaskID
	contextID string
}

// toUserFunctionCall returns a non-nil struct when the last event in the session has a FunctionResponse
// with user-provided data. The struct contains both call and response events.
func getUserFunctionCallAt(events session.Events, index int) *userFunctionCall {
	if index < 0 || index >= events.Len() {
		return nil
	}
	candidate := events.At(index)
	if candidate.Author != "user" {
		return nil
	}
	fnCallID, ok := getFunctionResponseCallID(candidate)
	if !ok {
		return nil
	}
	for i := index - 1; i >= 0; i-- {
		event := events.At(i)
		if !isFunctionCallEvent(event, fnCallID) {
			continue
		}
		result := &userFunctionCall{event: candidate}
		tid, ctxID := adka2a.GetA2ATaskInfo(event)
		result.taskID = tid
		result.contextID = ctxID
		return result
	}
	return nil
}

func isFunctionCallEvent(event *session.Event, callID string) bool {
	if event == nil || event.Content == nil {
		return false
	}
	return slices.ContainsFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part.FunctionCall != nil && part.FunctionCall.ID == callID
	})
}

// getFunctionResponseCallID finds the first part with non-nil FunctionResponse and returns the call ID.
func getFunctionResponseCallID(event *session.Event) (string, bool) {
	if event.Content == nil {
		return "", false
	}
	responsePartIndex := slices.IndexFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part.FunctionResponse != nil
	})
	if responsePartIndex < 0 {
		return "", false
	}
	return event.Content.Parts[responsePartIndex].FunctionResponse.ID, true
}

// toMissingRemoteSessionParts returns content parts for all events we think are not present in the remote session
// and a2a contextID if it was found in a remote agent event metadata.
// We iterate session events backward until all events are processed or an event authored by a remote agent is found.
// Parts from all events we processed are returned as a single list.
// The returned contextID might be an empty string. This means the current remote agent invocation is not associates with
// any of the previous one. In this case a new contextID will be generated on the remote server.
func toMissingRemoteSessionParts(ctx agent.InvocationContext, events session.Events) ([]a2a.Part, string) {
	partCount, contextID := 0, ""
	// only events after this index are not in the remote session
	lastRemoteResponseIndex := -1
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		if event.LLMResponse.Content != nil {
			partCount += len(event.Content.Parts)
		}
		if event.Author == ctx.Agent().Name() {
			lastRemoteResponseIndex = i
			_, contextID = adka2a.GetA2ATaskInfo(event)
			break
		}
	}

	result := make([]a2a.Part, 0, partCount)
	for i := lastRemoteResponseIndex + 1; i < events.Len(); i++ {
		event := events.At(i)
		if event.Author != "user" && event.Author != ctx.Agent().Name() {
			event = presentAsUserMessage(ctx, event)
		}
		if event.Content == nil || len(event.Content.Parts) == 0 {
			continue
		}
		parts, err := adka2a.ToA2AParts(event.Content.Parts, event.LongRunningToolIDs)
		if err != nil {
			// TODO(yarolegovich): log error
			continue
		}
		result = append(result, parts...)
	}
	return result, contextID
}

func presentAsUserMessage(ctx agent.InvocationContext, agentEvent *session.Event) *session.Event {
	event := session.NewEvent(ctx.InvocationID())
	event.Author = "user"

	if agentEvent.Content == nil {
		return event
	}

	parts := make([]*genai.Part, 0, len(agentEvent.Content.Parts)+1)
	parts = append(parts, &genai.Part{Text: "For context:"})
	for _, part := range agentEvent.Content.Parts {
		if part.Thought {
			continue
		}
		if part.Text != "" {
			text := fmt.Sprintf("[%s] said: %s", agentEvent.Author, part.Text)
			parts = append(parts, genai.NewPartFromText(text))
		} else if part.FunctionCall != nil {
			call := part.FunctionCall
			text := fmt.Sprintf("[%s] called tool %s with parameters: %v", agentEvent.Author, call.Name, call.Args)
			parts = append(parts, genai.NewPartFromText(text))
		} else if part.FunctionResponse != nil {
			resp := part.FunctionResponse
			text := fmt.Sprintf("[%s] %s tool returned result: %v", agentEvent.Author, resp.Name, resp.Response)
			parts = append(parts, genai.NewPartFromText(text))
		} else {
			parts = append(parts, part)
		}
	}
	if len(parts) > 1 { // not only "For context:" part
		event.Content = genai.NewContentFromParts(parts, genai.RoleUser)
	}
	return event
}
