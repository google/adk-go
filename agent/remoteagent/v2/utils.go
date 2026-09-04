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
	"encoding/json"
	"fmt"
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/log"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

type userFunctionCall struct {
	response  *session.Event
	taskID    a2a.TaskID
	contextID string
}

// toUserFunctionCall returns a non-nil struct when the last event in the session has a FunctionResponse
// with user-provided data. The struct contains both call and response events.
// agentName is the remote peer's name; when non-empty, only function calls authored by that peer match
// (aligned with adk-python's event.author == self.name check).
func getUserFunctionCallAt(events session.Events, index int, agentName string) *userFunctionCall {
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
		request := events.At(i)
		if !isFunctionCallEvent(request, fnCallID, agentName) {
			continue
		}
		result := &userFunctionCall{response: candidate}
		tid, ctxID := adka2a.GetA2ATaskInfo(request)
		result.taskID = tid
		result.contextID = ctxID
		return result
	}
	return nil
}

func isFunctionCallEvent(event *session.Event, callID, agentName string) bool {
	if event == nil || event.Content == nil {
		return false
	}
	// Empty agentName skips the author gate (anonymous wrappers / harnesses).
	if agentName != "" && event.Author != agentName {
		return false
	}
	return slices.ContainsFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part.FunctionCall != nil && part.FunctionCall.ID == callID
	})
}

// collectRemoteFunctionCallIDs returns call IDs this remote peer itself emitted.
// Function responses whose IDs are not in this set must not be forwarded as A2A
// function responses — the peer has no invocation to resume for a call it never made.
func collectRemoteFunctionCallIDs(events session.Events, agentName string) map[string]struct{} {
	ids := make(map[string]struct{})
	if agentName == "" {
		return ids
	}
	for i := 0; i < events.Len(); i++ {
		event := events.At(i)
		if event.Author != agentName || event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.FunctionCall != nil && part.FunctionCall.ID != "" {
				ids[part.FunctionCall.ID] = struct{}{}
			}
		}
	}
	return ids
}

// unmatchedFunctionResponsesAsText rewrites function responses whose call ID was
// not issued by this peer into plain text, matching adk-python remote_a2a_agent.
func unmatchedFunctionResponsesAsText(parts []*genai.Part, remoteFCIDs map[string]struct{}) []*genai.Part {
	needRewrite := false
	for _, part := range parts {
		if part.FunctionResponse == nil {
			continue
		}
		if _, ok := remoteFCIDs[part.FunctionResponse.ID]; !ok {
			needRewrite = true
			break
		}
	}
	if !needRewrite {
		return parts
	}
	out := make([]*genai.Part, 0, len(parts))
	for _, part := range parts {
		if part.FunctionResponse == nil {
			out = append(out, part)
			continue
		}
		if _, ok := remoteFCIDs[part.FunctionResponse.ID]; ok {
			out = append(out, part)
			continue
		}
		respJSON, err := json.Marshal(part.FunctionResponse.Response)
		if err != nil {
			respJSON = []byte(fmt.Sprintf("%v", part.FunctionResponse.Response))
		}
		text := fmt.Sprintf("Tool %s returned: %s", part.FunctionResponse.Name, respJSON)
		out = append(out, genai.NewPartFromText(text))
	}
	return out
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
func toMissingRemoteSessionParts(ctx agent.InvocationContext, events session.Events, cfg A2AConfig) ([]*a2a.Part, string) {
	partCount, contextID := 0, ""
	// only events after this index are not in the remote session
	lastRemoteResponseIndex := -1
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		if event.Author == ctx.Agent().Name() {
			lastRemoteResponseIndex = i
			_, contextID = adka2a.GetA2ATaskInfo(event)
			break
		}
		if event.LLMResponse.Content != nil {
			partCount += len(event.Content.Parts)
		}
	}

	result := make([]*a2a.Part, 0, partCount)
	for i := lastRemoteResponseIndex + 1; i < events.Len(); i++ {
		event := events.At(i)
		// Only wrap foreign agent events as user messages when the current agent has an explicit name.
		// If Agent().Name() is empty (e.g., in anonymous wrappers or conformance harnesses), event.Author != ""
		// would falsely match and attribute events as foreign turns.
		if ctx.Agent().Name() != "" && event.Author != "user" && event.Author != ctx.Agent().Name() {
			event = presentAsUserMessage(ctx, event)
		}
		if event.Content == nil || len(event.Content.Parts) == 0 {
			continue
		}
		parts, err := convertParts(ctx, cfg, event)
		if err != nil {
			log.Warn(ctx, "failed to convert parts for session event", "index", i, "error", err)
			continue
		}
		result = append(result, parts...)
	}
	return result, contextID
}

func presentAsUserMessage(ctx agent.InvocationContext, agentEvent *session.Event) *session.Event {
	event := session.NewEvent(ctx, ctx.InvocationID())
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

func convertParts(ctx agent.InvocationContext, cfg A2AConfig, event *session.Event) ([]*a2a.Part, error) {
	remoteFCIDs := collectRemoteFunctionCallIDs(ctx.Session().Events(), ctx.Agent().Name())
	genaiParts := unmatchedFunctionResponsesAsText(event.Content.Parts, remoteFCIDs)
	parts := make([]*a2a.Part, 0, len(genaiParts))
	if cfg.GenAIPartConverter != nil {
		for _, part := range genaiParts {
			cp, err := cfg.GenAIPartConverter(ctx, event, part)
			if err != nil {
				return nil, err
			}
			if cp != nil {
				parts = append(parts, cp)
			}
		}
	} else {
		var err error
		parts, err = adka2a.ToA2AParts(genaiParts, event.LongRunningToolIDs)
		if err != nil {
			return nil, fmt.Errorf("event part conversion failed: %w", err)
		}
	}
	return parts, nil
}
