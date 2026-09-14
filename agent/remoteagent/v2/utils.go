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
	"bytes"
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

// getUserFunctionCallAt returns a non-nil struct when the event at index has a FunctionResponse
// with user-provided data. The struct contains both call and response events.
// agentName is the remote peer's name; when non-empty, only function calls authored by that peer match
// (aligned with adk-python's event.author == self.name check).
// scope is the invocation IsolationScope so a sibling-scope function call cannot
// leak its TaskID or contextID into this invocation.
func getUserFunctionCallAt(events session.Events, index int, agentName, scope string) *userFunctionCall {
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
		if request.IsolationScope != scope || !isFunctionCallEvent(request, fnCallID, agentName) {
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
	if event == nil || event.Content == nil || callID == "" {
		return false
	}
	// Empty agentName skips the author gate (anonymous wrappers / harnesses).
	if agentName != "" && event.Author != agentName {
		return false
	}
	return slices.ContainsFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part != nil && part.FunctionCall != nil && part.FunctionCall.ID == callID
	})
}

// collectRemoteFunctionCallIDs returns call IDs this remote peer itself emitted
// within the given IsolationScope. Function responses whose IDs are not in this
// set must not be forwarded as A2A function responses — the peer has no
// invocation to resume for a call it never made in this scope.
// When agentName is empty, the author gate is skipped (same as isFunctionCallEvent),
// so calls from any author — including coordinators — are collected.
// Events whose IsolationScope differs from scope are skipped (aligned with
// toMissingRemoteSessionParts and adk-python's task_scope filter).
func collectRemoteFunctionCallIDs(events session.Events, agentName, scope string) map[string]struct{} {
	ids := make(map[string]struct{})
	for i := 0; i < events.Len(); i++ {
		event := events.At(i)
		if event == nil || event.Content == nil {
			continue
		}
		if event.IsolationScope != scope {
			continue
		}
		// Empty agentName skips the author gate (anonymous wrappers / harnesses).
		if agentName != "" && event.Author != agentName {
			continue
		}
		for _, part := range event.Content.Parts {
			if part == nil {
				continue
			}
			if part.FunctionCall != nil && part.FunctionCall.ID != "" {
				ids[part.FunctionCall.ID] = struct{}{}
			}
		}
	}
	return ids
}

// marshalFunctionResponseJSON encodes a function-response payload without HTML
// escaping. On encode failure it returns a fixed placeholder that does not
// embed the payload (cyclic or non-JSON values such as NaN).
func marshalFunctionResponseJSON(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "<unserializable>"
	}
	return string(bytes.TrimSuffix(buf.Bytes(), []byte("\n")))
}

func unmatchedFunctionResponseText(fr *genai.FunctionResponse) string {
	name := fr.Name
	if name == "" {
		name = "<unnamed>"
	}
	return fmt.Sprintf("Tool %s returned: %s", name, marshalFunctionResponseJSON(fr.Response))
}

// getFunctionResponseCallID finds the first part with non-nil FunctionResponse and returns the call ID.
func getFunctionResponseCallID(event *session.Event) (string, bool) {
	if event.Content == nil {
		return "", false
	}
	responsePartIndex := slices.IndexFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part != nil && part.FunctionResponse != nil
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
		// Isolation scopes require an exact match, so an unscoped invocation replays
		// only unscoped events, per session.Event.IsolationScope. adk-python's remote
		// agent gates this on task mode instead
		// (remote_a2a_agent.py::_construct_message_parts_from_session); we follow the
		// Go contract, which its own prompt-history filter already uses.
		if event.IsolationScope != ctx.IsolationScope() {
			continue
		}
		if event.Author == ctx.Agent().Name() {
			lastRemoteResponseIndex = i
			_, contextID = adka2a.GetA2ATaskInfo(event)
			break
		}
		if event.LLMResponse.Content != nil {
			partCount += len(event.Content.Parts)
		}
	}

	remoteFCIDs := collectRemoteFunctionCallIDs(events, ctx.Agent().Name(), ctx.IsolationScope())
	result := make([]*a2a.Part, 0, partCount)
	for i := lastRemoteResponseIndex + 1; i < events.Len(); i++ {
		event := events.At(i)
		// Same exact-match rule as above.
		if event.IsolationScope != ctx.IsolationScope() {
			continue
		}
		// Only wrap foreign agent events as user messages when the current agent has an explicit name.
		// If Agent().Name() is empty (e.g., in anonymous wrappers or conformance harnesses), event.Author != ""
		// would falsely match and attribute events as foreign turns.
		if ctx.Agent().Name() != "" && event.Author != "user" && event.Author != ctx.Agent().Name() {
			event = presentAsUserMessage(ctx, event)
		}
		if event.Content == nil || len(event.Content.Parts) == 0 {
			continue
		}
		parts, err := convertParts(ctx, cfg, event, remoteFCIDs)
		if err != nil {
			log.Warn(ctx, "failed to convert parts for session event", "index", i, "error", err)
			continue
		}
		result = append(result, parts...)
	}
	return result, contextID
}

// hasIsolationScopeHistory reports whether the session already holds at
// least one event in the given scope. A scoped node dispatch seeds its first
// turn from UserContent because AgentNode.Run seeds it without appending an
// event, so the shared session has nothing in scope yet on that first
// dispatch. Once the scope has any history, further seeding is wrong even if
// every in-scope event has already been sent to the remote agent -- that
// case should produce an empty message so run() short-circuits, not a resend
// of the original input.
func hasIsolationScopeHistory(events session.Events, scope string) bool {
	for i := range events.Len() {
		if events.At(i).IsolationScope == scope {
			return true
		}
	}
	return false
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
		if part == nil {
			continue
		}
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

// convertParts converts genai parts to A2A parts. When remoteFCIDs is non-nil
// (history path), function responses whose IDs are not in the set are rewritten
// as text. A nil remoteFCIDs skips rewrite entirely (resume path), matching
// Python's preserve_as_resume rule that forbids mixing data FR with flattened text.
func convertParts(ctx agent.InvocationContext, cfg A2AConfig, event *session.Event, remoteFCIDs map[string]struct{}) ([]*a2a.Part, error) {
	parts := make([]*a2a.Part, 0, len(event.Content.Parts))
	for _, part := range event.Content.Parts {
		if part == nil {
			continue
		}
		if remoteFCIDs != nil && part.FunctionResponse != nil {
			if _, ok := remoteFCIDs[part.FunctionResponse.ID]; !ok {
				text := unmatchedFunctionResponseText(part.FunctionResponse)
				log.Warn(ctx, "rewrote unmatched function response as text for remote peer",
					"tool", part.FunctionResponse.Name, "id", part.FunctionResponse.ID)
				parts = append(parts, a2a.NewTextPart(text))
				continue
			}
		}
		if cfg.GenAIPartConverter != nil {
			cp, err := cfg.GenAIPartConverter(ctx, event, part)
			if err != nil {
				return nil, err
			}
			if cp != nil {
				parts = append(parts, cp)
			}
			continue
		}
		converted, err := adka2a.ToA2APart(part, event.LongRunningToolIDs)
		if err != nil {
			return nil, fmt.Errorf("event part conversion failed: %w", err)
		}
		parts = append(parts, converted)
	}
	return parts, nil
}
