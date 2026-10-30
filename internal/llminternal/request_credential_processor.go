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

package llminternal

import (
	"encoding/json"
	"fmt"
	"iter"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/authconsent"
)

// RequestCredentialRequestProcessor resumes tool calls paused for interactive
// (3-legged) OAuth consent. It is the credential twin of
// [RequestConfirmationRequestProcessor] and the Go port of adk-python's
// auth_preprocessor: when the latest user turn carries adk_request_credential
// function responses, it matches each back to the paused tool call and re-runs
// that tool with the consent response threaded in.
func RequestCredentialRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if asLLMAgent(ctx.Agent()) == nil {
			return // Matching the confirmation processor: no error is yielded.
		}

		toolsmap := make(map[string]tool.Tool, len(f.Tools))
		for _, t := range f.Tools {
			toolsmap[t.Name()] = t
		}

		var events []*session.Event
		if ctx.Session() != nil {
			for e := range ctx.Session().Events().All() {
				events = append(events, e)
			}
		}

		// Collect the adk_request_credential responses from the latest user turn,
		// keyed by the id of the consent call each answers.
		consentResponses := make(map[string]authconsent.AuthConfig)
		for k := len(events) - 1; k >= 0; k-- {
			event := events[k]
			if event.Author != "user" {
				continue
			}
			responses := utils.FunctionResponses(event.Content)
			if len(responses) == 0 {
				return
			}
			for _, funcResp := range responses {
				if funcResp.Name != authconsent.FunctionCallName {
					continue
				}
				cfg, err := decodeConsentResponse(funcResp, event.ID)
				if err != nil {
					yield(nil, err)
					return
				}
				consentResponses[funcResp.ID] = cfg
			}
			break
		}

		if len(consentResponses) == 0 {
			return
		}

		agentName := ctx.Agent().Name()

		// Resolve each consent call to the tool call it paused, reading
		// functionCallId off the agent-authored request rather than off the
		// client's reply. The same reasoning as the confirmation processor's
		// tampering check applies, one step earlier: the user's reply carries only
		// the consent call's id, so the binding from consent to tool call comes
		// entirely from an event in the session. Honor that binding only from a
		// consent call this flow emitted — authored by this agent and marked
		// long-running, which generateRequestCredentialEvent always does and a
		// model emitting a call under the reserved name cannot do — and refuse a
		// consent id that two events bind to different tool calls: a single
		// consent cannot be known to cover both, so fail closed and resume
		// neither.
		resumeTarget := make(map[string]string, len(consentResponses))
		credentialKey := make(map[string]string, len(consentResponses))
		requestIndex := make(map[string]int, len(consentResponses))
		conflicting := make(map[string]bool)
		// Every consent id that binds each tool call, answered or not. A call
		// bound to two different consent ids is ambiguous whichever of them the
		// client answers, so the count has to run over the whole history rather
		// than over the ids in this reply.
		consentsByTarget := make(map[string]map[string]bool)
		for i, event := range events {
			if event.Author != agentName {
				continue
			}
			for _, functionCall := range utils.FunctionCalls(event.Content) {
				if functionCall.Name != authconsent.FunctionCallName {
					continue
				}
				if !slices.Contains(event.LongRunningToolIDs, functionCall.ID) {
					continue
				}
				targetID, ok := pausedCallID(functionCall)
				if !ok {
					continue
				}
				if consentsByTarget[targetID] == nil {
					consentsByTarget[targetID] = make(map[string]bool)
				}
				consentsByTarget[targetID][functionCall.ID] = true
				if _, ok := consentResponses[functionCall.ID]; !ok {
					continue
				}
				if prev, seen := resumeTarget[functionCall.ID]; seen {
					// The key is part of the binding: two requests under one consent
					// id that disagree on it are as ambiguous as two that disagree on
					// the tool call.
					if prev != targetID || credentialKey[functionCall.ID] != requestedCredentialKey(functionCall) {
						conflicting[functionCall.ID] = true
					}
					continue
				}
				resumeTarget[functionCall.ID] = targetID
				credentialKey[functionCall.ID] = requestedCredentialKey(functionCall)
				requestIndex[functionCall.ID] = i
			}
		}

		// Key the consent responses by the tool call to resume. The credential key
		// comes from the agent's own request, not from the client's reply, so a
		// client cannot point the resumed tool at another credential.
		authResponses := make(map[string]*authconsent.AuthConfig, len(resumeTarget))
		for consentID, targetID := range resumeTarget {
			// The mirror ambiguity: one tool call named by two consent ids. Picking
			// one would resume the tool under one credential key or the other,
			// decided by which id the client happened to answer, or by map order
			// when both arrive together. Neither consent is known to cover the
			// other, so resume under neither.
			if conflicting[consentID] || len(consentsByTarget[targetID]) > 1 {
				continue
			}
			if answeredAfter(events, requestIndex[consentID], targetID) {
				// The paused call already has a response newer than the consent
				// request, so this consent was spent. A reply sent again later — a
				// retried POST, a reloaded consent popup — must not run the tool a
				// second time. Bounding by the reply's own position instead would
				// miss it, because a replayed reply is the newest event.
				continue
			}
			cfg := consentResponses[consentID]
			cfg.CredentialKey = credentialKey[consentID]
			authResponses[targetID] = &cfg
		}
		if len(authResponses) == 0 {
			return
		}

		for k := len(events) - 2; k >= 0 && len(authResponses) > 0; k-- {
			event := events[k]
			// Only this agent's own tool calls may be resumed. Function call parts
			// reach the session from elsewhere too, notably an A2A peer response
			// converted into a model-role event, and resuming one of those would let
			// the peer choose which local tool runs and with what arguments — behind
			// a consent the user granted for something else.
			if event.Author != agentName {
				continue
			}
			calls := utils.FunctionCalls(event.Content)
			if len(calls) == 0 {
				continue
			}

			// Iterate calls (an ordered slice) rather than the authResponses map, so
			// the re-dispatched parts, the merged response and the resulting
			// StateDelta last-writer-wins merge are deterministic across runs.
			toResume := make(map[string]*authconsent.AuthConfig, len(authResponses))
			var resumeOrder []string
			for _, functionCall := range calls {
				cfg, ok := authResponses[functionCall.ID]
				if !ok {
					continue
				}
				if _, seen := toResume[functionCall.ID]; seen {
					continue
				}
				resumeOrder = append(resumeOrder, functionCall.ID)
				toResume[functionCall.ID] = cfg
			}
			if len(toResume) == 0 {
				continue
			}

			parts := make([]*genai.Part, 0, len(toResume))
			for _, callID := range resumeOrder {
				if _, ok := toResume[callID]; !ok {
					continue
				}
				for _, functionCall := range calls {
					if functionCall.ID == callID {
						// Copy the call, args included to any depth: functionCall points
						// into a stored session event, and a tool or callback that edits
						// its args in place — a nested map as much as a top-level key —
						// would otherwise rewrite history.
						call := *functionCall
						call.Args = cloneArgs(functionCall.Args)
						parts = append(parts, &genai.Part{FunctionCall: &call})
						break
					}
				}
			}

			ev, err := f.handleFunctionCalls(ctx, toolsmap, &model.LLMResponse{
				Content: &genai.Content{Parts: parts, Role: genai.RoleUser},
			}, nil, toResume, nil)
			if !yield(ev, err) || err != nil {
				return
			}
			// A consent is spent once its call has been resumed, so a call whose
			// id also appears in an earlier event is not dispatched again. The
			// scan goes on for the consents this reply answered in earlier turns:
			// a client may consent to two prompts at once, and adk-python's early
			// return would drop the older one.
			for _, callID := range resumeOrder {
				delete(authResponses, callID)
			}
		}
	}
}

// answeredAfter reports whether any event after index i carries a function
// response for callID.
func answeredAfter(events []*session.Event, i int, callID string) bool {
	for _, ev := range events[i+1:] {
		for _, resp := range utils.FunctionResponses(ev.Content) {
			if resp.ID == callID {
				return true
			}
		}
	}
	return false
}

// pausedCallID returns the id of the tool call an adk_request_credential call is
// asking consent for. It is the "functionCallId" argument adk-python's
// AuthToolArguments carries.
func pausedCallID(functionCall *genai.FunctionCall) (string, bool) {
	if functionCall == nil || functionCall.Args == nil {
		return "", false
	}
	id, ok := functionCall.Args[authconsent.FunctionCallIDArg].(string)
	if !ok || id == "" {
		return "", false
	}
	return id, true
}

// requestedCredentialKey returns the credential key the agent put in the consent
// request, or "" when the request carried none. The request is re-encoded
// through JSON because a call read back from a session store holds its args as
// plain maps, while one still in memory holds the original struct.
func requestedCredentialKey(functionCall *genai.FunctionCall) string {
	if functionCall == nil || functionCall.Args == nil {
		return ""
	}
	raw, ok := functionCall.Args[authconsent.AuthConfigArg]
	if !ok {
		return ""
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return ""
	}
	var cfg authconsent.AuthConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return ""
	}
	return cfg.CredentialKey
}

// decodeConsentResponse extracts an [authconsent.AuthConfig] from a client's
// function response, accepting two encodings: a plain object, which is what the
// ADK web client sends for consent, and the payload wrapped in a single
// "response" JSON string, which is what it sends for confirmation. Both are
// accepted because the confirmation processor accepts both and a client is free
// to use either.
func decodeConsentResponse(funcResp *genai.FunctionResponse, eventID string) (authconsent.AuthConfig, error) {
	var cfg authconsent.AuthConfig
	if funcResp.Response == nil {
		return cfg, nil
	}
	if raw, ok := funcResp.Response["response"]; ok && len(funcResp.Response) == 1 {
		s, ok := raw.(string)
		if !ok {
			return cfg, fmt.Errorf("credential response for event id %q: 'response' key is not a string", eventID)
		}
		if err := json.Unmarshal([]byte(s), &cfg); err != nil {
			return cfg, fmt.Errorf("credential response for event id %q: unmarshal 'response': %w", eventID, err)
		}
		return cfg, nil
	}
	b, err := json.Marshal(funcResp.Response)
	if err != nil {
		return cfg, fmt.Errorf("credential response for event id %q: marshal: %w", eventID, err)
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, fmt.Errorf("credential response for event id %q: unmarshal: %w", eventID, err)
	}
	return cfg, nil
}

// cloneArgs deep-copies JSON-shaped tool arguments: maps and slices are
// rebuilt, scalars are shared, which is what a value decoded from JSON holds.
func cloneArgs(args map[string]any) map[string]any {
	if args == nil {
		return nil
	}
	return cloneJSONValue(args).(map[string]any)
}

func cloneJSONValue(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			out[k] = cloneJSONValue(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = cloneJSONValue(val)
		}
		return out
	default:
		return v
	}
}
