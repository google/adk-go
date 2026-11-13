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
	"maps"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// authPreprocessor stores the user's answers to adk_request_credential calls
// and runs the tool calls that asked for those credentials again, as
// adk-python's auth_preprocessor does.
func authPreprocessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if asLLMAgent(ctx.Agent()) == nil || ctx.Session() == nil {
			return
		}
		var events []*session.Event
		for ev := range ctx.Session().Events().All() {
			if utils.EventBelongsToBranch(ctx.Branch(), ev.Branch) {
				events = append(events, ev)
			}
		}
		answers := credentialAnswers(events)
		if len(answers) == 0 {
			return
		}
		agentName := ctx.Agent().Name()
		requests, err := issuedCredentialRequests(events, agentName, answers)
		if err != nil {
			yield(nil, err)
			return
		}
		dropResumedRequests(events, agentName, requests)
		// Like adk-python, the answer goes into the session's state directly
		// rather than through an event, so it is never persisted or streamed.
		authorized, err := storeCredentialAnswers(ctx.Session().State(), requests, answers)
		if err != nil {
			yield(nil, err)
			return
		}
		callEventIdx, parts := partsToRerun(events, agentName, toolCallsToResume(events, agentName, requests, authorized))
		if len(parts) == 0 {
			return
		}
		confirmations, err := answeredConfirmations(events, agentName, callEventIdx, parts)
		if err != nil {
			yield(nil, err)
			return
		}
		tools := make(map[string]tool.Tool, len(f.Tools))
		for _, t := range f.Tools {
			tools[t.Name()] = t
		}
		callEvent := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Parts: parts, Role: genai.RoleModel},
			},
		}
		ev, err := f.handleFunctionCalls(ctx, tools, &callEvent.LLMResponse, confirmations, nil)
		if err != nil {
			yield(nil, err)
			return
		}
		if ev != nil {
			// A tool that runs again can ask for another credential or for confirmation.
			if !yieldWithAuthRequest(ctx, ev, yield) {
				return
			}
			if confirmationEvent := generateRequestConfirmationEvent(ctx, callEvent, ev); confirmationEvent != nil {
				if !yield(confirmationEvent, nil) {
					return
				}
			}
		}
	}
}

// credentialAnswers returns the answers to adk_request_credential calls by
// call ID, when the latest event with content is the user's. Any later event
// means the answers were already handled earlier in the invocation.
func credentialAnswers(events []*session.Event) map[string]map[string]any {
	for _, ev := range slices.Backward(events) {
		if ev.Content == nil {
			continue
		}
		if ev.Author != "user" {
			return nil
		}
		answers := map[string]map[string]any{}
		for _, resp := range utils.FunctionResponses(ev.Content) {
			if resp.Name == toolauth.RequestCredentialFunctionCallName {
				answers[resp.ID] = resp.Response
			}
		}
		return answers
	}
	return nil
}

// issuedCredentialRequests returns, by call ID, the arguments of the
// adk_request_credential calls agentName made that answers replies to. A
// request from anyone else, such as a remote agent whose events reached this
// session, is not this agent's to resume.
func issuedCredentialRequests(events []*session.Event, agentName string, answers map[string]map[string]any) (map[string]*toolauth.ToolArguments, error) {
	requests := map[string]*toolauth.ToolArguments{}
	for _, ev := range events {
		if ev.Author != agentName {
			continue
		}
		for _, call := range utils.FunctionCalls(ev.Content) {
			if _, ok := answers[call.ID]; !ok || call.Name != toolauth.RequestCredentialFunctionCallName {
				continue
			}
			args, err := converters.FromMapStructure[toolauth.ToolArguments](call.Args)
			if err != nil {
				return nil, fmt.Errorf("read %s call %q: %w", call.Name, call.ID, err)
			}
			requests[call.ID] = args
		}
	}
	return requests, nil
}

// dropResumedRequests deletes from requests the ones an earlier answer already
// resumed: agentName responded to the call a request names after the first
// answer to the request. Answering such a request again then neither stores a
// credential nor runs the tool. adk-python resumes the call again, which
// redeems a code twice and repeats the tool's side effects. A request whose
// first answer failed before the tool ran stays open, and so does one whose
// call only another party answered.
func dropResumedRequests(events []*session.Event, agentName string, requests map[string]*toolauth.ToolArguments) {
	firstAnswer := map[string]int{}
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(ev.Content) {
			if _, seen := firstAnswer[resp.ID]; !seen {
				firstAnswer[resp.ID] = i
			}
		}
	}
	for id, req := range requests {
		for _, ev := range events[firstAnswer[id]+1:] {
			if ev.Author == agentName && slices.ContainsFunc(utils.FunctionResponses(ev.Content), func(r *genai.FunctionResponse) bool { return r.ID == req.FunctionCallID }) {
				delete(requests, id)
				break
			}
		}
	}
}

// storeCredentialAnswers stores each answer to a request in requests and
// returns the credential keys it stored.
func storeCredentialAnswers(state session.State, requests map[string]*toolauth.ToolArguments, answers map[string]map[string]any) (map[string]bool, error) {
	authorized := map[string]bool{}
	for _, id := range slices.Sorted(maps.Keys(answers)) {
		req, ok := requests[id]
		if !ok {
			continue
		}
		key, err := authhandler.StoreAuthResponse(state, &req.AuthConfig, answers[id])
		if err != nil {
			return nil, fmt.Errorf("store answer to %s call %q: %w", toolauth.RequestCredentialFunctionCallName, id, err)
		}
		authorized[key] = true
	}
	return authorized, nil
}

// toolCallsToResume returns the IDs of the tool calls to run again: the ones
// the answered requests name, and the ones that asked for a credential the
// answers provided in the same tool response as a named call, since only the
// first call per credential key was named.
//
// Only agentName's tool responses count, and a call is resumed only while its
// latest response still asks for a credential: a call that never asked, or ran
// to an answer after it asked, has nothing to resume. adk-python resumes every
// call that shared an answered key in such a response, so a call that already
// finished runs again and repeats its side effects. The REST server accepts a
// new session's events from the client, calls and requests in this agent's
// name included, but never the credential requests a tool records.
func toolCallsToResume(events []*session.Event, agentName string, requests map[string]*toolauth.ToolArguments, authorized map[string]bool) map[string]bool {
	named := map[string]bool{}
	for _, req := range requests {
		named[req.FunctionCallID] = true
	}
	resume := maps.Clone(named)
	asking := map[string]bool{}
	for _, ev := range events {
		if ev.Author != agentName {
			continue
		}
		pending := ev.Actions.RequestedAuthConfigs
		for _, resp := range utils.FunctionResponses(ev.Content) {
			_, asked := pending[resp.ID]
			asking[resp.ID] = asked
		}
		if !slices.ContainsFunc(slices.Collect(maps.Keys(pending)), func(id string) bool { return named[id] }) {
			continue
		}
		for id, cfg := range pending {
			if authorized[cfg.CredentialKey] {
				resume[id] = true
			}
		}
	}
	maps.DeleteFunc(resume, func(id string, _ bool) bool { return !asking[id] })
	return resume
}

// partsToRerun returns the event index and function call parts in resume from
// the latest event with any of them, when agentName made that event. Keeping
// the parts rather than the bare calls preserves ThoughtSignature when a rerun
// asks for confirmation.
func partsToRerun(events []*session.Event, agentName string, resume map[string]bool) (int, []*genai.Part) {
	// The last event is the user's answer.
	for i, ev := range slices.Backward(events[:len(events)-1]) {
		if ev.Content == nil || !slices.ContainsFunc(ev.Content.Parts, func(p *genai.Part) bool {
			return p != nil && p.FunctionCall != nil && resume[p.FunctionCall.ID]
		}) {
			continue
		}
		if ev.Author != agentName {
			continue
		}
		var parts []*genai.Part
		for _, p := range ev.Content.Parts {
			if p != nil && p.FunctionCall != nil && resume[p.FunctionCall.ID] {
				parts = append(parts, p)
			}
		}
		return i, parts
	}
	return -1, nil
}

type confirmationRef struct {
	confID   string
	origID   string
	eventIdx int
}

// answeredConfirmations returns the user's answers to agentName's
// adk_request_confirmation calls for the function call parts being rerun, keyed
// by original function call ID. A tool that was already approved before it
// asked for sign-in receives its approval again when authPreprocessor reruns
// it, rather than asking for confirmation a second time.
//
// Each approval is matched against the full canonical FunctionCall (ID, name,
// and arguments) emitted after callEventIdx, so a later call that reuses an
// earlier call ID cannot inherit an approval meant for a different tool,
// different arguments, or an earlier turn. Only responses targeting the
// rerun's active confirmations are decoded, from newest to oldest, so an
// unrelated or superseded malformed confirmation response does not block a
// valid sign-in.
func answeredConfirmations(events []*session.Event, agentName string, callEventIdx int, parts []*genai.Part) (map[string]*toolconfirmation.ToolConfirmation, error) {
	rerunCanonicalByID := make(map[string]string, len(parts))
	ambiguousRerunIDs := map[string]bool{}
	for _, p := range parts {
		if p == nil || p.FunctionCall == nil {
			continue
		}
		id := p.FunctionCall.ID
		if ambiguousRerunIDs[id] {
			continue
		}
		canonical, err := json.Marshal(p.FunctionCall)
		if err != nil {
			ambiguousRerunIDs[id] = true
			delete(rerunCanonicalByID, id)
			continue
		}
		if prev, seen := rerunCanonicalByID[id]; seen && prev != string(canonical) {
			ambiguousRerunIDs[id] = true
			delete(rerunCanonicalByID, id)
			continue
		}
		rerunCanonicalByID[id] = string(canonical)
	}
	if len(rerunCanonicalByID) == 0 {
		return nil, nil
	}

	originalCallByConfirmationID := map[string]string{}
	conflicting := map[string]bool{}
	latestConfByOrigID := map[string]confirmationRef{}
	for i, ev := range events {
		if ev.Author != agentName {
			continue
		}
		for _, call := range utils.FunctionCalls(ev.Content) {
			if call.Name != toolconfirmation.FunctionCallName {
				continue
			}
			orig, err := toolconfirmation.OriginalCallFrom(call)
			if err != nil {
				conflicting[call.ID] = true
				continue
			}
			canonical, err := json.Marshal(orig)
			if err != nil {
				conflicting[call.ID] = true
				continue
			}
			if prev, seen := originalCallByConfirmationID[call.ID]; seen {
				if prev != string(canonical) {
					conflicting[call.ID] = true
				}
			} else {
				originalCallByConfirmationID[call.ID] = string(canonical)
			}
			if i <= callEventIdx {
				continue
			}
			wantCanonical, ok := rerunCanonicalByID[orig.ID]
			if !ok {
				continue
			}
			if string(canonical) != wantCanonical {
				delete(latestConfByOrigID, orig.ID)
				continue
			}
			latestConfByOrigID[orig.ID] = confirmationRef{
				confID:   call.ID,
				origID:   orig.ID,
				eventIdx: i,
			}
		}
	}

	targetByConfID := make(map[string]confirmationRef, len(latestConfByOrigID))
	for _, ref := range latestConfByOrigID {
		if conflicting[ref.confID] {
			continue
		}
		targetByConfID[ref.confID] = ref
	}
	if len(targetByConfID) == 0 {
		return nil, nil
	}

	var confirmations map[string]*toolconfirmation.ToolConfirmation
	for i, ev := range slices.Backward(events) {
		if ev.Author != "user" || ev.Content == nil {
			continue
		}
		for _, p := range slices.Backward(ev.Content.Parts) {
			if p == nil || p.FunctionResponse == nil {
				continue
			}
			resp := p.FunctionResponse
			if resp.Name != toolconfirmation.FunctionCallName {
				continue
			}
			target, ok := targetByConfID[resp.ID]
			if !ok || i <= target.eventIdx {
				continue
			}
			if _, decided := confirmations[target.origID]; decided {
				continue
			}
			tc, err := parseToolConfirmation(ev.ID, resp.Response)
			if err != nil {
				return nil, err
			}
			if confirmations == nil {
				confirmations = map[string]*toolconfirmation.ToolConfirmation{}
			}
			confirmations[target.origID] = &tc
		}
	}
	return confirmations, nil
}
