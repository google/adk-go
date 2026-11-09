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
		calls := callsToRerun(events, agentName, toolCallsToResume(events, agentName, requests, authorized))
		if len(calls) == 0 {
			return
		}
		tools := make(map[string]tool.Tool, len(f.Tools))
		for _, t := range f.Tools {
			tools[t.Name()] = t
		}
		parts := make([]*genai.Part, len(calls))
		for i, call := range calls {
			parts[i] = &genai.Part{FunctionCall: call}
		}
		ev, err := f.handleFunctionCalls(ctx, tools, &model.LLMResponse{
			Content: &genai.Content{Parts: parts, Role: genai.RoleModel},
		}, nil, nil)
		if err != nil {
			yield(nil, err)
			return
		}
		if ev != nil {
			// A tool that runs again can ask for another credential.
			yieldWithAuthRequest(ctx, ev, yield)
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

// callsToRerun returns the calls in resume from the latest event with any of
// them, when agentName made that event.
func callsToRerun(events []*session.Event, agentName string, resume map[string]bool) []*genai.FunctionCall {
	// The last event is the user's answer.
	for _, ev := range slices.Backward(events[:len(events)-1]) {
		calls := utils.FunctionCalls(ev.Content)
		if !slices.ContainsFunc(calls, func(c *genai.FunctionCall) bool { return resume[c.ID] }) {
			continue
		}
		if ev.Author != agentName {
			continue
		}
		return slices.DeleteFunc(calls, func(c *genai.FunctionCall) bool { return !resume[c.ID] })
	}
	return nil
}
