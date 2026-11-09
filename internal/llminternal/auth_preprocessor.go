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
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/url"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolauthinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// authPreprocessor answers the client's responses to the
// adk_request_credential calls this agent issued. When the latest event with
// content is the user's and carries such responses, it stores each credential
// in session state under "temp:" followed by the credential key of the
// request ADK issued, then runs again the tool calls that asked for those
// credentials, and yields their function response event.
//
// A resumed tool that asks for a credential again gets a new
// adk_request_credential call, as in the flow, and the invocation ends.
//
// The client's response supplies only the credential. The scheme, the
// configured credential and the key come from the issued request, so the
// client cannot choose where the credential is stored or which token
// endpoint it is exchanged at. Responses that are malformed, or answer a call
// this agent did not issue on this branch, are ignored.
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

		var last *session.Event
		for i := len(events) - 1; i >= 0; i-- {
			if events[i].Content != nil {
				last = events[i]
				break
			}
		}
		if last == nil || last.Author != "user" {
			return
		}

		var ids []string
		responses := make(map[string]map[string]any)
		for _, fr := range utils.FunctionResponses(last.Content) {
			if fr.Name != toolauth.RequestCredentialFunctionCallName || fr.ID == "" {
				continue
			}
			if _, ok := responses[fr.ID]; !ok {
				ids = append(ids, fr.ID)
			}
			responses[fr.ID] = fr.Response
		}
		if len(ids) == 0 {
			return
		}

		agentName := ctx.Agent().Name()
		targets, err := collectResumeTargets(ctx, events, agentName, ids, responses, ctx.Session().State())
		if err != nil {
			yield(nil, err)
			return
		}
		if len(targets) == 0 {
			return
		}

		for i := len(events) - 2; i >= 0; i-- {
			ev := events[i]
			var parts []*genai.Part
			for _, fc := range utils.FunctionCalls(ev.Content) {
				if targets[fc.ID] {
					call := *fc
					parts = append(parts, &genai.Part{FunctionCall: &call})
				}
			}
			if len(parts) == 0 {
				continue
			}
			// A call another agent made is left to that agent, which runs it
			// with its own tools.
			if ev.Author != agentName {
				continue
			}
			// toolProcessor runs first, so f.Tools also holds the tools of the
			// agent's toolsets.
			tools := make(map[string]tool.Tool, len(f.Tools))
			for _, t := range f.Tools {
				tools[t.Name()] = t
			}
			respEv, err := f.handleFunctionCalls(ctx, tools, &model.LLMResponse{
				Content: &genai.Content{Parts: parts, Role: ev.Content.Role},
			}, nil, nil)
			if err != nil {
				yield(nil, err)
				return
			}
			if respEv == nil {
				return
			}
			authEvent, err := generateAuthEvent(ctx, ev, respEv)
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(respEv, nil) {
				return
			}
			if authEvent != nil {
				if yield(authEvent, nil) {
					ctx.EndInvocation()
				}
			}
			return
		}
	}
}

// collectResumeTargets stores the credential in each of responses, keyed by
// adk_request_credential call ID, that answers a call in ids that agentName
// issued, and returns the IDs of the tool calls to run again: every tool call
// an issued request names, and every tool call in events that asked for a
// credential under a key that was stored.
//
// An adk_request_credential call counts as issued only when it names a tool
// call that asked for a credential in a function response event's
// Actions.RequestedAuthConfigs, which only a tool run writes. A session event
// a client seeded cannot carry RequestedAuthConfigs, so the config taken from
// that event's entry rather than from the function call arguments cannot
// supply a client-chosen token endpoint, OAuth2 client or credential key.
func collectResumeTargets(ctx context.Context, events []*session.Event, agentName string, ids []string, responses map[string]map[string]any, state session.State) (map[string]bool, error) {
	requested := findRequestedAuthConfigs(events, ids, agentName)
	dropResumedRequests(events, agentName, requested)

	authorizedKeys := make(map[string]bool)
	for _, id := range ids {
		args, ok := requested[id]
		if !ok {
			continue
		}
		stored, err := storeAuthResponse(ctx, &args.AuthConfig, responses[id], state)
		if err != nil {
			return nil, err
		}
		if stored == nil {
			continue
		}
		if key, err := stored.Key(); err == nil {
			authorizedKeys[key] = true
		}
	}

	// As in adk-python, a tool call whose request was issued is run again even
	// when its response was not stored, so the tool can ask again. adk-python
	// leaves out requests made for a toolset before it lists its tools; ADK Go
	// makes no such requests.
	targets := make(map[string]bool)
	for _, args := range requested {
		targets[args.FunctionCallID] = true
	}
	asking := make(map[string]bool)
	var matching []*session.Event
	for _, ev := range events {
		if ev.Author != agentName {
			continue
		}
		pending := ev.Actions.RequestedAuthConfigs
		for _, resp := range utils.FunctionResponses(ev.Content) {
			_, asked := pending[resp.ID]
			asking[resp.ID] = asked
		}
		for id := range pending {
			if targets[id] {
				matching = append(matching, ev)
				break
			}
		}
	}
	for _, ev := range matching {
		for id, cfg := range ev.Actions.RequestedAuthConfigs {
			if _, seen := asking[id]; seen && !asking[id] {
				continue
			}
			if key, err := cfg.Key(); err == nil && authorizedKeys[key] {
				targets[id] = true
			}
		}
	}
	return targets, nil
}

// dropResumedRequests deletes from requested the ones an earlier answer already
// resumed: agentName responded to the call a request names after the first
// answer to the request. Answering such a request again then neither stores a
// credential nor runs the tool. adk-python resumes the call again, which
// redeems a code twice and repeats the tool's side effects. A request whose
// first answer failed before the tool ran stays open, and so does one whose
// call only another party answered.
func dropResumedRequests(events []*session.Event, agentName string, requested map[string]toolauth.ToolArguments) {
	firstAnswer := make(map[string]int)
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(ev.Content) {
			if _, seen := firstAnswer[resp.ID]; !seen {
				firstAnswer[resp.ID] = i
			}
		}
	}
	for id, req := range requested {
		idx, seen := firstAnswer[id]
		if !seen {
			continue
		}
		for _, ev := range events[idx+1:] {
			if ev.Author == agentName && slices.ContainsFunc(utils.FunctionResponses(ev.Content), func(r *genai.FunctionResponse) bool { return r.ID == req.FunctionCallID }) {
				delete(requested, id)
				break
			}
		}
	}
}

// findRequestedAuthConfigs returns the arguments of the adk_request_credential
// calls in events whose IDs are in ids, keyed by call ID. Calls whose
// arguments do not decode are skipped.
//
// Only calls agentName made count, and only when the tool call they name has a
// request recorded in an event's Actions.RequestedAuthConfigs, which replaces
// the config in the call's arguments. adk-python reads the config off the
// function call arguments without checking the author; ADK Go checks both so
// that neither another agent sharing the session nor a session event a client
// seeded can issue a request this agent trusts.
func findRequestedAuthConfigs(events []*session.Event, ids []string, agentName string) map[string]toolauth.ToolArguments {
	byToolCall := make(map[string]toolauth.Config)
	for _, ev := range events {
		if ev.Author != agentName {
			continue
		}
		for callID, cfg := range ev.Actions.RequestedAuthConfigs {
			byToolCall[callID] = cfg
		}
	}

	want := make(map[string]bool, len(ids))
	for _, id := range ids {
		want[id] = true
	}
	out := make(map[string]toolauth.ToolArguments)
	for _, ev := range events {
		if ev.Author != agentName {
			continue
		}
		for _, fc := range utils.FunctionCalls(ev.Content) {
			if fc.Name != toolauth.RequestCredentialFunctionCallName || !want[fc.ID] {
				continue
			}
			b, err := json.Marshal(fc.Args)
			if err != nil {
				continue
			}
			var args toolauth.ToolArguments
			if err := json.Unmarshal(b, &args); err != nil {
				continue
			}
			cfg, ok := byToolCall[args.FunctionCallID]
			if !ok {
				continue
			}
			args.AuthConfig = cfg
			out[fc.ID] = args
		}
	}
	return out
}

// storeAuthResponse stores the credential in response, the client's answer to
// the adk_request_credential call that carried requested, and returns the
// config it stored the credential for. It returns nil, and stores nothing,
// when response is neither a [toolauth.Config] with an exchanged credential
// nor a [toolauth.Credential], or fails the OAuth2 state check. It returns an
// error when storing the credential, or exchanging it for a token, fails.
//
// Only the credential is taken from response. Unset OAuth2 client fields,
// including a PKCE code verifier, are filled in from the issued request.
//
// Configured secrets on requested are stripped with WithoutSecrets, and any
// OAuth2 clientSecret in the client's response is cleared before merging, so
// neither the request nor the response can supply a client secret to trigger a
// server-side token exchange when requested has no configured OAuth2 client.
func storeAuthResponse(ctx context.Context, requested *toolauth.Config, response map[string]any, state session.State) (*toolauth.Config, error) {
	cred := decodeAuthResponse(response)
	if cred == nil {
		if _, ok := response["authScheme"].(string); ok {
			return nil, errors.New("auth response is neither an auth config nor a credential")
		}
		return nil, nil
	}
	// adk-python checks the state against one it saved in session state,
	// which only its workflow path saves; ADK Go checks it against the state
	// in the issued request, which this agent wrote, and also against the
	// state parameter of the redirect URI when the response carries one.
	if ex := requested.ExchangedAuthCredential; ex != nil && ex.OAuth2 != nil && ex.OAuth2.State != "" {
		if !matchingOAuth2State(cred.OAuth2, ex.OAuth2.State) {
			return nil, nil
		}
	}

	stored, err := requested.WithoutSecrets()
	if err != nil {
		return nil, err
	}
	source := stored.ExchangedAuthCredential
	if source == nil {
		source = stored.RawAuthCredential
	}
	if cred.OAuth2 != nil {
		o := *cred.OAuth2
		o.ClientSecret = ""
		cred.OAuth2 = &o
	}
	stored.ExchangedAuthCredential = mergeOAuth2Fields(cred, source)
	if err := toolauthinternal.ParseAndStoreResponse(ctx, stored, state); err != nil {
		return nil, err
	}
	return stored, nil
}

// matchingOAuth2State reports whether o carries state in State and, when
// AuthResponseURI is set, in the URI's state query parameter.
func matchingOAuth2State(o *toolauth.OAuth2Auth, state string) bool {
	if o == nil || o.State != state {
		return false
	}
	if o.AuthResponseURI != "" {
		u, err := url.Parse(strings.TrimSuffix(o.AuthResponseURI, "#"))
		if err != nil || u.Query().Get("state") != state {
			return false
		}
	}
	return true
}

// decodeAuthResponse returns the credential in a client's response: the
// exchanged credential of a [toolauth.Config], or a bare
// [toolauth.Credential]. It returns nil when response is neither, or is a
// config without an exchanged credential.
func decodeAuthResponse(response map[string]any) *toolauth.Credential {
	b, err := json.Marshal(response)
	if err != nil {
		return nil
	}
	var cfg toolauth.Config
	if err := json.Unmarshal(b, &cfg); err == nil {
		return cfg.ExchangedAuthCredential
	}
	var cred toolauth.Credential
	if err := json.Unmarshal(b, &cred); err != nil {
		return nil
	}
	return &cred
}

// mergeOAuth2Fields fills the OAuth2 client fields that target leaves empty
// from source, and returns target, or source when target is nil. target is
// changed in place; source is not. adk-python fills a field only when it was
// not set at all, which Go cannot tell from empty.
func mergeOAuth2Fields(target, source *toolauth.Credential) *toolauth.Credential {
	if source == nil {
		return target
	}
	if target == nil {
		return source
	}
	if source.OAuth2 == nil {
		return target
	}
	if target.OAuth2 == nil {
		o := *source.OAuth2
		target.OAuth2 = &o
		return target
	}
	t, s := target.OAuth2, source.OAuth2
	for _, f := range []struct{ dst, src *string }{
		{&t.ClientID, &s.ClientID},
		{&t.ClientSecret, &s.ClientSecret},
		{&t.RedirectURI, &s.RedirectURI},
		{&t.CodeVerifier, &s.CodeVerifier},
		{&t.CodeChallengeMethod, &s.CodeChallengeMethod},
		{&t.TokenEndpointAuthMethod, &s.TokenEndpointAuthMethod},
	} {
		if *f.dst == "" {
			*f.dst = *f.src
		}
	}
	return target
}
