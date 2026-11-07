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
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolauthinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

const (
	authAgentName = "test_agent"
	credToolName  = "cred_tool"
)

// toolRun is one run of a credToolRecorder tool.
type toolRun struct {
	callID     string
	args       map[string]any
	credential any
}

// credToolRecorder records the runs of the tools it makes.
type credToolRecorder struct {
	mu   sync.Mutex
	runs []toolRun
}

func (r *credToolRecorder) recorded() []toolRun {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]toolRun(nil), r.runs...)
}

// tool returns a tool that records each run with the value it reads from
// state under "temp:" followed by key.
func (r *credToolRecorder) tool(name, key string) tool.Tool {
	return &mockFunctionTool{name: name, runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
		cred, _ := ctx.State().Get(session.KeyPrefixTemp + key)
		r.mu.Lock()
		r.runs = append(r.runs, toolRun{callID: ctx.FunctionCallID(), args: args, credential: cred})
		r.mu.Unlock()
		return map[string]any{"ok": true}, nil
	}}
}

func newAuthTestAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return &mockLLMAgent{Agent: a, s: &State{}}
}

// newAuthSession returns a session holding events.
func newAuthSession(t *testing.T, events []*session.Event) session.Session {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	for _, ev := range events {
		if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}
	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return got.Session
}

// runAuthPreprocessor runs authPreprocessor for agnt on a session holding
// events, and returns the events it yields and the session.
func runAuthPreprocessor(t *testing.T, agnt agent.Agent, branch string, events []*session.Event, tools ...tool.Tool) ([]*session.Event, session.Session) {
	t.Helper()
	got, sess, _ := runAuthPreprocessorContext(t, agnt, branch, events, tools...)
	return got, sess
}

// runAuthPreprocessorContext is runAuthPreprocessor that also returns the
// invocation context.
func runAuthPreprocessorContext(t *testing.T, agnt agent.Agent, branch string, events []*session.Event, tools ...tool.Tool) ([]*session.Event, session.Session, agent.InvocationContext) {
	t.Helper()
	sess := newAuthSession(t, events)
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: agnt, Session: sess, Branch: branch})
	var got []*session.Event
	for ev, err := range authPreprocessor(ctx, &model.LLMRequest{}, &Flow{Tools: tools}) {
		if err != nil {
			t.Fatalf("authPreprocessor() error = %v", err)
		}
		got = append(got, ev)
	}
	return got, sess, ctx
}

// stateCredential returns the credential stored under "temp:" followed by
// key, or nil.
func stateCredential(t *testing.T, sess session.Session, key string) *toolauth.Credential {
	t.Helper()
	v, err := sess.State().Get(session.KeyPrefixTemp + key)
	if err != nil {
		return nil
	}
	c, _ := v.(*toolauth.Credential)
	return c
}

func oauth2TestConfig(key, tokenURL string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://example.com/auth",
				TokenURL:         tokenURL,
				Scopes:           map[string]string{"read": ""},
			}},
		},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "real-client-id",
			ClientSecret: "server-secret",
			RedirectURI:  "https://example.com/callback",
		}},
		CredentialKey: key,
	}
}

// issuedRequest returns the auth request ADK issues for cfg.
func issuedRequest(t *testing.T, cfg *toolauth.Config) toolauth.Config {
	t.Helper()
	req, err := toolauthinternal.GenerateRequest(cfg)
	if err != nil {
		t.Fatalf("GenerateRequest() error = %v", err)
	}
	return *req
}

// clientResponse returns what a client sends back for issued: the request
// with the auth response URI the provider redirected to.
func clientResponse(t *testing.T, issued toolauth.Config) map[string]any {
	t.Helper()
	ex := *issued.ExchangedAuthCredential
	o := *ex.OAuth2
	o.AuthResponseURI = "https://example.com/callback?code=code-1&state=" + o.State
	ex.OAuth2 = &o
	issued.ExchangedAuthCredential = &ex
	return jsonMap(t, issued)
}

func jsonMap(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func originalCallEvent(author string, callIDs ...string) *session.Event {
	var parts []*genai.Part
	for _, id := range callIDs {
		parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: credToolName, Args: map[string]any{"q": id}}})
	}
	return &session.Event{Author: author, LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}}
}

// toolResponseEvent is the function response event of tools that asked for
// the credentials in requested, keyed by function call ID.
func toolResponseEvent(requested map[string]toolauth.Config) *session.Event {
	ev := &session.Event{Author: authAgentName}
	ev.Actions.RequestedAuthConfigs = requested
	var parts []*genai.Part
	for id := range requested {
		parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: credToolName, Response: map[string]any{"status": "needs auth"}}})
	}
	ev.Content = &genai.Content{Role: genai.RoleUser, Parts: parts}
	return ev
}

func requestEvent(t *testing.T, author, requestID, callID string, issued toolauth.Config) *session.Event {
	t.Helper()
	args := jsonMap(t, toolauth.ToolArguments{FunctionCallID: callID, AuthConfig: issued})
	return &session.Event{
		Author:             author,
		LongRunningToolIDs: []string{requestID},
		LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: requestID, Name: toolauth.RequestCredentialFunctionCallName, Args: args}},
		}}},
	}
}

func userResponseEvent(responses map[string]map[string]any) *session.Event {
	var parts []*genai.Part
	for id, resp := range responses {
		parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: toolauth.RequestCredentialFunctionCallName, Response: resp}})
	}
	return &session.Event{Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: parts}}}
}

// respondedCallIDs returns the IDs of the function responses in events.
func respondedCallIDs(events []*session.Event) []string {
	var ids []string
	for _, ev := range events {
		for _, fr := range utils.FunctionResponses(ev.Content) {
			ids = append(ids, fr.ID)
		}
	}
	return ids
}

// authFlowEvents returns the events of one tool call that asked for a
// credential and the client's answer to the request.
func authFlowEvents(t *testing.T, key string) ([]*session.Event, toolauth.Config) {
	t.Helper()
	issued := issuedRequest(t, oauth2TestConfig(key, "https://example.com/token"))
	return []*session.Event{
		originalCallEvent(authAgentName, "call-1"),
		toolResponseEvent(map[string]toolauth.Config{"call-1": issued}),
		requestEvent(t, authAgentName, "req-1", "call-1", issued),
		userResponseEvent(map[string]map[string]any{"req-1": clientResponse(t, issued)}),
	}, issued
}

func TestAuthPreprocessor(t *testing.T) {
	const key = "cred_key"
	flow, _ := authFlowEvents(t, key)
	textEvent := func(author, text string) *session.Event {
		return &session.Event{Author: author, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleUser)}}
	}
	plainAgent, err := agent.New(agent.Config{Name: authAgentName})
	if err != nil {
		t.Fatal(err)
	}

	// Each case yields nothing, stores nothing and runs no tool.
	noop := []struct {
		name   string
		agent  agent.Agent
		branch string
		events []*session.Event
	}{
		// reference: test_non_llm_agent_returns_early
		// reference: test_isinstance_check_for_llm_agent
		{name: "agent is not an LLM agent", agent: plainAgent, events: flow},
		// reference: test_empty_events_returns_early
		{name: "no events"},
		// reference: test_no_events_with_content_returns_early
		{name: "no event with content", events: []*session.Event{{Author: "user"}}},
		// reference: test_last_event_with_content_not_user_authored_returns_early
		{name: "last event with content is not the user's", events: func() []*session.Event {
			evs := append([]*session.Event{}, flow...)
			c := *evs[3]
			c.Author = "other_agent"
			evs[3] = &c
			return evs
		}()},
		// reference: test_last_event_no_responses_returns_early
		{name: "last event has no function responses", events: append(append([]*session.Event{}, flow[:3]...), textEvent("user", "hi"))},
		// reference: test_last_event_no_auth_responses_returns_early
		{name: "last event has no credential responses", events: append(append([]*session.Event{}, flow[:3]...), &session.Event{
			Author: "user",
			LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "req-1", Name: "other_function", Response: map[string]any{}}},
			}}},
		})},
		// reference: test_ignores_auth_responses_outside_current_branch
		{name: "response on another branch", branch: "root.a", events: func() []*session.Event {
			var evs []*session.Event
			for i, ev := range flow {
				c := *ev
				c.Branch = "root.a"
				if i == len(flow)-1 {
					c.Branch = "root.b"
				}
				evs = append(evs, &c)
			}
			return evs
		}()},
		// reference: test_no_matching_system_function_calls_returns_early
		{name: "response to a call that was never issued", events: append(append([]*session.Event{}, flow[:2]...),
			userResponseEvent(map[string]map[string]any{"req-unknown": flow[3].Content.Parts[0].FunctionResponse.Response}))},
		// A request another agent issued is not trusted.
		{name: "request issued by another agent", events: func() []*session.Event {
			evs := append([]*session.Event{}, flow...)
			c := *evs[2]
			c.Author = "other_agent"
			evs[2] = &c
			return evs
		}()},
	}
	for _, tc := range noop {
		t.Run(tc.name, func(t *testing.T) {
			agnt := tc.agent
			if agnt == nil {
				agnt = newAuthTestAgent(t, authAgentName)
			}
			rec := &credToolRecorder{}
			got, sess := runAuthPreprocessor(t, agnt, tc.branch, tc.events, rec.tool(credToolName, key))
			if len(got) != 0 {
				t.Errorf("authPreprocessor() yielded %d events, want 0", len(got))
			}
			if c := stateCredential(t, sess, key); c != nil {
				t.Errorf("state holds a credential, want none")
			}
			if n := len(rec.recorded()); n != 0 {
				t.Errorf("tool ran %d times, want 0", n)
			}
		})
	}

	// reference: test_processes_auth_response_successfully
	t.Run("stores the credential and resumes the tool", func(t *testing.T) {
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", flow, rec.tool(credToolName, key))
		if diff := cmp.Diff([]string{"call-1"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
		stored := stateCredential(t, sess, key)
		if stored == nil || stored.OAuth2 == nil {
			t.Fatalf("state holds no OAuth2 credential under the key")
		}
		if stored.OAuth2.ClientSecret != "" {
			t.Errorf("stored credential has a client secret")
		}
		if stored.OAuth2.AuthResponseURI == "" || stored.OAuth2.ClientID != "real-client-id" {
			t.Errorf("stored credential lacks the auth response URI or the client ID")
		}
		runs := rec.recorded()
		if len(runs) != 1 {
			t.Fatalf("tool ran %d times, want 1", len(runs))
		}
		if runs[0].callID != "call-1" || runs[0].args["q"] != "call-1" {
			t.Errorf("tool ran as call %q with args %v, want call-1 with its original args", runs[0].callID, runs[0].args)
		}
		if runs[0].credential == nil {
			t.Errorf("tool did not see the stored credential")
		}
	})

	// reference: test_processes_multiple_auth_responses_and_resumes_tools
	t.Run("resumes every tool that got a credential", func(t *testing.T) {
		issued1 := issuedRequest(t, oauth2TestConfig("key1", "https://example.com/token"))
		issued2 := issuedRequest(t, oauth2TestConfig("key2", "https://example.com/token"))
		events := []*session.Event{
			originalCallEvent(authAgentName, "call-1", "call-2"),
			toolResponseEvent(map[string]toolauth.Config{"call-1": issued1, "call-2": issued2}),
			requestEvent(t, authAgentName, "req-1", "call-1", issued1),
			requestEvent(t, authAgentName, "req-2", "call-2", issued2),
			userResponseEvent(map[string]map[string]any{
				"req-1": clientResponse(t, issued1),
				"req-2": clientResponse(t, issued2),
			}),
		}
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, "key1"))
		if diff := cmp.Diff([]string{"call-1", "call-2"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
		if stateCredential(t, sess, "key1") == nil || stateCredential(t, sess, "key2") == nil {
			t.Errorf("state does not hold both credentials")
		}
	})

	t.Run("resumed tool that asks again gets a new request", func(t *testing.T) {
		again := &mockFunctionTool{name: credToolName, runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
			ctx.Actions().RequestedAuthConfigs = map[string]toolauth.Config{
				ctx.FunctionCallID(): *oauth2TestConfig("other_key", "https://example.com/token"),
			}
			return map[string]any{"status": "needs auth"}, nil
		}}
		sess := newAuthSession(t, flow)
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			Agent:   newAuthTestAgent(t, authAgentName),
			Session: sess,
		})
		var got []*session.Event
		var hasSecretAtYield bool
		for ev, err := range authPreprocessor(ctx, &model.LLMRequest{}, &Flow{Tools: []tool.Tool{again}}) {
			if err != nil {
				t.Fatalf("authPreprocessor() error = %v", err)
			}
			if len(got) == 0 {
				if req, ok := ev.Actions.RequestedAuthConfigs["call-1"]; !ok || req.RawAuthCredential == nil || req.RawAuthCredential.OAuth2.ClientSecret != "" {
					hasSecretAtYield = true
				}
			}
			got = append(got, ev)
		}
		if len(got) != 2 {
			t.Fatalf("authPreprocessor() yielded %d events, want the function response and an auth request", len(got))
		}
		if hasSecretAtYield {
			t.Errorf("function response carried the client secret when it was yielded")
		}
		req, ok := got[0].Actions.RequestedAuthConfigs["call-1"]
		if !ok || req.RawAuthCredential == nil || req.RawAuthCredential.OAuth2.ClientSecret != "" || req.CredentialKey != "other_key" {
			t.Errorf("function response does not carry the generated request without the secret")
		}
		var requests int
		for _, fc := range utils.FunctionCalls(got[1].Content) {
			if fc.Name == toolauth.RequestCredentialFunctionCallName {
				requests++
			}
		}
		if requests != 1 {
			t.Errorf("auth event has %d credential requests, want 1", requests)
		}
		if !ctx.Ended() {
			t.Errorf("invocation did not end after the new auth request")
		}
	})

	t.Run("stops yielding when consumer breaks after the resumed function response", func(t *testing.T) {
		again := &mockFunctionTool{name: credToolName, runFunc: func(ctx agent.Context, args map[string]any) (map[string]any, error) {
			ctx.Actions().RequestedAuthConfigs = map[string]toolauth.Config{
				ctx.FunctionCallID(): *oauth2TestConfig("other_key", "https://example.com/token"),
			}
			return map[string]any{"status": "needs auth"}, nil
		}}
		f := &Flow{
			RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{authPreprocessor},
			Tools:             []tool.Tool{again},
		}
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			Agent:   newAuthTestAgent(t, authAgentName),
			Session: newAuthSession(t, flow),
		})
		var count int
		for range f.preprocess(ctx, &model.LLMRequest{}) {
			count++
			break
		}
		if count != 1 {
			t.Errorf("preprocess() yielded %d events before break, want 1", count)
		}
	})

	// reference: test_does_not_resume_tool_call_authored_by_another_agent
	t.Run("does not resume a call another agent made", func(t *testing.T) {
		events := append([]*session.Event{}, flow...)
		events[0] = originalCallEvent("other_agent", "call-1")
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, key))
		if len(got) != 0 || len(rec.recorded()) != 0 {
			t.Errorf("authPreprocessor() resumed another agent's call")
		}
		if stateCredential(t, sess, key) == nil {
			t.Errorf("state holds no credential, want it stored")
		}
	})

	// reference: test_handles_missing_original_function_calls
	t.Run("original call is missing", func(t *testing.T) {
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", flow[1:], rec.tool(credToolName, key))
		if len(got) != 0 || len(rec.recorded()) != 0 {
			t.Errorf("authPreprocessor() resumed a call that is not in the session")
		}
		if stateCredential(t, sess, key) == nil {
			t.Errorf("state holds no credential, want it stored")
		}
	})

	// reference: test_resumes_tools_by_credential_key
	t.Run("resumes every call that asked for the stored key", func(t *testing.T) {
		issued := issuedRequest(t, oauth2TestConfig(key, "https://example.com/token"))
		events := []*session.Event{
			originalCallEvent(authAgentName, "call-1", "call-2"),
			toolResponseEvent(map[string]toolauth.Config{"call-1": issued, "call-2": issued}),
			requestEvent(t, authAgentName, "req-1", "call-1", issued),
			userResponseEvent(map[string]map[string]any{"req-1": clientResponse(t, issued)}),
		}
		rec := &credToolRecorder{}
		got, _ := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, key))
		if diff := cmp.Diff([]string{"call-1", "call-2"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("does not resume a call whose different credential key was not answered", func(t *testing.T) {
		issued1 := issuedRequest(t, oauth2TestConfig("key1", "https://example.com/token"))
		issued2 := issuedRequest(t, oauth2TestConfig("key2", "https://example.com/token"))
		events := []*session.Event{
			originalCallEvent(authAgentName, "call-1", "call-2"),
			toolResponseEvent(map[string]toolauth.Config{"call-1": issued1, "call-2": issued2}),
			requestEvent(t, authAgentName, "req-1", "call-1", issued1),
			userResponseEvent(map[string]map[string]any{"req-1": clientResponse(t, issued1)}),
		}
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, "key1"))
		if diff := cmp.Diff([]string{"call-1"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
		if stateCredential(t, sess, "key1") == nil || stateCredential(t, sess, "key2") != nil {
			t.Errorf("state does not hold only key1")
		}
	})

	// reference: test_does_not_resume_stale_tools_from_older_events
	t.Run("does not resume stale calls from older events", func(t *testing.T) {
		issued := issuedRequest(t, oauth2TestConfig(key, "https://example.com/token"))
		events := []*session.Event{
			toolResponseEvent(map[string]toolauth.Config{"stale-call": issued}),
			originalCallEvent(authAgentName, "call-1", "stale-call"),
			toolResponseEvent(map[string]toolauth.Config{"call-1": issued}),
			requestEvent(t, authAgentName, "req-1", "call-1", issued),
			userResponseEvent(map[string]map[string]any{"req-1": clientResponse(t, issued)}),
		}
		rec := &credToolRecorder{}
		got, _ := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, key))
		if diff := cmp.Diff([]string{"call-1"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_malformed_auth_response_is_skipped
	t.Run("malformed response is skipped", func(t *testing.T) {
		events := append([]*session.Event{}, flow[:3]...)
		events = append(events, userResponseEvent(map[string]map[string]any{"req-1": {"authScheme": 7}}))
		rec := &credToolRecorder{}
		got, sess := runAuthPreprocessor(t, newAuthTestAgent(t, authAgentName), "", events, rec.tool(credToolName, key))
		if stateCredential(t, sess, key) != nil {
			t.Errorf("state holds a credential, want none")
		}
		// As in adk-python, the issued call is still resumed, so the tool can
		// ask again.
		if diff := cmp.Diff([]string{"call-1"}, respondedCallIDs(got)); diff != "" {
			t.Errorf("resumed calls mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestCollectResumeTargets(t *testing.T) {
	// reference: test_scheme_comes_from_the_request_not_the_response
	t.Run("scheme comes from the request, not the response, and seeded secrets are stripped", func(t *testing.T) {
		var realHits, attackerHits atomic.Int32
		tokenHandler := func(hits *atomic.Int32) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"access_token":"tok","token_type":"bearer"}`))
			})
		}
		realSrv := httptest.NewServer(tokenHandler(&realHits))
		defer realSrv.Close()
		attacker := httptest.NewServer(tokenHandler(&attackerHits))
		defer attacker.Close()

		// Even when a client-seeded request still carries a client secret,
		// storeAuthResponse strips configured secrets before storing, so
		// neither the request's nor the response's token endpoint is contacted.
		issued := *oauth2TestConfig("pinned_key", realSrv.URL)
		issued.ExchangedAuthCredential = issued.RawAuthCredential
		forged := *oauth2TestConfig("attacker_key", attacker.URL)
		forged.ExchangedAuthCredential = &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "attacker-client", AuthCode: "code-1",
		}}
		events := []*session.Event{requestEvent(t, authAgentName, "req-1", "call-1", issued)}
		sess := newAuthSession(t, nil)
		targets, err := collectResumeTargets(t.Context(), events, authAgentName, []string{"req-1"},
			map[string]map[string]any{"req-1": jsonMap(t, forged)}, sess.State())
		if err != nil {
			t.Fatalf("collectResumeTargets() error = %v", err)
		}
		if realHits.Load() != 0 || attackerHits.Load() != 0 {
			t.Errorf("real endpoint got %d requests and forged one %d, want 0 and 0", realHits.Load(), attackerHits.Load())
		}
		stored := stateCredential(t, sess, "pinned_key")
		if stored == nil || stateCredential(t, sess, "attacker_key") != nil {
			t.Fatalf("credential is not stored under the issued key alone")
		}
		if stored.OAuth2.ClientSecret != "" {
			t.Errorf("stored credential carries a client secret")
		}
		if diff := cmp.Diff(map[string]bool{"call-1": true}, targets); diff != "" {
			t.Errorf("targets mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_response_to_an_unrequested_call_id_is_ignored
	t.Run("response to an unrequested call ID is ignored", func(t *testing.T) {
		issued := issuedRequest(t, oauth2TestConfig("k", "https://example.com/token"))
		sess := newAuthSession(t, nil)
		targets, err := collectResumeTargets(t.Context(), nil, authAgentName, []string{"never-issued"},
			map[string]map[string]any{"never-issued": clientResponse(t, issued)}, sess.State())
		if err != nil {
			t.Fatalf("collectResumeTargets() error = %v", err)
		}
		if len(targets) != 0 || stateCredential(t, sess, "k") != nil {
			t.Errorf("collectResumeTargets() acted on a response nobody asked for")
		}
	})
}

func TestFindRequestedAuthConfigs(t *testing.T) {
	issued := issuedRequest(t, oauth2TestConfig("k", "https://example.com/token"))

	// reference: test_returns_matching_requests_and_ignores_unknown_ids
	t.Run("returns matching requests only", func(t *testing.T) {
		events := []*session.Event{
			requestEvent(t, authAgentName, "req-1", "call-1", issued),
			requestEvent(t, authAgentName, "req-2", "call-2", issued),
		}
		got := findRequestedAuthConfigs(events, []string{"req-1", "req-unknown"}, authAgentName)
		if len(got) != 1 || got["req-1"].FunctionCallID != "call-1" {
			t.Errorf("findRequestedAuthConfigs() = %v, want only req-1 for call-1", got)
		}
	})

	// reference: test_skips_malformed_auth_tool_arguments
	t.Run("skips malformed arguments", func(t *testing.T) {
		bad := requestEvent(t, authAgentName, "req-1", "call-1", issued)
		bad.Content.Parts[0].FunctionCall.Args = map[string]any{"functionCallId": 7}
		got := findRequestedAuthConfigs([]*session.Event{bad}, []string{"req-1"}, authAgentName)
		if len(got) != 0 {
			t.Errorf("findRequestedAuthConfigs() = %v, want none", got)
		}
	})

	t.Run("skips requests another agent issued", func(t *testing.T) {
		got := findRequestedAuthConfigs([]*session.Event{requestEvent(t, "other_agent", "req-1", "call-1", issued)}, []string{"req-1"}, authAgentName)
		if len(got) != 0 {
			t.Errorf("findRequestedAuthConfigs() = %v, want none", got)
		}
	})
}

func TestStoreAuthResponse(t *testing.T) {
	issued := issuedRequest(t, oauth2TestConfig("k", "https://example.com/token"))
	issuedState := issued.ExchangedAuthCredential.OAuth2.State

	store := func(t *testing.T, requested toolauth.Config, response map[string]any) (*toolauth.Config, session.Session) {
		t.Helper()
		sess := newAuthSession(t, nil)
		got, err := storeAuthResponse(t.Context(), &requested, response, sess.State())
		if err != nil {
			t.Fatalf("storeAuthResponse() error = %v", err)
		}
		return got, sess
	}

	// reference: test_accepts_auth_config_payload
	t.Run("config payload", func(t *testing.T) {
		got, sess := store(t, issued, clientResponse(t, issued))
		if got == nil || stateCredential(t, sess, "k") == nil {
			t.Fatalf("storeAuthResponse() did not store the credential")
		}
	})

	// reference: test_accepts_auth_credential_payload
	t.Run("credential payload", func(t *testing.T) {
		resp := map[string]any{"authType": "oauth2", "oauth2": map[string]any{"authCode": "code-1", "state": issuedState}}
		got, sess := store(t, issued, resp)
		if got == nil {
			t.Fatalf("storeAuthResponse() = nil, want the stored config")
		}
		stored := stateCredential(t, sess, "k")
		if stored == nil || stored.OAuth2.AuthCode != "code-1" {
			t.Errorf("state does not hold the credential with its code")
		}
	})

	// reference: test_rejects_auth_config_with_missing_exchanged_credential
	t.Run("config payload without an exchanged credential", func(t *testing.T) {
		resp := clientResponse(t, issued)
		delete(resp, "exchangedAuthCredential")
		if got, sess := store(t, issued, resp); got != nil || stateCredential(t, sess, "k") != nil {
			t.Errorf("storeAuthResponse() stored a response without a credential")
		}
	})

	// reference: test_malformed_response_returns_none_and_does_not_store
	t.Run("malformed payload", func(t *testing.T) {
		if got, sess := store(t, issued, map[string]any{"authType": 7}); got != nil || stateCredential(t, sess, "k") != nil {
			t.Errorf("storeAuthResponse() stored a malformed response")
		}
	})

	// reference: test_pins_credential_key_and_token_endpoint_to_requested
	t.Run("credential key and scheme come from the request", func(t *testing.T) {
		forged := clientResponse(t, issued)
		forged["credentialKey"] = "attacker_key"
		forged["authScheme"] = map[string]any{"type": "oauth2", "flows": map[string]any{
			"authorizationCode": map[string]any{"authorizationUrl": "https://attacker.example/auth", "tokenUrl": "https://attacker.example/token", "scopes": map[string]any{}},
		}}
		got, sess := store(t, issued, forged)
		if got == nil || got.CredentialKey != "k" || got.AuthScheme.Flows.AuthorizationCode.TokenURL != "https://example.com/token" {
			t.Errorf("stored config does not carry the issued key and scheme")
		}
		if stateCredential(t, sess, "k") == nil || stateCredential(t, sess, "attacker_key") != nil {
			t.Errorf("credential is not stored under the issued key alone")
		}
	})

	// reference: test_backfills_oauth2_fields_from_raw_auth_credential
	t.Run("fills unset OAuth2 fields from the request", func(t *testing.T) {
		requested := *oauth2TestConfig("k", "https://example.com/token")
		requested.RawAuthCredential.OAuth2.ClientSecret = ""
		requested.RawAuthCredential.OAuth2.CodeVerifier = "server-verifier"
		requested.RawAuthCredential.OAuth2.CodeChallengeMethod = "S256"
		resp := map[string]any{"authType": "oauth2", "oauth2": map[string]any{"authCode": "code-456"}}
		got, sess := store(t, requested, resp)
		if got == nil {
			t.Fatalf("storeAuthResponse() = nil, want the stored config")
		}
		o := stateCredential(t, sess, "k").OAuth2
		want := toolauth.OAuth2Auth{
			ClientID:            "real-client-id",
			RedirectURI:         "https://example.com/callback",
			AuthCode:            "code-456",
			CodeVerifier:        "server-verifier",
			CodeChallengeMethod: "S256",
		}
		if diff := cmp.Diff(want, *o); diff != "" {
			t.Errorf("stored OAuth2 mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_restores_and_clears_pkce_code_verifier_from_session_state
	// adk-python restores the verifier from a session state entry its
	// workflow path writes; ADK Go restores it from the issued request.
	t.Run("restores the PKCE verifier from the issued request", func(t *testing.T) {
		cfg := oauth2TestConfig("k", "https://example.com/token")
		cfg.RawAuthCredential.OAuth2.ClientSecret = ""
		pkce := issuedRequest(t, cfg)
		resp := map[string]any{"authType": "oauth2", "oauth2": map[string]any{"authCode": "code-789", "state": pkce.ExchangedAuthCredential.OAuth2.State}}
		_, sess := store(t, pkce, resp)
		o := stateCredential(t, sess, "k").OAuth2
		if o.CodeVerifier == "" || o.CodeVerifier != pkce.ExchangedAuthCredential.OAuth2.CodeVerifier || o.CodeChallengeMethod != "S256" {
			t.Errorf("stored credential does not carry the issued PKCE verifier")
		}
	})

	// reference: test_rejects_mismatched_oauth_state_from_session_state
	// reference: test_rejects_missing_oauth_state_when_expected
	// adk-python checks a state it saved in session state; ADK Go checks the
	// state in the issued request.
	for _, tc := range []struct {
		name  string
		oauth map[string]any
	}{
		{"mismatched state", map[string]any{"authCode": "code-123", "state": "wrong-state"}},
		{"missing state", map[string]any{"authCode": "code-123"}},
		{"mismatched state in auth response URI", map[string]any{"state": issuedState, "authResponseUri": "https://example.com/callback?code=code-123&state=wrong-state"}},
		{"missing state in auth response URI", map[string]any{"state": issuedState, "authResponseUri": "https://example.com/callback?code=code-123"}},
		{"malformed auth response URI", map[string]any{"state": issuedState, "authResponseUri": "http://[::1/?code=code-123&state=" + issuedState}},
	} {
		t.Run("rejects "+tc.name, func(t *testing.T) {
			resp := map[string]any{"authType": "oauth2", "oauth2": tc.oauth}
			if got, sess := store(t, issued, resp); got != nil || stateCredential(t, sess, "k") != nil {
				t.Errorf("storeAuthResponse() stored a response with the wrong state")
			}
		})
	}
	t.Run("rejects a credential without oauth2 when a state was issued", func(t *testing.T) {
		resp := map[string]any{"authType": "apiKey", "apiKey": "k"}
		if got, _ := store(t, issued, resp); got != nil {
			t.Errorf("storeAuthResponse() stored a response with no state")
		}
	})
}

func TestMergeOAuth2Fields(t *testing.T) {
	source := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
		ClientID: "id", ClientSecret: "secret", RedirectURI: "https://example.com/cb",
		CodeVerifier: "v", CodeChallengeMethod: "S256", TokenEndpointAuthMethod: "client_secret_post",
	}}

	// reference: test_store_auth_merges_oauth2_fields
	t.Run("fills only empty fields", func(t *testing.T) {
		target := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{ClientID: "client-id", AuthCode: "c"}}
		got := mergeOAuth2Fields(target, source)
		want := toolauth.OAuth2Auth{
			ClientID: "client-id", ClientSecret: "secret", RedirectURI: "https://example.com/cb", AuthCode: "c",
			CodeVerifier: "v", CodeChallengeMethod: "S256", TokenEndpointAuthMethod: "client_secret_post",
		}
		if diff := cmp.Diff(want, *got.OAuth2); diff != "" {
			t.Errorf("mergeOAuth2Fields() mismatch (-want +got):\n%s", diff)
		}
		if source.OAuth2.ClientID != "id" {
			t.Errorf("mergeOAuth2Fields() changed the source")
		}
	})

	// reference: test_merge_credential_oauth2_fields_when_target_oauth2_is_none
	t.Run("target without oauth2 gets a copy of the source's", func(t *testing.T) {
		target := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2}
		got := mergeOAuth2Fields(target, source)
		if got.OAuth2 == nil || got.OAuth2 == source.OAuth2 {
			t.Fatalf("mergeOAuth2Fields() did not copy the source's oauth2 section")
		}
		if diff := cmp.Diff(source.OAuth2, got.OAuth2); diff != "" {
			t.Errorf("mergeOAuth2Fields() oauth2 mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil target or source", func(t *testing.T) {
		if mergeOAuth2Fields(nil, source) != source {
			t.Errorf("mergeOAuth2Fields(nil, source) is not source")
		}
		target := &toolauth.Credential{}
		if mergeOAuth2Fields(target, nil) != target {
			t.Errorf("mergeOAuth2Fields(target, nil) is not target")
		}
	})
}

// oneCallModel answers its first request with a call to fetch_data and every
// later one with text.
type oneCallModel struct{ calls int }

func (m *oneCallModel) Name() string { return "one-call-model" }

func (m *oneCallModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "fetch_data", Args: map[string]any{}}},
		}}}, nil)
	}
}

func TestRunEndsInvocationOnAuthRequest(t *testing.T) {
	ft, err := functiontool.New(functiontool.Config{Name: "fetch_data", Description: "fetches data"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			ctx.Actions().RequestedAuthConfigs = map[string]toolauth.Config{
				ctx.FunctionCallID(): *oauth2TestConfig("k", "https://example.com/token"),
			}
			return map[string]any{"status": "needs auth"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	m := &oneCallModel{}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   newAuthTestAgent(t, authAgentName),
		Session: newAuthSession(t, nil),
	})
	var last *session.Event
	for ev, err := range (&Flow{Model: m, Tools: []tool.Tool{ft}}).Run(ctx) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		last = ev
	}
	if len(credentialRequestsIn(last)) != 1 {
		t.Fatalf("Run() did not end with an auth request")
	}
	if !ctx.Ended() {
		t.Errorf("invocation did not end after the auth request")
	}
	if m.calls != 1 {
		t.Errorf("model was called %d times, want 1", m.calls)
	}
}

// credentialRequestsIn returns the adk_request_credential calls in ev.
func credentialRequestsIn(ev *session.Event) []*genai.FunctionCall {
	var calls []*genai.FunctionCall
	if ev == nil {
		return nil
	}
	for _, fc := range utils.FunctionCalls(ev.Content) {
		if fc.Name == toolauth.RequestCredentialFunctionCallName {
			calls = append(calls, fc)
		}
	}
	return calls
}
