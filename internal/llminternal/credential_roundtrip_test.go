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

package llminternal_test

import (
	"context"
	"encoding/json"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/toolutils"
)

const consentToolCallID = "consent_call_1"

// consentGatedTool asks for consent the first time it is called and reports the
// consent response it was resumed with the second time. It stands in for a tool
// whose credential provider returned auth.ConsentRequiredError.
type consentGatedTool struct {
	runs         int
	sawResponses []*authconsent.AuthConfig
}

func (t *consentGatedTool) Name() string        { return "needs_consent" }
func (t *consentGatedTool) Description() string { return "a tool that needs interactive consent" }
func (t *consentGatedTool) IsLongRunning() bool { return false }
func (t *consentGatedTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "needs_consent"}
}

func (t *consentGatedTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func (t *consentGatedTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	t.runs++
	resp := ctx.AuthResponse()
	t.sawResponses = append(t.sawResponses, resp)
	if resp == nil {
		if err := ctx.RequestCredential(authconsent.OAuth2Consent("https://consent.example/auth", "n0nce", "cred-key")); err != nil {
			return nil, err
		}
		return nil, tool.ErrCredentialRequired
	}
	return map[string]any{"executed": true}, nil
}

// consentMockModel calls the gated tool on the first turn and summarizes on the
// second, the same two-turn shape the confirmation end-to-end test uses.
type consentMockModel struct {
	model.LLM
	calls int
}

func (m *consentMockModel) Name() string { return "consent-mock-model" }

func (m *consentMockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{Content: &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText("Done, after you consented.")},
				Role:  "model",
			}}, nil)
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: consentToolCallID, Name: "needs_consent", Args: map[string]any{},
			}}},
			Role: "model",
		}}, nil)
	}
}

// TestConsentRoundTripEndToEnd drives the whole flow through the runner: the
// tool asks for consent, ADK emits adk_request_credential and pauses, the client
// replies, and ADK re-runs the original tool with the response threaded in.
//
// The intermediate assertions matter as much as the final one. The emitted
// call's name, its long-running marking and its args are the entire contract
// with the client, so this is the test that fails if any of them drifts.
func TestConsentRoundTripEndToEnd(t *testing.T) {
	gated := &consentGatedTool{}
	mockModel := &consentMockModel{}

	a, err := llmagent.New(llmagent.Config{
		Name:  "consent_tester",
		Model: mockModel,
		Tools: []tool.Tool{gated},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName: "testApp", UserID: "testUser", SessionID: "testSession",
	}); err != nil {
		t.Fatalf("session Create() error = %v", err)
	}

	r, err := runner.New(runner.Config{Agent: a, SessionService: sessionService, AppName: "testApp"})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	// Turn 1: the tool asks for consent.
	var turn1 []*session.Event
	for ev, err := range r.Run(t.Context(), "testUser", "testSession",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("do the thing")}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("turn 1 run error = %v", err)
		}
		turn1 = append(turn1, ev)
	}

	if gated.runs != 1 {
		t.Fatalf("tool ran %d times in turn 1, want 1", gated.runs)
	}

	consentCall, consentEvent := findConsentCall(turn1)
	if consentCall == nil {
		t.Fatalf("turn 1 emitted no %s call. Events: %s", authconsent.FunctionCallName, describeEvents(turn1))
	}
	if consentCall.ID == "" {
		t.Fatal("emitted consent call has no id, so the client cannot address its reply")
	}
	if !containsString(consentEvent.LongRunningToolIDs, consentCall.ID) {
		t.Errorf("LongRunningToolIDs = %v, want the consent call id %q: without it the client "+
			"does not treat the call as pending", consentEvent.LongRunningToolIDs, consentCall.ID)
	}
	if got := consentCall.Args[authconsent.FunctionCallIDArg]; got != consentToolCallID {
		t.Errorf("args[%s] = %v, want %q", authconsent.FunctionCallIDArg, got, consentToolCallID)
	}
	if uri := consentURIFrom(t, consentCall); uri != "https://consent.example/auth" {
		t.Errorf("consent URL = %q, want the one the tool asked for. This is the path the "+
			"ADK web client reads (authConfig.exchangedAuthCredential.oauth2.authUri).", uri)
	}

	// Turn 2: the client replies the way the ADK web client does — the authConfig
	// it received, cloned, with the OAuth2 response fields filled in, sent as a
	// plain object. Checked against adk-web's sendOAuthResponse, which assigns
	// `response: authConfig` directly; the "response"-wrapped JSON string is the
	// confirmation widget's encoding, not this one, and is covered separately by
	// TestRequestCredentialAcceptsWrappedReply.
	reply := authconsent.AuthConfig{
		ExchangedAuthCredential: &authconsent.AuthCredential{
			AuthType: authconsent.AuthTypeOAuth2,
			OAuth2: &authconsent.OAuth2Auth{
				AuthURI:         "https://consent.example/auth",
				AuthResponseURI: "http://localhost:4200/dev-ui/?code=abc",
				RedirectURI:     "http://localhost:4200/dev-ui/",
			},
		},
	}
	replyJSON, err := json.Marshal(reply)
	if err != nil {
		t.Fatalf("Marshal(reply) error = %v", err)
	}
	var replyObject map[string]any
	if err := json.Unmarshal(replyJSON, &replyObject); err != nil {
		t.Fatalf("Unmarshal(reply) error = %v", err)
	}

	for _, err := range r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID:       consentCall.ID,
			Name:     authconsent.FunctionCallName,
			Response: replyObject,
		}}},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("turn 2 run error = %v", err)
		}
	}

	if gated.runs != 2 {
		t.Fatalf("tool ran %d times overall, want 2 (once asking for consent, once resumed)", gated.runs)
	}
	resumed := gated.sawResponses[1]
	if resumed == nil {
		t.Fatal("the resumed call saw AuthResponse() = nil, so it cannot tell it was resumed and would ask for consent again")
	}
	if resumed.CredentialKey != "cred-key" {
		t.Errorf("resumed call saw CredentialKey = %q, want %q from the agent's own request",
			resumed.CredentialKey, "cred-key")
	}
	if resumed.ExchangedAuthCredential == nil || resumed.ExchangedAuthCredential.OAuth2 == nil ||
		resumed.ExchangedAuthCredential.OAuth2.AuthResponseURI != "http://localhost:4200/dev-ui/?code=abc" {
		t.Errorf("resumed call saw %+v, want the client's callback URL", resumed)
	}
}

func findConsentCall(events []*session.Event) (*genai.FunctionCall, *session.Event) {
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == authconsent.FunctionCallName {
				return p.FunctionCall, ev
			}
		}
	}
	return nil, nil
}

// consentURIFrom reads the consent URL back out of the emitted args the way a
// client does: through JSON, not through the Go type.
func consentURIFrom(t *testing.T, call *genai.FunctionCall) string {
	t.Helper()
	b, err := json.Marshal(call.Args[authconsent.AuthConfigArg])
	if err != nil {
		t.Fatalf("Marshal(authConfig) error = %v", err)
	}
	var decoded struct {
		Exchanged struct {
			OAuth2 struct {
				AuthURI string `json:"authUri"`
			} `json:"oauth2"`
		} `json:"exchangedAuthCredential"`
	}
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("Unmarshal(authConfig) error = %v", err)
	}
	return decoded.Exchanged.OAuth2.AuthURI
}

func containsString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func describeEvents(events []*session.Event) string {
	var out []string
	for _, ev := range events {
		if ev.Content == nil {
			out = append(out, ev.Author+":<no content>")
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				out = append(out, ev.Author+":call "+p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				out = append(out, ev.Author+":response "+p.FunctionResponse.Name)
			default:
				out = append(out, ev.Author+":text")
			}
		}
	}
	b, _ := json.Marshal(out)
	return string(b)
}
