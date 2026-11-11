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

package models_test

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// TestEventRoundTripPreservesWorkflowFields asserts the REST event model
// round-trips every v2 workflow field added to session.Event so external
// readers and re-submitted events do not lose them.
func TestEventRoundTripPreservesWorkflowFields(t *testing.T) {
	ev := session.Event{
		ID:             "e1",
		Timestamp:      time.UnixMicro(1733040000123456).UTC(),
		Author:         "agent",
		IsolationScope: "scope-1",
		Routes:         []string{"next-node", "fallback"},
		RequestedInput: &session.RequestInput{InterruptID: "approve", Message: "ok?"},
		NodeInfo:       &session.NodeInfo{Path: "root/child", MessageAsOutput: true},
		Output:         "hello",
		Actions: session.EventActions{
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call-1": {Hint: "please confirm"},
			},
		},
	}

	restEvent := models.FromSessionEvent(ev)
	if restEvent.Timestamp != 1733040000.123456 {
		t.Errorf("FromSessionEvent().Timestamp = %f, want 1733040000.123456", restEvent.Timestamp)
	}
	rawJSON, err := json.Marshal(restEvent)
	if err != nil {
		t.Fatalf("json.Marshal(restEvent) failed: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(rawJSON, &wire); err != nil {
		t.Fatalf("json.Unmarshal(rawJSON) failed: %v", err)
	}
	if gotTS, ok := wire["timestamp"].(float64); !ok || gotTS != 1733040000.123456 {
		t.Errorf("wire[\"timestamp\"] = %v, want 1733040000.123456", wire["timestamp"])
	}
	var decodedRestEvent models.Event
	if err := json.Unmarshal(rawJSON, &decodedRestEvent); err != nil {
		t.Fatalf("json.Unmarshal(rawJSON, &decodedRestEvent) failed: %v", err)
	}

	back, err := models.ToSessionEvent(decodedRestEvent)
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}

	if !back.Timestamp.Equal(ev.Timestamp) {
		t.Errorf("Timestamp = %v, want %v", back.Timestamp, ev.Timestamp)
	}
	zeroBack, err := models.ToSessionEvent(models.FromSessionEvent(session.Event{}))
	if err != nil {
		t.Fatalf("ToSessionEvent(zero event) failed: %v", err)
	}
	if !zeroBack.Timestamp.IsZero() {
		t.Errorf("zero Timestamp round-trip = %v, want zero time.Time", zeroBack.Timestamp)
	}

	if back.IsolationScope != ev.IsolationScope {
		t.Errorf("IsolationScope = %q, want %q", back.IsolationScope, ev.IsolationScope)
	}
	if len(back.Routes) != len(ev.Routes) || back.Routes[0] != "next-node" {
		t.Errorf("Routes = %v, want %v", back.Routes, ev.Routes)
	}
	if back.RequestedInput == nil || back.RequestedInput.InterruptID != "approve" {
		t.Errorf("RequestedInput = %+v, want InterruptID=approve", back.RequestedInput)
	}
	if back.NodeInfo == nil || back.NodeInfo.Path != "root/child" {
		t.Errorf("NodeInfo = %+v, want Path=root/child", back.NodeInfo)
	}
	if got, ok := back.Actions.RequestedToolConfirmations["call-1"]; !ok || got.Hint != "please confirm" {
		t.Errorf("RequestedToolConfirmations = %+v, want call-1 hint", back.Actions.RequestedToolConfirmations)
	}
}

// TestCompactionIsReadOnlyOverREST pins the direction the compaction record may
// travel across the REST boundary.
//
// A record tells prompt assembly to drop a span of history and show its own
// content instead. The create-session body maps onto a stored event, so
// accepting one inbound would let a client erase part of a conversation and
// speak into the gap. Reads are fine and useful: a client should be able to see
// summaries the framework wrote.
func TestCompactionIsReadOnlyOverREST(t *testing.T) {
	t.Parallel()

	record := &session.EventCompaction{
		StartTimestamp:   time.Unix(1, 0),
		EndTimestamp:     time.Unix(9, 0),
		CompactedContent: genai.NewContentFromText("planted", "model"),
	}

	// Inbound: a client-supplied record must not survive.
	in := models.Event{Actions: models.EventActions{Compaction: record}}
	got, err := models.ToSessionEvent(in)
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}
	if got.Actions.Compaction != nil {
		t.Error("ToSessionEvent kept a client-supplied compaction record; it must be dropped")
	}

	// Outbound: a stored record must be returned.
	stored := session.Event{Actions: session.EventActions{Compaction: record}}
	out := models.FromSessionEvent(stored)
	if out.Actions.Compaction == nil {
		t.Fatal("FromSessionEvent dropped the stored compaction record; reads should show it")
	}
	if !out.Actions.Compaction.EndTimestamp.Equal(record.EndTimestamp) {
		t.Errorf("EndTimestamp = %v, want %v", out.Actions.Compaction.EndTimestamp, record.EndTimestamp)
	}
}

// TestAgentStateIsReadOnlyOverREST pins that a client can read a workflow
// agent's checkpoint but not plant one. A planted state would tell a resumed
// sequence where to continue from, letting a client skip a sub-agent such as
// an approval step.
func TestAgentStateIsReadOnlyOverREST(t *testing.T) {
	t.Parallel()

	state := map[string]any{"current_sub_agent": "after_approval"}

	in := models.Event{Actions: models.EventActions{AgentState: state, EndOfAgent: true}}
	got, err := models.ToSessionEvent(in)
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}
	if got.Actions.AgentState != nil || got.Actions.EndOfAgent {
		t.Errorf("ToSessionEvent kept a client-supplied checkpoint (AgentState=%v, EndOfAgent=%v); it must be dropped", got.Actions.AgentState, got.Actions.EndOfAgent)
	}

	stored := session.Event{Actions: session.EventActions{AgentState: state, EndOfAgent: true}}
	out := models.FromSessionEvent(stored)
	if diff := cmp.Diff(state, out.Actions.AgentState); diff != "" {
		t.Errorf("FromSessionEvent AgentState mismatch (-want +got):\n%s", diff)
	}
	if !out.Actions.EndOfAgent {
		t.Error("FromSessionEvent dropped EndOfAgent; reads should show it")
	}
}

// decodeKeepingNumbers decodes data into v with numbers as json.Number, so two
// numbers compare equal only when their digits do.
func decodeKeepingNumbers(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	return dec.Decode(v)
}

// TestRequestedAuthConfigsGoOutWithoutSecrets pins what a client sees of a
// pending credential request: the whole config except the secrets in it,
// wherever they sit.
func TestRequestedAuthConfigsGoOutWithoutSecrets(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string // a toolauth.Config as JSON
		want string // in without its secrets
	}{
		{
			name: "oauth2 client and flow state",
			in: `{
				"authScheme": {"type": "oauth2", "flows": {"authorizationCode": {
					"authorizationUrl": "https://idp.example/auth", "tokenUrl": "https://idp.example/token",
					"scopes": {"files": "Read files"}}}},
				"rawAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "clientSecret": "s3cret", "tokenEndpointAuthMethod": "client_secret_post"}},
				"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "authUri": "https://idp.example/auth?state=st", "state": "st",
					"authResponseUri": "https://app.example/cb?code=c0de", "authCode": "c0de",
					"accessToken": "at", "refreshToken": "rt", "idToken": "it", "codeVerifier": "cv",
					"expiresAt": 1700000000, "tokenEndpointAuthMethod": "client_secret_post"}},
				"credentialKey": "drive"
			}`,
			want: `{
				"authScheme": {"type": "oauth2", "flows": {"authorizationCode": {
					"authorizationUrl": "https://idp.example/auth", "tokenUrl": "https://idp.example/token",
					"scopes": {"files": "Read files"}}}},
				"rawAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "tokenEndpointAuthMethod": "client_secret_post"}},
				"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "authUri": "https://idp.example/auth?state=st", "state": "st",
					"expiresAt": 1700000000, "tokenEndpointAuthMethod": "client_secret_post"}},
				"credentialKey": "drive"
			}`,
		},
		{
			name: "http password, token and headers",
			in: `{
				"authScheme": {"type": "http", "scheme": "basic"},
				"rawAuthCredential": {"authType": "http", "http": {"scheme": "basic",
					"credentials": {"username": "ann", "password": "pw", "token": "tk"},
					"additionalHeaders": {"X-Api-Key": "hk"}}},
				"credentialKey": "basic"
			}`,
			want: `{
				"authScheme": {"type": "http", "scheme": "basic"},
				"rawAuthCredential": {"authType": "http", "http": {"scheme": "basic",
					"credentials": {"username": "ann"}}},
				"credentialKey": "basic"
			}`,
		},
		{
			name: "api key next to a service account key",
			in: `{
				"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
				"rawAuthCredential": {"authType": "apiKey", "apiKey": "ak"},
				"exchangedAuthCredential": {"authType": "serviceAccount", "serviceAccount": {
					"serviceAccountCredential": {"type": "service_account", "projectId": "p",
						"privateKeyId": "kid", "privateKey": "pk", "clientEmail": "sa@p.example",
						"clientId": "123", "authUri": "https://accounts.example/auth",
						"tokenUri": "https://accounts.example/token",
						"authProviderX509CertUrl": "https://certs.example/p",
						"clientX509CertUrl": "https://certs.example/c", "universeDomain": "example.com"},
					"scopes": ["cloud-platform"], "useDefaultCredential": false, "useIdToken": false}},
				"credentialKey": "mixed"
			}`,
			want: `{
				"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
				"rawAuthCredential": {"authType": "apiKey"},
				"exchangedAuthCredential": {"authType": "serviceAccount", "serviceAccount": {
					"serviceAccountCredential": {"type": "service_account", "projectId": "p",
						"clientEmail": "sa@p.example",
						"clientId": "123", "authUri": "https://accounts.example/auth",
						"tokenUri": "https://accounts.example/token",
						"authProviderX509CertUrl": "https://certs.example/p",
						"clientX509CertUrl": "https://certs.example/c", "universeDomain": "example.com"},
					"scopes": ["cloud-platform"], "useDefaultCredential": false, "useIdToken": false}},
				"credentialKey": "mixed"
			}`,
		},
		{
			// Unknown keys keep the spelling they arrived with, so each secret
			// name is dropped here only if the list holds that exact spelling.
			name: "every secret key name as an unknown key",
			in: `{
				"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
				"credentialKey": "extra",
				"keep": "me",
				"password": "s", "token": "s",
				"additionalHeaders": "s", "additional_headers": "s",
				"clientSecret": "s", "client_secret": "s",
				"authResponseUri": "s", "auth_response_uri": "s",
				"authCode": "s", "auth_code": "s",
				"accessToken": "s", "access_token": "s",
				"refreshToken": "s", "refresh_token": "s",
				"idToken": "s", "id_token": "s",
				"codeVerifier": "s", "code_verifier": "s",
				"privateKeyId": "s", "private_key_id": "s",
				"privateKey": "s", "private_key": "s",
				"apiKey": "s", "api_key": "s"
			}`,
			want: `{
				"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
				"credentialKey": "extra",
				"keep": "me"
			}`,
		},
		{
			name: "unknown key keeps its digits and loses nested secrets",
			in: `{
				"authScheme": {"type": "oauth2", "flows": {"clientCredentials": {
					"tokenUrl": "https://idp.example/token", "scopes": {}}}},
				"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "tokenEndpointAuthMethod": "client_secret_post",
					"provider": {"user": "ann", "id": 9007199254740993, "refresh_token": "rt",
						"sessions": [{"access_token": "at", "scope": "files"}]}}},
				"credentialKey": "cc"
			}`,
			want: `{
				"authScheme": {"type": "oauth2", "flows": {"clientCredentials": {
					"tokenUrl": "https://idp.example/token", "scopes": {}}}},
				"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "tokenEndpointAuthMethod": "client_secret_post",
					"provider": {"user": "ann", "id": 9007199254740993,
						"sessions": [{"scope": "files"}]}}},
				"credentialKey": "cc"
			}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var cfg toolauth.Config
			if err := json.Unmarshal([]byte(tc.in), &cfg); err != nil {
				t.Fatalf("json.Unmarshal(in) failed: %v", err)
			}
			ev := session.Event{Actions: session.EventActions{
				RequestedAuthConfigs: map[string]toolauth.Config{"call-1": cfg},
			}}

			b, err := json.Marshal(models.FromSessionEvent(ev))
			if err != nil {
				t.Fatalf("json.Marshal(event) failed: %v", err)
			}

			var got struct {
				Actions struct {
					RequestedAuthConfigs map[string]any `json:"requestedAuthConfigs"`
				} `json:"actions"`
			}
			if err := decodeKeepingNumbers(b, &got); err != nil {
				t.Fatalf("decodeKeepingNumbers(event) failed: %v", err)
			}
			var want any
			if err := decodeKeepingNumbers([]byte(tc.want), &want); err != nil {
				t.Fatalf("decodeKeepingNumbers(want) failed: %v", err)
			}
			if diff := cmp.Diff(want, got.Actions.RequestedAuthConfigs["call-1"]); diff != "" {
				t.Errorf("requestedAuthConfigs[call-1] mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRequestedAuthConfigsEncodeFailureFailsTheEvent pins that a config that
// cannot be encoded stops the event from being written, instead of the request
// silently going missing.
func TestRequestedAuthConfigsEncodeFailureFailsTheEvent(t *testing.T) {
	t.Parallel()

	ev := session.Event{Actions: session.EventActions{
		RequestedAuthConfigs: map[string]toolauth.Config{"call-1": {
			AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"},
			CredentialKey: "key",
			Extra:         map[string]any{"ratio": math.NaN()},
		}},
	}}

	if _, err := json.Marshal(models.FromSessionEvent(ev)); err == nil {
		t.Error("json.Marshal(event) succeeded, want an error for a config that cannot be encoded")
	}
}

// TestRequestedAuthConfigsAreNotTakenFromRequests pins that a client cannot
// plant a credential request through a create-session body.
func TestRequestedAuthConfigsAreNotTakenFromRequests(t *testing.T) {
	t.Parallel()

	body := `{"events": [{"author": "user", "actions": {"requestedAuthConfigs": {
		"call-1": {"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"}}}}}]}`
	var req models.CreateSessionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("json.Unmarshal(body) failed: %v", err)
	}
	if len(req.Events) != 1 || len(req.Events[0].Actions.RequestedAuthConfigs) != 1 {
		t.Fatalf("decoded %+v, want one event carrying one credential request", req.Events)
	}

	got, err := models.ToSessionEvent(req.Events[0])
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}
	if got.Actions.RequestedAuthConfigs != nil {
		t.Error("ToSessionEvent kept a client-supplied credential request; it must be dropped")
	}
}

// encodedParts returns the content parts of ev as a client receives them.
func encodedParts(t *testing.T, ev models.Event) []any {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("json.Marshal(event) failed: %v", err)
	}
	var got struct {
		Content struct {
			Parts []any `json:"parts"`
		} `json:"content"`
	}
	if err := decodeKeepingNumbers(b, &got); err != nil {
		t.Fatalf("decodeKeepingNumbers(event) failed: %v", err)
	}
	return got.Content.Parts
}

// TestRequestCredentialPartsGoOutWithoutSecrets pins that an
// adk_request_credential call, and the client's answer to it, reach the client
// without credential secrets such as the PKCE verifier and the code.
func TestRequestCredentialPartsGoOutWithoutSecrets(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		part *genai.Part
		want string // the parts as the client should see them
	}{
		{
			name: "request whose args hold a config struct",
			part: &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   "req-1",
				Name: toolauth.RequestCredentialFunctionCallName,
				Args: map[string]any{
					"functionCallId": "call-1",
					"authConfig": toolauth.Config{
						AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
							AuthorizationCode: &toolauth.OAuthFlow{
								AuthorizationURL: "https://idp.example/auth",
								TokenURL:         "https://idp.example/token",
								Scopes:           map[string]string{"files": "Read files"},
							},
						}},
						RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
							ClientID: "app", ClientSecret: "s3cret", TokenEndpointAuthMethod: "client_secret_post",
						}},
						ExchangedAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
							ClientID: "app", AuthURI: "https://idp.example/auth?state=st", State: "st",
							CodeVerifier: "cv", TokenEndpointAuthMethod: "client_secret_post",
						}},
						CredentialKey: "drive",
					},
				},
			}},
			want: `[{"functionCall": {"id": "req-1", "name": "adk_request_credential", "args": {
				"functionCallId": "call-1",
				"authConfig": {
					"authScheme": {"type": "oauth2", "flows": {"authorizationCode": {
						"authorizationUrl": "https://idp.example/auth", "tokenUrl": "https://idp.example/token",
						"scopes": {"files": "Read files"}}}},
					"rawAuthCredential": {"authType": "oauth2", "oauth2": {
						"clientId": "app", "tokenEndpointAuthMethod": "client_secret_post"}},
					"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
						"clientId": "app", "authUri": "https://idp.example/auth?state=st", "state": "st",
						"tokenEndpointAuthMethod": "client_secret_post"}},
					"credentialKey": "drive"}}}}]`,
		},
		{
			name: "request whose args are plain JSON",
			part: &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   "req-1",
				Name: toolauth.RequestCredentialFunctionCallName,
				Args: map[string]any{
					"functionCallId": "call-1",
					"authConfig": map[string]any{
						"authScheme":        map[string]any{"type": "apiKey", "in": "header", "name": "X-Key"},
						"rawAuthCredential": map[string]any{"authType": "apiKey", "apiKey": "ak"},
						"credentialKey":     "key",
					},
				},
			}},
			want: `[{"functionCall": {"id": "req-1", "name": "adk_request_credential", "args": {
				"functionCallId": "call-1",
				"authConfig": {
					"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
					"rawAuthCredential": {"authType": "apiKey"},
					"credentialKey": "key"}}}}]`,
		},
		{
			name: "answer with the redirect URL and the verifier",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{
				ID:   "req-1",
				Name: toolauth.RequestCredentialFunctionCallName,
				Response: map[string]any{
					"exchangedAuthCredential": map[string]any{"authType": "oauth2", "oauth2": map[string]any{
						"clientId":        "app",
						"state":           "st",
						"redirectUri":     "https://app.example/cb",
						"authResponseUri": "https://app.example/cb?state=st&code=c0de",
						"codeVerifier":    "cv",
					}},
					"credentialKey": "drive",
				},
			}},
			want: `[{"functionResponse": {"id": "req-1", "name": "adk_request_credential", "response": {
				"exchangedAuthCredential": {"authType": "oauth2", "oauth2": {
					"clientId": "app", "state": "st", "redirectUri": "https://app.example/cb"}},
				"credentialKey": "drive"}}}]`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ev := models.Event{Content: &genai.Content{Parts: []*genai.Part{tc.part}}}

			got := encodedParts(t, ev)

			var want []any
			if err := decodeKeepingNumbers([]byte(tc.want), &want); err != nil {
				t.Fatalf("decodeKeepingNumbers(want) failed: %v", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestOtherFunctionPartsKeepSecretLookingFields pins that only
// adk_request_credential parts are cleaned: a tool's own args and results
// reach the client as they are, even under a key such as password.
func TestOtherFunctionPartsKeepSecretLookingFields(t *testing.T) {
	t.Parallel()

	ev := models.Event{Content: &genai.Content{Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "login", Args: map[string]any{"user": "ann", "password": "pw"}}},
		{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "login", Response: map[string]any{"token": "tk"}}},
	}}}

	got := encodedParts(t, ev)

	var want []any
	if err := decodeKeepingNumbers([]byte(`[
		{"functionCall": {"id": "call-1", "name": "login", "args": {"user": "ann", "password": "pw"}}},
		{"functionResponse": {"id": "call-1", "name": "login", "response": {"token": "tk"}}}]`), &want); err != nil {
		t.Fatalf("decodeKeepingNumbers(want) failed: %v", err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("parts mismatch (-want +got):\n%s", diff)
	}
}

// TestRequestCredentialRedactionLeavesTheStoredEventAlone pins that encoding an
// event for a client does not strip the event the session holds. ADK reads the
// PKCE verifier and the code from it when the user's answer comes in.
func TestRequestCredentialRedactionLeavesTheStoredEventAlone(t *testing.T) {
	t.Parallel()

	args := func() map[string]any {
		return map[string]any{"functionCallId": "call-1", "authConfig": map[string]any{
			"exchangedAuthCredential": map[string]any{"authType": "oauth2", "oauth2": map[string]any{
				"clientId": "app", "codeVerifier": "cv",
			}},
		}}
	}
	response := func() map[string]any {
		return map[string]any{"exchangedAuthCredential": map[string]any{"authType": "oauth2", "oauth2": map[string]any{
			"clientId": "app", "authResponseUri": "https://app.example/cb?code=c0de",
		}}}
	}
	content := &genai.Content{Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "req-1", Name: toolauth.RequestCredentialFunctionCallName, Args: args()}},
		{FunctionResponse: &genai.FunctionResponse{ID: "req-1", Name: toolauth.RequestCredentialFunctionCallName, Response: response()}},
	}}

	if _, err := json.Marshal(models.Event{Content: content}); err != nil {
		t.Fatalf("json.Marshal(event) failed: %v", err)
	}

	if diff := cmp.Diff(args(), content.Parts[0].FunctionCall.Args); diff != "" {
		t.Errorf("stored call args changed (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(response(), content.Parts[1].FunctionResponse.Response); diff != "" {
		t.Errorf("stored answer changed (-want +got):\n%s", diff)
	}
}

// TestStateDeltaCredentialsGoOutWithoutSecrets pins that a credential kept in
// state, such as the tokens a tool's sign-in produced, reaches the client
// without its secrets, while ordinary state keeps every key.
func TestStateDeltaCredentialsGoOutWithoutSecrets(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		value any
		want  string // the value as the client should see it
	}{
		{
			name: "credential struct, as a token exchange stores it",
			value: toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
				ClientID: "app", AccessToken: "at", RefreshToken: "rt", ExpiresAt: 1700000000,
				TokenEndpointAuthMethod: "client_secret_post",
			}},
			want: `{"authType": "oauth2", "oauth2": {
				"clientId": "app", "expiresAt": 1700000000, "tokenEndpointAuthMethod": "client_secret_post"}}`,
		},
		{
			name: "credential as JSON, nested in another value",
			value: map[string]any{"profile": map[string]any{"name": "ann", "cred": map[string]any{
				"auth_type": "OAUTH2", "oauth2": map[string]any{"access_token": "at", "scope": "files"},
			}}},
			want: `{"profile": {"name": "ann", "cred": {"auth_type": "OAUTH2", "oauth2": {"scope": "files"}}}}`,
		},
		{
			name: "auth config",
			value: map[string]any{
				"authScheme":              map[string]any{"type": "apiKey", "in": "header", "name": "X-Key"},
				"rawAuthCredential":       map[string]any{"apiKey": "ak"},
				"exchangedAuthCredential": map[string]any{"oauth2": map[string]any{"clientId": "app", "authCode": "c0de"}},
			},
			want: `{
				"authScheme": {"type": "apiKey", "in": "header", "name": "X-Key"},
				"rawAuthCredential": {},
				"exchangedAuthCredential": {"oauth2": {"clientId": "app"}}}`,
		},
		{
			name:  "ordinary value with keys that look secret",
			value: map[string]any{"password": "pw", "token": "tk", "authType": "not a credential type"},
			want:  `{"password": "pw", "token": "tk", "authType": "not a credential type"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ev := session.Event{Actions: session.EventActions{StateDelta: map[string]any{"temp:cred": tc.value}}}

			b, err := json.Marshal(models.FromSessionEvent(ev))
			if err != nil {
				t.Fatalf("json.Marshal(event) failed: %v", err)
			}

			var got struct {
				Actions struct {
					StateDelta map[string]any `json:"stateDelta"`
				} `json:"actions"`
			}
			if err := decodeKeepingNumbers(b, &got); err != nil {
				t.Fatalf("decodeKeepingNumbers(event) failed: %v", err)
			}
			var want any
			if err := decodeKeepingNumbers([]byte(tc.want), &want); err != nil {
				t.Fatalf("decodeKeepingNumbers(want) failed: %v", err)
			}
			if diff := cmp.Diff(want, got.Actions.StateDelta["temp:cred"]); diff != "" {
				t.Errorf("stateDelta[temp:cred] mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestToSessionEventTimestampBoundsMatchSessionDecoder pins the timestamps
// ToSessionEvent accepts to the ones the session.Event JSON decoder accepts, so
// the two bounds cannot drift apart. Zero is left out: the REST model reads it
// as "no timestamp", while the session decoder reads it as the epoch.
func TestToSessionEventTimestampBoundsMatchSessionDecoder(t *testing.T) {
	for _, secs := range []float64{
		0.5,
		-0.5,
		1,
		1733040000.123456,
		1733040000123, // epoch milliseconds, as JavaScript's Date.now() returns
		1e13,          // overflows the int64 conversion without a bound
		-1e13,
		1e300,
		-1e300,
		-62135596800,        // first second of year 1
		-62135596800.000001, // just before it
		253402300799,        // last second of year 9999
		253402300799.999999, // rounds up into year 10000
		253402300800,        // first second of year 10000
	} {
		_, restErr := models.ToSessionEvent(models.Event{Timestamp: secs})
		var decoded session.Event
		raw, err := json.Marshal(map[string]any{"timestamp": secs})
		if err != nil {
			t.Fatalf("json.Marshal(%v) failed: %v", secs, err)
		}
		decodeErr := json.Unmarshal(raw, &decoded)
		if (restErr == nil) != (decodeErr == nil) {
			t.Errorf("timestamp %v: ToSessionEvent error = %v, session.Event decoder error = %v; want both nil or both non-nil", secs, restErr, decodeErr)
			continue
		}
		if restErr != nil {
			continue
		}
		got, _ := models.ToSessionEvent(models.Event{Timestamp: secs})
		if !got.Timestamp.Equal(decoded.Timestamp) {
			t.Errorf("timestamp %v: ToSessionEvent = %v, session.Event decoder = %v", secs, got.Timestamp, decoded.Timestamp)
		}
	}
}

// TestMarshalEventDropsNilParts: a nil entry in Content.Parts (a null part in
// a stored create-session body reaches storage verbatim) must serialize, not
// panic every endpoint that returns the session — and the serialized output
// must not contain a null part: REST clients, including the bundled web UI,
// read fields off every part they render. This pins the exact output shape:
// the nil slot is dropped, the real parts keep their order and content.
func TestMarshalEventDropsNilParts(t *testing.T) {
	ev := session.Event{
		ID: "e1",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{
				genai.NewPartFromText("kept"),
				nil,
				{FunctionCall: &genai.FunctionCall{Name: "f"}},
			}},
		},
	}
	rawJSON, err := json.Marshal(models.FromSessionEvent(ev))
	if err != nil {
		t.Fatalf("json.Marshal(event with nil part) failed: %v", err)
	}
	var got struct {
		Content struct {
			Parts []json.RawMessage `json:"parts"`
		} `json:"content"`
	}
	if err := json.Unmarshal(rawJSON, &got); err != nil {
		t.Fatalf("unmarshal marshaled event: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("parts count = %d, want 2 (nil slot dropped): %s", len(got.Content.Parts), rawJSON)
	}
	for i, p := range got.Content.Parts {
		if string(p) == "null" {
			t.Errorf("parts[%d] is null; output must never carry a null part: %s", i, rawJSON)
		}
	}
	if !bytes.Contains(got.Content.Parts[0], []byte(`"kept"`)) {
		t.Errorf("parts[0] = %s, want the kept text part first", got.Content.Parts[0])
	}
	if !bytes.Contains(got.Content.Parts[1], []byte(`"name":"f"`)) || !bytes.Contains(got.Content.Parts[1], []byte(`"args":{}`)) {
		t.Errorf("parts[1] = %s, want the function call with empty args object", got.Content.Parts[1])
	}
}
