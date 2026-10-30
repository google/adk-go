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

package agentanalytics

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestConsentRoundTripIsRedacted pins that the interactive OAuth consent
// round-trip never reaches a row. The framework's adk_request_credential call
// carries the consent URL (which names the acting user) and a nonce, and the
// client's reply carries the callback URL with the authorization code. The
// key-based redaction does not know those names, so the round-trip is
// recognized by its function name instead, and only the id of the paused call
// is kept.
func TestConsentRoundTripIsRedacted(t *testing.T) {
	p, requests, mCtx := newTestPluginWithConfig(t, nil)

	args := map[string]any{
		"functionCallId": "paused-call",
		"authConfig": map[string]any{
			"authScheme":    map[string]any{"type": "oauth2"},
			"credentialKey": "cred-key",
			"exchangedAuthCredential": map[string]any{
				"authType": "oauth2",
				"oauth2": map[string]any{
					"authUri": "https://consent.example/auth?login_hint=someone%40example.com&state=st",
					"nonce":   "s3cr3t-nonce",
				},
			},
		},
	}
	consentCall := &session.Event{Author: "agent", InvocationID: "inv", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "cc-1", Name: hitlCredentialFunctionCallName, Args: args},
	}}}}}
	if _, err := p.OnEventCallback()(mCtx, consentCall); err != nil {
		t.Fatalf("onEvent: %v", err)
	}
	reply := &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: "cc-1", Name: hitlCredentialFunctionCallName,
		Response: map[string]any{"exchangedAuthCredential": map[string]any{"oauth2": map[string]any{"authResponseUri": "http://localhost/cb?code=AUTHCODE123&state=st"}}},
	}}}}
	if _, err := p.OnUserMessageCallback()(mCtx, reply); err != nil {
		t.Fatalf("onUserMessage: %v", err)
	}
	// The model response that carried the consent call is logged as content
	// parts too, through the formatter rather than the HITL rows.
	if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}); err != nil {
		t.Fatalf("beforeModel: %v", err)
	}
	if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: consentCall.Content}, nil); err != nil {
		t.Fatalf("afterModel: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 4 })
	if n := len(rowsOfType(rows, EventLLMResponse)); n != 1 {
		t.Fatalf("got %d LLM_RESPONSE rows, want 1 (types: %v)", n, eventTypes(rows))
	}
	dump, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, leaked := range []string{"consent.example", "login_hint", "s3cr3t-nonce", "AUTHCODE123", "cred-key"} {
		if strings.Contains(string(dump), leaked) {
			t.Errorf("rows contain %q", leaked)
		}
	}
	for _, kept := range []string{hitlCredentialFunctionCallName, "paused-call"} {
		if !strings.Contains(string(dump), kept) {
			t.Errorf("rows lost %q: the round-trip should still be visible by name and paused call id", kept)
		}
	}
}
