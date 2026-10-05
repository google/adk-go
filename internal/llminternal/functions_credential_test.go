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
	"maps"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func credentialTestContext(t *testing.T) agent.InvocationContext {
	t.Helper()
	return &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1", branch: "b"}
}

func callEvent(calls ...*genai.FunctionCall) *session.Event {
	parts := make([]*genai.Part, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, &genai.Part{FunctionCall: c})
	}
	return &session.Event{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: parts}}}
}

func responseEvent(requested map[string]authconsent.AuthConfig) *session.Event {
	return &session.Event{Actions: session.EventActions{RequestedCredentials: requested}}
}

// TestGenerateRequestCredentialEventArgs pins the args of the emitted call
// against what an ADK client reads. The two keys and the nested authConfig
// object are adk-python's AuthToolArguments, and a client matches on them
// literally.
func TestGenerateRequestCredentialEventArgs(t *testing.T) {
	ctx := credentialTestContext(t)
	fc := &genai.FunctionCall{ID: "call-1", Name: "tool_a"}
	cfg := authconsent.OAuth2Consent("https://consent.example/auth", "n", "key-1")

	ev := generateRequestCredentialEvent(ctx, callEvent(fc), responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg}))
	if ev == nil {
		t.Fatal("generateRequestCredentialEvent() = nil, want an event")
	}
	if len(ev.Content.Parts) != 1 {
		t.Fatalf("emitted %d parts, want 1", len(ev.Content.Parts))
	}
	emitted := ev.Content.Parts[0].FunctionCall
	if emitted.Name != authconsent.FunctionCallName {
		t.Errorf("emitted call name = %q, want %q", emitted.Name, authconsent.FunctionCallName)
	}
	if got := emitted.Args["functionCallId"]; got != "call-1" {
		t.Errorf("args[functionCallId] = %v, want %q", got, "call-1")
	}

	// Compare the marshalled form, not the Go value: the client reads JSON, and
	// a struct comparison here would pass for any field tags that agree with
	// themselves.
	b, err := json.Marshal(emitted.Args["authConfig"])
	if err != nil {
		t.Fatalf("Marshal(authConfig) error = %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal(authConfig) error = %v", err)
	}
	want := map[string]any{
		"authScheme": map[string]any{"type": "oauth2"},
		"exchangedAuthCredential": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"authUri": "https://consent.example/auth", "nonce": "n"},
		},
		"credentialKey": "key-1",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("args[authConfig] diff (-want +got):\n%s", diff)
	}

	if !slices.Contains(ev.LongRunningToolIDs, emitted.ID) {
		t.Errorf("LongRunningToolIDs = %v, want it to contain the emitted call id %q. "+
			"Without it the client never treats the call as pending and the run does not pause.",
			ev.LongRunningToolIDs, emitted.ID)
	}
}

// TestGenerateRequestCredentialEventOrder pins that emission follows the order
// of the model's function calls rather than Go's randomized map iteration. A
// single run would pass by luck, so this asserts over repeated builds of the
// same input.
func TestGenerateRequestCredentialEventOrder(t *testing.T) {
	ctx := credentialTestContext(t)
	calls := []*genai.FunctionCall{
		{ID: "c1", Name: "a"},
		{ID: "c2", Name: "b"},
		{ID: "c3", Name: "c"},
		{ID: "c4", Name: "d"},
		{ID: "c5", Name: "e"},
		{ID: "c6", Name: "f"},
	}
	requested := map[string]authconsent.AuthConfig{}
	for _, c := range calls {
		requested[c.ID] = authconsent.OAuth2Consent("https://consent.example/"+c.ID, "", "")
	}

	want := []string{"c1", "c2", "c3", "c4", "c5", "c6"}
	for i := range 50 {
		ev := generateRequestCredentialEvent(ctx, callEvent(calls...), responseEvent(requested))
		if ev == nil {
			t.Fatalf("iteration %d: got nil event", i)
		}
		var got []string
		for _, p := range ev.Content.Parts {
			got = append(got, p.FunctionCall.Args["functionCallId"].(string))
		}
		if !slices.Equal(got, want) {
			t.Fatalf("iteration %d: emitted order = %v, want %v", i, got, want)
		}
	}
}

func TestGenerateRequestCredentialEventEmpty(t *testing.T) {
	ctx := credentialTestContext(t)
	fc := &genai.FunctionCall{ID: "call-1", Name: "tool_a"}
	cfg := authconsent.OAuth2Consent("https://consent.example/auth", "", "")

	tests := []struct {
		name         string
		callEvent    *session.Event
		responseEven *session.Event
	}{
		{"nil response event", callEvent(fc), nil},
		{"no requested credentials", callEvent(fc), responseEvent(nil)},
		{"nil call event", nil, responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg})},
		{"call event without content", &session.Event{}, responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg})},
		{"no matching call id", callEvent(&genai.FunctionCall{ID: "other"}), responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg})},
		{"part carries no function call", &session.Event{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "hi"}}}}}, responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if ev := generateRequestCredentialEvent(ctx, tt.callEvent, tt.responseEven); ev != nil {
				t.Errorf("generateRequestCredentialEvent() = %+v, want nil", ev)
			}
		})
	}
}

// TestMergeEventActionsKeepsRequestedCredentials pins that a consent request
// raised by any tool call in a parallel batch reaches the merged event.
//
// Tool calls in one turn run concurrently, each writing its own EventActions,
// and mergeParallelFunctionResponseEvents folds them into the first one. A
// field missing from mergeEventActions is dropped for every call but the first,
// so consent would work or not depending on the order the model emitted its
// calls. The confirmation twin is asserted alongside as the control.
func TestMergeEventActionsKeepsRequestedCredentials(t *testing.T) {
	first := &session.EventActions{StateDelta: map[string]any{}}
	second := &session.EventActions{
		StateDelta:                 map[string]any{},
		RequestedCredentials:       map[string]authconsent.AuthConfig{"call-2": authconsent.OAuth2Consent("https://consent.example/auth", "n", "key-2")},
		RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{"call-2": {Hint: "approve?"}},
	}

	got := mergeEventActions(first, second)

	if _, ok := got.RequestedToolConfirmations["call-2"]; !ok {
		t.Errorf("RequestedToolConfirmations = %v, want call-2 (control)", got.RequestedToolConfirmations)
	}
	cfg, ok := got.RequestedCredentials["call-2"]
	if !ok {
		t.Fatalf("RequestedCredentials = %v, want call-2. A consent raised by any call but the "+
			"first is dropped here, and the run then halts with no consent request emitted.",
			got.RequestedCredentials)
	}
	if cfg.CredentialKey != "key-2" {
		t.Errorf("merged CredentialKey = %q, want %q", cfg.CredentialKey, "key-2")
	}
}

// TestGenerateRequestCredentialEventForSecondParallelCall is the end of the same
// path: the merged event must still produce a consent call when the requesting
// tool was not the first one the model called.
func TestGenerateRequestCredentialEventForSecondParallelCall(t *testing.T) {
	ctx := credentialTestContext(t)
	first := &genai.FunctionCall{ID: "call-1", Name: "tool_a"}
	second := &genai.FunctionCall{ID: "call-2", Name: "tool_b"}

	// What the flow builds: one EventActions per concurrent call, folded into
	// the first.
	merged := mergeEventActions(
		&session.EventActions{StateDelta: map[string]any{}},
		&session.EventActions{
			StateDelta:           map[string]any{},
			RequestedCredentials: map[string]authconsent.AuthConfig{"call-2": authconsent.OAuth2Consent("https://consent.example/auth", "n", "key-2")},
		},
	)

	ev := generateRequestCredentialEvent(ctx, callEvent(first, second), &session.Event{Actions: *merged})
	if ev == nil {
		t.Fatal("generateRequestCredentialEvent() = nil, want a consent call for the second parallel tool call")
	}
	if got := ev.Content.Parts[0].FunctionCall.Args["functionCallId"]; got != "call-2" {
		t.Errorf("args[functionCallId] = %v, want %q", got, "call-2")
	}
}

// TestRequestCredentialArgsSurviveTheAgentEngineEncoder pins that the emitted
// args reach a client with the field names it reads.
//
// The Agent Engine event encoder rewrites Go struct field names to snake_case
// and leaves map keys alone, so an AuthConfig struct placed straight into
// FunctionCall.Args is delivered as exchanged_auth_credential.oauth2.auth_uri
// over that transport, which no ADK client looks at. Nothing else in the suite
// can see this: the encoder is in another package with no fixture carrying a
// struct inside Args.
func TestRequestCredentialArgsSurviveTheAgentEngineEncoder(t *testing.T) {
	ctx := credentialTestContext(t)
	fc := &genai.FunctionCall{ID: "call-1", Name: "tool_a"}
	cfg := authconsent.OAuth2Consent("https://consent.example/auth", "n", "key-1")

	ev := generateRequestCredentialEvent(ctx, callEvent(fc), responseEvent(map[string]authconsent.AuthConfig{"call-1": cfg}))
	if ev == nil {
		t.Fatal("generateRequestCredentialEvent() = nil")
	}
	authConfig := ev.Content.Parts[0].FunctionCall.Args["authConfig"]

	if _, isStruct := authConfig.(authconsent.AuthConfig); isStruct {
		t.Fatal("args[authConfig] is an authconsent.AuthConfig struct. The Agent Engine encoder " +
			"snake-cases struct field names, so the client receives exchanged_auth_credential " +
			"instead of exchangedAuthCredential and never opens the consent prompt.")
	}
	m, ok := authConfig.(map[string]any)
	if !ok {
		t.Fatalf("args[authConfig] is %T, want map[string]any", authConfig)
	}
	if _, ok := m["exchangedAuthCredential"]; !ok {
		t.Errorf("args[authConfig] keys = %v, want exchangedAuthCredential", slices.Sorted(maps.Keys(m)))
	}
}
