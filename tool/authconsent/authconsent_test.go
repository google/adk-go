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

package authconsent_test

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/tool/authconsent"
)

// TestOAuth2ConsentMatchesClientContract pins the exact JSON an ADK client
// receives. The path below is not ours to choose: the ADK web client gates its
// whole consent popup on args.authConfig.exchangedAuthCredential.oauth2 being
// present and reads the URL from .authUri, so renaming any field in it silently
// stops the popup from ever opening and the run hangs on a long-running call
// that never gets a response.
//
// Asserting on the decoded map rather than on a Go struct is deliberate: a test
// that round-trips through authconsent's own types passes for any pair of
// matching field tags, including a wrong pair.
func TestOAuth2ConsentMatchesClientContract(t *testing.T) {
	b, err := json.Marshal(authconsent.OAuth2Consent("https://consent.example/auth?state=s", "n0nce", "cred-key"))
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	want := map[string]any{
		"authScheme": map[string]any{"type": "oauth2"},
		"exchangedAuthCredential": map[string]any{
			"authType": "oauth2",
			"oauth2": map[string]any{
				"authUri": "https://consent.example/auth?state=s",
				"nonce":   "n0nce",
			},
		},
		"credentialKey": "cred-key",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("OAuth2Consent() JSON diff (-want +got):\n%s", diff)
	}
}

// TestDecodeClientReply decodes the payload the ADK web client sends back: the
// authConfig it was given, cloned, with the OAuth2 response fields filled in. It
// carries fields Go never sets, and rawAuthCredential, which Go does not model
// at all, so this also pins that an unmodelled field is ignored rather than
// failing the decode.
func TestDecodeClientReply(t *testing.T) {
	const reply = `{
	  "exchangedAuthCredential": {
	    "authType": "oauth2",
	    "oauth2": {
	      "authUri": "https://consent.example/auth?state=s",
	      "nonce": "n0nce",
	      "redirectUri": "http://localhost:4200/dev-ui/",
	      "authResponseUri": "http://localhost:4200/dev-ui/?code=abc&state=s"
	    }
	  },
	  "credentialKey": "cred-key",
	  "authScheme": {"type": "oauth2"},
	  "rawAuthCredential": {"authType": "oauth2"}
	}`

	var got authconsent.AuthConfig
	if err := json.Unmarshal([]byte(reply), &got); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	want := authconsent.AuthConfig{
		AuthScheme: &authconsent.AuthScheme{Type: authconsent.AuthTypeOAuth2},
		ExchangedAuthCredential: &authconsent.AuthCredential{
			AuthType: authconsent.AuthTypeOAuth2,
			OAuth2: &authconsent.OAuth2Auth{
				AuthURI:         "https://consent.example/auth?state=s",
				Nonce:           "n0nce",
				RedirectURI:     "http://localhost:4200/dev-ui/",
				AuthResponseURI: "http://localhost:4200/dev-ui/?code=abc&state=s",
			},
		},
		CredentialKey: "cred-key",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("decoded client reply diff (-want +got):\n%s", diff)
	}
}

// TestZeroAuthConfigOmitsEverything pins that an empty config serializes to an
// empty object. The client's presence checks are what decide whether it prompts,
// so an empty config must not look like a consent request.
func TestZeroAuthConfigOmitsEverything(t *testing.T) {
	b, err := json.Marshal(authconsent.AuthConfig{})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if got := string(b); got != "{}" {
		t.Errorf("Marshal(zero AuthConfig) = %s, want {}", got)
	}
}

// TestFunctionCallNameIsThePythonName pins the wire name. It is the only thing
// a client matches on to recognise a consent request, and adk-python's
// REQUEST_EUC_FUNCTION_CALL_NAME fixes its value.
func TestFunctionCallNameIsThePythonName(t *testing.T) {
	if authconsent.FunctionCallName != "adk_request_credential" {
		t.Errorf("FunctionCallName = %q, want %q", authconsent.FunctionCallName, "adk_request_credential")
	}
	if authconsent.FunctionCallIDArg != "functionCallId" {
		t.Errorf("FunctionCallIDArg = %q, want %q", authconsent.FunctionCallIDArg, "functionCallId")
	}
	if authconsent.AuthConfigArg != "authConfig" {
		t.Errorf("AuthConfigArg = %q, want %q", authconsent.AuthConfigArg, "authConfig")
	}
}
