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
// a client matches on to recognize a consent request, and adk-python's
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

// TestWireArgsAlwaysCarriesAuthScheme pins the one field adk-python's AuthConfig
// requires. A tool that builds its config by hand rather than through
// OAuth2Consent must still produce an event an adk-python reader accepts, so
// WireArgs fills the scheme type from the credential type, defaulting to oauth2.
func TestWireArgsAlwaysCarriesAuthScheme(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  authconsent.AuthConfig
		want string
	}{
		{name: "no scheme", cfg: authconsent.AuthConfig{ExchangedAuthCredential: &authconsent.AuthCredential{
			AuthType: authconsent.AuthTypeOAuth2, OAuth2: &authconsent.OAuth2Auth{AuthURI: "https://c/a"}}}, want: "oauth2"},
		{name: "empty scheme type", cfg: authconsent.AuthConfig{AuthScheme: &authconsent.AuthScheme{}, ExchangedAuthCredential: &authconsent.AuthCredential{
			AuthType: authconsent.AuthTypeOAuth2}}, want: "oauth2"},
		{name: "scheme given", cfg: authconsent.OAuth2Consent("https://c/a", "n", "k"), want: "oauth2"},
		{name: "credential without type", cfg: authconsent.AuthConfig{ExchangedAuthCredential: &authconsent.AuthCredential{
			OAuth2: &authconsent.OAuth2Auth{AuthURI: "https://c/a"}}}, want: "oauth2"},
		{name: "zero config", cfg: authconsent.AuthConfig{}, want: "oauth2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args, err := authconsent.WireArgs("call-1", tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			ac, _ := args[authconsent.AuthConfigArg].(map[string]any)
			scheme, _ := ac["authScheme"].(map[string]any)
			if got, _ := scheme["type"].(string); got != tc.want {
				t.Errorf("authConfig.authScheme.type = %q, want %q (authConfig = %v)", got, tc.want, ac)
			}
		})
	}
}

// TestAuthConfigCloneSharesNoPointers pins the deep copy a store relies on.
// AuthConfig carries three pointers; a value copy shares all of them, so a
// caller holding the original could rewrite a stored consent URL or nonce.
func TestAuthConfigCloneSharesNoPointers(t *testing.T) {
	orig := authconsent.OAuth2Consent("https://consent.example/auth", "n0nce", "key")
	clone := orig.Clone()
	orig.AuthScheme.Type = "changed"
	orig.ExchangedAuthCredential.AuthType = "changed"
	orig.ExchangedAuthCredential.OAuth2.AuthURI = "https://attacker.example/auth"
	orig.ExchangedAuthCredential.OAuth2.Nonce = "changed"
	if clone.AuthScheme.Type != authconsent.AuthTypeOAuth2 ||
		clone.ExchangedAuthCredential.AuthType != authconsent.AuthTypeOAuth2 ||
		clone.ExchangedAuthCredential.OAuth2.AuthURI != "https://consent.example/auth" ||
		clone.ExchangedAuthCredential.OAuth2.Nonce != "n0nce" {
		t.Errorf("Clone() shares state with the original: %+v / %+v", clone.AuthScheme, clone.ExchangedAuthCredential.OAuth2)
	}
	var zero authconsent.AuthConfig
	if got := zero.Clone(); got.AuthScheme != nil || got.ExchangedAuthCredential != nil {
		t.Errorf("Clone() of a zero config = %+v, want zero", got)
	}
}
