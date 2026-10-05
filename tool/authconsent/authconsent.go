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

// Package authconsent defines the wire contract for ADK's interactive
// (3-legged) OAuth consent round-trip: the adk_request_credential function call
// a tool emits when it needs the end user to grant consent, and the response the
// client returns once consent is complete.
//
// It is the credential analog of [google.golang.org/adk/v2/tool/toolconfirmation]
// and, like it, is a dependency-light leaf so that session, agent, and the LLM
// flow can reference the wire types without importing the heavier auth package
// (which would form an import cycle).
//
// The JSON encoding here is not ours to choose: it mirrors adk-python's
// AuthToolArguments/AuthConfig so that an ADK client written against any
// implementation drives a Go agent's consent flow unchanged.
package authconsent

import (
	"encoding/json"
	"fmt"
)

// FunctionCallName is the name of the function call ADK emits to ask the client
// to drive the end user through interactive (3-legged) OAuth consent. It is the
// credential analog of toolconfirmation.FunctionCallName and matches
// adk-python's REQUEST_EUC_FUNCTION_CALL_NAME.
//
// A client interacting with an ADK agent must:
//  1. Watch for a long-running function call with this name.
//  2. Read the consent URL from
//     args.authConfig.exchangedAuthCredential.oauth2.authUri and send the user
//     there.
//  3. Reply with a FunctionResponse carrying the same id and name, whose
//     payload is the [AuthConfig] it received with the OAuth2 response fields
//     filled in.
//
// ADK then resumes the original tool call. This holds on the standard run loop;
// a live (bidi) session does not emit the call.
const FunctionCallName = "adk_request_credential"

// Argument keys of the adk_request_credential function call. They are the
// camelCase aliases adk-python emits (AuthToolArguments serialized with
// by_alias), and are what existing ADK clients read.
const (
	// FunctionCallIDArg names the id of the tool call that is paused for consent.
	FunctionCallIDArg = "functionCallId"
	// AuthConfigArg names the [AuthConfig] payload.
	AuthConfigArg = "authConfig"
)

// AuthTypeOAuth2 is the only credential type ADK Go raises consent for. It is
// adk-python's AuthCredentialTypes.OAUTH2.
const AuthTypeOAuth2 = "oauth2"

// AuthConfig is the credential configuration exchanged with the client during
// consent. ADK sends one inside the adk_request_credential call and the client
// returns it, amended, in the matching function response.
//
// It is a deliberate subset of adk-python's AuthConfig: rawAuthCredential is
// omitted because no ADK Go provider produces one, and authScheme carries only
// the type. Unknown fields a client sends back are ignored rather than
// rejected, so a fuller client payload still decodes.
type AuthConfig struct {
	// AuthScheme names how the target authenticates. ADK Go carries only the
	// scheme type, but the field is not optional: adk-python's AuthConfig
	// requires auth_scheme, so omitting it makes an adk-python reader reject the
	// whole event rather than just the field — including on the Vertex session
	// store, which both runtimes read from.
	AuthScheme *AuthScheme `json:"authScheme,omitempty"`
	// ExchangedAuthCredential carries the consent URL on the way out and the
	// client's authorization response on the way back.
	ExchangedAuthCredential *AuthCredential `json:"exchangedAuthCredential,omitempty"`
	// CredentialKey identifies the credential to resume under. It is
	// auth.ConsentRequiredError.Key, passed through untouched.
	CredentialKey string `json:"credentialKey,omitempty"`
}

// AuthScheme is the scheme half of an [AuthConfig]. adk-python models a full
// security scheme here; ADK Go sends only the discriminator, which adk-python
// accepts as a CustomAuthScheme.
type AuthScheme struct {
	// Type is the scheme type, [AuthTypeOAuth2] for a consent round-trip.
	Type string `json:"type,omitempty"`
}

// AuthCredential is the credential half of an [AuthConfig].
type AuthCredential struct {
	// AuthType is the credential kind. ADK Go only raises consent for
	// [AuthTypeOAuth2].
	AuthType string `json:"authType,omitempty"`
	// OAuth2 holds the OAuth2 consent details.
	OAuth2 *OAuth2Auth `json:"oauth2,omitempty"`
}

// OAuth2Auth holds the OAuth2 fields of a consent round-trip. ADK fills AuthURI
// and Nonce; the client fills RedirectURI and AuthResponseURI once the user has
// authorized.
type OAuth2Auth struct {
	// AuthURI is the URL the end user must visit to grant consent.
	//
	// Hand it to that user and to nobody else: a 3-legged authorization URI
	// normally carries the acting user in a login_hint parameter and binds the
	// request through State and Nonce. Do not log it and do not put it in a span.
	// It does reach the session store, because the client needs it, so treat a
	// session containing one as carrying user data.
	AuthURI string `json:"authUri,omitempty"`
	// Nonce is an opaque value echoed back to correlate the consent response.
	// Sensitive on the same terms as AuthURI, which embeds it.
	Nonce string `json:"nonce,omitempty"`
	// State binds the authorization request to this session. Sensitive on the
	// same terms as AuthURI. ADK Go does not set it today; a client that echoes
	// one back is preserved here.
	State string `json:"state,omitempty"`
	// RedirectURI is where the client asked the provider to return the user.
	// Set by the client.
	RedirectURI string `json:"redirectUri,omitempty"`
	// AuthResponseURI is the full callback URL the provider redirected to, query
	// string included. Set by the client.
	AuthResponseURI string `json:"authResponseUri,omitempty"`
}

// OAuth2Consent builds the [AuthConfig] a tool passes to
// agent.Context.RequestCredential for an interactive OAuth2 consent round-trip.
// Its arguments are the fields of auth.ConsentRequiredError.
func OAuth2Consent(authURI, nonce, credentialKey string) AuthConfig {
	return AuthConfig{
		AuthScheme: &AuthScheme{Type: AuthTypeOAuth2},
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AuthURI: authURI,
				Nonce:   nonce,
			},
		},
		CredentialKey: credentialKey,
	}
}

// WireArgs returns the arguments of the adk_request_credential function call
// that asks a client to collect consent for the tool call with id
// functionCallID.
//
// The authConfig value is a decoded map rather than the [AuthConfig] struct,
// and that is load-bearing rather than tidiness. The Agent Engine event encoder
// rewrites Go struct field names to snake_case while leaving map keys alone
// (server/agentengine/internal/helper/encode.go), so a struct here reaches the
// client as exchanged_auth_credential.oauth2.auth_uri, which no ADK client
// looks for.
func WireArgs(functionCallID string, cfg AuthConfig) (map[string]any, error) {
	b, err := json.Marshal(cfg)
	if err != nil {
		return nil, fmt.Errorf("authconsent: marshal auth config: %w", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		return nil, fmt.Errorf("authconsent: decode auth config: %w", err)
	}
	return map[string]any{
		FunctionCallIDArg: functionCallID,
		AuthConfigArg:     decoded,
	}, nil
}
