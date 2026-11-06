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

package authhandler

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/adk/v2/tool/toolauth"
)

// parseAuthResponse reads the credential out of the client's answer to an
// adk_request_credential call: the config ADK sent with ExchangedAuthCredential
// filled in, or a bare credential, as adk-python's store_auth_response accepts.
func parseAuthResponse(resp map[string]any) (*toolauth.Credential, error) {
	b, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("authhandler: encode auth response: %w", err)
	}
	var cfg toolauth.Config
	// A bare credential has no authScheme, so it fails here and is read below.
	if err := json.Unmarshal(b, &cfg); err == nil {
		if cfg.ExchangedAuthCredential == nil {
			return nil, errors.New("authhandler: auth response carries no exchanged credential")
		}
		return cfg.ExchangedAuthCredential, nil
	}
	var cred toolauth.Credential
	if err := json.Unmarshal(b, &cred); err != nil {
		return nil, fmt.Errorf("authhandler: auth response is neither an auth config nor a credential: %w", err)
	}
	return &cred, nil
}

// pinAuthResponse returns requested, the config ADK sent, with answer as its
// exchanged credential. OAuth2 fields the answer leaves empty are filled from
// the request, so a PKCE verifier the client did not echo back comes back.
// The result shares requested's other fields; nothing here writes to them.
func pinAuthResponse(requested *toolauth.Config, answer *toolauth.Credential) *toolauth.Config {
	out := *requested
	merged := *answer
	source := cmp.Or(requested.ExchangedAuthCredential, requested.RawAuthCredential)
	if source != nil && source.OAuth2 != nil {
		merged.OAuth2 = mergeOAuth2(answer.OAuth2, source.OAuth2)
	}
	out.ExchangedAuthCredential = &merged
	return &out
}

// mergeOAuth2 returns a copy of source when target is nil, as adk-python does,
// and otherwise a copy of target with six fields filled from source where
// empty. The state always comes from source, so the redirect URL is checked
// against the state ADK issued rather than one the client names. adk-python's
// agent path keeps the answer's state, so there the check proves nothing.
func mergeOAuth2(target, source *toolauth.OAuth2Auth) *toolauth.OAuth2Auth {
	if target == nil {
		o := *source
		return &o
	}
	o := *target
	o.State = source.State
	o.ClientID = cmp.Or(o.ClientID, source.ClientID)
	o.ClientSecret = cmp.Or(o.ClientSecret, source.ClientSecret)
	o.RedirectURI = cmp.Or(o.RedirectURI, source.RedirectURI)
	o.CodeVerifier = cmp.Or(o.CodeVerifier, source.CodeVerifier)
	o.CodeChallengeMethod = cmp.Or(o.CodeChallengeMethod, source.CodeChallengeMethod)
	o.TokenEndpointAuthMethod = cmp.Or(o.TokenEndpointAuthMethod, source.TokenEndpointAuthMethod)
	return &o
}
