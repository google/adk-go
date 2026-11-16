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
	"fmt"

	"google.golang.org/adk/v2/tool/toolauth"
)

// GenerateAuthRequest returns the config ADK sends to the client in an
// adk_request_credential call: with an authorization URI for OAuth2 and OpenID
// Connect, without configured secrets, and under cfg's key.
func GenerateAuthRequest(cfg *toolauth.Config) (*toolauth.Config, error) {
	t := cfg.AuthScheme.Type
	if t != toolauth.SchemeTypeOAuth2 && t != toolauth.SchemeTypeOpenIDConnect {
		return cfg.WithoutSecrets()
	}
	if ex := cfg.ExchangedAuthCredential; ex != nil && ex.OAuth2 != nil && ex.OAuth2.AuthURI != "" {
		return cfg.WithoutSecrets()
	}
	raw := cfg.RawAuthCredential
	if raw == nil {
		return nil, fmt.Errorf("authhandler: %s auth scheme requires a raw auth credential", t)
	}
	if raw.OAuth2 == nil {
		return nil, fmt.Errorf("authhandler: %s auth scheme requires oauth2 in the raw auth credential", t)
	}
	// WithoutSecrets copies raw and exchanged separately, so sharing the
	// pointer here does not tie them together.
	exchanged := raw
	if raw.OAuth2.AuthURI == "" {
		if raw.OAuth2.ClientID == "" {
			return nil, fmt.Errorf("authhandler: %s auth scheme requires a client ID in the raw auth credential", t)
		}
		generated, err := GenerateAuthURI(cfg)
		if err != nil {
			return nil, err
		}
		exchanged = generated
	}
	req := &toolauth.Config{
		AuthScheme:              cfg.AuthScheme,
		RawAuthCredential:       raw,
		ExchangedAuthCredential: exchanged,
		CredentialKey:           cfg.CredentialKey,
	}
	return req.WithoutSecrets()
}
