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

package sessionutils_test

import (
	"encoding/json"
	"math"
	"testing"

	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/tool/toolauth"
)

// TestCheckRequestedAuthConfigsReadsNumbersAsFloat64 pins the part of the
// check that goes beyond a plain JSON round trip: the Vertex AI service stores
// numbers as float64, so a config that conversion breaks is refused by every
// service that calls the check.
func TestCheckRequestedAuthConfigsReadsNumbersAsFloat64(t *testing.T) {
	scheme := toolauth.Scheme{
		Type:  toolauth.SchemeTypeOAuth2,
		Flows: &toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{TokenURL: "https://example.com/token"}},
	}
	tests := []struct {
		name    string
		config  toolauth.Config
		wantErr bool
	}{
		{
			// 2^53 + 1 loses its last digit as a float64 but still fits.
			name: "expiresAt that float64 rounds within range",
			config: toolauth.Config{AuthScheme: scheme, RawAuthCredential: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "cid", ExpiresAt: 9007199254740993},
			}},
			wantErr: false,
		},
		{
			// As a float64 this is 2^63, which does not fit expiresAt.
			name: "expiresAt that float64 rounds out of range",
			config: toolauth.Config{AuthScheme: scheme, RawAuthCredential: &toolauth.Credential{
				AuthType: toolauth.CredentialTypeOAuth2,
				OAuth2:   &toolauth.OAuth2Auth{ClientID: "cid", ExpiresAt: math.MaxInt64},
			}},
			wantErr: true,
		},
		{
			// Extra keeps a number as written, but no float64 holds it.
			name: "number in extra beyond float64",
			config: toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"},
				Extra:      map[string]any{"big": json.Number("1e400")},
			},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := sessionutils.CheckRequestedAuthConfigs(map[string]toolauth.Config{"call-1": tc.config})
			if (err != nil) != tc.wantErr {
				t.Errorf("CheckRequestedAuthConfigs() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
