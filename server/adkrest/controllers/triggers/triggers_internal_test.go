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

package triggers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/api/idtoken"
)

func TestVerifyPushRequestAuth(t *testing.T) {
	const audience = "https://example-agent.example.com"

	tests := []struct {
		name             string
		expectedAudience string
		authHeader       string
		validate         func(ctx context.Context, idToken, audience string) (*idtoken.Payload, error)
		wantErr          bool
	}{
		{
			name:             "audience not configured: no-op regardless of header",
			expectedAudience: "",
			authHeader:       "",
			wantErr:          false,
		},
		{
			name:             "audience configured, no Authorization header",
			expectedAudience: audience,
			authHeader:       "",
			wantErr:          true,
		},
		{
			name:             "audience configured, non-Bearer scheme",
			expectedAudience: audience,
			authHeader:       "Basic dXNlcjpwYXNz",
			wantErr:          true,
		},
		{
			name:             "audience configured, valid bearer token",
			expectedAudience: audience,
			authHeader:       "Bearer valid-token",
			validate: func(ctx context.Context, idToken, aud string) (*idtoken.Payload, error) {
				if idToken != "valid-token" || aud != audience {
					t.Errorf("validate called with (%q, %q), want (%q, %q)", idToken, aud, "valid-token", audience)
				}
				return &idtoken.Payload{Audience: aud}, nil
			},
			wantErr: false,
		},
		{
			name:             "audience configured, token fails verification",
			expectedAudience: audience,
			authHeader:       "Bearer forged-token",
			validate: func(ctx context.Context, idToken, aud string) (*idtoken.Payload, error) {
				return nil, errors.New("idtoken: invalid token")
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := validateIDToken
			defer func() { validateIDToken = original }()
			if tt.validate != nil {
				validateIDToken = tt.validate
			} else {
				validateIDToken = func(context.Context, string, string) (*idtoken.Payload, error) {
					t.Fatal("validateIDToken should not be called")
					return nil, nil
				}
			}

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			err := verifyPushRequestAuth(req, tt.expectedAudience)
			if (err != nil) != tt.wantErr {
				t.Errorf("verifyPushRequestAuth() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
