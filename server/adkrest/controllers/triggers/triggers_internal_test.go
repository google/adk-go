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
	"strings"
	"testing"

	"google.golang.org/api/idtoken"
)

const (
	testAudience       = "https://example-agent.example.com"
	testServiceAccount = "pubsub-push@example-project.iam.gserviceaccount.com"
)

// googlePayload builds the payload a Google-minted OIDC token for a push
// subscription carries: Google issuer plus a verified service account email.
func googlePayload(aud string) *idtoken.Payload {
	return &idtoken.Payload{
		Audience: aud,
		Issuer:   "https://accounts.google.com",
		Claims: map[string]any{
			"email":          testServiceAccount,
			"email_verified": true,
		},
	}
}

func TestVerifyPushRequestAuth(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		expectedAudience string
		allowedAccounts  []string
		authHeader       string
		validate         tokenValidator
		wantStatus       int // 0 means the request is accepted
	}{
		{
			name:             "audience not configured: no-op regardless of header",
			expectedAudience: "",
			authHeader:       "",
		},
		{
			name:             "audience configured, no Authorization header",
			expectedAudience: testAudience,
			authHeader:       "",
			wantStatus:       http.StatusUnauthorized,
		},
		{
			name:             "audience configured, non-Bearer scheme",
			expectedAudience: testAudience,
			authHeader:       "Basic dXNlcjpwYXNz",
			wantStatus:       http.StatusUnauthorized,
		},
		{
			name:             "audience configured, Bearer with empty credential",
			expectedAudience: testAudience,
			authHeader:       "Bearer   ",
			wantStatus:       http.StatusUnauthorized,
		},
		{
			// RFC 9110 makes the auth scheme case-insensitive.
			name:             "lowercase bearer scheme is accepted",
			expectedAudience: testAudience,
			authHeader:       "bearer valid-token",
			validate: func(_ context.Context, idToken, aud string) (*idtoken.Payload, error) {
				if idToken != "valid-token" {
					t.Errorf("validate called with token %q, want %q", idToken, "valid-token")
				}
				return googlePayload(aud), nil
			},
		},
		{
			name:             "audience configured, valid bearer token",
			expectedAudience: testAudience,
			authHeader:       "Bearer valid-token",
			validate: func(_ context.Context, idToken, aud string) (*idtoken.Payload, error) {
				if idToken != "valid-token" || aud != testAudience {
					t.Errorf("validate called with (%q, %q), want (%q, %q)", idToken, aud, "valid-token", testAudience)
				}
				return googlePayload(aud), nil
			},
		},
		{
			name:             "audience configured, token fails verification",
			expectedAudience: testAudience,
			authHeader:       "Bearer forged-token",
			validate: func(context.Context, string, string) (*idtoken.Payload, error) {
				return nil, errors.New("idtoken: invalid token")
			},
			wantStatus: http.StatusUnauthorized,
		},
		{
			// idtoken.Validate parses iss but leaves it to the caller.
			name:             "token from a non-Google issuer is rejected",
			expectedAudience: testAudience,
			authHeader:       "Bearer other-issuer",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Issuer = "https://accounts.example.com"
				return p, nil
			},
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:             "bare issuer spelling is accepted",
			expectedAudience: testAudience,
			authHeader:       "Bearer bare-issuer",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Issuer = "accounts.google.com"
				return p, nil
			},
		},
		{
			name:             "allow-list configured, matching principal",
			expectedAudience: testAudience,
			allowedAccounts:  []string{"other@example.iam.gserviceaccount.com", testServiceAccount},
			authHeader:       "Bearer valid-token",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				return googlePayload(aud), nil
			},
		},
		{
			// The audience is chosen by whoever mints the token, so a
			// Google-signed token for this audience can belong to an unrelated
			// principal. Without the allow-list this request is accepted.
			name:             "allow-list configured, unlisted principal",
			expectedAudience: testAudience,
			allowedAccounts:  []string{testServiceAccount},
			authHeader:       "Bearer valid-token",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email"] = "someone-else@gmail.com"
				return p, nil
			},
			wantStatus: http.StatusForbidden,
		},
		{
			// getOpenIdToken defaults includeEmail to false, so a valid token
			// may carry no email claim at all. That must not pass the pin.
			name:             "allow-list configured, token has no email claim",
			expectedAudience: testAudience,
			allowedAccounts:  []string{testServiceAccount},
			authHeader:       "Bearer no-email",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				return &idtoken.Payload{Audience: aud, Issuer: "https://accounts.google.com", Claims: map[string]any{}}, nil
			},
			wantStatus: http.StatusForbidden,
		},
		{
			name:             "allow-list configured, email not verified",
			expectedAudience: testAudience,
			allowedAccounts:  []string{testServiceAccount},
			authHeader:       "Bearer unverified-email",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email_verified"] = false
				return p, nil
			},
			wantStatus: http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			validate := tt.validate
			if validate == nil {
				validate = func(context.Context, string, string) (*idtoken.Payload, error) {
					t.Error("validateIDToken should not have been called")
					return nil, errors.New("unexpected call")
				}
			}
			r := &RetriableRunner{
				triggerConfig: TriggerConfig{
					ExpectedAudience:       tt.expectedAudience,
					AllowedServiceAccounts: tt.allowedAccounts,
				},
				validateIDToken: validate,
			}

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			err := r.verifyPushRequestAuth(req)
			if tt.wantStatus == 0 {
				if err != nil {
					t.Fatalf("verifyPushRequestAuth() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("verifyPushRequestAuth() = nil, want an error with status %d", tt.wantStatus)
			}
			var authErr *authError
			if !errors.As(err, &authErr) {
				t.Fatalf("verifyPushRequestAuth() = %v, want an *authError", err)
			}
			if authErr.status != tt.wantStatus {
				t.Errorf("status = %d, want %d", authErr.status, tt.wantStatus)
			}
		})
	}
}

// The rejection body must not tell an anonymous caller which check failed.
func TestRespondAuthErrorHidesDetail(t *testing.T) {
	t.Parallel()

	r := &RetriableRunner{
		triggerConfig: TriggerConfig{ExpectedAudience: testAudience},
		validateIDToken: func(context.Context, string, string) (*idtoken.Payload, error) {
			return nil, errors.New("idtoken: token expired at 2026-01-01")
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
	req.Header.Set("Authorization", "Bearer expired")

	w := httptest.NewRecorder()
	respondAuthError(w, r.verifyPushRequestAuth(req))

	if got := w.Code; got != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", got, http.StatusUnauthorized)
	}
	if got := w.Header().Get("WWW-Authenticate"); got == "" {
		t.Error("401 response is missing a WWW-Authenticate header")
	}
	if body := w.Body.String(); strings.Contains(body, "expired") {
		t.Errorf("response body leaks the verification failure reason: %q", body)
	}
}

func TestBearerToken(t *testing.T) {
	t.Parallel()

	tests := []struct {
		header  string
		want    string
		wantErr bool
	}{
		{header: "Bearer abc", want: "abc"},
		{header: "bearer abc", want: "abc"},
		{header: "BEARER abc", want: "abc"},
		{header: "Bearer  abc ", want: "abc"},
		{header: "", wantErr: true},
		{header: "Bearer", wantErr: true},
		{header: "Bearer ", wantErr: true},
		{header: "Basic abc", wantErr: true},
		{header: "abc", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.header, func(t *testing.T) {
			t.Parallel()
			got, err := bearerToken(tt.header)
			if (err != nil) != tt.wantErr {
				t.Fatalf("bearerToken(%q) error = %v, wantErr %v", tt.header, err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("bearerToken(%q) = %q, want %q", tt.header, got, tt.want)
			}
		})
	}
}
