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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
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
		// extraAuthHeader is appended as a second Authorization header when
		// non-empty.
		extraAuthHeader string
		validate        tokenValidator
		wantStatus      int // 0 means the request is accepted
		// oidcConfigured overrides the OIDC block for the misconfiguration
		// case, where an audience is absent but verification is still on.
		oidcConfigured *OIDCConfig
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
		{
			// A non-boolean email_verified must fail closed. adk-python uses
			// Python truthiness here and would accept the string "true"; this
			// pins the stricter behavior against a later "friendlier"
			// coercion.
			name:             "allow-list configured, email_verified is a non-boolean",
			expectedAudience: testAudience,
			allowedAccounts:  []string{testServiceAccount},
			authHeader:       "Bearer stringy-verified",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email_verified"] = "true"
				return p, nil
			},
			wantStatus: http.StatusForbidden,
		},
		{
			// Header.Values, not Header.Get: a second credential is ambiguous
			// rather than ignored, so this handler cannot disagree with a
			// proxy in front that reads the last value.
			name:             "duplicate Authorization headers are rejected",
			expectedAudience: testAudience,
			authHeader:       "Bearer valid-token",
			extraAuthHeader:  "Bearer second-token",
			wantStatus:       http.StatusUnauthorized,
		},
		{
			// Verification is switched off by leaving OIDC nil. A non-nil
			// block with no audience is a misconfiguration, and must not
			// silently fall through to the unverified path.
			name:           "OIDC set with no audience denies rather than disabling",
			oidcConfigured: &OIDCConfig{AllowedServiceAccounts: []string{testServiceAccount}},
			authHeader:     "Bearer valid-token",
			wantStatus:     http.StatusInternalServerError,
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
			oidc := tt.oidcConfigured
			if oidc == nil && tt.expectedAudience != "" {
				oidc = &OIDCConfig{
					ExpectedAudience:       tt.expectedAudience,
					AllowedServiceAccounts: tt.allowedAccounts,
				}
			}
			r := &RetriableRunner{oidc: oidc, validateIDToken: validate}

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
			if tt.authHeader != "" {
				req.Header.Add("Authorization", tt.authHeader)
			}
			if tt.extraAuthHeader != "" {
				req.Header.Add("Authorization", tt.extraAuthHeader)
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
		oidc: &OIDCConfig{ExpectedAudience: testAudience},
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
		name    string
		headers []string
		want    string
		wantErr bool
	}{
		{name: "canonical", headers: []string{"Bearer abc"}, want: "abc"},
		{name: "lowercase scheme", headers: []string{"bearer abc"}, want: "abc"},
		{name: "uppercase scheme", headers: []string{"BEARER abc"}, want: "abc"},
		{name: "surrounding space", headers: []string{"Bearer  abc "}, want: "abc"},
		{name: "no header", wantErr: true},
		{name: "empty header", headers: []string{""}, wantErr: true},
		{name: "scheme only", headers: []string{"Bearer"}, wantErr: true},
		{name: "empty credential", headers: []string{"Bearer "}, wantErr: true},
		{name: "other scheme", headers: []string{"Basic abc"}, wantErr: true},
		{name: "no scheme", headers: []string{"abc"}, wantErr: true},
		{name: "two headers", headers: []string{"Bearer abc", "Bearer def"}, wantErr: true},
		{name: "two headers, second unusable", headers: []string{"Bearer abc", "Basic def"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := bearerToken(tt.headers)
			if (err != nil) != tt.wantErr {
				t.Fatalf("bearerToken(%q) error = %v, wantErr %v", tt.headers, err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("bearerToken(%q) = %q, want %q", tt.headers, got, tt.want)
			}
		})
	}
}

// A 403 tells the caller its token verified and an allow-list rejected it. The
// status carries that much by design and matches adk-python; the body must not
// add to it, or an anonymous caller learns from the wording alone that its
// audience guess was right.
func TestRespondAuthErrorBodyDoesNotVaryWithStatus(t *testing.T) {
	t.Parallel()

	body := func(err error) string {
		w := httptest.NewRecorder()
		respondAuthError(w, err)
		return w.Body.String()
	}

	unauthorized := body(&authError{status: http.StatusUnauthorized, err: errors.New("invalid identity token: expired")})
	forbidden := body(&authError{status: http.StatusForbidden, err: errors.New(`principal "attacker@example.com" is not an allowed service account`)})
	if unauthorized != forbidden {
		t.Errorf("401 body %q differs from 403 body %q; the wording tells the caller which check it got past", unauthorized, forbidden)
	}
	for _, leak := range []string{"expired", "attacker@example.com", "allowed service account"} {
		if strings.Contains(forbidden, leak) {
			t.Errorf("response body %q leaks %q", forbidden, leak)
		}
	}
}

// Verification is disabled by leaving TriggerConfig.OIDC nil. A non-nil block
// with no audience reads as "pin these principals" but would verify nothing,
// so both constructors reject it rather than serving the pre-change behavior.
func TestControllersRejectOIDCWithoutAudience(t *testing.T) {
	t.Parallel()

	cfg := ControllerConfig{
		SessionService: newSessionService(),
		AgentLoader:    agent.NewSingleLoader(countingAgent(t, new(int))),
		TriggerConfig: TriggerConfig{
			MaxConcurrentRuns: 1,
			OIDC:              &OIDCConfig{AllowedServiceAccounts: []string{testServiceAccount}},
		},
	}
	if _, err := NewPubSubControllerWithConfig(cfg); err == nil {
		t.Error("NewPubSubControllerWithConfig() = nil error, want a rejection of OIDC without an audience")
	}
	if _, err := NewEventarcControllerWithConfig(cfg); err == nil {
		t.Error("NewEventarcControllerWithConfig() = nil error, want a rejection of OIDC without an audience")
	}
}

// The deprecated constructors have no error to return. They must not hand back
// a nil controller, and must not fall through to the unverified path either.
func TestDeprecatedConstructorsFailClosedOnOIDCWithoutAudience(t *testing.T) {
	t.Parallel()

	triggerConfig := TriggerConfig{
		MaxConcurrentRuns: 1,
		OIDC:              &OIDCConfig{AllowedServiceAccounts: []string{testServiceAccount}},
	}
	loader := agent.NewSingleLoader(countingAgent(t, new(int)))

	pubsubController := NewPubSubController(newSessionService(), loader, nil, nil, runner.PluginConfig{}, triggerConfig)
	if pubsubController == nil {
		t.Fatal("NewPubSubController() = nil, which panics on the first request")
	}
	eventarcController := NewEventarcController(newSessionService(), loader, nil, nil, runner.PluginConfig{}, triggerConfig)
	if eventarcController == nil {
		t.Fatal("NewEventarcController() = nil, which panics on the first request")
	}

	for name, r := range map[string]*RetriableRunner{"pubsub": pubsubController.runner, "eventarc": eventarcController.runner} {
		req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
		req.Header.Set("Authorization", "Bearer any-token")
		err := r.verifyPushRequestAuth(req)
		var authErr *authError
		if !errors.As(err, &authErr) {
			t.Errorf("%s: verifyPushRequestAuth() = %v, want a denial", name, err)
			continue
		}
		if authErr.status != http.StatusInternalServerError {
			t.Errorf("%s: status = %d, want %d", name, authErr.status, http.StatusInternalServerError)
		}
	}
}

// The controller copies the OIDC block, so a caller that keeps its config and
// mutates it later does not change what a running handler enforces.
func TestOIDCConfigIsCopiedAtConstruction(t *testing.T) {
	t.Parallel()

	oidc := &OIDCConfig{
		ExpectedAudience:       testAudience,
		AllowedServiceAccounts: []string{testServiceAccount},
	}
	c, err := NewPubSubControllerWithConfig(ControllerConfig{
		SessionService: newSessionService(),
		AgentLoader:    agent.NewSingleLoader(countingAgent(t, new(int))),
		TriggerConfig:  TriggerConfig{MaxConcurrentRuns: 1, OIDC: oidc},
	})
	if err != nil {
		t.Fatalf("NewPubSubControllerWithConfig() failed: %v", err)
	}

	// In place, not append: appending reallocates and so would leave the
	// controller's slice alone whether or not it was copied.
	const intruder = "intruder@example.iam.gserviceaccount.com"
	oidc.AllowedServiceAccounts[0] = intruder
	oidc.ExpectedAudience = "https://somewhere-else.example.com"

	c.runner.validateIDToken = func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
		if aud != testAudience {
			t.Errorf("validate called with audience %q, want the audience captured at construction %q", aud, testAudience)
		}
		p := googlePayload(aud)
		p.Claims["email"] = intruder
		return p, nil
	}

	req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
	req.Header.Set("Authorization", "Bearer valid-token")

	err = c.runner.verifyPushRequestAuth(req)
	var authErr *authError
	if !errors.As(err, &authErr) || authErr.status != http.StatusForbidden {
		t.Errorf("verifyPushRequestAuth() = %v, want 403: the allow-list changed under a running handler", err)
	}
}
