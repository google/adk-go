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

package authn

import (
	"context"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/api/idtoken"
)

const (
	testAudience       = "https://example-agent.example.com"
	testServiceAccount = "pubsub-push@example-project.iam.gserviceaccount.com"
	testSubject        = "108100000000000000000"
)

// googlePayload builds the payload a Google-minted OIDC token for a push
// subscription carries: Google issuer, a subject, and a verified service
// account email.
func googlePayload(aud string) *idtoken.Payload {
	return &idtoken.Payload{
		Audience: aud,
		Issuer:   "https://accounts.google.com",
		Subject:  testSubject,
		Claims: map[string]any{
			"email":          testServiceAccount,
			"email_verified": true,
		},
	}
}

// quietLogs silences the rejection log lines for the duration of a test. The
// provider logs every denial on purpose, which would otherwise bury the test
// output.
func quietLogs(t *testing.T) {
	t.Helper()
	prev := log.Writer()
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(prev) })
}

func TestGoogleOIDCAuthenticate(t *testing.T) {
	quietLogs(t)

	tests := []struct {
		name            string
		allowedAccounts []string
		authHeader      string
		// extraAuthHeader is appended as a second Authorization header when
		// non-empty.
		extraAuthHeader string
		// validate stands in for idtoken.Validate. Nil means the provider must
		// not reach it.
		validate tokenValidator
		wantErr  bool
		// wantForbidden marks a rejection that must wrap ErrForbidden (a 403)
		// rather than ErrUnauthenticated (a 401): the token verified but named a
		// principal outside the allow-list. Only meaningful with wantErr.
		wantForbidden bool
		// wantUserID is checked only when the request is accepted.
		wantUserID string
	}{
		{
			name:       "no Authorization header",
			authHeader: "",
			wantErr:    true,
		},
		{
			name:       "non-Bearer scheme",
			authHeader: "Basic dXNlcjpwYXNz",
			wantErr:    true,
		},
		{
			name:       "Bearer with empty credential",
			authHeader: "Bearer   ",
			wantErr:    true,
		},
		{
			// RFC 9110 makes the auth scheme case-insensitive.
			name:            "lowercase bearer scheme is accepted",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "bearer valid-token",
			validate: func(_ context.Context, idToken, aud string) (*idtoken.Payload, error) {
				if idToken != "valid-token" {
					t.Errorf("validate called with token %q, want %q", idToken, "valid-token")
				}
				return googlePayload(aud), nil
			},
			wantUserID: testSubject,
		},
		{
			name:            "valid bearer token",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer valid-token",
			validate: func(_ context.Context, idToken, aud string) (*idtoken.Payload, error) {
				if idToken != "valid-token" || aud != testAudience {
					t.Errorf("validate called with (%q, %q), want (%q, %q)", idToken, aud, "valid-token", testAudience)
				}
				return googlePayload(aud), nil
			},
			wantUserID: testSubject,
		},
		{
			name:       "token fails verification",
			authHeader: "Bearer forged-token",
			validate: func(context.Context, string, string) (*idtoken.Payload, error) {
				return nil, errors.New("idtoken: invalid token")
			},
			wantErr: true,
		},
		{
			name:       "verification returns no payload",
			authHeader: "Bearer empty-payload",
			validate: func(context.Context, string, string) (*idtoken.Payload, error) {
				return nil, nil
			},
			wantErr: true,
		},
		{
			// idtoken.Validate parses iss but leaves it to the caller.
			name:       "token from a non-Google issuer is rejected",
			authHeader: "Bearer other-issuer",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Issuer = "https://accounts.example.com"
				return p, nil
			},
			wantErr: true,
		},
		{
			name:            "bare issuer spelling is accepted",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer bare-issuer",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Issuer = "accounts.google.com"
				return p, nil
			},
			wantUserID: testSubject,
		},
		{
			name:            "allow-list configured, matching principal",
			allowedAccounts: []string{"other@example.iam.gserviceaccount.com", testServiceAccount},
			authHeader:      "Bearer valid-token",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				return googlePayload(aud), nil
			},
			wantUserID: testSubject,
		},
		{
			// The audience is chosen by whoever mints the token, so a
			// Google-signed token for this audience can belong to an unrelated
			// principal. The allow-list is what refuses it.
			name:            "allow-list configured, unlisted principal",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer valid-token",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email"] = "someone-else@gmail.com"
				return p, nil
			},
			wantErr:       true,
			wantForbidden: true,
		},
		{
			// getOpenIdToken defaults includeEmail to false, so a valid token
			// may carry no email claim at all. That must not pass the pin.
			name:            "allow-list configured, token has no email claim",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer no-email",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				return &idtoken.Payload{
					Audience: aud,
					Issuer:   "https://accounts.google.com",
					Subject:  testSubject,
					Claims:   map[string]any{},
				}, nil
			},
			wantErr:       true,
			wantForbidden: true,
		},
		{
			// A blank email cannot match a listed account (the constructor keeps
			// blanks out of the list), so even a token that claims the email is
			// verified is refused when it carries no email string.
			name:            "allow-list configured, verified but empty email",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer empty-email",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				return &idtoken.Payload{
					Audience: aud,
					Issuer:   "https://accounts.google.com",
					Subject:  testSubject,
					Claims:   map[string]any{"email_verified": true},
				}, nil
			},
			wantErr:       true,
			wantForbidden: true,
		},
		{
			name:            "allow-list configured, email not verified",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer unverified-email",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email_verified"] = false
				return p, nil
			},
			wantErr:       true,
			wantForbidden: true,
		},
		{
			// A non-boolean email_verified must fail closed. adk-python uses
			// Python truthiness here and would accept the string "true"; this
			// pins the stricter behavior against a later "friendlier"
			// coercion.
			name:            "allow-list configured, email_verified is a non-boolean",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer stringy-verified",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email_verified"] = "true"
				return p, nil
			},
			wantErr:       true,
			wantForbidden: true,
		},
		{
			// Header.Values, not Header.Get: a second credential is ambiguous
			// rather than ignored, so this provider cannot disagree with a
			// proxy in front that reads the last value.
			name:            "duplicate Authorization headers are rejected",
			authHeader:      "Bearer valid-token",
			extraAuthHeader: "Bearer second-token",
			wantErr:         true,
		},
		{
			// A verified, listed email with no subject still names the caller:
			// the Caller falls back to the email.
			name:            "no subject falls back to the email",
			allowedAccounts: []string{testServiceAccount},
			authHeader:      "Bearer no-subject",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Subject = ""
				return p, nil
			},
			wantUserID: testServiceAccount,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validate := tt.validate
			if validate == nil {
				validate = func(context.Context, string, string) (*idtoken.Payload, error) {
					t.Error("validate should not have been called")
					return nil, errors.New("unexpected call")
				}
			}
			g := &googleOIDC{
				audience:               testAudience,
				allowedServiceAccounts: tt.allowedAccounts,
				validate:               validate,
			}

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", nil)
			if tt.authHeader != "" {
				req.Header.Add("Authorization", tt.authHeader)
			}
			if tt.extraAuthHeader != "" {
				req.Header.Add("Authorization", tt.extraAuthHeader)
			}

			caller, err := g.Authenticate(req)
			if tt.wantErr {
				wantSentinel, wantName := ErrUnauthenticated, "ErrUnauthenticated"
				if tt.wantForbidden {
					wantSentinel, wantName = ErrForbidden, "ErrForbidden"
				}
				if !errors.Is(err, wantSentinel) {
					t.Fatalf("Authenticate() error = %v, want it to wrap %s", err, wantName)
				}
				// The two sentinels stay disjoint, so anything classifying a
				// rejection with errors.Is gets one answer. Middleware would cope
				// either way, since it matches ErrForbidden first, but nothing
				// else should have to know that order.
				if tt.wantForbidden && errors.Is(err, ErrUnauthenticated) {
					t.Errorf("Authenticate() error = %v, want it not to also wrap ErrUnauthenticated", err)
				}
				if caller != nil {
					t.Errorf("Authenticate() caller = %+v, want nil on error", caller)
				}
				return
			}
			if err != nil {
				t.Fatalf("Authenticate() error = %v, want nil", err)
			}
			if got := caller.UserID; got != tt.wantUserID {
				t.Errorf("Caller.UserID = %q, want %q", got, tt.wantUserID)
			}
		})
	}
}

// A rejection must not be reported as an internal provider failure, which is
// what Middleware answers 500 to. Pub/Sub push resends on any non-2xx, so the
// status does not change whether a forged token is redelivered; the point is
// that a bad credential is the caller's error, and reporting it as a 5xx would
// charge it to the server in the operator's logs and metrics and hide a real
// outage among forged traffic.
func TestGoogleOIDCRejectionIsNotAnInternalFailure(t *testing.T) {
	quietLogs(t)

	g := &googleOIDC{
		audience: testAudience,
		validate: func(context.Context, string, string) (*idtoken.Payload, error) {
			return nil, errors.New("idtoken: invalid token")
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer forged-token")

	rec := httptest.NewRecorder()
	Middleware(g)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the next handler ran on a rejected request")
	})).ServeHTTP(rec, req)

	if got := rec.Code; got != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", got, http.StatusUnauthorized)
	}
}

// The response body must not tell a caller which check it failed. The
// unauthenticated reasons must be indistinguishable from one another, and no
// rejection body — 401 or 403 — may name the allow-list or the refused
// principal. The 401/403 status is deliberately allowed to differ: that is the
// one bit an operator is meant to see.
func TestGoogleOIDCRejectionBodyDoesNotVaryWithReason(t *testing.T) {
	quietLogs(t)

	// Genuinely-unauthenticated reasons: their status and body must both match.
	unauthReasons := map[string]tokenValidator{
		"bad signature": func(context.Context, string, string) (*idtoken.Payload, error) {
			return nil, errors.New("idtoken: invalid token")
		},
		"untrusted issuer": func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
			p := googlePayload(aud)
			p.Issuer = "https://accounts.example.com"
			return p, nil
		},
	}

	var first string
	for name, validate := range unauthReasons {
		g := &googleOIDC{
			audience:               testAudience,
			allowedServiceAccounts: []string{testServiceAccount},
			validate:               validate,
		}
		req := httptest.NewRequest(http.MethodPost, "/", nil)
		req.Header.Set("Authorization", "Bearer some-token")

		rec := httptest.NewRecorder()
		Middleware(g)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})).ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s: status = %d, want %d", name, rec.Code, http.StatusUnauthorized)
		}
		body := rec.Body.String()
		if first == "" {
			first = body
			continue
		}
		if body != first {
			t.Errorf("%s: body = %q, want it identical to every other unauthenticated rejection (%q)", name, body, first)
		}
	}
	if first == "" {
		t.Fatal("no unauthenticated rejection was exercised")
	}
	if strings.Contains(strings.ToLower(first), "service account") {
		t.Errorf("rejection body %q names the allow-list", first)
	}

	// An unlisted principal is authenticated but not permitted: a 403, not a
	// 401. The status is meant to differ; the body still must not name the
	// allow-list nor the principal that was refused.
	g := &googleOIDC{
		audience:               testAudience,
		allowedServiceAccounts: []string{testServiceAccount},
		validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
			p := googlePayload(aud)
			p.Claims["email"] = "someone-else@gmail.com"
			return p, nil
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer some-token")

	rec := httptest.NewRecorder()
	Middleware(g)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})).ServeHTTP(rec, req)

	if rec.Code != http.StatusForbidden {
		t.Errorf("unlisted principal: status = %d, want %d", rec.Code, http.StatusForbidden)
	}
	forbiddenBody := strings.ToLower(rec.Body.String())
	if strings.Contains(forbiddenBody, "service account") {
		t.Errorf("403 body %q names the allow-list", rec.Body.String())
	}
	if strings.Contains(forbiddenBody, "someone-else@gmail.com") {
		t.Errorf("403 body %q names the refused principal", rec.Body.String())
	}
}

// A verified token reaches the next handler with its principal on the request
// context, which is what an authz.Authorizer downstream reads.
func TestGoogleOIDCPutsCallerOnContext(t *testing.T) {
	g := &googleOIDC{
		audience:               testAudience,
		allowedServiceAccounts: []string{testServiceAccount},
		validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
			return googlePayload(aud), nil
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer valid-token")

	var got *Caller
	rec := httptest.NewRecorder()
	Middleware(g)(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		c, ok := CallerFromContext(r.Context())
		if !ok {
			t.Error("CallerFromContext() ok = false, want true")
			return
		}
		got = c
	})).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got == nil {
		t.Fatal("no caller reached the next handler")
	}
	if got.UserID != testSubject {
		t.Errorf("Caller.UserID = %q, want %q", got.UserID, testSubject)
	}
	if email := got.Claims["email"]; email != testServiceAccount {
		t.Errorf("Claims[email] = %v, want %q", email, testServiceAccount)
	}
	if verified := got.Claims["email_verified"]; verified != true {
		t.Errorf("Claims[email_verified] = %v, want true", verified)
	}
	if aud := got.Claims["aud"]; aud != testAudience {
		t.Errorf("Claims[aud] = %v, want %q", aud, testAudience)
	}
}

// Neither field has a usable zero value, and the two fail in opposite
// directions. An audience-less authenticator would take a token minted for any
// audience, since idtoken.Validate skips that check when the audience is empty.
// An empty allow-list matches nobody, so that authenticator refuses every
// caller while looking configured. Running without authentication is done by
// leaving the Authenticator unset instead.
func TestNewGoogleOIDCRequiresAudienceAndAllowList(t *testing.T) {
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{AllowedServiceAccounts: []string{testServiceAccount}}); err == nil {
		t.Error("NewGoogleOIDC(no audience) error = nil, want an error for the empty audience")
	}
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience}); err == nil {
		t.Errorf("NewGoogleOIDC(no allow-list) error = nil, want an error for the missing allow-list")
	}
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{}}); err == nil {
		t.Errorf("NewGoogleOIDC(empty allow-list) error = nil, want an error for the empty allow-list")
	}
	// A blank or space-padded entry must be rejected, not stored: []string{""}
	// is what strings.Split of an unset value returns, and a trailing comma
	// leaves one alongside a real entry.
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{""}}); err == nil {
		t.Errorf("NewGoogleOIDC([\"\"]) error = nil, want an error for the blank entry")
	}
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{testServiceAccount, ""}}); err == nil {
		t.Errorf("NewGoogleOIDC([acct, \"\"]) error = nil, want an error for the blank entry")
	}
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{" " + testServiceAccount + " "}}); err == nil {
		t.Errorf("NewGoogleOIDC([padded]) error = nil, want an error for the surrounding whitespace")
	}
	if _, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{testServiceAccount}}); err != nil {
		t.Errorf("NewGoogleOIDC(audience, [acct]) error = %v, want nil", err)
	}
}

// The constructor must install the production validator. Without the assignment
// an authenticator it returns would carry a nil validate, which validateToken
// only survives by recovering the nil call into a rejection -- every request
// would be refused. The fallback that used to hide this is gone, so the
// assignment is load-bearing and pinned here.
func TestNewGoogleOIDCInstallsValidator(t *testing.T) {
	a, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: []string{testServiceAccount}})
	if err != nil {
		t.Fatalf("NewGoogleOIDC() error = %v", err)
	}
	g, ok := a.(*googleOIDC)
	if !ok {
		t.Fatalf("NewGoogleOIDC() returned %T, want *googleOIDC", a)
	}
	if g.validate == nil {
		t.Error("NewGoogleOIDC() left validate nil; production would reject every request")
	}
}

// An authenticator built by NewGoogleOIDC must refuse a token minted for any
// audience other than the configured one. idtoken.Validate skips the audience
// check entirely when the audience it is handed is empty, so dropping the
// constructor's audience wiring would fail silently rather than loudly: it
// would leave an authenticator that accepts a listed principal's token whatever
// it was minted for. Going through the constructor is what pins that, since a
// googleOIDC literal sets the audience itself.
//
// The stand-in validator reproduces only idtoken.Validate's audience rule: an
// empty audience accepts anything, and otherwise the token's own audience has
// to match.
func TestNewGoogleOIDCEnforcesAudience(t *testing.T) {
	quietLogs(t)

	// authenticate runs a token minted for mintedFor through an authenticator
	// built by the constructor and configured with testAudience.
	authenticate := func(t *testing.T, mintedFor string) error {
		t.Helper()
		a, err := NewGoogleOIDC(GoogleOIDCConfig{
			Audience:               testAudience,
			AllowedServiceAccounts: []string{testServiceAccount},
		})
		if err != nil {
			t.Fatalf("NewGoogleOIDC() error = %v", err)
		}
		g, ok := a.(*googleOIDC)
		if !ok {
			t.Fatalf("NewGoogleOIDC() returned %T, want *googleOIDC", a)
		}
		g.validate = func(_ context.Context, _, audience string) (*idtoken.Payload, error) {
			// Checked directly as well as through the rule below, so that
			// weakening the rule cannot quietly leave this test asserting
			// nothing about the audience.
			if audience != testAudience {
				t.Errorf("validate called with audience %q, want the configured %q", audience, testAudience)
			}
			if audience != "" && audience != mintedFor {
				return nil, errors.New("idtoken: audience provided does not match aud claim in the JWT")
			}
			return googlePayload(mintedFor), nil
		}

		req := httptest.NewRequest(http.MethodPost, "/", nil)
		req.Header.Set("Authorization", "Bearer some-token")
		_, err = a.Authenticate(req)
		return err
	}

	// The accepted case keeps the refusal below honest: without it a validator
	// that rejected everything would pass just as well.
	if err := authenticate(t, testAudience); err != nil {
		t.Errorf("Authenticate(token for the configured audience) error = %v, want nil", err)
	}
	if err := authenticate(t, "https://another-agent.example.com"); !errors.Is(err, ErrUnauthenticated) {
		t.Errorf("Authenticate(token for another audience) error = %v, want it to wrap ErrUnauthenticated: "+
			"the constructor did not install the audience", err)
	}
}

// The allow-list is copied at construction, so a caller that reuses its slice
// cannot change what a running authenticator enforces.
func TestNewGoogleOIDCCopiesAllowList(t *testing.T) {
	quietLogs(t)

	allowed := []string{testServiceAccount}
	a, err := NewGoogleOIDC(GoogleOIDCConfig{Audience: testAudience, AllowedServiceAccounts: allowed})
	if err != nil {
		t.Fatalf("NewGoogleOIDC() error = %v", err)
	}
	allowed[0] = "attacker@evil.example.com"

	g, ok := a.(*googleOIDC)
	if !ok {
		t.Fatalf("NewGoogleOIDC() returned %T, want *googleOIDC", a)
	}
	g.validate = func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
		p := googlePayload(aud)
		p.Claims["email"] = "attacker@evil.example.com"
		return p, nil
	}

	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer valid-token")
	// The attacker's mutation must not have entered the allow-list, so the token
	// verifies but its principal is refused: ErrForbidden, not acceptance.
	if _, err := g.Authenticate(req); !errors.Is(err, ErrForbidden) {
		t.Errorf("Authenticate() error = %v, want it to wrap ErrForbidden: the allow-list was not copied", err)
	}
}

// No request may crash the handler. idtoken.Validate panics on some malformed
// tokens before it can return an error (for an ES256 token it slices the
// signature to 32 bytes with no length check), so Authenticate must turn a
// panic in the validator into an ordinary rejection rather than let it unwind
// through the handler.
func TestGoogleOIDCValidatePanicIsRejected(t *testing.T) {
	quietLogs(t)

	g := &googleOIDC{
		audience:               testAudience,
		allowedServiceAccounts: []string{testServiceAccount},
		validate: func(context.Context, string, string) (*idtoken.Payload, error) {
			panic("slice bounds out of range [:32] with capacity 0")
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer malformed-token")

	rec := httptest.NewRecorder()
	// If Authenticate let the panic through, this ServeHTTP call would panic and
	// fail the test rather than record a status.
	Middleware(g)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the next handler ran on a token that failed validation")
	})).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
}

// validate has no default, so a googleOIDC built without one -- which only a
// test does, since NewGoogleOIDC always assigns it -- holds a nil func. The
// same recover that absorbs a panic inside the validator absorbs the nil call,
// so such a struct refuses every request instead of crashing the handler on the
// first one. That is asserted rather than assumed, because it is what makes the
// constructor's assignment the only thing standing between production and a
// provider that rejects everything.
func TestGoogleOIDCNilValidatorIsRejected(t *testing.T) {
	quietLogs(t)

	g := &googleOIDC{
		audience:               testAudience,
		allowedServiceAccounts: []string{testServiceAccount},
	}
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer some-token")

	rec := httptest.NewRecorder()
	// A nil call that escaped the recover would panic here rather than record a
	// status.
	Middleware(g)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the next handler ran with no validator installed")
	})).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
}

func TestBearerToken(t *testing.T) {
	tests := []struct {
		name    string
		headers []string
		want    string
		wantErr bool
	}{
		{name: "no header", headers: nil, wantErr: true},
		{name: "empty header value", headers: []string{""}, wantErr: true},
		{name: "scheme only", headers: []string{"Bearer"}, wantErr: true},
		{name: "empty credential", headers: []string{"Bearer "}, wantErr: true},
		{name: "whitespace credential", headers: []string{"Bearer    "}, wantErr: true},
		{name: "wrong scheme", headers: []string{"Basic dXNlcjpwYXNz"}, wantErr: true},
		{name: "two headers", headers: []string{"Bearer a", "Bearer b"}, wantErr: true},
		{name: "canonical", headers: []string{"Bearer abc"}, want: "abc"},
		{name: "lowercase scheme", headers: []string{"bearer abc"}, want: "abc"},
		{name: "mixed-case scheme", headers: []string{"BeArEr abc"}, want: "abc"},
		{name: "padded credential", headers: []string{"Bearer   abc  "}, want: "abc"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := bearerToken(tt.headers)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("bearerToken(%q) = %q, want an error", tt.headers, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("bearerToken(%q) error = %v, want nil", tt.headers, err)
			}
			if got != tt.want {
				t.Errorf("bearerToken(%q) = %q, want %q", tt.headers, got, tt.want)
			}
		})
	}
}
