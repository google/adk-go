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

package auth_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/auth"
)

func TestStaticToken(t *testing.T) {
	if got := resolvedAuth(t, auth.StaticToken("abc")); got != "Bearer abc" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer abc")
	}
}

func TestAPIKey(t *testing.T) {
	cred, err := auth.APIKey("X-Api-Key", "secret").Credential(t.Context())
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	if got := h.Get("X-Api-Key"); got != "secret" {
		t.Errorf("X-Api-Key = %q, want %q", got, "secret")
	}
}

func TestTokenSourceProvider(t *testing.T) {
	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"})
	if got := resolvedAuth(t, auth.TokenSourceProvider(ts)); got != "Bearer tok" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer tok")
	}
}

func TestTokenSourceProviderNil(t *testing.T) {
	if _, err := auth.TokenSourceProvider(nil).Credential(t.Context()); err == nil {
		t.Fatal("Credential() = nil error, want error for nil token source")
	}
}

func TestServiceAccountInvalidKey(t *testing.T) {
	// Scopes are set so the invalid-key path (not the scopes gate) is exercised.
	p := auth.ServiceAccount(auth.ServiceAccountConfig{
		JSONKey: []byte("not-json"),
		Scopes:  []string{"https://www.googleapis.com/auth/cloud-platform"},
	})
	if _, err := p.Credential(t.Context()); err == nil {
		t.Fatal("Credential() = nil error, want error for invalid service-account key")
	}
}

func TestServiceAccountAccessTokenRequiresScopes(t *testing.T) {
	// Explicit key + no scopes must error (parity with adk-python), before any I/O.
	p := auth.ServiceAccount(auth.ServiceAccountConfig{JSONKey: []byte("{}")})
	if _, err := p.Credential(t.Context()); err == nil {
		t.Fatal("Credential() = nil error, want error for missing scopes")
	}
}

func TestADCReturnsProvider(t *testing.T) {
	// Only assert construction; resolving ADC depends on the ambient
	// environment and would perform I/O, which the offline test suite avoids.
	if auth.ADC("https://www.googleapis.com/auth/cloud-platform") == nil {
		t.Fatal("ADC() = nil provider")
	}
}

func TestProviderFunc(t *testing.T) {
	want := auth.BearerCredential{Token: "abc"}
	var p auth.CredentialProvider = auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return want, nil
	})
	got, err := p.Credential(t.Context())
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	if got != want {
		t.Errorf("Credential() = %v, want %v", got, want)
	}
}

func TestConsentRequiredError(t *testing.T) {
	err := error(&auth.ConsentRequiredError{AuthURI: "https://consent.example", Nonce: "n", Key: "k"})

	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Fatalf("errors.As failed for %v", err)
	}
	if consent.AuthURI != "https://consent.example" {
		t.Errorf("AuthURI = %q, want %q", consent.AuthURI, "https://consent.example")
	}
	if !strings.Contains(err.Error(), "consent.example") {
		t.Errorf("Error() = %q, want it to mention the auth URI", err.Error())
	}
}

// resolvedAuth resolves the provider and returns the Authorization header its
// credential produces.
func resolvedAuth(t *testing.T, p auth.CredentialProvider) string {
	t.Helper()
	cred, err := p.Credential(t.Context())
	if err != nil {
		t.Fatalf("Credential() error = %v", err)
	}
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	return h.Get("Authorization")
}
