// Copyright 2025 Google LLC
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

package auth

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestAuthCredential_Copy_Nil(t *testing.T) {
	var cred *AuthCredential
	got := cred.Copy()
	if got != nil {
		t.Errorf("Copy() of nil = %v, want nil", got)
	}
}

func TestAuthCredential_Copy_APIKey(t *testing.T) {
	cred := &AuthCredential{
		AuthType:    AuthCredentialTypeAPIKey,
		ResourceRef: "my-ref",
		APIKey:      "secret-key",
	}
	got := cred.Copy()

	if got == cred {
		t.Error("Copy() returned same pointer")
	}
	if diff := cmp.Diff(cred, got); diff != "" {
		t.Errorf("Copy() mismatch (-want +got):\n%s", diff)
	}
}

func TestAuthCredential_Copy_HTTP(t *testing.T) {
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeHTTP,
		HTTP: &HTTPAuth{
			Scheme: "bearer",
			Credentials: &HTTPCredentials{
				Username: "user",
				Password: "pass",
				Token:    "token123",
			},
		},
	}
	got := cred.Copy()

	if got == cred {
		t.Error("Copy() returned same pointer")
	}
	if got.HTTP == cred.HTTP {
		t.Error("Copy() returned same HTTP pointer")
	}
	if got.HTTP.Credentials == cred.HTTP.Credentials {
		t.Error("Copy() returned same Credentials pointer")
	}
	if diff := cmp.Diff(cred, got); diff != "" {
		t.Errorf("Copy() mismatch (-want +got):\n%s", diff)
	}
}

func TestAuthCredential_Copy_OAuth2(t *testing.T) {
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeOAuth2,
		OAuth2: &OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			AuthURI:      "https://example.com/auth",
			State:        "random-state",
			RedirectURI:  "https://example.com/callback",
			AccessToken:  "access-token",
			RefreshToken: "refresh-token",
			ExpiresAt:    1234567890,
		},
	}
	got := cred.Copy()

	if got == cred {
		t.Error("Copy() returned same pointer")
	}
	if got.OAuth2 == cred.OAuth2 {
		t.Error("Copy() returned same OAuth2 pointer")
	}
	if diff := cmp.Diff(cred, got); diff != "" {
		t.Errorf("Copy() mismatch (-want +got):\n%s", diff)
	}
}

func TestAuthCredential_Copy_ServiceAccount(t *testing.T) {
	cred := &AuthCredential{
		AuthType: AuthCredentialTypeServiceAccount,
		ServiceAccount: &ServiceAccount{
			Scopes:               []string{"scope1", "scope2"},
			UseDefaultCredential: true,
			ServiceAccountCredential: &ServiceAccountCredential{
				Type:        "service_account",
				ProjectID:   "my-project",
				PrivateKey:  "private-key-data",
				ClientEmail: "sa@example.iam.gserviceaccount.com",
			},
		},
	}
	got := cred.Copy()

	if got == cred {
		t.Error("Copy() returned same pointer")
	}
	if got.ServiceAccount == cred.ServiceAccount {
		t.Error("Copy() returned same ServiceAccount pointer")
	}
	if got.ServiceAccount.ServiceAccountCredential == cred.ServiceAccount.ServiceAccountCredential {
		t.Error("Copy() returned same ServiceAccountCredential pointer")
	}
	// Verify scopes are deep copied
	got.ServiceAccount.Scopes[0] = "modified"
	if cred.ServiceAccount.Scopes[0] == "modified" {
		t.Error("Copy() did not deep copy Scopes slice")
	}
}
