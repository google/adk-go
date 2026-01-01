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

import "testing"

func TestServiceAccountCredentialsParse(t *testing.T) {
	jsonCred := []byte(`{
		"type": "service_account",
		"client_email": "service@example.com",
		"token_uri": "https://oauth2.example.com/token"
	}`)

	scheme, cred, err := ServiceAccountCredentials(jsonCred, []string{"scope1"})
	if err != nil {
		t.Fatalf("ServiceAccountCredentials() error = %v", err)
	}
	if scheme == nil {
		t.Fatal("scheme is nil")
	}
	if cred.ServiceAccount == nil || cred.ServiceAccount.ServiceAccountCredential == nil {
		t.Fatal("service account credential was not parsed")
	}
	if got := cred.ServiceAccount.ServiceAccountCredential.ClientEmail; got != "service@example.com" {
		t.Fatalf("client email = %s, want service@example.com", got)
	}
}

func TestServiceAccountCredentialsInvalidJSON(t *testing.T) {
	if _, _, err := ServiceAccountCredentials([]byte("{invalid json"), nil); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}
