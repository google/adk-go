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
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestIdentityAwareProxyStripsPrefix checks that IdentityAwareProxy removes the
// "accounts.google.com:" namespace prefix IAP prepends to both header values,
// so the resolved UserID and email match what a REST path carries and an
// authorizer compares against.
func TestIdentityAwareProxyStripsPrefix(t *testing.T) {
	const prefix = "accounts.google.com:"

	tests := []struct {
		name string
		// id and email are the raw header values; an empty string means the
		// header is not set.
		id        string
		email     string
		wantUser  string
		wantEmail string
		wantErr   bool
	}{
		{
			name:      "both prefixed values are stripped",
			id:        prefix + "108100000000000000000",
			email:     prefix + "alice@example.com",
			wantUser:  "108100000000000000000",
			wantEmail: "alice@example.com",
		},
		{
			name:      "only the first prefix occurrence is stripped",
			id:        prefix + "108",
			email:     prefix + prefix + "weird@example.com",
			wantUser:  "108",
			wantEmail: prefix + "weird@example.com",
		},
		{
			name:    "id without the prefix is rejected",
			id:      "108100000000000000000",
			email:   prefix + "alice@example.com",
			wantErr: true,
		},
		{
			name:    "email without the prefix is rejected",
			id:      prefix + "108100000000000000000",
			email:   "alice@example.com",
			wantErr: true,
		},
		{
			name:    "missing id header is unauthenticated",
			email:   prefix + "alice@example.com",
			wantErr: true,
		},
		{
			name:    "missing email header is unauthenticated",
			id:      prefix + "108100000000000000000",
			wantErr: true,
		},
		{
			name:    "both headers missing is unauthenticated",
			wantErr: true,
		},
	}

	iap := &identityAwareProxy{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.id != "" {
				r.Header.Set(UserIDHeader, tt.id)
			}
			if tt.email != "" {
				r.Header.Set(UserEMailHeader, tt.email)
			}

			id, err := iap.Authenticate(r)
			if tt.wantErr {
				if !errors.Is(err, ErrUnauthenticated) {
					t.Fatalf("Authenticate() error = %v, want it to wrap ErrUnauthenticated", err)
				}
				if id != nil {
					t.Errorf("Authenticate() identity = %+v, want nil on error", id)
				}
				return
			}
			if err != nil {
				t.Fatalf("Authenticate() error = %v, want nil", err)
			}
			if got := id.UserID; got != tt.wantUser {
				t.Errorf("Caller.UserID = %q, want %q", got, tt.wantUser)
			}
			if got := id.Claims["email"]; got != tt.wantEmail {
				t.Errorf("Claims[email] = %v, want %q", got, tt.wantEmail)
			}
		})
	}
}
