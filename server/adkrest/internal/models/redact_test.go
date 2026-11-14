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

package models

import "testing"

func TestMergeRedactedValueCredentialIdentity(t *testing.T) {
	tests := []struct {
		name      string
		stored    map[string]any
		submitted map[string]any
		wantToken string
	}{
		{
			name: "authType spelling still restores",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": "at"},
			},
			submitted: map[string]any{
				"authType": "OAUTH2",
				"oauth2":   map[string]any{"clientId": "app"},
			},
			wantToken: "at",
		},
		{
			name: "snake case client id change is a replacement",
			stored: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app", "access_token": "at"},
			},
			submitted: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "other"},
			},
		},
		{
			name: "camel and snake client ids are the same principal",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": "at"},
			},
			submitted: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app"},
			},
			wantToken: "at",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged, restored, err := MergeRedactedValue(tt.stored, tt.submitted)
			if err != nil {
				t.Fatalf("MergeRedactedValue() failed: %v", err)
			}
			oauth := merged.(map[string]any)["oauth2"].(map[string]any)
			if tt.wantToken == "" {
				if restored {
					t.Error("MergeRedactedValue() restored a secret, want none")
				}
				if _, ok := oauth["accessToken"]; ok {
					t.Error("merged value has an access token, want none")
				}
				return
			}
			if !restored {
				t.Error("MergeRedactedValue() did not report a restore")
			}
			if got := oauth["accessToken"]; got != tt.wantToken {
				t.Errorf("accessToken = %v, want %q", got, tt.wantToken)
			}
		})
	}
}

func TestCredentialIdentityStableWithMultipleCredentials(t *testing.T) {
	v := map[string]any{
		"a": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"clientId": "a"},
		},
		"b": map[string]any{
			"authType": "http",
			"http":     map[string]any{"scheme": "bearer"},
		},
	}
	want := credentialIdentity(v)
	if want == "" {
		t.Fatal("credentialIdentity() = \"\", want an identity")
	}
	for range 100 {
		if got := credentialIdentity(v); got != want {
			t.Fatal("credentialIdentity() changed between calls")
		}
	}
}
