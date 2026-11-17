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

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
)

// TestMergeRedactedValueCredentialIdentity checks that secrets are restored
// only into the same credential: the same authType and the same visible
// principal.
func TestMergeRedactedValueCredentialIdentity(t *testing.T) {
	tests := []struct {
		name      string
		stored    any
		submitted any
		want      []string
		wantNone  []string
	}{
		{
			name: "authType spelling still restores",
			stored: map[string]any{"drive": map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": "AT-RESTORE"},
			}},
			submitted: map[string]any{"drive": map[string]any{
				"authType": "OAUTH2",
				"oauth2":   map[string]any{"clientId": "app"},
			}},
			want: []string{"AT-RESTORE"},
		},
		{
			name: "snake case client id change is a replacement",
			stored: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app", "access_token": "AT-SNAKE"},
			},
			submitted: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "other"},
			},
			wantNone: []string{"AT-SNAKE"},
		},
		{
			name: "camel and snake client ids are the same principal",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": "AT-CAMEL"},
			},
			submitted: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app"},
			},
			want: []string{"AT-CAMEL"},
		},
		{
			name:   "http username change is a replacement",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "bearer",
				"credentials": map[string]any{"username": "bob"},
			}},
			wantNone: []string{"HTTP-PASSWORD"},
		},
		{
			name:   "http scheme change is a replacement",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "basic",
				"credentials": map[string]any{"username": "alice"},
			}},
			wantNone: []string{"HTTP-PASSWORD"},
		},
		{
			name:   "unchanged http principal restores",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "bearer",
				"credentials": map[string]any{"username": "alice"},
			}},
			want: []string{"HTTP-PASSWORD"},
		},
		{
			name:   "service account email change is a replacement",
			stored: serviceAccountCredential("alice@proj.iam.gserviceaccount.com", "proj", "SA-PRIVATE-KEY"),
			submitted: map[string]any{"authType": "serviceAccount", "serviceAccount": map[string]any{
				"serviceAccountCredential": map[string]any{
					"clientEmail": "bob@proj.iam.gserviceaccount.com",
					"projectId":   "proj",
				},
			}},
			wantNone: []string{"SA-PRIVATE-KEY"},
		},
		{
			name:   "service account project id change is a replacement",
			stored: serviceAccountCredential("alice@proj.iam.gserviceaccount.com", "proj", "SA-PRIVATE-KEY"),
			submitted: map[string]any{"authType": "serviceAccount", "serviceAccount": map[string]any{
				"serviceAccountCredential": map[string]any{
					"clientEmail": "alice@proj.iam.gserviceaccount.com",
					"projectId":   "other",
				},
			}},
			wantNone: []string{"SA-PRIVATE-KEY"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged, restored, err := MergeRedactedValue(context.Background(), tt.stored, tt.submitted)
			if err != nil {
				t.Fatalf("MergeRedactedValue() failed: %v", err)
			}
			encoded, err := json.Marshal(merged)
			if err != nil {
				t.Fatalf("Marshal() failed: %v", err)
			}
			if len(tt.want) > 0 && !restored {
				t.Error("MergeRedactedValue() did not report a restore")
			}
			for _, secret := range tt.want {
				if !bytes.Contains(encoded, []byte(secret)) {
					t.Errorf("merged value does not carry %q, want it restored", secret)
				}
			}
			for _, secret := range tt.wantNone {
				if restored {
					t.Error("MergeRedactedValue() reported a restore, want none")
				}
				if bytes.Contains(encoded, []byte(secret)) {
					t.Errorf("merged value carries %q, want it not restored", secret)
				}
			}
		})
	}
}

// TestMergeRedactedValueAmbiguousIdentities checks that an identity appearing
// more than once on either side restores nothing, because the match would be a
// guess.
func TestMergeRedactedValueAmbiguousIdentities(t *testing.T) {
	tests := []struct {
		name      string
		stored    any
		submitted any
		secrets   []string
	}{
		{
			name: "removing the first of two API keys",
			stored: map[string]any{"creds": []any{
				apiKeyCredential("API-KEY-ONE"),
				apiKeyCredential("API-KEY-TWO"),
			}},
			submitted: map[string]any{"creds": []any{redactedAPIKeyCredential()}},
			secrets:   []string{"API-KEY-ONE", "API-KEY-TWO"},
		},
		{
			name:      "duplicated submitted API key",
			stored:    map[string]any{"creds": []any{apiKeyCredential("API-KEY-ONE")}},
			submitted: map[string]any{"creds": []any{redactedAPIKeyCredential(), redactedAPIKeyCredential()}},
			secrets:   []string{"API-KEY-ONE"},
		},
		{
			name: "reordered lookalike OAuth2 credentials",
			stored: map[string]any{"creds": []any{
				oauth2Credential("app", "https://one.example/cb", "AT-ONE"),
				oauth2Credential("app", "https://two.example/cb", "AT-TWO"),
			}},
			submitted: map[string]any{"creds": []any{
				redactedOAuth2Credential("app", "https://two.example/cb"),
				redactedOAuth2Credential("app", "https://one.example/cb"),
			}},
			secrets: []string{"AT-ONE", "AT-TWO"},
		},
		{
			name: "prepended lookalike OAuth2 credential",
			stored: map[string]any{"creds": []any{
				oauth2Credential("app", "https://one.example/cb", "AT-ONE"),
				oauth2Credential("app", "https://two.example/cb", "AT-TWO"),
			}},
			submitted: map[string]any{"creds": []any{
				redactedOAuth2Credential("app", "https://new.example/cb"),
				redactedOAuth2Credential("app", "https://one.example/cb"),
				redactedOAuth2Credential("app", "https://two.example/cb"),
			}},
			secrets: []string{"AT-ONE", "AT-TWO"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged, restored, err := MergeRedactedValue(context.Background(), tt.stored, tt.submitted)
			if err != nil {
				t.Fatalf("MergeRedactedValue() failed: %v", err)
			}
			if restored {
				t.Error("MergeRedactedValue() reported a restore, want none")
			}
			encoded, err := json.Marshal(merged)
			if err != nil {
				t.Fatalf("Marshal() failed: %v", err)
			}
			for _, secret := range tt.secrets {
				if bytes.Contains(encoded, []byte(secret)) {
					t.Errorf("merged value carries %q, want it not restored", secret)
				}
			}
		})
	}
}

// TestMergeRedactedValueUniqueArrayIdentities checks that an array of
// credentials with unique identities still restores into an entry whose
// neighboring field changed, and that reordering keeps each credential's own
// secrets.
func TestMergeRedactedValueUniqueArrayIdentities(t *testing.T) {
	stored := map[string]any{"creds": []any{
		oauth2Credential("a", "", "AT-A"),
		oauth2Credential("b", "", "AT-B"),
	}}
	submitted := map[string]any{"creds": []any{
		redactedOAuth2Credential("b", ""),
		redactedOAuth2Credential("a", ""),
	}}
	merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
	if err != nil {
		t.Fatalf("MergeRedactedValue() failed: %v", err)
	}
	if !restored {
		t.Fatal("MergeRedactedValue() did not report a restore")
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	for _, want := range []string{`"clientId":"a"`, `"accessToken":"AT-A"`, `"clientId":"b"`, `"accessToken":"AT-B"`} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Errorf("merged value does not carry %s after a reorder", want)
		}
	}
}

// TestMergeRedactedValueObjectArraySiblingEdit checks an array of objects that
// each hold a credential: editing a sibling field inside one object keeps that
// object's own credential secrets.
func TestMergeRedactedValueObjectArraySiblingEdit(t *testing.T) {
	stored := map[string]any{"creds": []any{
		map[string]any{"name": "one", "cred": oauth2Credential("a", "", "AT-A")},
		map[string]any{"name": "two", "cred": oauth2Credential("b", "", "AT-B")},
	}}
	submitted := map[string]any{"creds": []any{
		map[string]any{"name": "renamed", "cred": redactedOAuth2Credential("a", "")},
		map[string]any{"name": "two", "cred": redactedOAuth2Credential("b", "")},
	}}
	merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
	if err != nil {
		t.Fatalf("MergeRedactedValue() failed: %v", err)
	}
	if !restored {
		t.Fatal("MergeRedactedValue() did not report a restore")
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	for _, want := range []string{`"name":"renamed"`, `"accessToken":"AT-A"`, `"accessToken":"AT-B"`} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Errorf("merged value does not carry %s", want)
		}
	}
}

// TestCredentialIdentityOfSpellings checks that the visible principal matches
// across the camelCase and snake_case spellings.
func TestCredentialIdentityOfSpellings(t *testing.T) {
	tests := []struct {
		name      string
		stored    map[string]any
		submitted map[string]any
		wantEqual bool
	}{
		{
			name:      "service account in both spellings",
			stored:    serviceAccountCredential("alice@proj.iam.gserviceaccount.com", "proj", "SA-PRIVATE-KEY"),
			submitted: snakeCaseServiceAccountCredential("alice@proj.iam.gserviceaccount.com", "proj"),
			wantEqual: true,
		},
		{
			name:      "http scheme is part of the identity",
			stored:    httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: httpPasswordCredential("basic", "alice", "HTTP-PASSWORD"),
			wantEqual: false,
		},
		{
			name:      "http username is part of the identity",
			stored:    httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: httpPasswordCredential("bearer", "bob", "HTTP-PASSWORD"),
			wantEqual: false,
		},
		{
			name:      "api key has no visible principal",
			stored:    apiKeyCredential("API-KEY-ONE"),
			submitted: apiKeyCredential("API-KEY-TWO"),
			wantEqual: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &restoreWalk{ctx: context.Background()}
			if got := credentialIdentityOf(w, tt.stored) == credentialIdentityOf(w, tt.submitted); got != tt.wantEqual {
				t.Errorf("credentialIdentityOf() equal = %v, want %v", got, tt.wantEqual)
			}
			if w.err != nil {
				t.Fatalf("credentialIdentityOf() walk failed: %v", w.err)
			}
		})
	}
}

// TestMergeRedactedValueCanceledContext checks that a canceled request stops
// the merge instead of walking a large value to the end.
func TestMergeRedactedValueCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := MergeRedactedValue(ctx, map[string]any{"creds": []any{apiKeyCredential("API-KEY")}}, map[string]any{"creds": []any{redactedAPIKeyCredential()}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("MergeRedactedValue() error = %v, want context.Canceled", err)
	}
}

// BenchmarkMergeRedactedValueLookalikes measures the merge of 2n look-alike
// submitted entries against n stored ones. They are all one identity, so
// nothing is restored and the work should stay linear.
func BenchmarkMergeRedactedValueLookalikes(b *testing.B) {
	for _, n := range []int{5000, 20000} {
		stored := make([]any, 0, n)
		for range n {
			stored = append(stored, redactedAPIKeyCredential())
		}
		submitted := make([]any, 0, 2*n)
		for range 2 * n {
			submitted = append(submitted, redactedAPIKeyCredential())
		}
		storedValue := map[string]any{"creds": stored}
		submittedValue := map[string]any{"creds": submitted}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := MergeRedactedValue(context.Background(), storedValue, submittedValue); err != nil {
					b.Fatalf("MergeRedactedValue() failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkMergeRedactedValueDeepNesting measures a value nested in
// single-entry arrays around a wide array. Identity is only computed for array
// entries that are themselves credentials, so the depth should not multiply
// the work.
func BenchmarkMergeRedactedValueDeepNesting(b *testing.B) {
	for _, depth := range []int{500, 2000} {
		inner := make([]any, 0, 20000)
		for range 19999 {
			inner = append(inner, map[string]any{"x": 1})
		}
		// A credential without secrets: matching it changes nothing, so the
		// benchmark measures the walk itself.
		inner = append(inner, oauth2Credential("app", "", ""))
		var storedValue, submittedValue any = inner, inner
		for range depth {
			storedValue = []any{storedValue}
			submittedValue = []any{submittedValue}
		}
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := MergeRedactedValue(context.Background(), storedValue, submittedValue); err != nil {
					b.Fatalf("MergeRedactedValue() failed: %v", err)
				}
			}
		})
	}
}

func apiKeyCredential(secret string) map[string]any {
	return map[string]any{"authType": "apiKey", "apiKey": secret}
}

func redactedAPIKeyCredential() map[string]any {
	return map[string]any{"authType": "apiKey"}
}

func oauth2Credential(clientID, redirectURI, accessToken string) map[string]any {
	oauth2 := map[string]any{"clientId": clientID}
	if redirectURI != "" {
		oauth2["redirectUri"] = redirectURI
	}
	if accessToken != "" {
		oauth2["accessToken"] = accessToken
	}
	return map[string]any{"authType": "oauth2", "oauth2": oauth2}
}

func redactedOAuth2Credential(clientID, redirectURI string) map[string]any {
	return oauth2Credential(clientID, redirectURI, "")
}

func httpPasswordCredential(scheme, username, password string) map[string]any {
	return map[string]any{"authType": "http", "http": map[string]any{
		"scheme":      scheme,
		"credentials": map[string]any{"username": username, "password": password},
	}}
}

func serviceAccountCredential(clientEmail, projectID, privateKey string) map[string]any {
	return map[string]any{"authType": "serviceAccount", "serviceAccount": map[string]any{
		"serviceAccountCredential": map[string]any{
			"clientEmail": clientEmail,
			"projectId":   projectID,
			"privateKey":  privateKey,
		},
	}}
}

func snakeCaseServiceAccountCredential(clientEmail, projectID string) map[string]any {
	return map[string]any{"auth_type": "service_account", "service_account": map[string]any{
		"service_account_credential": map[string]any{
			"client_email": clientEmail,
			"project_id":   projectID,
		},
	}}
}
