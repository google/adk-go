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
	"strings"
	"testing"
)

// TestMergeRedactedValueCredentialIdentity checks that secrets are restored
// only inside the same credential: the same authType and the same visible
// principal.
func TestMergeRedactedValueCredentialIdentity(t *testing.T) {
	tests := []struct {
		name       string
		stored     any
		submitted  any
		wantSecret string
		wantNone   bool
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
			wantSecret: "AT-RESTORE",
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
			wantNone: true,
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
			wantSecret: "AT-CAMEL",
		},
		{
			name: "snake case stored secret still restores",
			stored: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app", "access_token": "AT-SNAKE-POSITIVE"},
			},
			submitted: map[string]any{
				"auth_type": "oauth2",
				"oauth2":    map[string]any{"client_id": "app"},
			},
			wantSecret: "AT-SNAKE-POSITIVE",
		},
		{
			name: "explicit null secret is a replacement",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": "AT-NULL"},
			},
			submitted: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app", "accessToken": nil},
			},
			wantNone: true,
		},
		{
			name:   "http username change is a replacement",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "bearer",
				"credentials": map[string]any{"username": "bob"},
			}},
			wantNone: true,
		},
		{
			name:   "http scheme change is a replacement",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "basic",
				"credentials": map[string]any{"username": "alice"},
			}},
			wantNone: true,
		},
		{
			name:   "unchanged http principal restores",
			stored: httpPasswordCredential("bearer", "alice", "HTTP-PASSWORD"),
			submitted: map[string]any{"authType": "http", "http": map[string]any{
				"scheme":      "bearer",
				"credentials": map[string]any{"username": "alice"},
			}},
			wantSecret: "HTTP-PASSWORD",
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
			wantNone: true,
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
			wantNone: true,
		},
		{
			name: "resource ref change is a replacement",
			stored: map[string]any{
				"authType":    "oauth2",
				"resourceRef": "projects/one",
				"oauth2":      map[string]any{"clientId": "app", "accessToken": "AT-RESOURCE"},
			},
			submitted: map[string]any{
				"authType":    "oauth2",
				"resourceRef": "projects/two",
				"oauth2":      map[string]any{"clientId": "app"},
			},
			wantNone: true,
		},
		{
			name: "changed non-string client id is a replacement",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": 123, "accessToken": "AT-NUMBER"},
			},
			submitted: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": 456},
			},
			wantNone: true,
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
			if tt.wantSecret != "" {
				if !restored {
					t.Error("MergeRedactedValue() did not report a restore")
				}
				if !bytes.Contains(encoded, []byte(tt.wantSecret)) {
					t.Error("merged value does not carry the restored secret")
				}
			}
			if tt.wantNone {
				if restored {
					t.Error("MergeRedactedValue() reported a restore, want none")
				}
				if bytes.Contains(encoded, []byte("AT-")) || bytes.Contains(encoded, []byte("HTTP-PASSWORD")) ||
					bytes.Contains(encoded, []byte("SA-PRIVATE-KEY")) {
					t.Error("merged value carries a secret, want it not restored")
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
			name: "reordered look-alike OAuth2 credentials",
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
			name: "prepended look-alike OAuth2 credential",
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
					t.Error("merged value carries a secret, want it not restored")
				}
			}
		})
	}
}

// TestMergeRedactedValueNestedCredentials checks credentials nested inside
// another credential. They are matched on their own, so a look-alike pair stays
// ambiguous, and a replaced credential blocks the restore inside it.
func TestMergeRedactedValueNestedCredentials(t *testing.T) {
	t.Run("nested look-alikes restore nothing", func(t *testing.T) {
		stored := map[string]any{"outer": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"clientId": "app", "accessToken": "OUTER-TOKEN"},
			"extras": []any{
				apiKeyCredential("NESTED-ONE"),
				apiKeyCredential("NESTED-TWO"),
			},
		}}
		submitted := map[string]any{"outer": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"clientId": "app"},
			"extras":   []any{redactedAPIKeyCredential(), redactedAPIKeyCredential()},
		}}
		merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
		if err != nil {
			t.Fatalf("MergeRedactedValue() failed: %v", err)
		}
		if !restored {
			t.Error("MergeRedactedValue() did not report the outer credential's restore")
		}
		encoded, err := json.Marshal(merged)
		if err != nil {
			t.Fatalf("Marshal() failed: %v", err)
		}
		if !bytes.Contains(encoded, []byte("OUTER-TOKEN")) {
			t.Error("the outer credential's secret was not restored")
		}
		for _, secret := range []string{"NESTED-ONE", "NESTED-TWO"} {
			if bytes.Contains(encoded, []byte(secret)) {
				t.Error("a nested look-alike credential received a secret")
			}
		}
	})

	t.Run("replaced credential does not restore inside it", func(t *testing.T) {
		stored := map[string]any{"outer": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"clientId": "app"},
			"extra":    apiKeyCredential("NESTED-ONE"),
		}}
		submitted := map[string]any{"outer": map[string]any{
			"authType": "oauth2",
			"oauth2":   map[string]any{"clientId": "other"},
			"extra":    redactedAPIKeyCredential(),
		}}
		merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
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
		if bytes.Contains(encoded, []byte("NESTED-ONE")) {
			t.Error("a nested credential received its secret under a replaced credential")
		}
	})
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
	creds, ok := merged.(map[string]any)["creds"].([]any)
	if !ok {
		t.Fatalf("merged creds has type %T, want []any", merged.(map[string]any)["creds"])
	}
	want := []string{"b", "a"}
	for i, item := range creds {
		oauth := item.(map[string]any)["oauth2"].(map[string]any)
		if oauth["clientId"] != want[i] {
			t.Errorf("credential %d has the wrong clientId", i)
		}
		wantToken := map[string]string{"a": "AT-A", "b": "AT-B"}[want[i]]
		if oauth["accessToken"] != wantToken {
			t.Errorf("credential %d has the wrong access token", i)
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
	creds, ok := merged.(map[string]any)["creds"].([]any)
	if !ok {
		t.Fatalf("merged creds has type %T, want []any", merged.(map[string]any)["creds"])
	}
	wantNames := []string{"renamed", "two"}
	wantTokens := []string{"AT-A", "AT-B"}
	for i, item := range creds {
		entry := item.(map[string]any)
		if entry["name"] != wantNames[i] {
			t.Errorf("entry %d has the wrong name", i)
		}
		oauth := entry["cred"].(map[string]any)["oauth2"].(map[string]any)
		if oauth["accessToken"] != wantTokens[i] {
			t.Errorf("entry %d has the wrong access token", i)
		}
	}
}

// TestMergeRedactedValueRestoresNothingOutsideCredentials covers the review
// finding that the walk used to start with restoring switched on, so secrets
// were filled in outside any credential: a pairing entry that is not a
// credential, a credential field without an authType, and a visible secret next
// to a credential.
func TestMergeRedactedValueRestoresNothingOutsideCredentials(t *testing.T) {
	t.Run("deleting the first of two non-credential entries", func(t *testing.T) {
		stored := map[string]any{"configs": []any{
			map[string]any{"name": "a", "rawAuthCredential": map[string]any{"apiKey": "AK-A"}},
			map[string]any{"name": "b", "rawAuthCredential": map[string]any{"apiKey": "AK-B"}},
		}}
		submitted := map[string]any{"configs": []any{
			map[string]any{"name": "b", "rawAuthCredential": map[string]any{}},
		}}
		merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
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
		for _, secret := range []string{"AK-A", "AK-B"} {
			if bytes.Contains(encoded, []byte(secret)) {
				t.Errorf("merged value carries %s, want it not restored", secret)
			}
		}
	})

	t.Run("changing a client id under a credential field without authType", func(t *testing.T) {
		stored := map[string]any{"auth": map[string]any{"rawAuthCredential": map[string]any{
			"oauth2": map[string]any{"clientId": "app", "accessToken": "AT-OLD"},
		}}}
		submitted := map[string]any{"auth": map[string]any{"rawAuthCredential": map[string]any{
			"oauth2": map[string]any{"clientId": "other"},
		}}}
		merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
		if err != nil {
			t.Fatalf("MergeRedactedValue() failed: %v", err)
		}
		if restored {
			t.Error("MergeRedactedValue() reported a restore, want none")
		}
		if encoded, _ := json.Marshal(merged); bytes.Contains(encoded, []byte("AT-OLD")) {
			t.Error("the old token was restored under the new client id")
		}
	})

	t.Run("deleting a visible secret next to a credential", func(t *testing.T) {
		stored := map[string]any{
			"drive":   oauth2Credential("app", "", "AT-DRIVE"),
			"profile": map[string]any{"name": "ann", "token": "page-2"},
		}
		submitted := map[string]any{
			"drive":   redactedOAuth2Credential("app", ""),
			"profile": map[string]any{"name": "ann"},
		}
		merged, _, err := MergeRedactedValue(context.Background(), stored, submitted)
		if err != nil {
			t.Fatalf("MergeRedactedValue() failed: %v", err)
		}
		encoded, err := json.Marshal(merged)
		if err != nil {
			t.Fatalf("Marshal() failed: %v", err)
		}
		if !bytes.Contains(encoded, []byte("AT-DRIVE")) {
			t.Error("the credential's token was not restored")
		}
		if bytes.Contains(encoded, []byte("page-2")) {
			t.Error("a visible secret outside the credential was restored")
		}
	})

	t.Run("a credential field without authType inside a credential", func(t *testing.T) {
		stored := map[string]any{
			"authType":          "oauth2",
			"oauth2":            map[string]any{"clientId": "app", "accessToken": "AT-TOP"},
			"rawAuthCredential": map[string]any{"apiKey": "AK-NESTED"},
		}
		submitted := map[string]any{
			"authType":          "oauth2",
			"oauth2":            map[string]any{"clientId": "app"},
			"rawAuthCredential": map[string]any{},
		}
		merged, _, err := MergeRedactedValue(context.Background(), stored, submitted)
		if err != nil {
			t.Fatalf("MergeRedactedValue() failed: %v", err)
		}
		encoded, err := json.Marshal(merged)
		if err != nil {
			t.Fatalf("Marshal() failed: %v", err)
		}
		if !bytes.Contains(encoded, []byte("AT-TOP")) {
			t.Error("the credential's own token was not restored")
		}
		if bytes.Contains(encoded, []byte("AK-NESTED")) {
			t.Error("a credential field without an authType had a secret restored")
		}
	})
}

// TestMergeRedactedValueNullSecretClears checks that a secret sent as JSON null
// counts as sent: the stored secret is replaced by the null, so a client can
// sign out as it does on main.
func TestMergeRedactedValueNullSecretClears(t *testing.T) {
	stored := map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "app", "accessToken": "AT-CLEAR", "refreshToken": "RT-CLEAR"},
	}
	submitted := map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": "app", "accessToken": nil},
	}
	merged, restored, err := MergeRedactedValue(context.Background(), stored, submitted)
	if err != nil {
		t.Fatalf("MergeRedactedValue() failed: %v", err)
	}
	if restored {
		t.Error("MergeRedactedValue() reported a restore, want none")
	}
	oauth, ok := merged.(map[string]any)["oauth2"].(map[string]any)
	if !ok {
		t.Fatalf("merged oauth2 has type %T, want map[string]any", merged.(map[string]any)["oauth2"])
	}
	if value, ok := oauth["accessToken"]; !ok || value != nil {
		t.Errorf("merged accessToken = %v (present %v), want an explicit null", value, ok)
	}
}

// TestMergeRedactedValueNestedCredentialWorkIsLinear checks that a chain of
// nested credentials costs linear work. Every level passes the identity checks,
// so the restore walk descends through all of them, and every credential is
// counted in the same pass that computes its secret flag.
func TestMergeRedactedValueNestedCredentialWorkIsLinear(t *testing.T) {
	const depth = 2000
	build := func() any {
		value := any(map[string]any{"x": 1})
		for i := range depth {
			value = map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": fmt.Sprintf("app-%d", i)},
				"inner":    value,
			}
		}
		return value
	}
	stored := build()
	submitted := build()
	_, _, visited, err := mergeRedactedValue(context.Background(), stored, submitted)
	if err != nil {
		t.Fatalf("mergeRedactedValue() failed: %v", err)
	}
	// A linear merge visits each node a small constant number of times. The
	// broken version walked every credential's whole subtree at every level.
	if limit := 10 * (depth * 3); visited > limit {
		t.Errorf("mergeRedactedValue() visited %d nodes, want at most %d", visited, limit)
	}
}

// TestCredentialIdentityOfSpellings checks that the visible principal matches
// across the camelCase and snake_case spellings, and takes part in the identity
// when it changes or is not a string.
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
		{
			name: "non-string principal takes part in the identity",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "123"},
			},
			submitted: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": 123},
			},
			wantEqual: false,
		},
		{
			// The authType spelling that is not a credential type must not stand
			// in for the stored one: "oauth_2" normalizes onto "oauth2", so only
			// a type check keeps it from matching a stored OAuth2 credential.
			name: "unknown authType spelling is not the stored type",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": "app"},
			},
			submitted: map[string]any{
				"authType":  "oauth_2",
				"auth_type": "http",
				"http":      map[string]any{"scheme": "bearer", "credentials": map[string]any{"username": "app"}},
			},
			wantEqual: false,
		},
		{
			name: "object principal is not read",
			stored: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": map[string]any{"nested": "one"}},
			},
			submitted: map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": map[string]any{"nested": "two"}},
			},
			wantEqual: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storedIdentity := credentialIdentityOf(tt.stored)
			submittedIdentity := credentialIdentityOf(tt.submitted)
			if storedIdentity == "" {
				t.Fatal("credentialIdentityOf() = \"\", want an identity")
			}
			if got := storedIdentity == submittedIdentity; got != tt.wantEqual {
				t.Errorf("credentialIdentityOf() equal = %v, want %v", got, tt.wantEqual)
			}
		})
	}
}

// TestCredentialIdentityOfDoesNotReadNonScalarPrincipal checks that a principal
// field holding a large object is not encoded into the identity: doing that is
// what let one request allocate depth times body size.
func TestCredentialIdentityOfDoesNotReadNonScalarPrincipal(t *testing.T) {
	payload := strings.Repeat("x", 200_000)
	credential := map[string]any{
		"authType": "oauth2",
		"oauth2":   map[string]any{"clientId": map[string]any{"nested": payload}},
	}
	identity := credentialIdentityOf(credential)
	if len(identity) > 100 {
		t.Errorf("credentialIdentityOf() returned %d bytes, want the non-scalar principal left out", len(identity))
	}
	if strings.Contains(identity, payload) {
		t.Error("credentialIdentityOf() encoded the non-scalar principal")
	}
}

// TestMergeRedactedValueCanceledContext checks a merge that starts with a
// canceled context.
func TestMergeRedactedValueCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := MergeRedactedValue(ctx, map[string]any{"creds": []any{apiKeyCredential("API-KEY")}}, map[string]any{"creds": []any{redactedAPIKeyCredential()}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("MergeRedactedValue() error = %v, want context.Canceled", err)
	}
}

// secondErrContext reports a cancellation from its second Err call on, so a
// merge can be canceled after it has started walking.
type secondErrContext struct {
	context.Context
	calls int
}

func (c *secondErrContext) Err() error {
	c.calls++
	if c.calls > 1 {
		return context.Canceled
	}
	return nil
}

// TestMergeRedactedValueCancellationDuringWalk checks that a merge notices a
// cancellation after it has started, not only before it.
func TestMergeRedactedValueCancellationDuringWalk(t *testing.T) {
	ctx := &secondErrContext{Context: context.Background()}
	stored := make([]any, 0, 2001)
	for i := range 2000 {
		stored = append(stored, map[string]any{"n": fmt.Sprint(i)})
	}
	stored = append(stored, apiKeyCredential("API-KEY"))
	submitted := append([]any{}, stored[:len(stored)-1]...)
	submitted = append(submitted, redactedAPIKeyCredential())
	_, _, err := MergeRedactedValue(ctx, map[string]any{"creds": stored}, map[string]any{"creds": submitted})
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

// BenchmarkMergeRedactedValueDeepNesting measures a chain of nested
// credentials. Every level is a credential with its own identity, and its
// identity and secret flag are computed once per request, so the depth should
// not multiply the work.
func BenchmarkMergeRedactedValueDeepNesting(b *testing.B) {
	build := func(depth int) any {
		wide := make([]any, 0, 20000)
		for range 19999 {
			wide = append(wide, map[string]any{"x": 1})
		}
		var value any = wide
		for i := range depth {
			value = map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"clientId": fmt.Sprintf("app-%d", i)},
				"inner":    value,
			}
		}
		return value
	}
	for _, depth := range []int{500, 2000} {
		storedValue := build(depth)
		submittedValue := build(depth)
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
