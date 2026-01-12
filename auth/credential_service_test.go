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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestInMemoryCredentialService_SaveAndLoad(t *testing.T) {
	svc := NewInMemoryCredentialService()
	ctx := context.Background()

	cfg := &AuthConfig{
		CredentialKey: "test-key",
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AccessToken:  "access-token",
				RefreshToken: "refresh-token",
			},
		},
	}

	// Save
	if err := svc.SaveCredential(ctx, cfg); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}

	// Load
	loaded, err := svc.LoadCredential(ctx, cfg)
	if err != nil {
		t.Fatalf("LoadCredential() error = %v", err)
	}
	if loaded == nil {
		t.Fatal("LoadCredential() returned nil")
	}
	if diff := cmp.Diff(cfg.ExchangedAuthCredential, loaded); diff != "" {
		t.Errorf("LoadCredential() mismatch (-want +got):\n%s", diff)
	}
}

func TestInMemoryCredentialService_LoadCredential_NotFound(t *testing.T) {
	svc := NewInMemoryCredentialService()
	ctx := context.Background()

	cfg := &AuthConfig{
		CredentialKey: "non-existent-key",
	}

	loaded, err := svc.LoadCredential(ctx, cfg)
	if err != nil {
		t.Fatalf("LoadCredential() error = %v", err)
	}
	if loaded != nil {
		t.Errorf("LoadCredential() = %v, want nil for non-existent key", loaded)
	}
}

// Note: SaveCredential requires non-nil config - passing nil will panic.

func TestInMemoryCredentialService_Overwrite(t *testing.T) {
	svc := NewInMemoryCredentialService()
	ctx := context.Background()

	cfg := &AuthConfig{
		CredentialKey: "test-key",
		ExchangedAuthCredential: &AuthCredential{
			AuthType: AuthCredentialTypeOAuth2,
			OAuth2: &OAuth2Auth{
				AccessToken: "token-1",
			},
		},
	}

	// Save first credential
	if err := svc.SaveCredential(ctx, cfg); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}

	// Overwrite with new credential
	cfg.ExchangedAuthCredential.OAuth2.AccessToken = "token-2"
	if err := svc.SaveCredential(ctx, cfg); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}

	// Load should return new credential
	loaded, err := svc.LoadCredential(ctx, cfg)
	if err != nil {
		t.Fatalf("LoadCredential() error = %v", err)
	}
	if loaded.OAuth2.AccessToken != "token-2" {
		t.Errorf("AccessToken = %q, want %q", loaded.OAuth2.AccessToken, "token-2")
	}
}
