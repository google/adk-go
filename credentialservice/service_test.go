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

package credentialservice_test

import (
	"fmt"
	"sync"
	"testing"

	"google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/tool/toolauth"
)

// oauthConfig returns a config stored under key whose exchanged credential is
// an OAuth2 client with clientID.
func oauthConfig(key, clientID string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		CredentialKey: key,
		ExchangedAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: clientID, ClientSecret: "secret"},
		},
	}
}

func save(t *testing.T, svc credentialservice.Service, appName, userID string, cfg *toolauth.Config) {
	t.Helper()
	if err := svc.SaveCredential(t.Context(), appName, userID, cfg); err != nil {
		t.Fatalf("SaveCredential(%q, %q) error = %v", appName, userID, err)
	}
}

func load(t *testing.T, svc credentialservice.Service, appName, userID string, cfg *toolauth.Config) *toolauth.Credential {
	t.Helper()
	got, err := svc.LoadCredential(t.Context(), appName, userID, cfg)
	if err != nil {
		t.Fatalf("LoadCredential(%q, %q) error = %v", appName, userID, err)
	}
	return got
}

// clientID returns the OAuth2 client ID of c, or "<none>" when c is nil.
func clientID(c *toolauth.Credential) string {
	if c == nil {
		return "<none>"
	}
	return c.OAuth2.ClientID
}

func TestInMemoryServicePreventsAliasing(t *testing.T) {
	t.Run("caller changes the saved credential", func(t *testing.T) {
		svc := credentialservice.InMemoryService()
		cfg := oauthConfig("key-a", "client-a")
		cfg.ExchangedAuthCredential.Extra = map[string]any{"note": "orig"}
		save(t, svc, "app-1", "user-1", cfg)

		cfg.ExchangedAuthCredential.OAuth2.ClientID = "mutated"
		cfg.ExchangedAuthCredential.Extra["note"] = "mutated"

		got := load(t, svc, "app-1", "user-1", cfg)
		if clientID(got) != "client-a" {
			t.Fatalf("loaded ClientID = %q, want %q", clientID(got), "client-a")
		}
		if got.Extra["note"] != "orig" {
			t.Errorf("loaded Extra[note] = %v, want orig", got.Extra["note"])
		}
	})

	t.Run("caller changes a loaded credential", func(t *testing.T) {
		svc := credentialservice.InMemoryService()
		cfg := oauthConfig("key-a", "client-a")
		save(t, svc, "app-1", "user-1", cfg)

		first := load(t, svc, "app-1", "user-1", cfg)
		if first == nil {
			t.Fatal("LoadCredential() = nil, want the saved credential")
		}
		first.OAuth2.ClientID = "mutated"

		second := load(t, svc, "app-1", "user-1", cfg)
		if clientID(second) != "client-a" {
			t.Errorf("second load ClientID = %q, want %q", clientID(second), "client-a")
		}
		if second == first {
			t.Error("two loads returned the same pointer, want a copy each")
		}
	})
}

func TestInMemoryServiceSaveWithoutExchangedCredentialErases(t *testing.T) {
	erase := &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2}, CredentialKey: "key-a"}

	t.Run("saved credential", func(t *testing.T) {
		svc := credentialservice.InMemoryService()
		cfgA := oauthConfig("key-a", "client-a")
		cfgB := oauthConfig("key-b", "client-b")
		save(t, svc, "app-1", "user-1", cfgA)
		save(t, svc, "app-1", "user-1", cfgB)

		save(t, svc, "app-1", "user-1", erase)

		if got := load(t, svc, "app-1", "user-1", cfgA); got != nil {
			t.Errorf("load of the erased key = %+v, want nil", got)
		}
		if got := load(t, svc, "app-1", "user-1", cfgB); clientID(got) != "client-b" {
			t.Errorf("load of the other key ClientID = %q, want %q", clientID(got), "client-b")
		}
	})

	t.Run("fresh service", func(t *testing.T) {
		svc := credentialservice.InMemoryService()
		save(t, svc, "app-1", "user-1", erase)
		if got := load(t, svc, "app-1", "user-1", erase); got != nil {
			t.Errorf("load after saving no credential = %+v, want nil", got)
		}
	})
}

func TestInMemoryServiceBucketsDoNotCollide(t *testing.T) {
	tests := []struct {
		name              string
		saveApp, saveUser string
		loadApp, loadUser string
		wantClientID      string
	}{
		{name: "other app, same user", saveApp: "app-1", saveUser: "user-1", loadApp: "app-2", loadUser: "user-1", wantClientID: "<none>"},
		{name: "same app, other user", saveApp: "app-1", saveUser: "user-1", loadApp: "app-1", loadUser: "user-2", wantClientID: "<none>"},
		{name: "split moved between app and user", saveApp: "ab", saveUser: "c", loadApp: "a", loadUser: "bc", wantClientID: "<none>"},
		{name: "same app and user", saveApp: "ab", saveUser: "c", loadApp: "ab", loadUser: "c", wantClientID: "client-a"},
		{name: "empty app and user swapped", saveApp: "", saveUser: "u", loadApp: "u", loadUser: "", wantClientID: "<none>"},
		{name: "empty app, same user", saveApp: "", saveUser: "u", loadApp: "", loadUser: "u", wantClientID: "client-a"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := credentialservice.InMemoryService()
			cfg := oauthConfig("key-a", "client-a")
			save(t, svc, tc.saveApp, tc.saveUser, cfg)
			if got := load(t, svc, tc.loadApp, tc.loadUser, cfg); clientID(got) != tc.wantClientID {
				t.Errorf("LoadCredential(%q, %q) ClientID = %q, want %q", tc.loadApp, tc.loadUser, clientID(got), tc.wantClientID)
			}
		})
	}
}

func TestInMemoryServiceConcurrentSaveLoad(t *testing.T) {
	const goroutines, iterations = 32, 50
	svc := credentialservice.InMemoryService()
	shared := oauthConfig("key-shared", "client-shared")

	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Add(2)
		go func() {
			defer wg.Done()
			cfg := oauthConfig(fmt.Sprintf("key-%d", i), fmt.Sprintf("client-%d", i))
			if err := svc.SaveCredential(t.Context(), "app-1", "user-1", cfg); err != nil {
				t.Errorf("SaveCredential(key-%d) error = %v", i, err)
				return
			}
			for range iterations {
				if _, err := svc.LoadCredential(t.Context(), "app-1", "user-1", cfg); err != nil {
					t.Errorf("LoadCredential(key-%d) error = %v", i, err)
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			for range iterations {
				if err := svc.SaveCredential(t.Context(), "app-1", "user-1", shared); err != nil {
					t.Errorf("SaveCredential(key-shared) error = %v", err)
					return
				}
				if _, err := svc.LoadCredential(t.Context(), "app-1", "user-1", shared); err != nil {
					t.Errorf("LoadCredential(key-shared) error = %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	for i := range goroutines {
		cfg := oauthConfig(fmt.Sprintf("key-%d", i), "")
		want := fmt.Sprintf("client-%d", i)
		if got := load(t, svc, "app-1", "user-1", cfg); clientID(got) != want {
			t.Errorf("key-%d ClientID = %q, want %q", i, clientID(got), want)
		}
	}
	if got := load(t, svc, "app-1", "user-1", shared); clientID(got) != "client-shared" {
		t.Errorf("key-shared ClientID = %q, want %q", clientID(got), "client-shared")
	}
}

func TestInMemoryServiceKeyError(t *testing.T) {
	// A channel in an Extra map cannot be encoded as JSON, so Key fails.
	bad := &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey},
		RawAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "c", Extra: map[string]any{"bad": make(chan int)}},
		},
		ExchangedAuthCredential: oauthConfig("", "client-a").ExchangedAuthCredential,
	}
	svc := credentialservice.InMemoryService()

	if err := svc.SaveCredential(t.Context(), "app-1", "user-1", bad); err == nil {
		t.Error("SaveCredential() error = nil, want the error of Config.Key")
	}
	got, err := svc.LoadCredential(t.Context(), "app-1", "user-1", bad)
	if err == nil {
		t.Error("LoadCredential() error = nil, want the error of Config.Key")
	}
	if got != nil {
		t.Errorf("LoadCredential() = %+v, want nil", got)
	}
}

func TestInMemoryServiceNilConfig(t *testing.T) {
	svc := credentialservice.InMemoryService()
	t.Run("save", func(t *testing.T) {
		if err := svc.SaveCredential(t.Context(), "app-1", "user-1", nil); err == nil {
			t.Error("SaveCredential(nil config) error = nil, want an error")
		}
	})
	t.Run("load", func(t *testing.T) {
		got, err := svc.LoadCredential(t.Context(), "app-1", "user-1", nil)
		if err == nil {
			t.Error("LoadCredential(nil config) error = nil, want an error")
		}
		if got != nil {
			t.Errorf("LoadCredential(nil config) = %+v, want nil", got)
		}
	})
}
