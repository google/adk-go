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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/adk/v2/auth"
)

// TestServiceAccountKeepsTheIDTokenExchangeOutOfDebugLogs:
// GOOGLE_SDK_GO_LOGGING_LEVEL is shared with the other Google client libraries
// in the process, so turning it on to debug one of them must not put the
// signed assertion and the minted ID token on stderr.
func TestServiceAccountKeepsTheIDTokenExchangeOutOfDebugLogs(t *testing.T) {
	const audience = "https://service.example.test"
	claims := fmt.Sprintf(`{"aud":%q,"exp":%d}`, audience, time.Now().Add(time.Hour).Unix())
	idToken := "eyJhbGciOiJSUzI1NiJ9." + base64.RawURLEncoding.EncodeToString([]byte(claims)) + ".c2ln"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id_token":%q,"token_type":"Bearer","expires_in":3600}`, idToken)
	}))
	defer srv.Close()
	key := idTokenServiceAccountKey(t, srv.URL)
	keyFile := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(keyFile, key, 0o600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		cfg  auth.ServiceAccountConfig
		// newAuthLib is GOOGLE_API_GO_EXPERIMENTAL_ENABLE_NEW_AUTH_LIB. An
		// explicit key moves google.golang.org/api onto cloud.google.com/go/auth,
		// the library that logs. Default credentials reach it only with the
		// variable set.
		newAuthLib string
	}{
		{name: "explicit key", cfg: auth.ServiceAccountConfig{JSONKey: key, Audience: audience}},
		{name: "default credentials", cfg: auth.ServiceAccountConfig{Audience: audience}, newAuthLib: "true"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_SDK_GO_LOGGING_LEVEL", "debug")
			t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", keyFile)
			t.Setenv("GOOGLE_API_GO_EXPERIMENTAL_ENABLE_NEW_AUTH_LIB", tc.newAuthLib)
			// Set on the host, this keeps both rows off that library, and they
			// pass with the leak in place.
			t.Setenv("GOOGLE_API_GO_EXPERIMENTAL_DISABLE_NEW_AUTH_LIB", "")

			stderr, err := os.CreateTemp(t.TempDir(), "stderr")
			if err != nil {
				t.Fatal(err)
			}
			saved := os.Stderr
			os.Stderr = stderr
			h := http.Header{}
			cred, err := auth.ServiceAccount(tc.cfg).Credential(t.Context())
			if err == nil {
				err = cred.Apply(h)
			}
			os.Stderr = saved
			if err != nil {
				t.Fatalf("minting an ID token: %v", err)
			}
			if got, want := h.Get("Authorization"), "Bearer "+idToken; got != want {
				t.Fatalf("Authorization = %q, want the minted ID token %q", got, want)
			}
			if err := stderr.Close(); err != nil {
				t.Fatal(err)
			}
			out, err := os.ReadFile(stderr.Name())
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 0 {
				t.Errorf("stderr with debug logging on = %q, want nothing from the auth library", out)
			}
		})
	}
}

// idTokenServiceAccountKey returns a service-account key file whose token
// requests go to tokenURI.
func idTokenServiceAccountKey(t *testing.T, tokenURI string) []byte {
	t.Helper()
	b, err := json.Marshal(map[string]string{
		"type":           "service_account",
		"project_id":     "p",
		"private_key_id": "key-id",
		"private_key":    auth.ServiceAccountTestKeyPEM(t),
		"client_email":   "sa@p.iam.gserviceaccount.com",
		"client_id":      "1",
		"token_uri":      tokenURI,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}
