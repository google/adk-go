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

package triggerauth

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/server/authn"
)

// captureNewGoogleOIDC records every config Authenticator hands to
// authn.NewGoogleOIDC for the rest of the test.
func captureNewGoogleOIDC(t *testing.T) *[]authn.GoogleOIDCConfig {
	t.Helper()
	var got []authn.GoogleOIDCConfig
	orig := newGoogleOIDC
	t.Cleanup(func() { newGoogleOIDC = orig })
	newGoogleOIDC = func(cfg authn.GoogleOIDCConfig) (authn.Authenticator, error) {
		got = append(got, cfg)
		return orig(cfg)
	}
	return &got
}

func TestAuthenticatorHandsFlagValuesToNewGoogleOIDC(t *testing.T) {
	for _, tc := range []struct {
		audience, serviceAccounts string
		want                      authn.GoogleOIDCConfig
	}{
		{
			audience:        "https://svc.run.app",
			serviceAccounts: "a@p.iam.gserviceaccount.com, b@p.iam.gserviceaccount.com",
			want: authn.GoogleOIDCConfig{
				Audience:               "https://svc.run.app",
				AllowedServiceAccounts: []string{"a@p.iam.gserviceaccount.com", "b@p.iam.gserviceaccount.com"},
			},
		},
		{
			// A second audience, mixed case, and an entry with a trailing space.
			audience:        "https://Trigger.Example/Path",
			serviceAccounts: "c@p.iam.gserviceaccount.com ,d@p.iam.gserviceaccount.com",
			want: authn.GoogleOIDCConfig{
				Audience:               "https://Trigger.Example/Path",
				AllowedServiceAccounts: []string{"c@p.iam.gserviceaccount.com", "d@p.iam.gserviceaccount.com"},
			},
		},
	} {
		t.Run(tc.audience, func(t *testing.T) {
			got := captureNewGoogleOIDC(t)
			f := &Flags{audience: tc.audience, serviceAccounts: tc.serviceAccounts}

			a, err := f.Authenticator()
			if err != nil || a == nil {
				t.Fatalf("Authenticator() = %v, %v, want an authenticator", a, err)
			}
			if diff := cmp.Diff([]authn.GoogleOIDCConfig{tc.want}, *got); diff != "" {
				t.Errorf("config handed to NewGoogleOIDC mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAuthenticatorNilWithoutFlags(t *testing.T) {
	got := captureNewGoogleOIDC(t)

	a, err := (&Flags{}).Authenticator()
	if err != nil || a != nil {
		t.Errorf("Authenticator() = %v, %v, want nil, nil", a, err)
	}
	if len(*got) != 0 {
		t.Errorf("NewGoogleOIDC called with %v, want no call", *got)
	}
}

func TestAuthenticatorRejectsPaddedAudience(t *testing.T) {
	got := captureNewGoogleOIDC(t)
	for _, aud := range []string{" https://svc.run.app", "https://svc.run.app ", "https://svc.run.app\t", " "} {
		f := &Flags{audience: aud, serviceAccounts: "a@p.iam.gserviceaccount.com"}
		if _, err := f.Authenticator(); err == nil {
			t.Errorf("Authenticator() with audience %q succeeded, want an error", aud)
		}
	}
	if len(*got) != 0 {
		t.Errorf("NewGoogleOIDC called with %v, want the padded audience rejected before it", *got)
	}
}
