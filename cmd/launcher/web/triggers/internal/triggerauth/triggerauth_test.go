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
		{
			// Three entries, one mixed case: matching is exact.
			audience:        "https://svc.run.app",
			serviceAccounts: "E@p.iam.gserviceaccount.com,f@p.iam.gserviceaccount.com,g@p.iam.gserviceaccount.com",
			want: authn.GoogleOIDCConfig{
				Audience:               "https://svc.run.app",
				AllowedServiceAccounts: []string{"E@p.iam.gserviceaccount.com", "f@p.iam.gserviceaccount.com", "g@p.iam.gserviceaccount.com"},
			},
		},
	} {
		t.Run(tc.serviceAccounts, func(t *testing.T) {
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

// Compares every allow-list of one to six letters, commas and spaces with a
// parser written independently, rather than relying on a few hand-picked
// cases.
func TestAuthenticatorAllowListMatchesReference(t *testing.T) {
	got := captureNewGoogleOIDC(t)
	var lists []string
	var gen func(prefix string, n int)
	gen = func(prefix string, n int) {
		lists = append(lists, prefix)
		if n == 0 {
			return
		}
		for _, c := range []string{"a", "B", ",", " "} {
			gen(prefix+c, n-1)
		}
	}
	gen("", 6)

	for _, list := range lists[1:] {
		for _, audience := range []string{"https://svc.run.app", "https://Other.example/a b"} {
			*got = nil
			_, err := (&Flags{audience: audience, serviceAccounts: list}).Authenticator()
			want, ok := referenceAllowList(list)
			switch {
			case !ok && err == nil:
				t.Fatalf("Authenticator() with -oidc_service_accounts %q succeeded, want an error for its empty entry", list)
			case !ok:
				continue
			case err != nil:
				t.Fatalf("Authenticator() with -oidc_service_accounts %q = %v, want no error", list, err)
			}
			wantCfg := []authn.GoogleOIDCConfig{{Audience: audience, AllowedServiceAccounts: want}}
			if diff := cmp.Diff(wantCfg, *got); diff != "" {
				t.Fatalf("-oidc_service_accounts %q: config handed to NewGoogleOIDC mismatch (-want +got):\n%s", list, diff)
			}
		}
	}
}

// referenceAllowList splits s at each comma and trims spaces from both ends
// of every entry, byte by byte. ok is false when an entry is left empty.
func referenceAllowList(s string) (entries []string, ok bool) {
	start := 0
	for i := 0; i <= len(s); i++ {
		if i < len(s) && s[i] != ',' {
			continue
		}
		lo, hi := start, i
		for lo < hi && s[lo] == ' ' {
			lo++
		}
		for hi > lo && s[hi-1] == ' ' {
			hi--
		}
		if lo == hi {
			return nil, false
		}
		entries = append(entries, s[lo:hi])
		start = i + 1
	}
	return entries, true
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
