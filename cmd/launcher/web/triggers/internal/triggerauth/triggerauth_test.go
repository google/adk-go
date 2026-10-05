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
	"strings"
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

// triggerFlagAlphabet is the input domain of TestAuthenticatorMatchesReference:
// letters of both cases, the list separator, and four characters that
// strings.TrimSpace removes (space, tab, U+00A0 and U+3000).
var triggerFlagAlphabet = []string{"a", "B", ",", " ", "\t", "\u00a0", "\u3000"}

func isReferenceSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\u00a0' || r == '\u3000'
}

// allStrings returns every string of zero to n symbols from alphabet.
func allStrings(alphabet []string, n int) []string {
	out := []string{""}
	last := []string{""}
	for range n {
		var next []string
		for _, prefix := range last {
			for _, c := range alphabet {
				next = append(next, prefix+c)
			}
		}
		out = append(out, next...)
		last = next
	}
	return out
}

// referenceConfig states the flag rules independently of Authenticator. ok is
// false when the flags must be rejected. cfg is nil when neither flag is set.
func referenceConfig(audience, accounts string) (cfg *authn.GoogleOIDCConfig, ok bool) {
	if audience == "" && accounts == "" {
		return nil, true
	}
	if audience == "" || accounts == "" {
		return nil, false
	}
	aud := []rune(audience)
	if isReferenceSpace(aud[0]) || isReferenceSpace(aud[len(aud)-1]) {
		return nil, false
	}
	var entries []string
	var entry []rune
	for _, r := range append([]rune(accounts), ',') {
		if r != ',' {
			entry = append(entry, r)
			continue
		}
		lo, hi := 0, len(entry)
		for lo < hi && isReferenceSpace(entry[lo]) {
			lo++
		}
		for hi > lo && isReferenceSpace(entry[hi-1]) {
			hi--
		}
		if lo == hi {
			return nil, false
		}
		entries = append(entries, string(entry[lo:hi]))
		entry = entry[:0]
	}
	return &authn.GoogleOIDCConfig{Audience: audience, AllowedServiceAccounts: entries}, true
}

// R1-a, bounded: over every audience of up to two symbols and every allow-list
// of up to five symbols from triggerFlagAlphabet, Authenticator rejects exactly
// what the reference rejects, and otherwise hands NewGoogleOIDC exactly the
// reference config.
func TestAuthenticatorMatchesReference(t *testing.T) {
	got := captureNewGoogleOIDC(t)
	check := func(audience, accounts string) {
		t.Helper()
		*got = nil
		a, err := (&Flags{audience: audience, serviceAccounts: accounts}).Authenticator()
		want, ok := referenceConfig(audience, accounts)
		switch {
		case !ok:
			if err == nil {
				t.Fatalf("Authenticator(%q, %q) succeeded, want an error", audience, accounts)
			}
		case err != nil:
			t.Fatalf("Authenticator(%q, %q) = %v, want no error", audience, accounts, err)
		case want == nil:
			if a != nil || len(*got) != 0 {
				t.Fatalf("Authenticator(%q, %q) built an authenticator, want none", audience, accounts)
			}
		default:
			if diff := cmp.Diff([]authn.GoogleOIDCConfig{*want}, *got); diff != "" {
				t.Fatalf("Authenticator(%q, %q): config handed to NewGoogleOIDC mismatch (-want +got):\n%s", audience, accounts, diff)
			}
		}
	}
	for _, audience := range allStrings(triggerFlagAlphabet, 2) {
		check(audience, "a")
		check(audience, "")
	}
	for _, accounts := range allStrings(triggerFlagAlphabet, 5) {
		check("https://svc.run.app", accounts)
		check("B", accounts)
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

// A lone flag must name the missing one, not surface NewGoogleOIDC's error for
// the empty value.
func TestAuthenticatorNamesTheMissingFlag(t *testing.T) {
	for _, tc := range []struct {
		f    Flags
		want string
	}{
		{Flags{audience: "https://svc.run.app"}, "requires -oidc_service_accounts"},
		{Flags{serviceAccounts: "a@p.iam.gserviceaccount.com"}, "requires -oidc_audience"},
	} {
		if _, err := tc.f.Authenticator(); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("Authenticator() with %+v = %v, want an error containing %q", tc.f, err, tc.want)
		}
	}
}
