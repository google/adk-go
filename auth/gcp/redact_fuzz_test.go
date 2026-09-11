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

package gcp

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// FuzzRedact drives the two-secret shape every call site uses.
//
// It is checked in because redact is the whole of contract invariant 3's
// enforcement and it took several attempts to get right: a literal scrub, an
// ASCII-only fold, an offset splice, one pass per value, and two range-merge
// bugs each shipped a way for the acting user to survive.
//
// The seeds are those inputs. They seed the corpus rather than standing in for
// the regression suite: the exact-output assertions in TestRedactAcrossSeveralSecrets
// are what pin those bugs, because several of them leaked a FRAGMENT of the
// identifier and a whole-secret containment check cannot see a fragment. The
// property below removes the markers before looking, which recovers most of that
// — a surviving fragment of two bytes or more now fails it.
//
// Run it with:
//
//	go test ./auth/gcp/ -run=FuzzRedact -fuzz=FuzzRedact -fuzztime=2m
func FuzzRedact(f *testing.F) {
	for _, seed := range [][3]string{
		{"error https://app.example.test/cb user e", "https://app.example.test/cb", "e"},
		{"bad continueUri: https://example.test/cb?user=alice", "alice", "https://example.test/cb?user=alice"},
		{"invalid: https://app.example.test/cb?login=alice@example.test", "alice@example.test", "https://app.example.test/cb?login=al"},
		{"\u0130\u0130\u0130\u0130\u0130alice\u023a\u023a\u023a\u023a\u023a", "alice", ""},
		{"Alice@Example.test echoed as alice@example.test", "alice@example.test", ""},
		{"xalice", "xa", "alice"},
		{"https://cb.test/u/alice@example.test/alice@example.test", "alice@example.test", "https://cb.test/u/alice@example.test/a"},
		{"aaa", "aa", ""},
		{"\xdb\xd0[", "0", "\xc2["},
		{`invalid userId: alice\ud83d\ude00`, "alice\U0001F600", ""},
		{`bad continueUri: https:\/\/app.example.test\/cb`, "https://app.example.test/cb", ""},
	} {
		f.Add(seed[0], seed[1], seed[2])
	}

	f.Fuzz(func(t *testing.T, s, a, b string) {
		ls, got := strings.ToLower(s), redact(s, a, b)

		// Survival is only expressible over valid UTF-8. strings.ToLower maps every
		// invalid byte to U+FFFD, so two unrelated invalid bytes compare equal, and
		// the marker's own "[" can complete a secret ending in one — both produce a
		// substring match that is an artifact rather than a disclosure.
		if utf8.ValidString(s) && utf8.ValidString(a) && utf8.ValidString(b) {
			// Checked between the markers rather than across them. Substituting a
			// placeholder for the marker was tried and is wrong — any byte chosen
			// can occur in the input, and the fuzzer found "\x00" doing exactly
			// that. Splitting needs no placeholder. It also makes the check see a
			// secret that is a piece of "[redacted]", and a fragment left next to
			// one, which is the shape three of the historical bugs produced.
			between := strings.Split(strings.ToLower(got), "[redacted]")
			survives := func(needle string) bool {
				for _, part := range between {
					if strings.Contains(part, needle) {
						return true
					}
				}
				return false
			}
			for _, secret := range []string{a, b} {
				lsec := strings.ToLower(secret)
				if secret == "" || !strings.Contains(ls, lsec) {
					continue
				}
				if survives(lsec) {
					t.Errorf("redact(%q, %q, %q) = %q, %q survives", s, a, b, got, secret)
				}
				// A fragment is a leak too — every historical bug that was not a
				// clean miss left one. The longest proper prefix and the longest
				// proper suffix are the ones a bad splice or an unmerged overlap
				// produce, and a fragment that appears in the input outside the
				// secret is not evidence of anything.
				if len(lsec) < 3 {
					continue
				}
				elsewhere := strings.Split(strings.ToLower(s), lsec)
				inInputOutsideTheSecret := func(needle string) bool {
					for _, part := range elsewhere {
						if strings.Contains(part, needle) {
							return true
						}
					}
					return false
				}
				for _, frag := range []string{lsec[:len(lsec)-1], lsec[1:]} {
					if !inInputOutsideTheSecret(frag) && survives(frag) {
						t.Errorf("redact(%q, %q, %q) = %q, the fragment %q of %q survives", s, a, b, got, frag, secret)
					}
				}
			}
			if !utf8.ValidString(got) {
				t.Errorf("redact(%q, %q, %q) = %q, valid UTF-8 in and invalid out", s, a, b, got)
			}
		}
		// Each match costs at most one marker for the bytes it consumes, so a
		// rewrite that nests markers or rescans its own output shows up as growth.
		if len(got) > 10*len(ls)+len("[redacted]") {
			t.Errorf("redact(%q, %q, %q) grew %d bytes to %d", s, a, b, len(s), len(got))
		}
	})
}
