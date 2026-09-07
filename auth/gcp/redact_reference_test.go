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
)

// This file exists because redact was rewritten seven times over one review and
// six of those rewrites shipped a way for the acting user to survive. Opinion and
// example-based tests kept missing the next one, so the property is settled here
// by exhaustion against a definition simple enough to be read and believed.

// reference is the obviously-correct definition: mark every byte covered by any
// occurrence of any secret in the lowered text, then emit the uncovered runs
// separated by one marker per maximal covered run. Quadratic and unusable in
// production, which is the point — it exists only to say what the answer is.
func reference(s string, values ...string) string {
	var lowered []string
	for _, v := range values {
		if v != "" {
			lowered = append(lowered, strings.ToLower(v))
		}
	}
	if len(lowered) == 0 {
		return s
	}
	ls := strings.ToLower(s)
	covered := make([]bool, len(ls))
	hit := false
	for _, lv := range lowered {
		for i := 0; i+len(lv) <= len(ls); i++ {
			if ls[i:i+len(lv)] == lv {
				hit = true
				for j := i; j < i+len(lv); j++ {
					covered[j] = true
				}
			}
		}
	}
	if !hit {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(ls); {
		if covered[i] {
			b.WriteString("[redacted]")
			for i < len(ls) && covered[i] {
				i++
			}
			continue
		}
		b.WriteByte(ls[i])
		i++
	}
	return b.String()
}

// TestRedactMatchesReferenceExhaustively compares redact against reference over
// every string of length <= 4 over {a,b,A,B} crossed with every secret pair of
// length <= 3 — 2,463,725 combinations, about a second.
//
// The alphabet is deliberately tiny and repetitive: overlap, containment,
// self-overlap and case difference are exactly what four-character strings over
// two letters in two cases produce in bulk, and those are the shapes every past
// bug needed.
//
// What it provably cannot reach, so that a green run here is not read as more
// than it is: any byte outside those four, hence no multi-byte rune, no invalid
// UTF-8 and no length-changing fold — the table and the fuzz carry those. And
// nothing that needs a body longer than four bytes. A concrete example of the
// second, since it is the easy one to overlook: bounding the extension walk to a
// fixed lookahead ("for k := at + 1; k < end && k < at+4") passes all 2,463,725
// combinations here and still leaves a byte of the secret on redact("aaaaaa",
// "aa"). TestRedactAcrossSeveralSecrets carries that case for exactly this
// reason.
func TestRedactMatchesReferenceExhaustively(t *testing.T) {
	alphabet := []byte("abAB")
	var words []string
	var gen func(prefix string, n int)
	gen = func(prefix string, n int) {
		words = append(words, prefix)
		if n == 0 {
			return
		}
		for _, c := range alphabet {
			gen(prefix+string(c), n-1)
		}
	}
	// Length 4 is where the depth-2 extension chain lives: redact("aaaa","aa","aa")
	// grows end 2 -> 3 -> 4, so the fresh re-read of end is exercised. Short mode
	// drops to 3, which keeps overlap and containment and loses that one chain.
	maxLen := 4
	if testing.Short() {
		maxLen = 3
	}
	gen("", maxLen)

	secrets := []string{}
	for _, w := range words {
		if len(w) <= maxLen-1 {
			secrets = append(secrets, w)
		}
	}

	checked, mismatches, adjacencyOnly := 0, 0, 0
	for _, body := range words {
		for _, s1 := range secrets {
			for _, s2 := range secrets {
				checked++
				got, want := redact(body, s1, s2), reference(body, s1, s2)
				if got == want {
					continue
				}
				// Adjacency is counted apart from substance, and then required to
				// be zero. Redact emits one marker per redacted RUN, so it should
				// match the reference byte for byte — separating the two says which
				// property broke rather than only that something did.
				//
				// Tolerating adjacency was tried and is wrong. Reverting the
				// run-merge makes redact emit a marker per MATCH, which turns a
				// megabyte of a one-character user id into ten megabytes. With the
				// difference merely counted, this test passed with 122,304 of them.
				collapse := func(x string) string {
					for strings.Contains(x, "[redacted][redacted]") {
						x = strings.ReplaceAll(x, "[redacted][redacted]", "[redacted]")
					}
					return x
				}
				if collapse(got) == collapse(want) {
					if adjacencyOnly < 3 {
						t.Errorf("ADJACENCY redact(%q, %q, %q) = %q, reference says %q — same bytes "+
							"redacted, different marker count, so the run-merge is gone", body, s1, s2, got, want)
					}
					adjacencyOnly++
					continue
				}
				if mismatches < 5 {
					t.Errorf("SUBSTANTIVE redact(%q, %q, %q) = %q, reference says %q", body, s1, s2, got, want)
				}
				mismatches++
			}
		}
	}
	t.Logf("%d combinations checked, %d substantive mismatches, %d marker-adjacency differences", checked, mismatches, adjacencyOnly)
	if adjacencyOnly != 0 {
		t.Errorf("%d combinations differ from the reference in marker count; redact must emit one "+
			"marker per redacted run", adjacencyOnly)
	}
}
