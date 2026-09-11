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
	"encoding/json"
	"strings"
	"testing"
)

// unescapeJSON had no direct test until this file, and neither FuzzRedact nor the
// exhaustive differential reach it: both drive redact, which does not call it.
// redactedForError is its only caller, so every escape case was covered only through
// end-to-end table rows that happened to use a well-formed escape.

// TestUnescapeJSONMatchesEncodingJSON compares it against the standard library
// over every string of length <= 6 on an alphabet chosen to build the three
// escapes it handles, plus the ways they can be malformed.
//
// Inputs the standard library rejects are skipped rather than asserted: an escape
// JSON does not define is outside this function's contract, which is to leave
// what it does not decode alone. TestUnescapeJSONLeavesTheRestAlone covers those.
func TestUnescapeJSONMatchesEncodingJSON(t *testing.T) {
	// "n" earns its place by being the one symbol that forms an escape this
	// package deliberately does not decode, which is what exercises the
	// hasForeignEscape filter below. Without it the filter never fires and the
	// skip it accounts for never happens.
	alphabet := []byte(`\/un0dA8`)
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
	gen("", 6)

	checked, escaped, foreign, rejected := 0, 0, 0, 0
	for _, w := range words {
		var want string
		if err := json.Unmarshal([]byte(`"`+w+`"`), &want); err != nil {
			rejected++
			continue
		}
		// Only inputs whose escapes are all ours. \n, \t and friends are real JSON
		// and deliberately not decoded here, so they are not a disagreement.
		if hasForeignEscape(w) {
			foreign++
			continue
		}
		checked++
		if strings.Contains(w, `\`) {
			escaped++
		}
		if got := unescapeJSON(w); got != want {
			t.Fatalf("unescapeJSON(%q) = %q, encoding/json says %q", w, got, want)
		}
	}
	t.Logf("%d inputs compared (%d of them escaped), %d rejected by the oracle, %d skipped as foreign escapes",
		checked, escaped, rejected, foreign)
	// Counting comparisons proves nothing: the escape-free words alone number
	// tens of thousands and would clear any such floor while asserting only that
	// plain text is copied. The floors that matter are on the two populations the
	// test exists for.
	if escaped < 1000 {
		t.Errorf("only %d comparable inputs contained an escape, so the alphabet stopped producing them", escaped)
	}
	if foreign == 0 {
		t.Error("no input was skipped as a foreign escape, so that filter went unexercised")
	}
}

// TestUnescapeJSONMatchesEncodingJSONOnPairs extends the oracle to inputs the
// byte-level sweep above cannot reach. That one stops at 6 bytes, which is one
// \uXXXX escape, so a surrogate PAIR — 12 bytes, and the case the decoder was
// wrong about — is never compared against the standard library there.
//
// Building from whole escape tokens instead of bytes reaches 30-byte inputs at a
// fraction of the cost, and every sequence of a high and a low surrogate, of two
// highs, of a high followed by a BMP escape, and of a doubled backslash in front
// of any of them falls out of the enumeration.
func TestUnescapeJSONMatchesEncodingJSONOnPairs(t *testing.T) {
	tokens := []string{`\ud83d`, `\ude00`, `\u0041`, `\\`, `\/`, "q"}
	var words []string
	var gen func(prefix string, n int)
	gen = func(prefix string, n int) {
		words = append(words, prefix)
		if n == 0 {
			return
		}
		for _, tok := range tokens {
			gen(prefix+tok, n-1)
		}
	}
	gen("", 5)

	var pairs int
	for _, w := range words {
		var want string
		if err := json.Unmarshal([]byte(`"`+w+`"`), &want); err != nil {
			t.Fatalf("the oracle rejected %q, so the token set is wrong: %v", w, err)
		}
		if strings.Contains(w, `\ud83d\ude00`) {
			pairs++
		}
		if got := unescapeJSON(w); got != want {
			t.Fatalf("unescapeJSON(%q) = %q, encoding/json says %q", w, got, want)
		}
	}
	t.Logf("%d token sequences compared against encoding/json", len(words))
	// Guards the reason this test exists: a token set that stopped producing
	// adjacent surrogate halves would still pass every assertion above.
	if pairs == 0 {
		t.Error("no input contained a complete surrogate pair, so the pair path went uncompared")
	}
}

// hasForeignEscape reports whether s carries a JSON escape this package does not
// decode, which is every one except \\, \/ and \uXXXX.
func hasForeignEscape(s string) bool {
	for i := 0; i < len(s); {
		if s[i] != '\\' || i+1 >= len(s) {
			i++
			continue
		}
		switch s[i+1] {
		case '\\', '/':
			i += 2
		case 'u':
			i += 6
		default:
			return true
		}
	}
	return false
}

// TestUnescapeJSONLeavesTheRestAlone covers the malformed shapes the standard
// library rejects outright, so no oracle can speak for them.
//
// Each is a way the new surrogate lookahead can be entered and not completed. The
// property is the same throughout: nothing panics, nothing is dropped, and what
// cannot be decoded comes back as it went in.
func TestUnescapeJSONLeavesTheRestAlone(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"a high surrogate at the end", `\ud83d`, "\uFFFD"},
		{"a high surrogate then a truncated escape", `\ud83d\ude0`, "\uFFFD" + `\ude0`},
		{"a high surrogate then a non-escape", `\ud83dX`, "\uFFFD" + "X"},
		{"a high surrogate then a non-surrogate", `\ud83d\u0041`, "\uFFFD" + "A"},
		{"two high surrogates", `\ud83d\ud83d`, "\uFFFD\uFFFD"},
		{"a low surrogate first", `\ude00\ud83d`, "\uFFFD\uFFFD"},
		{"a second escape that is not hex", `\ud83d\uZZZZ`, "\uFFFD" + `\uZZZZ`},
		{"non-hex in the first escape", `\uZZZZ`, `\uZZZZ`},
		{"a bare backslash-u at the end", `\u`, `\u`},
		{"an escape this package does not decode", `\n`, `\n`},
		{"a lone trailing backslash", `abc\`, `abc\`},
		{"no backslash at all", "plain text", "plain text"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := unescapeJSON(tc.in); got != tc.want {
				t.Errorf("unescapeJSON(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestUnescapeJSONNeverGrows pins the assumption b.Grow(len(s)) rests on.
func TestUnescapeJSONNeverGrows(t *testing.T) {
	for _, in := range []string{
		`\ud83d\ude00`, `\u0041`, `\/`, `\\`, `\uFFFF`, strings.Repeat(`\u0040`, 1000),
	} {
		if got := unescapeJSON(in); len(got) > len(in) {
			t.Errorf("unescapeJSON(%q) grew %d bytes to %d", in, len(in), len(got))
		}
	}
}

// TestServiceTextReturnsOnlyWhatItCanShowClean pins the choice between the two
// candidate outputs, which three earlier revisions of redactedForError got wrong.
//
// Each of those decided from a property of the INPUTS, and the service writes the
// inputs. The last row is the shape that broke them: a decoy occurrence the decode
// destroys costs the service nothing and moves any such comparison wherever it
// likes. Deciding from the output instead is what these rows hold in place.

func TestServiceTextReturnsOnlyWhatItCanShowClean(t *testing.T) {
	for _, tc := range []struct {
		name    string
		in      string
		secrets []string
		want    string
	}{{
		// The commonest real shape, and the one a presence test gets wrong: the
		// service quotes what it received and then what it normalized to, so the
		// identifier is in the body twice, escaped once and plain once. Both copies
		// then CONTAIN it, and only counting shows the decoded copy holds two where
		// the original holds one. Getting this wrong ships the escaped spelling in
		// a body whose [redacted] marker makes it look scrubbed.
		name:    "the same secret escaped once and plain once",
		in:      `invalid userId: alice\u0040example.test (normalized: alice@example.test)`,
		secrets: []string{"alice@example.test"},
		want:    "invalid userid: [redacted] (normalized: [redacted])",
	}, {
		// Only the decoded copy matches, so it is the one that can remove the
		// identifier. This is the case the decode exists for.
		name:    "the decode reveals the only secret",
		in:      `bad user alice\u0040example.test`,
		secrets: []string{"alice@example.test"},
		want:    "bad user [redacted]",
	}, {
		// Already-decoded text, which is what three of the four callers pass. The
		// service really did report two backslashes and the secret is plainly
		// visible without decoding, so decoding buys nothing and must not rewrite
		// the path.
		name:    "the decode reveals nothing and would rewrite the text",
		in:      `invalid path C:\\logs for alice@example.test`,
		secrets: []string{"alice@example.test"},
		want:    `invalid path c:\\logs for [redacted]`,
	}, {
		// Neither candidate can be shown clean. The identifier matches the original
		// only, the URI the decoded copy only, so scrubbing either one leaves the
		// other readable — the URI decodes straight out of the first, and the
		// identifier survives the second as the slash it decodes to. Withholding is
		// the answer rather than picking the less bad leak.
		name:    "no candidate is clean, so nothing is returned",
		in:      `bad user alice\/bob, uri https:\/\/app.test\/cb`,
		secrets: []string{`alice\/bob`, "https://app.test/cb"},
		want:    withheldText,
	}, {
		// A decoy occurrence the decode destroys. \u004a decodes to "J" and eats the
		// "a" after it, so this contains the identifier and decodes to something
		// that does not. Every revision that compared the two inputs read that as a
		// reason to keep the original, where the real echo is still escaped, and
		// shipped it under a [redacted] marker that made the body look scrubbed.
		name:    "a decoy occurrence the decode destroys",
		in:      `invalid userId: alice\u0040example.test (also \u004alice@example.test)`,
		secrets: []string{"alice@example.test"},
		want:    "invalid userid: [redacted] (also jlice@example.test)",
	}, {
		// The scan must skip the marker redact itself wrote, or a secret spelled
		// with letters of "redacted" finds itself inside it and every response is
		// withheld. A one-character id makes that permanent: with the marker in
		// scope, every body containing an "e" is suppressed, adversary or not.
		name:    "a secret made of letters the marker also contains",
		in:      "denied",
		secrets: []string{"e"},
		want:    "d[redacted]ni[redacted]d",
	}, {
		// The cap has to be applied before the check, because its ellipsis is text
		// this code appends and appended text can finish a secret the untruncated
		// string only started. Here the body ends the identifier with a "y", so
		// nothing matches until the cut replaces that tail with "...", which
		// completes the dot the identifier ends in.
		name:    "the cap's ellipsis completes the secret",
		in:      strings.Repeat("x", 1006) + "alice@example.test" + "y",
		secrets: []string{"alice@example.test."},
		want:    withheldText,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			if got := redactedForError(tc.in, tc.secrets...); got != tc.want {
				t.Errorf("redactedForError(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestServiceTextNeverReturnsARecoverableSecret is the invariant the four earlier
// revisions of redactedForError each violated, stated once and checked over every body
// an adversary can assemble from the pieces that broke them.
//
// It does not call recoverable. Calling it made an earlier version circular:
// redactedForError returns a candidate only when recoverable says no, so asserting
// the same predicate held by construction and the test was green on a body it was
// already generating — a doubly escaped identifier, which the then-single-pass
// recoverable could not see. The oracle below is written from the attacker's side
// instead: strip the markers, decode until nothing changes, look for the identifier.
//
// Removing the circularity is not the same as buying independence, and the oracle
// does not claim the second — see readable, which is a frozen copy of recoverable
// making the same structural choices, re-frozen whenever it moves. So this catches a future weakening of
// recoverable, and not a blind spot the two share today.
func TestServiceTextNeverReturnsARecoverableSecret(t *testing.T) {
	const user = "alice@example.test"
	const uri = "https://app.test/cb"
	pieces := []string{
		"",
		user,                       // plain
		`alice\u0040example.test`,  // escaped
		`alice\\u0040example.test`, // doubly escaped: one decode pass is not enough
		`\u004alice@example.test`,  // decoy: decodes to Jlice@…, contains the plain form
		uri,                        // the other secret, plain
		`https:\/\/app.test\/cb`,   // the other secret, escaped
		strings.Repeat("x", 600),   // pushes a later piece past the 1024-byte cap
	}
	secrets := []string{user, uri}

	var checked, withheld int
	var gen func(prefix string, n int)
	gen = func(prefix string, n int) {
		if n == 0 {
			checked++
			got := redactedForError(prefix, secrets...)
			if got == withheldText {
				withheld++
				return
			}
			if readable(t, got, secrets) {
				t.Fatalf("redactedForError(%q) = %q, out of which a secret is still readable", prefix, got)
			}
			return
		}
		for _, p := range pieces {
			gen(prefix+p, n-1)
		}
	}
	for n := 1; n <= 3; n++ {
		gen("", n)
	}

	t.Logf("%d bodies checked, %d withheld", checked, withheld)
	// Both outcomes have to occur, or the invariant is being satisfied trivially.
	if withheld == checked {
		t.Error("every body was withheld, so the clean path went unexercised")
	}
	if withheld == 0 {
		t.Error("no body was withheld, so the fail-closed path went unexercised")
	}
}

// readable is the test's own oracle for "an attacker can read a secret out of this
// string". It is a frozen COPY of recoverable, not an independent one: it makes
// the same three structural choices — exclude the marker by position, decode then
// fold, the same unescapeJSON. So it catches a future weakening of recoverable,
// which is worth having, and it cannot catch a blind spot the two share today.
// \U0040 is one they share: unescapeJSON matches the introducer case-sensitively
// while redact folds case, and neither this nor recoverable folds before decoding.
//
// Being a copy is the whole contract, so it has to be re-frozen when the original
// moves. It split on the marker while recoverable split, and went on splitting for
// a commit after recoverable stopped — which cost it exactly the case that commit
// was about, since a value spanning a marker lands whole in no part.
//
// Marker bytes are excused rather than searched, since text redact itself inserted
// is not something the service disclosed. Decoding runs to a fixpoint under an
// explicit bound: relying on the production loop's termination argument here would
// hand the oracle the same assumption it is supposed to be testing.
func readable(t *testing.T, x string, secrets []string) bool {
	t.Helper()
	decode := func(s string) string {
		for range 64 {
			u := unescapeJSON(s)
			if u == s {
				return s
			}
			s = u
		}
		t.Fatalf("decoding %q never reached a fixpoint in 64 passes", x)
		return ""
	}
	const marker = "[redacted]"
	for _, text := range []string{strings.ToLower(x), strings.ToLower(decode(x))} {
		ours := make([]bool, len(text))
		for i := 0; ; {
			j := strings.Index(text[i:], marker)
			if j < 0 {
				break
			}
			for k := i + j; k < i+j+len(marker); k++ {
				ours[k] = true
			}
			i += j + len(marker)
		}
		for _, v := range secrets {
			for _, form := range []string{strings.ToLower(v), strings.ToLower(decode(v))} {
				if form == "" {
					continue
				}
				for i := 0; i+len(form) <= len(text); {
					j := strings.Index(text[i:], form)
					if j < 0 {
						break
					}
					at, all := i+j, true
					for k := at; k < at+len(form); k++ {
						all = all && ours[k]
					}
					if !all {
						return true
					}
					i = at + 1
				}
			}
		}
	}
	return false
}

// TestDecodeFullyStopsAtItsPassBound pins the bound itself, which the end-to-end
// timing test cannot see: the straddle window keeps the input small enough that
// even an unbounded loop finishes inside its budget.
//
// Terminating is not the same as affordable. "\u005c" decodes to a backslash that
// re-forms the introducer for the next one, so the loop shortens by five bytes a
// pass and needs one pass per five bytes of input.
//
// The boundary rows pin where the bound actually falls, which is one short of the
// constant: maxDecodePasses counts reads, and a text is only reported finished by
// a read that changes nothing, so the last of the maxDecodePasses reads has no
// successor to confirm it. Fail-closed either way, and pinned so it cannot drift.
func TestDecodeFullyStopsAtItsPassBound(t *testing.T) {
	// Well inside a pass bound, and a fixpoint it can actually reach.
	if got, done := decodeFully(`\u005c` + strings.Repeat("u005c", 4)); !done {
		t.Errorf("decodeFully() gave up on %d passes' worth of input, got %q", 5, got)
	}
	// The last input reported finished, and the first one not.
	if _, done := decodeFully(`\u005c` + strings.Repeat("u005c", maxDecodePasses-2)); !done {
		t.Errorf("decodeFully() gave up %d changing passes in, want it to report the fixpoint", maxDecodePasses-1)
	}
	if _, done := decodeFully(`\u005c` + strings.Repeat("u005c", maxDecodePasses-1)); done {
		t.Errorf("decodeFully() reported a fixpoint %d changing passes in, want it to run out of reads", maxDecodePasses)
	}
	// Past it. What matters is the flag, not the text: the caller reads "not
	// finished" as "assume a secret is in there".
	if _, done := decodeFully(`\u005c` + strings.Repeat("u005c", maxDecodePasses+50)); done {
		t.Error("decodeFully() reported a fixpoint on input needing more than maxDecodePasses")
	}
	// And the caller does fail closed on it, on BOTH the text it examines and the
	// secrets it examines it for. Two arms, so two rows: one arm covered left the
	// other free to be flipped to fail open with the suite still green.
	long := `\u005c` + strings.Repeat("u005c", maxDecodePasses+50)
	if !recoverable(long, []string{"nowhere-in-this-string"}) {
		t.Error("recoverable() = false on text it could not finish decoding, want it to assume the worst")
	}
	if !recoverable("carries no escape at all", []string{long}) {
		t.Error("recoverable() = false on a secret it could not finish decoding, want it to assume the worst")
	}
}

// TestDecodeFullyShrinksWheneverItChanges pins the termination argument decodeFully
// relies on, which is not the same claim as TestUnescapeJSONNeverGrows: an equal
// length that still changed would loop forever.
func TestDecodeFullyShrinksWheneverItChanges(t *testing.T) {
	for _, in := range []string{
		`\\u0040`, `\u0040`, `\/`, `\\`, `\ud83d\ude00`, `\uZZZZ`, `plain`, ``,
		`\\\\u0040example`, `a\\b\/c\u0041d`,
	} {
		if u := unescapeJSON(in); u != in && len(u) >= len(in) {
			t.Errorf("unescapeJSON(%q) = %q changed without shrinking, %d to %d bytes", in, u, len(in), len(u))
		}
	}
}

// TestSecretSpanningOurOwnMarkerIsNotShown pins the half of the marker rule that
// excluding the marker got wrong.
//
// The marker has to be kept out of the search somehow: a one-character user id of
// "e" occurs inside "[redacted]" and would otherwise suppress every response that
// redacted anything. Keeping it out by splitting the text on it kept out too much
// — a value whose own bytes contain or abut the marker spans a split, lands whole
// in no part, and is matched by nothing, so `a[redacted]b` came back verbatim.
//
// What every row asserts is the contract itself: the value is not readable out of
// what comes back. Withholding satisfies that too, and so does scrubbing, so
// neither outcome is prescribed — the two leak rows are in fact scrubbed rather
// than withheld, because rejecting the first candidate lets the second one decode
// and match. The controls additionally assert the response is SHOWN, since
// suppressing them is the failure the split was introduced to prevent.
func TestSecretSpanningOurOwnMarkerIsNotShown(t *testing.T) {
	for _, tc := range []struct {
		name     string
		body     string
		secrets  []string
		mustShow bool
	}{{
		name:    "the marker sits inside the value",
		body:    `\u0061[redacted]b`,
		secrets: []string{"a[redacted]b"},
	}, {
		name:    "the value abuts the marker and shares its opening",
		body:    `\u0061lice[redacted]`,
		secrets: []string{"alice[red"},
	}, {
		// The marker here is one WE wrote, over the first value, and the second
		// value starts inside it and runs on into the service's own bytes. Every
		// byte of the occurrence has to be checked, and both ends of it: the first
		// byte is ours, the last is not, and either one alone answers wrong.
		name:    "the value begins inside our marker and runs into service text",
		body:    "uabc",
		secrets: []string{"u", "ted]a"},
	}, {
		name:     "CONTROL a one-character value that occurs inside the marker",
		body:     "denied for user e, retry later",
		secrets:  []string{"e"},
		mustShow: true,
	}, {
		name:     "CONTROL an ordinary value, ordinary body",
		body:     "denied for alice@example.test",
		secrets:  []string{"alice@example.test"},
		mustShow: true,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			got := redactedForError(tc.body, tc.secrets...)
			t.Logf("returned %q", got)
			if tc.mustShow && got == withheldText {
				t.Fatal("withheld, want the body shown with the value redacted out of it")
			}
			if got == withheldText {
				return
			}
			if !strings.Contains(got, redactedMarker) {
				t.Errorf("nothing was redacted from %q", got)
			}
			decoded, _ := decodeFully(got)
			for _, secret := range tc.secrets {
				// A value that fits inside the marker is readable out of the marker
				// by construction, and excusing exactly that is why the marker is
				// excluded at all. Every other value must be gone.
				if strings.Contains(redactedMarker, strings.ToLower(secret)) {
					continue
				}
				if strings.Contains(strings.ToLower(decoded), strings.ToLower(secret)) {
					t.Errorf("%q is readable out of %q", secret, decoded)
				}
			}
		})
	}
}
