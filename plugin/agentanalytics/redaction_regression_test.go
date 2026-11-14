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

package agentanalytics

import (
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

// TestCompoundKeyMemberListHeaderAndBackstopRegressions covers inputs that
// kept a secret that the legacy regular expressions redacted or failed
// closed, or that a second pass changed: quote-led member lists whose keys
// are escaped or single-quoted, whose values are unquoted, or whose strings
// follow each other without a delimiter; "{"- or "["-led text with an
// Authorization or API-key header whose value starts on the next line;
// header lines that end in a joined or padded nested key; words after a
// placeholder or before a path-joined redacted key; container tails whose
// nested quoted value has text glued after it; line-start names with a
// non-ASCII space before a key; Unicode case folds of key names; URIs with
// credentials; and over-budget values with a dangling key quote.
func TestCompoundKeyMemberListHeaderAndBackstopRegressions(t *testing.T) {
	runInParallel(t)
	const (
		s0 = "LEAK_SECRET_ZERO"
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
		s3 = "LEAK_SECRET_THREE"
	)
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		// Quote-led member lists fail closed, as the legacy expressions
		// failed them closed.
		{"escaped_member_key", `"token": "[REDACTED]", \"note\": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"single_quoted_member_key", `"token": "[REDACTED]", 'note': "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"unquoted_placeholder_member", `"token": [REDACTED], "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"unquoted_placeholder_member_trailing_comma", `"token": [REDACTED], "note": "` + s1 + `",`, unparseableJSONPlaceholder, []string{s1}},
		{"number_member_before_placeholder", `"count": 1, "token": [REDACTED], "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"adjacent_placeholder_without_delimiter", `"` + s1 + `" "[REDACTED]"; "token": "[REDACTED]"`, unparseableJSONPlaceholder, []string{s1}},
		{"mixed_quotes_and_unicode_member_list", "password: \\'credentials\\':\t\"" + s0 + "\"\u00a0\\\"authorization\\\"\v:\"" + s1 + "\u00a0pa\u017f\u017fword\\\":\r\n  \"[REDACTED]\", \"note\": \"" + s2 + "\"", redactedSensitiveTextPlaceholder, []string{s0, s1, s2}},

		// A header whose value starts on the next line and reads as
		// redacted counts as a credential marker in malformed JSON text.
		{"brace_led_multiline_header", "{api_key:\n  \"[REDACTED]\" \"" + s1 + "\"}", unparseableJSONPlaceholder, []string{s1}},
		{"bracket_led_multiline_header", "[x]\napi_key:\n  \"[REDACTED]\" \"" + s1 + "\"", unparseableJSONPlaceholder, []string{s1}},
		{"brace_led_multiline_kelvin_header", "{api_\u212aey:\r\n  \"[REDACTED]\" \"" + s1 + "\"", unparseableJSONPlaceholder, []string{s1}},

		// A header line that ends in a nested key has its value redacted
		// up to the key that nestedKeyName reads, and the nested value keeps
		// its line break, so later fields stay and a second pass agrees.
		{"header_nested_key_after_joined_value", "DB_PASSWORD: a/api_key: v1 " + s1 + "/token:\n  " + s2, "DB_PASSWORD:[REDACTED] token:\n  [REDACTED]", []string{s1, s2}},
		{"header_padded_quoted_nested_key", "api_key: " + s1 + " \" token \":\n  " + s2, "api_key: [REDACTED] \"token\":\n  \"[REDACTED]\"", []string{s1, s2}},
		{"header_nested_key_after_vertical_tab", "api_key: v1\v" + s1 + "/token:\n  " + s2, "api_key: [REDACTED] token:\n  [REDACTED]", []string{s1, s2}},
		{"header_quoted_nested_key_keeps_fields", "api_key: x \"token\":\n  " + s1 + " other=ok", "api_key: [REDACTED] \"token\":\n  \"[REDACTED]\" other=ok", []string{s1}},
		{"header_bare_nested_key_keeps_fields", "api_key: x password:\n  " + s1 + " other=ok", "api_key: [REDACTED] password:\n  [REDACTED] other=ok", []string{s1}},
		{"header_nested_key_after_placeholder", "ok=[REDACTED],api_key:v1\v" + s1 + "/token:\n  " + s2, "ok=[REDACTED],api_key: [REDACTED] token:\n  [REDACTED]", []string{s1, s2}},

		// Words that the scanners keep before a path-joined redacted key
		// or inside a quoted key span count against the backstop.
		{"quoted_key_span_across_placeholder", `: "password:[a v0]/\"monkey\" ` + s1 + ` /token"=v1`, redactedSensitiveTextPlaceholder, []string{s1}},
		{"word_before_redacted_key_after_vertical_tab", "token=" + s1 + "\vpassword:[REDACTED]", "token=[REDACTED]", []string{s1}},

		// A nested quoted value with text glued after it runs on to where
		// it really ends, whatever the case fold of the outer key.
		{"container_tail_glued_nested_quote", `log: "api_key"={"k":"` + s1 + `"}/foo'x/temp:a="a token: "x ` + s2 + `""t`, `log: "api_key"=[REDACTED]`, []string{s1, s2}},
		{"container_tail_glued_nested_quote_kelvin", `log: \"api_` + "\u212a" + `ey\"={"k":"` + s1 + `"}/foo'x/temp:a="a token: "x ` + s2 + `""t`, `log: \"api_` + "\u212a" + `ey\"=[REDACTED]`, []string{s1, s2}},
		{"line_start_name_with_nbsp_before_key", "msg: \"\\'private_key\\':\n  S0/token:\n  " + s1 + "\u00a0x/signing_key:\tS2/token:\n  " + s3, "msg: \"\\'private_key\\':\n  S0/token:[REDACTED]", []string{s1, s3}},
		{"long_s_quoted_key_before_next_line_value", "parsing \"x\u00a0\\\"pa\u017f\u017fword\\\":\r\n  " + s1 + "/token:\n  " + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},

		// Text that starts with a redacted URI is text, also to
		// SmartTruncate, and a URI value keeps its key.
		{"uri_led_text", "https://user:" + s1 + "@example.com/a hello", redactedURIPlaceholder + " hello", []string{s1}},
		{"uri_led_text_with_quoted_key", "https://example.com/token/" + s0 + "\vapp:dbPassword\\\":\n  \" password \": " + s1 + " dbPassword\"\v:'" + s2 + "'", unparseableJSONPlaceholder, []string{s0, s1, s2}},
		{"nested_key_with_uri_value", "credentials:\n  dbPassword: https://admin:" + s1 + "@example.com", "credentials:\n  dbPassword: " + redactedURIPlaceholder, []string{s1}},
		{"quoted_nested_key_with_uri_value", "config:\n  \"DB_PASSWORD\":\n    password: https://admin:" + s1 + "@example.com", "config:\n  \"DB_PASSWORD\":\n    password: " + redactedURIPlaceholder, []string{s1}},

		// An over-budget value whose dangling key quote is followed by a
		// quoted value that runs past it is redacted through that value.
		{"over_budget_dangling_quote_compound_key", "DB_PASSWORD: " + a135 + `=S0/my_api_key":"a ` + s1 + `"`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_dangling_quote_exact_key", "DB_PASSWORD: " + a135 + `=S0/secret_key":"a ` + s1 + `" ok=1`, "DB_PASSWORD:[REDACTED] ok=1", []string{s1}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		if got, _ := sanitizeRawText(tc.in); got != tc.want {
			t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		raw, _, err := SmartTruncate(tc.in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", tc.label, tc.in, raw, err, tc.want)
		}
	}

	// Redacted text, structured text and prose stay byte-identical.
	for _, in := range []string{
		"api_key: [REDACTED] \"token\":\n  \"[REDACTED]\" other=ok",
		"api_key: [REDACTED] token:\n  [REDACTED]",
		"credentials:\n  dbPassword: " + redactedURIPlaceholder,
		"credentials:\n  spring.datasource.password: [REDACTED]\n  X-Amz-Security-Token: [REDACTED]",
		`"X-Amz-Security-Token": '[REDACTED]' other=ok`,
		`"token": "[REDACTED]"; other="ok"`,
		`"Hello," she said, "see you" [later].`,
		redactedURIPlaceholder + " hello",
		"x=[REDACTED]/a/token:[REDACTED]",
		"open /etc/ssl/private.key: no such file or directory",
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
	}
}

// TestMalformedQuotedMemberLists checks that hasMalformedQuotedMembers
// reads every member separator and adjacent string, whatever the quotes and
// the value before it, and that isBalancedAlreadyRedactedProse rejects
// escaped quotes.
func TestMalformedQuotedMemberLists(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		in   string
		want bool
	}{
		{`"token": [REDACTED], "note": "v"`, true},
		{`"count": 1, "token": "v"`, true},
		{`"a": "b", 'c': "d"`, true},
		{`"a": "b", \"c\": "d"`, true},
		{`"v" "[REDACTED]"`, true},
		{`"password=" "[REDACTED]"x`, true},
		{`"a" 'b'`, true},
		{`"password=" "[REDACTED]"`, false},
		{`"api.key":"[REDACTED]" other=ok`, false},
		{`"Hello," she said "bye"`, false},
		{`"token": "[REDACTED]", see logs`, false},
	} {
		if got := hasMalformedQuotedMembers(tc.in); got != tc.want {
			t.Errorf("hasMalformedQuotedMembers(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
	for _, in := range []string{`"token": "[REDACTED]" \"note\": "v"`, `"token": "[REDACTED]" \'v\'`} {
		if isBalancedAlreadyRedactedProse(in) {
			t.Errorf("isBalancedAlreadyRedactedProse(%q) = true, want false", in)
		}
	}
}

// TestNestedKeyBeforeNextLineReturnsKeyStart checks that
// nestedKeyBeforeNextLine returns where nestedKeyName reads the key, not
// where the name that ends in it starts, and reads padded quoted keys.
func TestNestedKeyBeforeNextLineReturnsKeyStart(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		in, key string
	}{
		{"api_key: v1 SECRET/token:\n  v", "token:"},
		{"api_key: v1\vSECRET/token:\n  v", "token:"},
		{"api_key: v1 a=b:token:\n  v", "token:"},
		{"api_key: SECRET \" token \":\n  v", "\" token \":"},
		{"api_key: x \"token\":\n  v", "\"token\":"},
	} {
		valStart := len("api_key: ")
		valEnd := strings.IndexByte(tc.in, '\n')
		got, ok := nestedKeyBeforeNextLine(tc.in, valStart, valEnd)
		if want := strings.Index(tc.in, tc.key); !ok || got != want {
			t.Errorf("nestedKeyBeforeNextLine(%q) = (%d, %v), want (%d, true)", tc.in, got, ok, want)
		}
	}
}

// TestBackstopExemptionsStayNarrow checks that wordsAfterPlaceholder counts
// only the words of the value token that a placeholder starts, and that
// redactedKeyBytes marks only the key of a redacted pair: not the words
// before a path-joined key, nor a quoted span across other words, and also
// the key of a redacted URI.
func TestBackstopExemptionsStayNarrow(t *testing.T) {
	runInParallel(t)
	glued := func(s string) []string {
		words := legacyWords(s, true)
		var out []string
		for j, g := range wordsAfterPlaceholder(s, words) {
			if g {
				out = append(out, s[words[j].start:words[j].end])
			}
		}
		return out
	}
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{"x=[REDACTED];token=SECRET", nil},
		{"ok=[REDACTED],api_key:v1\vSECRET", nil},
		{"[REDACTED]\u00a0password=SECRET", nil},
		{`ok=[REDACTED],"token":"[REDACTED]", "note":"SECRET"`, nil},
		{"password: [REDACTED]/aaa/token:[REDACTED]", []string{"aaa", "token"}},
		{"[REDACTED]x", []string{"x"}},
	} {
		if got := glued(tc.in); !slices.Equal(got, tc.want) {
			t.Errorf("wordsAfterPlaceholder(%q) marks %q, want %q", tc.in, got, tc.want)
		}
	}
	marked := func(s string) string {
		var b strings.Builder
		for i, m := range redactedKeyBytes(s) {
			if m {
				b.WriteByte(s[i])
			} else if b.Len() > 0 && !strings.HasSuffix(b.String(), " ") {
				b.WriteByte(' ')
			}
		}
		return strings.TrimSpace(b.String())
	}
	for _, tc := range []struct {
		in, want string
	}{
		{"SECRET/token:[REDACTED]", "token"},
		{`x "token":[REDACTED]`, "token"},
		{`: "password:[REDACTED] SECRET /token"=[REDACTED]`, "password"},
		{"token=SECRET\vpassword:[REDACTED]", "password"},
		{"credentials:\n  C:\\ssl\\password:[REDACTED]", `C:\ssl\password`},
		{"dbPassword: " + redactedURIPlaceholder, "dbPassword"},
	} {
		if got := marked(tc.in); got != tc.want {
			t.Errorf("redactedKeyBytes(%q) marks %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestLegacyPatternUnicodeFolds checks that legacyScrubTextPatterns matches
// key names and schemes as the reference expressions do under RE2 (?i),
// which folds U+017F (long s) to "s" and U+212A (Kelvin sign) to "k" but not
// U+0130 to "i", and reads only ASCII word bytes for \b, and that
// leavesLegacyValue flags scanner output that keeps a value they redact.
func TestLegacyPatternUnicodeFolds(t *testing.T) {
	runInParallel(t)
	refScrub, _ := legacyReferenceScrub()
	keptWords := func(raw, redacted string) []string {
		words := legacyWords(raw, true)
		var kept []string
		for j, at := range legacyKeptWords(raw, words, redacted) {
			if at >= 0 {
				kept = append(kept, raw[words[j].start:words[j].end])
			}
		}
		return kept
	}
	for _, in := range []string{
		"api_\u212aey: v w",
		"x-api-\u212aey: v",
		"pa\u017f\u017fword=abc x",
		"x\u017fecret=abc x",
		"-\u017fecret=abc x",
		"AP\u0130_KEY=abc x",
		"ba\u017fic dXNlcjpwYXNz",
		`"pa` + "\u017f\u017f" + `word": "abc" x`,
		`log: \"api_` + "\u212a" + `ey\"=abc x`,
	} {
		if got, want := keptWords(in, legacyScrubTextPatterns(in, false)), keptWords(in, refScrub(in)); !slices.Equal(got, want) {
			t.Errorf("legacyScrubTextPatterns(%q) keeps %q, reference expressions keep %q", in, got, want)
		}
	}
	if !leavesLegacyValue("pa\u017f\u017fword=abc x", "pa\u017f\u017fword=abc x") {
		t.Errorf("leavesLegacyValue keeps a long-s password value")
	}
}

// TestSmartTruncateIdempotentOnRedactedText checks that SmartTruncate,
// which runs its pass until the text stops changing, leaves its own result
// unchanged, as redactErrorMessage and sanitizeRawText do.
func TestSmartTruncateIdempotentOnRedactedText(t *testing.T) {
	runInParallel(t)
	for _, in := range []string{
		"https://user:SECRET@example.com/a hello",
		"https://example.com/token/SECRET\vapp:dbPassword\\\":\n  \" password \": SECRET dbPassword\"\v:'SECRET'",
		"https://example.com/cb?secret_key=SECRET&ok=1\vcredentials\\\" : name/token:\n  SECRET",
		"https://example.com/token/SECRET\vhttps://example.com/cb?x-goog-api-key=SECRET&ok=1\n  \\'DB_PASSWORD\\'\u00a0:\"a token: \"x SECRET\"",
		`{"msg":"password=SECRET"}`,
		`[{"token": "SECRET"}] trailing`,
	} {
		first, _, err := SmartTruncate(in, -1)
		var out string
		if err != nil || json.Unmarshal(first, &out) != nil {
			t.Fatalf("SmartTruncate(%q) = (%s, %v)", in, first, err)
		}
		if strings.Contains(out, "SECRET") {
			t.Errorf("SmartTruncate(%q) = %q kept SECRET", in, out)
		}
		if second, _, err := SmartTruncate(out, -1); err != nil || string(second) != string(first) {
			t.Errorf("SmartTruncate second pass(%q) = (%s, %v), want %s", out, second, err, first)
		}
	}
}

// TestMayMatchTextPatternsMatchesFoldedText checks mayMatchTextPatterns,
// which folds the text as it reads it, against patternWords in the folded
// text that foldPatternText builds.
func TestMayMatchTextPatternsMatchesFoldedText(t *testing.T) {
	runInParallel(t)
	ref := func(s string) bool {
		folded := foldPatternText(s)
		return slices.ContainsFunc(patternWords, func(w string) bool { return strings.Contains(folded, w) })
	}
	inputs := []string{
		"", "design: monkey=keyboard;", "pass-word", "PASS_WORD", "pa.ss.wo.rd", "pa\u017f\u017fword",
		"api_\u212aey", "AP\u0130KEY", "tem-p:", "temp:", "TEMP:x", "si-g", "b.e.a.r.e.r", "pass\u00a0word",
		"x" + strings.Repeat("-", 40) + "token", "ToKeN", "credentia", "s", "\xff\xfepassword",
		"\u00e9token", "\u65e5\u672cpassword", "pa\u00e9ssword", "\u017f\u212a\u0130", "to\U0001F600ken", "\U0001F600-token",
		"\xe6\x97password", "\xf0\x9f\x98token", "pass\xe2\x80\x8bword", "\u017fecret", "\u212aey",
	}
	for _, w := range patternWords {
		inputs = append(inputs, w, strings.ToUpper(w), w[:len(w)-1], "_"+w[:1]+"-"+w[1:]+".")
	}
	for _, in := range inputs {
		if got, want := mayMatchTextPatterns(in), ref(in); got != want {
			t.Errorf("mayMatchTextPatterns(%q) = %v, folded text gives %v", in, got, want)
		}
	}
}

// TestAdversarialScalingBackstopShapes checks that the member-list,
// placeholder, redacted-key, nested-header and over-budget shapes, including
// unquoted placeholders before quoted members, header lines that end in
// compound keys and over-budget values with closed containers after dangling
// key quotes, stay linear through all three entry points.
func TestAdversarialScalingBackstopShapes(t *testing.T) {
	runInParallel(t)
	a135 := strings.Repeat("a", 135)
	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"unquoted placeholder members", func(n int) string { return strings.Repeat(`"token": [REDACTED], `, n) }},
		{"words after a placeholder", func(n int) string { return "x=[REDACTED]," + strings.Repeat("a,b,", n) + "token=v" }},
		{"redacted quoted keys", func(n int) string { return strings.Repeat(`"a":"[REDACTED]",`, n) }},
		{"redacted path-joined keys", func(n int) string { return strings.Repeat("a/password:[REDACTED] ", n) }},
		{"header lines ending in joined nested keys", func(n int) string { return strings.Repeat("api_key: v1 s/token:\n  v other=ok\n", n) }},
		{"header line with many colons", func(n int) string { return "api_key: " + strings.Repeat("a:", n) + " token:\n  v" }},
		{"over-budget dangling quoted values", func(n int) string {
			return strings.Repeat("DB_PASSWORD: "+a135+`=S0/my_api_key":"a v" `, n)
		}},
		{"over-budget dangling quote chain", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`=S0/my_api_key":"a `, n) + `v"`
		}},
		{"brace-led multiline headers", func(n int) string { return "{" + strings.Repeat("api_key:\n  \"[REDACTED]\" \"v\"\n", n) }},
		{"redacted URI led lines", func(n int) string { return strings.Repeat("https://u:p@example.com/a token=v\n", n) }},
		{"unquoted placeholders before a quoted member", func(n int) string {
			return `"note": ` + strings.Repeat("[REDACTED] x ", n) + `"token": "v"`
		}},
		{"header lines ending in compound keys", func(n int) string {
			return strings.Repeat("api_key: v1 temp:a_s_b:\n  v\nX-Api-Key: v1 sk_s_token:\n  v other=ok\n", n)
		}},
		{"over-budget closed containers after dangling quotes", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`=S0/password":["a"]/`, n) + `token":["v" ]`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
}

// TestBarePlaceholderHeaderCompoundAndContainerRegressions covers inputs
// that kept a value that the legacy regular expressions redacted or failed
// closed: quote-led text with a placeholder followed by another member with
// no "," in between, or with a quoted sensitive key whose value the passes
// do not read; Authorization and API-key header lines
// that end in a compound, quoted compound or "temp:" nested key, whose
// words before the fixed key the header redacted; and over-budget values
// whose dangling key quote is followed directly by a container that runs
// past the value.
func TestBarePlaceholderHeaderCompoundAndContainerRegressions(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		// An unquoted placeholder followed by a quote of any kind, with or
		// without text in between, is a malformed member list.
		{"bare_placeholder_quoted_member", `"token": [REDACTED] "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_single_quoted_member", `"token": [REDACTED] 'note': "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_semicolon_member", `"token": [REDACTED]; "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_number_member", `"token": [REDACTED] 1 "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_newline_member", "\"token\": [REDACTED]\n\"note\": \"" + s1 + "\"", unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_state_key_and_quoted_member", "\"token\": [REDACTED] user:token\u00a0:[REDACTED]\t'count':\n  '" + s1 + "'", unparseableJSONPlaceholder, []string{s1}},
		{"word_before_bare_placeholder", `"token": x [REDACTED] "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"single_quoted_placeholder_member", `"token": '[REDACTED]' "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"bare_placeholder_after_quoted_key_colon", `"token": [REDACTED] note=` + s1, unparseableJSONPlaceholder, []string{s1}},
		{"quoted_placeholder_number_member", `"temp:a": "[REDACTED]" 1 "author": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"quoted_placeholder_semicolon_member", `"token": "[REDACTED]"; "note": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},
		{"single_quoted_placeholder_word_member", `"api_key": '[REDACTED]' x "path": "` + s1 + `"`, unparseableJSONPlaceholder, []string{s1}},

		// A quoted sensitive key whose value the passes do not read, as
		// after a non-ASCII space, keeps a placeholder elsewhere in the
		// text from counting as redacted prose.
		{"unread_quoted_key_value_with_placeholder", "\"user:token\"\u00a0:" + s1 + "\nnext_page_token = [REDACTED]", unparseableJSONPlaceholder, []string{s1}},
		{"unread_quoted_key_value_glued_placeholder", "\"awsSecretAccessKey\"\u00a0:" + s1 + "'[REDACTED]'", unparseableJSONPlaceholder, []string{s1}},

		// An escaped quote before a placeholder keeps quote-led text from
		// counting as redacted prose, as on main.
		{"escaped_quote_before_placeholder", "\"note\" : " + s1 + "&\\\"token_count\\\"\v:[REDACTED]", unparseableJSONPlaceholder, []string{s1}},

		// A header line that ends in a nested key redacts its value up to
		// the fixed key at the end of the name, or as a whole when the name
		// is a quoted compound or "temp:" name or ends in no fixed key.
		{"header_underscore_compound_key", "Authorization: Bearer sk_" + s1 + "_token:\n  unauthorized", "Authorization: [REDACTED] token:\n  [REDACTED]", []string{s1}},
		{"header_hyphen_compound_key", "api_key: v1 sk-" + s1 + "-token:\n  " + s2, "api_key: [REDACTED] token:\n  [REDACTED]", []string{s1, s2}},
		{"header_camel_compound_key", "api_key: v1 leaksecretoneToken:\n  " + s2, "api_key: [REDACTED] Token:\n  [REDACTED]", []string{"leaksecretone", s2}},
		{"header_temp_compound_key", "api_key: v1 temp:a_" + s1 + "_token:\n  " + s2, "api_key: [REDACTED] token:\n  [REDACTED]", []string{s1, s2}},
		{"header_path_compound_key", "X-Api-Key: " + s1 + "/sk_" + s2 + "_api_key:\n  " + s1, "X-Api-Key: [REDACTED] api_key:\n  [REDACTED]", []string{s1, s2}},
		{"header_temp_key", "api_key: v1 temp:" + s1 + ":\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_temp_key_without_fixed_key", "api_key: v1 temp:a_" + s1 + "_b:\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_quoted_compound_key", "api_key: v1 \"dev-" + s1 + "-password\":\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_quoted_underscore_compound_key", "api_key: v1 \"sk_" + s1 + "_token\":\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_quoted_temp_key", "api_key: v1 \"temp:" + s1 + "\":\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_compound_key_without_boundary", "api_key: v1 " + s1 + "PASSWORD:\n  v2", "api_key: [REDACTED]\n  v2", []string{s1}},
		{"header_quoted_state_key_keeps_fields", "api_key: x \"user:token\":\n  " + s1 + " other=ok", "api_key: [REDACTED] \"user:token\":\n  \"[REDACTED]\" other=ok", []string{s1}},
		{"header_fixed_hyphen_key_keeps_fields", "api_key: x x-api-key:\n  " + s1 + " other=ok", "api_key: [REDACTED] x-api-key:\n  [REDACTED] other=ok", []string{s1}},

		// A dangling key quote followed directly by a container that runs
		// past the over-budget value redacts through that container.
		{"over_budget_dangling_quote_object", "DB_PASSWORD: " + a135 + `=S0/credentials":{"value": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_dangling_quote_array", "dbPassword=" + a135 + `=S0/password":["a", "` + s1 + `"] other=ok`, "dbPassword=[REDACTED] other=ok", []string{s1}},
		{"over_budget_closed_container_then_dangling_quote_array", "dbPassword=" + a135 + `=S0/password":["a"]/token":["` + s1 + `" ] other=ok`, "dbPassword=[REDACTED] other=ok", []string{s1}},
		// The quotes inside a container that closes inside the value are
		// not read again, so they do not pair with the quotes after it.
		{"over_budget_closed_quoted_container_then_array", "DB_PASSWORD: " + a135 + `=S0/password":["'x'"]/password:["a", "` + s1 + `"] other=ok`, "DB_PASSWORD:[REDACTED] other=ok", []string{s1}},

		// Against the scrub with every key that isSensitiveKey accepts, a
		// compound name within a line is a key as a whole, so the
		// dbPassword after a non-ASCII space does not fail the text closed.
		{"compound_key_after_non_ascii_space", "USER:TOKEN:\n  \"OPENAI_API_KEY\"=[REDACTED]\u00a0dbPassword: \"" + s1 + "\"", "USER:TOKEN:\n  \"OPENAI_API_KEY\"=[REDACTED]\u00a0dbPassword:\"[REDACTED]\"", []string{s1}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		if got, _ := sanitizeRawText(tc.in); got != tc.want {
			t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		raw, _, err := SmartTruncate(tc.in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", tc.label, tc.in, raw, err, tc.want)
		}
	}

	// Redacted text and prose stay byte-identical.
	for _, in := range []string{
		`"token": "[REDACTED]"; other="ok"`,
		`"api.key":"[REDACTED]" other=ok`,
		`"token": "[REDACTED]", see logs`,
		`"X-Amz-Security-Token"=[REDACTED] other=ok`,
		`"X-Amz-Security-Token": '[REDACTED]' other=ok`,
		`"S0:password="[REDACTED]""`,
		"credentials: user:private.key:[REDACTED]",
		"Authorization: [REDACTED] token:\n  [REDACTED]",
		"api_key: [REDACTED] Token:\n  [REDACTED]",
		"api_key: [REDACTED] x-api-key:\n  [REDACTED] other=ok",
		`dbPassword=[REDACTED]/"` + strings.Repeat("a", 130) + `-password": "[REDACTED]"`,
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		raw, _, err := SmartTruncate(in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != in {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want unchanged", in, raw, err)
		}
	}
}

// TestBarePlaceholderMemberRules checks that hasMalformedQuotedMembers reads
// a quote after a placeholder, unquoted or in quotes, and an unquoted
// placeholder as the value of a quoted sensitive key and ":", as malformed
// members, but not the quoted value of an unquoted key, unquoted text, or a
// placeholder glued between a string that ends in a delimiter and the next.
func TestBarePlaceholderMemberRules(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		in   string
		want bool
	}{
		{`"token": [REDACTED] "note": "v"`, true},
		{`"token": [REDACTED] 'note': "v"`, true},
		{`"token": [REDACTED]; "note": "v"`, true},
		{`"token": [REDACTED] 1 "note": "v"`, true},
		{`"a": [REDACTED] x \"note\": "v"`, true},
		{`"token": '[REDACTED]' "note": "v"`, true},
		{`"token": [REDACTED]`, true},
		{`"api.key":[REDACTED] other=ok`, true},
		{`"token":"[REDACTED]"[REDACTED]"note": "v"`, true},
		{`"token": "[REDACTED]" 1 "note": "v"`, true},
		{`"token": "[REDACTED]"; "note": "v"`, true},
		{`"token": '[REDACTED]' x 'note': "v"`, true},
		{`"token": "[REDACTED]"; other="ok"`, false},
		{`"api.key":"[REDACTED]" other=ok`, false},
		{`"token": "[REDACTED]", see logs`, false},
		{`"X-Amz-Security-Token"=[REDACTED] other=ok`, false},
		{`"X-Amz-Security-Token"=[REDACTED] other="ok"`, false},
		{`"X-Amz-Security-Token": '[REDACTED]' other=ok`, false},
		{`"S0:password="[REDACTED]""`, false},
		{`"note": [REDACTED] other=ok`, false},
	} {
		if got := hasMalformedQuotedMembers(tc.in); got != tc.want {
			t.Errorf("hasMalformedQuotedMembers(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
	for _, in := range []string{`"token": "[REDACTED]"; other="ok"`, `"api.key":"[REDACTED]" other=ok`, `"token": "[REDACTED]", see logs`} {
		if !isBalancedAlreadyRedactedProse(in) {
			t.Errorf("isBalancedAlreadyRedactedProse(%q) = false, want true", in)
		}
	}
	// An escaped quote before the placeholder, which the member rules do
	// not read, still keeps the text from counting as redacted prose.
	for _, in := range []string{"\"note\" : v&\\\"token_count\\\"\v:[REDACTED]", `"}token\", sig"[REDACTED]" [REDACTED]`} {
		if isBalancedAlreadyRedactedProse(in) {
			t.Errorf("isBalancedAlreadyRedactedProse(%q) = true, want false", in)
		}
	}
}

// TestHeaderNestedKeysAndBackstopKeepOnlyFixedKeys checks that
// nestedKeyBeforeNextLine returns the fixed key at the end of a compound
// name, and no key for a quoted compound or "temp:" name or a name that
// ends in no fixed key, and that redactedKeyBytes marks, within a line, only
// the fixed key at the end of a compound or "temp:" name, so that the
// backstop flags the words before it that the header expressions removed.
func TestHeaderNestedKeysAndBackstopKeepOnlyFixedKeys(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		in, key string
	}{
		{"api_key: v1 sk_V4LUE_token:\n  v", "token:"},
		{"api_key: v1 secret123Token:\n  v", "Token:"},
		{"api_key: v1 temp:a_V4LUE_token:\n  v", "token:"},
		{"api_key: v1 x-api-key:\n  v", "x-api-key:"},
		{"api_key: v1 \"user:token\":\n  v", "\"user:token\":"},
		{"api_key: v1 password\":\n  v", "password\":"},
	} {
		valStart := len("api_key: ")
		valEnd := strings.IndexByte(tc.in, '\n')
		got, ok := nestedKeyBeforeNextLine(tc.in, valStart, valEnd)
		if want := strings.Index(tc.in, tc.key); !ok || got != want {
			t.Errorf("nestedKeyBeforeNextLine(%q) = (%d, %v), want (%d, true)", tc.in, got, ok, want)
		}
	}
	for _, in := range []string{
		"api_key: v1 temp:V4LUE:\n  v",
		"api_key: v1 \"dev-V4LUE-password\":\n  v",
		"api_key: v1 \"sk_V4LUE_token\":\n  v",
		"api_key: v1 \"temp:V4LUE\":\n  v",
		"api_key: v1 DBPASSWORD:\n  v",
	} {
		if got, ok := nestedKeyBeforeNextLine(in, len("api_key: "), strings.IndexByte(in, '\n')); ok {
			t.Errorf("nestedKeyBeforeNextLine(%q) = (%d, true), want false", in, got)
		}
	}
	marked := func(s string, compoundKeys bool) string {
		var b strings.Builder
		for i, m := range redactedKeyBytesWith(s, compoundKeys) {
			if m {
				b.WriteByte(s[i])
			} else if b.Len() > 0 && !strings.HasSuffix(b.String(), " ") {
				b.WriteByte(' ')
			}
		}
		return strings.TrimSpace(b.String())
	}
	for _, tc := range []struct {
		in, want, wantCompound string
	}{
		{"api_key: [REDACTED] sk_V4LUE_token:[REDACTED]", "api_key token", "api_key sk_V4LUE_token"},
		{"api_key: [REDACTED] temp:V4LUE:[REDACTED]", "api_key", "api_key temp:V4LUE"},
		{`api_key: [REDACTED] "dev-V4LUE-password":"[REDACTED]"`, "api_key password", "api_key dev-V4LUE-password"},
		{"api_key: [REDACTED] x/temp:V4LUE:[REDACTED]", "api_key", "api_key temp:V4LUE"},
		{"api_key: [REDACTED] secret123Token:[REDACTED]", "api_key Token", "api_key secret123Token"},
		{"credentials: user:private.key:[REDACTED]", "user:private.key", "user:private.key"},
		{"x api.key=[REDACTED]", "api.key", "api.key"},
		{"DB_PASSWORD:[REDACTED]", "DB_PASSWORD", "DB_PASSWORD"},
		{"x\n  temp:a:[REDACTED]", "temp:a", "temp:a"},
	} {
		if got := marked(tc.in, false); got != tc.want {
			t.Errorf("redactedKeyBytes(%q) marks %q, want %q", tc.in, got, tc.want)
		}
		if got := marked(tc.in, true); got != tc.wantCompound {
			t.Errorf("redactedKeyBytesWith(%q, true) marks %q, want %q", tc.in, got, tc.wantCompound)
		}
	}
	// Scanner output that keeps the words of a compound or "temp:" key
	// after a header, which the expressions on main redact with the header
	// line, fails the backstop.
	for _, tc := range []struct{ raw, scrubbed string }{
		{"Authorization: Bearer sk_V4LUE123_token:\n  unauthorized", "Authorization: [REDACTED] sk_V4LUE123_token:\n  [REDACTED]"},
		{"api_key: v1 \"dev-V4LUE123-password\":\n  v2", "api_key: [REDACTED] \"dev-V4LUE123-password\":\n  \"[REDACTED]\""},
		{"api_key: v1 temp:V4LUE123:\n  v2", "api_key: [REDACTED] temp:V4LUE123:\n  [REDACTED]"},
	} {
		if !leavesLegacyValue(tc.raw, tc.scrubbed) {
			t.Errorf("leavesLegacyValue(%q, %q) = false, want true", tc.raw, tc.scrubbed)
		}
	}
	for _, tc := range []struct {
		name string
		want int
	}{
		{"sk_V4LUE_token", 9},
		{"secret123Token", 9},
		{"x-api-key", 0},
		{"dev-V4LUE/token", 10},
		{"temp:V4LUE", -1},
		{"DBPASSWORD", -1},
		{"temp:a_token", 7},
		{"mytoken", -1},
	} {
		if got := fixedKeySuffixStart(tc.name); got != tc.want {
			t.Errorf("fixedKeySuffixStart(%q) = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestValueEndsPastContainers checks that valueEndsPast reads a "{" or "["
// container that is still open at end, or a quoted string that closes past
// it, as running past it, and returns the end of a container or quoted string
// that closes before end, so that its quotes are not read again, but not of a
// quoted string whose closing quote opens a quoted key of a container.
func TestValueEndsPastContainers(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		s         string
		end       int
		past      bool
		closedEnd int
	}{
		{`{"value": "v"}`, 5, true, 0},
		{`["a", "v"] x`, 4, true, 0},
		{`["a"] x`, 7, false, 5},
		{`{"k": {"v": 1}} x`, 17, false, 15},
		{`"a v" x`, 2, true, 0},
		{`"a" x`, 5, false, 3},
		{`'a' x`, 5, false, 3},
		{`\"a\" x`, 7, false, 5},
		{`"a\"b" x`, 8, false, 6},
		{`"S0/x:{"k": "v"}`, 16, false, 0},
		{`'S0/x:['k': 'v']`, 16, false, 0},
		{`"S0/x:{"k" x`, 12, false, 8},
		{`"S0 x:{"k": "v"}`, 3, true, 0},
		{`abc`, 2, false, 0},
		{`abc`, 0, true, 0},
	} {
		st := newQuoteScanState(len(tc.s))
		past, closedEnd := valueEndsPast(tc.s, 0, tc.end, &st)
		if past != tc.past || closedEnd != tc.closedEnd {
			t.Errorf("valueEndsPast(%q, 0, %d) = (%v, %d), want (%v, %d)", tc.s, tc.end, past, closedEnd, tc.past, tc.closedEnd)
		}
	}
}

// TestGluedPlaceholderBeforeUnredactedPair covers text in which a word glued
// to a placeholder runs into the key of a pair that is not redacted yet: the
// legacy expressions redacted that word with the value of the outer key, so
// text whose scrub keeps it fails closed, as does a chain that runs into a
// header whose value is not a placeholder as a whole, while an
// already-redacted value chain, whose pairs are all redacted, keeps its
// words.
func TestGluedPlaceholderBeforeUnredactedPair(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		{"glued_word_before_bracket_header", "password: [REDACTED]" + s1 + "[api_key: " + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"glued_word_before_quoted_authorization", "password: [REDACTED]" + s1 + `"["Authorization: Bearer ` + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"glued_word_after_quoted_temp_key", `'temp:x': [REDACTED]` + s1 + `\"["Authorization: Bearer ` + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"path_word_before_header", "password: [REDACTED]/" + s1 + "/api_key: " + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"underscore_word_before_header", "password: [REDACTED]_" + s1 + "[api_key: " + s2, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"chain_before_header_with_glued_value", "password: [REDACTED]/" + a130 + "-api_key: [REDACTED]" + s1 + " region=us", redactedSensitiveTextPlaceholder, []string{s1}},
		{"redacted_chain_and_other_pair", "password: [REDACTED]/x/token: [REDACTED] api_key=" + s1, "password: [REDACTED]/x/token: [REDACTED] api_key=[REDACTED]", []string{s1}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		if got, _ := sanitizeRawText(tc.in); got != tc.want {
			t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		raw, _, err := SmartTruncate(tc.in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", tc.label, tc.in, raw, err, tc.want)
		}
	}

	kept := func(s string) []string {
		words := legacyWords(s, true)
		glued := wordsAfterPlaceholder(s, words)
		dropGluedBeforeUnredactedPair(s, words, glued)
		var out []string
		for j, g := range glued {
			if g {
				out = append(out, s[words[j].start:words[j].end])
			}
		}
		return out
	}
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{"password: [REDACTED]SECRET[api_key: v", nil},
		{"password: [REDACTED]/x/token:\n  v", nil},
		{"password: [REDACTED]/x/token:v", nil},
		{"password: [REDACTED]/x/api_key: [REDACTED]v", []string{"v"}},
		{"password: [REDACTED]/x/api_key: [REDACTED]\nv", []string{"x", "api", "key"}},
		{"password: [REDACTED]/x/token:'[REDACTED]'", []string{"x", "token"}},
		{"[REDACTED]x", []string{"x"}},
	} {
		if got := kept(tc.in); !slices.Equal(got, tc.want) {
			t.Errorf("dropGluedBeforeUnredactedPair(%q) keeps %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestOverBudgetDanglingKeyQuoteRegressions covers over-budget values in
// which a quote closes a key, as the password" of password":S0/..., whose
// value is a bare token, a quoted string or a container that ends inside the
// value: that quote opens no string, so it does not pair with a later quote
// and skip the containers and quoted pairs after it, and a quoted string
// whose closing quote opens a quoted key of a container, as the "S0/x:{" of
// "S0/x:{"k": "SECRET"}, is not skipped as a whole. The same inputs under the
// nested-key budget redact the same secrets.
func TestOverBudgetDanglingKeyQuoteRegressions(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	a130 := strings.Repeat("a", 130)
	a10 := strings.Repeat("a", 10)
	creds := `/credentials:{"AccessKeyId": "AKIA_SECRET"}`
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		// A dangling key quote before a bare value or a closed quoted
		// value, past the nested-key budget.
		{"over_budget_bare_value_then_container", `DB_PASSWORD: password":S0/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_prefix_then_bare_value_container", `DB_PASSWORD: ` + a130 + `/password":S0` + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_closed_quoted_value_then_container", `DB_PASSWORD: password":"S0"/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_single_quoted_key_then_container", `DB_PASSWORD: password':S0/` + a130 + `/x:{'password': 'a MY_SECRET_123'}`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
		{"over_budget_single_quoted_key_then_padded_key", `DB_PASSWORD: password':S0/` + a130 + `/' token ': 'a MY_SECRET_123'`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
		{"over_budget_bare_value_then_quoted_pair", `DB_PASSWORD: password":S0/` + a130 + `/"token "= "a MY_SECRET_123"`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
		{"over_budget_closed_single_quoted_value", `DB_PASSWORD: password":'S0'/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_closed_value_with_escaped_quote", `DB_PASSWORD: password":"S\"0"/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_closed_value_with_other_quote", `DB_PASSWORD: password":'S"0'/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_closed_values_in_a_row", `DB_PASSWORD: password":"S0"/"S1"/'S2'/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_two_dangling_key_quotes", `DB_PASSWORD: password":S0/token":"S1"/` + a130 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_closed_container_then_container", `DB_PASSWORD: password":S0/` + a130 + `/x:{"a":"b"}` + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_escaped_dangling_key_quote", `DB_PASSWORD: password\":S0/` + a130 + `/credentials:{\"AccessKeyId\": \"AKIA_SECRET\"}`, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},

		// A quoted string whose closing quote opens a quoted key of a
		// container is not skipped as a whole, whether it is the value of
		// a dangling key quote or starts at a stray quote.
		{"over_budget_quoted_value_opens_keyed_object", `DB_PASSWORD: password":"S0/` + a130 + `/x:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_prefix_then_quoted_value_opens_keyed_object", `DB_PASSWORD: ` + a130 + `/password":"S0/x:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_single_quoted_value_opens_keyed_object", `DB_PASSWORD: password':'S0/` + a130 + `/x:{'k': '` + s1 + `'}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_quoted_value_opens_keyed_array", `DB_PASSWORD: ` + a130 + `/password":"S0/x:["k": "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_stray_quote_opens_keyed_object", `DB_PASSWORD: ` + a130 + `/x"S0/creds:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"over_budget_stray_quote_before_credentials", `DB_PASSWORD: ` + a130 + `/x"y` + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"over_budget_quoted_value_past_end_opens_keyed_object", `DB_PASSWORD: ` + a130 + `/password":"S0 x:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},

		// Fields after the value stay.
		{"over_budget_bare_value_keeps_fields", `DB_PASSWORD: password":` + s1 + `/` + a130 + `/x:y other=ok`, "DB_PASSWORD:[REDACTED] other=ok", []string{s1}},
		{"over_budget_closed_quoted_value_keeps_fields", `DB_PASSWORD: password":"` + s1 + `"/` + a130 + ` other=ok`, "DB_PASSWORD:[REDACTED] other=ok", []string{s1}},
		{"over_budget_closed_keyed_quoted_value_keeps_fields", `DB_PASSWORD: ` + a130 + `/password":"S0/x:{"k":"` + s2 + `"}"/z other=ok`, "DB_PASSWORD:[REDACTED] other=ok", []string{s2}},

		// Under the nested-key budget.
		{"bare_value_then_container", `DB_PASSWORD: password":S0/` + a10 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"prefix_then_bare_value_container", `DB_PASSWORD: ` + a10 + `/password":S0` + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"closed_quoted_value_then_container", `DB_PASSWORD: password":"S0"/` + a10 + creds, "DB_PASSWORD:[REDACTED]", []string{"AKIA_SECRET"}},
		{"single_quoted_key_then_container", `DB_PASSWORD: password':S0/` + a10 + `/x:{'password': 'a MY_SECRET_123'}`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
		{"single_quoted_key_then_padded_key", `DB_PASSWORD: password':S0/` + a10 + `/' token ': 'a MY_SECRET_123'`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
		{"bare_value_then_quoted_pair", `DB_PASSWORD: password":S0/` + a10 + `/"token "= "a MY_SECRET_123"`, "DB_PASSWORD:[REDACTED]", []string{"MY_SECRET_123"}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		if got, _ := sanitizeRawText(tc.in); got != tc.want {
			t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		raw, _, err := SmartTruncate(tc.in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", tc.label, tc.in, raw, err, tc.want)
		}
	}

	// Redacted text and text without a sensitive key stay byte-identical.
	for _, in := range []string{
		"DB_PASSWORD:[REDACTED] other=ok",
		`DB_PASSWORD:"[REDACTED]" other=ok`,
		"password: [REDACTED]/" + a130 + "/token:[REDACTED]",
		"note: " + a130 + `/x"y/ok:{"a":1} other=ok`,
		"note: " + a130 + `/x"y/ok:{"a": "b"}/"c":'d' other=ok`,
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		raw, _, err := SmartTruncate(in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != in {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want unchanged", in, raw, err)
		}
	}
}

// TestEscapedQuoteLineStartKeys covers redacted pairs whose key starts a
// line after an escaped opening quote, as the \"DB_PASSWORD\" of
// credentials:\n  \"DB_PASSWORD\"=\"[REDACTED]\": the backstop marks the
// whole key, as it does for an unescaped quote, so the redacted text stays
// byte-identical, while a value that is not redacted yet is redacted and a
// name after an escaped backslash, or within a line, is still narrowed to
// the fixed key at its end.
func TestEscapedQuoteLineStartKeys(t *testing.T) {
	runInParallel(t)
	// Redacted text stays byte-identical.
	for _, in := range []string{
		"credentials:\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n  \\'DB_PASSWORD\\'=\\'[REDACTED]\\'",
		"credentials:\n  \\\"OPENAI_API_KEY\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n  \\\"sk_v1_token\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n\t\\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"",
		"credentials:\r\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n\\\"DB_PASSWORD\\\":\\\"[REDACTED]\\\"",
		"credentials:\n  \\\"C:\\\\ssl\\\\password\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n  \\'temp:a\\'=\\'[REDACTED]\\'",
		"credentials:\n  \\\"temp:a\\\"=\\\"[REDACTED]\\\"",
		"credentials:\n  \\'user:temp:refresh_token\\'=\\'[REDACTED]\\'",
		"credentials:\n  \"DB_PASSWORD\"=\"[REDACTED]\"",
		"x \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"",
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		raw, _, err := SmartTruncate(in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != in {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want unchanged", in, raw, err)
		}
	}

	// A value that is not redacted yet is redacted.
	const s1 = "LEAK_SECRET_ONE"
	for _, tc := range []struct{ label, in, want string }{
		{"escaped_double_quoted_compound_key", "credentials:\n  \\\"DB_PASSWORD\\\"=\\\"" + s1 + "\\\"", "credentials:\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\""},
		{"escaped_single_quoted_temp_key", "credentials:\n  \\'temp:token\\'=\\'" + s1 + "\\'", "credentials:\n  \\'temp:token\\'=\\'[REDACTED]\\'"},
		{"escaped_double_quoted_api_key", "credentials:\n  \\\"OPENAI_API_KEY\\\":\\\"" + s1 + "\\\"", "credentials:\n  \\\"OPENAI_API_KEY\\\":\\\"[REDACTED]\\\""},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, s1)
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		if got, _ := sanitizeRawText(tc.in); got != tc.want {
			t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		raw, _, err := SmartTruncate(tc.in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", tc.label, tc.in, raw, err, tc.want)
		}
	}

	// Only spaces or tabs and an opening quote, escaped by a single
	// backslash or not, may stand before a name that starts a line.
	for _, tc := range []struct {
		s    string
		want bool
	}{
		{"credentials:\n  \\\"DB_PASSWORD", true},
		{"credentials:\n  \\'temp:a", true},
		{"credentials:\n\t\\\"DB_PASSWORD", true},
		{"credentials:\r\n\\\"DB_PASSWORD", true},
		{"\\\"DB_PASSWORD", true},
		{"credentials:\n  \"DB_PASSWORD", true},
		{"credentials:\n  DB_PASSWORD", true},
		{"credentials:\n  \\\\\"DB_PASSWORD", false},
		{"credentials:\n  \\\\\\\"DB_PASSWORD", false},
		{"credentials:\n  \\\"\\\"DB_PASSWORD", false},
		{"credentials:\n  \\DB_PASSWORD", false},
		{"credentials:\n  x\\\"DB_PASSWORD", false},
		{"x \\\"DB_PASSWORD", false},
	} {
		i := strings.LastIndex(tc.s, "DB_PASSWORD")
		if i < 0 {
			i = strings.LastIndex(tc.s, "temp:a")
		}
		if got := startsLineAt(tc.s, i); got != tc.want {
			t.Errorf("startsLineAt(%q, %d) = %v, want %v", tc.s, i, got, tc.want)
		}
	}

	// The backstop marks the whole name at the start of a line, and only
	// the fixed key at its end after an escaped backslash or within a line.
	marked := func(s string) string {
		var b strings.Builder
		for i, m := range redactedKeyBytes(s) {
			if m {
				b.WriteByte(s[i])
			}
		}
		return b.String()
	}
	for _, tc := range []struct{ in, want string }{
		{"credentials:\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", "DB_PASSWORD\\"},
		{"credentials:\n  \\'temp:a\\'=\\'[REDACTED]\\'", "temp:a\\"},
		{"credentials:\n  \\\"OPENAI_API_KEY\\\"=\\\"[REDACTED]\\\"", "OPENAI_API_KEY\\"},
		{"credentials:\n  \\\\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", "PASSWORD\\"},
		{"x \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", "PASSWORD\\"},
		{"x \\'temp:a\\'=\\'[REDACTED]\\'", ""},
	} {
		if got := marked(tc.in); got != tc.want {
			t.Errorf("redactedKeyBytes(%q) marks %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestAdversarialScalingDanglingKeyQuoteShapes checks that over-budget values
// with dangling key quotes before bare and closed quoted values, quoted
// strings that open keyed containers, escaped line-start quoted keys and
// backslash runs before dangling key quotes stay linear through all three
// entry points.
func TestAdversarialScalingDanglingKeyQuoteShapes(t *testing.T) {
	runInParallel(t)
	a135 := strings.Repeat("a", 135)
	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"over-budget dangling key quotes before bare values", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`/password":S0/`, n) + `credentials:{"k": "v"}`
		}},
		{"over-budget dangling key quotes before closed quoted values", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`/password":"S0"/`, n) + `credentials:{"k": "v"}`
		}},
		{"over-budget quoted values that open keyed containers", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`/password":"S0/x:{"k":"v"}"/`, n) + ` other=ok`
		}},
		{"over-budget stray quotes before keyed containers", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat(a135+`/x"S0/creds:{"k":`, n) + ` "v"}`
		}},
		{"over-budget quote runs before a keyed container", func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`"x{"`, n) + `k": "v"}`
		}},
		{"over-budget backslash runs before dangling key quotes", func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`\`, n) + `password":S0/credentials:{"k": "v"}`
		}},
		{"escaped line-start quoted keys", func(n int) string {
			return "credentials:\n" + strings.Repeat("  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"\n", n)
		}},
		{"escaped quotes after backslash runs at line starts", func(n int) string {
			return "credentials:\n" + strings.Repeat("  "+strings.Repeat(`\`, 9)+"\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"\n", n)
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
}

// TestLegacyPatternBackstopOnUnchangedText covers text that the scanners
// leave as it is while the legacy expressions redact a word of it, as the
// V4LUE glued to a placeholder before a key name that holds a vertical tab or
// a no-break space: such text fails closed in all three entry points, while
// redacted text that the expressions leave as it is stays byte-identical.
func TestLegacyPatternBackstopOnUnchangedText(t *testing.T) {
	runInParallel(t)
	for _, in := range []string{
		"log: '\vtemp:abc'=[REDACTED]V4LUE/'\vDB_PASSWORD'=\"[REDACTED]\"",
		"config:\n  \"password\":[REDACTED]V4LUE/\vtemp:abc:[REDACTED]",
		"log: \"\vpassword\"=[REDACTED]V4LUE/\" api_Key\"\v: \"[REDACTED]\"",
		"credentials:\n  '\vapi_key'=[REDACTED]V4LUE/temp:abc : [REDACTED]",
	} {
		if got := scrubErrorTokens(in); got != in {
			t.Errorf("scrubErrorTokens(%q) = %q, want unchanged", in, got)
		}
		assertRedactedIdempotent(t, "unchanged_text", in, "V4LUE")
		if got := redactErrorMessage(in); got != redactedSensitiveTextPlaceholder {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, redactedSensitiveTextPlaceholder)
		}
		if got, _ := sanitizeRawText(in); got != redactedSensitiveTextPlaceholder {
			t.Errorf("sanitizeRawText(%q) = %q, want %q", in, got, redactedSensitiveTextPlaceholder)
		}
		raw, _, err := SmartTruncate(in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != redactedSensitiveTextPlaceholder {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want %q", in, raw, err, redactedSensitiveTextPlaceholder)
		}
	}

	for _, in := range []string{
		"config:\n  \"password\":[REDACTED]",
		"log: 'api_key'=[REDACTED] other=ok",
		"credentials:\n  'api_key'=[REDACTED]/token : [REDACTED]",
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		raw, _, err := SmartTruncate(in, -1)
		var got string
		if err != nil || json.Unmarshal(raw, &got) != nil || got != in {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want unchanged", in, raw, err)
		}
	}
}

// assertAllEntryPoints checks that redactErrorMessage, sanitizeRawText and
// SmartTruncate all return want for in, and that each is idempotent and keeps
// none of forbidden (see assertRedactedIdempotent).
func assertAllEntryPoints(t *testing.T, label, in, want string, forbidden ...string) {
	t.Helper()
	assertRedactedIdempotent(t, label, in, forbidden...)
	if got := redactErrorMessage(in); got != want {
		t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", label, in, got, want)
	}
	if got, _ := sanitizeRawText(in); got != want {
		t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", label, in, got, want)
	}
	raw, _, err := SmartTruncate(in, -1)
	var got string
	if err != nil || json.Unmarshal(raw, &got) != nil || got != want {
		t.Errorf("[%s] SmartTruncate(%q) = (%s, %v), want %q", label, in, raw, err, want)
	}
}

// assertUnchangedAllEntryPoints checks that redactErrorMessage,
// sanitizeRawText and SmartTruncate all leave in byte-identical.
func assertUnchangedAllEntryPoints(t *testing.T, in string) {
	t.Helper()
	if got := redactErrorMessage(in); got != in {
		t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
	}
	if got, lost := sanitizeRawText(in); got != in || lost {
		t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
	}
	raw, _, err := SmartTruncate(in, -1)
	var got string
	if err != nil || json.Unmarshal(raw, &got) != nil || got != in {
		t.Errorf("SmartTruncate(%q) = (%s, %v), want unchanged", in, raw, err)
	}
}

// TestStrayQuoteContainersUnderBudget covers values under the nested-key
// budget with a container after a delimiter that is still open where the
// value seems to end, possibly after a stray quote whose quoted string ends in
// the "{" or "[" of that container, as the "S0/creds:{" of
// DB_PASSWORD: x"S0/creds:{"k": "SECRET"}: the string is not skipped as a
// whole, and the container runs past the value and is redacted with it, as
// past the budget.
func TestStrayQuoteContainersUnderBudget(t *testing.T) {
	runInParallel(t)
	const s1 = "LEAK_SECRET_ONE"
	for _, tc := range []struct{ label, in, want string }{
		{"stray_quote_keyed_object_compound_key", `DB_PASSWORD: x"S0/creds:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_keyed_object_exact_key", `password: x"S0/creds:{"k": "` + s1 + `"}`, "password:[REDACTED]"},
		{"stray_quote_array_of_strings", `DB_PASSWORD: x"S0/creds:["k", "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_array_single_element", `DB_PASSWORD: x"S0/creds:["` + s1 + `" ]`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_array_without_spaces", `password: x"S0/creds:["a","` + s1 + `"]`, "password:[REDACTED]"},
		{"stray_quote_unquoted_object", `DB_PASSWORD: x"S0/creds:{k: ` + s1 + `}`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_single_quoted_object", `api_key=x"S0/creds:{'k': '` + s1 + `'}`, "api_key=[REDACTED]"},
		{"stray_quote_object_keeps_fields", `DB_PASSWORD: x"S0/creds:{"k": "` + s1 + `"} other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"open_object_after_path", `DB_PASSWORD: x/creds:{"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"open_object_after_delimiter", `password: x:{"k": "` + s1 + `"}`, "password:[REDACTED]"},
		{"open_array_after_delimiter", `password: x:["a", "` + s1 + `"]`, "password:[REDACTED]"},
		{"quoted_value_opens_array", `password: "a:token:["a", "` + s1 + `"]"`, `password:"[REDACTED]"`},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, s1)
	}

	// Text without a sensitive key stays byte-identical.
	for _, in := range []string{
		`note: x"S0/creds:{"k": "v"} other=ok`,
		`note: x/creds:{"k": "v"} other=ok`,
		`note: x"S0/creds:["a", "b"] other=ok`,
		`DB_PASSWORD:[REDACTED] other=ok`,
	} {
		assertUnchangedAllEntryPoints(t, in)
	}
}

// TestQuotedContainerMembersWithAnyBytes covers quoted strings whose closing
// quote opens a quoted key that holds bytes other than compound key bytes,
// as the "user:foo", "a b", "a/b" or "a\"b" key of a JSON object: the key is
// read as one (see isQuotedMemberAt), so the string before it is not skipped
// as a whole, past the nested-key budget and under it.
func TestQuotedContainerMembersWithAnyBytes(t *testing.T) {
	runInParallel(t)
	const s1 = "LEAK_SECRET_ONE"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct{ label, in, want string }{
		{"over_budget_stray_quote_colon_key", `DB_PASSWORD: ` + a135 + `/x"S0/creds:{"user:foo": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote_spaced_key", `DB_PASSWORD: ` + a135 + `/x"S0/creds:{"a b": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote_slash_key", `DB_PASSWORD: ` + a135 + `/x"S0/creds:{"a/b": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote_equals_key", `DB_PASSWORD: ` + a135 + `/x"S0/creds:{"a b"= "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote_escaped_quote_key", `DB_PASSWORD: ` + a135 + `/x"S0/creds:{"a\"b": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_dangling_quote_colon_key", `DB_PASSWORD: ` + a135 + `/S0/creds":{"user:foo": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_dangling_quoted_value_colon_key", `DB_PASSWORD: ` + a135 + `/password":"S0/creds:{"user:foo": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_dangling_quoted_value_padded_key", `DB_PASSWORD: ` + a135 + `/password":"S0/creds:{" a ": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_slash_key", `DB_PASSWORD: x"S0/creds:{"a/b": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"quoted_value_opens_colon_key", `password: "a:token:{"user:foo": "` + s1 + `"}"`, `password:"[REDACTED]"`},
		{"quoted_value_opens_spaced_key", `password: "a:token:{"a b": "` + s1 + `"}"`, `password:"[REDACTED]"`},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, s1)
	}

	// Text without a sensitive key stays byte-identical.
	for _, in := range []string{
		"note: " + a135 + `/x"y/ok:{"a b": "c"} other=ok`,
		"note: " + a135 + `/x"y/ok:{"user:foo": "c"}/"d":'e' other=ok`,
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	for _, tc := range []struct {
		s    string
		open byte
		want bool
	}{
		{`"k": 1}`, '{', true},
		{`"user:foo": 1}`, '{', true},
		{`"a b" = 1}`, '{', true},
		{`"a/b":1}`, '{', true},
		{`'k': 1}`, '{', true},
		{`"a\"b": 1}`, '{', true},
		{"\"k\"\n: 1}", '{', true},
		{"\"a\nb\": 1}", '{', true},
		{"\"a\tb\": 1}", '{', true},
		{`"a", 1]`, '[', true},
		{`"a"]`, '[', true},
		{`"", 1]`, '[', true},
		{`"k": 1]`, '[', true},
		{`"": 1}`, '{', false},
		{`"a", 1}`, '{', false},
		{`"k"`, '{', false},
		{`"k" x: 1}`, '{', false},
		{`"k: 1}`, '{', false},
		{`"a\`, '{', false},
	} {
		if got := isQuotedMemberAt(tc.s, 0, tc.open); got != tc.want {
			t.Errorf("isQuotedMemberAt(%q, 0, %q) = %v, want %v", tc.s, tc.open, got, tc.want)
		}
	}
}

// TestPaddedLineStartQuotedKeys covers quoted keys padded with spaces at the
// start of a line, whatever their quote (", ', \" or \') and delimiter: a
// redacted pair stays byte-identical instead of being redacted with the key
// of the line before it, and a value that is not redacted yet is redacted.
func TestPaddedLineStartQuotedKeys(t *testing.T) {
	runInParallel(t)
	for _, in := range []string{
		"credentials:\n  \"   DB_PASSWORD   \"=\"[REDACTED]\"",
		"credentials:\n  '   DB_PASSWORD   ':'[REDACTED]'",
		"credentials:\n  \\\"   DB_PASSWORD   \\\"=\\\"[REDACTED]\\\"",
		"credentials:\n  \\'   DB_PASSWORD   \\'=\\'[REDACTED]\\'",
		"credentials:\n  \"   DB_PASSWORD   \" = \"[REDACTED]\"",
		"credentials:\n  '  token  ': '[REDACTED]'",
		"credentials:\n  \"  token  \"=[REDACTED]",
		"credentials:\n  \"   DB_PASSWORD   \":\"[REDACTED]\"",
		"credentials:\n  \"   DB_PASSWORD   \"=\"[REDACTED]\"\n  '  token  ':'[REDACTED]'",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	const s1 = "LEAK_SECRET_ONE"
	for _, tc := range []struct{ label, in, want string }{
		{"padded_double_quoted_equals", "credentials:\n  \"   DB_PASSWORD   \"=\"" + s1 + "\"", "credentials:\n  \"DB_PASSWORD\"=\"[REDACTED]\""},
		{"padded_single_quoted_colon", "credentials:\n  '   DB_PASSWORD   ':'" + s1 + "'", "credentials:\n  'DB_PASSWORD':'[REDACTED]'"},
		{"padded_escaped_double_quoted", "credentials:\n  \\\"   DB_PASSWORD   \\\"=\\\"" + s1 + "\\\"", "credentials:\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\""},
		{"padded_escaped_single_quoted", "credentials:\n  \\'   DB_PASSWORD   \\'=\\'" + s1 + "\\'", "credentials:\n  \\'DB_PASSWORD\\'=\\'[REDACTED]\\'"},
		{"padded_bare_value_keeps_fields", "credentials:\n  \"  token  \"=" + s1 + " other=ok", "credentials:\n  \"token\"=[REDACTED] other=ok"},
		{"padded_temp_key", "credentials:\n  '  temp:abc  ' = '" + s1 + "'", "credentials:\n  'temp:abc'='[REDACTED]'"},
		{"padded_name_that_is_not_a_key", "credentials:\n  \"  note  \"=" + s1, "credentials:\"[REDACTED]\""},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, s1)
	}
}

// TestGluedPlaceholderBeforeAnyUnredactedPair covers words glued to a
// placeholder in a run that holds a pair that is not redacted before a pair
// that is, as the api_key:SECRET of
// password: [REDACTED]/SECRET/api_key:SECRET,other=[REDACTED], or whose key
// ends the run before whitespace and a delimiter, as the token = SECRET of
// password = [REDACTED]SECRET/token = SECRET: the legacy expressions redacted
// those words, so such text fails closed, while already-redacted chains keep
// their words, also before a header whose redacted value ends its line.
func TestGluedPlaceholderBeforeAnyUnredactedPair(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct{ label, in, want string }{
		{"redacted_pair_after_value_stop", "password: [REDACTED]/" + s1 + "/api_key:" + s2 + ",other=[REDACTED]", redactedSensitiveTextPlaceholder},
		{"redacted_pair_later_in_run", "password: [REDACTED]" + s1 + "[api_key:" + s2 + "/token:[REDACTED]", redactedSensitiveTextPlaceholder},
		{"redacted_query_pair_in_uri_value", "password: [REDACTED]/" + s1 + "/token:https://" + s2 + "@example.com?x=[REDACTED]", redactedSensitiveTextPlaceholder},
		{"spaced_equals_after_run", "password = [REDACTED]" + s1 + "/token = " + s2, redactedSensitiveTextPlaceholder},
		{"spaced_colon_after_run", "password: [REDACTED]" + s1 + "/token : " + s2, redactedSensitiveTextPlaceholder},
		{"tab_before_equals_after_run", "password: [REDACTED]" + s1 + "/token\t= " + s2, redactedSensitiveTextPlaceholder},
		{"redacted_chain_before_redacted_header", "password: [REDACTED]/" + a130 + "-api_key: [REDACTED]\napi_key=" + s1, "password: [REDACTED]/" + a130 + "-api_key: [REDACTED]\napi_key=[REDACTED]"},
		{"redacted_chain_then_other_pair", "x=[REDACTED]/y=[REDACTED] api_key=" + s1, "x=[REDACTED]/y=[REDACTED] api_key=[REDACTED]"},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, s1)
	}

	for _, in := range []string{
		"x=[REDACTED]/y=[REDACTED]",
		"password: [REDACTED]/x/token = [REDACTED] other=ok",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	kept := func(s string) []string {
		words := legacyWords(s, true)
		glued := wordsAfterPlaceholder(s, words)
		dropGluedBeforeUnredactedPair(s, words, glued)
		var out []string
		for j, g := range glued {
			if g {
				out = append(out, s[words[j].start:words[j].end])
			}
		}
		return out
	}
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{"password: [REDACTED]/S/api_key:v,other=[REDACTED]", nil},
		{"password: [REDACTED]S[api_key:v/token:[REDACTED]", nil},
		{"password: [REDACTED]/S/token:https://v@example.com?x=[REDACTED]", nil},
		{"password = [REDACTED]S/token = v", nil},
		{"password = [REDACTED]S/token\n= v", nil},
		{"password = [REDACTED]S/token = [REDACTED]", []string{"S", "token"}},
		{"x=[REDACTED]/y=[REDACTED]", []string{"y"}},
		{"password: [REDACTED]/x/token:'[REDACTED]' other=v", []string{"x", "token"}},
	} {
		if got := kept(tc.in); !slices.Equal(got, tc.want) {
			t.Errorf("dropGluedBeforeUnredactedPair(%q) keeps %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestURIComponentsCheckedAsText covers credentials in the components of a
// URI: each decoded component is scrubbed until it stops changing and checked
// against the legacy expressions and its escapes, as scrubErrorText checks
// text (see redactURIComponent), so a URI that is formatted on its own, as
// the file URI of a content part, keeps no word that the expressions redact;
// and the legacy-pattern backstop reads the components of a URI as the
// expressions on main did (see legacyRedactURIComponent).
func TestURIComponentsCheckedAsText(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	failedQuery := "https://example.com/cb?err=" + url.QueryEscape(redactedSensitiveTextPlaceholder)
	formatURI := func(uri string) string {
		parts := FormatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: uri}}}}, -1)
		if len(parts) != 1 {
			t.Fatalf("FormatContentParts(%q) = %v, want one part", uri, parts)
		}
		got, _ := parts[0]["uri"].(string)
		return got
	}
	for _, tc := range []struct{ label, uri, wantFormatted string }{
		{"query_glued_word_before_redacted_pair_and_other_pair", "https://example.com/cb?err=password+%3D+%5BREDACTED%5D" + s1 + "%2Ftoken+%3D+%5BREDACTED%5D%2Capi_key%3D" + s2, failedQuery},
		{"query_glued_word_before_spaced_pair", "https://example.com/cb?err=password+%3D+%5BREDACTED%5D" + s1 + "%2Ftoken+%3D+" + s2, failedQuery},
		{"query_glued_word_before_bracket_header", "https://example.com/cb?err=password%3A+%5BREDACTED%5D" + s1 + "%5Bapi_key%3A+" + s2, failedQuery},
		{"query_escaped_tab_in_quoted_key", "https://example.com/cb?err=x+%22client_secret%22%27%3A+token%5Ct%3A%27" + s1 + "%27+%3D+" + s2, failedQuery},
		// The expressions on main fail this text closed as a whole, so it
		// fails closed as error text, while the file URI, whose components
		// are checked one at a time, keeps the other parameter.
		{"query_that_main_fails_closed_as_a_whole", "https://example.com/cb?err=v&q=password%22%3A%5C%22%22+DB_PASSWORD+%22+%3D+" + s2 + "%5C%22", "https://example.com/cb?err=v&q=password%22%3A%5BREDACTED%5D"},
	} {
		assertAllEntryPoints(t, tc.label, tc.uri, redactedSensitiveTextPlaceholder, s1, s2)
		got := formatURI(tc.uri)
		if got != tc.wantFormatted {
			t.Errorf("[%s] FormatContentParts file URI %q = %q, want %q", tc.label, tc.uri, got, tc.wantFormatted)
		}
		if again := formatURI(got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	// A component that the scrub redacts in part keeps the rest, and
	// redacted or harmless components stay byte-identical.
	for _, tc := range []struct{ uri, want string }{
		{"https://example.com/cb?q=Bearer+" + s1, "https://example.com/cb?q=Bearer+%5BREDACTED%5D"},
		{"https://example.com/a/password%3A+%5BREDACTED%5D" + s1 + "%5Bapi_key%3A+" + s2 + "/c", "https://example.com/a/password:%5BREDACTED%5D/c"},
		{"https://example.com/a#password%3A+%5BREDACTED%5D" + s1 + "%5Bapi_key%3A+" + s2, "https://example.com/a#password:%5BREDACTED%5D"},
		{"https://example.com/cb?q=hello+world", "https://example.com/cb?q=hello+world"},
		{"https://example.com/cb?q=password+%3D+%5BREDACTED%5D&token=%5BREDACTED%5D", "https://example.com/cb?q=password+%3D+%5BREDACTED%5D&token=%5BREDACTED%5D"},
	} {
		got := formatURI(tc.uri)
		if got != tc.want {
			t.Errorf("FormatContentParts file URI %q = %q, want %q", tc.uri, got, tc.want)
		}
		if again := formatURI(got); again != got {
			t.Errorf("FormatContentParts second pass(%q) = %q, want unchanged", got, again)
		}
	}
	for _, in := range []string{
		"https://example.com/cb?q=hello+world",
		"https://example.com/cb?q=password+%3D+%5BREDACTED%5D&token=%5BREDACTED%5D",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	// The decoded component scrub is a fixed point.
	for _, in := range []string{
		"password = [REDACTED]" + s1 + "/token = [REDACTED],api_key=" + s2,
		"x \"client_secret\"': token\\t:'" + s1 + "' = " + s2,
		"Bearer " + s1,
		"hello world",
	} {
		got := scrubURIComponentText(in)
		if strings.Contains(got, s1) || strings.Contains(got, s2) {
			t.Errorf("scrubURIComponentText(%q) = %q, kept a secret", in, got)
		}
		if again := scrubURIComponentText(got); again != got {
			t.Errorf("scrubURIComponentText(%q) = %q, then %q, want a fixed point", in, got, again)
		}
	}

	// The legacy scrubs read URI components as the expressions on main did
	// (outputs of scrubErrorTokens on main).
	for _, tc := range []struct{ in, want string }{
		{"https://example.com/cb?err=password+%3D+%5BREDACTED%5DSECRET1%2Ftoken+%3D+%5BREDACTED%5D%2Capi_key%3DSECRET2", "https://example.com/cb?err=password%3D%5BREDACTED%5D+%3D+%5BREDACTED%5D%2Capi_key%3D%5BREDACTED%5D"},
		{"https://example.com/cb?err=password+%3D+%5BREDACTED%5DSECRET1%2Ftoken+%3D+SECRET2", "https://example.com/cb?err=password%3D%5BREDACTED%5D+%3D+SECRET2"},
		{"https://example.com/cb?err=password%3A+%5BREDACTED%5DSECRET1%5Bapi_key%3A+SECRET2", "https://example.com/cb?err=password%3A%5BREDACTED%5D+%5BREDACTED%5D"},
		{"https://example.com/cb?err=x+%22client_secret%22%27%3A+token%5Ct%3A%27SECRET1%27+%3D+SECRET2", "https://example.com/cb?err=x+%22client_secret%22%27%3A+token%5Ct%3A%27SECRET1%27+%3D+SECRET2"},
		{"https://example.com/cb?q=Bearer+SECRET1&r=Basic+U0VDUkVUMTpTRUNSRVQy", "https://example.com/cb?q=Bearer+%5BREDACTED%5D&r=Basic+%5BREDACTED%5D"},
		{"https://example.com/a/x?err=client_secret%3A%0A++sk_v1_token%3A+SECRET1", "https://example.com/a/x?err=client_secret%3A%5BREDACTED%5D+SECRET1"},
		{"https://example.com/a/x?err=DB_PASSWORD%3A+SECRET1", "https://example.com/a/x?err=DB_PASSWORD%3A+SECRET1"},
		{"see https://example.com/cb?err=Authorization%3A+Bearer+SECRET1 and password=SECRET2", "see https://example.com/cb?err=Authorization%3A+%5BREDACTED%5D and password=[REDACTED]"},
	} {
		if got := legacyScrubErrorTokens(tc.in); got != tc.want {
			t.Errorf("legacyScrubErrorTokens(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if got, want := compoundLegacyScrubErrorTokens("https://example.com/a/x?err=DB_PASSWORD%3A+SECRET1"), "https://example.com/a/x?err=DB_PASSWORD%3A%5BREDACTED%5D"; got != want {
		t.Errorf("compoundLegacyScrubErrorTokens = %q, want %q", got, want)
	}
	// Without a compound key, the scrub with every key that isSensitiveKey
	// accepts reads URI components as the expressions on main did too.
	for _, in := range []string{
		"https://example.com/cb?err=password+%3D+%5BREDACTED%5DSECRET1%2Ftoken+%3D+SECRET2",
		"https://example.com/cb?err=password%3A+%5BREDACTED%5DSECRET1%5Bapi_key%3A+SECRET2",
		"https://example.com/cb?err=x+%22client_secret%22%27%3A+token%5Ct%3A%27SECRET1%27+%3D+SECRET2",
	} {
		if got, want := compoundLegacyScrubErrorTokens(in), legacyScrubErrorTokens(in); got != want {
			t.Errorf("compoundLegacyScrubErrorTokens(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestKnownSentinelLengths checks that maxKnownSentinelLen is the length of
// the longest known placeholder, so that knownSentinelLenAt finds each of
// them in text.
func TestKnownSentinelLengths(t *testing.T) {
	runInParallel(t)
	longest := 0
	for _, s := range knownSentinels {
		if !isKnownSentinel(s) {
			t.Errorf("isKnownSentinel(%q) = false", s)
		}
		if len(s) > maxKnownSentinelLen {
			t.Errorf("len(%q) = %d > maxKnownSentinelLen = %d", s, len(s), maxKnownSentinelLen)
		}
		longest = max(longest, len(s))
		if got := knownSentinelLenAt("x"+s+"y", 1); got != len(s) {
			t.Errorf("knownSentinelLenAt(%q, 1) = %d, want %d", "x"+s+"y", got, len(s))
		}
	}
	if longest != maxKnownSentinelLen || maxKnownSentinelLen != len(sanitizeBudgetPlaceholder) {
		t.Errorf("maxKnownSentinelLen = %d, want the longest length %d (%q)", maxKnownSentinelLen, longest, sanitizeBudgetPlaceholder)
	}
	if isKnownSentinel(sanitizeBudgetPlaceholder+"]") || knownSentinelLenAt("[REDACTED", 0) != 0 {
		t.Error("isKnownSentinel or knownSentinelLenAt accepted a text that is no placeholder")
	}
}

// TestAdversarialScalingContainerGluedAndURIShapes checks that stray quotes
// before open containers under the budget, quoted members with any bytes,
// padded line-start keys, glued placeholder runs with many delimiters, URI
// components checked as text, and over-budget values whose nested containers
// close at the end of the text after dangling key quotes stay linear through
// all three entry points, and through FormatContentParts for URIs.
func TestAdversarialScalingContainerGluedAndURIShapes(t *testing.T) {
	runInParallel(t)
	a135 := strings.Repeat("a", 135)
	for _, shape := range []struct {
		name string
		uri  bool
		mk   func(n int) string
	}{
		{"stray quotes before open containers under the budget", false, func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"S0/creds:{"k": "v"} `, n)
		}},
		{"over-budget quoted members with any bytes", false, func(n int) string {
			return strings.Repeat("DB_PASSWORD: "+a135+`/x"S0/creds:{"user:a b/c": "v"}`+"\n", n)
		}},
		{"escaped quotes in a quoted member", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`"x{"a\"b`, n) + `": "v"}`
		}},
		{"padded line-start keys", false, func(n int) string {
			return "credentials:\n" + strings.Repeat("  '   DB_PASSWORD   ':'[REDACTED]'\n", n)
		}},
		{"glued placeholder runs with many delimiters", false, func(n int) string {
			return "password: [REDACTED]" + strings.Repeat("/x:[REDACTED]", n) + " api_key=v"
		}},
		{"glued placeholder runs before spaced delimiters", false, func(n int) string {
			return strings.Repeat("password = [REDACTED]x/token = [REDACTED] ", n) + "api_key=v"
		}},
		{"URI query components checked as text", true, func(n int) string {
			return "https://example.com/cb?" + strings.Repeat("err=password%3A+%5BREDACTED%5Dx%5Bapi_key%3A+v&", n) + "z=1"
		}},
		{"one long URI query component", true, func(n int) string {
			return "https://example.com/cb?err=" + strings.Repeat("password+%3D+%5BREDACTED%5Dx%2F", n)
		}},
		// Each level holds two dangling key quotes, whose quotes pair up in
		// the containers, so every container closes, at the end of the text;
		// a scan that read the container after each dangling key quote again
		// would take quadratic time.
		{"over-budget nested containers closing at the end", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`/password":x/token":{`, n) + strings.Repeat("}", n)
		}},
		{"over-budget nested single-quoted containers closing at the end", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`/password':x/token':{`, n) + strings.Repeat("}", n)
		}},
		{"over-budget nested arrays closing at the end", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`/password":x/token":[`, n) + strings.Repeat("]", n)
		}},
		{"stray escaped double quote with inner escaped quotes before object", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\\"" + strings.Repeat("a\\\\\\\"", n) + "/creds:{ \"k\": \"v\"}"
		}},
		{"stray escaped single quote with inner escaped quotes before object", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\'" + strings.Repeat("a\\\\\\'", n) + "/creds:{ 'k': 'v'}"
		}},
		{"stray escaped double quote with inner escaped quotes before escaped object", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\\"" + strings.Repeat("a\\\\\\\"", n) + "/creds:{ \\\"k\\\": \\\"v\\\"}"
		}},
		{"stray escaped single quote with inner escaped quotes before escaped object", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\'" + strings.Repeat("a\\\\\\'", n) + "/creds:{ \\'k\\': \\'v\\'}"
		}},
		{"stray escaped double quote with inner escaped quotes in array", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\\":[" + strings.Repeat("1,a\\\\\\\",", n) + "\\\"v\\\"]"
		}},
		{"stray escaped single quote with inner escaped quotes in array", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\':[" + strings.Repeat("1,a\\\\\\',", n) + "\\'v\\']"
		}},
		{"stray escaped double quote with inner escaped quotes before long array", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\\"" + strings.Repeat("a\\\\\\\"", n) + ":[" + strings.Repeat("1,", n) + "\\\"v\\\"]"
		}},
		{"stray escaped single quote with inner escaped quotes before long array", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + "/x\\'" + strings.Repeat("a\\\\\\'", n) + ":[" + strings.Repeat("1,", n) + "\\'v\\']"
		}},
		{"alternating inner-quote parity across escaped quotes before object", false, func(n int) string {
			return "token:x\\\" " + strings.Repeat("' token:a\\\\\\\" ' token:b\\\\\\\" ", n) + "{ \\\"k\\\": \\\"v\\\"}"
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
				if shape.uri {
					_ = FormatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: s}}}}, -1)
				}
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
}

// TestGluedPlaceholderBeforeUnredactedValues covers words glued to a
// placeholder in a run whose pair has a value that is not a redaction
// placeholder as a whole: one after an empty quoted string or a line break,
// a placeholder that stands for content, as [BINARY DATA], and a placeholder
// with more text in its quotes, glued to it, or after spaces in a header
// value. The expressions on main redacted the glued words with the value of
// the outer key, so such text fails closed, also when the scrub leaves it as
// it is, while already-redacted chains whose pairs end cleanly keep their
// words.
func TestGluedPlaceholderBeforeUnredactedValues(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	for _, tc := range []struct{ label, in string }{
		{"empty_quotes_before_placeholder", "password: [REDACTED]" + s1 + `/api_key: "" [REDACTED] ` + s2},
		{"line_break_before_placeholder", "password: [REDACTED]" + s1 + "/api_key:\n  [REDACTED] " + s2},
		{"content_placeholder", "password: [REDACTED]" + s1 + "/api_key: [BINARY DATA] " + s2},
		{"text_in_quotes_after_placeholder", "password: [REDACTED]" + s1 + `/"token": "[REDACTED] ` + s2 + `"`},
		{"text_glued_after_quoted_placeholder", "password: [REDACTED]" + s1 + `/token: "[REDACTED]"/` + s2},
		{"word_after_header_placeholder", "password: [REDACTED]" + s1 + "/api_key: [REDACTED] " + s2},
		{"line_break_before_glued_header", "password: [REDACTED]" + s1 + "/token:\n  [REDACTED][api_key: " + s2},
		{"word_glued_to_header_placeholder", "password: [REDACTED]" + s1 + "/api_key: [REDACTED]" + s2},
		{"pair_after_header_placeholder", "password: [REDACTED]" + s1 + "/api_key: [REDACTED] other=" + s2},
		{"escaped_empty_quotes_before_placeholder", "password: [REDACTED]" + s1 + `/api_key: \"\" [REDACTED] ` + s2},
		{"crlf_before_placeholder", "password: [REDACTED]" + s1 + "/token:\r\n[REDACTED] api_key=" + s2},
		{"word_after_placeholder_before_pair", "password: [REDACTED]" + s1 + "/token: [REDACTED] " + s2 + " api_key=v"},
		{"word_after_placeholder_unchanged", "password: [REDACTED]" + s1 + "/token: [REDACTED] " + s2},
		{"depth_placeholder_before_pair", "password: [REDACTED]" + s1 + "/token: [MAX_DEPTH_EXCEEDED] api_key=" + s2},
		{"word_glued_to_content_placeholder", `'access_token': /secret:  [BINARY DATA]` + s1 + `[\'secret\' : \"[REDACTED]\"`},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, redactedSensitiveTextPlaceholder, s1, s2)
	}
	// The same shapes with other words glued to the placeholder.
	for _, in := range []string{
		`password: [REDACTED]SECRET1/api_key: "" [REDACTED] SECRET2`,
		"password: [REDACTED]SECRET1/api_key:\n  [REDACTED] SECRET2",
		"password: [REDACTED]SECRET1/api_key: [BINARY DATA] SECRET2",
		`password: [REDACTED]SECRET1/"token": "[REDACTED] SECRET2"`,
		`password: [REDACTED]SECRET1/token: "[REDACTED]"/SECRET2`,
		"password: [REDACTED]SECRET1/api_key: [REDACTED] SECRET2",
		"password: [REDACTED]SECRET1/token:\n  [REDACTED][api_key: SECRET2",
	} {
		assertAllEntryPoints(t, in, in, redactedSensitiveTextPlaceholder, "SECRET1", "SECRET2")
	}

	// Without the placeholder, the pass redacts the whole value, as the
	// expressions on main do.
	for _, in := range []string{
		"password: " + s1 + `/api_key: "" [REDACTED] ` + s2,
		"password: " + s1 + "/api_key: [BINARY DATA] " + s2,
		"password: " + s1 + "/api_key: [REDACTED] " + s2,
	} {
		assertAllEntryPoints(t, "unredacted_"+in, in, "password:[REDACTED]", s1, s2)
	}

	// Values that are placeholders as a whole, and pairs that end cleanly,
	// keep the text as it is or as main redacts it.
	for _, tc := range []struct{ in, want string }{
		{"\"x/token\":\"[REDACTED]\"\n api_key=" + s1, unparseableJSONPlaceholder},
		{"password: [REDACTED]/x-api_key: [REDACTED]" + s1, "password: [REDACTED]/x-api_key: [REDACTED]"},
		{"password: [REDACTED] api_key=" + s1, "password: [REDACTED] api_key=[REDACTED]"},
		{"password: [REDACTED]/x:[REDACTED] api_key=" + s1, "password:[REDACTED] api_key=[REDACTED]"},
		{"x password: [REDACTED]v/token:[REDACTED] api_key=" + s1, "x password: [REDACTED]v/token:[REDACTED] api_key=[REDACTED]"},
		{`password: [REDACTED]/x/token:"[REDACTED]" api_key=` + s1, `password: [REDACTED]/x/token:"[REDACTED]" api_key=[REDACTED]`},
		{`password: [REDACTED]/x/token: \"[REDACTED]\" api_key=` + s1, `password: [REDACTED]/x/token: \"[REDACTED]\" api_key=[REDACTED]`},
		{"password: [REDACTED]/x/token: [REDACTED]\napi_key=" + s1, "password: [REDACTED]/x/token: [REDACTED]\napi_key=[REDACTED]"},
		{"password: [REDACTED]/x/token: [REDACTED_SENSITIVE_URI] api_key=" + s1, "password: [REDACTED]/x/token: [REDACTED_SENSITIVE_URI] api_key=[REDACTED]"},
		{"password: [REDACTED]/x/api_key: [REDACTED]\napi_key=" + s1, "password: [REDACTED]/x/api_key: [REDACTED]\napi_key=[REDACTED]"},
	} {
		assertAllEntryPoints(t, "clean_"+tc.in, tc.in, tc.want, s1)
	}
	for _, in := range []string{
		`"password": "[REDACTED]"`,
		`\"password\": \"[REDACTED]\"`,
		"password: [REDACTED]/x/token:'[REDACTED]'",
		"password: [REDACTED]/x/token: [REDACTED],other=ok",
		"password: [REDACTED]/x/token: [REDACTED]}",
		"password: [REDACTED]/x/api_key: [REDACTED]\nother=ok",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	for _, tc := range []struct {
		s    string
		want bool
	}{
		{"k: [REDACTED]", true},
		{"k:[REDACTED]", true},
		{"k:\t[REDACTED]", true},
		{`k: "[REDACTED]"`, true},
		{`k: \"[REDACTED]\"`, true},
		{"k: '[REDACTED]'", true},
		{"k: [REDACTED_SENSITIVE_URI]", true},
		{"k: [REDACTED] x=1", true},
		{"k: [REDACTED]\nx", true},
		{"k: [REDACTED]  \r\nx", true},
		{"k: [REDACTED],x", true},
		{`k: "[REDACTED]"}`, true},
		{"k: [REDACTED].", true},
		{"api_key: [REDACTED]", true},
		{"api_key: [REDACTED] \nx", true},
		{"api_key= [REDACTED] x=1", true},
		{`k: "" [REDACTED]`, false},
		{"k:\n  [REDACTED]", false},
		{"k:\r[REDACTED]", false},
		{"k: [BINARY DATA]", false},
		{"k: [MAX_DEPTH_EXCEEDED]", false},
		{"k: [REDACTED_SENSITIVE_TEXT]", false},
		{`k: "[REDACTED] x"`, false},
		{`k: "[REDACTED]"/x`, false},
		{"k: [REDACTED] x", false},
		{"k: [REDACTED]x", false},
		{`k: "[REDACTED]`, false},
		{`k: "[REDACTED]'`, false},
		{`k: \"[REDACTED]"`, false},
		{"api_key: [REDACTED] x=1", false},
		{"api_key: [REDACTED],x", false},
		{"x-api-key :[REDACTED] y", false},
		{"Authorization: [REDACTED] x", false},
		{"api_\u212aey: [REDACTED] x", false},
	} {
		if got := isRedactedPairValueAt(tc.s, strings.IndexAny(tc.s, ":=")); got != tc.want {
			t.Errorf("isRedactedPairValueAt(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}

	kept := func(s string) []string {
		words := legacyWords(s, true)
		glued := wordsAfterPlaceholder(s, words)
		dropGluedBeforeUnredactedPair(s, words, glued)
		var out []string
		for j, g := range glued {
			if g {
				out = append(out, s[words[j].start:words[j].end])
			}
		}
		return out
	}
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{`password: [REDACTED]S/api_key: "" [REDACTED] v`, nil},
		{"password: [REDACTED]S/api_key:\n  [REDACTED] v", nil},
		{"password: [REDACTED]S/api_key: [BINARY DATA] v", nil},
		{"password: [REDACTED]S/token: [REDACTED] v", nil},
		{"password: [REDACTED]S/api_key: [REDACTED] other=1", nil},
		{"password: [REDACTED]S/token: [REDACTED] api_key=v", []string{"S", "token"}},
		{"password: [REDACTED]S/api_key: [REDACTED]", []string{"S", "api", "key"}},
	} {
		if got := kept(tc.in); !slices.Equal(got, tc.want) {
			t.Errorf("dropGluedBeforeUnredactedPair(%q) keeps %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestSpacedAndMixedContainerMembers covers stray quotes before containers
// whose quoted members start after spaces or line breaks, as in
// { "k": "SECRET"}, after other members and a ",", as in [1,"SECRET"] or
// [true, "SECRET"], or as the quoted value of an unquoted key, as in
// {k: "SECRET"}: the value seems to end at the space or "," before the
// closing quote of the stray string, which opens a member of the container,
// so the container is read and redacted with the value, under the nested-key
// budget and past it; and a quoted value whose closing quote opens such a
// member runs on to the end of the container.
func TestSpacedAndMixedContainerMembers(t *testing.T) {
	runInParallel(t)
	const s1 = "SPACED_MEMBER_SECRET"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct{ label, in, want string }{
		{"spaced_object", `DB_PASSWORD: x"S0/creds:{ "k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"multiline_object", "DB_PASSWORD: x\"S0/creds:{\n  \"k\": \"" + s1 + "\"\n}", "DB_PASSWORD:[REDACTED]"},
		{"spaced_array", `DB_PASSWORD: x"S0/creds:[ "a", "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"number_then_string", `DB_PASSWORD: x"S0/creds:[1,"` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"number_then_object", `DB_PASSWORD: x"S0/creds:[1, {"k": "` + s1 + `"}]`, "DB_PASSWORD:[REDACTED]"},
		{"bool_then_string", `DB_PASSWORD: x"S0/creds:[true, "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_spaced_object", "DB_PASSWORD: " + a135 + `/x"S0/creds:{ "k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_number_then_string", "DB_PASSWORD: " + a135 + `/x"S0/creds:[1,"` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"quoted_value_opens_spaced_object", `password: "a:token:{ "k": "` + s1 + `"}"`, `password:"[REDACTED]"`},
		{"quoted_value_opens_spaced_array", `password: "a:token:[ "a", "` + s1 + `"]"`, `password:"[REDACTED]"`},
		{"quoted_value_opens_mixed_array", `password: "a:token:[1,"` + s1 + `"]"`, `password:"[REDACTED]"`},
		{"quoted_value_opens_spaced_mixed_array", `password: "a:token:[1, "` + s1 + `"]"`, `password:"[REDACTED]"`},
		{"quoted_value_opens_multiline_object", "password: \"a:token:{\n  \"k\": \"" + s1 + "\"\n}\"", `password:"[REDACTED]"`},
		{"single_quoted_value_opens_spaced_object", `password: 'a:token:{ 'k': '` + s1 + `'}'`, `password:'[REDACTED]'`},
		{"single_quoted_spaced_object", `DB_PASSWORD: x'S0/creds:{ 'k': '` + s1 + `'}`, "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_number_then_string", `DB_PASSWORD: x'S0/creds:[1,'` + s1 + `']`, "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_spaced_array", `DB_PASSWORD: x'S0/creds:[ 'a', '` + s1 + `']`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_single_quoted_members", `DB_PASSWORD: x\'S0/creds:{ \'k\': \'` + s1 + `\'}`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_double_quoted_members", `DB_PASSWORD: x"S0/creds:{\"k\": \"` + s1 + `\"}`, "DB_PASSWORD:[REDACTED]"},
		{"tab_before_member", "DB_PASSWORD: x\"S0/creds:{\t\"k\": \"" + s1 + "\"}", "DB_PASSWORD:[REDACTED]"},
		{"crlf_before_member", "DB_PASSWORD: x\"S0/creds:{\r\n  \"k\": \"" + s1 + "\"\r\n}", "DB_PASSWORD:[REDACTED]"},
		{"unquoted_key_quoted_value", `DB_PASSWORD: x"S0/creds:{k: "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"unquoted_key_spaced_equals", `DB_PASSWORD: x"S0/creds:{k = "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"unquoted_member_then_quoted_key", `DB_PASSWORD: x"S0/creds:{a: 1, "k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"nested_array_then_string", `DB_PASSWORD: x"S0/creds:[[1,2], "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"null_then_string", `DB_PASSWORD: x"S0/creds:[null,"` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"exact_key_spaced_object", `password: x"S0/creds:{ "k": "` + s1 + `"}`, "password:[REDACTED]"},
		{"exact_key_number_then_string", `password: x"S0/creds:[1,"` + s1 + `"]`, "password:[REDACTED]"},
		{"over_budget_multiline_object", "DB_PASSWORD: " + a135 + "/x\"S0/creds:{\n  \"k\": \"" + s1 + "\"\n}", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_spaced_array", "DB_PASSWORD: " + a135 + `/x"S0/creds:[ "a", "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_bool_then_string", "DB_PASSWORD: " + a135 + `/x"S0/creds:[true, "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_number_then_object", "DB_PASSWORD: " + a135 + `/x"S0/creds:[1, {"k": "` + s1 + `"}]`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_dangling_quoted_value_spaced_object", "DB_PASSWORD: " + a135 + `/password":"S0/creds:{ "k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]"},
		{"spaced_object_keeps_fields", `DB_PASSWORD: x"S0/creds:{ "k": "` + s1 + `"} other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"mixed_array_keeps_fields", `DB_PASSWORD: x"S0/creds:[1,"` + s1 + `"] other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, s1)
	}

	// Braces and brackets in strings that are not stray stay as they are.
	for _, in := range []string{
		`{"msg": "text with { only"}`,
		`{"msg": "text with [ only", "n": [1, "x"]}`,
		`note: x"S0/creds:{ "k": "v"} other=ok`,
		`note: x"S0/creds:[1,"v"] other=ok`,
		`msg: "a { b" ok`,
		`note: [1, "a", {"b": "c"}] fine`,
	} {
		assertUnchangedAllEntryPoints(t, in)
	}
	for _, tc := range []struct{ in, want string }{
		{`password: "text with { only" other=ok`, `password:"[REDACTED]" other=ok`},
		{`password: x"a:{ "b" c`, `password:[REDACTED] "b" c`},
	} {
		assertAllEntryPoints(t, tc.in, tc.in, tc.want)
	}

	for _, tc := range []struct {
		s    string
		open string
	}{
		{`"S0:{ "k": 1}`, "{"},
		{"\"S0:{\n  \"k\": 1}", "{"},
		{`"S0:[ "a", 1]`, "["},
		{`"S0:[1,"a"]`, "["},
		{`"S0:[true, "a", 2]`, "["},
		{`"S0:[1, {"k": 1}]`, "{"},
		{`"S0:{k: "v"}`, "{"},
		{`"S0:{a: 1, "k": 2}`, "{"},
		{`'S0:[1,'a']`, "["},
		{`\'S0:{ \'k\': 1}`, "{"},
		{`"S0:{k: "v" x}`, ""},
		{`"S0:[1, "a" x]`, ""},
		{`"S0:{ "k" x}`, ""},
		{`"S0:[1], "a"]`, ""},
		{`"a, "b"]`, ""},
		{`"S0:{ "`, ""},
		{`"S0:x: "v"}`, ""},
		{`"S0:[k: "v"]`, ""},
	} {
		qEnd, _ := scanQuotedAt(tc.s, 0, nil)
		want := -1
		if tc.open != "" {
			want = strings.LastIndex(tc.s[:qEnd], tc.open)
		}
		if got := containerMemberOpener(tc.s, 0, qEnd, nil); got != want {
			t.Errorf("containerMemberOpener(%q, 0, %d, nil) = %d, want %d", tc.s, qEnd, got, want)
		}
	}
	s := `x"S0/creds:[1, {"k": "v"}]`
	qEnd, _ := scanQuotedAt(s, 1, nil)
	if got, want := strayQuoteOuter(s, 1, qEnd, strings.IndexByte(s, ','), nil), strings.IndexByte(s, '['); got != want {
		t.Errorf("strayQuoteOuter(%q) = %d with the [ in the value, want %d", s, got, want)
	}
	if got := strayQuoteOuter(s, 1, qEnd, strings.IndexByte(s, '['), nil); got >= 0 {
		t.Errorf("strayQuoteOuter(%q) = %d with no container in the value, want < 0", s, got)
	}
}

// TestPlusPaddedQueryKeys covers query keys padded with a "+", which a query
// reads as a space, as the +token of ?+token=SECRET: FormatContentParts
// redacts its value, as it does for ?%20token=SECRET and as the text scrub
// of an error message does, and so it does for a key that ends in a
// sensitive key name after a "+" or an escaped one, while ?q=a+b and
// ?a+b=c stay as they are; the legacy scrubs read such keys as main did.
func TestPlusPaddedQueryKeys(t *testing.T) {
	runInParallel(t)
	const s1 = "LEAK_SECRET_ONE"
	formatURI := func(uri string) string {
		parts := FormatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: uri}}}}, -1)
		if len(parts) != 1 {
			t.Fatalf("FormatContentParts(%q) = %v, want one part", uri, parts)
		}
		got, _ := parts[0]["uri"].(string)
		return got
	}
	for _, tc := range []struct{ label, uri, want string }{
		{"plus_before_key", "https://example.com/cb?+token=" + s1, "https://example.com/cb?+token=%5BREDACTED%5D"},
		{"escaped_space_before_key", "https://example.com/cb?%20token=" + s1, "https://example.com/cb?%20token=%5BREDACTED%5D"},
		{"plus_after_key", "https://example.com/cb?token+=" + s1, "https://example.com/cb?token+=%5BREDACTED%5D"},
		{"plus_around_key", "https://example.com/cb?+api_key+=" + s1 + "&q=a+b", "https://example.com/cb?+api_key+=%5BREDACTED%5D&q=a+b"},
		{"plus_before_key_parameter", "https://example.com/cb?+key=" + s1, "https://example.com/cb?+key=%5BREDACTED%5D"},
		{"word_and_plus_before_key", "https://example.com/cb?my+token=" + s1, "https://example.com/cb?my+token=%5BREDACTED%5D"},
		{"plus_key_after_prose", "https://example.com/cb?q=hello+world&+token=" + s1, "https://example.com/cb?q=hello+world&+token=%5BREDACTED%5D"},
	} {
		assertAllEntryPoints(t, tc.label, tc.uri, tc.want, s1)
		if got := formatURI(tc.uri); got != tc.want {
			t.Errorf("[%s] FormatContentParts file URI %q = %q, want %q", tc.label, tc.uri, got, tc.want)
		} else if again := formatURI(got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	// An escaped "+" before a key fails the error text closed as a whole, as
	// on main, and FormatContentParts redacts the value.
	escapedPlus := "https://example.com/cb?%2Btoken=" + s1
	assertAllEntryPoints(t, "escaped_plus_before_key", escapedPlus, redactedSensitiveTextPlaceholder, s1)
	if got, want := formatURI(escapedPlus), "https://example.com/cb?%2Btoken=%5BREDACTED%5D"; got != want {
		t.Errorf("FormatContentParts file URI %q = %q, want %q", escapedPlus, got, want)
	}

	for _, uri := range []string{
		"https://example.com/cb?q=a+b",
		"https://example.com/cb?a+b=c",
		"https://example.com/cb?q=hello+world&sort=a+b",
		"https://example.com/cb?+token=%5BREDACTED%5D",
	} {
		assertUnchangedAllEntryPoints(t, uri)
		if got := formatURI(uri); got != uri {
			t.Errorf("FormatContentParts file URI %q = %q, want unchanged", uri, got)
		}
	}

	// The legacy scrubs read query keys as the expressions on main did
	// (outputs of scrubErrorTokens on main).
	for _, tc := range []struct{ in, want string }{
		{"https://example.com/cb?+token=" + s1, "https://example.com/cb?+token=[REDACTED]"},
		{"https://example.com/cb?%20token=" + s1, "https://example.com/cb?%20token=%5BREDACTED%5D"},
		{"https://example.com/cb?token+=" + s1, "https://example.com/cb?token+=" + s1},
		{"https://example.com/cb?my+token=" + s1, "https://example.com/cb?my+token=[REDACTED]"},
		{"https://example.com/cb?q=a+b", "https://example.com/cb?q=a+b"},
	} {
		if got := legacyScrubErrorTokens(tc.in); got != tc.want {
			t.Errorf("legacyScrubErrorTokens(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct {
		key  string
		want bool
	}{
		{"my token", true},
		{"+token", true},
		{"x/api_key ", true},
		{"a b", false},
		{"next_page_token", false},
		{"", false},
	} {
		if got := endsInSensitiveTextKey(tc.key); got != tc.want {
			t.Errorf("endsInSensitiveTextKey(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
}

// TestAdversarialScalingSpacedContainersAndRedactedValues checks that stray
// quotes before spaced and mixed containers, long member lists before a ",",
// unquoted keys with quoted values, header values after glued placeholders,
// long words after a placeholder, chains that the scrub leaves as they are,
// and "+"-padded query keys stay linear through all three entry points, and
// through FormatContentParts for URIs.
func TestAdversarialScalingSpacedContainersAndRedactedValues(t *testing.T) {
	runInParallel(t)
	a135 := strings.Repeat("a", 135)
	for _, shape := range []struct {
		name string
		uri  bool
		mk   func(n int) string
	}{
		{"stray quotes before spaced containers", false, func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"S0/creds:{ "k": "v"} `, n)
		}},
		{"over-budget stray quotes before mixed arrays", false, func(n int) string {
			return strings.Repeat("DB_PASSWORD: "+a135+`/x"S0/creds:[1, {"k": "v"}]`+"\n", n)
		}},
		{"long member list before a comma", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + `/x"` + strings.Repeat("[1],", n) + `"v"]`
		}},
		{"nested unquoted keys with quoted values", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`/x"S0:{k: `, n) + `"v"` + strings.Repeat("}", n)
		}},
		{"stray quotes before many commas", false, func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`/x"S0:[1,`, n) + `"v"` + strings.Repeat("]", n)
		}},
		{"quoted value opening a long array", false, func(n int) string {
			return `password: "a:token:[` + strings.Repeat("1, ", n) + `"v"]"`
		}},
		{"header values after glued placeholders", false, func(n int) string {
			return strings.Repeat("password: [REDACTED]x/api_key: [REDACTED] ", n) + "api_key=v"
		}},
		{"long word after a placeholder", false, func(n int) string {
			return "password: [REDACTED]" + strings.Repeat("/x:[REDACTED]", n) + " " + strings.Repeat("a", 8*n) + " api_key=v"
		}},
		{"chains that the scrub leaves as they are", false, func(n int) string {
			return "password: [REDACTED]x/token:\n" + strings.Repeat("  [REDACTED]x/token: [REDACTED] y=1\n", n)
		}},
		{"plus-padded query keys", true, func(n int) string {
			return "https://example.com/cb?" + strings.Repeat("+token+=v&", n) + "z=1"
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
				if shape.uri {
					_ = FormatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: s}}}}, -1)
				}
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
}

// TestEscapedMemberQuotesAndOppositeQuoteContainerBrackets covers inner
// escaped quotes (\\\" and \\\', m&3 == 3) inside escaped-quoted container
// members in quotedMemberEnd, brackets inside inner quoted strings in
// containerMemberOpener and strayQuoteOuter, and delimited containers in
// other quotes enclosed by a spanning quote.
func TestEscapedMemberQuotesAndOppositeQuoteContainerBrackets(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "SECRET123"
		s2 = "AKIA_SECRET"
	)
	a130 := strings.Repeat("a", 130)
	a135 := strings.Repeat("a", 135)

	for _, tc := range []struct {
		label, in, want, secret string
	}{
		{"spaced_object_escaped_double_inner_quote", `DB_PASSWORD: x\"S0/creds:{ \"a\\\"b\": \"` + s1 + `\"}`, "DB_PASSWORD:[REDACTED]", s1},
		{"compact_object_escaped_double_inner_quote", `DB_PASSWORD: x\"S0/creds:{\"a\\\"b\": \"` + s1 + `\"}`, "DB_PASSWORD:[REDACTED]", s1},
		{"spaced_array_escaped_double_inner_quote", `DB_PASSWORD: x\"S0/creds:[ \"a\\\"b\", \"` + s1 + `\"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_dangling_key_escaped_double_inner_quote", `DB_PASSWORD: password\":S0/` + a130 + `/credentials:{ \"Access\\\"KeyId\": \"` + s2 + `\"}`, "DB_PASSWORD:[REDACTED]", s2},
		{"spaced_object_escaped_single_inner_quote", `DB_PASSWORD: x\'S0/creds:{ \'a\\\'b\': \'` + s1 + `\'}`, "DB_PASSWORD:[REDACTED]", s1},
		{"compact_object_escaped_single_inner_quote", `DB_PASSWORD: x\'S0/creds:{\'a\\\'b\': \'` + s1 + `\'}`, "DB_PASSWORD:[REDACTED]", s1},
		{"spaced_array_escaped_single_inner_quote", `DB_PASSWORD: x\'S0/creds:[ \'a\\\'b\', \'` + s1 + `\']`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_dangling_key_escaped_single_inner_quote", `DB_PASSWORD: password\':S0/` + a130 + `/credentials:{ \'Access\\\'KeyId\': \'` + s2 + `\'}`, "DB_PASSWORD:[REDACTED]", s2},
		{"escaped_member_trailing_literal_backslash", `DB_PASSWORD: x\"S0/creds:{ \"a\\\\\": \"` + s1 + `\"}`, "DB_PASSWORD:[REDACTED]", s1},
		{"escaped_member_seven_backslash_inner_quote", `DB_PASSWORD: x\"S0/creds:{ \"a\\\\\\\"b\": \"` + s1 + `\"}`, "DB_PASSWORD:[REDACTED]", s1},
		{"stray_escaped_quote_inner_escape_before_nested_object", `DB_PASSWORD: x\"[1, a\\\", creds:{ \"k\": \"` + s1 + `\"}]`, "DB_PASSWORD:[REDACTED]", s1},
		{"single_quoted_closing_bracket_in_double", `DB_PASSWORD: x"S0/creds:[']', "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"double_quoted_closing_brace_in_single", `DB_PASSWORD: x'S0/creds:["}", '` + s1 + `']`, "DB_PASSWORD:[REDACTED]", s1},
		{"single_quoted_opening_brace_in_double", `DB_PASSWORD: x"S0/creds:[ '{', "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_single_quoted_closing_bracket_in_double", "DB_PASSWORD: " + a135 + `/x"S0/creds:[']', "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_double_quoted_closing_brace_in_single", "DB_PASSWORD: " + a135 + `/x'S0/creds:["}", '` + s1 + `']`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_single_quoted_opening_brace_in_double", "DB_PASSWORD: " + a135 + `/x"S0/creds:[ '{', "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"escaped_double_quoted_closing_brace_in_double", `DB_PASSWORD: x"S0/creds:[ \"}\", "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"triple_escaped_closing_brace_in_escaped_double", `DB_PASSWORD: x\"S0/creds:[\\\"}\\\", \"` + s1 + `\"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"apostrophe_contraction_before_container", `DB_PASSWORD: x"it's/creds:[1, "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"possessive_apostrophe_before_quoted_array_element", `DB_PASSWORD: x"users'/creds:[1, 'a', "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"mid_token_escaped_quote_before_container", `DB_PASSWORD: x"S0\"/creds:[1, a\", "` + s1 + `"]`, "DB_PASSWORD:[REDACTED]", s1},
		{"mixed_outer_double_inner_single_object", `DB_PASSWORD: x"S0/creds:{ 'k': '` + s1 + `'}"`, "DB_PASSWORD:[REDACTED]", s1},
		{"mixed_outer_single_inner_double_object", `DB_PASSWORD: x'S0/creds:{ "k": "` + s1 + `"}'`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_mixed_outer_double_inner_single_object", "DB_PASSWORD: " + a135 + `/x"S0/creds:{ 'k': '` + s1 + `'}"`, "DB_PASSWORD:[REDACTED]", s1},
		{"over_budget_mixed_outer_single_inner_double_object", "DB_PASSWORD: " + a135 + `/x'S0/creds:{ "k": "` + s1 + `"}'`, "DB_PASSWORD:[REDACTED]", s1},
		{"mixed_outer_double_inner_single_keeps_fields", `DB_PASSWORD: x"S0/creds:{ 'k': '` + s1 + `'}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok", s1},
		{"mixed_outer_single_inner_double_keeps_fields", `DB_PASSWORD: x'S0/creds:{ "k": "` + s1 + `"}' other=ok`, "DB_PASSWORD:[REDACTED] other=ok", s1},
	} {
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, tc.secret)
	}

	for _, tc := range []struct {
		s         string
		wantEnd   int
		wantEmpty bool
	}{
		{`\"\"`, 4, true},
		{`\"a\\\"b\"`, 10, false},
		{`\'a\\\'b\'`, 10, false},
		{`\"a\\\\\"`, 9, false},
		{`\"a\\\\\\\"b\"`, 14, false},
		{`\"a\\"`, -1, false},
		{`\"a\\\"`, -1, false},
		{`\"a\\\b\"`, -1, false},
		{"\\\"a\nb\\\"", 7, false},
	} {
		end, empty := quotedMemberEnd(tc.s, 0)
		if end != tc.wantEnd || empty != tc.wantEmpty {
			t.Errorf("quotedMemberEnd(%q, 0) = (%d, %v), want (%d, %v)", tc.s, end, empty, tc.wantEnd, tc.wantEmpty)
		}
	}

	for _, tc := range []struct {
		s    string
		open string
	}{
		{`"S0:[']', "a"]`, "["},
		{`'S0:["}", 'a']`, "["},
		{`"S0:[ '{', "a"]`, "["},
		{`"S0:[ \"}\", "a"]`, "["},
		{`\"S0:[\\\"}\\\", \"a\"]`, "["},
		{`"S0:{ 'a:b': '}', "k": 1}`, "{"},
	} {
		qEnd, _ := scanQuotedAt(tc.s, 0, nil)
		want := strings.Index(tc.s, tc.open)
		if got := containerMemberOpener(tc.s, 0, qEnd, nil); got != want {
			t.Errorf("containerMemberOpener(%q, 0, %d, nil) = %d, want %d", tc.s, qEnd, got, want)
		}
	}
}

// TestDelimSpaceTrimmedQueryKeys checks that endsInSensitiveTextKey trims all
// trailing ASCII delimiter whitespace matching redactTextKeyValues, so query
// keys ending in %09, %0a, %0b, %0c, %0d, or other delimiter whitespace redact
// their values in FormatContentParts as well as the text entry points while
// benign "+" query pairs and URL path segments stay unchanged.
func TestDelimSpaceTrimmedQueryKeys(t *testing.T) {
	runInParallel(t)
	const s1 = "SECRET_VAL"
	formatURI := func(uri string) string {
		parts := FormatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: uri}}}}, -1)
		if len(parts) != 1 {
			t.Fatalf("FormatContentParts(%q) = %v, want one part", uri, parts)
		}
		got, _ := parts[0]["uri"].(string)
		return got
	}

	for _, tc := range []struct {
		label, uri, wantText, wantURI string
	}{
		{"plus_key_tab", "https://example.com/cb?my+token%09=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%09=%5BREDACTED%5D"},
		{"plus_key_lf", "https://example.com/cb?my+token%0a=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%0a=%5BREDACTED%5D"},
		{"plus_key_vt", "https://example.com/cb?my+token%0b=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%0b=%5BREDACTED%5D"},
		{"plus_key_ff", "https://example.com/cb?my+token%0c=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%0c=%5BREDACTED%5D"},
		{"plus_key_cr", "https://example.com/cb?my+token%0d=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%0d=%5BREDACTED%5D"},
		{"slash_key_tab", "https://example.com/cb?x/token%09=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?x/token%09=%5BREDACTED%5D"},
		{"pct20_key_tab", "https://example.com/cb?my%20token%09=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my%20token%09=%5BREDACTED%5D"},
		{"leading_plus_key_tab", "https://example.com/cb?+token%09=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?+token%09=%5BREDACTED%5D"},
		{"leading_tab_exact_key", "https://example.com/cb?%09token=" + s1, "https://example.com/cb?%09token=%5BREDACTED%5D", "https://example.com/cb?%09token=%5BREDACTED%5D"},
		{"plus_key_param_tab", "https://example.com/cb?+key%09=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?+key%09=%5BREDACTED%5D"},
		{"plus_key_nul", "https://example.com/cb?my+token%00=" + s1, redactedSensitiveTextPlaceholder, "https://example.com/cb?my+token%00=%5BREDACTED%5D"},
	} {
		assertAllEntryPoints(t, tc.label, tc.uri, tc.wantText, s1)
		if got := formatURI(tc.uri); got != tc.wantURI {
			t.Errorf("[%s] FormatContentParts file URI %q = %q, want %q", tc.label, tc.uri, got, tc.wantURI)
		} else if again := formatURI(got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	for _, uri := range []string{
		"https://example.com/cb?q=a+b",
		"https://example.com/cb?a+b=c",
		"https://example.com/account/reset-password/confirm",
		"https://example.com/my+token/public",
	} {
		assertUnchangedAllEntryPoints(t, uri)
		if got := formatURI(uri); got != uri {
			t.Errorf("FormatContentParts file URI %q = %q, want unchanged", uri, got)
		}
	}

	for _, tc := range []struct {
		key  string
		want bool
	}{
		{"my token\t", true},
		{"my token\n", true},
		{"my token\v", true},
		{"my token\f", true},
		{"my token\r", true},
		{"x/token\t", true},
		{"my token\x00", true},
		{"my token\x7f", true},
		{"\t\r\n ", false},
		{"a b\t", false},
	} {
		if got := endsInSensitiveTextKey(tc.key); got != tc.want {
			t.Errorf("endsInSensitiveTextKey(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
}

// formatContentTextPart returns the text that FormatContentParts keeps of a
// single text part that holds in.
func formatContentTextPart(t *testing.T, in string) string {
	t.Helper()
	parts := FormatContentParts(&genai.Content{Parts: []*genai.Part{{Text: in}}}, -1)
	if len(parts) != 1 {
		t.Fatalf("FormatContentParts(%.80q) = %d parts, want 1", in, len(parts))
	}
	got, _ := parts[0]["text"].(string)
	return got
}

// TestMixedQuoteContainerMembersAndBoundaries covers spanning quotes that
// enclose delimited containers whose members use other quotes: a delimiter
// right after the opening quote, as the ": " of x": {'k': 'SECRET'}", a
// member in other quotes after the first member or element, in a nested
// container, or in a container next to another one, within the nested-key
// budget and past it. Prose with an apostrophe and containers without such a
// member are not extended, and the scan of long, many or deeply nested
// containers stays linear.
func TestMixedQuoteContainerMembersAndBoundaries(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_123"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"delim_at_zero_with_space", `DB_PASSWORD: x": {'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"delim_at_zero_compact", `DB_PASSWORD: x":{'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"delim_at_zero_eq_with_space", `DB_PASSWORD: x"= ['` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"second_object_member_quoted", `DB_PASSWORD: x"S0/creds:{ k1: 1, 'k2': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"second_object_member_unquoted_key_quoted_val", `DB_PASSWORD: x"S0/creds:{ k1: 1, k2: '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"third_array_element_quoted", `DB_PASSWORD: x"S0/creds:[1, 2, '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"nested_subcontainer_quoted_member", `DB_PASSWORD: x"S0/creds:{ a: { 'k': '` + secret + `' }}"`, "DB_PASSWORD:[REDACTED]"},
		{"nested_array_in_object_quoted_member", `DB_PASSWORD: x"S0/creds:{ a: [1, 2, '` + secret + `'] }"`, "DB_PASSWORD:[REDACTED]"},
		{"trailing_adjacent_container", `DB_PASSWORD: x"S0/creds:{ 'k': '` + secret + `'} {a:1}"`, "DB_PASSWORD:[REDACTED]"},
		{"leading_adjacent_container", `DB_PASSWORD: x"S0/creds:{a:1} { 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"earlier_container_in_path_prefix", `DB_PASSWORD: x"S0:{a:1}/creds:{ 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_delim_at_zero", "DB_PASSWORD: " + a135 + `": {'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_later_member", "DB_PASSWORD: " + a135 + `/x"S0/creds:{ k1: 1, 'k2': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_nested_subcontainer", "DB_PASSWORD: " + a135 + `/x"S0/creds:{ a: { 'k': '` + secret + `' }}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_trailing_adjacent_container", "DB_PASSWORD: " + a135 + `/x"S0/creds:{ 'k': '` + secret + `'} {a:1}"`, "DB_PASSWORD:[REDACTED]"},
		{"single_outer_double_inner_later_member", `DB_PASSWORD: x'S0/creds:{ k1: 1, "k2": "` + secret + `"}' other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
	} {
		if got := scrubErrorTokens(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorTokens(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	// Prose with an apostrophe, and containers without a member in other
	// quotes, end the value where it seems to end.
	for _, tc := range []struct {
		label, in, want string
	}{
		{"prose_contraction", `DB_PASSWORD: x"it's not a container" other=ok`, `DB_PASSWORD:[REDACTED] not a container" other=ok`},
		{"unquoted_container_members_not_extended", `DB_PASSWORD: x"S0/creds:{a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] 1}" other=ok`},
		{"unquoted_container_after_delim_quote", `msg: "DB_PASSWORD: x": {a: 1} "`, `msg: "DB_PASSWORD:[REDACTED] {a: 1} "`},
		{"apostrophe_in_container_word", `DB_PASSWORD: x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
	} {
		if got := scrubErrorTokens(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorTokens(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"long member list before the member in other quotes", func(n int) string {
			return `DB_PASSWORD: x"S0/creds:{ ` + strings.Repeat(`k: 1, `, n) + `'k2': '` + secret + `'}"`
		}},
		{"many containers before the one with a member in other quotes", func(n int) string {
			return `DB_PASSWORD: x"S0/creds:` + strings.Repeat(`{a:1} `, n) + `{ 'k': '` + secret + `'}"`
		}},
		{"deeply nested member in other quotes", func(n int) string {
			return `DB_PASSWORD: x"S0/creds:` + strings.Repeat(`{ a: `, n) + `'` + secret + `'` + strings.Repeat(` }`, n) + `"`
		}},
		{"repeated spanning quotes around mixed-quote containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"S0/creds:{ k1: 1, 'k2': '`+secret+`'}" `, n)
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
}

// TestExtendGluedValueEndOuterQuoteBoundaries covers a value that runs on
// through a spanning quoted string around a mixed-quote container and is
// followed by the closing quote of a string that encloses the pair, a ":"
// and a space, or an escaped line break (see hasGluedSuffixAfterQuotedValue).
// One scrubErrorTokens pass ends the value before that boundary rather than
// taking it and the text after it. A further pass then reads the placeholder
// as it reads any unquoted value: it takes an enclosing closing quote, or an
// escaped line break and the word glued to it, as main does for
// msg: "password=abc" for user "bob", so the entry points keep the prose after
// it. Quote-led text whose first string decodes to a credential fails closed,
// as on main, and a word glued to the spanning quote is still redacted with
// the value.
func TestExtendGluedValueEndOuterQuoteBoundaries(t *testing.T) {
	runInParallel(t)
	const (
		secret = "SECRET_PROBE_123"
		glued  = "GLUED_TAIL_SECRET_999"
	)
	container := `x"S0/creds:{ 'k': '` + secret + `'}"`
	for _, tc := range []struct {
		label, in string
		// tokens is the result of one scrubErrorTokens pass, and want the
		// result of the entry points, which scrub to a fixed point.
		tokens, want string
	}{
		{
			"enclosing_quote_before_prose",
			`msg: "DB_PASSWORD: ` + container + `" for user "bob"`,
			`msg: "DB_PASSWORD:[REDACTED]" for user "bob"`,
			`msg: "DB_PASSWORD:[REDACTED] for user "bob"`,
		},
		{
			"prefixed_enclosing_quote_before_pair",
			`err: "DB_PASSWORD: ` + container + `" other=ok`,
			`err: "DB_PASSWORD:[REDACTED]" other=ok`,
			`err: "DB_PASSWORD:[REDACTED] other=ok`,
		},
		{
			"enclosing_quote_before_colon_and_space",
			`parsing "DB_PASSWORD: ` + container + `": invalid`,
			`parsing "DB_PASSWORD:[REDACTED]": invalid`,
			`parsing "DB_PASSWORD:[REDACTED] invalid`,
		},
		{
			"colon_and_space_after_spanning_quote",
			`DB_PASSWORD: ` + container + `: invalid`,
			`DB_PASSWORD:[REDACTED]: invalid`,
			`DB_PASSWORD:[REDACTED]: invalid`,
		},
		{
			"escaped_line_break_after_spanning_quote",
			`DB_PASSWORD: ` + container + `\nretry`,
			`DB_PASSWORD:[REDACTED]\nretry`,
			`DB_PASSWORD:[REDACTED]`,
		},
		{
			"quote_led_enclosing_quote_before_pair",
			`"DB_PASSWORD: ` + container + `" other=ok`,
			`"DB_PASSWORD:[REDACTED]" other=ok`,
			unparseableJSONPlaceholder,
		},
		{
			"word_glued_to_spanning_quote",
			`msg: "DB_PASSWORD: ` + container + glued + `" for user "bob"`,
			`msg: "DB_PASSWORD:[REDACTED] for user "bob"`,
			`msg: "DB_PASSWORD:[REDACTED] for user "bob"`,
		},
	} {
		if got := scrubErrorTokens(tc.in); got != tc.tokens {
			t.Errorf("[%s] scrubErrorTokens(%q) = %q, want %q", tc.label, tc.in, got, tc.tokens)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret, glued)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}
}

// TestOpenerScanBudgetLinearScaling covers spans of escaped quotes that pair
// one way once a "{" or "[" is open in a quoted string and another way before,
// next to an escaped or bare single quote, so that each check of a quoted
// string made from inside it lands in an inner string of every scan cached for
// that string and scans the rest of the text again. The container scans of a
// text stop after a budget linear in its length (see
// openerScanBudgetMultiplier) and fail closed, so the work stays linear, a
// JSON-escaped span of 224 KiB fails closed through every entry point in well
// under a second, no secret before, inside or after such a span is kept, and a
// second pass changes nothing.
func TestOpenerScanBudgetLinearScaling(t *testing.T) {
	runInParallel(t)
	const (
		sBefore = "SECRET_BEFORE_789"
		sInside = "SECRET_INSIDE_456"
		sAfter  = "SECRET_AFTER_123"
	)
	jsonInner := func(s string) string {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}
		return string(b[1 : len(b)-1])
	}
	span := func(unit string, n int) string {
		return `msg " ` + strings.Repeat(unit, n) + `{ "k": "v"}`
	}
	maxWall := 2 * time.Second
	if raceEnabled {
		maxWall = 20 * time.Second
	}
	for _, u := range []struct{ name, unit string }{
		{"escaped quotes around an escaped single quote", `\" [ \" \' token:a\" `},
		{"escaped quotes around a single quote", `\" [ \" ' token:a\" `},
		{"bracket before escaped quotes around an escaped single quote", `[ \" \' token:a\" \" `},
	} {
		t.Run(u.name, func(t *testing.T) {
			// The JSON-escaped span is the one whose checks keep missing the
			// cached scans; at 512 units and more, rescanning the rest of the
			// text dominates, so 16 times the units would take some 256 times
			// as long.
			for _, form := range []struct {
				name         string
				small, large int
				mk           func(n int) string
			}{
				{"raw", 128, 2048, func(n int) string { return span(u.unit, n) }},
				{"JSON-escaped", 512, 8192, func(n int) string { return jsonInner(span(u.unit, n)) }},
			} {
				small, large := form.mk(form.small), form.mk(form.large)
				scrub := func(s string) {
					_ = redactErrorMessage(s)
					_, _ = sanitizeRawText(s)
					_, _, _ = SmartTruncate(s, -1)
				}
				scrub(small)
				scrub(large)
				if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
					t.Errorf("%s: %d units took %v, %d units took %v: ratio %.0f, want linear", form.name, form.small, tSmall, form.large, tLarge, ratio)
				}
			}

			// A JSON-escaped span of 224 KiB runs the container scans out of
			// budget, so every entry point fails it closed, idempotently.
			unitLen := len(jsonInner(u.unit))
			in := jsonInner(span(u.unit, (224<<10+unitLen-1)/unitLen))
			if got := scrubErrorTokens(in); got != redactedSensitiveTextPlaceholder {
				t.Errorf("scrubErrorTokens(%d bytes) = %.80q, want %q", len(in), got, redactedSensitiveTextPlaceholder)
			}
			start := time.Now()
			for _, ep := range []struct {
				name string
				f    func(string) string
			}{
				{"redactErrorMessage", redactErrorMessage},
				{"sanitizeRawText", func(s string) string { out, _ := sanitizeRawText(s); return out }},
				{"SmartTruncate", func(s string) string {
					raw, _, err := SmartTruncate(s, -1)
					var out string
					if err != nil || json.Unmarshal(raw, &out) != nil {
						t.Fatalf("SmartTruncate(%d bytes) = (%.80s, %v), want a JSON string", len(s), raw, err)
					}
					return out
				}},
				{"FormatContentParts", func(s string) string { return formatContentTextPart(t, s) }},
			} {
				if got := ep.f(in); got != redactedSensitiveTextPlaceholder {
					t.Errorf("%s(%d bytes) = %.80q, want %q", ep.name, len(in), got, redactedSensitiveTextPlaceholder)
				} else if again := ep.f(got); again != got {
					t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
				}
			}
			if elapsed := time.Since(start); elapsed > maxWall {
				t.Errorf("%d bytes took %v through the entry points, want at most %v", len(in), elapsed, maxWall)
			}

			// Secrets before, inside and after the span, raw and JSON-escaped.
			planted := "api_key=" + sBefore + ` msg " ` +
				strings.Repeat(strings.Replace(u.unit, "token:a", "token:"+sInside, 1), 512) +
				`{ "password": "` + sAfter + `"}`
			for _, in := range []string{planted, jsonInner(planted)} {
				assertRedactedIdempotent(t, u.name, in, sBefore, sInside, sAfter)
				got := formatContentTextPart(t, in)
				for _, secret := range []string{sBefore, sInside, sAfter} {
					if strings.Contains(got, secret) {
						t.Errorf("FormatContentParts(%.80q) = %.80q, kept %q", in, got, secret)
					}
				}
				if again := formatContentTextPart(t, got); again != got {
					t.Errorf("FormatContentParts second pass(%.80q) = %.80q, want unchanged", got, again)
				}
			}
		})
	}
}

// TestOpenerScanBudget checks that consumeOpenerBudget charges a scan state
// until its budget, at least minOpenerScanBudget and otherwise
// openerScanBudgetMultiplier bytes per byte of text, runs out, that the state
// then stays exhausted, and that a nil state has no budget.
func TestOpenerScanBudget(t *testing.T) {
	runInParallel(t)
	for _, textLen := range []int{0, 1, minOpenerScanBudget / openerScanBudgetMultiplier, 1 << 20} {
		st := newQuoteScanState(textLen)
		budget := max(openerScanBudgetMultiplier*textLen, minOpenerScanBudget)
		if st.openerScanBudget != budget {
			t.Errorf("newQuoteScanState(%d) budget = %d, want %d", textLen, st.openerScanBudget, budget)
		}
		if !st.consumeOpenerBudget(budget-1) || st.openerBudgetExhausted {
			t.Errorf("newQuoteScanState(%d): consuming %d of %d stopped the scan", textLen, budget-1, budget)
		}
		if st.consumeOpenerBudget(1) || !st.openerBudgetExhausted {
			t.Errorf("newQuoteScanState(%d): consuming all of %d did not stop the scan", textLen, budget)
		}
		if st.consumeOpenerBudget(0) {
			t.Errorf("newQuoteScanState(%d): an exhausted state let the scan go on", textLen)
		}
	}
	var st *quoteScanState
	if !st.consumeOpenerBudget(1 << 40) {
		t.Errorf("nil quoteScanState stopped the scan")
	}
}

// TestSpacedStrayQuoteContainers covers a value that ends in a ":" or "="
// and whitespace inside a quoted string that encloses a "{" or "[" container
// with a member in that same quote, as in x": {'msg': "it's ok", 'k': 'v'}",
// the repr Python gives a dict that holds a string with an apostrophe. The
// string seems to end at that member, inside the container, which opens past
// the value. When the container closes and the quote closes the string right
// after it, possibly after whitespace, every entry point redacts the value
// together with the container, within the nested-key budget and past it, as
// it does when no whitespace follows the delimiter, and also when other text
// comes before the container. An unclosed container, or one that the quote
// does not close right after, keeps the text after it, and repeated, long or
// deeply nested containers stay linear.
func TestSpacedStrayQuoteContainers(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_123"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"dict_repr_with_apostrophe", `DB_PASSWORD: x": {'msg': "it's ok", 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"member_value_in_outer_quote", `DB_PASSWORD: x": {'a': '1', 'b': "` + secret + `"}"`, "DB_PASSWORD:[REDACTED]"},
		{"array_element_in_outer_quote", `DB_PASSWORD: x": [1, "` + secret + `"]"`, "DB_PASSWORD:[REDACTED]"},
		{"array_element_after_other_quote", `DB_PASSWORD: x": [1, 'a', "` + secret + `"]"`, "DB_PASSWORD:[REDACTED]"},
		{"padded_object_unquoted_key", `DB_PASSWORD: x": { a: "1", 'b': '` + secret + `' }"`, "DB_PASSWORD:[REDACTED]"},
		{"single_outer_quote", `DB_PASSWORD: x': { a: '1', "b": "` + secret + `" }'`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_quote_in_member", `DB_PASSWORD: x": {'k': "a\"b", 'm': '` + secret + `'}"`, redactedSensitiveTextPlaceholder},
		{"enclosing_string_before_prose", `msg: "DB_PASSWORD: x": {'a': "1", 'b': '` + secret + `'}" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"nested_object_with_apostrophe", `DB_PASSWORD: x": {'a': {'b': "it's ok"}, 'c': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"nested_arrays", `DB_PASSWORD: x": [[1, "a"], ['b', '` + secret + `']]"`, "DB_PASSWORD:[REDACTED]"},
		{"tab_after_delimiter", "DB_PASSWORD: x\":\t{'a': \"1\", 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"spaces_after_delimiter", `DB_PASSWORD: x":   {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"line_break_after_delimiter", "DB_PASSWORD: x\":\n{'a': \"1\", 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"eq_delimiter", `DB_PASSWORD: x"= {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"eq_key_delimiter", `DB_PASSWORD= x": {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD=[REDACTED]"},
		{"path_prefixed_delimiter", `DB_PASSWORD: x"S0/creds: {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"space_before_closing_quote", `DB_PASSWORD: x": {'a': "1", 'b': '` + secret + `'} "`, `DB_PASSWORD:[REDACTED] "`},
		{"fields_after_closing_quote", `DB_PASSWORD: x": {'a': "1", 'b': '` + secret + `'}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"escaped_spanning_quote", `DB_PASSWORD: x\": {'a': \"1\", 'k': '` + secret + `'}\"`, redactedSensitiveTextPlaceholder},
		{"over_budget_object", "DB_PASSWORD: " + a135 + `": {'a': '1', 'b': "` + secret + `"}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_array", "DB_PASSWORD: " + a135 + `": [1, "` + secret + `"]"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_nested_object", "DB_PASSWORD: " + a135 + `": {'a': {'b': "it's ok"}, 'c': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_path_prefixed", "DB_PASSWORD: " + a135 + `/x"S0/creds: {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"object_after_other_text", `DB_PASSWORD: x": abc {'a': "1"}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"object_after_no_delimiter", `DB_PASSWORD: x"abc {'a': "1"}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	// A container that stays open or that the quote does not close right
	// after, and prose around a closing quote, keep the text after the value.
	for _, tc := range []struct {
		label, in, want string
	}{
		{"unclosed_object", `msg: "DB_PASSWORD: x": {"a": 1 for user "bob" other=ok`, `msg: "DB_PASSWORD:[REDACTED] {"a": 1 for user "bob" other=ok`},
		{"object_without_closing_quote", `msg: "DB_PASSWORD: x": {"a": 1} for user "bob" other=ok`, `msg: "DB_PASSWORD:[REDACTED] {"a": 1} for user "bob" other=ok`},
		{"unclosed_array", `DB_PASSWORD: x": [1, "a" and then other=ok`, `DB_PASSWORD:[REDACTED] [1, "a" and then other=ok`},
		{"object_closed_by_other_quote", `DB_PASSWORD: x": {'a': "1"}' other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"}' other=ok`},
		{"closing_quote_before_prose", `msg: "DB_PASSWORD: x" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"closing_quote_before_colon", `msg: "DB_PASSWORD: x": invalid`, `msg: "DB_PASSWORD:[REDACTED] invalid`},
		{"over_budget_object_without_closing_quote", "DB_PASSWORD: " + a135 + `": {"a": 1} for user "bob" other=ok`, `DB_PASSWORD:[REDACTED] {"a": 1} for user "bob" other=ok`},
		{"over_budget_unclosed_array", "DB_PASSWORD: " + a135 + `": [1, "a" and then other=ok`, `DB_PASSWORD:[REDACTED] [1, "a" and then other=ok`},
		{"over_budget_object_closed_by_other_quote", "DB_PASSWORD: " + a135 + `": {'a': "1"}' other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"}' other=ok`},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
	}
	for _, in := range []string{
		`it's fine, don't worry about password_count=3 and token_limit: 5`,
		"password=[REDACTED]\nretry=ok",
		`note: x": {'msg': "it's ok", 'k': 'v'}" other=ok`,
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated closed containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x": {'a': "1", 'b': '`+secret+`'}" `, n)
		}},
		{"repeated unclosed containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x": {'a': "1", `, n)
		}},
		{"repeated over-budget unclosed containers", func(n int) string {
			return strings.Repeat("DB_PASSWORD: "+a135+`": {'a': "1", `, n)
		}},
		{"long member list in one container", func(n int) string {
			return `DB_PASSWORD: x": {` + strings.Repeat(`'k': "v", `, n) + `'b': '` + secret + `'}"`
		}},
		{"deeply nested containers", func(n int) string {
			return `DB_PASSWORD: x": ` + strings.Repeat(`{'a': "1", 'b': `, n) + `'` + secret + `'` + strings.Repeat(`}`, n) + `"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}

	// 224 KiB of containers that never close runs the container scans out of
	// budget, so every entry point fails the text closed, idempotently.
	unit := `DB_PASSWORD: x": {'a': "1", 'b': '` + secret + `', `
	in := strings.Repeat(unit, (224<<10+len(unit)-1)/len(unit))
	if got := scrubErrorTokens(in); got != redactedSensitiveTextPlaceholder {
		t.Errorf("scrubErrorTokens(%d bytes) = %.80q, want %q", len(in), got, redactedSensitiveTextPlaceholder)
	}
	maxWall := 2 * time.Second
	if raceEnabled {
		maxWall = 20 * time.Second
	}
	start := time.Now()
	for _, ep := range []struct {
		name string
		f    func(string) string
	}{
		{"scrubErrorText", scrubErrorText},
		{"redactErrorMessage", redactErrorMessage},
		{"sanitizeRawText", func(s string) string { out, _ := sanitizeRawText(s); return out }},
		{"SmartTruncate", func(s string) string {
			raw, _, err := SmartTruncate(s, -1)
			var out string
			if err != nil || json.Unmarshal(raw, &out) != nil {
				t.Fatalf("SmartTruncate(%d bytes) = (%.80s, %v), want a JSON string", len(s), raw, err)
			}
			return out
		}},
		{"FormatContentParts", func(s string) string { return formatContentTextPart(t, s) }},
	} {
		if got := ep.f(in); got != redactedSensitiveTextPlaceholder {
			t.Errorf("%s(%d bytes) = %.80q, want %q", ep.name, len(in), got, redactedSensitiveTextPlaceholder)
		} else if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
		}
	}
	if elapsed := time.Since(start); elapsed > maxWall {
		t.Errorf("%d bytes took %v through the entry points, want at most %v", len(in), elapsed, maxWall)
	}
}

// TestSpacedStrayQuoteAfterApostropheOrOtherQuote covers values in which a
// quote that opens no string the spanning-quote checks can place comes before
// the quote that opens the string around a container: an apostrophe joined to
// the word before it, as in user's": {'msg': "it's ok", 'k': 'SECRET'}", or a
// quote of the other kind, as the ' of a'x": {'a': "1", 'b': 'SECRET'}". The
// scan goes on to the later quote, so every entry point redacts the value
// together with the container, within the nested-key budget and past it, as it
// does without the first quote. Prose after the enclosing string, containers
// that the string does not close right after, and a container after a quote
// whose string closes after it, which a later quote does not open, are kept,
// within the budget and past it, and repeated values with such quotes stay
// linear.
func TestSpacedStrayQuoteAfterApostropheOrOtherQuote(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_123"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"apostrophe_before_spaced_object", `DB_PASSWORD: user's": {'msg': "it's ok", 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_apostrophe_before_spaced_object", "DB_PASSWORD: " + a135 + `user's": {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"apostrophe_before_compact_object", `DB_PASSWORD: user's":{"msg": "it's ok", "k": "` + secret + `"}"`, "DB_PASSWORD:[REDACTED]"},
		{"single_quote_before_double_quoted_string", `DB_PASSWORD: a'x": {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_single_quote_before_double_quoted_string", "DB_PASSWORD: " + a135 + `a'x": {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"double_quote_before_single_quoted_string", `DB_PASSWORD: a"x': { a: '1', "b": "` + secret + `" }'`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_double_quote_before_single_quoted_string", "DB_PASSWORD: " + a135 + `a"x': { a: '1', "b": "` + secret + `" }'`, "DB_PASSWORD:[REDACTED]"},
		{"apostrophe_enclosing_string_before_prose", `msg: "DB_PASSWORD: user's": {'a': "1", 'b': '` + secret + `'}" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"apostrophe_before_adjacent_objects", `DB_PASSWORD: user's": {'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"closing_quote_before_prose", `msg: "DB_PASSWORD: x" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"apostrophe_closing_quote_before_prose", `msg: "DB_PASSWORD: user's" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"apostrophe_object_without_closing_quote", `DB_PASSWORD: user's": {'a': "1"} other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"} other=ok`},
		{"other_quote_object_without_closing_quote", `DB_PASSWORD: a'x": {'a': "1"} other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"} other=ok`},
		{"string_closing_after_compact_container", `DB_PASSWORD: x"S0/creds:{a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] 1}" other=ok`},
		{"string_closing_after_container_with_apostrophe", `DB_PASSWORD: x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
		{"over_budget_string_closing_after_compact_container", "DB_PASSWORD: " + a135 + `/x"S0/creds:{a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] 1}" other=ok`},
		{"over_budget_string_closing_after_container_with_apostrophe", "DB_PASSWORD: " + a135 + `/x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
	}
	for _, in := range []string{
		`note: user's": {'msg': "it's ok", 'k': 'v'}" other=ok`,
		`it's the user's "x": {'a': 1} {'b': 2}" note`,
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated quotes of both kinds before unclosed containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: a'x"a'x": {'a': "1", `, n)
		}},
		{"repeated quotes of both kinds before closed containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: a'x"a'x": {'a': "1", 'b': '`+secret+`'}" `, n)
		}},
		{"repeated apostrophes before closed containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: user's": {'msg': "it's ok", 'k': '`+secret+`'}" `, n)
		}},
		{"many unplaced quotes in one over-budget value", func(n int) string {
			return "DB_PASSWORD: " + a135 + strings.Repeat(`a'x"`, n) + `": {'a': "1", 'b': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: a'x"a'x": {'a': "1", 'b': '`+secret+`', `, secret)
}

// TestSpacedStrayQuoteContainerRuns covers quoted strings that enclose one or
// more "{" or "[" containers after a ":" or "=" and seem to close inside them:
// at an outer quote inside a member in other quotes, as in the Python repr
// x": {'msg': 'say "hi"', 'k': 'SECRET'}", whether whitespace follows the
// delimiter or not, also before prose and past the nested-key budget; at a
// member of a container followed by other containers, as in
// x": {'a': "1"} {'b': 'SECRET'}"; or before delimiter whitespace other than
// the regular-expression spaces, as the \v of x": {...}\v". Every entry point
// redacts the value together with all the containers. Containers that the
// string does not close right after, that other text or another quote follows,
// or that stay open keep the text after the value, and long, many or repeated
// containers stay linear.
func TestSpacedStrayQuoteContainerRuns(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_123"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"repr_outer_quote_in_member", `DB_PASSWORD: x": {'msg': 'say "hi"', 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"repr_enclosing_string_before_prose", `msg: "DB_PASSWORD: x": {'msg': 'say "hi"', 'k': '` + secret + `'}" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"over_budget_repr_outer_quote_in_member", "DB_PASSWORD: " + a135 + `x": {'msg': 'say "hi"', 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"compact_repr_outer_quote_in_member", `DB_PASSWORD: x":{'msg': 'say "hi"', 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_compact_repr_outer_quote_in_member", "DB_PASSWORD: " + a135 + `x":{'msg': 'say "hi"', 'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"repr_outer_quote_in_array_element", `DB_PASSWORD: x": ['say "hi"', '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"single_outer_quote_in_member", `DB_PASSWORD: x': {"msg": "it's ok", "k": "` + secret + `"}'`, "DB_PASSWORD:[REDACTED]"},
		{"vertical_tab_before_closing_quote", "DB_PASSWORD: x\": {'a': \"1\", 'b': '" + secret + "'}\v\"", "DB_PASSWORD:[REDACTED]"},
		{"adjacent_objects", `DB_PASSWORD: x": {'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"compact_adjacent_objects", `DB_PASSWORD: x":{'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"three_adjacent_objects", `DB_PASSWORD: x": {'a': "1"} {'c': 2} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"adjacent_array_and_object", `DB_PASSWORD: x": [1, "a"] {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"line_break_between_objects", "DB_PASSWORD: x\": {'a': \"1\"}\n{'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"adjacent_objects_before_prose", `msg: "DB_PASSWORD: x": {'a': "1"} {'b': '` + secret + `'}" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"path_prefixed_adjacent_objects", `DB_PASSWORD: x"S0/creds: {'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"compact_path_prefixed_adjacent_objects", `DB_PASSWORD: x"S0/creds:{'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"closed_container_before_spaced_delimiter", `DB_PASSWORD: x"S0:[0]/creds: {'a': "1", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"compact_repr_adjacent_objects", `DB_PASSWORD: x":{'msg': 'say "hi"'} {'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_adjacent_objects", "DB_PASSWORD: " + a135 + `x": {'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_compact_adjacent_objects", "DB_PASSWORD: " + a135 + `x":{'a': "1"} {'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_adjacent_objects", `DB_PASSWORD: x\": {'a': \"1\"} {'b': '` + secret + `'}\"`, redactedSensitiveTextPlaceholder},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"adjacent_objects_without_closing_quote", `msg: "DB_PASSWORD: x": {'a': "1"} {'b': "2"} for user "bob" other=ok`, `msg: "DB_PASSWORD:[REDACTED] {'a': "1"} {'b': "2"} for user "bob" other=ok`},
		{"compact_adjacent_objects_without_closing_quote", `DB_PASSWORD: x":{'a': "1"} {'b': "2"} other=ok`, `DB_PASSWORD:[REDACTED] {'b': "2"} other=ok`},
		{"adjacent_objects_closed_by_other_quote", `DB_PASSWORD: x": {'a': "1"} {'b': "2"}' other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"} {'b': "2"}' other=ok`},
		{"text_between_objects", `DB_PASSWORD: x": {'a': "1"} and {'b': "2"}" other=ok`, `DB_PASSWORD:[REDACTED] {'a': "1"} and {'b': "2"}" other=ok`},
		{"repr_object_without_closing_quote", `DB_PASSWORD: x": {'msg': 'say "hi"'} other=ok`, `DB_PASSWORD:[REDACTED] {'msg': 'say "hi"'} other=ok`},
		{"unclosed_repr_object", `DB_PASSWORD: x": {'msg': 'say "hi"' and more other=ok`, `DB_PASSWORD:[REDACTED] {'msg': 'say "hi"' and more other=ok`},
		{"vertical_tab_before_text", "DB_PASSWORD: x\": {'a': \"1\"}\vtext\" other=ok", "DB_PASSWORD:[REDACTED] {'a': \"1\"}\vtext\" other=ok"},
		{"string_closing_after_container", `msg: "DB_PASSWORD: x": {a: 1} "`, `msg: "DB_PASSWORD:[REDACTED] {a: 1} "`},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
	}
	assertUnchangedAllEntryPoints(t, `note: x": {'msg': 'say "hi"'} {'k': 'v'}" other=ok`)

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"many adjacent containers in one string", func(n int) string {
			return `DB_PASSWORD: x": ` + strings.Repeat(`{'a': "1"} `, n) + `{'b': '` + secret + `'}"`
		}},
		{"repeated adjacent containers without closing quote", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x": {'a': "1"} {'b': "2"} `, n)
		}},
		{"repeated closed repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x": {'msg': 'say "hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated unclosed repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x": {'msg': 'say "hi"', `, n)
		}},
		{"long repr member list in one container", func(n int) string {
			return `DB_PASSWORD: x": {` + strings.Repeat(`'msg': 'say "hi"', `, n) + `'k': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: x": {'a': "1"} {'b': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: x": {'a': "1"} {'b': "2"} `)
	assertLargeTextLinear(t, `DB_PASSWORD: x": {'msg': 'say "hi"', 'k': '`+secret+`', `, secret)
}

// TestOpenerScanBudgetGluedTokenKeys covers a quoted string glued to the word
// after its closing quote whose text holds many ":" or "=" delimiters, before
// or after its last whitespace. Telling whether the value after one of them
// runs into that word reads the text before it, so those reads are charged to
// the container scan budget: the work stays linear, a text that runs the
// budget out fails closed through every entry point, idempotently, and a
// pair at the end of such a string still takes the word glued to it.
func TestOpenerScanBudgetGluedTokenKeys(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_123"
	for _, tc := range []struct {
		label, in string
	}{
		{"key_glued_to_closing_quote", `DB_PASSWORD: /' x token:abc'` + secret},
		{"key_after_delimiters_glued_to_closing_quote", `DB_PASSWORD: /' a:b:c token=abc'` + secret},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, secret)
	}
	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"delimiters after the last space", func(n int) string {
			return "DB_PASSWORD: /' " + strings.Repeat("a:", n) + "'x"
		}},
		{"delimiters before the last space", func(n int) string {
			return "DB_PASSWORD: /'" + strings.Repeat(":", n) + " 'x"
		}},
		{"equals signs in a double-quoted string", func(n int) string {
			return `DB_PASSWORD: /" ` + strings.Repeat("a=", n) + `"x`
		}},
		{"delimiters in an over-budget value", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat("b", 135) + "/' " + strings.Repeat("a:", n) + "'x"
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			// At 1024 units and more, rereading the text before each
			// delimiter dominates, so 16 times the units would take some 256
			// times as long.
			small, large := shape.mk(1024), shape.mk(16384)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("1024 units took %v, 16384 units took %v: ratio %.0f, want linear", tSmall, tLarge, ratio)
			}
		})
	}
	unit := "a:"
	in := "DB_PASSWORD: /' " + strings.Repeat(unit, (224<<10)/len(unit)) + "'" + secret
	assertLargeTextFailsClosed(t, in)
}

// TestValueEndingInsideEnclosedContainers covers a sensitive value that is a
// quoted string, in a quote that may be escaped, whose closing quote seems to
// come inside the "{" or "[" containers that the string encloses: at a quote
// in a member in other quotes, as the "hi" of
// DB_PASSWORD: "{'msg': 'say "hi"', 'k': 'SECRET'}", the repr Python gives a
// dict, or the it's of DB_PASSWORD: '{"msg": "it's ok", "k": "SECRET"}'; or
// at a member of the first of several containers separated by whitespace, as
// in DB_PASSWORD: "{'a': "1"} {'b': 'SECRET'}". The containers open right
// after the opening quote or after a ":" or "=" in the string, also after a
// path, as in "S0/x: {...}", within the nested-key budget and past it, and in
// a JSON member. Every entry point redacts the value together with all the
// containers, up to the quote that closes the string, and keeps the text
// after that quote. Containers that stay open, that the string does not close
// right after, or that the string closes right after keep the text after the
// value, and long, many or repeated containers stay linear.
func TestValueEndingInsideEnclosedContainers(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a160 := strings.Repeat("a", 160)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"repr_object", `DB_PASSWORD: "{'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"adjacent_objects", `DB_PASSWORD: "{'a': "1"} {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"json_member_repr_object", `msg={"DB_PASSWORD": "{'msg': 'say "hi"', 'k': '` + secret + `'}"}`, `msg={"DB_PASSWORD":"[REDACTED]"}`},
		{"single_quoted_object_with_apostrophe", `DB_PASSWORD: '{"msg": "it's ok", "k": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"repr_array", `DB_PASSWORD: "['say "hi"', '` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"delimiter_then_repr_object", `DB_PASSWORD: ": {'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"delimiter_then_adjacent_objects", `DB_PASSWORD: ": {'msg': "it's ok"} {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_delimiter_then_repr_object", `DB_PASSWORD: \": {'msg': 'say \"hi\"', 'k': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"over_budget_repr_object", `DB_PASSWORD: "{` + a160 + ` 'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"path_then_repr_object", `DB_PASSWORD: "S0/x: {'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"compact_path_then_repr_object", `DB_PASSWORD: "S0/x:{'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"path_then_repr_array", `DB_PASSWORD: "S0/x: ['say "hi"', '` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"single_quoted_path_then_object_with_apostrophe", `DB_PASSWORD: 'S0/x: {"msg": "it's ok", "k": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"path_then_adjacent_objects", `DB_PASSWORD: "S0/x: {'a': "1"} {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_repr_object", `DB_PASSWORD: \"{'msg': 'say \"hi\"', 'k': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"escaped_path_then_adjacent_objects", `DB_PASSWORD: \"S0/x: {'a': \"1\"} {'b': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"over_budget_adjacent_objects", `DB_PASSWORD: "{` + a160 + ` 'a': "1"} {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_path_then_repr_object", `DB_PASSWORD: "S0/` + a160 + `: {'msg': 'say "hi"', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"nested_object_then_member", `DB_PASSWORD: "{'a': {"b": 1}, 'c': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"nested_object_then_adjacent_object", `DB_PASSWORD: "{'a': {'b': "1"}} {'c': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"repr_object_before_fields", `DB_PASSWORD: "{'msg': 'say "hi"', 'k': '` + secret + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"escaped_repr_object_in_enclosing_string", `msg: "DB_PASSWORD: \"{'msg': 'say \"hi\"', 'k': '` + secret + `'}\"" for user "bob"`, `msg: "DB_PASSWORD:\"[REDACTED]\"" for user "bob"`},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}

	// Quote-led text with a quoted sensitive key and a malformed member fails
	// closed in every entry point but scrubErrorText, which redacts the value.
	jsonLed := `"password": "S0/x: {'msg': 'say "hi"', 'k': '` + secret + `'}"`
	if got, want := scrubErrorText(jsonLed), `"password":"[REDACTED]"`; got != want {
		t.Errorf("scrubErrorText(%q) = %q, want %q", jsonLed, got, want)
	}
	assertAllEntryPoints(t, "quote_led_json_member", jsonLed, unparseableJSONPlaceholder, secret)
	if got := formatContentTextPart(t, jsonLed); got != unparseableJSONPlaceholder {
		t.Errorf("FormatContentParts(%q) = %q, want %q", jsonLed, got, unparseableJSONPlaceholder)
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"repr_object_without_closing_quote", `DB_PASSWORD: "{'msg': 'say "hi"'} other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"path_then_repr_object_without_closing_quote", `DB_PASSWORD: "S0/x: {'msg': 'say "hi"'} other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"unclosed_repr_object", `DB_PASSWORD: "{'msg': 'say "hi"' and more other=ok`, `DB_PASSWORD:"[REDACTED]" and more other=ok`},
		{"adjacent_objects_without_closing_quote", `DB_PASSWORD: "{'a': "1"} {'b': "2"} other=ok`, `DB_PASSWORD:"[REDACTED]" {'b': "2"} other=ok`},
		{"adjacent_objects_closed_by_other_quote", `DB_PASSWORD: "{'a': "1"} {'b': "2"}' other=ok`, `DB_PASSWORD:"[REDACTED]" {'b': "2"}' other=ok`},
		{"text_between_objects", `DB_PASSWORD: "{'a': "1"} and {'b': "2"}" other=ok`, `DB_PASSWORD:"[REDACTED]" and {'b': "2"}" other=ok`},
		{"object_closed_by_its_own_quote", `DB_PASSWORD: "{a: 1}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"object_with_apostrophe_closed_by_its_own_quote", `DB_PASSWORD: "{a: it's}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"escaped_json_object_value", `password: "{\"a\": 1}" other=ok`, `password:"[REDACTED]" other=ok`},
		{"value_opening_unclosed_brace", `password: "{abc" user=bob`, `password:"[REDACTED]" user=bob`},
		{"apostrophes_in_values_closed_by_their_quotes", `DB_PASSWORD: "{it's}" user: "{bob's}" other=ok`, `DB_PASSWORD:"[REDACTED]" user: "{bob's}" other=ok`},
		{"enclosing_string_closing_after_value", `msg: "DB_PASSWORD: "{a: 1}" for user "bob"`, `msg: "DB_PASSWORD:"[REDACTED]" for user "bob"`},
		{"string_closing_after_container", `msg: "DB_PASSWORD: ": {a: 1} "`, `msg: "DB_PASSWORD:"[REDACTED]"`},
		{"prose_after_enclosing_string", `msg: "DB_PASSWORD: x" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"stray_quote_container_with_apostrophe", `DB_PASSWORD: x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
	}
	assertUnchangedAllEntryPoints(t, `note: "{'msg': 'say "hi"'} {'k': 'v'}" other=ok`)
	assertUnchangedAllEntryPoints(t, `x"S0/creds:{a: it's}" other=ok`)

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"many adjacent containers in one quoted value", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat(`{'a': "1"} `, n) + `{'b': '` + secret + `'}"`
		}},
		{"repeated quoted adjacent containers without closing quote", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': "1"} {'b': "2"} `, n)
		}},
		{"repeated quoted repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'msg': 'say "hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated unclosed quoted repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'msg': 'say "hi"', `, n)
		}},
		{"repeated quoted path-prefixed repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "S0/x: {'msg': 'say "hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated unclosed quoted path-prefixed repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "S0/x: {'msg': 'say "hi"', `, n)
		}},
		{"long repr member list in one quoted value", func(n int) string {
			return `DB_PASSWORD: "{` + strings.Repeat(`'msg': 'say "hi"', `, n) + `'k': '` + secret + `'}"`
		}},
		{"many delimiters before containers in one quoted value", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat(`a: {b} `, n) + `{'msg': 'say "hi"', 'k': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': "1"} {'b': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': "1"} {'b': "2"} `)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'msg': 'say "hi"', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'msg': 'say "hi"', 'k': '`+secret+`', `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "S0/x: {'msg': 'say "hi"', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "S0/x: {'msg': 'say "hi"', 'k': '`+secret+`', `, secret)

	// A value that opens a "{" that never closes, as the {abc of
	// password: "{abc", is not scanned for containers when a stop follows
	// its closing quote, so that many such values are redacted one by one
	// rather than running the scan budget out and failing the text closed.
	unit := `password: "{abc" user=bob `
	in := strings.Repeat(unit, largeTextLen()/len(unit))
	want := strings.Repeat(`password:"[REDACTED]" user=bob `, largeTextLen()/len(unit))
	for _, ep := range largeTextEntryPoints(t) {
		if got := ep.f(in); got != want {
			t.Errorf("%s(%d bytes of %q) = %.80q, want each value redacted", ep.name, len(in), unit, got)
		}
	}
}

// TestMixedQuoteContainersAfterControlSpace covers a quoted string that spans
// past a value and encloses delimited containers with members in other quotes,
// as in DB_PASSWORD: x": {'a': '1', 'b': 'SECRET'}", when delimiter whitespace
// other than the regular-expression spaces, as a vertical tab or another
// control byte, comes before the closing quote or between the containers,
// also between compact containers that open in the value, as in
// DB_PASSWORD: x":{'a':'1'}\v{'b':'1', 'c':'SECRET'}": every entry point
// redacts the value together with the containers, within the nested-key
// budget and past it, as it does for a space, while text after such
// whitespace keeps the text after the value.
func TestMixedQuoteContainersAfterControlSpace(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"vertical_tab_before_closing_quote", "DB_PASSWORD: x\": {'a': '1', 'b': '" + secret + "'}\v\"", "DB_PASSWORD:[REDACTED]"},
		{"vertical_tab_between_containers", "DB_PASSWORD: x\": {'a': '1'}\v{'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"unit_separator_before_closing_quote", "DB_PASSWORD: x\": {'a': '1', 'b': '" + secret + "'}\x1f\"", "DB_PASSWORD:[REDACTED]"},
		{"delete_between_containers", "DB_PASSWORD: x\": {'a': '1'}\x7f{'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"compact_vertical_tab_before_closing_quote", "DB_PASSWORD: x\":{'a': '1', 'b': '" + secret + "'}\v\"", "DB_PASSWORD:[REDACTED]"},
		{"path_prefixed_vertical_tab_between_containers", "DB_PASSWORD: x\"S0/creds: {'a': '1'}\v{'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_vertical_tab_before_closing_quote", "DB_PASSWORD: " + a135 + "x\": {'a': '1', 'b': '" + secret + "'}\v\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_vertical_tab_between_containers", "DB_PASSWORD: " + a135 + "x\": {'a': '1'}\v{'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"compact_vertical_tab_between_containers", "DB_PASSWORD: x\":{'a':'1'}\v{'b':'1', 'c':'" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"compact_vertical_tab_between_arrays", "DB_PASSWORD: x\":['a']\v['b', '" + secret + "']\"", "DB_PASSWORD:[REDACTED]"},
		{"compact_path_prefixed_vertical_tab_between_containers", "DB_PASSWORD: x\"S0/creds:{'a':'1'}\v{'b':'1', 'c':'" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"compact_vertical_tab_between_containers_before_prose", "msg: \"DB_PASSWORD: x\":{'a':'1'}\v{'b':'1', 'c':'" + secret + "'}\" for user \"bob\"", "msg: \"DB_PASSWORD:[REDACTED] for user \"bob\""},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want, secret)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		} else if again := formatContentTextPart(t, got); again != got {
			t.Errorf("[%s] FormatContentParts second pass(%q) = %q, want unchanged", tc.label, got, again)
		}
	}
	for _, tc := range []struct {
		label, in, want string
	}{
		{"vertical_tab_before_text", "DB_PASSWORD: x\": {'a': '1'}\vtext\" other=ok", "DB_PASSWORD:[REDACTED] {'a': '1'}\vtext\" other=ok"},
		{"vertical_tab_before_text_between_containers", "DB_PASSWORD: x\": {'a': '1'}\vtext {'b': '2'}\" other=ok", "DB_PASSWORD:[REDACTED] {'a': '1'}\vtext {'b': '2'}\" other=ok"},
	} {
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("[%s] scrubErrorText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.label, tc.in, tc.want)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("[%s] FormatContentParts(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
		}
	}
}

// TestValueEndingInsideEnclosedContainersBeforeStops covers a sensitive value
// that is a quoted string, in a quote that may be escaped, enclosing "{" or "["
// containers, whose first closing quote comes inside a member in other quotes
// and is followed by whitespace or punctuation rather than a word, as the
// " hi" of DB_PASSWORD: "{'msg': 'say " hi"', 'b': 'SECRET'}", or opens a
// member string that holds a line break, as in
// DB_PASSWORD: "{'a': " 1\n2", 'b': 'SECRET'}", and containers separated by a
// "," in the string, as in DB_PASSWORD: "{'msg': 'say "hi"'}, {'b': 'SECRET'}"
// or after a stray quote, as in DB_PASSWORD: x": {'a': '1'}, {'b': 'SECRET'}".
// They come within the nested-key budget and past it, in a JSON member, in
// single quotes and in prose. Every entry point redacts the value together
// with all the containers, up to the quote that closes the string, and keeps
// the text after that quote. A "," that no container follows, a string that
// closes right after its first container, as in
// password: "{'a': 1}", {'b': 'benign'}, and values whose closing quote opens
// no member, as "{abc", "{" or "{'a': 'x", keep the text after the value, also
// when many of them make up a large text.
func TestValueEndingInsideEnclosedContainersBeforeStops(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a160 := strings.Repeat("a", 160)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"space_after_inner_quote", `DB_PASSWORD: "{'msg': 'say " hi"', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"lowercase_key_space_after_inner_quote", `password: "{'msg': 'say " hi"', 'b': '` + secret + `'}"`, `password:"[REDACTED]"`},
		{"array_space_after_inner_quote", `DB_PASSWORD: "['say " "', '` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"comma_after_inner_quote", `DB_PASSWORD: "{'msg': 'say ", hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_space_after_inner_quote", `DB_PASSWORD: \"{'a': 'x \" \"', 'b': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"json_member_space_after_inner_quote", `err: {"password": "{'a': 'x " "', 'b': '` + secret + `'}"}`, `err: {"password":"[REDACTED]"}`},
		{"member_string_with_line_break", "DB_PASSWORD: \"{'a': \" 1\n2\", 'b': '" + secret + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"set_member_after_space", `DB_PASSWORD: "{" a", "` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"path_then_set_member_after_space", `DB_PASSWORD: "S0/x: {" a", "` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"semicolon_after_inner_quote", `DB_PASSWORD: "{'msg': 'say "; hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"ampersand_after_inner_quote", `DB_PASSWORD: "{'msg': 'say "& hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"period_after_inner_quote", `DB_PASSWORD: "{'msg': 'say ". hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"parenthesis_after_inner_quote", `DB_PASSWORD: "{'msg': 'say ") hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"bracket_after_inner_quote", `DB_PASSWORD: "{'msg': 'say "] hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"brace_after_inner_quote", `DB_PASSWORD: "{'msg': 'say "} hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"tab_after_inner_quote", "DB_PASSWORD: \"{'msg': 'say \"\t hi', 'b': '" + secret + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_space_after_inner_quote", `DB_PASSWORD: "{` + a160 + ` 'msg': 'say " hi"', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_comma_after_inner_quote", `DB_PASSWORD: "{` + a160 + ` 'msg': 'say ", hi', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_path_then_space_after_inner_quote", `DB_PASSWORD: "S0/` + a160 + `: {'msg': 'say " hi"', 'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_array_space_after_inner_quote", `DB_PASSWORD: "[` + a160 + `, 'say " "', '` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"json_value_space_after_inner_quote", `msg={"DB_PASSWORD": "{'msg': 'say " hi"', 'b': '` + secret + `'}"}`, `msg={"DB_PASSWORD":"[REDACTED]"}`},
		{"json_value_comma_after_inner_quote", `msg={"DB_PASSWORD": "{'msg': 'say ", hi', 'b': '` + secret + `'}"}`, `msg={"DB_PASSWORD":"[REDACTED]"}`},
		{"single_quoted_space_after_inner_quote", `DB_PASSWORD: '{"msg": "say ' hi'", "b": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"single_quoted_comma_after_inner_quote", `DB_PASSWORD: '{"msg": "say ', hi", "b": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"single_quoted_array_space_after_inner_quote", `DB_PASSWORD: '["say ' '", "` + secret + `"]'`, `DB_PASSWORD:'[REDACTED]'`},
		{"space_after_inner_quote_before_fields", `DB_PASSWORD: "{'msg': 'say " hi"', 'b': '` + secret + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"escaped_space_after_inner_quote_in_enclosing_string", `msg: "DB_PASSWORD: \"{'msg': 'say \" hi\"', 'b': '` + secret + `'}\"" for user "bob"`, `msg: "DB_PASSWORD:\"[REDACTED]\"" for user "bob"`},
		{"comma_separated_objects", `DB_PASSWORD: "{'msg': 'say "hi"'}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"comma_separated_objects_before_fields", `DB_PASSWORD: "{'msg': 'say "hi"'}, {'b': '` + secret + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"three_comma_separated_objects", `DB_PASSWORD: "{'a': 'x "y'}, {'c': 1}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"comma_separated_arrays", `DB_PASSWORD: "['say "hi"'], ['` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"same_quote_member_comma_separated_objects", `DB_PASSWORD: "{'a': "1"}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"same_quote_key_comma_separated_objects", `DB_PASSWORD: "{"a": 1}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"path_then_comma_separated_objects", `DB_PASSWORD: "S0/x: {'a': "1"}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_comma_separated_objects", `DB_PASSWORD: \"{'msg': 'say \"hi\"'}, {'b': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"over_budget_comma_separated_objects", `DB_PASSWORD: "{` + a160 + `: 'say "hi"'}, {'b': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"json_value_comma_separated_objects", `msg={"DB_PASSWORD": "{'msg': 'say "hi"'}, {'b': '` + secret + `'}"}`, `msg={"DB_PASSWORD":"[REDACTED]"}`},
		{"single_quoted_comma_separated_objects_with_apostrophe", `DB_PASSWORD: '{"msg": "it's"}, {"b": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"stray_quote_comma_separated_mixed_quote_objects", `DB_PASSWORD: x": {'a': '1'}, {'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_path_comma_separated_mixed_quote_objects", `DB_PASSWORD: x"S0/creds: {'a': '1'}, {'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_compact_comma_separated_mixed_quote_objects", `DB_PASSWORD: x":{'a': '1'},{'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_comma_after_object_closed_in_value", `DB_PASSWORD: x":{'a':'1'},{'b':'` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_comma_separated_same_quote_member", `DB_PASSWORD: x":{'a': "1"}, {'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"spaced_stray_quote_comma_separated_objects", `DB_PASSWORD: x": {'msg': "it's ok"}, {'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"string_closed_before_comma_and_object", `password: "{'a': 1}", {'b': 'benign'} user=bob`, `password:"[REDACTED]", {'b': 'benign'} user=bob`},
		{"string_with_inner_quote_closed_before_comma_and_object", `password: "{'a': 'x "y'}", {'b': 'benign'} user=bob`, `password:"[REDACTED]", {'b': 'benign'} user=bob`},
		{"comma_then_text", `DB_PASSWORD: "{'a': "1"}, and {'b': "2"}" other=ok`, `DB_PASSWORD:"[REDACTED]", and {'b': "2"}" other=ok`},
		{"comma_before_closing_quote", `DB_PASSWORD: "{'a': "1"}, " other=ok`, `DB_PASSWORD:"[REDACTED]", " other=ok`},
		{"comma_separated_objects_without_closing_quote", `DB_PASSWORD: "{'a': "1"}, {'b': "2"} other=ok`, `DB_PASSWORD:"[REDACTED]", {'b': "2"} other=ok`},
		{"comma_separated_objects_closed_by_other_quote", `DB_PASSWORD: "{'a': "1"}, {'b': "2"}' other=ok`, `DB_PASSWORD:"[REDACTED]", {'b': "2"}' other=ok`},
		{"single_quoted_comma_separated_objects", `DB_PASSWORD: '{"a": 1}, {"b": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"stray_quote_comma_then_text", `DB_PASSWORD: x": {'a': '1'}, text {'b': '2'}" other=ok`, `DB_PASSWORD:[REDACTED] {'a': '1'}, text {'b': '2'}" other=ok`},
		{"stray_quote_comma_before_closing_quote", `DB_PASSWORD: x": {'a': '1'}, " other=ok`, `DB_PASSWORD:[REDACTED] {'a': '1'}, " other=ok`},
		{"unclosed_member_string", `password: "{'a': 'x" user=bob`, `password:"[REDACTED]" user=bob`},
		{"lone_brace", `password: "{" user=bob`, `password:"[REDACTED]" user=bob`},
		{"unclosed_brace_before_words", `password: "{not a container" user=bob`, `password:"[REDACTED]" user=bob`},
		{"apostrophe_word_before_space", `DB_PASSWORD: "{msg: it's " ok}" user=bob`, `DB_PASSWORD:"[REDACTED]" ok}" user=bob`},
		{"value_before_quoted_brace", `DB_PASSWORD: "{abc" or "}" other=ok`, `DB_PASSWORD:"[REDACTED]" or "}" other=ok`},
		{"colon_before_closing_quote_without_container", `password: "a: " other=ok`, `password:"[REDACTED]" other=ok`},
		{"colon_in_array_before_closing_quote", `DB_PASSWORD: "[note: " see "]" other=ok`, `DB_PASSWORD:"[REDACTED]" see "]" other=ok`},
		{"prose_after_spaced_quote", `password: "say " hi there "bob"`, `password:"[REDACTED]" hi there "bob"`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	// A JSON object keeps the members after a value that the string closes
	// right after, or whose closing quote opens no member.
	for _, tc := range []struct {
		in, keep string
	}{
		{`{"password": "{'a': 1}", "other": {"b": "benign"}}`, `"other":{"b":"benign"}`},
		{`{"password": "{not a container", "user": "bob"}`, `"user":"bob"`},
		{`{"password": "{", "user": {"name": "bob"}}`, `"user":{"name":"bob"}`},
		{`{"password": "[1, ", "user": "bob"}`, `"user":"bob"`},
	} {
		for _, ep := range largeTextEntryPoints(t) {
			got := strings.ReplaceAll(ep.f(tc.in), " ", "")
			if !strings.Contains(got, `"password":"[REDACTED]"`) || !strings.Contains(got, tc.keep) {
				t.Errorf("%s(%q) = %q, want the value redacted and %s kept", ep.name, tc.in, got, tc.keep)
			}
		}
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"many comma-separated containers in one quoted value", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat(`{'a': "1"}, `, n) + `{'b': '` + secret + `'}"`
		}},
		{"many comma-separated mixed-quote containers after a stray quote", func(n int) string {
			return `DB_PASSWORD: x": ` + strings.Repeat(`{'a': '1'}, `, n) + `{'b': '` + secret + `'}"`
		}},
		{"repeated quoted repr containers with a space after the inner quote", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'msg': 'say " hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated unclosed quoted repr containers with a space after the inner quote", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'msg': 'say " hi"', `, n)
		}},
		{"repeated comma-separated containers without closing quote", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': "1"}, {'b': "2"}, `, n)
		}},
		{"long repr member list with spaces after inner quotes", func(n int) string {
			return `DB_PASSWORD: "{` + strings.Repeat(`'msg': 'say " hi"', `, n) + `'k': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			small, large := shape.mk(smallN), shape.mk(largeN)
			assertRedactedIdempotent(t, shape.name, large, secret)
			scrub := func(s string) {
				_ = redactErrorMessage(s)
				_, _ = sanitizeRawText(s)
				_, _, _ = SmartTruncate(s, -1)
			}
			scrub(small)
			scrub(large)
			if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
				t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
			}
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "{'msg': 'say " hi"', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'msg': 'say " hi"', 'k': '`+secret+`', `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': "1"}, {'b': '`+secret+`'}" `, secret)

	// Values whose closing quote opens no member are redacted one by one
	// rather than running the scan budget out and failing the text closed.
	for _, unit := range []string{`password: "{'a': 'x" user=bob `, `password: "{" user=bob `} {
		in := strings.Repeat(unit, largeTextLen()/len(unit))
		want := strings.Repeat(`password:"[REDACTED]" user=bob `, largeTextLen()/len(unit))
		for _, ep := range largeTextEntryPoints(t) {
			if got := ep.f(in); got != want {
				t.Errorf("%s(%d bytes of %q) = %.80q, want each value redacted", ep.name, len(in), unit, got)
			}
		}
	}
}

// TestMixedQuoteMembersAfterControlSpace covers quoted members of containers
// that a spanning or quoted string encloses, when delimiter whitespace other
// than the regular-expression spaces, as a vertical tab or another control
// byte, comes between such a member and the "{", "[", ",", ":" or "=" before
// it or after it: before a quoted key, as in
// password: x"S0/creds:{\v'k': 1, m: SECRET}", between a key and its ":",
// before or after a quoted value or element, between a triple-escaped key and
// its ":", and before or after a member that the closing quote of the string
// opens, as in password: x"S0/creds:{\v"k": SECRET}" or
// DB_PASSWORD: "S0/x: [\v" a", "SECRET"]", within the nested-key budget and
// past it. Every entry point redacts the value together with the containers,
// as it does for a space, while a container without such a member, text after
// it and text without a sensitive key keep their text.
func TestMixedQuoteMembersAfterControlSpace(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"vertical_tab_before_quoted_key", "password: x\"S0/creds:{\v'k': 1, m: " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_after_quoted_key", "password: x\"S0/creds:{'k'\v: 1, m: " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_before_quoted_value", "password: x\"S0/creds:{k:\v'a b', m: " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_before_quoted_element", "password: x\"S0/creds:[\v'a b', " + secret + "]\"", "password:[REDACTED]"},
		{"vertical_tab_after_quoted_value", "password: x\"S0/creds:{k: 'a b'\v, m: " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_before_same_quote_key", "password: x\"S0/creds:{\v\"k\": " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_before_same_quote_element", "password: x\"S0/creds:[\v\"a\", " + secret + "]\"", "password:[REDACTED]"},
		{"vertical_tab_before_same_quote_value", "password: x\"S0/creds:{k:\v\"a\", m: " + secret + "}\"", "password:[REDACTED]"},
		{"vertical_tab_after_triple_escaped_key", `password: x\"S0/creds:{\\\"k\\\"` + "\v" + `: 1, m: ` + secret + `}\"`, "password:[REDACTED]"},
		{"vertical_tab_before_closing_quote_set_member", "DB_PASSWORD: \"S0/x: {\v\" a\", \"" + secret + "\"}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"vertical_tab_before_closing_quote_element", "DB_PASSWORD: \"S0/x: [\v\" a\", \"" + secret + "\"]\"", `DB_PASSWORD:"[REDACTED]"`},
		{"vertical_tab_after_closing_quote_member", "password: \"S0/x: {'k': \" " + secret + "\"\v}\"", `password:"[REDACTED]"`},
		{"vertical_tab_after_closing_quote_member_in_container_after_text", "DB_PASSWORD: \"x {k: \" a\"\v, m: '" + secret + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"unit_separator_before_quoted_key", "password: x\"S0/creds:{\x1f'k': 1, m: " + secret + "}\"", "password:[REDACTED]"},
		{"delete_after_quoted_key", "password: x\"S0/creds:{'k'\x7f: 1, m: " + secret + "}\"", "password:[REDACTED]"},
		{"over_budget_vertical_tab_before_quoted_key", "password: " + a135 + "x\"S0/creds:{\v'k': 1, m: " + secret + "}\"", "password:[REDACTED]"},
		{"over_budget_vertical_tab_before_quoted_element", "password: " + a135 + "x\"S0/creds:[\v'a b', " + secret + "]\"", "password:[REDACTED]"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	for _, tc := range []struct {
		label, in, want string
	}{
		{"vertical_tab_before_unquoted_key", "password: x\"S0/creds:{\vk: v}\" other=ok", "password:[REDACTED] v}\" other=ok"},
		{"vertical_tab_before_text", "password: x\"S0/creds:{\v'k': 1}\vtext\" other=ok", "password:[REDACTED] 1}\vtext\" other=ok"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedAllEntryPoints(t, "note: x\"S0/creds:{\v'k': 1}\" other=ok")
}

// TestMixedQuoteContainersWithStringPrefixes covers a quoted member with a
// Python string prefix (b, r, u or f, or br, rb, fr or rf, in either case) in a
// container that a spanning string encloses, as the b'x}SECRET' of
// DB_PASSWORD: x"S0/creds:{b: b'x}SECRET'}" or the u'k' of a Python 2 repr: the
// quote after the prefix opens a string when it closes before the end of the
// containers, so that a bracket in it does not close a container, and every
// entry point redacts the value together with the containers. A quote joined
// to a longer word, as in it's, don't or herb'x, a prefix whose string does not
// close or is not followed by the end of a member, as the B's of
// Plan B's budget, and text without a sensitive key keep their text.
func TestMixedQuoteContainersWithStringPrefixes(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a135 := strings.Repeat("a", 135)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"bytes_prefix", `DB_PASSWORD: x"S0/creds:{b: b'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"raw_prefix", `DB_PASSWORD: x"S0/creds:{b: r'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"raw_bytes_prefix", `DB_PASSWORD: x"S0/creds:{b: rb'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"upper_case_bytes_raw_prefix", `DB_PASSWORD: x"S0/creds:{b: BR'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"unicode_prefix", `DB_PASSWORD: x"S0/creds:{b: u'x]` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"format_prefix_in_array", `DB_PASSWORD: x"S0/creds:[1, f'x]` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"python2_repr", `DB_PASSWORD: x"S0/creds:{u'k': u'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"prefix_before_member_that_closing_quote_opens", `DB_PASSWORD: "x {b: b'x}y', k: " ` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"python2_repr_before_member_that_closing_quote_opens", `DB_PASSWORD: "x {u'k': u'x]y', k: " ` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_member_around_closing_quote", `DB_PASSWORD: "{k: b'say " hi', 'n': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_bytes_prefix", "DB_PASSWORD: " + a135 + `x"S0/creds:{b: b'x}` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	for _, tc := range []struct {
		label, in, want string
	}{
		{"contraction", `DB_PASSWORD: x"S0/creds:{a: don't}" other=ok`, `DB_PASSWORD:[REDACTED] don't}" other=ok`},
		{"possessive", `DB_PASSWORD: x"S0/creds:{a: Bob's}" other=ok`, `DB_PASSWORD:[REDACTED] Bob's}" other=ok`},
		{"longer_word_before_quote", `DB_PASSWORD: x"S0/creds:{a: herb'x}y'}" other=ok`, `DB_PASSWORD:[REDACTED] herb'x}y'}" other=ok`},
		{"unclosed_prefixed_string", `DB_PASSWORD: x"S0/creds:{b: b'unclosed}" other=ok`, `DB_PASSWORD:[REDACTED] b'unclosed}" other=ok`},
		{"prefix_letter_before_apostrophe_word", `DB_PASSWORD: x"S0/creds:{a: b's, c: it's}" other=ok`, `DB_PASSWORD:[REDACTED] b's, c: it's}" other=ok`},
		{"prefix_letter_word_before_quoted_member", `DB_PASSWORD: x"S0/creds:{a: Plan B's budget, c: '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedAllEntryPoints(t, `note: x"S0/creds:{b: b'x}y'}" other=ok`)
	assertUnchangedAllEntryPoints(t, `x"S0/creds:{a: it's}" other=ok`)
}

// TestClosingQuoteAfterStringPrefix covers a sensitive value that is a quoted
// string, in a quote that may be escaped, enclosing "{" or "[" containers,
// whose first closing quote comes right after a Python string prefix (b, r, u
// or f, or br, rb, fr or rf, in either case) where a member starts, as the b"
// of DB_PASSWORD: "{'k': b" SECRET", 'm': 'SECRET'}", and is followed by
// whitespace, a line break or punctuation, and a member with such a prefix
// whose string holds a line break and a bracket, as the b'x}\nSECRET' of
// DB_PASSWORD: "{'a': b'x}\nSECRET', 'b': 'say " hi"', 'c': 'SECRET'}". They
// come in arrays and in nested, multi-line or path-prefixed objects, within
// the nested-key budget and past it, in a JSON member, in single and escaped
// quotes, in prose and after a stray quote. Every entry point redacts the
// value together with the containers, up to the quote that closes the string,
// and keeps the text after that quote. A quote joined to a word that is no
// string prefix, as in it's, Bob's, herb'x, ub" or Bs", a prefix after another
// word, and a prefixed string that does not close or that no end of a member
// follows, as the B's of Plan B's budget, keep the text after the value, and
// long, repeated and unclosed prefixed strings stay linear.
func TestClosingQuoteAfterStringPrefix(t *testing.T) {
	runInParallel(t)
	const secret, secret2 = "SECRET_PROBE_98765", "SECRET_PROBE_24680"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"space_after_bytes_prefix_quote", `DB_PASSWORD: "{'k': b" ` + secret + `", 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"unicode_prefix_before_brace", `DB_PASSWORD: "{'k': u" x}` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"bytes_prefix_in_array", `DB_PASSWORD: "[b" ` + secret + `", '` + secret2 + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"raw_bytes_prefix_line_break", "DB_PASSWORD: \"{'k': rb\"\n" + secret + "\", 'm': '" + secret2 + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"single_quoted_bytes_prefix", `DB_PASSWORD: '{k: b' ` + secret + `', 'm': '` + secret2 + `'}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"multiline_prefixed_member_before_inner_quote", "DB_PASSWORD: \"{'a': b'x}\n" + secret + "', 'b': 'say \" hi\"', 'c': '" + secret2 + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_outer_quote", `DB_PASSWORD: \"{'k': b\" ` + secret + `\", 'm': '` + secret2 + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"escaped_outer_quote_line_break", `DB_PASSWORD: \"{'k': rb\"` + "\n" + secret + `\", 'm': '` + secret2 + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"upper_unicode_prefix_line_break", "DB_PASSWORD: \"{'k': U\"\n" + secret + "\", 'm': '" + secret2 + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"tab_after_prefix_quote", "DB_PASSWORD: \"{'k': b\"\t" + secret + "\", 'm': '" + secret2 + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"crlf_after_prefix_quote", "DB_PASSWORD: \"{'k': b\"\r\n" + secret + "\", 'm': '" + secret2 + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"comma_after_prefix_quote", `DB_PASSWORD: "{'k': b", ` + secret + `", 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"semicolon_after_prefix_quote", `DB_PASSWORD: "{'k': b"; ` + secret + `", 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_after_comma_in_array", `DB_PASSWORD: "[1, b" ` + secret + `", '` + secret2 + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_in_nested_object", `DB_PASSWORD: "{'a': {'k': b" ` + secret + `"}, 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_after_equals", `DB_PASSWORD: "{k= b" ` + secret + `", m= '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_in_multiline_object", "DB_PASSWORD: \"{\n  'k': b\" " + secret + "\",\n  'm': '" + secret2 + "'\n}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_prefix", `DB_PASSWORD: "{` + a140 + `: 1, 'k': b" ` + secret + `", 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"json_member_prefix", `msg={"DB_PASSWORD": "{'k': b" ` + secret + `", 'm': '` + secret2 + `'}"}`, `msg={"DB_PASSWORD":"[REDACTED]"}`},
		{"single_quoted_multiline_prefixed_member", "DB_PASSWORD: '{\"a\": b\"x}\n" + secret + "\", \"b\": \"say ' hi'\", \"c\": \"" + secret2 + "\"}'", `DB_PASSWORD:'[REDACTED]'`},
		{"path_then_prefix", `DB_PASSWORD: "S0/x: {'k': b" ` + secret + `", 'm': '` + secret2 + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"container_after_text_then_prefix", `DB_PASSWORD: "x {k: b" a", m: '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_key", `DB_PASSWORD: "{'x': 1, b" k": '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_set_member", `DB_PASSWORD: "{b" a", "` + secret + `"}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_before_fields", `DB_PASSWORD: "{'k': b" ` + secret + `", 'm': '` + secret2 + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"prefix_in_enclosing_string", `msg: "DB_PASSWORD: \"{'k': b\" ` + secret + `\", 'm': '` + secret2 + `'}\"" for user "bob"`, `msg: "DB_PASSWORD:\"[REDACTED]\"" for user "bob"`},
		{"stray_quote_then_prefix", `DB_PASSWORD: x"{'k': b" ` + secret + `", 'm': '` + secret2 + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_multiline_prefixed_member", "DB_PASSWORD: x\"{'a': b'x}\n" + secret + "', 'b': 'say \" hi\"', 'c': '" + secret2 + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_multiline_prefixed_member_mixed", "DB_PASSWORD: x\"{'a': b'x}\n" + secret + "', 'c': '" + secret2 + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"stray_path_multiline_prefixed_member_mixed", "DB_PASSWORD: x\"S0/creds:{'a': b'x}\n" + secret + "', 'c': '" + secret2 + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote_then_prefix", "DB_PASSWORD: " + a140 + `"{'k': b" ` + secret + `", 'm': '` + secret2 + `'}"`, "DB_PASSWORD:[REDACTED]"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret, secret2)
		})
	}
	for _, prefix := range []string{"r", "f", "br", "fr", "rf", "B", "RB", "Rb", "U", "F"} {
		in := `DB_PASSWORD: "{'k': ` + prefix + `" ` + secret + `", 'm': '` + secret2 + `'}"`
		t.Run("prefix_"+prefix, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, in, `DB_PASSWORD:"[REDACTED]"`, secret, secret2)
		})
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"prefix_letter_word_before_quoted_member", `DB_PASSWORD: "{a: Plan B's budget, c: '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefix_letter_word_before_line_break", "DB_PASSWORD: \"{a: Plan B's\nbudget', c: '" + secret + "'}\"", `DB_PASSWORD:"[REDACTED]"`},
		{"contraction", `DB_PASSWORD: "{msg: it's " ok}" user=bob`, `DB_PASSWORD:"[REDACTED]" ok}" user=bob`},
		{"possessive", `DB_PASSWORD: "{msg: Bob's " ok}" user=bob`, `DB_PASSWORD:"[REDACTED]" ok}" user=bob`},
		{"longer_word_before_quote", `DB_PASSWORD: "{a: herb'x}y'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"prose_after_closing_quote", `msg: "DB_PASSWORD: x" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"prose_container_after_closing_quote", `msg: "DB_PASSWORD: x": {a: 1} "`, `msg: "DB_PASSWORD:[REDACTED] {a: 1} "`},
		{"unclosed_string_after_prefix", `password: "{x: b" other=ok`, `password:"[REDACTED]" other=ok`},
		{"no_member_end_after_prefixed_string", `password: "{x: b" other "bob"`, `password:"[REDACTED]" other "bob"`},
		{"prefix_after_other_word", `DB_PASSWORD: "{see b" said bob, "x": 1}" other=ok`, `DB_PASSWORD:"[REDACTED]" said bob, "x": 1}" other=ok`},
		{"two_letter_word_that_is_no_prefix", `password: "{x: Bs" other=ok, "bob": 1}`, `password:"[REDACTED]" other=ok, "bob": 1}`},
		{"prefix_letters_that_are_no_prefix", `DB_PASSWORD: "{'k': ub" a", 'm': 'b'}" other=ok`, `DB_PASSWORD:"[REDACTED]" a", 'm': 'b'}" other=ok`},
		{"prefix_letter_in_prose", `msg: "DB_PASSWORD: b" for user "bob"`, `msg: "DB_PASSWORD:[REDACTED] for user "bob"`},
		{"prefix_letter_before_apostrophe_word", `DB_PASSWORD: "[ b's ], text [ v , '1'" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"prefix_letters_in_prose", `password: "see rb" in docs, "ok"} other`, `password:"[REDACTED]" in docs, "ok"} other`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedOnEveryEntryPoint(t, `note: "{'k': b" x", 'm': 'y'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, "note: \"{'a': b'x}\ny', 'b': 'say \" hi\"'}\" other=ok")

	// The closing quote after a prefix starts a member after the byte before
	// that prefix, and a prefixed member string read past its line break is
	// charged to the scan budget.
	s := `"{'k': b" ` + secret + `", 'm': 1}"`
	qEnd := strings.Index(s, `b"`) + len(`b"`)
	if inner, outer := quotedContainerOpeners(s, 0, qEnd, nil); inner != 1 || outer != 1 {
		t.Errorf("quotedContainerOpeners(%q) = (%d, %d), want (1, 1)", s[:qEnd], inner, outer)
	}
	s = "\"{'a': b'x}\n" + secret + "', 'b': 'say \" hi\"'}\""
	qEnd = strings.Index(s, `say "`) + len(`say "`)
	st := newQuoteScanState(len(s))
	budget := st.openerScanBudget
	if outer, ok := closingQuoteInMember(s, 0, qEnd, &st); !ok || outer != 1 {
		t.Errorf("closingQuoteInMember(%q) = (%d, %v), want (1, true)", s[:qEnd], outer, ok)
	}
	if charged, least := budget-st.openerScanBudget, qEnd-2+len("'x}\n"+secret+"'"); charged < least {
		t.Errorf("closingQuoteInMember(%q) charged %d bytes, want at least %d", s[:qEnd], charged, least)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated closing quotes after a prefix", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'k': b" `+secret+`", 'm': 1}" `, n)
		}},
		{"repeated multiline prefixed members", func(n int) string {
			return strings.Repeat("DB_PASSWORD: \"{'a': b'x}\ny', 'b': 'say \" hi\"', 'c': '"+secret+"'}\" ", n)
		}},
		{"long prefixed member list", func(n int) string {
			return `DB_PASSWORD: "{` + strings.Repeat(`'a': b'x}', `, n) + `'k': b" ` + secret + `"}"`
		}},
		{"long multiline prefixed member list", func(n int) string {
			return `DB_PASSWORD: "{` + strings.Repeat("'a': b'x}\ny', ", n) + `'k': b" ` + secret + `"}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "{'k': b" `+secret+`", 'm': '`+secret2+`'}" `, secret, secret2)
	assertLargeTextLinear(t, "DB_PASSWORD: \"{'a': b'x}\n"+secret+"', 'b': 'say \" hi\"', 'c': '"+secret2+"'}\" ", secret, secret2)
	assertLargeTextLinear(t, `DB_PASSWORD: "{k: b" `)
	assertLargeTextLinear(t, "DB_PASSWORD: \"{k: b'\n")
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{a: b'x}\n")
}

// TestContainersRightAfterStrayQuote covers "{" or "[" containers that open
// right after a quote in a sensitive value, possibly after delimiter
// whitespace, where that quote is joined to a word or comes past the
// nested-key budget, as the x" of DB_PASSWORD: x"{'k': 'SECRET'}" or of
// DB_PASSWORD: x"{'msg': 'say " hi"', 'b': 'SECRET'}": the containers hold a
// member in other quotes and the string closes right after them, or the
// string seems to close inside them, in a member in other quotes, at a member
// string with a line break, or in a later container after a ",". They come in
// single and escaped quotes, in arrays, after a vertical tab or a line break,
// before other fields, after other keys and after other text in the string.
// Every entry point redacts the value together with the containers, as it
// does when a ":" comes between the quote and the containers, and keeps the
// text after the closing quote. A container that stays open, one without a
// quoted member, one that text follows inside the string, and text without a
// sensitive key keep their text, and repeated, long and unclosed containers
// stay linear, with unclosed ones redacted value by value.
func TestContainersRightAfterStrayQuote(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"inner_quote_then_space", `DB_PASSWORD: x"{'msg': 'say " hi"', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"member_string_line_break", "DB_PASSWORD: x\"{'a': \" 1\n2\", 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"comma_separated_objects", `DB_PASSWORD: x"{'a': '1'}, {'b': 'say " hi"', 'c': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_inner_quote", "DB_PASSWORD: " + a140 + `"{'msg': 'say " hi"', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_object", "DB_PASSWORD: " + a140 + `"{'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"object", `DB_PASSWORD: x"{'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_object", `DB_PASSWORD: x'{"k": "` + secret + `"}'`, "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_inner_quote", `DB_PASSWORD: x'{"msg": "say ' hi'", "b": "` + secret + `"}'`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_object", `DB_PASSWORD: x\"{'k': '` + secret + `'}\"`, "DB_PASSWORD:[REDACTED]"},
		{"escaped_inner_quote_fails_closed", `DB_PASSWORD: x\"{'msg': 'say \" hi\"', 'b': '` + secret + `'}\"`, redactedSensitiveTextPlaceholder},
		{"array", `DB_PASSWORD: x"['a', '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"array_inner_quote", `DB_PASSWORD: x"['say " hi"', '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"vertical_tab_before_object", "DB_PASSWORD: x\" \v{'k': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"vertical_tab_before_inner_quote_object", "DB_PASSWORD: x\" \v{'msg': 'say \" hi\"', 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"line_break_before_object", "DB_PASSWORD: x\"\n{'k': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"object_before_fields", `DB_PASSWORD: x"{'k': '` + secret + `'}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"inner_quote_glued", `DB_PASSWORD: x"{'msg': 'say "hi"', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"glued_quote_in_unquoted_member", `DB_PASSWORD: x"{msg: say "hi", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_glued_quote_in_unquoted_member", "DB_PASSWORD: " + a140 + `"{msg: say "hi", 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"equals_key", `DB_PASSWORD= x"{'k': '` + secret + `'}"`, "DB_PASSWORD=[REDACTED]"},
		{"lowercase_key", `password: x"{'k': '` + secret + `'}"`, "password:[REDACTED]"},
		{"two_letter_word", `DB_PASSWORD: ab"{'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"path_word", `DB_PASSWORD: S0/x"{'k': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"parenthesis", `DB_PASSWORD: ("{'k': '` + secret + `'}") other=ok`, "DB_PASSWORD:[REDACTED]) other=ok"},
		{"over_budget_single_quoted", "DB_PASSWORD: " + a140 + `'{"k": "` + secret + `"}'`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_escaped", "DB_PASSWORD: " + a140 + `\"{'k': '` + secret + `'}\"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_array_inner_quote", "DB_PASSWORD: " + a140 + `"['say " hi"', '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_vertical_tab", "DB_PASSWORD: " + a140 + "\" \v{'k': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_member_string_line_break", "DB_PASSWORD: " + a140 + "\"{'a': \" 1\n2\", 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget_comma_separated_objects", "DB_PASSWORD: " + a140 + `"{'a': '1'}, {'b': 'say " hi"', 'c': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"text_before_container", `DB_PASSWORD: x"abc {'a': "1"}"`, "DB_PASSWORD:[REDACTED]"},
		{"text_before_container_before_fields", `DB_PASSWORD: x"abc {'a': "1"}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
		{"over_budget_text_before_container", "DB_PASSWORD: " + a140 + `"abc {'a': "1"}" other=ok`, "DB_PASSWORD:[REDACTED] other=ok"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"prose_braces", `DB_PASSWORD: x"hello {world}" other=ok`, `DB_PASSWORD:[REDACTED] {world}" other=ok`},
		{"apostrophe_in_path_container", `DB_PASSWORD: x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
		{"unclosed_container", `DB_PASSWORD: x"{not closed" other=ok`, `DB_PASSWORD:[REDACTED] closed" other=ok`},
		{"unclosed_container_glued_close", `DB_PASSWORD: x"{not closed"x other=ok`, `DB_PASSWORD:[REDACTED] closed"x other=ok`},
		{"over_budget_unclosed_container", "DB_PASSWORD: " + a140 + `"{not closed" other=ok`, `DB_PASSWORD:[REDACTED] closed" other=ok`},
		{"no_quoted_member", `DB_PASSWORD: x"{a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] 1}" other=ok`},
		{"text_after_container", `DB_PASSWORD: x"{'a': '1'} text" other=ok`, `DB_PASSWORD:[REDACTED] '1'} text" other=ok`},
		{"quote_after_brace", `DB_PASSWORD: x"{" said bob" other=ok`, `DB_PASSWORD:[REDACTED] said bob" other=ok`},
		{"unclosed_repr_container", `DB_PASSWORD: x"{'msg': 'say " hi"' and more other=ok`, `DB_PASSWORD:[REDACTED] 'say " hi"' and more other=ok`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedOnEveryEntryPoint(t, `x"hello {world}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: x"{'k': 'v'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: x"{'msg': 'say " hi"', 'k': 'v'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `x"S0/creds:{a: it's}" other=ok`)

	// The containers are found from the quote, and a string that closes
	// before a stop and outside any member is not scanned for them, which
	// would read the rest of the text.
	s := `x"{'msg': 'say " hi"', 'k': '` + secret + `'}"`
	qEnd := strings.Index(s, `say "`) + len(`say "`)
	if got := spacedStrayQuoteContainer(s, 1, qEnd, strings.IndexByte(s, ' '), nil); got != 2 {
		t.Errorf("spacedStrayQuoteContainer(%q) = %d, want 2", s, got)
	}
	s = `x"{'k': '` + secret + `'}"`
	if !enclosesMixedQuoteContainer(s, 1, len(s), strings.IndexByte(s, ' '), nil) {
		t.Errorf("enclosesMixedQuoteContainer(%q) = false, want true", s)
	}
	s = `x"{not closed" other=ok ` + strings.Repeat(`{"a": 1} `, 1024)
	qEnd = strings.Index(s, `closed"`) + len(`closed"`)
	st := newQuoteScanState(len(s))
	budget := st.openerScanBudget
	if got := spacedStrayQuoteContainer(s, 1, qEnd, strings.IndexByte(s, ' '), &st); got >= 0 {
		t.Errorf("spacedStrayQuoteContainer(%.40q) = %d, want < 0", s, got)
	}
	if charged := budget - st.openerScanBudget; charged > 2*qEnd {
		t.Errorf("spacedStrayQuoteContainer(%.40q) charged %d bytes, want at most %d", s, charged, 2*qEnd)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated stray-quoted repr containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"{'msg': 'say " hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated stray-quoted objects", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"{'k': '`+secret+`'}" `, n)
		}},
		{"repeated over-budget repr containers", func(n int) string {
			return strings.Repeat("DB_PASSWORD: "+a140+`"{'msg': 'say " hi"', 'k': '`+secret+`'}" `, n)
		}},
		{"long member list after a stray quote", func(n int) string {
			return `DB_PASSWORD: x"{` + strings.Repeat(`'msg': 'say " hi"', `, n) + `'k': '` + secret + `'}"`
		}},
		{"many comma-separated objects after a stray quote", func(n int) string {
			return `DB_PASSWORD: x"` + strings.Repeat(`{'a': '1'}, `, n) + `{'b': 'say " hi"', 'c': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: x"{'msg': 'say " hi"', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: x"{'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, "DB_PASSWORD: "+a140+`"{'msg': 'say " hi"', 'k': '`+secret+`'}" `, secret)

	// Values whose string closes before a stop, outside any member, are
	// redacted one by one rather than running the scan budget out.
	for _, unit := range []string{`DB_PASSWORD: x"{not closed" other=ok `, "DB_PASSWORD: " + a140 + `"{not closed" other=ok `} {
		n := largeTextLen() / len(unit)
		in := strings.Repeat(unit, n)
		want := strings.Repeat(`DB_PASSWORD:[REDACTED] closed" other=ok `, n)
		for _, ep := range largeTextEntryPoints(t) {
			if got := ep.f(in); got != want {
				t.Errorf("%s(%d bytes of %.40q) = %.80q, want each value redacted", ep.name, len(in), unit, got)
			}
		}
	}
}

// TestStrayQuoteContainerMembersWithControlBytes covers "{" or "[" containers
// in the string of a stray quote in a sensitive value whose only quoted
// member, a value after an unquoted key or an array element, with or without
// a string prefix, holds a line break, a tab, a vertical tab or another
// control byte, as the 'x}\nSECRET' of DB_PASSWORD: x"S0/creds:{a:
// 'x}\nSECRET'}", and a stray quote that never closes whose closing quote
// opens such a member, as the "1\n2" of DB_PASSWORD: x"S0/creds:{'a': "1\n2",
// 'b': 'SECRET'}. They come after a path and right after the quote, nested,
// in single and escaped quotes, past the nested-key budget and before other
// fields. Every entry point redacts the value together with the containers,
// as it does when the member is on one line. A word with an apostrophe, a
// prefixed string that does not close, a container that stays open, a member
// string that no end of a member follows and text after the containers in
// the string keep the text after the value, text without a sensitive key
// stays unchanged, and repeated, long and unclosed such containers stay
// linear.
func TestStrayQuoteContainerMembersWithControlBytes(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"prefixed_value_line_break", "DB_PASSWORD: x\"S0/creds:{a: b'x}\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"value_line_break", "DB_PASSWORD: x\"S0/creds:{a: 'x}\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"prefixed_element_line_break", "DB_PASSWORD: x\"S0/creds:[b'x]\n" + secret + "']\"", "DB_PASSWORD:[REDACTED]"},
		{"element_line_break", "DB_PASSWORD: x\"S0/creds:['x]\n" + secret + "']\"", "DB_PASSWORD:[REDACTED]"},
		{"prefixed_value_vertical_tab", "DB_PASSWORD: x\"S0/creds:{a: b'x}\v" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"prefixed_value_tab", "DB_PASSWORD: x\"S0/creds:{a: b'x}\t" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"prefixed_value_carriage_return", "DB_PASSWORD: x\"S0/creds:{a: b'x}\r" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"prefixed_value_form_feed", "DB_PASSWORD: x\"S0/creds:{a: b'x}\f" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"value_crlf", "DB_PASSWORD: x\"S0/creds:{a: 'x}\r\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"value_unit_separator", "DB_PASSWORD: x\"S0/creds:{a: 'x}\x1f" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"value_delete_byte", "DB_PASSWORD: x\"S0/creds:{a: 'x}\x7f" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"element_after_other_element", "DB_PASSWORD: x\"S0/creds:[1, 'x]\n" + secret + "']\"", "DB_PASSWORD:[REDACTED]"},
		{"nested_object_value", "DB_PASSWORD: x\"S0/creds:{a: {b: 'x}\n" + secret + "'}}\"", "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_stray_quote", "DB_PASSWORD: x'S0/creds:{a: \"x}\n" + secret + "\"}'", "DB_PASSWORD:[REDACTED]"},
		{"escaped_stray_quote", "DB_PASSWORD: x\\\"S0/creds:{a: 'x}\n" + secret + "'}\\\"", "DB_PASSWORD:[REDACTED]"},
		{"escaped_member_quote", "DB_PASSWORD: x\\\"S0/creds:{a: \\'x}\n" + secret + "\\'}\\\"", "DB_PASSWORD:[REDACTED]"},
		{"container_right_after_quote", "DB_PASSWORD: x\"{a: b'x}\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"unprefixed_container_right_after_quote", "DB_PASSWORD: x\"{a: 'x}\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"over_budget", "DB_PASSWORD: " + a140 + "\"S0/creds:{a: 'x}\n" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"before_fields", "DB_PASSWORD: x\"S0/creds:{a: 'x}\n" + secret + "'}\" other=ok", "DB_PASSWORD:[REDACTED] other=ok"},
		{"unclosed_stray_quote_multiline_value", "DB_PASSWORD: x\"S0/creds:{'a': \"1\n2\", 'b': '" + secret + "'}", "DB_PASSWORD:[REDACTED]"},
		{"unclosed_stray_quote_tabbed_value", "DB_PASSWORD: x\"S0/creds:{'a': \"1\t2\", 'b': '" + secret + "'}", "DB_PASSWORD:[REDACTED]"},
		{"unclosed_stray_quote_multiline_element", "DB_PASSWORD: x\"S0/creds:[1, \"1\n2\", '" + secret + "']", "DB_PASSWORD:[REDACTED]"},
		{"unclosed_stray_quote_before_fields", "DB_PASSWORD: x\"S0/creds:{'a': \"1\n2\", 'b': '" + secret + "'} other=ok", "DB_PASSWORD:[REDACTED] other=ok"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"apostrophe_in_path_container", `DB_PASSWORD: x"S0/creds:{a: it's}" other=ok`, `DB_PASSWORD:[REDACTED] it's}" other=ok`},
		{"unclosed_prefixed_string", "DB_PASSWORD: x\"S0/creds:{a: b'unclosed\nother=ok", "DB_PASSWORD:[REDACTED]"},
		{"unclosed_container", `DB_PASSWORD: x"{not closed" other=ok`, `DB_PASSWORD:[REDACTED] closed" other=ok`},
		{"no_member_end_after_string", "DB_PASSWORD: x\"S0/creds:{a: 'x}\ny' z}\" other=ok", "DB_PASSWORD:[REDACTED] 'x}\ny' z}\" other=ok"},
		{"text_after_container", "DB_PASSWORD: x\"S0/creds:{a: 'x}\ny'} text\" other=ok", "DB_PASSWORD:[REDACTED] 'x}\ny'} text\" other=ok"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedOnEveryEntryPoint(t, `x"S0/creds:{a: it's}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, "x\"S0/creds:{a: b'unclosed\nother=ok")
	assertUnchangedOnEveryEntryPoint(t, `x"{not closed" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, "note: x\"S0/creds:{a: 'x}\nv'}\" other=ok")

	// A member string may span lines and hold any byte, as the strings that
	// the container scans skip may.
	for _, tc := range []struct {
		s       string
		wantEnd int
	}{
		{"'x}\ny'}", 6},
		{"\"1\t2\",", 5},
		{"\\'x\v\\'}", 6},
	} {
		if end, _ := quotedMemberEnd(tc.s, 0); end != tc.wantEnd {
			t.Errorf("quotedMemberEnd(%q, 0) = %d, want %d", tc.s, end, tc.wantEnd)
		}
	}
	s := "\"S0/creds:{'a': \"1\n2\", 'b': 'v'}"
	qEnd := strings.Index(s, ": \"") + len(": \"")
	if inner, outer := quotedContainerOpeners(s, 0, qEnd, nil); inner != 10 || outer != 10 {
		t.Errorf("quotedContainerOpeners(%q) = (%d, %d), want (10, 10)", s[:qEnd], inner, outer)
	}
	s = "{a: 'x}\nv'}"
	if !scanMixedQuoteContainers(s, 0, len(s), '"', false, nil) {
		t.Errorf("scanMixedQuoteContainers(%q) = false, want true", s)
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated multiline members", func(n int) string {
			return strings.Repeat("DB_PASSWORD: x\"S0/creds:{a: 'x}\n"+secret+"'}\" ", n)
		}},
		{"repeated tabbed prefixed members", func(n int) string {
			return strings.Repeat("DB_PASSWORD: x\"S0/creds:{a: b'x}\t"+secret+"'}\" ", n)
		}},
		{"repeated vertical-tab prefixed members", func(n int) string {
			return strings.Repeat("DB_PASSWORD: x\"S0/creds:{a: b'x}\v"+secret+"'}\" ", n)
		}},
		{"repeated unclosed stray quotes", func(n int) string {
			return strings.Repeat("DB_PASSWORD: x\"S0/creds:{'a': \"1\n2\", 'b': '"+secret+"'}\n", n)
		}},
		{"long multiline member list", func(n int) string {
			return "DB_PASSWORD: x\"S0/creds:{" + strings.Repeat("a: 'x}\ny', ", n) + "k: 'x\n" + secret + "'}\""
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{a: 'x}\n"+secret+"'}\" ", secret)
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{a: b'x}\t"+secret+"'}\" ", secret)
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{a: b'x}\v"+secret+"'}\" ", secret)
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{'a': \"1\n2\", 'b': '"+secret+"'}\n", secret)
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{a: '\n")
	assertLargeTextLinear(t, "DB_PASSWORD: x\"S0/creds:{'a': \"1\n")
}

// TestPrefixedStringsInParenthesesAndConcatenations covers members of "{" or
// "[" containers in a sensitive value whose string has a Python string prefix
// and holds a bracket, inside parentheses or concatenated with another
// string, implicitly or with a "+", as the b'x}y' of DB_PASSWORD: "{'a':
// (b'x}y'), 'b': 'say " hi"', 'c': 'SECRET'}" or of DB_PASSWORD:
// x"S0/creds:{'a': b'x}y' + 'z', 'b': 'SECRET'}", and quoted values whose
// closing quote comes inside such a string, as the 'say " hi' of
// DB_PASSWORD: "{'a': ('say " hi'), 'k': 'SECRET'}". They come with several
// prefixes, in tuples and arrays, across tabs and line breaks, in single and
// escaped quotes, after a stray quote, past the nested-key budget and before
// other fields. Every entry point redacts the value together with the
// containers. Words with an apostrophe after a prefix letter, as the B's of
// Plan B's budget, also next to parentheses, a "+" or a quoted word, and other
// contractions and possessives keep the text after the value, text without a
// sensitive key stays unchanged, and repeated, long and unclosed such
// strings stay linear.
func TestPrefixedStringsInParenthesesAndConcatenations(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	tail := `, 'b': 'say " hi"', 'c': '` + secret + `'}"`
	singleTail := `, "b": "say ' hi'", "c": "` + secret + `"}'`
	for _, tc := range []struct {
		label, in, want string
	}{
		{"parenthesized", `DB_PASSWORD: "{'a': (b'x}y')` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"implicitly_concatenated", `DB_PASSWORD: "{'a': b'x}y' 'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"plus_concatenated", `DB_PASSWORD: "{'a': b'x}y' + 'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_strings_concatenated", `DB_PASSWORD: "{'a': b'x}y' b'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"two_letter_prefixes_concatenated", `DB_PASSWORD: "{'a': rb'x}y' rb'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"upper_case_prefix_plus", `DB_PASSWORD: "{'a': B'x}y' + 'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"format_prefix_plus_bytes", `DB_PASSWORD: "{'a': f'x}y' + b'z'` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"parenthesized_unicode_double_quoted", `DB_PASSWORD: '{"a": (u"x}y")` + singleTail, `DB_PASSWORD:'[REDACTED]'`},
		{"double_quoted_bytes_plus", `DB_PASSWORD: '{"a": b"x}y" + "z"` + singleTail, `DB_PASSWORD:'[REDACTED]'`},
		{"parenthesized_multiline", "DB_PASSWORD: \"{'a': (b'x}\ny')" + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"tabs_around_plus", "DB_PASSWORD: \"{'a': b'x}y'\t+\t'z'" + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"vertical_tab_before_adjacent_string", "DB_PASSWORD: \"{'a': b'x}y'\v'z'" + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"tuple_last_element", `DB_PASSWORD: "{'a': (1, b'x}y')` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"array_parenthesized", `DB_PASSWORD: "[(b'x]y'), 'say " hi', '` + secret + `']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"before_fields", `DB_PASSWORD: "{'a': (b'x}y')` + tail + ` other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"escaped_outer_quote", `DB_PASSWORD: \"{'a': (b'x}y'), 'b': 'say \" hi\"', 'c': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"single_quoted_outer_quote", `DB_PASSWORD: '{"a": (b"x}y")` + singleTail, `DB_PASSWORD:'[REDACTED]'`},
		{"over_budget", `DB_PASSWORD: "{` + a140 + `: 1, 'a': (b'x}y')` + tail, `DB_PASSWORD:"[REDACTED]"`},
		{"stray_quote_parenthesized", `DB_PASSWORD: x"S0/creds:{'a': (b'x}y'), 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_plus_concatenated", `DB_PASSWORD: x"S0/creds:{'a': b'x}y' + 'z', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_implicitly_concatenated", `DB_PASSWORD: x"S0/creds:{'a': b'x}y' 'z', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_prefixed_strings_concatenated", `DB_PASSWORD: x"S0/creds:{'a': b'x}y' b'z', 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_plus_after_line_break", "DB_PASSWORD: x\"S0/creds:{'a': b'x}y'\n+ 'z', 'b': '" + secret + "'}\"", "DB_PASSWORD:[REDACTED]"},
		{"stray_quote_array_parenthesized", `DB_PASSWORD: x"S0/creds:[(b'x]y'), '` + secret + `']"`, "DB_PASSWORD:[REDACTED]"},
		{"single_quoted_stray_quote_parenthesized_unicode", `DB_PASSWORD: x'S0/creds:{"a": (u"x}y"), "b": "` + secret + `"}'`, "DB_PASSWORD:[REDACTED]"},
		{"container_right_after_stray_quote", `DB_PASSWORD: x"{'a': (b'x}y'), 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"over_budget_stray_quote", "DB_PASSWORD: " + a140 + `"{'a': (b'x}y'), 'b': '` + secret + `'}"`, "DB_PASSWORD:[REDACTED]"},
		{"closing_quote_in_parenthesized_string", `DB_PASSWORD: "{'a': ('say " hi'), 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_parenthesized_bytes", `DB_PASSWORD: "{'a': (b'say " hi'), 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_tuple_element", `DB_PASSWORD: "{'a': (1, 'say " hi'), 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_plus_concatenated_string", `DB_PASSWORD: "{'a': 'say " hi' + 'z', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_implicitly_concatenated_string", `DB_PASSWORD: "{'a': 'say " hi' 'z', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_plus_concatenated_bytes", `DB_PASSWORD: "{'a': b'say " hi' + 'z', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct {
		label, in, want string
	}{
		{"possessive_after_prefix_letter", `DB_PASSWORD: x"S0/creds:{a: Plan B's budget}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's budget}" other=ok`},
		{"two_possessives_after_prefix_letter", `DB_PASSWORD: x"S0/creds:{a: Plan B's budget, b: Plan B's other}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's budget, b: Plan B's other}" other=ok`},
		{"possessives_around_parentheses", `DB_PASSWORD: x"S0/creds:{a: Plan B's (budget), b: Plan B's) cost}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's (budget), b: Plan B's) cost}" other=ok`},
		{"possessives_around_plus", `DB_PASSWORD: x"S0/creds:{a: Plan B's budget + Plan B's cost}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's budget + Plan B's cost}" other=ok`},
		{"possessive_before_quoted_word", `DB_PASSWORD: x"S0/creds:{a: Plan B's 'budget', b: Plan B's) cost}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's 'budget', b: Plan B's) cost}" other=ok`},
		{"contractions_and_possessives", `DB_PASSWORD: x"S0/creds:{a: it's (don't) + Bob's, b: herb'x}" other=ok`, `DB_PASSWORD:[REDACTED] it's (don't) + Bob's, b: herb'x}" other=ok`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}
	assertUnchangedOnEveryEntryPoint(t, `note: "{a: (Plan B's budget), b: Plan B's cost}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: "{'a': (b'x}y'), 'b': 'say " hi"'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: x"{'a': b'x}y' + 'z', 'b': 'v'}" other=ok`)

	// A prefixed string that a ")", a "+" or another string follows opens a
	// string, while an apostrophe after a prefix letter whose string closes
	// before a word does not.
	for _, tc := range []struct {
		s    string
		want bool
	}{
		{"{a: (b'x}y'), c: 1}", true},
		{"{a: b'x}y' 'z', c: 1}", true},
		{"{a: b'x}y' + 'z', c: 1}", true},
		{"{a: b'x}y' rb'z', c: 1}", true},
		{"{a: b'x}y'\v\"z\", c: 1}", true},
		{"{a: Plan B's budget, b: Plan B's cost}", false},
		{"{a: Plan B's budget}", false},
		{"{a: b'x}y' z, c: 1}", false},
		{"{a: b'x}y' bz'q', c: 1}", false},
	} {
		if got := isStringPrefixQuote(tc.s, 1, strings.IndexByte(tc.s, '\''), nil); got != tc.want {
			t.Errorf("isStringPrefixQuote(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated parenthesized prefixed members", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': (b'x}y')`+tail+` `, n)
		}},
		{"repeated plus-concatenated prefixed members", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': b'x}y' + 'z'`+tail+` `, n)
		}},
		{"repeated implicitly concatenated prefixed members", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': b'x}y' 'z'`+tail+` `, n)
		}},
		{"long parenthesized prefixed member list", func(n int) string {
			return `DB_PASSWORD: "{` + strings.Repeat(`'a': (b'x}y'), `, n) + `'b': 'say " hi"', 'c': '` + secret + `'}"`
		}},
		{"repeated stray-quote concatenations", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"S0/creds:{'a': b'x}y' + 'z', 'b': '`+secret+`'}" `, n)
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': (b'x}y')`+tail+` `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': b'x}y' + 'z'`+tail+` `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': b'x}y' 'z'`+tail+` `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': (b'x}y`)
	assertLargeTextLinear(t, `DB_PASSWORD: x"S0/creds:{a: b'x}y' + `)
}

// TestConcatenatedStringsInEnclosedContainers covers a closing quote that
// seems to end a quoted or stray-quoted sensitive value inside a string that
// is a later operand of a "+" or of an implicit concatenation in a container
// member, as the 'say " hi' of
// DB_PASSWORD: "{'a': 'x' + 'say " hi', 'k': 'SECRET'}" or of
// DB_PASSWORD: "{'a': 'x' 'say " hi', 'k': 'SECRET'}", or that opens such an
// operand after a string prefix, as the b" of
// DB_PASSWORD: "{'a': 'x' + b" SECRET", 'k': 'v'}": that operand adds a part
// to the member that the string before it starts (see opensMemberAfter and
// quotedContainerOpeners), so every entry point redacts the value together
// with the containers, in parentheses, after a line break or a tab, in single
// and escaped quotes, in arrays, before other fields and past the nested-key
// budget, and up to the end of the innermost container when text follows the
// containers in the string. Contractions and possessives around a "+" keep
// the text after the value, text without a sensitive key stays unchanged, and
// repeated and long concatenations stay linear.
func TestConcatenatedStringsInEnclosedContainers(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"plus_concatenated_second_operand", `DB_PASSWORD: "{'a': 'x' + 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"implicitly_concatenated_second_operand", `DB_PASSWORD: "{'a': 'x' 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"parenthesized_plus_concatenation", `DB_PASSWORD: "{'a': ('x' + 'say " hi'), 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_closing_quote_after_plus", `DB_PASSWORD: "{'a': 'x' + b" ` + secret + `", 'k': 'v'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_closing_quote_after_string", `DB_PASSWORD: "{'a': 'x' b" ` + secret + `", 'k': 'v'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"prefixed_closing_quote_after_number_plus", `DB_PASSWORD: "{'a': (1 + b" ` + secret + `"), 'k': 'v'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"stray_quote_plus_concatenated", `DB_PASSWORD: x"{'a': 'x' + 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"third_operand", `DB_PASSWORD: "{'a': 'x' + 'y' + 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"line_break_before_plus", "DB_PASSWORD: \"{'a': 'x'\n+ 'say \" hi', 'k': '" + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"tab_between_strings", "DB_PASSWORD: \"{'a': 'x'\t'say \" hi', 'k': '" + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"single_quoted_outer_quote", `DB_PASSWORD: '{"a": "x" + "say ' hi", "k": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"escaped_outer_quote", `DB_PASSWORD: \"{'a': 'x' + 'say \" hi', 'k': '` + secret + `'}\"`, `DB_PASSWORD:\"[REDACTED]\"`},
		{"array_element_after_plus", `DB_PASSWORD: "[1 + b" ` + secret + `", 'v']"`, `DB_PASSWORD:"[REDACTED]"`},
		{"stray_quote_prefixed_after_plus", `DB_PASSWORD: x"{'a': 'x' + b" ` + secret + `", 'k': 'v'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_prefixed_after_string", `DB_PASSWORD: x"{'a': 'x' b" ` + secret + `", 'k': 'v'}"`, `DB_PASSWORD:[REDACTED]`},
		{"before_fields", `DB_PASSWORD: "{'a': 'x' + 'say " hi', 'k': '` + secret + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"text_after_containers_after_plus", `DB_PASSWORD: "{'a': 'x' + b" ` + secret + `", 'k': 'v'} tail" other=ok`, `DB_PASSWORD:"[REDACTED]" tail" other=ok`},
		{"text_after_containers_after_string", `DB_PASSWORD: "{'a': 'x' b" ` + secret + `", 'k': 'v'} tail" other=ok`, `DB_PASSWORD:"[REDACTED]" tail" other=ok`},
		{"over_budget", `DB_PASSWORD: "{` + a140 + `: 1, 'a': 'x' + 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_stray_quote", `DB_PASSWORD: ` + a140 + `"{'a': 'x' + 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct{ in, want string }{
		{`DB_PASSWORD: x"S0/creds:{a: a's + b's}" other=ok`, `DB_PASSWORD:[REDACTED] a's + b's}" other=ok`},
		{`DB_PASSWORD: x"S0/creds:{a: it's + 'x', b: don't}" other=ok`, `DB_PASSWORD:[REDACTED] it's + 'x', b: don't}" other=ok`},
		{`DB_PASSWORD: "{a: 'x' + it's}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{`DB_PASSWORD: "{'a': 'x' " other="ok"`, `DB_PASSWORD:"[REDACTED]" other="ok"`},
		{`DB_PASSWORD: "{'a': 'x' + ", b: "y"} other=ok`, `DB_PASSWORD:"[REDACTED]", b: "y"} other=ok`},
	} {
		assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
	}
	assertUnchangedOnEveryEntryPoint(t, `note: "{'a': 'x' + 'say " hi', 'k': 'v'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: "{'a': 'x' b" y", 'k': 'v'}" other=ok`)

	// A "+" or a string that closes starts a part of a member, while a quote
	// joined to a word, which opens no string, does not.
	for _, tc := range []struct {
		stack string
		prev  byte
		want  bool
	}{
		{"{", '+', true},
		{"[", '+', true},
		{"{", closedString, true},
		{"[", closedString, true},
		{"", '+', false},
		{"{", '\'', false},
		{"{", '"', false},
	} {
		if got := opensMemberAfter([]byte(tc.stack), tc.prev); got != tc.want {
			t.Errorf("opensMemberAfter(%q, %q) = %v, want %v", tc.stack, tc.prev, got, tc.want)
		}
	}
	for _, tc := range []struct {
		s          string
		outer      int
		wantMember bool
	}{
		{`"{'a': 'x' + 'say " hi', 'k': 'v'}"`, 1, true},
		{`"{'a': 'x' 'say " hi', 'k': 'v'}"`, 1, true},
		{`"{'a': 'x' + b" y", 'k': 'v'}"`, 1, true},
		{`"{'a': it's 'say " hi'}"`, -1, false},
	} {
		qEnd := strings.Index(tc.s[1:], `"`) + 2
		if outer, ok := closingQuoteInMember(tc.s, 0, qEnd, nil); outer != tc.outer || ok != tc.wantMember {
			t.Errorf("closingQuoteInMember(%q) = (%d, %v), want (%d, %v)", tc.s[:qEnd], outer, ok, tc.outer, tc.wantMember)
		}
	}
	for _, tc := range []struct {
		s     string
		inner int
	}{
		{`"{'a': (1 + b" y"), 'k': 'v'}"`, 1},
		{`"{'a': 'x' b" y", 'k': 'v'}"`, 1},
		{`"[1, {'a': 'x' + b" y", 'k': 'v'}]"`, 5},
		{`"{'a': it' " y", 'k': 'v'}"`, -1},
		{`"{'a': 'x' + b" y" z, 'k': 'v'}"`, -1},
	} {
		qEnd := strings.Index(tc.s[1:], `"`) + 2
		if inner, _ := quotedContainerOpeners(tc.s, 0, qEnd, nil); inner != tc.inner {
			t.Errorf("quotedContainerOpeners(%q) = %d, want %d", tc.s[:qEnd], inner, tc.inner)
		}
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated plus-concatenated closing quotes", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': 'x' + 'say " hi', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated prefixed closing quotes after strings", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': 'x' b" `+secret+`", 'k': 'v'}" `, n)
		}},
		{"repeated stray-quote concatenations", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"{'a': 'x' + 'say " hi', 'k': '`+secret+`'}" `, n)
		}},
		{"long concatenation", func(n int) string {
			return `DB_PASSWORD: "{'a': ` + strings.Repeat(`'x' + `, n) + `'say " hi', 'k': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': 'x' + 'say " hi', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': 'x' b" `+secret+`", 'k': 'v'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': 'x' + 'say " hi', `)
}

// TestContainersAfterTextGroupsAndBackslashes covers containers that open
// after other text in a quoted or stray-quoted sensitive value, in groups
// whose ")" come before the closing quote, or after a backslash, as the { of
// DB_PASSWORD: "prefix {'a': 'say " hi', 'k': 'SECRET'}", of
// DB_PASSWORD: "({'a': 'say " hi', 'k': 'SECRET'})", of
// DB_PASSWORD: "\{'a': 'say " hi', 'k': 'SECRET'}" or of
// DB_PASSWORD: x"prefix {'a': '1', 'k': 'SECRET'}": the containers start at
// the outermost one open at the closing quote that seems to end the value,
// wherever it opens (see closingQuoteInMember), or at the first one in a
// stray-quoted string that closes right after them (see
// enclosesMixedQuoteContainer), so every entry point redacts the value
// together with them, also after a tuple member, in single quotes, before
// other fields and past the nested-key budget, while a closing quote glued to
// a word in an unquoted member still has them read from the opening quote on.
// A container that stays open, one without a quoted member, text between
// containers and prose with an apostrophe keep the text after the value, text
// without a sensitive key stays unchanged, and repeated and long such values
// stay linear.
func TestContainersAfterTextGroupsAndBackslashes(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"parenthesized_prefix_after_text", `DB_PASSWORD: "x {a: (b" ` + secret + `"), m: 'v'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"text_before_container", `DB_PASSWORD: "prefix {'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"container_in_group", `DB_PASSWORD: "({'a': 'say " hi', 'k': '` + secret + `'})"`, `DB_PASSWORD:"[REDACTED]"`},
		{"escaped_backslash_before_container", `DB_PASSWORD: "\\{'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"backslash_before_container", `DB_PASSWORD: "\{'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"stray_quote_text_before_container", `DB_PASSWORD: x"prefix {'a': '1', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"words_before_container", `DB_PASSWORD: "prefix more words {'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"nested_groups", `DB_PASSWORD: "(({'a': 'say " hi', 'k': '` + secret + `'}))"`, `DB_PASSWORD:"[REDACTED]"`},
		{"spaced_group", `DB_PASSWORD: "( {'a': 'say " hi', 'k': '` + secret + `'} )"`, `DB_PASSWORD:"[REDACTED]"`},
		{"tuple_before_container", `DB_PASSWORD: "(1, {'a': 'say " hi', 'k': '` + secret + `'})"`, `DB_PASSWORD:"[REDACTED]"`},
		{"unbalanced_group_closer", `DB_PASSWORD: "{'a': 'say " hi', 'k': '` + secret + `'})"`, `DB_PASSWORD:"[REDACTED]"`},
		{"closing_quote_in_parenthesized_member_after_text", `DB_PASSWORD: "x {a: ('say " hi'), m: '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"text_after_parenthesized_prefix", `DB_PASSWORD: "x {a: (b" ` + secret + `"), m: 'v'} tail" other=ok`, `DB_PASSWORD:"[REDACTED]" tail" other=ok`},
		{"stray_quote_text_before_inner_quote", `DB_PASSWORD: x"prefix {'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_group", `DB_PASSWORD: x"({'a': 'say " hi', 'k': '` + secret + `'})"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_group_without_inner_quote", `DB_PASSWORD: x"({'a': '1', 'k': '` + secret + `'})"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_words_before_container", `DB_PASSWORD: x"prefix more words {'a': '1', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"stray_quote_parenthesized_prefix", `DB_PASSWORD: x"x {a: (b" ` + secret + `"), m: 'v'}" other=ok`, `DB_PASSWORD:[REDACTED] other=ok`},
		{"stray_quote_backslash", `DB_PASSWORD: x"\{'a': '1', 'k': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"single_quoted_text_before_container", `DB_PASSWORD: 'prefix {"a": "say ' hi", "k": "` + secret + `"}'`, `DB_PASSWORD:'[REDACTED]'`},
		{"before_fields", `DB_PASSWORD: "prefix {'a': 'say " hi', 'k': '` + secret + `'}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{"stray_quote_before_fields", `DB_PASSWORD: x"prefix {'a': '1', 'k': '` + secret + `'}" other=ok`, `DB_PASSWORD:[REDACTED] other=ok`},
		{"over_budget_text_before_container", `DB_PASSWORD: "` + a140 + ` {'a': 'say " hi', 'k': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"over_budget_stray_quote_text_before_container", `DB_PASSWORD: ` + a140 + `"prefix {'a': '1', 'k': '` + secret + `'}" other=ok`, `DB_PASSWORD:[REDACTED] other=ok`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct{ in, want string }{
		{`DB_PASSWORD: "prefix it's fine" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{`DB_PASSWORD: x"prefix {not closed" other=ok`, `DB_PASSWORD:[REDACTED] {not closed" other=ok`},
		{`DB_PASSWORD: x"prefix" other=ok`, `DB_PASSWORD:[REDACTED] other=ok`},
		{`DB_PASSWORD: "prefix {a: 1}" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{`DB_PASSWORD: "({a: 1})" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{`DB_PASSWORD: x"note {a: 1} and {b: 2}" other=ok`, `DB_PASSWORD:[REDACTED] {a: 1} and {b: 2}" other=ok`},
		{`DB_PASSWORD: x"prefix {a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] {a: 1}" other=ok`},
		{`DB_PASSWORD: x"Bob's note {a: 1}" other=ok`, `DB_PASSWORD:[REDACTED] note {a: 1}" other=ok`},
		{`DB_PASSWORD: "(x)" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
		{`DB_PASSWORD: x"(note) more" other=ok`, `DB_PASSWORD:[REDACTED] more" other=ok`},
		{`DB_PASSWORD: "\{abc" other=ok`, `DB_PASSWORD:"[REDACTED]" other=ok`},
	} {
		assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
	}
	assertUnchangedOnEveryEntryPoint(t, `note: "prefix {'a': 'say " hi', 'k': 'v'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: x"prefix {'a': '1', 'k': 'v'}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: "({'a': 'say " hi', 'k': 'v'})" other=ok`)

	// A closing quote glued to the word after it in an unquoted member, which
	// opens no member, still has the containers read from the opening quote
	// on.
	for _, in := range []string{
		`DB_PASSWORD: "{msg: say "hi", 'b': '` + secret + `'}"`,
		`DB_PASSWORD: "[say "hi", '` + secret + `']"`,
		`DB_PASSWORD: "{a: {b: say "hi"}, 'c': '` + secret + `'}"`,
	} {
		assertRedactedOnEveryEntryPoint(t, in, `DB_PASSWORD:"[REDACTED]"`, secret)
	}

	// The containers start at the outermost one open at the closing quote,
	// after any text and after a backslash that escapes no quote, and ")"
	// may close groups around them before the quote.
	for _, tc := range []struct {
		s     string
		outer int
	}{
		{`"prefix {'a': 'say " hi'}"`, 8},
		{`"({'a': 'say " hi'})"`, 2},
		{`"\{'a': 'say " hi'}"`, 2},
		{`"\\{'a': 'say " hi'}"`, 3},
		{`"x {a: (b" y"), m: 'v'}"`, 3},
		{`"prefix {a: it's " hi'}"`, -1},
	} {
		qEnd := strings.Index(tc.s[1:], `"`) + 2
		if outer, _ := closingQuoteInMember(tc.s, 0, qEnd, nil); outer != tc.outer {
			t.Errorf("closingQuoteInMember(%q) = %d, want %d", tc.s[:qEnd], outer, tc.outer)
		}
	}
	for _, tc := range []struct {
		s    string
		i    int
		want int
	}{
		{"}) \" ", 1, 3},
		{"} ) )\"", 1, 5},
		{"}\"", 1, 1},
		{"} x\"", 1, 2},
	} {
		if got := skipGroupClosers(tc.s, tc.i); got != tc.want {
			t.Errorf("skipGroupClosers(%q, %d) = %d, want %d", tc.s, tc.i, got, tc.want)
		}
	}
	for _, tc := range []struct {
		s    string
		want bool
	}{
		{`x"prefix {'k': 'v'}"`, true},
		{`x"({'k': 'v'})"`, true},
		{`x"prefix {k: 1}"`, false},
		{`x"(note) more"`, false},
	} {
		if got := enclosesMixedQuoteContainer(tc.s, 1, len(tc.s), strings.IndexAny(tc.s, " {("), nil); got != tc.want {
			t.Errorf("enclosesMixedQuoteContainer(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated containers after text", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "prefix {'a': 'say " hi', 'k': '`+secret+`'}" `, n)
		}},
		{"repeated groups", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "({'a': 'say " hi', 'k': '`+secret+`'})" `, n)
		}},
		{"repeated stray quotes before text and containers", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"prefix {'a': '1', 'k': '`+secret+`'}" `, n)
		}},
		{"long text before a container", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat("prefix ", n) + `{'a': 'say " hi', 'k': '` + secret + `'}"`
		}},
		{"many groups", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat("(", n) + `{'a': 'say " hi', 'k': '` + secret + `'}` + strings.Repeat(")", n) + `"`
		}},
		{"many backslashes before a container", func(n int) string {
			return `DB_PASSWORD: "` + strings.Repeat(`\\`, n) + `{'a': 'say " hi', 'k': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: "prefix {'a': 'say " hi', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "({'a': 'say " hi', 'k': '`+secret+`'})" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: x"prefix {'a': '1', 'k': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "prefix {'a': 'say " hi', `)
	assertLargeTextLinear(t, `DB_PASSWORD: x"prefix {'a': '1', `)
}

// TestPrefixedStringsWithSubscriptsAndAttributes covers a prefixed string in a
// container member that a subscript or an attribute follows right after its
// closing quote, as the b'x}SECRET'.decode() of
// DB_PASSWORD: x"S0/creds:{'a': b'x}SECRET'.decode(), 'b': '1'}" or the
// b'x}y'[0] of DB_PASSWORD: "{'a': b'x}y'[0], 'b': 'say " hi', 'c': 'SECRET'}":
// that string ends a part of the member (see memberStringEnds), so the
// container scans read it as one and its "}" closes nothing, and every entry
// point redacts the value together with the containers, with slices and
// method calls, right after the stray quote, in arrays, in double quotes,
// before other fields and past the nested-key budget. A possessive or a
// contraction before a "." or a "[", which opens no string, keeps the text
// after the value, text without a sensitive key stays unchanged, and
// repeated such members stay linear.
func TestPrefixedStringsWithSubscriptsAndAttributes(t *testing.T) {
	runInParallel(t)
	const secret = "SECRET_PROBE_98765"
	a140 := strings.Repeat("a", 140)
	for _, tc := range []struct {
		label, in, want string
	}{
		{"attribute_after_prefixed_string_holding_secret", `DB_PASSWORD: x"S0/creds:{'a': b'x}` + secret + `'.decode(), 'b': '1'}"`, `DB_PASSWORD:[REDACTED]`},
		{"subscript_after_prefixed_string_holding_secret", `DB_PASSWORD: x"S0/creds:{'a': b'x}` + secret + `'[0], 'b': '1'}"`, `DB_PASSWORD:[REDACTED]`},
		{"attribute_before_later_member", `DB_PASSWORD: x"S0/creds:{'a': b'x}y'.decode(), 'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"attribute_before_closing_quote_member", `DB_PASSWORD: "{'a': b'x}y'.decode(), 'b': 'say " hi', 'c': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"subscript_before_closing_quote_member", `DB_PASSWORD: "{'a': b'x}y'[0], 'b': 'say " hi', 'c': '` + secret + `'}"`, `DB_PASSWORD:"[REDACTED]"`},
		{"format_prefix_method", `DB_PASSWORD: x"S0/creds:{'a': f'x}y'.strip(), 'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"slice_after_prefixed_string", `DB_PASSWORD: x"S0/creds:{'a': b'x}y'[0:2], 'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"right_after_stray_quote", `DB_PASSWORD: x"{'a': b'x}y'.decode(), 'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
		{"array_attribute", `DB_PASSWORD: x"S0/creds:[b'x]y'.decode(), '` + secret + `']"`, `DB_PASSWORD:[REDACTED]`},
		{"unicode_prefix_double_quoted", `DB_PASSWORD: x'S0/creds:{"a": u"x}y".upper(), "b": "` + secret + `"}'`, `DB_PASSWORD:[REDACTED]`},
		{"before_fields", `DB_PASSWORD: x"S0/creds:{'a': b'x}y'.decode(), 'b': '` + secret + `'}" other=ok`, `DB_PASSWORD:[REDACTED] other=ok`},
		{"over_budget", `DB_PASSWORD: ` + a140 + `"S0/creds:{'a': b'x}y'.decode(), 'b': '` + secret + `'}"`, `DB_PASSWORD:[REDACTED]`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
		})
	}

	for _, tc := range []struct{ in, want string }{
		{`DB_PASSWORD: x"S0/creds:{a: Plan B's.txt, b: Plan B's [note]}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's.txt, b: Plan B's [note]}" other=ok`},
		{`DB_PASSWORD: x"S0/creds:{a: it's.}" other=ok`, `DB_PASSWORD:[REDACTED] it's.}" other=ok`},
		{`DB_PASSWORD: x"S0/creds:{a: Plan B's budget. Plan B's cost}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's budget. Plan B's cost}" other=ok`},
		{`DB_PASSWORD: x"S0/creds:{a: Plan B's x, b: it'. done}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's x, b: it'. done}" other=ok`},
		{`DB_PASSWORD: x"S0/creds:{a: Plan B's [x] and B'[y]}" other=ok`, `DB_PASSWORD:[REDACTED] Plan B's [x] and B'[y]}" other=ok`},
	} {
		assertRedactedOnEveryEntryPoint(t, tc.in, tc.want, secret)
	}
	assertUnchangedOnEveryEntryPoint(t, `note: "{'a': f'x}y'.strip(), 'b': b'x}y'[0:2]}" other=ok`)
	assertUnchangedOnEveryEntryPoint(t, `note: x"S0/creds:{'a': b'x}y'.decode(), 'b': 'v'}" other=ok`)

	// A subscript or an attribute right after the closing quote ends a part
	// of a member, while one after a space or a "." that no word follows
	// does not.
	for _, tc := range []struct {
		s    string
		want bool
	}{
		{"{a: b'x}y'.decode(), c: 1}", true},
		{"{a: b'x}y'[0], c: 1}", true},
		{"{a: f'x}y'.strip(), c: 1}", true},
		{"{a: b'x}y'[0:2], c: 1}", true},
		{"{a: b'x}y' .decode(), c: 1}", false},
		{"{a: b'x}y'. c: 1}", false},
		{"{a: Plan B's.txt, b: Plan B's [note]}", false},
		{"{a: Plan B's x, b: it'. done}", false},
	} {
		if got := isStringPrefixQuote(tc.s, 1, strings.IndexByte(tc.s, '\''), nil); got != tc.want {
			t.Errorf("isStringPrefixQuote(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}

	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"repeated attributes after prefixed strings", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: x"S0/creds:{'a': b'x}y'.decode(), 'b': '`+secret+`'}" `, n)
		}},
		{"repeated subscripts before closing quotes", func(n int) string {
			return strings.Repeat(`DB_PASSWORD: "{'a': b'x}y'[0], 'b': 'say " hi', 'c': '`+secret+`'}" `, n)
		}},
		{"long member list with attributes", func(n int) string {
			return `DB_PASSWORD: x"S0/creds:{` + strings.Repeat(`'a': b'x}y'.decode(), `, n) + `'b': '` + secret + `'}"`
		}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			assertLinearAndRedacted(t, shape.mk, secret)
		})
	}
	assertLargeTextLinear(t, `DB_PASSWORD: x"S0/creds:{'a': b'x}y'.decode(), 'b': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: "{'a': b'x}y'[0], 'b': 'say " hi', 'c': '`+secret+`'}" `, secret)
	assertLargeTextLinear(t, `DB_PASSWORD: x"S0/creds:{a: b'x}y'.`)
}

// assertLinearAndRedacted checks that every entry point (see
// largeTextEntryPoints) keeps none of forbidden in the large text of mk (see
// scalingUnits), and leaves its own output unchanged on a second pass, that
// sanitizeRawText reports the change, and that redactErrorMessage,
// sanitizeRawText and SmartTruncate take at most 64 times as long on that
// text as on the small one, 16 times shorter.
func assertLinearAndRedacted(t *testing.T, mk func(n int) string, forbidden ...string) {
	t.Helper()
	smallN, largeN := scalingUnits()
	small, large := mk(smallN), mk(largeN)
	for _, ep := range largeTextEntryPoints(t) {
		got := ep.f(large)
		for _, f := range forbidden {
			if strings.Contains(got, f) {
				t.Errorf("%s(%d bytes of %.40q) kept %q", ep.name, len(large), large, f)
			}
		}
		if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%d bytes of %.40q) = %.80q, want unchanged", ep.name, len(large), large, again)
		}
	}
	if _, lost := sanitizeRawText(large); !lost {
		t.Errorf("sanitizeRawText(%d bytes of %.40q) reported no change", len(large), large)
	}
	scrub := func(s string) {
		_ = redactErrorMessage(s)
		_, _ = sanitizeRawText(s)
		_, _, _ = SmartTruncate(s, -1)
	}
	scrub(small)
	scrub(large)
	if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
		t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
	}
}

// assertUnchangedOnEveryEntryPoint checks that every entry point (see
// largeTextEntryPoints) leaves in byte-identical and that sanitizeRawText
// reports no change.
func assertUnchangedOnEveryEntryPoint(t *testing.T, in string) {
	t.Helper()
	for _, ep := range largeTextEntryPoints(t) {
		if got := ep.f(in); got != in {
			t.Errorf("%s(%q) = %q, want unchanged", ep.name, in, got)
		}
	}
	if _, lost := sanitizeRawText(in); lost {
		t.Errorf("sanitizeRawText(%q) reported a change", in)
	}
}

// assertRedactedOnEveryEntryPoint checks that every entry point (see
// largeTextEntryPoints) turns in into want, keeps none of forbidden, and leaves
// its own output unchanged on a second pass, and that sanitizeRawText reports
// the change.
func assertRedactedOnEveryEntryPoint(t *testing.T, in, want string, forbidden ...string) {
	t.Helper()
	for _, ep := range largeTextEntryPoints(t) {
		got := ep.f(in)
		kept := false
		for _, f := range forbidden {
			if strings.Contains(got, f) {
				t.Errorf("%s(%q) = %q kept %q", ep.name, in, got, f)
				kept = true
			}
		}
		if !kept && got != want {
			t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
		}
		if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
		}
	}
	if _, lost := sanitizeRawText(in); !lost {
		t.Errorf("sanitizeRawText(%q) reported no change", in)
	}
}

// assertLargeTextLinear checks that a large text of repeated unit (see
// largeTextLen) runs through every entry point within a wall-clock limit, keeps none of forbidden, and is
// unchanged by a second pass. A text that the scans redact rather than fail
// closed does the work of every pass, so the limit is wider than that of
// assertLargeTextFailsClosed, while a scan that is quadratic in such a text
// takes minutes.
func assertLargeTextLinear(t *testing.T, unit string, forbidden ...string) {
	t.Helper()
	in := strings.Repeat(unit, (largeTextLen()+len(unit)-1)/len(unit))
	maxWall := 10 * time.Second
	if raceEnabled {
		maxWall = 60 * time.Second
	}
	start := time.Now()
	for _, ep := range largeTextEntryPoints(t) {
		got := ep.f(in)
		for _, f := range forbidden {
			if strings.Contains(got, f) {
				t.Errorf("%s(%d bytes of %.40q) kept %q", ep.name, len(in), unit, f)
			}
		}
		if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%d bytes of %.40q) = %.80q, want unchanged", ep.name, len(in), unit, again)
		}
	}
	if elapsed := time.Since(start); elapsed > maxWall {
		t.Errorf("%d bytes of %.40q took %v through the entry points, want at most %v", len(in), unit, elapsed, maxWall)
	}
}

// assertLargeTextFailsClosed checks that every entry point returns
// redactedSensitiveTextPlaceholder for in within the wall-clock limit of the
// large-text tests, and that a second pass changes nothing.
func assertLargeTextFailsClosed(t *testing.T, in string) {
	t.Helper()
	maxWall := 2 * time.Second
	if raceEnabled {
		maxWall = 20 * time.Second
	}
	start := time.Now()
	for _, ep := range largeTextEntryPoints(t) {
		if got := ep.f(in); got != redactedSensitiveTextPlaceholder {
			t.Errorf("%s(%d bytes) = %.80q, want %q", ep.name, len(in), got, redactedSensitiveTextPlaceholder)
		} else if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
		}
	}
	if elapsed := time.Since(start); elapsed > maxWall {
		t.Errorf("%d bytes took %v through the entry points, want at most %v", len(in), elapsed, maxWall)
	}
}

// largeTextEntryPoints returns the entry points that the large-text checks run
// a text through, each returning the redacted text.
func largeTextEntryPoints(t *testing.T) []struct {
	name string
	f    func(string) string
} {
	return []struct {
		name string
		f    func(string) string
	}{
		{"scrubErrorText", scrubErrorText},
		{"redactErrorMessage", redactErrorMessage},
		{"sanitizeRawText", func(s string) string { out, _ := sanitizeRawText(s); return out }},
		{"SmartTruncate", func(s string) string {
			raw, _, err := SmartTruncate(s, -1)
			var out string
			if err != nil || json.Unmarshal(raw, &out) != nil {
				t.Fatalf("SmartTruncate(%d bytes) = (%.80s, %v), want a JSON string", len(s), raw, err)
			}
			return out
		}},
		{"FormatContentParts", func(s string) string { return formatContentTextPart(t, s) }},
	}
}
