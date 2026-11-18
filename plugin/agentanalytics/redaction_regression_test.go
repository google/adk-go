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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
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
		{"container_tail_glued_nested_quote", `log: "api_key"={"k":"` + s1 + `"}/foo'x/temp:a="a token: "x ` + s2 + `""t`, redactedSensitiveTextPlaceholder, []string{s1, s2}},
		{"container_tail_glued_nested_quote_kelvin", `log: \"api_` + "\u212a" + `ey\"={"k":"` + s1 + `"}/foo'x/temp:a="a token: "x ` + s2 + `""t`, redactedSensitiveTextPlaceholder, []string{s1, s2}},
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

	// Redacted text that does not start with a quote, structured text and
	// prose stay byte-identical.
	for _, in := range []string{
		"api_key: [REDACTED] \"token\":\n  \"[REDACTED]\" other=ok",
		"api_key: [REDACTED] token:\n  [REDACTED]",
		"credentials:\n  dbPassword: " + redactedURIPlaceholder,
		"credentials:\n  spring.datasource.password: [REDACTED]\n  X-Amz-Security-Token: [REDACTED]",
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

	// Text that starts with a quote and holds a quoted sensitive key fails
	// closed, as on main, also when the value of that key reads as redacted.
	for _, in := range []string{
		`"X-Amz-Security-Token": '[REDACTED]' other=ok`,
		`"token": "[REDACTED]"; other="ok"`,
	} {
		assertAllEntryPoints(t, in, in, unparseableJSONPlaceholder)
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

// TestBackstopExemptionsStayNarrow checks that wholePlaceholderBytes marks
// only a placeholder that stands as a whole value, not one glued to other
// bytes, and that redactedKeyBytes marks only the key of a redacted pair, its
// quotes and its delimiter: not the words before a path-joined key, on a line
// of its own too, nor a quoted span across other words, and also the key of a
// redacted URI.
func TestBackstopExemptionsStayNarrow(t *testing.T) {
	runInParallel(t)
	markedText := func(s string, marks []bool) string {
		var b strings.Builder
		for i, m := range marks {
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
		{"x=[REDACTED];token=SECRET", "[REDACTED]"},
		{`token: "[REDACTED]", x`, `"[REDACTED]"`},
		{"password: [REDACTED]/aaa/token:[REDACTED]", "[REDACTED]"},
		{"password: [REDACTED]hunter2/token: [REDACTED]", "[REDACTED]"},
		{"[REDACTED]x", ""},
		{"x[REDACTED] y", ""},
		{"x/[REDACTED] y", ""},
		{"a [REDACTED_SENSITIVE_URI]. b", "[REDACTED_SENSITIVE_URI]"},
		{"token:'[REDACTED]\"", ""},
	} {
		if got := markedText(tc.in, wholePlaceholderBytes(tc.in)); got != tc.want {
			t.Errorf("wholePlaceholderBytes(%q) marks %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct {
		in, want string
	}{
		{"SECRET/token:[REDACTED]", "token:"},
		{`x "token":[REDACTED]`, `"token":`},
		{`: "password:[REDACTED] SECRET /token"=[REDACTED]`, `"password:`},
		{"token=SECRET\vpassword:[REDACTED]", "password:"},
		{"credentials:\n  C:\\ssl\\password:[REDACTED]", "password:"},
		{"dbPassword: " + redactedURIPlaceholder, "dbPassword:"},
		{`api_key = "[REDACTED]"`, `api_key = "`},
	} {
		if got := markedText(tc.in, redactedKeyBytes(tc.in)); got != tc.want {
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
		if got, want := keptWords(legacyScrubTextPatterns(in, false)), keptWords(refScrub(in)); !slices.Equal(got, want) {
			t.Errorf("legacyScrubTextPatterns(%q) keeps %q, reference expressions keep %q", in, got, want)
		}
	}
	if !testLeavesLegacyValue("pa\u017f\u017fword=abc x", "pa\u017f\u017fword=abc x") {
		t.Errorf("leavesLegacyValue keeps a long-s password value")
	}
}

// keptWords returns the words of s, runs of ASCII letters and digits or of
// non-ASCII bytes, outside redaction placeholders, to compare what two
// redactions of a text keep whatever spaces and quotes each writes around a
// placeholder.
func keptWords(s string) []string {
	var words []string
	start := -1
	for i := 0; i <= len(s); i++ {
		if i < len(s) {
			if n := placeholderLenAt(s, i); n > 0 {
				if start >= 0 {
					words = append(words, s[start:i])
					start = -1
				}
				i += n - 1
				continue
			}
			if c := s[i]; ('a' <= c|0x20 && c|0x20 <= 'z') || ('0' <= c && c <= '9') || c >= 0x80 {
				if start < 0 {
					start = i
				}
				continue
			}
		}
		if start >= 0 {
			words = append(words, s[start:i])
			start = -1
		}
	}
	return words
}

// testLeavesLegacyValue is leavesLegacyValue for scrubbed, text that a scrub
// of raw may return, whose bytes copied from raw alignedKeptRuns finds.
func testLeavesLegacyValue(raw, scrubbed string) bool {
	var m scrubMemo
	return m.leavesLegacyValue(raw, scrubbed, alignedKeptRuns(raw, scrubbed))
}

// alignedKeptRuns returns the runs of scrubbed copied from raw for a test,
// aligning the texts as a part of a rewritten URI is aligned, byte by byte
// as they are (see uriAligner.alignText).
func alignedKeptRuns(raw, scrubbed string) []copyRun {
	var tr passTrace
	newURIAligner(&tr, 0, 0, raw, scrubbed).alignText(0, len(raw), 0, len(scrubbed), false)
	return tr.runs
}

// TestSmartTruncateSecondCallOnRedactedText checks what SmartTruncate returns
// for its own result. Within a call it runs its pass until the text stops
// changing, and a later pass keeps what an earlier one left malformed (see
// redactJSONTextPass). A second call reads that result as any other value:
// plain text and JSON documents come back unchanged, and a value that starts
// with "{" or "[" and is no JSON fails closed, as on main, since nothing
// tells it from text that came with its placeholders.
func TestSmartTruncateSecondCallOnRedactedText(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct{ in, want, wantSecond string }{
		{"password=SECRET other=ok", "password=[REDACTED] other=ok", "password=[REDACTED] other=ok"},
		{`{"token":"SECRET","n":1}`, `{"n":1,"token":"[REDACTED]"}`, `{"n":1,"token":"[REDACTED]"}`},
		{"https://user:SECRET@example.com/a hello", redactedURIPlaceholder + " hello", unparseableJSONPlaceholder},
		{"https://example.com/cb?secret_key=SECRET&ok=1\vcredentials\\\" : name/token:\n  SECRET", redactedURIPlaceholder + `" : name/token:[REDACTED]`, unparseableJSONPlaceholder},
		{`{"msg":"password=SECRET"}`, `{"msg":"password=[REDACTED]}`, unparseableJSONPlaceholder},
		{"https://example.com/token/SECRET\vapp:dbPassword\\\":\n  \" password \": SECRET dbPassword\"\v:'SECRET'", unparseableJSONPlaceholder, unparseableJSONPlaceholder},
		{"https://example.com/token/SECRET\vhttps://example.com/cb?x-goog-api-key=SECRET&ok=1\n  \\'DB_PASSWORD\\'\u00a0:\"a token: \"x SECRET\"", unparseableJSONPlaceholder, unparseableJSONPlaceholder},
		{`[{"token": "SECRET"}] trailing`, unparseableJSONPlaceholder, unparseableJSONPlaceholder},
	} {
		first, _, err := SmartTruncate(tc.in, -1)
		var out string
		if err != nil || json.Unmarshal(first, &out) != nil {
			t.Fatalf("SmartTruncate(%q) = (%s, %v)", tc.in, first, err)
		}
		if out != tc.want {
			t.Errorf("SmartTruncate(%q) = %q, want %q", tc.in, out, tc.want)
		}
		second, truncated, err := SmartTruncate(out, -1)
		var again string
		if err != nil || json.Unmarshal(second, &again) != nil {
			t.Fatalf("SmartTruncate second call(%q) = (%s, %v)", out, second, err)
		}
		if again != tc.wantSecond || truncated != (again != out) {
			t.Errorf("SmartTruncate second call(%q) = (%q, %v), want (%q, %v)", out, again, truncated, tc.wantSecond, tc.wantSecond != out)
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
		{"over_budget_dangling_quote_object", "DB_PASSWORD: " + a135 + `=S0/credentials":{"value": "` + s1 + `"}`, redactedSensitiveTextPlaceholder, []string{s1}},
		{"over_budget_dangling_quote_array", "dbPassword=" + a135 + `=S0/password":["a", "` + s1 + `"] other=ok`, redactedSensitiveTextPlaceholder, []string{s1}},
		{"over_budget_closed_container_then_dangling_quote_array", "dbPassword=" + a135 + `=S0/password":["a"]/token":["` + s1 + `" ] other=ok`, "dbPassword=[REDACTED] ] other=ok", []string{s1}},
		// The quotes inside a container that closes inside the value are
		// not read again, so they do not pair with the quotes after it.

		// Against the scrub with every key that isSensitiveKey accepts, a
		// compound name within a line is a key as a whole, so the
		// dbPassword after a non-ASCII space does not fail the text closed.
		{"compound_key_after_non_ascii_space", "USER:TOKEN:\n  \"OPENAI_API_KEY\"=[REDACTED]\u00a0dbPassword: \"" + s1 + "\"", "USER:TOKEN:\n  \"OPENAI_API_KEY\"=[REDACTED]", []string{s1}},
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

	// Redacted text that does not start with a quote stays byte-identical.
	for _, in := range []string{
		"credentials: user:private.key:[REDACTED]",
		"Authorization: [REDACTED] token:\n  [REDACTED]",
		"api_key: [REDACTED] Token:\n  [REDACTED]",
		"api_key: [REDACTED] x-api-key:\n  [REDACTED] other=ok",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}

	// Redacted text that starts with a quote is read as any other text that
	// does. With a quoted sensitive key in it, it fails closed, as on main,
	// whatever the value of that key reads as.
	for _, in := range []string{
		`"token": "[REDACTED]"; other="ok"`,
		`"api.key":"[REDACTED]" other=ok`,
		`"token": "[REDACTED]", see logs`,
		`"X-Amz-Security-Token"=[REDACTED] other=ok`,
		`"X-Amz-Security-Token": '[REDACTED]' other=ok`,
	} {
		assertAllEntryPoints(t, in, in, unparseableJSONPlaceholder)
	}
	// With none, it is prose to redactErrorMessage and sanitizeRawText, and
	// malformed JSON text to SmartTruncate.
	assertTextKeptAndValueFailsClosed(t, `"S0:password="[REDACTED]""`)
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
		{"api_key: [REDACTED] sk_V4LUE_token:[REDACTED]", "api_key: token:", "api_key: sk_V4LUE_token:"},
		{"api_key: [REDACTED] temp:V4LUE:[REDACTED]", "api_key:", "api_key: temp:V4LUE:"},
		{`api_key: [REDACTED] "dev-V4LUE-password":"[REDACTED]"`, `api_key: password":"`, `api_key: "dev-V4LUE-password":"`},
		{"api_key: [REDACTED] x/temp:V4LUE:[REDACTED]", "api_key:", "api_key: temp:V4LUE:"},
		{"api_key: [REDACTED] secret123Token:[REDACTED]", "api_key: Token:", "api_key: secret123Token:"},
		{"credentials: user:private.key:[REDACTED]", "user:private.key:", "user:private.key:"},
		{"x api.key=[REDACTED]", "api.key=", "api.key="},
		{"DB_PASSWORD:[REDACTED]", "DB_PASSWORD:", "DB_PASSWORD:"},
		{"x\n  temp:a:[REDACTED]", "temp:a:", "temp:a:"},
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
		if !testLeavesLegacyValue(tc.raw, tc.scrubbed) {
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
		{"credentials:\n  \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", `\"DB_PASSWORD\"=\"`},
		{"credentials:\n  \\'temp:a\\'=\\'[REDACTED]\\'", `\'temp:a\'=\'`},
		{"credentials:\n  \\\"OPENAI_API_KEY\\\"=\\\"[REDACTED]\\\"", `\"OPENAI_API_KEY\"=\"`},
		{"credentials:\n  \\\\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", `PASSWORD\"=\"`},
		{"x \\\"DB_PASSWORD\\\"=\\\"[REDACTED]\\\"", `PASSWORD\"=\"`},
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

// TestLegacyPatternBackstopOnUnchangedText covers a word glued to a
// placeholder before a key name that holds a vertical tab or a no-break
// space, which the scanners kept as it was while the legacy expressions
// redacted the word: a placeholder glued to other text is no whole value, so
// the scanners redact the word with the value too, while redacted text that
// the expressions leave as it is stays byte-identical.
func TestLegacyPatternBackstopOnUnchangedText(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct{ in, want string }{
		{"log: '\vtemp:abc'=[REDACTED]V4LUE/'\vDB_PASSWORD'=\"[REDACTED]\"", "log: '\vtemp:abc'=[REDACTED]"},
		{"config:\n  \"password\":[REDACTED]V4LUE/\vtemp:abc:[REDACTED]", "config:\n  \"password\":\"[REDACTED]\""},
		{"log: \"\vpassword\"=[REDACTED]V4LUE/\"\u00a0api_Key\"\v: \"[REDACTED]\"", "log: \"\vpassword\"=[REDACTED]"},
		{"credentials:\n  '\vapi_key'=[REDACTED]V4LUE/temp:abc : [REDACTED]", "credentials:\n  '\vapi_key'=[REDACTED]"},
		{"credentials:\n  'api_key'=[REDACTED]/token : [REDACTED]", "credentials:\n  'api_key'=[REDACTED]"},
	} {
		if got := scrubErrorTokens(tc.in); got != tc.want {
			t.Errorf("scrubErrorTokens(%q) = %q, want %q", tc.in, got, tc.want)
		}
		assertAllEntryPoints(t, tc.in, tc.in, tc.want, "V4LUE")
	}
	for _, in := range []string{
		"config:\n  \"password\":[REDACTED]",
		"log: 'api_key'=[REDACTED] other=ok",
	} {
		assertUnchangedAllEntryPoints(t, in)
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

// assertTextKeptAndValueFailsClosed checks that redactErrorMessage and
// sanitizeRawText leave in, malformed JSON text that is prose (see
// isPositivelyHarmlessProse), byte-identical, and that SmartTruncate, which
// keeps no malformed JSON text, replaces it with the placeholder and reports
// the loss, as on main.
func assertTextKeptAndValueFailsClosed(t *testing.T, in string) {
	t.Helper()
	if got := redactErrorMessage(in); got != in {
		t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
	}
	if got, lost := sanitizeRawText(in); got != in || lost {
		t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
	}
	raw, truncated, err := SmartTruncate(in, -1)
	var got string
	if err != nil || json.Unmarshal(raw, &got) != nil || got != unparseableJSONPlaceholder || !truncated {
		t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%q, true, <nil>)", in, raw, truncated, err, unparseableJSONPlaceholder)
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
		// A placeholder glued to other text in a component is no whole
		// value: the file URI has the value redacted with what is glued
		// to it, as the error text has it redacted.
		{"query_glued_word_before_redacted_pair_and_other_pair", "https://example.com/cb?err=password+%3D+%5BREDACTED%5D" + s1 + "%2Ftoken+%3D+%5BREDACTED%5D%2Capi_key%3D" + s2, "https://example.com/cb?err=password%3D%5BREDACTED%5D%2Capi_key%3D%5BREDACTED%5D"},
		{"query_glued_word_before_spaced_pair", "https://example.com/cb?err=password+%3D+%5BREDACTED%5D" + s1 + "%2Ftoken+%3D+" + s2, "https://example.com/cb?err=password%3D%5BREDACTED%5D"},
		{"query_glued_word_before_bracket_header", "https://example.com/cb?err=password%3A+%5BREDACTED%5D" + s1 + "%5Bapi_key%3A+" + s2, "https://example.com/cb?err=password%3A%5BREDACTED%5D"},
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
// the longest known placeholder, so that isKnownSentinel accepts each of
// them.
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
	}
	if longest != maxKnownSentinelLen || maxKnownSentinelLen != len(sanitizeBudgetPlaceholder) {
		t.Errorf("maxKnownSentinelLen = %d, want the longest length %d (%q)", maxKnownSentinelLen, longest, sanitizeBudgetPlaceholder)
	}
	if isKnownSentinel(sanitizeBudgetPlaceholder+"]") || isKnownSentinel("[REDACTED") {
		t.Error("isKnownSentinel accepted a text that is no placeholder")
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

// TestGluedPlaceholderBeforeUnredactedValues covers placeholders that an
// earlier pass wrote with other bytes glued to them: such a placeholder is no
// whole value, so the value it starts is redacted as a whole, with the pair
// that it runs into, whatever the glued bytes are (a word, digits,
// punctuation, a non-ASCII letter or a control byte) and whatever text
// follows the chain, as the reproducers of the review of this change show:
// password: [REDACTED]hunter2ZQ/token: [REDACTED] loses hunter2ZQ with or
// without " for user bob" after it. Where the scanners read such text
// otherwise than the expressions on main, it fails closed (see
// leavesLegacyValue). Each row forbids the secrets that main removes. A
// placeholder that stands as a whole value stays as it is, except in text
// that starts with a quote and is no JSON, as "password": "[REDACTED]",
// which fails closed as on main.
func TestGluedPlaceholderBeforeUnredactedValues(t *testing.T) {
	runInParallel(t)
	const (
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
	)
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct {
		in, want  string
		forbidden []string
	}{
		{"password: [REDACTED]hunter2ZQ/token: [REDACTED]", "password:[REDACTED]", []string{"hunter2ZQ"}},
		{"password: [REDACTED]hunter2ZQ/token: [REDACTED] for user bob", "password:[REDACTED] for user bob", []string{"hunter2ZQ"}},
		{"DB_PASSWORD:x\v'password :[REDACTED]RVIEWz9MARK", redactedSensitiveTextPlaceholder, []string{"RVIEWz9MARK"}},
		{"credential:\\'temp:\\\": '= [REDACTED]RVIEWz9MARK", redactedSensitiveTextPlaceholder, []string{"RVIEWz9MARK"}},
		{"api.key:\"secret:'\" ~!@$^*~'", redactedSensitiveTextPlaceholder, []string{"~!@$^*~"}},
		{"x password: [REDACTED]v/token:[REDACTED] api_key=" + s1, "x password:[REDACTED] api_key=[REDACTED]", []string{s1}},
		{"x password: [REDACTED]" + s2 + "/token:[REDACTED] api_key=" + s1, "x password:[REDACTED] api_key=[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/api_key: \"\" [REDACTED] " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/api_key:\n  [REDACTED] " + s2, "password:[REDACTED] " + s2, []string{s1}},
		{"password: [REDACTED]" + s1 + "/api_key: [BINARY DATA] " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/\"token\": \"[REDACTED] " + s2 + "\"", "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/token: \"[REDACTED]\"/" + s2, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]" + s1 + "/api_key: [REDACTED] " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/token:\n  [REDACTED][api_key: " + s2, "password:[REDACTED] [REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/api_key: [REDACTED]" + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/api_key: [REDACTED] other=" + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/token:\r\n[REDACTED] api_key=" + s2, "password:[REDACTED] api_key=[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "/token: [REDACTED] " + s2 + " api_key=v", "password:[REDACTED] " + s2 + " api_key=[REDACTED]", []string{s1}},
		{"password: [REDACTED]" + s1 + "/token: [MAX_DEPTH_EXCEEDED] api_key=" + s2, "password:[REDACTED] api_key=[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]" + s1 + "[api_key: " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]/" + s1 + "/api_key: " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]_" + s1 + "[api_key: " + s2, "password:[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]/" + s1 + "/api_key:" + s2 + ",other=[REDACTED]", "password:[REDACTED]", []string{s1, s2}},
		{"password = [REDACTED]" + s1 + "/token = " + s2, "password=[REDACTED]", []string{s1}},
		{"password: [REDACTED]" + s1 + "/token\t= " + s2, "password:[REDACTED]", []string{s1}},
		{"'temp:x': [REDACTED]" + s1 + "\\\"[\"Authorization: Bearer " + s2, "'temp:x':[REDACTED]", []string{s1, s2}},
		{"password: [REDACTED]/x/token: [REDACTED] api_key=" + s1, "password:[REDACTED] api_key=[REDACTED]", []string{s1}},
		{"password: [REDACTED]/x-api_key: [REDACTED]" + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]/x/token:\"[REDACTED]\" api_key=" + s1, "password:[REDACTED] api_key=[REDACTED]", []string{s1}},
		{"password: [REDACTED]/x/token: [REDACTED]\napi_key=" + s1, "password:[REDACTED]\napi_key=[REDACTED]", []string{s1}},
		{"password: [REDACTED]/" + a130 + "-api_key: [REDACTED]\napi_key=" + s1, "password:[REDACTED]\napi_key=[REDACTED]", []string{s1}},
		{"x=[REDACTED]/y=[REDACTED] api_key=" + s1, "x=[REDACTED]/y=[REDACTED] api_key=[REDACTED]", []string{s1}},
		{"password: [REDACTED]/x/token:'[REDACTED]'", "password:[REDACTED]", []string{}},
		{"password: [REDACTED]/x/token: [REDACTED],other=ok", "password:[REDACTED],other=ok", []string{}},
		{"password: [REDACTED]" + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]123456", "password:[REDACTED]", []string{"123456"}},
		{"password: [REDACTED]_" + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]-" + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]." + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]~!@$^*~", "password:[REDACTED]", []string{"~!@$^*~"}},
		{"password: [REDACTED]é" + s1, "password:[REDACTED]", []string{s1}},
		{"password: [REDACTED]\v" + s1, "password:[REDACTED]", []string{s1}},
		{"token=[REDACTED][REDACTED]" + s1, "token=[REDACTED]", []string{s1}},
		{"token=" + s1 + "[REDACTED]", "token=[REDACTED]", []string{s1}},
		{"token=[REDACTED]" + s1 + "[REDACTED]", "token=[REDACTED]", []string{s1}},
		{"password: password", "password:[REDACTED]", []string{}},
		{"api_key: api_key other=1", "api_key: [REDACTED]", []string{}},
		{"secret=\"secret\" ok", "secret=\"[REDACTED]\" ok", []string{}},
		{"password: ~!@$^*~' x", "password:[REDACTED] x", []string{"~!@$^*~"}},
		{"token: !@#$%^&*() y", "token:[REDACTED]&*() y", []string{"!@#$%^&*()"}},
		{"secret=---===+++ z", "secret=[REDACTED] z", []string{"---===+++"}},
	} {
		assertAllEntryPoints(t, tc.in, tc.in, tc.want, tc.forbidden...)
		if got := formatContentTextPart(t, tc.in); got != tc.want {
			t.Errorf("FormatContentParts(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if got := scrubErrorText(tc.in); got != tc.want {
			t.Errorf("scrubErrorText(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, in := range []string{
		"password: [REDACTED]",
		"password: [REDACTED] other=ok",
		"token=\"[REDACTED]\", user=bob",
		"\\\"password\\\": \\\"[REDACTED]\\\"",
		"api_key:\n  [REDACTED] other=ok",
		"x=[REDACTED];token=[REDACTED]",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}
	assertAllEntryPoints(t, "quote_led_redacted_pair", "\"password\": \"[REDACTED]\"", unparseableJSONPlaceholder)
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

// TestOpenerScanBudgetLinearScaling covers spans of escaped quotes that pair
// one way once a "{" or "[" is open in a quoted string and another way before,
// next to an escaped or bare single quote, which made an earlier scan of the
// containers in quoted strings read the rest of the text again from each
// quote. The scans read no container in a quoted string, so the work stays
// linear: a JSON-escaped span of 224 KiB goes through every entry point
// within the limits of the large-text checks with the value of every pair in
// it redacted, no secret before, inside or after such a span is kept, and a
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
	maxWall := 10 * time.Second
	if raceEnabled {
		maxWall = 60 * time.Second
	}
	for _, u := range []struct{ name, unit string }{
		{"escaped quotes around an escaped single quote", `\" [ \" \' token:a\" `},
		{"escaped quotes around a single quote", `\" [ \" ' token:a\" `},
		{"bracket before escaped quotes around an escaped single quote", `[ \" \' token:a\" \" `},
	} {
		t.Run(u.name, func(t *testing.T) {
			smallN, largeN := scalingUnits()
			for _, form := range []struct {
				name string
				mk   func(n int) string
			}{
				{"raw", func(n int) string { return span(u.unit, n) }},
				{"JSON-escaped", func(n int) string { return jsonInner(span(u.unit, n)) }},
			} {
				small, large := form.mk(smallN), form.mk(largeN)
				scrub := func(s string) {
					_ = redactErrorMessage(s)
					_, _ = sanitizeRawText(s)
					_, _, _ = SmartTruncate(s, -1)
				}
				scrub(small)
				scrub(large)
				if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
					t.Errorf("%s: %d units took %v, %d units took %v: ratio %.0f, want linear", form.name, smallN, tSmall, largeN, tLarge, ratio)
				}
			}

			// A JSON-escaped span of 224 KiB (see largeTextLen) has the value
			// of every pair in it redacted by every entry point, idempotently.
			unitLen := len(jsonInner(u.unit))
			in := jsonInner(span(u.unit, (largeTextLen()+unitLen-1)/unitLen))
			want := scrubErrorTokens(in)
			if strings.Contains(want, "token:a") || !strings.Contains(want, "token:"+redactedValue) {
				t.Errorf("scrubErrorTokens(%d bytes) = %.80q, want every token value redacted", len(in), want)
			}
			eps := []struct {
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
			}
			got, again, elapsed := timeEntryPoints(maxWall, in, eps)
			for i, ep := range eps {
				if got[i] != want {
					t.Errorf("%s(%d bytes) = %.80q, want %.80q", ep.name, len(in), got[i], want)
				} else if again[i] != got[i] {
					t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got[i], again[i])
				}
			}
			if elapsed > maxWall {
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

// assertLargeTextLinear checks that a large text of repeated unit (see
// largeTextLen) runs through every entry point within a wall-clock limit (see
// timeEntryPoints), keeps none of forbidden, and is unchanged by a second
// pass. A text that the scans redact rather than fail closed does the work of
// every pass, so the limit is wider than that of assertLargeTextFailsClosed,
// while a scan that is quadratic in such a text takes minutes.
func assertLargeTextLinear(t *testing.T, unit string, forbidden ...string) {
	t.Helper()
	in := strings.Repeat(unit, (largeTextLen()+len(unit)-1)/len(unit))
	maxWall := 10 * time.Second
	if raceEnabled {
		maxWall = 60 * time.Second
	}
	eps := largeTextEntryPoints(t)
	got, again, elapsed := timeEntryPoints(maxWall, in, eps)
	for i, ep := range eps {
		for _, f := range forbidden {
			if strings.Contains(got[i], f) {
				t.Errorf("%s(%d bytes of %.40q) kept %q", ep.name, len(in), unit, f)
			}
		}
		if again[i] != got[i] {
			t.Errorf("%s second pass(%d bytes of %.40q) = %.80q, want unchanged", ep.name, len(in), unit, again[i])
		}
	}
	if elapsed > maxWall {
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
	eps := largeTextEntryPoints(t)
	got, again, elapsed := timeEntryPoints(maxWall, in, eps)
	for i, ep := range eps {
		if got[i] != redactedSensitiveTextPlaceholder {
			t.Errorf("%s(%d bytes) = %.80q, want %q", ep.name, len(in), got[i], redactedSensitiveTextPlaceholder)
		} else if again[i] != got[i] {
			t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got[i], again[i])
		}
	}
	if elapsed > maxWall {
		t.Errorf("%d bytes took %v through the entry points, want at most %v", len(in), elapsed, maxWall)
	}
}

// timeEntryPoints runs in through each of eps, and each output through it
// again, and returns the outputs, those of the second passes and how long the
// runs took, timed as withinWallClock times them against limit, so that a
// scheduler pause on a loaded host or another test running in parallel does
// not fail a wall-clock check that a quadratic scan fails on every run.
func timeEntryPoints(limit time.Duration, in string, eps []struct {
	name string
	f    func(string) string
},
) (got, again []string, elapsed time.Duration) {
	got, again = make([]string, len(eps)), make([]string, len(eps))
	elapsed = withinWallClock(limit, func() {
		for i, ep := range eps {
			got[i] = ep.f(in)
			again[i] = ep.f(got[i])
		}
	})
	return got, again, elapsed
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

// assertSecondCall checks what f, the entry point named name, returns for
// got, a result of its own: got unchanged, or got with the text that main
// fails closed in it failed closed (see failsClosedAsOnMain). SmartTruncate,
// which keeps no malformed JSON text, must return the placeholder for a
// result that starts with "{" or "[" and is no JSON, as it does on main.
func assertSecondCall(t *testing.T, name string, f func(string) string, got string) {
	t.Helper()
	again := f(got)
	if again != got && !failsClosedAsOnMain(got, again) {
		t.Errorf("%s second call(%q) = %q, want unchanged or failed closed as on main", name, got, again)
	}
	trimmed := trimBOMAndSpace(got)
	malformed := trimmed != "" && (trimmed[0] == '{' || trimmed[0] == '[') && !isKnownSentinel(trimmed) && !json.Valid([]byte(trimmed))
	if name == "SmartTruncate" && malformed && again != unparseableJSONPlaceholder {
		t.Errorf("%s second call(%q) = %q, want %q", name, got, again, unparseableJSONPlaceholder)
	}
}

// TestJSONDocumentValuesKeepNoSecret covers string values that are a JSON
// object or array as a whole, as the arguments of a tool call written as JSON
// text, whose leaves hold credentials: a URL leaf that ends in a credential,
// whether as the last field or not and in nested objects and arrays, a pair
// in a leaf, and a sensitive key. Every entry point keeps no secret. A
// second call changes nothing, except that SmartTruncate, which keeps no
// malformed JSON text, fails closed a result that the text scrub left
// malformed, as on main. SmartTruncate and FormatContentParts scrub
// such a document as text, as sanitizeRawText does and as main did, rather
// than leaf by leaf, malformed JSON text fails closed, a document with nothing
// to redact stays byte-identical, and a long document is redacted before it
// is truncated.
func TestJSONDocumentValuesKeepNoSecret(t *testing.T) {
	runInParallel(t)
	eps := largeTextEntryPoints(t)
	for _, in := range []string{
		`{"url":"https://example.com/path?a=1&access_token=SECRET1"}`,
		`{"url":"https://example.com/path?a=1&access_token=SECRET1","other":"ok"}`,
		`{"args":{"url":"https://example.com/cb?api_key=SECRET1"},"n":1}`,
		`[{"url":"https://example.com/cb#token=SECRET1"},"ok"]`,
		`{"msg":"foo&access_token=SECRET1"}`,
		`{"x":"bar password=SECRET1"}`,
		`{"cfg":["a","DB_PASSWORD: SECRET1"]}`,
		`{"access_token":"SECRET1","other":"ok"}`,
		`{"msg":"password=SECRET1","detail":{"url":"https://h/p?token=SECRET2&x=1"}}`,
		`{"msg":"password=SECRET1"`,
		`{"msg":"password=SECRET1"}} trailing`,
		`[1, 2, {"api_key": "SECRET1"}`,
	} {
		srt, _ := sanitizeRawText(in)
		for _, ep := range eps {
			got := ep.f(in)
			if strings.Contains(got, "SECRET") {
				t.Errorf("%s(%q) = %q kept a secret", ep.name, in, got)
			}
			assertSecondCall(t, ep.name, ep.f, got)
			if (ep.name == "SmartTruncate" || ep.name == "FormatContentParts") && got != srt {
				t.Errorf("%s(%q) = %q, sanitizeRawText = %q, want the same", ep.name, in, got, srt)
			}
		}
	}
	for _, in := range []string{`{"msg":"password=SECRET1"`, `{"msg":"password=SECRET1"}} trailing`, `[1, 2, {"api_key": "SECRET1"}`} {
		for _, ep := range eps[2:] {
			if got := ep.f(in); got != unparseableJSONPlaceholder {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, unparseableJSONPlaceholder)
			}
		}
	}
	assertUnchangedAllEntryPoints(t, `{"note":"talk to Ken tomorrow","n":[1,2,3]}`)

	long := `{"url":"https://example.com/path?a=1&access_token=SECRET1","pad":"` + strings.Repeat("x", 200) + `"}`
	raw, truncated, err := SmartTruncate(long, 100)
	var got string
	if err != nil || json.Unmarshal(raw, &got) != nil || !truncated || !strings.HasSuffix(got, truncationSuffix) || strings.Contains(got, "SECRET") {
		t.Errorf("SmartTruncate(%d bytes, 100) = (%s, %v, %v), want the redacted text truncated", len(long), raw, truncated, err)
	}
}

// TestTokenizerSpecialTokenNames checks that the names of the special tokens
// of a tokenizer, as eos_token or bosToken, are not sensitive, as pad_token
// is not, while credential names that end in "token" stay sensitive, also
// through every entry point. The expressions on main read the token of
// eos-token or eos.token as a key on its own, after a "-" or ".", so those
// forms stay redacted as on main.
func TestTokenizerSpecialTokenNames(t *testing.T) {
	runInParallel(t)
	for _, word := range []string{"eos", "eot", "bos", "pad", "unk", "cls", "sep", "mask"} {
		for _, k := range []string{
			word + "_token", word + "Token", strings.ToUpper(word) + "_TOKEN",
			strings.ToUpper(word[:1]) + word[1:] + "Token",
		} {
			if isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = true, want false", k)
			}
			assertUnchangedAllEntryPoints(t, k+": <tok>")
			assertUnchangedAllEntryPoints(t, `{"`+k+`": "<tok>"}`)
		}
		for _, k := range []string{word + "-token", word + ".token"} {
			if isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = true, want false", k)
			}
			assertAllEntryPoints(t, k, k+": TOKSECRET1", k+":[REDACTED]", "TOKSECRET1")
		}
	}
	for _, k := range []string{
		"access_token", "auth_token", "refresh_token", "api_token", "session_token", "bearer_token",
		"id_token", "oauth_token", "csrf_token", "secret_token", "authToken", "CSRF_TOKEN", "my_eos_token",
	} {
		if !isSensitiveKey(k) {
			t.Errorf("isSensitiveKey(%q) = false, want true", k)
		}
		assertAllEntryPoints(t, k, k+": TOKSECRET1", k+":[REDACTED]", "TOKSECRET1")
	}
}

// TestAdversarialScalingGluedPairRuns checks that runs of pairs glued to one
// another with no value stop in between, which made the redaction of glued
// values read the rest of the run again from each pair, stay linear through
// every entry point, keep no secret and are unchanged by a second pass: quoted
// JSON pairs joined by "/", pairs after placeholders, placeholder-led runs,
// bare pairs, and quoted pairs joined by ";".
func TestAdversarialScalingGluedPairRuns(t *testing.T) {
	runInParallel(t)
	for _, shape := range []struct {
		name, unit string
		// want is what scrubErrorText returns for a run of the unit: the
		// value of the first pair runs on over the rest of the run, which
		// is redacted with it as one value, rather than failing closed.
		want      string
		forbidden []string
	}{
		{"glued quoted pairs", `"token":"GLUEDSECRET"/`, `"token":"[REDACTED]"`, []string{"GLUEDSECRET"}},
		{"pairs after placeholders", `password=[REDACTED]/`, "password=[REDACTED]", nil},
		{"placeholder-led runs", `[REDACTED]x/token:[REDACTED]/`, "[REDACTED]x/token:[REDACTED]", nil},
		{"placeholder-led glued values", `token:[REDACTED]GLUEDSECRET/`, "token:[REDACTED]", []string{"GLUEDSECRET"}},
		{"glued bare pairs", `api_key:GLUEDSECRET/token:GLUEDSECRET2/`, "api_key: [REDACTED]", []string{"GLUEDSECRET"}},
		{"quoted pairs joined by semicolons", `secret="GLUEDSECRET";token="GLUEDSECRET2";`, "", []string{"GLUEDSECRET"}},
	} {
		t.Run(shape.name, func(t *testing.T) {
			if len(shape.forbidden) > 0 {
				assertLinearAndRedacted(t, func(n int) string { return strings.Repeat(shape.unit, n) }, shape.forbidden...)
			}
			assertLargeTextLinear(t, shape.unit, shape.forbidden...)
			if shape.want != "" {
				in := strings.Repeat(shape.unit, (largeTextLen()+len(shape.unit)-1)/len(shape.unit))
				if got := scrubErrorText(in); got != shape.want {
					t.Errorf("scrubErrorText(%d bytes of %q) = %.80q, want %q", len(in), shape.unit, got, shape.want)
				}
			}
		})
	}
}

// TestSpanTracesMapCopiesToEqualBytes checks the traces of the scanners and
// of the legacy expressions (see passTrace) on texts that every kind of pass
// rewrites, URIs whose rewrite escapes bytes included: each run that a trace
// records as a copy maps bytes of the result to equal bytes of the text, in
// order, or, in a rewritten URI, to the bytes that it writes escaped or
// unescaped (see copyRun and sameRunBytes), each span that a trace records
// as removed lies in the text, and no trace is unaligned. It also checks
// composeRuns and mapSpans on a small chain of passes.
func TestSpanTracesMapCopiesToEqualBytes(t *testing.T) {
	runInParallel(t)
	checkRuns := func(label, raw, out string, runs []copyRun) {
		t.Helper()
		lastOut, lastIn := -1, -1
		for _, r := range runs {
			if r.n <= 0 || r.out < 0 || r.in < 0 || r.out+r.n > len(out) || r.in+r.n > len(raw) {
				t.Errorf("[%s] run %+v out of range (out %d, raw %d bytes)", label, r, len(out), len(raw))
				return
			}
			if r.out <= lastOut || r.in <= lastIn {
				t.Errorf("[%s] run %+v out of order", label, r)
			}
			lastOut, lastIn = r.out+r.n-1, r.in+r.n-1
			if !sameRunBytes(out, raw, r) {
				t.Errorf("[%s] run %+v maps %q to %q", label, r, out[r.out:r.out+r.n], raw[r.in:r.in+r.n])
			}
		}
	}
	for _, raw := range []string{
		"password: [REDACTED]hunter2ZQ/token: [REDACTED] for user bob",
		"Authorization: Bearer abc123 api_key: v1 v2\nx-api-key:\n  v3",
		`log: "token": "abc", ' client_secret ': v, "  api.key  ": [1, 2]`,
		"failed: Basic dXNlcjpwYXNz retry password=x https://u:p@h/a?token=t&ok=1#k=v",
		"credentials:\n  temp:a:b: v1\n  C:\\ssl\\password: v2 dbPassword={a: b}/password: v3",
		`msg: "password=" "SECRET" api.key:"secret:'" ~!@$^*~'`,
		"see https://h/^/secret_key/password:#secret_hunter2&5 now",
		"GET HTTPS://h/a|b?q=a%20b+c&token=x&Bearer+SECRET+token=y#k=v failed",
		"https://h/caf\u00e9/%7Bx%7D?key=1&err=password%3A+abc+def",
	} {
		var tr passTrace
		out := scrubErrorTokensTraced(raw, &tr)
		if out != scrubErrorTokens(raw) {
			t.Errorf("scrubErrorTokensTraced(%q) = %q, scrubErrorTokens = %q", raw, out, scrubErrorTokens(raw))
		}
		checkRuns("scanners "+raw, raw, out, tr.runs)
		if tr.unaligned {
			t.Errorf("scrubErrorTokensTraced(%q) left a URI unaligned", raw)
		}
		for _, compound := range []bool{false, true} {
			var lt passTrace
			lout := legacyScrubTokensTraced(raw, compound, &lt)
			if want := legacyScrubErrorTokens(raw); !compound && lout != want {
				t.Errorf("legacyScrubTokensTraced(%q) = %q, legacyScrubErrorTokens = %q", raw, lout, want)
			}
			checkRuns(fmt.Sprintf("legacy(%v) %s", compound, raw), raw, lout, lt.runs)
			for _, sp := range lt.removed {
				if sp[0] < 0 || sp[0] >= sp[1] || sp[1] > len(raw) {
					t.Errorf("[legacy(%v) %s] removed span %v out of range", compound, raw, sp)
				}
			}
			if lt.unaligned {
				t.Errorf("legacyScrubTokensTraced(%q, %v) left a URI unaligned", raw, compound)
			}
		}
	}

	// Two passes over "ab-cd-ef": the first writes X for "cd", the second Y
	// for "ab", so that "-", "X"'s "-" and "ef" stay copies of the text.
	inner := []copyRun{{0, 0, 3}, {4, 5, 3}} // "ab-" and "-ef" around "X"
	outer := []copyRun{{1, 2, 5}}            // "Y" + "-X-ef"[...]
	if got, want := composeRuns(outer, inner), []copyRun{{1, 2, 1}, {3, 5, 3}}; !slices.Equal(got, want) {
		t.Errorf("composeRuns = %v, want %v", got, want)
	}
	if got, want := mapSpans([][2]int{{0, 2}, {3, 7}}, inner), [][2]int{{0, 2}, {5, 8}}; !slices.Equal(got, want) {
		t.Errorf("mapSpans = %v, want %v", got, want)
	}
	if from, to := trimRemovedEdges(`a \"x y\"`, 2, 9); from != 4 || to != 7 {
		t.Errorf("trimRemovedEdges = (%d, %d), want (4, 7)", from, to)
	}
}

// TestLegacyURIKeysAsOnMain checks that the legacy scrubs read the keys of a
// URI as the expressions on main did, without compound names (see
// legacyURIKeys), so that they leave for their text passes what main left: a
// fragment key that only the scanners read as sensitive, with its value after
// a space, has that value redacted by the text pass on main, so the text
// fails closed when the scanners, which redact the empty value in the URI,
// keep the word after it.
func TestLegacyURIKeysAsOnMain(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct {
		key  string
		want bool
	}{
		{"password", true},
		{"API-Key", true},
		{"user:token", true},
		{"app:x_api_key", true},
		{"temp:x", true},
		{"spring.datasource.password", false},
		{"dbPassword", false},
		{"secret_key", false},
		{"signing_key", false},
		{"api.key", false},
		{"pad_token", false},
	} {
		if got := isLegacySensitiveKey(tc.key); got != tc.want {
			t.Errorf("isLegacySensitiveKey(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
	in := "https://h/cb#token=URISECRET1&x=1&spring.datasource.password= URISECRET2 for user bob"
	if got := legacyScrubErrorTokens(in); strings.Contains(got, "URISECRET2") || !strings.Contains(got, "spring.datasource.password=[REDACTED]") {
		t.Errorf("legacyScrubErrorTokens(%q) = %q, want the value after the URI redacted as on main", in, got)
	}
	assertAllEntryPoints(t, in, in, redactedSensitiveTextPlaceholder, "URISECRET1", "URISECRET2")
	if got := formatContentTextPart(t, in); got != redactedSensitiveTextPlaceholder {
		t.Errorf("FormatContentParts(%q) = %q, want %q", in, got, redactedSensitiveTextPlaceholder)
	}
}

// sameRunBytes reports whether the run r maps each byte of
// out[r.out:r.out+r.n] to an equal byte of raw[r.in:r.in+r.n], to a letter
// of a URI scheme in another case, or, where either side starts an escape or
// writes a space as "+", to the first byte of how the other side writes the
// same byte (see copyRun).
func sameRunBytes(out, raw string, r copyRun) bool {
	unit := func(s string, i int) byte {
		switch {
		case s[i] == '%' && i+2 < len(s) && isHexByte(s[i+1]) && isHexByte(s[i+2]):
			return unhexByte(s[i+1])<<4 | unhexByte(s[i+2])
		case s[i] == '+':
			return ' '
		}
		return s[i]
	}
	for j := range r.n {
		a, b := out[r.out+j], raw[r.in+j]
		if a == b || (a|0x20 == b|0x20 && 'a' <= a|0x20 && a|0x20 <= 'z') {
			continue
		}
		if unit(out, r.out+j) != unit(raw, r.in+j) {
			return false
		}
	}
	return true
}

// spansCover reports whether spans hold every byte of [from, from+n).
func spansCover(spans [][2]int, from, n int) bool {
	for p := from; p < from+n; p++ {
		held := false
		for _, sp := range spans {
			held = held || (sp[0] <= p && p < sp[1])
		}
		if !held {
			return false
		}
	}
	return n > 0
}

// runsCoverIn reports whether runs copy every byte of [from, from+n) of
// their input.
func runsCoverIn(runs []copyRun, from, n int) bool {
	spans := make([][2]int, len(runs))
	for i, r := range runs {
		spans[i] = [2]int{r.in, r.in + r.n}
	}
	return spansCover(spans, from, n)
}

// TestURIKeyCredentialsRedactedWithTheirKey covers a credential written into
// a URI key that reads as sensitive, as the Bearer SECRET_ONE of
// ?Bearer+SECRET_ONE+token=SECRET_TWO, which reads as the key
// "Bearer SECRET_ONE token" with each "+" as a space: the scanners scrub such
// a key as text, as main scrubbed a key that it did not read as sensitive,
// also when its value is redacted already, so that no entry point keeps the
// credential, and each output is unchanged by a second pass. The legacy scrub
// records what it removes from a URI component, so that the backstop also
// fails closed a scrub that keeps such a credential.
func TestURIKeyCredentialsRedactedWithTheirKey(t *testing.T) {
	runInParallel(t)
	for _, tc := range []struct{ in, want string }{
		{"https://example.com/cb?Bearer+SECRET_ONE+token=SECRET_TWO", "https://example.com/cb?Bearer+%5BREDACTED%5D+token=%5BREDACTED%5D"},
		{"https://example.com/cb?Basic+dXNlcjpwYXNz+token=SECRET_TWO", "https://example.com/cb?Basic+%5BREDACTED%5D+token=%5BREDACTED%5D"},
		{"https://example.com/cb?Bearer+SECRET_ONE+token=[REDACTED]", "https://example.com/cb?Bearer+%5BREDACTED%5D+token=[REDACTED]"},
		{"https://example.com/cb?x=1&Bearer+SECRET_ONE+token=SECRET_TWO&y=2", "https://example.com/cb?x=1&Bearer+%5BREDACTED%5D+token=%5BREDACTED%5D&y=2"},
		{"https://example.com/cb?my+Bearer+SECRET_ONE+secret=x", "https://example.com/cb?my+Bearer+%5BREDACTED%5D+secret=%5BREDACTED%5D"},
		{"https://example.com/cb?Basic+dXNlcjpwYXNz+api_key=[REDACTED]", "https://example.com/cb?Basic+%5BREDACTED%5D+api_key=[REDACTED]"},
		{"see https://example.com/cb?Bearer+SECRET_ONE+token=SECRET_TWO now", "see https://example.com/cb?Bearer+%5BREDACTED%5D+token=%5BREDACTED%5D now"},
		{"https://example.com/cb#Bearer%20SECRET_ONE%20token=SECRET_TWO", "https://example.com/cb#Bearer%20%5BREDACTED%5D%20token=%5BREDACTED%5D"},
		{"https://example.com/cb?api_key%3DSECRET_ONE+token=[REDACTED]", redactedSensitiveTextPlaceholder},
		{"https://example.com/a;Bearer%20SECRET_ONE%20token=SECRET_TWO/b", redactedSensitiveTextPlaceholder},
	} {
		for _, ep := range largeTextEntryPoints(t) {
			got := ep.f(tc.in)
			if got != tc.want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, tc.in, got, tc.want)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	// sanitizeURI scrubs a sensitive key of a path segment or a fragment
	// pair as text too.
	for _, tc := range []struct{ in, want string }{
		{"https://example.com/a/Bearer%20SECRET_ONE%20token=[REDACTED]", "https://example.com/a/Bearer%20%5BREDACTED%5D%20token=%5BREDACTED%5D"},
		{"https://example.com/a/Bearer%20SECRET_ONE%20token=SECRET_TWO", "https://example.com/a/Bearer%20%5BREDACTED%5D%20token=%5BREDACTED%5D"},
		{"https://example.com/cb#x=1&Bearer%20SECRET_ONE%20token=[REDACTED]", "https://example.com/cb#x=1&Bearer%20%5BREDACTED%5D%20token=%5BREDACTED%5D"},
	} {
		if got, _ := sanitizeURI(tc.in); got != tc.want {
			t.Errorf("sanitizeURI(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	// The legacy scrub reads the key as main did, as no sensitive key, and
	// redacts the credential in it; it records that, so that a scrub that
	// keeps the credential keeps a byte that the legacy scrub removes.
	raw := "https://example.com/cb?Bearer+SECRET_ONE+token=SECRET_TWO"
	var tr passTrace
	legacyScrubTokensTraced(raw, false, &tr)
	if !spansCover(tr.removed, strings.Index(raw, "SECRET_ONE"), len("SECRET_ONE")) {
		t.Errorf("legacyScrubTokensTraced(%q) removed %v, want SECRET_ONE among them", raw, tr.removed)
	}
	kept := "https://example.com/cb?Bearer+SECRET_ONE+token=%5BREDACTED%5D"
	var ktr passTrace
	w := spanWriter{s: raw, tr: &ktr}
	w.rewriteURI(0, len(raw), kept)
	var m scrubMemo
	if !m.leavesLegacyValue(raw, kept, ktr.runs) {
		t.Errorf("leavesLegacyValue(%q, %q) = false, want true", raw, kept)
	}
}

// TestURIRewriteAlignment covers the alignment of a URI that sanitizeURIWith
// rewrites with the text (see uriAligner). url.URL.String escapes bytes of a
// path, as ^, |, {, }, a backtick, a backslash and non-ASCII bytes, when it
// writes a path that the rewrite of a newly sensitive segment changed, and a
// byte after the URI that equals a hex digit of such an escape no longer
// pulls the alignment past the fragment or query that the rewrite keeps, so
// that the backstop sees them kept and fails closed the text whose
// password: value main redacted there. The parts of a URI align apart: a
// byte that the rewrite escapes or unescapes maps to the byte that it stands
// for, the scheme maps whatever its case, a value replaced with a
// placeholder counts as removed, a URI replaced as a whole keeps nothing, and
// an alignment whose searches run out of budget marks its trace unaligned,
// which fails the text closed.
func TestURIRewriteAlignment(t *testing.T) {
	runInParallel(t)
	const secret = "secret_hunter2"
	for _, path := range []struct{ seg, tails string }{
		{"^", "5E"}, {"|", "7C"}, {"`", "6"}, {"{", "7B"}, {"}", "7D"}, {"\\", "5C"}, {"café", "C9A"}, {"%7B", "7B"},
	} {
		for _, key := range []string{"secret_key", "signing_key", "user:temp:x"} {
			for _, sep := range []string{"#", "?"} {
				for _, tail := range path.tails {
					in := "https://h/" + path.seg + "/" + key + "/password:" + sep + secret + "&" + string(tail)
					for _, ep := range largeTextEntryPoints(t) {
						got := ep.f(in)
						if strings.Contains(got, secret) {
							t.Errorf("%s(%q) = %q kept %q", ep.name, in, got, secret)
						}
						if again := ep.f(got); again != got {
							t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
						}
					}
				}
			}
		}
	}

	for _, tc := range []struct {
		raw           string
		kept, removed []string
	}{
		{"https://h/^/secret_key/password:#secret_hunter2&5", []string{"https://h/", "^", "#secret_hunter2&5"}, []string{"secret_key", "password:"}},
		{"HTTPS://h/a|b?q=a%20b+c&token=x#k=v", []string{"HTTPS://h/a", "|", "b?q=a%20b+c&token=", "#k=v"}, []string{"x"}},
		{"https://h/café?key=1", []string{"https://h/café?key="}, []string{"1"}},
		{"https://h/?err=password%3Aabc+def&ok=1", []string{"https://h/?err=password%3A", "+def&ok=1"}, []string{"abc"}},
		{"https://u:p@h/a?x=1", nil, []string{"https://u:p@h/a?x=1"}},
	} {
		out, changed := sanitizeURI(tc.raw)
		if !changed {
			t.Fatalf("sanitizeURI(%q) changed nothing", tc.raw)
		}
		var tr passTrace
		w := spanWriter{s: tc.raw, tr: &tr}
		w.rewriteURI(0, len(tc.raw), out)
		for _, r := range tr.runs {
			if !sameRunBytes(out, tc.raw, r) {
				t.Errorf("alignment of %q with %q: run %+v maps %q to %q", out, tc.raw, r, out[r.out:r.out+r.n], tc.raw[r.in:r.in+r.n])
			}
		}
		for _, k := range tc.kept {
			if !runsCoverIn(tr.runs, strings.Index(tc.raw, k), len(k)) {
				t.Errorf("alignment of %q with %q (runs %v) does not keep %q", out, tc.raw, tr.runs, k)
			}
		}
		for _, r := range tc.removed {
			if !spansCover(tr.removed, strings.Index(tc.raw, r), len(r)) {
				t.Errorf("alignment of %q with %q (removed %v) does not remove %q", out, tc.raw, tr.removed, r)
			}
		}
		if tr.unaligned || (tc.kept == nil && len(tr.runs) > 0) {
			t.Errorf("alignment of %q with %q: unaligned %v, runs %v", out, tc.raw, tr.unaligned, tr.runs)
		}
	}

	// An alignment out of budget is unaligned, and so is the scrub.
	raw := "https://h/?err=password%3Aabc+def&ok=1"
	out, _ := sanitizeURI(raw)
	var tr passTrace
	a := newURIAligner(&tr, 0, 0, raw, out)
	a.budget = -1
	a.alignURI()
	if !tr.unaligned || !spansCover(tr.removed, strings.Index(raw, "def"), len("def")) {
		t.Errorf("alignment of %q with %q out of budget: unaligned %v, removed %v", out, raw, tr.unaligned, tr.removed)
	}
	// A scrub whose trace is unaligned fails closed, even for text that
	// it keeps as it is.
	for _, e := range []memoEntry{
		{in: raw, out: out, trace: tr},
		{in: "see https://h/a for details", out: "see https://h/a for details", trace: passTrace{unaligned: true}},
	} {
		var m scrubMemo
		m.tokenRuns.e[0] = e
		m.tokenRuns.n = 1
		if got := m.scrubText(e.in); got != redactedSensitiveTextPlaceholder {
			t.Errorf("scrubText(%q) with an unaligned URI = %q, want %q", e.in, got, redactedSensitiveTextPlaceholder)
		}
	}
}

// repeatUnitTo returns unit repeated to at least n bytes.
func repeatUnitTo(unit string, n int) string {
	return strings.Repeat(unit, max(1, (n+len(unit)-1)/len(unit)))
}

// TestAdversarialScalingURIRewrites checks that URIs whose rewrite writes
// many bytes escaped or changed, as runs of pairs whose values are pairs, of
// password pairs, of bearer values and of "+"-delimited sensitive keys, and
// paths of bytes that url.URL.String escapes, stay linear, keep no secret and
// are unchanged by a second pass of each entry point. Aligning such a URI
// with the text searched the rest of the URI from each such byte, which took
// quadratic time. scrubErrorText, which every entry point runs, is timed at
// 16, 64 and 224 KiB, and every entry point runs the middle size; under the
// race detector, whose instrumentation slows these scans more than ten
// times, scrubErrorText is timed at 2 and 32 KiB and the entry points run
// the smaller size.
func TestAdversarialScalingURIRewrites(t *testing.T) {
	runInParallel(t)
	for _, shape := range []struct {
		name      string
		mk        func(n int) string
		forbidden []string
	}{
		{"pairs whose values are pairs", func(n int) string { return "https://h/?a=1" + repeatUnitTo("&k=token=1", n-14) }, nil},
		{"pairs whose values are password pairs", func(n int) string { return "https://h/?a=1" + repeatUnitTo("&k=password=1", n-14) }, nil},
		{"bearer values", func(n int) string { return "https://h/?a=1" + repeatUnitTo("&k=Bearer%20x", n-14) }, nil},
		{"plus-delimited sensitive keys", func(n int) string {
			return "https://h/?a=1" + repeatUnitTo("&Bearer+PLUSKEYSECRET+token=1", n-14)
		}, []string{"PLUSKEYSECRET"}},
		{"escaped path bytes", func(n int) string { return "https://h/" + strings.Repeat("^", n-16) + "?key=1" }, nil},
	} {
		t.Run(shape.name, func(t *testing.T) {
			large := largeTextLen()
			small, mid, big := shape.mk(large/14), shape.mk(large*2/7), shape.mk(large)
			checked := mid
			if raceEnabled {
				checked = small
			}
			for _, ep := range largeTextEntryPoints(t) {
				got := ep.f(checked)
				for _, f := range shape.forbidden {
					if strings.Contains(got, f) {
						t.Errorf("%s(%d bytes of %.40q) kept %q", ep.name, len(checked), checked, f)
					}
				}
				if again := ep.f(got); again != got {
					t.Errorf("%s second pass(%d bytes of %.40q) = %.80q, want unchanged", ep.name, len(checked), checked, again)
				}
			}
			scrub := func(s string) func() { return func() { _ = scrubErrorText(s) } }
			if !raceEnabled {
				if ratio, tSmall, tMid := linearRatio(16, scrub(small), scrub(mid)); ratio > 16 {
					t.Errorf("%d bytes took %v, %d bytes took %v: ratio %.0f, want linear", len(small), tSmall, len(mid), tMid, ratio)
				}
			}
			ratio, tSmall, tBig := linearRatio(56, scrub(small), scrub(big))
			if ratio > 56 {
				t.Errorf("%d bytes took %v, %d bytes took %v: ratio %.0f, want linear", len(small), tSmall, len(big), tBig, ratio)
			}
			maxWall := 4 * time.Second
			if raceEnabled {
				maxWall = 40 * time.Second
			}
			if tBig > maxWall {
				// Timed again, alone if need be (see withinWallClock).
				tBig = min(tBig, withinWallClock(maxWall, scrub(big)))
			}
			if tBig > maxWall {
				t.Errorf("scrubErrorText(%d bytes of %.40q) took %v, want at most %v", len(big), big, tBig, maxWall)
			}
		})
	}
}

// TestJSONValuesInStringsKeepNoSecret covers valid JSON whose string leaves a
// pass redacts: a leaf that starts with a URI with credentials and goes on
// with quoted pairs or escaped quotes, which a later pass reads as malformed
// JSON text; a leaf that is the JSON text of another document; a JSON string
// literal that holds a document; and a leaf whose text scrub takes its
// closing quote with the value. No entry point keeps a credential and a
// second call changes nothing but what SmartTruncate fails closed as
// malformed JSON text (see assertSecondCall), JSON that does not parse fails
// closed, a credential that the scrub of the text reads across strings is not
// kept, and documents nested in strings 12 levels deep stay fast.
func TestJSONValuesInStringsKeepNoSecret(t *testing.T) {
	runInParallel(t)
	eps := largeTextEntryPoints(t)
	for _, in := range []string{
		`{"log":"https://u:p@h/a \"token\":\"SECRET\""}`,
		`{"log":"https://u:p@h/a 'token':'SECRET'"}`,
		`{"log":"https://u:p@h/a \\\"hi\\\" token=SECRET"}`,
		`{"nested":"{\"url\":\"https://example.com/path?a=1&access_token=SECRET\"}"}`,
		`"{\"url\":\"https://example.com/path?a=1&access_token=SECRET\"}"`,
		`{"msg":"say \"hi\" &access_token=SECRET"}`,
		`[{"url":"https://example.com/cb?x=1&token=SECRET"},"say \"hi\" &access_token=SECRET2"]`,
	} {
		for _, ep := range eps {
			got := ep.f(in)
			if strings.Contains(got, "SECRET") {
				t.Errorf("%s(%q) = %q kept a secret", ep.name, in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	jsonEntryPoints := eps[1:]
	for _, tc := range []struct{ in, want string }{
		{`{"log":"x","nested":"{\"url\":\"https://example.com/?access_token=SECRET\""}`, `{"log":"x","nested":"[UNPARSEABLE_JSON_BLOB]"}`},
		{`{"args":"{\"url\":\"https://h/p?access_token=SECRETZQ\"}"}`, `{"args":"[UNPARSEABLE_JSON_BLOB]"}`},
		{`{"args":"{\"credentials\":{\"AccessKeyId\":\"SECRETZQ\"}"}`, `{"args":"[UNPARSEABLE_JSON_BLOB]"}`},
		{`{"args":"{\"temp:k\":[\"SECRETZQ\"]"}`, `{"args":"[UNPARSEABLE_JSON_BLOB]"}`},
		{`{"msg":"say \"hi\" &access_token=SECRET"`, unparseableJSONPlaceholder},
	} {
		for _, ep := range jsonEntryPoints {
			got := ep.f(tc.in)
			if got != tc.want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, tc.in, got, tc.want)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		if san, lost1 := sanitizeRawText(tc.in); !lost1 {
			t.Errorf("sanitizeRawText(%q) lost=false, want true", tc.in)
		} else if _, lost2 := sanitizeRawText(san); lost2 {
			t.Errorf("sanitizeRawText second pass(%q) reported contentLost", san)
		}
	}
	malformedHeaderLeaf := `{"k":"{ \"a\": \"Authorization: [REDACTED]\n\", \"b\": \"SECRETZQ\""}`
	if got := redactErrorMessage(malformedHeaderLeaf); got != `{"k":"[UNPARSEABLE_JSON_BLOB]"}` {
		t.Errorf("redactErrorMessage(%q) = %q, want malformed leaf replaced", malformedHeaderLeaf, got)
	}
	for _, ep := range jsonEntryPoints[1:] {
		got := ep.f(malformedHeaderLeaf)
		if strings.Contains(got, "SECRETZQ") {
			t.Errorf("%s(%q) = %q kept SECRETZQ", ep.name, malformedHeaderLeaf, got)
		}
		if again := ep.f(got); again != got {
			t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
		}
	}
	// The scrub of the text reads the header value of the first string on
	// into the next, and the credentials in the keys; sanitizeRawText,
	// SmartTruncate and FormatContentParts keep none of them, as on main,
	// while redactErrorMessage redacts the leaves of a JSON document one by
	// one, as it did on main.
	for _, in := range []string{
		`["x-api-key:", "SECRET1", "say \"hi\" &access_token=SECRET2"]`,
		`{"password=SECRET1":"x","c":"say \"hi\" &access_token=SECRET2"}`,
		`{"temp:Bearer SECRET1":"x","c":"say \"hi\" &access_token=SECRET2"}`,
		`{"n":"[\"x-api-key:\", \"SECRET1\"]","c":"say \"hi\" &access_token=SECRET2"}`,
	} {
		for _, ep := range jsonEntryPoints[1:] {
			got := ep.f(in)
			if strings.Contains(got, "SECRET1") || strings.Contains(got, "SECRET2") {
				t.Errorf("%s(%q) = %q kept a secret", ep.name, in, got)
			}
			assertSecondCall(t, ep.name, ep.f, got)
		}
	}
	nested := `{"url":"https://example.com/p?a=1&access_token=NESTEDSECRET"}`
	for range 12 {
		b, err := json.Marshal(map[string]string{"nested": nested})
		if err != nil {
			t.Fatal(err)
		}
		nested = string(b)
	}
	maxWall := 2 * time.Second
	if raceEnabled {
		maxWall = 20 * time.Second
	}
	got, again, elapsed := timeEntryPoints(maxWall, nested, eps)
	for i, ep := range eps {
		if strings.Contains(got[i], "NESTEDSECRET") {
			t.Errorf("%s(%d bytes nested 12 deep) = %.80q kept the secret", ep.name, len(nested), got[i])
		}
		if again[i] != got[i] {
			t.Errorf("%s second pass(%d bytes nested 12 deep) = %.80q, want unchanged", ep.name, len(nested), again[i])
		}
	}
	if elapsed > maxWall {
		t.Errorf("%d bytes nested 12 deep took %v through the entry points, want at most %v", len(nested), elapsed, maxWall)
	}
}

// TestJSONLeavesReadStringsAsSpelled covers JSON documents whose strings the
// scrub of the text reads otherwise than the strings decoded: a header that a
// line break ends in a string, which the text spells as the escape \n, \r,
// \u000a or \u000D, so that its value runs on into the next string, in an
// object, an array, a key, a document in a string and a JSON string literal,
// with or without a placeholder in the header already; a quote that a later
// string closes; a document in a string whose header runs on into the next
// string of the outer document once the redaction has failed the inner one
// closed; quotes that a JSON string literal spells as \u0022, past which a
// bearer credential runs on; a document that the redaction writes again with
// its keys sorted, as for a compound key, past which a header value would run
// on over other strings than on main; a document in a string whose sensitive
// key main redacts, which writes the escaped colon of a header as a colon in
// the text that main scrubs; a header in the value of a compound key, which
// main keeps and runs on over the strings after it once it writes the
// document again with its keys sorted; and a header in the path of a URL
// leaf, which the expressions of main without compound keys read on past the
// leaf. No entry point keeps a secret that main removes from the text,
// redactErrorMessage keeps one only where main scrubs the string that holds
// it on its own, and a second call changes nothing but what SmartTruncate
// fails closed as malformed JSON text (see assertSecondCall), also for a URL
// leaf in a document in a string and a document whose sensitive keys may run
// on into the placeholders of their values.
func TestJSONLeavesReadStringsAsSpelled(t *testing.T) {
	runInParallel(t)
	marshal := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	type spelledCase struct {
		in string
		// keptByLeaves reports that redactErrorMessage scrubs the string
		// that holds the secret on its own, as it did on main, and keeps it.
		keptByLeaves bool
	}
	var cases []spelledCase
	for _, header := range []string{"Authorization:", "x-api-key:", "api-key:", "proxy-authorization:", "Authorization: [REDACTED]", "x-api-key: [REDACTED]"} {
		for _, brk := range []string{`\n`, `\r`, `\u000a`, `\u000D`} {
			str := `"` + header + brk + `"`
			var decoded string
			if err := json.Unmarshal([]byte(str), &decoded); err != nil {
				t.Fatal(err)
			}
			cases = append(cases,
				spelledCase{`{"a":` + str + `,"b":"SECRET"}`, true},
				spelledCase{`[` + str + `,"SECRET"]`, true},
				spelledCase{`{` + str + `:"x","b":"SECRET"}`, true},
				spelledCase{marshal(map[string]string{"k": marshal(map[string]string{"a": decoded, "b": "SECRET"})}), false},
				spelledCase{marshal(marshal([]string{decoded, "SECRET"})), false},
				spelledCase{marshal(marshal(map[string]string{"a": decoded, "b": "SECRET"})), false},
			)
		}
	}
	cases = append(cases,
		spelledCase{`["access_token='","SECRET","'"]`, true},
		spelledCase{`{"a":"token:'","b":"SECRET","c":["token:'","x"]}`, true},
		spelledCase{`{"k":"{\"a\":\"x-api-key:\\r\",\"b\":\"x\"}","c":"SECRET"}`, true},
		spelledCase{`{"x":{"x":"proxy-authorization:  [REDACTED]"},"data":"SECRET","zz_password":"hunter2"}`, true},
		spelledCase{`"{\u0022x Bearer [REDACTED]\u005cr\u0022:\u0022SECRET\u0022}"`, false},
		spelledCase{`["{\"token\":\"x\",\"x_api_key\\u003a\\r\":\"x\"}","SECRET"]`, true},
		spelledCase{`{"password":"p","b":"SECRET","aa_secret":"Authorization: x"}`, true},
		spelledCase{`{"k":"{\"password\":\"p\",\"b\":\"SECRET\",\"aa_secret\":\"Authorization: x\"}"}`, false},
		spelledCase{`{"url":"https://h/b/secret_key/api_key:?x","n":"SECRET"}`, true},
	)
	eps := largeTextEntryPoints(t)
	for _, c := range cases {
		for _, ep := range eps[1:] {
			got := ep.f(c.in)
			if strings.Contains(got, "SECRET") && (!c.keptByLeaves || ep.name != "redactErrorMessage") {
				t.Errorf("%s(%q) = %q kept the secret", ep.name, c.in, got)
			}
			assertSecondCall(t, ep.name, ep.f, got)
		}
	}
	for _, in := range []string{
		`{"nested":"{\"url\":\"https://example.com/path?a=1&access_token=SECRET\",\"b\":\"PLANTED\"}"}`,
		`{"token":"SECRET1","n":"{\"url\":\"https://h/p?access_token=SECRET2\"}"}`,
		`{"password":"SECRET1","url":"https://example.com/p?a=1&access_token=SECRET2"}`,
	} {
		for _, ep := range eps {
			got := ep.f(in)
			if strings.Contains(got, "SECRET") {
				t.Errorf("%s(%q) = %q kept a secret", ep.name, in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
}

// TestExpressionsKeepStringsApartChecks covers the checks behind
// keyRedactionHidesRunOn one by one: the expressions of main applied to a
// document as it is, which may remove no part of more than one value, at each
// level of a document in a string; the decoding of a JSON string that maps
// each decoded byte to its escape; and the spans of the values of a document
// in a string, which start where the outer text spells them and whose closing
// quotes end where their escapes end, so that a reading whose span ends at
// the decoded closing quote of a value ends at that quote, as in a JSON string
// literal that holds {"m":"Bearer \"SECRET;key=b;token=c"}.
func TestExpressionsKeepStringsApartChecks(t *testing.T) {
	runInParallel(t)
	quote := func(s string) string {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	for _, tc := range []struct {
		doc  string
		want bool
	}{
		{`{"a":"x","b":"y"}`, true},
		{`{"a":"x-api-key: v"}`, true},
		{`["x-api-key: v","SECRET"]`, false},
		{`["x-api-key: v",1]`, false},
		{`["access_token='","SECRET","'"]`, false},
		{`{"k":"{\"a\":\"x-api-key: v\",\"b\":\"SECRET\"}"}`, false},
		{`{"k":"[\"x-api-key: v\"]","b":"x"}`, false},
		{`{"k":"[\"token=v\"]","b":"x"}`, true},
		{`{"url":"https://h/b/secret_key/api_key:?x","n":"SECRET"}`, false},
		{`"{\u0022x Bearer [REDACTED]\u005cr\u0022:\u0022SECRET\u0022}"`, false},
		{`{"m":"Bearer \"SECRET;key=b;token=c"}`, false},
		{quote(`{"m":"Bearer \"SECRET;key=b;token=c"}`), false},
		{quote(quote(`{"m":"Bearer \"SECRET;key=b;token=c"}`)), false},
		{quote(`{"m":"Bearer \"SECRET;token=c"}`), false},
		{quote(`{"m":"say \"hi\" to Ken","n":"x"}`), true},
	} {
		if got := expressionsKeepStringsApart(tc.doc); got != tc.want {
			t.Errorf("expressionsKeepStringsApart(%q) = %v, want %v", tc.doc, got, tc.want)
		}
	}
	raw := `a\"b\u00e9\ud83d\ude00\ud800x\/`
	got, pos, ok := decodeJSONStringMapped(raw)
	wantPos := []int{0, 1, 3, 4, 4, 10, 10, 10, 10, 22, 22, 22, 28, 29, 31}
	if want := "a\"b\u00e9\U0001F600\ufffdx/"; !ok || got != want || !slices.Equal(pos, wantPos) {
		t.Errorf("decodeJSONStringMapped(%q) = (%q, %v, %v), want (%q, %v, true)", raw, got, pos, ok, want, wantPos)
	}
	for _, bad := range []string{`a\`, `\x41`, `\u12`, `\u12G4`} {
		if _, _, ok := decodeJSONStringMapped(bad); ok {
			t.Errorf("decodeJSONStringMapped(%q) reports true, want false", bad)
		}
	}
	// The key and the value of the outer document, then those of the
	// document in its string, whose quotes the outer text spells as \".
	var values []jsonValueSpan
	wantSpans := []jsonValueSpan{{1, 3, 4, false}, {5, 15, 16, false}, {7, 10, 12, false}, {13, 13, 14, false}}
	if doc := `{"k":"{\"a\":1}"}`; !appendValueSpans(doc, nil, &values, 0) || !slices.Equal(values, wantSpans) {
		t.Errorf("appendValueSpans(%q) = %v, want %v", doc, values, wantSpans)
	}
}

// TestJSONDocumentsScrubbedAsText covers JSON documents, and JSON string
// literals that hold one, whose credentials every entry point that reads JSON
// text redacts by scrubbing the text, as main did: a header and a key-value
// pair after the escape \n of a line break, next to a word that no scrub
// redacts; a bearer credential that an escaped quote opens, in a JSON string
// literal once and twice; and a quoted placeholder that a word follows after
// the escape \n. No entry point keeps the credential, and a second call
// changes nothing but what SmartTruncate fails closed as malformed JSON text
// (see assertSecondCall). SmartTruncate keeps what sanitizeRawText keeps of a
// document whose scrub takes the closing quote of a header value, as main
// did, rather than failing it closed, and a large array of strings, whose
// last strings the scrub reads as one credential, goes through the node
// budget once, as on main. Text that starts with "{" or "[" and is no JSON
// is prose to redactErrorMessage, sanitizeRawText and FormatContentParts
// when it holds nothing to redact, and fails closed in SmartTruncate
// whatever it holds, as on main.
func TestJSONDocumentsScrubbedAsText(t *testing.T) {
	runInParallel(t)
	marshal := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	eps := largeTextEntryPoints(t)
	for _, tc := range []struct{ doc, secret string }{
		{`{"m":"zqxSentinel7 Authorization:\n;key=SECRETZQ"}`, "SECRETZQ"},
		{`{"m":"Bearer \"SECRETZQ;key=b;token=c"}`, "SECRETZQ"},
		{`{"m":"Bearer \"SECRETZQ;token=c"}`, "SECRETZQ"},
		{`{"m":"Bearer \"SECRETZQ;api_key=c"}`, "SECRETZQ"},
		{`{"k":"{\"m\":\"Bearer \\\"SECRETZQ;key=b;token=c\"}","password":"p"}`, "SECRETZQ"},
		{`{"note":"password:\n\"[REDACTED]\".hunter2ZQ"}`, "hunter2ZQ"},
		{`{"k": "{\"zz_a\":\"Authorization: x\",\"aa_b\":\"SECRETZQ\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`, "SECRETZQ"},
		{`["{\"zz_a\":\"Authorization: x\",\"aa_b\":\"SECRETZQ\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"]`, "SECRETZQ"},
		{`{"k": "{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`, "SECRETZQ"},
		{`["{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"]`, "SECRETZQ"},
		{`{"a": "{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}", "b": "plain"}`, "SECRETZQ"},
		{`{"k":"{\"inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\"}\",\"other\":\"SECRETZQ\"}"}`, "SECRETZQ"},
		{`["{\"inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\"}\",\"other\":\"SECRETZQ\"}"]`, "SECRETZQ"},
		{`"{\"k\":\"{\\\"inner\\\":\\\"{\\\\\\\"zz_password\\\\\\\":\\\\\\\"Authorization: x\\\\\\\"}\\\",\\\"other\\\":\\\"SECRETZQ\\\"}\"}"`, "SECRETZQ"},
		{`{"outer":"{\"k\":\"{\\\"inner\\\":\\\"{\\\\\\\"zz_password\\\\\\\":\\\\\\\"Authorization: x\\\\\\\"}\\\",\\\"other\\\":\\\"SECRETZQ\\\"}\"}"}`, "SECRETZQ"},
		{`{"k":"[\"{\\\"zz_password\\\":\\\"Bearer '\\\"}\",\"SECRETZQ\",\"'\"]"}`, "SECRETZQ"},
		{`{"k":"{\"other\":\"SECRETZQ\",\"a_inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\",\\\"password\\\":\\\"p\\\"}\"}"}`, "SECRETZQ"},
		{`{"a":"{\"other\":\"SECRETZQ\",\"a_inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\",\\\"password\\\":\\\"p\\\"}\"}","b":"plain"}`, "SECRETZQ"},
	} {
		for _, in := range []string{tc.doc, marshal(tc.doc), marshal(marshal(tc.doc))} {
			for _, ep := range eps {
				got := ep.f(in)
				// redactErrorMessage scrubs the strings of a document that
				// is not in a JSON string literal one by one, as main did,
				// and keeps what main kept of each string.
				if strings.Contains(got, tc.secret) &&
					((ep.name != "redactErrorMessage" || in != tc.doc) || strings.Contains(tc.doc, "zz_password") || strings.Contains(tc.doc, "DB_PASSWORD")) &&
					(ep.name != "scrubErrorText" || !strings.Contains(tc.doc, "a_inner")) {
					t.Errorf("%s(%q) = %q kept %q", ep.name, in, got, tc.secret)
				}
				assertSecondCall(t, ep.name, ep.f, got)
			}
		}
	}
	// The header value of the log line, which the text spells with the
	// escape \n, runs on to the end of the text, as on main.
	in := `{"log": "x-api-key: SECRETZQ\nrequest id: 42"}`
	for _, ep := range eps {
		want := `{"log": "x-api-key: [REDACTED]`
		if ep.name == "redactErrorMessage" {
			want = `{"log":"x-api-key: [REDACTED]\nrequest id: 42"}`
		}
		if got := ep.f(in); got != want {
			t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
		} else {
			assertSecondCall(t, ep.name, ep.f, got)
		}
	}
	items := make([]string, 40_000)
	for i := range items {
		items[i] = fmt.Sprintf("item %d", i)
	}
	for _, tail := range [][]string{nil, {"token="}, {"access_token='", "SECRETZQ", "'"}} {
		in := marshal(append(slices.Clone(items), tail...))
		got := eps[3].f(in)
		if strings.Contains(got, sanitizeBudgetPlaceholder) || !strings.Contains(got, `"item 39999"`) || strings.Contains(got, "SECRETZQ") {
			t.Errorf("SmartTruncate(%d strings and %q) = %.60q...%.60q, want every string and no secret", len(items), tail, got, got[max(0, len(got)-60):])
		}
	}
	// Malformed and valid JSON strings with Unicode-escaped case-folded
	// credential keys (\u212a, \u017f, \u0130) keep no secret on any entry
	// point, and a second pass changes nothing.
	for _, in := range []string{
		`{"msg": "to\u212aen=SECRETZQ"`,
		`{"msg": "pa\u017f\u017fword=SECRETZQ"`,
		`[INFO] "pa\u017f\u017fword=SECRETZQ"`,
		`{"msg": "ap\u0130_key=SECRETZQ"`,
		"{\"log\": \"x-api-key: [REDACTED]\n\"pa\\u017f\\u017fword=SECRETZQ\"",
		`[REDACTED_SENSITIVE_URI] "pa\u017f\u017fword=SECRETZQ"`,
		`{"msg": "to\u212aen=SECRETZQ"}`,
		`{"msg": "pa\u017f\u017fword=SECRETZQ"}`,
		`{"msg": "ap\u0130_key=SECRETZQ"}`,
		`{"msg": "unclosed: "'token'\u00a0:\u00a0SECRETZQ"`,
		`[INFO] "unclosed: "'token'\u00a0:\u00a0SECRETZQ"`,
		`[INFO] "unclosed: "'token'\u00a0:\u00a0SECRETZQ`,
		`{"msg": "unclosed: 'token'\u00a0:\u00a0SECRETZQ`,
		"{\"\t\\u00a0token\": \"SECRETZQ\"}",
		`{'\u00a0token': 'SECRETZQ'}`,
		`{'token'\u00a0:\u00a0'SECRETZQ'}`,
		`[INFO] 'token'\u00a0:\u00a0SECRETZQ`,
		`[REDACTED_SENSITIVE_URI] 'token'\u00a0:\u00a0SECRETZQ`,
		"{\"msg\": \"line1\n'token'\\u00a0:\\u00a0SECRETZQ\"}",
		`"{\" password \":\" SECRETZQ \"}" and 'hi'`,
		`"\u007b\u0022password\u0022:\u0022SECRETZQ\u0022\u007d" ok`,
		`"\u007b\u0022a\u0022:1,\u0022a\u0022:2,\u0022password\u0022:\u0022[REDACTED]\u0022\u007d" ok`,
	} {
		for _, ep := range eps {
			got := ep.f(in)
			if strings.Contains(got, "SECRETZQ") &&
				(!strings.Contains(in, `\u00a0`) || ep.name != "scrubErrorText") {
				t.Errorf("%s(%q) = %q kept SECRETZQ", ep.name, in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		if san, _ := sanitizeRawText(in); san != "" {
			if _, againLost := sanitizeRawText(san); againLost {
				t.Errorf("sanitizeRawText second pass(%q) reported contentLost", san)
			}
		}
	}
	// Quote-led JSON string literals whose inner credential runs across the
	// closing quote into a trailing suffix fail closed to
	// redactedSensitiveTextPlaceholder across every JSON entry point.
	for _, in := range []string{
		`"\u007b\u0022DB_PASSWORD\u0022:\u0022Authorization: x\u0022\u007d" SECRETZQ`,
		`"{\" DB_PASSWORD \":\" Authorization: x\"}" SECRETZQ`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ'`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ' " }"`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ%27`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ\u0027`,
	} {
		for _, ep := range eps[1:] {
			got := ep.f(in)
			if got != redactedSensitiveTextPlaceholder {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, redactedSensitiveTextPlaceholder)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	// Malformed JSON text with a BOM, literal or as the escape \ufeff, between
	// a quoted sensitive key and its delimiter, a header after a line
	// separator, literal or percent-encoded, or a bearer credential after a
	// form feed fails closed to unparseableJSONPlaceholder on every entry
	// point.
	for _, in := range []string{
		"{\"token\"\ufeff:\ufeff\"SECRETZQ\"}",
		"{\"credentials\"\ufeff:\ufeff{",
		`{"token"\ufeff:\ufeff"SECRETZQ"}`,
		`{"credentials"\ufeff:\ufeff{`,
		"{note\u2028Authorization:\u00a0SECRETZQ}",
		"[INFO] Bearer\fSECRETZQ",
		"{\"k\":\"Bearer\fSECRETZQ\"",
		`{"note"%E2%80%A8Authorization:%C2%A0SECRETZQ}`,
		`{"password\ufeff":"SECRETZQ"`,
		"{\"password\"\ufeff:\ufeff",
		"[INFO] \"password\"\ufeff:\ufeff",
		"{\"password\"\ufeff:\ufeff[REDACTED]",
	} {
		for _, ep := range eps[1:] {
			got := ep.f(in)
			if got != unparseableJSONPlaceholder {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, unparseableJSONPlaceholder)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		if san, lost1 := sanitizeRawText(in); san != unparseableJSONPlaceholder || !lost1 {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, san, lost1, unparseableJSONPlaceholder)
		} else if _, lost2 := sanitizeRawText(san); lost2 {
			t.Errorf("sanitizeRawText second pass(%q) reported contentLost", san)
		}
		if raw, trunc1, err := SmartTruncate(map[string]any{"v": in}, -1); err != nil || !trunc1 || string(raw) != `{"v":"[UNPARSEABLE_JSON_BLOB]"}` {
			t.Errorf("SmartTruncate(map{v:%q}) = (%s, %v, %v), want unparseable placeholder with truncated=true", in, raw, trunc1, err)
		}
	}
	// Malformed JSON text with a non-ASCII space, a BOM, a line separator or
	// a vertical tab, literal, escaped or percent-encoded, around a delimiter
	// or after a scheme, or with a bearer credential behind escaped quotes
	// that do not decode, is text in which no scanner reads a key or a
	// scheme, on main or here. SmartTruncate, which keeps no malformed JSON
	// text, fails it closed, as a value and in a map, as it does on main;
	// what redactErrorMessage, sanitizeRawText and FormatContentParts make of
	// such text is left to the scanners.
	for _, in := range []string{
		"{password\u00a0:\u00a0SECRETZQ}",
		"{password\u2000:\u2000SECRETZQ}",
		"{password\u3000=\u3000SECRETZQ}",
		"{password\ufeff:\ufeffSECRETZQ}",
		"{password\u00a0:\u00a0SECRETZQ—}",
		"[INFO] password\u00a0:\u00a0SECRETZQ",
		"[INFO] password\u2000=\u2000SECRETZQ",
		"[INFO] password\ufeff:\ufeffSECRETZQ",
		"[REDACTED_SENSITIVE_URI] password\u3000:\u3000SECRETZQ",
		`{"msg": "password\u00a0:\u00a0SECRETZQ"`,
		`{"msg": "password\u2000:\u2000SECRETZQ"`,
		`{"msg": "password\u3000=\u3000SECRETZQ"`,
		`{"msg": "password\ufeff:\ufeffSECRETZQ"`,
		"{DB_PASSWORD\u00a0=\u00a0SECRETZQ}",
		"[INFO] Bearer\u2000SECRETZQ",
		"[INFO] Bearer\ufeffSECRETZQ",
		"[INFO] 'a\u0085db_password=' '[REDACTED]'",
		"[INFO] 'a\u2028db_password=' '[REDACTED]'",
		"[INFO] \"a\u2029db_password=\" \"[REDACTED]\"",
		`[INFO] 'a\u2028db_password=' '[REDACTED]'`,
		"[INFO] Bearer\vSECRETZQ",
		"{\"k\":\"Bearer\vSECRETZQ\"",
		`{"k":"Bearer\u000bSECRETZQ"`,
		`{"password"%C2%A0:%C2%A0"SECRETZQ"}`,
		`{"password"%c2%a0:%c2%a0"SECRETZQ"}`,
		`[INFO] Bearer%C2%A0SECRETZQ`,
		`{"token"%E2%80%80:%E2%80%80"SECRETZQ"}`,
		`{"password"%EF%BB%BF:%EF%BB%BF"SECRETZQ"}`,
		`"[\u0022Bearer \'\u0022,\u0022SECRETZQ\u0022]" ok`,
	} {
		if got := eps[3].f(in); got != unparseableJSONPlaceholder {
			t.Errorf("SmartTruncate(%q) = %q, want %q", in, got, unparseableJSONPlaceholder)
		}
		if raw, trunc1, err := SmartTruncate(map[string]any{"v": in}, -1); err != nil || !trunc1 || string(raw) != `{"v":"[UNPARSEABLE_JSON_BLOB]"}` {
			t.Errorf("SmartTruncate(map{v:%q}) = (%s, %v, %v), want unparseable placeholder with truncated=true", in, raw, trunc1, err)
		}
	}
	for _, in := range []string{
		`{"k":"{\"password\"%C2%A0:%C2%A0\"SECRETZQ\"}"}`,
		`{"k":"{\"k\":\"Bearer\u000bSECRETZQ\"}"}`,
	} {
		for _, ep := range eps[1:] {
			got := ep.f(in)
			if strings.Contains(got, "SECRETZQ") {
				t.Errorf("%s(%q) = %q kept SECRETZQ", ep.name, in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	// Escaped-scheme URIs keep merge-base fail-closed parity in the backstop
	// without allowing BOM or Unicode escape folding to disable URI matching.
	for _, esc := range []string{`\ufeff`, `\u00a0`, `\u212a`, `\u017f`, `\u0130`} {
		in := fmt.Sprintf(`{"s":"https\%s://h/a;Bearer+SECRETZQ+token=x"}`, esc)
		for _, ep := range eps {
			got := ep.f(in)
			if ep.name == "redactErrorMessage" {
				want := fmt.Sprintf(`{"s":"https\%s://h/a%%3BBearer+SECRETZQ+token=%%5BREDACTED%%5D"}`, esc)
				if got != want {
					t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
				}
			} else if got != redactedSensitiveTextPlaceholder {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, redactedSensitiveTextPlaceholder)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	// BOM-padded sensitive keys are recognized across structured and text paths.
	for _, in := range []string{
		"{\"password\ufeff\":\"SECRETZQ\"}",
		"{\"\ufeffpassword\":\"SECRETZQ\"}",
		"{\"\ufeffapi_key\ufeff\":\"SECRETZQ\"}",
		"{\"user:\ufeffpassword\ufeff\":\"SECRETZQ\"}",
		"{\"\ufeffuser:password\":\"SECRETZQ\"}",
	} {
		for _, ep := range eps {
			got := ep.f(in)
			if strings.Contains(got, "SECRETZQ") {
				t.Errorf("%s(%q) = %q kept SECRETZQ", ep.name, in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	for _, prose := range []string{
		"[INFO] hello\u00a0world",
		"[INFO] hello\ufeffworld",
		"[INFO] hello\vworld",
		"\ufeff[INFO] hello world",
		"[INFO] café — ok",
		"[Help] forgot my\u2000password\u3000again",
		"\"Hello,\u00a0\" she said [laughing].",
		`{"msg":"hello%C2%A0world"`,
		`{"msg":"hello%E2%80%A8world"`,
		`{"msg":"cafe%C3%A9%F0%9F%98%80"`,
		`{"status": 500, "url": "https://example.com/a%20b"`,
		`{"msg":"100% ok %80 %C2 %C2%ZZ %E0%80%80 %"`,
		`{"msg":"100%`,
		`{"msg":"a%C2`,
	} {
		// Prose that starts with "{", "[" or a quote and is no JSON:
		// redactErrorMessage, sanitizeRawText and FormatContentParts keep
		// it, and SmartTruncate fails it closed, as on main.
		for _, ep := range eps[1:] {
			want := prose
			if ep.name == "SmartTruncate" {
				want = unparseableJSONPlaceholder
			}
			if got := ep.f(prose); got != want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, prose, got, want)
			}
		}
		if san, lost := sanitizeRawText(prose); san != prose || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", prose, san, lost, prose)
		}
		if raw, trunc, err := SmartTruncate(prose, -1); err != nil || !trunc {
			t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want trunc=true", prose, raw, trunc, err)
		}
	}
}

// TestNestedJSONTextDepthScaling covers documents nested in JSON string
// literals, and in string leaves, 12 levels deep, with runs of "&" in the
// innermost document and no credential, a sensitive key, a pair in a leaf, or
// a URL leaf that ends in a credential next to them, and such a nest in a leaf
// next to a sensitive key. Writing a redacted document again escapes the "&"
// as \u0026, and each level of JSON string encoding around it escapes the
// escapes inside it once more, so that the text would grow by a factor for
// each level, as main wrote it (see maxRewriteGrowth); the redaction of a
// sensitive key reads the document once more with its keys sorted and every
// level written again (see sortedJSONTextDepth). Every entry point keeps no
// secret, returns no more than a value written again may hold (see
// rewritesTooLong), and changes nothing on a second pass, within a wall-clock
// limit, and the time grows linearly with the number of "&". A list of documents with sensitive keys, which the redaction
// reads once more as written and with its keys sorted (see
// keyRedactionHidesRunOn), scales linearly too.
func TestNestedJSONTextDepthScaling(t *testing.T) {
	runInParallel(t)
	encode := func(v any) string {
		var b strings.Builder
		enc := json.NewEncoder(&b)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(v); err != nil {
			t.Fatal(err)
		}
		return strings.TrimSuffix(b.String(), "\n")
	}
	nest := func(doc string, inLeaves bool) string {
		for range 12 {
			if inLeaves {
				doc = encode(map[string]string{"k": doc})
			} else {
				doc = encode(doc)
			}
		}
		return doc
	}
	// runAll and the subtests build the entry points with the t of the
	// subtest, whose Fatalf the SmartTruncate and FormatContentParts entry
	// points call.
	runAll := func(t *testing.T, in string) func() {
		eps := largeTextEntryPoints(t)
		return func() {
			for _, ep := range eps {
				ep.f(in)
			}
		}
	}
	maxWall := 2 * time.Second
	if raceEnabled {
		maxWall = 20 * time.Second
	}
	type shapeCase struct {
		name string
		mk   func(n int) string
	}
	var shapes []shapeCase
	for _, s := range []struct {
		name  string
		inner func(amp string) map[string]string
	}{
		{"ampersands", func(amp string) map[string]string { return map[string]string{"a": amp} }},
		{"sensitive key", func(amp string) map[string]string { return map[string]string{"a": amp, "password": "SECRET1"} }},
		{"pair in a leaf", func(amp string) map[string]string { return map[string]string{"a": amp, "b": "password=SECRET1"} }},
		{"URL leaf", func(amp string) map[string]string {
			return map[string]string{"pad": amp, "url": "https://h/p?a=1&access_token=SECRET1"}
		}},
	} {
		for _, inLeaves := range []bool{false, true} {
			name := s.name + " in literals"
			if inLeaves {
				name = s.name + " in string leaves"
			}
			shapes = append(shapes, shapeCase{name, func(n int) string {
				return nest(encode(s.inner(strings.Repeat("&", 4*n))), inLeaves)
			}})
		}
	}
	shapes = append(shapes,
		shapeCase{"sensitive key beside literals", func(n int) string {
			return encode(map[string]string{"password": "SECRET1", "k": nest(encode(map[string]string{"a": strings.Repeat("&", 4*n)}), false)})
		}},
		shapeCase{"plain ASCII with sensitive key in literals", func(n int) string {
			return nest(encode(map[string]string{"a": strings.Repeat("a", 4*n), "password": "SECRET1"}), false)
		}},
		shapeCase{"plain ASCII with sensitive key in string leaves", func(n int) string {
			return nest(encode(map[string]string{"a": strings.Repeat("a", 4*n), "password": "SECRET1"}), true)
		}},
	)
	smallN, largeN := scalingUnits()
	for _, sc := range shapes {
		t.Run(sc.name, func(t *testing.T) {
			eps := largeTextEntryPoints(t)
			small, large := sc.mk(smallN), sc.mk(largeN)
			got, again, elapsed := timeEntryPoints(maxWall, large, eps)
			if elapsed > maxWall {
				t.Errorf("%d bytes took %v through the entry points, want at most %v", len(large), elapsed, maxWall)
			}
			for i, ep := range eps {
				for _, tc := range []struct{ in, got, again string }{{large, got[i], again[i]}, {small, ep.f(small), ""}} {
					if tc.again == "" {
						tc.again = ep.f(tc.got)
					}
					if strings.Contains(tc.got, "SECRET") {
						t.Errorf("%s(%d bytes) = %.80q kept the secret", ep.name, len(tc.in), tc.got)
					}
					if limit := maxRewriteGrowth*len(tc.in) + maxRewriteSlack; len(tc.got) > limit {
						t.Errorf("%s(%d bytes) returned %d bytes, want at most %d", ep.name, len(tc.in), len(tc.got), limit)
					}
					if tc.again != tc.got {
						t.Errorf("%s second pass(%.80q) = %.80q, want unchanged", ep.name, tc.got, tc.again)
					}
				}
			}
			if ratio, tSmall, tLarge := linearRatio(64, runAll(t, small), runAll(t, large)); ratio > 64 {
				t.Errorf("entry points scale %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, tSmall, len(small), tLarge, len(large))
			}
		})
	}
	// A document whose sensitive keys the redaction changes is read once
	// more, as written and with its keys sorted (see keyRedactionHidesRunOn),
	// through every entry point that redacts it.
	t.Run("objects with sensitive keys through SmartTruncate", func(t *testing.T) {
		unit := `{"password":"x","note":"y <a&b>","n":1},`
		small := "[" + strings.Repeat(unit, smallN) + `"z"]`
		large := "[" + strings.Repeat(unit, largeN) + `"z"]`
		truncate := func(s string) func() { return func() { _, _, _ = SmartTruncate(s, -1) } }
		if ratio, tSmall, tLarge := linearRatio(64, truncate(small), truncate(large)); ratio > 64 {
			t.Errorf("SmartTruncate scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, tSmall, len(small), tLarge, len(large))
		}
	})
	t.Run("large ampersand leaves 15 levels deep", func(t *testing.T) {
		eps := largeTextEntryPoints(t)
		n := 700_000
		if raceEnabled {
			n = 70_000
		}
		doc := encode(map[string]string{"a": strings.Repeat("&", n), "password": "SECRET1"})
		for range 15 {
			doc = encode(map[string]string{"k": doc})
		}
		for i, ep := range eps {
			got := ep.f(doc)
			again := ep.f(got)
			if strings.Contains(got, "SECRET") {
				t.Errorf("%s(%d bytes) = %.80q kept the secret", ep.name, len(doc), got)
			}
			if i >= 2 && got != redactedSensitiveTextPlaceholder {
				t.Errorf("%s(%d bytes) = %.80q, want %q", ep.name, len(doc), got, redactedSensitiveTextPlaceholder)
			}
			if again != got {
				t.Errorf("%s second pass(%.80q) = %.80q, want unchanged", ep.name, got, again)
			}
		}
		amp15 := encode(map[string]string{"a": strings.Repeat("&", 4*largeN), "password": "SECRET1"})
		for range 15 {
			amp15 = encode(map[string]string{"k": amp15})
		}
		plain1 := encode(map[string]string{"a": strings.Repeat("a", len(amp15)), "password": "SECRET1"})
		if ratio, t1, t15 := linearRatio(64, runAll(t, plain1), runAll(t, amp15)); ratio > 64 {
			t.Errorf("15-level ampersand leaves vs equal-length 1-level document scale %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, t1, len(plain1), t15, len(amp15))
		}
	})
	t.Run("large plain ASCII 12 levels deep", func(t *testing.T) {
		eps := largeTextEntryPoints(t)
		base := encode(map[string]string{"a": strings.Repeat("a", 8*largeN), "password": "SECRET1"})
		for _, inLeaves := range []bool{false, true} {
			doc := nest(base, inLeaves)
			for _, ep := range eps {
				got := ep.f(doc)
				again := ep.f(got)
				if strings.Contains(got, "SECRET") {
					t.Errorf("%s(%d bytes, inLeaves=%v) = %.80q kept the secret", ep.name, len(doc), inLeaves, got)
				}
				if again != got {
					t.Errorf("%s second pass(%.80q) = %.80q, want unchanged", ep.name, got, again)
				}
			}
			plain1 := encode(map[string]string{"a": strings.Repeat("a", len(doc)), "password": "SECRET1"})
			if ratio, t1, t12 := linearRatio(64, runAll(t, plain1), runAll(t, doc)); ratio > 64 {
				t.Errorf("12-level plain ASCII (inLeaves=%v) vs equal-length 1-level document scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", inLeaves, ratio, t1, len(plain1), t12, len(doc))
			}
		}
	})
	t.Run("large plain ASCII 15 levels all-levels key-redacted", func(t *testing.T) {
		eps := largeTextEntryPoints(t)
		mkAllLevels := func(levels, n int) string {
			doc := encode(map[string]string{"a": strings.Repeat("a", n), "password": "SECRET1"})
			for range levels - 1 {
				doc = encode(map[string]any{"z": doc, "password": "SECRET1"})
			}
			return doc
		}
		n := 256_000
		if raceEnabled {
			n = 128_000
		}
		doc15 := mkAllLevels(15, n)
		plain1 := mkAllLevels(1, len(doc15))
		sanFn := func(in string) func() { return func() { _, _ = sanitizeRawText(in) } }
		if ratio, t1, t15 := linearRatio(64, sanFn(plain1), sanFn(doc15)); ratio > 64 {
			t.Errorf("sanitizeRawText 15-level vs equal-length 1-level document scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, t1, len(plain1), t15, len(doc15))
		}
		doc15Small := mkAllLevels(15, n/16)
		if ratio, ts, tl := linearRatio(64, sanFn(doc15Small), sanFn(doc15)); ratio > 64 {
			t.Errorf("sanitizeRawText 15-level growing payload scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, ts, len(doc15Small), tl, len(doc15))
		}
		// Keep keyRedactionHidesRunOn outer-only: nesting a 49k-int leaf 12
		// levels deep repeats node decoding only once for redaction and once
		// at depth 0, staying well below 64x of equal-length 1-level input
		// (measures ~22-23x without -race, leaving headroom while still
		// catching per-level exponential blowup).
		nums := 49_000
		if raceEnabled {
			nums = 20_000
		}
		arrDoc12 := fmt.Sprintf(`{"a":%s,"password":"SECRET1"}`, encode(make([]int, nums)))
		for range 11 {
			arrDoc12 = fmt.Sprintf(`{"z":%s,"password":"SECRET1"}`, encode(arrDoc12))
		}
		arrPlain1 := mkAllLevels(1, len(arrDoc12))
		if ratio, t1, t12 := linearRatio(64, sanFn(arrPlain1), sanFn(arrDoc12)); ratio > 64 {
			t.Errorf("sanitizeRawText 12-level array-leaf vs equal-length 1-level document scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, t1, len(arrPlain1), t12, len(arrDoc12))
		}
		san, _ := sanitizeRawText(doc15)
		if strings.Contains(san, "SECRET") {
			t.Errorf("sanitizeRawText(%d bytes) = %.80q kept the secret", len(doc15), san)
		}
		if san2, lost2 := sanitizeRawText(san); san2 != san || lost2 {
			t.Errorf("sanitizeRawText(%d bytes) second pass changed=%v lost=%v", len(doc15), san2 != san, lost2)
		}
		smallDoc15 := mkAllLevels(15, 8*largeN)
		smallPlain1 := mkAllLevels(1, len(smallDoc15))
		for _, ep := range eps {
			got := ep.f(smallDoc15)
			again := ep.f(got)
			if strings.Contains(got, "SECRET") {
				t.Errorf("%s(%d bytes) = %.80q kept the secret", ep.name, len(smallDoc15), got)
			}
			if again != got {
				t.Errorf("%s second pass(%.80q) = %.80q, want unchanged", ep.name, got, again)
			}
		}
		if ratio, t1, t15 := linearRatio(64, runAll(t, smallPlain1), runAll(t, smallDoc15)); ratio > 64 {
			t.Errorf("15-level all-levels key-redacted vs equal-length 1-level document scales %.1fx (%v for %d bytes, %v for %d bytes), want at most 64x", ratio, t1, len(smallPlain1), t15, len(smallDoc15))
		}
	})
}

// TestEscapedRunOnAcrossJSONStringsFailsClosed covers JSON documents whose
// strings the expressions of main read as one credential only once the
// escapes of the text are decoded: a quote or a header delimiter spelled as
// \u0027, \u0022, \u003a, \u003d, %27 or \", which a later string closes or
// runs on into, as the quote after Bearer in ["Bearer '","SECRET","\u0027"],
// a key that is sensitive only decoded, as the temp%3aa of
// {"temp%3aa":"SECRET"}, and a quote spelled %22 that the closing quote of
// its string closes.
// scrubErrorText on main failed the text of such a document closed as a
// whole, so no entry point that scrubs it as text there may redact it string
// by string and keep the secret in another string (see
// expressionsKeepStringsApart, which reads the text with its escapes
// decoded), and a document whose redaction of a compound key would take the
// header that runs on fails closed too (see keyRedactionHidesRunOn).
// redactErrorMessage scrubs the strings of a document that is not in a JSON
// string literal or a string one by one, as main did, and keeps the secret
// where main kept it.
func TestEscapedRunOnAcrossJSONStringsFailsClosed(t *testing.T) {
	runInParallel(t)
	quote := func(doc string) string {
		b, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	type escapedCase struct {
		in string
		// keptByLeaves reports that redactErrorMessage scrubs the string
		// that holds the secret on its own, as it did on main, and keeps it.
		keptByLeaves bool
	}
	var cases []escapedCase
	for _, doc := range []string{
		`{"zz_secret":"Authorization\u003a\n","b":"RVIEWz9SECRET"}`,
		`["access_token\u003d'","SECRET","'"]`,
		`["Bearer '","SECRET","\u0027"]`,
		`{"a":"Bearer '","b":"SECRET%27"}`,
		`["password: \u0027token:[REDACTED] ","SECRET","'"]`,
		// A header whose decoded value runs on into a later string that
		// it holds as a whole, and a key that main read as sensitive only
		// with its escapes decoded, whose value no leaf redacts.
		`["Authorization%3A token:[REDACTED]\t" , "SECRET"]`,
		`["Authorization%3A x\t","SECRET"]`,
		`{"temp%3aa":"SECRET"}`,
		`{"temp%3aa\u003D[REDACTED] ":"SECRET","x_token":""}`,
		`{"temp%3aa\u003D\r\n[REDAC\u0054ED] ":" SECRET\\u0027","page_token":"%22"}`,
		// A document in a JSON string literal whose redaction fails
		// closed fails the literal closed as a whole.
		`{"db_password": "api_key%3d\"token:[REDACTED]%20" , "page_token": "x" , "c": "SECRET"}`,
		"{\"x_token\": \"see Authorization\\u003A \\r\\ntoke\\u006E\\u003A[REDACTED]\\t\",\n  \"msg\": \"SECRET\",\n  \"note\": \"%0a\"}",
	} {
		cases = append(cases,
			escapedCase{doc, true},
			escapedCase{quote(doc), false},
			escapedCase{`{"nested":` + quote(doc) + `}`, false},
		)
	}
	for _, doc := range []string{
		`["access_token=\u0027","RVIEWz9SECRET","\u0027"]`,
		`{"a":"token=\u0027","b":"RVIEWz9SECRET","c":"\u0027"}`,
		`{"a":"token\u003d\u0027","b":"RVIEWz9SECRET","c":"\u0027"}`,
		`["Bearer \"","RVIEWz9SECRET","\""]`,
		`{"a":"Bearer \"x","b":"RVIEWz9SECRET","c":"\""}`,
		`["Bearer \u0027","RVIEWz9SECRET","\u0027"]`,
		`["Bearer \u0022","RVIEWz9SECRET","\u0022"]`,
		`{"a":"Bearer\u0020\u0022","b":"RVIEWz9SECRET","c":"\u0022"}`,
		`["Bearer \u0027","SECRET","'"]`,
		`["\u0042earer '","SECRET","'"]`,
		`["Bearer '","SECRET","%27"]`,
		`{"a":"Bearer '","b":"SECRET%27","password":"x"}`,
		`{"a":"x-api-key\u003a","b":"SECRET"}`,
		`{"db_password":"Bearer\u0020\u0027","b":"SECRET","c":"'"}`,
		// A quote that %22 spells opens a value that the closing quote of
		// its string closes, which main read as one credential.
		`{"m":"access_token=%22a;key=SECRET;token=c"}`,
		`{"url":"https://h/p#a=https%3A%2F%2Fh%2Fcb%3Faccess_token%3DSECRET;access_token=%22SECRET;key%3DSECRET;Bearer+x+token:token=SECRET","n":"x"}`,
	} {
		cases = append(cases, escapedCase{doc, true})
	}
	cases = append(cases, escapedCase{`{"payload":"[\"Bearer \\\"\", \"RVIEWz9SECRET\", \"\\\"\"]"}`, false})
	eps := largeTextEntryPoints(t)
	for _, c := range cases {
		for _, ep := range eps {
			got := ep.f(c.in)
			if strings.Contains(got, "SECRET") && (!c.keptByLeaves || ep.name != "redactErrorMessage") {
				t.Errorf("%s(%q) = %q kept the secret", ep.name, c.in, got)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	// A document whose escapes, decoded, read as no credential across its
	// strings is kept: HTML escapes, and a quote that no later string
	// closes, as on main.
	for _, in := range []string{
		`{"a":"x \u003c y \u0026 z","b":"w"}`,
		`["it\u0027s","fine"]`,
	} {
		for _, ep := range eps[2:] {
			if got := ep.f(in); got != in {
				t.Errorf("%s(%q) = %q, want it unchanged", ep.name, in, got)
			}
		}
	}
	// Read with their escapes decoded, the expressions of main may remove no
	// first or last byte of a value that they keep as written, but for those
	// of one value as a whole, or around a placeholder.
	for _, tc := range []struct {
		doc  string
		want bool
	}{
		{`["Bearer '","SECRET","\u0027"]`, false},
		{`["Bearer '","SECRET","%27"]`, false},
		{`["Bearer \"","SECRET","\""]`, false},
		{`{"a":"token\u003d\u0027","b":"x","c":"\u0027"}`, false},
		{`{"zz_secret":"Authorization\u003a\n","b":"x"}`, false},
		{`["db_password\u003d'","x","'"]`, false},
		{`{"a":"Bearer \u0022abc"}`, false},
		{`["Authorization%3A x\t","SECRET"]`, false},
		{`{"temp%3aa":"SECRET"}`, false},
		{`{"m":"access_token=%22a;key=b;token=c"}`, false},
		{`{"a":"x \u003c y \u0026 z","b":"w"}`, true},
		{`["it\u0027s","fine"]`, true},
		{`{"msg":"say \"hi\" \u0026access_token=[REDACTED]"}`, true},
		{`{"url":"https://example.com/p?a=1\u0026access_token=%5BREDACTED%5D"}`, true},
		{`{"pass\u0077ord":"x"}`, true},
		{`{"a":"Bearer\u0020tok","b":"x"}`, true},
		{`{"temp:a":"x\u0026y","b":"z"}`, true},
		{`{"password":"x\u0026y","b":"z"}`, true},
	} {
		if got := expressionsKeepStringsApart(tc.doc); got != tc.want {
			t.Errorf("expressionsKeepStringsApart(%q) = %v, want %v", tc.doc, got, tc.want)
		}
	}
}

// TestLineStartPathPrefixedKeysFollowMain covers a key name on a line of its
// own with words before the key that the expressions of main read as part of
// a value: a path that holds a backslash, which no key name of main holds,
// before a key of main, a compound key or a state key, and a path before a
// key that main did not have, as secret_key, signing_key, api.key or
// db_password. main redacted such a name as the value of the pair on the line
// above, so it is redacted here too, with the value of the pair that it
// starts, rather than kept as a nested key or failed closed, and the backstop
// exempts no word before the key of such a name (see redactedKeyBytesWith).
// So is a stacked bare temp: key, as temp:SECRET:x, a segment of which may be
// a value. A path before a key of main, as /etc/ssl/password, stays a nested
// key, as on main.
func TestLineStartPathPrefixedKeysFollowMain(t *testing.T) {
	runInParallel(t)
	eps := largeTextEntryPoints(t)
	// Each text is the pair of a key on the line above whose value runs on
	// over the name, which is redacted with it rather than failed closed.
	check := func(in string) {
		t.Helper()
		want := in[:strings.IndexByte(in, ':')] + ":" + redactedValue
		for _, ep := range eps {
			got := ep.f(in)
			if got != want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
	}
	for _, in := range []string{
		"password:\n  \\SECRET/token:\"a b\"",
		"password:\n  C:\\SECRET/token:\"a b\"",
		"password:\n  /SECRET\\x/token:\"a b\"",
		"password:\n  \\SECRET/token:\"[REDACTED] \"",
		"password:\n  /SECRET/secret_key: [REDACTED]",
		"password:\n  /SECRET/signing_key: [REDACTED]",
		"password:\n  C:\\SECRET\\secret_key: [REDACTED]",
	} {
		check(in)
	}
	for _, head := range []string{"password:\n  ", "credentials:\n  ", "password:\r\n\t"} {
		for _, prefix := range []string{`\SECRET/`, `C:\SECRET\`, `C:\SECRET/`, `/SECRET\x/`, `\\SECRET/`} {
			for _, key := range []string{"token", "password", "client_secret", "secret_key", "signing_key", "api.key", "db_password", "my_api_key", "temp:x"} {
				for _, value := range []string{": [REDACTED]", ":[REDACTED]", `:"a b"`, `:"[REDACTED] "`, ": v", `='a b'`} {
					check(head + prefix + key + value)
				}
			}
		}
		for _, prefix := range []string{"/SECRET/", "/a/SECRET/", "C:/SECRET/"} {
			for _, key := range []string{"secret_key", "signing_key", "api.key", "private.key", "db_password", "my_api_key"} {
				for _, value := range []string{": [REDACTED]", `:"a b"`, ": v"} {
					check(head + prefix + key + value)
				}
			}
		}
	}
	// A stacked bare temp: key on a line of its own, a segment of which may
	// be a value, is redacted with the value of the pair above, as main
	// redacted it, and with its own value.
	for _, head := range []string{"password:\n  ", "credentials:\n  ", "password:\r\n\t", "password: \n", "password:\n"} {
		for _, key := range []string{"temp:SECRET:x", "TEMP:SECRET:x", "user:temp:SECRET:x", "app:temp:SECRET:x", "temp:SECRET:x:y", "temp:a:hunter2"} {
			for _, value := range []string{": bar", ":bar", ": [REDACTED]"} {
				check(head + key + value)
			}
		}
	}
	for _, tc := range []struct{ in, want string }{
		{"credentials:\n  /etc/ssl/password: v", "credentials:\n  /etc/ssl/password:[REDACTED]"},
		{"credentials:\n  C:/ssl/token: v", "credentials:\n  C:/ssl/token:[REDACTED]"},
		{"password:\n  \"temp:SECRET:x\": bar", "password:\"[REDACTED]\""},
		{"password:\n  'temp:SECRET:x': bar", "password:'[REDACTED]'"},
		{"password:\n  \"user:temp:SECRET:x\": bar", "password:\"[REDACTED]\""},
	} {
		for _, ep := range eps {
			if got := ep.f(tc.in); got != tc.want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, tc.in, got, tc.want)
			}
		}
	}
}

// TestEscapedReadingsMatchMain checks that the readings that
// expressionsKeepStringsApart reads a document in are the texts that
// scrubErrorText decodes it into before it fails it closed (see
// escapedReadings): each mapped decoder returns what
// canonicalizeCredentialMarker, unescapeBackslashControls and
// unescapeCredentialStep return, with one offset of the input for each byte
// of the result and for its end, in order, so that what the expressions
// remove from a reading maps back to the bytes that it is decoded from.
func TestEscapedReadingsMatchMain(t *testing.T) {
	runInParallel(t)
	frags := []string{
		`\u0027`, `\u0022`, `\u003a`, `\u003D`, `\U0041`, `\u00e9`, `\ud83d`, `\u004`, `\x41`, `\xZZ`,
		`\u0130`, `\u017f`, `\u212a`, `\u00a0`, `\u2000`, `\ufeff`, `\u2028`, `\xA0`, `%A0`,
		`%27`, `%22`, `%3a`, `%2F`, `%5BREDACTED%5D`, `%4`, `%`, `%255C`,
		`\"`, `\'`, `\\`, `\n`, `\r`, `\t`, `\/`, `\b`, `\f`, `\`, `\\u0041`, `\\\"`,
		"a", " ", `"`, "'", "\u00e9", "\ufeff", "\xff", "token=", "Bearer ", "[REDACTED]",
	}
	inputs := []string{"", `["Bearer '","SECRET","\u0027"]`, `{"m":"access_token=%22a;key=b;token=c"}`, `["Authorization%3A x\t","SECRET"]`}
	state := uint64(2250)
	next := func(n int) int {
		state = state*6364136223846793005 + 1442695040888963407
		return int(state>>33) % n
	}
	for range 3000 {
		var b strings.Builder
		for n := 1 + next(8); n > 0; n-- {
			b.WriteString(frags[next(len(frags))])
		}
		inputs = append(inputs, b.String())
	}
	checkPos := func(name, in, text string, pos []int) {
		t.Helper()
		if len(pos) != len(text)+1 || pos[len(text)] != len(in) || pos[0] < 0 || !slices.IsSorted(pos) {
			t.Errorf("%s(%q) = %q maps to %v, want an offset of the input in order for each byte and the end", name, in, text, pos)
		}
	}
	for _, in := range inputs {
		if text, pos := canonicalizeMapped(in); text != canonicalizeCredentialMarker(in) {
			t.Errorf("canonicalizeMapped(%q) = %q, want %q", in, text, canonicalizeCredentialMarker(in))
		} else {
			checkPos("canonicalizeMapped", in, text, pos)
		}
		if text, pos := unescapeBackslashControlsMapped(in); text != unescapeBackslashControls(in) {
			t.Errorf("unescapeBackslashControlsMapped(%q) = %q, want %q", in, text, unescapeBackslashControls(in))
		} else {
			checkPos("unescapeBackslashControlsMapped", in, text, pos)
		}
		if r := unescapeCredentialStepMapped(in); r.text != unescapeCredentialStep(in) {
			t.Errorf("unescapeCredentialStepMapped(%q) = %q, want %q", in, r.text, unescapeCredentialStep(in))
		} else {
			checkPos("unescapeCredentialStepMapped", in, r.text, r.pos)
		}
		seen := map[string]bool{in: true}
		for _, r := range escapedReadings(in) {
			if seen[r.text] {
				t.Errorf("escapedReadings(%q) holds %q twice, or the input itself", in, r.text)
			}
			seen[r.text] = true
			checkPos("escapedReadings", in, r.text, r.pos)
		}
	}
}

// TestOuterRunOnBudgetIdempotencyAndCandidateRecheck covers three invariants:
// (1) outer quote-led JSON string literals and single-leaf JSON wrappers run
// keyRedactionHidesRunOn at the outermost level so mixed %22/\" escapes cannot
// leak sibling secrets; (2) maxSanitizeNodes truncation in redactJSONValueInner
// shrinks containers strictly so Pass 2 is idempotent with contentLost=false
// across the 49,997..50,001 boundary on maps and arrays across all five entry
// points; and (3) quote-led literal candidate re-checks run at the same depth
// with a fresh pass budget.
func TestOuterRunOnBudgetIdempotencyAndCandidateRecheck(t *testing.T) {
	runInParallel(t)

	t.Run("outer mixed escape run-on across literals and single-leaf wrappers", func(t *testing.T) {
		eps := largeTextEntryPoints(t)
		marshal := func(v any) string {
			b, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		}
		decoded := `{"DB_PASSWORD":"Bearer%20%22x\" tail","b":"SECRETZQ"}`
		in1 := marshal(decoded)
		in2 := marshal(in1)
		in3 := marshal(in2)
		wrapped1 := marshal(map[string]any{"k": in1})
		wrapped2 := marshal(map[string]any{"k": wrapped1})
		wrappedIn2 := marshal(map[string]any{"k": in2})
		decodedWithClose := `{"DB_PASSWORD":"Bearer%20%22x\" tail","b":"SECRETZQ","c":"%22"}`
		in2WithClose := marshal(marshal(decodedWithClose))

		for _, tc := range []struct {
			in           string
			keptByLeaves bool
		}{
			{in2, false},
			{in3, false},
			{wrapped1, true},
			{wrapped2, false},
			{wrappedIn2, false},
			{in2 + " ok", false},
			{in2WithClose + " ok", false},
		} {
			for _, ep := range eps {
				got := ep.f(tc.in)
				if strings.Contains(got, "SECRETZQ") && (!tc.keptByLeaves || ep.name != "redactErrorMessage") {
					t.Errorf("%s(%q) = %q kept SECRETZQ", ep.name, tc.in, got)
				}
				if again := ep.f(got); again != got {
					t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
				}
			}
			san, _ := sanitizeRawText(tc.in)
			if _, lost2 := sanitizeRawText(san); lost2 {
				t.Errorf("sanitizeRawText second pass(%q) reported contentLost", san)
			}
		}
	})

	t.Run("node budget boundary idempotency and contentLost stability", func(t *testing.T) {
		mkMap := func(n int) string {
			var b strings.Builder
			b.WriteByte('{')
			for i := range n {
				if i > 0 {
					b.WriteByte(',')
				}
				fmt.Fprintf(&b, `"k%05d":"v"`, i)
			}
			b.WriteByte('}')
			return b.String()
		}
		mkArr := func(n int) string {
			var b strings.Builder
			b.WriteByte('[')
			for i := range n {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(`"v"`)
			}
			b.WriteByte(']')
			return b.String()
		}

		type boundaryCase struct {
			name string
			n    int
			mk   func(int) string
		}
		var cases []boundaryCase
		for _, n := range []int{49_997, 49_998, 49_999, 50_000, 50_001} {
			cases = append(cases,
				boundaryCase{"map", n, mkMap},
				boundaryCase{"arr", n, mkArr},
			)
		}
		errs := make([][]string, len(cases)+1)
		sem := make(chan struct{}, min(6, max(1, runtime.GOMAXPROCS(0))))
		var wg sync.WaitGroup
		for idx, bc := range cases {
			wg.Go(func() {
				sem <- struct{}{}
				defer func() { <-sem }()
				in := bc.mk(bc.n)

				rem1 := redactErrorMessage(in)
				if rem2 := redactErrorMessage(rem1); rem2 != rem1 {
					errs[idx] = append(errs[idx], fmt.Sprintf("redactErrorMessage(%s, n=%d) not idempotent on second pass", bc.name, bc.n))
				}

				san1, _ := sanitizeRawText(in)
				if san2, lost2 := sanitizeRawText(san1); san2 != san1 || lost2 {
					errs[idx] = append(errs[idx], fmt.Sprintf("sanitizeRawText(%s, n=%d) second pass changed=%v lost=%v", bc.name, bc.n, san2 != san1, lost2))
				}
				if once1, onceLost1, _ := sanitizeRawTextOnce(in, false); onceLost1 {
					if once2, onceLost2, onceSettled2 := sanitizeRawTextOnce(once1, true); once2 != once1 || onceLost2 || !onceSettled2 {
						errs[idx] = append(errs[idx], fmt.Sprintf("sanitizeRawTextOnce(%s, n=%d) not settled in one pass: changed=%v lost=%v settled=%v", bc.name, bc.n, once2 != once1, onceLost2, onceSettled2))
					}
				}

				raw1, _, err := SmartTruncate(in, -1)
				if err != nil {
					errs[idx] = append(errs[idx], fmt.Sprintf("SmartTruncate(%s, n=%d) err: %v", bc.name, bc.n, err))
					return
				}
				var st1 string
				if err := json.Unmarshal(raw1, &st1); err != nil {
					errs[idx] = append(errs[idx], fmt.Sprintf("SmartTruncate(%s, n=%d) unmarshal: %v", bc.name, bc.n, err))
					return
				}
				raw2, lost2, err := SmartTruncate(st1, -1)
				if err != nil {
					errs[idx] = append(errs[idx], fmt.Sprintf("SmartTruncate second pass(%s, n=%d) err: %v", bc.name, bc.n, err))
					return
				}
				var st2 string
				if err := json.Unmarshal(raw2, &st2); err != nil {
					errs[idx] = append(errs[idx], fmt.Sprintf("SmartTruncate second pass(%s, n=%d) unmarshal: %v", bc.name, bc.n, err))
					return
				}
				if st2 != st1 || lost2 {
					errs[idx] = append(errs[idx], fmt.Sprintf("SmartTruncate(%s, n=%d) second pass changed=%v lost=%v", bc.name, bc.n, st2 != st1, lost2))
				}

				parts1 := FormatContentParts(&genai.Content{Parts: []*genai.Part{{Text: in}}}, -1)
				if len(parts1) != 1 {
					errs[idx] = append(errs[idx], fmt.Sprintf("FormatContentParts(%s, n=%d) = %d parts, want 1", bc.name, bc.n, len(parts1)))
					return
				}
				fcp1, _ := parts1[0]["text"].(string)
				parts2 := FormatContentParts(&genai.Content{Parts: []*genai.Part{{Text: fcp1}}}, -1)
				if len(parts2) != 1 {
					errs[idx] = append(errs[idx], fmt.Sprintf("FormatContentParts second pass(%s, n=%d) = %d parts, want 1", bc.name, bc.n, len(parts2)))
					return
				}
				if fcp2, _ := parts2[0]["text"].(string); fcp2 != fcp1 {
					errs[idx] = append(errs[idx], fmt.Sprintf("FormatContentParts(%s, n=%d) not idempotent on second pass", bc.name, bc.n))
				}
			})
		}
		nestedIdx := len(cases)
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			// Nested JSON-encoded boundary map for redactErrorMessage (outer unmarshal + traverse takes 4 nodes).
			nestedMap := fmt.Sprintf(`{"s":%q}`, mkMap(49_997))
			got := redactErrorMessage(nestedMap)
			if again := redactErrorMessage(got); again != got {
				errs[nestedIdx] = append(errs[nestedIdx], "redactErrorMessage(nested 49997-key map) not idempotent on second pass")
			}
		})
		wg.Wait()
		for _, batch := range errs {
			for _, msg := range batch {
				t.Error(msg)
			}
		}
	})

	t.Run("candidate recheck preserves same depth and fresh pass budget", func(t *testing.T) {
		// 44 array wrappers put deepLit at depth=45, decoded at 46, map at 47, numbers at 48 (< maxSanitizeDepth 50).
		deepLit := `"\u007b\u0022a\u0022:1,\u0022a\u0022:2\u007d" ok`
		var deep any = deepLit
		for range 44 {
			deep = []any{deep}
		}
		deepBytes, err := json.Marshal(deep)
		if err != nil {
			t.Fatal(err)
		}
		deepOut, deepLost := sanitizeRawText(string(deepBytes))
		if deepLost || strings.Contains(deepOut, maxDepthPlaceholder) {
			t.Errorf("sanitizeRawText(depth-46 literal+suffix) lost=%v hasMaxDepth=%v", deepLost, strings.Contains(deepOut, maxDepthPlaceholder))
		}
		if again, againLost := sanitizeRawText(deepOut); again != deepOut || againLost {
			t.Errorf("sanitizeRawText(depth-46 literal+suffix) second pass changed=%v lost=%v", again != deepOut, againLost)
		}

		// 26,000-node array inside a quote-led literal + suffix (>52k nodes on Pass 1, >100k if budget shared).
		var arr strings.Builder
		arr.WriteByte('[')
		for i := range 26_000 {
			if i > 0 {
				arr.WriteByte(',')
			}
			arr.WriteByte('0')
		}
		arr.WriteByte(']')
		bigLit := `"\u007b\u0022arr\u0022:` + arr.String() + `,\u0022a\u0022:1,\u0022a\u0022:2\u007d" ok`
		bigOut, bigLost := sanitizeRawText(bigLit)
		if bigLost || strings.Contains(bigOut, sanitizeBudgetPlaceholder) {
			t.Errorf("sanitizeRawText(26k-arr literal+suffix) lost=%v hasBudget=%v", bigLost, strings.Contains(bigOut, sanitizeBudgetPlaceholder))
		}
		if again, againLost := sanitizeRawText(bigOut); again != bigOut || againLost {
			t.Errorf("sanitizeRawText(26k-arr literal+suffix) second pass changed=%v lost=%v", again != bigOut, againLost)
		}
	})
}

// TestOverlappingSensitiveURIPathSegments verifies that when a URI path segment
// consumed as the value of an earlier sensitive segment is itself a sensitive
// path segment name (such as /secret_key/password/SECRET, /signing_key/token/SECRET,
// or /user:temp:a/password/SECRET), redactNext stays armed so the following
// secret segment is still redacted across both direct sanitizeURI and the
// FileData -> content_parts[].uri route.
func TestOverlappingSensitiveURIPathSegments(t *testing.T) {
	runInParallel(t)
	cases := []struct {
		name      string
		uri       string
		want      string
		keepToken string
	}{
		{
			name: "secret_key then password",
			uri:  "https://h/secret_key/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "signing_key then token",
			uri:  "https://h/signing_key/token/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "stacked state prefix then password",
			uri:  "https://h/user:temp:a/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "three adjacent sensitive segments",
			uri:  "https://h/secret_key/signing_key/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "percent-encoded adjacent sensitive segments",
			uri:  "https://h/secret%5Fkey/pa%73sword/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "four state-prefixed sensitive segments",
			uri:  "https://h/user:api_key/app:secret_key/temp:x/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name:      "non-sensitive value stops redactNext",
			uri:       "https://h/secret_key/password/val/SAFE_TAIL",
			want:      "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D/SAFE_TAIL",
			keepToken: "SAFE_TAIL",
		},
		{
			name:      "legacy pair boundary preserves trailing segment",
			uri:       "https://example.com/v1/api_key/secret/items",
			want:      "https://example.com/v1/%5BREDACTED%5D/%5BREDACTED%5D/items",
			keepToken: "items",
		},
		{
			name: "leading escaped NBSP api_key then password",
			uri:  "https://h/%5Cu00a0api_key/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "trailing escaped NBSP api_key then password",
			uri:  "https://h/api_key%5Cu00a0/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "kelvin-folded api_key then password",
			uri:  "https://h/api_%5Cu212aey/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "ideographic space api_key then password",
			uri:  "https://h/api_key%5Cu3000/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "zwsp api_key then password",
			uri:  "https://h/api_key%5Cu200b/password/SECRET123",
			want: "https://h/api_key%5Cu200b/%5BREDACTED%5D/%5BREDACTED%5D",
		},
		{
			name: "bom api_key then password",
			uri:  "https://h/%5Cufeffapi_key/password/SECRET123",
			want: "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, changed := sanitizeURI(tc.uri)
			if !changed || got != tc.want {
				t.Errorf("sanitizeURI(%q) = (%q, %v), want (%q, true)", tc.uri, got, changed, tc.want)
			}
			if tc.keepToken == "" && strings.Contains(got, "SECRET123") {
				t.Errorf("sanitizeURI(%q) = %q exposed SECRET123", tc.uri, got)
			}
			if tc.keepToken != "" && !strings.Contains(got, tc.keepToken) {
				t.Errorf("sanitizeURI(%q) = %q dropped %q", tc.uri, got, tc.keepToken)
			}
			if again, againChanged := sanitizeURI(got); againChanged || again != got {
				t.Errorf("sanitizeURI second pass(%q) = (%q, %v), want (%q, false)", got, again, againChanged, got)
			}

			parts, lost := formatContentParts(&genai.Content{
				Parts: []*genai.Part{{
					FileData: &genai.FileData{MIMEType: "application/octet-stream", FileURI: tc.uri},
				}},
			}, -1, true)
			if !lost || len(parts) != 1 {
				t.Fatalf("formatContentParts(%q) = (%v, %v), want 1 part with lost=true", tc.uri, parts, lost)
			}
			gotURI, _ := parts[0]["uri"].(string)
			if gotURI != tc.want {
				t.Errorf("file-part content_parts[0].uri = %q, want %q", gotURI, tc.want)
			}
			if tc.keepToken == "" && strings.Contains(gotURI, "SECRET123") {
				t.Errorf("file-part content_parts[0].uri = %q exposed SECRET123", gotURI)
			}
			exported := FormatContentParts(&genai.Content{Parts: []*genai.Part{{
				FileData: &genai.FileData{MIMEType: "application/octet-stream", FileURI: tc.uri},
			}}}, -1)
			if len(exported) != 1 || exported[0]["uri"] != tc.want {
				t.Errorf("FormatContentParts(FileData %q) = %v, want uri %q", tc.uri, exported, tc.want)
			}
		})
	}
}

// TestURIPathSegmentsMatchKeyNamesOnly pins what the comment of
// Config.DisableSensitiveKeyRedaction says of URIs. A path segment redacts
// the segment after it only when it is a credential name itself, whatever its
// case and "_" or "-", or a "temp:" key, alone or behind "user:" or "app:"
// prefixes. The "." and compound-name matching of map keys and text applies
// to the keys of query, fragment and ";" parameter pairs, not to path
// segments, which stay as on main, through sanitizeURI and the FileData ->
// content_parts[].uri route.
func TestURIPathSegmentsMatchKeyNamesOnly(t *testing.T) {
	runInParallel(t)
	const r = "%5BREDACTED%5D"
	for _, tc := range []struct{ uri, want string }{
		{"https://h/api_key/abc123", "https://h/" + r + "/" + r},
		{"https://h/api-key/abc123", "https://h/" + r + "/" + r},
		{"https://h/apiKey/abc123", "https://h/" + r + "/" + r},
		{"https://h/temp:x/abc123", "https://h/" + r + "/" + r},
		{"https://h/user:api_key/abc123", "https://h/" + r + "/" + r},
		{"https://h/app:user:temp:x/abc123", "https://h/" + r + "/" + r},

		{"https://h/api.key/abc123", "https://h/api.key/abc123"},
		{"https://h/DB_PASSWORD/abc123", "https://h/DB_PASSWORD/abc123"},
		{"https://h/dbPassword/abc123", "https://h/dbPassword/abc123"},
		{"https://h/spring.datasource.password/abc123", "https://h/spring.datasource.password/abc123"},
		{"https://h/certs/private.key/download", "https://h/certs/private.key/download"},
		{"https://h/account/reset-password/confirm", "https://h/account/reset-password/confirm"},

		{"https://h/p?api.key=abc123", "https://h/p?api.key=" + r},
		{"https://h/p?DB_PASSWORD=abc123", "https://h/p?DB_PASSWORD=" + r},
		{"https://h/p#dbPassword=abc123", "https://h/p#dbPassword=" + r},
		{"https://h/p;api.key=abc123", "https://h/p%3Bapi.key=" + r},
	} {
		if got, changed := sanitizeURI(tc.uri); got != tc.want || changed != (tc.want != tc.uri) {
			t.Errorf("sanitizeURI(%q) = (%q, %v), want (%q, %v)", tc.uri, got, changed, tc.want, tc.want != tc.uri)
		}
		exported := FormatContentParts(&genai.Content{Parts: []*genai.Part{{
			FileData: &genai.FileData{MIMEType: "application/octet-stream", FileURI: tc.uri},
		}}}, -1)
		if len(exported) != 1 || exported[0]["uri"] != tc.want {
			t.Errorf("FormatContentParts(FileData %q) = %v, want uri %q", tc.uri, exported, tc.want)
		}
	}
}

// TestURIEncodedPaddingRedactsAcrossComponentsAndEntryPoints covers multibyte
// percent-encoded BOM (%EF%BB%BF) and no-break space (%C2%A0) padding on
// query keys (from url.Values.Encode), path segments, matrix key=val segments,
// and fragment keys (from url.URL.String), plus invalid-escape fallback and
// legacyURIKeys vs scannerURIKeys parity across SmartTruncate, sanitizeRawText,
// redactErrorMessage and FormatContentParts.
func TestURIEncodedPaddingRedactsAcrossComponentsAndEntryPoints(t *testing.T) {
	runInParallel(t)
	const (
		secret = "S3CR3T_URI_PAD_99"
		r      = "%5BREDACTED%5D"
	)

	queryBOM := (&url.URL{
		Scheme:   "https",
		Host:     "example.com",
		Path:     "/api",
		RawQuery: url.Values{"\ufeffDB_PASSWORD": {secret}}.Encode() + "&status=ready",
	}).String()
	queryNBSPAndKey := (&url.URL{
		Scheme:   "https",
		Host:     "example.com",
		Path:     "/api",
		RawQuery: url.Values{"DB_PASSWORD\u00a0": {secret}, "\ufeffkey\u00a0": {secret}}.Encode(),
	}).String()
	pathSegments := (&url.URL{
		Scheme: "https",
		Host:   "example.com",
		Path:   "/v1/\ufeffsigning_key/" + secret + "/\u00a0api_key\ufeff/" + secret + "/ready",
	}).String()
	matrixPairs := (&url.URL{
		Scheme: "https",
		Host:   "example.com",
		Path:   "/v1/\ufeffDB_PASSWORD=" + secret + "/signing_key\u00a0=" + secret + "/\ufeffkey=" + secret,
	}).String()
	fragmentPairs := (&url.URL{
		Scheme:   "https",
		Host:     "example.com",
		Path:     "/cb",
		Fragment: "\ufeffDB_PASSWORD=" + secret + "&signing_key\u00a0=" + secret + "&\u00a0key\ufeff=" + secret + "&state=ready",
	}).String()

	cases := []struct {
		name string
		uri  string
		want string
		// wantText, when set, is the URI as the text entry points write it
		// (see the matrix case below); want is used otherwise.
		wantText string
	}{
		{
			name: "url.Values.Encode BOM-padded DB_PASSWORD",
			uri:  queryBOM,
			want: "https://example.com/api?%EF%BB%BFDB_PASSWORD=" + r + "&status=ready",
		},
		{
			name: "url.Values.Encode NBSP DB_PASSWORD and BOM-NBSP key",
			uri:  queryNBSPAndKey,
			want: "https://example.com/api?DB_PASSWORD%C2%A0=" + r + "&%EF%BB%BFkey%C2%A0=" + r,
		},
		{
			name: "plus-prefixed query key with trailing encoded BOM",
			uri:  "https://example.com/api?my+DB_PASSWORD%EF%BB%BF=" + secret,
			want: "https://example.com/api?my+DB_PASSWORD%EF%BB%BF=" + r,
		},
		{
			name: "credential before encoded-BOM query key scrubs key too",
			uri:  "https://example.com/api?Bearer+" + secret + "+%EF%BB%BFDB_PASSWORD=" + secret,
			want: "https://example.com/api?Bearer+" + r + "+%EF%BB%BFDB_PASSWORD=" + r,
		},
		{
			name: "invalid percent escape in query key falls back cleanly",
			uri:  "https://example.com/api?%ZZ+DB_PASSWORD=" + secret,
			want: "https://example.com/api?%ZZ+DB_PASSWORD=" + r,
		},
		{
			name: "url.URL.String BOM signing_key and NBSP api_key path segments",
			uri:  pathSegments,
			want: "https://example.com/v1/" + r + "/" + r + "/" + r + "/" + r + "/ready",
		},
		{
			name: "encoded NBSP api_key before password keeps following secret armed",
			uri:  "https://example.com/%C2%A0api_key/password/" + secret,
			want: "https://example.com/" + r + "/" + r + "/" + r,
		},
		{
			name: "url.URL.String matrix key=val with BOM and NBSP padding",
			uri:  matrixPairs,
			want: "https://example.com/v1/%EF%BB%BFDB_PASSWORD=" + r + "/signing_key%C2%A0=" + r + "/%EF%BB%BFkey=" + r,
			// In text, the value of DB_PASSWORD= runs to the next space,
			// so the rest of the path is redacted with it, as main and
			// the unpadded password=v/token=v/key=v do (see
			// TestURIMatrixPairsInTextRedactRestOfPath).
			wantText: "https://example.com/v1/%EF%BB%BFDB_PASSWORD=[REDACTED]",
		},
		{
			name: "url.URL.String fragment keys with BOM and NBSP padding",
			uri:  fragmentPairs,
			want: "https://example.com/cb#%EF%BB%BFDB_PASSWORD=" + r + "&signing_key%C2%A0=" + r + "&%C2%A0key%EF%BB%BF=" + r + "&state=ready",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotURI, changed := sanitizeURI(tc.uri)
			if !changed || gotURI != tc.want {
				t.Errorf("sanitizeURI(%q) = (%q, %v), want (%q, true)", tc.uri, gotURI, changed, tc.want)
			}
			if strings.Contains(gotURI, secret) {
				t.Errorf("sanitizeURI(%q) = %q leaked %q", tc.uri, gotURI, secret)
			}
			if again, againChanged := sanitizeURI(gotURI); againChanged || again != gotURI {
				t.Errorf("sanitizeURI second pass(%q) = (%q, %v), want (%q, false)", gotURI, again, againChanged, gotURI)
			}

			msg := "fetch " + tc.uri + " done"
			wantText := tc.want
			if tc.wantText != "" {
				wantText = tc.wantText
			}
			wantMsg := "fetch " + wantText + " done"

			if gotErr := redactErrorMessage(msg); gotErr != wantMsg {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", msg, gotErr, wantMsg)
			} else if again := redactErrorMessage(gotErr); again != gotErr {
				t.Errorf("redactErrorMessage second pass(%q) = %q, want unchanged", gotErr, again)
			}

			if gotSan, lost := sanitizeRawText(msg); !lost || gotSan != wantMsg {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", msg, gotSan, lost, wantMsg)
			} else if again, lost2 := sanitizeRawText(gotSan); lost2 || again != gotSan {
				t.Errorf("sanitizeRawText second pass(%q) = (%q, %v), want (%q, false)", gotSan, again, lost2, gotSan)
			}

			wantTrunc := `"` + strings.ReplaceAll(wantMsg, "&", `\u0026`) + `"`
			if gotTrunc, truncated, err := SmartTruncate(msg, -1); err != nil || !truncated || string(gotTrunc) != wantTrunc {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%s, true, nil)", msg, gotTrunc, truncated, err, wantTrunc)
			}

			if gotTextPart := formatContentTextPart(t, msg); gotTextPart != wantMsg {
				t.Errorf("FormatContentParts(Text %q) = %q, want %q", msg, gotTextPart, wantMsg)
			}
			fileParts := FormatContentParts(&genai.Content{Parts: []*genai.Part{{
				FileData: &genai.FileData{MIMEType: "application/octet-stream", FileURI: tc.uri},
			}}}, -1)
			if len(fileParts) != 1 || fileParts[0]["uri"] != tc.want {
				t.Errorf("FormatContentParts(FileData %q) = %v, want uri %q", tc.uri, fileParts, tc.want)
			}
		})
	}

	// Verify legacyURIKeys preserves merge-base ASCII-only matching while
	// scannerURIKeys redacts multibyte percent-escaped padding cleanly without
	// tripping leavesLegacyValue.
	if leg, legChanged := sanitizeURIWith(queryBOM, legacyRedactURIComponent, legacyURIKeys); legChanged || leg != queryBOM {
		t.Errorf("legacyURIKeys on %q = (%q, %v), want unchanged", queryBOM, leg, legChanged)
	}
	if leg, legChanged := sanitizeURIWith(pathSegments, legacyRedactURIComponent, legacyURIKeys); legChanged || leg != pathSegments {
		t.Errorf("legacyURIKeys on %q = (%q, %v), want unchanged", pathSegments, leg, legChanged)
	}
}

// TestURIMatrixPairsInTextRedactRestOfPath covers a sensitive matrix pair of
// a URI in text. The text entry points read the value of the pair up to the
// next space, so the rest of the path is redacted with it, as on main, even
// when sanitizeURI already scrubbed every later pair: the scrubbed path is
// never kept in place of the value. A later segment that holds a secret, a
// non-sensitive key, a bare value or an unclosed quote is redacted the same
// way, and scrubbing the result again changes nothing.
func TestURIMatrixPairsInTextRedactRestOfPath(t *testing.T) {
	runInParallel(t)
	const (
		secret = "S3CR3T_URI_MATRIX_7"
		r      = "%5BREDACTED%5D"
	)
	for _, tc := range []struct{ name, in, want string }{
		{
			name: "unpadded legacy names",
			in:   "fetch https://example.com/v1/password=" + secret + "/token=" + secret + "/key=" + secret + " done",
			want: "fetch https://example.com/v1/password=[REDACTED] done",
		},
		{
			name: "padded names already scrubbed",
			in:   "fetch https://example.com/v1/%EF%BB%BFDB_PASSWORD=" + r + "/signing_key%C2%A0=" + r + "/%EF%BB%BFkey=" + r + " done",
			want: "fetch https://example.com/v1/%EF%BB%BFDB_PASSWORD=[REDACTED] done",
		},
		{
			name: "scrubbed pairs then secret behind non-sensitive key",
			in:   "fetch https://example.com/v1/DB_PASSWORD=" + r + "/signing_key=" + r + "/x=" + secret + " done",
			want: "fetch https://example.com/v1/DB_PASSWORD=[REDACTED] done",
		},
		{
			name: "scrubbed pairs then bare secret segment",
			in:   "fetch https://example.com/v1/DB_PASSWORD=" + r + "/signing_key=" + r + "/" + secret + " done",
			want: "fetch https://example.com/v1/DB_PASSWORD=[REDACTED] done",
		},
		{
			name: "scrubbed pair then unclosed quote",
			in:   "fetch https://example.com/v1/DB_PASSWORD=" + r + "/token='" + secret + " done",
			want: "fetch https://example.com/v1/DB_PASSWORD=[REDACTED] done",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			} else if again := redactErrorMessage(got); again != got {
				t.Errorf("redactErrorMessage second pass(%q) = %q, want unchanged", got, again)
			}
			if got, lost := sanitizeRawText(tc.in); !lost || got != tc.want {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, got, lost, tc.want)
			} else if again, lost2 := sanitizeRawText(got); lost2 || again != got {
				t.Errorf("sanitizeRawText second pass(%q) = (%q, %v), want (%q, false)", got, again, lost2, got)
			}
			wantTrunc := `"` + tc.want + `"`
			if got, truncated, err := SmartTruncate(tc.in, -1); err != nil || !truncated || string(got) != wantTrunc {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%s, true, nil)", tc.in, got, truncated, err, wantTrunc)
			}
			if got := formatContentTextPart(t, tc.in); got != tc.want {
				t.Errorf("FormatContentParts(Text %q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestSensitiveKeyContainerValuesKeepDocument covers JSON documents in which
// a sensitive key holds an object or an array, as the Credentials of an AWS
// STS response: the value is redacted as a whole and the rest of the document
// is kept, as on main, by every entry point that reads JSON text, for a
// document in a string of another too. The expressions of main remove a part
// of such a container, which holds the start of the strings inside it; none
// of them is a value that the redaction keeps, so that hides no run-on (see
// appendValueSpans and keyRedactionHidesRunOn).
func TestSensitiveKeyContainerValuesKeepDocument(t *testing.T) {
	runInParallel(t)
	eps := largeTextEntryPoints(t)[1:]
	for _, tc := range []struct{ in, want string }{
		{`{"Credentials":{"AccessKeyId":"EXAMPLEKEYID","SecretAccessKey":"SECRET123","SessionToken":"SECRET123"},"Message":"ok"}`, `{"Credentials":"[REDACTED]","Message":"ok"}`},
		{`{"Message":"ok","Credentials":{"AccessKeyId":"EXAMPLEKEYID","SecretAccessKey":"SECRET123"}}`, `{"Credentials":"[REDACTED]","Message":"ok"}`},
		{`{"credentials":{"header":"Authorization: sample"},"status":"ready"}`, `{"credentials":"[REDACTED]","status":"ready"}`},
		{`{"status":"ready","credentials":{"header":"Authorization: sample"}}`, `{"credentials":"[REDACTED]","status":"ready"}`},
		{`{"credentials":"Authorization: sample","status":"ready"}`, `{"credentials":"[REDACTED]","status":"ready"}`},
		{`{"status":"ready","credentials":"Authorization: sample"}`, `{"credentials":"[REDACTED]","status":"ready"}`},
		{`{"wrapper":"{\"credentials\":{\"header\":\"Authorization: sample%20x\"},\"status\":\"ready\"}"}`, `{"wrapper":"{\"credentials\":\"[REDACTED]\",\"status\":\"ready\"}"}`},
		{`{"wrapper":"{\"status\":\"ready\",\"credentials\":{\"header\":\"Authorization: sample%20x\"}}"}`, `{"wrapper":"{\"credentials\":\"[REDACTED]\",\"status\":\"ready\"}"}`},
		{`"{\"credentials\":{\"header\":\"Authorization: sample\"},\"status\":\"ready\"}"`, `"{\"credentials\":\"[REDACTED]\",\"status\":\"ready\"}"`},
		{`"{\"status\":\"ready\",\"credentials\":{\"header\":\"Authorization: sample\"}}"`, `"{\"credentials\":\"[REDACTED]\",\"status\":\"ready\"}"`},
		{`{"token":{"value":"SECRET123"},"status":"ok"}`, `{"status":"ok","token":"[REDACTED]"}`},
		{`{"token":["SECRET123","b"],"status":"ok"}`, `{"status":"ok","token":"[REDACTED]"}`},
		{`{"token":{},"a":1}`, `{"a":1,"token":"[REDACTED]"}`},
		{`{"token":[],"a":1}`, `{"a":1,"token":"[REDACTED]"}`},
		{`{"resp":{"Credentials":{"SessionToken":"SECRET123","Nested":{"k":["SECRET123"]}},"RequestId":"r-1"}}`, `{"resp":{"Credentials":"[REDACTED]","RequestId":"r-1"}}`},
		{`[{"credentials":[{"k":"SECRET123"}],"n":1}]`, `[{"credentials":"[REDACTED]","n":1}]`},
		{`{"credentials": {"a": "password=SECRET123"}, "next": "x"}`, `{"credentials":"[REDACTED]","next":"x"}`},
		{`{"Credentials":{"note":"a } b ] c","SessionToken":"SECRET123"},"Message":"ok"}`, `{"Credentials":"[REDACTED]","Message":"ok"}`},
		{`{"user:temp:conn":{"dsn":"SECRET123"},"Message":"ok"}`, `{"Message":"ok","user:temp:conn":"[REDACTED]"}`},
		{`{"args":"{\"token\":{\"value\":\"SECRET123\"},\"status\":\"ok\"}"}`, `{"args":"{\"status\":\"ok\",\"token\":\"[REDACTED]\"}"}`},
	} {
		for _, ep := range eps {
			got := ep.f(tc.in)
			if got != tc.want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, tc.in, got, tc.want)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second call(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		// Redacting the value of a sensitive key is no loss of content,
		// as on main: none is reported.
		if _, lost := sanitizeRawText(tc.in); lost {
			t.Errorf("sanitizeRawText(%q) reported a loss, want lost=false", tc.in)
		}
		if _, truncated, err := SmartTruncate(tc.in, -1); err != nil || truncated {
			t.Errorf("SmartTruncate(%q) = (truncated=%v, %v), want truncated=false", tc.in, truncated, err)
		}
	}

	// appendValueSpans reads the object or array of a sensitive key as one
	// value that the redaction replaces, from its opening to its closing
	// bracket, and none of the values inside it.
	var values []jsonValueSpan
	wantSpans := []jsonValueSpan{{1, 7, 8, false}, {9, 19, 20, true}, {21, 23, 24, false}, {25, 28, 29, false}}
	if doc := `{"token":{"a":["b"]},"m":"ok"}`; !appendValueSpans(doc, nil, &values, 0) || !slices.Equal(values, wantSpans) {
		t.Errorf("appendValueSpans(%q) = %v, want %v", doc, values, wantSpans)
	}
	// The container of a key that is not sensitive is read value by value.
	values = values[:0]
	wantSpans = []jsonValueSpan{{1, 6, 7, false}, {9, 11, 12, false}, {13, 15, 16, false}}
	if doc := `{"data":{"a":"b"}}`; !appendValueSpans(doc, nil, &values, 0) || !slices.Equal(values, wantSpans) {
		t.Errorf("appendValueSpans(%q) = %v, want %v", doc, values, wantSpans)
	}
	// A container of a sensitive key that does not end is no value.
	for _, doc := range []string{`{"token":{"a":1`, `{"token":["a","b"`, `{"token":{"a":"}"`} {
		values = values[:0]
		if appendValueSpans(doc, nil, &values, 0) {
			t.Errorf("appendValueSpans(%q) = true with %v, want false", doc, values)
		}
	}
	for _, tc := range []struct {
		text string
		open int
		want int
	}{
		{`{"a":1}`, 0, 6},
		{`{"a":"}"}`, 0, 8},
		{`{"a":"\"}"}x`, 0, 10},
		{`[1,[2,{"k":"]"}]]x`, 0, 16},
		{`[1,[2,{"k":"]"}]]x`, 3, 15},
		{`{"a":{"b":[]}}`, 5, 12},
		{`{"a":1`, 0, -1},
		{`["a`, 0, -1},
	} {
		if got := jsonContainerEnd(tc.text, tc.open); got != tc.want {
			t.Errorf("jsonContainerEnd(%q, %d) = %d, want %d", tc.text, tc.open, got, tc.want)
		}
	}
}

// TestMalformedJSONProseKeptAsOnMain covers text that starts with "{", "["
// or a quote and is no JSON, in which a scheme word or a key name stands next
// to an escape, a non-ASCII space, a quote or a path and no credential does.
// redactErrorMessage, sanitizeRawText and FormatContentParts keep it, as on
// main, and SmartTruncate, which keeps no malformed JSON text, fails it
// closed. The same words in text that starts otherwise are kept by every
// entry point.
func TestMalformedJSONProseKeptAsOnMain(t *testing.T) {
	runInParallel(t)
	for _, in := range []string{
		`[INFO] Basic mode=Okta`,
		`[CONFIG] Basic mode=Only over TLS`,
		`[INFO] password\t=[REDACTED]`,
		`[INFO] api_key\n: [REDACTED]`,
		`{"log":"Enter your password\n:`,
		"[INFO] Saisissez le token\u00a0:",
		"{\"log\":\"Saisissez le token\u00a0:",
		`[INFO] Plan: Basic 'monthly'`,
		"[Basic `curl` usage](https://example.com/docs)",
		`[INFO] opened C:\Tools\Basic\bin\app.exe`,
		"[INFO] `password`: must be at least 8 characters",
		`"[object Object]" is not valid JSON`,
	} {
		assertTextKeptAndValueFailsClosed(t, in)
		if got := formatContentTextPart(t, in); got != in {
			t.Errorf("FormatContentParts(%q) = %q, want unchanged", in, got)
		}
	}
	for _, in := range []string{
		"Saisissez le token\u00a0:",
		`INFO Basic mode=Okta`,
		`INFO Plan: Basic 'monthly'`,
		`INFO opened C:\Tools\Basic\bin\app.exe`,
		"INFO `password`: must be at least 8 characters",
	} {
		assertUnchangedAllEntryPoints(t, in)
	}
}

// TestURITokenSpansMatchExpression checks uriTokenSpans, which finds the URIs
// of a text without running uriTokenRe over it, against that expression: over
// every text of up to six of the bytes and runes that decide a match (a
// letter, a digit, a scheme sign, the bytes of "://", a space, a quote, a
// byte that ends no URI and a non-ASCII rune), over longer texts, and over
// text with many URIs.
func TestURITokenSpansMatchExpression(t *testing.T) {
	runInParallel(t)
	check := func(s string) {
		t.Helper()
		got, want := uriTokenSpans(s), uriTokenRe.FindAllStringIndex(s, -1)
		if len(got) != len(want) || !slices.EqualFunc(got, want, func(a, b []int) bool { return slices.Equal(a, b) }) {
			t.Fatalf("uriTokenSpans(%q) = %v, want %v", s, got, want)
		}
	}
	alphabet := []string{"a", "Z", "9", "+", ":", "/", " ", `"`, "é"}
	checked := 0
	var walk func(prefix string, left int)
	walk = func(prefix string, left int) {
		check(prefix)
		checked++
		if left == 0 {
			return
		}
		for _, c := range alphabet {
			walk(prefix+c, left-1)
		}
	}
	walk("", 6)
	if want := 597_871; checked != want {
		t.Fatalf("checked %d texts, want %d", checked, want)
	}
	for _, s := range []string{
		"see https://example.com/a?b=c, then ftp://h/x.",
		"https://a/b?u=http://c/d e://f",
		"1://b://c d",
		"a://",
		"a:// b",
		"x.y-z+w://h\tnext",
		"9a://h<b://i>'c://j'",
		"user:temp://x",
		"://a://b",
		"://://http://h/a",
		"http://://https://h/b",
		"123://456://http://h/c",
		"a://b\vc://d\x00e://f\ufefff",
		"\xffhttp://h/\xff\xfe x",
		strings.Repeat("[INFO] fetched https://example.com/docs/page?id=42 in 12 ms\\tok\u00a0\n", 2000),
		strings.Repeat("a", 5000) + "://" + strings.Repeat("b", 5000),
		strings.Repeat("://", 5000),
		strings.Repeat("123://", 5000),
		strings.Repeat("://abc/def", 5000),
		strings.Repeat("a://", 5000),
	} {
		check(s)
	}

	const smallReps, largeReps = 2 << 10, 32 << 10
	for _, shape := range []struct {
		name string
		unit string
	}{
		{"repeated delimiter", "://"},
		{"scheme-less delimiter with body", "123://456/789"},
	} {
		small := strings.Repeat(shape.unit, smallReps)
		large := strings.Repeat(shape.unit, largeReps)
		if got := uriTokenSpans(large); len(got) != 0 {
			t.Fatalf("uriTokenSpans(%s) = %v, want empty", shape.name, got)
		}
		if ratio, tSmall, tLarge := linearRatio(64, func() { _ = uriTokenSpans(small) }, func() { _ = uriTokenSpans(large) }); ratio > 64 {
			t.Errorf("uriTokenSpans(%s): %d bytes took %v, %d bytes took %v: ratio %.0f, want linear", shape.name, len(small), tSmall, len(large), tLarge, ratio)
		}
		if ratio, tSmall, tLarge := linearRatio(64, func() { _ = scrubErrorText(small) }, func() { _ = scrubErrorText(large) }); ratio > 64 {
			t.Errorf("scrubErrorText(%s): %d bytes took %v, %d bytes took %v: ratio %.0f, want linear", shape.name, len(small), tSmall, len(large), tLarge, ratio)
		}
	}
}

// TestLargeMalformedJSONProseScaling covers a megabyte of text that starts
// with "[" and is no JSON, with a URI, an escaped tab and a no-break space on
// each line and no credential. redactErrorMessage and sanitizeRawText keep
// it, SmartTruncate fails it closed, and the time that they take grows
// linearly with the text and stays within a wall-clock limit: what tells
// such prose from JSON reads the text a fixed number of times, and finds its
// URIs without trying an expression from every letter (see uriTokenSpans).
func TestLargeMalformedJSONProseScaling(t *testing.T) {
	runInParallel(t)
	const unit = "[INFO] fetched https://example.com/docs/page?id=42 in 12 ms\\tstatus ok,\u00a03 rows\n"
	largeN := 12 << 10
	if raceEnabled {
		largeN = 1536
	}
	small, large := strings.Repeat(unit, largeN/16), strings.Repeat(unit, largeN)
	if got := redactErrorMessage(large); got != large {
		t.Errorf("redactErrorMessage(%d bytes of %q) = %.80q (%d bytes), want unchanged", len(large), unit, got, len(got))
	}
	if got, lost := sanitizeRawText(large); got != large || lost {
		t.Errorf("sanitizeRawText(%d bytes of %q) = (%.80q (%d bytes), %v), want unchanged and lost=false", len(large), unit, got, len(got), lost)
	}
	wantClosed := `"` + unparseableJSONPlaceholder + `"`
	if raw, truncated, err := SmartTruncate(large, -1); err != nil || !truncated || string(raw) != wantClosed {
		t.Errorf("SmartTruncate(%d bytes of %q) = (%.80s, %v, %v), want (%s, true, <nil>)", len(large), unit, raw, truncated, err, wantClosed)
	}
	scrub := func(s string) {
		_ = redactErrorMessage(s)
		_, _ = sanitizeRawText(s)
		_, _, _ = SmartTruncate(s, -1)
	}
	scrub(small)
	scrub(large)
	if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(small) }, func() { scrub(large) }); ratio > 64 {
		t.Errorf("%d bytes took %v, %d bytes took %v: ratio %.0f, want linear", len(small), tSmall, len(large), tLarge, ratio)
	}
	maxWall := 10 * time.Second
	if raceEnabled {
		maxWall = 60 * time.Second
	}
	if elapsed := withinWallClock(maxWall, func() { scrub(large) }); elapsed > maxWall {
		t.Errorf("%d bytes of %q took %v, want at most %v", len(large), unit, elapsed, maxWall)
	}
}

// TestSmartTruncateMalformedJSONTextFailsClosed checks that SmartTruncate
// keeps no string that starts with "{" or "[" and is no JSON document,
// whatever the string holds, as on main: as a value, as a map value and in
// the part_attributes of a function response, at any length limit, it is
// replaced by the placeholder for unparseable JSON and reported as lost.
// The scanners read no credential in most of these strings, and a
// placeholder in a string that comes with one says nothing of the text next
// to it, so nothing in such a string shows that it holds no credential. The
// same holds for text that starts with a quote and holds such a string, its
// bracket spelled as an escape, as a later JSON string.
func TestSmartTruncateMalformedJSONTextFailsClosed(t *testing.T) {
	runInParallel(t)
	const secret = "SECRETZQ"
	// base64("user:SECRETZQ" + 13000 * "x"), longer than
	// maxBasicCredentialBytes, and what the secret starts with in it.
	oversizedBasic := base64.StdEncoding.EncodeToString([]byte("user:" + secret + strings.Repeat("x", 13000)))
	oversizedBasicPrefix := oversizedBasic[:16]
	if len(oversizedBasic) <= maxBasicCredentialBytes {
		t.Fatalf("oversized Basic credential is %d bytes, want more than %d", len(oversizedBasic), maxBasicCredentialBytes)
	}
	cases := []struct {
		name, in string
	}{
		// A scheme or a key with no delimiter that the scanners read
		// after it.
		{"bearer glued to a quote", `[INFO] Bearer'` + secret},
		{"basic glued to a quote", `[INFO] Basic'` + oversizedBasicPrefix},
		{"bearer before a placeholder and a glued value", `[INFO] Bearer '[REDACTED]'&` + secret},
		{"key before a variation selector", "[INFO] password\ufe0f=" + secret},
		{"key before a combining grapheme joiner", "[INFO] password\u034f=" + secret},
		{"key before a hangul filler", "[INFO] password\u3164=" + secret},
		{"key before a no-break space", "[INFO] password\u00a0:\u00a0" + secret},
		{"key before an escaped vertical tab", `[INFO] password\v=` + secret},
		{"bearer before a vertical tab", "[INFO] Bearer\v" + secret},
		{"bearer before a zero width space", "[INFO] Bearer\u200b" + secret},
		{"bearer before an invalid percent escape", `[INFO] Bearer%ZZ` + secret},
		{"bearer before an unterminated quote", `[INFO] Bearer '` + secret},
		{"oversized basic with extra padding", "[INFO] Basic " + oversizedBasic + "==="},
		{"oversized basic with glued padding", "[INFO] Basic " + oversizedBasic + "=A"},

		// Truncated JSON documents.
		{"document with an unterminated bearer value", `{"log":"Bearer '` + secret},
		{"document with an escaped vertical tab", `{"log":"password\v=` + secret},
		{"document with an escaped quote in a quoted key", `{"log":"[INFO] 'temp:a\\'x'=` + secret},
		{"document with an escaped backtick in a quoted key", "{\"log\":\"[INFO] `temp:a,b\\\\`x`=" + secret},

		// Text that comes with a placeholder.
		{"redacted URI before a glued bearer", redactedURIPlaceholder + ` Bearer'` + secret},
		{"placeholder before an unread key", "[REDACTED] password\ufe0f=" + secret},
		{"redacted URI and pair before an unterminated bearer value", redactedURIPlaceholder + ` "api_key":"[REDACTED]" Bearer '` + secret},
		{"scrubbed document before a glued bearer", `{"msg":"password=[REDACTED]} Bearer'` + secret},

		// Text that starts with a quote and holds such a string, its
		// bracket spelled as the escape \u005b, as a later JSON string.
		{"later string that decodes to a log line", `"hello" "\u005bINFO\u005d Bearer'` + secret + `"`},
		{"later string that decodes to a redacted URI and a glued bearer", `"hello" "\u005bREDACTED_SENSITIVE_URI\u005d Bearer'` + secret + `"`},
		{"later string that decodes to a placeholder and an unread key", `"hello" "\u005bREDACTED\u005d password` + "\ufe0f=" + secret + `"`},

		// Text without a credential.
		{"log line", `[INFO] ready`},
		{"markdown link", `[docs](https://example.com)`},
		{"template", `{{template}} rendered`},
		{"scheme word in prose", `[INFO] Basic mode=Okta`},
		{"redacted pair after an escaped tab", `[INFO] password\t=[REDACTED]`},
	}
	const (
		wantDirect = `"` + unparseableJSONPlaceholder + `"`
		wantMap    = `{"log":"` + unparseableJSONPlaceholder + `"}`
		wantAttrs  = `{"function_response":{"name":"fetch_log","response":{"log":"` + unparseableJSONPlaceholder + `"}}}`
	)
	for _, limit := range []int{-1, 4096} {
		for _, tc := range cases {
			t.Run(tc.name+"/limit="+strconv.Itoa(limit), func(t *testing.T) {
				if direct, lost, err := SmartTruncate(tc.in, limit); err != nil || !lost || string(direct) != wantDirect {
					t.Errorf("SmartTruncate(%.80q, %d) = (%.120s, %v, %v), want (%s, true, <nil>)", tc.in, limit, direct, lost, err, wantDirect)
				}
				if inMap, lost, err := SmartTruncate(map[string]any{"log": tc.in}, limit); err != nil || !lost || string(inMap) != wantMap {
					t.Errorf("SmartTruncate(map with %.80q, %d) = (%.120s, %v, %v), want (%s, true, <nil>)", tc.in, limit, inMap, lost, err, wantMap)
				}
				parts, lost := formatContentParts(&genai.Content{
					Parts: []*genai.Part{{
						FunctionResponse: &genai.FunctionResponse{
							Name:     "fetch_log",
							Response: map[string]any{"log": tc.in},
						},
					}},
				}, limit, true)
				if !lost || len(parts) != 1 {
					t.Fatalf("formatContentParts(function response with %.80q, %d) = (%v, %v), want 1 part and lost=true", tc.in, limit, parts, lost)
				}
				if attrs, _ := parts[0]["part_attributes"].(string); attrs != wantAttrs {
					t.Errorf("part_attributes of a function response with %.80q (limit %d) = %.160s, want %s", tc.in, limit, attrs, wantAttrs)
				}
			})
		}
	}

	// A string of a JSON document is read as the document is: one that
	// starts with "{" or "[" and is no JSON fails closed in the document
	// that every entry point returns, also when it starts with a placeholder.
	for _, tc := range []struct{ in, want string }{
		{`{"a":"[INFO] Bearer'` + secret + `"}`, `{"a":"` + unparseableJSONPlaceholder + `"}`},
		{`{"a":"` + redactedURIPlaceholder + ` Bearer'` + secret + `"}`, `{"a":"` + unparseableJSONPlaceholder + `"}`},
		{`{"a":"[REDACTED] password=` + secret + `"}`, `{"a":"` + unparseableJSONPlaceholder + `"}`},
		{`["{\"msg\":\"password=[REDACTED]} Bearer'` + secret + `"]`, `["` + unparseableJSONPlaceholder + `"]`},
		{`{"a":"\"hello\" \"\\u005bREDACTED_SENSITIVE_URI\\u005d Bearer'` + secret + `\""}`, `{"a":"` + unparseableJSONPlaceholder + `"}`},
	} {
		if got := redactErrorMessage(tc.in); got != tc.want {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if got, lost := sanitizeRawText(tc.in); got != tc.want || !lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, got, lost, tc.want)
		}
		var got string
		if raw, lost, err := SmartTruncate(tc.in, -1); err != nil || !lost || json.Unmarshal(raw, &got) != nil || got != tc.want {
			t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%q, true, <nil>)", tc.in, raw, lost, err, tc.want)
		}
	}
}

// TestJSONLiteralBeforeTextScrubbedAfterRedaction covers text that starts
// with a JSON string literal and goes on after it, where the redaction
// writes the document in the literal again, with a string of it failed
// closed, and the scrub of the text then redacts a credential in another
// string of that document. Every entry point that reads JSON text returns
// the literal written again with the credential redacted and the text after
// it kept, as on main, and a second call changes nothing.
func TestJSONLiteralBeforeTextScrubbedAfterRedaction(t *testing.T) {
	runInParallel(t)
	eps := largeTextEntryPoints(t)[1:]
	for _, tc := range []struct{ in, want string }{
		{`"[\"[INFO] x\",\"a\"]" ok`, `"[\"[UNPARSEABLE_JSON_BLOB]\",\"a\"]" ok`},
		{`"[\"[INFO] x\",\"password=hunter2ZQ def\"]" ok`, `"[\"[UNPARSEABLE_JSON_BLOB]\",\"password=[REDACTED] def\"]" ok`},
		{`"[\"[INFO] x\",\"see https://u:hunter2ZQ@h/a now\"]" ok`, `"[\"[UNPARSEABLE_JSON_BLOB]\",\"see [REDACTED_SENSITIVE_URI] now\"]" ok`},
		{`"{\"k\":\"[INFO] x\",\"m\":\"password=hunter2ZQ and more\"}" ok`, `"{\"k\":\"[UNPARSEABLE_JSON_BLOB]\",\"m\":\"password=[REDACTED] and more\"}" ok`},
		{`"{\"k\":\"[INFO] x\",\"m\":\"see https://u:hunter2ZQ@h/a now\"}" tail`, `"{\"k\":\"[UNPARSEABLE_JSON_BLOB]\",\"m\":\"see [REDACTED_SENSITIVE_URI] now\"}" tail`},
		{`{"a":"\"[\\\"[INFO] x\\\",\\\"see https://u:hunter2ZQ@h/a now\\\"]\" ok"}`, `{"a":"\"[\\\"[UNPARSEABLE_JSON_BLOB]\\\",\\\"see [REDACTED_SENSITIVE_URI] now\\\"]\" ok"}`},
	} {
		for _, ep := range eps {
			got := ep.f(tc.in)
			if got != tc.want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, tc.in, got, tc.want)
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second call(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		if _, lost := sanitizeRawText(tc.in); !lost {
			t.Errorf("sanitizeRawText(%q) reported no loss, want lost=true", tc.in)
		}
	}
}

// TestQuotedKeyPastInspectedPrefixCountsAsMarker checks that a quoted
// sensitive key with no value after it that stands past the first
// maxInspectableJSONTextBytes bytes of a text, which
// hasCandidateQuotedSensitiveKey does not read, still counts as a credential
// marker (see hasAnyCredentialMarker), so that redactErrorMessage fails
// malformed JSON text that ends in one closed, as on main.
func TestQuotedKeyPastInspectedPrefixCountsAsMarker(t *testing.T) {
	runInParallel(t)
	text := strings.Repeat("x", maxInspectableJSONTextBytes) + ` "token":`
	if hasCandidateQuotedSensitiveKey(text, 0) {
		t.Fatalf("hasCandidateQuotedSensitiveKey reads a key past the first %d bytes", maxInspectableJSONTextBytes)
	}
	if !hasAnyCredentialMarker(text) {
		t.Errorf("hasAnyCredentialMarker(%d bytes that end in a quoted sensitive key) = false, want true", len(text))
	}
	if got := redactErrorMessage("{" + text); got != unparseableJSONPlaceholder {
		t.Errorf("redactErrorMessage(%d bytes of malformed JSON that end in a quoted sensitive key) = %.40q, want %q", len(text)+1, got, unparseableJSONPlaceholder)
	}
}
