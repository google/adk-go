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
	"errors"
	"fmt"
	"net/url"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode"
)

func TestScrubTextPatterns_BearerAndBasicWithUnicodeWhitespace(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"bearer ASCII space", "Bearer sk-live-SECRET", "Bearer [REDACTED]"},
		{"bearer no-break space", "Bearer\u00a0sk-live-SECRET", "Bearer [REDACTED]"},
		{"bearer ideographic space", "Bearer\u3000eyJSECRET", "Bearer [REDACTED]"},
		{"bearer em space", "Bearer\u2003SECRET", "Bearer [REDACTED]"},
		{"bearer next line", "Bearer\u0085SECRET", "Bearer [REDACTED]"},
		{"bearer in a sentence", "auth failed: Bearer\u00a0SECRET, retry", "auth failed: Bearer [REDACTED], retry"},
		{"basic no-break space", "x Basic\u00a0dXNlcjpwYXNz y", "x Basic [REDACTED] y"},
		{"basic ideographic space", "Basic\u3000dXNlcjpwYXNz", "Basic [REDACTED]"},
		{"bearer of is still prose", "the bearer\u00a0of bad news", "the bearer\u00a0of bad news"},
		{"basic prose is kept", "a basic\u00a0HTTP server", "a basic\u00a0HTTP server"},
		{"bearer vertical tab", "Bearer\vSYNTHETIC_SECRET", "Bearer [REDACTED]"},
		{"bearer line separator", "Bearer\u2028SYNTHETIC_SECRET", "Bearer [REDACTED]"},
		{"bearer paragraph separator", "Bearer\u2029SYNTHETIC_SECRET", "Bearer [REDACTED]"},
		{"basic vertical tab", "Basic\vdXNlcjpwYXNz", "Basic [REDACTED]"},
		{"basic line separator", "Basic\u2028dXNlcjpwYXNz", "Basic [REDACTED]"},
		{"basic paragraph separator", "Basic\u2029dXNlcjpwYXNz", "Basic [REDACTED]"},
		{"bearer token stops at a vertical tab", "x Bearer SYNTHETIC_SECRET\vy", "x Bearer [REDACTED]\vy"},
		{"bearer token stops at a line separator", "x Bearer SYNTHETIC_SECRET\u2028y", "x Bearer [REDACTED]\u2028y"},
		// Bearer token after NBSP must not swallow trailing Unicode whitespace.
		{"bearer token stops at trailing nbsp", "x Bearer\u00a0sk-live-SECRET\u00a0y", "x Bearer [REDACTED]\u00a0y"},
		// "bearer of" prose with NBSP must not swallow a following real Bearer token.
		{"bearer of prose then real bearer token", "the bearer\u00a0of\u00a0Bearer\u00a0sk-live-SECRET", "the bearer\u00a0of\u00a0Bearer [REDACTED]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := scrubTextPatterns(tt.in); got != tt.want {
				t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
			}
			if got, _ := sanitizeRawText(tt.in); got != tt.want {
				t.Errorf("sanitizeRawText(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestIsSensitiveKey_CompoundNames(t *testing.T) {
	tests := []struct {
		key  string
		want bool
	}{
		// Environment variables and prefixed names that end in a credential.
		{"DB_PASSWORD", true},
		{"db_password", true},
		{"MYSQL_ROOT_PASSWORD", true},
		{"GITHUB_TOKEN", true},
		{"SLACK_BOT_TOKEN", true},
		{"OPENAI_API_KEY", true},
		{"openai-api-key", true},
		{"dbPassword", true},
		{"DbPassword", true},
		{"myClientSecret", true},
		{"x_auth_token", true},
		{"X-Amz-Security-Token", true},
		{"spring.datasource.password", true},
		{"oauth2Config.code", true},
		{"status.oauth2Config.code", true},
		{"user:DB_PASSWORD", true},
		{"app:db_password", true},
		// Names that only contain a credential name.
		{"token_count", false},
		{"prompt_tokens", false},
		{"completion_tokens", false},
		{"total_tokens", false},
		{"max_output_tokens", false},
		{"thinking_tokens", false},
		{"cached_content_token_count", false},
		{"password_policy", false},
		{"secret_name", false},
		{"tokenizer", false},
		// Pagination tokens are not credentials.
		{"next_page_token", false},
		{"page_token", false},
		{"nextPageToken", false},
		{"app:next_page_token", false},
		// Ordinary words in a payload.
		{"method_signature", false},
		{"design", false},
		{"mypassword", false},
		{"code", false},
		{"status.code", false},
		{"", false},
		// Names that were sensitive before.
		{"password", true},
		{"api_key", true},
		{"apiKey", true},
		{"APIKey", true},
		{"temp:anything", true},
		{"user:api_key", true},
	}
	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			if got := isSensitiveKey(tt.key); got != tt.want {
				t.Errorf("isSensitiveKey(%q) = %v, want %v", tt.key, got, tt.want)
			}
		})
	}
}

func TestScrubTextPatterns_CompoundKeyNames(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"openai api key", "OPENAI_API_KEY=sk-SECRET", "OPENAI_API_KEY=[REDACTED]"},
		{"lowercase api key", "openai_api_key: SECRET", "openai_api_key:[REDACTED]"},
		{"db password", "DB_PASSWORD: SECRET", "DB_PASSWORD:[REDACTED]"},
		{"camel case password", "dbPassword=SECRET", "dbPassword=[REDACTED]"},
		{"github token with a following pair", "GITHUB_TOKEN=SECRET x=1", "GITHUB_TOKEN=[REDACTED] x=1"},
		{"env listing", "ENV: DB_PASSWORD=SECRET", "ENV: DB_PASSWORD=[REDACTED]"},
		{"oauth config code", "status.oauth2Config.code=SECRET", "status.oauth2Config.code=[REDACTED]"},
		{"hyphenated header name", "X-Amz-Security-Token: SECRET", "X-Amz-Security-Token:[REDACTED]"},
		{"quoted value", `MYSQL_ROOT_PASSWORD="se cret"`, `MYSQL_ROOT_PASSWORD="[REDACTED]"`},
		{"plain name is unchanged", "api_key=SECRET", "api_key=[REDACTED]"},
		{"name after a hyphen was always matched", "foo-signature=abc", "foo-signature=[REDACTED]"},
		{"counts are kept", "token_count=5 prompt_tokens: 7 max_output_tokens=9", "token_count=5 prompt_tokens: 7 max_output_tokens=9"},
		{"pagination tokens are kept", "next_page_token=abc page_token: def", "next_page_token=abc page_token: def"},
		{"ordinary words are kept", "method_signature=abc mypassword=abc", "method_signature=abc mypassword=abc"},
		{"credential after a kept name", "page_token=abc DB_PASSWORD=SECRET", "page_token=abc DB_PASSWORD=[REDACTED]"},
		// CamelCase compound keys at non-zero offset (P1-2 fix).
		{"camel case after prefix", "ENV: dbPassword=supersecret", "ENV: dbPassword=[REDACTED]"},
		{"camel case after another pair", "x=1 myClientSecret=supersecret", "x=1 myClientSecret=[REDACTED]"},
		{"camel case in truncated json", `err: {"dbPassword": "supersecret"`, `err: {"dbPassword":"[REDACTED]"`},
		// Non-boundary substrings must NOT be redacted (false-positive guard).
		{"non-boundary mypassword", "mypassword=abc", "mypassword=abc"},
		{"non-boundary token count", "not_a_token_count=5", "not_a_token_count=5"},
		{"non-boundary prompt tokens", "prompt_tokens=100", "prompt_tokens=100"},
		{"camel case acronym prefix", "x DBPassword=supersecret", "x DBPassword=[REDACTED]"},
		{"camel case before a bare credential", "dbPassword=a token=b", "dbPassword=[REDACTED] token=[REDACTED]"},
		{"camel case before a compound credential", "see dbPassword=a DB_PASSWORD=b", "see dbPassword=[REDACTED] DB_PASSWORD=[REDACTED]"},
		{"camel case page token is kept", "x nextPageToken=abc", "x nextPageToken=abc"},
		{"camel case name that is not a credential", "x myLabel=abc", "x myLabel=abc"},
		{"hyphenated page token is kept", "page-token=cursor-2", "page-token=cursor-2"},
		{"hyphenated next page token is kept", "next-page-token=cursor-3", "next-page-token=cursor-3"},
		{"hyphenated previous page token is kept", "x previous-page-token: cursor-4", "x previous-page-token: cursor-4"},
		{"hyphenated signature is redacted", "foo-signature=SECRET", "foo-signature=[REDACTED]"},
		{"bare token is redacted", "token=SECRET", "token=[REDACTED]"},
		{"hyphenated credential after a page token", "page-token=cursor-2 foo-signature=SECRET", "page-token=cursor-2 foo-signature=[REDACTED]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := scrubTextPatterns(tt.in)
			if got != tt.want {
				t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
			}
			if strings.Contains(got, "SECRET") {
				t.Errorf("scrubTextPatterns(%q) = %q, which still holds the secret", tt.in, got)
			}
			if again := scrubTextPatterns(got); again != got {
				t.Errorf("scrubbing %q again gave %q, want it unchanged", got, again)
			}
		})
	}
}

func TestSanitizeValue_CompoundMapKeys(t *testing.T) {
	in := map[string]any{
		"DB_PASSWORD":       "SECRET",
		"GITHUB_TOKEN":      "SECRET",
		"OPENAI_API_KEY":    "SECRET",
		"dbPassword":        "SECRET",
		"oauth2Config.code": "SECRET",
		"x_auth_token":      "SECRET",
		"prompt_tokens":     12,
		"total_tokens":      30,
		"next_page_token":   "page-2",
		"model":             "gemini",
	}
	got, _ := sanitizeValue(in, 1<<20, true)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("sanitizeValue returned %T, want a map", got)
	}
	for _, k := range []string{"DB_PASSWORD", "GITHUB_TOKEN", "OPENAI_API_KEY", "dbPassword", "oauth2Config.code", "x_auth_token"} {
		if m[k] != redactedValue {
			t.Errorf("%s = %v, want %q", k, m[k], redactedValue)
		}
	}
	for k, want := range map[string]any{"next_page_token": "page-2", "model": "gemini"} {
		if m[k] != want {
			t.Errorf("%s = %v, want %v", k, m[k], want)
		}
	}
	for _, k := range []string{"prompt_tokens", "total_tokens"} {
		if m[k] == redactedValue {
			t.Errorf("%s was redacted", k)
		}
	}
}

func TestSanitizeValue_CompoundStructFieldNames(t *testing.T) {
	type config struct {
		DBPassword    string
		GithubToken   string
		NextPageToken string
		PromptTokens  int
	}
	got, _ := sanitizeValue(config{DBPassword: "SECRET", GithubToken: "SECRET", NextPageToken: "p2", PromptTokens: 5}, 1<<20, true)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("sanitizeValue returned %T, want a map", got)
	}
	if m["DBPassword"] != redactedValue || m["GithubToken"] != redactedValue {
		t.Errorf("credential fields were not redacted: %v", m)
	}
	if m["NextPageToken"] != "p2" {
		t.Errorf("NextPageToken = %v, want p2", m["NextPageToken"])
	}
}

func TestSmartTruncate_RedactsUserinfoPassword(t *testing.T) {
	// url.Userinfo implements fmt.Stringer and publishes its password via
	// String(); the formatter must redact it instead.
	got, truncated, err := SmartTruncate(url.UserPassword("alice", "SYNTHETIC_SECRET"), 1000)
	if err != nil {
		t.Fatalf("SmartTruncate() error = %v", err)
	}
	if truncated {
		t.Errorf("SmartTruncate() truncated = true, want false")
	}
	s := string(got)
	if strings.Contains(s, "SYNTHETIC_SECRET") {
		t.Errorf("SmartTruncate() = %q, password leaked", s)
	}
	if !strings.Contains(s, "alice") {
		t.Errorf("SmartTruncate() = %q, username lost", s)
	}
	if !strings.Contains(s, redactedValue) {
		t.Errorf("SmartTruncate() = %q, want [REDACTED]", s)
	}
}

func TestSmartTruncate_RedactsUserinfoPasswordNestedMap(t *testing.T) {
	type holder struct {
		Auth *url.Userinfo
		Val  url.Userinfo
	}
	pw := url.UserPassword("alice", "SYNTHETIC_SECRET")
	for _, tt := range []struct {
		name string
		in   any
	}{
		{"map", map[string]any{"userinfo": pw}},
		{"slice", []any{pw}},
		{"url", &url.URL{Scheme: "https", Host: "example.com", User: pw}},
		{"struct field", holder{Auth: pw, Val: *pw}},
		{"value", *pw},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, _, err := SmartTruncate(tt.in, 1000)
			if err != nil {
				t.Fatalf("SmartTruncate() error = %v", err)
			}
			s := string(got)
			if strings.Contains(s, "SYNTHETIC_SECRET") {
				t.Errorf("SmartTruncate() = %s, password leaked", s)
			}
			if !strings.Contains(s, "alice") {
				t.Errorf("SmartTruncate() = %s, user name lost", s)
			}
		})
	}
	t.Run("user name only", func(t *testing.T) {
		got, _, err := SmartTruncate(url.User("alice"), 1000)
		if err != nil || string(got) != `"alice"` {
			t.Errorf("SmartTruncate(url.User) = (%s, %v), want \"alice\"", got, err)
		}
	})
}

type credentialError map[string]string

func (e credentialError) Error() string {
	return "operation failed: " + e["password"]
}

type credentialStringerSlice []string

func (s credentialStringerSlice) String() string { return "joined: " + strings.Join(s, ",") }

func TestSmartTruncate_ErrorImplementingMapStaysStructural(t *testing.T) {
	// A map, slice or array that implements error or fmt.Stringer keeps its
	// structural traversal, which redacts by key, instead of its text, which
	// has lost the key.
	for _, tt := range []struct {
		name string
		in   any
		want string
	}{
		{"map error", credentialError{"password": "SYNTHETIC_SECRET"}, `{"password":"[REDACTED]"}`},
		{"nested map error", map[string]any{"e": credentialError{"password": "SYNTHETIC_SECRET"}}, `{"e":{"password":"[REDACTED]"}}`},
		{"slice stringer", credentialStringerSlice{"a", "b"}, `["a","b"]`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, _, err := SmartTruncate(tt.in, 1000)
			if err != nil {
				t.Fatalf("SmartTruncate() error = %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("SmartTruncate() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestRedactsCompoundCredentialsNestedUnderEnvelope(t *testing.T) {
	envelopes := []struct{ name, in string }{
		{"DB_PASSWORD", "credentials:\n DB_PASSWORD: SYNTHETIC_SECRET"},
		{"OPENAI_API_KEY", "credentials:\n OPENAI_API_KEY: SYNTHETIC_SECRET"},
		{"GITHUB_TOKEN", "credentials:\n GITHUB_TOKEN: SYNTHETIC_SECRET"},
		{"dbPassword", "credentials:\n dbPassword: SYNTHETIC_SECRET"},
		{"oauth2Config.code", "oauth2Config:\n oauth2Config.code: SYNTHETIC_SECRET"},
		{"oauth2Config block", "credentials:\n oauth2Config.code: SYNTHETIC_SECRET"},
		{"same line", "credentials: DB_PASSWORD: SYNTHETIC_SECRET"},
	}
	for _, tt := range envelopes {
		t.Run(tt.name, func(t *testing.T) {
			if got := scrubTextPatterns(tt.in); strings.Contains(got, "SYNTHETIC_SECRET") {
				t.Errorf("scrubTextPatterns(%q) = %q, which holds the secret", tt.in, got)
			}
			if got, _ := sanitizeRawText(tt.in); strings.Contains(got, "SYNTHETIC_SECRET") {
				t.Errorf("sanitizeRawText(%q) = %q, which holds the secret", tt.in, got)
			}
			got, _, err := SmartTruncate(errors.New(tt.in), 1000)
			if err != nil {
				t.Fatalf("SmartTruncate(error) error = %v", err)
			}
			if strings.Contains(string(got), "SYNTHETIC_SECRET") {
				t.Errorf("SmartTruncate(error %q) = %s, which holds the secret", tt.in, got)
			}
		})
	}
	t.Run("a child that is not a credential is kept", func(t *testing.T) {
		in := "credentials:\n DB_PASSWORD: SYNTHETIC_SECRET\n region: us-east1"
		if got := scrubTextPatterns(in); !strings.Contains(got, "region: us-east1") {
			t.Errorf("scrubTextPatterns(%q) = %q, want the region kept", in, got)
		}
	})
}

func TestScrubsEveryUnicodeSpaceAfterScheme(t *testing.T) {
	var separators []rune
	for r := rune(0); r <= 0x10FFFF; r++ {
		if r != '\n' && r != '\r' && unicode.IsSpace(r) {
			separators = append(separators, r)
		}
	}
	separators = append(separators, '\n', '\r')
	for _, sep := range separators {
		t.Run(fmt.Sprintf("U+%04X", sep), func(t *testing.T) {
			for _, in := range []string{
				"Bearer" + string(sep) + "SYNTHETIC_SECRET",
				"Basic" + string(sep) + "dXNlcjpwYXNz",
			} {
				if got := scrubTextPatterns(in); strings.Contains(got, "SYNTHETIC_SECRET") || strings.Contains(got, "dXNlcjpwYXNz") {
					t.Errorf("scrubTextPatterns(%q) = %q, which holds the credential", in, got)
				}
				if got, _ := sanitizeRawText(in); strings.Contains(got, "SYNTHETIC_SECRET") || strings.Contains(got, "dXNlcjpwYXNz") {
					t.Errorf("sanitizeRawText(%q) = %q, which holds the credential", in, got)
				}
			}
		})
	}
}

func TestIsCompoundSensitiveKey_LongKey(t *testing.T) {
	prefix := strings.Repeat("a_", 16000)
	for _, tt := range []struct {
		name, key string
		want      bool
	}{
		{"credential suffix", prefix + "password", true},
		{"camel credential suffix", prefix + "Password", true},
		{"low precision suffix", prefix + "signature", false},
		{"page token suffix", prefix + "page_token", false},
		{"no credential suffix", prefix + "label", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := isCompoundSensitiveKey(tt.key); got != tt.want {
				t.Errorf("isCompoundSensitiveKey(long key) = %v, want %v", got, tt.want)
			}
			// Only the last maxSensitiveKeyWords suffixes are joined, so the
			// bytes allocated grow with the key, not with its square (137 MB
			// for this key when every suffix was joined).
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			isCompoundSensitiveKey(tt.key)
			runtime.ReadMemStats(&after)
			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 16<<20 {
				t.Errorf("isCompoundSensitiveKey(long key) allocated %d bytes, want a small multiple of the key", allocated)
			}
		})
	}
	t.Run("time", func(t *testing.T) {
		start := time.Now()
		isCompoundSensitiveKey(prefix + "password")
		isCompoundSensitiveKey(prefix + "signature")
		if d := time.Since(start); d > 200*time.Millisecond {
			t.Errorf("isCompoundSensitiveKey(long key) took %v, want well under the quadratic cost", d)
		}
	})
}

func TestScrubTextPatterns_EnvelopeChildren(t *testing.T) {
	longKey := strings.Repeat("x_", 70) + "PASSWORD"
	dottedKey := "x.y_" + longKey
	tests := []struct {
		name string
		in   string
	}{
		{"state-prefixed user child", "credentials:\n user:DB_PASSWORD: SYNTHETIC_SECRET"},
		{"state-prefixed app child", "credentials:\n app:OPENAI_API_KEY: SYNTHETIC_SECRET"},
		{"state-prefixed camel case child", "credentials:\n user:dbPassword: SYNTHETIC_SECRET"},
		{"quoted state-prefixed child", "credentials:\n \"user:DB_PASSWORD\": SYNTHETIC_SECRET"},
		{"state-prefixed child on the same line", "credentials: user:DB_PASSWORD: SYNTHETIC_SECRET"},
		{"upper-case state prefix", "credentials:\n USER:DB_PASSWORD: SYNTHETIC_SECRET"},
		{"oversized child", "credentials:\n " + longKey + ": SYNTHETIC_SECRET"},
		{"oversized child with a delimiter before its value", "credentials:\n " + longKey + " = SYNTHETIC_SECRET"},
		{"oversized child on the same line", "credentials: " + longKey + ": SYNTHETIC_SECRET"},
		{"quoted oversized child", "credentials:\n \"" + longKey + "\": SYNTHETIC_SECRET"},
		{"oversized state-prefixed child", "credentials:\n user:" + longKey + ": SYNTHETIC_SECRET"},
		{"oversized child after another", "credentials:\n a_PASSWORD: " + longKey + ": SYNTHETIC_SECRET"},
		{"oversized key, quoted multiword value after =", longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized key, quoted multiword value after :", longKey + `: "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized key, single-quoted multiword value", longKey + `='prefix SYNTHETIC_SECRET suffix'`},
		{"oversized key, single-quoted multiword value after :", longKey + `: 'prefix SYNTHETIC_SECRET suffix'`},
		{"oversized user key, quoted multiword value", "user:" + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized app key, quoted multiword value", "app:" + longKey + `: "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized child, quoted multiword value", "credentials:\n " + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized key, quoted multiword value after a word", "call failed " + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"two oversized children, then a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ": " + longKey + ": " + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"three oversized children, then a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ": " + longKey + ": " + longKey + ": " + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children, then a delimiter and a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ": " + longKey + ": " + longKey + `: "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a delimiter and a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `: "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a single-quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `='prefix SYNTHETIC_SECRET suffix'`},
		{"oversized children in one word, then a bare value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `: SYNTHETIC_SECRET`},
		{"oversized children in one word, then a spaced = and a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a spaced : and a quoted multiword value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` : "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a spaced = and a single-quoted value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` =  'prefix SYNTHETIC_SECRET suffix'`},
		{"oversized children in one word, then a delimiter and a spaced quoted value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `= "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children in one word, then a spaced = and a bare value", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = SYNTHETIC_SECRET`},
		{"oversized dotted child, quoted multiword value", "credentials:\n a_PASSWORD: " + dottedKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized dotted child, delimiter and quoted multiword value", "credentials:\n a_PASSWORD: " + dottedKey + `: "prefix SYNTHETIC_SECRET suffix"`},
		{"oversized dotted child, single-quoted multiword value", "credentials:\n a_PASSWORD: " + dottedKey + `='prefix SYNTHETIC_SECRET suffix'`},
		{"oversized dotted child with several dots", "credentials:\n a_PASSWORD: a.b.c." + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized dotted child with a state prefix", "credentials:\n a_PASSWORD: user:" + dottedKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized dotted child after another", "credentials:\n a_PASSWORD: " + longKey + ": " + dottedKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized dotted key on its own", dottedKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"oversized children with state prefixes", "credentials:\n user:" + longKey + ": app:" + longKey + ": " + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check := func(path, got string) {
				t.Helper()
				if strings.Contains(got, "SYNTHETIC_SECRET") {
					t.Errorf("%s of %q = %q, which still holds the secret", path, tt.in, got)
				}
			}
			got := scrubTextPatterns(tt.in)
			check("scrubTextPatterns", got)
			if again := scrubTextPatterns(got); again != got {
				t.Errorf("scrubbing %q again gave %q, want it unchanged", got, again)
			}
			raw, _ := sanitizeRawText(tt.in)
			check("sanitizeRawText", raw)
			for path, in := range map[string]any{
				"error":                    errors.New(tt.in),
				"struct with a text field": struct{ Text string }{tt.in},
				"error in a map":           map[string]any{"err": errors.New(tt.in)},
				"error in a struct":        struct{ Err error }{errors.New(tt.in)},
				"wrapped error":            fmt.Errorf("outer: %w", errors.New(tt.in)),
			} {
				b, _, err := SmartTruncate(in, 1<<20)
				if err != nil {
					t.Fatalf("SmartTruncate(%s) error: %v", path, err)
				}
				check("SmartTruncate "+path, string(b))
			}
		})
	}
}

func TestNestedKeyName_StatePrefix(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		lineBeg  bool
		wantKind nestedKeyKind
		wantAt   int
	}{
		{"user prefix on a line", "user:DB_PASSWORD", true, nestedKey, 5},
		{"app prefix on a line", "app:OPENAI_API_KEY", true, nestedKey, 4},
		{"user prefix after the key's line", "user:dbPassword", false, valueEndingInKeyName, 5},
		{"upper-case prefix", "APP:dbPassword", true, nestedKey, 4},
		{"prefix of a name that is not a credential", "user:name", true, notNestedKey, 0},
		{"fixed name with prefix", "user:password", true, nestedKey, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kind, at := nestedKeyName(tt.in, 0, tt.lineBeg)
			if kind != tt.wantKind || at != tt.wantAt {
				t.Errorf("nestedKeyName(%q) = (%v, %d), want (%v, %d)", tt.in, kind, at, tt.wantKind, tt.wantAt)
			}
		})
	}
}

// scrubTime returns how long scrubbing takes.
func scrubTime(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

// TestScrubTextPatterns_CamelCaseScanIsLinear checks that the search for
// camelCase keys does not rescan the text for each key or boundary in it: a
// run of n repeats must take about n times as long as one of n/4, not n^2.
func TestScrubTextPatterns_CamelCaseScanIsLinear(t *testing.T) {
	tests := []struct {
		name string
		gen  func(n int) string
		run  func(in string)
		n    int
	}{
		{
			"repeated camel case pairs through an error", func(n int) string { return strings.Repeat("x dbPassword=synthetic ", n) },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 2000,
		},
		{
			"repeated camel case pairs", func(n int) string { return strings.Repeat("x dbPassword=synthetic ", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one long run of oauth boundaries", func(n int) string { return strings.Repeat("aOauth", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"long run of oauth boundaries after a word", func(n int) string { return "x " + strings.Repeat("xOauth", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"long run of oauth boundaries with an inner dot", func(n int) string { return "x " + strings.Repeat("xOauth", n) + ".x.code=x" },
			func(in string) { _ = scrubTextPatterns(in) }, 40000,
		},
		{
			"long run of oauth boundaries with an inner dot through an error", func(n int) string { return "x " + strings.Repeat("xOauth", n) + ".x.code=x" },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 40000,
		},
		{
			"long run of oauth boundaries before a delimiter", func(n int) string { return strings.Repeat("aOauth", n) + "=x" },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"long run of oauth boundaries before a delimiter without a value", func(n int) string { return strings.Repeat("aOauth", n) + ".code=" },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"oversized key chain", func(n int) string { return "password=SECRET123-" + strings.Repeat("token:", n) + " x" },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"spaced oversized keys", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat(strings.Repeat("x_", 70)+"PASSWORD: ", n) + `x_PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 400,
		},
		{
			"spaced oversized keys through an error", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat(strings.Repeat("x_", 70)+"PASSWORD=", n) + " x"
			},
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 400,
		},
		{
			"oversized key with many dots", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("a.", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"repeated oversized dotted children", func(n int) string {
				return strings.Repeat("credentials:\n a_PASSWORD: x.y_"+strings.Repeat("x_", 70)+`PASSWORD="a b" `+"\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 150,
		},
		{
			"repeated runs of oversized keys with a spaced delimiter", func(n int) string {
				key := strings.Repeat("x_", 70) + "PASSWORD"
				return strings.Repeat("credentials:\n a_PASSWORD: "+key+":"+key+" = \"a b\"\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 150,
		},
		{
			"oversized children of an envelope", func(n int) string {
				return "credentials:\n" + strings.Repeat(" "+strings.Repeat("x_", 70)+"PASSWORD: s\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 500,
		},
		{
			"repeated nested camel case envelopes", func(n int) string { return strings.Repeat("credentials:\n user:dbPassword: s\n", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			small, large := tt.gen(tt.n/4), tt.gen(tt.n)
			tt.run(small) // warm up
			// The best of three runs of each size, so that a pause of the
			// scheduler is not read as growth.
			best := func(in string) time.Duration {
				d := scrubTime(func() { tt.run(in) })
				for range 2 {
					d = min(d, scrubTime(func() { tt.run(in) }))
				}
				return d
			}
			ds, dl := best(small), best(large)
			// Linear growth gives 4x; quadratic 16x. Allow generous noise
			// and a floor for timer resolution.
			if dl > 10*ds && dl > 100*time.Millisecond {
				t.Errorf("scrubbing %d repeats took %v and %d repeats took %v, which is more than linear", tt.n/4, ds, tt.n, dl)
			}
			if dl > 4*time.Second {
				t.Errorf("scrubbing %d repeats took %v, want well under 4s", tt.n, dl)
			}
		})
	}
}
