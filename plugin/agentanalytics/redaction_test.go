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
	"errors"
	"fmt"
	"net/url"
	"regexp"
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
		// Dotted names whose last segments are a credential name of several
		// words.
		{"stripe.api.key", true},
		{"jwt.private.key", true},
		{"app.client.secret", true},
		// Credential names of several words written with dots.
		{"api.key", true},
		{"auth.code", true},
		{"code.verifier", true},
		{"google.access.id", true},
		{"x.goog.api.key", true},
		{"aws.secret.access.key", true},
		{"secret.access.key", true},
		{"x.amz.credential", true},
		{"API.Key", true},
		// Acronyms in a row, such as the "AIAPI" of OpenAIAPIKey, that end in
		// the first word of a credential name.
		{"OpenAIAPIKey", true},
		{"AWSAPIKey", true},
		{"GCPAPIKey", true},
		{"BearerToken", true},
		// A token named after a page that is not a pagination cursor.
		{"homepage-token", true},
		{"fb-page-token", true},
		{"fb_page_token", true},
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
		{"TokenCount", false},
		{"maxAPIKeyCount", false},
		// Pagination tokens are not credentials.
		{"next_page_token", false},
		{"page_token", false},
		{"nextPageToken", false},
		{"NextPageToken", false},
		{"next-page-token", false},
		{"prev_page_token", false},
		{"previous_page_token", false},
		{"x.next_page_token", false},
		{"app:next_page_token", false},
		// Ordinary words in a payload.
		{"method_signature", false},
		{"design", false},
		{"mypassword", false},
		{"code", false},
		{"status.code", false},
		{"access.id", false},
		{"api.key.file", false},
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
		// A token named after a page that is not a pagination cursor.
		{"homepage token is redacted", "homepage-token=SECRET", "homepage-token=[REDACTED]"},
		{"page access token is redacted", "fb-page-token=SECRET fb_page_token: SECRET", "fb-page-token=[REDACTED] fb_page_token:[REDACTED]"},
		{"previous page cursors are kept", "prev_page_token=p1 previous-page-token=p2 NextPageToken=p3", "prev_page_token=p1 previous-page-token=p2 NextPageToken=p3"},
		// Acronyms in a row, such as the "AIAPI" of OpenAIAPIKey.
		{"acronyms in a row", "OpenAIAPIKey=SECRET AWSAPIKey=SECRET GCPAPIKey=SECRET", "OpenAIAPIKey=[REDACTED] AWSAPIKey=[REDACTED] GCPAPIKey=[REDACTED]"},
		{"acronyms in a row after a prefix", "x AWSAPIKey: SECRET", "x AWSAPIKey:[REDACTED]"},
		{"camel case bearer token", "x BearerToken=SECRET", "x BearerToken=[REDACTED]"},
		{"camel case counts are kept", "TokenCount=5 maxAPIKeyCount=3", "TokenCount=5 maxAPIKeyCount=3"},
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
		"stripe.api.key":    "SECRET",
		"jwt.private.key":   "SECRET",
		"OpenAIAPIKey":      "SECRET",
		"AWSAPIKey":         "SECRET",
		"fb-page-token":     "SECRET",
		"homepage-token":    "SECRET",
		"prompt_tokens":     12,
		"total_tokens":      30,
		"next_page_token":   "page-2",
		"prev_page_token":   "page-0",
		"model":             "gemini",
	}
	got, _ := sanitizeValue(in, 1<<20, true)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("sanitizeValue returned %T, want a map", got)
	}
	for _, k := range []string{"DB_PASSWORD", "GITHUB_TOKEN", "OPENAI_API_KEY", "dbPassword", "oauth2Config.code", "x_auth_token", "stripe.api.key", "jwt.private.key", "OpenAIAPIKey", "AWSAPIKey", "fb-page-token", "homepage-token"} {
		if m[k] != redactedValue {
			t.Errorf("%s = %v, want %q", k, m[k], redactedValue)
		}
	}
	for k, want := range map[string]any{"next_page_token": "page-2", "prev_page_token": "page-0", "model": "gemini"} {
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
		OpenAIAPIKey  string
		AWSAPIKey     string
		GCPAPIKey     string
		NextPageToken string
		PromptTokens  int
		TokenCount    int
	}
	got, _ := sanitizeValue(config{DBPassword: "SECRET", GithubToken: "SECRET", OpenAIAPIKey: "SECRET", AWSAPIKey: "SECRET", GCPAPIKey: "SECRET", NextPageToken: "p2", PromptTokens: 5, TokenCount: 7}, 1<<20, true)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("sanitizeValue returned %T, want a map", got)
	}
	for _, k := range []string{"DBPassword", "GithubToken", "OpenAIAPIKey", "AWSAPIKey", "GCPAPIKey"} {
		if m[k] != redactedValue {
			t.Errorf("credential field %s = %v, want %q", k, m[k], redactedValue)
		}
	}
	if m["NextPageToken"] != "p2" {
		t.Errorf("NextPageToken = %v, want p2", m["NextPageToken"])
	}
	if m["TokenCount"] == redactedValue {
		t.Errorf("TokenCount was redacted: %v", m)
	}
}

func TestSmartTruncate_RedactsUserinfoPassword(t *testing.T) {
	// url.Userinfo implements fmt.Stringer and publishes its password via
	// String(); the formatter must redact it instead. The user name is
	// redacted with it: it may be the credential itself, as the token of a
	// token-authenticated git remote is.
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
	if strings.Contains(s, "alice") {
		t.Errorf("SmartTruncate() = %q, user name kept", s)
	}
	if want := `"` + redactedValue + `"`; s != want {
		t.Errorf("SmartTruncate() = %q, want %q", s, want)
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
		{"user name only in a url", &url.URL{Scheme: "https", Host: "example.com", User: url.User("alice")}},
		{"token as the user name", &url.URL{Scheme: "https", Host: "github.com", Path: "/o/r", User: url.User("SYNTHETIC_SECRET")}},
		{"token as the user name with a password", url.UserPassword("SYNTHETIC_SECRET", "x-oauth-basic")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, _, err := SmartTruncate(tt.in, 1000)
			if err != nil {
				t.Fatalf("SmartTruncate() error = %v", err)
			}
			s := string(got)
			if strings.Contains(s, "SYNTHETIC_SECRET") {
				t.Errorf("SmartTruncate() = %s, credential leaked", s)
			}
			if strings.Contains(s, "alice") {
				t.Errorf("SmartTruncate() = %s, user name kept", s)
			}
			if !strings.Contains(s, redactedValue) {
				t.Errorf("SmartTruncate() = %s, want %s", s, redactedValue)
			}
		})
	}
	t.Run("user name only", func(t *testing.T) {
		got, _, err := SmartTruncate(url.User("alice"), 1000)
		if want := `"` + redactedValue + `"`; err != nil || string(got) != want {
			t.Errorf("SmartTruncate(url.User) = (%s, %v), want %s", got, err, want)
		}
	})
	t.Run("empty user name", func(t *testing.T) {
		got, _, err := SmartTruncate(url.User(""), 1000)
		if err != nil || string(got) != `""` {
			t.Errorf(`SmartTruncate(url.User("")) = (%s, %v), want ""`, got, err)
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

// TestCanonicalizeCredentialMarker_EscapedUnicodeWhitespace checks that the
// \u and \x escapes of whitespace beyond ASCII are decoded, so that a scheme
// is told apart from its credential as in the decoded text, that other
// escapes beyond ASCII are left alone, and that a credential after such an
// escape in JSON text is kept out on the raw text and error paths.
func TestCanonicalizeCredentialMarker_EscapedUnicodeWhitespace(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`Bearer\u00a0X`, "Bearer\u00a0X"},
		{`Bearer\u3000X`, "Bearer\u3000X"},
		{`Bearer\u2028X`, "Bearer\u2028X"},
		{`Bearer\U2029X`, "Bearer\u2029X"},
		{`Bearer\xa0X`, "Bearer\u00a0X"},
		{`Bearer\\u00a0X`, `Bearer\` + "\u00a0X"},
		{`Bearer\u0020X`, "Bearer X"},
		{`caf\u00e9`, `caf\u00e9`},
		{`caf\xe9`, `caf\xe9`},
		{`100%a0`, `100%a0`},
	} {
		if got := canonicalizeCredentialMarker(tt.in); got != tt.want {
			t.Errorf("canonicalizeCredentialMarker(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
	// c3ludGhldGljOnNlY3JldA== is the base64 of "synthetic:secret", a basic
	// credential.
	for _, tt := range []struct{ in, secret string }{
		{`{"h":"Bearer\u00a0SYNTHETIC_SECRET"}`, "SYNTHETIC_SECRET"},
		{`{"h":"Bearer\u3000SYNTHETIC_SECRET"}`, "SYNTHETIC_SECRET"},
		{`{"h":"Bearer\u2028SYNTHETIC_SECRET"}`, "SYNTHETIC_SECRET"},
		{`{"h":"caf\u00e9 Bearer\u00a0SYNTHETIC_SECRET"}`, "SYNTHETIC_SECRET"},
		{`{"password":"x","h":"Bearer\u00a0SYNTHETIC_SECRET"}`, "SYNTHETIC_SECRET"},
		{"{\"password\":\"x\",\"h\":\"Bearer\u2028SYNTHETIC_SECRET\"}", "SYNTHETIC_SECRET"},
		{`auth: Bearer\xa0SYNTHETIC_SECRET`, "SYNTHETIC_SECRET"},
		{`{"h":"Basic\u00a0c3ludGhldGljOnNlY3JldA=="}`, "c3ludGhldGljOnNlY3JldA"},
		{`auth: Basic\u3000c3ludGhldGljOnNlY3JldA==`, "c3ludGhldGljOnNlY3JldA"},
	} {
		if got, _ := sanitizeRawText(tt.in); strings.Contains(got, tt.secret) {
			t.Errorf("sanitizeRawText(%q) = %q, which holds the credential", tt.in, got)
		}
		b, _, err := SmartTruncate(errors.New(tt.in), 1000)
		if err != nil || strings.Contains(string(b), tt.secret) {
			t.Errorf("SmartTruncate(errors.New(%q)) = (%s, %v), which holds the credential", tt.in, b, err)
		}
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
		{"page token suffix", prefix + "next_page_token", false},
		{"page token after another word", prefix + "page_token", true},
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
	// longName makes the name of an object field longer than maxNestedKeyLen.
	longName := strings.Repeat("a", 150)
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
		{"oversized children in one word, then a spaced = and an object", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = {"value": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then a spaced : and an object", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` : {"value": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then a delimiter and an object", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `: {"value": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then an object of two fields", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `={"value": "x", "other": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then an object without spaces", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `={"value":"x","other":"SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then a nested object", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = {"a": {"b": "x"}, "c": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then a spaced = and an array", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = ["x", "SYNTHETIC_SECRET"]`},
		{"oversized children in one word, then an object that is not closed", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = {"value": "SYNTHETIC_SECRET"`},
		{"oversized children, then a delimiter and an object", "credentials:\n a_PASSWORD: " + longKey + ": " + longKey + ": " + longKey + `: {"value": "SYNTHETIC_SECRET"}`},
		{"oversized key, then a spaced = and an object", longKey + ` = {"value": "x", "other": "SYNTHETIC_SECRET"}`},
		{"short children in one word, then a quoted multiword value", "credentials:\n a_PASSWORD: x_TOKEN:y_PASSWORD=\"prefix SYNTHETIC_SECRET suffix\""},
		{"short child in one word, then an object of two fields", "credentials:\n a_PASSWORD: x_PASSWORD:{\"value\":\"x\", \"other\":\"SYNTHETIC_SECRET\"}"},
		{"bare value that opens a quoted multiword value", `password=a="prefix SYNTHETIC_SECRET suffix"`},
		{"bare value that opens an object", `token=a:{"value": "x", "other": "SYNTHETIC_SECRET"}`},
		{"bare value that opens an escaped quoted multiword value", `password=a=\"prefix SYNTHETIC_SECRET suffix\"`},
		{"dotted names that are not credentials, then an oversized key", "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", 3) + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"dotted names that are not credentials, then a dotted key", "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", 3) + dottedKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"dotted name that is not a credential, then a short key", "credentials:\n a_PASSWORD: x.next_page_token:x_PASSWORD=\"prefix SYNTHETIC_SECRET suffix\""},
		{"name that is not a credential, then an oversized key", "a_PASSWORD: next_page_token:" + longKey + `="prefix SYNTHETIC_SECRET suffix"`},
		{"run whose last value is an object", `token=a_TOKEN:b_TOKEN:{"` + longName + `": "x", "value": "SYNTHETIC_SECRET"}`},
		{"run whose last value is an object of numbers", `token=a_TOKEN:b_TOKEN:{"` + longName + `": 1, "value": "SYNTHETIC_SECRET"}`},
		{"run whose last value is a Python dict", `token=a_TOKEN:b_TOKEN:{'` + longName + `': 'x', 'value': 'SYNTHETIC_SECRET'}`},
		{"run whose last value is an array", `token=a_TOKEN:b_TOKEN:[{"` + longName + `": "x", "value": "SYNTHETIC_SECRET"}]`},
		{"oversized children whose last value is an object", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + `:{"` + longName + `": "x", "value": "SYNTHETIC_SECRET"}`},
		{"oversized children in one word, then an object with a brace in a string", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = {"a": "}", "other": "SYNTHETIC_SECRET"}`},
		{"dotted name that is not a credential, then an oversized key and an object", "credentials:\n a_PASSWORD: x.next_page_token:" + longKey + ` = {"b": "SYNTHETIC_SECRET"}`},
		{"dotted name that is not a credential, then a dotted oversized key and an object", "credentials:\n a_PASSWORD: x.next_page_token:y." + longKey + ` = {"b": "SYNTHETIC_SECRET"}`},
		{"name that is not a credential, then an oversized key and a spaced quoted value", "next_page_token:" + longKey + ` = "prefix SYNTHETIC_SECRET suffix"`},
		{"label, then an oversized key and an object", "label:" + longKey + ` = {"b": "SYNTHETIC_SECRET"}`},
		{"child label, then an oversized key and an object", "credentials:\n a_PASSWORD: label:" + longKey + ` = {"b": "SYNTHETIC_SECRET"}`},
		{"child name that is not a credential, then an oversized key and a spaced quoted value", "credentials:\n a_PASSWORD: next_page_token:" + longKey + ` = "prefix SYNTHETIC_SECRET suffix"`},
		{"child name that is not a credential, then oversized keys and an object", "credentials:\n a_PASSWORD: next_page_token:" + longKey + ":" + longKey + ` = {"b": "SYNTHETIC_SECRET"}`},
		{"name that is not a credential, then a short key and an object", `token=next_page_token:x_TOKEN = {"b": "SYNTHETIC_SECRET"}`},
		{"child label, then a short key and an object", "credentials:\n a_PASSWORD: label:x_TOKEN = {\"b\": \"SYNTHETIC_SECRET\"}"},
		{"child name that is not a credential, then a short key and a spaced quoted value", "credentials:\n a_PASSWORD: next_page_token:x_TOKEN = \"prefix SYNTHETIC_SECRET suffix\""},
		{"dotted name that is not a credential, then a key in one word", "x.next_page_token:x_TOKEN=SYNTHETIC_SECRET"},
		{"dotted label, then a camel case key in one word", "foo.bar:myPassword=SYNTHETIC_SECRET"},
		{"camel case name that is not a credential, then a camel case key", "nextPageToken:dbPassword=SYNTHETIC_SECRET"},
		{"run of names that are not credentials, then a key", strings.Repeat("next_page_token:", 50) + "x_TOKEN=SYNTHETIC_SECRET"},
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

// TestScrubTextPatterns_OpenedValueEnds checks where the redaction of a value
// that opens a quoted value, an object or an array after a delimiter ends: at
// the closing quote or bracket, not before it, which would leave the rest of
// the value without a key, and not after it, which would drop the text that
// follows. A value that is opened and closed in the bare value is redacted
// with it, and an object that is not closed is redacted to the end.
func TestScrubTextPatterns_OpenedValueEnds(t *testing.T) {
	longKey := strings.Repeat("x_", 70) + "PASSWORD"
	run := longKey + ":" + longKey + ":" + longKey
	tests := []struct {
		name, in, want string
	}{
		{"quoted value", `password=a="b c" kept`, `password=[REDACTED] kept`},
		{"object", `token=a:{"b": "c", "d": "e"}, kept=1`, `token=[REDACTED], kept=1`},
		{"array", `token=a=["b", "c"] kept`, `token=[REDACTED] kept`},
		{"quoted value closed in the bare value", `password=a:"b",kept=1`, `password=[REDACTED],kept=1`},
		{"object closed in the bare value", `password=a:{"b":"c"},kept=1`, `password=[REDACTED],kept=1`},
		{"oversized run, then a spaced = and an object", "credentials:\n a_PASSWORD: " + run + ` = {"b": "c"} kept`, "credentials:\n a_PASSWORD:[REDACTED] kept"},
		{"oversized run, then an object in the word", "credentials:\n a_PASSWORD: " + run + `={"b": "c"} kept`, "credentials:\n a_PASSWORD:[REDACTED] kept"},
		{"oversized run, then an array after a delimiter", "credentials:\n a_PASSWORD: " + run + `: ["b", "c"] kept`, "credentials:\n a_PASSWORD:[REDACTED] kept"},
		{"oversized run, then an object that is not closed", "credentials:\n a_PASSWORD: " + run + ` = {"b": "c" kept`, "credentials:\n a_PASSWORD:[REDACTED]"},
		// A quote at the end of an oversized run whose last key is followed
		// by a spaced delimiter opens a value, even when a letter follows
		// the quoted value; read from the delimiter instead, the value
		// would end at its first space.
		{"oversized run, then a quote that ends the word", "credentials:\n a_PASSWORD: " + run + `=" = b c"x`, "credentials:\n a_PASSWORD:[REDACTED]x"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := scrubTextPatterns(tt.in); got != tt.want {
				t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestAllowedKeyAt_AgreesWithTextKeyValTokenAtRe checks that allowedKeyAt,
// which reads only the name of a key, the delimiter and the start of the
// value, tells what textKeyValTokenAtRe and textKeyAllowed tell from the
// whole pair, for names, quotes, whitespace, delimiters and value starts of
// every kind that they read.
func TestAllowedKeyAt_AgreesWithTextKeyValTokenAtRe(t *testing.T) {
	names := []string{"token", "TOKEN", "next_page_token", "x_PASSWORD", "dbPassword", "temp:a", "temp:a:b", "user:token", "oauth2.code", "oauth2.x.code", "xtoken", "temptoken", "a.b", "label"}
	quotes := []string{"", `"`, `\"`, `'`}
	spaces := []string{"", " ", "\t", "\n", "\x01", " \x0b ", "\u00a0", "\u3000"}
	delims := []string{"=", ":", ""}
	values := []string{"v", "", " ", ",", ";", "&x", " ,x", `"a b"`, "'a", "\x01", "\t\x02y"}
	allowed := 0
	for _, name := range names {
		for _, q := range quotes {
			for _, sp := range spaces {
				for _, d := range delims {
					for _, v := range values {
						s := name + q + sp + d + v
						loc := textKeyValTokenAtRe.FindStringSubmatchIndex(s)
						want := loc != nil && textKeyAllowed(s[loc[2]:loc[3]])
						if got := allowedKeyAt("x "+s, 2); got != want {
							t.Errorf("allowedKeyAt(%q) = %v, want %v", s, got, want)
						}
						if want {
							allowed++
						}
					}
				}
			}
		}
	}
	if allowed == 0 {
		t.Error("no pair was allowed, so nothing was compared")
	}
}

// TestKeyNameRegexps_FindThePairsOfPairRegexps checks that keyValFinder,
// which reads the name and delimiter of a key before its value, finds the
// pairs that regular expressions of whole pairs find: nextKeyName the first
// pair that textKeyNameRe followed by a value finds, and textKeyNameCamelAtRe
// followed by a value start the pairs that textKeyValBareAtRe, which pairAt
// reads them with, matches, with the same key and delimiter.
func TestKeyNameRegexps_FindThePairsOfPairRegexps(t *testing.T) {
	pairRe := regexp.MustCompile(textKeyNameRe.String() + `[` + delimSpaceClass + `]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
	prefixes := []string{"", "x ", "x_", "(", "db"}
	// The "temp:" names that end in "temp" make the colon of a second
	// "temp:" key their delimiter (see nextKeyName).
	names := []string{"token", "Token", "next_page_token", "x_PASSWORD", "dbPassword", "APIKey", "temp:a", "temp:a:b", "temp:x&temp", "temp:a;temp", "oauth2.code", "oauth2.x.code", "label", "sig"}
	quotes := []string{"", `"`, `\"`, `'`}
	spaces := []string{"", " ", "\n", "\x01", "\u00a0", "\u3000"}
	delims := []string{"=", ":", ""}
	values := []string{"v", "", " ", ",", "&x", "&y=v", ";b=v", " ,x", `"a b"`, "'a", "\x01", " ,token=v"}
	compared := 0
	for _, p := range prefixes {
		for _, name := range names {
			for _, q := range quotes {
				for _, sp := range spaces {
					for _, d := range delims {
						for _, v := range values {
							s := p + name + q + sp + d + v
							want, got := pairRe.FindStringSubmatchIndex(s), nextKeyName(s, 0)
							if (want == nil) != (got == nil) || want != nil && (want[0] != got[0] || want[2] != got[2] || want[3] != got[3] || want[7] != got[7]) {
								t.Errorf("nextKeyName(%q) = %v, want the key and delimiter of %v", s, got, want)
							}
							if p != "" {
								// The anchored expressions read s[i:] only.
								continue
							}
							// Where a credential name may start in the name:
							// at it, at a capital and after "_" or ":".
							for i := range len(name) {
								if i > 0 && (s[i] < 'A' || s[i] > 'Z') && s[i-1] != '_' && s[i-1] != ':' {
									continue
								}
								want := textKeyValBareAtRe.FindStringSubmatchIndex(s[i:])
								got := textKeyNameCamelAtRe.FindStringSubmatchIndex(s[i:])
								found := got != nil && textValueStartRe.MatchString(s[i+got[1]:])
								if (want != nil) != found || found && (want[3] != got[3] || want[7] != got[7]) {
									t.Errorf("key name at %d of %q = %v (value %v), want the key and delimiter of %v", i, s, got, found, want)
								}
								if want != nil {
									compared++
								}
							}
						}
					}
				}
			}
		}
	}
	if compared == 0 {
		t.Error("no pair was found, so nothing was compared")
	}
}

// TestKeyValFinder_SkipsKeysWithoutValues checks that a key with no value
// after its delimiter is passed over, as a regular expression of a whole pair
// passes over it, on each of the paths of keyValFinder.
func TestKeyValFinder_SkipsKeysWithoutValues(t *testing.T) {
	const at = "token: , password=x"
	f := keyValFinder{whole: at}
	if loc := f.find(at); loc == nil || at[loc[2]:loc[3]] != "password" || loc[8] < 0 || at[loc[8]:loc[9]] != "x" {
		t.Errorf("keyValFinder.find(%q) = %v, want the password pair", at, loc)
	}
	const later = "x token: , password=x"
	if loc := nextKeyName(later, 0); loc == nil || later[loc[2]:loc[3]] != "password" {
		t.Errorf("nextKeyName(%q) = %v, want the password key", later, loc)
	}
	// A key without a value whose delimiter is the colon of a "temp:" key
	// that starts inside it, which has a value.
	for _, inner := range []string{"temp:x&temp:&y=v", "temp:a;temp:;b=v"} {
		if loc := nextKeyName(inner, 0); loc == nil || loc[2] != 7 || inner[loc[1]:] != "v" {
			t.Errorf("nextKeyName(%q) = %v, want the key at 7 that ends before v", inner, loc)
		}
	}
	const camel = "dbPassword: , myToken=x"
	if loc := findCamelKeyName(camel, len(camel)); loc == nil || camel[loc[2]:loc[3]] != "Token" {
		t.Errorf("findCamelKeyName(%q) = %v, want the Token of myToken", camel, loc)
	}
	// A pair without a value, as pairAt returns for a key whose value it
	// does not read, is left in place up to its delimiter, whatever its key.
	var brackets openBrackets
	var quotes openQuotes
	if got, end := redactTextKeyValue("token= rest", []int{0, 6, 0, 5, 5, 5, 5, 6, -1, -1}, &brackets, &quotes); got != "token=" || end != 6 {
		t.Errorf(`redactTextKeyValue(pair without a value) = (%q, %d), want ("token=", 6)`, got, end)
	}
	// pairAt does not take a pair whose key ends elsewhere than the key it
	// was given, here the first token for the second.
	const two = "token=v token=w"
	if loc := pairAt(two, []int{8, 14, 8, 13, 13, 13, 13, 14}, 0, textKeyValBareAtRe); loc[8] >= 0 {
		t.Errorf("pairAt(%q, second key, read at the first) = %v, want no value", two, loc)
	}
	if loc := pairAt(two, []int{8, 14, 8, 13, 13, 13, 13, 14}, 8, textKeyValBareAtRe); loc[8] < 0 || two[loc[8]:loc[9]] != "w" {
		t.Errorf("pairAt(%q, second key) = %v, want its value w", two, loc)
	}
}

// TestScrubTextPatterns_KeysThatAreNotCredentials checks that a key that
// textKeyAllowed does not accept, whose value keyValFinder does not read, is
// written as it is, with its value, and that a key in its value is read.
func TestScrubTextPatterns_KeysThatAreNotCredentials(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"next_page_token=abc123 kept", "next_page_token=abc123 kept"},
		{"x.next_page_token:abc, method_signature=xyz;nextPageToken: q", "x.next_page_token:abc, method_signature=xyz;nextPageToken: q"},
		{strings.Repeat("next_page_token:", 3) + "abc", strings.Repeat("next_page_token:", 3) + "abc"},
		{"next_page_token:x_TOKEN=SECRET kept", "next_page_token:x_TOKEN=[REDACTED] kept"},
		{"nextPageToken:dbPassword=SECRET kept", "nextPageToken:dbPassword=[REDACTED] kept"},
	} {
		if got := scrubTextPatterns(tt.in); got != tt.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestScrubTextPatterns_TempKeysInsideTempKeys checks that a "temp:" key
// that starts inside a "temp:" key without a value, whose delimiter is the
// colon of the inner key (see nextKeyName), is redacted on every path.
func TestScrubTextPatterns_TempKeysInsideTempKeys(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"temp:x&temp:&y=SYNTHETIC_SECRET", "temp:x&temp:&y=[REDACTED]"},
		{"temp:a;temp:;b=SYNTHETIC_SECRET", "temp:a;temp:;b=[REDACTED]"},
		{"temp:temp:&a=SYNTHETIC_SECRET kept", "temp:temp:&a=[REDACTED] kept"},
		{"state temp:cache&temp:&session_id=SYNTHETIC_SECRET", "state temp:cache&temp:&session_id=[REDACTED]"},
		// Each key has a value, which is redacted, as for any "temp:" key.
		{"temp:x=1&temp:y=SYNTHETIC_SECRET", "temp:x=[REDACTED]&temp:y=[REDACTED]"},
	} {
		if got := scrubTextPatterns(tt.in); got != tt.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
		}
		if got, _ := sanitizeRawText(tt.in); got != tt.want {
			t.Errorf("sanitizeRawText(%q) = %q, want %q", tt.in, got, tt.want)
		}
		b, _, err := SmartTruncate(errors.New(tt.in), 1000)
		var got string
		if err != nil || json.Unmarshal(b, &got) != nil || got != tt.want {
			t.Errorf("SmartTruncate(errors.New(%q)) = (%s, %v), want %q", tt.in, b, err, tt.want)
		}
	}
}

// checkRedactionPaths checks that scrubTextPatterns, sanitizeRawText and the
// error serializer, SmartTruncate of errors.New, all write want for in, and
// that scrubbing want again changes nothing.
func checkRedactionPaths(t *testing.T, in, want string) {
	t.Helper()
	if got := scrubTextPatterns(in); got != want {
		t.Errorf("scrubTextPatterns(%q) = %q, want %q", in, got, want)
	}
	if again := scrubTextPatterns(want); again != want {
		t.Errorf("scrubbing %q again gave %q, want it unchanged", want, again)
	}
	if got, _ := sanitizeRawText(in); got != want {
		t.Errorf("sanitizeRawText(%q) = %q, want %q", in, got, want)
	}
	b, _, err := SmartTruncate(errors.New(in), 1000)
	var got string
	if err != nil || json.Unmarshal(b, &got) != nil || got != want {
		t.Errorf("SmartTruncate(errors.New(%q)) = (%s, %v), want %q", in, b, err, want)
	}
}

// TestScrubTextPatterns_ValuesOpenedAfterSpacedDelimiters checks that a bare
// value that ends in or is followed after horizontal whitespace by a
// delimiter, followed by whitespace (including Unicode spaces and vertical
// tab) and a quoted value, an object or an array, is redacted with that
// value, that a quote after a delimiter in or after a bare value that nothing
// closes is redacted to the end of the text, and that a quoted value that the
// closing quote of the string around the pair follows is redacted whole, on
// every path.
func TestScrubTextPatterns_ValuesOpenedAfterSpacedDelimiters(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`password=a= "prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix"`, `password=[REDACTED]`},
		{`password=a= 'prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix'`, `password=[REDACTED]`},
		{"password=a=\t\"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\" kept", "password=[REDACTED] kept"},
		{`password=a= \"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\" kept`, `password=[REDACTED] kept`},
		{`token=a: {"value": "R7_SYNTHETIC_CREDENTIAL_9327"}`, `token=[REDACTED]`},
		{`token=a: {"value": "R7_SYNTHETIC_CREDENTIAL_9327"}, kept=1`, `token=[REDACTED], kept=1`},
		{`token=a: ["prefix", "SYNTHETIC_R7_CREDENTIAL"]`, `token=[REDACTED]`},
		{`token=a: ["prefix", "R7_SYNTHETIC_CREDENTIAL_9327"]`, `token=[REDACTED]`},
		{`token=a:  ["prefix", "R7_SYNTHETIC_CREDENTIAL_9327"] kept`, `token=[REDACTED] kept`},
		{`token=a: {"value": "R7_SYNTHETIC_CREDENTIAL_9327`, `token=[REDACTED]`},
		{`token=a: ["prefix", "R7_SYNTHETIC_CREDENTIAL_9327`, `token=[REDACTED]`},
		{`password=x-token: "prefix R7_SYNTHETIC_CREDENTIAL_9327`, `password=[REDACTED]`},
		{`password=x-token:"prefix R7_SYNTHETIC_CREDENTIAL_9327`, `password=[REDACTED]`},
		{`password=a:'prefix R7_SYNTHETIC_CREDENTIAL_9327`, `password=[REDACTED]`},
		// Spaced inner delimiter (whitespace before and/or after ":" or "=").
		{`password=a = "prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix"`, `password=[REDACTED]`},
		{`password=abc :"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix"`, `password=[REDACTED]`},
		{`token=a : {"value": "R7_SYNTHETIC_CREDENTIAL_9327"}`, `token=[REDACTED]`},
		{`password=a = "unclosed R7_SYNTHETIC_CREDENTIAL_9327`, `password=[REDACTED]`},
		{`failed: password: R7_SYNTHETIC_CREDENTIAL_9327 : retry`, `failed: password:[REDACTED] : retry`},
		{`password="R7_SYNTHETIC_CREDENTIAL_9327" :`, `password="[REDACTED]" :`},
		{`token=dGVzdA== "user@example.com" logged in`, `token=[REDACTED] logged in`},
		// Unicode delimiter whitespace and control characters.
		{"password=\u00a0\"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\"", "password=\"[REDACTED]\""},
		{"password=\u3000\"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\"", "password=\"[REDACTED]\""},
		{"token\u00a0= R7_SYNTHETIC_CREDENTIAL_9327", "token=[REDACTED]"},
		{"token=a\u00a0b kept", "token=[REDACTED] kept"},
		{"password=a=\u00a0\"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\"", "password=[REDACTED]"},
		{"password=a=\v\"prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix\"", "password=[REDACTED]"},
		// The value of a quoted key, as JSON text writes it.
		{`failed: {"password": a= "prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix"}`, `failed: {"password":"[REDACTED]"}`},
		{`failed: {"password": a = "prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix"}`, `failed: {"password":"[REDACTED]"}`},
		{`failed: {"token": a:{"value": "R7_SYNTHETIC_CREDENTIAL_9327"}}`, `failed: {"token":"[REDACTED]"}`},
		// Inner quoted value immediately followed by the enclosing string's
		// closing quote, which may end a sentence, and negative controls where
		// adjacent quotes do not close an enclosing string.
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password=[REDACTED]"`},
		{`failed: "password=a = "R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password=[REDACTED]"`},
		{`failed: 'password=a: 'R7_SYNTHETIC_CREDENTIAL_9327''`, `failed: 'password=[REDACTED]'`},
		{`failed: 'password=a = 'R7_SYNTHETIC_CREDENTIAL_9327''`, `failed: 'password=[REDACTED]'`},
		{`failed: \"password=a: \"R7_SYNTHETIC_CREDENTIAL_9327\"\"`, `failed: \"password=[REDACTED]\"`},
		{`failed: "password=a = "R7_SYNTHETIC_CREDENTIAL_9327"" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL"`, `failed: "password=[REDACTED]" DB_PASSWORD="[REDACTED]"`},
		{`failed: "password=a = "R7_SYNTHETIC_CREDENTIAL_9327"" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `failed: "password=[REDACTED]" DB_PASSWORD=[REDACTED]`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327"".`, `failed: "password=[REDACTED]".`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327""!`, `failed: "password=[REDACTED]"!`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327""?`, `failed: "password=[REDACTED]"?`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327"". DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `failed: "password=[REDACTED]". DB_PASSWORD=[REDACTED]`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327""! DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `failed: "password=[REDACTED]"! DB_PASSWORD=[REDACTED]`},
		{`failed: "password=a: "R7_SYNTHETIC_CREDENTIAL_9327""? DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `failed: "password=[REDACTED]"? DB_PASSWORD=[REDACTED]`},
		{`failed: "password="!R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password="[REDACTED]""`},
		// The inner quoted value starts with whitespace.
		{`failed: "password="   R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password="[REDACTED]""`},
		{`failed: "password: "   R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password:"[REDACTED]""`},
		{`failed: "password = "   R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password="[REDACTED]""`},
		{`failed: "password=a = "   R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "password=[REDACTED]"`},
		{`failed: "password=a: "   R7_SYNTHETIC_CREDENTIAL_9327"".`, `failed: "password=[REDACTED]".`},
		{`failed: "password=a: "   R7_SYNTHETIC_CREDENTIAL_9327 and more words""`, `failed: "password=[REDACTED]"`},
		{`failed: 'password=a: '   R7_SYNTHETIC_CREDENTIAL_9327''`, `failed: 'password=[REDACTED]'`},
		{`failed: "{"password": "   R7_SYNTHETIC_CREDENTIAL_9327"}"`, `failed: "{"password":"[REDACTED]"}"`},
		{`failed: \"password=\"   R7_SYNTHETIC_CREDENTIAL_9327\"\"`, `failed: \"password=\"[REDACTED]\"\"`},
		{`failed: "password="   R7_SYNTHETIC_CREDENTIAL_9327"" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `failed: "password="[REDACTED]"" DB_PASSWORD=[REDACTED]`},
		{`failed: "password="   R7_SYNTHETIC_CREDENTIAL_9327"" for user "bob"`, `failed: "password="[REDACTED]"" for user "bob"`},
		{`failed: "password=abc-token: " for user "bob""`, `failed: "password=[REDACTED] " for user "bob""`},
		{`got "password=abc-token=", want "x"`, `got "password=[REDACTED], want "x"`},
		{`password=a: "x""y"`, `password=[REDACTED] "x""y"`},
		{`failed: "password=" "x"`, `failed: "password=" "x"`},
		{`failed: "password=" and "y""`, `failed: "password=" and "y""`},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
}

// TestScrubTextPatterns_UnclosedQuotedValues checks that a value that starts
// with a quote that nothing closes is redacted to the end of the text on
// every path, as no closing quote tells where its words end, and that a quote
// that closes the string around the pair, or that only punctuation follows,
// is not read as one, nor paired with a later string's opening quote.
func TestScrubTextPatterns_UnclosedQuotedValues(t *testing.T) {
	longKey := strings.Repeat("x_", 70) + "PASSWORD"
	run := longKey + ":" + longKey + ":" + longKey
	for _, tt := range []struct{ in, want string }{
		{`DB_PASSWORD="prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix`, `DB_PASSWORD=[REDACTED]`},
		{`DB_PASSWORD="   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `DB_PASSWORD=[REDACTED]`},
		{`dbPassword='prefix R7_SYNTHETIC_CREDENTIAL_9327`, `dbPassword=[REDACTED]`},
		{`password="prefix R7_SYNTHETIC_CREDENTIAL_9327`, `password=[REDACTED]`},
		{`token: 'prefix R7_SYNTHETIC_CREDENTIAL_9327`, `token:[REDACTED]`},
		{`DB_PASSWORD=\"prefix R7_SYNTHETIC_CREDENTIAL_9327`, `DB_PASSWORD=[REDACTED]`},
		{`DB_PASSWORD=\'prefix R7_SYNTHETIC_CREDENTIAL_9327`, `DB_PASSWORD=[REDACTED]`},
		{`err="failed: DB_PASSWORD="prefix R7_SYNTHETIC_CREDENTIAL_9327`, `err="failed: DB_PASSWORD=[REDACTED]`},
		{"credentials:\n a_PASSWORD: " + run + ` = "prefix R7_SYNTHETIC_CREDENTIAL_9327`, "credentials:\n a_PASSWORD:[REDACTED]"},
		{"credentials:\n a_PASSWORD: " + run + `: 'prefix R7_SYNTHETIC_CREDENTIAL_9327`, "credentials:\n a_PASSWORD:[REDACTED]"},
		// Unclosed spaced quotes under envelope and nested keys.
		{`credentials: token = "   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials: token=[REDACTED]`},
		{`credentials: token = '   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials: token=[REDACTED]`},
		{`credentials: token = \"   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials: token=[REDACTED]`},
		{`credentials: token = \'   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials: token=[REDACTED]`},
		{`credentials: password="correct R7_SYNTHETIC_CREDENTIAL_9327 battery"`, `credentials: password="[REDACTED]"`},
		{"credentials:\n  token: \"   prefix R7_SYNTHETIC_CREDENTIAL_9327", "credentials:\n  token:[REDACTED]"},
		// The quote closes the string that the pair is in, including when
		// another quoted pair follows on the same line.
		{`failed: "password=" in config`, `failed: "password=" in config`},
		{`failed: "password=" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=" DB_PASSWORD="[REDACTED]"`},
		{`failed: "password=" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "password=" DB_PASSWORD=[REDACTED]`},
		{`failed: "password=   " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=   " DB_PASSWORD="[REDACTED]"`},
		{`failed: "password=abc:" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=[REDACTED] DB_PASSWORD="[REDACTED]"`},
		{`failed: "password=abc:" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "password=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: "password=abc: " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=[REDACTED] " DB_PASSWORD="[REDACTED]"`},
		{`failed: 'password=abc: ' dbPassword='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'password=[REDACTED] ' dbPassword='[REDACTED]'`},
		{`failed: "credentials: token = " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "credentials:[REDACTED] = " DB_PASSWORD="[REDACTED]"`},
		{"failed: \"credentials= " + run + ` = " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "credentials=[REDACTED] = " DB_PASSWORD="[REDACTED]"`},
		{"failed: \"credentials= " + run + `=" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "credentials=[REDACTED] DB_PASSWORD="[REDACTED]"`},
		{"failed: \"credentials= " + run + ` = " in config`, `failed: "credentials=[REDACTED] = " in config`},
		{"failed: \"a_PASSWORD: " + run + `:[REDACTED] = " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "a_PASSWORD:[REDACTED] = " DB_PASSWORD="[REDACTED]"`},
		{"failed: \"a_PASSWORD: " + run + `:[REDACTED] = " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "a_PASSWORD:[REDACTED] = " DB_PASSWORD=[REDACTED]`},
		{`parsing "password=SECRETP-token: ": invalid syntax`, `parsing "password=[REDACTED] ": invalid syntax`},
		{`parsing 'password=SECRETP-token: ': invalid syntax`, `parsing 'password=[REDACTED] ': invalid syntax`},
		{`failed: \"password=SECRETP-token: \" in config`, `failed: \"password=[REDACTED] \" in config`},
		// Lexical quote state: prose apostrophes, cross-quotes, escaped
		// backslashes and line breaks.
		{`couldn't connect: dbPassword='   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `couldn't connect: dbPassword=[REDACTED]`},
		{`err="can't connect" dbPassword='   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `err="can't connect" dbPassword=[REDACTED]`},
		{`x='a "b' DB_PASSWORD="   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `x='a "b' DB_PASSWORD=[REDACTED]`},
		{`dir="C:\\" DB_PASSWORD="   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `dir="C:\\" DB_PASSWORD=[REDACTED]`},
		{"\\\"Authorization: R7_SYNTHETIC_CREDENTIAL_9327\\\"\nDB_PASSWORD=\\\"   prefix R7_SYNTHETIC_CREDENTIAL_9327", "\\\"Authorization: [REDACTED]\nDB_PASSWORD=[REDACTED]"},
		{"msg=\"line1\nDB_PASSWORD=\"   prefix R7_SYNTHETIC_CREDENTIAL_9327", "msg=\"line1\nDB_PASSWORD=[REDACTED]"},
		// Apostrophes inside a single-quoted string, within a word, after
		// a plural or a non-ASCII letter, or inside a redacted value, do not
		// close the string.
		{`failed: 'I don't know' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'I don't know' DB_PASSWORD='[REDACTED]'`},
		{`users' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `users' DB_PASSWORD=[REDACTED]`},
		{`l'été DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `l'été DB_PASSWORD=[REDACTED]`},
		{`failed: 'token={"a":"don't"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token="don't"' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token="[REDACTED]"' DB_PASSWORD=[REDACTED]`},
		{`failed: '{"dbPassword": "don't"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: '{"dbPassword":"[REDACTED]"}' DB_PASSWORD=[REDACTED]`},
		{`failed: 'token={"a":"x'y","b":"don't"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'token=[REDACTED]' DB_PASSWORD='[REDACTED]'`},
		{`failed: 'token=don't' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token=l'été' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token=abc:don't' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'password=it's-a-secret' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'password=[REDACTED] DB_PASSWORD=[REDACTED]`},
		// Nor does a quote of the outer kind that is not an apostrophe,
		// inside a quoted value, object or array that starts the redacted
		// value and closes inside it: the quotes in it are its own.
		{`failed: 'token={"a":"'x"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token={"a":"x'"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token=["'"]' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: 'token="'x"' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'token="[REDACTED]"' DB_PASSWORD=[REDACTED]`},
		{`failed: '{"dbPassword": "'x"}' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: '{"dbPassword":"[REDACTED]"}' DB_PASSWORD=[REDACTED]`},
		{`failed: 'token={"a":"'x"}' DB_PASSWORD='R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'token=[REDACTED] DB_PASSWORD='[REDACTED]'`},
		// A quote in a redacted value leaves no string open, and balanced
		// quotes inside a redacted value inside an outer string preserve the
		// outer string's active quote state.
		{`password=a"b x "token=SECRETP-token: " in config`, `password=[REDACTED] x "token=[REDACTED] " in config`},
		{`failed: "credentials={"user":"alice","pass":"x"}" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "credentials=[REDACTED]" DB_PASSWORD=[REDACTED]`},
		{`failed: "password="x"" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "password="[REDACTED]"" DB_PASSWORD=[REDACTED]`},
		{`failed: "credentials: token = "x"" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "credentials: token="[REDACTED]"" DB_PASSWORD=[REDACTED]`},
		{`failed: 'password='x'' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'password='[REDACTED]'' DB_PASSWORD=[REDACTED]`},
		{`failed: "token={"a":"b"} password=" in config`, `failed: "token=[REDACTED] password=" in config`},
		// A quote inside the bare value, which a value may follow in the
		// same word, does not close the string.
		{`x 'password=abc:':{"k": "R7_SYNTHETIC_CREDENTIAL_9327"}`, `x 'password=[REDACTED]`},
		{`failed: "password=abc:{\"k\":\"R7_SYNTHETIC_CREDENTIAL_9327\"}"`, `failed: "password=[REDACTED]`},
		{`token=abc:'s note password='SECRET VALUE' rest`, `token=[REDACTED] note password='[REDACTED]' rest`},
		{`token=abc:'s scope missing; the user's request`, `token=[REDACTED] scope missing; the user's request`},
		// Nor does it open a value when its closing quote opens the value of
		// a later key, sensitive or not.
		{`token=abc:'s label=' x'`, `token=[REDACTED] label=' x'`},
		{`token=abc:'s id=' 123' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `token=[REDACTED] id=' 123' password=[REDACTED]`},
		{`token=abc:' id=' 123' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `token=[REDACTED] id=' 123' password=[REDACTED]`},
		{`token=abc:'s note=' rest`, `token=[REDACTED] note=' rest`},
		// The quote after a Python string prefix or a short command-line
		// option, before a letter, opens a string, but an apostrophe before
		// the ending of a possessive or a contraction, after a prime or after
		// a letter that follows a "-" inside a word does not.
		{`b'note=123' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `b'note=123' password=[REDACTED]`},
		{`r'path=/a/1' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `r'path=/a/1' password=[REDACTED]`},
		{`f'id={1}' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `f'id={1}' password=[REDACTED]`},
		{`f'd={1}' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `f'd={1}' password=[REDACTED]`},
		{`curl -H'X-Custom: 1' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `curl -H'X-Custom: 1' password=[REDACTED]`},
		{`Rb'x=1' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `Rb'x=1' password=[REDACTED]`},
		{`B's note password='   R7_SYNTHETIC_CREDENTIAL_9327`, `B's note password=[REDACTED]`},
		{`x -I'm told password='   R7_SYNTHETIC_CREDENTIAL_9327`, `x -I'm told password=[REDACTED]`},
		{`Smith-O'Neil said password='   R7_SYNTHETIC_CREDENTIAL_9327`, `Smith-O'Neil said password=[REDACTED]`},
		{`x b' password='   R7_SYNTHETIC_CREDENTIAL_9327`, `x b' password=[REDACTED]`},
		{`a', b'. password='   R7_SYNTHETIC_CREDENTIAL_9327`, `a', b'. password=[REDACTED]`},
		// The value of a quoted key or bearer token, including when an outer
		// quote closes right after the prefix or pairs forward with a later
		// string's opening quote.
		{`failed: {"password": "prefix R7_SYNTHETIC_CREDENTIAL_9327`, `failed: {"password":"[REDACTED]"`},
		{`x "temp:cache": "prefix R7_SYNTHETIC_CREDENTIAL_9327 suffix}`, `x "temp:cache":"[REDACTED]"`},
		{`failed: 'error at "token": ' user=alice`, `failed: 'error at "token": ' user=alice`},
		{`failed: 'error at "token": ' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'error at "token": ' DB_PASSWORD='[REDACTED]'`},
		{`failed: 'error at "token": ' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'error at "token": ' DB_PASSWORD=[REDACTED]`},
		{`got "token": " in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`failed: "Bearer " api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "Bearer " api_key="[REDACTED]"`},
		{`Bearer " in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer " in config, api_key="[REDACTED]"`},
		{`failed: "Bearer "R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "Bearer [REDACTED]"`},
		// The later string's value starts with a character that a key does
		// not, or the value before it ends in a key and a delimiter.
		{`got "token": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="$2b$12$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="/R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="-R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="★R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="{\"k\":\"R7_SYNTHETIC_CREDENTIAL_9327\"}"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key=""R7_SYNTHETIC_CREDENTIAL_9327""`, `got "token": " in config, api_key="[REDACTED]"`},
		{`got "token": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `got "token": " in config, api_key="[REDACTED]" DB_PASSWORD=[REDACTED]`},
		{`got "password": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `got "password": " in config, api_key="[REDACTED]"`},
		{`got "password": " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "password": " in config, api_key="[REDACTED]"`},
		{`got "password": " in config, api_key=""R7_SYNTHETIC_CREDENTIAL_9327""`, `got "password": " in config, api_key="[REDACTED]"`},
		{`Bearer " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer " in config, api_key="[REDACTED]"`},
		{`Bearer " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer " in config, api_key="[REDACTED]"`},
		{`Bearer " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `Bearer " in config, api_key="[REDACTED]" DB_PASSWORD=[REDACTED]`},
		{`failed: "Bearer "   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "Bearer [REDACTED]`},
		{`failed: "Bearer "   R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "Bearer [REDACTED]"`},
		{`password=" in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `password="[REDACTED]"`},
		{`password=" in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `password="[REDACTED]"`},
		{`password=" in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `password="[REDACTED]" DB_PASSWORD=[REDACTED]`},
		{`api_key=""R7_SYNTHETIC_CREDENTIAL_9327""`, `api_key="[REDACTED]"`},
		{`failed: {"token":"R7_SYNTHETIC_CREDENTIAL_9327""api_key":"SYNTHETIC_R7_CREDENTIAL"}`, `failed: {"token":"[REDACTED]""api_key":"[REDACTED]"}`},
		{`failed: '{"token": "R7_SYNTHETIC_CREDENTIAL_9327"}' DB_PASSWORD='   SYNTHETIC_R7_CREDENTIAL'`, `failed: '{"token":"[REDACTED]"}' DB_PASSWORD='[REDACTED]'`},
		{`failed: '{"token": "R7_SYNTHETIC_CREDENTIAL_9327"}' DB_PASSWORD='   SYNTHETIC_R7_CREDENTIAL`, `failed: '{"token":"[REDACTED]"}' DB_PASSWORD=[REDACTED]`},
		// Only punctuation follows the quote, including after an oversized key run.
		{`failed: {Password:"}`, `failed: {Password:[REDACTED]}`},
		{`failed: {Password:SECRETP-token: '}`, `failed: {Password:[REDACTED] '}`},
		{"failed: {" + longKey + `: "}`, "failed: {" + longKey + `:[REDACTED]}`},
		{"failed: {a_PASSWORD: " + run + `:[REDACTED] = "}`, `failed: {a_PASSWORD:[REDACTED] = "}`},
		{"failed: {a_PASSWORD: " + run + `:[REDACTED] = ""}`, `failed: {a_PASSWORD:[REDACTED] = ""}`},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
	adjJSON := `{"token":"R7_SYNTHETIC_CREDENTIAL_9327""api_key":"SYNTHETIC_R7_CREDENTIAL"}`
	if got, want := scrubTextPatterns(adjJSON), `{"token":"[REDACTED]""api_key":"[REDACTED]"}`; got != want {
		t.Errorf("scrubTextPatterns(%q) = %q, want %q", adjJSON, got, want)
	}
	if raw, _ := sanitizeRawText(adjJSON); strings.Contains(raw, "R7_SYNTHETIC_CREDENTIAL_9327") || strings.Contains(raw, "SYNTHETIC_R7_CREDENTIAL") {
		t.Errorf("sanitizeRawText(%q) = %q, leaked credential", adjJSON, raw)
	}
	// Quote-led multiline input where authHeaderRe consumes the closing quote
	// on line 1 (sanitizeRawText and SmartTruncate fail closed as unparseable
	// JSON; scrubTextPatterns resets quote state at the newline).
	quoteLedAuth := "\"Authorization: R7_SYNTHETIC_CREDENTIAL_9327\"\nDB_PASSWORD=\"   prefix R7_SYNTHETIC_CREDENTIAL_9327"
	if got, want := scrubTextPatterns(quoteLedAuth), "\"Authorization: [REDACTED]\nDB_PASSWORD=[REDACTED]"; got != want {
		t.Errorf("scrubTextPatterns(%q) = %q, want %q", quoteLedAuth, got, want)
	}
	if raw, _ := sanitizeRawText(quoteLedAuth); strings.Contains(raw, "R7_SYNTHETIC_CREDENTIAL_9327") {
		t.Errorf("sanitizeRawText(%q) = %q, leaked credential", quoteLedAuth, raw)
	}
	if b, _, err := SmartTruncate(errors.New(quoteLedAuth), 1000); err != nil || strings.Contains(string(b), "R7_SYNTHETIC_CREDENTIAL_9327") {
		t.Errorf("SmartTruncate(errors.New(%q)) = (%s, %v), leaked credential", quoteLedAuth, b, err)
	}
}

// TestScrubTextPatterns_DottedOAuthNames checks that the value of a dotted
// OAuth name of any number of words, which isCompoundSensitiveKey accepts as
// a map key, is redacted in text on every path, wherever the name starts in
// its dotted words and whatever key or value surrounds it, and that a dotted
// name that isCompoundSensitiveKey does not accept is kept.
func TestScrubTextPatterns_DottedOAuthNames(t *testing.T) {
	longKey := strings.Repeat("x_", 70) + "PASSWORD"
	for _, tt := range []struct{ in, want string }{
		{"oauth2Config.extra.code=R7_SYNTHETIC_CREDENTIAL_9327", "oauth2Config.extra.code=[REDACTED]"},
		{"status.oauth2Config.extra.code=R7_SYNTHETIC_CREDENTIAL_9327", "status.oauth2Config.extra.code=[REDACTED]"},
		{"a.b.oauth_client.sub.code=R7_SYNTHETIC_CREDENTIAL_9327 kept", "a.b.oauth_client.sub.code=[REDACTED] kept"},
		{"oauth2.x.code: R7_SYNTHETIC_CREDENTIAL_9327", "oauth2.x.code:[REDACTED]"},
		{"x OAuth2.Extra.CODE = R7_SYNTHETIC_CREDENTIAL_9327", "x OAuth2.Extra.CODE=[REDACTED]"},
		{`x oauth2Config.extra.code="R7_SYNTHETIC_CREDENTIAL_9327 tail"`, `x oauth2Config.extra.code="[REDACTED]"`},
		{`x "status.oauth2Config.extra.code": "R7_SYNTHETIC_CREDENTIAL_9327"`, `x "status.oauth2Config.extra.code":"[REDACTED]"`},
		{"oauth..oauth.x.code=R7_SYNTHETIC_CREDENTIAL_9327", "oauth..oauth.x.code=[REDACTED]"},
		{"oauth..x.code=R7_SYNTHETIC_CREDENTIAL_9327", "oauth..x.code=[REDACTED]"},
		{"a.oauth.b..c.code=R7_SYNTHETIC_CREDENTIAL_9327", "a.oauth.b..c.code=[REDACTED]"},
		{"status.oauth2Config..extra.code=R7_SYNTHETIC_CREDENTIAL_9327", "status.oauth2Config..extra.code=[REDACTED]"},
		{"credentials: oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:[REDACTED]"},
		{"credentials: status.oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:[REDACTED]"},
		{"password=secretPrefix.oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327", "password=[REDACTED]"},
		{"password=secretPrefix.dbPassword: R7_SYNTHETIC_CREDENTIAL_9327", "password=[REDACTED]"},
		{"credentials:\n  oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327\n  region: us", "credentials:\n  oauth2Config.extra.code:[REDACTED]\n  region: us"},
		{"credentials:\n a_PASSWORD: " + longKey + ".oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:\n a_PASSWORD:[REDACTED]"},
		// Names that isCompoundSensitiveKey does not accept.
		{"xoauth.extra.code=kept", "xoauth.extra.code=kept"},
		{"oauth2Config.extra.codes=kept", "oauth2Config.extra.codes=kept"},
		{"status.extra.code=kept", "status.extra.code=kept"},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
	// keyValFinder finds the name from its word that starts with "oauth".
	for _, tt := range []struct{ in, key string }{
		{"oauth2.code=v", "oauth2.code"},
		{"oauth2.x.code=v", "oauth2.x.code"},
		{"oauth..x.code=v", "oauth..x.code"},
		{"a.oauth2.x.code=v", "oauth2.x.code"},
		{"oauth2Config.extra.code=v", "oauth2Config.extra.code"},
		{"status.oauth2Config.extra.code=v", "oauth2Config.extra.code"},
	} {
		f := keyValFinder{whole: tt.in}
		if loc := f.find(tt.in); loc == nil || tt.in[loc[2]:loc[3]] != tt.key || loc[8] < 0 || tt.in[loc[8]:loc[9]] != "v" {
			t.Errorf("keyValFinder.find(%q) = %v, want the key %q and the value v", tt.in, loc, tt.key)
		}
	}
}

// TestScrubTextPatterns_DottedCredentialNames checks that the value of a
// credential name of several words written with dots, such as api.key or
// aws.secret.access.key, which isSensitiveKey accepts as a map key, is
// redacted in text on every path: after the name alone or after other dotted
// words, under a parent key, and at the end of the value of another key,
// whose value it is then redacted with.
func TestScrubTextPatterns_DottedCredentialNames(t *testing.T) {
	for _, name := range []string{
		"stripe.api.key", "jwt.private.key", "aws.secret.access.key", "x.goog.api.key",
		"auth.code", "code.verifier", "google.access.id", "api.key", "API.Key", "x.stripeApi.key",
	} {
		checkRedactionPaths(t, name+"=R7_SYNTHETIC_CREDENTIAL_9327", name+"=[REDACTED]")
		checkRedactionPaths(t, "x "+name+": R7_SYNTHETIC_CREDENTIAL_9327 y", "x "+name+":[REDACTED] y")
	}
	for _, tt := range []struct{ in, want string }{
		{"credentials:\n  stripe.api.key: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:\n  stripe.api.key:[REDACTED]"},
		{"credentials:\n  my.stripe_api.key: R7_SYNTHETIC_CREDENTIAL_9327\n  region: us", "credentials:\n  my.stripe_api.key:[REDACTED]\n  region: us"},
		{`x "api.key": "R7_SYNTHETIC_CREDENTIAL_9327"`, `x "api.key":"[REDACTED]"`},
		{"password=my.stripe.api.key: R7_SYNTHETIC_CREDENTIAL_9327 tcp", "password=[REDACTED] tcp"},
		{"password=my.stripe_api.key: R7_SYNTHETIC_CREDENTIAL_9327 tcp", "password=[REDACTED] tcp"},
		// A value that ends in a compound key name that the text patterns
		// do not read, as in api..key, is redacted to its end.
		{"password=R7_SYNTHETIC_CREDENTIAL_9327.API..key: dial tcp", "password=[REDACTED] dial tcp"},
		// Dotted names that are not credential names are kept.
		{"my.api.key.file=/path access.id=7 status.code=200", "my.api.key.file=/path access.id=7 status.code=200"},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
}

// TestScrubTextPatterns_DottedNamesAgreeWithMapKeys checks that a dotted name
// has its value redacted in text exactly when isSensitiveKey redacts the
// value of a map key of that name, for names of up to four words: dotted
// OAuth names, and names made of the words of credential names and others,
// as in stripe.api.key or x.goog.api.key. It leaves out two kinds of names
// that text reads otherwise: those with an empty dotted word that are not
// OAuth names, as in api..key, which text does not read as a credential
// name, and a pagination cursor name after a dot, as in fb.page_token, of
// which text reads only the words after the last dot (see isPageTokenName).
func TestScrubTextPatterns_DottedNamesAgreeWithMapKeys(t *testing.T) {
	var names []string
	var build func(words, lasts []string, prefix string, n int)
	build = func(words, lasts []string, prefix string, n int) {
		for _, last := range lasts {
			names = append(names, prefix+last)
		}
		if n > 0 {
			for _, w := range words {
				build(words, lasts, prefix+w+".", n-1)
			}
		}
	}
	build([]string{"", "oauth", "OAuth2", "oauth_client", "status", "xoauth", "code"}, []string{"code", "Code", "codes", "xcode"}, "", 3)
	build([]string{"stripe", "jwt", "aws", "x", "goog", "google", "api", "private", "secret", "access", "auth", "code"}, []string{"key", "Key", "id", "verifier", "code", "token"}, "", 3)
	sensitive := 0
	for _, name := range names {
		want := isSensitiveKey(name)
		if want {
			sensitive++
		}
		for _, in := range []string{name + "=R7_SYNTHETIC_CREDENTIAL_9327", "x " + name + ": R7_SYNTHETIC_CREDENTIAL_9327"} {
			if got := !strings.Contains(scrubTextPatterns(in), "R7_SYNTHETIC_CREDENTIAL_9327"); got != want {
				t.Errorf("scrubTextPatterns(%q) redacted the value: %v, want %v as for the map key", in, got, want)
			}
			b, _, err := SmartTruncate(errors.New(in), 1000)
			if got := err == nil && !strings.Contains(string(b), "R7_SYNTHETIC_CREDENTIAL_9327"); got != want {
				t.Errorf("SmartTruncate(errors.New(%q)) = (%s, %v), redacted the value: %v, want %v as for the map key", in, b, err, got, want)
			}
		}
	}
	if sensitive == 0 || sensitive == len(names) {
		t.Errorf("%d of the %d names are sensitive, want some of each", sensitive, len(names))
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
		{"prefixed dotted oauth key", "secretPrefix.oauth2Config.extra.code", false, valueEndingInKeyName, 13},
		{"prefixed dotted camel key", "secretPrefix.dbPassword", false, valueEndingInKeyName, 13},
		{"dotted multi-word key", "my.stripe_api.key", false, valueEndingInKeyName, 3},
		{"dotted credential name on a line", "stripe.api.key", true, nestedKey, 7},
		{"dotted name that text does not read", "R7.API..key", false, valueEndingInKeyName, 8},
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
		{
			// Each dot starts a name that is not a credential and whose
			// value runs to the end of the text.
			"oversized run of dotted names that are not credentials", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"oversized run of dotted names that are not credentials through an error", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 1600,
		},
		{
			"oversized run of dotted names, then a dotted key", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", n) + "x.y_" + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			// No key before the run reads it whole, so each key in it is
			// found and passed over on its own.
			"bare run of dotted names that are not credentials", func(n int) string { return strings.Repeat("x.next_page_token:", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"bare run of dotted names that are not credentials through an error", func(n int) string { return strings.Repeat("x.next_page_token:", n) },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 1<<20) }, 1600,
		},
		{
			"bare run of names that are not credentials", func(n int) string { return strings.Repeat("next_page_token:", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"bare run of low-precision names", func(n int) string { return strings.Repeat("method_signature=", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"bare run of camel case names that are not credentials", func(n int) string { return strings.Repeat("nextPageToken:", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			// Each "temp:" key runs on to the colon of the next.
			"bare run of temp keys", func(n int) string { return strings.Repeat("temp:x&", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			// None of them has a value, and the search goes on from the
			// byte after each.
			"bare run of temp keys without values", func(n int) string { return strings.Repeat("temp:&", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"repeated values that open quoted values", func(n int) string { return strings.Repeat(`password=a="b c" `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated values that open objects", func(n int) string { return strings.Repeat(`token=a:{"b": "c"} `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated values with delimiters before an unclosed quote", func(n int) string {
				return `password=a:'b ` + strings.Repeat(`token=a:b:c:d `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated values opened after spaced delimiters", func(n int) string { return strings.Repeat(`password=a= "b c" token=a: {"d": "e"} `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quotes that close the strings around pairs", func(n int) string { return strings.Repeat(`parsing "password=a-token: ": x `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated dotted OAuth names", func(n int) string { return strings.Repeat("x status.oauth2Config.extra.code=v ", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			// Each word starts a dotted OAuth name, which no value follows.
			"bare run of dotted OAuth words without a value", func(n int) string { return strings.Repeat("oauth.a.", n) + "code=" },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"bare run of hyphenated OAuth words without a value", func(n int) string { return strings.Repeat("oauth-", n) + "x.code=," },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"bare run of underscored OAuth words without a value", func(n int) string { return strings.Repeat("oauth_", n) + "x.code=" },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"bare run of double-dotted OAuth words without a value", func(n int) string { return strings.Repeat("oauth..", n) + "x.code=" },
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			// Each word starts a dotted OAuth name, which runs on to the
			// end of the run, where no ".code" ends it.
			"oversized run of dotted OAuth words", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("oauth.", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"oversized run of double-dotted OAuth words", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("oauth..", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"oversized run of dotted OAuth words through an error", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("oauth.", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 8000,
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
