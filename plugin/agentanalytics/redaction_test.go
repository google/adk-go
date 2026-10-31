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
		// The same names after dotted words, which are read whole.
		{"fb.page_token", true},
		{"home.page_token", true},
		{"x.y.page_token", true},
		{"fb.page.token", true},
		{"x.fb-page-token", true},
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
		// Pagination cursor names written with dots, after dotted words or
		// not.
		{"page.token", false},
		{"next.page.token", false},
		{"prev.page.token", false},
		{"previous.page.token", false},
		{"x.next.page.token", false},
		{"Next.Page.Token", false},
		{"x.nextPageToken", false},
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
		// The same names after dotted words: the whole dotted name says
		// whether it is a cursor, not only its last words.
		{"dotted page access tokens are redacted", "fb.page_token=SECRET home.page_token: SECRET x.y.page_token=SECRET", "fb.page_token=[REDACTED] home.page_token:[REDACTED] x.y.page_token=[REDACTED]"},
		{"dotted page words are redacted", "fb.page.token=SECRET x.fb-page-token: SECRET", "fb.page.token=[REDACTED] x.fb-page-token:[REDACTED]"},
		{"dotted cursors are kept", "page.token=c1 next.page.token: c2 prev.page.token=c3 previous.page.token: c4 x.next.page.token=c5", "page.token=c1 next.page.token: c2 prev.page.token=c3 previous.page.token: c4 x.next.page.token=c5"},
		{"dotted cursor before a dotted page access token", "next.page.token=c1 fb.page_token=SECRET", "next.page.token=c1 fb.page_token=[REDACTED]"},
		{"dotted page access token before a dotted cursor", "fb.page_token=SECRET next.page.token=c1", "fb.page_token=[REDACTED] next.page.token=c1"},
		{"dotted page access token after a credential", "DB_PASSWORD=SECRET home.page_token=SECRET", "DB_PASSWORD=[REDACTED] home.page_token=[REDACTED]"},
		{"dotted page access token in truncated json", `err: {"fb.page_token": "SECRET"`, `err: {"fb.page_token":"[REDACTED]"`},
		{"dotted cursor in truncated json", `err: {"next.page.token": "c1"`, `err: {"next.page.token": "c1"`},
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
		"DB_PASSWORD":         "SECRET",
		"GITHUB_TOKEN":        "SECRET",
		"OPENAI_API_KEY":      "SECRET",
		"dbPassword":          "SECRET",
		"oauth2Config.code":   "SECRET",
		"x_auth_token":        "SECRET",
		"stripe.api.key":      "SECRET",
		"jwt.private.key":     "SECRET",
		"OpenAIAPIKey":        "SECRET",
		"AWSAPIKey":           "SECRET",
		"fb-page-token":       "SECRET",
		"homepage-token":      "SECRET",
		"fb.page_token":       "SECRET",
		"home.page_token":     "SECRET",
		"prompt_tokens":       12,
		"total_tokens":        30,
		"next_page_token":     "page-2",
		"prev_page_token":     "page-0",
		"page.token":          "page-1",
		"next.page.token":     "page-3",
		"prev.page.token":     "page-4",
		"previous.page.token": "page-5",
		"x.next.page.token":   "page-6",
		"model":               "gemini",
	}
	got, _ := sanitizeValue(in, 1<<20, true)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("sanitizeValue returned %T, want a map", got)
	}
	for _, k := range []string{"DB_PASSWORD", "GITHUB_TOKEN", "OPENAI_API_KEY", "dbPassword", "oauth2Config.code", "x_auth_token", "stripe.api.key", "jwt.private.key", "OpenAIAPIKey", "AWSAPIKey", "fb-page-token", "homepage-token", "fb.page_token", "home.page_token"} {
		if m[k] != redactedValue {
			t.Errorf("%s = %v, want %q", k, m[k], redactedValue)
		}
	}
	for k, want := range map[string]any{"next_page_token": "page-2", "prev_page_token": "page-0", "page.token": "page-1", "next.page.token": "page-3", "prev.page.token": "page-4", "previous.page.token": "page-5", "x.next.page.token": "page-6", "model": "gemini"} {
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

// TestSanitizeValue_DottedPageTokenNames checks that a map key, and a struct
// field that a json tag names, which is a dotted name that ends in a page
// token is redacted exactly when isSensitiveKey says so: the access token of a
// page, as fb.page_token is, is a credential, and a pagination cursor, written
// with underscores or with dots, is not.
func TestSanitizeValue_DottedPageTokenNames(t *testing.T) {
	type pageTokens struct {
		FB       string `json:"fb.page_token"`
		Home     string `json:"home.page_token"`
		Dotted   string `json:"fb.page.token"`
		Page     string `json:"page.token"`
		Next     string `json:"next.page.token"`
		Prev     string `json:"prev.page.token"`
		Previous string `json:"previous.page.token"`
		XNext    string `json:"x.next.page.token"`
		XCursor  string `json:"x.next_page_token"`
	}
	want := map[string]any{
		"fb.page_token":       redactedValue,
		"home.page_token":     redactedValue,
		"fb.page.token":       redactedValue,
		"page.token":          "page-1",
		"next.page.token":     "page-2",
		"prev.page.token":     "page-3",
		"previous.page.token": "page-4",
		"x.next.page.token":   "page-5",
		"x.next_page_token":   "page-6",
	}
	for k, v := range want {
		if got := isSensitiveKey(k); got != (v == redactedValue) {
			t.Errorf("isSensitiveKey(%q) = %v, want %v", k, got, v == redactedValue)
		}
	}
	values := map[string]any{
		"fb.page_token": "SECRET", "home.page_token": "SECRET", "fb.page.token": "SECRET",
		"page.token": "page-1", "next.page.token": "page-2", "prev.page.token": "page-3",
		"previous.page.token": "page-4", "x.next.page.token": "page-5", "x.next_page_token": "page-6",
	}
	for name, in := range map[string]any{
		"map": values,
		"struct": pageTokens{
			FB: "SECRET", Home: "SECRET", Dotted: "SECRET", Page: "page-1", Next: "page-2",
			Prev: "page-3", Previous: "page-4", XNext: "page-5", XCursor: "page-6",
		},
	} {
		got, _ := sanitizeValue(in, 1<<20, true)
		m, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("%s: sanitizeValue returned %T, want a map", name, got)
		}
		for k, v := range want {
			if m[k] != v {
				t.Errorf("%s: %s = %v, want %v", name, k, m[k], v)
			}
		}
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
		{"oversized key, quoted value that a letter runs into", longKey + `="prefix SYNTHETIC_SECRET"x`},
		{"oversized key, quoted value that a quote runs into", longKey + `="prefix SYNTHETIC_SECRET""`},
		{"oversized key, quoted value after : that a letter runs into", longKey + `: "prefix SYNTHETIC_SECRET"x`},
		{"oversized key, single-quoted value that a letter runs into", longKey + `='prefix SYNTHETIC_SECRET'x`},
		{"oversized key, spaced = and a quoted value that a letter runs into", longKey + ` = "prefix SYNTHETIC_SECRET"x`},
		{"oversized key, spaced : and a quoted value that a quote runs into", longKey + ` : "prefix SYNTHETIC_SECRET""`},
		{"oversized dotted child, quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + dottedKey + `="prefix SYNTHETIC_SECRET"x`},
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
		{"oversized children in one word, then a quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `="prefix SYNTHETIC_SECRET"x`},
		{"oversized children in one word, then a delimiter and a quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + `: "prefix SYNTHETIC_SECRET"x`},
		{"oversized children in one word, then a spaced = and a quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` = "prefix SYNTHETIC_SECRET"x`},
		{"oversized children in one word, then a spaced : and a quoted value that a quote runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` : "prefix SYNTHETIC_SECRET""`},
		{"oversized children in one word, then a spaced = and a single-quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + ` =  'prefix SYNTHETIC_SECRET'x`},
		{"oversized children apart, then a spaced = and a quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ": " + longKey + ": " + longKey + ` = "prefix SYNTHETIC_SECRET"x`},
		{"oversized children in one word, then a delimiter on the next line and a quoted value that a letter runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + "\n = \"prefix SYNTHETIC_SECRET\"x"},
		{"oversized children in one word, then a delimiter on the next line and a quoted value that a quote runs into", "credentials:\n a_PASSWORD: " + longKey + ":" + longKey + ":" + longKey + "\n : \"prefix SYNTHETIC_SECRET\"\""},
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
		// the quoted value, which the quote runs into: the letter is
		// redacted with the value, as it may be part of it (see
		// runOnValueEnd). Read from the delimiter instead, the value would
		// end at its first space.
		{"oversized run, then a quote that ends the word", "credentials:\n a_PASSWORD: " + run + `=" = b c"x`, "credentials:\n a_PASSWORD:[REDACTED]"},
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

// checkAllRedactionPaths checks, as checkRedactionPaths does, that
// scrubTextPatterns, sanitizeRawText and SmartTruncate of an error write want
// for in, and that scrubbing want again changes nothing, and that
// redactErrorMessage, which scrubs the message of an error before
// error_message holds it, writes want for in too, and for want itself.
func checkAllRedactionPaths(t *testing.T, in, want string) {
	t.Helper()
	checkRedactionPaths(t, in, want)
	if got := redactErrorMessage(in); got != want {
		t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, want)
	}
	if again := redactErrorMessage(want); again != want {
		t.Errorf("redactErrorMessage(%q) = %q, want it unchanged", want, again)
	}
}

// checkNoLeak checks that none of scrubTextPatterns, sanitizeRawText,
// redactErrorMessage and SmartTruncate of an error writes the secret in in.
// It is for texts that the paths write differently, as they do for a JSON
// object that is not valid, which only the text patterns read as pairs.
func checkNoLeak(t *testing.T, in, secret string) {
	t.Helper()
	raw, _ := sanitizeRawText(in)
	smart, _, err := SmartTruncate(errors.New(in), 1000)
	if err != nil {
		t.Errorf("SmartTruncate(errors.New(%q)) error = %v", in, err)
	}
	for _, p := range []struct{ name, got string }{
		{"scrubTextPatterns", scrubTextPatterns(in)},
		{"sanitizeRawText", raw},
		{"redactErrorMessage", redactErrorMessage(in)},
		{"SmartTruncate", string(smart)},
	} {
		if strings.Contains(p.got, secret) {
			t.Errorf("%s(%q) = %q, which still holds the secret", p.name, in, p.got)
		}
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
		// A nested quoted value whose closing quote runs into another quote
		// is read to the end of that run, as that quote may be part of the
		// credential: the run was kept after "[REDACTED]" before (see
		// TestScrubTextPatterns_QuotedCredentialsRunningIntoText).
		{`password=a: "x""y"`, `password=[REDACTED]`},
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

// TestScrubTextPatterns_QuotedCredentialsRunningIntoText checks that a quoted
// credential is redacted whole on every path when what follows its closing
// quote makes the quote read as one that belongs to something else: a letter
// or another quote that runs into it, a credential that is padded with a
// quote or that ends in the name of a key and a delimiter, as
// R8_..._TOKEN= does, whose closing quote then looks like the one that opens
// the value of a later key, a credential that starts with whitespace and has
// several words, whose text reads as the words between two strings, and a
// value in which a delimiter and a quote open a nested quoted value. The text
// that the quote runs into is redacted with the credential, as it may be part
// of it, and never copied after the redacted value, whereas the words between
// two strings, which a second string follows, and the pairs that a quote opens
// the value of, are kept and redacted as before.
func TestScrubTextPatterns_QuotedCredentialsRunningIntoText(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		// A bearer token that its closing quote does not end.
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" expired`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173_PASSWORD:" expired`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173 more_TOKEN=" expired`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173_api_key=" and more`, `Bearer [REDACTED] more`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" "R8_SYNTHETIC_SECRET_4801"`, `Bearer [REDACTED]`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" "R8_SYNTHETIC_SECRET_4802"`, `Bearer [REDACTED]`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired in "prod"`, `Bearer [REDACTED] in "prod"`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801 "suffix; echo "done"`, `Bearer [REDACTED]; echo "done"`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173"expired`, `Bearer [REDACTED]`},
		{`Bearer 'R8_SYNTHETIC_CREDENTIAL_49173'expired`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173"x"y"`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173""second"`, `Bearer [REDACTED]`},
		{`Authorization: Bearer "R8_SYNTHETIC_CREDENTIAL_49173"expired`, `Authorization: [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_CREDENTIAL_49173" expired`, `Bearer [REDACTED] expired`},
		// The value of a quoted or an unquoted key in a shell command or an
		// environment, with the quotes of the shell.
		{`export "DB_PASSWORD"="R8_SYNTHETIC_CREDENTIAL_49173"suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"='R8_SYNTHETIC_CREDENTIAL_49173'suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export 'DB_PASSWORD'="R8_SYNTHETIC_CREDENTIAL_49173"suffix`, `export 'DB_PASSWORD'=[REDACTED]`},
		{`export "DB_PASSWORD" = "R8_SYNTHETIC_CREDENTIAL_49173"suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"="R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" expired`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"="R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" "R8_SYNTHETIC_SECRET_4801"`, `export "DB_PASSWORD"="[REDACTED]"`},
		{`password="R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" "R8_SYNTHETIC_SECRET_4801"`, `password="[REDACTED]"`},
		{`export "DB_PASSWORD" = " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801 "suffix; echo "done"`, `export "DB_PASSWORD"=[REDACTED]; echo "done"`},
		{`cmd='export "DB_PASSWORD"=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801 "suffix; echo "done"'`, `cmd='export "DB_PASSWORD"=[REDACTED]; echo "done"'`},
		{`export DB_PASSWORD="R8_SYNTHETIC_CREDENTIAL_49173"suffix`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD='R8_SYNTHETIC_CREDENTIAL_49173'suffix`, `export DB_PASSWORD=[REDACTED]`},
		{`token="R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" expired`, `token=[REDACTED]`},
		{`password="R8_SYNTHETIC_CREDENTIAL_49173_PASSWORD:" expired`, `password=[REDACTED]`},
		{`password="R8_SYNTHETIC_CREDENTIAL_49173"suffix`, `password=[REDACTED]`},
		{`export "DB_PASSWORD"="R8_SYNTHETIC_CREDENTIAL_49173"; echo ok`, `export "DB_PASSWORD"="[REDACTED]"; echo ok`},
		// The value of a JSON key that is written in text.
		{`failed: "password":"R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" expired`, `failed: "password":"[REDACTED]"`},
		{`failed: "password": "R8_SYNTHETIC_CREDENTIAL_49173"x rest`, `failed: "password":"[REDACTED]" rest`},
		{`failed: "token": "R8_SYNTHETIC_CREDENTIAL_49173"expired`, `failed: "token":"[REDACTED]"`},
		{`failed: "token": "R8_SYNTHETIC_CREDENTIAL_49173_api_key=" expired`, `failed: "token":"[REDACTED]"`},
		{`failed: "token": 'R8_SYNTHETIC_CREDENTIAL_49173'expired`, `failed: "token":"[REDACTED]"`},
		{`failed: 'token': 'R8_SYNTHETIC_CREDENTIAL_49173'expired`, `failed: 'token':[REDACTED]`},
		// A credential of several words that starts with whitespace reads as
		// the words between two strings, but no second string follows it, so
		// it is redacted whole, whatever its closing quote runs into or opens.
		{`export "DB_PASSWORD"=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"=' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export 'DB_PASSWORD'=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `export 'DB_PASSWORD'=[REDACTED]`},
		{`export "DB_PASSWORD" = " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired`, `export "DB_PASSWORD"=[REDACTED]`},
		{`export "DB_PASSWORD"=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801""`, `export "DB_PASSWORD"="[REDACTED]"`},
		{`export "DB_PASSWORD"=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801" expired`, `export "DB_PASSWORD"="[REDACTED]" expired`},
		{`failed: "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `failed: "password":"[REDACTED]"`},
		{`failed: "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired`, `failed: "password":"[REDACTED]"`},
		{`failed: "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired in "prod"`, `failed: "password":"[REDACTED]" in "prod"`},
		{`failed: "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801 "suffix, "user": "bob"`, `failed: "password":"[REDACTED]", "user": "bob"`},
		{`failed: "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801""`, `failed: "password":"[REDACTED]"`},
		{`failed: "token": ' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'suffix`, `failed: "token":"[REDACTED]"`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `Bearer [REDACTED]`},
		{`Bearer ' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'suffix`, `Bearer [REDACTED]`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801""`, `Bearer [REDACTED]`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired`, `Bearer [REDACTED]`},
		{`Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801 more_TOKEN=" expired`, `Bearer [REDACTED]`},
		{`Authorization: Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `Authorization: [REDACTED]`},
		{`password=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `password=[REDACTED]`},
		// The value of a quoted key whose "=" a quote follows at once starts
		// with that quote, so the text in front of the closing quote is not
		// the text between two strings: it is redacted, along with a string
		// that follows it.
		{`export "DB_PASSWORD"=" in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `export "DB_PASSWORD"="[REDACTED]"`},
		{`export "DB_PASSWORD"=" for user "bob"`, `export "DB_PASSWORD"="[REDACTED]"`},
		// After a ":" and whitespace, as in JSON, the same text is the text
		// between two strings, which is kept, and the pair in the second
		// string is redacted on its own.
		{`failed: "password": " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `failed: "password": " in config, api_key="[REDACTED]"`},
		{`failed: "password": " for user "bob"`, `failed: "password": " for user "bob"`},
		// A nested quoted value after a delimiter, whose closing quote runs
		// into a letter or into more quotes, with or without spaces around
		// the delimiter.
		{`password=a= "prefix R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`password=a= "prefix R8_SYNTHETIC_SECRET_4800""`, `password=[REDACTED]`},
		{`token=a: 'prefix R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`token=a: 'prefix R8_SYNTHETIC_SECRET_4800''`, `token=[REDACTED]`},
		{`password=a="prefix R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`token=a:'prefix R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`password=a = "prefix R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`password=a = "prefix R8_SYNTHETIC_SECRET_4800""`, `password=[REDACTED]`},
		{`token=a : 'prefix R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`password=abc :"prefix R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`password=a= "   R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`password=a= \"prefix R8_SYNTHETIC_SECRET_4800\"x`, `password=[REDACTED]`},
		{`password=a: "R8_SYNTHETIC_SECRET_4800""x"`, `password=[REDACTED]`},
		{`password=a:"R8_SYNTHETIC_SECRET_4800""`, `password=[REDACTED]`},
		{`password=a: "R8_SYNTHETIC_SECRET_4800"x`, `password=[REDACTED]`},
		{`credentials: token = "prefix R8_SYNTHETIC_SECRET_4800"x`, `credentials: token=[REDACTED]`},
		{`token=a: "R8_SYNTHETIC_SECRET_4800"x token=b`, `token=[REDACTED] token=[REDACTED]`},
		// What follows the word that the quote runs into is kept.
		{`password=a= "prefix R8_SYNTHETIC_SECRET_4800"x y z`, `password=[REDACTED] y z`},
		{`failed: "password=a: "R8_SYNTHETIC_SECRET_4800"x"`, `failed: "password=[REDACTED]`},
		// The words between two strings are not a credential, and the pair
		// that a closing quote opens the value of is redacted on its own.
		{`got "token": " for user "bob"`, `got "token": " for user "bob"`},
		{`Bearer " for user "bob"`, `Bearer " for user "bob"`},
		{`got "token": " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `got "token": " in config, api_key="[REDACTED]"`},
		{`Bearer " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `Bearer " in config, api_key="[REDACTED]"`},
		{`password=" in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `password="[REDACTED]"`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	// A quote of the other kind after the closing quote, and a JSON object
	// that is not valid, which the paths write differently, as only the text
	// patterns read it as pairs and the others fail closed, are checked for
	// the credential alone.
	for _, tt := range []struct{ in, secret string }{
		{`password=a= "prefix R8_SYNTHETIC_SECRET_4800"'`, "R8_SYNTHETIC_SECRET_4800"},
		{`password=a: "R8_SYNTHETIC_SECRET_4800"'x'`, "R8_SYNTHETIC_SECRET_4800"},
		{`password=a: 'R8_SYNTHETIC_SECRET_4800'"`, "R8_SYNTHETIC_SECRET_4800"},
		{`{"password":"R8_SYNTHETIC_CREDENTIAL_49173"suffix}`, "R8_SYNTHETIC_CREDENTIAL_49173"},
		{`{"password":"R8_SYNTHETIC_CREDENTIAL_49173_TOKEN=" expired}`, "R8_SYNTHETIC_CREDENTIAL_49173"},
		{`{"token": a = "prefix R8_SYNTHETIC_SECRET_4800"x}`, "R8_SYNTHETIC_SECRET_4800"},
		{`{"password": a: 'prefix R8_SYNTHETIC_SECRET_4800''}`, "R8_SYNTHETIC_SECRET_4800"},
		{`{"Authorization": "Bearer \"R8_SYNTHETIC_CREDENTIAL_49173\"expired"}`, "R8_SYNTHETIC_CREDENTIAL_49173"},
		{`failed: {"password": a = "prefix R8_SYNTHETIC_SECRET_4800"x}`, "R8_SYNTHETIC_SECRET_4800"},
		// The last word of a credential of several words that starts with
		// whitespace is the one that a partial redaction leaves in place.
		{`{"password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix}`, "R8_SYNTHETIC_SECRET_4801"},
		{`{"token": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801_TOKEN=" expired}`, "R8_SYNTHETIC_SECRET_4801"},
		{`{"password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801""}`, "R8_SYNTHETIC_SECRET_4801"},
	} {
		checkNoLeak(t, tt.in, tt.secret)
	}
}

// TestScrubTextPatterns_QuotedValueFollowedByKey checks that a quoted value
// that another key and its delimiter follow at once, as the keys of a shell
// word or a log line that joins pairs do, is redacted up to its closing quote
// on every path, and the pair after the quote is read on its own: redacted if
// its key is a credential name, kept if it is not, and kept whole if it is an
// ordinary pair. A quote that a word of another kind follows still runs into
// it, as the padding of base64 text, a bare letter run, digits, a name whose
// delimiter has no value after it, as in tail= or tail:, or one whose ":" is
// that of a port or of a URL do: that word is the end of the credential, and
// is redacted with it.
func TestScrubTextPatterns_QuotedValueFollowedByKey(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`password="benign"token="R8_SYNTHETIC_SECRET_4700" rest`, `password="[REDACTED]"token="[REDACTED]" rest`},
		{`password="R8_SYNTHETIC_SECRET_4700"user="bob" rest`, `password="[REDACTED]"user="bob" rest`},
		{`password="R8_SYNTHETIC_SECRET_4700"token="R8_SYNTHETIC_SECRET_4701"`, `password="[REDACTED]"token="[REDACTED]"`},
		{`password="benign"token=R8_SYNTHETIC_SECRET_4700 rest`, `password="[REDACTED]"token=[REDACTED] rest`},
		{`password='benign'api_key='R8_SYNTHETIC_SECRET_4700'`, `password='[REDACTED]'api_key='[REDACTED]'`},
		{`password="benign"user=bob rest`, `password="[REDACTED]"user=bob rest`},
		{`password="benign"label:"bob" rest`, `password="[REDACTED]"label:"bob" rest`},
		{`password="benign"user="bob"token="R8_SYNTHETIC_SECRET_4700"`, `password="[REDACTED]"user="bob"token="[REDACTED]"`},
		{"password=\"benign\"\ttoken=\"R8_SYNTHETIC_SECRET_4700\"", "password=\"[REDACTED]\"\ttoken=\"[REDACTED]\""},
		{`password="benign"a.b=c token="R8_SYNTHETIC_SECRET_4700"`, `password="[REDACTED]"a.b=c token="[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN="R8_SYNTHETIC_SECRET_4700"`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN="[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN=SYNTHETIC_FOLLOWING_6401 rest`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN=[REDACTED] rest`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN:"SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN:"[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN:'SYNTHETIC_FOLLOWING_6401'`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN:'[REDACTED]'`},
		{`DB_PASSWORD="benign"dbPassword="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"dbPassword="[REDACTED]"`},
		{`DB_PASSWORD="benign"stripe.api.key="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"stripe.api.key="[REDACTED]"`},
		{`OPENAI_API__KEY="benign"stripe.api..key="SYNTHETIC_FOLLOWING_6401"`, `OPENAI_API__KEY="[REDACTED]"stripe.api..key="[REDACTED]"`},
		{`DB_PASSWORD="benign"fb.page_token="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"fb.page_token="[REDACTED]"`},
		{`DB_PASSWORD="benign"next.page.token="cursor-1"`, `DB_PASSWORD="[REDACTED]"next.page.token="cursor-1"`},
		{`Bearer "benign"GITHUB_TOKEN="SYNTHETIC_FOLLOWING_6401"`, `Bearer [REDACTED]GITHUB_TOKEN="[REDACTED]"`},
		{`failed: "password": "benign"GITHUB_TOKEN="SYNTHETIC_FOLLOWING_6401"`, `failed: "password":"[REDACTED]"GITHUB_TOKEN="[REDACTED]"`},
		{"DB_PASSWORD=\"benign\"\nGITHUB_TOKEN=\"R8_SYNTHETIC_SECRET_4700\"", "DB_PASSWORD=\"[REDACTED]\"\nGITHUB_TOKEN=\"[REDACTED]\""},
		// Standalone sensitive keys after a closed quote with whitespace after
		// ":" or "=", and compound credential tails ending in a key suffix.
		{`password="benign"token: R8_SYNTHETIC_SECRET_4700`, `password="[REDACTED]"token:[REDACTED]`},
		{`password="benign"token: "R8_SYNTHETIC_SECRET_4700 R8_SYNTHETIC_SECRET_4701"`, `password="[REDACTED]"token:"[REDACTED]"`},
		{`password="benign"api_key= R8_SYNTHETIC_SECRET_4700`, `password="[REDACTED]"api_key=[REDACTED]`},
		{`Bearer "benign"token: R8_SYNTHETIC_SECRET_4700`, `Bearer [REDACTED]token:[REDACTED]`},
		{`failed: "password": "benign"token: R8_SYNTHETIC_SECRET_4700`, `failed: "password":"[REDACTED]"token:[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_TOKEN:8080`, `password=[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_TOKEN: R8_SYNTHETIC_SECRET_4801`, `password=[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_TOKEN= R8_SYNTHETIC_SECRET_4801`, `password=[REDACTED]`},
		{`Bearer "R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_TOKEN:8080`, `Bearer [REDACTED]`},
		{`failed: "password": "R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_PASSWORD:8080`, `failed: "password":"[REDACTED]"`},
		// The quoted value of a JSON key that another quoted key follows,
		// whether the keys are credential names or not.
		{`failed: {"password":"R8_SYNTHETIC_SECRET_4700""user":"bob"}`, `failed: {"password":"[REDACTED]""user":"bob"}`},
		{`failed: {"user":"bob""password":"R8_SYNTHETIC_SECRET_4700"}`, `failed: {"user":"bob""password":"[REDACTED]"}`},
		{`failed: {"password":"benign""token":"R8_SYNTHETIC_SECRET_4700"}`, `failed: {"password":"[REDACTED]""token":"[REDACTED]"}`},
		// A word that is not a pair runs into the quote: base64 padding, a
		// run of letters, or a name that starts with a digit.
		{`password="abc"ZGVmZ2g==`, `password=[REDACTED]`},
		{`password="abc"def==ghi`, `password=[REDACTED]`},
		{`password="abc"1=b rest`, `password=[REDACTED] rest`},
		{`password="abc"def rest`, `password=[REDACTED] rest`},
		// A word that is a name followed by a delimiter that no value follows
		// at once, or a host or a port, is not a pair either: it is the end
		// of a credential that the quote runs into, whose tail a "=" or a ":"
		// with nothing or a space after it does not make a pair of its own.
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:8080`, `password=[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802= rest`, `password=[REDACTED] rest`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:`, `password=[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802=`, `password=[REDACTED]`},
		{`password='R8_SYNTHETIC_SECRET_4800'R8_SYNTHETIC_TAIL_4802:8080`, `password=[REDACTED]`},
		{`DB_PASSWORD="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802= rest`, `DB_PASSWORD=[REDACTED] rest`},
		{`Bearer "R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:`, `Bearer [REDACTED]`},
		{`Bearer "R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:8080`, `Bearer [REDACTED]`},
		{`failed: "password": "R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:8080`, `failed: "password":"[REDACTED]"`},
		{`export "DB_PASSWORD"="R8_SYNTHETIC_SECRET_4800"R8_SYNTHETIC_TAIL_4802:8080`, `export "DB_PASSWORD"=[REDACTED]`},
		{`password="x"a:b`, `password=[REDACTED]`},
		{`password="x"https://example.com/path`, `password=[REDACTED]`},
		// A name whose delimiter a value follows at once is still a pair: a
		// credential name is redacted on its own, a ":" starts a pair of
		// another name only if a quote follows it, and an "=" always does.
		{`password="x"key=val rest`, `password="[REDACTED]"key=val rest`},
		{`password="x"url=http://h/p`, `password="[REDACTED]"url=http://h/p`},
		{`password="x"user="bob" rest`, `password="[REDACTED]"user="bob" rest`},
		{`password="x"label:"bob"`, `password="[REDACTED]"label:"bob"`},
		{`password="x"label:'bob'`, `password="[REDACTED]"label:'bob'`},
		{`password="x"token:R8_SYNTHETIC_SECRET_4801 rest`, `password="[REDACTED]"token:[REDACTED] rest`},
		{`password="x"token:'R8_SYNTHETIC_SECRET_4801'`, `password="[REDACTED]"token:'[REDACTED]'`},
		// A closing quote that a space follows ends the value, as before.
		{`password="R8_SYNTHETIC_SECRET_4700" token="R8_SYNTHETIC_SECRET_4701"`, `password="[REDACTED]" token="[REDACTED]"`},
		{`password="R8_SYNTHETIC_SECRET_4700" user="bob"`, `password="[REDACTED]" user="bob"`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	// A URL with a user name after the quote: the other paths redact the URL
	// before they read the pair, and the text patterns alone read it as the
	// end of the credential that the quote runs into.
	url := `password="R8_SYNTHETIC_SECRET_4800"https://R8_SYNTHETIC_TAIL_4802@example.com`
	if got, want := scrubTextPatterns(url), `password=[REDACTED]`; got != want {
		t.Errorf("scrubTextPatterns(%q) = %q, want %q", url, got, want)
	}
	for _, secret := range []string{"R8_SYNTHETIC_SECRET_4800", "R8_SYNTHETIC_TAIL_4802"} {
		checkNoLeak(t, url, secret)
	}
}

// TestScrubTextPatterns_NestedAndConcatenatedQuotes checks that a credential
// is redacted whole on every path when its quotes are not those of the one
// quoted value that its key opens: a quoted value that a delimiter opens
// inside a bare value, after an apostrophe that starts a contraction, as the
// one of token=a:'s x'y does; a value made of strings that follow each other,
// of the same kind of quote or of both, as the output of the quoting function
// of a shell, 'a'"'"'b', does; an SQL string that doubles its quote; and such
// a value after an oversized key. Every string that is joined to the
// credential is part of it, and its words are not left in place after the
// redacted value, whereas the apostrophes of prose after a bare value, which
// no quote closes, are not quotes, and the prose is kept.
func TestScrubTextPatterns_NestedAndConcatenatedQuotes(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		// An apostrophe that starts a contraction opens a quoted value
		// after the delimiter, whose closing quote runs into a letter or
		// into another quote.
		{`token=a:'s R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`password=a='d R8_SYNTHETIC_SECRET_4800'x`, `password=[REDACTED]`},
		{`api_key=a='re R8_SYNTHETIC_SECRET_4800'x`, `api_key=[REDACTED]`},
		{`token=a:'ll R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`token=a:'ve R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`token=a:'t R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`token=a='s-part1 R8_SYNTHETIC_SECRET_4800''`, `token=[REDACTED]`},
		{`token=a:'s R8_SYNTHETIC_SECRET_4800abc'x`, `token=[REDACTED]`},
		{`token=a:'s R8_SECRET_ABCD's`, `token=[REDACTED]`},
		{`token=a:'d R8_SECRET_ABCD'd`, `token=[REDACTED]`},
		{`token=a:'s R8_SECRET_ABCD're rest`, `token=[REDACTED] rest`},
		{`token=a: 's R8_SYNTHETIC_SECRET_4800'x`, `token=[REDACTED]`},
		{`token=a:'s R8_SYNTHETIC_SECRET_4800' x y`, `token=[REDACTED] x y`},
		// Values opened at the end of a bare match after nested ":'x_token='"
		// or unclosed joined quotes with whitespace inside the tail.
		{`password=a:'x_token=' "R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"`, `password=[REDACTED]`},
		{`password=a:'x_token=' R8_SYNTHETIC_SECRET_4800`, `password=[REDACTED]`},
		{`token=a:"x_api_key=" "R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"`, `token=[REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"'R8_SYNTHETIC_SECRET_4801 R8_SYNTHETIC_SECRET_4802`, `password=[REDACTED]`},
		{`DB_PASSWORD='my'"'"'R8_SYNTHETIC_SECRET_4801 R8_SYNTHETIC_SECRET_4802`, `DB_PASSWORD=[REDACTED]`},
		{`Bearer "R8_SYNTHETIC_SECRET_4800"'R8_SYNTHETIC_SECRET_4801 R8_SYNTHETIC_SECRET_4802`, `Bearer [REDACTED]`},
		// Strings of both kinds of quote that are joined, as a shell joins
		// them, with a word, with the same kind of quote or with the other.
		{`DB_PASSWORD='my'"'"'R8_SYNTHETIC_TAIL_4802'`, `DB_PASSWORD='[REDACTED]'`},
		{`export "DB_PASSWORD"='my'"'"'R8_SYNTHETIC_TAIL_4802'`, `export "DB_PASSWORD"='[REDACTED]'`},
		{`export DB_PASSWORD='my'"'"'R8_SYNTHETIC_TAIL_4802' rest`, `export DB_PASSWORD='[REDACTED]' rest`},
		{`Bearer 'my'"'"'R8_SYNTHETIC_TAIL_4802'`, `Bearer [REDACTED]`},
		{`password="R8_SYNTHETIC_SECRET_4800"'R8_SYNTHETIC_TAIL_4802'`, `password=[REDACTED]`},
		{`password='R8_SYNTHETIC_SECRET_4800'"R8_SYNTHETIC_TAIL_4802"`, `password=[REDACTED]`},
		{`password=a: "R8_SYNTHETIC_SECRET_4800"'R8_SYNTHETIC_TAIL_4802'`, `password=[REDACTED]`},
		{`password=a: 'R8_SYNTHETIC_SECRET_4800'"R8_SYNTHETIC_TAIL_4802"`, `password=[REDACTED]`},
		// An SQL string whose doubled quote is followed by more words, or a
		// string of the other kind that follows a closed one.
		{`token=a:'R8_SYNTHETIC_SECRET_4800'' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'`, `token=[REDACTED]`},
		{`token='R8_SYNTHETIC_SECRET_4800'' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'`, `token='[REDACTED]'`},
		{`token=a:'x'' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'`, `token=[REDACTED]`},
		{`password=a: 'R8_SYNTHETIC_SECRET_4800'' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801' rest`, `password=[REDACTED] rest`},
		{`password=a= 'x'' R8_SYNTHETIC_SECRET_4800'`, `password=[REDACTED]`},
		{`token=a:'my'' R8_SYNTHETIC_SECRET_4800'`, `token=[REDACTED]`},
		{`token=a:'x'" R8_SYNTHETIC_SECRET_4800"`, `token=[REDACTED]`},
		{`token=a:'x'"R8_SYNTHETIC_SECRET_4800 x"`, `token=[REDACTED]`},
		{`token=a:"x"' R8_SYNTHETIC_SECRET_4800'`, `token=[REDACTED]`},
		// The apostrophes of prose after the value are not quotes, and no
		// credential follows the value, so the prose is kept.
		{`token=abc:'s missing; the user's request`, `token=[REDACTED] missing; the user's request`},
		{`token=abc:'s note password='x' rest`, `token=[REDACTED] note password='[REDACTED]' rest`},
		{`token=abc:'s note Bearer ' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'suffix`, `token=[REDACTED] note Bearer [REDACTED]`},
		{`token=abc:'s note 'password': ' R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801'suffix`, `token=[REDACTED] note 'password':[REDACTED]`},
		{`token=abc:'s Bearer " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `token=[REDACTED] Bearer [REDACTED]`},
		{`token=abc:'s "password": " R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"suffix`, `token=[REDACTED] "password":"[REDACTED]"`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	// A quoted value after an oversized key, which an envelope key nests,
	// with a whitespace-led string or an apostrophe-led string joined to it.
	longKey := strings.Repeat("x_", 70) + "PASSWORD"
	for _, key := range []string{longKey, "x.y_" + longKey} {
		for _, value := range []string{
			`=" R8_SYNTHETIC_SECRET_4800"'R8_SYNTHETIC_TAIL_4802'`,
			`=" R8_SYNTHETIC_SECRET_4800 R8_SYNTHETIC_SECRET_4801"x`,
			`='my'"'"'R8_SYNTHETIC_TAIL_4802'`,
			`=a:'s R8_SYNTHETIC_SECRET_4800'x`,
		} {
			checkAllRedactionPaths(t, "credentials:\n a_PASSWORD: "+key+value, "credentials:\n a_PASSWORD:[REDACTED]")
		}
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
		// A value that ends in a compound key name with repeated dots, as in
		// API..key, is redacted together with the value of the pair it starts.
		{"password=R7_SYNTHETIC_CREDENTIAL_9327.API..key: dial tcp", "password=[REDACTED] tcp"},
		{`password=R7_SYNTHETIC_CREDENTIAL_9327.API..key: "R7_SYNTHETIC_SECRET_1 R7_SYNTHETIC_SECRET_2"`, "password=[REDACTED]"},
		// Dotted names that are not credential names are kept.
		{"my.api.key.file=/path access.id=7 status.code=200", "my.api.key.file=/path access.id=7 status.code=200"},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
}

// TestScrubTextPatterns_DottedPageTokens checks that text reads the complete
// dotted name that ends in a page token, as isSensitiveKey reads a map key, on
// every path, to tell the access token of a page from a pagination cursor:
// the value of fb.page_token or home.page_token is redacted although that of
// page_token is a cursor, and the value of page.token, next.page.token,
// prev.page.token, previous.page.token or x.next.page.token is kept although
// that of token is a credential, whichever of the two comes first and whatever
// pair or key surrounds them. A dotted name that is neither is read by its
// last words, as before.
func TestScrubTextPatterns_DottedPageTokens(t *testing.T) {
	longDotted := strings.Repeat("abcdefgh.", 16)
	for _, tt := range []struct{ in, want string }{
		// The access token of a page, after dotted words.
		{"fb.page_token=R8_SYNTHETIC_CREDENTIAL_49173", "fb.page_token=[REDACTED]"},
		{"home.page_token=R8_SYNTHETIC_CREDENTIAL_49173", "home.page_token=[REDACTED]"},
		{"x fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173 y", "x fb.page_token:[REDACTED] y"},
		{"a.b.fb.page_token=R8_SYNTHETIC_CREDENTIAL_49173", "a.b.fb.page_token=[REDACTED]"},
		{`fb.page_token="R8_SYNTHETIC_CREDENTIAL_49173"`, `fb.page_token="[REDACTED]"`},
		{"fb.page.token=R8_SYNTHETIC_CREDENTIAL_49173", "fb.page.token=[REDACTED]"},
		{"x.page.token=R8_SYNTHETIC_CREDENTIAL_49173", "x.page.token=[REDACTED]"},
		{"x.fb-page-token=R8_SYNTHETIC_CREDENTIAL_49173", "x.fb-page-token=[REDACTED]"},
		{"FB.PAGE_TOKEN=R8_SYNTHETIC_CREDENTIAL_49173", "FB.PAGE_TOKEN=[REDACTED]"},
		{"page.token.fb.page_token=R8_SYNTHETIC_CREDENTIAL_49173", "page.token.fb.page_token=[REDACTED]"},
		{"next.page.token.token=R8_SYNTHETIC_CREDENTIAL_49173", "next.page.token.token=[REDACTED]"},
		{`failed: {"fb.page_token": "R8_SYNTHETIC_CREDENTIAL_49173"}`, `failed: {"fb.page_token":"[REDACTED]"}`},
		{"credentials:\n  fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173", "credentials:\n  fb.page_token:[REDACTED]"},
		// An oversized run of dotted words before the name.
		{"credentials:\n  " + longDotted + "z.fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173", "credentials:\n  " + longDotted + "z.fb.page_token:[REDACTED]"},
		// The access token and the cursor in either order.
		{"a.b.fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173 next.page.token: cursor-1", "a.b.fb.page_token:[REDACTED] next.page.token: cursor-1"},
		{"next.page.token: cursor-1 fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173", "next.page.token: cursor-1 fb.page_token:[REDACTED]"},
		{"home.page_token=R8_SYNTHETIC_CREDENTIAL_49173 x.next.page.token=cursor-2 DB_PASSWORD=R8_SYNTHETIC_CREDENTIAL_49173", "home.page_token=[REDACTED] x.next.page.token=cursor-2 DB_PASSWORD=[REDACTED]"},
		// A dotted name that is neither a page token nor a cursor is read by
		// its last words: x.signature is, but method_signature is not.
		{"x.signature=R8_SYNTHETIC_CREDENTIAL_49173", "x.signature=[REDACTED]"},
		{"x.method_signature=kept", "x.method_signature=kept"},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	for _, in := range []string{
		"page_token=cursor-0",
		"page.token=cursor-1",
		"next.page.token=cursor-2",
		"prev.page.token: cursor-3",
		"previous.page.token=cursor-4",
		"x.next.page.token=cursor-5",
		"x.next_page_token=cursor-6",
		"Next.Page.Token=cursor-7",
		"x.nextPageToken=cursor-8",
		"page.token=cursor-9 next.page.token: cursor-10 prev.page.token=cursor-11 previous.page.token: cursor-12 x.next.page.token=cursor-13",
		`failed: {"next.page.token": "cursor-14"}`,
		`failed: {"x.next.page.token": "cursor-15"}`,
		"credentials:\n  next.page.token: cursor-16",
	} {
		checkRedactionPaths(t, in, in)
	}
	// textKeyAllowedAt reads the dotted name that the key found at
	// s[start:end] ends: the key of x.fb.page_token is page_token, which is a
	// cursor alone, and that of x.next.page.token is token, a credential
	// alone.
	for _, tt := range []struct {
		in, key string
		want    bool
	}{
		{"x.fb.page_token: v", "page_token", true},
		{"x.home.page_token: v", "page_token", true},
		{"x.fb.page.token: v", "token", true},
		{"x.page.token: v", "token", true},
		{"x.next.page.token: v", "token", false},
		{"x.prev.page.token: v", "token", false},
		{"x.previous.page.token: v", "token", false},
		{"page.token: v", "token", false},
		{"x.next_page_token: v", "next_page_token", false},
		{"x.signature: v", "signature", true},
		{"x.method_signature: v", "method_signature", false},
		{"page_token: v", "page_token", false},
		{"token: v", "token", true},
	} {
		start := strings.Index(tt.in, tt.key)
		if got := textKeyAllowedAt(tt.in, start, start+len(tt.key)); got != tt.want {
			t.Errorf("textKeyAllowedAt(%q, %d, %d) = %v, want %v", tt.in, start, start+len(tt.key), got, tt.want)
		}
	}
}

// TestScrubTextPatterns_DottedNamesAfterLongRunsOfSeparators checks that a
// dotted name that ends in a page token is read whole, as a map key is, when
// the run of dots, underscores or hyphens before its last words is longer than
// maxNestedKeyLen: the bound that keeps the search for names linear does not
// cut the name inside that run, so the value of a.....page_token, which the
// word "a" makes the access token of a page, is redacted whatever the length
// of the run, as that of the map key is, and the value of
// next.....page_token, which is a cursor, is kept.
func TestScrubTextPatterns_DottedNamesAfterLongRunsOfSeparators(t *testing.T) {
	for _, n := range []int{124, 125, 126, 127, 128, 129, 130, 131, 300} {
		dots := strings.Repeat(".", n)
		checkAllRedactionPaths(t, "a"+dots+"page_token=R8_SYNTHETIC_CREDENTIAL_49173", "a"+dots+"page_token=[REDACTED]")
		checkAllRedactionPaths(t, "x fb"+dots+"page_token: R8_SYNTHETIC_CREDENTIAL_49173 y", "x fb"+dots+"page_token:[REDACTED] y")
		checkAllRedactionPaths(t, "a"+dots+"page.token=R8_SYNTHETIC_CREDENTIAL_49173", "a"+dots+"page.token=[REDACTED]")
		checkAllRedactionPaths(t, "fb.page"+dots+"token=R8_SYNTHETIC_CREDENTIAL_49173", "fb.page"+dots+"token=[REDACTED]")
		checkAllRedactionPaths(t, "fb_page"+dots+"token=R8_SYNTHETIC_CREDENTIAL_49173", "fb_page"+dots+"token=[REDACTED]")
		checkRedactionPaths(t, "x next"+dots+"page_token: cursor-1", "x next"+dots+"page_token: cursor-1")
		checkRedactionPaths(t, "prev"+dots+"page.token=cursor-2", "prev"+dots+"page.token=cursor-2")
		checkRedactionPaths(t, "x.next.page"+dots+"token=cursor-3", "x.next.page"+dots+"token=cursor-3")
	}
	var sensitive, kept int
	for _, prefix := range []string{"a", "fb", "fb.page", "fb_page", "fb-page", "a.b.c.page", "next", "abcnext", "x.next", "x.next.page"} {
		for _, sep := range []string{".", "_.", "-."} {
			for _, n := range []int{1, 64, 124, 125, 127, 128, 129, 130, 200, 300} {
				for _, last := range []string{"page_token", "next_page_token", "token", "page.token"} {
					name := prefix + strings.Repeat(sep, n) + last
					want := isSensitiveKey(name)
					if want {
						sensitive++
					} else {
						kept++
					}
					for _, in := range []string{name + "=R8_SYNTHETIC_CREDENTIAL_49173", "x " + name + ": R8_SYNTHETIC_CREDENTIAL_49173"} {
						if got := !strings.Contains(scrubTextPatterns(in), "R8_SYNTHETIC_CREDENTIAL_49173"); got != want {
							t.Errorf("scrubTextPatterns redacted the value of the name %q (%d bytes): %v, want %v as for the map key", prefix+"…"+last, len(name), got, want)
						}
					}
					in := name + "=R8_SYNTHETIC_CREDENTIAL_49173"
					b, _, err := SmartTruncate(errors.New(in), 1<<20)
					if got := err == nil && !strings.Contains(string(b), "R8_SYNTHETIC_CREDENTIAL_49173"); got != want {
						t.Errorf("SmartTruncate(errors.New(%q…%q)) redacted the value: %v, want %v as for the map key (err %v)", prefix, last, got, want, err)
					}
				}
			}
		}
	}
	if sensitive == 0 || kept == 0 {
		t.Errorf("%d names are sensitive and %d are kept, want some of each", sensitive, kept)
	}
}

// TestScrubTextPatterns_DottedNamesAgreeWithMapKeys checks that a dotted name
// has its value redacted in text exactly when isSensitiveKey redacts the
// value of a map key of that name, for names of up to four dotted words
// (including empty segments from repeated dots, as in api..key): dotted OAuth
// names, names made of the words of credential names and others, as in
// stripe.api.key, stripe.api..key or x.goog.api.key, and names that end in a
// page token or in a pagination cursor, as fb.page_token, the access token of
// a page, and x.next.page.token, a cursor, do.
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
	build([]string{"", "stripe", "jwt", "aws", "x", "goog", "google", "api", "private", "secret", "access", "auth", "code"}, []string{"key", "Key", "id", "verifier", "code", "token"}, "", 3)
	build([]string{"", "page", "next", "prev", "previous", "fb", "home", "x"}, []string{"token", "Token", "page_token", "next_page_token", "prev_page_token", "previous_page_token", "nextPageToken"}, "", 3)
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
		{"dotted credential name with repeated dots", "R7.API..key", false, valueEndingInKeyName, 3},
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
		{
			// Each dotted name ends in a name that its complete dotted name
			// decides on, which is read from the last bytes of the run.
			"bare run of dotted cursor names", func(n int) string { return strings.Repeat("x.next.page.token:", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"bare run of dotted cursor names through an error", func(n int) string { return strings.Repeat("x.next.page.token:", n) },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 1<<20) }, 1600,
		},
		{
			"repeated dotted page access tokens", func(n int) string { return strings.Repeat("x.fb.page_token: v ", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one long dotted run before a page access token", func(n int) string { return strings.Repeat("a.", n) + "fb.page_token=v" },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"one long dotted run before a page access token through an error", func(n int) string { return strings.Repeat("a.", n) + "fb.page_token=v" },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 10000,
		},
		{
			// The name is read whole, past the bound, from the first
			// word before the dots.
			"one long run of dots before a page access token", func(n int) string { return "a" + strings.Repeat(".", n) + "page_token=v" },
			func(in string) { _ = scrubTextPatterns(in) }, 40000,
		},
		{
			"one long run of dots before a page access token through an error", func(n int) string { return "a" + strings.Repeat(".", n) + "page_token=v" },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 40000,
		},
		{
			"one long word before a dotted page token", func(n int) string { return strings.Repeat("a", n) + ".page_token=v" },
			func(in string) { _ = scrubTextPatterns(in) }, 40000,
		},
		{
			"repeated runs of dots before a page access token", func(n int) string {
				return strings.Repeat("a"+strings.Repeat(".", 200)+"page_token=v ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated quoted values that run into a letter", func(n int) string { return strings.Repeat(`password="a"b `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quoted values that run into a key", func(n int) string { return strings.Repeat(`password="a"token="b" `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated bearer tokens that run into a letter", func(n int) string { return strings.Repeat(`Bearer "a"b `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated nested quoted values that run into a letter", func(n int) string { return strings.Repeat(`password=a= "b c"d `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one long run of quoted values that run into each other", func(n int) string { return "password=" + strings.Repeat(`"a"`, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			// Each closing quote opens the value of a later key, which the
			// bearer token runs on through.
			"one long run of bearer values that end in a key and a delimiter", func(n int) string { return "Bearer " + strings.Repeat(`"a_TOKEN="`, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"repeated whitespace-led bearer values that run into a letter", func(n int) string { return strings.Repeat(`Bearer " a b"c `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated whitespace-led quoted values of a quoted key that run into a letter", func(n int) string { return strings.Repeat(`export "DB_PASSWORD"=" a b"c `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated whitespace-led quoted values of a JSON key that run into a letter", func(n int) string { return strings.Repeat(`failed: "password": " a b"c `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one long run of whitespace-led bearer values", func(n int) string { return "Bearer " + strings.Repeat(`" a b"`, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"one long run of whitespace-led bearer values that end in a key and a delimiter", func(n int) string { return "Bearer " + strings.Repeat(`" a b_TOKEN="`, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"repeated quoted values that run into a name and a port", func(n int) string { return strings.Repeat(`password="a"b:8080 `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quoted values that run into a name and a delimiter", func(n int) string { return strings.Repeat(`password="a"b= `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated nested quoted values after an apostrophe", func(n int) string { return strings.Repeat(`token=a:'s b c'x `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated shell-quoted values", func(n int) string { return strings.Repeat(`DB_PASSWORD='a'"'"'b' `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated SQL strings with a doubled quote", func(n int) string { return strings.Repeat(`token=a:'b'' c d' `, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one long run of strings joined to a credential", func(n int) string { return "password=" + strings.Repeat(`'a'"b"`, n) },
			func(in string) { _ = scrubTextPatterns(in) }, 10000,
		},
		{
			"one long run of strings joined to a credential through an error", func(n int) string { return "password=" + strings.Repeat(`'a'"b"`, n) },
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 10000,
		},
		{
			"repeated multi-word dotted page tokens after long dot runs", func(n int) string {
				return strings.Repeat("fb.page"+strings.Repeat(".", 130)+"token=v ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated closed quotes followed by spaced sensitive keys", func(n int) string {
				return strings.Repeat(`password="a"token: b `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated prose apostrophes after colon before quoted keys", func(n int) string {
				return strings.Repeat(`token=abc:'s note 'password': ' x y'z `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated camelCase keys with long separator runs", func(n int) string {
				return strings.Repeat("stripeApi"+strings.Repeat(".", 150)+"key=v ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated multi-separator credential keys through an error", func(n int) string {
				return strings.Repeat("stripe.api..key=v ", n)
			},
			func(in string) { _, _, _ = SmartTruncate(errors.New(in), 100) }, 2000,
		},
		{
			"repeated punctuation-continued quoted values", func(n int) string {
				return strings.Repeat(`export DB_PASSWORD="a"+b `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated adjacent compound credential pairs", func(n int) string {
				return strings.Repeat(`DB_PASSWORD="a"GITHUB_TOKEN="b" `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
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

// TestScrubTextPatterns_RepeatedAndMixedSeparatorKeys checks that credential
// names written with repeated or mixed "_", "-" and "." separators inside the
// name (including camelCase and acronym prefixes) are redacted in free text
// across all paths just as isSensitiveKey redacts them as structured map keys,
// while non-credential names and pagination cursors remain preserved.
func TestScrubTextPatterns_RepeatedAndMixedSeparatorKeys(t *testing.T) {
	sensitiveNames := []string{
		"api__key",
		"api..key",
		"api--key",
		"api._-key",
		"private..key",
		"stripe.api..key",
		"OPENAI_API__KEY",
		"stripe.api--key",
		"foo.private..key",
		"foo.auth..code",
		"aws..secret..access..key",
		"x..goog__api--key",
		"client.-_secret",
		"code...verifier",
		"google...access...id",
		"stripeApi..key",
		"OpenAIAPI__Key",
		"OpenAIAPI..key",
		"OpenAIAPI-Key",
		"stripeApi" + strings.Repeat(".", 200) + "key",
		"user:api__key",
		"app:private..key",
	}
	for _, name := range sensitiveNames {
		if !isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = false, want true", name)
		}
		checkAllRedactionPaths(t, name+"=SYNTHETIC_CREDENTIAL_6301", name+"=[REDACTED]")
		checkAllRedactionPaths(t, "failed: "+name+": SYNTHETIC_CREDENTIAL_6301 retry", "failed: "+name+":[REDACTED] retry")
		checkAllRedactionPaths(t, `failed: "`+name+`": "SYNTHETIC_CREDENTIAL_6301"`, `failed: "`+name+`":"[REDACTED]"`)
	}
	for _, tt := range []struct{ in, want string }{
		{`{"api__key":"SYNTHETIC_CREDENTIAL_6301","note":"ok"}`, `{"api__key":"[REDACTED]","note":"ok"}`},
		{"credentials:\n  stripe.api..key: SYNTHETIC_CREDENTIAL_6301\n  region: us", "credentials:\n  stripe.api..key:[REDACTED]\n  region: us"},
		{"credentials:\n  foo.auth..code: SYNTHETIC_CREDENTIAL_6301\n  region: us", "credentials:\n  foo.auth..code:[REDACTED]\n  region: us"},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
	for _, name := range []string{
		"oauth..config.codes",
		"status..code",
		"access..id",
		"api.x.key",
		"my.api..key.file",
		"page..token",
		"next..page..token",
		"prev__page--token",
		"x.next..page__token",
	} {
		if isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = true, want false", name)
		}
		in := name + "=kept-value"
		checkRedactionPaths(t, in, in)
	}
}

// TestScrubTextPatterns_JoinedContractionShapedCredentials checks that joined,
// quoted or unclosed credential assignments containing contraction-shaped words
// are redacted across their entire credential span rather than truncated at an
// apostrophe, while preserving independent outer prose clauses and following
// key-value or Bearer constructs.
func TestScrubTextPatterns_JoinedContractionShapedCredentials(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`export DB_PASSWORD=a:'s correct horse battery's`, `export DB_PASSWORD=[REDACTED]`},
		{`password=a:'s prefix CORRECTHORSE's`, `password=[REDACTED]`},
		{`password=a:'s correct horse battery`, `password=[REDACTED]`},
		{`export DB_PASSWORD=a:'d correct horse battery'd`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'re correct horse battery're rest`, `export DB_PASSWORD=[REDACTED] rest`},
		{`password=a: 's missing; the SYNTHETIC_CREDENTIAL_6301's request`, `password=[REDACTED] request`},
		{`export DB_PASSWORD=a:'s part1; SYNTHETIC_CREDENTIAL_6301's`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s missing; the user's+SYNTHETIC_CREDENTIAL_6301`, `export DB_PASSWORD=[REDACTED]`},
		{`password=a:'s bearer of bad news`, `password=[REDACTED]`},
		{`password=a:'s SYNTHETIC_CREDENTIAL_6301; retry later`, `password=[REDACTED]`},
		// Verified outer-prose boundaries and following constructs stay intact.
		{`token=abc:'s scope missing; the user's request`, `token=[REDACTED] scope missing; the user's request`},
		{`token=abc:'s; the user's request`, `token=[REDACTED]; the user's request`},
		{`token=abc:'s scope missing; retry later`, `token=[REDACTED] scope missing; retry later`},
		{`token=abc:'s note password='SYNTHETIC_CREDENTIAL_6301' rest`, `token=[REDACTED] note password='[REDACTED]' rest`},
		{`token=abc:'s note Bearer ' SYNTHETIC_CREDENTIAL_6301 SYNTHETIC_CREDENTIAL_6302'suffix`, `token=[REDACTED] note Bearer [REDACTED]`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
}

// TestScrubTextPatterns_PunctuationContinuedQuotedValues checks that a closing
// quote immediately followed by a punctuation continuation (+, -, /, ., @, %,
// !, ?, =, :, \, etc.) is consumed as part of the credential value across all
// paths, while genuine delimiters and enclosing punctuation boundaries remain
// intact.
func TestScrubTextPatterns_PunctuationContinuedQuotedValues(t *testing.T) {
	for _, op := range []string{"+", "-", "/", ".", "..", "@", "%", "!", "?", "=", ":", `\`, `\n`, `\\`, "~", "|"} {
		checkAllRedactionPaths(t, `export DB_PASSWORD="prefix"`+op+`SYNTHETIC_CREDENTIAL_91827`, `export DB_PASSWORD=[REDACTED]`)
		checkAllRedactionPaths(t, `export DB_PASSWORD='prefix'`+op+`SYNTHETIC_CREDENTIAL_91827`, `export DB_PASSWORD=[REDACTED]`)
		checkAllRedactionPaths(t, `export "DB_PASSWORD"="prefix"`+op+`SYNTHETIC_CREDENTIAL_91827`, `export "DB_PASSWORD"=[REDACTED]`)
		checkAllRedactionPaths(t, `failed: "password": "prefix"`+op+`SYNTHETIC_CREDENTIAL_91827`, `failed: "password":"[REDACTED]"`)
		checkAllRedactionPaths(t, `Bearer "prefix"`+op+`SYNTHETIC_CREDENTIAL_91827`, `Bearer [REDACTED]`)
		checkAllRedactionPaths(t, `password=a="prefix"`+op+`SYNTHETIC_CREDENTIAL_91827`, `password=[REDACTED]`)
	}
	for _, tt := range []struct{ in, want string }{
		{`export DB_PASSWORD="prefix"+"SYNTHETIC_CREDENTIAL_91827"`, `export DB_PASSWORD="[REDACTED]"`},
		{`export DB_PASSWORD="prefix"+'SYNTHETIC_CREDENTIAL_91827 tail'`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD='prefix'/"SYNTHETIC_CREDENTIAL_91827"`, `export DB_PASSWORD=[REDACTED]`},
		{`failed: {'api_key': 'prefix'@SYNTHETIC_CREDENTIAL_91827}`, `failed: {'api_key':[REDACTED]}`},
		{`(password="prefix")+SYNTHETIC_CREDENTIAL_91827`, `(password=[REDACTED]`},
		// Verified delimiters and enclosing brackets/sentence punctuation.
		{`export DB_PASSWORD="SYNTHETIC_CREDENTIAL_91827"; echo ok`, `export DB_PASSWORD="[REDACTED]"; echo ok`},
		{`password="SYNTHETIC_CREDENTIAL_91827", user="bob"`, `password="[REDACTED]", user="bob"`},
		{`password="SYNTHETIC_CREDENTIAL_91827"&user=bob`, `password="[REDACTED]"&user=bob`},
		{`(password="SYNTHETIC_CREDENTIAL_91827")`, `(password="[REDACTED]")`},
		{`failed: [password="SYNTHETIC_CREDENTIAL_91827"]`, `failed: [password="[REDACTED]"]`},
		{`failed: {Password:"SYNTHETIC_CREDENTIAL_91827"}.`, `failed: {Password:"[REDACTED]"}.`},
		{`password="SYNTHETIC_CREDENTIAL_91827": dial tcp`, `password="[REDACTED]": dial tcp`},
		{`password="SYNTHETIC_CREDENTIAL_91827"! retry`, `password="[REDACTED]"! retry`},
		{`password="SYNTHETIC_CREDENTIAL_91827"?`, `password="[REDACTED]"?`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
}
