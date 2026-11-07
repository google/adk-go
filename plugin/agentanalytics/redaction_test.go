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
	"math/rand/v2"
	"net/url"
	"os/exec"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"
	"unsafe"
	"weak"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
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
		{"bearer token stops at a vertical tab", "x Bearer SYNTHETIC_SECRET\vy", "x Bearer [REDACTED]"},
		{"bearer token stops at a line separator", "x Bearer SYNTHETIC_SECRET\u2028y", "x Bearer [REDACTED]"},
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
			got, _, err := smartTruncateText(t, tt.in, 1000)
			if err != nil {
				t.Fatalf("SmartTruncate(%q) error = %v", tt.in, err)
			}
			if strings.Contains(string(got), "SYNTHETIC_SECRET") {
				t.Errorf("SmartTruncate(%q) = %s, which holds the secret", tt.in, got)
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
// escape in JSON text is kept out on the raw text and SmartTruncate paths.
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
		b, _, err := smartTruncateText(t, tt.in, 1000)
		if err != nil || strings.Contains(string(b), tt.secret) {
			t.Errorf("SmartTruncate(%q) = (%s, %v), which holds the credential", tt.in, b, err)
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
				"text":                     tt.in,
				"text in a map":            map[string]any{"msg": tt.in},
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
		{"oversized run, then a spaced = and an object", "credentials:\n a_PASSWORD: " + run + ` = {"b": "c"} kept`, "credentials:[REDACTED] kept"},
		{"oversized run, then an object in the word", "credentials:\n a_PASSWORD: " + run + `={"b": "c"} kept`, "credentials:[REDACTED] kept"},
		{"oversized run, then an array after a delimiter", "credentials:\n a_PASSWORD: " + run + `: ["b", "c"] kept`, "credentials:[REDACTED] kept"},
		{"oversized run, then an object that is not closed", "credentials:\n a_PASSWORD: " + run + ` = {"b": "c" kept`, "credentials:[REDACTED]"},
		// A quote at the end of an oversized run whose last key is followed
		// by a spaced delimiter opens a value, even when a letter follows
		// the quoted value, which the quote runs into: the letter is
		// redacted with the value, as it may be part of it (see
		// runOnValueEnd). Read from the delimiter instead, the value would
		// end at its first space.
		{"oversized run, then a quote that ends the word", "credentials:\n a_PASSWORD: " + run + `=" = b c"x`, "credentials:[REDACTED]"},
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
	for _, name := range names {
		for _, q := range []string{`"`, `'`, `\"`, `\'`} {
			for _, pad := range []string{"", " ", "\t", "\u00a0"} {
				for _, sp := range []string{"", " ", "\t"} {
					for _, d := range delims {
						for _, v := range values {
							s := q + pad + name + pad + q + sp + d + v
							pair := quotedPairAtStart(s)
							want := pair != nil && pair[8] >= 0
							if got := allowedKeyAt("x "+s, 2); got != want {
								t.Errorf("allowedKeyAt(%q) = %v, want %v", s, got, want)
							}
						}
					}
				}
			}
		}
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
		b, _, err := smartTruncateText(t, tt.in, 1000)
		var got string
		if err != nil || json.Unmarshal(b, &got) != nil || got != tt.want {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want %q", tt.in, b, err, tt.want)
		}
	}
}

// checkRedactionPaths checks that scrubTextPatterns, sanitizeRawText and
// SmartTruncate of in as a string all write want for in, that scrubbing want
// again changes nothing, and that SmartTruncate writes errors.New(in) as {}
// (see smartTruncateText).
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
	b, _, err := smartTruncateText(t, in, 1000)
	var got string
	if err != nil || json.Unmarshal(b, &got) != nil || got != want {
		t.Errorf("SmartTruncate(%q) = (%s, %v), want %q", in, b, err, want)
	}
}

// smartTruncateText returns what SmartTruncate writes for in as a string,
// which the sanitizer scrubs as it scrubs every string value, after checking
// that SmartTruncate writes errors.New(in) as {}: under redaction the
// sanitizer does not call the Error method of an error, so no text of an
// error, scrubbed or not, reaches its output (see sanitizer.marshaled).
func smartTruncateText(t *testing.T, in string, maxLength int) ([]byte, bool, error) {
	t.Helper()
	if b, _, err := SmartTruncate(errors.New(in), maxLength); err != nil || string(b) != "{}" {
		t.Errorf("SmartTruncate(errors.New(%s)) = (%s, %v), want {}", elideText(in), elideText(string(b)), err)
	}
	return SmartTruncate(in, maxLength)
}

// checkAllRedactionPaths checks, as checkRedactionPaths does, that
// scrubTextPatterns, sanitizeRawText and SmartTruncate of in as a string
// write want for in, that scrubbing want again changes nothing, and that
// SmartTruncate writes errors.New(in) as {}, and that redactErrorMessage,
// which scrubs the message of an error before error_message holds it, writes
// want for in too, and for want itself.
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
// redactErrorMessage and SmartTruncate of in as a string writes the secret in
// in, and that SmartTruncate writes errors.New(in) as {}. It is for texts that
// the paths write differently, as they do for a JSON object that is not
// valid, which only the text patterns read as pairs.
func checkNoLeak(t *testing.T, in, secret string) {
	t.Helper()
	raw, _ := sanitizeRawText(in)
	smart, _, err := smartTruncateText(t, in, 1000)
	if err != nil {
		t.Errorf("SmartTruncate(%q) error = %v", in, err)
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

// baselineRedacts reports whether the baseline pass, which scrubTextPatterns
// runs under its own passes so that text loses at least what that pass
// redacts, removes the secret from in.
func baselineRedacts(in, secret string) bool {
	out, _ := baselineScrub(in)
	return !strings.Contains(out, secret)
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
		{`failed: "password=" and "y""`, `failed: "password=" [REDACTED] "y""`},
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
		{"credentials:\n a_PASSWORD: " + run + ` = "prefix R7_SYNTHETIC_CREDENTIAL_9327`, "credentials:[REDACTED]"},
		{"credentials:\n a_PASSWORD: " + run + `: 'prefix R7_SYNTHETIC_CREDENTIAL_9327`, "credentials:[REDACTED]"},
		// Unclosed spaced quotes under envelope and nested keys.
		{`credentials: token = "   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials:[REDACTED]`},
		{`credentials: token = '   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials:[REDACTED]`},
		{`credentials: token = \"   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials:[REDACTED]`},
		{`credentials: token = \'   prefix R7_SYNTHETIC_CREDENTIAL_9327`, `credentials:[REDACTED]`},
		{`credentials: password="correct R7_SYNTHETIC_CREDENTIAL_9327 battery"`, `credentials: password="[REDACTED]"`},
		{"credentials:\n  token: \"   prefix R7_SYNTHETIC_CREDENTIAL_9327", "credentials:[REDACTED]"},
		// The quote closes the string that the pair is in, including when
		// another quoted pair follows on the same line.
		{`failed: "password=" in config`, `failed: "password=" in config`},
		{`failed: "password=" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=" [REDACTED]="[REDACTED]"`},
		{`failed: "password=" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "password=" [REDACTED]=[REDACTED]`},
		{`failed: "password=   " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=   " [REDACTED]="[REDACTED]"`},
		{`failed: "password=abc:" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=[REDACTED] DB_PASSWORD="[REDACTED]"`},
		{`failed: "password=abc:" DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: "password=[REDACTED] DB_PASSWORD=[REDACTED]`},
		{`failed: "password=abc: " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "password=[REDACTED] " DB_PASSWORD="[REDACTED]"`},
		{`failed: 'password=abc: ' dbPassword='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'password=[REDACTED] ' dbPassword='[REDACTED]'`},
		{`failed: "credentials: token = " DB_PASSWORD="   R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "credentials:[REDACTED] = " [REDACTED]="[REDACTED]"`},
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
		{"\\\"Authorization: R7_SYNTHETIC_CREDENTIAL_9327\\\"\nDB_PASSWORD=\\\"   prefix R7_SYNTHETIC_CREDENTIAL_9327", "\\\"Authorization: [REDACTED]\\\"\nDB_PASSWORD=[REDACTED]"},
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
		{`token=abc:'s scope missing; the user's request`, `token=[REDACTED] request`},
		{`failed: "token=abc:'s"; the user's request`, `failed: "token=[REDACTED]; the user's request`},
		{`failed: {"msg": "token=abc:'s"}; the user's request`, `failed: {"msg": "token=[REDACTED]}; the user's request`},
		{`failed: {token=abc:'s}; the user's request`, `failed: {token=[REDACTED] request`},
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
		{`failed: 'error at "token": ' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327'`, `failed: 'error at "token": ' [REDACTED]='[REDACTED]'`},
		{`failed: 'error at "token": ' DB_PASSWORD='   R7_SYNTHETIC_CREDENTIAL_9327`, `failed: 'error at "token": ' [REDACTED]=[REDACTED]`},
		{`got "token": " in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`failed: "Bearer " api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `failed: "Bearer " [REDACTED]="[REDACTED]"`},
		{`Bearer " in config, api_key="R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer [REDACTED]`},
		{`failed: "Bearer "R7_SYNTHETIC_CREDENTIAL_9327""`, `failed: "Bearer [REDACTED]"`},
		// The later string's value starts with a character that a key does
		// not, or the value before it ends in a key and a delimiter.
		{`got "token": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="$2b$12$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="/R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="-R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="★R7_SYNTHETIC_CREDENTIAL_9327"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="{\"k\":\"R7_SYNTHETIC_CREDENTIAL_9327\"}"`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key=""R7_SYNTHETIC_CREDENTIAL_9327""`, `got "token":"[REDACTED]"`},
		{`got "token": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `got "token":"[REDACTED]"`},
		{`got "password": " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `got "password":"[REDACTED]"`},
		{`got "password": " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `got "password":"[REDACTED]"`},
		{`got "password": " in config, api_key=""R7_SYNTHETIC_CREDENTIAL_9327""`, `got "password":"[REDACTED]"`},
		{`Bearer " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer [REDACTED]`},
		{`Bearer " in config, api_key="$R7_SYNTHETIC_CREDENTIAL_9327"`, `Bearer [REDACTED]`},
		{`Bearer " in config, api_key="   R7_SYNTHETIC_CREDENTIAL_9327" DB_PASSWORD="   SYNTHETIC_R7_CREDENTIAL`, `Bearer [REDACTED]`},
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
	// Quote-led multiline input where authHeaderRe reads the header up to the
	// closing quote on line 1 and keeps that quote (sanitizeRawText and
	// SmartTruncate fail closed as unparseable JSON; scrubTextPatterns resets
	// quote state at the newline).
	quoteLedAuth := "\"Authorization: R7_SYNTHETIC_CREDENTIAL_9327\"\nDB_PASSWORD=\"   prefix R7_SYNTHETIC_CREDENTIAL_9327"
	if got, want := scrubTextPatterns(quoteLedAuth), "\"Authorization: [REDACTED]\"\nDB_PASSWORD=[REDACTED]"; got != want {
		t.Errorf("scrubTextPatterns(%q) = %q, want %q", quoteLedAuth, got, want)
	}
	if raw, _ := sanitizeRawText(quoteLedAuth); strings.Contains(raw, "R7_SYNTHETIC_CREDENTIAL_9327") {
		t.Errorf("sanitizeRawText(%q) = %q, leaked credential", quoteLedAuth, raw)
	}
	if b, _, err := smartTruncateText(t, quoteLedAuth, 1000); err != nil || strings.Contains(string(b), "R7_SYNTHETIC_CREDENTIAL_9327") {
		t.Errorf("SmartTruncate(%q) = (%s, %v), leaked credential", quoteLedAuth, b, err)
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
		{`failed: "password": " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `failed: "password":"[REDACTED]"`},
		{`failed: "password": " for user "bob"`, `failed: "password":"[REDACTED]"`},
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
		{`got "token": " for user "bob"`, `got "token":"[REDACTED]"`},
		{`Bearer " for user "bob"`, `Bearer [REDACTED]`},
		{`got "token": " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `got "token":"[REDACTED]"`},
		{`Bearer " in config, api_key="R8_SYNTHETIC_CREDENTIAL_49173"`, `Bearer [REDACTED]`},
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
// on every path, and the pair after the quote is read on its own if its key
// is a credential name, whose pair is redacted on its own. A pair or a quoted
// key whose key is not a credential name is redacted with the value, as a
// shell joins it to the value into one word, whose credential it may be part
// of: the pair is no proof that the value ended at the quote. A quote that a
// word of another kind follows runs into it too, as the padding of base64
// text, a bare letter run, digits, a name whose delimiter has no value after
// it, as in tail= or tail:, or one whose ":" is that of a port or of a URL
// do: that word is the end of the credential, and is redacted with it.
func TestScrubTextPatterns_QuotedValueFollowedByKey(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`password="benign"token="R8_SYNTHETIC_SECRET_4700" rest`, `password="[REDACTED]"token="[REDACTED]" rest`},
		{`password="R8_SYNTHETIC_SECRET_4700"user="bob" rest`, `password="[REDACTED]" rest`},
		{`password="R8_SYNTHETIC_SECRET_4700"token="R8_SYNTHETIC_SECRET_4701"`, `password="[REDACTED]"token="[REDACTED]"`},
		{`password="benign"token=R8_SYNTHETIC_SECRET_4700 rest`, `password="[REDACTED]"token=[REDACTED] rest`},
		{`password='benign'api_key='R8_SYNTHETIC_SECRET_4700'`, `password='[REDACTED]'api_key='[REDACTED]'`},
		{`password="benign"user=bob rest`, `password=[REDACTED] rest`},
		{`password="benign"label:"bob" rest`, `password="[REDACTED]" rest`},
		{`password="benign"user="bob"token="R8_SYNTHETIC_SECRET_4700"`, `password="[REDACTED]"`},
		{"password=\"benign\"\ttoken=\"R8_SYNTHETIC_SECRET_4700\"", "password=\"[REDACTED]\"\ttoken=\"[REDACTED]\""},
		{`password="benign"a.b=c token="R8_SYNTHETIC_SECRET_4700"`, `password=[REDACTED] token="[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN="R8_SYNTHETIC_SECRET_4700"`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN="[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN=SYNTHETIC_FOLLOWING_6401 rest`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN=[REDACTED] rest`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN:"SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN:"[REDACTED]"`},
		{`DB_PASSWORD="benign"GITHUB_TOKEN:'SYNTHETIC_FOLLOWING_6401'`, `DB_PASSWORD="[REDACTED]"GITHUB_TOKEN:'[REDACTED]'`},
		{`DB_PASSWORD="benign"dbPassword="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"dbPassword="[REDACTED]"`},
		{`DB_PASSWORD="benign"stripe.api.key="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"stripe.api.key="[REDACTED]"`},
		{`OPENAI_API__KEY="benign"stripe.api..key="SYNTHETIC_FOLLOWING_6401"`, `OPENAI_API__KEY="[REDACTED]"stripe.api..key="[REDACTED]"`},
		{`DB_PASSWORD="benign"fb.page_token="SYNTHETIC_FOLLOWING_6401"`, `DB_PASSWORD="[REDACTED]"fb.page_token="[REDACTED]"`},
		// A preserved pagination cursor or integer token counter pair (see startsPreservedTokenPair) deliberately ends the quoted credential value glued before it, so non-secret cursor and counter telemetry stays intact.
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
		// The quoted value of a JSON key that another quoted key follows: a
		// key that is not a credential name is redacted with the value, the
		// closing brace too, as the JSON-text pass counts no brackets.
		{`failed: {"password":"R8_SYNTHETIC_SECRET_4700""user":"bob"}`, `failed: {"password":"[REDACTED]"`},
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
		// credential name is redacted on its own, and another name is
		// redacted with the value, as a shell joins it to the value.
		{`password="x"key=val rest`, `password=[REDACTED] rest`},
		{`password="x"url=http://h/p`, `password=[REDACTED]`},
		{`password="x"user="bob" rest`, `password="[REDACTED]" rest`},
		{`password="x"label:"bob"`, `password="[REDACTED]"`},
		{`password="x"label:'bob'`, `password=[REDACTED]`},
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
		// An enclosing outer quote closing before a clause delimiter keeps
		// following prose outside the credential, whereas a bare bracket without
		// an enclosing quote conservatively consumes an unclosed credential.
		{`failed: "token=abc:'s"; the user's request`, `failed: "token=[REDACTED]; the user's request`},
		{`failed: {"msg": "token=abc:'s"}; the user's request`, `failed: {"msg": "token=[REDACTED]}; the user's request`},
		{`failed: {token=abc:'s}; the user's request`, `failed: {token=[REDACTED] request`},
		{`token=abc:'s missing; the user's request`, `token=[REDACTED] request`},
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
			checkAllRedactionPaths(t, "credentials:\n a_PASSWORD: "+key+value, "credentials:[REDACTED]")
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
		{"credentials:\n  oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327\n  region: us", "credentials:[REDACTED]\n  region: us"},
		{"credentials:\n a_PASSWORD: " + longKey + ".oauth2Config.extra.code: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:[REDACTED]"},
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
		{"credentials:\n  stripe.api.key: R7_SYNTHETIC_CREDENTIAL_9327", "credentials:[REDACTED]"},
		{"credentials:\n  my.stripe_api.key: R7_SYNTHETIC_CREDENTIAL_9327\n  region: us", "credentials:[REDACTED]\n  region: us"},
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
// page_token is a cursor, whichever of the two comes first and whatever pair
// or key surrounds them. The value of a dotted cursor name whose last word is
// token, as page.token, next.page.token or x.next.page.token, is redacted in
// text, although isSensitiveKey keeps it in a map, because the baseline pass
// reads that last word as the credential key token. A dotted name that is
// neither is read by its last words, as before.
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
		{"credentials:\n  fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173", "credentials:[REDACTED]"},
		// An oversized run of dotted words before the name.
		{"credentials:\n  " + longDotted + "z.fb.page_token: R8_SYNTHETIC_CREDENTIAL_49173", "credentials:[REDACTED]"},
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
		"x.next_page_token=cursor-6",
		"x.nextPageToken=cursor-8",
		"page.token=cursor-1",
		"next.page.token=cursor-2",
		"prev.page.token: cursor-3",
		"previous.page.token=cursor-4",
		"x.next.page.token=cursor-5",
		"Next.Page.Token=cursor-7",
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
// next.....page_token, which is a cursor, is kept. The value of
// prev.....page.token or x.next.page.....token, whose last word is token, is
// redacted, as the baseline pass redacts it, and so is that of any name that
// the baseline pass redacts although isSensitiveKey keeps it.
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
						want := want || baselineRedacts(in, "R8_SYNTHETIC_CREDENTIAL_49173")
						if got := !strings.Contains(scrubTextPatterns(in), "R8_SYNTHETIC_CREDENTIAL_49173"); got != want {
							t.Errorf("scrubTextPatterns redacted the value of the name %q (%d bytes): %v, want %v as for the map key or the baseline pass", prefix+"…"+last, len(name), got, want)
						}
					}
					in := name + "=R8_SYNTHETIC_CREDENTIAL_49173"
					want = want || baselineRedacts(in, "R8_SYNTHETIC_CREDENTIAL_49173")
					b, _, err := smartTruncateText(t, in, 1<<20)
					if got := err == nil && !strings.Contains(string(b), "R8_SYNTHETIC_CREDENTIAL_49173"); got != want {
						t.Errorf("SmartTruncate(%q…%q) redacted the value: %v, want %v as for the map key or the baseline pass (err %v)", prefix, last, got, want, err)
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
// value of a map key of that name or the baseline pass redacts it in the same
// text, for names of up to four dotted words
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
	build([]string{"", "stripe", "api", "private", "secret", "auth", "code", "page", "next", "fb", "x"}, []string{"key_", "Key_", "verifier..", "code_", "token_", "Token--", "page_token_", "next_page_token_", "nextPageToken."}, "", 2)
	sensitive := 0
	for _, name := range names {
		want := isSensitiveKey(name)
		if want {
			sensitive++
		}
		for _, in := range []string{name + "=R7_SYNTHETIC_CREDENTIAL_9327", "x " + name + ": R7_SYNTHETIC_CREDENTIAL_9327"} {
			want := want || baselineRedacts(in, "R7_SYNTHETIC_CREDENTIAL_9327")
			if got := !strings.Contains(scrubTextPatterns(in), "R7_SYNTHETIC_CREDENTIAL_9327"); got != want {
				t.Errorf("scrubTextPatterns(%q) redacted the value: %v, want %v as for the map key or the baseline pass", in, got, want)
			}
			b, _, err := smartTruncateText(t, in, 1000)
			if got := err == nil && !strings.Contains(string(b), "R7_SYNTHETIC_CREDENTIAL_9327"); got != want {
				t.Errorf("SmartTruncate(%q) = (%s, %v), redacted the value: %v, want %v as for the map key or the baseline pass", in, b, err, got, want)
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

// scrubEveryPath scrubs in as scrubTextPatterns, sanitizeRawText,
// redactErrorMessage and SmartTruncate do, which read text with the same
// passes under their own entry points.
func scrubEveryPath(in string) {
	_ = scrubTextPatterns(in)
	_, _ = sanitizeRawText(in)
	_ = redactErrorMessage(in)
	_, _, _ = SmartTruncate(in, 1<<20)
}

// TestScrubTextPatterns_CamelCaseScanIsLinear checks that the search for
// camelCase keys does not rescan the text for each key or boundary in it: a
// run of n repeats must take about n times as long as one of n/4, not n^2.
func TestScrubTextPatterns_CamelCaseScanIsLinear(t *testing.T) {
	// The race detector slows the scrubs about twentyfold. A row whose
	// large run would take longer than the limit below under it uses a
	// sixteenth of its size there, so that the run takes about as long as
	// it does at full size without the race detector. For such a row, only
	// the run at full size tells linear growth from quadratic.
	raceScaled := func(n int) int {
		if raceEnabled {
			return n / 16
		}
		return n
	}
	tests := []struct {
		name string
		gen  func(n int) string
		run  func(in string)
		n    int
	}{
		{
			"repeated camel case pairs through SmartTruncate", func(n int) string { return strings.Repeat("x dbPassword=synthetic ", n) },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 2000,
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
			"long run of oauth boundaries with an inner dot through SmartTruncate", func(n int) string { return "x " + strings.Repeat("xOauth", n) + ".x.code=x" },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 40000,
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
			"spaced oversized keys through SmartTruncate", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat(strings.Repeat("x_", 70)+"PASSWORD=", n) + " x"
			},
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 400,
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
			"oversized run of dotted names that are not credentials through SmartTruncate", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("x.next_page_token:", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 1600,
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
			"bare run of dotted names that are not credentials through SmartTruncate", func(n int) string { return strings.Repeat("x.next_page_token:", n) },
			func(in string) { _, _, _ = SmartTruncate(in, 1<<20) }, 1600,
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
		// Each word of a run of OAuth words starts a dotted OAuth name that
		// no ".code" and delimiter end, which the pattern reads to the end
		// of the run before it fails: the run is read once, not for each
		// word, whichever way the paths read the text.
		{
			"one run of hyphenated OAuth words with no dotted name", func(n int) string { return strings.Repeat("oauth-", n) },
			scrubEveryPath, 8000,
		},
		{
			"one run of underscored OAuth words with no dotted name", func(n int) string { return strings.Repeat("oauth_", n) },
			scrubEveryPath, 8000,
		},
		{
			"one run of mixed OAuth words with no dotted name", func(n int) string { return strings.Repeat("oauth-oauth_", n/2) },
			scrubEveryPath, 8000,
		},
		{
			"one run of hyphenated OAuth words before a code with no delimiter", func(n int) string { return strings.Repeat("oauth-", n) + ".code" },
			scrubEveryPath, 8000,
		},
		{
			"one run of underscored OAuth words before a code with no value", func(n int) string { return strings.Repeat("oauth_", n) + ".code " },
			scrubEveryPath, 8000,
		},
		{
			"one run of OAuth words before a code that a quote and no delimiter follow", func(n int) string { return strings.Repeat("oauth-", n) + `.code"` },
			scrubEveryPath, 8000,
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
			"oversized run of dotted OAuth words through SmartTruncate", func(n int) string {
				return "credentials:\n a_PASSWORD: " + strings.Repeat("oauth.", n) + strings.Repeat("x_", 70) + `PASSWORD="a b"`
			},
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 8000,
		},
		{
			// Each dotted name ends in a name that its complete dotted name
			// decides on, which is read from the last bytes of the run.
			"bare run of dotted cursor names", func(n int) string { return strings.Repeat("x.next.page.token:", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 1600,
		},
		{
			"bare run of dotted cursor names through SmartTruncate", func(n int) string { return strings.Repeat("x.next.page.token:", n) },
			func(in string) { _, _, _ = SmartTruncate(in, 1<<20) }, 1600,
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
			"one long dotted run before a page access token through SmartTruncate", func(n int) string { return strings.Repeat("a.", n) + "fb.page_token=v" },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 10000,
		},
		{
			// The name is read whole, past the bound, from the first
			// word before the dots.
			"one long run of dots before a page access token", func(n int) string { return "a" + strings.Repeat(".", n) + "page_token=v" },
			func(in string) { _ = scrubTextPatterns(in) }, 40000,
		},
		{
			"one long run of dots before a page access token through SmartTruncate", func(n int) string { return "a" + strings.Repeat(".", n) + "page_token=v" },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 40000,
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
			"one long run of strings joined to a credential through SmartTruncate", func(n int) string { return "password=" + strings.Repeat(`'a'"b"`, n) },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 10000,
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
			"repeated multi-separator credential keys through SmartTruncate", func(n int) string {
				return strings.Repeat("stripe.api..key=v ", n)
			},
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 2000,
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
		{
			"repeated camelCase keys with long trailing separator runs", func(n int) string {
				return strings.Repeat("OpenAIAPIKey"+strings.Repeat("_", 150)+"=v ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated compound keys with mixed trailing separators through SmartTruncate", func(n int) string {
				return strings.Repeat("stripe.api..key_.-=v ", n)
			},
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 2000,
		},
		{
			"repeated semicolon-containing contraction credentials", func(n int) string {
				return strings.Repeat("export DB_PASSWORD=a:'s correct staple horse; the user's; echo ok\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1500,
		},
		{
			"repeated Bearer matches inside an unclosed non-sensitive single quote", func(n int) string {
				return "x='" + strings.Repeat(" Bearer 'a", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated alternating single-quoted Bearer tokens", func(n int) string {
				return strings.Repeat("a='b Bearer 'c ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated empty-quoted Bearer tokens inside open single quote", func(n int) string {
				return "p='" + strings.Repeat("Bearer ''", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated joined Bearer credentials on one line", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=a:'s Bearer '"x" `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 500,
		},
		{
			"plain prefix lines before repeated open-quote Bearer matches", func(n int) string {
				return strings.Repeat("plain log line without credentials\n", 2000) + "y='" + strings.Repeat("Bearer ''", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated state-prefixed and padded temp keys in single-quoted dicts", func(n int) string {
				return strings.Repeat(`failed: {' user: DB_PASSWORD ': 'x', ' temp:a b ': 'y'} `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated temp keys that hold or end in a delimiter", func(n int) string {
				return strings.Repeat(`{'temp:a=': 'x', "temp:=b": "y"} `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated padded quoted temp keys after contractions", func(n int) string {
				return strings.Repeat(`token=abc:'s note ' user: temp:a=b ': ' x y'z `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated joined quotes after later labels", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=a:'s note label='"x" `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quotes joined inside bare values", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=pre"a b"c'd e'"f" `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quotes joined inside bare values before pairs", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=pre"a b"token=x `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated redacted values before pairs", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=[REDACTED]token=[REDACTED] `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		// A run of redacted pairs in one word, with no whitespace between
		// them, whose bare values would each run on to the end of the word.
		{
			"one word of redacted pairs", func(n int) string { return strings.Repeat("password=[REDACTED]", n) },
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one word of redacted pairs through SmartTruncate", func(n int) string { return strings.Repeat("password=[REDACTED]", n) },
			func(in string) { _, _, _ = SmartTruncate(in, 100) }, 2000,
		},
		{
			"one word of redacted compound and plain pairs", func(n int) string {
				return strings.Repeat("DB_PASSWORD=[REDACTED]token=[REDACTED]", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"one word of redacted pairs and quoted keys", func(n int) string {
				return strings.Repeat(`password=[REDACTED]"token":"[REDACTED]"`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"one word of quoted temp keys that hold delimiters and redacted values", func(n int) string {
				return strings.Repeat(`'temp:a=b':[REDACTED]`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one bare value joined to many quoted parts", func(n int) string {
				return "export DB_PASSWORD=pre" + strings.Repeat(`"a b"c`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one bare value joined to many quoted parts and an unclosed quote", func(n int) string {
				return "export DB_PASSWORD=pre" + strings.Repeat(`"a b"c`, n) + `"tail`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one bare value joined to many quoted parts of both kinds", func(n int) string {
				return "export DB_PASSWORD=pre" + strings.Repeat(`'a b'"c d"`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one bare value joined to many single-quoted parts and an unclosed quote", func(n int) string {
				return "export DB_PASSWORD=pre" + strings.Repeat(`'a b'c`, n) + `'tail`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated bare parts joined to quoted parts that end in later labels", func(n int) string {
				return strings.Repeat(`DB_PASSWORD=a:'s note label='x `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		// Pairs and quoted keys that are not sensitive, joined to a redacted
		// value, end no value, so each word is one value read once.
		{
			"one word of redacted values before plain pairs", func(n int) string {
				return strings.Repeat("password=[REDACTED]x=1", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one word of redacted values before plain quoted keys", func(n int) string {
				return strings.Repeat(`password=[REDACTED]"x": "1"`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"one word of quoted values that run on over delimited values", func(n int) string {
				return strings.Repeat(`password="a"x: "b"`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quoted temp keys that hold commas", func(n int) string {
				return strings.Repeat(`{'temp:a,b': 'x', 'a,b': 'y'} `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated quoted arguments after brackets that hold pairs", func(n int) string {
				return strings.Repeat(`credentials: token("x token=y z") `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one quoted argument after a bracket with many words before a pair", func(n int) string {
				return `password=f("` + strings.Repeat("a ", n) + `token=x")`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			"repeated closing quotes that open quoted pairs", func(n int) string {
				return strings.Repeat(`password=a"b x "token=y-token: " in config `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1000,
		},
		{
			"repeated cursor keys whose direction words are split", func(n int) string {
				return strings.Repeat("n_e_x_t_page_token=c p_r_e_v_i_o_u_s_page_token=d ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"one word of quoted pairs after a quoted credential", func(n int) string {
				return `password="x"` + strings.Repeat(`"token":"x"`, n)
			},
			func(in string) {
				_ = scrubTextPatterns(in)
				_, _, _ = SmartTruncate(in, 100)
			}, 1200,
		},
		{
			"one word of empty-valued quoted pairs after a quoted credential", func(n int) string {
				return `password="x"` + strings.Repeat(`"token":""`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1200,
		},
		{
			"one word of single-quoted pairs after a single-quoted credential", func(n int) string {
				return `password='x'` + strings.Repeat(`'token':'x'`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1200,
		},
		{
			"one word of preserved token-counter pairs after a quoted credential", func(n int) string {
				return `password="x"` + strings.Repeat(`"prompt_tokens":1,`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1200,
		},
		{
			"one word of inter-string prose after quoted credentials", func(n int) string {
				return `password="x"` + strings.Repeat(`"token":" a b "x"`, n)
			},
			func(in string) {
				_ = scrubTextPatterns(in)
				_, _, _ = SmartTruncate(in, 100)
			}, 1200,
		},
		{
			"one word of single-quoted inter-string prose after quoted credentials", func(n int) string {
				return `password='x'` + strings.Repeat(`'token':' a b 'x'`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 1200,
		},
		{
			"closing parentheses after a single-quoted bearer token", func(n int) string {
				return `x='Bearer abc'` + strings.Repeat(")", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 128000,
		},
		{
			"closers and dots after a word joined to a single-quoted bearer token", func(n int) string {
				return `echo 'Bearer abc'x` + strings.Repeat(").", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 64000,
		},
		{
			"closing braces after a word joined to a bearer token", func(n int) string {
				return `Bearer abc'x` + strings.Repeat("}", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 128000,
		},
		{
			"mixed closers after a word joined to a single-quoted bearer token", func(n int) string {
				return `echo 'Bearer abc'x` + strings.Repeat(").]}.)", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 24000,
		},
		{
			// A quote of the token that its string does not close must not
			// pair with the next object's and read on for each token,
			// whether or not what joins the objects ends a value.
			"glued JSON objects whose bearer token holds a single quote", func(n int) string {
				return strings.Repeat(`{"auth": "Bearer a'b"}`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"glued single-quoted objects whose bearer token holds a double quote", func(n int) string {
				return strings.Repeat(`{'auth': 'Bearer a"b'}`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"JSON objects joined by a plus whose bearer token holds a single quote", func(n int) string {
				return strings.Repeat(`{"auth": "Bearer a'b"}+`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			// A string, and the strings nested in it, are read once however
			// many tokens they hold (see openQuotes.innerAt).
			"bearer tokens in strings nested in one string", func(n int) string {
				return `msg="` + strings.Repeat(`{\"auth\": \"Bearer a'b\"}`, n) + `"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 4000,
		},
		{
			// A line is searched for the ends of its strings once however
			// many strings that do not end on it hold tokens (see
			// openQuotes.endOnLine).
			"bearer tokens in strings that do not end on their line", func(n int) string {
				return strings.Repeat(`"Bearer a'b"x`, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 8000,
		},
		{
			// Telling a shell's single-quoted part from quotedValueRegex's
			// reads a bounded part of the line (see
			// findQuotedOrShellSegment).
			"single-quoted shell parts closed by an escaped quote on one line", func(n int) string {
				return strings.Repeat(`export DB_PASSWORD=pre' S\'; export API_KEY=pre2' T'; `, n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 2000,
		},
		{
			"repeated single-quoted contraction in double quotes", func(n int) string {
				return `message="` + strings.Repeat("password='it's ", n) + `"`
			},
			func(in string) { _ = scrubTextPatterns(in) }, 16000,
		},
		{
			"repeated single-quoted contraction in double quotes on every path", func(n int) string {
				return `message="` + strings.Repeat("password='it's ", n) + `"`
			},
			scrubEveryPath, raceScaled(8000),
		},
		{
			"multiline unclosed call-wrapped single-quoted contraction", func(n int) string {
				return strings.Repeat("password=f('it's \n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, 4000,
		},
		{
			"multiline unclosed call-wrapped single-quoted contraction on every path", func(n int) string {
				return strings.Repeat("password=f('it's \n", n)
			},
			scrubEveryPath, raceScaled(4000),
		},
		{
			// Where the quote that holds each Cookie header closes is
			// found once for all the headers in it (see
			// headerQuoteScanner.closeAt).
			"repeated quoted cookie prose in double quotes", func(n int) string {
				return `message="` + strings.Repeat("The cookie: chocolate was delicious. ", n) + `"`
			},
			scrubEveryPath, raceScaled(8000),
		},
		{
			"repeated JSON cookie prose with quantities", func(n int) string {
				return `{"note":"` + strings.Repeat("The cookie: 2.5 ounces remained. ", n) + `"}`
			},
			scrubEveryPath, raceScaled(16000),
		},
		{
			"repeated hyphen-prefixed cookie prose on one line", func(n int) string {
				return strings.Repeat("my-cookie: Monster ate x-set-cookie: Monster ate ", n)
			},
			scrubEveryPath, raceScaled(4000),
		},
		{
			"repeated delimiter-space-before-colon cookie prose on one line", func(n int) string {
				return strings.Repeat("Cookie\u00a0: Monster ate Cookie\u2003: Monster ate Cookie\x0b: Monster ate ", n)
			},
			scrubEveryPath, raceScaled(4000),
		},
		{
			"repeated unclosed Bearer word-apostrophe quotes on one line", func(n int) string {
				return "p='" + strings.Repeat("Bearer 'words' ", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"nested joined apostrophe and colon quote lines", func(n int) string {
				return strings.Repeat("curl -H \"X-Api-Key: \"pre' ok\nAuthorization: 'Bearer ok'; echo x\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"unclosed joined apostrophe with unpaired trailing double quote", func(n int) string {
				return strings.Repeat("curl -H \"X-Api-Key: \"pre' ok\nCookie: 'tail'; echo \"unpaired\nPUBLIC_KEEP\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"plural possessive tails with CLI flags on one line", func(n int) string {
				return "curl -H \"X-Api-Key: \"users' -a " + strings.Repeat("users' -a ", n) + "-H 'X-Api-Key: SYNTH_SECRET'"
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"multiline nested headers with newline continuation", func(n int) string {
				return strings.Repeat("curl -H \"X-Api-Key: \"pre' s1\nAuthorization: 'Bearer tok1\ns2'; echo PUBLIC_KEEP\n", n)
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"plural possessive tails with same-line multi-header double-quoted target", func(n int) string {
				return "curl -H \"X-Api-Key: \"users' -a " + strings.Repeat("users' -a; curl -H \"X-Api-Key: \"users' -a ", n) + "-H \"Authorization: Bearer SYNTH_SECRET\""
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"many leading open parens before subshell closing apostrophe", func(n int) string {
				return strings.Repeat("(", n) + "curl -H \"X-Api-Key: \"users') && echo PUBLIC_KEEP"
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"multi JSON fields on one line with subshell and plural possessive", func(n int) string {
				return `{"pre":"(x)` + strings.Repeat(`","k":"curl -H \"X-Api-Key: \"users') && echo y`, n) + `"}`
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(1500),
		},
		{
			"many unclosed inline parens before semicolon", func(n int) string {
				return `curl -H "X-Api-Key: "pre` + strings.Repeat("(", n) + `tok; echo PUBLIC_KEEP`
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"many unclosed inline parens before single close paren", func(n int) string {
				return `curl -H "X-Api-Key: "pre` + strings.Repeat("(", n) + `x); echo PUBLIC_KEEP`
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
		},
		{
			"many single-line JSON fields with apostrophe subshell tail", func(n int) string {
				return `{"pre":"(x)` + strings.Repeat(`","k":"curl -H \"X-Api-Key: \"users') && echo y"`, n) + `"}`
			},
			func(in string) { _ = scrubTextPatterns(in) }, raceScaled(2000),
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
// while non-credential names and pagination cursors remain preserved, but for
// the cursors whose last word is token after a separator, whose value the
// baseline pass redacts.
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
		{"credentials:\n  stripe.api..key: SYNTHETIC_CREDENTIAL_6301\n  region: us", "credentials:[REDACTED]\n  region: us"},
		{"credentials:\n  foo.auth..code: SYNTHETIC_CREDENTIAL_6301\n  region: us", "credentials:[REDACTED]\n  region: us"},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
	}
	for _, name := range []string{
		"oauth..config.codes",
		"status..code",
		"access..id",
		"api.x.key",
		"my.api..key.file",
		"x.next..page__token",
	} {
		if isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = true, want false", name)
		}
		in := name + "=kept-value"
		checkRedactionPaths(t, in, in)
	}
	for _, name := range []string{
		"page..token",
		"next..page..token",
		"prev__page--token",
	} {
		if isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = true, want false", name)
		}
		checkRedactionPaths(t, name+"=kept-value", name+"=kept-value")
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
		{`export DB_PASSWORD=a:'s correct staple horse; the user's`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s correct; horse battery's`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s correct staple horse; the user's; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:'s correct; horse battery's && echo 'ok'`, `export DB_PASSWORD=[REDACTED] && echo 'ok'`},
		{`export DB_PASSWORD=a:'s part1; part2; the user's+tail`, `export DB_PASSWORD=[REDACTED]`},
		{`cmd="export DB_PASSWORD=a:'s correct staple horse; the user's"`, `cmd="export DB_PASSWORD=[REDACTED]`},
		{`password=a:'s prefix CORRECTHORSE's`, `password=[REDACTED]`},
		{`password=a:'s correct horse battery`, `password=[REDACTED]`},
		{`password=a:'s correct; horse battery`, `password=[REDACTED]`},
		{`password=a:'s; horse battery`, `password=[REDACTED]`},
		{`password=a:'s val=SYNTHETIC_CREDENTIAL_6301`, `password=[REDACTED]`},
		{`export DB_PASSWORD=a:'d correct horse battery'd`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'re correct horse battery're rest`, `export DB_PASSWORD=[REDACTED] rest`},
		{`password=a: 's missing; the SYNTHETIC_CREDENTIAL_6301's request`, `password=[REDACTED] request`},
		{`export DB_PASSWORD=a:'s part1; SYNTHETIC_CREDENTIAL_6301's`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s missing; the user's+SYNTHETIC_CREDENTIAL_6301`, `export DB_PASSWORD=[REDACTED]`},
		{`password=a:'s bearer of bad news`, `password=[REDACTED]`},
		{`password=a:'s SYNTHETIC_CREDENTIAL_6301; retry later`, `password=[REDACTED]`},
		// Outer quote context closing before ";" preserves outer prose, whereas
		// bare brackets without an enclosing quote do not terminate a quoted
		// credential early.
		{`failed: "token=abc:'s"; the user's request`, `failed: "token=[REDACTED]; the user's request`},
		{`failed: "token=abc:'s"; retry later`, `failed: "token=[REDACTED]; retry later`},
		{`failed: {"msg": "token=abc:'s"}; the user's request`, `failed: {"msg": "token=[REDACTED]}; the user's request`},
		{`failed: ["token=abc:'s"]; retry later`, `failed: ["token=[REDACTED]]; retry later`},
		{`failed: {token=abc:'s}; the user's request`, `failed: {token=[REDACTED] request`},
		{`failed: [token=abc:'s]; retry later`, `failed: [token=[REDACTED]`},
		{`failed: {token=abc:'s}; Bearer ' SYNTHETIC_CREDENTIAL_6301'`, `failed: {token=[REDACTED]`},
		{`token=abc:'s; Bearer ' SYNTHETIC_CREDENTIAL_6301'`, `token=[REDACTED]; Bearer [REDACTED]`},
		{`token=abc:'s note password='SYNTHETIC_CREDENTIAL_6301' rest`, `token=[REDACTED] note password='[REDACTED]' rest`},
		{`token=abc:'s note Bearer ' SYNTHETIC_CREDENTIAL_6301 SYNTHETIC_CREDENTIAL_6302'suffix`, `token=[REDACTED] note Bearer [REDACTED]`},
		{`token=abc:'s DB_PASSWORD=SYNTHETIC_CREDENTIAL_6301`, `token=[REDACTED] DB_PASSWORD=[REDACTED]`},
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

// TestTrailingSeparatorKeyPolicyAgreement checks that structured maps, structs,
// free text, raw text, error messages, and string values agree on key names with
// trailing "_", "-" and "." separators (including compound, camelCase, acronym,
// dotted, state-prefixed, and oversized keys) while preserving counters,
// pagination cursors, and non-credential names with trailing separators.
func TestTrailingSeparatorKeyPolicyAgreement(t *testing.T) {
	const secret = "SYNTHETIC_CREDENTIAL_7319"
	sensitiveNames := []string{
		"DB_PASSWORD_",
		"GITHUB_TOKEN_",
		"OpenAIAPIKey_",
		"stripe.api..key_",
		"DB_PASSWORD___",
		"GITHUB_TOKEN---",
		"DB_PASSWORD.",
		"OpenAIAPIKey_.-",
		"OpenAIAPI_Key_",
		"OpenAI.APIKey_",
		"stripe.api..key.._",
		"stripeApi..key____",
		"awsSecretAccessKey_",
		"fb.page_token_",
		"foo.auth..code_",
		"user:DB_PASSWORD_",
		"app:api_key_",
		"_password_",
		"---token___",
		"OpenAIAPIKey" + strings.Repeat("_", 150),
		strings.Repeat("x_", 70) + "PASSWORD_",
	}
	for _, name := range sensitiveNames {
		if !isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = false, want true", name)
		}
		gotMap, _, err := SmartTruncate(map[string]string{name: secret}, 10000)
		if err != nil || strings.Contains(string(gotMap), secret) || !strings.Contains(string(gotMap), redactedValue) {
			t.Errorf("SmartTruncate(map[%q]) = (%s, %v), want redacted value", name, gotMap, err)
		}
		checkAllRedactionPaths(t, name+"="+secret, name+"=[REDACTED]")
		checkAllRedactionPaths(t, `export `+name+`="`+secret+`"`, `export `+name+`="[REDACTED]"`)
		checkAllRedactionPaths(t, "failed: "+name+": "+secret+" retry", "failed: "+name+":[REDACTED] retry")
		checkAllRedactionPaths(t, `failed: "`+name+`": "`+secret+`"`, `failed: "`+name+`":"[REDACTED]"`)
		checkAllRedactionPaths(t, `failed: '`+name+`': '`+secret+`'`, `failed: '`+name+`':'[REDACTED]'`)
		checkRedactionPaths(t, `{"`+name+`":"`+secret+`","~ok":"1"}`, `{"`+name+`":"[REDACTED]","~ok":"1"}`)
		checkRedactionPaths(t, "credentials:\n  "+name+": "+secret+"\n  region: us", "credentials:[REDACTED]\n  region: us")
	}

	// Leading-dot standalone and camelCase keys preserve the leading dot in text.
	for _, name := range []string{".api_key_", "...api_key...", ".OpenAIAPIKey_"} {
		if !isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = false, want true", name)
		}
		checkAllRedactionPaths(t, name+"="+secret, name+"=[REDACTED]")
		checkRedactionPaths(t, `{"`+name+`":"`+secret+`"}`, `{"`+name+`":"[REDACTED]"}`)
		checkRedactionPaths(t, "credentials:\n  "+name+": "+secret+"\n  region: us", "credentials:[REDACTED]\n  region: us")
	}

	// Adjacent compound pairs with trailing separators preserve both keys.
	checkAllRedactionPaths(t, `DB_PASSWORD_="benign"GITHUB_TOKEN_="`+secret+`"`, `DB_PASSWORD_="[REDACTED]"GITHUB_TOKEN_="[REDACTED]"`)
	checkAllRedactionPaths(t, `DB_PASSWORD_="benign"stripe.api..key_="`+secret+`"`, `DB_PASSWORD_="[REDACTED]"stripe.api..key_="[REDACTED]"`)

	// Struct fields and json tags with trailing separators agree with text/map policy.
	type trailingSepRecord struct {
		DB_PASSWORD_  string
		OpenAIAPIKey_ string
		StripeKey     string `json:"stripe.api..key_"`
		PageToken_    string `json:"page_token_"`
		TokenCount_   int    `json:"token_count_"`
	}
	gotStruct, _, err := SmartTruncate(trailingSepRecord{
		DB_PASSWORD_:  secret,
		OpenAIAPIKey_: secret,
		StripeKey:     secret,
		PageToken_:    "cursor-42",
		TokenCount_:   7,
	}, 10000)
	if err != nil {
		t.Fatalf("SmartTruncate(trailingSepRecord) error = %v", err)
	}
	if strings.Contains(string(gotStruct), secret) {
		t.Errorf("SmartTruncate(trailingSepRecord) leaked %s: %s", secret, gotStruct)
	}
	if !strings.Contains(string(gotStruct), `"page_token_":"cursor-42"`) || !strings.Contains(string(gotStruct), `"token_count_":7`) {
		t.Errorf("SmartTruncate(trailingSepRecord) over-redacted cursor/counter: %s", gotStruct)
	}

	// Non-credential keys, counters, OAuth non-code suffixes, and pagination cursors
	// with trailing separators stay unredacted across structured and text paths.
	for _, name := range []string{
		"page_token_",
		"next_page_token_",
		"next-page-token-",
		"prev..page..token..",
		"x.next.page.token_",
		"nextPageToken.",
		"token_count_",
		"prompt_tokens_",
		"max_output_tokens_",
		"method_signature_",
		"status..code_",
		"oauth2Config.code_",
		"my.api..key.file_",
		"mypassword_",
	} {
		if isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = true, want false", name)
		}
		gotMap, _, err := SmartTruncate(map[string]string{name: "kept-value"}, 10000)
		if err != nil || !strings.Contains(string(gotMap), "kept-value") {
			t.Errorf("SmartTruncate(map[%q]) = (%s, %v), want kept-value preserved", name, gotMap, err)
		}
		in := name + "=kept-value"
		checkRedactionPaths(t, in, in)
	}
}

// TestScrubTextPatterns_QuotedPasswordWithInnerCloserInsideOuterBrackets
// checks that "}" and "]" inside a quoted credential opened after ":" or "="
// never decrement outer bracket depth or terminate redaction early, while
// genuine enclosing quotes still preserve outer prose.
func TestScrubTextPatterns_QuotedPasswordWithInnerCloserInsideOuterBrackets(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`command: { export DB_PASSWORD=a:'s} R12_SYNTHETIC_90517'; }`, `command: { export DB_PASSWORD=[REDACTED]; }`},
		{`command: { export DB_PASSWORD=a:'s} R12_SYNTHETIC_90517'; echo ok; }`, `command: { export DB_PASSWORD=[REDACTED]; echo ok; }`},
		{`failed: {password=a:'s} R12_SYNTHETIC_90517'suffix}`, `failed: {password=[REDACTED]}`},
		{`command: [ export DB_PASSWORD=a:'s] R12_SYNTHETIC_90517'; ]`, `command: [ export DB_PASSWORD=[REDACTED]; ]`},
		{`failed: [password=a:'s] R12_SYNTHETIC_90517'suffix]`, `failed: [password=[REDACTED]]`},
		{`command: { export DB_PASSWORD=a:'s} R12_SYNTHETIC_90517; }`, `command: { export DB_PASSWORD=[REDACTED]`},
		{`command: [ export DB_PASSWORD=a:'s] R12_SYNTHETIC_90517; ]`, `command: [ export DB_PASSWORD=[REDACTED]`},
		{`cmd="command: { export DB_PASSWORD=a:'s} R12_SYNTHETIC_90517'; }"`, `cmd="command: { export DB_PASSWORD=[REDACTED]; }"`},
		{`failed: "token=abc:'s"; the user's request`, `failed: "token=[REDACTED]; the user's request`},
		{`failed: {"msg": "token=abc:'s"}; retry later`, `failed: {"msg": "token=[REDACTED]}; retry later`},
		{`failed: {note=a:'s} 'password': ' R12_SYNTHETIC_90517'suffix`, `failed: {note=a:'s} 'password':[REDACTED]`},
		{`failed: {note=a:'s} Bearer ' R12_SYNTHETIC_90517'suffix`, `failed: {note=a:'s} Bearer [REDACTED]`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
}

// TestIntraWordSplitCompoundKeysPolicyAgreement checks that compound,
// camelCase, acronym, dotted, and state-prefixed keys whose credential suffix
// contains "_" / "-" / "." separators between letters (not only at dictionary
// underscores) are redacted identically in structured maps/structs and text
// discovery while rejecting non-credential suffixes and pagination cursors.
func TestIntraWordSplitCompoundKeysPolicyAgreement(t *testing.T) {
	const secret = "R12_SYNTHETIC_90517"
	sensitiveNames := []string{
		"DB_PASS_WORD",
		"DB_PASS__WORD",
		"dbPass_Word",
		"DBPASS_WORD",
		"OpenAIAP_IKey",
		"stripe.api.k_ey",
		"AWS_SEC_RET_ACC_ESS_KEY",
		"DB_P_A_S_S_W_O_R_D",
		"x_goog_sec_urity_to_ken",
		"fb.pa_ge.to_ken",
		"homepage_to_ken",
		"user:DB_PASS_WORD",
	}
	for _, name := range sensitiveNames {
		if !isSensitiveKey(name) {
			t.Fatalf("isSensitiveKey(%q) = false, want true", name)
		}
		gotMap, _, err := SmartTruncate(map[string]string{name: secret}, 10000)
		if err != nil || strings.Contains(string(gotMap), secret) {
			t.Errorf("SmartTruncate(map[%q]) = (%s, %v), leaked secret", name, gotMap, err)
		}
		checkAllRedactionPaths(t, name+"="+secret, name+"=[REDACTED]")
		checkAllRedactionPaths(t, "failed: "+name+": "+secret, "failed: "+name+":[REDACTED]")
		checkAllRedactionPaths(t, `failed: {"`+name+`": "`+secret+`"}`, `failed: {"`+name+`":"[REDACTED]"}`)
	}

	type splitKeyRecord struct {
		DBPassWord string `json:"DB_PASS_WORD"`
		CamelKey   string `json:"dbPass_Word"`
		DottedKey  string `json:"stripe.api.k_ey"`
		PageCursor string `json:"next_page_to_ken"`
		TokenCount int    `json:"to_ken_count"`
	}
	gotStruct, _, err := SmartTruncate(splitKeyRecord{
		DBPassWord: secret,
		CamelKey:   secret,
		DottedKey:  secret,
		PageCursor: "cursor-99",
		TokenCount: 12,
	}, 10000)
	if err != nil {
		t.Fatalf("SmartTruncate(splitKeyRecord) error = %v", err)
	}
	if strings.Contains(string(gotStruct), secret) {
		t.Errorf("SmartTruncate(splitKeyRecord) leaked %s: %s", secret, gotStruct)
	}
	if !strings.Contains(string(gotStruct), `"next_page_to_ken":"cursor-99"`) || !strings.Contains(string(gotStruct), `"to_ken_count":12`) {
		t.Errorf("SmartTruncate(splitKeyRecord) over-redacted cursor/counter: %s", gotStruct)
	}

	for _, safe := range []string{
		"mypass_word",
		"method_sig_nature",
		"to_ken_count",
		"next_page_to_ken",
		"nextPageTo_ken",
		"x.next.page.to_ken",
	} {
		if isSensitiveKey(safe) {
			t.Fatalf("isSensitiveKey(%q) = true, want false", safe)
		}
		in := safe + "=kept-value"
		checkRedactionPaths(t, in, in)
	}
}

// TestPaddedQuotedCompoundKeysPolicyAgreement checks that interior padding
// (ASCII spaces, tabs, vertical tabs, and Unicode spaces) inside quoted keys is
// trimmed for classification across compound, camelCase, dotted, OAuth, temp:,
// and state-prefixed keys, while preserving padded non-sensitive keys,
// counters, and pagination cursors.
func TestPaddedQuotedCompoundKeysPolicyAgreement(t *testing.T) {
	const secret = "R12_SYNTHETIC_90517"
	pads := []struct{ l, r string }{
		{" ", " "},
		{"  ", ""},
		{"", "  "},
		{"\t", "\t"},
		{"\v", "\v"},
		{"\u00a0", "\u00a0"},
		{"\u3000", " "},
	}
	sensitiveKeysToTest := []string{
		"password",
		"DB_PASSWORD",
		"DB_PASS_WORD",
		"dbPassword",
		"dbPass_Word",
		"OpenAIAPIKey",
		"stripe.api.key",
		"oauth2Config.code",
		"status.oauth2Config.extra.code",
		"user:DB_PASSWORD",
		"app:dbPassword",
		"user:api_key",
		"temp:scratch/val",
		"user:temp:scratch",
		strings.Repeat("x_", 70) + "PASSWORD",
	}
	for _, key := range sensitiveKeysToTest {
		for _, pad := range pads {
			padded := pad.l + key + pad.r
			checkAllRedactionPaths(t, `failed: {"`+padded+`": "`+secret+`"}`, `failed: {"`+padded+`":"[REDACTED]"}`)
			checkAllRedactionPaths(t, `failed: {'`+padded+`': '`+secret+`'}`, `failed: {'`+padded+`':'[REDACTED]'}`)
			checkAllRedactionPaths(t, `failed: "`+padded+`" = "`+secret+`"`, `failed: "`+padded+`"="[REDACTED]"`)
			checkAllRedactionPaths(t, `err="{\"`+padded+`\": \"`+secret+`\"}"`, `err="{\"`+padded+`\":\"[REDACTED]\"}"`)
		}
	}
	// Valid JSON, nested YAML envelope, stray preceding quote, and malformed JSON fail-closed.
	checkAllRedactionPaths(t, `{" DB_PASSWORD ":"`+secret+`"}`, `{" DB_PASSWORD ":"[REDACTED]"}`)
	checkAllRedactionPaths(t, "credentials:\n  \" DB_PASSWORD \": \""+secret+"\"", "credentials:\"[REDACTED]\"")
	checkAllRedactionPaths(t, `failed: "credentials": " DB_PASSWORD ": "`+secret+`"`, `failed: "credentials":"[REDACTED]"`)
	if got := scrubTextPatterns(`"credentials": " DB_PASSWORD ": "` + secret + `"`); got != `"credentials":"[REDACTED]"` {
		t.Errorf(`scrubTextPatterns("credentials": " DB_PASSWORD ": ...) = %q, want %q`, got, `"credentials":"[REDACTED]"`)
	}
	checkAllRedactionPaths(t, `bad " quote " DB_PASSWORD ": "`+secret+`"`, `bad " quote " DB_PASSWORD ":"[REDACTED]"`)
	if got, _ := sanitizeRawText(`{" DB_PASSWORD ":"` + secret + `"`); got != unparseableJSONPlaceholder {
		t.Errorf("sanitizeRawText(malformed padded compound JSON) = %q, want %q", got, unparseableJSONPlaceholder)
	}
	if !hasQuotedSensitiveKey(`{" user:temp:scratch ": }`) {
		t.Errorf("hasQuotedSensitiveKey(padded state-prefixed temp key) = false, want true")
	}
	for _, malformed := range []string{
		`[INFO] {" user:temp:scratch ": }`,
		`[INFO] %7B%22%20DB_PASSWORD%20%22%3A%7D`,
	} {
		if got, _ := sanitizeRawText(malformed); got != unparseableJSONPlaceholder {
			t.Errorf("sanitizeRawText(%q) = %q, want %q", malformed, got, unparseableJSONPlaceholder)
		}
		if got, _, err := SmartTruncate(malformed, 10000); err != nil || string(got) != `"`+unparseableJSONPlaceholder+`"` {
			t.Errorf("SmartTruncate(%q) = (%s, %v), want %q", malformed, got, err, `"`+unparseableJSONPlaceholder+`"`)
		}
	}

	// Non-sensitive padded quoted keys, cursors, and counters remain intact.
	for _, safe := range []string{
		"next_page_token",
		"nextPageToken",
		"page_token",
		"x.next_page_token",
		"user:next_page_token",
		"token_count",
		"prompt_tokens",
		"mypassword",
		"method_signature",
		"label",
	} {
		for _, pad := range pads {
			padded := pad.l + safe + pad.r
			in := `failed: {"` + padded + `": "kept-cursor-123"}`
			checkAllRedactionPaths(t, in, in)
		}
	}
}

// TestScrubTextPatterns_JoinedBearerInShellPassword checks that a quoted shell
// credential fragment ending in "Bearer " immediately joined (with no
// intervening delimiter whitespace after the closing quote) to another quote or
// word is redacted as a single credential, while whitespace-separated
// independent Bearer credentials continue to redact separately.
func TestScrubTextPatterns_JoinedBearerInShellPassword(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{`export DB_PASSWORD=a:'s correct; horse Bearer '"R12_SYNTHETIC_7319"`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s correct; horse Bearer '"R12_SYNTHETIC_7319"'_tail'`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s Bearer ''R12_SYNTHETIC_7319'`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=a:'s Bearer 'joined'R12_SYNTHETIC_7319'`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD='my Bearer ''R12_SYNTHETIC_7319'; echo ok`, `export DB_PASSWORD='[REDACTED]'; echo ok`},
		{`export DB_PASSWORD=a:'s Bearer 'R12_SYNTHETIC_7319`, `export DB_PASSWORD=[REDACTED]`},
		{`token=abc:'s note Bearer ' R12_SYNTHETIC_7319'`, `token=[REDACTED] note Bearer [REDACTED]`},
		{`token=abc:'s Bearer " R12_SYNTHETIC_7319"`, `token=[REDACTED] Bearer [REDACTED]`},
		{`failed: "Bearer ""R12_SYNTHETIC_7319""`, `failed: "Bearer [REDACTED]`},
		{`failed: "Bearer ""R12_SYNTHETIC_7319"""`, `failed: "Bearer [REDACTED]`},
		{`next_page_token='a Bearer ''R12_SYNTHETIC_7319'`, `next_page_token='a Bearer [REDACTED]`},
		{`echo 'Bearer '"R12_SYNTHETIC_7319"`, `echo 'Bearer [REDACTED]'`},
		{`echo 'Bearer '"R12_SYNTHETIC_7319"; echo 'done'`, `echo 'Bearer [REDACTED]'; echo 'done'`},
		{`echo "Bearer "'R12_SYNTHETIC_7319'`, `echo "Bearer [REDACTED]"`},
		{`echo 'Bearer 'R12_SYNTHETIC_7319`, `echo 'Bearer [REDACTED]`},
		{`x='Bearer '"R12_SYNTHETIC_7319"`, `x='Bearer [REDACTED]'`},
		{`note='x Bearer '"R12_SYNTHETIC_7319"`, `note='x Bearer [REDACTED]'`},
		{`password=hunter2 && echo 'Bearer '"R12_SYNTHETIC_7319"`, `password=[REDACTED] && echo 'Bearer [REDACTED]'`},
		{`password=a:'s Bearer '"R12_SYNTHETIC_7319" ; echo 'Bearer '"R12_SYNTHETIC_8420"`, `password=[REDACTED] ; echo 'Bearer [REDACTED]'`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	checkAllRedactionPaths(t, `{"msg":"ask bearer "}`, `{"msg":"ask bearer "}`)
	if got := scrubTextPatterns(`{"msg": "ask bearer "}`); got != `{"msg": "ask bearer "}` {
		t.Errorf(`scrubTextPatterns({"msg": "ask bearer "}) = %q, want unchanged`, got)
	}

	// Verify keyTrie handles digits and punctuation in future sensitive key entries without panicking.
	tr := &keyTrie{}
	for _, sample := range []string{"oauth2code", "x-api-key", "p12_secret"} {
		cur := tr
		for i := 0; i < len(sample); i++ {
			if cur.children[sample[i]] == nil {
				cur.children[sample[i]] = &keyTrie{}
			}
			cur = cur.children[sample[i]]
		}
		cur.end = true
	}
	if re := tr.tailRegex(); re == "" {
		t.Errorf("keyTrie.tailRegex() = empty, want non-empty pattern")
	}
}

// checkRedactedMap checks that SmartTruncate writes the value of key in a map
// as [REDACTED] and keeps the key.
func checkRedactedMap(t *testing.T, key, secret string) {
	t.Helper()
	b, _, err := SmartTruncate(map[string]string{key: secret}, 10000)
	var got map[string]string
	if err != nil || json.Unmarshal(b, &got) != nil || len(got) != 1 || got[key] != redactedValue {
		t.Errorf("SmartTruncate(map[%q]) = (%s, %v), want {%q: %q}", key, b, err, key, redactedValue)
	}
}

// TestStatePrefixedAndDelimiterTempKeysPolicyAgreement checks that a "temp:"
// key is sensitive whatever its name holds, after an optional "user:" or
// "app:" state prefix and padding, even when the name is empty or holds "=",
// ":", ";", "&" or whitespace, and that maps, structs, valid JSON and text
// agree on it: a quoted key that isSensitiveKey accepts is one that
// quotedKeyAllowed accepts, so the text paths read it whole and redact its
// value, quoted or not, rather than redacting part of the key and keeping the
// value. A state-prefixed key with whitespace after the prefix, such as
// " user: DB_PASSWORD ", is read whole too, and so is a key that holds a
// comma, as 'temp:a,b' does, whose quotes, not the comma, tell where it ends.
// The exceptions are the keys that hold a quote, "\" or a line break,
// which single-line quoted-key scanning reads as the structure of the text
// around a key: their values are redacted in maps, structs and valid JSON
// only.
func TestStatePrefixedAndDelimiterTempKeysPolicyAgreement(t *testing.T) {
	const secret = "R13_SYNTHETIC_52816"
	for _, key := range []string{
		"user:temp:scratch",
		"app:temp:scratch",
		" user: temp:scratch ",
		"user: temp:a b",
		"temp:",
		"user:temp:",
		"user:temp:=",
		"temp:a b",
		" temp:scratch=x ",
		" temp:a=b ",
		" temp:a:b ",
		" user:temp:a=b ",
		" user: temp:a=b ",
		" user: DB_PASSWORD ",
		"user: DB_PASSWORD",
		"temp:a:",
		"temp:a=",
		"temp:a=b c",
		"temp:a=b c=d",
		"temp:=",
		"temp::",
		"temp:=b",
		"temp::b",
		"temp:a=;",
		"temp:a=;b",
		"temp:a=&b",
		"temp:a=b;",
		"temp:a=b&",
		"temp:a=b;c",
		"temp:a=b&c",
		"user:temp:a=",
		"app:temp:a:",
		"TEMP:a=",
		"Temp:a=",
		"temp:a=b",
		"temp:a:b",
		"user:temp:a=b",
		"temp:cache=foo",
		"temp:a,b",
		"user:temp:a,b",
		" temp:a,b ",
		"user: temp:a,b",
		"temp:,",
		"temp:cache/{user}",
		"user:temp:a}b",
		" temp:a{b,c ",
		"temp:a}b",
		"temp:a{b",
		"temp:a{b c",
		"temp:a{b,c",
	} {
		if !isSensitiveKey(key) || !quotedKeyAllowed(key) {
			t.Errorf("isSensitiveKey(%q), quotedKeyAllowed(%q) = %v, %v, want true, true", key, key, isSensitiveKey(key), quotedKeyAllowed(key))
		}
		checkRedactedMap(t, key, secret)
		checkAllRedactionPaths(t, `failed: {'`+key+`': '`+secret+`'}`, `failed: {'`+key+`':'[REDACTED]'}`)
		checkAllRedactionPaths(t, `failed: {"`+key+`": "`+secret+`"}`, `failed: {"`+key+`":"[REDACTED]"}`)
	}
	// Valid JSON is read as JSON on every path.
	for _, key := range []string{"user:temp:scratch", " user: temp:a=b ", " user: DB_PASSWORD ", "temp:", "temp:a="} {
		in := `{"` + key + `": "` + secret + `"}`
		checkAllRedactionPaths(t, in, `{"`+key+`":"[REDACTED]"}`)
	}
	// A quoted key is read whole whatever follows its delimiter: a value
	// that is not quoted, as in the repr of a Python dict that holds a
	// bare word, in YAML or after a contraction, is redacted to its end,
	// and a key that holds the delimiter of a pair, as temp:a=b does, is
	// kept whole rather than read as a pair of its own.
	for _, tt := range []struct{ in, want string }{
		{`failed: {'temp:a=b': ` + secret + `}`, `failed: {'temp:a=b':[REDACTED]}`},
		{`failed: {'temp:a:b': ` + secret + `}`, `failed: {'temp:a:b':[REDACTED]}`},
		{`failed: {'user:temp:a=b': ` + secret + `}`, `failed: {'user:temp:a=b':[REDACTED]}`},
		{`failed: {'temp:cache=foo': ` + secret + `}`, `failed: {'temp:cache=foo':[REDACTED]}`},
		{`failed: {'temp:a,b': ` + secret + `}`, `failed: {'temp:a,b':[REDACTED]}`},
		{`failed: {"temp:a,b": ` + secret + `}`, `failed: {"temp:a,b":"[REDACTED]"}`},
		{`failed: {'user:temp:a,b': ` + secret + `}`, `failed: {'user:temp:a,b':[REDACTED]}`},
		{`failed: {'temp:cache/{user}': '` + secret + `'}`, `failed: {'temp:cache/{user}':'[REDACTED]'}`},
		{`failed: {'user:temp:a}b': '` + secret + `'}`, `failed: {'user:temp:a}b':'[REDACTED]'}`},
		{`failed: {' temp:a{b,c ': '` + secret + `'}`, `failed: {' temp:a{b,c ':'[REDACTED]'}`},
		// A comma-bearing or brace-bearing key between other entries: the entries around it
		// keep their keys and values.
		{`failed: {'label': 'keep', 'temp:a,b': '` + secret + `'}`, `failed: {'label': 'keep', 'temp:a,b':'[REDACTED]'}`},
		{`failed: {'temp:a,b': '` + secret + `', 'label': 'keep'}`, `failed: {'temp:a,b':'[REDACTED]', 'label': 'keep'}`},
		{`failed: {'label': 'keep', 'temp:cache/{user}': '` + secret + `'}`, `failed: {'label': 'keep', 'temp:cache/{user}':'[REDACTED]'}`},
		{`failed: {"label": "keep", "temp:a,b": "` + secret + `", "x,y": "z"}`, `failed: {"label": "keep", "temp:a,b":"[REDACTED]", "x,y": "z"}`},
		{`failed: {"temp:a=b": ` + secret + `}`, `failed: {"temp:a=b":"[REDACTED]"}`},
		{`failed: {"temp:a:b": ` + secret + `}`, `failed: {"temp:a:b":"[REDACTED]"}`},
		{`failed: {"user:temp:a=b": ` + secret + `}`, `failed: {"user:temp:a=b":"[REDACTED]"}`},
		{`failed: {" user: temp:scratch ": ` + secret + `}`, `failed: {" user: temp:scratch ":"[REDACTED]"}`},
		{`failed: {' user: temp:scratch ': ` + secret + `}`, `failed: {' user: temp:scratch ':[REDACTED]}`},
		{`failed: {" user: DB_PASSWORD ": ` + secret + `}`, `failed: {" user: DB_PASSWORD ":[REDACTED]}`},
		{`failed: {' user: DB_PASSWORD ': ` + secret + `}`, `failed: {' user: DB_PASSWORD ':[REDACTED]}`},
		{`failed: "temp:a=b" = ` + secret, `failed: "temp:a=b"=[REDACTED]`},
		{`failed: 'temp:a=b' = ` + secret, `failed: 'temp:a=b'=[REDACTED]`},
		{"credentials:\n  'temp:a=b': " + secret, "credentials:[REDACTED]"},
		{"credentials:\n  'temp:a=b': '" + secret + "'", "credentials:'[REDACTED]'"},
		{"credentials:\n  \"temp:a=b\": " + secret, "credentials:\"[REDACTED]\""},
		{`token=abc:'s note 'temp:a=b': ` + secret, `token=[REDACTED] note 'temp:a=b':[REDACTED]`},
		{`token=abc:'s note 'temp:a=b': '` + secret + `'`, `token=[REDACTED] note 'temp:a=b':'[REDACTED]'`},
		// A quoted "temp:" key that no delimiter follows is no key: the
		// text patterns read the pair inside it, as adk-python does.
		{`'temp:cache=` + secret + `'`, `'temp:cache=[REDACTED]`},
		{`value is 'temp:cache=` + secret + `' ok`, `value is 'temp:cache=[REDACTED] ok`},
		// Bare "temp:" keys that hold "{" or balanced "{...}" are redacted
		// in text and malformed-dict contexts, while template placeholders
		// such as "{temp:x}: hello world" keep their text.
		{`temp:a{b=` + secret, `temp:a{b=[REDACTED]`},
		{`temp:a{b: ` + secret, `temp:a{b:[REDACTED]`},
		{`user:temp:a{b=` + secret, `user:temp:a{b=[REDACTED]`},
		{`app:temp:a{b: ` + secret, `app:temp:a{b:[REDACTED]`},
		{`failed: {temp:a{b=` + secret + `}`, `failed: {temp:a{b=[REDACTED]}`},
		{`failed: {'temp:a{b: '` + secret + `'}`, `failed: {'temp:a{b:'[REDACTED]'}`},
		{`temp:cache/{user}=` + secret, `temp:cache/{user}=[REDACTED]`},
		{`map[temp:cache/{user}:` + secret + `]`, `map[temp:cache/{user}:[REDACTED]]`},
		{`template {temp:x}: hello world`, `template {temp:x}: hello world`},
		{`template {temp:cache/{user}}: hello world`, `template {temp:cache/{user}}: hello world`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}

	// A key that holds a quote, "\" or a line break is one that
	// isSensitiveKey accepts and quotedKeyAllowed does not: single-line
	// quoted-key scanning reads those bytes as the structure of the text
	// around a key. The value of such a key is redacted where it is read by
	// its key, in maps, structs and valid JSON, but not in text.
	for _, key := range []string{"temp:it's", `temp:a"b`, `temp:a\b`, "temp:a\nb", "temp:a\rb"} {
		if !isSensitiveKey(key) || quotedKeyAllowed(key) {
			t.Errorf("isSensitiveKey(%q), quotedKeyAllowed(%q) = %v, %v, want true, false", key, key, isSensitiveKey(key), quotedKeyAllowed(key))
		}
		checkRedactedMap(t, key, secret)
	}
	for _, key := range []string{"temp:a}b", "temp:a,b", "user: temp:a,b"} {
		in := `{"` + key + `": "` + secret + `"}`
		want := `{"` + key + `":"[REDACTED]"}`
		if got, _ := sanitizeRawText(in); got != want {
			t.Errorf("sanitizeRawText(%q) = %q, want %q", in, got, want)
		}
		if got := redactErrorMessage(in); got != want {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, want)
		}
	}
	for _, tt := range []struct{ in, want string }{
		{`failed: {'temp:a}b': '` + secret + `'}`, `failed: {'temp:a}b':'[REDACTED]'}`},
		{`failed: {"temp:a}b": "` + secret + `"}`, `failed: {"temp:a}b":"[REDACTED]"}`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}

	type tempKeyRecord struct {
		UserScratch string `json:"user:temp:scratch"`
		AppScratch  string `json:"app:temp:scratch"`
		PaddedKey   string `json:" user: DB_PASSWORD "`
		EmptyName   string `json:"temp:"`
		Assignment  string `json:"temp:a=b"`
		Label       string `json:"label"`
	}
	const wantRecord = `{" user: DB_PASSWORD ":"[REDACTED]","app:temp:scratch":"[REDACTED]","label":"kept-label","temp:":"[REDACTED]","temp:a=b":"[REDACTED]","user:temp:scratch":"[REDACTED]"}`
	gotStruct, _, err := SmartTruncate(tempKeyRecord{secret, secret, secret, secret, secret, "kept-label"}, 10000)
	if err != nil || string(gotStruct) != wantRecord {
		t.Errorf("SmartTruncate(tempKeyRecord) = (%s, %v), want %s", gotStruct, err, wantRecord)
	}
	gotMap, _, err := SmartTruncate(map[string]string{
		"user:temp:scratch":   secret,
		"app:temp:scratch":    secret,
		" user: DB_PASSWORD ": secret,
		"temp:":               secret,
		"temp:a=b":            secret,
		"label":               "kept-label",
	}, 10000)
	if err != nil || string(gotMap) != wantRecord {
		t.Errorf("SmartTruncate(map of tempKeyRecord keys) = (%s, %v), want %s", gotMap, err, wantRecord)
	}

	// Quoted keys nested in a value, after a contraction or in YAML.
	for _, tt := range []struct{ in, want string }{
		{`credentials: ' user: DB_PASSWORD ': '` + secret + `'`, `credentials:'[REDACTED]'`},
		{`credentials: 'temp:a=b': '` + secret + `'`, `credentials:'[REDACTED]'`},
		{"credentials:\n  ' user: DB_PASSWORD ': '" + secret + "'", "credentials:'[REDACTED]'"},
		{"credentials:\n  ' temp:a b ': '" + secret + "'", "credentials:'[REDACTED]'"},
		{`password=' user: DB_PASSWORD ': '` + secret + `'`, `password='[REDACTED]'`},
		{`token=abc:'s note ' user: DB_PASSWORD ': ' ` + secret + `'`, `token=[REDACTED] note ' user: DB_PASSWORD ':'[REDACTED]'`},
		{`token=abc:'s note ' temp:a=b ': ' ` + secret + `'`, `token=[REDACTED] note ' temp:a=b ':'[REDACTED]'`},
		{`token=abc:'s note 'temp:a b': ' ` + secret + `'`, `token=[REDACTED] note 'temp:a b':'[REDACTED]'`},
		// A delimiter-bearing temp key longer than a nested key may be is
		// read whole with the value it is nested in.
		{`credentials: 'temp:` + strings.Repeat("a", 200) + `=b;c': ` + secret, `credentials:'[REDACTED]'`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}

	// Keys that are not sensitive keep their values.
	for _, safe := range []string{"next_page_token", "user: next_page_token", "label", "temp", "user:temp", "my password"} {
		if isSensitiveKey(safe) || quotedKeyAllowed(safe) {
			t.Errorf("isSensitiveKey(%q), quotedKeyAllowed(%q) = %v, %v, want false, false", safe, safe, isSensitiveKey(safe), quotedKeyAllowed(safe))
		}
	}
	for _, safe := range []string{"next_page_token", "user: next_page_token", "label", "temp", "user:temp", "a,b", "label, x"} {
		for _, in := range []string{`failed: {"` + safe + `": "kept-value"}`, `failed: {'` + safe + `': 'kept-value'}`} {
			checkAllRedactionPaths(t, in, in)
		}
	}
}

// TestScrubTextPatterns_ShellJoinedQuoteAfterLaterLabel checks that the
// closing quote of a quoted part that ends in a later key and its delimiter
// does not open that key's value when a quote or a bare part follows it at
// once: a shell joins the parts into one word, so the credential they hold
// is redacted with the value rather than left after it. Only a later key
// that is itself sensitive, or one set apart from what follows by
// whitespace, still opens a pair of its own there.
func TestScrubTextPatterns_ShellJoinedQuoteAfterLaterLabel(t *testing.T) {
	const secret = "R13_SYNTHETIC_52816"
	for _, tt := range []struct{ in, want string }{
		{`export DB_PASSWORD=a:'s note label='"` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:'s note label=''` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:"s note label="'` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:'s note label='""` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=" note label="'` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// A bare part joined at once to the closing quote belongs to the
		// same word as well.
		{`export DB_PASSWORD=a:'s note label='` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse label="` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:'s note label='` + secret + ` rest`, `export DB_PASSWORD=[REDACTED] rest`},
		// A sensitive later key still opens its own pair.
		{`token=abc:'s note password='` + secret + ` VALUE' rest`, `token=[REDACTED] note password='[REDACTED]' rest`},
		// A later key whose value no quote joins keeps its value, and the
		// quoted value of a key that starts with a quote of its own is read
		// as before.
		{`DB_PASSWORD=a:'s note label='"x"`, `DB_PASSWORD=[REDACTED]`},
		{`token=abc:'s label=' x'`, `token=[REDACTED] label=' x'`},
		{`api_key=""x""`, `api_key="[REDACTED]"`},
		{`'error: token=abc; label="x"'`, `'error: token=[REDACTED]; label="x"'`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
}

// TestScrubTextPatterns_ShellQuoteJoinedInsideBareValue checks that a quote
// joined to the bytes of a bare value before it, with no delimiter between
// them, opens a quoted part of the same value, as a shell joins them, so that
// the words of the quoted part are redacted with the value, and that a value
// whose joined quote is not closed runs on to the end of the text. A quote
// that ends the bare value, so that what it quotes starts with whitespace, is
// read so too, and one that ends it at another delimiter or whose quoted part
// goes on to another line is read so when its closing quote ends the value or
// another quote follows it. A quote after an opening bracket opens an
// argument that is joined whole up to its closing quote, whatever pairs it
// holds, so that no closing quote is left in the text without the opening
// quote that the redacted value took, a closing quote that opens a quoted
// pair is left to it, the placeholder that a quoted part ends before a
// sensitive pair or quoted key is kept when the text is scrubbed again, a
// pair or a quoted key that is not sensitive is redacted with the value, and
// a single quote opens a quoted part unless it starts a contraction, so that
// apostrophes and quotes that close an outer string are read as before.
func TestScrubTextPatterns_ShellQuoteJoinedInsideBareValue(t *testing.T) {
	const secret = "R13_SYNTHETIC_52816"
	for _, tt := range []struct{ in, want string }{
		{`export DB_PASSWORD=pre"correct horse ` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'correct horse'"` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse ` + secret, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=pre" correct horse ` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre\"correct horse ` + secret + `\"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"a"b"c ` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'a'"b ` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse"token=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]token=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse "token=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]token=[REDACTED]; echo ok`},
		{`password=a"b x "token=` + secret + ` in config`, `password=[REDACTED]token=[REDACTED] in config`},
		// A quote that ends the bare value, so that what it quotes starts
		// with the whitespace that ended it, opens a quoted part too,
		// whether it is a single or a double quote, it is not closed, its
		// closing quote runs into a word, a quote or a pair, or it quotes
		// only whitespace.
		{`export DB_PASSWORD=pre' correct horse ` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre' correct horse'"` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre" correct horse""` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre" correct horse"` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre" correct horse ` + secret, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=pre" correct horse"token=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]token=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre" " rest`, `export DB_PASSWORD=[REDACTED] rest`},
		// A quote that ends the bare value at a delimiter that is not
		// whitespace, or whose quoted part goes on to another line, opens a
		// quoted part only if its closing quote ends the value or another
		// quote follows it.
		{`export DB_PASSWORD=pre",correct horse ` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre",correct horse""` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{"export DB_PASSWORD=pre\" correct\nhorse " + secret + "\" ok", `export DB_PASSWORD=[REDACTED] ok`},
		{"export DB_PASSWORD=pre\" correct\nhorse\"\"" + secret + "\"; echo ok", `export DB_PASSWORD=[REDACTED]; echo ok`},
		{"got password=abc\" \nwant \"x\"", "got password=[REDACTED] \nwant \"x\""},
		// The placeholder that a quoted part ends before a sensitive pair or
		// quoted key stands for the whole value, so scrubbing the text again
		// leaves it as it is. A pair or a quoted key that is not sensitive
		// is a part of the word that a shell joins to the value, and is
		// redacted with it.
		{`export DB_PASSWORD=pre"correct horse"label=x; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse"label=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse"token: ` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]token:[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse""token": "` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]"token":"[REDACTED]"; echo ok`},
		{`export DB_PASSWORD=pre"correct horse"'token'='` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]'token'='[REDACTED]'; echo ok`},
		{`export DB_PASSWORD=pre"correct horse""label": "x"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct horse""label": "` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// A quote after an opening bracket opens an argument, which is
		// joined whole, up to and including its closing quote, whatever
		// pairs it holds: the redacted value never takes the opening quote
		// while it leaves the closing quote, which would then open a string
		// that hides the text after it from the quote tracking, so that a
		// later credential on the line is kept or the rest of the text is
		// redacted with the value.
		{`password=f("correct horse ` + secret + `")`, `password=[REDACTED])`},
		{`export DB_PASSWORD=pre("correct horse ` + secret + `"); echo ok`, `export DB_PASSWORD=[REDACTED]); echo ok`},
		{`credentials: token("x ` + secret + `")`, `credentials:[REDACTED])`},
		{`password=f("correct horse token=` + secret + `")`, `password=[REDACTED])`},
		{`credentials: token("SECRET123 token=` + secret + `")`, `credentials:[REDACTED])`},
		{`password=f("SECRET123 token=SECRET456 ok"); DB_PASSWORD=pre" correct horse ` + secret + `"`, `password=[REDACTED]); DB_PASSWORD=[REDACTED]`},
		{`credentials: token("SECRET123 token=SECRET456 ok"); DB_PASSWORD=pre" correct horse ` + secret + `"`, `credentials:[REDACTED]); DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=f("SECRET123 token=SECRET456"); echo ok done`, `export DB_PASSWORD=[REDACTED]); echo ok done`},
		{`credentials: token("SECRET123 token=SECRET456"); echo ok done`, `credentials:[REDACTED]); echo ok done`},
		{`password=f('correct horse token=` + secret + `'); echo ok`, `password=[REDACTED]); echo ok`},
		// A closing quote that whitespace precedes and a sensitive pair
		// follows opens a quoted pair that a later quote closes.
		{`password=a"b x "token=SECRETP-token: " in config`, `password=[REDACTED] x "token=[REDACTED] " in config`},
		// Apostrophes and quotes that close an outer string open nothing.
		{`got password=abc", want "x"`, `got password=[REDACTED], want "x"`},
		{`got "password=abc" want "x"`, `got "password=[REDACTED] want "x"`},
		{`password=abc' OR '1'='1`, `password=[REDACTED] OR '1'='1`},
		{`SELECT * FROM users WHERE password=abc' AND '1'='1`, `SELECT * FROM users WHERE password=[REDACTED] AND '1'='1`},
		{`the password=users' wasn't changed`, `the password=[REDACTED] wasn't changed`},
		{`password=it's Bob's`, `password=[REDACTED] Bob's`},
		// A word that an "=" joins to the key before 's is the start of the
		// value unless 's makes a contraction of it, as it does of it, so the
		// single quote after abc opens a quoted part, closed or not, as the one
		// after pre in password=pre's synthlow's changed does.
		{`password=abc's foo bar's`, `password=[REDACTED]`},
		{`password=it's foo bar's`, `password=[REDACTED] foo bar's`},
		{`password=abc's`, `password=[REDACTED]`},
		{`the user's password=abc's value`, `the user's password=[REDACTED]`},
		{`the user's password=it's value`, `the user's password=[REDACTED] value`},
		{`token=abc"def" next`, `token=[REDACTED] next`},
		{`export DB_PASSWORD=pre"a b"; echo ok done`, `export DB_PASSWORD=[REDACTED]; echo ok done`},
		// A single quote inside the bare value that starts no contraction
		// opens a quoted part, which its closing quote closes wherever that
		// stands, also between two letters, and the text joined to it runs
		// on with the value, as a shell joins them; a part that is not
		// closed runs on to the end of the text. A single quote that ends
		// the bare value joins the text after its closing quote unless that
		// quote starts a contraction or follows whitespace.
		{`password=pre'a b'c R14_after`, `password=[REDACTED] R14_after`},
		{`password=pre'correct horse'staple`, `password=[REDACTED]`},
		{`export DB_PASSWORD=pre'correct horse ` + secret + `'suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'correct horse ` + secret + `'"x"suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'correct horse ` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + `'suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix; echo 'done'`, `export DB_PASSWORD=[REDACTED]; echo 'done'`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix && echo 'done'`, `export DB_PASSWORD=[REDACTED] && echo 'done'`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix || echo 'fail'`, `export DB_PASSWORD=[REDACTED] || echo 'fail'`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix | tr 'a' 'b'`, `export DB_PASSWORD=[REDACTED] | tr 'a' 'b'`},
		{`export DB_PASSWORD=pre' correct horse ` + secret + ` 'suffix --label 'ok'`, `export DB_PASSWORD=[REDACTED] --label 'ok'`},
		{`cmd --password=pre' ` + secret + ` 'suffix --label 'ok'`, `cmd --password=[REDACTED] --label 'ok'`},
		{`export DB_PASSWORD=pre' part1 'mid'` + secret + `'end`, `export DB_PASSWORD=[REDACTED]`},
		{`DB_PASSWORD=pre' part1 'mid' ` + secret + ` 'end`, `DB_PASSWORD=[REDACTED]`},
		{`DB_PASSWORD=pre'   '` + secret, `DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD=pre"x y": "` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`password=a : "x y": "` + secret + `"`, `password=[REDACTED]`},
		{`password=a = "x y": "` + secret + `"`, `password=[REDACTED]`},
		{`failed: "Bearer "'x y': '` + secret + `'"`, `failed: "Bearer [REDACTED]`},
		{`password=" in config, api_key=" "x y": "` + secret + `"`, `password="[REDACTED]"`},
		{`failed: "password=a:'s note 'temp:` + secret + `': "; retry`, `failed: "password=[REDACTED]`},
		{`failed: "password=a:'s note Bearer "; retry`, `failed: "password=[REDACTED]`},
		{`failed: "password=a:'s note token="; retry`, `failed: "password=[REDACTED]`},
		{`failed: "password=a:'s note 'temp:k': '` + secret + `'"; retry`, `failed: "password=[REDACTED] note 'temp:k':'[REDACTED]'"; retry`},
		{`password=doesn't work`, `password=[REDACTED] work`},
		{`the password=users' don't know`, `the password=[REDACTED] don't know`},
		{`password=chris' laptop's rest`, `password=[REDACTED] laptop's rest`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
		checkAllRedactionPaths(t, tt.want, tt.want)
	}
}

// TestIsPageTokenName_SplitDirectionWords checks that a pagination cursor
// key whose direction word, "next", "prev" or "previous", is split by
// separators, as in n_e_x_t_page_token or p_r_e_v_i_o_u_s_page_token, is read
// as a cursor and kept in maps, structs and text, while a credential key
// whose words only end in such a split, as in arn_ext_page_token, or that
// holds no direction word, as fb.page_token and page_access_token do, is
// still redacted. In text, the value of x.n_e_x_t.page.token is redacted
// too, as the baseline pass reads its last word as the credential key token.
func TestIsPageTokenName_SplitDirectionWords(t *testing.T) {
	const cursor = "cursor-123"
	const secret = "R13_SYNTHETIC_52816"
	for _, key := range []string{
		"n_e_x_t_p_a_g_e_t_o_k_e_n",
		"n_e_x_t_page_token",
		"x.n_e_x_t.page.token",
		"p_r_e_v_p_a_g_e_t_o_k_e_n",
		"p_r_e_v_page_token",
		"p_r_e_v_i_o_u_s_page_token",
		"N_E_X_T_PAGE_TOKEN",
		"n_ext_page_token",
	} {
		if !isPageTokenName(key) || isSensitiveKey(key) {
			t.Errorf("isPageTokenName(%q), isSensitiveKey(%q) = %v, %v, want true, false", key, key, isPageTokenName(key), isSensitiveKey(key))
		}
		b, _, err := SmartTruncate(map[string]string{key: cursor}, 10000)
		var got map[string]string
		if err != nil || json.Unmarshal(b, &got) != nil || len(got) != 1 || got[key] != cursor {
			t.Errorf("SmartTruncate(map[%q]) = (%s, %v), want {%q: %q}", key, b, err, key, cursor)
		}
		checkAllRedactionPaths(t, key+"="+cursor, key+"="+cursor)
		checkAllRedactionPaths(t, `failed: {"`+key+`": "`+cursor+`"}`, `failed: {"`+key+`": "`+cursor+`"}`)
	}
	type cursorRecord struct {
		Next     string `json:"n_e_x_t_page_token"`
		Previous string `json:"p_r_e_v_i_o_u_s_page_token"`
		Joined   string `json:"n_ext_page_token"`
		Facebook string `json:"fb.page_token"`
	}
	const wantRecord = `{"fb.page_token":"[REDACTED]","n_e_x_t_page_token":"cursor-123","n_ext_page_token":"cursor-123","p_r_e_v_i_o_u_s_page_token":"cursor-123"}`
	if got, _, err := SmartTruncate(cursorRecord{cursor, cursor, cursor, secret}, 10000); err != nil || string(got) != wantRecord {
		t.Errorf("SmartTruncate(cursorRecord) = (%s, %v), want %s", got, err, wantRecord)
	}
	for _, key := range []string{"fb.page_token", "page_access_token", "arn_ext_page_token", "pin_ext_page_token", "secret_ext_page_token"} {
		if isPageTokenName(key) || !isSensitiveKey(key) {
			t.Errorf("isPageTokenName(%q), isSensitiveKey(%q) = %v, %v, want false, true", key, key, isPageTokenName(key), isSensitiveKey(key))
		}
		checkRedactedMap(t, key, secret)
		checkAllRedactionPaths(t, key+"="+secret, key+"=[REDACTED]")
	}
}

// TestSensitivePairSpans_RedactedValueBeforePair checks that the value spans
// that redactBearerToken consults end where redactTextKeyValue ends a value:
// the placeholder that a quoted part joined inside a bare value leaves before
// a pair, as in DB_PASSWORD=[REDACTED]token=[REDACTED], ends its value as the
// value that it stands for does, rather than running on over the pair.
func TestSensitivePairSpans_RedactedValueBeforePair(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{`export DB_PASSWORD=pre"correct horse"token=X; echo ok`, "[{7 19 37} {37 43 44}]"},
		{"export DB_PASSWORD=[REDACTED]token=[REDACTED]; echo ok", "[{7 19 29} {29 35 45}]"},
		{"DB_PASSWORD=[REDACTED]token=[REDACTED] x", "[{0 12 22} {22 28 38}]"},
		{`DB_PASSWORD=[REDACTED]token='my Bearer ''X'`, "[{0 12 22} {22 28 43}]"},
		{"password=[REDACTED]password=[REDACTED]", "[{0 9 19} {19 28 38}]"},
		{"password=[REDACTED]x b=[REDACTED]", "[{0 9 20}]"},
	} {
		var p sensitivePairSpans
		p.build(tc.in)
		if got := fmt.Sprint(p.spans); got != tc.want {
			t.Errorf("sensitivePairSpans.build(%q) spans = %s, want %s", tc.in, got, tc.want)
		}
	}
}

// TestScrubTextPatterns_RedactedPlaceholderBeforePair checks that a value
// that starts with the redaction placeholder ends with it when a pair whose
// value the text patterns redact on its own follows it at once, as a
// sensitive pair or quoted key does, so that scrubbing again what a scrub
// wrote, as password=[REDACTED]token=[REDACTED] for password=x"a b"token=y,
// leaves it as it is. Text can hold the placeholder without any scrub having
// written it, so it is no proof that a value ended there: a pair or a quoted
// key that is not sensitive, or whose key only ends in a credential name, is
// the rest of a credential that a shell reads as one word with the
// placeholder, quoted or not, and it is redacted with it, as are the words of
// a value whose placeholder no pair follows.
func TestScrubTextPatterns_RedactedPlaceholderBeforePair(t *testing.T) {
	const secret = "R14_PLACEHOLDER_SECRET"
	for _, tt := range []struct{ in, want string }{
		{`password=pre"a b"token=` + secret, `password=[REDACTED]token=[REDACTED]`},
		{"password=[REDACTED]token=[REDACTED]", "password=[REDACTED]token=[REDACTED]"},
		{"DB_PASSWORD=[REDACTED]token=[REDACTED]", "DB_PASSWORD=[REDACTED]token=[REDACTED]"},
		{`password=[REDACTED]"token":"[REDACTED]"`, `password=[REDACTED]"token":"[REDACTED]"`},
		{`password=[REDACTED]'token'='[REDACTED]'`, `password=[REDACTED]'token'='[REDACTED]'`},
		{"password=[REDACTED]x_token=" + secret, "password=[REDACTED]x_token=[REDACTED]"},
		{"password=[REDACTED]GITHUB_TOKEN=" + secret, "password=[REDACTED]GITHUB_TOKEN=[REDACTED]"},
		// No pair or quoted key follows the placeholder.
		{"password=[REDACTED]x", "password=[REDACTED]"},
		{"password=[REDACTED]-token=" + secret, "password=[REDACTED]"},
		{"password=[REDACTED]'" + secret + "'", "password=[REDACTED]"},
		{"password=[REDACTED]x b=" + secret, "password=[REDACTED] b=" + secret},
		// A pair or a quoted key that is not sensitive, or a key that only
		// ends in a credential name, which the text patterns do not read,
		// is redacted with the value.
		{`password=pre"a ` + secret + `"label=keep`, `password=[REDACTED]`},
		{"password=[REDACTED]label=keep", "password=[REDACTED]"},
		{"password=[REDACTED]x=" + secret, "password=[REDACTED]"},
		{"password=[REDACTED]label=" + secret, "password=[REDACTED]"},
		{`password=[REDACTED]"x": "` + secret + `"`, `password=[REDACTED]`},
		{`password=[REDACTED]"x y": "` + secret + `"`, `password=[REDACTED]`},
		{`password=[REDACTED]"x":"` + secret + `"; echo ok`, `password=[REDACTED]; echo ok`},
		{`password=[REDACTED]Q13password=("` + secret + `")`, `password=[REDACTED]`},
		{"password=[REDACTED]foo.signature=" + secret, "password=[REDACTED]"},
		{"export DB_PASSWORD=[REDACTED]label=" + secret + "; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=[REDACTED]x=" + secret + "; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{`export DB_PASSWORD=[REDACTED]"label": "` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=[REDACTED]"label name": "` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`password=[REDACTED]'temp:a{b c': '` + secret + `'`, `password=[REDACTED]'temp:a{b c':'[REDACTED]'`},
		{`password=[REDACTED]'temp:a{b,c': '` + secret + `'`, `password=[REDACTED]'temp:a{b,c':'[REDACTED]'`},
		{`export DB_PASSWORD="x"'temp:a{b,c':'` + secret + `'`, `export DB_PASSWORD="[REDACTED]"'temp:a{b,c':'[REDACTED]'`},
		{`export DB_PASSWORD=[REDACTED]'temp:` + secret + `':; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="x"'temp:` + secret + `'=; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=[REDACTED]'temp:` + secret + `':}); echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=[REDACTED]'temp:` + secret + `'==1; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`failed: "password=[REDACTED]'temp:` + secret + `': "; retry`, `failed: "password=[REDACTED] "; retry`},
		{`failed: "export DB_PASSWORD='x''temp:` + secret + `': "; retry`, `failed: "export DB_PASSWORD=[REDACTED] "; retry`},
		{`failed: "export DB_PASSWORD=\"x\"'temp:` + secret + `': "; retry`, `failed: "export DB_PASSWORD=[REDACTED] "; retry`},
		{`failed: "Bearer [REDACTED]'temp:` + secret + `': "; retry`, `failed: "Bearer [REDACTED] "; retry`},
		{`failed: 'password=[REDACTED]"temp:` + secret + `": '; retry`, `failed: 'password=[REDACTED] '; retry`},
		{`failed: "password=[REDACTED]'temp:` + secret + `'="; retry`, `failed: "password=[REDACTED]; retry`},
		{`failed: "password=[REDACTED]'temp:k': '` + secret + `'"; retry`, `failed: "password=[REDACTED]'temp:k':'[REDACTED]'"; retry`},
		{`password=[REDACTED]' part1 'mid'` + secret + `'end`, `password=[REDACTED]`},
		{`Bearer [REDACTED]' part1 'mid'` + secret + `'end`, `Bearer [REDACTED]`},
		{`Bearer [REDACTED]' correct horse 'suffix && echo 'done'`, `Bearer [REDACTED] && echo 'done'`},
		{`password=[REDACTED]' correct horse 'suffix --label 'ok'`, `password=[REDACTED] --label 'ok'`},
		{`password=[REDACTED]("x y"): "` + secret + `"`, `password=[REDACTED]`},
		{`password=[REDACTED]["x y"]: "` + secret + `"`, `password=[REDACTED]`},
		{`password=[REDACTED]{"x y"}: "` + secret + `"`, `password=[REDACTED]`},
		{`got {password=[REDACTED]"x y"}: "` + secret + `"`, `got {password=[REDACTED]`},
		// The placeholder quoted, alone or with more of the credential.
		{`export DB_PASSWORD="[REDACTED]x=` + secret + `"; echo ok`, `export DB_PASSWORD="[REDACTED]"; echo ok`},
		{`export DB_PASSWORD="[REDACTED]"x=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="[REDACTED]""x": "` + secret + `"; echo ok`, `export DB_PASSWORD="[REDACTED]"; echo ok`},
		{`export DB_PASSWORD="[REDACTED]""x y": "` + secret + `"; echo ok`, `export DB_PASSWORD="[REDACTED]"; echo ok`},
		{`export DB_PASSWORD='[REDACTED]'label=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="[REDACTED]"token=` + secret + `; echo ok`, `export DB_PASSWORD="[REDACTED]"token=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="[REDACTED]"; echo ok`, `export DB_PASSWORD="[REDACTED]"; echo ok`},
		// A Bearer token that starts with the placeholder.
		{"Bearer [REDACTED]x=" + secret, "Bearer [REDACTED]"},
		{`Bearer [REDACTED]"x"="` + secret + `"`, `Bearer [REDACTED]`},
		{`Bearer [REDACTED]"x": "` + secret + `"`, `Bearer [REDACTED]`},
		{`Bearer [REDACTED]"x y": "` + secret + `"`, `Bearer [REDACTED]`},
		{`Bearer [REDACTED]'x'='` + secret + `'`, `Bearer [REDACTED]`},
		{`Bearer [REDACTED]"api_key"="` + secret + `"`, `Bearer [REDACTED]"api_key"="[REDACTED]"`},
		{"Bearer [REDACTED]token=" + secret, "Bearer [REDACTED]token=[REDACTED]"},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
		checkAllRedactionPaths(t, tt.want, tt.want)
	}
}

// TestValueStartsAt_ReadsOnlyUntilItCanTell checks that valueStartsAt tells
// whether a bare value holds a value as it would by reading the whole value,
// up to the first whitespace, ",", ";" or "&", while it reads the value only
// up to the first rune that tells: a check of each pair in a run of pairs in
// one word, as in password=[REDACTED]token=[REDACTED]..., would otherwise read
// the rest of the word for each pair, in a time that grows with the square of
// the length of the word.
func TestValueStartsAt_ReadsOnlyUntilItCanTell(t *testing.T) {
	wholeValue := func(s string) bool {
		rest := s[skipDelimSpace(s, 0):]
		if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
			rest = rest[:n]
		}
		return holdsValue(rest)
	}
	for _, in := range []string{"x", " x", "", " ", "\u00a0x", "\x00x", ",x", ";x", "&x", "}x", "})", "}) x", ".:", ".:x", ":) ,x", "a b", "\xffx"} {
		if got, want := valueStartsAt(in, 0, false, 0), wholeValue(in); got != want {
			t.Errorf("valueStartsAt(%q) = %v, want %v", in, got, want)
		}
	}
	long := "x" + strings.Repeat("y", 4<<20)
	start := time.Now()
	for range 1000 {
		if !valueStartsAt(long, 0, false, 0) {
			t.Fatal("valueStartsAt(long word) = false, want true")
		}
	}
	if d := time.Since(start); d > 100*time.Millisecond {
		t.Errorf("1000 checks of a 4 MiB word took %v, want well under 100ms", d)
	}
}

func TestRound15ShellAndTempKeyBoundaries(t *testing.T) {
	const secret = "ASTRA_SYNTHETIC_935817"
	for _, tt := range []struct{ in, want string }{
		// R15-1: Contraction-like text at the start of a single-quoted shell part.
		{`export DB_PASSWORD=pre's R15_CONTRACTION_S'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'm R15_CONTRACTION_M'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'd R15_CONTRACTION_D'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'll R15_CONTRACTION_LL'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre're R15_CONTRACTION_RE'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre've R15_CONTRACTION_VE'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pren't R15_CONTRACTION_NT'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`Bearer pre's R15_CONTRACTION_S' ok`, `Bearer [REDACTED] ok`},
		{`password=it's Bob's`, `password=[REDACTED] Bob's`},
		{`password=abc's foo bar's`, `password=[REDACTED]`},
		{`password=it's foo bar's`, `password=[REDACTED] foo bar's`},
		// R15-2: A newline immediately after an opening single quote.
		{"export DB_PASSWORD=pre'\nR15_MULTILINE_INITIAL'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre'\r\nR15_MULTILINE_INITIAL'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre\"\nR15_MULTILINE_INITIAL\"; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"Bearer pre'\nR15_MULTILINE_INITIAL' ok", "Bearer [REDACTED] ok"},
		// R15-3: Adjacent token-counter and pagination-cursor pairs after [REDACTED] and closed quotes.
		// These preserved pairs (see startsPreservedTokenPair) deliberately end a quoted credential value glued before them, so non-secret cursor and counter telemetry stays intact.
		{`password=[REDACTED]token_count=7`, `password=[REDACTED]token_count=7`},
		{`password=[REDACTED]prompt_tokens=10`, `password=[REDACTED]prompt_tokens=10`},
		{`password=[REDACTED]completion_tokens=20`, `password=[REDACTED]completion_tokens=20`},
		{`password=[REDACTED]total_tokens=30`, `password=[REDACTED]total_tokens=30`},
		{`password=[REDACTED]max_output_tokens=100`, `password=[REDACTED]max_output_tokens=100`},
		{`password=[REDACTED]thinking_tokens=50`, `password=[REDACTED]thinking_tokens=50`},
		{`password=[REDACTED]cached_content_token_count=5`, `password=[REDACTED]cached_content_token_count=5`},
		{`password=[REDACTED]page_token=cursor-1`, `password=[REDACTED]page_token=cursor-1`},
		{`password=[REDACTED]next_page_token=cursor-1`, `password=[REDACTED]next_page_token=cursor-1`},
		{`password=[REDACTED]prev_page_token=cursor-1`, `password=[REDACTED]prev_page_token=cursor-1`},
		{`password=[REDACTED]previous_page_token=cursor-1`, `password=[REDACTED]previous_page_token=cursor-1`},
		{`password=[REDACTED]next.page.token="cursor-1"`, `password=[REDACTED]next.page.token="cursor-1"`},
		{`password=[REDACTED]x.next.page.token="cursor-1"`, `password=[REDACTED]x.next.page.token="cursor-1"`},
		{`DB_PASSWORD="benign"token_count=7`, `DB_PASSWORD="[REDACTED]"token_count=7`},
		{`DB_PASSWORD="benign"prompt_tokens=10`, `DB_PASSWORD="[REDACTED]"prompt_tokens=10`},
		{`DB_PASSWORD="benign"completion_tokens=20`, `DB_PASSWORD="[REDACTED]"completion_tokens=20`},
		{`DB_PASSWORD="benign"total_tokens=30`, `DB_PASSWORD="[REDACTED]"total_tokens=30`},
		{`DB_PASSWORD="benign"max_output_tokens=100`, `DB_PASSWORD="[REDACTED]"max_output_tokens=100`},
		{`DB_PASSWORD="benign"thinking_tokens=50`, `DB_PASSWORD="[REDACTED]"thinking_tokens=50`},
		{`DB_PASSWORD="benign"cached_content_token_count=5`, `DB_PASSWORD="[REDACTED]"cached_content_token_count=5`},
		{`DB_PASSWORD="benign"page_token="cursor-1"`, `DB_PASSWORD="[REDACTED]"page_token="cursor-1"`},
		{`DB_PASSWORD="benign"next_page_token="cursor-1"`, `DB_PASSWORD="[REDACTED]"next_page_token="cursor-1"`},
		{`DB_PASSWORD="benign"prev_page_token="cursor-1"`, `DB_PASSWORD="[REDACTED]"prev_page_token="cursor-1"`},
		{`DB_PASSWORD="benign"previous_page_token="cursor-1"`, `DB_PASSWORD="[REDACTED]"previous_page_token="cursor-1"`},
		{`DB_PASSWORD="benign"next.page.token="cursor-1"`, `DB_PASSWORD="[REDACTED]"next.page.token="cursor-1"`},
		{`DB_PASSWORD="benign"x.next.page.token="cursor-1"`, `DB_PASSWORD="[REDACTED]"x.next.page.token="cursor-1"`},
		{`DB_PASSWORD="benign""token_count":7`, `DB_PASSWORD="[REDACTED]""token_count":7`},
		{`DB_PASSWORD="benign""next_page_token":"cursor-1"`, `DB_PASSWORD="[REDACTED]""next_page_token":"cursor-1"`},
		// R15-4: Later single-quoted fragments in a recognized shell credential.
		{`export DB_PASSWORD="a"pre'correct horse ` + secret + `'suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`DB_PASSWORD=pre'a b'mid'correct horse ` + secret + `'end`, `DB_PASSWORD=[REDACTED]`},
		{`export DB_PASSWORD="a"pre'correct horse ` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]`},
		{`Bearer "a"pre'correct horse ` + secret + `'suffix ok`, `Bearer [REDACTED] ok`},
		{`export DB_PASSWORD=pre'part1 ''` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"part1 ""` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre'part1 '"part2 ""` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// R15-5: Sensitive inner label with an attached bare suffix.
		{`export DB_PASSWORD=pre'correct ` + secret + ` token='suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=pre"correct ` + secret + ` token="suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=a:'s note ` + secret + ` token='suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// R15-6: Braces inside a quoted temporary-state key.
		{`failed: {'temp:cache/{user}': '` + secret + `'}`, `failed: {'temp:cache/{user}':'[REDACTED]'}`},
		{`failed: {'user:temp:a}b': '` + secret + `'}`, `failed: {'user:temp:a}b':'[REDACTED]'}`},
		{`failed: {' temp:a{b,c ': '` + secret + `'}`, `failed: {' temp:a{b,c ':'[REDACTED]'}`},
		{`failed: {'label': 'keep', 'temp:cache/{user}': '` + secret + `'}`, `failed: {'label': 'keep', 'temp:cache/{user}':'[REDACTED]'}`},
		// R15-7: Quoted sensitive keys with bare values after [REDACTED].
		{`password=[REDACTED]"token": ` + secret, `password=[REDACTED]"token":"[REDACTED]"`},
		{`Bearer [REDACTED]"api_key"= ` + secret, `Bearer [REDACTED]"api_key"=[REDACTED]`},
		// P1-1: Non-counter/non-cursor token* keys and non-numeric counter values must be redacted as joined continuations.
		{`DB_PASSWORD="a"token_hint=` + secret, `DB_PASSWORD=[REDACTED]`},
		{`password=[REDACTED]token_hint=` + secret, `password=[REDACTED]`},
		{`export DB_PASSWORD=pre'x y'token_note=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`DB_PASSWORD="a""token_hint":"` + secret + `"`, `DB_PASSWORD="[REDACTED]"`},
		{`password=[REDACTED]"my_tokens": "` + secret + `"`, `password=[REDACTED]`},
		{`export DB_PASSWORD="a"access_tokens=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=[REDACTED]"access_tokens":"` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="a"token_value=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="a"token_count=` + secret + `; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=[REDACTED]prompt_tokens="` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`password=[REDACTED]refresh_token_id=` + secret, `password=[REDACTED]`},
		{`password=[REDACTED]next_page_token_x=` + secret, `password=[REDACTED]`},
		{`export DB_PASSWORD="a"token_hint=123456; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="a"my_tokens=7; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="a"access_tokens=314159; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`password=[REDACTED]token_hint=123456`, `password=[REDACTED]`},
		// P1-2: Inter-string prose after a quoted key following [REDACTED] or a closed quote must not be preserved.
		// A second string with no lowercase prose word is the quoted key's own value, so that pair is redacted
		// on its own, as in R15-7.
		{`password=[REDACTED]"token": " for user "` + secret + `"`, `password=[REDACTED]"token":"[REDACTED]"`},
		{`password=[REDACTED]'token': ' for user '` + secret + `'`, `password=[REDACTED]'token':'[REDACTED]'`},
		{`password=[REDACTED]"token": " in config, label="` + secret + `"`, `password=[REDACTED]`},
		{`export DB_PASSWORD="x""token": " for user "` + secret + `"; echo ok`, `export DB_PASSWORD="[REDACTED]""token":"[REDACTED]"`},
		// P1-3 & P2-2: Bearer tokens with joined quotes on Pass 1 and Pass 2.
		{`echo Bearer abc"` + secret + `"`, `echo Bearer [REDACTED]`},
		{`curl -H 'x: Bearer abc'"` + secret + `"`, `curl -H 'x: Bearer [REDACTED]'`},
		{`curl -H 'x: Bearer abc'"` + secret + `"; export DB_PASSWORD=pre' ASTRA_SECOND_935817'; echo ok`, `curl -H 'x: Bearer [REDACTED]'; export DB_PASSWORD=[REDACTED]; echo ok`},
		{`curl -H "x: Bearer abc"'` + secret + `'; export DB_PASSWORD=pre" ASTRA_SECOND_935817"; echo ok`, `curl -H "x: Bearer [REDACTED]"; export DB_PASSWORD=[REDACTED]; echo ok`},
		{`curl -H 'x: Bearer '"` + secret + `"; export DB_PASSWORD=pre' ASTRA_SECOND_935817'; echo ok`, `curl -H 'x: Bearer [REDACTED]'; export DB_PASSWORD=[REDACTED]; echo ok`},
		{`curl -H "x: Bearer "'` + secret + `'; export DB_PASSWORD=pre" ASTRA_SECOND_935817"; echo ok`, `curl -H "x: Bearer [REDACTED]"; export DB_PASSWORD=[REDACTED]; echo ok`},
		{`Bearer pre's_` + secret + `'`, `Bearer [REDACTED]`},
		{`Bearer pre'll_` + secret + `' ok`, `Bearer [REDACTED] ok`},
		{`Bearer pren't_` + secret + `' ok`, `Bearer [REDACTED] ok`},
		{`Bearer pre'` + secret + `'`, `Bearer [REDACTED]`},
		{`Bearer pre"` + secret + `"`, `Bearer [REDACTED]`},
		{`Bearer pre'` + secret + `'suffix`, `Bearer [REDACTED]`},
		{`Bearer user's request`, `Bearer [REDACTED]'s request`},
		// P1-4: Contraction-led and later multiline quoted fragments in shell values.
		{"export DB_PASSWORD=pre's\n" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's \n" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\n" + secret + "'suffix; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre'll\n" + secret + "'x; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre'\n" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre\"\n" + secret + "\"; echo ok", "export DB_PASSWORD=\"[REDACTED]\"; echo ok"},
		{"export DB_PASSWORD=pre'part1 ''\n" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre'first\n" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre's\nhttps://host:8080/" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre's\nYWJjZGVmZ2hpamtsbW5vcA=='; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre's\nlabel=" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=\"a\"pre's\nlabel=" + secret + "'\"more\"; echo ok", "export DB_PASSWORD=\"[REDACTED]\"; echo ok"},
		{"Bearer \"a\"pre's\nhttps://host:8080/" + secret + "' ok", "Bearer [REDACTED] ok"},
		{"export DB_PASSWORD=pre's\nlabel=" + secret + "'tail; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\nhost:" + secret + "'tail; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"Bearer pre's\nlabel=" + secret + "'tail ok", "Bearer [REDACTED] ok"},
		{"Bearer \"a\"pre'\n" + secret + "' ok", "Bearer [REDACTED] ok"},
		{"password=it'\nKEEP", "password=[REDACTED]\nKEEP"},
		{"password=it's\nuser='bob'", "password=[REDACTED]\nuser='bob'"},
		{"password=\"a\"it's\nuser='bob'", "password=[REDACTED]\nuser='bob'"},
		// P2-1 Q11: Direct initial quoted value ending in a sensitive inner label with an attached bare suffix.
		{`export DB_PASSWORD='correct ` + secret + ` token='suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD="correct ` + secret + ` token="suffix; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// Round 15d Item 1: Bare temp: keys with "{" or balanced "{...}", and template placeholder controls.
		{`temp:a{b=` + secret, `temp:a{b=[REDACTED]`},
		{`temp:a{b: ` + secret, `temp:a{b:[REDACTED]`},
		{`user:temp:a{b=` + secret, `user:temp:a{b=[REDACTED]`},
		{`app:temp:a{b: ` + secret, `app:temp:a{b:[REDACTED]`},
		{`failed: {temp:a{b=` + secret + `}`, `failed: {temp:a{b=[REDACTED]}`},
		{`failed: {'temp:a{b: '` + secret + `'}`, `failed: {'temp:a{b:'[REDACTED]'}`},
		{`temp:cache/{user}=` + secret, `temp:cache/{user}=[REDACTED]`},
		{`map[temp:cache/{user}:` + secret + `]`, `map[temp:cache/{user}:[REDACTED]]`},
		{`template {temp:x}: hello world`, `template {temp:x}: hello world`},
		{`template {temp:cache/{user}}: hello world`, `template {temp:cache/{user}}: hello world`},
		// Round 15d Item 3: Multiline contraction-led shell splices with inner quotes, multiple lines, or extra words.
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "'; export API_KEY=pre2' ASTRA_SECOND_935817'", "export DB_PASSWORD=[REDACTED]; export API_KEY=[REDACTED]"},
		{"export DB_PASSWORD=\"a\"pre's\nfoo='bar' " + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\n" + secret + "+foo='ASTRA_SECOND_935817'", "export DB_PASSWORD=[REDACTED]"},
		{"export DB_PASSWORD=pre's\nx=1; label=" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's first\nlabel=" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\n" + secret + "\nuser='bob'", "export DB_PASSWORD=[REDACTED]"},
		{"export DB_PASSWORD=pre's\nline1\nlabel=" + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"Bearer pre's\nfoo='bar' " + secret + "' ok", "Bearer [REDACTED] ok"},
		// Round 15d Item 4: Escaped quotes before joined quotes in Bearer, outer-quoted Bearer with glued suffix, and two-Bearer boundaries.
		{`curl -H 'x: Bearer abc\'"` + secret + `"; export DB_PASSWORD=pre' ASTRA_SECOND_935817'`, `curl -H 'x: Bearer [REDACTED]'; export DB_PASSWORD=[REDACTED]`},
		{`curl -H 'x: Bearer abc\\'"` + secret + `"; export DB_PASSWORD=pre' ASTRA_SECOND_935817'`, `curl -H 'x: Bearer [REDACTED]'; export DB_PASSWORD=[REDACTED]`},
		{`curl -H 'x: Bearer abc\'"` + secret + `"; echo 'hello world'`, `curl -H 'x: Bearer [REDACTED]'; echo 'hello world'`},
		{`x='Bearer ` + secret + `'x y='Bearer ASTRA_SECOND_935817'`, `x='Bearer [REDACTED]' y='Bearer [REDACTED]'`},
		{`Bearer ` + secret + `'x1 y='Bearer ASTRA_SECOND_935817'`, `Bearer [REDACTED] y='Bearer [REDACTED]'`},
		{`echo Bearer ` + secret + `'x1; echo 'Bearer ASTRA_SECOND_935817'`, `echo Bearer [REDACTED]; echo 'Bearer [REDACTED]'`},
		{`export H='Bearer ` + secret + `'x1 ok`, `export H='Bearer [REDACTED]' ok`},
		{`echo "Bearer abc'` + secret + `'" more`, `echo "Bearer [REDACTED]" more`},
		{`{"h":"x: Bearer abc'` + secret + `'"}`, `{"h":"x: Bearer [REDACTED]"}`},
		// Round 15d Item 5: Multiline double-quoted joined shell segments with attached bare suffix.
		{"export DB_PASSWORD=pre\"\n" + secret + "\"end; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export password=suffix's'a.b\"\n" + secret + "\"pre; echo ok", "export password=[REDACTED]; echo ok"},
		// Round 15d NIT-4: Empty initial quoted value immediately followed by a joined quote.
		{`export DB_PASSWORD=""'` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{`export DB_PASSWORD=''"` + secret + `"; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// Round 15e Item 1: Closers after a single-quoted bearer token are trimmed only where nothing before them in the word opens them.
		{`x='Bearer ` + secret + `')()`, `x='Bearer [REDACTED]'`},
		{`Bearer ` + secret + `'x)()`, `Bearer [REDACTED]`},
		{`echo 'Bearer ` + secret + `'x)() ok`, `echo 'Bearer [REDACTED]' ok`},
		{`Bearer ` + secret + `'x(y))`, `Bearer [REDACTED])`},
		{`x='Bearer ` + secret + `'a(b).)`, `x='Bearer [REDACTED]'.)`},
		{`echo 'Bearer ` + secret + `'x)`, `echo 'Bearer [REDACTED]')`},
		// Round 15e Item 2: The word joined to the quote that closes a bearer token stops before an adjacent pair, reads its joined quoted parts as a shell does, and keeps the closing quote.
		{`x='Bearer abc'DB_PASSWORD='` + secret + ` WITH SPACE'`, `x='Bearer [REDACTED]'DB_PASSWORD='[REDACTED]'`},
		{`curl -H 'x-hdr: Bearer abc'mid' ` + secret + `'; echo ok`, `curl -H 'x-hdr: Bearer [REDACTED]'; echo ok`},
		{`export H='Bearer abc's' ` + secret + `'; echo ok`, `export H='Bearer [REDACTED]'; echo ok`},
		{`curl -H 'x-hdr: Bearer abc'mid" ` + secret + `"; export DB_PASSWORD=pre" ASTRA_SECOND_935817"`, `curl -H 'x-hdr: Bearer [REDACTED]'; export DB_PASSWORD=[REDACTED]`},
		{`x='Bearer abc'mid' ` + secret + `'.)`, `x='Bearer [REDACTED]'.)`},
		{`echo 'Bearer abc'x' 9'; echo 'Bearer ` + secret + `'`, `echo 'Bearer [REDACTED]'; echo 'Bearer [REDACTED]'`},
		{`x='Bearer abc'prompt_tokens=5 ok`, `x='Bearer [REDACTED]'prompt_tokens=5 ok`},
		{`x='Bearer abc'next.page.token="cursor-1"`, `x='Bearer [REDACTED]'next.page.token="cursor-1"`},
		{`x='Bearer abc'"token":"` + secret + `"`, `x='Bearer [REDACTED]'"token":"[REDACTED]"`},
		// Round 15e Item 2: The joined word also stops before a later Bearer keyword, in unbalanced lines and exotic glue too.
		{`ARGS=(x='Bearer ZQAAA7'tail;set a Bearer ZQBBB8';'Bearer ZQCCC9'x1) ok`, `ARGS=(x='Bearer [REDACTED]';set a Bearer [REDACTED]Bearer [REDACTED]`},
		{`ARGS=(curl -H 'x: Bearer ZQAAA7''\'\;echo 'Bearer ZQBBB8',) 'q' w`, `ARGS=(curl -H 'x: Bearer [REDACTED]'Bearer [REDACTED]',) 'q' w`},
		{`export H='x: Bearer ZQAAA7''-x'-x -H run: "Bearer ZQBBB8"" -H export H="Bearer ZQCCC9"_`, `export H='x: Bearer [REDACTED]' -H run: "Bearer [REDACTED]"Bearer [REDACTED]`},
		{`Bearer a';'Bearer ` + secret + `'x1`, `Bearer [REDACTED]Bearer [REDACTED]`},
		{`x='Bearer a'Bearer ` + secret + `' ok`, `x='Bearer [REDACTED]'Bearer [REDACTED]' ok`},
		{`'Bearer abc'QQSA'Bearer QQSB'`, `'Bearer [REDACTED]'`},
		// Round 15e Item 3: Quote pairing on a later line of a contraction-led splice skips quotes inside double-quoted text and reads \' as a shell does.
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "'\"1'2\"; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "'\"ASTRA_SECOND_935817'3\"; export API_KEY=pre2' ASTRA_THIRD_935817'", "export DB_PASSWORD=[REDACTED]; export API_KEY=[REDACTED]"},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "\\'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre' " + secret + "\\'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"password=it's\nuser='bob' \"it's\"", "password=[REDACTED]\nuser='bob' \"it's\""},
		{"password=it's\nuser='bob' \"a'b\" ok", "password=[REDACTED]\nuser='bob' \"a'b\" ok"},
		{"password=it's\nuser='bob' don't", "password=[REDACTED]\nuser='bob' don't"},
		// Round 15e Item 4: A quoted value glued to the delimiter of its key, or a second string with no lowercase word in it, is a credential rather than the text between two strings.
		{`got "token": " for user "` + secret + `"`, `got "token":"[REDACTED]"`},
		{`got 'token': ' for user '` + secret + `'`, `got 'token':'[REDACTED]'`},
		{`got "token": " for user "Bob"`, `got "token":"[REDACTED]"`},
		{`msg "token": " a b "` + secret + `"`, `msg "token":"[REDACTED]"`},
		{`msg 'token': ' a b '` + secret + `'`, `msg 'token':'[REDACTED]'`},
		{`password="x"` + secret + `""user":"bob"`, `password="[REDACTED]"`},
		{`msg "token":" a b "` + secret + `""next_page_token":"cursor-1"`, `msg "token":"[REDACTED]""next_page_token":"cursor-1"`},
		{`msg "token":" a b "` + secret + `""token_count":7`, `msg "token":"[REDACTED]""token_count":7`},
		{`msg "token": " for user "bob""token":" a b "` + secret + `"`, `msg "token":"[REDACTED]""token":"[REDACTED]"`},
		{`got "token": " for user "bob"`, `got "token":"[REDACTED]"`},
		{`failed: "password": " for user "bob"`, `failed: "password":"[REDACTED]"`},
		{`got 'token': ' for user 'bob'`, `got 'token':'[REDACTED]'`},
		// Round 15e: A brace key that is not closed before its delimiter is a key even before prose, failing closed.
		{`say {temp:a{b}: hello`, `say {temp:a{b}:[REDACTED]`},
		// Round 15f Item 2: On the rest of a contraction-led line, and where a quoted part of a value closes, a quote between two letters is an
		// apostrophe only if it ends a prose word made of letters and no quote follows its ending, so the parts are read as a shell reads them.
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "_it's; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "_it's' ASTRA_SECOND_935817'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{"export DB_PASSWORD=pre's\nfoo='bar' a's' " + secret + "'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		// Round 15f Item 3: A single-quoted part that a shell closes at the ' of a \' leaves the quotes after it paired as the shell pairs them.
		{`export DB_PASSWORD=pre' ` + secret + `\'; export API_KEY=pre2' ASTRA_SECOND_935817'`, `export DB_PASSWORD=[REDACTED]; export API_KEY=[REDACTED]`},
		{`curl -H 'x: Bearer abc'mid' ` + secret + `\'; curl -H 'y: Bearer def'mid2' ASTRA_SECOND_935817'`, `curl -H 'x: Bearer [REDACTED]'; curl -H 'y: Bearer [REDACTED]'`},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "\\'; export API_KEY=pre2' ASTRA_SECOND_935817'", "export DB_PASSWORD=[REDACTED]; export API_KEY=[REDACTED]"},
		{`export DB_PASSWORD=pre' ` + secret + `\'; echo 'benign_ok'`, `export DB_PASSWORD=[REDACTED]; echo 'benign_ok'`},
		{`export DB_PASSWORD=pre' ` + secret + `\'; export API_KEY=pre2' ASTRA_SECOND_935817' x'; y'`, `export DB_PASSWORD=[REDACTED]; export API_KEY=[REDACTED] x'; y'`},
		{`export DB_PASSWORD=pre'\''s ` + secret + `'; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// Round 15f Item 3 controls: an escaped quote that the rest of the line reads as one of a Python string stays in its string.
		{`token=b'` + secret + `\'b'; echo 'benign_ok'`, `token=[REDACTED]; echo 'benign_ok'`},
		{`token=b'it\'s; ` + secret + `'`, `token=[REDACTED]`},
		// Round 15f Item 1: Glued JSON objects and arrays keep their structure after a bearer token or a quoted value, and the backslash of an
		// escaped quote that closes the string of a token stays with that quote, on the first scrub and on the next.
		{`got {"auth": "Bearer ` + secret + `"}{"user": "bob"}`, `got {"auth": "Bearer [REDACTED]"}{"user": "bob"}`},
		{`got {"auth": "Bearer a'` + secret + `"}{"user": "bob"}`, `got {"auth": "Bearer [REDACTED]"}{"user": "bob"}`},
		{`got ["Bearer ` + secret + `"]["x"]`, `got ["Bearer [REDACTED]"]["x"]`},
		{`password="` + secret + `"}{"user": "bob"}`, `password="[REDACTED]"}{"user": "bob"}`},
		{`x {\"auth\": \"Bearer abc'` + secret + `\"}`, `x {\"auth\": \"Bearer [REDACTED]\"}`},
		{`x {\"auth\": \"Bearer ` + secret + `\"}`, `x {\"auth\": \"Bearer [REDACTED]\"}`},
		{`got [{"auth": "Bearer ` + secret + `"}][{"user": "bob"}]`, `got [{"auth": "Bearer [REDACTED]"}][{"user": "bob"}]`},
		{`got {'auth': 'Bearer ` + secret + `'}{'user': 'bob'}`, `got {'auth': 'Bearer [REDACTED]'}{'user': 'bob'}`},
		// Round 15f Item 1: Brackets glued to a word that does not open a JSON object or array with a quoted key or element are the rest of
		// the value, as a shell reads them, and are redacted with it.
		{`x='Bearer abc'}{` + secret, `x='Bearer [REDACTED]'`},
		{`x='Bearer abc'][` + secret, `x='Bearer [REDACTED]'`},
		{`x="Bearer abc"}{` + secret, `x="Bearer [REDACTED]"`},
		{`x='Bearer abc'}{'` + secret + `'`, `x='Bearer [REDACTED]'`},
		{`password='abc'}{` + secret, `password=[REDACTED]`},
		{`password="abc"}{` + secret, `password=[REDACTED]`},
		{`got {"auth": "Bearer abc"}{` + secret, `got {"auth": "Bearer [REDACTED]"`},
		// A quoted word after the brackets that no ":" (object) or "," or "]" (array) follows, or that does not close on its line, is not a
		// key or an element either.
		{`x='Bearer abc'}{'` + secret + `' ok`, `x='Bearer [REDACTED]' ok`},
		{`x='Bearer abc']['` + secret + `' ok`, `x='Bearer [REDACTED]' ok`},
		{"x='Bearer abc'}{'" + secret + "\n': 1}", "x='Bearer [REDACTED]: 1}"},
		// Round 15g: A single quote that closes a shell part after a digit or a symbol joins the part to the word even where a contraction
		// ending follows it, as a shell joins them; only the apostrophe of a prose word does not.
		{`export DB_PASSWORD="a"pre' ` + secret + `'s; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		{"export DB_PASSWORD=pre's\nfoo='bar' " + secret + "'s; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok"},
		{`export DB_PASSWORD=pre' ` + secret + `'s; echo ok`, `export DB_PASSWORD=[REDACTED]; echo ok`},
		// Round 15g: A quote after an escaped backslash that punctuation follows ends the string, so the next pair is matched on its own.
		{`got {"auth": "Bearer a'` + secret + `\\", "api_key": "ASTRA_SECOND_935817"}`, `got {"auth": "Bearer [REDACTED]", "api_key":"[REDACTED]"}`},
		// Round 15g: A word joined to the quote that closes the string of a token that holds a quote is redacted with it, as after any token.
		{`got {"auth": "Bearer a'b"}` + secret, `got {"auth": "Bearer [REDACTED]"`},
		// Round 15g: Neither an object whose first string a "," follows nor an object glued to a ")" opens a JSON object after brackets.
		{`got {"auth": "Bearer ` + secret + `"}{"ASTRA_SECOND_935817", 1]`, `got {"auth": "Bearer [REDACTED]", 1]`},
		{`("Bearer ` + secret + `"){"ASTRA_SECOND_935817": 1}`, `("Bearer [REDACTED]": 1}`},
		// Round 15g: Brackets and an escaped line break before a word that opens no object are read on with the value, as before.
		{`msg="{\"Authorization\": \"Bearer abc\"}\nBearer ` + secret + `"`, `msg="{\"Authorization\":[REDACTED]`},
		// Round 15h: A key-value pair that the text patterns redact, right after the quote that closes a bearer token or after brackets
		// joined to it, is not read as the rest of the token: it is left in place and its value is redacted on its own, after a plain
		// token and after one that holds a quote, with a bare or a quoted key and a spaced value.
		{`x='Bearer abc'}{password: ` + secret + `}`, `x='Bearer [REDACTED]'}{password:[REDACTED]}`},
		{`x='Bearer abc'}{"password": ` + secret + `}`, `x='Bearer [REDACTED]'}{"password":"[REDACTED]"}`},
		{`x='Bearer abc']{password: ` + secret + `}`, `x='Bearer [REDACTED]']{password:[REDACTED]}`},
		{`x='Bearer abc']{"password": ` + secret + `}`, `x='Bearer [REDACTED]']{"password":"[REDACTED]"}`},
		{`x='Bearer abc')password: ` + secret, `x='Bearer [REDACTED]')password:[REDACTED]`},
		{`x='Bearer abc')"password": ` + secret, `x='Bearer [REDACTED]')"password":"[REDACTED]"`},
		{`x='Bearer abc'}password: ` + secret, `x='Bearer [REDACTED]'}password:[REDACTED]`},
		{`x='Bearer abc'}"password": ` + secret, `x='Bearer [REDACTED]'}"password":"[REDACTED]"`},
		{`x='Bearer abc'][password: ` + secret + `]`, `x='Bearer [REDACTED]'][password:[REDACTED]]`},
		{`x='Bearer abc']["password": ` + secret + `]`, `x='Bearer [REDACTED]']["password":"[REDACTED]"`},
		{`got {"auth": "Bearer a'b"}{password: ` + secret + `}`, `got {"auth": "Bearer [REDACTED]"}{password:[REDACTED]}`},
		{`got {"auth": "Bearer a'b"}{"password": ` + secret + `}`, `got {"auth": "Bearer [REDACTED]"}{"password":"[REDACTED]"}`},
		{`got {"auth": "Bearer a'b"]{password: ` + secret + `}`, `got {"auth": "Bearer [REDACTED]"]{password:[REDACTED]}`},
		{`got {"auth": "Bearer a'b"]{"password": ` + secret + `}`, `got {"auth": "Bearer [REDACTED]"]{"password":"[REDACTED]"}`},
		{`got {"auth": "Bearer a'b")password: ` + secret, `got {"auth": "Bearer [REDACTED]")password:[REDACTED]`},
		{`got {"auth": "Bearer a'b")"password": ` + secret, `got {"auth": "Bearer [REDACTED]")"password":"[REDACTED]"`},
		{`got {"auth": "Bearer a'b"}password: ` + secret, `got {"auth": "Bearer [REDACTED]"}password:[REDACTED]`},
		{`got {"auth": "Bearer a'b"}"password": ` + secret, `got {"auth": "Bearer [REDACTED]"}"password":"[REDACTED]"`},
		{`got {"auth": "Bearer a'b"][password: ` + secret + `]`, `got {"auth": "Bearer [REDACTED]"][password:[REDACTED]]`},
		{`got {"auth": "Bearer a'b"]["password": ` + secret + `]`, `got {"auth": "Bearer [REDACTED]"]["password":"[REDACTED]"`},
		{`x='Bearer abc'][\"password\": \"` + secret + `\"}`, `x='Bearer [REDACTED]'][\"password\":\"[REDACTED]\"}`},
		// Round 15h: A compound key with a spaced value, which the text patterns read with their key-value finder, is left in place too.
		{`x='Bearer abc'DB_PASSWORD: ` + secret, `x='Bearer [REDACTED]'DB_PASSWORD:[REDACTED]`},
		{`x='Bearer abc'}{GITHUB_TOKEN: ` + secret, `x='Bearer [REDACTED]'}{GITHUB_TOKEN:[REDACTED]`},
		{`got {"auth": "Bearer a'b")DB_PASSWORD = ` + secret, `got {"auth": "Bearer [REDACTED]")DB_PASSWORD=[REDACTED]`},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
		checkAllRedactionPaths(t, tt.want, tt.want)
	}
	// Round 15e Item 2: In a double-quoted JSON value, the word joined to a bearer token ends at the quote that closes the value, which stays in place.
	jsonIn := `{"auth": "Bearer abc'` + secret + `"}`
	raw, _ := sanitizeRawText(jsonIn)
	smart, _, smartErr := smartTruncateText(t, jsonIn, 1000)
	var smartText string
	if smartErr != nil || json.Unmarshal(smart, &smartText) != nil {
		t.Errorf("SmartTruncate(%q) = (%s, %v), want a JSON string", jsonIn, smart, smartErr)
	}
	for _, p := range []struct{ name, got, want string }{
		{"scrubTextPatterns", scrubTextPatterns(jsonIn), `{"auth": "Bearer [REDACTED]"}`},
		{"sanitizeRawText", raw, `{"auth": "Bearer [REDACTED]"}`},
		{"SmartTruncate", smartText, `{"auth": "Bearer [REDACTED]"}`},
		{"redactErrorMessage", redactErrorMessage(jsonIn), `{"auth":"Bearer [REDACTED]"}`},
	} {
		if p.got != p.want {
			t.Errorf("%s(%q) = %q, want %q", p.name, jsonIn, p.got, p.want)
		}
		if again := scrubTextPatterns(p.want); again != p.want {
			t.Errorf("scrubbing %q again gave %q, want it unchanged", p.want, again)
		}
	}
	// Round 15f Item 1: In a string nested in another, as JSON text quoted into a JSON string is, the word joined to a bearer token ends at the
	// escaped quote that closes the nested string, which stays in place, and so does the backslash of an escaped quote that closes a string of
	// escaped quotes. The other paths read these texts as JSON that is not valid and write a placeholder, so the text patterns are checked on
	// their own, with a leak check of every path.
	for _, tc := range []struct{ in, want string }{
		{`msg="{\"auth\": \"Bearer abc'` + secret + `\"}"`, `msg="{\"auth\": \"Bearer [REDACTED]\"}"`},
		{`{"body": "{\"auth\": \"Bearer abc'` + secret + `\"}"}`, `{"body": "{\"auth\": \"Bearer [REDACTED]\"}"}`},
		{`{\"auth\": \"Bearer abc'` + secret + `\"}`, `{\"auth\": \"Bearer [REDACTED]\"}`},
	} {
		if got := scrubTextPatterns(tc.in); got != tc.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if again := scrubTextPatterns(tc.want); again != tc.want {
			t.Errorf("scrubbing %q again gave %q, want it unchanged", tc.want, again)
		}
		checkNoLeak(t, tc.in, secret)
	}
	// Round 15g: A quote is read after the backslashes before it: after three it is a byte of a nested string, and after two, where
	// whitespace follows it, it may be escaped, so the value runs on to the quote that ends the string. A word joined to the closing quote
	// of a token that holds a quote is redacted with it. A nested string that closing brackets end, before an escaped line break and an
	// object or before a glued object, ends the value, so the object after it is read on its own, and a bearer keyword after an escaped
	// line break is matched on its own. The other paths read these texts as JSON that is not valid, or as text that holds an encoded
	// credential, and write a placeholder, so the text patterns are checked on their own, with a leak check of every path.
	for _, tc := range []struct{ in, want string }{
		{`msg="{\"auth\": \"Bearer a'` + secret + `\\\", ASTRA_SECOND_935817\"}"`, `msg="{\"auth\": \"Bearer [REDACTED]\"}"`},
		{`{\"auth\": \"Bearer a'` + secret + `\\\" ASTRA_SECOND_935817\"}`, `{\"auth\": \"Bearer [REDACTED]\"}`},
		{`"Bearer a'b\\" ` + secret + `"`, `"Bearer [REDACTED]"`},
		{`{"auth": "Bearer a'` + secret + `"}ASTRA_SECOND_935817`, `{"auth": "Bearer [REDACTED]"`},
		{`{"auth": "Bearer ` + secret + `"}{"ASTRA_SECOND_935817", 1]`, `{"auth": "Bearer [REDACTED]", 1]`},
		{`msg="{\"Authorization\": \"Bearer abc\"}\n{\"access_token\": \"` + secret + `\", \"expires_in\": 3599}"`, `msg="{\"Authorization\":\"[REDACTED]\"}\n{\"access_token\":\"[REDACTED]\", \"expires_in\": 3599}"`},
		{`msg="{\"Authorization\": \"Bearer abc\"}\n\n{\"access_token\": \"` + secret + `\"}"`, `msg="{\"Authorization\":\"[REDACTED]\"}\n\n{\"access_token\":\"[REDACTED]\"}"`},
		{`msg='{\'Authorization\': \'Bearer abc\'}\n{\'access_token\': \'` + secret + `\'}'`, `msg='{\'Authorization\':\'[REDACTED]\'}\n{\'access_token\':\'[REDACTED]\'}'`},
		{`{"log": "{\"Authorization\": \"Bearer abc\"}{\"password\": \"` + secret + `\"}"}`, `{"log": "{\"Authorization\":\"[REDACTED]\"}{\"password\":\"[REDACTED]\"}"}`},
		{`{"headers": {"Authorization": "Bearer abc"}}\n{"access_token": "` + secret + `"}`, `{"headers": {"Authorization":"[REDACTED]"}}\n{"access_token":"[REDACTED]"}`},
		{`got {"auth": "Bearer a'b"}\nBearer ` + secret, `got {"auth": "Bearer [REDACTED]"Bearer [REDACTED]`},
	} {
		if got := scrubTextPatterns(tc.in); got != tc.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if again := scrubTextPatterns(tc.want); again != tc.want {
			t.Errorf("scrubbing %q again gave %q, want it unchanged", tc.want, again)
		}
		checkNoLeak(t, tc.in, secret)
		checkNoLeak(t, tc.in, "ASTRA_SECOND_935817")
	}
	// Round 15g: A \' part that the line ends is not read with a quoted part on the next line; the part of the line that
	// findQuotedOrShellSegment reads after a quoted value never crosses a line break, which would read a slice that ends before it
	// starts. The first value is redacted on every path, and the other paths fail closed on the second line; the text patterns keep the
	// second line's quoted value there, as they have since before Round 15.
	{
		in := "export DB_PASSWORD=pre' " + secret + "\\'\nexport API_KEY=pre2' ASTRA_SECOND_935817'"
		checkNoLeak(t, in, secret)
		raw, _ := sanitizeRawText(in)
		smart, _, _ := smartTruncateText(t, in, 1000)
		for _, got := range []string{raw, redactErrorMessage(in), string(smart)} {
			if strings.Contains(got, "ASTRA_SECOND_935817") {
				t.Errorf("redacting %q gave %q, which still holds the second secret", in, got)
			}
		}
	}
	for _, tmpl := range []string{`{temp:x}: hello world`, `{temp:cache/{user}}: hello world`} {
		if got := scrubTextPatterns(tmpl); got != tmpl {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tmpl, got, tmpl)
		}
	}

	if _, err := exec.LookPath("sh"); err == nil {
		for _, tc := range []struct {
			script string
			want   string
		}{
			{`export DB_PASSWORD=pre's R15_CONTRACTION_S'; printf '%s' "$DB_PASSWORD"`, "pres R15_CONTRACTION_S"},
			{"export DB_PASSWORD=pre'\nR15_MULTILINE_INITIAL'; printf '%s' \"$DB_PASSWORD\"", "pre\nR15_MULTILINE_INITIAL"},
			{`export DB_PASSWORD="a"pre'correct horse ASTRA_SYNTHETIC_935817'suffix; printf '%s' "$DB_PASSWORD"`, "aprecorrect horse ASTRA_SYNTHETIC_935817suffix"},
			{`export DB_PASSWORD=pre'correct ASTRA_SYNTHETIC_935817 token='suffix; printf '%s' "$DB_PASSWORD"`, "precorrect ASTRA_SYNTHETIC_935817 token=suffix"},
			// Round 15e: the shell joins the quoted parts after a bearer token into its word, pairs quotes on a later line outside double-quoted text, and keeps \ inside single quotes.
			{`H='x-hdr: Bearer abc'mid' ASTRA_SYNTHETIC_935817'; printf '%s' "$H"`, "x-hdr: Bearer abcmid ASTRA_SYNTHETIC_935817"},
			{"export DB_PASSWORD=pre's\nfoo='bar' ASTRA_SYNTHETIC_935817'\"1'2\"; printf '%s' \"$DB_PASSWORD\"", "pres\nfoo=bar ASTRA_SYNTHETIC_9358171'2"},
			{`export DB_PASSWORD=pre' ASTRA_SYNTHETIC_935817\'; printf '%s' "$DB_PASSWORD"`, `pre ASTRA_SYNTHETIC_935817\`},
			// Round 15f: a quote between the letters of an identifier closes a part, \' ends a part and leaves the next ones paired, and '\''
			// joins a part that holds a backslash to the next one.
			{"export DB_PASSWORD=pre's\nfoo='bar' ASTRA_SYNTHETIC_935817_it's; printf '%s' \"$DB_PASSWORD\"", "pres\nfoo=bar ASTRA_SYNTHETIC_935817_its"},
			{`export DB_PASSWORD=pre' ASTRA_SYNTHETIC_935817\'; export API_KEY=pre2' ASTRA_SECOND_935817'; printf '%s|%s' "$DB_PASSWORD" "$API_KEY"`, `pre ASTRA_SYNTHETIC_935817\|pre2 ASTRA_SECOND_935817`},
			{`export DB_PASSWORD=pre'\''s ASTRA_SYNTHETIC_935817'; printf '%s' "$DB_PASSWORD"`, `pre\s ASTRA_SYNTHETIC_935817`},
		} {
			out, err := exec.Command("sh", "-c", tc.script).Output()
			if err != nil || string(out) != tc.want {
				t.Errorf("sh -c %q = (%q, %v), want %q", tc.script, out, err, tc.want)
			}
		}
	}
}

// TestScrubTextPatterns_QuotedKeyChainsAfterPrefixes checks that a word of one
// to four quoted token pairs, each of whose values holds words before a second
// string with a credential in it, is redacted pair by pair (or, for some
// prefixes, from its first value on as one value) after each kind of prefix,
// in the glued and spaced forms with double and single quotes, so that no
// value of the word survives however many pairs it holds.
func TestScrubTextPatterns_QuotedKeyChainsAfterPrefixes(t *testing.T) {
	secrets := []string{"QQSA", "QQSB", "QQSC", "QQSD"}
	forms := []struct {
		pair func(secret string) string
		want string
	}{
		{func(s string) string { return `"token":" a b "` + s + `"` }, `"token":"[REDACTED]"`},
		{func(s string) string { return `"token": " a b "` + s + `"` }, `"token":"[REDACTED]"`},
		{func(s string) string { return `'token':' a b '` + s + `'` }, `'token':'[REDACTED]'`},
		{func(s string) string { return `'token': ' a b '` + s + `'` }, `'token':'[REDACTED]'`},
	}
	// A word starts each text that would otherwise start with a double quote,
	// which SmartTruncate reads as a JSON text that does not parse; those
	// texts are checked on the text patterns alone below.
	prefixes := []struct{ in, want string }{
		{"msg ", "msg "},
		{`password="x"`, `password="[REDACTED]"`},
		{`msg "password":"x"`, `msg "password":"[REDACTED]"`},
		{`password='x'`, `password='[REDACTED]'`},
		{`password=[REDACTED]`, `password=[REDACTED]`},
	}
	// The baseline pass (see baselineCuts) reads the rest of a word of
	// double-quoted pairs after password=[REDACTED] as the value of
	// password=, so the text paths redact the word after its first key as
	// one value.
	merged := func(prefix, pairWant string) bool {
		return prefix == `password=[REDACTED]` && strings.HasPrefix(pairWant, `"`)
	}
	for _, f := range forms {
		for _, p := range prefixes {
			for n := 1; n <= len(secrets); n++ {
				in, want := p.in, p.want
				for i, s := range secrets[:n] {
					in += f.pair(s)
					if i == 0 || !merged(p.in, f.want) {
						want += f.want
					}
				}
				checkAllRedactionPaths(t, in, want)
				checkAllRedactionPaths(t, want, want)
			}
		}
	}
	// The same words with a value of literal x in place of each credential,
	// glued to the delimiter of their keys, are redacted pair by pair too. (A
	// spaced value whose second string is a lowercase word reads as the words
	// between two strings, as " for user "bob" does.)
	for _, f := range forms {
		if strings.Contains(f.pair(""), ": ") {
			continue
		}
		for _, p := range prefixes {
			for n := 1; n <= len(secrets); n++ {
				in, want := p.in, p.want
				for i := range n {
					in += f.pair("x")
					if i == 0 || !merged(p.in, f.want) {
						want += f.want
					}
				}
				checkAllRedactionPaths(t, in, want)
			}
		}
	}
	for _, f := range forms {
		for _, p := range []struct{ in, want string }{{"", ""}, {`"password":"x"`, `"password":"[REDACTED]"`}} {
			for n := 1; n <= len(secrets); n++ {
				in, want := p.in, p.want
				for _, s := range secrets[:n] {
					in += f.pair(s)
					want += f.want
				}
				if got := scrubTextPatterns(in); got != want {
					t.Errorf("scrubTextPatterns(%q) = %q, want %q", in, got, want)
				}
				if again := scrubTextPatterns(want); again != want {
					t.Errorf("scrubbing %q again gave %q, want it unchanged", want, again)
				}
				for _, s := range append(secrets[:n:n], " a b ") {
					checkNoLeak(t, in, s)
				}
			}
		}
	}
}

// TestRound16ReviewRegressions checks the Round 16 review reproductions on
// every text path, with a second scrub that leaves each output unchanged and
// a leak check of every marker:
//   - a closed quoted part that holds a backslash-newline is one part of the
//     value;
//   - the quote that closes a quoted part before a contraction ending is a
//     close in a shell assignment, a compound shell word, or whenever the
//     part's opening quote is neither a contraction nor a plural possessive
//     followed by prose, whatever the case of the word before it;
//   - an unclosed contraction-shaped quote that a line break follows, that is
//     not after a contraction stem, or that stands in a shell assignment, an
//     unclosed trailing quote before a line break in a shell assignment, and
//     an unclosed part after a closed one, run on to the end of the text;
//   - a token counter with a quoted decimal value after a credential is kept,
//     while a quoted value that is not a count or that more text is glued to
//     is redacted;
//   - an Authorization or API-key header whose quote (including an escaped
//     quote or an enclosing outer quote) closes on a later line, including
//     over interior lines containing colons or indented obs-fold lines, or is
//     not closed, is redacted over those lines up to the next unindented
//     header, while an enclosing outer quote that closes on the header's line
//     preserves subsequent lines.
func TestRound16ReviewRegressions(t *testing.T) {
	for _, tt := range []struct {
		in, want string
		allPaths bool
		secrets  []string
	}{
		{"export DB_PASSWORD=\"a\"pre'\\\nSYNTHETIC_REVIEW_984731'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"export DB_PASSWORD=\"a\"pre\"\\\nSYNTHETIC_REVIEW_984731\"; echo ok", "export DB_PASSWORD=\"[REDACTED]\"; echo ok", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"export DB_PASSWORD=\"a\"pre'x \\\nREVIEW_MATRIX_SECRET_5317'suffix; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"REVIEW_MATRIX_SECRET_5317"}},
		{"export DB_PASSWORD=\"a\"pre'x \\\r\nREVIEW_MATRIX_SECRET_5317'suffix; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"REVIEW_MATRIX_SECRET_5317"}},
		{"export DB_PASSWORD=\"a\"pre\"x \\\nREVIEW_MATRIX_SECRET_5317\"suffix; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"REVIEW_MATRIX_SECRET_5317"}},
		{"curl -H 'x: Bearer abc'mid'x \\\nREVIEW_MATRIX_SECRET_5317'; echo ok", "curl -H 'x: Bearer [REDACTED]; echo ok", true, []string{"REVIEW_MATRIX_SECRET_5317"}},
		{"export DB_PASSWORD=pre' violetfern's; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"violetfern"}},
		{"export DB_PASSWORD=\"a\"pre' violetfern's; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre's\nfoo='bar' violetfern's; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre'D\nSYNTHETIC_REVIEW_984731\ntoken=TAIL_984731; echo ok", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731", "TAIL_984731"}},
		{"export DB_PASSWORD=pre's\nSYNTHETIC_REVIEW_984731\ntoken=TAIL_984731; echo ok", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731", "TAIL_984731"}},
		{"export DB_PASSWORD=pre'll\nSYNTHETIC_REVIEW_984731\ntoken=TAIL_984731; echo ok", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731", "TAIL_984731"}},
		{"export DB_PASSWORD=pre'd REVIEW_SYNTHETIC_7319", "export DB_PASSWORD=[REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"export DB_PASSWORD=pre's\nREVIEW_SYNTHETIC_7319", "export DB_PASSWORD=[REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"export DB_PASSWORD=\"a\"pre'correct\nSYNTHETIC_REVIEW_984731", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"export DB_PASSWORD=\"a\"pre's\nSYNTHETIC_REVIEW_984731", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"export DB_PASSWORD=\"a\"pre'x\nREVIEW_SYNTHETIC_7319\ntoken=SECOND_SECRET_7319", "export DB_PASSWORD=[REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319", "SECOND_SECRET_7319"}},
		{"Authorization: Bearer \"a\"pre'x\nREVIEW_SYNTHETIC_7319", "Authorization: [REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"export DB_PASSWORD=pre'D\r\nSYNTHETIC_REVIEW_984731\r\ntoken=TAIL_984731; echo ok", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731", "TAIL_984731"}},
		{"password=[REDACTED]prompt_tokens=\"10\"", "password=[REDACTED]prompt_tokens=\"10\"", true, []string{}},
		{"password=[REDACTED]prompt_tokens='10'", "password=[REDACTED]prompt_tokens='10'", true, []string{}},
		{"DB_PASSWORD=\"a\"prompt_tokens=\"10\"", "DB_PASSWORD=\"[REDACTED]\"prompt_tokens=\"10\"", true, []string{}},
		{"password=[REDACTED]\"prompt_token_count\":\"10\"", "password=[REDACTED]\"prompt_token_count\":\"10\"", true, []string{}},
		{"DB_PASSWORD=\"a\"\"prompt_token_count\":\"10\"", "DB_PASSWORD=\"[REDACTED]\"\"prompt_token_count\":\"10\"", true, []string{}},
		{"password=\"abc\nprompt_tokens=\"SYNTHETIC_REVIEW_984731\"", "password=\"[REDACTED]\"", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"password=\"abc\nprompt_tokens=\"10\"SYNTHETIC_REVIEW_984731", "password=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"password=[REDACTED]prompt_tokens=\"10\"SYNTHETIC_REVIEW_984731", "password=[REDACTED]", true, []string{"SYNTHETIC_REVIEW_984731"}},
		{"Authorization: abc'x\nREVIEW_SYNTHETIC_7319'", "Authorization: [REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"Authorization: Bearer abc'x\nREVIEW_SYNTHETIC_7319'", "Authorization: [REDACTED]", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"x-api-key: \"abc\nREVIEW_SYNTHETIC_7319\nContent-Type: text/plain", "x-api-key: [REDACTED]\nContent-Type: text/plain", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"Proxy-Authorization: Basic \"abc\nREVIEW_SYNTHETIC_7319\"\nAccept: */*", "Proxy-Authorization: [REDACTED]\nAccept: */*", true, []string{"REVIEW_SYNTHETIC_7319"}},
		{"api-key: 'abc\nREVIEW_SYNTHETIC_7319' tail\nHost: example.com", "api-key: [REDACTED]\nHost: example.com", true, []string{"REVIEW_SYNTHETIC_7319"}},
		// A quoted bearer token that holds a backslash-newline, and a header whose
		// apostrophe a quote on the next header line would close, which ends at its line.
		{"run Bearer 'abc\\\nSYNTHETIC_BEARER_7319' tail", "run Bearer [REDACTED] tail", true, []string{"SYNTHETIC_BEARER_7319"}},
		{"run Bearer \"abc\\\nSYNTHETIC_BEARER_7319\" tail", "run Bearer [REDACTED] tail", true, []string{"SYNTHETIC_BEARER_7319"}},
		{"Authorization: Bob's key\nX-Note: 'hi'", "Authorization: [REDACTED]\nX-Note: 'hi'", true, []string{}},
		{"Authorization: Bearer abc's\nX-Request-Id: 'r-1'", "Authorization: [REDACTED]\nX-Request-Id: 'r-1'", true, []string{}},
		// Closing-contraction shell credentials without ;&|, including compound words with quoted spaces and quoted variable names.
		{"export DB_PASSWORD=pre' violetfern's", "export DB_PASSWORD=[REDACTED]", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre' violetfern's\necho ok", "export DB_PASSWORD=[REDACTED]\necho ok", true, []string{"violetfern"}},
		{"DB_PASSWORD=pre' violetfern's python3 app.py", "DB_PASSWORD=[REDACTED] python3 app.py", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre' violetfern's APP_ENV=prod", "export DB_PASSWORD=[REDACTED] APP_ENV=prod", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre' violetfern's # comment", "export DB_PASSWORD=[REDACTED] # comment", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre' violetfern's > /tmp/out", "export DB_PASSWORD=[REDACTED] > /tmp/out", true, []string{"violetfern"}},
		{"(export DB_PASSWORD=pre' violetfern's )", "(export DB_PASSWORD=[REDACTED] )", true, []string{"violetfern"}},
		{"(export DB_PASSWORD=pre' violetfern's)", "(export DB_PASSWORD=[REDACTED]", true, []string{"violetfern"}},
		{"export DB_PASSWORD=\"a\"pre' violetfern's", "export DB_PASSWORD=[REDACTED]", true, []string{"violetfern"}},
		{"password=\"a\"users' violetfern's changed", "password=[REDACTED] changed", true, []string{"violetfern"}},
		{"export DB_PASSWORD=pre's\nfoo='bar' violetfern's\necho ok", "export DB_PASSWORD=[REDACTED]\necho ok", true, []string{"violetfern"}},
		{"export DB_PASSWORD=\"a b\"pre's\nfoo='bar' violetfern's\necho ok", "export DB_PASSWORD=[REDACTED]\necho ok", true, []string{"violetfern"}},
		{"export \"DB_PASSWORD\"=pre's\nfoo='bar' violetfern's\necho ok", "export \"DB_PASSWORD\"=[REDACTED]\necho ok", true, []string{"violetfern"}},
		{"export \"db_password\"=\"a b\"pre' violetfern's\necho ok", "export \"db_password\"=[REDACTED]\necho ok", true, []string{"violetfern"}},
		{"curl -H 'x: y' Bearer pre' violetfern's", "curl -H 'x: y' Bearer [REDACTED]", true, []string{"violetfern"}},
		{"the password=users' wasn't changed", "the password=[REDACTED] wasn't changed", true, []string{}},
		{"password=it's Bob's", "password=[REDACTED] Bob's", true, []string{}},
		{"password=abc' (rejected)\ntoken_count=3 next_page_token=c1", "password=[REDACTED] (rejected)\ntoken_count=3 next_page_token=c1", true, []string{}},
		// Same-line lowercase unclosed quotes and trailing-quote newline continuations.
		{"export DB_PASSWORD=pre'd synthetic_lower_991", "export DB_PASSWORD=[REDACTED]", true, []string{"synthetic_lower_991"}},
		{"export DB_PASSWORD=\"a\"pre'd synthetic_lower_991", "export DB_PASSWORD=[REDACTED]", true, []string{"synthetic_lower_991"}},
		{"x=\"Bearer abc'd zqxsekreta\"", "x=\"Bearer [REDACTED]\"", true, []string{"zqxsekreta"}},
		{"export DB_PASSWORD=\"a\"pre'\nSYNTHETIC_TRAIL_991", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_TRAIL_991"}},
		{"export DB_PASSWORD=pre'\nSYNTHETIC_TRAIL_991", "export DB_PASSWORD=[REDACTED]", true, []string{"SYNTHETIC_TRAIL_991"}},
		{"export \"DB_PASSWORD\"=pre'\nSYNTHETIC_QVAR_NL_902", "export \"DB_PASSWORD\"=[REDACTED]", true, []string{"SYNTHETIC_QVAR_NL_902"}},
		{"password=doesn't work", "password=[REDACTED] work", true, []string{}},
		{"Bearer user's request", "Bearer [REDACTED]'s request", true, []string{}},
		// Closed, folded, escaped-quote, and outer-quoted single-line and multiline auth headers.
		{"Authorization: \"abc\npart2: SYNTHETIC_HDR_995\"", "Authorization: [REDACTED]", true, []string{"SYNTHETIC_HDR_995"}},
		{"Authorization: 'abc\npart2: SYNTHETIC_HDR_995'", "Authorization: [REDACTED]", true, []string{"SYNTHETIC_HDR_995"}},
		{"x-api-key: \"{\n  sub_field: SYNTHETIC_HDR_995\n}\"", "x-api-key: [REDACTED]", true, []string{"SYNTHETIC_HDR_995"}},
		{"Authorization: \"abc\n  folded_part: SYNTHETIC_HDR_995", "Authorization: [REDACTED]", true, []string{"SYNTHETIC_HDR_995"}},
		{"x-api-key: \\\"abc\nSYNTHETIC_HDR_995\\\"", "x-api-key: [REDACTED]", false, []string{"SYNTHETIC_HDR_995"}},
		{"'Authorization: SYNTHETIC_HDR_995'\nKEEP_NEXT_LINE", "'Authorization: [REDACTED]'\nKEEP_NEXT_LINE", true, []string{"SYNTHETIC_HDR_995"}},
		{"msg=\"Authorization: SYNTHETIC_HDR_995\"\nKEEP_NEXT_LINE", "msg=\"Authorization: [REDACTED]\"\nKEEP_NEXT_LINE", true, []string{"SYNTHETIC_HDR_995"}},
		{"curl -H 'Authorization: Bearer abc\"def'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"abc\"def"}},
		{"curl -H 'Authorization: Bearer abc\\'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"abc\\"}},
		{"curl -H \"Authorization: Bearer abc'def\"\necho KEEP_LINE_2", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_LINE_2", true, []string{"abc'def"}},
		{"curl -H \"Authorization: Bearer it's\"\necho KEEP_LINE_2", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_LINE_2", true, []string{}},
		{"{\"h\": \\\"Authorization: Bearer abc'def\\\"}\necho KEEP_LINE_2", "{\"h\": \\\"Authorization: [REDACTED]\necho KEEP_LINE_2", false, []string{"abc'def"}},
		{"curl -H 'Authorization: Bearer abc\nSYNTHETIC_CURL_901'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_CURL_901"}},
		{"curl -H \"x-api-key: abc\nSYNTHETIC_CURL_901\"\necho KEEP_LINE_2", "curl -H \"x-api-key: [REDACTED]\"\necho KEEP_LINE_2", true, []string{"SYNTHETIC_CURL_901"}},
		{"curl -H 'x-api-key: abc\nSYNTHETIC_OUTER_UNCL_901\nHost: example.com", "curl -H 'x-api-key: [REDACTED]\nHost: example.com", true, []string{"SYNTHETIC_OUTER_UNCL_901"}},
		{"{\"h\": \\\"Authorization: Bearer abc\nSYNTHETIC_INNER_ESC_901\\\"}", "{\"h\": \\\"Authorization: [REDACTED]", false, []string{"SYNTHETIC_INNER_ESC_901"}},
		{"req=\"Authorization: Bearer SYNTHETIC_REQ_901\nHost: example.com\"", "req=\"Authorization: [REDACTED]\"", true, []string{"SYNTHETIC_REQ_901"}},
		{"req=\"Authorization: Bearer SYNTHETIC_REQ_901\nHost: example.com", "req=\"Authorization: [REDACTED]\nHost: example.com", true, []string{"SYNTHETIC_REQ_901"}},
		{"Couldn't authenticate. Authorization: Bearer zqsecretb\nZQKEEP_L2\nZQKEEP_L3", "Couldn't authenticate. Authorization: [REDACTED]\nZQKEEP_L2\nZQKEEP_L3", true, []string{"zqsecretb"}},
		{"Error: couldn't call API (Authorization: Bearer zqsecretbb)\n  at foo.go:12\n  at bar.go:34 ZQKEEP_D", "Error: couldn't call API (Authorization: [REDACTED]\n  at foo.go:12\n  at bar.go:34 ZQKEEP_D", true, []string{"zqsecretbb"}},
		{"WARN don't retry; Authorization: Bearer zqsecretbc\n2024-01-01 INFO next ZQKEEP_E", "WARN don't retry; Authorization: [REDACTED]\n2024-01-01 INFO next ZQKEEP_E", true, []string{"zqsecretbc"}},
		{"It's broken: x-api-key: zqsecretbd\nZQKEEP_A\nZQKEEP_B", "It's broken: x-api-key: [REDACTED]\nZQKEEP_A\nZQKEEP_B", true, []string{"zqsecretbd"}},
		{"'quoted' prefix then Authorization: Bearer zqsecretbe\nZQKEEP_K\nZQKEEP_L", "'quoted' prefix then Authorization: [REDACTED]\nZQKEEP_K\nZQKEEP_L", true, []string{"zqsecretbe"}},
		{"user's request: Authorization: Bearer zqsecretbf\nHost: example.com\nZQKEEP_C", "user's request: Authorization: [REDACTED]\nHost: example.com\nZQKEEP_C", true, []string{"zqsecretbf"}},
		{"users' Authorization: Bearer zqsecretbg\nZQKEEP_L2", "users' Authorization: [REDACTED]\nZQKEEP_L2", true, []string{"zqsecretbg"}},
		{"curl -H 'Authorization: \"abc\n\nzqsecreta\"'\necho ZQKEEP_X", "curl -H 'Authorization: [REDACTED]'\necho ZQKEEP_X", true, []string{"zqsecreta"}},
		{"msg=\"Authorization: 'abc\npart2: zqsecretab'\necho ZQKEEP_X", "msg=\"Authorization: [REDACTED]\necho ZQKEEP_X", true, []string{"zqsecretab"}},
		{"msg=\"Authorization: 'abc\npart2: zqsecretab2'\"\necho ZQKEEP_X", "msg=\"Authorization: [REDACTED]\"\necho ZQKEEP_X", true, []string{"zqsecretab2"}},
		{"echo 'x-api-key: \"abc\npart2: zqsecretac\"'\necho ZQKEEP_X", "echo 'x-api-key: [REDACTED]'\necho ZQKEEP_X", true, []string{"zqsecretac"}},
		{"curl -H 'Authorization: \\'abc\nzqsecretad\\''\necho ZQKEEP_X", "curl -H 'Authorization: [REDACTED]'\necho ZQKEEP_X", false, []string{"zqsecretad"}},
		{"Authorization: \"abc\n\nzqsecretae\"\necho ZQKEEP_X", "Authorization: [REDACTED]\necho ZQKEEP_X", true, []string{"zqsecretae"}},
		{"curl -H 'Authorization: \"abc zqsecretaf\"'\necho ZQKEEP_X", "curl -H 'Authorization: [REDACTED]'\necho ZQKEEP_X", true, []string{"zqsecretaf"}},
		{"curl -H 'Authorization: Bearer abc\n\nzqsecretc'\necho ZQKEEP_X", "curl -H 'Authorization: [REDACTED]'\necho ZQKEEP_X", true, []string{"zqsecretc"}},
		{"curl -H 'Authorization: Bearer abc\nline2\n\nzqsecretcb'\necho ZQKEEP_X", "curl -H 'Authorization: [REDACTED]'\necho ZQKEEP_X", true, []string{"zqsecretcb"}},
		{"msg=\"Authorization: Bearer abc\n\nzqsecretcc\"", "msg=\"Authorization: [REDACTED]\"", true, []string{"zqsecretcc"}},
		{"curl -H 'Authorization: Bearer abc\r\nSYNTHETIC_CRLF_901'\r\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\r\necho KEEP_LINE_2", true, []string{"SYNTHETIC_CRLF_901"}},
		{"export a=\\;1 db_password=pre's violetfern's", "export a=\\;1 db_password=[REDACTED]", true, []string{"violetfern"}},
		{"export; password=users' wasn't changed", "export; password=[REDACTED] wasn't changed", true, []string{}},
		{"export \" password=users' wasn't changed", "export \" password=[REDACTED] wasn't changed", true, []string{}},
		{"export -\"a\n\" password=users' wasn't changed", "export -\"a\n\" password=[REDACTED] wasn't changed", true, []string{}},
		{"curl -H 'Authorization: \"part1\npart2\"\nSYNTHETIC_LINE3_905'\necho KEEP_LINE_4", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_4", true, []string{"SYNTHETIC_LINE3_905"}},
		{"curl -H 'x-api-key: \"{\n  a: 1\n}\"\nSYNTHETIC_TAIL_905'\necho KEEP_LINE_4", "curl -H 'x-api-key: [REDACTED]'\necho KEEP_LINE_4", true, []string{"SYNTHETIC_TAIL_905"}},
		{"curl -H 'Authorization: \"part1\npart2: sub\"\nSYNTHETIC_LINE3_905'\necho KEEP_LINE_4", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_4", true, []string{"SYNTHETIC_LINE3_905"}},
		{"msg=\"Authorization: 'part1\npart2: x'\nSYNTHETIC_MSG3_905\"\nKEEP_LINE_4", "msg=\"Authorization: [REDACTED]\"\nKEEP_LINE_4", true, []string{"SYNTHETIC_MSG3_905"}},
		{"{\"h\": \\\"Authorization: 'part1\npart2: s1'\nSYNTHETIC_JSON_LINE3_905\\\"}", "{\"h\": \\\"Authorization: [REDACTED]", false, []string{"SYNTHETIC_JSON_LINE3_905"}},
		{"curl -H 'Authorization: Bearer\u2003\"abc\npart2: SYNTHETIC_EMSPACE_905\"'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_EMSPACE_905"}},
		{"curl -H 'Authorization: Bearer\u3000\"abc\npart2: SYNTHETIC_IDEO_905\"'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_IDEO_905"}},
		{"curl -H 'Authorization: Bearer \\'\necho KEEP_LINE_2\\'", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2\\'", false, []string{}},
		{"msg = b'Authorization: Bearer abc\nSYNTHETIC_PREF_905'\necho KEEP_LINE_2", "msg = b'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_PREF_905"}},
		{"curl -H'Authorization: Bearer abc\nSYNTHETIC_FLAG_905'\necho KEEP_LINE_2", "curl -H'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_FLAG_905"}},
		{"export X=pre's Authorization: Bearer abc\nSYNTHETIC_SH_HDR_905'\necho KEEP_LINE_2", "export X=pre's Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_SH_HDR_905"}},
		{"curl -H 'Authorization: Bearer abc\n\n\nSYNTHETIC_3NL_905'\necho KEEP_LINE_2", "curl -H 'Authorization: [REDACTED]'\necho KEEP_LINE_2", true, []string{"SYNTHETIC_3NL_905"}},
		{"{\"h\": \\\"Authorization: 'abc\\\"}\npart2: KEEP_LINE_2'", "{\"h\": \\\"Authorization: [REDACTED]\npart2: KEEP_LINE_2'", false, []string{}},
		{"curl -H 'Authorization: \"abc'\npart2: KEEP_LINE_2\"", "curl -H 'Authorization: [REDACTED]'\npart2: KEEP_LINE_2\"", true, []string{}},
		{"msg=\"Authorization: 'abc\"\npart2: KEEP_LINE_2'", "msg=\"Authorization: [REDACTED]\"\npart2: KEEP_LINE_2'", true, []string{}},
	} {
		if tt.allPaths {
			checkAllRedactionPaths(t, tt.in, tt.want)
			checkAllRedactionPaths(t, tt.want, tt.want)
		} else {
			if got := scrubTextPatterns(tt.in); got != tt.want {
				t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
			}
			if again := scrubTextPatterns(tt.want); again != tt.want {
				t.Errorf("scrubbing %q again gave %q, want it unchanged", tt.want, again)
			}
		}
		for _, secret := range tt.secrets {
			checkNoLeak(t, tt.in, secret)
		}
	}
}

func TestRound17ReviewRegressions(t *testing.T) {
	for _, n := range []int{255, 256, 257, 300} {
		for _, tc := range []struct {
			in, want, secret string
		}{
			{
				"export DB_PASSWORD=" + strings.Repeat("a", n) + "'s syntheticvioletfern's",
				"export DB_PASSWORD=[REDACTED]",
				"syntheticvioletfern",
			},
			{
				"export DB_PASSWORD=" + strings.Repeat("a", n) + "'s syntheticvioletfern",
				"export DB_PASSWORD=[REDACTED]",
				"syntheticvioletfern",
			},
			{
				"export DB_PASSWORD=" + strings.Repeat("x", n) + "users' violetfern's changed",
				"export DB_PASSWORD=[REDACTED] changed",
				"violetfern",
			},
			{
				"export db_password=" + strings.Repeat("a", n) + "'s syntheticvioletfern's",
				"export db_password=[REDACTED]",
				"syntheticvioletfern",
			},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
			checkAllRedactionPaths(t, tc.want, tc.want)
			checkNoLeak(t, tc.in, tc.secret)
		}
	}

	for _, tt := range []struct {
		in      string
		want    string
		allPath bool
		secrets []string
	}{
		// Lowercase shell assignments without export and backslash-newline continuations.
		{"db_password=pre's syntheticvioletfern's", "db_password=[REDACTED]", true, []string{"syntheticvioletfern"}},
		{"api_key=pre's syntheticvioletfern's", "api_key=[REDACTED]", true, []string{"syntheticvioletfern"}},
		{"password=pre's syntheticvioletfern's", "password=[REDACTED]", true, []string{"syntheticvioletfern"}},
		{"export\\\nDB_PASSWORD=pre's syntheticvioletfern's", "export\\\nDB_PASSWORD=[REDACTED]", true, []string{"syntheticvioletfern"}},
		{"export \\\ndb_password=pre's syntheticvioletfern's", "export \\\ndb_password=[REDACTED]", true, []string{"syntheticvioletfern"}},
		{"export \\\r\ndb_password=pre's syntheticvioletfern's", "export \\\r\ndb_password=[REDACTED]", true, []string{"syntheticvioletfern"}},
		// Long ordinary prose controls (>300 bytes before prose contractions/possessives).
		{strings.Repeat("word ", 70) + "the password=users' wasn't changed", strings.Repeat("word ", 70) + "the password=[REDACTED] wasn't changed", true, nil},
		{strings.Repeat("a", 300) + " password=it's Bob's", strings.Repeat("a", 300) + " password=[REDACTED] Bob's", true, nil},
		// Multiline headers: colon-led continuation inside closed outer quote, empty first line, prior-line outer quote.
		{"curl -H 'Authorization: abc\npart2: syntheticvioletfern'\necho ok", "curl -H 'Authorization: [REDACTED]'\necho ok", true, []string{"syntheticvioletfern"}},
		{"curl -H 'Authorization: \nsyntheticvioletfern'\necho ok", "curl -H 'Authorization: [REDACTED]'\necho ok", true, []string{"syntheticvioletfern"}},
		{"curl -H '\nAuthorization: abc\nsyntheticvioletfern'\necho ok", "curl -H '\nAuthorization: [REDACTED]'\necho ok", true, []string{"syntheticvioletfern"}},
		{"curl -H \"\nAuthorization: abc\npart2: syntheticvioletfern\"\necho ok", "curl -H \"\nAuthorization: [REDACTED]\"\necho ok", true, []string{"syntheticvioletfern"}},
		{"{\"h\": \\\"\nAuthorization: abc\nsyntheticvioletfern\\\"}\necho ok", "{\"h\": \\\"\nAuthorization: [REDACTED]\necho ok", false, []string{"syntheticvioletfern"}},
		// Unquoted RFC 9112 whitespace-folded header continuations.
		{"Authorization: Bearer abc\n  SYNTHETIC_FOLD_1601\nContent-Type: text/plain", "Authorization: [REDACTED]\nContent-Type: text/plain", true, []string{"SYNTHETIC_FOLD_1601"}},
		{"Proxy-Authorization: Basic abc\r\n\tSYNTHETIC_FOLD_1602\r\nAccept: */*", "Proxy-Authorization: [REDACTED]\r\nAccept: */*", true, []string{"SYNTHETIC_FOLD_1602"}},
		{"X-Api-Key: part1\n  SYNTHETIC_FOLD_1603", "X-Api-Key: [REDACTED]", true, []string{"SYNTHETIC_FOLD_1603"}},
		{"Api-Key: part1\n\tSYNTHETIC_FOLD_1607\nX-Request-Id: 1", "Api-Key: [REDACTED]\nX-Request-Id: 1", true, []string{"SYNTHETIC_FOLD_1607"}},
		{"X-Goog-Api-Key: part1\n  SYNTHETIC_FOLD_1609\nAccept: application/json", "X-Goog-Api-Key: [REDACTED]\nAccept: application/json", true, []string{"SYNTHETIC_FOLD_1609"}},
		// Cookie & Set-Cookie headers and JSON fields.
		{"Cookie: sid=syntheticvioletfern; preference=light", "Cookie: [REDACTED]", true, []string{"syntheticvioletfern", "preference=light"}},
		{"Cookie: session=SYNTHETIC_COOKIE_1604", "Cookie: [REDACTED]", true, []string{"SYNTHETIC_COOKIE_1604"}},
		{"Set-Cookie: session=SYNTHETIC_COOKIE_1608; Path=/; HttpOnly", "Set-Cookie: [REDACTED]", true, []string{"SYNTHETIC_COOKIE_1608"}},
		{"Cookie: sid=part1;\n  part2=SYNTHETIC_COOKIE_FOLD\nAccept: */*", "Cookie: [REDACTED]\nAccept: */*", true, []string{"SYNTHETIC_COOKIE_FOLD"}},
		{`{"headers":{"Cookie":"SYNTHETIC_COOKIE_1605","Set-Cookie":"SYNTHETIC_COOKIE_1608","Accept":"text/plain"}}`, `{"headers":{"Accept":"text/plain","Cookie":"[REDACTED]","Set-Cookie":"[REDACTED]"}}`, false, []string{"SYNTHETIC_COOKIE_1605", "SYNTHETIC_COOKIE_1608"}},
		// Unclosed outer quote across subsequent header line before a later paired string.
		{"req=\"Authorization: Bearer SYNTHETIC_REQ_901\nContent-Type: application/json\n\n{\"status\": \"ok\"}\nKEEP_AFTER", "req=\"Authorization: [REDACTED]\nContent-Type: application/json\n\n{\"status\": \"ok\"}\nKEEP_AFTER", true, []string{"SYNTHETIC_REQ_901"}},
		{"req='Authorization: Bearer SYNTHETIC_REQ_901\nContent-Type: application/json\n\nX-Note: 'hello'\nKEEP_AFTER", "req='Authorization: [REDACTED]\nContent-Type: application/json\n\nX-Note: 'hello'\nKEEP_AFTER", true, []string{"SYNTHETIC_REQ_901"}},
		// Prior-line closed multiline strings and mid-lookback-window closing quotes before an unquoted header.
		{"msg=\"hello\nworld!\"\nAuthorization: Bearer SYNTHETIC_SECRET\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", "msg=\"hello\nworld!\"\nAuthorization: [REDACTED]\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET"}},
		{"msg=\"hello\nworld \"\nAuthorization: Bearer SYNTHETIC_SECRET\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", "msg=\"hello\nworld \"\nAuthorization: [REDACTED]\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET"}},
		{"msg='hello\nworld2'\nAuthorization: Bearer SYNTHETIC_SECRET\nHost: example.com\nstatus='ok'\nKEEP_AFTER", "msg='hello\nworld2'\nAuthorization: [REDACTED]\nHost: example.com\nstatus='ok'\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET"}},
		{"msg=\"hello\n\"\nAuthorization: Bearer SYNTHETIC_SECRET\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", "msg=\"hello\n\"\nAuthorization: [REDACTED]\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET"}},
		{"msg=\"" + strings.Repeat("line_of_prose_text_1234567890\n", 160) + "last_word\"\nAuthorization: Bearer SYNTHETIC_SECRET\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", "msg=\"" + strings.Repeat("line_of_prose_text_1234567890\n", 160) + "last_word\"\nAuthorization: [REDACTED]\nHost: example.com\nstatus=\"ok\"\nKEEP_AFTER", false, []string{"SYNTHETIC_SECRET"}},
		// Escaped quotes inside paired double-quoted arguments on the closing line of a multiline header.
		{"curl -H \"Authorization: abc\npart2: SYNTHETIC_SECRET_ESC_1\" -H \"X-Quote: a\\\"b\"\necho KEEP_OK", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_ESC_1"}},
		{"curl -H \"Authorization: abc\npart2: SYNTHETIC_SECRET_ESC_2\" -d \"{\\\"msg\\\": \\\"a\\\\\\\"b\\\"}\"\necho KEEP_OK", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_ESC_2"}},
		// Unmatched quote on an earlier line before a multiline outer-quoted header.
		{"Error: unexpected \" in input\ncurl -H \"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_1\"\necho KEEP_OK", "Error: unexpected \" in input\ncurl -H \"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_1"}},
		{"Error: unexpected ' in input\ncurl -H 'Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_2'\necho KEEP_OK", "Error: unexpected ' in input\ncurl -H 'Authorization: [REDACTED]'\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_2"}},
		{"Error: unexpected \" in input\ncurl -H \"\nAuthorization: abc\npart2: SYNTHETIC_SECRET_TAINT_3\"\necho KEEP_OK", "Error: unexpected \" in input\ncurl -H \"\nAuthorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_3"}},
		{"Error: unexpected \" in input\n{\\\"h\\\": \\\"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_4\\\"}\necho KEEP_OK", "Error: unexpected \" in input\n{\\\"h\\\": \\\"Authorization: [REDACTED]\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_4"}},
		{"Error: unexpected \" in input\n\\\"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_5\\\"\necho KEEP_OK", "Error: unexpected \" in input\n\\\"Authorization: [REDACTED]\\\"\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_5"}},
		{"Error: unexpected ' in input\n{\\'h\\': \\'Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_6\\'}\necho KEEP_OK", "Error: unexpected ' in input\n{\\'h\\': \\'Authorization: [REDACTED]\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_6"}},
		{"Error: unexpected ' in input\n\\'Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_7\\'\necho KEEP_OK", "Error: unexpected ' in input\n\\'Authorization: [REDACTED]\\'\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_7"}},
		{"Error: unexpected ' in input\ncurl -H \"X-Note: a'b\" -H \"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_8\"\necho KEEP_OK", "Error: unexpected ' in input\ncurl -H \"X-Note: a'b\" -H \"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_8"}},
		{"Error: unexpected ' in input\ncurl -H \"X-Note: 'hi'\" -H \"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_9\"\necho KEEP_OK", "Error: unexpected ' in input\ncurl -H \"X-Note: 'hi'\" -H \"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_9"}},
		{"Error: unexpected \" in input\ncurl -H 'X-Note: \"hi\"' -H 'Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_10'\necho KEEP_OK", "Error: unexpected \" in input\ncurl -H 'X-Note: \"hi\"' -H 'Authorization: [REDACTED]'\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_10"}},
		{"Error: unexpected \\\" in input\ncurl -d \"{\\\"h\\\": \\\"Authorization: abc\npart2: SYNTHETIC_SECRET_TAINT_11\\\"}\"\necho KEEP_OK", "Error: unexpected \\\" in input\ncurl -d \"{\\\"h\\\": \\\"Authorization: [REDACTED]\"\necho KEEP_OK", true, []string{"SYNTHETIC_SECRET_TAINT_11"}},
		{"Error: unexpected \" in input\n{\\\"h\\\": \\\"\nAuthorization: abc\npart2: SYNTHETIC_SECRET_TAINT_12\\\"}\necho KEEP_OK", "Error: unexpected \" in input\n{\\\"h\\\": \\\"\nAuthorization: [REDACTED]\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_12"}},
		// Decoding the escaped quotes of the X-Json header pairs the quotes
		// of this text differently, so the error paths, which check that
		// decoding a scrubbed text exposes nothing, fail closed on it (see
		// taint13 below).
		{"Error: unexpected ' in input\ncurl -H \"X-Json: {\\\"k\\\": \\\"\\\"}\nAuthorization: abc\npart2: SYNTHETIC_SECRET_TAINT_13\"\necho KEEP_OK", "Error: unexpected ' in input\ncurl -H \"X-Json: {\\\"k\\\": \\\"\\\"}\nAuthorization: [REDACTED]\"\necho KEEP_OK", false, []string{"SYNTHETIC_SECRET_TAINT_13"}},
		// Prior-line closed multiline string and unclosed header quote before a later single-opening-quote header line.
		{"msg=\"hello\nworld \"\nAuthorization: Bearer SYNTHETIC_SECRET_END_1\nHost: example.com\nX-End: \"\nKEEP_AFTER", "msg=\"hello\nworld \"\nAuthorization: [REDACTED]\nHost: example.com\nX-End: \"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET_END_1"}},
		{"req=\"Authorization: Bearer SYNTHETIC_SECRET_END_2\nHost: example.com\nX-End: \"\nKEEP_AFTER", "req=\"Authorization: [REDACTED]\nHost: example.com\nX-End: \"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET_END_2"}},
		// Colon-led continuation lines ending with Base64 '=' / '==' padding or trailing ':' before closing quote.
		{"curl -H \"Authorization: Basic dXNlcjpwYXNz\npart2: c2VjcmV0dG9rZW4=\"\necho KEEP_TOK", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_TOK", true, []string{"c2VjcmV0dG9rZW4="}},
		{"curl -H \"Authorization: Bearer part1\npart2: c2VjcmV0==\"\necho KEEP_TOK", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_TOK", true, []string{"c2VjcmV0=="}},
		{"curl -H \"Authorization: Basic dXNlcj\npart2: pass:\"\necho KEEP_TOK", "curl -H \"Authorization: [REDACTED]\"\necho KEEP_TOK", true, []string{"pass:"}},
		// Column-0 standalone closing quote line of a prior-line multiline literal vs. standalone opener after stray prose quote.
		{"msg=\"hello\n\"\nAuthorization: Bearer SYNTHETIC_SECRET_COL0_1\nHost: KEEP_TOK\necho \"\nKEEP_AFTER", "msg=\"hello\n\"\nAuthorization: [REDACTED]\nHost: KEEP_TOK\necho \"\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET_COL0_1"}},
		{"msg='hello\n'\nAuthorization: Bearer SYNTHETIC_SECRET_COL0_2\nHost: KEEP_TOK\necho '\nKEEP_AFTER", "msg='hello\n'\nAuthorization: [REDACTED]\nHost: KEEP_TOK\necho '\nKEEP_AFTER", true, []string{"SYNTHETIC_SECRET_COL0_2"}},
		{"Error: unexpected \" in input\n\"\nAuthorization: Bearer SYNTHETIC_SECRET_COL0_3\n   part2\"\necho KEEP_TOK", "Error: unexpected \" in input\n\"\nAuthorization: [REDACTED]\"\necho KEEP_TOK", true, []string{"SYNTHETIC_SECRET_COL0_3", "part2"}},
	} {
		if tt.allPath {
			checkAllRedactionPaths(t, tt.in, tt.want)
			checkAllRedactionPaths(t, tt.want, tt.want)
		} else if got, _ := sanitizeRawText(tt.in); got != tt.want && scrubTextPatterns(tt.in) != tt.want {
			t.Errorf("scrub(%q) = %q / %q, want %q", tt.in, scrubTextPatterns(tt.in), got, tt.want)
		}
		for _, secret := range tt.secrets {
			checkNoLeak(t, tt.in, secret)
		}
	}
	taint13 := "Error: unexpected ' in input\ncurl -H \"X-Json: {\\\"k\\\": \\\"\\\"}\nAuthorization: abc\npart2: SYNTHETIC_SECRET_TAINT_13\"\necho KEEP_OK"
	if want := "Error: unexpected ' in input\ncurl -H \"X-Json: {\\\"k\\\": \\\"\\\"}\nAuthorization: [REDACTED]\"\necho KEEP_OK"; scrubTextPatterns(want) != want {
		t.Errorf("scrubbing %q again gave %q, want it unchanged", want, scrubTextPatterns(want))
	}
	if got, _ := sanitizeRawText(taint13); got != redactedSensitiveTextPlaceholder {
		t.Errorf("sanitizeRawText(%q) = %q, want %q", taint13, got, redactedSensitiveTextPlaceholder)
	}
	if got := redactErrorMessage(taint13); got != redactedSensitiveTextPlaceholder {
		t.Errorf("redactErrorMessage(%q) = %q, want %q", taint13, got, redactedSensitiveTextPlaceholder)
	}

	// Verify linear O(N) scaling on multiline outer-quoted headers and prior-line lookback windows.
	for _, sc := range []struct {
		name string
		in   string
		want string
	}{
		{
			"multiline_outer_closed",
			"curl -H 'Authorization: Bearer abc" + strings.Repeat("\nline_of_secret_payload_12345", 8000) + "'",
			"curl -H 'Authorization: [REDACTED]'",
		},
		{
			"multiline_outer_unclosed",
			"curl -H 'Authorization: Bearer abc" + strings.Repeat("\nline_of_secret_payload_12345", 8000),
			"curl -H 'Authorization: [REDACTED]",
		},
		{
			"prior_blank_lines_with_tail",
			"curl -H '" + strings.Repeat("\n", 2000) + "\nAuthorization: Bearer abc\n" + strings.Repeat("x", 256000) + "'",
			"curl -H '" + strings.Repeat("\n", 2000) + "\nAuthorization: [REDACTED]'",
		},
	} {
		start := time.Now()
		got := scrubTextPatterns(sc.in)
		elapsed := time.Since(start)
		if got != sc.want {
			t.Errorf("%s: got len %d, want len %d", sc.name, len(got), len(sc.want))
		}
		if elapsed > 500*time.Millisecond {
			t.Errorf("%s: scrubTextPatterns took %v, want <= 500ms", sc.name, elapsed)
		}
	}
}

// checkLongRedactionPaths checks, as checkAllRedactionPaths does, that
// scrubTextPatterns, sanitizeRawText, redactErrorMessage and SmartTruncate of
// in as a string write want for in, that SmartTruncate writes errors.New(in)
// as {}, and that the scrubs leave want as it is, with a SmartTruncate budget
// that holds all of in, for texts longer than the budget of
// checkRedactionPaths allows, and that none of them writes the secrets.
func checkLongRedactionPaths(t *testing.T, in, want string, secrets ...string) {
	t.Helper()
	raw, _ := sanitizeRawText(in)
	var smart string
	if b, _, err := smartTruncateText(t, in, 2*len(in)+1000); err != nil || json.Unmarshal(b, &smart) != nil {
		t.Errorf("SmartTruncate(%s) = (%s, %v), want a JSON string", elideText(in), elideText(string(b)), err)
	}
	for _, p := range []struct{ name, in, got string }{
		{"scrubTextPatterns", in, scrubTextPatterns(in)},
		{"sanitizeRawText", in, raw},
		{"redactErrorMessage", in, redactErrorMessage(in)},
		{"SmartTruncate", in, smart},
		{"scrubTextPatterns", want, scrubTextPatterns(want)},
		{"redactErrorMessage", want, redactErrorMessage(want)},
	} {
		if p.got != want {
			t.Errorf("%s(%s) = %s, want %s", p.name, elideText(p.in), elideText(p.got), elideText(want))
		}
		for _, secret := range secrets {
			if strings.Contains(p.got, secret) {
				t.Errorf("%s(%s) = %s, which still holds %q", p.name, elideText(p.in), elideText(p.got), secret)
			}
		}
	}
}

// elideText quotes s for a test message, with the middle of a long s left
// out.
func elideText(s string) string {
	if len(s) <= 240 {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%q...<%d bytes>...%q", s[:80], len(s)-240, s[len(s)-160:])
}

// TestRound22ReviewRegressions checks the boundaries of the text scrub that
// the direct, error message and SmartTruncate text paths share, for headers,
// shell assignments and prose:
//
//   - a whitespace-folded header value runs on over a line of spaces or tabs
//     alone to the folded lines after it, and past a first line that holds
//     only the redaction placeholder, for every credential header, after LF
//     and CRLF, while an empty line, or a line indented no further than the
//     header, ends it;
//   - a quote that opens any number of bytes or lines before a header and
//     stays open into it carries the header's value on to where it closes;
//   - a quote after a run of options and assignments long enough to push the
//     command's declaration out of the shell lookbehind window is a shell
//     quote, while a long word whose visible part shows that the name before
//     its "=" is not a shell name keeps the contraction after it;
//   - a possessive after it's is prose whatever the length or the script of
//     its word, while a word glued to "=" that 's does not make a
//     contraction of opens a quoted part of the value, however short the
//     secret in the part;
//   - the word cookie and a colon in prose, with no cookie pair after them,
//     are not a Cookie header;
//
// and that the scrub takes time linear in the length of the text for runs of
// the headers and assignments that these boundaries read.
func TestRound22ReviewRegressions(t *testing.T) {
	t.Run("folded headers", func(t *testing.T) {
		const secret = "SYNTHETIC_CANARY_82517"
		for _, h := range []struct{ name, first string }{
			{"Authorization", "Bearer abc"},
			{"Proxy-Authorization", "Basic abc"},
			{"Api-Key", "part1"},
			{"X-Api-Key", "part1"},
			{"X-Goog-Api-Key", "part1"},
			{"Api-Key-Secret", "part1"},
			{"Api-Key-String", "part1"},
			{"Cookie", "sid=part1;"},
			{"Set-Cookie", "session=part1;"},
		} {
			for _, nl := range []string{"\n", "\r\n"} {
				for _, ws := range []string{" ", "  ", "\t", "\t\t", " \t"} {
					head := h.name + ": " + h.first
					placeholder := h.name + ": [REDACTED]"
					next := nl + "Accept: text/plain"
					for _, in := range []string{
						head + nl + ws + nl + "\t" + secret + next,
						head + nl + ws + nl + ws + secret + next,
						head + nl + ws + nl + ws + nl + ws + secret + next,
						placeholder + nl + ws + secret + next,
						placeholder + nl + ws + nl + ws + secret + next,
					} {
						checkAllRedactionPaths(t, in, placeholder+next)
						checkNoLeak(t, in, secret)
					}
					for _, in := range []string{
						head + nl + ws + nl + ws + secret,
						placeholder + nl + ws + secret,
					} {
						checkAllRedactionPaths(t, in, placeholder)
						checkNoLeak(t, in, secret)
					}
				}
			}
		}
		for _, tt := range []struct{ in, want string }{
			{"Authorization: Bearer abc\r\n \r\n\tSYNTHETIC_CANARY_82517\r\nAccept: text/plain", "Authorization: [REDACTED]\r\nAccept: text/plain"},
			{"Authorization: [REDACTED]\r\n\tSYNTHETIC_CANARY_82517\r\nAccept: text/plain", "Authorization: [REDACTED]\r\nAccept: text/plain"},
			{"Authorization: Bearer abc\n  \n  syntheticfold\nContent-Type: text/plain", "Authorization: [REDACTED]\nContent-Type: text/plain"},
			// A folded Cookie value whose first line holds no cookie pair.
			{"Cookie: abc\n\tsid=SYNTHETIC_COOKIE_2201\nAccept: text/plain", "Cookie: [REDACTED]\nAccept: text/plain"},
			{"cookie: Monster\n\tsid=SYNTHETIC_COOKIE_2201\nAccept: */*", "cookie: [REDACTED]\nAccept: */*"},
			// An indented header folds over the lines indented further.
			{"  Authorization: Bearer abc\n    \n    SYNTHETIC_FOLD_2202\n  Accept: text/plain", "  Authorization: [REDACTED]\n  Accept: text/plain"},
			// An empty line, or a line indented no further than the header,
			// ends the header and is kept.
			{"Authorization: Bearer abc\r\n\r\n{\"status\":\"ok\"}", "Authorization: [REDACTED]\r\n\r\n{\"status\":\"ok\"}"},
			{"Authorization: Bearer abc\n\n{\"status\":\"ok\"}", "Authorization: [REDACTED]\n\n{\"status\":\"ok\"}"},
			{"Authorization: [REDACTED]\r\n\r\n\tbody", "Authorization: [REDACTED]\r\n\r\n\tbody"},
			{"  Authorization: Bearer abc\n  \n  Accept: text/plain", "  Authorization: [REDACTED]\n  \n  Accept: text/plain"},
		} {
			checkAllRedactionPaths(t, tt.in, tt.want)
			for _, secret := range []string{"SYNTHETIC_CANARY_82517", "syntheticfold", "SYNTHETIC_COOKIE_2201", "SYNTHETIC_FOLD_2202"} {
				checkNoLeak(t, tt.in, secret)
			}
		}
	})

	t.Run("prior-line quote lookback", func(t *testing.T) {
		const secret = "syntheticlookback"
		for _, n := range []int{100, 4095, 4096, 4097, 5000} {
			pad := "X-Pad: " + strings.Repeat("p", n)
			for _, prefix := range []string{"'", "curl -H '", "curl -H \""} {
				q := prefix[len(prefix)-1:]
				in := prefix + pad + "\nAuthorization: Bearer abc\n" + secret + q + "\necho KEEP_HEADER_TAIL"
				// The quote that closes the string the header is in is
				// kept, so a scrub of the result reads that string again.
				checkLongRedactionPaths(t, in, prefix+pad+"\nAuthorization: [REDACTED]"+q+"\necho KEEP_HEADER_TAIL", secret)
			}
			// A text that opens with a double quote is a JSON blob that does
			// not parse for the paths that try it, which write a placeholder
			// for all of it; the scrub of its text keeps the rest.
			in := "\"" + pad + "\nAuthorization: Bearer abc\n" + secret + "\"\necho KEEP_HEADER_TAIL"
			if got, want := scrubTextPatterns(in), "\""+pad+"\nAuthorization: [REDACTED]\"\necho KEEP_HEADER_TAIL"; got != want {
				t.Errorf("scrubTextPatterns(%s) = %s, want %s", elideText(in), elideText(got), elideText(want))
			}
			raw, _ := sanitizeRawText(in)
			smart, _, err := smartTruncateText(t, in, 2*len(in)+1000)
			if err != nil || strings.Contains(raw, secret) || strings.Contains(string(smart), secret) || strings.Contains(redactErrorMessage(in), secret) {
				t.Errorf("a path wrote %q for %s: %s / %s / %v", secret, elideText(in), elideText(raw), elideText(string(smart)), err)
			}
		}
		// A quote that opens hundreds of header lines before the header.
		var pad strings.Builder
		for i := range 300 {
			fmt.Fprintf(&pad, "X-Pad-%d: value\n", i)
		}
		for _, prefix := range []string{"'", "curl -H '", "curl -H \""} {
			q := prefix[len(prefix)-1:]
			in := prefix + pad.String() + "Authorization: Bearer abc\n" + secret + q + "\necho KEEP_HEADER_TAIL"
			checkLongRedactionPaths(t, in, prefix+pad.String()+"Authorization: [REDACTED]"+q+"\necho KEEP_HEADER_TAIL", secret)
		}
	})

	t.Run("shell context", func(t *testing.T) {
		const secret = "violetfern"
		const redacted = "password=[REDACTED] changed"
		// export and the options and assignments after it own the quotes
		// of the assignment that follows them, however many there are,
		// past the run (61 assignments and more) after which the window
		// of inShellAssignment no longer reaches export. The possessive
		// and the contraction after the plural possessive are prose outside
		// a shell assignment, so that each pins the window's guard.
		for _, pair := range []string{"password=users' violetfern's changed", "password=users' violetfern wasn't changed"} {
			for _, k := range []int{1, 50, 58, 59, 60, 61, 62, 63, 64, 100} {
				for _, arg := range []string{"X=1 ", "-x "} {
					prefix := "export " + strings.Repeat(arg, k)
					checkLongRedactionPaths(t, prefix+pair, prefix+redacted, secret)
				}
			}
			for _, prefix := range []string{
				"export" + strings.Repeat(" ", 300),
				"export \\\n" + strings.Repeat("X=1 \\\n", 60),
				"export \\\r\n" + strings.Repeat("X=1 \\\r\n", 60),
				"export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/custom/very/long/directory/path/that/exceeds/two/hundred/and/fifty/six/bytes/by/quite/a/lot/so/winstart/lands/inside/the/value/after/colon:/usr/games:/usr/local/games:/snap/bin ",
				"export " + strings.Repeat("--very-long-shell-option-flag-name ", 8),
				"export " + strings.Repeat("--very-long-shell-option-flag-name ", 59),
				"export A\\\n=1 ",
				"export \\\n-x ",
				"ex\\\nport ",
				"ex\\\r\nport ",
			} {
				checkLongRedactionPaths(t, prefix+pair, prefix+redacted, secret)
			}
		}
		// Prose before the pair, or a command that ends before it, keeps
		// the contraction after it.
		for _, k := range []int{40, 50, 51, 52, 70} {
			prefix := strings.Repeat("word ", k)
			checkLongRedactionPaths(t, prefix+"the password=users' wasn't changed", prefix+"the password=[REDACTED] wasn't changed")
		}
		for _, tt := range []struct{ in, want string }{
			{"export; password=users' wasn't changed", "export; password=[REDACTED] wasn't changed"},
			{"export X=1; password=users' wasn't changed", "export X=1; password=[REDACTED] wasn't changed"},
			{"echo hi\npassword=users' wasn't changed", "echo hi\npassword=[REDACTED] wasn't changed"},
		} {
			checkLongRedactionPaths(t, tt.in, tt.want)
		}
		// A long word whose visible part shows that the name before its
		// "=" is not a shell name, as the ":" before password does, keeps
		// its contraction and the pairs after it, while a long value of a
		// credential assignment is redacted whole.
		for _, n := range []int{255, 256, 257, 300} {
			prefix := strings.Repeat("a", n)
			tail := " wasn't changed; prompt_token_count=10 next_page_token=c1"
			checkLongRedactionPaths(t, prefix+":password=users'"+tail, prefix+":password=[REDACTED]"+tail)
			for _, tt := range []struct{ in, want string }{
				{"export DB_PASSWORD=" + prefix + "users' violetfern's changed", "export DB_PASSWORD=[REDACTED] changed"},
				{"export db_password=" + prefix + "users' violetfern's changed", "export db_password=[REDACTED] changed"},
				{"DB_PASSWORD=" + prefix + "users' violetfern's changed", "DB_PASSWORD=[REDACTED] changed"},
				{prefix + " db_password=users' violetfern's changed", prefix + " db_password=[REDACTED] changed"},
			} {
				checkLongRedactionPaths(t, tt.in, tt.want, secret)
			}
		}
	})

	t.Run("shell guard witnesses", func(t *testing.T) {
		// Short closing words, which no rule on the length of a word
		// redacts on its own, and which outside a shell assignment are
		// prose after a plural possessive, so that each pins the guard it
		// names.
		for _, closing := range []string{"violetfern's", "violetfern wasn't", "violetfern it's"} {
			for _, tt := range []struct{ in, want string }{
				// A snake_case name, and a name with a digit, with no declaration.
				{"db_password=users' " + closing + " changed", "db_password=[REDACTED] changed"},
				{"oauth2_token=users' " + closing + " changed", "oauth2_token=[REDACTED] changed"},
				// A declaration, a backslash-newline and whitespace before the name.
				{"export\\\n password=users' " + closing + " changed", "export\\\n password=[REDACTED] changed"},
				{"export\\\r\n password=users' " + closing + " changed", "export\\\r\n password=[REDACTED] changed"},
				{"export \\\n password=users' " + closing + " changed", "export \\\n password=[REDACTED] changed"},
				{"export A\\\n=1 password=users' " + closing + " changed", "export A\\\n=1 password=[REDACTED] changed"},
				{"export \\\n-x password=users' " + closing + " changed", "export \\\n-x password=[REDACTED] changed"},
				{"ex\\\nport password=users' " + closing + " changed", "ex\\\nport password=[REDACTED] changed"},
			} {
				checkAllRedactionPaths(t, tt.in, tt.want)
				checkNoLeak(t, tt.in, "violetfern")
			}
		}
		// The same contractions outside a shell assignment are prose.
		for _, tt := range []struct{ in, want string }{
			{"the password=users' wasn't changed", "the password=[REDACTED] wasn't changed"},
			{"the password=users' it's changed", "the password=[REDACTED] it's changed"},
		} {
			checkAllRedactionPaths(t, tt.in, tt.want)
		}
	})

	t.Run("possessives after it's", func(t *testing.T) {
		for _, name := range []string{"Bob", "administrator", "Bartholomewsmith", "Екатерина", "José", strings.Repeat("x", 64)} {
			for _, tt := range []struct{ in, want string }{
				{"the password=it's " + name + "'s", "the password=[REDACTED] " + name + "'s"},
				{"password=it's " + name + "'s", "password=[REDACTED] " + name + "'s"},
				{"the password=it's " + name + "'s now", "the password=[REDACTED] " + name + "'s now"},
				{"the password=It's " + name + "'s", "the password=[REDACTED] " + name + "'s"},
			} {
				checkAllRedactionPaths(t, tt.in, tt.want)
			}
		}
	})

	t.Run("short secrets in quoted parts", func(t *testing.T) {
		// A word glued to "=" that 's does not make a contraction of, as
		// pre is not, opens a quoted part of the value, which the next
		// apostrophe closes, so that the words up to it are the secret,
		// however short, and the word after it is kept. A part that no
		// apostrophe closes owns the rest of the text, as an unclosed
		// shell quote does.
		for _, key := range []string{"password", "token", "apikey"} {
			for _, secret := range []string{"z", "zqxjkv", "zqxjkvwpbnmh", "zqxjkvwpbnmhy"} {
				for _, tt := range []struct{ in, want string }{
					{key + "=pre's " + secret + "'s changed", key + "=[REDACTED] changed"},
					{key + "=pre's " + secret + "'s", key + "=[REDACTED]"},
					{key + "=pre's " + secret + " changed", key + "=[REDACTED]"},
					{key + "=pre's " + secret, key + "=[REDACTED]"},
				} {
					checkAllRedactionPaths(t, tt.in, tt.want)
					checkNoLeak(t, tt.in, secret)
				}
			}
		}
		for _, tt := range []struct{ in, want, secret string }{
			{"password=pre's synthlow's changed", "password=[REDACTED] changed", "synthlow"},
			{"token=pre's secret's changed", "token=[REDACTED] changed", "secret"},
			{"apikey=pre's secret's changed", "apikey=[REDACTED] changed", "secret"},
			{"password=pre's synthlow", "password=[REDACTED]", "synthlow"},
			{"password=v1_pre's synthlow's changed", "password=[REDACTED] changed", "synthlow"},
			{"token=my-pre's synthtok's changed", "token=[REDACTED] changed", "synthtok"},
			{"apikey=2fapre's synthkey's changed", "apikey=[REDACTED] changed", "synthkey"},
			{"secret=a.b's synthdot's changed", "secret=[REDACTED] changed", "synthdot"},
			{"password=v1_pre's synthunclosed", "password=[REDACTED]", "synthunclosed"},
		} {
			checkAllRedactionPaths(t, tt.in, tt.want)
			checkNoLeak(t, tt.in, tt.secret)
		}
		// A word that 's makes a contraction of, glued to "=", and a
		// contraction or a possessive after a plural possessive outside a
		// shell assignment, are prose.
		for _, tt := range []struct{ in, want string }{
			{"password=it's foo bar's", "password=[REDACTED] foo bar's"},
			{"the password=that's Bob's", "the password=[REDACTED] Bob's"},
			{"the password=users' it's changed", "the password=[REDACTED] it's changed"},
			{"the password=users' isn't changed", "the password=[REDACTED] isn't changed"},
			{"the password=users' they're changed", "the password=[REDACTED] they're changed"},
			{"the password=users' manager's changed", "the password=[REDACTED] manager's changed"},
		} {
			checkAllRedactionPaths(t, tt.in, tt.want)
		}
	})

	t.Run("cookie prose", func(t *testing.T) {
		for _, in := range []string{
			"cookie: Monster",
			"The cookie: Monster was here.",
			"Cookie: Monster",
			"Set-Cookie: Monster",
			"cookie: Monster cookie: Elmo",
			"I ate a cookie: Monster says hi\nThen another.",
			"The cookie: Monster said \"hello\nworld\" today.",
			"Cookie: Monster said \"hello\nworld\" today.",
			"The cookie: Monster was here " + strings.Repeat("ordinary words ", 25),
			"Set-Cookie2: x",
			"X-Cookie-Jar: x",
			"Cookie-Count: 3",
		} {
			checkAllRedactionPaths(t, in, in)
		}
		for _, tail := range []string{"it's", `"hi"`} {
			for _, n := range []int{1500, 6000} {
				in := strings.Repeat("cookie: Monster ", n) + tail
				checkLongRedactionPaths(t, in, in)
			}
		}
		manyCookies := strings.Repeat("cookie: Monster ", 50)
		for _, tt := range []struct{ in, want, secret string }{
			{"Cookie: sid=synthetic; preference=light", "Cookie: [REDACTED]", "synthetic"},
			{"cookie: sid=SYNTHETIC_COOKIE_2203", "cookie: [REDACTED]", "SYNTHETIC_COOKIE_2203"},
			{"The cookie: sid=SYNTHETIC_COOKIE_2204 was set.", "The cookie: [REDACTED]", "SYNTHETIC_COOKIE_2204"},
			{"Set-Cookie: session=SYNTHETIC_COOKIE_2205; Path=/", "Set-Cookie: [REDACTED]", "SYNTHETIC_COOKIE_2205"},
			{"Cookie: sid=part1;\n  part2=SYNTHETIC_COOKIE_2206\nAccept: */*", "Cookie: [REDACTED]\nAccept: */*", "SYNTHETIC_COOKIE_2206"},
			// A quoted value, or a value in a quote that runs over lines, is
			// a header value whatever it holds.
			{"Cookie: \"Monster SYNTHETIC_COOKIE_2207\"", "Cookie: [REDACTED]", "SYNTHETIC_COOKIE_2207"},
			{"Cookie: 'Monster SYNTHETIC_COOKIE_2209'", "Cookie: [REDACTED]", "SYNTHETIC_COOKIE_2209"},
			{"curl -H 'Cookie: Monster\nSYNTHETIC_COOKIE_2208'\necho ok", "curl -H 'Cookie: [REDACTED]'\necho ok", "SYNTHETIC_COOKIE_2208"},
			{"cookie=SYNTHETIC_COOKIE_2210", "cookie=[REDACTED]", "SYNTHETIC_COOKIE_2210"},
			// Cookie prose before another credential on the same line does not
			// mask that credential or become redacted on a second pass.
			{"The cookie: pre\"Bearer \"secret456\"tail was here", "The cookie: pre\"Bearer [REDACTED] was here", "secret456"},
			{"The cookie: pre\"Bearer \"secret456\" tail was here", "The cookie: pre\"Bearer [REDACTED] tail was here", "secret456"},
			{"The cookie: Monster and Bearer secret789 were here.", "The cookie: Monster and Bearer [REDACTED] were here.", "secret789"},
			{"Cookie: Monster Bearer secret789", "Cookie: Monster Bearer [REDACTED]", "secret789"},
			{`The cookie: Monster and {"api_key": "secret789"} were here.`, `The cookie: Monster and {"api_key":"[REDACTED]"} were here.`, "secret789"},
			{"The cookie: Monster; curl -H \"Authorization: Bearer abc\npart2: SECRET\"\necho KEEP_OK", "The cookie: Monster; curl -H \"Authorization: [REDACTED]\"\necho KEEP_OK", "SECRET"},
			{manyCookies + "Cookie: sid=secret2211", manyCookies + "Cookie: [REDACTED]", "secret2211"},
			{manyCookies + `"Cookie: sid=secret2211"`, "cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster cookie: Monster \"Cookie: [REDACTED]\"", "secret2211"},
			{manyCookies + `Cookie: "secret2212"`, manyCookies + "Cookie: [REDACTED]", "secret2212"},
			{manyCookies + "Authorization: Bearer secret2213", manyCookies + "Authorization: [REDACTED]", "secret2213"},
			{manyCookies + "Bearer secret2214", manyCookies + "Bearer [REDACTED]", "secret2214"},
			{manyCookies + "Basic dXNlcjpwYXNz", manyCookies + "Basic [REDACTED]", "dXNlcjpwYXNz"},
			{manyCookies + "password=secret2215", manyCookies + "password=[REDACTED]", "secret2215"},
		} {
			checkAllRedactionPaths(t, tt.in, tt.want)
			checkNoLeak(t, tt.in, tt.secret)
		}
		// Structured Cookie and Set-Cookie values are redacted by key,
		// whatever they hold.
		for _, key := range []string{"Cookie", "cookie", "Set-Cookie", "set_cookie"} {
			checkRedactedMap(t, key, "Monster")
		}
		for _, in := range []string{`{"cookie": "Monster"}`, `{"Set-Cookie": "Monster"}`} {
			checkNoLeak(t, in, "Monster")
		}
	})

	t.Run("linear scaling", func(t *testing.T) {
		scrub := func(in string) { _ = scrubTextPatterns(in) }
		for _, tt := range []struct {
			name string
			gen  func(n int) string
			run  func(in string)
			n    int
		}{
			{"whitespace-folded headers", func(n int) string {
				return strings.Repeat("Authorization: Bearer abc\r\n \r\n\tSYNTHETIC_CANARY_82517\r\n", n)
			}, scrub, 3200},
			{"placeholder-folded headers", func(n int) string {
				return strings.Repeat("Authorization: [REDACTED]\r\n\t\r\n\tSYNTHETIC_CANARY_82517\r\n", n)
			}, scrub, 3200},
			{"headers in prior-line quotes", func(n int) string {
				return strings.Repeat("'\nAuthorization: Bearer abc\nsyntheticlookback'\n", n)
			}, scrub, 3200},
			{"headers in prior-line quotes through SmartTruncate", func(n int) string {
				return strings.Repeat("'\nAuthorization: Bearer abc\nsyntheticlookback'\n", n)
			}, func(in string) { _, _, _ = SmartTruncate(in, 1<<20) }, 3200},
			{"headers after an unclosed quote", func(n int) string {
				return "'" + strings.Repeat("\nAuthorization: Bearer abc", n)
			}, scrub, 3200},
			{"headers in a quote that opens on a header line", func(n int) string {
				return "'X-Pad: x" + strings.Repeat("\nAuthorization: Bearer abc", n) + "\nsyntheticlookback'\necho ok"
			}, scrub, 3200},
			{"header lines between a quote and its header", func(n int) string {
				return "'" + strings.Repeat("\nX-Pad: value", n) + "\nAuthorization: Bearer abc\nsyntheticlookback'\necho ok"
			}, scrub, 12800},
			{"headers after escaped-quote lines", func(n int) string {
				return strings.Repeat("\\\"\nAuthorization: Bearer abc\n", n)
			}, scrub, 3200},
			{"headers after double-quote lines", func(n int) string {
				return strings.Repeat("\"\nAuthorization: Bearer abc\nsynthetic\"\n", n)
			}, scrub, 3200},
			{"continued shell assignments", func(n int) string {
				return strings.Repeat("export X=1 \\\n password=users' violetfern's changed\n", n)
			}, scrub, 3200},
			{"assignments after long runs of assignments", func(n int) string {
				return strings.Repeat("export "+strings.Repeat("X=1 ", 70)+"password=users' violetfern's changed\n", n)
			}, scrub, 400},
			{"possessive chains", func(n int) string {
				return strings.Repeat("password=users' synthlow's changed the password=it's Bob's ", n)
			}, scrub, 2000},
			{"cookie prose on one line", func(n int) string { return strings.Repeat("cookie: Monster ", n) }, scrub, 4000},
			{"cookie prose on one line with apostrophe", func(n int) string { return strings.Repeat("cookie: Monster ", n) + "it's" }, func(in string) { _ = scrubErrorTokens(in) }, 6000},
			{"cookie prose on one line with quotes", func(n int) string { return strings.Repeat("cookie: Monster ", n) + `"hi"` }, func(in string) { _ = scrubErrorTokens(in) }, 6000},
			{"cookie prose on lines", func(n int) string { return strings.Repeat("The cookie: Monster was here.\n", n) }, scrub, 4000},
		} {
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
				if strings.HasPrefix(tt.name, "cookie prose on one line with") {
					if dl > 6*ds && dl > 50*time.Millisecond {
						t.Errorf("scrubbing %d repeats took %v and %d repeats took %v, want < 6x ratio", tt.n/4, ds, tt.n, dl)
					}
				}
				if dl > 10*ds && dl > 100*time.Millisecond {
					t.Errorf("scrubbing %d repeats took %v and %d repeats took %v, which is more than linear", tt.n/4, ds, tt.n, dl)
				}
				if dl > 4*time.Second {
					t.Errorf("scrubbing %d repeats took %v, want well under 4s", tt.n, dl)
				}
			})
		}
	})
}

// redactAuthHeadersFreshScanner is a reference implementation of
// redactAuthHeaders that constructs a fresh headerQuoteScanner for each header
// match, used to verify that headerQuoteScanner's carried state is bit-for-bit
// equivalent to a from-scratch scan on every match.
func redactAuthHeadersFreshScanner(s string) string {
	var b strings.Builder
	last := 0
	curLineStart, curLineEnd, curIndentLen := -1, -1, 0
	for pos := 0; ; {
		loc := findAtKeys(s, pos, authHeaderKeys, authHeaderPrefixAtRe)
		if loc == nil {
			break
		}
		quotes := newHeaderQuoteScanner(s)
		outer, inner := quotes.openBefore(loc[0], loc[1])
		if loc[1] > curLineEnd {
			curLineStart = strings.LastIndexAny(s[:loc[0]], "\r\n") + 1
			curLineEnd = len(s)
			if nl := strings.IndexAny(s[loc[1]:], "\r\n"); nl >= 0 {
				curLineEnd = loc[1] + nl
			}
			curIndentLen = 0
			for curLineStart+curIndentLen < loc[0] && (s[curLineStart+curIndentLen] == ' ' || s[curLineStart+curIndentLen] == '\t') {
				curIndentLen++
			}
		}
		lineStart, lineEnd, indentLen := curLineStart, curLineEnd, curIndentLen
		canFold := outer == "" && inner == "" && lineStart+indentLen == loc[0]
		if isCookieHeaderKey(s[loc[2]:loc[3]]) && cookieLineProseAllowed(s, loc[1], lineEnd, outer, inner, quotes) && !hasFoldedHeaderContinuation(s, lineEnd, canFold, indentLen) {
			if sub := findAtKeys(s[loc[1]:lineEnd], 0, authHeaderKeys, authHeaderPrefixAtRe); sub != nil {
				pos = loc[1] + sub[0]
			} else {
				pos = lineEnd
			}
			continue
		}
		end, closedQuotes := authHeaderValueEnd(s, loc[1], lineEnd, outer, inner, canFold, indentLen, quotes)
		repl := s[loc[2]:loc[3]] + ": " + redactedValue
		if outer != "" || inner != "" {
			cutEnd := closingQuotesStart(s, loc[1], end, outer, inner)
			if closedQuotes != "" && cutEnd == end {
				repl += closedQuotes
			}
			end = cutEnd
		}
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func TestRound23ReviewRegressions(t *testing.T) {
	t.Run("placeholder-first multiline quoted headers", func(t *testing.T) {
		const secret = "synthcanary"
		for _, hdr := range []string{
			"Authorization",
			"Proxy-Authorization",
			"Api-Key",
			"X-Api-Key",
			"X-Goog-Api-Key",
			"Api-Key-Secret",
			"Api-Key-String",
			"Cookie",
			"Set-Cookie",
		} {
			in := "curl -H '" + hdr + ": [REDACTED]\n\t" + secret + "'\necho KEEP"
			want := "curl -H '" + hdr + ": [REDACTED]'\necho KEEP"
			checkAllRedactionPaths(t, in, want)
			checkNoLeak(t, in, secret)
		}
		for _, hdr := range []string{"Authorization", "Cookie", "X-Api-Key"} {
			for _, nl := range []string{"\n", "\r\n", "\r"} {
				for _, cont := range []string{"\t" + secret, "  " + secret, secret, "part2: " + secret} {
					for _, q := range []struct{ open, close string }{
						{"curl -H '", "'"},
						{`curl -H "`, `"`},
						{`curl -H \'`, `\'`},
						{`curl -H \"`, `\"`},
						{"'X-Pad: pad" + nl, "'"},
						{`curl -H "X-Pad: pad` + nl, `"`},
					} {
						for _, tail := range []string{nl + "echo KEEP", nl + "echo 'ok'", nl + "Accept: text/plain"} {
							in := q.open + hdr + ": [REDACTED]" + nl + cont + q.close + tail
							want := q.open + hdr + ": [REDACTED]" + q.close + tail
							checkLongRedactionPaths(t, in, want, secret)
						}
					}
				}
			}
		}
		// Leading double quote on line 0 is scrubbed by scrubTextPatterns while
		// JSON-aware paths replace the unclosed top-level quote with placeholder.
		leadDQ := "\"X-Pad: pad\nAuthorization: [REDACTED]\n\t" + secret + "\"\necho KEEP"
		if got, want := scrubTextPatterns(leadDQ), "\"X-Pad: pad\nAuthorization: [REDACTED]\"\necho KEEP"; got != want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", leadDQ, got, want)
		}
		checkNoLeak(t, leadDQ, secret)
		pad5000 := "'X-Pad: " + strings.Repeat("p", 5000) + "\nAuthorization: [REDACTED]\n" + secret + "'\necho KEEP"
		wantPad5000 := "'X-Pad: " + strings.Repeat("p", 5000) + "\nAuthorization: [REDACTED]'\necho KEEP"
		checkLongRedactionPaths(t, pad5000, wantPad5000, secret)

		// Unclosed-quote placeholder outputs from an earlier scrub remain idempotent.
		for _, idemp := range []string{
			"curl -H 'Cookie: [REDACTED]\necho ok",
			"curl -H 'Authorization: [REDACTED]\necho KEEP",
			"curl -H 'Authorization: [REDACTED]\necho 'ok'",
			`curl -H "Authorization: [REDACTED]` + "\necho \"ok\"",
		} {
			checkAllRedactionPaths(t, idemp, idemp)
		}
	})

	t.Run("shell assignment lookbehind and quoted assignments", func(t *testing.T) {
		const secret = "synthcanary"
		for _, n := range []int{10, 255, 256, 257, 300, 1000, 5000} {
			pad := strings.Repeat("a", n)
			for _, closing := range []string{"synthcanary's changed", "synthcanary wasn't changed", "synthcanary it's changed"} {
				for _, tc := range []struct{ prefix, wantPrefix string }{
					{"export DB_PASSWORD=" + pad + ":password=users' ", "export DB_PASSWORD=[REDACTED] changed"},
					{"export db_password=" + pad + ":password=users' ", "export db_password=[REDACTED] changed"},
					{"DB_PASSWORD=" + pad + ":password=users' ", "DB_PASSWORD=[REDACTED] changed"},
					{"db_password=" + pad + ":password=users' ", "db_password=[REDACTED] changed"},
					{"export password=" + pad + ":password=users' ", "export password=[REDACTED] changed"},
					{"export \"DB_PASSWORD\"=" + pad + ":password=users' ", "export \"DB_PASSWORD\"=[REDACTED] changed"},
				} {
					checkLongRedactionPaths(t, tc.prefix+closing, tc.wantPrefix, secret)
				}
			}
			// Long non-assignment prefix before :password= preserves prose contraction and following metadata.
			proseTail := " wasn't changed; prompt_token_count=10 next_page_token=c1"
			checkLongRedactionPaths(t, pad+":password=users'"+proseTail, pad+":password=[REDACTED]"+proseTail)
		}

		for _, k := range []int{1, 10, 35, 40, 60, 100} {
			for _, arg := range []string{
				"A='foo bar' ",
				`A="foo bar" `,
				`A=\'foo bar\' `,
				`A=\"foo bar\" `,
				"--opt='foo bar' ",
				`--opt="foo bar" `,
			} {
				prefix := "export " + strings.Repeat(arg, k)
				for _, closing := range []string{"synthcanary's changed", "synthcanary wasn't changed"} {
					checkLongRedactionPaths(t, prefix+"password=users' "+closing, prefix+"password=[REDACTED] changed", secret)
				}
			}
		}

		longQuotedPrefix := "export A='" + strings.Repeat("foo bar ", 80) + "' "
		checkLongRedactionPaths(t, longQuotedPrefix+"password=users' synthcanary's changed", longQuotedPrefix+"password=[REDACTED] changed", secret)

		// Outer quoted command followed by top-level non-assignment key-value pair:
		// the inner password= inside msg="..." is in an export command (redacts synthcanary),
		// while the outer token= after msg="..." is not in a declaration (preserves wasn't changed).
		outerAndInner := `msg="export ` + strings.Repeat("A='foo bar' ", 40) + `password=users' synthcanary's changed" token=users' wasn't changed`
		wantOuterAndInner := `msg="export ` + strings.Repeat("A='foo bar' ", 40) + `password=[REDACTED] changed" token=[REDACTED] wasn't changed`
		checkLongRedactionPaths(t, outerAndInner, wantOuterAndInner, secret)
	})

	t.Run("same-line header quote scanner state", func(t *testing.T) {
		for _, tc := range []struct {
			in     string
			want   string
			secret string
		}{
			{"\"\nCookie:r'Cookie:", "\"\nCookie:r'Cookie: [REDACTED]", ""},
			{"\"\nCookie:r'Cookie: Monster\nKEEP_LINE_ONE\nKEEP_LINE_TWO'\nKEEP_LINE_THREE", "\"\nCookie:r'Cookie: [REDACTED]'\nKEEP_LINE_THREE", ""},
			{"\"\nCookie:r'Cookie: sid=synthcanary\nKEEP_LINE_ONE", "\"\nCookie:r'Cookie: [REDACTED]", "synthcanary"},
			{"'\nCookie:r\"Cookie: Monster\nKEEP_LINE_ONE\nKEEP_LINE_TWO\"\nKEEP_LINE_THREE", "'\nCookie:r\"Cookie: [REDACTED]\"\nKEEP_LINE_THREE", ""},
			{"'\nCookie:r\"Cookie: sid=synthcanary\nKEEP_LINE_ONE", "'\nCookie:r\"Cookie: [REDACTED]", "synthcanary"},
		} {
			if got := scrubTextPatterns(tc.in); got != tc.want {
				t.Errorf("scrubTextPatterns(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if got := redactAuthHeaders(tc.in); got != redactAuthHeadersFreshScanner(tc.in) {
				t.Errorf("redactAuthHeaders(%q) = %q != fresh scanner %q", tc.in, got, redactAuthHeadersFreshScanner(tc.in))
			}
			if tc.secret != "" {
				checkNoLeak(t, tc.in, tc.secret)
			}
		}

		// Differential fuzz/token-soup test: sequential headerQuoteScanner vs fresh per-header scanner.
		tokens := []string{
			"\"", "'", `\"`, "\n", "\r\n", " ", "Cookie:", "Cookie: Monster ", "Cookie: sid=x ",
			"r'", `r"`, "Authorization: Bearer x", "KEEP_LINE",
		}
		var state uint64 = 0x9e3779b97f4a7c15
		for i := range 2000 {
			var sb strings.Builder
			for range 12 {
				state ^= state << 13
				state ^= state >> 7
				state ^= state << 17
				sb.WriteString(tokens[state%uint64(len(tokens))])
			}
			in := sb.String()
			gotSeq := redactAuthHeaders(in)
			gotFresh := redactAuthHeadersFreshScanner(in)
			if gotSeq != gotFresh {
				t.Fatalf("case %d: redactAuthHeaders(%q) = %q, want fresh-scanner result %q", i, in, gotSeq, gotFresh)
			}
		}
	})

	t.Run("concurrent race stress", func(t *testing.T) {
		inputs := []struct {
			in     string
			want   string
			secret string
		}{
			{
				"export DB_PASSWORD=" + strings.Repeat("a", 300) + ":password=users' synthcanary's changed",
				"export DB_PASSWORD=[REDACTED] changed",
				"synthcanary",
			},
			{
				"export " + strings.Repeat("A='foo bar' ", 40) + "password=users' synthcanary's changed",
				"export " + strings.Repeat("A='foo bar' ", 40) + "password=[REDACTED] changed",
				"synthcanary",
			},
			{
				"curl -H 'Authorization: [REDACTED]\n\tsynthcanary'\necho KEEP",
				"curl -H 'Authorization: [REDACTED]'\necho KEEP",
				"synthcanary",
			},
			{
				strings.Repeat("a", 300) + ":password=users' wasn't changed",
				strings.Repeat("a", 300) + ":password=[REDACTED] wasn't changed",
				"",
			},
		}
		done := make(chan struct{})
		for range 8 {
			go func() {
				defer func() { done <- struct{}{} }()
				for range 25 {
					for _, tc := range inputs {
						if got := scrubTextPatterns(tc.in); got != tc.want {
							t.Errorf("concurrent scrubTextPatterns() = %q, want %q", elideText(got), elideText(tc.want))
							return
						}
						if b, _, err := SmartTruncate(tc.in, 4096); err != nil || (tc.secret != "" && strings.Contains(string(b), tc.secret)) {
							t.Errorf("concurrent SmartTruncate() = (%s, %v)", elideText(string(b)), err)
							return
						}
					}
				}
			}()
		}
		for range 8 {
			<-done
		}
	})

	t.Run("gate2 p1-1 multi-pass idempotency on nested outer+inner quotes", func(t *testing.T) {
		// The quote that closes the header string is kept after the
		// placeholder, so that scrubbing the result again keeps the lines
		// after it.
		for _, tc := range []struct{ in, want string }{
			{
				"curl -H 'Authorization: Bearer secret'\necho 'hello\nworld'",
				"curl -H 'Authorization: [REDACTED]'\necho 'hello\nworld'",
			},
			{
				"curl -H \"Authorization: 'Bearer secret'\n\tsynthcanary'\"\necho KEEP",
				"curl -H \"Authorization: [REDACTED]\"\necho KEEP",
			},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
		}
	})

	t.Run("gate2 p1-2 trailing backslash in placeholderQuoteCloses", func(t *testing.T) {
		const secret = "synthcanary"
		// A backslash is literal inside single quotes, so the quote after
		// it closes the header string, whether or not the continuation
		// line is indented.
		for _, in := range []string{
			"curl -H 'Authorization: [REDACTED]\n\tsynthcanary\\'\necho KEEP",
			"curl -H 'Authorization: [REDACTED]\nsynthcanary\\'\necho KEEP",
		} {
			if got := redactAuthHeaders(in); strings.Contains(got, secret) {
				t.Errorf("redactAuthHeaders(%q) = %q, which still holds the secret", in, got)
			}
			checkAllRedactionPaths(t, in, "curl -H 'Authorization: [REDACTED]'\necho KEEP")
		}
	})

	t.Run("gate2 p1-3 CRLF line continuation in shell assignment", func(t *testing.T) {
		const secret = "synthcanary"
		for _, tc := range []struct {
			name string
			in   string
		}{
			{
				"DB_PASSWORD with CRLF continuation",
				"DB_PASSWORD=foo\\\r\npassword:users' synthcanary's changed",
			},
			{
				"DB_PASSWORD with CRLF continuation and auth header",
				"DB_PASSWORD=foo\\\r\nbar'\r\nAuthorization: Bearer x\r\n\tsynthcanary'\r\necho KEEP",
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkNoLeak(t, tc.in, secret)
				// Verify idempotency.
				got := scrubTextPatterns(tc.in)
				if got != scrubTextPatterns(got) {
					t.Errorf("scrubTextPatterns not idempotent: scrub(scrub(%q)) = %q, scrub(%q) = %q", tc.in, scrubTextPatterns(got), tc.in, got)
				}
			})
		}
	})

	t.Run("linear scaling", func(t *testing.T) {
		scrub := func(in string) { _ = scrubTextPatterns(in) }
		for _, tt := range []struct {
			name string
			gen  func(n int) string
			n    int
		}{
			{"placeholder-first quoted multiline headers", func(n int) string {
				return strings.Repeat("curl -H 'Authorization: [REDACTED]\n\tsynthcanary'\necho KEEP\n", n)
			}, 3200},
			{"placeholder headers after unclosed quote", func(n int) string {
				return "curl -H '" + strings.Repeat("\nAuthorization: [REDACTED]", n)
			}, 3200},
			{"quoted shell assignments before password", func(n int) string {
				return "export " + strings.Repeat("A='foo bar' ", n) + "password=users' synthcanary's changed"
			}, 3200},
			{"long embedded-equals shell assignment", func(n int) string {
				return "export DB_PASSWORD=" + strings.Repeat("aaaa", n) + ":password=users' synthcanary's changed"
			}, 16000},
			{"same-line cookie prose after rejected prior quote", func(n int) string {
				return "\"\n" + strings.Repeat("Cookie:r' ", n) + "Cookie: Monster"
			}, 3200},
		} {
			t.Run(tt.name, func(t *testing.T) {
				small, large := tt.gen(tt.n/4), tt.gen(tt.n)
				scrub(small)
				best := func(in string) time.Duration {
					d := scrubTime(func() { scrub(in) })
					for range 2 {
						d = min(d, scrubTime(func() { scrub(in) }))
					}
					return d
				}
				ds, dl := best(small), best(large)
				if dl > 10*ds && dl > 100*time.Millisecond {
					t.Errorf("scrubbing %d repeats took %v and %d repeats took %v, which is more than linear", tt.n/4, ds, tt.n, dl)
				}
				if dl > 4*time.Second {
					t.Errorf("scrubbing %d repeats took %v, want well under 4s", tt.n, dl)
				}
			})
		}
	})
}

// TestRound24ReviewRegressions checks inputs from the twenty-fourth round of
// review on the paths that scrub text: values with a quoted sensitive key
// written straight after them, quoted passphrases with apostrophes, colons or
// equals signs in them, line splices in keys and values, a closing quote
// after a redacted header, cookie values that do not read as prose, an
// apostrophe in a bare value, values with a later sensitive key in them, and
// text in JSON string values.
func TestRound24ReviewRegressions(t *testing.T) {
	const secret = "synthcanary"

	t.Run("R24-2 bearer glued to sensitive quoted key", func(t *testing.T) {
		// A token or a bare value that a quoted sensitive key follows with
		// no space ends before the key's quote, and the key's value is
		// redacted too.
		for _, tc := range []struct {
			name string
			in   string
			want string
		}{
			{
				"redacted bearer + quoted password key",
				`x='Bearer [REDACTED]'"password": ZQXSEKRETN}`,
				`x='Bearer [REDACTED]'"password":"[REDACTED]"}`,
			},
			{
				"bearer abc + quoted password key",
				`x='Bearer abc'"password": ZQXSEKRETN}`,
				`x='Bearer [REDACTED]'"password":"[REDACTED]"}`,
			},
			{
				"bearer abc + quoted label key",
				`x='Bearer abc'"label": ZQXSEKRETN}`,
				`x='Bearer [REDACTED]'}`,
			},
			{
				"redacted bearer + quoted token key",
				`x='Bearer [REDACTED]'"token": ZQXSEKRETN}`,
				`x='Bearer [REDACTED]'"token":"[REDACTED]"}`,
			},
			{
				"redacted bearer + single-quoted password key",
				`x="Bearer [REDACTED]"'password': ZQXSEKRETN}`,
				`x="Bearer [REDACTED]"'password':[REDACTED]`,
			},
			{
				"bare bearer + quoted password key",
				`Bearer abc"password": ZQXSEKRETN`,
				`Bearer [REDACTED]"password":"[REDACTED]"`,
			},
			{
				"bare bearer + single-quoted password key",
				`Bearer abc'password': ZQXSEKRETN`,
				`Bearer [REDACTED]'password':[REDACTED]`,
			},
			{
				"bare bearer + quoted password key and spaced equals sign",
				`Bearer abc"password" = ZQXSEKRETN`,
				`Bearer [REDACTED]"password"=[REDACTED]`,
			},
			{
				"bare token value + single-quoted password key",
				`token=abc'password': ZQXSEKRETN`,
				`token=[REDACTED]`,
			},
			{
				"bare password value + single-quoted token key",
				`password=abc'token': ZQXSEKRETN`,
				`password=[REDACTED]`,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkAllRedactionPaths(t, tc.in, tc.want)
			})
		}
	})

	t.Run("R24-3 glued prefix before env assignment", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			in   string
		}{
			{
				"note prefix before DB_PASSWORD",
				"note:DB_PASSWORD=users' synthcanary's changed",
			},
			{
				"cmd prefix before db_password",
				"cmd:db_password=users' synthcanary's changed",
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkNoLeak(t, tc.in, secret)
			})
		}
	})

	t.Run("R24-4 interior apostrophe in quoted passphrase", func(t *testing.T) {
		// A quoted passphrase is redacted up to its closing quote, whatever
		// apostrophes, colons or equals signs it holds, also when it is
		// longer than 256 bytes.
		for _, tc := range []struct {
			name string
			in   string
		}{
			{"password equals it's SECRET", "password='it's " + secret + "' user=x"},
			{"password equals Bob's SECRET", "password='Bob's " + secret + "' user=x"},
			{"password equals I'm SECRET", "password='I'm " + secret + "' user=x"},
			{"password equals O'Brien SECRET", "password='O'Brien " + secret + "' user=x"},
			{"password equals JSON's SECRET", "password='JSON's " + secret + "' user=x"},
			{"time in the passphrase", "password='it's 10:30 " + secret + "' user=x"},
			{"two interior apostrophes", "password='it's a friend's " + secret + "' user=x"},
			{"interior apostrophes in two words", "password='it's Bob's " + secret + "' user=x"},
			{"equals sign in the passphrase", "password='it's a=b " + secret + "' user=x"},
			{"colon in the passphrase", "password='it's a:b " + secret + "' user=x"},
			{"passphrase of 300 bytes", "password='it's " + strings.Repeat("pad ", 70) + secret + "' user=x"},
			{"passphrase of 420 bytes", "password='it's " + strings.Repeat("pad ", 100) + secret + "' user=x"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkAllRedactionPaths(t, tc.in, "password='[REDACTED]' user=x")
			})
		}
		// The shell-assignment path (export DB_PASSWORD=pre' it's SECRET'; echo ok)
		// goes through findQuotedOrShellSegment and joinedQuoteEnd, where the
		// interior apostrophe fix is now wired in (see interiorApostropheEnd).
		t.Run("export DB_PASSWORD pre it's SECRET", func(t *testing.T) {
			checkAllRedactionPaths(t, "export DB_PASSWORD=pre' it's "+secret+"'; echo ok", "export DB_PASSWORD=[REDACTED]; echo ok")
		})
		t.Run("json password with escaped apostrophes", func(t *testing.T) {
			in := `{"password": "\'it\'s ` + secret + `'", "user": "x"}`
			checkNoLeak(t, in, secret)
		})
	})

	t.Run("R24-5 unquoted line splices in key=value text", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			in   string
			want string
		}{
			{
				"DB_PASSWORD equals pre LF SECRET value continuation",
				"DB_PASSWORD=pre\\\n" + secret,
				"DB_PASSWORD=[REDACTED]",
			},
			{
				"DB_PASSWORD equals pre CRLF SECRET value continuation",
				"DB_PASSWORD=pre\\\r\n" + secret,
				"DB_PASSWORD=[REDACTED]",
			},
			{
				"DB_PASS LF WORD equals SECRET (spliced key name)",
				"DB_PASS\\\nWORD=" + secret,
				"DB_PASS\\\nWORD=[REDACTED]",
			},
			{
				"DB_PASS CRLF WORD equals SECRET (spliced key name)",
				"DB_PASS\\\r\nWORD=" + secret,
				"DB_PASS\\\r\nWORD=[REDACTED]",
			},
			{
				"DB_PASSWORD LF equals SECRET (spliced before delimiter)",
				"DB_PASSWORD\\\n=" + secret,
				"DB_PASSWORD\\\n=[REDACTED]",
			},
			{
				"DB_PASSWORD CRLF equals SECRET (spliced before delimiter)",
				"DB_PASSWORD\\\r\n=" + secret,
				"DB_PASSWORD\\\r\n=[REDACTED]",
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkAllRedactionPaths(t, tc.in, tc.want)
			})
		}
		// A map key or a JSON object key with a line splice in it is read
		// as the name that the splice joins.
		for _, key := range []string{"DB_PASS\\\nWORD", "DB_PASS\\\r\nWORD"} {
			checkRedactedMap(t, key, secret)
			b, err := json.Marshal(map[string]string{key: secret})
			if err != nil {
				t.Fatal(err)
			}
			if got, _ := sanitizeRawText(string(b)); strings.Contains(got, secret) {
				t.Errorf("sanitizeRawText(%s) = %q, which still holds the secret", b, got)
			}
		}
	})

	t.Run("closing quote after a redacted header is kept", func(t *testing.T) {
		// The quote that closes the string around a header value is kept
		// after the placeholder, so that scrubbing the result again reads
		// the lines after it as the first scrub did and keeps them.
		for _, tc := range []struct{ in, want string }{
			{
				"'X-Pad: pad\nAuthorization: Bearer abc'\n\tKEEP_TAIL",
				"'X-Pad: pad\nAuthorization: [REDACTED]'\n\tKEEP_TAIL",
			},
			{
				"'X-Pad: pad\nAuthorization: Bearer abc'\n\tKEEP_TAIL\n",
				"'X-Pad: pad\nAuthorization: [REDACTED]'\n\tKEEP_TAIL\n",
			},
			{
				"curl -H 'Authorization: Bearer secret\n\t" + secret + "'\necho 'hello\nworld'",
				"curl -H 'Authorization: [REDACTED]'\necho 'hello\nworld'",
			},
			// Each path also keeps the text when it ends in a backslash
			// and an "n".
			{
				"'X-Pad: pad\nAuthorization: Bearer abc'\n\tKEEP_TAIL\\n",
				"'X-Pad: pad\nAuthorization: [REDACTED]'\n\tKEEP_TAIL\\n",
			},
			{
				"curl -H 'Authorization: Bearer secret\n\t" + secret + "'\necho 'hello\nworld'\\n",
				"curl -H 'Authorization: [REDACTED]'\necho 'hello\nworld'\\n",
			},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
		}
	})

	t.Run("cookie values that do not read as prose are redacted", func(t *testing.T) {
		for _, tc := range []struct{ in, want string }{
			{"Cookie: 5f2b9c0e7d1a44e3", "Cookie: [REDACTED]"},
			{"Set-Cookie: 9f8e7d6c5b4a; HttpOnly; Secure", "Set-Cookie: [REDACTED]"},
			{"  cookie: 5f2b9c0e7d1a", "  cookie: [REDACTED]"},
			{"Cookie: app:sid=" + secret, "Cookie: [REDACTED]"},
			{"Set-Cookie: Secure, sid=" + secret + "; Path=/", "Set-Cookie: [REDACTED]"},
			{"Cookie: Monster, sid=" + secret, "Cookie: [REDACTED]"},
			// A quote on an earlier line that nothing closes does not hide
			// the quoted header after cookie prose.
			{
				"'\nCookie: Monster; curl -H \"Authorization: Bearer abc\npart2: " + secret + "\"",
				"'\nCookie: Monster; curl -H \"Authorization: [REDACTED]\"",
			},
			{"The cookie: Monster was here.", "The cookie: Monster was here."},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
		}
		// Only the text patterns read a struct dump as pairs (see
		// checkNoLeak).
		checkNoLeak(t, "{Host:example.com Cookie:5f2b9c0e User:bob}", "5f2b9c0e")
	})

	t.Run("apostrophe in a bare value does not take the pairs after it", func(t *testing.T) {
		for _, tc := range []struct{ in, want string }{
			{"password=abc's prompt_tokens=10 next_page_token=c1", "password=[REDACTED] prompt_tokens=10 next_page_token=c1"},
			{"password=pre's token=" + secret + " tail", "password=[REDACTED] token=[REDACTED] tail"},
			{"password=pre's a:b " + secret, "password=[REDACTED]"},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
		}
	})

	t.Run("value that takes a later sensitive key takes its value too", func(t *testing.T) {
		for _, tc := range []struct{ in, want string }{
			{"token=O'Brien password='" + secret + " VALUE' rest", "token=[REDACTED] rest"},
			{`DB_PASSWORD="a"prompt_tokens="10"token: ` + secret, "DB_PASSWORD=[REDACTED] [REDACTED]"},
			{`password="benign"user="bob"token: ` + secret, "password=[REDACTED] [REDACTED]"},
			{`credentials: token = "` + secret + `"`, `credentials: token="[REDACTED]"`},
		} {
			checkAllRedactionPaths(t, tc.in, tc.want)
		}
	})

	t.Run("text in a JSON string value is scrubbed", func(t *testing.T) {
		for _, tc := range []struct{ in, want string }{
			{"password='it's a friend's " + secret + "' user=x", "password='[REDACTED]' user=x"},
			{"DB_PASS\\\nWORD=" + secret, "DB_PASS\\\nWORD=[REDACTED]"},
			{"DB_PASS\\\r\nWORD=" + secret, "DB_PASS\\\r\nWORD=[REDACTED]"},
		} {
			in, err := json.Marshal(map[string]string{"note": tc.in})
			if err != nil {
				t.Fatal(err)
			}
			want, err := json.Marshal(map[string]string{"note": tc.want})
			if err != nil {
				t.Fatal(err)
			}
			if got, _ := sanitizeRawText(string(in)); got != string(want) {
				t.Errorf("sanitizeRawText(%s) = %s, want %s", in, got, want)
			}
			b, _, err := SmartTruncate(string(in), 4096)
			var got string
			if err != nil || json.Unmarshal(b, &got) != nil || got != string(want) {
				t.Errorf("SmartTruncate(%q) = (%s, %v), want %q", in, b, err, want)
			}
			if b, _, err := SmartTruncate(map[string]string{"note": tc.in}, 4096); err != nil || string(b) != string(want) {
				t.Errorf("SmartTruncate(map[note:%q]) = (%s, %v), want %s", tc.in, b, err, want)
			}
		}
	})

	t.Run("R24-1 bearer token linear scaling", func(t *testing.T) {
		fragment := `msg="Bearer DB_PASSWORD=users' synthcanary's changed"; `
		small := strings.Repeat(fragment, 1000)
		large := strings.Repeat(fragment, 4000)
		// Warm up.
		_ = scrubTextPatterns(small)
		_, _ = sanitizeRawText(small)
		_ = redactErrorMessage(small)
		// Time small and large across all three entry points, taking the
		// best of 2 runs each.
		bestOf := func(in string, fn func(string)) func() time.Duration {
			return func() time.Duration {
				d := scrubTime(func() { fn(in) })
				for range 2 {
					d = min(d, scrubTime(func() { fn(in) }))
				}
				return d
			}
		}
		for _, p := range []struct {
			name string
			fn   func(string)
		}{
			{"scrubTextPatterns", func(s string) { _ = scrubTextPatterns(s) }},
			{"sanitizeRawText", func(s string) { _, _ = sanitizeRawText(s) }},
			{"redactErrorMessage", func(s string) { _ = redactErrorMessage(s) }},
		} {
			s, l := bestOf(small, p.fn)(), bestOf(large, p.fn)()
			// The primary check: scaling should be roughly linear (4x data → ≤10x time).
			if l > 10*s && l > 100*time.Millisecond {
				t.Errorf("%s: 1000 fragments took %v, 4000 took %v, more than 10x linear", p.name, s, l)
			}
			// Secondary check: absolute time should be reasonable.
			if l > 4*time.Second {
				t.Errorf("%s: 4000 bearer fragments took %v, want < 4s", p.name, l)
			}
		}
	})
}

// TestRedactionKeepsWhatMainRedacts checks that the plugin keeps out of the
// content_parts and error_message columns every secret that the baseline pass
// redacts, for texts on which the passes once read a quote, a header or a
// value differently from it and wrote the secret.
func TestRedactionKeepsWhatMainRedacts(t *testing.T) {
	for in, secret := range map[string]string{
		`login failed: {"password": " correct horse "battery"}`:          "correct horse",
		`failed: "{"password":"   Passw0rd","user":"bob"}"`:              "Passw0rd",
		`msg: "got Bearer " SECRET1"x`:                                   "SECRET1",
		`export DB_PASSWORD='a "password": "b Bearer '" users' SECRET2"`: "SECRET2",
		`export DB_PASSWORD=pre"one two "token=" SECRET3"suffix`:         "SECRET3",
		`password="benign"user="bob"token: SECRET4`:                      "SECRET4",
		`id="1" temp:ok password="= word1 SECRET5"`:                      "SECRET5",
		`token=O'Brien password='SECRET6 VALUE6' rest`:                   "VALUE6",
		`got {"auth": "Bearer abc"}+{"password": "SECRET7"}`:             "SECRET7",
		`x='Bearer abc'|password: SECRET8`:                               "SECRET8",
		"failed: {\"credentials\": {\n  \"hdr\": \"Authorization: Bearer abc\nx\",\n  \"note\": \"} b\",\n  \"AccessKeyId\": \"SECRET9\"\n}}": "SECRET9",
		"req=\"Authorization: Bearer abc\nHost: example.com\n\npassword=\"line1\nSECRET10\"":                                                  "SECRET10",
		"data=\"YWJj\nZGVm=\"\nAuthorization: Bearer abc\necho \"hello world\"\npassword=\"line1\nSECRET11\"":                                 "SECRET11",
		"Error: api_key_secret: x, credentials: {\n  \"AccessKeyId\": \"SECRET12\"\n}":                                                        "SECRET12",
		`x='Bearer abc'"label": password="part1 SECRET13"`:                                                                                    "SECRET13",
		`x="Bearer ""password": SECRET14`:                  "SECRET14",
		`DB_PASSWORD="a"prompt_tokens="10"token: SECRET15`: "SECRET15",
	} {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: in}}}))
		errs.add(p.OnModelErrorCallback()(mCtx, &model.LLMRequest{}, errors.New(in)))
		errs.check(t)
		p.AfterRunCallback()(mCtx)
		rows := collectRows(t, requests, untilInvocationEnd)
		parts, _ := onlyRowOfType(t, rows, EventUserMessageReceived)["content_parts"].([]map[string]any)
		text, _ := parts[0]["text"].(string)
		msg, _ := onlyRowOfType(t, rows, EventLLMError)["error_message"].(string)
		if strings.Contains(text, secret) || strings.Contains(msg, secret) {
			t.Errorf("%q: text %q, error_message %q", in, text, msg)
		}
	}
}

// TestScrubbingStaysLinear checks that sanitizeRawText takes time linear in
// the length of texts that repeat shapes on which the passes once read the
// rest of the text again from each repeat.
func TestScrubbingStaysLinear(t *testing.T) {
	for _, c := range []struct {
		prefix, unit, suffix string
		small, large         int
	}{
		{"", `'temp:a=`, " password=x", 50_000, 200_000},
		{"", `Bearer " in a, token="{a"x "" `, "", 50_000, 200_000},
		{"password=a = ", `"x": `, "", 5_000, 20_000},
		{`password="a"`, `x\\\'`, " \n \\'z", 10_000, 40_000},
		{"", "Authorization: a\\\\\\'s\n", "", 10_000, 40_000},
		{"a=it's cookie: Monster ", "password=users' synthlow's changed ", "", 100_000, 800_000},
		{"x ", `{"a":"b"pass\` + "\n" + `word:[REDACTED]token:"[REDACTED]"}`, "", 240_000, 960_000},
	} {
		var took [2]time.Duration
		for i, n := range []int{c.small, c.large} {
			in := c.prefix + strings.Repeat(c.unit, n/len(c.unit)+1)[:n] + c.suffix
			start := time.Now()
			sanitizeRawText(in)
			took[i] = time.Since(start)
		}
		if limit := 2 * time.Duration(c.large/c.small) * took[0]; took[1] > limit && took[1] > 500*time.Millisecond {
			t.Errorf("%q: %d bytes took %v, %d bytes took %v", c.unit, c.small, took[0], c.large, took[1])
		}
	}
}

// TestPlaceholderHeadersStayLinear checks that sanitizeRawText takes time
// linear in the number of redacted headers in a quoted string that a long
// line ends.
func TestPlaceholderHeadersStayLinear(t *testing.T) {
	var took [2]time.Duration
	for i, n := range []int{1_000, 4_000} {
		in := "'" + strings.Repeat("\nAuthorization: [REDACTED]", n) + "\n" + strings.Repeat("a", 50*n) + "\\'"
		start := time.Now()
		sanitizeRawText(in)
		took[i] = time.Since(start)
	}
	if took[1] > 8*took[0] && took[1] > 500*time.Millisecond {
		t.Errorf("1,000 headers took %v, 4,000 headers took %v", took[0], took[1])
	}
}

// TestScrubbingStaysLinearUnderLoad checks that sanitizeRawText stays linear
// in the length of its text when sixteen goroutines scrub at once, each its
// own copy, since every scrub reads the text with scanners of its own.
func TestScrubbingStaysLinearUnderLoad(t *testing.T) {
	var took [2]time.Duration
	for i, n := range []int{25_000, 200_000} {
		in := strings.Repeat("password=users' synthlow's changed ", n/35)
		start := time.Now()
		var wg sync.WaitGroup
		for range 16 {
			wg.Add(1)
			go func(s string) {
				defer wg.Done()
				sanitizeRawText(s)
			}(strings.Clone(in))
		}
		wg.Wait()
		took[i] = time.Since(start)
	}
	if took[1] > 16*took[0] && took[1] > 500*time.Millisecond {
		t.Errorf("16 goroutines: 25 KB each took %v, 200 KB each took %v", took[0], took[1])
	}
}

// TestScrubbingKeepsNoReferenceToItsInput checks that a scrub keeps nothing
// that holds its text reachable after it returns, so that no secret in the
// text outlives the scrub in memory, as it would in a cache of scanners that
// the package keeps.
func TestScrubbingKeepsNoReferenceToItsInput(t *testing.T) {
	text := "export DB_PASSWORD=SYNTH_RETAINED_SECRET_77 " + strings.Repeat("password=users' synthlow's changed ", 40)
	for _, p := range []struct {
		name string
		fn   func(string)
	}{
		{"scrubTextPatterns", func(s string) { _ = scrubTextPatterns(s) }},
		{"sanitizeRawText", func(s string) { _, _ = sanitizeRawText(s) }},
		{"redactErrorMessage", func(s string) { _ = redactErrorMessage(s) }},
	} {
		wp := func() weak.Pointer[byte] {
			s := strings.Clone(text)
			p.fn(s)
			return weak.Make(unsafe.StringData(s))
		}()
		for range 10 {
			if runtime.GC(); wp.Value() == nil {
				break
			}
		}
		if wp.Value() != nil {
			t.Errorf("%s keeps its text reachable after it returns", p.name)
		}
	}
}

// TestConcurrentScrubbingAgrees checks that scrubs that run at once on
// sixteen goroutines, each on its own copy of a text, write what one scrub
// alone writes for it.
func TestConcurrentScrubbingAgrees(t *testing.T) {
	const secret = "SYNTH_RETAINED_SECRET_77"
	texts := []string{
		"export DB_PASSWORD=" + secret + " " + strings.Repeat("password=users' synthlow's changed ", 40),
		"a=it's cookie: Monster " + strings.Repeat("password=users' synthlow's changed ", 40) + "token=" + secret,
		"password='it's a friend's " + secret + "' user=x",
		"curl -H 'Authorization: Bearer " + secret + "\n\tmore'\necho 'hello\nworld'",
	}
	want := make([]string, len(texts))
	for i, text := range texts {
		want[i], _ = sanitizeRawText(text)
		if strings.Contains(want[i], secret) {
			t.Fatalf("sanitizeRawText(%q) = %q, which still holds the secret", text, want[i])
		}
	}
	var wg sync.WaitGroup
	for g := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := range 4 * len(texts) {
				i := (g + k) % len(texts)
				if got, _ := sanitizeRawText(strings.Clone(texts[i])); got != want[i] {
					t.Errorf("concurrent sanitizeRawText(%q) = %q, want %q", texts[i], got, want[i])
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestBaselineScrubMatchesMain checks that baselineScrub writes what
// scrubTextPatterns wrote before the passes that read quotes, shell
// assignments, line splices and prose more closely were added (commit
// 40f132d6), for texts built at random from words that the passes read, and
// for texts on which those passes once wrote a secret that it redacts.
func TestBaselineScrubMatchesMain(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"hello=\u00e9\r\nBearer X-Api-Key: \"password\"\"note\"%5BREDACTED%5D\"note\"\u00e9", "hello=\u00e9\r\nBearer [REDACTED] [REDACTED]"},
		{"export SECRET\t: \n{: [\r\n_temp:xbasic %5BREDACTED%5D]'token''passphrase\u65e5\u672c", "export SECRET:[REDACTED]"},
		{"\n\"Authorization: \\\"token\\\"", "\n\"Authorization: [REDACTED]"},
		{"Basic authorization:Set-Cookie: 'token' \nAuthorization:  =", "Basic authorization: [REDACTED]\nAuthorization: [REDACTED]"},
		{"[REDACTED]token=abc%5BREDACTED%5D\"auth\"\u00a0passphraseSECRET[REDACTED]", "[REDACTED]token=[REDACTED]"},
		{"\"note\".\"auth\"=\\\n}\\'secret\\': password\t", "\"note\".\"auth\"=\\\n}\\'secret\\':[REDACTED]\t"},
		{"(,Bob'ssecretbearer &]O'Brien'token': credentials : ", "(,Bob'ssecretbearer &]O'Brien'token':[REDACTED] : "},
		{"\"|user:tokenX-Api-Key: app:password\\\\authorization: = ", "\"|user:tokenX-Api-Key: [REDACTED]"},
		{"\\'secret\\'Bearer bearer users'&\u2028\u65e5\u672c : Bob'sDB_PASSWORD{", "\\'secret\\'Bearer [REDACTED] users'&\u2028\u65e5\u672c : Bob'sDB_PASSWORD{"},
		{"\\_+'bearer curl -H cookie:  _dGVzdDp0ZXN0cookie: +.Authorization: echo]", "\\_+'bearer [REDACTED] -H cookie:  _dGVzdDp0ZXN0cookie: +.Authorization: [REDACTED]"},
		{"api_key \\\":temp:xhello_ :  \\\\Bearer \u00a0temp:xpassphraseit's\n\\\"\nbasic \\\\", "api_key \\\":temp:xhello_:[REDACTED] [REDACTED]'s\n\\\"\nbasic \\\\"},
		{"\\\\temp:x = \"note\"[REDACTED]:apiKeyprompt_tokens", "\\\\temp:x=\"[REDACTED]\"[REDACTED]:apiKeyprompt_tokens"},
		{"of bearer \nhello&access_tokennext_page_token%5BREDACTED%5D{", "of bearer [REDACTED]"},
		{"Set-Cookie: &(O'Brien", "Set-Cookie: &(O'Brien"},
		{"a=b\\\"token\\\"api_keytemp:xexport : ", "a=b\\\"token\\\"api_keytemp:xexport : "},
		{"next_page_token\u00e9curl -H \\\\\\\"token\\\"{{", "next_page_token\u00e9curl -H \\\\\\\"token\\\"{{"},
		{"login failed: {\"password\": \" correct horse \"battery\"}", "login failed: {\"password\":\"[REDACTED]\"battery\"}"},
		{"failed: \"{\"password\":\"   Passw0rd\",\"user\":\"bob\"}\"", "failed: \"{\"password\":\"[REDACTED]\",\"user\":\"bob\"}\""},
		{"msg: \"got Bearer \" SECRET1\"x", "msg: \"got Bearer [REDACTED]x"},
		{"export DB_PASSWORD='a \"password\": \"b Bearer '\" users' SECRET2\"", "export DB_PASSWORD='a \"password\":\"[REDACTED]\""},
		{"export DB_PASSWORD=pre\"one two \"token=\" SECRET3\"suffix", "export DB_PASSWORD=pre\"one two \"token=\"[REDACTED]\"suffix"},
		{"password=\"benign\"user=\"bob\"token: SECRET4", "password=\"[REDACTED]\"user=\"bob\"token:[REDACTED]"},
		{"id=\"1\" temp:ok password=\"= word1 SECRET5\"", "id=\"1\" temp:ok password=\"[REDACTED]\""},
		{"token=O'Brien password='SECRET6 VALUE6' rest", "token=[REDACTED] password='[REDACTED]' rest"},
		{"got {\"auth\": \"Bearer abc\"}+{\"password\": \"SECRET7\"}", "got {\"auth\": \"Bearer [REDACTED]\"}+{\"password\":\"[REDACTED]\"}"},
		{"x='Bearer abc'|password: SECRET8", "x='Bearer [REDACTED]'|password:[REDACTED]"},
		{"failed: {\"credentials\": {\n  \"hdr\": \"Authorization: Bearer abc\nx\",\n  \"note\": \"} b\",\n  \"AccessKeyId\": \"SECRET9\"\n}}", "failed: {\"credentials\":\"[REDACTED]\"}"},
		{"req=\"Authorization: Bearer abc\nHost: example.com\n\npassword=\"line1\nSECRET10\"", "req=\"Authorization: [REDACTED]\nHost: example.com\n\npassword=\"[REDACTED]\""},
		{"data=\"YWJj\nZGVm=\"\nAuthorization: Bearer abc\necho \"hello world\"\npassword=\"line1\nSECRET11\"", "data=\"YWJj\nZGVm=\"\nAuthorization: [REDACTED]\necho \"hello world\"\npassword=\"[REDACTED]\""},
		{"Error: api_key_secret: x, credentials: {\n  \"AccessKeyId\": \"SECRET12\"\n}", "Error: api_key_secret:[REDACTED], credentials:[REDACTED]"},
		{"x='Bearer abc'\"label\": password=\"part1 SECRET13\"", "x='Bearer [REDACTED]'\"label\": password=\"[REDACTED]\""},
		{"x=\"Bearer \"\"password\": SECRET14", "x=\"Bearer [REDACTED]password\":[REDACTED]"},
		{"DB_PASSWORD=\"a\"prompt_tokens=\"10\"token: SECRET15", "DB_PASSWORD=\"a\"prompt_tokens=\"10\"token:[REDACTED]"},
	} {
		if got, _ := baselineScrub(tc.in); got != tc.want {
			t.Errorf("baselineScrub(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// redactionVocabulary holds the keys, header names, quotes, delimiters,
// escapes, spaces and words that TestScrubbingOnlyAddsToBaseline builds its
// texts from. SECRET stands for a value that is checked.
var redactionVocabulary = []string{
	"password", "token", "api_key", "apiKey", "Bearer ", "bearer ", "Basic ", "basic ", "Authorization: ", "authorization:",
	"X-Api-Key: ", "temp:x", "credentials", "secret", "passphrase", "access_token", "user:token", "app:password",
	"cookie: ", "Cookie: ", "Set-Cookie: ", "DB_PASSWORD", "export ", "prompt_tokens", "next_page_token",
	`"password"`, `'token'`, `\"token\"`, `\'secret\'`, `"auth"`, `"note"`,
	"=", ":", ": ", " = ", " : ", "=", "=",
	`"`, `'`, `\"`, `\'`, `"`, `'`, `\\`, `\`,
	"{", "}", "[", "]", "(", ")",
	" ", " ", " ", "\n", "\t", "\r\n", "\\\n", "  ",
	"abc", "SECRET", "it's", "O'Brien", "users'", "Bob's", "[REDACTED]", "dGVzdDp0ZXN0", "%5BREDACTED%5D",
	"x", "of", "a b", "10:30", "a=b", "pad", "hello", "world", "echo", "curl -H ",
	",", ";", "&", ".", "|", "+", "/", "-", "_",
	"é", "\u00a0", "\u2028", "日本",
}

// TestScrubbingOnlyAddsToBaseline checks, on texts built at random from
// redactionVocabulary, that scrubTextPatterns, sanitizeRawText and
// redactErrorMessage each redact every value that the baseline pass redacts:
// a value that baselineScrub leaves out of its result is left out of theirs.
func TestScrubbingOnlyAddsToBaseline(t *testing.T) {
	r := rand.New(rand.NewPCG(24, 2019))
	for checked := 0; checked < 1500; {
		var b strings.Builder
		for range 1 + r.IntN(24) {
			b.WriteString(redactionVocabulary[r.IntN(len(redactionVocabulary))])
		}
		text := b.String()
		// Give each SECRET a value of its own, to tell which one is written.
		var values []string
		for k := 0; strings.Contains(text, "SECRET"); k++ {
			v := fmt.Sprintf("Q%dZCANARY", k)
			text = strings.Replace(text, "SECRET", v, 1)
			values = append(values, v)
		}
		if len(values) == 0 {
			continue
		}
		checked++
		base, _ := baselineScrub(text)
		raw, _ := sanitizeRawText(text)
		for _, p := range []struct{ name, got string }{
			{"scrubTextPatterns", scrubTextPatterns(text)},
			{"sanitizeRawText", raw},
			{"redactErrorMessage", redactErrorMessage(text)},
		} {
			for _, v := range values {
				if !strings.Contains(base, v) && strings.Contains(p.got, v) {
					t.Errorf("%s(%q) = %q, which holds %s, which baselineScrub redacts: %q", p.name, text, p.got, v, base)
				}
			}
		}
	}
}

// randomText returns up to maxLen strings from alphabet, picked by r.
func randomText(r *rand.Rand, alphabet []string, maxLen int) string {
	var b strings.Builder
	for range r.IntN(maxLen + 1) {
		b.WriteString(alphabet[r.IntN(len(alphabet))])
	}
	return b.String()
}

// TestQuotedValueScanMatchesRegexp checks that findQuotedValue, and
// quotedValueAt at each offset of a text, queried in order or out of order on
// one scanner, find the quoted value that quotedValueRe finds there, on
// random texts of quotes, backslashes, line breaks and other bytes.
func TestQuotedValueScanMatchesRegexp(t *testing.T) {
	alphabet := []string{`"`, `'`, `\`, `\`, "x", "\n", "é", "\xff"}
	r := rand.New(rand.NewPCG(1, 2))
	for range 100_000 {
		s := randomText(r, alphabet, 14)
		if got, want := findQuotedValue(s), quotedValueRe.FindString(s); got != want {
			t.Fatalf("findQuotedValue(%q) = %q, want %q", s, got, want)
		}
		sh := newShellAssignScanner(s)
		order := r.Perm(len(s))
		if r.IntN(2) == 0 {
			for i := range order {
				order[i] = i
			}
		}
		for _, i := range order {
			if got, want := quotedValueAt(sh, s, i), quotedValueRe.FindString(s[i:]); got != want {
				t.Fatalf("quotedValueAt(%q, %d) = %q, want %q", s, i, got, want)
			}
		}
	}
}

// TestBracketEndMatchesBracketValueEnd checks that bracketEnd finds, for each
// opening bracket of a text, the end that bracketValueEnd finds, on random
// texts of brackets, quotes and backslashes.
func TestBracketEndMatchesBracketValueEnd(t *testing.T) {
	alphabet := []string{`"`, `'`, `\`, `\`, "x", "{", "}", "[", "]"}
	r := rand.New(rand.NewPCG(3, 4))
	for range 100_000 {
		s := randomText(r, alphabet, 16)
		sh := newShellAssignScanner(s)
		for i := range len(s) {
			if s[i] != '{' && s[i] != '[' {
				continue
			}
			ge, gc := bracketEnd(sh, s, i)
			we, wc := bracketValueEnd(s, i)
			if ge != we || gc != wc {
				t.Fatalf("bracketEnd(%q, %d) = %d, %v, want %d, %v", s, i, ge, gc, we, wc)
			}
		}
	}
}

// TestLineBreakFrom checks that lineBreakFrom, queried at each offset of a
// text out of order on one scanner, finds the first line break at or after
// that offset, or the end of the text.
func TestLineBreakFrom(t *testing.T) {
	alphabet := []string{"\r", "\n", "x", "y"}
	r := rand.New(rand.NewPCG(5, 6))
	for range 50_000 {
		s := randomText(r, alphabet, 10)
		sh := newShellAssignScanner(s)
		for _, i := range r.Perm(len(s) + 1) {
			want := len(s)
			if n := strings.IndexAny(s[i:], "\r\n"); n >= 0 {
				want = i + n
			}
			if got := lineBreakFrom(sh, s, i); got != want {
				t.Fatalf("lineBreakFrom(%q, %d) = %d, want %d", s, i, got, want)
			}
		}
	}
}

// TestKeyFindersMatchRegexps checks that the searches that try anchored
// patterns only where the key names that they start with may start (see
// keyNameSet) find what a regular expression search for each pattern finds,
// unanchored and at a word boundary where the pattern reads one: in a text
// that starts at an offset of a text, as replaceMatchesFunc and
// baselineJSONKeyValues search, and in the whole text from the end of each
// match, as baselineAuthHeaders and redactAuthHeadersRecorded search. It also
// checks that the checks that keyValFinder and keyNameAt make before they
// match textKeyNameAtRe and textKeyNameCamelAtRe pass wherever those match,
// and that isBareTextKey and isWholeTextKeyName agree with the expressions
// that they match only after such a check.
// The texts are built at random from key names and parts of them, written in
// either case, with or without their separators, or with the runes that case
// folding matches with their letters, and from delimiters, quotes, escapes,
// spaces, base64 values, other bytes and invalid UTF-8.
func TestKeyFindersMatchRegexps(t *testing.T) {
	authAtRe := regexp.MustCompile(`(?i)^` + baselineAuthHeaderPattern)
	authRe := regexp.MustCompile(`(?im)\b` + baselineAuthHeaderPattern)
	bearerRe := regexp.MustCompile(`(?i)\b` + baselineBearerTokenPattern)
	basicRe := regexp.MustCompile(`(?i)\b` + baselineBasicTokenPattern)
	jsonRe := regexp.MustCompile(`(?i)` + baselineJSONKeyValTokenPattern)
	textRe := regexp.MustCompile(`(?i)\b` + baselineTextKeyValTokenPattern)
	alphabet := []string{
		"password", "PassWord", "pa\u017f\u017fword", "token", "TOKEN", "temp:", "Temp:x", "secret", "\u017fecret", "sig",
		"api_key", "api-key", "apikey", "API_\u212aEY", "x_api_key", "X-Api-Key", "xapikey", "x_goog_api_key", "access_token",
		"authorization", "Proxy-Authorization", "proxy_authorization", "bearer", "Bearer", "basic", "BASIC", "ba\u017fic",
		"cookie", "Set-Cookie", "set_cookie", "x-goog-api-key", "api_key_secret", "API-KEY-STRING", "\u017fet-coo\u212aie",
		"user:", "app:", "User:temp:", "oauth", "oauth2Config.code", ".code", "OAuth.x.code", "credentials", "signature",
		"pass", "word", "to", "ken", "sec", "ret", "api", "key", "goog", "auth", "code", "db_", "DB", "my",
		"p_assword", "pass.word", "PASS-WORD", "__", "YWxh", "Zm9v==", "a+b/c",
		"=", ":", ": ", " ", "\t", "\n", "\r", "\v", "\x00", "\u00a0", "\u2028", "\u3000",
		`"`, `'`, `\`, "[", "]", "{", "}", ",", ";", "&", ")", "_", "-", ".", "+", "/", "==",
		"a", "b", "x", "Z9", "\u00e9", "\xff", "\xc5", "\xe2\x84", "\u212a", "\u017f",
		`" password"`, "\"\u00a0token\u3000\"", "\"\t\u017fecret \"", `\"`, `"_.api-key"`,
		`"user:temp:a"`, `"app: temp:b"`, `"User:Temp:x"`, `"temp:y"`,
	}
	offset := func(loc []int, by int) []int {
		for k, v := range loc {
			if v >= 0 {
				loc[k] = v + by
			}
		}
		return loc
	}
	finders := []struct {
		name string
		find func(string) []int
		re   *regexp.Regexp
	}{
		{"baseline bearer", func(s string) []int { return findAtKeys(s, 0, bearerKeys, baselineBearerTokenAtRe) }, bearerRe},
		{"baseline basic", func(s string) []int { return findAtKeys(s, 0, basicKeys, baselineBasicTokenAtRe) }, basicRe},
		{"bearer", findBearerToken, bearerTokenRe},
		{"basic", findBasicToken, basicTokenRe},
		{"JSON", findJSONKeyValToken, jsonKeyValTokenRe},
		{"text key name", findTextKeyName, textKeyNameRe},
		{"auth header prefix", func(s string) []int { return findAtKeys(s, 0, authHeaderKeys, authHeaderPrefixAtRe) }, authHeaderPrefixRe},
	}
	n := 30_000
	if raceEnabled {
		n = 3_000
	}
	r := rand.New(rand.NewPCG(7, 8))
	for range n {
		s := randomText(r, alphabet, 12)
		for _, p := range []int{0, r.IntN(len(s) + 1)} {
			rest := s[p:]
			want := baselineTextKeyValTokenAtRe.FindStringSubmatchIndex(rest)
			if want == nil {
				want = textRe.FindStringSubmatchIndex(rest)
			}
			if got := baselineFindKeyValToken(rest); !slices.Equal(got, want) {
				t.Fatalf("baselineFindKeyValToken(%q) = %v, want %v", rest, got, want)
			}
			for _, f := range finders {
				if got, want := f.find(rest), f.re.FindStringSubmatchIndex(rest); !slices.Equal(got, want) {
					t.Fatalf("%s search in %q = %v, want %v", f.name, rest, got, want)
				}
			}
			if got, want := baselineFindJSONKeyValToken(s, p), offset(jsonRe.FindStringSubmatchIndex(rest), p); !slices.Equal(got, want) {
				t.Fatalf("baselineFindJSONKeyValToken(%q, %d) = %v, want %v", s, p, got, want)
			}
			if textKeyNameAtRe.MatchString(rest) && !mayStartTextKeyName(rest) {
				t.Fatalf("mayStartTextKeyName(%q) = false, want true: textKeyNameAtRe matches", rest)
			}
			if textKeyNameCamelAtRe.MatchString(rest) && !textKeyNames.mayStartAt(rest, 0) {
				t.Fatalf("textKeyNames.mayStartAt(%q, 0) = false, want true: textKeyNameCamelAtRe matches", rest)
			}
		}
		if got, want := findAllAtKeys(s, baselineAuthHeaderKeys, authAtRe), authRe.FindAllStringSubmatchIndex(s, -1); !slices.EqualFunc(got, want, slices.Equal) {
			t.Fatalf("baseline auth headers in %q = %v, want %v", s, got, want)
		}
		if got, want := findAllAtKeys(s, authHeaderKeys, authHeaderAtRe), authHeaderRe.FindAllStringSubmatchIndex(s, -1); !slices.EqualFunc(got, want, slices.Equal) {
			t.Fatalf("auth headers in %q = %v, want %v", s, got, want)
		}
	}
	// Keys of a few parts, so that whole keys match often.
	kr := rand.New(rand.NewPCG(9, 10))
	for range n {
		key := randomText(kr, alphabet, 3)
		if got, want := isBareTextKey(key), bareTextKeyRe.MatchString(key); got != want {
			t.Fatalf("isBareTextKey(%q) = %v, want %v", key, got, want)
		}
		if got, want := isWholeTextKeyName(key), textKeyNameWholeRe.MatchString(key); got != want {
			t.Fatalf("isWholeTextKeyName(%q) = %v, want %v", key, got, want)
		}
	}
}

// TestFindTextKeyNameMatchesRegexpOnOAuthRuns checks that findTextKeyName
// finds what textKeyNameRe finds in texts built at random from the parts of
// dotted OAuth names and from the bytes that may end them. It reads the run of
// bytes that such a name starts in once, rather than for each "oauth" in the
// run (see findTextKeyName), so its answer for a run that holds many of them
// must be the one that the pattern gives.
func TestFindTextKeyNameMatchesRegexpOnOAuthRuns(t *testing.T) {
	alphabet := []string{
		"oauth", "OAuth", "oauth-", "oauth_", "oauth.", "OAUTH2", "-", "_", ".", "..", ".code", ".CODE", "code", "x", "1",
		"=", ":", " ", "\t", "\n", `"`, `'`, `\"`, `\`, "password", "pass", "word", "token", "api_key", "temp:",
		"\u212a", "\u017f", "\u00e9", "\xff",
	}
	n := 100_000
	if raceEnabled {
		n = 10_000
	}
	r := rand.New(rand.NewPCG(11, 12))
	for range n {
		s := randomText(r, alphabet, 12)
		if got, want := findTextKeyName(s), textKeyNameRe.FindStringSubmatchIndex(s); !slices.Equal(got, want) {
			t.Fatalf("findTextKeyName(%q) = %v, want %v", s, got, want)
		}
	}
}

// TestShellScannerNoCloseMemoLimits checks that closedQuoteLen and
// closedEscapedQuoteLen give, for each query in turn on one scanner, the
// answer that a fresh scanner gives. A record of no closing quote that a scan
// under a narrow limit made must not be used under a wider limit, under which
// the closing quote can be in reach: whether that scan ended at the limit, or
// at an escape or a line splice that the limit cut, or the query is for a
// later opening quote that the record covers. Records that a line break or a
// quote ended, and queries under narrower limits, find no close either way.
func TestShellScannerNoCloseMemoLimits(t *testing.T) {
	type query struct{ pos, limit int }
	quoteLen := (*shellAssignScanner).closedQuoteLen
	escapedQuoteLen := (*shellAssignScanner).closedEscapedQuoteLen
	for _, tt := range []struct {
		name    string
		s       string
		f       func(sc *shellAssignScanner, pos, limit int) int
		queries []query
	}{
		{"single quote cut by a narrow limit", "'abc'", quoteLen, []query{{0, 3}, {0, 5}}},
		{"double quote cut by a narrow limit", `"abc"`, quoteLen, []query{{0, 3}, {0, 5}}},
		{"double quote with an escape that the limit cuts", `"a\"b"`, quoteLen, []query{{0, 3}, {0, 6}}},
		{"double quote with a line splice that the limit cuts", "\"a\\\r\nb\"", quoteLen, []query{{0, 4}, {0, 7}}},
		{"later double quote that a narrow record covers", `"a\"bc"`, quoteLen, []query{{0, 5}, {3, 7}}},
		{"escaped single quote cut by a narrow limit", `\'abc\'`, escapedQuoteLen, []query{{0, 5}, {0, 7}}},
		{"escaped double quote cut by a narrow limit", `\"abc\"`, escapedQuoteLen, []query{{0, 5}, {0, 7}}},
		{"escaped quote with a line splice that the limit cuts", "\\'a\\\r\nb\\'", escapedQuoteLen, []query{{0, 5}, {0, 9}}},
		{"single quote that a line break ends", "'ab\ncd'", quoteLen, []query{{0, 7}, {0, 2}, {0, 3}, {0, 7}}},
		{"escaped quote that a quote ends", `\'ab"cd\'`, escapedQuoteLen, []query{{0, 9}, {0, 4}, {0, 9}}},
		{"wide record, then a narrow query", "'abc'", quoteLen, []query{{0, 5}, {0, 3}, {0, 5}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var sc shellAssignScanner
			sc.reset(tt.s)
			for i, q := range tt.queries {
				var fresh shellAssignScanner
				fresh.reset(tt.s)
				if got, want := tt.f(&sc, q.pos, q.limit), tt.f(&fresh, q.pos, q.limit); got != want {
					t.Errorf("query %d (pos %d, limit %d) on %q = %d, want %d as from a fresh scanner", i, q.pos, q.limit, tt.s, got, want)
				}
			}
		})
	}
}

func TestPlaceholderAndQuoteBoundaryRegressions(t *testing.T) {
	const secret = "SYNTH_BOUNDARY_7Q9"
	for _, tt := range []struct {
		in, want string
	}{
		// Placeholders embedded inside credentials do not disable redaction of surrounding secret bytes.
		{`failed: "{"password":" [REDACTED] ` + secret + `","user":"bob"}"`, `failed: "{"password":" [REDACTED]","user":"bob"}"`},
		{`failed: "{"password":" %5BREDACTED%5D ` + secret + `","user":"bob"}"`, `failed: "{"password":" [REDACTED]","user":"bob"}"`},
		{`failed: "{"password":"` + secret + ` [REDACTED]","user":"bob"}"`, `failed: "{"password":"[REDACTED]","user":"bob"}"`},
		{`failed: "{"password":"abc [REDACTED] ` + secret + `","user":"bob"}"`, `failed: "{"password":"[REDACTED]","user":"bob"}"`},
		{`failed: "{"password":" ` + secret + ` [REDACTED]","user":"bob"}"`, `failed: "{"password":" [REDACTED]","user":"bob"}"`},
		// Hyphenated words, parenthesized words, and numeric quantities in cookie prose stay intact while cookie headers/tokens redact.
		{`The cookie: chocolate-chip was delicious.`, `The cookie: chocolate-chip was delicious.`},
		{`The cookie: (chocolate) was delicious.`, `The cookie: (chocolate) was delicious.`},
		{`The cookie: 2 pieces were left.`, `The cookie: 2 pieces were left.`},
		{`The cookie: 2.5 pieces remained.`, `The cookie: 2.5 pieces remained.`},
		{`The cookie: 1/2 piece remained.`, `The cookie: 1/2 piece remained.`},
		{`The cookie: ½ piece remained.`, `The cookie: ½ piece remained.`},
		{`The cookie: 2,000 pieces remained.`, `The cookie: 2,000 pieces remained.`},
		{`Cookie: 2.5`, `Cookie: [REDACTED]`},
		{`Cookie: sid=` + secret, `Cookie: [REDACTED]`},
		{`Cookie: 5f2b9c0e7d1a44e3`, `Cookie: [REDACTED]`},
		// Single-quoted passphrases with interior apostrophes past 4096 bytes, across newlines, with interior key=value pairs, or unterminated.
		{"password='it's " + strings.Repeat("pad ", 1250) + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's token=prefix " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's token='x' " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's " + secret + "\nstill here' user=x", "password='[REDACTED]' user=x"},
		{"password='it's pad " + secret, "password=[REDACTED]"},
		{"password='it's a trap; token='" + secret + "'", "password='[REDACTED]'"},
		{"password=pre'it's secret=f('it's token='x' " + secret + "')", "password=[REDACTED] secret=[REDACTED])"},
		{"password=pre'it's " + strings.Repeat("a_", 33) + "secret='it's token='x' " + secret + "'", "password=[REDACTED] " + strings.Repeat("a_", 33) + "secret='[REDACTED]'"},
		// Multiline quoted strings with ordinary content before a header line keep their closing quote.
		{"'pad\nAuthorization: [REDACTED]\n" + secret + "'\necho KEEP", "'pad\nAuthorization: [REDACTED]'\necho KEEP"},
		{"request='GET / HTTP/1.1\nAuthorization: [REDACTED]\n" + secret + "'\necho KEEP", "request='GET / HTTP/1.1\nAuthorization: [REDACTED]'\necho KEEP"},
		// Dotted pagination cursors and shell-quoted Bearer prefixes stay preserved.
		{`x.n_e_x_t.page.token=cursor-1`, `x.n_e_x_t.page.token=cursor-1`},
		{`next.page.token=cursor-1 prev.page.token=cursor-0`, `next.page.token=cursor-1 prev.page.token=cursor-0`},
		{`fb.page_token=` + secret, `fb.page_token=[REDACTED]`},
		{`page_access_token=` + secret, `page_access_token=[REDACTED]`},
		{`echo 'Bearer '"` + secret + `"; echo 'done'`, `echo 'Bearer [REDACTED]'; echo 'done'`},
		// Single-line quoted temp: keys containing = or : keep their full key name.
		{`failed: {'temp:a=b': '` + secret + `'}`, `failed: {'temp:a=b':'[REDACTED]'}`},
		{`failed: {'temp:a:b': '` + secret + `'}`, `failed: {'temp:a:b':'[REDACTED]'}`},
		// Quoted token counter or pagination cursor keys glued to a Bearer token keep their key name.
		{`Bearer ` + secret + `"prompt_token_count": 42`, `Bearer [REDACTED]"prompt_token_count": 42`},
		{`Bearer ` + secret + `"next_page_token": "cursor-1"`, `Bearer [REDACTED]"next_page_token": "cursor-1"`},
		// JSON wrapping does not bypass cookie-prose fix or destroy JSON text.
		{`{"note":"The cookie: 2.5 ounces remained."}`, `{"note":"The cookie: 2.5 ounces remained."}`},
		{`{"note":"Cookie: sid=` + secret + `"}`, `{"note":"Cookie: [REDACTED]"}`},
		// Expanded cookie prose classifier: percentages, hyphenated units, parenthesized quantities, negative quantities, quoted prose.
		{`The cookie: 50% was left.`, `The cookie: 50% was left.`},
		{`The cookie: 2.5-ounce pieces remained.`, `The cookie: 2.5-ounce pieces remained.`},
		{`The cookie: (2 pieces) was delicious.`, `The cookie: (2 pieces) was delicious.`},
		{`The cookie: "chocolate" was great.`, `The cookie: "chocolate" was great.`},
		{`The cookie: -2 degrees outside.`, `The cookie: -2 degrees outside.`},
		{`The cookie: “chocolate” was delicious.`, `The cookie: “chocolate” was delicious.`},
		// Punctuation and incomplete multiline apostrophe values.
		{"password='it's pad\n" + secret, "password=[REDACTED]"},
		{"password='it's pad\nline2\n" + secret, "password=[REDACTED]"},
		{"password='it's\n" + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's, " + secret + "' user=x", "password='[REDACTED]' user=x"},
		// Disproof cases: quoted cookie values, contractions in cookie prose, and
		// JSON-wrapped multiline apostrophe values that must not over-extend.
		{"Cookie: \"" + secret + "\"", "Cookie: [REDACTED]"},
		{"The cookie: 'chocolate' was great.", "The cookie: 'chocolate' was great."},
		{"The cookie: \"chocolate\" wasn't bad.", "The cookie: \"chocolate\" wasn't bad."},
		{`"The cookie: \"chocolate\" was great."`, `"The cookie: \"chocolate\" was great."`},
		{"note=\"password='it's " + secret + "\" user='bob'", "note=\"password=[REDACTED]\" user='bob'"},
		{`{"a":"password='it's pad","b":"secret='it's ` + secret + `"}`, `{"a":"password=[REDACTED]","b":"secret=[REDACTED]"}`},
		{`{"a":"password='it's pad","b":"quote 'here'"}`, `{"a":"password=[REDACTED]","b":"quote 'here'"}`},

		// Apostrophe cache must not swallow continuation-line key=value or closing outer quotes.
		{"password=('it's a) secret='it's pad\nuser=bob", "password=[REDACTED] a) secret=[REDACTED]\nuser=bob"},
		{"password=('it's a) \"secret='it's pad\"", "password=[REDACTED] a) \"secret=[REDACTED]\""},
		// Multiline JSON cookie prose is preserved across all paths.
		{`{"note":"The cookie: 2.5 ounces remained.\nKEEP"}`, `{"note":"The cookie: 2.5 ounces remained.\nKEEP"}`},
		{`{"note":"The cookie: chocolate-chip was delicious.\nLine 2."}`, `{"note":"The cookie: chocolate-chip was delicious.\nLine 2."}`},
		{`"The cookie: 2.5 ounces remained.\nKEEP"`, `"The cookie: 2.5 ounces remained.\nKEEP"`},
		// Apostrophe passphrase shapes that must redact their secret.
		{"password='it's... " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's) " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's," + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"DB_PASSWORD=pre'it's correct " + secret, "DB_PASSWORD=[REDACTED]"},
		// Quoted Authorization headers keep their closing quote and trailing line.
		{"'Authorization: Bearer " + secret + "' user=bob", "'Authorization: [REDACTED]' user=bob"},
		{"curl -H 'Authorization: Bearer " + secret + "' https://example.com", "curl -H 'Authorization: [REDACTED]' https://example.com"},
		// Cookie prose with quantity and punctuation shapes stays intact,
		// while a cookie pair after quoted prose, or a lone time, redacts.
		{"The cookie: \"chocolate\", not vanilla, was great.", "The cookie: \"chocolate\", not vanilla, was great."},
		{"The cookie: 2:30 bake time worked well.", "The cookie: 2:30 bake time worked well."},
		{"The cookie: 2:30pm batch was fresh.", "The cookie: 2:30pm batch was fresh."},
		{"The cookie: 2-3 pieces remained.", "The cookie: 2-3 pieces remained."},
		{"The cookie: $2 each at the stand.", "The cookie: $2 each at the stand."},
		{"The cookie: #2 on the menu.", "The cookie: #2 on the menu."},
		{"The cookie: 2nd batch was better.", "The cookie: 2nd batch was better."},
		{"The cookie: A+ texture and flavor.", "The cookie: A+ texture and flavor."},
		{"The cookie: x2 recipe for the party.", "The cookie: x2 recipe for the party."},
		{"Cookie: \"word\", sid=" + secret, "Cookie: [REDACTED]"},
		{"Cookie: 2:30", "Cookie: [REDACTED]"},
		// An escaped line break in a JSON string of an object with more
		// fields, or of an array, keeps the cookie prose in it.
		{`{"a":"x","note":"The cookie: 2.5 ounces remained.\nKEEP"}`, `{"a":"x","note":"The cookie: 2.5 ounces remained.\nKEEP"}`},
		{`["The cookie: 2.5 ounces remained.\nKEEP","x"]`, `["The cookie: 2.5 ounces remained.\nKEEP","x"]`},
		// A run of punctuation after an interior apostrophe still ends in
		// the quoted passphrase's closing quote.
		{"password='it's?! " + secret + "' user=x", "password='[REDACTED]' user=x"},
		// Existing controls that must still pass.
		{"password='it's pad\nuser=bob", "password=[REDACTED]\nuser=bob"},
		{"password='it's pad\n" + secret + "\nuser=bob", "password=[REDACTED]\nuser=bob"},
		{"note: \"password='it's\n" + secret + "\"", "note: \"password=[REDACTED]\""},
		// A quoted value whose interior contraction is followed by a quote
		// that a later quote on its line follows, with no key between
		// them, ends at that later quote, as a shell joins the parts
		// between them and as a value written into the quotes as it is
		// reads; a key, or a quote that opens a key's value, ends it at
		// the first quote.
		{"password='it's' " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"note=\"password='it's' " + secret + "' user=x\"", "note=\"password='[REDACTED]' user=x\""},
		{"DB_PASSWORD='it's' correct " + secret + "' user=x", "DB_PASSWORD='[REDACTED]' user=x"},
		{"password='it's' user=x", "password='[REDACTED]' user=x"},
		{"password='it's' note='x'", "password='[REDACTED]' note='x'"},
		// A joined contraction value that punctuation continues, and an
		// explicitly opened one whose contraction ending a digit, "_" or
		// "=" follows, are redacted whole.
		{"DB_PASSWORD=pre'it's," + secret + " x", "DB_PASSWORD=[REDACTED]"},
		{"DB_PASSWORD=pre'it's) " + secret + "' x", "DB_PASSWORD=[REDACTED] x"},
		{"password='it's_x " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's=x " + secret + "' user=x", "password='[REDACTED]' user=x"},
		{"password='it's1 " + secret + "' user=x", "password='[REDACTED]' user=x"},
		// A cookie pair after words that read as prose, whether quoted,
		// after a space or after an escaped backslash, redacts the Cookie
		// value.
		{"curl -H 'Cookie: chocolate, \"sid=" + secret + "\"' https://x", "curl -H 'Cookie: [REDACTED]' https://x"},
		{"Cookie: \"word\" Monster sid=" + secret, "Cookie: [REDACTED]"},
		{`Cookie: Monster\\sid=` + secret, "Cookie: [REDACTED]"},
		// Joined-open contraction followed by a boundary quote or explicit-open
		// contraction followed by an adjacent/spaced unclosed quote redacts the
		// full secret without eating following keys.
		{"password=pre'it's' " + secret + "' user=x", "password=[REDACTED] user=x"},
		{"DB_PASSWORD=pre'it's' correct " + secret + "' user=x", "DB_PASSWORD=[REDACTED] user=x"},
		{"token=ab'cd's' " + secret + "' user=x", "token=[REDACTED] user=x"},
		{"note=\"password=pre'it's' " + secret + "' user=x\"", "note=\"password=[REDACTED] user=x\""},
		{"{\"msg\":\"password=pre'it's' " + secret + "' user=x\"}", "{\"msg\":\"password=[REDACTED] user=x\"}"},
		{"password=pre'it's' " + secret + "' token=pre'it's' " + secret + "2' user=x", "password=[REDACTED] token=[REDACTED] user=x"},
		{"password=pre'it's'' " + secret + "' user=x", "password=[REDACTED] user=x"},
		{"password=pre'it's'' '" + secret, "password=[REDACTED]"},
		{"DB_PASSWORD='Bob's5''" + secret, "DB_PASSWORD=[REDACTED]"},
		{"token='don't_'  '" + secret, "token=[REDACTED]"},
		// Benign quoted log lines with cookie/Set-Cookie prose remain unchanged
		// and idempotent when followed by key=value fields outside the quote.
		{"level=info msg=\"failed to parse cookie: invalid character\" code=400", "level=info msg=\"failed to parse cookie: invalid character\" code=400"},
		{"level=error msg=\"invalid cookie: expired\" path=/login", "level=error msg=\"invalid cookie: expired\" path=/login"},
		{"msg='bad cookie: expired' id=3", "msg='bad cookie: expired' id=3"},
		{"msg=\"bad cookie: expired\" id=3 other=\"x y\"", "msg=\"bad cookie: expired\" id=3 other=\"x y\""},
		{"msg=\"cookie: Monster was here\" k=v", "msg=\"cookie: Monster was here\" k=v"},
		{"msg=\"Set-Cookie: expired\" id=1", "msg=\"Set-Cookie: expired\" id=1"},
		{"curl -H 'Cookie: " + secret + "' k=v", "curl -H 'Cookie: [REDACTED]' k=v"},
		{"failed to read cookie: http: named cookie not present [id=3]", "failed to read cookie: http: named cookie not present [id=3]"},
		{"cookie: http: named cookie not present id=3", "cookie: http: named cookie not present id=3"},
		{"The cookie: chocolate chip, x=" + secret, "The cookie: [REDACTED]"},
		{"Cookie: Monster", "Cookie: Monster"},
		// Glued contraction suffixes without a closing quote and prose quotes
		// after a closed contraction value do not swallow following keys or prose.
		{"password='abc'd1 user=x", "password=[REDACTED] user=x"},
		{"password='abc'd=1 user='bob'", "password=[REDACTED] user='bob'"},
		{"password='abc'm_x user='bob'", "password=[REDACTED] user='bob'"},
		{"token='abc's=1 user='bob' k=v", "token=[REDACTED] user='bob' k=v"},
		{"password='it's' and then he said 'hello' to me", "password='[REDACTED]' and then he said 'hello' to me"},
		{"user said password='it's' ok then 'bye' ok", "user said password='[REDACTED]' ok then 'bye' ok"},
		// Colons inside non-header tokens (times, labels, JSON, URLs) do not
		// stop restHoldsCookiePair before a trailing cookie pair, while a later
		// Cookie or Authorization header on the same line is redacted on its own.
		{"note=\"Cookie: Monster at 10:30 sid=" + secret + "\"", "note=\"Cookie: [REDACTED]\""},
		{"note=\"Cookie: Monster note: sid=" + secret + "\"", "note=\"Cookie: [REDACTED]\""},
		{"note=\"Cookie: Monster {\\\"a\\\":1} sid=" + secret + "\"", "note=\"Cookie: [REDACTED]\""},
		{"note=\"Cookie: chocolate \\\"x\\\" at 10:30 sid=" + secret + "\"", "note=\"Cookie: [REDACTED]\""},
		{"Cookie: Monster https://example.com sid=" + secret, "Cookie: [REDACTED]"},
		{"Cookie: Monster: sid=" + secret, "Cookie: [REDACTED]"},
		{"Cookie: Monster at 10:30 was great. Cookie: sid=" + secret, "Cookie: Monster at 10:30 was great. Cookie: [REDACTED]"},
		{"Cookie: Monster at 10:30 was great. Authorization: Bearer " + secret, "Cookie: Monster at 10:30 was great. Authorization: [REDACTED]"},
	} {
		checkAllRedactionPaths(t, tt.in, tt.want)
	}
	// redactErrorMessage scrubs a JSON message by its decoded strings and
	// writes it compactly again, so its output is checked on its own.
	for _, tt := range []struct{ in, want, wantErr string }{
		{
			`{"note": "password='it's' ` + secret + `' user=x"}`,
			`{"note": "password='[REDACTED]' user=x"}`,
			`{"note":"password='[REDACTED]' user=x"}`,
		},
		{
			`{"note":"The cookie: chocolate\npassword=` + secret + `"}`,
			`{"note":"The cookie: [REDACTED]"}`,
			`{"note":"The cookie: chocolate\npassword=[REDACTED]"}`,
		},
		{
			`{"note":"The cookie: chocolate\nsecret: ` + secret + `"}`,
			`{"note":"The cookie: [REDACTED]"}`,
			`{"note":"The cookie: chocolate\nsecret:[REDACTED]"}`,
		},
	} {
		checkRedactionPaths(t, tt.in, tt.want)
		if got := redactErrorMessage(tt.in); got != tt.wantErr {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", tt.in, got, tt.wantErr)
		}
		if again := redactErrorMessage(tt.wantErr); again != tt.wantErr {
			t.Errorf("redactErrorMessage(%q) = %q, want it unchanged", tt.wantErr, again)
		}
	}
	// JSON-wrapped cookie-prose sentences are preserved across all redaction
	// paths, and a JSON object whose value is a cookie header redacts its
	// secret.
	cookieProseSentences := []string{
		"The cookie: chocolate-chip was delicious.",
		"The cookie: (chocolate) was delicious.",
		"The cookie: 2 pieces were left.",
		"The cookie: 2.5 pieces remained.",
		"The cookie: 1/2 piece remained.",
		"The cookie: ½ piece remained.",
		"The cookie: 2,000 pieces remained.",
		"The cookie: 50% was left.",
		"The cookie: 2.5-ounce pieces remained.",
		"The cookie: (2 pieces) was delicious.",
		"The cookie: \"chocolate\" was great.",
		"The cookie: 'chocolate' was great.",
		"The cookie: \"chocolate\" wasn't bad.",
		"The cookie: -2 degrees outside.",
		"The cookie: “chocolate” was delicious.",
		"The cookie: \"chocolate\", not vanilla, was great.",
		"The cookie: 2:30 bake time worked well.",
		"The cookie: 2:30pm batch was fresh.",
		"The cookie: 2-3 pieces remained.",
		"The cookie: $2 each at the stand.",
		"The cookie: #2 on the menu.",
		"The cookie: 2nd batch was better.",
		"The cookie: A+ texture and flavor.",
		"The cookie: x2 recipe for the party.",
		"The cookie: 2.5 ounces remained.\nKEEP",
	}
	for _, sentence := range cookieProseSentences {
		b, err := json.Marshal(map[string]string{"note": sentence})
		if err != nil {
			t.Fatalf("json.Marshal(%q): %v", sentence, err)
		}
		checkAllRedactionPaths(t, string(b), string(b))
		if !json.Valid(b) {
			t.Errorf("json.Marshal output not valid JSON: %s", b)
		}
	}
	cookieObj := []byte(`{"Cookie":"The cookie: 2.5 ounces remained."}`)
	checkAllRedactionPaths(t, string(cookieObj), `{"Cookie":"[REDACTED]"}`)
	// A cookie pair in a JSON string with an escaped line break, and a
	// quoted Cookie header value that its quote carries onto the next line,
	// are redacted on every path.
	for _, in := range []string{
		`{"note":"Cookie: sid=` + secret + `\nKEEP"}`,
		`{"a":"x","note":"Cookie: sid=` + secret + `\nKEEP"}`,
		"curl -H 'Cookie: Monster cake\n" + secret + "'",
	} {
		checkNoLeak(t, in, secret)
		if raw, _ := sanitizeRawText(in); strings.HasPrefix(in, "{") && !json.Valid([]byte(raw)) {
			t.Errorf("sanitizeRawText(%q) = %q, which is not valid JSON", in, raw)
		}
	}
	// Multiline apostrophe values that span into a new line with a key=value
	// pair, a joined-quote passphrase in a double-quoted string, which must
	// stop at the string's closing quote, and a double-quoted Authorization
	// header, which must keep its closing quote and the text after it:
	// scrubTextPatterns must redact the secret. The sanitizeRawText and
	// SmartTruncate paths return [UNPARSEABLE_JSON_BLOB] for these inputs
	// because they start with a quote that does not parse as JSON, or that
	// decodes to a credential, which is expected; only scrubTextPatterns is
	// checked exactly here, and no path may write the secret.
	for _, tt := range []struct{ in, want string }{
		{"\"password='it's pad\n" + secret + "==\"", "\"password=[REDACTED]\""},
		{"\"password='it's pad\nstill here a=b " + secret, "\"password=[REDACTED]"},
		{"\"DB_PASSWORD=pre'it's correct " + secret + "\"", "\"DB_PASSWORD=[REDACTED]\""},
		{`"Authorization: Bearer ` + secret + `" user=bob`, `"Authorization: [REDACTED]" user=bob`},
	} {
		checkNoLeak(t, tt.in, secret)
		if got := scrubTextPatterns(tt.in); got != tt.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tt.in, got, tt.want)
		}
		if got := scrubTextPatterns(tt.want); got != tt.want {
			t.Errorf("scrubbing %q again gave %q, want it unchanged", tt.want, got)
		}
	}
}

func TestJoinedHeaderQuotesRedacted(t *testing.T) {
	const secret = "SYNTH_REVIEW_ONLY_CREDENTIAL"
	for _, tt := range []struct{ in, want, wantErr string }{
		{
			in:   `curl -H "Authorization: Digest username="bob", response="` + secret + `""`,
			want: `curl -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix" ` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H 'Authorization: Digest username='bob', response='` + secret + `''`,
			want: `curl -H 'Authorization: [REDACTED]'`,
		},
		{
			in:   `curl -H 'X-Api-Key: 'prefix' ` + secret + `'`,
			want: `curl -H 'X-Api-Key: [REDACTED]'`,
		},
		{
			in:   `curl -H "Authorization: Bearer ""` + secret + `"; echo KEEP`,
			want: `curl -H "Authorization: [REDACTED]"; echo KEEP`,
		},
		{
			in:   `curl -H "Authorization: Digest username="bob", response="` + secret + `"" https://example.com`,
			want: `curl -H "Authorization: [REDACTED]" https://example.com`,
		},
		// Multiple auth/cookie headers on the same line.
		{
			in:   `curl -H "X-Api-Key: first_key" -H "Authorization: Digest username="bob", response="` + secret + `""`,
			want: `curl -H "X-Api-Key: [REDACTED]" -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "Authorization: Bearer first_tok" -H "Authorization: Digest username="bob", response="` + secret + `""`,
			want: `curl -H "Authorization: [REDACTED]" -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: first" -H "X-Goog-Api-Key: "prefix" ` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]" -H "X-Goog-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "Cookie: sid=first" -H "Cookie: "prefix" ` + secret + `"`,
			want: `curl -H "Cookie: [REDACTED]" -H "Cookie: [REDACTED]"`,
		},
		{
			in:   `curl -H "Authorization: Bearer first" -H "Proxy-Authorization: Digest username="bob", response="` + secret + `""`,
			want: `curl -H "Authorization: [REDACTED]" -H "Proxy-Authorization: [REDACTED]"`,
		},
		// Unspaced comma-joined shell fragments vs JSON sibling fields and array elements.
		{
			in:   `curl -H "Authorization: Digest username="bob,response="` + secret + `""`,
			want: `curl -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "Authorization: Digest username="bob,"` + secret + `""`,
			want: `curl -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix,` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix,"` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: ""prefix",` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: ""prefix","` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H 'X-Api-Key: 'prefix','` + secret + `'`,
			want: `curl -H 'X-Api-Key: [REDACTED]'`,
		},
		{
			in:      `{"h":"Authorization: Bearer ` + secret + `","b":"PUBLIC_KEEP"}`,
			want:    `{"h":"Authorization: [REDACTED]","b":"PUBLIC_KEEP"}`,
			wantErr: `{"b":"PUBLIC_KEEP","h":"Authorization: [REDACTED]"}`,
		},
		{
			in:   `["Authorization: Bearer ` + secret + `",123,"PUBLIC_KEEP"]`,
			want: `["Authorization: [REDACTED]",123,"PUBLIC_KEEP"]`,
		},
		{
			in:   `["Authorization: Bearer ` + secret + `",true,null,-1,"PUBLIC_KEEP"]`,
			want: `["Authorization: [REDACTED]",true,null,-1,"PUBLIC_KEEP"]`,
		},
		// Standalone and doubly-escaped quotes in log lines and JSON-wrapped commands.
		{
			in:   `ERROR: curl -H \"Authorization: Digest username=\"bob\", response=\"` + secret + `\"\" https://PUBLIC_KEEP.example.com`,
			want: `ERROR: curl -H \"Authorization: [REDACTED]\" https://PUBLIC_KEEP.example.com`,
		},
		{
			in:   `ERROR: curl -H \"Authorization: Digest username=\\\"bob\\\", response=\\\"` + secret + `\\\"\" https://PUBLIC_KEEP.example.com`,
			want: `ERROR: curl -H \"Authorization: [REDACTED]\" https://PUBLIC_KEEP.example.com`,
		},
		{
			in:   `ERROR: curl -H \"X-Api-Key: \\\"prefix\\\" ` + secret + `\" https://PUBLIC_KEEP.example.com`,
			want: `ERROR: curl -H \"X-Api-Key: [REDACTED]\" https://PUBLIC_KEEP.example.com`,
		},
		{
			in:   `{"b":"KEEP_B","h":"curl -H \"Authorization: Digest username=\\\"bob\\\", response=\\\"` + secret + `\\\"\" https://PUBLIC_KEEP.example.com"}`,
			want: `{"b":"KEEP_B","h":"curl -H \"Authorization: [REDACTED]\" https://PUBLIC_KEEP.example.com"}`,
		},
		{
			in:   `{"b":"KEEP_B","h":"curl -H \"Authorization: Digest username=\"bob\", response=\"` + secret + `\"\" https://PUBLIC_KEEP.example.com"}`,
			want: `{"b":"KEEP_B","h":"curl -H \"Authorization: [REDACTED]\" https://PUBLIC_KEEP.example.com"}`,
		},
		// Stranded opening quote idempotency and single-quote apostrophe disambiguation.
		{
			in:   `curl -H "Authorization: Bearer "` + secret + `; echo "PUBLIC_KEEP"`,
			want: `curl -H "Authorization: [REDACTED]"; echo "PUBLIC_KEEP"`,
		},
		{
			in:   `curl -H "Authorization: Bearer "'` + secret + `' -H "X-Public: PUBLIC_KEEP"`,
			want: `curl -H "Authorization: [REDACTED]" -H "X-Public: PUBLIC_KEEP"`,
		},
		{
			in:   `curl -H "Authorization: Bearer "bob's_` + secret + `; echo 'PUBLIC_KEEP'`,
			want: `curl -H "Authorization: [REDACTED]"; echo 'PUBLIC_KEEP'`,
		},
		{
			in:   `curl -H 'Authorization: Bearer 'bob's_` + secret + `; echo 'PUBLIC_KEEP'`,
			want: `curl -H 'Authorization: [REDACTED]'; echo 'PUBLIC_KEEP'`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix' ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix'a ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "Authorization: Digest username="bob', response='` + secret + `"`,
			want: `curl -H "Authorization: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix"bob's_` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "prefix"bob's`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		// Closing bracket and brace boundary variants in shell words.
		{
			in:   `curl -H "X-Api-Key: "pre}` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "X-Api-Key: "pre]` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Authorization: Bearer "pre}` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Authorization: Bearer "pre]` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Cookie: "pre}` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Cookie: "pre]` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Escaped quote multi-header command where bare scan must not consume next header.
		{
			in:   `curl -H "X-Api-Key: first\" -H "Authorization: Bearer x,` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]" [REDACTED]`,
		},
		// Multiline joined single quotes closing on a subsequent line.
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'" + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'\n" + secret + " tail'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre'" + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre'\n" + secret + " tail'; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre'" + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre'\n" + secret + " tail'; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Multiline joined single quotes with subsequent quoted fragments or separate commands on closing line.
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\ntail'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail'; echo 'public\nmore'; echo PUBLIC_KEEP",
			want: "curl -H \"X-Api-Key: [REDACTED]\"; echo 'public\nmore'; echo PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'; echo 'public\nmore'; echo PUBLIC_KEEP",
			want: "curl -H \"Authorization: [REDACTED]\"; echo 'public\nmore'; echo PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\ntail'; echo 'public\nmore'; echo PUBLIC_KEEP",
			want: "curl -H \"Cookie: [REDACTED]\"; echo 'public\nmore'; echo PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail'\"more\nend\"; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'\"more\nend\"; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\ntail'\"more\nend\"; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Multiline joined single quotes with colon preceding closing quote at end of line.
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nX-Note: '\necho PUBLIC_KEEP",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\nX-Note: '\necho PUBLIC_KEEP",
			want: "curl -H \"Authorization: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\nX-Note: '\necho PUBLIC_KEEP",
			want: "curl -H \"Cookie: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'\nX-Note: " + secret + "'\necho PUBLIC_KEEP",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre'\nX-Note: " + secret + "'\necho PUBLIC_KEEP",
			want: "curl -H \"Authorization: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"Cookie: \"pre'\nX-Note: " + secret + "'\necho PUBLIC_KEEP",
			want: "curl -H \"Cookie: [REDACTED]\"\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'\nX-Note: " + secret + "'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre'\nX-Note: " + secret + "'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre'\nX-Note: " + secret + "'foo' more\nend'; echo PUBLIC_KEEP",
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Escaped-quote multiline closure ending at colon-preceded quote at EOL.
		{
			in:   "curl -H \\\"X-Api-Key: \\\"pre' " + secret + "\nX-Note: '\necho PUBLIC_KEEP",
			want: "curl -H \\\"X-Api-Key: [REDACTED]\\\"\necho PUBLIC_KEEP",
		},
		// Same-line possessives and inch marks do not swallow a following quoted auth header.
		{
			in:   `curl -H "X-Api-Key: "Bob's Authorization: 'Bearer ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]" Authorization: [REDACTED]`,
		},
		{
			in:   `curl -H "X-Api-Key: "Bob's Cookie: 'theme=dark; sid=` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]" Cookie: [REDACTED]`,
		},
		{
			in:   `curl -H "X-Api-Key: "v2's Authorization: 'Bearer ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]" Authorization: [REDACTED]`,
		},
		{
			in:   `curl -H "X-Api-Key: "SECRET1's Authorization: 'Bearer ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]" Authorization: [REDACTED]`,
		},
		{
			in:   `curl -H "X-Api-Key: "v2' Authorization: 'Bearer ` + secret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]" Authorization: [REDACTED]`,
		},
		{
			in:   `I ran curl -H "X-Api-Key: "Bob's Authorization: 'Bearer ` + secret + `' and it failed`,
			want: `I ran curl -H "X-Api-Key: [REDACTED]" Authorization: [REDACTED]`,
		},
		{
			in:   "I ran curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer " + secret + "' and it failed\nPUBLIC_KEEP",
			want: "I ran curl -H \"X-Api-Key: [REDACTED]\" Authorization: [REDACTED]\nPUBLIC_KEEP",
		},
		{
			in:   `curl -H "X-Api-Key: "Bob's Authorization: 'Bearer ` + secret + `' didn't work`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "Bob's key Authorization: 'Bearer ` + secret + `' PUBLIC_KEEP`,
			want: `curl -H "X-Api-Key: [REDACTED]" PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's Authorization:'Bearer " + secret + "'\nPUBLIC_KEEP",
			want: "curl -H \"X-Api-Key: [REDACTED]\" Authorization: [REDACTED]\nPUBLIC_KEEP",
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer ` + secret + `'\""}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]\""}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\""}`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer ` + secret + `'\"", "note": "PUBLIC_KEEP"}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]\"", "note": "PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"","note":"PUBLIC_KEEP"}`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer ` + secret + `'", "note": "PUBLIC_KEEP"}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]\" Authorization: [REDACTED]", "note": "PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\" Authorization: [REDACTED]","note":"PUBLIC_KEEP"}`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's\nAuthorization: 'Bearer ` + secret + `'\""}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]\""}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"\nAuthorization: [REDACTED]"}`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's\nAuthorization: 'Bearer ` + secret + `'\"", "note": "PUBLIC_KEEP"}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]\"", "note": "PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"\nAuthorization: [REDACTED]","note":"PUBLIC_KEEP"}`,
		},
		{
			in:   `curl -H 'X-Api-Key: 'size12" Cookie: "theme=dark; sid=` + secret + `" PUBLIC_KEEP`,
			want: `curl -H 'X-Api-Key: [REDACTED]' PUBLIC_KEEP`,
		},
		// Multiline possessives and inch marks check quote-chain balance across subsequent headers and notes.
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\nCookie: 'theme=dark; sid=" + secret + "'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nCookie: [REDACTED]",
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\r\nCookie: 'theme=dark; sid=" + secret + "'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\r\nCookie: [REDACTED]",
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\nAuthorization: 'Bearer " + secret + "'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nAuthorization: [REDACTED]",
		},
		{
			in:   "> curl -H \"X-Api-Key: \"Bob's demo key\n< HTTP/1.1 200 OK\n< Set-Cookie: 'theme=dark; session=" + secret + "'",
			want: "> curl -H \"X-Api-Key: [REDACTED]\" demo key\n< HTTP/1.1 200 OK\n< Set-Cookie: [REDACTED]",
		},
		{
			in:   "curl -H 'X-Api-Key: 'size12\"\nCookie: \"theme=dark; sid=" + secret + "\"",
			want: "curl -H 'X-Api-Key: [REDACTED]'\nCookie: [REDACTED]",
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\nCookie: 'theme=dark; sid=" + secret + "'\nNote: Alice's PUBLIC_KEEP",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nNote: Alice's PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\nX-Note: 'PUBLIC_KEEP'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nX-Note: 'PUBLIC_KEEP'",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre1' " + secret + "\ntail'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Embedded mid-word header names where bare scan must not split credential without word boundary.
		{
			in:   `curl -H "X-Api-Key: "preAuthorization:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "X-Api-Key: "preauthorization:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Authorization: Bearer "preCookie:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Authorization: Bearer "precookie:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Authorization: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "X-Goog-Api-Key: "preAuthorization:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "X-Goog-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Cookie: "preAuthorization:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   `curl -H "Set-Cookie: "preAuthorization:` + secret + `; echo PUBLIC_KEEP`,
			want: `curl -H "Set-Cookie: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		// Closing lines containing prose contractions, bare digit stems, valid-shell continuations, and double-escaped JSON newlines.
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' it's",
			want: `curl -H "X-Api-Key: [REDACTED]" it's`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'; echo don't PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo don't PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\r\ntail' Bob's\r\necho PUBLIC_KEEP",
			want: "curl -H \"Cookie: [REDACTED]\" Bob's\r\necho PUBLIC_KEEP",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' it's Authorization: 'Bearer " + secret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]" it's Authorization: [REDACTED]`,
		},
		{
			in:   "msg=\"X-Api-Key: \"pre' " + secret + "\ntail' it's",
			want: `msg="X-Api-Key: [REDACTED]" it's`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'\n" + secret + " tail' it's",
			want: `curl -H "X-Api-Key: [REDACTED]" it's`,
		},
		{
			in:   `curl -H "X-Api-Key: "abc123' ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"v2' " + secret + "\nCookie: 'theme=dark; sid=" + secret + "'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nCookie: [REDACTED]",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'tail; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' it's PUBLIC_KEEP; Authorization: 'Bearer " + secret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]" it's PUBLIC_KEEP; Authorization: [REDACTED]`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's\\nAuthorization: 'Bearer ` + secret + `'"}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\""}`,
		},
		{
			in:      `{"cmd": "curl -H \"X-Api-Key: \"Bob's\\nAuthorization: 'Bearer ` + secret + `'", "note": "PUBLIC_KEEP"}`,
			want:    `{"cmd": "curl -H \"X-Api-Key: [REDACTED]", "note": "PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"","note":"PUBLIC_KEEP"}`,
		},
		// Closing lines containing trailing plural possessives (such as users', James', parents') and negative guards.
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' users'",
			want: `curl -H "X-Api-Key: [REDACTED]" users'`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' the users' keys",
			want: `curl -H "X-Api-Key: [REDACTED]" the users' keys`,
		},
		{
			in:   "curl -H \"Authorization: Bearer \"pre' " + secret + "\ntail'; echo users' PUBLIC_KEEP",
			want: `curl -H "Authorization: [REDACTED]"; echo users' PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"Cookie: \"pre' " + secret + "\r\ntail' James'\r\necho PUBLIC_KEEP",
			want: "curl -H \"Cookie: [REDACTED]\" James'\r\necho PUBLIC_KEEP",
		},
		{
			in:   "msg=\"X-Api-Key: \"pre' " + secret + "\ntail' parents'.",
			want: `msg="X-Api-Key: [REDACTED]" parents'.`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre'\n" + secret + " tail' users'",
			want: `curl -H "X-Api-Key: [REDACTED]" users'`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail' users' PUBLIC_KEEP; Authorization: 'Bearer " + secret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]" users' PUBLIC_KEEP; Authorization: [REDACTED]`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\ntoken: tail' " + secret + " users'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\ntoken:[REDACTED]",
		},
		{
			in:   "curl -H \"X-Api-Key: \"Bob's\npassword: tail' " + secret + " James'",
			want: "curl -H \"X-Api-Key: [REDACTED]\"\npassword:[REDACTED]",
		},
	} {
		checkNoLeak(t, tt.in, secret)
		checkRedactionPaths(t, tt.in, tt.want)
		if got, wantFresh := redactAuthHeaders(tt.in), redactAuthHeadersFreshScanner(tt.in); got != wantFresh {
			t.Errorf("redactAuthHeaders(%q) = %q, freshScanner = %q", tt.in, got, wantFresh)
		}
		wantErr := tt.want
		if tt.wantErr != "" {
			wantErr = tt.wantErr
		}
		if got := redactErrorMessage(tt.in); got != wantErr {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", tt.in, got, wantErr)
		}
		if got := scrubTextPatterns(tt.want); got != tt.want {
			t.Errorf("scrubTextPatterns idempotency(%q) = %q, want %q", tt.want, got, tt.want)
		}
		if raw, _, err := SmartTruncate(tt.in, 1<<20); err != nil || strings.Contains(string(raw), secret) {
			t.Errorf("SmartTruncate(%q, 1<<20) = (%q, %v), leaked or errored", tt.in, raw, err)
		}
	}
	for _, tt := range []struct {
		in           string
		wantValid    bool
		malformedErr bool
	}{
		{
			in:        `{"error": "curl -H \"Authorization: Digest username=\\\"bob\\\", response=\\\"` + secret + `\\\"\""}`,
			wantValid: true,
		},
		{
			in:        `{"h":"curl -H \"Authorization: Digest username=\\\"bob\\\", response=\\\"` + secret + `\\\"\"","b":"PUBLIC_KEEP"}`,
			wantValid: true,
		},
		{
			in:        `{"error": "curl -H \"Authorization: Digest username=\"bob\", response=\"` + secret + `\"\""}`,
			wantValid: true,
		},
		// Word-adjacent single-quoted shell fragments containing metacharacters or followed by unpaired/comment quotes.
		{in: `curl -H "X-Api-Key: "prefix'a;` + secret + `'`},
		{in: `curl -H "X-Api-Key: "prefix'a&` + secret + `'`},
		{in: `curl -H "Authorization: Bearer "tok'SYNTH;` + secret + `' https://PUBLIC_KEEP.example.com`},
		{in: `curl -H "X-Api-Key: "prefix'part1 ` + secret + `'; echo 'unclosed`},
		{in: `curl -H "X-Api-Key: "prefix'part1 ` + secret + `' # user's_note`},
		{in: `curl -H "Authorization: Bearer "tok'SYNTH ` + secret + `' # 'PUBLIC_KEEP`},
		// Multiline JSON objects and arrays preserve braceDepth and siblings.
		{
			in:        "{\n  \"h\": \"Authorization: Bearer " + secret + "\",\"b\": \"PUBLIC_KEEP\"\n}",
			wantValid: true,
		},
		{
			in:        "[\n  \"Authorization: Bearer " + secret + "\",123,\"PUBLIC_KEEP\"\n]",
			wantValid: true,
		},
		// JSON with sibling sensitive key and escaped auth header preserves valid JSON and public fields.
		{
			in:        `{"api_key":"k1","b":"PUBLIC_KEEP_B","h":"curl -H \"Authorization: Digest username=\\\"bob\\\", response=\\\"` + secret + `\\\"\" https://PUBLIC_KEEP.example.com"}`,
			wantValid: true,
		},
		// Word-joined single-quoted header preserves trailing URL.
		{in: `curl -H 'X-API-Key: prefix'` + secret + ` https://PUBLIC_KEEP.example.com`},
		// Standalone escaped quotes preserve subsequent lines and shell-separated commands.
		{in: "ERROR: curl -H \\\"Authorization: Bearer \\\"'" + secret + "' https://PUBLIC_KEEP.example.com\nNEXT_LINE_KEEP \"quoted\""},
		{in: `ERROR: curl -H \"X-Api-Key: \"prefix\"` + secret + `\"; echo PUBLIC_KEEP`},
		{in: `ERROR: curl -H \"X-Api-Key: \"prefix\"\"` + secret + `\"; echo unclosed`},
		// Escaped backslash before closing inner quote (bs%4 == 1) inside JSON string.
		{
			in:        `{"b":"PUBLIC_KEEP_B","h":"curl -H \"Authorization: Bearer ` + secret + `\\\\\" https://PUBLIC_KEEP.example.com"}`,
			wantValid: true,
		},
		// Word-adjacent single-quoted shell fragments ending in '=', space, or ';' with trailing apostrophe comments.
		{in: `curl -H "X-Api-Key: "prefix'a ` + secret + `=' # user's_note`},
		{in: `curl -H "X-Api-Key: "prefix'a ` + secret + ` ' # user's_note`},
		{in: `curl -H "X-Api-Key: "prefix'a ` + secret + ` '; echo 'unclosed`},
		{in: `curl -H "X-Api-Key: "prefix'a;` + secret + `;' # user's_note`},
		{in: `curl -H 'X-Api-Key: 'prefix'a ` + secret + ` ' # user's_note`},
		{in: `curl -H "X-Api-Key: "prefix's ` + secret + `=' # user's_note`},
		{in: `curl -H "X-Api-Key: "prefix's ` + secret + ` ' # user's_note`},
		{in: `curl -H "X-Api-Key: "prefix's;` + secret + `;' # user's_note`},
		// Unquoted possessive before a second -H header with even trailing single-quote parity.
		{in: `curl -H "X-Api-Key: "user's -H 'X-Api-Key: ` + secret + `' # bob's`},
		{in: `curl -H "X-Api-Key: "user's -H 'X-Api-Key: ` + secret + `'`},
		// Contraction/possessive before command separator keeps subsequent quoted argument.
		{in: `curl -H "Authorization: Bearer "bob's_` + secret + `; echo 'PUBLIC_KEEP' # bob's`},
		// Prior-line unclosed quote inside multiline JSON container preserves braceDepth.
		{
			in:           "{\n  \"unclosed\": \"x\n  \"h\": \"Authorization: Bearer " + secret + "\",\"b\": \"PUBLIC_KEEP\"\n}",
			malformedErr: true,
		},
		// Same-line possessives, JSON-escaped line breaks, inch marks, and multiline follow-on headers across all redaction paths.
		{in: `curl -H "X-Api-Key: "Bob's Authorization: 'Bearer ` + secret + `'`},
		{in: `curl -H "X-Api-Key: "Bob's Cookie: 'theme=dark; sid=` + secret + `'`},
		{in: `curl -H "X-Api-Key: "v2's Authorization: 'Bearer ` + secret + `'`},
		{in: `curl -H "X-Api-Key: "SECRET1's Authorization: 'Bearer ` + secret + `'`},
		{in: "I ran curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer " + secret + "' and it failed\nPUBLIC_KEEP"},
		{in: `curl -H "X-Api-Key: "Bob's Authorization: 'Bearer ` + secret + `' didn't work`},
		{in: `curl -H 'X-Api-Key: 'size12" Cookie: "theme=dark; sid=` + secret + `" PUBLIC_KEEP`},
		{
			in:        `{"cmd": "curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer ` + secret + `'\"", "note": "PUBLIC_KEEP"}`,
			wantValid: true,
		},
		{
			in:        `{"cmd": "curl -H \"X-Api-Key: \"Bob's Authorization: 'Bearer ` + secret + `'", "note": "PUBLIC_KEEP"}`,
			wantValid: true,
		},
		{
			in:        `{"cmd": "curl -H \"X-Api-Key: \"Bob's\nAuthorization: 'Bearer ` + secret + `'\"", "note": "PUBLIC_KEEP"}`,
			wantValid: true,
		},
		{in: "curl -H \"X-Api-Key: \"Bob's\nCookie: 'theme=dark; sid=" + secret + "'"},
		{in: "curl -H \"X-Api-Key: \"Bob's\r\nCookie: 'theme=dark; sid=" + secret + "'"},
		{in: "> curl -H \"X-Api-Key: \"Bob's demo key\n< HTTP/1.1 200 OK PUBLIC_KEEP\n< Set-Cookie: 'theme=dark; session=" + secret + "'"},
		{in: "curl -H 'X-Api-Key: 'size12\"\nCookie: \"theme=dark; sid=" + secret + "\""},
		{in: "curl -H \"X-Api-Key: \"Bob's\nCookie: 'theme=dark; sid=" + secret + "'\nNote: Alice's PUBLIC_KEEP"},
		{in: "curl -H \"X-Api-Key: \"Bob's\nX-Note: 'PUBLIC_KEEP'"},
		{in: "curl -H \\\"X-Api-Key: \\\"pre' " + secret + "\nX-Note: '\necho PUBLIC_KEEP"},
	} {
		checkNoLeak(t, tt.in, secret)
		if got, fresh := redactAuthHeaders(tt.in), redactAuthHeadersFreshScanner(tt.in); got != fresh {
			t.Errorf("redactAuthHeaders(%q) = %q != fresh %q", tt.in, got, fresh)
		} else if tt.malformedErr && !strings.Contains(got, "PUBLIC_KEEP") {
			t.Errorf("redactAuthHeaders(%q) = %q, lost PUBLIC_KEEP", tt.in, got)
		}
		if !tt.malformedErr && strings.Contains(tt.in, "PUBLIC_KEEP") {
			if errMsg := redactErrorMessage(tt.in); !strings.Contains(errMsg, "PUBLIC_KEEP") {
				t.Errorf("redactErrorMessage(%q) = %q, lost PUBLIC_KEEP", tt.in, errMsg)
			}
		}
		if tt.wantValid {
			raw, _ := sanitizeRawText(tt.in)
			if !json.Valid([]byte(raw)) {
				t.Errorf("sanitizeRawText(%q) = %q, want valid JSON", tt.in, raw)
			}
			if !strings.Contains(raw, "PUBLIC_KEEP") && strings.Contains(tt.in, "PUBLIC_KEEP") {
				t.Errorf("sanitizeRawText(%q) = %q, lost PUBLIC_KEEP", tt.in, raw)
			}
		}
		if raw, _, err := SmartTruncate(tt.in, 1<<20); err != nil || strings.Contains(string(raw), secret) {
			t.Errorf("SmartTruncate(%q, 1<<20) = (%q, %v), leaked or errored", tt.in, raw, err)
		}
		col, _, err := SmartTruncate(map[string]any{"err": tt.in}, 1000)
		if err != nil {
			t.Fatalf("SmartTruncate(%q) error = %v", tt.in, err)
		}
		if strings.Contains(string(col), secret) {
			t.Errorf("SmartTruncate(%q) = %q, which still holds the secret", tt.in, col)
		}
	}
	for _, tc := range []struct {
		in, want string
	}{
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\ntail'\\''x'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'tail'\\''x'; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
		},
	} {
		checkNoLeak(t, tc.in, secret)
		if got := scrubTextPatterns(tc.in); got != tc.want {
			t.Errorf("scrubTextPatterns(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if got := redactAuthHeaders(tc.in); got != tc.want {
			t.Errorf("redactAuthHeaders(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if got, fresh := redactAuthHeaders(tc.in), redactAuthHeadersFreshScanner(tc.in); got != fresh {
			t.Errorf("redactAuthHeaders(%q) = %q != fresh %q", tc.in, got, fresh)
		}
	}
	const secondSecret = "SYNTH_REVIEW_SECOND_CREDENTIAL"
	for _, tc := range []struct {
		in      string
		want    string
		wantErr string
		forbid  string
		keepPub bool
	}{
		// Nested auth or sensitive headers closing a joined or inner multiline quote and spanning a newline.
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'Bearer\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:     "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'Bearer tok1\n" + secondSecret,
			want:   `curl -H "X-Api-Key: [REDACTED]"`,
			forbid: "tok1",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'Bearer " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'Bearer tok1\n" + secondSecret + "'; echo \"unpaired\nPUBLIC_KEEP",
			want:    "curl -H \"X-Api-Key: [REDACTED]\"; echo \"unpaired\nPUBLIC_KEEP",
			forbid:  "tok1",
			keepPub: true,
		},
		{
			in:     "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; x=2\n" + secondSecret,
			want:   `curl -H "X-Api-Key: [REDACTED]"`,
			forbid: "x=2",
		},
		{
			in:     "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; x=" + secondSecret,
			want:   `curl -H "X-Api-Key: [REDACTED]"`,
			forbid: "sid=1",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'theme=dark; secret=" + secondSecret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' (" + secret + "); echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users'(" + secret + ")",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + "\"; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' --tok-" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -a " + secret + "\"; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' --custom=" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		// Final-line nested Cookie / Authorization tails with semicolons or pipes and trailing-apostrophe non-header flags / stray parens.
		{
			in:     "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; OPAQUE_KEY_FRAG_2222",
			want:   `curl -H "X-Api-Key: [REDACTED]"`,
			forbid: "OPAQUE_KEY_FRAG_2222",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; " + secondSecret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; true; " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; echo " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; env " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; sudo " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1; curl -u user:" + secondSecret + " https://example.com",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in: `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + "\nCookie: 'sid=1; true; " + secondSecret + `","ok":"PUBLIC_KEEP"}`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + `\nCookie: 'sid=1; true; ` + secondSecret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:     "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'tok1|tok_part2_alpha_9911",
			want:   `curl -H "X-Api-Key: [REDACTED]"`,
			forbid: "tok_part2_alpha_9911",
		},
		{
			in:   "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: 'Bearer tok1; " + secondSecret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -a \"" + secret + "\"; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -p " + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' --token " + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' --token \"` + secret + `\"","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' --header=X-Custom:" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -H=X-Custom:" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users')" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' )" + secret + "; echo PUBLIC_KEEP",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      "(curl -H \"X-Api-Key: \"users') && echo PUBLIC_KEEP",
			want:    `(curl -H "X-Api-Key: [REDACTED]") && echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:   "(curl -H \"X-Api-Key: \"users')" + secret + "; echo PUBLIC_KEEP",
			want: `(curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      "curl -H \"X-Api-Key: \"users' -H \"X-Api-Key: " + secret + "\" \"PUBLIC_KEEP\"",
			keepPub: true,
		},
		{
			in:      `wget --header="X-Api-Key: "users' --header="X-Api-Key: ` + secret + `" "PUBLIC_KEEP"`,
			keepPub: true,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + " -H \"Authorization: Bearer " + secondSecret + "\"",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -a " + secret + " -H \"Authorization: Bearer " + secondSecret + "\"",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -H \"X-Custom: " + secret + "\" -H \"Authorization: Bearer " + secondSecret + "\"",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + "; curl -H \"Authorization: Bearer " + secondSecret + "\"",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -a \"" + secret + "\"; curl -H \"Authorization: Bearer " + secondSecret + "\"",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -tok_` + secret + `; curl -H \"Authorization: Bearer ` + secondSecret + `\"","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -H \"Authorization: Bearer ` + secret + `\"","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:   `echo "(note)" && curl -H "X-Api-Key: "users') ` + secret + `; echo PUBLIC_KEEP`,
			want: `echo "(note)" && curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in: "curl -H \"X-Api-Key: \"pre(mid users') " + secret + "; echo PUBLIC_KEEP",
		},
		{
			in:   "(curl -H \"X-Api-Key: \"users') " + secret + "; echo PUBLIC_KEEP",
			want: `(curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + " -H 'Authorization: Bearer " + secondSecret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -a " + secret + " -H 'Authorization: Bearer " + secondSecret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -tok_" + secret + "; curl -H 'Authorization: Bearer " + secondSecret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' -H \"X-Custom: " + secret + "\" -H 'Authorization: Bearer " + secondSecret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users' " + secret + " -H 'Authorization: Bearer " + secondSecret + "'",
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -tok_` + secret + `; curl -H 'Authorization: Bearer ` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -H \"X-Custom: ` + secret + `\" -H 'Authorization: Bearer ` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -H \"Accept: text/html;q=0.9\" -H 'Authorization: Bearer ` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:   `echo "(note)" && curl -H "X-Api-Key: "users') && curl -H 'Authorization: Bearer ` + secret + `'`,
			want: `echo "(note)" && curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `(curl -H "X-Api-Key: "users') && curl -H "X-Api-Key: "users'); ` + secret,
			want: `(curl -H "X-Api-Key: [REDACTED]") && curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `(echo it's) && curl -H "X-Api-Key: "users'); ` + secret,
			want: `(echo it's) && curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"(echo it's) && curl -H \"X-Api-Key: \"users'); ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:   `echo "(note)" && curl -H "X-Api-Key: "users'); ` + secret,
			want: `echo "(note)" && curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users', ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users'} ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users'] ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "tok', ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "tok", ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "tok"} ` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"tok\", ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"(curl -H \"X-Api-Key: \"users'), ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"(curl -H \"X-Api-Key: \"users')] ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `curl -H "X-Api-Key: "pre(tok; echo PUBLIC_KEEP`,
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      `curl -H "X-Api-Key: "pre(` + secret + `; echo PUBLIC_KEEP`,
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      `$(curl -H "X-Api-Key: "users') && echo PUBLIC_KEEP`,
			want:    `$(curl -H "X-Api-Key: [REDACTED]") && echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      `echo $(curl -H "X-Api-Key: "users') && echo PUBLIC_KEEP`,
			want:    `echo $(curl -H "X-Api-Key: [REDACTED]") && echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"(curl -H \"X-Api-Key: \"users') && echo x","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'sid=1\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nX-Api-Key: 'part1\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nProxy-Authorization: 'Basic\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nX-Auth-Token: 'part1\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nAuthorization: '\nBearer " + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\r\nAuthorization: 'Bearer\r\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    `curl -H "X-Api-Key: [REDACTED]"; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      "curl -H 'X-Api-Key: 'pre\" " + secret + "\nAuthorization: \"Bearer\n" + secondSecret + "\"; echo PUBLIC_KEEP",
			want:    `curl -H 'X-Api-Key: [REDACTED]'; echo PUBLIC_KEEP`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + "\nAuthorization: 'Bearer\n" + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			wantErr: `[UNPARSEABLE_JSON_BLOB]`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + `\nAuthorization: 'Bearer\n` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			want:    `{"cmd":"curl -H \"X-Api-Key: [REDACTED]","ok":"PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: 'pre " + secret + "\nAuthorization: 'Bearer\n" + secondSecret + "'; echo PUBLIC_KEEP",
			want:    "curl -H \"X-Api-Key: [REDACTED]\nAuthorization: [REDACTED]; echo PUBLIC_KEEP",
			keepPub: true,
		},
		// Unclosed joined apostrophe/quote fragments followed by credential text before a mismatched or unpaired line.
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nX-Api-Key: \"part1\n" + secondSecret + "\"; echo PUBLIC_KEEP",
			want:    "curl -H \"X-Api-Key: [REDACTED]\"\nX-Api-Key: [REDACTED]; echo PUBLIC_KEEP",
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"users' " + secret + "\nX-Api-Key: \"part1\n" + secondSecret + "\"; echo PUBLIC_KEEP",
			want:    "curl -H \"X-Api-Key: [REDACTED]\"\nX-Api-Key: [REDACTED]; echo PUBLIC_KEEP",
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"pre' " + secret + "\nCookie: 'tail'; echo \"unpaired\nPUBLIC_KEEP",
			keepPub: true,
		},
		{
			in:      "curl -H \"X-Api-Key: \"users' " + secret + "\nCookie: 'tail'; echo \"unpaired\nPUBLIC_KEEP",
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + `\nCookie: 'tail'; echo \"unpaired PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"\nCookie: 'tail'; echo \"unpaired PUBLIC_KEEP"}`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"pre' ` + secret + `\nCookie: 'theme=dark; sid=` + secondSecret + `'","note":"PUBLIC_KEEP"}`,
			want:    `{"cmd":"curl -H \"X-Api-Key: [REDACTED]","note":"PUBLIC_KEEP"}`,
			wantErr: `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\"\nCookie: [REDACTED]","note":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      "Authorization: 'Bearer " + secret + "'; echo PUBLIC_KEEP",
			want:    "Authorization: [REDACTED]; echo PUBLIC_KEEP",
			keepPub: true,
		},
		{
			in:   "Cookie: 'theme=dark; sid=" + secret + "'; " + secondSecret,
			want: "Cookie: [REDACTED]",
		},
		{
			in:      `{"a":"(echo 'open","b":"' && curl -H \"X-Api-Key: \"users'); ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"a":"(curl -H \"X-Api-Key: \"it's 'open","b":"' && curl -H \"X-Api-Key: \"users'); ` + secret + `","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:   `curl -H "X-Api-Key: "users' -H "Accept: ` + secret + `"`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' -H "Accept: ` + secret + `" -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' -d x=` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' -d x=` + secret + ` -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' --data=x=` + secret + ` -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' -b my_val=` + secret + ` -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users'; env FOO=` + secret + ` curl -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:     `curl -H "X-Api-Key: "users'; env FOO="bar\" curl -H 'Authorization: Bearer ` + secondSecret + `'`,
			forbid: `bar`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' https://api.example.com/v1?x=` + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   `curl -H "X-Api-Key: "users' https://api.example.com/v1?x=` + secret + ` -H 'Authorization: Bearer ` + secondSecret + `'`,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -d x=` + secret + ` -H 'Authorization: Bearer ` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' -H \"Accept: ` + secret + `\" -H 'Authorization: Bearer ` + secondSecret + `'","ok":"PUBLIC_KEEP"}`,
			keepPub: true,
		},
		{
			in:      `{"cmd":"curl -H \"X-Api-Key: \"users' \"https://PUBLIC_KEEP.example.com/v1\"","ok":"OK_KEEP"}`,
			want:    `{"cmd":"curl -H \"X-Api-Key: [REDACTED]\" \"https://PUBLIC_KEEP.example.com/v1\"","ok":"OK_KEEP"}`,
			keepPub: true,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users'\ntok' The cookie: Monster 'unclosed " + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users'\nusers' " + secret,
			want: `curl -H "X-Api-Key: [REDACTED]"`,
		},
		{
			in:   "curl -H \"X-Api-Key: \"users'\nusers' " + secret + "\nAuthorization: Bearer " + secondSecret,
			want: "curl -H \"X-Api-Key: [REDACTED]\"\nAuthorization: [REDACTED]",
		},
	} {
		checkNoLeak(t, tc.in, secret)
		checkNoLeak(t, tc.in, secondSecret)
		authGot, fresh := redactAuthHeaders(tc.in), redactAuthHeadersFreshScanner(tc.in)
		if authGot != fresh || strings.Contains(authGot, secret) || strings.Contains(authGot, secondSecret) || (tc.forbid != "" && strings.Contains(authGot, tc.forbid)) {
			t.Errorf("redactAuthHeaders(%q) = %q (fresh %q), leaked or diverged", tc.in, authGot, fresh)
		}
		if redactAuthHeaders(authGot) != authGot {
			t.Errorf("redactAuthHeaders not idempotent on %q: pass1=%q, pass2=%q", tc.in, authGot, redactAuthHeaders(authGot))
		}
		scrubGot := scrubTextPatterns(tc.in)
		if scrubTextPatterns(scrubGot) != scrubGot || (tc.forbid != "" && strings.Contains(scrubGot, tc.forbid)) {
			t.Errorf("scrubTextPatterns(%q) = %q, pass2 = %q, leaked forbid=%q or not idempotent", tc.in, scrubGot, scrubTextPatterns(scrubGot), tc.forbid)
		}
		sanGot, _ := sanitizeRawText(tc.in)
		if san2, _ := sanitizeRawText(sanGot); san2 != sanGot || (tc.forbid != "" && strings.Contains(sanGot, tc.forbid)) {
			t.Errorf("sanitizeRawText(%q) = %q, pass2 = %q, leaked forbid=%q or not idempotent", tc.in, sanGot, san2, tc.forbid)
		}
		errGot := redactErrorMessage(tc.in)
		if redactErrorMessage(errGot) != errGot || (tc.forbid != "" && strings.Contains(errGot, tc.forbid)) {
			t.Errorf("redactErrorMessage(%q) = %q, pass2 = %q, leaked forbid=%q or not idempotent", tc.in, errGot, redactErrorMessage(errGot), tc.forbid)
		}
		if tc.want != "" {
			checkRedactionPaths(t, tc.in, tc.want)
			if authGot != tc.want {
				t.Errorf("redactAuthHeaders(%q) = %q, want %q", tc.in, authGot, tc.want)
			}
		}
		if tc.wantErr != "" {
			if errGot != tc.wantErr {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, errGot, tc.wantErr)
			}
		} else if tc.want != "" {
			if errGot != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, errGot, tc.want)
			}
		}
		if tc.keepPub {
			if !strings.Contains(scrubGot, "PUBLIC_KEEP") || !strings.Contains(scrubTextPatterns(scrubGot), "PUBLIC_KEEP") {
				t.Errorf("scrubTextPatterns(%q) = %q, lost PUBLIC_KEEP", tc.in, scrubGot)
			}
			if !strings.Contains(sanGot, "PUBLIC_KEEP") {
				t.Errorf("sanitizeRawText(%q) = %q, lost PUBLIC_KEEP", tc.in, sanGot)
			}
			if !strings.Contains(errGot, "PUBLIC_KEEP") {
				t.Errorf("redactErrorMessage(%q) = %q, lost PUBLIC_KEEP", tc.in, errGot)
			}
		}
		if raw, _, err := SmartTruncate(tc.in, 1<<20); err != nil || strings.Contains(string(raw), secret) || strings.Contains(string(raw), secondSecret) || (tc.forbid != "" && strings.Contains(string(raw), tc.forbid)) || (tc.keepPub && !strings.Contains(string(raw), "PUBLIC_KEEP")) {
			t.Errorf("SmartTruncate(%q, 1<<20) = (%q, %v), leaked or lost PUBLIC_KEEP", tc.in, raw, err)
		}
		if col, _, err := SmartTruncate(map[string]any{"err": tc.in}, 1000); err != nil || strings.Contains(string(col), secret) || strings.Contains(string(col), secondSecret) || (tc.keepPub && !strings.Contains(string(col), "PUBLIC_KEEP")) {
			t.Errorf("SmartTruncate(map, %q) = (%q, %v), leaked or lost PUBLIC_KEEP", tc.in, col, err)
		}
	}
	rawJSONFinal := `{"X-Api-Key: "pre' ` + secret + "\nnote Cookie: 'sid=1; " + secondSecret + `","ok":"PUBLIC_KEEP"}`
	if got, fresh := redactAuthHeaders(rawJSONFinal), redactAuthHeadersFreshScanner(rawJSONFinal); got != fresh || strings.Contains(got, secret) || strings.Contains(got, secondSecret) || !strings.Contains(got, `"ok":"PUBLIC_KEEP"`) {
		t.Errorf("redactAuthHeaders(rawJSONFinal) = %q (fresh %q), leaked or lost ok field", got, fresh)
	}
	if scrub := scrubTextPatterns(rawJSONFinal); strings.Contains(scrub, secret) || strings.Contains(scrub, secondSecret) || !strings.Contains(scrub, `"ok":"PUBLIC_KEEP"`) {
		t.Errorf("scrubTextPatterns(rawJSONFinal) = %q, leaked or lost ok field", scrub)
	}
}

func TestSameLineMultiHeaderScalesLinearly(t *testing.T) {
	cases := []struct {
		name     string
		makeLine func(n int) string
	}{
		{
			name: "multi-header",
			makeLine: func(n int) string {
				return strings.Repeat(`-H "X-Api-Key: k" `, n) + `-H "Authorization: Digest username="bob", response="SYNTH_REVIEW_ONLY_CREDENTIAL""`
			},
		},
		{
			name: "apostrophe-dense",
			makeLine: func(n int) string {
				return `curl -H "Authorization: Bearer "` + strings.Repeat("a'b", n) + ` 'unclosed`
			},
		},
		{
			name: "apostrophe-semicolon-200kb",
			makeLine: func(n int) string {
				return strings.Repeat("a'b; 'x' ", n*5) + `prefix's; -H 'X-Api-Key: ` + `SYNTH_REVIEW_ONLY_CREDENTIAL' 'PUBLIC_KEEP'`
			},
		},
		{
			name: "json-wrapped-multi-header",
			makeLine: func(n int) string {
				return `{"cmd":"curl ` + strings.Repeat(`-H \"X-Api-Key: k\" `, n) + `https://example.com"}`
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			small := tc.makeLine(500)
			large := tc.makeLine(5000)
			_ = scrubTextPatterns(small)

			t0 := time.Now()
			outSmall := scrubTextPatterns(small)
			dSmall := time.Since(t0)

			t1 := time.Now()
			outLarge := scrubTextPatterns(large)
			dLarge := time.Since(t1)

			if strings.Contains(outSmall, "SYNTH_REVIEW_ONLY_CREDENTIAL") || strings.Contains(outLarge, "SYNTH_REVIEW_ONLY_CREDENTIAL") {
				t.Fatalf("same-line %s input leaked secret", tc.name)
			}
			if dLarge > 50*time.Millisecond && (dSmall == 0 || dLarge > 30*dSmall) {
				t.Errorf("non-linear scaling for 10x %s: n=500 took %v, n=5000 took %v", tc.name, dSmall, dLarge)
			}
		})
	}

	t.Run("ampersand-dense-5kb-vs-50kb", func(t *testing.T) {
		makeAmpLine := func(n int) string {
			return `curl -H "X-Api-Key: "user's ` + strings.Repeat("a&", n) + ` -H 'X-Api-Key: SYNTH_R30J_AMP_SECRET'`
		}
		small := makeAmpLine(2500)
		large := makeAmpLine(25000)
		_ = scrubTextPatterns(small)

		t0 := time.Now()
		outSmall := scrubTextPatterns(small)
		dSmall := time.Since(t0)

		t1 := time.Now()
		outLarge := scrubTextPatterns(large)
		dLarge := time.Since(t1)

		if strings.Contains(outSmall, "SYNTH_R30J_AMP_SECRET") || strings.Contains(outLarge, "SYNTH_R30J_AMP_SECRET") {
			t.Fatalf("ampersand-dense input leaked secret")
		}
		if dLarge > 500*time.Millisecond || (dLarge > 50*time.Millisecond && (dSmall == 0 || dLarge > 30*dSmall)) {
			t.Errorf("non-linear scaling for 10x ampersand-dense: 5KB took %v, 50KB took %v", dSmall, dLarge)
		}
	})
}

func TestEncodedSensitiveJSONObjectKeys(t *testing.T) {
	const secret = "SYNTH_PROBE_SECRET_99"
	for _, in := range []string{
		`{"pass%77ord": "` + secret + `"}`,
		`{"api%5fkey": "` + secret + `"}`,
		`{"%74emp:x": "` + secret + `"}`,
		`{"pass\u0077ord": "` + secret + `"}`,
		`{"pass\x77ord": "` + secret + `"}`,
		`{"https%3A%2F%2Fadmin%3A` + secret + `%40db.internal":"ok"}`,
		`{"api_key%3D` + secret + `":"ok"}`,
		`{"Bearer\t` + secret + `":"ok"}`,
	} {
		if got := scrubErrorText(in); strings.Contains(got, secret) {
			t.Errorf("scrubErrorText(%q) = %q, leaked secret", in, got)
		}
		if got, _ := sanitizeRawText(in); strings.Contains(got, secret) {
			t.Errorf("sanitizeRawText(%q) = %q, leaked secret", in, got)
		}
		if got := redactErrorMessage(in); strings.Contains(got, secret) {
			t.Errorf("redactErrorMessage(%q) = %q, leaked secret", in, got)
		}
		if raw, _, err := SmartTruncate(in, 1<<20); err != nil || strings.Contains(string(raw), secret) {
			t.Errorf("SmartTruncate(%q, 1<<20) = (%q, %v), leaked or errored", in, raw, err)
		}
		if col, _, err := SmartTruncate(map[string]any{"v": in}, 1<<20); err != nil || strings.Contains(string(col), secret) {
			t.Errorf("SmartTruncate(map, 1<<20) for %q = (%q, %v), leaked or errored", in, col, err)
		}
	}
	for _, benign := range []string{
		`{"line1\nline2":"PUBLIC_KEEP"}`,
		`{"hello%20world":"PUBLIC_KEEP"}`,
	} {
		if got := scrubErrorText(benign); !strings.Contains(got, "PUBLIC_KEEP") {
			t.Errorf("scrubErrorText(%q) = %q, lost PUBLIC_KEEP", benign, got)
		}
	}
}

func TestJSONMapKeyCredentials(t *testing.T) {
	const secret = "SYNTH_GATE1_CRED_99"
	for _, tc := range []struct {
		in       string
		wantKeep bool
	}{
		{in: `{"curl -H \"X-Api-Key: \"prefix\" ` + secret + `\"":"ok"}`, wantKeep: false},
		{in: `{"Authorization: Bearer ` + secret + `":"ok","safe":"PUBLIC_KEEP"}`, wantKeep: true},
		{in: `{"https://admin:` + secret + `@db.internal":"ok"}`, wantKeep: false},
		{in: `{"api_key=` + secret + `":"ok"}`, wantKeep: false},
	} {
		if got := scrubErrorText(tc.in); strings.Contains(got, secret) {
			t.Errorf("scrubErrorText(%q) = %q, leaked secret", tc.in, got)
		}
		if got := redactErrorMessage(tc.in); strings.Contains(got, secret) {
			t.Errorf("redactErrorMessage(%q) = %q, leaked secret", tc.in, got)
		} else if tc.wantKeep && !strings.Contains(got, "PUBLIC_KEEP") {
			t.Errorf("redactErrorMessage(%q) = %q, lost PUBLIC_KEEP", tc.in, got)
		}
		if got, _ := sanitizeRawText(tc.in); strings.Contains(got, secret) {
			t.Errorf("sanitizeRawText(%q) = %q, leaked secret", tc.in, got)
		} else if tc.wantKeep && !strings.Contains(got, "PUBLIC_KEEP") {
			t.Errorf("sanitizeRawText(%q) = %q, lost PUBLIC_KEEP", tc.in, got)
		}
		if raw, _, err := SmartTruncate(tc.in, 1<<20); err != nil || strings.Contains(string(raw), secret) {
			t.Errorf("SmartTruncate(%q, 1<<20) = (%q, %v), leaked or errored", tc.in, raw, err)
		} else if tc.wantKeep && !strings.Contains(string(raw), "PUBLIC_KEEP") {
			t.Errorf("SmartTruncate(%q, 1<<20) = %q, lost PUBLIC_KEEP", tc.in, raw)
		}
		if col, _, err := SmartTruncate(map[string]any{"v": tc.in}, 1<<20); err != nil || strings.Contains(string(col), secret) {
			t.Errorf("SmartTruncate(map, 1<<20) for %q = (%q, %v), leaked or errored", tc.in, col, err)
		}
	}
	for _, m := range []map[string]any{
		{"pass%77ord": secret, "safe": "PUBLIC_KEEP"},
		{"\u0070assword": secret},
		{"api\x5fkey": secret},
		{"Authorization: Bearer " + secret: "ok", "safe": "PUBLIC_KEEP"},
	} {
		out, _, err := SmartTruncate(m, 1<<20)
		if err != nil {
			t.Fatalf("SmartTruncate(%#v) error = %v", m, err)
		}
		if strings.Contains(string(out), secret) {
			t.Errorf("SmartTruncate(%#v) = %q, leaked secret", m, out)
		}
		if _, hasSafe := m["safe"]; hasSafe && !strings.Contains(string(out), "PUBLIC_KEEP") {
			t.Errorf("SmartTruncate(%#v) = %q, lost PUBLIC_KEEP", m, out)
		}
	}
}

func TestJoinedHeaderQuotesApostropheFamilies(t *testing.T) {
	const secret = "SYNTH_REVIEW_ONLY_CREDENTIAL"
	const otherSecret = "SYNTH_SECOND_HEADER_CREDENTIAL"
	cases := []struct {
		name     string
		in       string
		wantKeep string
	}{
		{
			name: "familyB_unclosed_single_quote_after_contraction",
			in:   `user's -H 'X-Api-Key: ` + secret,
		},
		{
			name: "familyB_hash_inside_single_quote_after_contraction",
			in:   `user's -H 'X-Api-Key: pre # ` + secret + `'`,
		},
		{
			name: "familyB_trailing_contraction_after_single_quote",
			in:   `user's -H 'X-Api-Key: ` + secret + `'s`,
		},
		{
			name:     "familyA_semicolon_header_before_quoted_keep",
			in:       `prefix's; -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "familyA_flag_header_before_quoted_keep",
			in:       `prefix's -a -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "familyA_plural_possessive_semicolon_header_before_quoted_keep",
			in:       `prefixes'; -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_familyA_semicolon_header_before_quoted_keep",
			in:       `curl -H "X-Api-Key: "prefix's; ` + secret + `=' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_familyA_flag_header_before_quoted_keep",
			in:       `curl -H "X-Api-Key: "prefix's -a ` + secret + `=' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_familyA_semicolon_arg_header_before_quoted_keep",
			in:       `curl -H "X-Api-Key: "prefix's; ` + secret + ` ' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_familyA_plural_possessive_semicolon_header_before_quoted_keep",
			in:       `curl -H "X-Api-Key: "prefixes'; ` + secret + `=' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_familyA_space_inside_quoted_keep",
			in:       `curl -H "X-Api-Key: "prefix's; ` + secret + `=' ' PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name: "curl_familyB_unclosed_single_quote_after_contraction",
			in:   `curl -H "X-Api-Key: "user's -H 'X-Api-Key: ` + secret,
		},
		{
			name: "curl_familyB_hash_inside_single_quote_after_contraction",
			in:   `curl -H "X-Api-Key: "user's -H 'X-Api-Key: pre # ` + secret + `'`,
		},
		{
			name: "curl_familyB_trailing_contraction_after_single_quote",
			in:   `curl -H "X-Api-Key: "user's -H 'X-Api-Key: ` + secret + `'s`,
		},
		{
			name: "curl_p0_2_double_quote_hash_note_bypass",
			in:   `curl -H "X-Api-Key: "prefix's; ` + secret + `=' -d "users' # note"`,
		},
		{
			name: "curl_p0_2_semicolon_echo_unclosed_single_quote",
			in:   `curl -H "X-Api-Key: "prefix's; ` + secret + `='; echo ' unclosed`,
		},
		{
			name: "curl_p0_2_flag_echo_unclosed_single_quote",
			in:   `curl -H "X-Api-Key: "prefix's -a ` + secret + `='; echo ' unclosed`,
		},
		{
			name: "curl_p0_2_semicolon_echo_empty_single_quote",
			in:   `curl -H "X-Api-Key: "prefix's; ` + secret + `='; echo '`,
		},
		{
			name: "curl_p0_2_plural_possessive_semicolon_echo_empty_single_quote",
			in:   `curl -H "X-Api-Key: "prefixes'; ` + secret + `='; echo '`,
		},
		{
			name: "curl_multi_header_short_flag_cluster_sS",
			in:   `curl -H "X-Api-Key: "user's -sS -H 'X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_multi_header_long_flag_silent",
			in:   `curl -H "X-Api-Key: "user's --silent -H 'X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_multi_header_flag_with_arg_X_POST",
			in:   `curl -H "X-Api-Key: "user's -X POST -H 'X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_multi_header_url_arg_before_second_header",
			in:   `curl -H "X-Api-Key: "user's https://api.example.com/v1 -H 'X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_multi_header_plural_possessive_insecure_goog_key",
			in:   `curl -H "X-Api-Key: "users' --insecure -H 'X-Goog-Api-Key: ` + secret + `'`,
		},
		{
			name: "wget_multi_header_qO_long_header",
			in:   `wget --header="X-Api-Key: "user's -qO- --header='X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_multi_header_X_POST_unclosed_single_quote",
			in:   `curl -H "X-Api-Key: "user's -X POST -H 'X-Api-Key: ` + secret,
		},
		{
			name: "curl_multi_header_semicolon_curl_sS_H",
			in:   `curl -H "X-Api-Key: "prefix's; curl -sS -H 'X-Api-Key: ` + secret + `'`,
		},
		{
			name: "curl_semicolon_secret_before_H_header",
			in:   `curl -H "X-Api-Key: "prefix's; ` + secret + ` -H 'X-Api-Key: ` + otherSecret + `'`,
		},
		{
			name: "curl_semicolon_secret_eq_header",
			in:   `curl -H "X-Api-Key: "prefix's; ` + secret + `='X-Api-Key: ` + otherSecret + `'`,
		},
		{
			name:     "curl_multi_header_valued_flag_d_eq",
			in:       `curl -H "X-Api-Key: "user's -d a=1 -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_valued_flag_long_data_eq",
			in:       `curl -H "X-Api-Key: "user's --data foo=bar -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_valued_flag_F_eq",
			in:       `curl -H "X-Api-Key: "user's -F mode=prod -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_valued_flag_b_eq",
			in:       `curl -H "X-Api-Key: "user's -b theme=dark -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_non_auth_header_with_eq",
			in:       `curl -H "X-Api-Key: "user's -H "X-Trace: k=v" -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_quoted_semicolon_in_header",
			in:       `curl -H "X-Api-Key: "user's -H "Accept: text/html;q=0.9" -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_quoted_amp_in_data",
			in:       `curl -H "X-Api-Key: "user's -d "a=1&b=2" -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_quoted_url_with_amp",
			in:       `curl -H "X-Api-Key: "user's "https://api.example.com/v1?a=1&b=2" -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_fd_redirect_2_gt_amp_1",
			in:       `curl -H "X-Api-Key: "user's 2>&1; curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_fd_redirect_amp_gt_dev_null",
			in:       `curl -H "X-Api-Key: "user's &>/dev/null; curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_subshell_and_curl",
			in:       `(curl -H "X-Api-Key: "user's) && curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_and_env_DEBUG_curl",
			in:       `curl -H "X-Api-Key: "user's && env DEBUG=1 curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_semicolon_sudo_curl",
			in:       `curl -H "X-Api-Key: "user's; sudo curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_semicolon_usr_bin_curl",
			in:       `curl -H "X-Api-Key: "user's; /usr/bin/curl -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name: "curl_r30j_non_single_quote_double_quote_before_H",
			in:   `curl -H 'X-Api-Key: 'prefix"; SYNTH_R30J_S1 -H "X-Api-Key: SYNTH_R30J_O1"`,
		},
		{
			name: "curl_r30j_non_single_quote_double_quote_eq_header",
			in:   `curl -H 'X-Api-Key: 'prefix"; SYNTH_R30J_S2="X-Api-Key: SYNTH_R30J_O2"`,
		},
		{
			name: "curl_r30j_escaped_quote_before_H",
			in:   `curl -H \"X-Api-Key: \"prefix\'s; SYNTH_R30J_S3 -H \'X-Api-Key: SYNTH_R30J_O3\'`,
		},
		{
			name: "curl_r30j_long_space_padded_header_over_64_bytes",
			in:   `curl -H 'X-Api-Key: 'prefix's; SYNTH_R30J_S4 -H 'X-Api-Key` + strings.Repeat(" ", 70) + `: SYNTH_R30J_O4'`,
		},
		{
			name:     "curl_multi_header_plural_possessive_short_flag_H_keep",
			in:       `curl -H "X-Api-Key: "users' -a -H 'X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name:     "curl_multi_header_plural_possessive_long_header_eq_keep",
			in:       `wget --header="X-Api-Key: "users' --header='X-Api-Key: ` + secret + `' 'PUBLIC_KEEP'`,
			wantKeep: "PUBLIC_KEEP",
		},
		{
			name: "curl_multi_header_reject_custom_header_before_single_quoted",
			in:   `curl -H "X-Api-Key: "users' -H "X-Custom: ` + secret + `" -H 'Authorization: Bearer ` + otherSecret + `'`,
		},
		{
			name: "curl_multi_header_reject_userinfo_url_before_single_quoted",
			in:   `curl -H "X-Api-Key: "users' https://admin:` + secret + `@api.example.com/v1 -H 'X-Api-Key: ` + otherSecret + `'`,
		},
		{
			name: "curl_multi_header_reject_env_secret_var_before_curl",
			in:   `curl -H "X-Api-Key: "user's; env API_KEY=` + secret + ` curl -H 'X-Api-Key: ` + otherSecret + `'`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got, fresh := redactAuthHeaders(tc.in), redactAuthHeadersFreshScanner(tc.in); got != fresh {
				t.Errorf("redactAuthHeaders(%q) = %q != fresh %q", tc.in, got, fresh)
			}

			raw, _, err := SmartTruncate(tc.in, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(str) error = %v", err)
			}
			col, _, err := SmartTruncate(map[string]any{"err": tc.in}, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(map) error = %v", err)
			}
			san, _ := sanitizeRawText(tc.in)
			rem := redactErrorMessage(tc.in)
			scrub := scrubErrorText(tc.in)

			for label, out := range map[string]string{
				"SmartTruncate(str)": string(raw),
				"SmartTruncate(map)": string(col),
				"sanitizeRawText":    san,
				"redactErrorMessage": rem,
				"scrubErrorText":     scrub,
			} {
				if strings.Contains(out, secret) {
					t.Errorf("%s(%q) = %q, leaked secret", label, tc.in, out)
				}
				if strings.Contains(out, otherSecret) {
					t.Errorf("%s(%q) = %q, leaked otherSecret", label, tc.in, out)
				}
				if strings.Contains(out, "SYNTH_R30J_S") || strings.Contains(out, "SYNTH_R30J_O") {
					t.Errorf("%s(%q) = %q, leaked R30J secret", label, tc.in, out)
				}
				if tc.wantKeep != "" && !strings.Contains(out, tc.wantKeep) {
					t.Errorf("%s(%q) = %q, lost %q", label, tc.in, out, tc.wantKeep)
				}
			}
		})
	}
}

func TestMapKeyCollisionWithSliceOrMapValues(t *testing.T) {
	const secret = "SYNTH_G1_30E_SECRET_99"
	k1 := "Bearer s1_" + secret
	k2 := "Bearer s2_" + secret

	cases := []struct {
		name string
		m    map[string]any
		json string
	}{
		{
			name: "equal_slice_values",
			m:    map[string]any{k1: []any{"a", float64(1)}, k2: []any{"a", float64(1)}},
			json: fmt.Sprintf(`{%q:["a",1],%q:["a",1]}`, k1, k2),
		},
		{
			name: "unequal_slice_values",
			m:    map[string]any{k1: []any{"a"}, k2: []any{"b"}},
			json: fmt.Sprintf(`{%q:["a"],%q:["b"]}`, k1, k2),
		},
		{
			name: "equal_map_values",
			m:    map[string]any{k1: map[string]any{"x": "y"}, k2: map[string]any{"x": "y"}},
			json: fmt.Sprintf(`{%q:{"x":"y"},%q:{"x":"y"}}`, k1, k2),
		},
		{
			name: "unequal_map_values",
			m:    map[string]any{k1: map[string]any{"x": "y1"}, k2: map[string]any{"x": "y2"}},
			json: fmt.Sprintf(`{%q:{"x":"y1"},%q:{"x":"y2"}}`, k1, k2),
		},
		{
			name: "equal_typed_string_slices",
			m:    map[string]any{k1: []string{"p"}, k2: []string{"p"}},
			json: fmt.Sprintf(`{%q:["p"],%q:["p"]}`, k1, k2),
		},
		{
			name: "unequal_typed_string_slices",
			m:    map[string]any{k1: []string{"p"}, k2: []string{"q"}},
			json: fmt.Sprintf(`{%q:["p"],%q:["q"]}`, k1, k2),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for i := 0; i < 50; i++ {
				outMap, _, err := SmartTruncate(tc.m, 1<<20)
				if err != nil {
					t.Fatalf("SmartTruncate(map) error = %v", err)
				}
				outJSON, _, err := SmartTruncate(tc.json, 1<<20)
				if err != nil {
					t.Fatalf("SmartTruncate(json) error = %v", err)
				}
				errWrapped := "failed: " + tc.json
				san, _ := sanitizeRawText(errWrapped)
				rem := redactErrorMessage(errWrapped)
				scrub := scrubErrorText(errWrapped)

				for label, out := range map[string]string{
					"SmartTruncate(map)":  string(outMap),
					"SmartTruncate(json)": string(outJSON),
					"sanitizeRawText":     san,
					"redactErrorMessage":  rem,
					"scrubErrorText":      scrub,
				} {
					if strings.Contains(out, secret) {
						t.Errorf("%s leaked secret: %q", label, out)
					}
				}
			}
		})
	}
}

func TestTempPrefixedSensitiveMapKeys(t *testing.T) {
	const keySecret = "SYNTH_G1_30E_KEY_SECRET_99"
	const valSecret = "SYNTH_G1_30E_VAL_SECRET_88"
	mapCases := []struct {
		name string
		m    map[string]any
		raw  string
	}{
		{
			name: "temp_api_key_assignment",
			m:    map[string]any{"temp:api_key=" + keySecret: valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"temp:api_key=%s":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "percent_encoded_temp_bearer",
			m:    map[string]any{"%74emp:Bearer%20" + keySecret: valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"%%74emp:Bearer%%20%s":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "percent_encoded_temp_userinfo_url",
			m:    map[string]any{"%74emp:https%3A%2F%2Fadmin%3A" + keySecret + "%40db.internal": valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"%%74emp:https%%3A%%2F%%2Fadmin%%3A%s%%40db.internal":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "temp_bearer_header",
			m:    map[string]any{"temp:Bearer " + keySecret: valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"temp:Bearer %s":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "user_temp_authorization_bearer",
			m:    map[string]any{"user:temp:Authorization: Bearer " + keySecret: valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"user:temp:Authorization: Bearer %s":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "temp_redacted_prefix_encoded_userinfo_url",
			m:    map[string]any{"temp:[REDACTED] https%3A%2F%2Fadmin%3A" + keySecret + "%40db.internal": valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"temp:[REDACTED] https%%3A%%2F%%2Fadmin%%3A%s%%40db.internal":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "temp_redacted_prefix_encoded_bearer",
			m:    map[string]any{"temp:[REDACTED] Bearer%20" + keySecret: valSecret, "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"temp:[REDACTED] Bearer%%20%s":%q,"safe":"PUBLIC_KEEP"}`, keySecret, valSecret),
		},
		{
			name: "redacted_prefix_encoded_userinfo_url",
			m:    map[string]any{"[REDACTED] https%3A%2F%2Fadmin%3A" + keySecret + "%40db.internal": "ok", "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"[REDACTED] https%%3A%%2F%%2Fadmin%%3A%s%%40db.internal":"ok","safe":"PUBLIC_KEEP"}`, keySecret),
		},
		{
			name: "redacted_prefix_encoded_api_key",
			m:    map[string]any{"[REDACTED] api_key%3D" + keySecret: "ok", "safe": "PUBLIC_KEEP"},
			raw:  fmt.Sprintf(`{"[REDACTED] api_key%%3D%s":"ok","safe":"PUBLIC_KEEP"}`, keySecret),
		},
	}

	for _, tc := range mapCases {
		t.Run(tc.name, func(t *testing.T) {
			outMap, _, err := SmartTruncate(tc.m, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(map) error = %v", err)
			}
			outJSON, _, err := SmartTruncate(tc.raw, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(json) error = %v", err)
			}
			san, _ := sanitizeRawText(tc.raw)
			rem := redactErrorMessage(tc.raw)
			scrub := scrubErrorText(tc.raw)

			for label, out := range map[string]string{
				"SmartTruncate(map)":  string(outMap),
				"SmartTruncate(json)": string(outJSON),
				"sanitizeRawText":     san,
				"redactErrorMessage":  rem,
				"scrubErrorText":      scrub,
			} {
				if strings.Contains(out, keySecret) {
					t.Errorf("%s leaked keySecret: %q", label, out)
				}
				if strings.Contains(out, valSecret) {
					t.Errorf("%s leaked valSecret: %q", label, out)
				}
				if label != "scrubErrorText" && !strings.Contains(out, "PUBLIC_KEEP") {
					t.Errorf("%s lost PUBLIC_KEEP: %q", label, out)
				}
			}
		})
	}

	pyDict := fmt.Sprintf(`failed: {'temp:api_key=%s': '%s', 'safe': 'PUBLIC_KEEP'}`, keySecret, valSecret)
	sanPy, _ := sanitizeRawText(pyDict)
	for label, out := range map[string]string{
		"sanitizeRawText(py)":    sanPy,
		"redactErrorMessage(py)": redactErrorMessage(pyDict),
		"scrubErrorText(py)":     scrubErrorText(pyDict),
	} {
		if strings.Contains(out, keySecret) {
			t.Errorf("%s leaked keySecret: %q", label, out)
		}
		if strings.Contains(out, valSecret) {
			t.Errorf("%s leaked valSecret: %q", label, out)
		}
		if !strings.Contains(out, "PUBLIC_KEEP") {
			t.Errorf("%s lost PUBLIC_KEEP: %q", label, out)
		}
	}
}

// TestNestedJSONKeyScrubbingScalesLinearly checks that recursive scrubbing and
// redaction across nested JSON key documents scales linearly with input size
// without superlinear repetition.
func TestNestedJSONKeyScrubbingScalesLinearly(t *testing.T) {
	buildNestedDoc := func(levels int) string {
		s := "public"
		for range levels {
			b, err := json.Marshal(map[string]any{s: "public"})
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}
			s = string(b)
		}
		return s
	}

	doc9 := buildNestedDoc(9)
	doc11 := buildNestedDoc(11)

	if len(doc9) != 2131 {
		t.Fatalf("doc9 len = %d, want 2131", len(doc9))
	}
	if len(doc11) != 8293 {
		t.Fatalf("doc11 len = %d, want 8293", len(doc11))
	}

	// 1. Verify exact benign-text preservation across all four operations.
	for _, doc := range []string{doc9, doc11} {
		if got := redactErrorMessage(doc); got != doc {
			t.Errorf("redactErrorMessage changed benign nested JSON key document (len %d)", len(doc))
		}
		if got, changed := sanitizeRawText(doc); changed || got != doc {
			t.Errorf("sanitizeRawText changed benign nested JSON key document (len %d)", len(doc))
		}
		if got := scrubErrorText(doc); got != doc {
			t.Errorf("scrubErrorText changed benign nested JSON key document (len %d)", len(doc))
		}
		gotBytes, truncated, err := SmartTruncate(doc, len(doc)+1000)
		if err != nil || truncated {
			t.Errorf("SmartTruncate errored (%v) or truncated (%v) for len %d", err, truncated, len(doc))
		}
		var parsed string
		if err := json.Unmarshal(gotBytes, &parsed); err != nil || parsed != doc {
			t.Errorf("SmartTruncate JSON unmarshal mismatch for len %d", len(doc))
		}
	}

	// 2. Best-of-3 timings asserting levels=11 < 50ms and growth <= 6x.
	bestOf3 := func(doc string) time.Duration {
		var best time.Duration
		for i := range 3 {
			start := time.Now()
			_ = redactErrorMessage(doc)
			_, _ = sanitizeRawText(doc)
			_ = scrubErrorText(doc)
			_, _, _ = SmartTruncate(doc, len(doc)+1000)
			elapsed := time.Since(start)
			if i == 0 || elapsed < best {
				best = elapsed
			}
		}
		return best
	}

	t9 := bestOf3(doc9)
	t11 := bestOf3(doc11)

	t.Logf("nested JSON key scrubbing timings: levels=9 (2131 bytes) took %v; levels=11 (8293 bytes) took %v", t9, t11)

	if t11 >= 50*time.Millisecond {
		t.Errorf("levels=11 took %v, want < 50ms", t11)
	}

	// Bounded growth check (document length grew 8293 / 2131 = 3.89x; growth must be <= 6x).
	base9 := t9
	if minFloor := 100 * time.Microsecond; base9 < minFloor {
		base9 = minFloor
	}
	ratio := float64(t11) / float64(base9)
	if ratio > 6.0 {
		t.Errorf("timing growth levels=11 / levels=9 = %.2fx (%v / %v), want <= 6x", ratio, t11, t9)
	}
}
