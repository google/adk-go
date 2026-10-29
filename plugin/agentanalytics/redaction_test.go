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
	"strings"
	"testing"
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
