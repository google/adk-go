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
	"math"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// The tests in this file state the redaction rules as tables. A row names a
// rule and gives a text and the text that must come out of it.

// redactionCase is one row of a redaction table.
type redactionCase struct {
	name string
	in   string
	want string
}

// textEntryPoints are the functions that redact free text: error messages,
// raw text, the strings of structured content, and text parts.
var textEntryPoints = []struct {
	name   string
	redact func(t *testing.T, s string) string
}{
	{"redactErrorMessage", func(_ *testing.T, s string) string { return redactErrorMessage(s) }},
	{"sanitizeRawText", func(_ *testing.T, s string) string {
		out, _ := sanitizeRawText(s)
		return out
	}},
	{"smartTruncate", truncatedString},
	{"formatContentParts", textPartText},
}

// truncatedString returns what smartTruncate writes for the string s.
func truncatedString(t *testing.T, s string) string {
	t.Helper()
	raw, _, err := smartTruncate(s, -1, true)
	if err != nil {
		t.Fatalf("smartTruncate(%.80q): %v", s, err)
	}
	var out string
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("smartTruncate(%.80q) = %.80s, want a JSON string: %v", s, raw, err)
	}
	return out
}

// textPartText returns the text that formatContentParts writes for a text
// part that holds s.
func textPartText(t *testing.T, s string) string {
	t.Helper()
	parts, _ := formatContentParts(&genai.Content{Parts: []*genai.Part{{Text: s}}}, -1, true)
	if len(parts) != 1 {
		t.Fatalf("formatContentParts(%.80q) returned %d parts, want 1", s, len(parts))
	}
	text, _ := parts[0]["text"].(string)
	return text
}

// checkRedaction checks that every text entry point returns want for the in
// of each row.
func checkRedaction(t *testing.T, tests []redactionCase) {
	t.Helper()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, ep := range textEntryPoints {
				if got := ep.redact(t, tt.in); got != tt.want {
					t.Errorf("%s(%q) = %q, want %q", ep.name, tt.in, got, tt.want)
				}
			}
		})
	}
}

// TestIsSensitiveKey pins the names whose values are redacted wherever a key
// is read: map keys, struct fields, session state keys, JSON keys, URI
// parameters and the keys of key=value text.
func TestIsSensitiveKey(t *testing.T) {
	// The credential names. A key is sensitive when it is one of them in any
	// case, with "_", "-", "." or nothing between its words, alone or behind
	// "user:" and "app:" state prefixes.
	names := []string{
		"access_token", "api_key", "api_key_secret", "api_key_string", "auth_code", "authorization",
		"aws_secret_access_key", "aws_session_token", "bearer", "client_secret", "code_verifier", "credential",
		"credentials", "google_access_id", "id_token", "passphrase", "password", "private_key",
		"proxy_authorization", "refresh_token", "secret", "secret_access_key", "secret_key", "session_token",
		"sig", "signature", "signing_key", "token", "x_amz_credential", "x_amz_signature",
		"x_api_key", "x_auth_token", "x_goog_api_key", "x_goog_credential", "x_goog_security_token", "x_goog_signature",
	}
	if len(names) != len(sensitiveKeys) {
		t.Errorf("the test lists %d credential names and sensitiveKeys holds %d: keep them in step", len(names), len(sensitiveKeys))
	}
	for _, name := range names {
		words := strings.Split(name, "_")
		camel := words[0]
		for _, w := range words[1:] {
			camel += strings.ToUpper(w[:1]) + w[1:]
		}
		for _, key := range []string{
			name, strings.ToUpper(name), camel,
			strings.Join(words, "-"), strings.Join(words, "."), strings.Join(words, ""),
			"user:" + name, "app:" + name, "app:user:" + camel,
		} {
			if !isSensitiveKey(key) {
				t.Errorf("isSensitiveKey(%q) = false, want true", key)
			}
		}
	}

	tests := []struct {
		name string
		keys []string
		want bool
	}{
		{
			name: "compound name that ends in a credential name",
			keys: []string{
				"DB_PASSWORD", "db-password", "dbPassword", "DBPassword", "DBPASSWORD", "spring.datasource.password",
				"my_api_key", "OPENAI_API_KEY", "myAPIKey", "OpenAIAPIKey", "githubAccessToken", "GITHUB_TOKEN",
				"proxyAuthToken", "webhookSecret", "encryptionPassphrase", "sshPrivateKey", "awsSecretAccessKey",
				"HTTP_AUTHORIZATION", "X-Amz-Security-Token", "has_password", "hasPassword", "is_secret", "isSecret",
				"input_token", "output_token", "inputToken",
			},
			want: true,
		},
		{
			name: "token name that is no cursor",
			keys: []string{
				"auth_token", "csrf_token", "fb_page_token", "resync_token", "async_token", "app_sync_token",
				"start_token", "startToken", "newStartToken", "restartPageToken", "resumePageToken", "presumeToken",
				"my_eos_token",
			},
			want: true,
		},
		{
			name: "session state prefix",
			keys: []string{
				"temp:access_token", "temp:anything", "TEMP:x", "temp:", "user:temp:refresh_token", "app:temp:scratch",
				"user:api_key", "app:client_secret", "USER:API_KEY", "app:user:token", "user:db_password", "app:dbPassword",
				" user: temp:x ",
			},
			want: true,
		},
		{
			name: "letter that folds to ASCII",
			keys: []string{
				"AP\u0130_KEY", "CL\u0130ENT_SECRET", "DB_CL\u0130ENT_SECRET", "db_pa\u017f\u017fword", "api_\u212aey",
				"u\u017fer:temp:foo", "app:pa\u017f\u017fword",
			},
			want: true,
		},
		{
			name: "padding around the name",
			keys: []string{" password ", "\ufeffpassword"},
			want: true,
		},
		{
			name: "pagination cursor",
			keys: []string{
				"page_token", "next_page_token", "prev_page_token", "previous_page_token", "nextPageToken",
				"NEXT_PAGE_TOKEN", "NEXTPAGETOKEN", "user:next_page_token", "sync_token", "syncToken", "next_sync_token",
				"nextSyncToken", "continuation_token", "continuationToken", "NextToken", "next_token", "nextToken",
				"prev_token", "previous_token", "startPageToken", "newStartPageToken", "start_page_token",
				"resumeToken", "resume_token", "nextResumeToken",
			},
		},
		{
			name: "tokenizer special token",
			keys: []string{
				"pad_token", "padToken", "PAD_TOKEN", "eos_token", "eot_token", "eotToken", "bos_token", "unk_token",
				"cls_token", "sep_token", "mask_token",
			},
		},
		{
			name: "token counter",
			keys: []string{
				"token_count", "tokenCount", "TOKEN_COUNT", "prompt_tokens", "completion_tokens", "total_tokens",
				"max_tokens", "input_tokens", "output_tokens", "cached_tokens", "token_limit", "token_budget",
				"token_usage", "token_id", "token_type",
			},
		},
		{
			name: "credential name that is not the last word",
			keys: []string{"api_key_id", "password_hash", "secret_manager_project", "secretName", "passkey_enabled", "AP\u0130_KEY_ID"},
		},
		{
			name: "no word boundary before the credential name",
			keys: []string{
				"dbpassword", "xtoken", "XToken", "my_xtoken", "passwordless", "pas\u017fwordless", "monkey", "mon\u212aey",
				"keyboard", "turnkey", "author", "authority", "authorized", "reauthorization", "preauthorization",
				"deauthorization", "unauthorization", "xauthorization", "my_xauthorization",
			},
		},
		{
			name: "name that only matches alone",
			keys: []string{
				"my_bearer", "auth_credentials", "user_credential", "exchangedAuthCredential", "rawAuthCredential",
				"method_signature", "url_signature", "request_sig",
			},
		},
		{
			name: "key is no credential name",
			keys: []string{"key", "KEY", "primary_key", "public_key", "sort_key", "partition_key", "Sec-WebSocket-Key"},
		},
		{
			name: "other separator or no name",
			keys: []string{"db/password", "db password", "user:name", "app:theme", "u\u017fer:theme", "user:", "app:", ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, key := range tt.keys {
				if got := isSensitiveKey(key); got != tt.want {
					t.Errorf("isSensitiveKey(%q) = %v, want %v", key, got, tt.want)
				}
			}
		})
	}
}

// TestSanitizeURIComponents pins what is redacted in a URI: all of it when it
// holds user information or does not parse, a path segment that is a
// credential name with the segment after it, and the values of sensitive
// ";", query and fragment parameters. sanitizeURI redacts the URIs of file
// parts, and the URIs found in text follow the same rules.
func TestSanitizeURIComponents(t *testing.T) {
	const r = "%5BREDACTED%5D" // "[REDACTED]" in a URI component

	tests := []redactionCase{
		// The whole URI.
		{"user and password", "https://user:pass@example.com/x", redactedURIPlaceholder},
		{"user alone", "https://user@example.com/x", redactedURIPlaceholder},
		{"connection string", "postgres://app:hunter2@db:5432/prod?sslmode=require", redactedURIPlaceholder},
		{"invalid escape", "https://example.com/%zz", redactedURIPlaceholder},
		{"invalid host", "http://[::1", redactedURIPlaceholder},

		// Path: a segment that is a credential name, or a "temp:" key, is
		// redacted with the segment after it. A v0.1.0 key redacts only that
		// single following segment (keeping trailing segments like "/items"
		// after "/api_key/secret/items"); post-v0.1.0 keys (secret_key,
		// signing_key, stacked prefixes, or escaped non-ASCII keys) chain
		// across consecutive sensitive segments until the value ends the run.
		{"path key and value", "https://h/api_key/abc123", "https://h/" + r + "/" + r},
		{"path key in kebab case", "https://h/api-key/abc123", "https://h/" + r + "/" + r},
		{"path key in camel case", "https://h/apiKey/abc123", "https://h/" + r + "/" + r},
		{"path key without a value", "https://h/password", "https://h/" + r},
		{"path value before other segments", "https://h/token/SECRET123/details", "https://h/" + r + "/" + r + "/details"},
		{"path temp: key", "https://h/temp:x/abc123", "https://h/" + r + "/" + r},
		{"path key behind state prefixes", "https://h/app:user:temp:x/abc123", "https://h/" + r + "/" + r},
		{"percent-encoded path keys", "https://h/secret%5Fkey/pa%73sword/SECRET123", "https://h/" + r + "/" + r + "/" + r},
		{"path key after an encoded no-break space", "https://h/%C2%A0token/SECRET123", "https://h/" + r + "/" + r},
		{"path key after an encoded byte order mark", "https://h/%EF%BB%BFapi_key/SECRET123", "https://h/" + r + "/" + r},
		{"two path keys", "https://h/secret_key/password/SECRET123", "https://h/" + r + "/" + r + "/" + r},
		{"temp: key before a path key", "https://h/user:temp:a/password/SECRET123", "https://h/" + r + "/" + r + "/" + r},
		{"three path keys", "https://h/secret_key/signing_key/password/SECRET123", "https://h/" + r + "/" + r + "/" + r + "/" + r},
		{"segment after the value stays", "https://h/secret_key/password/val/SAFE_TAIL", "https://h/" + r + "/" + r + "/" + r + "/SAFE_TAIL"},
		{"credential name as the value", "https://example.com/v1/api_key/secret/items", "https://example.com/v1/" + r + "/" + r + "/items"},

		// Path segments match credential names only. Route slugs, file
		// names and compound names stay.
		{"route slug", "https://example.com/account/reset-password/confirm", "https://example.com/account/reset-password/confirm"},
		{"file name in a path", "https://example.com/certs/private.key/download", "https://example.com/certs/private.key/download"},
		{"dotted path segment", "https://h/api.key/abc123", "https://h/api.key/abc123"},
		{"compound path segment", "https://h/DB_PASSWORD/abc123", "https://h/DB_PASSWORD/abc123"},
		{"camelCase compound path segment", "https://h/dbPassword/abc123", "https://h/dbPassword/abc123"},
		{"key path segment", "https://h/key/abc123", "https://h/key/abc123"},
		{"file URI", "file:///etc/ssl/private.key", "file:///etc/ssl/private.key"},

		// Parameters in a path segment.
		{"matrix parameter", "https://h/p;api.key=SECRET123", "https://h/p%3Bapi.key=" + r},
		{"matrix parameter before a segment", "https://h/p;api.key=SECRET123/x", "https://h/p%3Bapi.key=" + r + "/x"},
		{"matrix parameters", "https://h/p;token=SECRET123;v=1", "https://h/p%3Btoken=" + r + "%3Bv=1"},
		{"key=value segment", "https://h/v1/password=SECRET123/next", "https://h/v1/password=" + r + "/next"},
		{"key:value segment", "https://h/token:SECRET123/file.png", "https://h/token:" + r + "/file.png"},

		// Query: a key is read like a map key, so dotted, compound and
		// state-prefixed names match, and cursors and counters do not.
		{"query key", "https://h/p?access_token=abc&x=1", "https://h/p?access_token=" + r + "&x=1"},
		{"dotted query key", "https://h/p?api.key=abc123", "https://h/p?api.key=" + r},
		{"compound query key", "https://h/p?DB_PASSWORD=abc123", "https://h/p?DB_PASSWORD=" + r},
		{"state-prefixed query keys", "https://h/p?user:temp:foo=SECRET123&app:client_secret=SECRET456", "https://h/p?user:temp:foo=" + r + "&app:client_secret=" + r},
		{"signed URL", "https://h/p?X-Goog-Signature=abc&X-Goog-Credential=def&X-Goog-Date=20260101", "https://h/p?X-Goog-Signature=" + r + "&X-Goog-Credential=" + r + "&X-Goog-Date=20260101"},
		{"service page token", "https://h/p?fb_page_token=SECRET123", "https://h/p?fb_page_token=" + r},
		{"query key without a value", "https://h/p?token", "https://h/p?token=" + r},
		{"percent-encoded query key", "https://h/p?access%5Ftoken=SECRET123", "https://h/p?access%5Ftoken=" + r},
		{"query key after a plus", "https://h/p?+token=SECRET123", "https://h/p?+token=" + r},
		{"query key after an encoded plus", "https://h/p?%2Btoken=SECRET123", "https://h/p?%2Btoken=" + r},
		{"query key after a word and a plus", "https://h/p?my+token=SECRET123", "https://h/p?my+token=" + r},
		{"query key after an encoded byte order mark", "https://h/p?%EF%BB%BFtoken=SECRET123", "https://h/p?%EF%BB%BFtoken=" + r},
		{"query key before an encoded no-break space", "https://h/p?token%C2%A0=SECRET123", "https://h/p?token%C2%A0=" + r},
		{"bearer token in a query key", "https://h/p?Bearer+SECRET123+token=VAL", "https://h/p?Bearer+" + r + "+token=" + r},
		{"bearer token in a query value", "https://h/p?q=Bearer+SECRET123", "https://h/p?q=Bearer+" + r},
		{"pair in a query value", "https://h/p?note=password%3DSECRET123", "https://h/p?note=password%3D" + r},
		{"URL in a query value", "https://h/redirect?next=https://inner/p?token=SECRET123", "https://h/redirect?next=https%3A%2F%2Finner%2Fp%3Ftoken%3D" + r},
		{"cursors and counters", "https://h/p?next_page_token=p2&sync_token=s1&token_count=3", "https://h/p?next_page_token=p2&sync_token=s1&token_count=3"},
		{"compound name that ends in key", "https://h/p?primary_key=abc", "https://h/p?primary_key=abc"},
		{"plus in a query value", "https://h/p?q=a+b&sort=asc", "https://h/p?q=a+b&sort=asc"},

		// Fragment.
		{"fragment key", "https://h/cb#access_token=SECRET123&state=ok", "https://h/cb#access_token=" + r + "&state=ok"},
		{"compound fragment key", "https://h/cb#dbPassword=abc123", "https://h/cb#dbPassword=" + r},
		{"query in a fragment route", "https://h/cb#/route?token=SECRET123", "https://h/cb#/route?token=" + r},
		{"plain fragment", "https://h/cb#section-1", "https://h/cb#section-1"},

		// Nothing to redact.
		{"redacted query value", "https://h/p?token=" + r, "https://h/p?token=" + r},
		{"redacted path", "https://h/" + r + "/" + r, "https://h/" + r + "/" + r},
		{"no credential", "gs://bucket/object.png", "gs://bucket/object.png"},
		{"relative reference", "/relative/path?token=SECRET123", "/relative/path?token=" + r},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wantChanged := tt.want != tt.in
			got, changed := sanitizeURI(tt.in)
			if got != tt.want || changed != wantChanged {
				t.Errorf("sanitizeURI(%q) = (%q, %v), want (%q, %v)", tt.in, got, changed, tt.want, wantChanged)
			}
			if again, changed := sanitizeURI(got); again != got || changed {
				t.Errorf("sanitizeURI(%q) = (%q, %v) on a second call, want no change", got, again, changed)
			}
			parts, _ := formatContentParts(&genai.Content{Parts: []*genai.Part{{FileData: &genai.FileData{FileURI: tt.in}}}}, -1, true)
			if len(parts) != 1 || parts[0]["uri"] != tt.want {
				t.Errorf("formatContentParts(file part %q) = %v, want uri %q", tt.in, parts, tt.want)
			}
		})
	}

	// An object_ref names an object that BigQuery can read. A URI that was
	// redacted names none.
	t.Run("redacted Cloud Storage URI gets no object_ref", func(t *testing.T) {
		for _, tt := range []struct {
			uri, want string
			wantRef   bool
		}{
			{"gs://bucket/doc.pdf", "gs://bucket/doc.pdf", true},
			{"gs://bucket/api_key/SECRET123/doc.pdf", "gs://bucket/" + r + "/" + r + "/doc.pdf", false},
		} {
			part := &genai.Part{FileData: &genai.FileData{FileURI: tt.uri}}
			c := collectParts(partsOptions{maxLen: -1, redact: true, offload: true, inlineLimit: 1 << 10}, singlePart(part))
			if len(c.parts) != 1 || c.parts[0]["uri"] != tt.want {
				t.Fatalf("content_parts for %q = %v, want uri %q", tt.uri, c.parts, tt.want)
			}
			if _, hasRef := c.parts[0]["object_ref"]; hasRef != tt.wantRef {
				t.Errorf("content_parts for %q has object_ref = %v, want %v", tt.uri, hasRef, tt.wantRef)
			}
		}
	})

	t.Run("in text", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"query credential", "GET https://h/p?db_password=hunter2&x=1", "GET https://h/p?db_password=" + r + "&x=1"},
			{"path credential", "GET https://h/token/SECRET123/details HTTP/1.1", "GET https://h/" + r + "/" + r + "/details HTTP/1.1"},
			{"user information", "see https://user:pass@example.com/x for details", "see [REDACTED_SENSITIVE_URI] for details"},
			{"connection string in an error", "dial postgres://app:hunter2@db:5432/prod?sslmode=require: timeout", "dial [REDACTED_SENSITIVE_URI]: timeout"},
			{"punctuation after a URI", "(https://h/p?token=SECRET123).", "(https://h/p?token=" + r + ")."},
			{"fragment before a full stop", "redirect to https://app.example.com/cb#access_token=SECRET123&state=1.", "redirect to https://app.example.com/cb#access_token=" + r + "&state=1."},
			{"pair in a path redacts the rest of the path", "fetch https://example.com/v1/password=SECRET123/token=SECRET123/key=SECRET123 done", "fetch https://example.com/v1/password=[REDACTED] done"},
			{"URL in a query value", "https://a/b?u=http://c/d?token=SECRET123 e://f", "https://a/b?u=http%3A%2F%2Fc%2Fd%3Ftoken%3D" + r + " e://f"},
			{"mixed parameters", "https://example.com?monkey=1&next_page_token=p2&token_count=3&my_api_key=SECRET1", "https://example.com?monkey=1&next_page_token=p2&token_count=3&my_api_key=" + r},
			{"vertical tab does not end a URI", "see https://u:p@h/x\vSECRET123 ok", "see [REDACTED_SENSITIVE_URI] ok"},
			{"vertical tab in a path value", "https://h/token/SEC\vRET123", redactedURIPlaceholder},
			{"harmless URIs", "see https://example.com/a?b=c, then ftp://h/x.", "see https://example.com/a?b=c, then ftp://h/x."},
			{"route slug", "err: https://example.com/account/reset-password/confirm failed", "err: https://example.com/account/reset-password/confirm failed"},
			{"file name in a path", "err: https://example.com/certs/private.key/download failed", "err: https://example.com/certs/private.key/download failed"},
			{"delimiter without a scheme", "123://456/789 and ://x stay", "123://456/789 and ://x stay"},
		})
	})
}

// TestRedactTextAndErrorMessages pins what is redacted in free text. Each row
// holds for an error message, raw text, a string of structured content and a
// text part.
func TestRedactTextAndErrorMessages(t *testing.T) {
	// The value of an Authorization, Proxy-Authorization, X-API-Key or
	// API-Key header is redacted to the end of its line.
	t.Run("header", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"Authorization", "Authorization: Bearer abc.def.ghi", "Authorization: [REDACTED]"},
			{"lower case with a Basic credential", "authorization: Basic dXNlcjpwYXNz", "authorization: [REDACTED]"},
			{"Proxy-Authorization", `Proxy-Authorization: Digest username="u", response="SECRET123"`, "Proxy-Authorization: [REDACTED]"},
			{"X-API-Key", "X-API-Key: sk-live-123", "X-API-Key: [REDACTED]"},
			{"API-Key with words after the value", "API-Key: SECRET123 trailing words", "API-Key: [REDACTED]"},
			{"api_key with a pair after the value", "api_key: s1 foo: bar", "api_key: [REDACTED]"},
			{"underscore spelling", "proxy_authorization: SECRET123", "proxy_authorization: [REDACTED]"},
			{"no space after the colon", "X-Api-Key:SECRET123", "X-Api-Key: [REDACTED]"},
			{"tab after the colon", "Proxy-Authorization:\tBasic dXNlcjpwYXNz", "Proxy-Authorization: [REDACTED]"},
			{"dotted capital I", "Author\u0130zation: SECRET123", "Author\u0130zation: [REDACTED]"},
			{"after other text", "request failed: x-goog-api-key: SECRET123", "request failed: x-goog-api-key: [REDACTED]"},
			{"in a command", "curl -H 'Authorization: Bearer SECRET123' https://example.com", "curl -H 'Authorization: [REDACTED]"},
			{"next line stays", "x-api-key: sk-live-123\nrequest id: 42", "x-api-key: [REDACTED]\nrequest id: 42"},
			{"CRLF lines", "HTTP/1.1 401\r\nAuthorization: Bearer SECRET123\r\nContent-Type: text/plain", "HTTP/1.1 401\r\nAuthorization: [REDACTED]\r\nContent-Type: text/plain"},
			{"value on the next line", "Authorization:\n  Bearer SECRET123", "Authorization: [REDACTED]"},
			{"quoted value on the next line", "Authorization:\n  \"SECRET123\"", "Authorization: [REDACTED]"},
			{"value on the next line before another pair", "api_key:\n  SECRET123\nregion: us", "api_key: [REDACTED]\nregion: us"},
			{"redacted value", "Authorization: [REDACTED]", "Authorization: [REDACTED]"},
			{"longer word", "reauthorization: ok my_xauthorization: ok", "reauthorization: ok my_xauthorization: ok"},
			{"no colon", "authorization failed for user bob", "authorization failed for user bob"},
		})
	})

	// The token after "Bearer" is redacted: a bare word, a quoted string, or
	// a bracketed word with what is glued to it.
	t.Run("Bearer credential", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"in a sentence", "sent Bearer abc123 to server", "sent Bearer [REDACTED] to server"},
			{"upper case", "BEARER SECRET123", "BEARER [REDACTED]"},
			{"after a tab", "Bearer\tSECRET123", "Bearer [REDACTED]"},
			{"double-quoted", `bearer "quoted secret" next`, "bearer [REDACTED] next"},
			{"single-quoted", "Bearer 'quoted secret' next", "Bearer [REDACTED] next"},
			{"bracketed", "Bearer [abc]def next", "Bearer [REDACTED] next"},
			{"in parentheses", "got (Bearer SECRET123) back", "got (Bearer [REDACTED]) back"},
			{"two tokens", "Bearer SECRET123, Bearer SECRET456; done", "Bearer [REDACTED], Bearer [REDACTED]; done"},
			{"token that starts with of", "Bearer of-SECRET123", "Bearer [REDACTED]"},
			{"text glued to a redacted token", "Bearer [REDACTED]SECRET123", "Bearer [REDACTED]"},
			{"redacted token", "Bearer [REDACTED]", "Bearer [REDACTED]"},
			{"bearer of", "the bearer of bad news arrived", "the bearer of bad news arrived"},
			{"no token after the word", "token type Bearer, expires soon", "token type Bearer, expires soon"},
		})
	})

	// The token after "Basic" is redacted when it is base64 that is padded
	// or decodes to user:password. A token too long to decode is redacted.
	t.Run("Basic credential", func(t *testing.T) {
		atLimit := strings.Repeat("QUJD", maxBasicCredentialBytes/4)
		checkRedaction(t, []redactionCase{
			{"user and password", "sent Basic dXNlcjpwYXNz to proxy", "sent Basic [REDACTED] to proxy"},
			{"padded", "Basic YWJjZA==", "Basic [REDACTED]"},
			{"too long to decode", "Basic " + atLimit + "QUJD", "Basic [REDACTED]"},
			{"no padding and no colon", "Basic YWJjZGVm", "Basic YWJjZGVm"},
			{"no padding and no colon at the longest decoded length", "Basic " + atLimit, "Basic " + atLimit},
			{"base64 text after the padding", "Basic dXNlcjpwYXNz=A", "Basic dXNlcjpwYXNz=A"},
			{"Basic auth", "Basic auth is not supported", "Basic auth is not supported"},
			{"Basic plan", "Basic plan costs 10 dollars", "Basic plan costs 10 dollars"},
		})
	})

	// A quoted sensitive key before ":" redacts its value: a string, a
	// number, or a whole object or array.
	t.Run("JSON pair in text", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"compound key", `bad config {"DB_PASSWORD": "hunter2"} rejected`, `bad config {"DB_PASSWORD":"[REDACTED]"} rejected`},
			{"padded key", `bad config {" DB_PASSWORD ": "hunter2"} rejected`, `bad config {"DB_PASSWORD":"[REDACTED]"} rejected`},
			{"state key next to a cursor", `bad config {" user:temp:api_key ": "sk-123", " next_page_token ": "p2"}`, `bad config {"user:temp:api_key":"[REDACTED]", " next_page_token ": "p2"}`},
			{"dotted keys", `err: {"api.key": "VAL987654", "private.key": "VAL987654"}`, `err: {"api.key":"[REDACTED]", "private.key":"[REDACTED]"}`},
			{"pairs without braces", `log: "api_key": "a" "token": "b" done`, `log: "api_key":"[REDACTED]" "token":"[REDACTED]" done`},
			{"object value", `upstream said {"credentials": {"user": "u", "pass": "p"}, "code": 7} and failed`, `upstream said {"credentials":"[REDACTED]", "code": 7} and failed`},
			{"key in the object of a kept key", `resp: {"exchangedAuthCredential": {"access_token": "SECRET123"}}`, `resp: {"exchangedAuthCredential": {"access_token":"[REDACTED]"}}`},
			{"number value", `got {"token": 12345, "ok": true} back`, `got {"token":"[REDACTED]", "ok": true} back`},
			{"value with spaces", `got {"password": "a b c", "n": 1} back`, `got {"password":"[REDACTED]", "n": 1} back`},
			{"value with an escaped quote", `got {"password": "a\"b", "n": 1} back`, `got {"password":"[REDACTED]", "n": 1} back`},
			{"array of objects", `rows: [{"api_key": "k1"}, {"api_key": "k2"}] end`, `rows: [{"api_key":"[REDACTED]"}, {"api_key":"[REDACTED]"}] end`},
			{"cursor and counter", `err: {"next_page_token": "p2", "token_count": 3}`, `err: {"next_page_token": "p2", "token_count": 3}`},
		})
	})

	// A sensitive key before "=" or ":" redacts its value, up to the next
	// space, ",", ";" or "&", or to the closing quote.
	t.Run("key=value and key: value", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"environment variable", "DB_PASSWORD=hunter2", "DB_PASSWORD=[REDACTED]"},
			{"export", "export GITHUB_TOKEN=ghp_abc", "export GITHUB_TOKEN=[REDACTED]"},
			{"camelCase key", "dbPassword=hunter2 other=ok", "dbPassword=[REDACTED] other=ok"},
			{"dotted key", "spring.datasource.password=hunter2", "spring.datasource.password=[REDACTED]"},
			{"colon", "my_api_key: sk-123", "my_api_key:[REDACTED]"},
			{"spaces around the equals sign", "password = hunter2", "password=[REDACTED]"},
			{"command-line flag", "mysql --password=hunter2 --host=db", "mysql --password=[REDACTED] --host=db"},
			{"three spellings", "token=a api-key=b apiKey=c", "token=[REDACTED] api-key=[REDACTED] apiKey=[REDACTED]"},
			{"value before a comma", "password:hunter2,user:bob", "password:[REDACTED],user:bob"},
			{"value before a semicolon", "password=hunter2;user=bob", "password=[REDACTED];user=bob"},
			{"value before an ampersand", "a=1&password=hunter2&b=2", "a=1&password=[REDACTED]&b=2"},
			{"one pair on each line", "client_secret=abc\nrefresh_token=def\nid_token=ghi", "client_secret=[REDACTED]\nrefresh_token=[REDACTED]\nid_token=[REDACTED]"},
			{"double-quoted value", `password="a=b" token=SECRET123`, `password="[REDACTED]" token=[REDACTED]`},
			{"single-quoted key and value", "'dbPassword': 'hunter2'", "'dbPassword':'[REDACTED]'"},
			{"escaped quotes", `msg: \"password\": \"SECRET123\" end`, `msg: \"password\":\"[REDACTED]\" end`},
			{"escaped quote in an escaped quoted value", `token=\"a\\\"SECRET\"`, `token=\"[REDACTED]\"`},
			{"state-prefixed key", "user:temp:api_key=sk-123 x=1", "user:temp:api_key=[REDACTED] x=1"},
			{"temp: key", "temp:session=abc next=1", "temp:session=[REDACTED] next=1"},
			{
				"sensitive and plain keys",
				"DB_PASSWORD=S1 my_api_key=S2 auth_token=S3 client-secret=S4 dbPassword=S5 temp:k=S6 user:temp:k=S7 app:client_secret=S8 app:region=us",
				"DB_PASSWORD=[REDACTED] my_api_key=[REDACTED] auth_token=[REDACTED] client-secret=[REDACTED] dbPassword=[REDACTED] temp:k=[REDACTED] user:temp:k=[REDACTED] app:client_secret=[REDACTED] app:region=us",
			},
			{"Go struct", "log: {DBPassword:hunter2 Count:1} done", "log: {DBPassword:[REDACTED] Count:1} done"},
			{"Go struct pointer", "cfg=&{Host:db Password:hunter2 Port:5432}", "cfg=&{Host:db Password:[REDACTED] Port:5432}"},
			{"Go map", "map[password:hunter2 user:bob]", "map[password:[REDACTED] user:bob]"},
			{"credential name after a hyphen", "next-page-token=abc", "next-page-token=[REDACTED]"},
			{"credential name after a dot", "x.sig=abc", "x.sig=[REDACTED]"},
			{"long s", "pa\u017f\u017fword=abc", "pa\u017f\u017fword=[REDACTED]"},
			{"dotted capital I", "AP\u0130_KEY=SECRET123 CL\u0130ENT_SECRET=SECRET456", "AP\u0130_KEY=[REDACTED] CL\u0130ENT_SECRET=[REDACTED]"},
			{"Kelvin sign", "x-goog-api_\u212aey: SECRET123", "x-goog-api_\u212aey: [REDACTED]"},
			{"key after a path", "kv/prod/db.password: SECRET123", "kv/prod/db.password:[REDACTED]"},
			{"key after a Windows path", `C:\certs\client.secret: SECRET123`, `C:\certs\client.secret:[REDACTED]`},
			{"redacted values", "password=[REDACTED] token: [REDACTED] and more", "password=[REDACTED] token: [REDACTED] and more"},
			{"no value", "password=", "password="},
		})
	})

	// When the value of a sensitive key is itself a sensitive key with a
	// value, the inner pair is redacted and the outer key stays. When the
	// value only ends in a key name, both are redacted as one value.
	t.Run("nested key", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"YAML block", "credentials:\n  password: SECRET123\n  username: bob", "credentials:\n  password:[REDACTED]\n  username: bob"},
			{"YAML block under a plain key", "auth:\n  token: SECRET123\n  type: oauth", "auth:\n  token:[REDACTED]\n  type: oauth"},
			{"YAML block with a compound key", "credentials:\n  DB_PASSWORD: SECRET123\n  region: us", "credentials:\n  DB_PASSWORD:[REDACTED]\n  region: us"},
			{"YAML block with a camelCase key", "credentials:\n  dbPassword: SECRET123\n  region: us", "credentials:\n  dbPassword:[REDACTED]\n  region: us"},
			{"YAML block with a quoted key", "credentials:\n  'DB_PASSWORD': 'SECRET123'\n  region: us", "credentials:\n  'DB_PASSWORD':'[REDACTED]'\n  region: us"},
			{"YAML block with a state-prefixed key", "credentials:\n  user:DB_PASSWORD: SECRET123\n  region: us", "credentials:\n  user:DB_PASSWORD:[REDACTED]\n  region: us"},
			{"YAML block with a dotted key", "credentials:\n  api.key: VAL987654\n  region: us", "credentials:\n  api.key:[REDACTED]\n  region: us"},
			{"YAML block with a temp: key", "credentials:\n  temp:foo: SECRET123", "credentials:\n  temp:foo:[REDACTED]"},
			{"credential names on one line", "password: token: SECRET123", "password: token:[REDACTED]"},
			{"state-prefixed dotted key on one line", "credentials: user:private.key: VAL987654", "credentials: user:private.key:[REDACTED]"},
			{"compound key on one line", "credentials: DB_PASSWORD: SECRET123", "credentials:[REDACTED]"},
			{"temp: key with a colon in its name", "credentials:\n  temp:a:b: SECRET123", "credentials:[REDACTED]"},
			{"value that ends in a key name", "password=foo_password: SECRET123 ok=1", "password=[REDACTED] ok=1"},
			{"value that ends in a key name before an error", "password=dev-SECRET123-password: dial tcp", "password=[REDACTED] tcp"},
			{"value that ends in a header name", "my_api_key=prodSECRET123-api-key: permission denied", "my_api_key=[REDACTED]"},
			{"value with a path before a key name", "password:\n  s3cr3t/dbPassword: SECRET123", "password:[REDACTED]"},
		})
	})

	// Names that only look like credential names, and file names that end
	// in ".key" in a path error, are kept.
	t.Run("kept text", func(t *testing.T) {
		kept := []redactionCase{
			{name: "pagination cursors", in: "next_page_token=abc nextPageToken=def sync_token=ghi syncToken=jkl continuation_token=mno NextToken=stu pad_token=pad0"},
			{name: "token counters", in: "prompt_tokens=5 token_count=5 max_tokens=1024 author=bob"},
			{name: "no word boundary", in: "xtoken=5 dbpassword=plain"},
			{name: "resource names", in: "tokenCount=5 api_key_id=abc secretName=projects/p/secrets/s"},
			{name: "credential envelope", in: "exchangedAuthCredential=struct"},
			{name: "signature names", in: "method_signature=foo request_sig=bar"},
			{name: "words that end in key", in: "monkey=ok donkey=ok turnkey=ok keyboard=ok passkey_enabled=true passwordless=true"},
			{name: "key alone", in: "key: value"},
			{name: "credential name without a delimiter", in: "the password is hunter2"},
			{name: "prompt without a value", in: "enter your password:"},
			{name: "time after a credential name", in: "token expired at 12:30"},
			{name: "words after a credential name", in: "error: secret not found"},
			{name: "path error", in: "open /etc/ssl/private.key: no such file or directory"},
			{name: "path error without a directory", in: "stat signing.key: permission denied"},
			{name: "path error with a Windows path", in: `open C:\ssl\secret.key: access denied`},
			{name: "wrapped path errors", in: "tls: failed to load /etc/ssl/api.key: open /etc/ssl/signing.key: permission denied"},
		}
		for i := range kept {
			kept[i].want = kept[i].in
		}
		checkRedaction(t, kept)
		checkRedaction(t, []redactionCase{
			{"credential next to cursors", "startPageToken=abc resumeToken=jkl eot_token=pqr startToken=vwx", "startPageToken=abc resumeToken=jkl eot_token=pqr startToken=[REDACTED]"},
			{".key name with a value", "private.key: SECRET123", "private.key:[REDACTED]"},
			{".key name assigned after a file verb", "open private.key=SECRET123", "open private.key=[REDACTED]"},
			{"quoted .key path with a value", `"/etc/ssl/private.key": "SECRET123"`, `"/etc/ssl/private.key":"[REDACTED]"`},
		})
	})

	// A key, a delimiter or a header that only shows after percent, hex or
	// Unicode escapes are decoded cannot be redacted in place. The whole
	// text is replaced.
	t.Run("escaped credential fails closed", func(t *testing.T) {
		escaped := []redactionCase{
			{name: "percent-encoded letter in a key", in: "pass%77ord=SECRET123"},
			{name: "percent-encoded underscore in a key", in: "api%5Fkey=SECRET123"},
			{name: "percent-encoded equals sign", in: "password%3DSECRET123"},
			{name: "percent-encoded header colon", in: "Authorization%3A Bearer SECRET123"},
			{name: "hex-escaped letter in a key", in: `pa\x73sword=SECRET123`},
			{name: "Unicode-escaped letter in a key", in: `pa\u0073sword=SECRET123`},
			{name: "Unicode-escaped header colon", in: `Authorization\u003a Bearer SECRET123`},
			{name: "Unicode-escaped space after Bearer", in: `Bearer\u0020SECRET123`},
			{name: "Unicode-escaped quotes after text", in: `upstream error: {\u0022dbPassword\u0022: \u0022SECRET123\u0022}`},
			{name: "quote that closes only after unescaping", in: `password=\"SECRET\\\ax monkey=ok tail\"`},
		}
		for i := range escaped {
			escaped[i].want = redactedSensitiveTextPlaceholder
		}
		checkRedaction(t, escaped)
	})

	// Text that a runtime comparison with the earlier patterns used to
	// replace as a whole is kept or redacted in place.
	t.Run("redacted in place", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"JSON error that names a key before a quoted colon", `{"error":"unexpected token \":\" at line 3"}`, `{"error":"unexpected token \":\" at line 3"}`},
			{"JSON error that names a key before a quoted equals sign", `{"msg":"expected secret \"=\" here","code":3}`, `{"msg":"expected secret \"=\" here","code":3}`},
			{"such an error next to a credential", `{"error":"unexpected token \":\" at line 3","password":"hunter2"}`, `{"error":"unexpected token \":\" at line 3","password":"[REDACTED]"}`},
			{"such an error after other text", `upstream: {"error":"unexpected token \":\" at line 3"}`, `upstream: {"error":"unexpected token \":\" at line 3"}`},
			{"JSON document in a string before other text", `"{\"password\":\"SECRETZQ\"}" trailing words`, `"{\"password\":\"[REDACTED]\"}" trailing words`},
			{"padded key in a string before other text", `"{\" password \":\" SECRETZQ \"}" and 'hi'`, `"{\" password \":\"[REDACTED]\"}" and 'hi'`},
			{"bearer token and state key in a query key", "https://example.com/cb?Bearer+SECRET_ONE+user:token=SECRET_TWO", "https://example.com/cb?Bearer+%5BREDACTED%5D+user%3Atoken=%5BREDACTED%5D"},
			{"path key before an escaped no-break space", "https://h/api_key%5Cu00a0/password/SECRETZQ", "https://h/%5BREDACTED%5D/%5BREDACTED%5D/%5BREDACTED%5D"},
		})
	})

	// textKeyPrefixSpans must locate the exact same sensitive key prefixes as
	// an equivalent RE2 expression on hand-picked edge cases and short strings
	// built from key, prefix, quote, escape, and delimiter tokens.
	t.Run("textKeyPrefixSpans matches reference regexp", func(t *testing.T) {
		refRe := regexp.MustCompile(`(?i)(temp:[^=:"'\t\n\f\r ,}]+|[A-Za-z0-9_.\x{0130}\x{017f}\x{212a}-]+)(\\?["'])?[\x00-\x20\x7f]*[=:]`)
		type span struct{ keyStart, keyEnd, quoteEnd, end int }
		fromScanner := func(s string) []span {
			var out []span
			for _, m := range textKeyPrefixSpans(s) {
				if ks := sensitiveKeyStart(s, m.start, m.keyEnd); ks >= 0 {
					out = append(out, span{ks, m.keyEnd, m.quoteEnd, m.end})
				}
			}
			return out
		}
		fromRegexp := func(s string) []span {
			var out []span
			for _, loc := range refRe.FindAllStringSubmatchIndex(s, -1) {
				keyStart, keyEnd, quoteEnd := loc[2], loc[3], loc[3]
				if loc[5] >= 0 {
					quoteEnd = loc[5]
				}
				if ks := sensitiveKeyStart(s, keyStart, keyEnd); ks >= 0 {
					out = append(out, span{ks, keyEnd, quoteEnd, loc[1]})
				}
			}
			return out
		}
		check := func(text string) {
			if got, want := fromScanner(text), fromRegexp(text); !slices.Equal(got, want) {
				t.Errorf("textKeyPrefixSpans(%q) = %v, want %v from reference regexp", text, got, want)
			}
		}
		for _, text := range []string{
			"password=1 token:2 db_password = 3",
			"user:temp:api_key=sk-1 app:user:password:x temp:foo=bar temp:x",
			`"password"=1 'dbPassword':2 \"token\"=3 \'secret\':4`,
			"AP\u0130_KEY=1 pa\u017f\u017fword=2 api_\u212aey=3 next-page-token=4 x.sig=5",
			"token\x01=1 token\x7f:2 token\\=3 token\\\\\"=4 temp:a\"=5 temp:a\\\"=6",
			"a=b c:d === ::: user:next_page_token=1 temp:a:token=2",
		} {
			check(text)
		}
		alpha := []string{"token", "temp", "user", "a", "=", ":", `"`, `'`, `\`, " ", "\x01", "\u0130", "."}
		for _, t1 := range alpha {
			for _, t2 := range alpha {
				for _, t3 := range alpha {
					check(t1 + t2 + t3)
					for _, t4 := range alpha {
						check(t1 + t2 + t3 + t4)
					}
				}
			}
		}
	})
}

// TestRedactJSONAndSmartTruncate pins how structured values and JSON text are
// redacted: by key, with the rest of the document kept.
func TestRedactJSONAndSmartTruncate(t *testing.T) {
	t.Run("map keys and struct fields", func(t *testing.T) {
		type config struct {
			DBPassword        string
			GithubAccessToken string
			NextPageToken     string
			TokenCount        int
		}
		for _, tt := range []struct {
			name  string
			value any
			want  string
		}{
			{
				name: "map",
				value: map[string]any{
					"DB_PASSWORD": "pw", "dbPassword": "pw", "my_api_key": "k", "user:temp:refresh_token": "r",
					"fb_page_token": "fb", "next_page_token": "p2", "pad_token": "pt", "token_count": 3,
					"nested": map[string]any{"GITHUB_TOKEN": "ghp", "credentials": map[string]any{"user": "u"}},
					"list":   []any{map[string]any{"password": "x"}, "password=hunter2"},
				},
				want: `{"DB_PASSWORD":"[REDACTED]","dbPassword":"[REDACTED]","fb_page_token":"[REDACTED]",` +
					`"list":[{"password":"[REDACTED]"},"password=[REDACTED]"],"my_api_key":"[REDACTED]",` +
					`"nested":{"GITHUB_TOKEN":"[REDACTED]","credentials":"[REDACTED]"},"next_page_token":"p2",` +
					`"pad_token":"pt","token_count":3,"user:temp:refresh_token":"[REDACTED]"}`,
			},
			{
				name:  "struct",
				value: config{DBPassword: "pw", GithubAccessToken: "t", NextPageToken: "p2", TokenCount: 3},
				want:  `{"DBPassword":"[REDACTED]","GithubAccessToken":"[REDACTED]","NextPageToken":"p2","TokenCount":3}`,
			},
		} {
			t.Run(tt.name, func(t *testing.T) {
				got, _, err := smartTruncate(tt.value, -1, true)
				if err != nil {
					t.Fatal(err)
				}
				if string(got) != tt.want {
					t.Errorf("smartTruncate = %s, want %s", got, tt.want)
				}
			})
		}
	})

	// A "temp:" key is dropped from a state delta. Behind "user:" or "app:"
	// it stays, and its value is redacted.
	t.Run("state delta", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-state-delta",
			Author: "test-agent",
			Actions: session.EventActions{StateDelta: map[string]any{
				"temp:access_token":       "dropped",
				"user:temp:refresh_token": "rt-secret",
				"app:dbPassword":          "pw-secret",
				"DB_PASSWORD":             "env-secret",
				"token_count":             7,
			}},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		attrs := jsonObjectColumn(t, onlyRowOfType(t, rows, EventStateDelta), "attributes")
		checkJSONAt(t, attrs, `{"DB_PASSWORD":"[REDACTED]","app:dbPassword":"[REDACTED]","token_count":7,"user:temp:refresh_token":"[REDACTED]"}`, "state_delta")
	})

	// Text that is a JSON document is redacted by key and written again
	// with its keys sorted.
	t.Run("JSON document", func(t *testing.T) {
		checkRedaction(t, []redactionCase{
			{"compound key", `{"dbPassword": "hunter2", "note": "ok"}`, `{"dbPassword":"[REDACTED]","note":"ok"}`},
			{"state key", `{"user:temp:api_key": "sk-123"}`, `{"user:temp:api_key":"[REDACTED]"}`},
			{"header map", `{"headers":{"Authorization":"Bearer SECRET123","Accept":"*/*"}}`, `{"headers":{"Accept":"*/*","Authorization":"[REDACTED]"}}`},
			{
				"object value",
				`{"Credentials":{"AccessKeyId":"EXAMPLEKEYID","SecretAccessKey":"SECRET123","SessionToken":"SECRET123"},"Message":"ok"}`,
				`{"Credentials":"[REDACTED]","Message":"ok"}`,
			},
			{"array value", `{"token":["SECRET123","b"],"status":"ok"}`, `{"status":"ok","token":"[REDACTED]"}`},
			{"temp: object", `{"temp:scratch":{"dsn":"SECRET123"},"Message":"ok"}`, `{"Message":"ok","temp:scratch":"[REDACTED]"}`},
			{"header as the value of a credential name", `{"credentials":"Authorization: sample","status":"ready"}`, `{"credentials":"[REDACTED]","status":"ready"}`},
			{"nested containers", `{"a":{"b":{"c":{"password":"x","d":[1,2,{"token":"y"}]}}}}`, `{"a":{"b":{"c":{"d":[1,2,{"token":"[REDACTED]"}],"password":"[REDACTED]"}}}}`},
			{"document in a string", `{"args":"{\"token\":{\"value\":\"SECRET123\"},\"status\":\"ok\"}"}`, `{"args":"{\"status\":\"ok\",\"token\":\"[REDACTED]\"}"}`},
			{"document that is a string", `"{\"credentials\":{\"header\":\"Authorization: sample\"},\"status\":\"ready\"}"`, `"{\"credentials\":\"[REDACTED]\",\"status\":\"ready\"}"`},
			{"numbers and literals", `{"n":1.0,"big":12345678901234567890,"t":true,"z":null,"token":"x"}`, `{"big":12345678901234567890,"n":1.0,"t":true,"token":"[REDACTED]","z":null}`},
			{"duplicate key", `{"a":1,"a":2,"password":"x"}`, `{"a":2,"password":"[REDACTED]"}`},
			{"padding and byte order mark", "\ufeff  {\"password\":\"x\"}  ", `{"password":"[REDACTED]"}`},
			{"string that starts like JSON", `{"a":"[INFO] ready","password":"x"}`, `{"a":"[UNPARSEABLE_JSON_BLOB]","password":"[REDACTED]"}`},
			{"URI with userinfo followed by text in a JSON string", `{"ok":1,"u":"https://u:p@example.com/p ok"}`, `{"ok":1,"u":"[REDACTED_SENSITIVE_URI] ok"}`},
			{"cursor and counter", `{"next_page_token":"p2","token_count":42,"monkey":"a"}`, `{"next_page_token":"p2","token_count":42,"monkey":"a"}`},
			{"error that names a credential", `{"error":{"code":401,"message":"Request had invalid authentication credentials."}}`, `{"error":{"code":401,"message":"Request had invalid authentication credentials."}}`},
		})
	})

	// An error message scrubs each string of a document on its own, so a
	// header in one string leaves the other fields in place. To preserve the
	// v0.1.0 redaction floor (where compound keys were not recognized as JSON
	// keys and a header inside one ran to end-of-line during the text scrub),
	// the other entry points collapse the whole document to
	// "[REDACTED_SENSITIVE_TEXT]" when a compound sensitive key holds a header.
	t.Run("header in a string of a document", func(t *testing.T) {
		for _, tt := range []struct {
			name      string
			in        string
			wantError string // redactErrorMessage
			wantText  string // sanitizeRawText, smartTruncate and formatContentParts
		}{
			{
				name:      "plain key",
				in:        `{"log":"x-api-key: SECRET123\nrequest id: 42","note":"kept"}`,
				wantError: `{"log":"x-api-key: [REDACTED]\nrequest id: 42","note":"kept"}`,
				wantText:  `{"log":"x-api-key: [REDACTED]`,
			},
			{
				name:      "compound sensitive key",
				in:        `{"password":"p","note":"kept","aa_secret":"Authorization: x"}`,
				wantError: `{"aa_secret":"[REDACTED]","note":"kept","password":"[REDACTED]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "compound sensitive key next to an object",
				in:        `{"password":"p","meta":{"note":"kept"},"aa_secret":"Authorization: x"}`,
				wantError: `{"aa_secret":"[REDACTED]","meta":{"note":"kept"},"password":"[REDACTED]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "compound sensitive key and an escaped colon",
				in:        `{"zz_secret":"Authorization\u003a\n","note":"kept"}`,
				wantError: `{"note":"kept","zz_secret":"[REDACTED]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "compound sensitive key in a document in a string",
				in:        `{"outer":"{\"password\":\"p\",\"note\":\"x\",\"aa_secret\":\"Authorization: x\"}","note":"kept"}`,
				wantError: `{"note":"kept","outer":"[REDACTED_SENSITIVE_TEXT]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "nested JSON header re-sorted before plain secret",
				in:        `{"k":"{\"zz_a\":\"Authorization: x\",\"aa_b\":\"SECRETZQ\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`,
				wantError: `{"k":"[UNPARSEABLE_JSON_BLOB]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "nested JSON pair re-sorted before plain secret",
				in:        `{"k":"{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`,
				wantError: `{"k":"[UNPARSEABLE_JSON_BLOB]"}`,
				wantText:  `{"k":"[UNPARSEABLE_JSON_BLOB]"}`,
			},
			{
				name:      "double-escaped BOM before URI token in JSON text",
				in:        `{"s":"https\\ufeff://h/a;Bearer+SECRETZQ+token=x"}`,
				wantError: `{"s":"https\\ufeff://h/a%3BBearer+SECRETZQ+token=%5BREDACTED%5D"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
		} {
			t.Run(tt.name, func(t *testing.T) {
				for _, ep := range textEntryPoints {
					want := tt.wantText
					if ep.name == "redactErrorMessage" {
						want = tt.wantError
					}
					if got := ep.redact(t, tt.in); got != want {
						t.Errorf("%s(%q) = %q, want %q", ep.name, tt.in, got, want)
					}
				}
			})
		}
	})

	// Text that starts with "{", "[" or a quote and is no JSON. An error
	// message, raw text and a text part keep it when nothing in it reads as
	// a credential. smartTruncate keeps no malformed JSON text.
	t.Run("malformed JSON without a credential", func(t *testing.T) {
		for _, tt := range []struct{ name, in string }{
			{"log line", "[INFO] ready"},
			{"log line with a URL", "[INFO] fetched https://example.com/docs/page?id=42 in 12 ms"},
			{"log line with a quoted path", `[INFO] saved to "C:\\" ok`},
			{"log line that names a key", "[INFO] `password`: must be at least 8 characters"},
			{"log line with a path error", "[ERROR] open private.key: no such file or directory"},
			{"Markdown link", "[docs](https://example.com/docs)"},
			{"object that does not end", `{"log":"Enter your password\n:`},
			{"trailing comma", `{"a": 1,}`},
			{"keys without quotes", "{unquoted: json}"},
			{"quoted word before other text", `"[object Object]" is not valid JSON`},
		} {
			t.Run(tt.name, func(t *testing.T) {
				for _, ep := range textEntryPoints {
					want := tt.in
					if ep.name == "smartTruncate" {
						want = unparseableJSONPlaceholder
					}
					if got := ep.redact(t, tt.in); got != want {
						t.Errorf("%s(%q) = %q, want %q", ep.name, tt.in, got, want)
					}
				}
			})
		}
	})

	// Malformed JSON with a credential in it cannot be redacted by key.
	// Every entry point replaces it.
	t.Run("malformed JSON with a credential", func(t *testing.T) {
		failed := []redactionCase{
			{name: "Go struct", in: "{DBPassword:hunter2 Count:1}"},
			{name: "object that does not end", in: `{"X-Amz-Security-Token": "SECRET123"`},
			{name: "text glued to a redacted value", in: `{"password": "[REDACTED]":SECRET123`},
			{name: "escaped quotes", in: `{\"password\": \"SECRET123\"}`},
			{name: "Unicode-escaped quotes", in: "{\\u0022dbPassword\\u0022\u00a0: \\u0022SECRET123\\u0022}"},
			{name: "log line with a pair", in: "[WARN] retrying password=hunter2"},
			{name: "log line with a credential in a URL", in: "[INFO] fetched https://example.com/p?token=SECRET123 in 12 ms"},
			{name: "quoted pair", in: `"password=hunter2"`},
			{name: "quoted URL with user information", in: `"see https://u:p@h/x"`},
			{name: "quoted key with a redacted value", in: `"password":"[REDACTED]" other=ok`},
			{name: "quoted key before a nested key", in: "\"DB_PASSWORD\":\n  password: SECRET123"},
		}
		for i := range failed {
			failed[i].want = unparseableJSONPlaceholder
		}
		checkRedaction(t, failed)
	})

	// Only the first maxInspectableJSONTextBytes bytes of a text are read
	// for quoted keys. A sensitive key after them still fails malformed JSON
	// closed.
	t.Run("sensitive key past the inspected bytes", func(t *testing.T) {
		in := "{" + strings.Repeat("x", maxInspectableJSONTextBytes) + ` "token":`
		for _, ep := range textEntryPoints {
			if got := ep.redact(t, in); got != unparseableJSONPlaceholder {
				t.Errorf("%s(%d bytes that end in a quoted sensitive key) = %.40q, want %q", ep.name, len(in), got, unparseableJSONPlaceholder)
			}
		}
	})
}

// TestRedactionIdempotency checks that redacting text that was redacted
// already changes nothing, through every text entry point and through
// scrubErrorText, the pattern pass that they share.
func TestRedactionIdempotency(t *testing.T) {
	entryPoints := append(slices.Clone(textEntryPoints), struct {
		name   string
		redact func(t *testing.T, s string) string
	}{"scrubErrorText", func(_ *testing.T, s string) string { return scrubErrorText(s) }})

	for _, in := range []string{
		// Headers, Bearer and Basic credentials.
		"Authorization: Bearer abc.def.ghi",
		"Authorization:\n  Bearer SECRET123",
		"sent Bearer abc123 to server",
		"sent Basic dXNlcjpwYXNz to proxy",
		// Pairs and nested keys.
		"DB_PASSWORD=hunter2 other=ok",
		`password="a=b" token=SECRET123`,
		`bad config {"DB_PASSWORD": "hunter2"} rejected`,
		"credentials:\n  DB_PASSWORD: SECRET123\n  region: us",
		"password=foo_password: SECRET123 ok=1",
		// URIs.
		"GET https://h/p?db_password=hunter2&x=1",
		"see https://user:pass@example.com/x for details",
		"https://h/token/SECRET123/details",
		"fetch https://example.com/v1/password=SECRET123/token=SECRET123/key=SECRET123 done",
		// JSON documents.
		`{"dbPassword": "hunter2", "note": "ok"}`,
		`{"Credentials":{"SessionToken":"SECRET123"},"Message":"ok"}`,
		`{"args":"{\"token\":{\"value\":\"SECRET123\"},\"status\":\"ok\"}"}`,
		`{"password":"p","note":"kept","aa_secret":"Authorization: x"}`,
		`{"error":"unexpected token \":\" at line 3"}`,
		`{"u":"https://u:p@example.com/p","ok":1}`,
		// Text that fails closed.
		"pass%77ord=SECRET123",
		"{DBPassword:hunter2 Count:1}",
		"[WARN] retrying password=hunter2",
		// Text that is kept.
		"[INFO] ready",
		"open /etc/ssl/private.key: no such file or directory",
		"next_page_token=abc prompt_tokens=5",
		"plain text",
		"",
		// Placeholders.
		redactedValue,
		redactedURIPlaceholder,
		redactedSensitiveTextPlaceholder,
		unparseableJSONPlaceholder,
	} {
		for _, ep := range entryPoints {
			once := ep.redact(t, in)
			if again := ep.redact(t, once); again != once {
				t.Errorf("%s(%q) = %q, and %q on a second call", ep.name, in, once, again)
			}
		}
	}

	// A second call reads its input like any other text. Output that starts
	// with a quote and is no JSON, with a quoted sensitive key in it, fails
	// closed whatever the value of that key reads as: a placeholder that a
	// text came with says nothing about the text around it.
	t.Run("quoted fragment fails closed when read again", func(t *testing.T) {
		const in, once = `"password": "hunter2" other=ok`, `"password":"[REDACTED]" other=ok`
		for _, ep := range textEntryPoints {
			if got := ep.redact(t, in); got != once {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, once)
			}
			if got := ep.redact(t, once); got != unparseableJSONPlaceholder {
				t.Errorf("%s(%q) = %q, want %q", ep.name, once, got, unparseableJSONPlaceholder)
			}
		}
	})

	// runFixedPoint fails closed if output still changes at maxRedactionPasses.
	t.Run("pass limit fails closed", func(t *testing.T) {
		got, lost := runFixedPoint(`{"a":1}`, func(s string, _ bool) (string, bool, bool) {
			return s + " ", false, false
		})
		if got != redactedSensitiveTextPlaceholder || !lost {
			t.Errorf("runFixedPoint at pass limit = (%q, %v), want (%q, true)", got, lost, redactedSensitiveTextPlaceholder)
		}
	})
}

// TestRedactionLinearTimeScaling checks that the time to redact a text grows
// linearly with its length, on shapes that make a pattern start, a quote
// search or a nested-key lookup rescan the text when they are written
// carelessly. It does not run in parallel with other tests, which would take
// the CPU from the runs that it times.
func TestRedactionLinearTimeScaling(t *testing.T) {
	// Each shape is timed at two lengths 16 times apart. A linear redaction
	// takes about 16 times longer on the large text, a quadratic one about
	// 256 times.
	const (
		sizeRatio = 16
		maxRatio  = 64
	)
	// The race detector slows these single-goroutine scans by more than ten
	// times and adds nothing to check here, so the texts are shorter under it.
	scale := 1
	if raceEnabled {
		scale = 4
	}
	shapes := []struct {
		name string
		// large is the length of the large text, in bytes.
		large int
		// text returns a text of about n bytes.
		text func(n int) string
	}{
		{"URI delimiters without a scheme", 256 << 10, func(n int) string { return strings.Repeat("123://456/789", n/13) }},
		{"bare URI delimiters", 256 << 10, func(n int) string { return strings.Repeat("://", n/3) }},
		{"URIs with a credential", 256 << 10, func(n int) string { return strings.Repeat("see https://h/p?token=SECRET123&x=1 ok\n", n/40) }},
		{"quotes that do not close", 100 << 10, func(n int) string { return strings.Repeat(`token="a `, n/9) }},
		{"quoted fragments after a quoted key", 100 << 10, func(n int) string { return `"api.key": failed: ` + strings.Repeat(`"x `, n/3) }},
		{"backslashes before a pair", 100 << 10, func(n int) string { return strings.Repeat(`\`, n) + " dbPassword=SECRET123" }},
		{"long key name", 256 << 10, func(n int) string { return strings.Repeat("a", n) + "password=x" }},
		{"key name with many dashes", 256 << 10, func(n int) string { return strings.Repeat("a-", n/2) + "page-token=SECRET123" }},
		{"chained nested keys", 100 << 10, func(n int) string { return strings.Repeat("password: ", n/10) + "SECRET123" }},
		{"quoted chain of nested keys", 64 << 10, func(n int) string { return `password="` + strings.Repeat("token: ", n/7) + `x" tail` }},
		{"short quoted values before colons", 256 << 10, func(n int) string { return strings.Repeat(`password:"x":`, n/13) }},
		{"JSON keys before unclosed arrays", 256 << 10, func(n int) string { return `err: "token":x` + strings.Repeat(`"token":[`, n/9) }},
		{"JSON keys before balanced arrays", 256 << 10, func(n int) string {
			return `err: "token":x` + strings.Repeat(`"token":[`, n/10) + strings.Repeat(`]`, n/10) + ` tail`
		}},
		{"quoted containers with nested keys", 128 << 10, func(n int) string {
			return `password="` + strings.Repeat("token={ ", n/10) + strings.Repeat("}", n/10) + `" tail`
		}},
		{"pairs with plain keys", 1 << 20, func(n int) string { return strings.Repeat("a=b ", n/4) }},
		{"bare delimiters", 1 << 20, func(n int) string { return strings.Repeat("=", n) }},
		{"redacted pairs", 256 << 10, func(n int) string { return strings.Repeat("DB_PASSWORD=SECRET1 dbPassword=SECRET2 ", n/39) }},
		{"header lines", 256 << 10, func(n int) string { return strings.Repeat("Authorization: Bearer SECRET\n", n/29) }},
		{"JSON document with pairs in its strings", 256 << 10, func(n int) string { return `{` + strings.Repeat(`"k":"password=abc",`, n/19) + `"password":"x"}` }},
		{"malformed JSON log lines", 1 << 20, func(n int) string {
			return strings.Repeat("[INFO] fetched https://example.com/docs/page?id=42 in 12 ms\n", n/60)
		}},
	}
	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			large := shape.text(shape.large / scale)
			small := shape.text(shape.large / scale / sizeRatio)
			redact := func(s string) {
				for _, ep := range textEntryPoints {
					if out := ep.redact(t, s); strings.Contains(out, "SECRET") {
						t.Fatalf("%s kept a secret: %.60q", ep.name, out)
					}
				}
			}
			// Load on the host only makes a run slower, so the best of
			// three rounds decides.
			ratio := math.Inf(1)
			var tSmall, tLarge time.Duration
			for range 3 {
				ts, tl := fastest(func() { redact(small) }), fastest(func() { redact(large) })
				if r := float64(tl) / float64(max(ts, 50*time.Microsecond)); r < ratio {
					ratio, tSmall, tLarge = r, ts, tl
				}
				if ratio <= maxRatio {
					break
				}
			}
			if ratio > maxRatio {
				t.Errorf("%d bytes took %v and %d bytes took %v: ratio %.0f, want at most %d for a linear redaction", len(small), tSmall, len(large), tLarge, ratio, maxRatio)
			}
		})
	}
}

// fastest returns the shortest of three runs of f.
func fastest(f func()) time.Duration {
	best := time.Duration(math.MaxInt64)
	for range 3 {
		start := time.Now()
		f()
		best = min(best, time.Since(start))
	}
	return best
}
