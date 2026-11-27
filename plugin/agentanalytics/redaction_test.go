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
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
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
		{"query key after an encoded no-break space for token", "https://h/p?my%C2%A0token=SECRET", "https://h/p?my%C2%A0token=" + r},
		{"query key after an encoded no-break space for temp:", "https://h/p?my%C2%A0temp:x=SECRET", "https://h/p?my%C2%A0temp:x=" + r},
		{"query key with plus-joined multi-colon temp:", "https://h/p?my+temp:a:b=SECRET", "https://h/p?my+temp%3Aa%3A" + r + "=" + r},
		{"query key with slash-joined multi-colon temp:", "https://h/p?a/temp:a:b=SECRET", "https://h/p?a%2Ftemp%3Aa%3A" + r + "=" + r},
		{"query key after encoded NBSP with multi-colon temp:", "https://h/p?my%C2%A0temp:a:b=SECRET", "https://h/p?my%C2%A0temp%3Aa%3A" + r + "=" + r},
		{"query key with temp: double colon stays unredacted at trailing equals", "https://h/p?my+temp:a:=PUBLIC", "https://h/p?my+temp:a:=PUBLIC"},
		{"query key after non-ASCII letter before temp: stays", "https://h/p?caf%C3%A9temp:x=PUBLIC", "https://h/p?caf%C3%A9temp:x=PUBLIC"},
		{"bearer token in a query key", "https://h/p?Bearer+SECRET123+token=VAL", "https://h/p?Bearer+" + r + "+token=" + r},
		{"bearer token in a query value", "https://h/p?q=Bearer+SECRET123", "https://h/p?q=Bearer+" + r},
		{"pair in a query value", "https://h/p?note=password%3DSECRET123", "https://h/p?note=password%3D" + r},
		{"query value with backslash-t after Bearer", "https://h/p?note=Bearer%5CtSECRET", "https://h/p?note=%5BREDACTED_SENSITIVE_TEXT%5D"},
		{"query key with plus-joined sensitive suffix", "https://h/p?my+X__API__KEY=V", "https://h/p?my+X__API__KEY=" + r},
		{"query key with slash-joined sensitive suffix", "https://h/p?a/X__API__KEY=V", "https://h/p?a/X__API__KEY=" + r},
		{"URL in a query value", "https://h/redirect?next=https://inner/p?token=SECRET123", "https://h/redirect?next=https%3A%2F%2Finner%2Fp%3Ftoken%3D" + r},
		{"percent-encoded redirect URL with userinfo", "https://app.example/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40evil.example%2Fp", "https://app.example/cb?next=%5BREDACTED_SENSITIVE_URI%5D"},
		{"benign percent-encoded redirect URL stays untouched", "https://app.example/cb?next=https%3A%2F%2Fapp.example%2Fdashboard", "https://app.example/cb?next=https%3A%2F%2Fapp.example%2Fdashboard"},
		{"percent-encoded IPv6 redirect URL stays untouched", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A1%5D", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A1%5D"},
		{"percent-encoded IPv6 redirect URL with userinfo leaves no dangling bracket", "https://example.com/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40%5B2001%3Adb8%3A%3A1%5D", "https://example.com/cb?next=%5BREDACTED_SENSITIVE_URI%5D"},
		{"double-redirect signed file URI", "https://a.example/redirect?next=https%3A%2F%2Fb.example%2Fr%3Fnext%3Dhttps%253A%252F%252Fc.example%252Ff%253Faccess_token%253DSECRET", "https://a.example/redirect?next=%5BREDACTED_SENSITIVE_TEXT%5D"},
		{"query key with empty value", "https://h/p?X-Amz-Security-Token=", "https://h/p?X-Amz-Security-Token="},
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
			{"percent-encoded redirect URL with userinfo", "err: https://app.example/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40evil.example%2Fp done", "err: https://app.example/cb?next=%5BREDACTED_SENSITIVE_URI%5D done"},
			{"percent-encoded IPv6 redirect URL stays untouched across entry points", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A1%5D", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A1%5D"},
			{"percent-encoded IPv6 double-colon redirect URL stays untouched across entry points", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A%5D", "https://example.com/cb?next=https%3A%2F%2F%5B2001%3Adb8%3A%3A%5D"},
			{"percent-encoded IPv6 redirect URL with userinfo across entry points", "https://example.com/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40%5B2001%3Adb8%3A%3A1%5D", "https://example.com/cb?next=%5BREDACTED_SENSITIVE_URI%5D"},
			{"percent-encoded IPv6 double-colon redirect URL with userinfo across entry points", "https://example.com/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40%5B2001%3Adb8%3A%3A%5D", "https://example.com/cb?next=%5BREDACTED_SENSITIVE_URI%5D"},
			{"bare IPv6 URL in prose stays untouched", "see https://[2001:db8::1].", "see https://[2001:db8::1]."},
			{"bare IPv6 double-colon URL in prose stays untouched", "see https://[2001:db8::].", "see https://[2001:db8::]."},
			{"bare IPv6 double-colon URL followed by double bracket stays untouched", "https://[2001:db8::]]", "https://[2001:db8::]]"},
			{"bare IPv6 double-colon URL followed by double bracket and punctuation stays untouched", "see https://[2001:db8::]]).", "see https://[2001:db8::]])."},
			{"IPv6 double-colon URL with userinfo and double bracket keeps outer bracket", "https://u:p@[2001:db8::]]", "[REDACTED_SENSITIVE_URI]]"},
			{"zoned IPv6 URL stays untouched", "https://[fe80::1%25en0]", "https://[fe80::1%25en0]"},
			{"percent-encoded zoned IPv6 redirect URL stays untouched", "https://example.com/cb?next=https%3A%2F%2F%5Bfe80%3A%3A1%2525en0%5D", "https://example.com/cb?next=https%3A%2F%2F%5Bfe80%3A%3A1%2525en0%5D"},
			{"zoned IPv6 URL with userinfo is redacted", "https://user:S3CR3T@[fe80::1%25en0]", redactedURIPlaceholder},
			{"percent-encoded zoned IPv6 redirect URL with userinfo is redacted", "https://example.com/cb?next=https%3A%2F%2Fuser%3AS3CR3T%40%5Bfe80%3A%3A1%2525en0%5D", "https://example.com/cb?next=%5BREDACTED_SENSITIVE_URI%5D"},
			{"zoned IPv6 URL with query credential redacts only query credential", "https://[fe80::1%25en0]/p?token=S3CR3T", "https://[fe80::1%25en0]/p?token=" + r},
			{"zoned IPv6 URL hiding encoded credential fails closed", "https://[fe80::1%2520token%3DS3CR3T]", redactedSensitiveTextPlaceholder},
			{"bracketed non-IPv6 URL trims outer closing bracket", "see [https://example.com]", "see [https://example.com]"},
			{"non-IPv6 host bracket trims closing bracket", "see https://host[a:b] ok", "see [REDACTED_SENSITIVE_URI]] ok"},
			{"mixed parameters", "https://example.com?monkey=1&next_page_token=p2&token_count=3&my_api_key=SECRET1", "https://example.com?monkey=1&next_page_token=p2&token_count=3&my_api_key=" + r},
			{"vertical tab does not end a URI", "see https://u:p@h/x\vSECRET123 ok", "see [REDACTED_SENSITIVE_URI] ok"},
			{"vertical tab in a path value", "https://h/token/SEC\vRET123", redactedURIPlaceholder},
			{"harmless URIs", "see https://example.com/a?b=c, then ftp://h/x.", "see https://example.com/a?b=c, then ftp://h/x."},
			{"route slug", "err: https://example.com/account/reset-password/confirm failed", "err: https://example.com/account/reset-password/confirm failed"},
			{"file name in a path", "err: https://example.com/certs/private.key/download failed", "err: https://example.com/certs/private.key/download failed"},
			{"delimiter without a scheme", "123://456/789 and ://x stay", "123://456/789 and ://x stay"},
			{"query key with empty value followed by space", "https://h/p?X-Amz-Security-Token= V", "https://h/p?X-Amz-Security-Token=[REDACTED]"},
			{"fragment key with empty value followed by space", "https://h/cb#token=URISECRET1&x=1&spring.datasource.password= URISECRET2 for user bob", "https://h/cb#token=" + r + "&x=1&spring.datasource.password=[REDACTED] for user bob"},
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
			{"single-quoted padded key", "' DB_PASSWORD ': 'v'", "' DB_PASSWORD ':'[REDACTED]'"},
			{"double-quoted padded key with equals", `" API_KEY " = "v"`, `" API_KEY "="` + redactedValue + `"`},
			{"prose inside padded quotes is kept", `he said "use DB_PASSWORD " = "v"`, `he said "use DB_PASSWORD " = "v"`},
			{"escaped quotes", `msg: \"password\": \"SECRET123\" end`, `msg: \"password\":\"[REDACTED]\" end`},
			{"escaped quote in an escaped quoted value", `token=\"a\\\"SECRET\"`, `token=\"[REDACTED]\"`},
			{"state-prefixed key", "user:temp:api_key=sk-123 x=1", "user:temp:api_key=[REDACTED] x=1"},
			{"temp: key", "temp:session=abc next=1", "temp:session=[REDACTED] next=1"},
			{"state.temp: key", "state.temp:x=V", "state.temp:x=[REDACTED]"},
			{"flag --temp: key", "run --temp:scratch=V --verbose", "run --temp:scratch=[REDACTED] --verbose"},
			{"flag -temp: key", "-temp:x=V", "-temp:x=[REDACTED]"},
			{"temp: key with colon before equals", "temp:cache:item = V", "temp:cache:[REDACTED]"},
			{"temp: key with colon before colon and space", "temp:cache:item: V", "temp:cache:[REDACTED]"},
			{"temp: key with padded base64 value", "temp:session:YWJjMTIz= expired", "temp:session:[REDACTED]"},
			{"temp: key with double-padded base64 value", "temp:session:YWJjMTIz== expired", "temp:session:[REDACTED] expired"},
			{"single-quoted padded multi-colon temp: key", "' temp:cache:item ': 'v'", "' temp:cache:item ':'[REDACTED]'"},
			{"double-quoted padded multi-colon temp: key with equals", `log: " temp:cache:item " = V`, `log: " temp:cache:item "=[REDACTED]`},
			{"single-quoted padded dotted multi-colon temp: key", "' state.temp:cache:item ': 'v'", "' state.temp:cache:item ':'[REDACTED]'"},
			{"prose inside padded multi-colon temp: quotes keeps value", `' see temp:x ': 'v'`, `' see temp:x ': 'v'`},
			{"temp: key with colon value", "temp:x:V", "temp:x:[REDACTED]"},
			{"temp: key with colon before sensitive key", "temp:a:token=2", "temp:a:[REDACTED]"},
			{"temp: key with nested colon key", "temp:a:b: V", "temp:a:[REDACTED]"},
			{"quoted multi-colon temp: key", `log: "temp:a:b": SECRET`, `log: "temp:a:b":"[REDACTED]"`},
			{"quoted multi-colon temp: key after prose", `foo "temp:a:b": SECRET`, `foo "temp:a:b":"[REDACTED]"`},
			{"quoted multi-colon temp: key after colon", `a: "temp:a:b": SECRET`, `a: "temp:a:b":"[REDACTED]"`},
			{"single-quoted multi-colon temp: key with equals", `'temp:a:b' = SECRET`, `'temp:a:b'=[REDACTED]`},
			{"padded quoted multi-colon temp: key after bare temp:", `temp:x: ' temp:a:b ': SECRET`, `temp:x:'[REDACTED]'`},
			{"padded quoted multi-colon temp: key after bare temp: pair", `temp:x=1 ' temp:a:b ': SECRET`, `temp:x=[REDACTED] ' temp:a:b ':[REDACTED]`},
			{"outer plain key wrapping quoted padded state.temp: key", `log: "x": " state.temp:x"="<value>"`, `log: "x": " state.temp:x"="[REDACTED]"`},
			{"outer sensitive JSON key wrapping quoted padded state.temp: key", `log: "password": " state.temp:x"="<value>"`, `log: "password":"[REDACTED]"`},
			{"outer sensitive text key wrapping quoted padded state.temp: key", `password=" state.temp:x"="<value>"`, `password="[REDACTED]"`},
			{"outer sensitive JSON key before Windows-style path key with colon", `log: "password": C:\ssl\token: V`, `log: "password":"[REDACTED]"`},
			{"outer sensitive JSON key before Windows-style path key with equals", `log: "password": C:\ssl\token= V`, `log: "password":"[REDACTED]"`},
			{"outer sensitive JSON key before quoted Windows-style path key", `log: "api.key": "C:\ssl\password": "V"`, `log: "api.key":"[REDACTED]"`},
			{"single-quoted slash-prefixed padded key", "' config/github.token  ' : '<value>'", "' config/github.token  ':'[REDACTED]'"},
			{"single-quoted slash-prefixed key", "'keys/api.key': <value>", "'keys/api.key':[REDACTED]"},
			{"single-quoted leading-only space padded key", `' api_key': 'SECRET'`, `' api_key':'[REDACTED]'`},
			{"single-quoted leading-only newline padded key", "'\napi_key': 'SECRET'", "'\napi_key':'[REDACTED]'"},
			{"double-quoted leading-only newline padded key with equals", "log: \"\napi_key\" = \"SECRET\"", "log: \"\napi_key\"=\"[REDACTED]\""},
			{"single-quoted leading-only NBSP padded key", "'\u00a0api_key': 'SECRET'", "'\u00a0api_key':'[REDACTED]'"},
			{"single-quoted slash-prefixed padded temp: key", "' config/temp:x ' : 'SECRET'", "' config/temp:x ':'[REDACTED]'"},
			{"single-quoted slash-prefixed newline temp: key", "'config/temp:x\n': 'SECRET'", "'config/temp:x\n':'[REDACTED]'"},
			{"single-quoted Windows-path-prefixed padded temp: key", `'C:\ssl\temp:x ': 'SECRET'`, redactedSensitiveTextPlaceholder},
			{"single-quoted slash-prefixed temp: key with vertical tab", "' config/temp:a\vb ': 'SECRET'", "' config/temp:a\vb ':'[REDACTED]'"},
			{"single-quoted slash-prefixed temp: key with NBSP", "' config/temp:a\u00a0b ': 'SECRET'", "' config/temp:a\u00a0b ':'[REDACTED]'"},
			{"spaced path before slash-prefixed temp: inside quotes redacts", `' see config/temp:x ': 'v'`, `' see config/temp:x ':'[REDACTED]'`},
			{"single-quoted slash-prefixed padded multi-colon temp: key", "' config/temp:a:b ': 'SECRET'", "' config/temp:a:b ':'[REDACTED]'"},
			{"single-quoted slash-prefixed multi-colon temp: key with NBSP", "' config/temp:a:b \u00a0': 'SECRET'", "' config/temp:a:b \u00a0':'[REDACTED]'"},
			{"single-quoted slash-prefixed multi-colon temp: key with BOM", "'config/temp:a:b \ufeff': 'SECRET'", "'config/temp:a:b \ufeff':'[REDACTED]'"},
			{"prose before slash-prefixed multi-colon temp: inside quotes stays unquoted key", `' see config/temp:a:b ': 'v'`, `' see config/temp:a:[REDACTED]`},
			{"JSON key before non-sensitive joined single-quoted prose keeps tail", `log: "password": abc&' hello '`, `log: "password":"[REDACTED]" hello '`},
			{"outer sensitive JSON key before slash-prefixed padded temp: key", `log: "password": " config/temp:x "="SECRET"`, `log: "password":"[REDACTED]"`},
			{"JSON container followed by nested sensitive pair", `log: "dbPassword": [SECRET1]password: SECRET2`, `log: "dbPassword":"[REDACTED]"`},
			{"long-padded JSON container followed by nested sensitive pair", `log: "` + strings.Repeat(" ", 2*maxNestedKeyLen+1) + `dbPassword ": [SECRET1]password: SECRET2`, `log: "dbPassword":"[REDACTED]"`},
			{"inline quoted state-prefixed key with leading padding", `log: "password": " user:api_key"="SECRET"`, `log: "password":"[REDACTED]"`},
			{"inline quoted exact key with leading padding", `password=" API_KEY"="SECRET"`, `password="[REDACTED]"`},
			{"nested temp: stacked key before container-joined pair", "password=temp:a:b: [x]token: SECRET", "password=[REDACTED]"},
			{"bare temp: key before container-joined pair", "temp:a:temp:b: [x]token: SECRET", "temp:a:[REDACTED]"},
			{"nested compound key before container-joined pair", "password=x_password: {a}/token: SECRET", "password=[REDACTED]"},
			{"JSON key before chained container-joined pairs", `log: "password": [a]token: [b]secret: SECRET`, `log: "password":"[REDACTED]"`},
			{"nested temp: key with NBSP", "password=temp:a\u00a0: SECRET", "password=[REDACTED]"},
			{"nested quoted temp: key with NBSP", "password='temp:a\u00a0': 'SECRET'", "password='[REDACTED]'"},
			{"nested temp: key with BOM", "password=temp:a\ufeff: SECRET", "password=[REDACTED]"},
			{"nested quoted temp: key with BOM", "password='temp:a\ufeff': 'SECRET'", "password='[REDACTED]'"},
			{"single-quoted key with trailing newline", "'api_key\n': 'v'", "'api_key\n':'[REDACTED]'"},
			{"single-quoted key with leading newline", "'\n api_key ': 'v'", "'\n api_key ':'[REDACTED]'"},
			{"double-quoted key with trailing newline and equals", "log: \"password\n\" = \"v\"", "log: \"password\n\"=\"[REDACTED]\""},
			{"prose inside newline-padded quotes is kept", "he said \"use DB_PASSWORD\n\" = \"v\"", "he said \"use DB_PASSWORD\n\" = \"v\""},
			{"bare slash-prefixed .key path error is kept", "/run/secrets/api.key: no such file", "/run/secrets/api.key: no such file"},
			{"nested temp: key with vertical tab", "password=temp:a\v: V", "password=[REDACTED]"},
			{"nested path-prefixed temp: key with vertical tab", "password=x/temp:a\v: V", "password=[REDACTED]"},
			{"nested quoted temp: key with vertical tab", "password='temp:a\vb': V", "password='[REDACTED]'"},
			{"punctuation-joined container and pair", "dbPassword={abc}/password: V", "dbPassword=[REDACTED]"},
			{"punctuation-joined colon container and pair", "dbPassword={a:b}/password: V", "dbPassword=[REDACTED]"},
			{"punctuation-joined spaced container and pair", "dbPassword={a: b}/password: V", "dbPassword=[REDACTED]"},
			{"punctuation-joined quoted value and pair", `password='foo 'x/token": V`, "password=[REDACTED]"},
			{"punctuation-joined backslash before container pair", `password=a\b/token={a: SECRET}`, "password=[REDACTED]"},
			{"punctuation-joined backslash before quoted pair", `password=a\b/token="V1 V2"`, "password=[REDACTED]"},
			{"punctuation-joined escaped-quote prefix before quoted pair", `password=\"foo \"x/token="V1 V2"`, "password=[REDACTED]"},
			{"long bare temp: key inside checkedBareEnd", "password=token=a.temp:a:" + strings.Repeat("b", 300) + " = S3CR3T", "password=[REDACTED]"},
			{"long bare temp: key inside quoted outer value", `password="token=a.temp:a:` + strings.Repeat("b", 300) + ` " = S3CR3T`, "password=[REDACTED]"},
			{"container-prefixed stacked bare temp: in text pair", "password=temp:a:[x]: S3CR3T", "password=[REDACTED]"},
			{"container-prefixed stacked bare temp: in JSON pair", `log: "password": temp:a:[x]: S3CR3T`, `log: "password":"[REDACTED]"`},
			{"brace-container stacked bare temp: in text pair", "password=temp:a:{x}: S3CR3T", "password=[REDACTED]"},
			{"brace-container stacked bare temp: in JSON pair", `log: "password": temp:a:{x}: S3CR3T`, `log: "password":"[REDACTED]"`},
			{"chained container-prefixed stacked bare temp: after ampersand", "password=x&temp:a:[x]: S3CR3T", "password=[REDACTED]&temp:a:[REDACTED]"},
			{"chained container-prefixed stacked bare temp: in bareCache path", "password=x/temp:a:[x]: S3CR3T", "password=[REDACTED]"},
			{"container-prefixed stacked bare temp: ampersand-joined to padded quoted key", `log: "password": temp:a:[x]: y&' token ': S3CR3T`, `log: "password":"[REDACTED]"&' token ':[REDACTED]`},
			{"container-prefixed stacked bare temp: slash-joined to padded quoted key", `log: "password": temp:a:[x]: y/' token ': S3CR3T`, `log: "password":"[REDACTED]"`},
			{"unquoted bare temp: does not cross newline into leading colon", "temp:a:hello\n: note", "temp:a:[REDACTED]\n: note"},
			{"unquoted bare temp: does not cross newline into leading equals", "temp:a:hello\n= note", "temp:a:[REDACTED]\n= note"},
			{"multiline single-quoted temp: still redacts across newline before equals", "'temp:a:hello\n' = S3CR3T", "'temp:a:hello\n'=[REDACTED]"},
			{"multiline log-prefixed quoted temp: still redacts across newline before equals", "log: \"temp:a:hello\n\" = S3CR3T", "log: \"temp:a:hello\n\"=[REDACTED]"},
			{"stacked bare temp: segment before brace container", "temp:a:b:{x}: S3CR3T", "temp:a:[REDACTED]"},
			{"outer pair wrapping stacked bare temp: segment before brace container", "password=temp:a:b:{x}: S3CR3T", "password=[REDACTED]"},
			{"stacked bare temp: segment before chained brace containers", "temp:a:b:{x}:{y}: S3CR3T", "temp:a:[REDACTED]"},
			{"stacked bare temp: segment before spaced bracket container", "temp:a:b:[x y]: S3CR3T", "temp:a:[REDACTED]"},
			{"stacked bare temp: chained bracket containers", "temp:a:[c]:[d e]: S3CR3T", "temp:a:[REDACTED]"},
			{"double-quoted padded key containing single quote", `log: "foo'bar/token " = S3CR3T`, `log: "foo'bar/token "=[REDACTED]`},
			{"single-quoted padded key containing double quote", `log: 'foo"bar/token ': S3CR3T`, `log: 'foo"bar/token ':[REDACTED]`},
			{"padded quoted slash path ending in compound uppercase key", `log: 'secrets/DB_PASSWORD ' = S3CR3T`, `log: 'secrets/DB_PASSWORD '=[REDACTED]`},
			{"padded quoted nested slash path ending in camelCase key", `log: 'svc/prod/dbPassword ' = S3CR3T`, `log: 'svc/prod/dbPassword '=[REDACTED]`},
			{"padded quoted Windows path ending in camelCase key", `log: 'svc\prod\dbPassword ' = S3CR3T`, `log: 'svc\prod\dbPassword '=[REDACTED]`},
			{"padded quoted key exceeding 256 bytes of trailing spaces", `log: 'password` + strings.Repeat(" ", 248) + `' = S3CR3T`, `log: 'password` + strings.Repeat(" ", 248) + `'=[REDACTED]`},
			{"padded quoted directory path exceeding 256 bytes", `log: '` + strings.Repeat("a/", 140) + `token ' = S3CR3T`, `log: '` + strings.Repeat("a/", 140) + `token '=[REDACTED]`},
			{"unterminated quote after indented bare temp: key", "credentials:\n  temp:a: 'xtemp:a = S3CR3T", "credentials:\n  temp:a:[REDACTED]"},
			{"unterminated quote immediately after indented bare temp: key", "credentials:\n  temp:a:'xtemp:a = S3CR3T", "credentials:[REDACTED]"},
			{"unterminated single quote after bare temp: prefix", "temp:a: 'xtemp:a = S3CR3T", "temp:a:[REDACTED]"},
			{"unterminated double quote after bare temp: prefix", `temp:a: "xtemp:a = S3CR3T`, "temp:a:[REDACTED]"},
			{"unterminated single quote before equals after bare temp: prefix", "temp:a: 'foo = S3CR3T", "temp:a:[REDACTED]"},
			{"already-redacted temp: segment before equals", "temp:a:[REDACTED] = S3CR3T", "temp:a:[REDACTED]"},
			{"already-redacted temp: segment stays idempotent", "temp:a:[REDACTED]", "temp:a:[REDACTED]"},
			{"closed single quote before equals after bare temp: prefix keeps tail", "temp:a: 'foo' = S3CR3T", "temp:a:'[REDACTED]' = S3CR3T"},
			{"closed backslash-double-quoted value before equals after bare temp: prefix keeps tail", `temp:a: \"foo\" = ok`, `temp:a:\"[REDACTED]\" = ok`},
			{"closed backslash-single-quoted value before equals after bare temp: prefix keeps tail", `temp:a: \'foo\' = ok`, `temp:a:\'[REDACTED]\' = ok`},
			{"already-redacted backslash-double-quoted temp: segment stays idempotent", `temp:a: \"[REDACTED]\" = ok`, `temp:a: \"[REDACTED]\" = ok`},
			{"already-redacted compact backslash-double-quoted temp: segment stays idempotent", `temp:a:\"[REDACTED]\" = ok`, `temp:a:\"[REDACTED]\" = ok`},
			{"closed backslash-double-quoted value with escaped quote after bare temp: prefix keeps tail", `temp:a: \"a\\\"b\" = ok`, `temp:a:\"[REDACTED]\" = ok`},
			{"unterminated backslash-double-quote after bare temp: prefix", `temp:a: \"xtemp:a = S3CR3T`, "temp:a:[REDACTED]"},
			{"unterminated backslash-single-quote after bare temp: prefix", `temp:a: \'xtemp:a = S3CR3T`, "temp:a:[REDACTED]"},
			{"doubled single closing key quote", "log: 'password'': S3CR3T", "log: 'password':[REDACTED]"},
			{"doubled double closing key quote", `log: "password"": S3CR3T`, `log: "password":[REDACTED]`},
			{"stray mixed closing key quote", `log: "client_secret"': S3CR3T`, `log: "client_secret":[REDACTED]`},
			{"leading padded doubled double closing key quote", `log: " password"": S3CR3T`, `log: " password":[REDACTED]`},
			{"both-sides padded doubled double closing key quote", `log: " password "": S3CR3T`, `log: " password ":[REDACTED]`},
			{"doubled double closing key quote on temp: key", `log: "temp:a"": S3CR3T`, `log: "temp:a":[REDACTED]`},
			{"doubled single closing key quote on temp: key", `log: 'temp:a'': S3CR3T`, `log: 'temp:a':[REDACTED]`},
			{"doubled double closing key quote on colon-prefixed key", `log: "app:password"": S3CR3T`, `log: "app:password":[REDACTED]`},
			{"doubled single closing key quote on colon-prefixed uppercase key", `log: 'user:DB_PASSWORD'': S3CR3T`, `log: 'user:DB_PASSWORD':[REDACTED]`},
			{"doubled double closing key quote on stacked temp: key", `log: "temp:a:b"": S3CR3T`, `log: "temp:a:b":[REDACTED]`},
			{"non-sensitive colon key with doubled quote is kept", `log: "a:b"": ok`, `log: "a:b"": ok`},
			{"indented quoted value glued to equals", "credentials:\n  \"label\"=V", `credentials:"[REDACTED]"`},
			{"indented quoted value glued to colon", "credentials:\n  \"account\":V", `credentials:"[REDACTED]"`},
			{"indented padded quoted value glued to equals", "credentials:\n  \"  note  \"=V", `credentials:"[REDACTED]"`},
			{"indented quoted value before colon and space keeps following value", "credentials:\n  \"label\": \"V\"", `credentials:"[REDACTED]": "V"`},
			{"indented quoted value glued to brace container", "credentials:\n  \"label\"={token: V} password=S3CR3T", `credentials:"[REDACTED]" password=[REDACTED]`},
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
			{"JSON pair value ending in compound key name", `log: "my_api_key": prod-api-key: SECRET`, `log: "my_api_key":"[REDACTED]"`},
			{"JSON pair value ending in key before container", `log: "password": foo_password: {a: b}`, `log: "password":"[REDACTED]"`},
			{"YAML block with Windows path before base sensitive key", "credentials:\n  C:\\ssl\\password: V", "credentials:[REDACTED]"},
			{"YAML block with quoted non-base key and equals", "credentials:\n  \"api.key\" = \"V\"", "credentials:\"[REDACTED]\""},
			{"YAML block with quoted temp key and equals", "credentials:\n  \"temp:x\" = \"V\"", "credentials:\"[REDACTED]\""},
		})
	})

	t.Run("punctuation-joined pairs redact joined sensitive pairs completely", func(t *testing.T) {
		const secret = "S3CR3T_JOIN_99"
		cases := []string{
			`log: "api.key": abc&"token": ` + secret,
			`log: "private.key": [abc];"client_secret" : "` + secret + `"`,
			`log: "my_api_key": abc/'password': '` + secret + `'`,
			`log: "dbPassword": [abc]&token: ` + secret,
			`log: "secret.key": abc+\"user:api_key\": \"` + secret + `\"`,
			`log: "X-Amz-Security-Token": abc|"api_key":"` + secret + ` more"`,
			"log: \"api.key\": [abc];\"\u00a0api_key\r\n\": \"" + secret + `"`,
			`dbPassword={abc}/password: ` + secret,
			`my_api_key={abc}|'token': '` + secret + `'`,
			`dbPassword={a:b}/password: ` + secret,
			`dbPassword={a: b}/password: ` + secret,
			`my_api_key={unclosed "password": ` + secret,
			`log: "dbPassword": [abc]&' token ': ` + secret,
			`log: "api.key": abc/' token ': ` + secret,
			`log: "api.key": [abc];' api_key ': '` + secret + `'`,
			`log: "dbPassword": [abc]&\" token \" = ` + secret,
			`log: "dbPassword": [abc]&'token ': ` + secret,
			`log: "api.key": abc/'token ': ` + secret,
			`log: "dbPassword": [abc]&"token " = ` + secret,
			"log: \"dbPassword\": [abc]&'api_key\n': " + secret,
			`log: "dbPassword": [abc]|'token':'` + secret + ` tail'`,
			`log: "dbPassword": [abc]&"token"="` + secret + ` tail"`,
			`log: "api.key": abc|'token':{a: ` + secret + `}`,
			`log: "password": "token": abc&"secret": "` + secret + `"`,
			`log: "password": "token": [a]&"secret": "` + secret + `"`,
			`log: "password": "token": abc&secret: ` + secret,
			`log: "password": "token": [a]&secret: ` + secret,
			`log: "password": "token": [a]&' token ': ` + secret,
			`log: "password": "token": abc&"secret": def&' token ': ` + secret,
			`log: "password": [a]token: [b]' token ': ` + secret,
			`log: "password": foo_password: x/' token ': ` + secret,
			`log: "dbPassword": [abc]&"a&' token ' = ` + secret,
			`log: "dbPassword": [abc]&\"foo\"&' token ' = ` + secret,
			`log: "dbPassword": [abc]&token : ` + secret,
			`log: password=abc&' token ': ` + secret,
			`log: "password": "x": abc&"secret": "` + secret + `"`,
			`log: "password": abc&"token": def&secret:'x"api_key":"y z" ` + secret + `'`,
			`log: "password": abc&"token": def&secret:'x"api_key":[y z] ` + secret + `'`,
			`log: "password": abc&"token": def&secret:'x"api_key":[y z]&foo ` + secret + `'`,
			`log: "password": abc&"token": def&secret:{x"api_key":"y z" ` + secret + `}`,
			`log: "dbPassword": [abc]&token` + strings.Repeat(" ", 2*maxNestedKeyLen+5) + `: ` + secret,
			`log: "password": foo_password` + strings.Repeat(" ", 2*maxNestedKeyLen+5) + `: ` + secret,
			`log: "dbPassword": [abc]&' token '` + strings.Repeat(" ", 2*maxNestedKeyLen+5) + `: ` + secret,
			`log: "password": "foo token"=` + secret,
			`log: "password": "foo token": ` + secret,
			`log: "password": 'x"secret": "' token"=` + secret,
			`log: "password": 'x"secret": "' token": ` + secret,
			`log: "dbPassword": [abc]&'password` + strings.Repeat(" ", 300) + `' = ` + secret,
			`log: "temp:a": 'x'' config/github.token  '=` + secret,
			`log: "password": 'x'' config/github.token  ': ` + secret,
			`log: "password": "x"" config/github.token  ": ` + secret,
			`log: "password": ''' config/github.token '=` + secret,
			`log: "password": 'x'' ` + strings.Repeat("a/", 300) + `token ' = ` + secret,
			`log: "password": 'x''` + strings.Repeat(" ", 600) + `token ' = ` + secret,
			`log: "dbPassword": [abc]&'password` + strings.Repeat(" ", 600) + `'' = ` + secret,
			`log: "password": \"x\"\" ` + strings.Repeat("a/", 300) + `token \" = ` + secret,
			`log: "password": \"x\"\" ` + strings.Repeat("a/", 127) + `token \" = ` + secret,
			`log: "password": a&token= '` + strings.Repeat(" ", 600) + `token ' = ` + secret,
			`log: "password": a&token= '` + strings.Repeat("a/", 300) + `token ' = ` + secret,
			`log: "password": a&token= '` + strings.Repeat(" ", 600) + `secrets/DB_PASSWORD ' = ` + secret,
			`log: "password": a&token= 'token=` + strings.Repeat("a/", 300) + `token ' = ` + secret,
			`log: "password": a&token= 'token=a/token ' = ` + secret,
			`log: "password": a&token= 'x token=y ` + strings.Repeat("a/", 300) + `token ' = ` + secret,
			`log: "password": 'token=a/token ' = ` + secret,
			`log: "password": "foo token ": ` + secret,
		}
		for _, in := range cases {
			gotErr := redactErrorMessage(in)
			if strings.Contains(gotErr, secret) {
				t.Errorf("redactErrorMessage(%q) leaked secret: %q", in, gotErr)
			}
			if twice := redactErrorMessage(gotErr); twice != gotErr {
				t.Errorf("redactErrorMessage(%q) not idempotent: pass1=%q pass2=%q", in, gotErr, twice)
			}
			gotSan, lost := sanitizeRawText(in)
			if !lost || strings.Contains(gotSan, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true without secret", in, gotSan, lost)
			}
			if twiceSan, lost2 := sanitizeRawText(gotSan); twiceSan != gotSan || lost2 {
				t.Errorf("sanitizeRawText(%q) not idempotent: pass1=(%q,%v) pass2=(%q,%v)", in, gotSan, lost, twiceSan, lost2)
			}
			truncBytes, _, err := smartTruncate(map[string]any{"message": in}, 1<<20, true)
			if err != nil {
				t.Fatalf("smartTruncate(%q): %v", in, err)
			}
			if strings.Contains(string(truncBytes), secret) {
				t.Errorf("smartTruncate(%q) leaked secret: %s", in, truncBytes)
			}
		}
		checkRedaction(t, []redactionCase{
			{"inner token in long quoted path followed by assignment", `log: "password": a&token= 'token=` + strings.Repeat("a/", 300) + `token ' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"inner token in short quoted path followed by assignment", `log: "password": a&token= 'token=a/token ' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"inner token with spaces and long quoted path followed by assignment", `log: "password": a&token= 'x token=y ` + strings.Repeat("a/", 300) + `token ' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"direct quoted path with inner token followed by assignment", `log: "password": 'token=a/token ' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"direct quoted spaced sensitive key followed by colon", `log: "password": "foo token ": S3CR3T`, `log: "password":"[REDACTED]"`},
			{"doubled single closing quote before assignment", `log: "password": 'token=a/token '' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"doubled double closing quote before assignment", `log: "password": "token=a/token "" = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"doubled escaped single closing quote before assignment", `log: "password": \'token=a/token \'\' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"doubled escaped double closing quote before assignment", `log: "password": \"token=a/token \"\" = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"mixed adjacent closing quotes before assignment", `log: "password": 'token=a/token '" = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"long inner token with doubled single closing quote", `log: "password": a&token= 'token=` + strings.Repeat("a/", 300) + `token '' = S3CR3T`, `log: "password":"[REDACTED]"`},
			{"triple closing quote stays unextended", `log: "password": 'token=a/token ''' = S3CR3T`, `log: "password":"[REDACTED]" = S3CR3T`},
			{"glued third single quote before equals", `log: "password": 'x'' token '''=S3CR3T`, `log: "password":"[REDACTED]"`},
			{"glued third double quote before equals", `log: "password": "x"" token """=S3CR3T`, `log: "password":"[REDACTED]"`},
			{"glued fourth single quote before equals", `log: "password": 'x'' token ''''=S3CR3T`, `log: "password":"[REDACTED]"`},
			{"glued mixed double quotes before equals", `log: "password": 'x'' token '""=S3CR3T`, `log: "password":"[REDACTED]"`},
			{"glued escaped single quotes before equals", `log: "password": 'x'' token '\'\'=S3CR3T`, `log: "password":"[REDACTED]"`},
			{"glued third quote with space before equals stays unextended", `log: "password": 'x'' token ''' = S3CR3T`, `log: "password":"[REDACTED]" = S3CR3T`},
			{"standalone triple quote before equals stays intact", `log: ' token '''=S3CR3T`, `log: ' token '''=S3CR3T`},
			{"quoted path with spaces in directory double-quoted", `log: "Application Support/DB_PASSWORD "=S3CR3T`, `log: "Application Support/DB_PASSWORD "=[REDACTED]`},
			{"quoted path with spaces in directory single-quoted", `log: 'Application Support/DB_PASSWORD '=S3CR3T`, `log: 'Application Support/DB_PASSWORD '=[REDACTED]`},
			{"quoted nested path with spaces in directories", `log: "a b/c d/DB_PASSWORD "=S3CR3T`, `log: "a b/c d/DB_PASSWORD "=[REDACTED]`},
			{"quoted path with spaces and base key", `log: "Application Support/token "=S3CR3T`, `log: "Application Support/token "=[REDACTED]`},
			{"quoted path with spaces and temp key", `log: "Application Support/temp:foo "=S3CR3T`, `log: "Application Support/temp:foo "=[REDACTED]`},
			{"quoted path with spaces inside final component stays intact", `log: "Application Support/my password "=S3CR3T`, `log: "Application Support/my password "=S3CR3T`},
			{"quoted embedded temp with double colon stays intact", `log: " a.temp::x "=S3CR3T`, `log: " a.temp::x "=S3CR3T`},
			{"prose with spaced quoted sensitive key stays intact", `he said "use DB_PASSWORD " = "v"`, `he said "use DB_PASSWORD " = "v"`},
			{"quoted value ending in non-sensitive word after inner token stays intact", `log: "password": 'x token=y foo ' = S3CR3T`, `log: "password":"[REDACTED]" = S3CR3T`},
		})
		spacedContainerJoined := `dbPassword={a: b}/password: ` + secret
		wantSpacedContainerJoined := `dbPassword=[REDACTED]`
		if got := redactErrorMessage(spacedContainerJoined); got != wantSpacedContainerJoined {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", spacedContainerJoined, got, wantSpacedContainerJoined)
		}
		if gotSan, lost := sanitizeRawText(spacedContainerJoined); !lost || gotSan != wantSpacedContainerJoined {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", spacedContainerJoined, gotSan, lost, wantSpacedContainerJoined)
		}
		for _, keptCase := range []redactionCase{
			{"prose inside quoted value is not extended", `log: "password": "foo token: " SECRET`, `log: "password":"[REDACTED]" SECRET`},
			{"trailing unclosed non-sensitive prose stays intact", `log: "password": abc&' hello '`, `log: "password":"[REDACTED]" hello '`},
			{"quoted non-sensitive key after sensitive JSON pair stays intact", `log: "password": "x": abc`, `log: "password":"[REDACTED]": abc`},
		} {
			if got := redactErrorMessage(keptCase.in); got != keptCase.want {
				t.Errorf("%s: redactErrorMessage(%q) = %q, want %q", keptCase.name, keptCase.in, got, keptCase.want)
			}
		}
	})

	t.Run("JSON string with state.temp: key", func(t *testing.T) {
		const in = `{"msg":"state.temp:x=V"}`
		for _, ep := range textEntryPoints {
			want := `{"msg":"state.temp:x=[REDACTED]}`
			if ep.name == "redactErrorMessage" {
				want = `{"msg":"state.temp:x=[REDACTED]"}`
			}
			if got := ep.redact(t, in); got != want {
				t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
			}
		}
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
			{name: "path error with quoted reason", in: `open signing.key: "ordinary47"`},
			{name: "path error with bracketed reason", in: `open signing.key: [ordinary47]`},
			{name: "wrapped path errors", in: "tls: failed to load /etc/ssl/api.key: open /etc/ssl/signing.key: permission denied"},
		}
		for i := range kept {
			kept[i].want = kept[i].in
		}
		checkRedaction(t, kept)
		checkRedaction(t, []redactionCase{
			{"credential next to cursors", "startPageToken=abc resumeToken=jkl eot_token=pqr startToken=vwx", "startPageToken=abc resumeToken=jkl eot_token=pqr startToken=[REDACTED]"},
			{".key name with a value", "private.key: SECRET123", "private.key:[REDACTED]"},
			{".key name with a quoted value", `private.key: "S3CR3T"`, `private.key:"[REDACTED]"`},
			{".key name after a slash with a quoted value directly after colon", `/etc/ssl/api.key:"S3CR3T"`, `/etc/ssl/api.key:"[REDACTED]"`},
			{".key name assigned after a file verb", "open private.key=SECRET123", "open private.key=[REDACTED]"},
			{"quoted .key path with a value", `"/etc/ssl/private.key": "SECRET123"`, `"/etc/ssl/private.key":"[REDACTED]"`},
			{"namespaced quoted compound key with trailing space", `"ConnectionStrings:dbPassword "=S3CR3T`, `"ConnectionStrings:dbPassword "=[REDACTED]`},
			{"prefixed namespaced quoted compound key with trailing space", `log: "ConnectionStrings:dbPassword "=S3CR3T`, `log: "ConnectionStrings:dbPassword "=[REDACTED]`},
			{"single-quoted namespaced compound key with trailing space", `log: 'ConnectionStrings:dbPassword '=S3CR3T`, `log: 'ConnectionStrings:dbPassword '=[REDACTED]`},
			{"multi-colon namespaced quoted key with trailing space", `log: "a:b:DB_PASSWORD "=S3CR3T`, `log: "a:b:DB_PASSWORD "=[REDACTED]`},
			{"slash and colon namespaced quoted key with trailing space", `log: "Application Support/ConnectionStrings:dbPassword "=S3CR3T`, `log: "Application Support/ConnectionStrings:dbPassword "=[REDACTED]`},
			{"spaced namespaced double-quoted key with trailing space", `"Connection Strings:dbPassword "=S3CR3T`, `"Connection Strings:dbPassword "=[REDACTED]`},
			{"spaced namespaced single-quoted key with trailing space", `'Connection Strings:dbPassword '=S3CR3T`, `'Connection Strings:dbPassword '=[REDACTED]`},
			{"slash and spaced namespaced double-quoted key with trailing space", `"Application Support/Connection Strings:dbPassword "=S3CR3T`, `"Application Support/Connection Strings:dbPassword "=[REDACTED]`},
			{"spaced namespaced double-quoted key with trailing colon", `"Connection Strings:dbPassword:"=S3CR3T`, `"Connection Strings:dbPassword:"=[REDACTED]`},
			{"prefixed spaced namespaced JSON key with trailing colon", `log: "Connection Strings:dbPassword:": S3CR3T`, `log: "Connection Strings:dbPassword:":[REDACTED]`},
			{"namespaced double-quoted key with trailing colon", `"ConnectionStrings:dbPassword:"=S3CR3T`, `"ConnectionStrings:dbPassword:"=[REDACTED]`},
			{"quoted double-colon non-sensitive suffix stays unredacted", `" a.temp::x "=S3CR3T`, `" a.temp::x "=S3CR3T`},
			{"quoted phrase with space and no slash stays unredacted", `he said "use DB_PASSWORD " = "v"`, `he said "use DB_PASSWORD " = "v"`},
			{"quoted space inside final path component stays unredacted", `"Application Support/my password "=S3CR3T`, `"Application Support/my password "=S3CR3T`},
			{"quoted key with space before colon and sensitive suffix redacts", `log: "foo bar:password"=S3CR3T`, `log: "foo bar:password"=[REDACTED]`},
			{"bare quoted key with space before colon and sensitive suffix redacts", `"foo bar:password"=S3CR3T`, `"foo bar:password"=[REDACTED]`},
			{"quoted key with space before colon and space inside suffix stays unredacted", `log: "foo bar:my password "=S3CR3T`, `log: "foo bar:my password "=S3CR3T`},
			{"glued empty quotes after single-quoted key redact trailing secret", `password=''' token':' 'S3CR3T`, `password='[REDACTED]'`},
			{"prefixed glued empty quotes after single-quoted key redact trailing secret", `log: password=''' token':' 'S3CR3T`, `log: password='[REDACTED]'`},
			{"glued empty quotes with non-empty single-quoted prefix redact trailing secret", `password=''' token':' a'S3CR3T`, `password='[REDACTED]'`},
			{"glued empty quotes with space-ended single-quoted prefix redact trailing secret", `password=''' token':' a 'S3CR3T`, `password='[REDACTED]'`},
			{"glued empty quotes with space-padded single-quoted prefix redact trailing secret", `password=''' token': ' a 'S3CR3T`, `password=[REDACTED]`},
			{"glued empty quotes with space-padded single-quoted prefix and trailing quote redact secret", `password=''' token': ' a 'S3CR3T'`, `password='[REDACTED]'`},
			{"glued empty quotes with comma-space single-quoted prefix redact trailing secret", `password=''' token':',a 'S3CR3T`, `password='[REDACTED]'`},
			{"colon followed by space-padded quote and unclosed secret redacts", `password: " a "S3CR3T`, `password:[REDACTED]`},
			{"colon followed by space-padded quote and closed secret redacts", `password: " a "S3CR3T"`, `password:"[REDACTED]"`},
			{"equals followed by comma-space quote and secret redacts", `password=",a "S3CR3T`, `password=[REDACTED]`},
			{"glued adjacent double then single quoted segments redact both", `password="foo"'S3CR3T'`, `password=[REDACTED]`},
			{"glued adjacent single then double quoted segments redact both", `password='foo'"S3CR3T"`, `password=[REDACTED]`},
			{"glued empty double then single quoted secret redacts", `password=""'S3CR3T'`, `password=[REDACTED]`},
			{"glued empty single then double quoted secret redacts", `password=''"S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then leading-space double quoted secret redacts both", `password='fixture-'" S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then double quoted secret redacts both", `password='fixture-'"S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then equals double quoted secret redacts both", `password='fixture-'"=S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then colon double quoted secret redacts both", `password='fixture-'":S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then comma double quoted secret redacts both", `password='fixture-'",S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then semicolon double quoted secret redacts both", `password='fixture-'";S3CR3T"`, `password=[REDACTED]`},
			{"glued single quote then space-padded double quoted secret redacts both", `password='fixture-'" S3CR3T "`, `password=[REDACTED]`},
			{"glued single quote then space-padded double quoted secret preserves trailing sibling", `password='fixture-'" S3CR3T " other="val"`, `password=[REDACTED] other="val"`},
			{"glued double quote then single quoted secret redacts both", `password="fixture-"'S3CR3T'`, `password=[REDACTED]`},
			{"glued double quote then empty single quotes redacts both", `password="fixture-"''`, `password=[REDACTED]`},
			{"glued three-segment single double empty-double shell secret redacts", `password='fixture-'" S3CR3T""`, `password=[REDACTED]`},
			{"glued three-segment single double suffix shell secret redacts", `password='fixture-'" S3CR3T"suffix"`, `password=[REDACTED]`},
			{"glued three-segment single comma-double empty-double shell secret redacts", `password='fixture-'",S3CR3T""`, `password=[REDACTED]`},
			{"contraction apostrophe before glued double then single quoted secret redacts", `it's password="fixture-"'S3CR3T'`, `it's password=[REDACTED]`},
			{"nested double quote inside earlier single-quoted value does not suppress glued shell redaction", `msg='say "hi' password='fixture-'"S3CR3T"`, `msg='say "hi' password=[REDACTED]`},
			{"nested single quote inside earlier double-quoted value does not suppress glued shell redaction", `msg="say 'hi" password="fixture-"'S3CR3T'`, `msg="say 'hi" password=[REDACTED]`},
			{"apostrophe inside earlier JSON string does not suppress glued shell redaction", `log: {"msg":"users' error"} password="fixture-"'S3CR3T'`, `log: {"msg":"users' error"} password=[REDACTED]`},
			{"nested double quote inside earlier single-quoted value does not corrupt outer double quote", `log: 'quote "' "password='fixture-'" other="val"`, `log: 'quote "' "password='[REDACTED]'" other="val"`},
			{"nested single quote inside earlier double-quoted value does not corrupt outer single quote", `log: "a'1" 'password="fixture-"' other='val'`, `log: "a'1" 'password="[REDACTED]"' other='val'`},
			{"outer double quote after single-quoted secret before plain key stays intact", `log: "password='fixture-'" other=val`, `log: "password='[REDACTED]'" other=val`},
			{"outer double quote after single-quoted secret before quoted word stays intact", `log: "password='fixture-'" "other"`, `log: "password='[REDACTED]'" "other"`},
			{"outer double quote after single-quoted secret before quoted sibling pair stays intact", `log: "password='fixture-'" other="val"`, `log: "password='[REDACTED]'" other="val"`},
			{"outer single quote after double-quoted secret before quoted sibling pair stays intact", `log: 'password="fixture-"' other='val'`, `log: 'password="[REDACTED]"' other='val'`},
			{"outer double quote after single-quoted secret before plain and quoted pairs stays intact", `log: "password='fixture-'" a=1 msg="ok"`, `log: "password='[REDACTED]'" a=1 msg="ok"`},
			{"outer double quote wrapping prose and single-quoted secret before quoted sibling stays intact", `log: "login failed: password='fixture-'" other="val"`, `log: "login failed: password='[REDACTED]'" other="val"`},
			{"JSON string ending in single-quoted secret preserves sibling fields", `{"a":"password='fixture-'","b":1}`, `{"a":"password='[REDACTED]'","b":1}`},
			{"outer single quote wrapping text and already-redacted double-quoted secret stays intact", `text: 'login failed: password="[REDACTED]"'`, `text: 'login failed: password="[REDACTED]"'`},
			{"empty space quote not glued to trailing secret leaves unredacted tail", `password=''' token':' ' `, `password='[REDACTED]' ' `},
			{"quoted greeting with escaped apostrophe stays intact", `"hello" it\'s fine`, `"hello" it\'s fine`},
			{"JSON key with balanced escaped double quotes before ok stays in place", `log: "password" : \"foo\" ok`, `log: "password":"[REDACTED]" ok`},
			{"JSON key with single escaped single quote does not fail closed", `"password":\' S3CR3T`, `"password":"[REDACTED]" S3CR3T`},
			{"JSON key with two escaped single quotes does not fail closed", `"password":\'' S3CR3T`, `"password":"[REDACTED]" S3CR3T`},
			{"JSON key with two backslashes and one escaped single quote does not fail closed", `"password":\\' S3CR3T`, `"password":"[REDACTED]" S3CR3T`},
			{"prefixed JSON key with two backslashes and two single quotes does not fail closed", `log: "password":\\'' S3CR3T`, `log: "password":"[REDACTED]" S3CR3T`},
			{"JSON key with three bare single quotes does not fail closed", `"password":''' S3CR3T`, `"password":"[REDACTED]" S3CR3T`},
			{"JSON key with escaped double quote does not fail closed", `"password":\"" S3CR3T`, `"password":"[REDACTED]" S3CR3T`},
			{"prefixed JSON key with escaped double quote does not fail closed", `log: "password":\"" S3CR3T`, `log: "password":"[REDACTED]" S3CR3T`},
		})
		if valueStartsAt("password=''' token':' ' ", 20) {
			t.Errorf("valueStartsAt(%q, 20) = true, want false for unglued empty space quote", "password=''' token':' ' ")
		}
		if !valueStartsAt("password=''' token':' 'S3CR3T", 20) {
			t.Errorf("valueStartsAt(%q, 20) = false, want true for glued secret after empty space quote", "password=''' token':' 'S3CR3T")
		}
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
			{name: "escaped quote before nested temp key", in: `password=\"'e /temp:'=x" S3CR3T`},
			{name: "prefixed escaped quote before nested temp key", in: `log: password=\"'e /temp:'=x" S3CR3T`},
			{name: "escaped quote with text before nested temp key", in: `password=\"a'e /temp:'=x" S3CR3T`},
			{name: "escaped quote before nested token key", in: `password=\"'e /token'=x" S3CR3T`},
			{name: "escaped quote before nested JSON password key", in: `token=abc\"password\": SECRET123`},
			{name: "escaped quote inside quoted nested temp key before password", in: `password=a'temp:\" /password'=x" S3CR3T`},
			{name: "prefixed JSON key with escaped quote before nested temp key", in: `log: "password" : \"'e /temp:'=x" S3CR3T`},
			{name: "prefixed JSON key with escaped single quote before nested temp key", in: `log: "password" : \'"e /temp:"=x' S3CR3T`},
			{name: "prefixed JSON key with escaped quote before nested quoted token key", in: `log: "password" : \"'e "token":x" S3CR3T`},
			{name: "prefixed JSON key with unclosed escaped quote before bare word", in: `log: "password" : \"foo bar" S3CR3T`},
			{name: "JSON key with unbalanced escaped single quote", in: `"password":\''' S3CR3T`},
			{name: "prefixed JSON key with unbalanced escaped single quote", in: `log: "password":\''' S3CR3T`},
			{name: "JSON key with two backslashes and three single quotes", in: `"password":\\''' S3CR3T`},
			{name: "prefixed JSON key with two backslashes and three single quotes", in: `log: "password":\\''' S3CR3T`},
			{name: "JSON key with three backslashes and three single quotes", in: `"password":\\\''' S3CR3T`},
			{name: "prefixed JSON key with three backslashes and three single quotes", in: `log: "password":\\\''' S3CR3T`},
			{name: "JSON key with four backslashes and three single quotes", in: `"password":\\\\''' S3CR3T`},
			{name: "JSON key with colon suffix and unbalanced escaped single quote", in: `"password:":\''' S3CR3T`},
			{name: "prefixed JSON key with colon suffix and unbalanced escaped single quote", in: `log: "password:":\''' S3CR3T`},
			{name: "namespaced JSON key with two escaped single quotes and one bare single quote", in: `"ConnectionStrings:dbPassword":\'\'' S3CR3T`},
			{name: "prefixed namespaced JSON key with two escaped single quotes and one bare single quote", in: `log: "ConnectionStrings:dbPassword":\'\'' S3CR3T`},
			{name: "namespaced JSON key with unbalanced escaped single quote", in: `"ConnectionStrings:dbPassword":\''' S3CR3T`},
			{name: "namespaced JSON key with two backslashes and three single quotes", in: `"ConnectionStrings:dbPassword":\\''' S3CR3T`},
			{name: "spaced namespaced JSON key with two escaped single quotes and one bare single quote", in: `"Connection Strings:dbPassword":\'\'' S3CR3T`},
			{name: "namespaced JSON key with trailing colon and unbalanced escaped single quote", in: `"ConnectionStrings:dbPassword:":\''' S3CR3T`},
			{name: "spaced namespaced JSON key with trailing colon and unbalanced escaped single quote", in: `"Connection Strings:dbPassword:":\''' S3CR3T`},
			{name: "namespaced quoted equals key with two escaped single quotes and one bare single quote", in: `"ConnectionStrings:dbPassword"=\'\'' S3CR3T`},
			{name: "quoted equals key with unbalanced escaped single quote", in: `"password"=\''' S3CR3T`},
			{name: "prefixed quoted equals key with unbalanced escaped single quote", in: `log: "password"=\''' S3CR3T`},
			{name: "JSON key with unbalanced escaped double quote", in: `"password":\""" S3CR3T`},
			{name: "prefixed JSON key with unbalanced escaped double quote", in: `log: "password":\""" S3CR3T`},
			{name: "JSON key with two backslashes and three double quotes", in: `"password":\\""" S3CR3T`},
			{name: "prefixed JSON key with two backslashes and three double quotes", in: `log: "password":\\""" S3CR3T`},
			{name: "JSON key with three backslashes and three double quotes", in: `"password":\\\""" S3CR3T`},
			{name: "prefixed JSON key with three backslashes and three double quotes", in: `log: "password":\\\""" S3CR3T`},
			{name: "JSON key with two escaped double quotes and one bare double quote", in: `"password":\"\"" S3CR3T`},
			{name: "prefixed JSON key with two escaped double quotes and one bare double quote", in: `log: "password":\"\"" S3CR3T`},
			{name: "JSON key with two escaped single quotes and one bare single quote", in: `"password":\'\'' S3CR3T`},
			{name: "prefixed JSON key with two escaped single quotes and one bare single quote", in: `log: "password":\'\'' S3CR3T`},
			{name: "JSON key with two bare single quotes and one escaped single quote", in: `"password":''\' S3CR3T`},
			{name: "JSON key with two bare double quotes and one escaped double quote", in: `"password":""\" S3CR3T`},
		}
		for i := range escaped {
			escaped[i].want = redactedSensitiveTextPlaceholder
			if got := scrubErrorText(escaped[i].in); got != redactedSensitiveTextPlaceholder {
				t.Errorf("scrubErrorText(%q) = %q, want %q", escaped[i].in, got, redactedSensitiveTextPlaceholder)
			}
		}
		checkRedaction(t, escaped)
		if got := scrubErrorText(`"password" : \"'e /temp:'=x" S3CR3T`); got != redactedSensitiveTextPlaceholder {
			t.Errorf("scrubErrorText top-level escaped quote before nested temp = %q, want %q", got, redactedSensitiveTextPlaceholder)
		}
		for _, in := range []string{
			`"password" : \"'e /temp:'=x" S3CR3T`,
			`"password" : \'"e /temp:"=x' S3CR3T`,
			`"password" : \"'e "token":x" S3CR3T`,
			`"password" : \"foo bar" S3CR3T`,
		} {
			for _, ep := range textEntryPoints {
				got := ep.redact(t, in)
				if strings.Contains(got, "S3CR3T") || (got != redactedSensitiveTextPlaceholder && got != unparseableJSONPlaceholder) {
					t.Errorf("%s(%q) = %q, want %q or %q without secret leak", ep.name, in, got, redactedSensitiveTextPlaceholder, unparseableJSONPlaceholder)
				}
			}
		}
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
		const (
			keyWord  = `[A-Za-z0-9_.\x{0130}\x{017f}\x{212a}-]`
			anyPad   = `[\t \n\x0b\f\r\x{00a0}\x{feff}]`
			tempSeg  = `[^=:"'\t\n\f\r ,}]+`
			tempHead = `(?:` + keyWord + `*[.-])?(temp:` + tempSeg + `)`
			tempCore = `temp:` + tempSeg + `(?::` + tempSeg + `)*`
		)
		quotedKeyAlt := `(?:(?:user|app):)*(?:` + tempCore + `|` + keyWord + `+)|` + keyWord + `*[.-]` + tempCore + `|[^\s"']*[/\\\x0b](?:` + tempCore + `|` + keyWord + `+)|[^"',{}]*[/\\](?:temp:` + tempSeg + `|` + keyWord + `+)`
		refPaddedRe := regexp.MustCompile(`(?i)(\\?["'])(?:` +
			anyPad + `+(` + quotedKeyAlt + `)` + anyPad + `*|` +
			anyPad + `*(` + quotedKeyAlt + `)` + anyPad + `+|` +
			`([^"',{}]*[/\\]` + keyWord + `+|[^\s"']*\x0b` + keyWord + `+)` +
			`)(\\?["'])[\x00-\x20\x7f]*[=:]`)
		refBareRe := regexp.MustCompile(`(?i)(?:` + tempHead + `|(` + keyWord + `+))(\\?["'])?[\x00-\x20\x7f]*[=:]`)
		type span struct{ keyStart, keyEnd, quoteEnd, end int }
		fromScanner := func(s string) []span {
			var out []span
			for _, m := range textKeyPrefixSpans(s) {
				out = append(out, span{m.start, m.keyEnd, m.quoteEnd, m.end})
			}
			return out
		}
		fromRegexp := func(s string) []span {
			var out []span
			for from := 0; from < len(s); {
				bestStart := len(s) + 1
				var keyStart, keyEnd, quoteEnd, end int
				if loc := refPaddedRe.FindStringSubmatchIndex(s[from:]); loc != nil && from+loc[2] < bestStart {
					openQ, closeQ := s[from+loc[2]:from+loc[3]], s[from+loc[10]:from+loc[11]]
					if openQ == closeQ {
						if len(openQ) == 2 || !strings.HasSuffix(s[from:from+loc[2]], `\`) || hasMatchingOpeningQuote(s, from+loc[3], from, openQ) {
							bestStart = from + loc[2]
							for _, g := range []int{4, 6, 8} {
								if loc[g] >= 0 {
									keyStart, keyEnd = from+loc[g], from+loc[g+1]
									break
								}
							}
							if sub := s[keyStart:keyEnd]; !strings.HasPrefix(strings.ToLower(sub), "temp:") {
								if idx := strings.Index(strings.ToLower(sub), "temp:"); idx > 0 && (sub[idx-1] == ':' || sub[idx-1] == '.' || sub[idx-1] == '-' || sub[idx-1] == '/' || sub[idx-1] == '\\') {
									keyStart += idx
								} else if sensitiveKeyStart(s, keyStart, keyEnd) < 0 {
									if slash := strings.LastIndexAny(sub, `/\`); slash >= 0 && slash+1 < len(sub) {
										keyStart += slash + 1
									}
								}
							}
							quoteEnd, end = from+loc[11], from+loc[1]
						}
					}
				}
				if loc := refBareRe.FindStringSubmatchIndex(s[from:]); loc != nil && from+loc[0] < bestStart {
					bestStart = from + loc[0]
					if loc[2] >= 0 {
						keyStart, keyEnd = from+loc[2], from+loc[3]
					} else {
						keyStart, keyEnd = from+loc[4], from+loc[5]
					}
					quoteEnd, end = keyEnd, from+loc[1]
					if loc[7] >= 0 {
						quoteEnd = from + loc[7]
					}
				}
				if bestStart > len(s) {
					break
				}
				if ks := sensitiveKeyStart(s, keyStart, keyEnd); ks >= 0 {
					out = append(out, span{ks, keyEnd, quoteEnd, end})
				}
				from = end
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
			"state.temp:x=V --temp:scratch=V -temp:x=V temp:cache:item = V temp:cache:item: V",
			`"password"=1 'dbPassword':2 \"token\"=3 \'secret\':4 ' DB_PASSWORD ': 'v' " API_KEY " = "v"`,
			`' temp:x ': 'v' " temp:cache:item " = 1 " user:api_key " = 1 ' state.temp:cache:item ': 'v' ' user:temp:cache:item ': 'v'`,
			"' config/github.token  ' : 'v' 'api_key\n': 'v' '\n api_key ': 'v' \"password\n\" = \"v\"",
			"' config/temp:x ' : 'v' 'config/temp:x\n': 'v' 'C:\\ssl\\temp:x ': 'v'",
			"' config/temp:a\vb ': 'v' ' config/temp:a\u00a0b ': 'v'",
			"' config/temp:a:b ': 'v' ' config/temp:a:b \u00a0': 'v' 'config/temp:a:b \ufeff': 'v' ' see config/temp:a:b ': 'v'",
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
			{"compound key mentioning single-quoted Bearer keeps siblings", `{"db_password": "use 'Bearer' auth", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key mentioning double-quoted Bearer keeps siblings", `{"db_password": "use \"Bearer\" auth", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with apostrophe and Bearer keeps siblings", `{"db_password": "don't mention Bearer", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with closed Bearer quote keeps siblings", `{"db_password": "Bearer 'tok'", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with percent-encoded space and balanced Bearer quotes keeps siblings", `{"db_password": "Bearer%20%22tok\"", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with percent-encoded trailing space and balanced Bearer quotes keeps siblings", `{"db_password": "Bearer%20%22tok\"%20", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with percent-encoded trailing tab and balanced Bearer quotes keeps siblings", `{"db_password": "Bearer%20%22tok\"%09", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with literal space and balanced percent-encoded Bearer quote keeps siblings", `{"db_password": "Bearer %22tok\"", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
			{"compound key with percent-encoded quote after literal quote and trailing encoded quote keeps siblings", `{"db_password": "Bearer%20%22x\"%20%22y%22", "request_id": "req-1"}`, `{"db_password":"[REDACTED]","request_id":"req-1"}`},
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
				name:      "compound sensitive key with percent-encoded space and unclosed Bearer quote",
				in:        `{"db_password": "Bearer%20\"unclosed", "request_id": "req-1"}`,
				wantError: `{"db_password":"[REDACTED]","request_id":"req-1"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "compound sensitive key with percent-encoded quote before literal quote and trailing text",
				in:        `{"db_password": "Bearer%20%22x\" tail", "request_id": "req-1"}`,
				wantError: `{"db_password":"[REDACTED]","request_id":"req-1"}`,
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
			{
				name:      "JSON string with unbalanced escaped single quote inside outer JSON",
				in:        `{"k":"\"password\":\\''' S3CR3T"}`,
				wantError: `{"k":"[REDACTED_SENSITIVE_TEXT]"}`,
				wantText:  redactedSensitiveTextPlaceholder,
			},
			{
				name:      "JSON string with two escaped and one bare single quote inside outer JSON",
				in:        `{"k":"\"password\":\\'\\'' S3CR3T"}`,
				wantError: `{"k":"[REDACTED_SENSITIVE_TEXT]"}`,
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
			{name: "temp key with comma in malformed JSON", in: `{"temp:a,b": "[REDACTED]"`},
			{name: "unicode-escaped key with redacted value in malformed JSON", in: `{"\u0070assword": "[REDACTED]"`},
			{name: "backslash-quote key boundary in malformed JSON", in: `{"a\\": "\"password": "[REDACTED]"`},
			{name: "double-quoted padded multi-colon temp: key with equals", in: `" temp:cache:item " = V`},
		}
		for i := range failed {
			failed[i].want = unparseableJSONPlaceholder
		}
		checkRedaction(t, failed)
	})

	t.Run("malformed text with unescaped quote escape fails closed outside error messages", func(t *testing.T) {
		for _, in := range []string{`{"error":"unexpected token \":\"`, `[INFO] "x\"`} {
			for _, ep := range textEntryPoints {
				want := unparseableJSONPlaceholder
				if ep.name == "redactErrorMessage" {
					want = in
				}
				if got := ep.redact(t, in); got != want {
					t.Errorf("%s(%q) = %q, want %q", ep.name, in, got, want)
				}
			}
		}
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
		{"sensitive key before dense plain pairs", 256 << 10, func(n int) string { return "password=SECRET " + strings.Repeat("a=1 ", n/4) }},
		{"backslash-terminated tokens in bare value", 128 << 10, func(n int) string { return "password=" + strings.Repeat(`token=\`, n/7) + " tail" }},
		{"repeated temp: prefixes", 64 << 10, func(n int) string { return strings.Repeat("temp:", n/5) + "x" }},
		{"repeated temp: prefixes with trailing space word", 64 << 10, func(n int) string { return strings.Repeat("temp:", n/5) + " x" }},
		{"repeated temp: prefixes with secret", 64 << 10, func(n int) string { return strings.Repeat("temp:", n/5) + "x=SECRET" }},
		{"repeated stacked bare temp segments", 64 << 10, func(n int) string { return "temp:a:" + strings.Repeat("b:", n/2) + " tail" }},
		{"repeated stacked bare temp open braces", 64 << 10, func(n int) string { return "temp:a:" + strings.Repeat("b:{", n/3) + " " }},
		{"repeated stacked bare temp balanced braces", 64 << 10, func(n int) string { return "temp:a:" + strings.Repeat("b:{", n/6) + strings.Repeat("}", n/6) + " tail" }},
		{"repeated temp: prefixes after pair with trailing words", 64 << 10, func(n int) string { return "password=" + strings.Repeat("temp:", n/5) + " foo bar" }},
		{"repeated temp: prefixes with unmatched quote", 64 << 10, func(n int) string { return strings.Repeat("temp:", n/5) + "a'" }},
		{"JSON string with repeated temp: prefixes", 64 << 10, func(n int) string { return `{"msg":"prefix ` + strings.Repeat("temp:", n/5) + `x"}` }},
		{"URI query with repeated temp: prefixes and double colon", 64 << 10, func(n int) string { return "https://h/p?" + strings.Repeat("temp:", n/5) + "a::b=PUBLIC" }},
		{"container-joined token chain in text pair", 64 << 10, func(n int) string { return "password=" + strings.Repeat("[x]token:", n/9) + " SECRET" }},
		{"container-joined token chain in JSON pair", 64 << 10, func(n int) string { return `log: "password": ` + strings.Repeat("[a]token: ", n/10) + "SECRET" }},
		{"chained JSON password and token keys", 64 << 10, func(n int) string { return strings.Repeat(`"password": "token": `, n/21) + "SECRET" }},
		{"repeated JSON pairs", 64 << 10, func(n int) string { return strings.Repeat(`"password": x `, n/14) + "password=SECRET" }},
		{"long bare temp: chain after checkedBareEnd", 64 << 10, func(n int) string {
			return "password=token=" + strings.Repeat("a.temp:a:", n/18) + strings.Repeat("b", max(300, n/2)) + " = SECRET"
		}},
		{"stacked bare temp with bracketed containers before closer", 64 << 10, func(n int) string { return "(temp:a:" + strings.Repeat("b:[a]:", n/6) + ") tail" }},
		{"repeated long-padded quoted keys", 64 << 10, func(n int) string {
			return strings.Repeat("'password"+strings.Repeat(" ", 300)+"' = S ", n/316) + "password=SECRET"
		}},
		{"repeated unmatched closing quotes", 64 << 10, func(n int) string { return strings.Repeat("x' = ", n/5) + "'password ' = SECRET" }},
		{"repeated escaped-backslash single quotes", 64 << 10, func(n int) string { return `\'a` + strings.Repeat(`\\'=`, n/4) + "password=SECRET" }},
		{"repeated escaped-backslash double quotes", 64 << 10, func(n int) string { return `\" ` + strings.Repeat(`\\"=x `, n/6) + "password=SECRET" }},
		{"repeated unterminated quotes after temp:", 64 << 10, func(n int) string { return "temp:a: '" + strings.Repeat("x:", n/2) + "a = SECRET" }},
		{"repeated bare temp pairs with escaped quotes", 64 << 10, func(n int) string { return strings.Repeat(`temp:k: x\' `, n/12) + "password=SECRET" }},
		{"repeated bare temp pairs with escaped quotes and trailing quote", 64 << 10, func(n int) string { return strings.Repeat(`temp:k: x\' `, n/12) + "' password=SECRET" }},
		{"repeated bare temp pairs with escaped backslash quotes", 64 << 10, func(n int) string { return strings.Repeat(`temp:k: \"x\\\" `, n/16) + "password=SECRET" }},
		{"repeated bare temp pairs with trailing escaped backslash quotes", 64 << 10, func(n int) string { return strings.Repeat(`temp:k: x\\\' `, n/15) + "password=SECRET" }},
		{"repeated JSON pairs with escaped quotes", 64 << 10, func(n int) string { return strings.Repeat(`"password": 'x\' `, n/17) + "password=SECRET" }},
		{"repeated JSON pairs with doubled quotes", 64 << 10, func(n int) string { return strings.Repeat(`"password": 'x'' `, n/17) + "password=SECRET" }},
		{"repeated window slides in extendOverlappingTextPair", 64 << 10, func(n int) string {
			return `log: "password": ` + strings.Repeat(`token="x `, n/9) + `x" password=SECRET`
		}},
		{"repeated quoted values with inner token in extendOverlappingTextPair", 64 << 10, func(n int) string {
			return `log: "password": a&` + strings.Repeat("token= 'token=a/b ' ", n/20) + "token=SECRET"
		}},
		{"repeated temp: segments inside quoted key ending in double colon", 64 << 10, func(n int) string {
			return `log: "` + strings.Repeat(".temp::", n/7) + ` "=x password=SECRET`
		}},
		{"repeated glued quoted tokens in extendOverlappingTextPair", 64 << 10, func(n int) string {
			return `log: "password": ` + strings.Repeat("'token='a '", n/11) + " password=SECRET"
		}},
		{"repeated glued quoted token pairs in extendOverlappingTextPair", 64 << 10, func(n int) string {
			return `log: "password": ` + strings.Repeat(`'x''token='a''`, n/14) + " password=SECRET"
		}},
		{"repeated colon-qualified JSON keys with escaped quotes", 64 << 10, func(n int) string {
			return "log: " + strings.Repeat(`"a:password":\\'';`, n/18) + " password=SECRET"
		}},
		{"repeated alternating long and short colon-qualified JSON keys with escaped quotes", 64 << 10, func(n int) string {
			return "log: " + strings.Repeat(`"a:password":\\'';"a:password":'x';`, n/35) + " password=SECRET"
		}},
		{"repeated colon-qualified JSON keys inside outer single-quoted span", 64 << 10, func(n int) string {
			return `log: "a:password":'` + strings.Repeat(`"a:password":\\"";`, n/18) + `,'`
		}},
		{"repeated chained quoted assigns in extendQuotedAssign", 64 << 10, func(n int) string {
			return `log: "password":'x'=` + strings.Repeat(`'x'"y"=`, n/7)
		}},
		{"repeated glued opposite-quote shell words", 64 << 10, func(n int) string {
			return strings.Repeat(`token='x'"y" `, n/13)
		}},
		{"repeated glued alternating empty quotes in valueStartsAt", 64 << 10, func(n int) string {
			return "password=" + strings.Repeat(`''""`, n/4) + "SECRET"
		}},
		{"repeated glued alternating empty quotes on nested key in valueStartsAt", 64 << 10, func(n int) string {
			return "password=token=" + strings.Repeat(`''""`, n/4) + "SECRET"
		}},
		{"repeated raw IPv6 scheme prefixes with percent-25 marker", 64 << 10, func(n int) string {
			return strings.Repeat("http://[", n/8) + "]%25"
		}},
		{"repeated percent-encoded IPv6 scheme prefixes with percent-25 marker", 64 << 10, func(n int) string {
			return strings.Repeat("%3A%2F%2F%5B", n/12) + "%25"
		}},
		{"repeated percent-encoded IPv6 hex-scheme prefixes with percent-25 marker", 64 << 10, func(n int) string {
			return strings.Repeat("a%3A%2F%2F%5B", n/13) + "%25"
		}},
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

// TestTextKeyPrefixSpansAllocs verifies that delimiter-dense non-sensitive text
// does not allocate per-delimiter match entries or per-key folding buffers.
func TestTextKeyPrefixSpansAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("allocation count is inflated under -race")
	}
	for _, tt := range []struct {
		name string
		in   string
	}{
		{"dense bare equals", "password=" + strings.Repeat("a=", 1<<17)},
		{"metrics dump", "password=SECRET " + strings.Repeat("metric_one=1 ", 1<<13)},
		{"uppercase colon metrics", "password=SECRET " + strings.Repeat("METRIC_ONE:1 ", 1<<13)},
		{"uppercase equals metrics", "password=SECRET " + strings.Repeat("METRIC_ONE=1 ", 1<<13)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(5, func() {
				_ = textKeyPrefixSpans(tt.in)
			})
			if allocs > 2 {
				t.Errorf("textKeyPrefixSpans allocs = %.0f, want <= 2", allocs)
			}
		})
	}
}

// TestReferenceRegexDifferential checks that redactErrorMessage and
// sanitizeRawText catch every secret removed by a standalone reference regexp
// pass over generated multi-pair, quoted, padded, temp:, and joined inputs.
func TestReferenceRegexDifferential(t *testing.T) {
	fixedParts := make([]string, 0, len(sensitiveKeys))
	compoundParts := make([]string, 0, len(sensitiveKeys))
	camelCompoundParts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		pat := strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[._-]?")
		fixedParts = append(fixedParts, pat)
		if _, low := lowPrecisionKeySuffixes[compactKey(k)]; !low {
			compoundParts = append(compoundParts, pat)
			camelCompoundParts = append(camelCompoundParts, "(?-i:"+strings.ToUpper(pat[:1])+")"+pat[1:])
		}
	}
	fixedAlt := strings.Join(fixedParts, "|")
	compoundAlt := strings.Join(compoundParts, "|")
	camelCompoundAlt := strings.Join(camelCompoundParts, "|")
	statePrefix := `(?:(?:user|app):)*`
	refKey := `(?:` +
		statePrefix + `(?:[A-Za-z0-9_.-]*[.-])?temp:[^=:"'\t\n\f\r ,}]+(?::[^=:"'\t\n\f\r ,}]+)*|` +
		statePrefix + `[A-Za-z0-9_.-]+[._-](?:` + compoundAlt + `)|` +
		statePrefix + `(?-i:[a-z0-9]+)(?:` + camelCompoundAlt + `)|` +
		statePrefix + `(?:[A-Za-z0-9_.-]*[.-])?(?:` + fixedAlt + `)` +
		`)`
	refQuote := `"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'`
	refBarePart := `(?:^|[^0-9A-Za-z_])` + refKey + `(["'])?[\x00-\x20\x7f]*[=:][\x00-\x20\x7f]*(?:` + refQuote + `|[^\s,;&]+)`
	refQuotedPart := `["'][\t \x{00a0}\x{feff}]*` + refKey + `[\t \x{00a0}\x{feff}]*["'][\x00-\x20\x7f]*[=:][\x00-\x20\x7f]*(?:` + refQuote + `|[^\s,;&]+)`
	refPairRe := regexp.MustCompile(`(?i)` + refQuotedPart + `|` + refBarePart)
	refScrub := func(s string) string {
		return refPairRe.ReplaceAllString(s, redactedValue)
	}

	rng := rand.New(rand.NewPCG(20261126, 2515))
	keys := []string{
		"password", "api_key", "api-key", "apiKey", "DB_PASSWORD", "dbPassword",
		"secret_key", "signing_key", "token", "access_token", "client_secret",
		"temp:x", "state.temp:x", "--temp:scratch", "temp:cache:item",
		"user:api_key", "private.key", "x.sig",
	}
	pads := []string{"", " ", "  ", "\t"}
	quotes := []string{"", `"`, `'`}
	delims := []string{":", "=", " : ", " = "}
	joins := []string{" ", ", ", "; ", "&", "/", "|"}
	prefixes := []string{"log: ", "err: {", ""}

	for _, k := range keys {
		const sec = "S3CR3T_SINGLE"
		for _, text := range []string{k + "=" + sec, `"` + k + `" = "` + sec + `"`} {
			if refOut := refScrub(text); strings.Contains(refOut, sec) {
				t.Fatalf("single-pair key %q: refScrub(%q) = %q failed to remove %s", k, text, refOut, sec)
			}
			if gotErr := redactErrorMessage(text); strings.Contains(gotErr, sec) {
				t.Fatalf("single-pair key %q: redactErrorMessage(%q) = %q kept %s", k, text, gotErr, sec)
			}
			if gotSan, _ := sanitizeRawText(text); strings.Contains(gotSan, sec) {
				t.Fatalf("single-pair key %q: sanitizeRawText(%q) = %q kept %s", k, text, gotSan, sec)
			}
		}
	}

	inputs := []string{
		`" temp:cache:item " = "S3CR3T_Q1"`,
		`' temp:cache:item ': 'S3CR3T_Q2'`,
		`' state.temp:cache:item ': 'S3CR3T_Q3'`,
	}
	secretsList := [][]string{{"S3CR3T_Q1"}, {"S3CR3T_Q2"}, {"S3CR3T_Q3"}}
	keysList := [][]string{{"temp:cache:item"}, {"temp:cache:item"}, {"state.temp:cache:item"}}

	for i := range 500 {
		nParts := 1 + rng.IntN(3)
		var sb strings.Builder
		sb.WriteString(prefixes[rng.IntN(len(prefixes))])
		var secrets []string
		var caseKeys []string
		for p := range nParts {
			if p > 0 {
				sb.WriteString(joins[rng.IntN(len(joins))])
			}
			k := keys[rng.IntN(len(keys))]
			q := quotes[rng.IntN(len(quotes))]
			d := delims[rng.IntN(len(delims))]
			sec := fmt.Sprintf("S3CR3T_%d_%d_END", i, p)
			var val string
			switch rng.IntN(4) {
			case 0:
				val = sec
			case 1:
				val = `"` + sec + `"`
			case 2:
				val = `'` + sec + `'`
			case 3:
				val = "{abc}/password: " + sec
			}
			if q == "" {
				sb.WriteString(k + d + val)
			} else {
				pad := pads[rng.IntN(len(pads))]
				sb.WriteString(q + pad + k + pad + q + d + val)
			}
			secrets = append(secrets, sec)
			caseKeys = append(caseKeys, k)
		}
		inputs = append(inputs, sb.String())
		secretsList = append(secretsList, secrets)
		keysList = append(keysList, caseKeys)
	}
	checkedByKey := make(map[string]int, len(keys))
	for i, text := range inputs {
		refOut := refScrub(text)
		gotErr := redactErrorMessage(text)
		gotSan, _ := sanitizeRawText(text)
		for j, sec := range secretsList[i] {
			if i < 3 && strings.Contains(refOut, sec) {
				t.Fatalf("case %d: refScrub(%q) = %q failed to remove %s", i, text, refOut, sec)
			}
			if !strings.Contains(refOut, sec) {
				checkedByKey[keysList[i][j]]++
				if strings.Contains(gotErr, sec) {
					t.Fatalf("case %d: redactErrorMessage(%q) = %q kept %s removed by refScrub", i, text, gotErr, sec)
				}
				if strings.Contains(gotSan, sec) {
					t.Fatalf("case %d: sanitizeRawText(%q) = %q kept %s removed by refScrub", i, text, gotSan, sec)
				}
			}
		}
	}
	for _, k := range keys {
		if checkedByKey[k] == 0 {
			t.Errorf("key %q was never exercised by refScrub in randomized inputs", k)
		}
	}
}
