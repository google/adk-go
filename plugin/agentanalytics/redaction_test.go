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
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"
)

// TestCompoundAndStatePrefixedSensitiveKeys covers key names that end in a
// credential name after other words (DB_PASSWORD, dbPassword, GITHUB_TOKEN,
// spring.datasource.password) and session state keys behind chained scope
// prefixes (user:temp:refresh_token), wherever a key is read: map keys and
// struct fields, STATE_DELTA rows, JSON text, URL queries and key=value or
// key: value text.
func TestCompoundAndStatePrefixedSensitiveKeys(t *testing.T) {
	t.Run("isSensitiveKey", func(t *testing.T) {
		sensitive := []string{
			"DB_PASSWORD", "dbPassword", "DBPassword", "DBPASSWORD", "db-password", "spring.datasource.password",
			"my_api_key", "OPENAI_API_KEY", "myAPIKey", "OpenAIAPIKey", "githubAccessToken", "GITHUB_TOKEN",
			"proxyAuthToken", "webhookSecret", "encryptionPassphrase", "sshPrivateKey", "awsSecretAccessKey",
			"HTTP_AUTHORIZATION", "X-Amz-Security-Token", "secret_key", "SECRET_KEY", "signing_key", "fb_page_token",
			"temp:access_token", "temp:anything", "user:api_key", "app:client_secret", "USER:API_KEY",
			"user:temp:refresh_token", "app:temp:scratch", " user: temp:x ", "user:db_password", "app:dbPassword",
		}
		for _, k := range sensitive {
			if !isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = false, want true", k)
			}
		}
		notSensitive := []string{
			"token_count", "prompt_tokens", "completion_tokens", "total_tokens", "max_tokens", "input_tokens",
			"output_tokens", "cached_tokens", "reasoning_tokens", "tool_tokens", "token_limit", "token_budget",
			"token_usage", "token_id", "token_type", "tokenCount", "author", "authority", "authorized",
			"cookie_jar", "secret_manager_project", "secretName", "api_key_id", "password_hash",
			"page_token", "next_page_token", "prev_page_token", "previous_page_token", "nextPageToken",
			"exchangedAuthCredential", "rawAuthCredential", "method_signature", "request_sig", "Sec-WebSocket-Key",
			"dbpassword", "xtoken", "user:name", "app:theme", "db/password", "db password", "",
		}
		for _, k := range notSensitive {
			if isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = true, want false", k)
			}
		}
	})

	t.Run("map keys and struct fields", func(t *testing.T) {
		type creds struct {
			DBPassword        string
			GithubAccessToken string
			TokenCount        int
			NextPageToken     string
		}
		tests := []struct {
			name string
			obj  any
			want string
		}{
			{
				name: "map",
				obj: map[string]any{
					"DB_PASSWORD": "pw", "dbPassword": "pw", "my_api_key": "k", "user:temp:refresh_token": "r",
					"token_count": 3, "next_page_token": "p2", "nested": map[string]any{"GITHUB_TOKEN": "ghp"},
				},
				want: `{"DB_PASSWORD":"[REDACTED]","dbPassword":"[REDACTED]","my_api_key":"[REDACTED]",` +
					`"nested":{"GITHUB_TOKEN":"[REDACTED]"},"next_page_token":"p2","token_count":3,"user:temp:refresh_token":"[REDACTED]"}`,
			},
			{
				name: "struct",
				obj:  creds{DBPassword: "pw", GithubAccessToken: "t", TokenCount: 3, NextPageToken: "p2"},
				want: `{"DBPassword":"[REDACTED]","GithubAccessToken":"[REDACTED]","NextPageToken":"p2","TokenCount":3}`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, _, err := SmartTruncate(tt.obj, 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				if string(got) != tt.want {
					t.Errorf("SmartTruncate = %s, want %s", got, tt.want)
				}
			})
		}
	})

	t.Run("STATE_DELTA", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-compound-state",
			Author: "test-agent",
			Actions: session.EventActions{
				StateDelta: map[string]any{
					"temp:access_token":       "dropped",
					"user:temp:refresh_token": "rt-secret",
					"app:dbPassword":          "pw-secret",
					"DB_PASSWORD":             "env-secret",
					"token_count":             7,
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		attrs := jsonObjectColumn(t, onlyRowOfType(t, rows, EventStateDelta), "attributes")
		checkJSONAt(t, attrs, `{"DB_PASSWORD":"[REDACTED]","app:dbPassword":"[REDACTED]","token_count":7,"user:temp:refresh_token":"[REDACTED]"}`, "state_delta")
	})

	t.Run("text", func(t *testing.T) {
		tests := []struct {
			name, in, want string
		}{
			{"env assignment", "DB_PASSWORD=hunter2", "DB_PASSWORD=[REDACTED]"},
			{"camelCase assignment", "dbPassword=hunter2 other=ok", "dbPassword=[REDACTED] other=ok"},
			{"colon delimiter", "my_api_key: sk-123", "my_api_key:[REDACTED]"},
			{"dotted key", "spring.datasource.password=hunter2", "spring.datasource.password=[REDACTED]"},
			{"env token", "export GITHUB_TOKEN=ghp_abc", "export GITHUB_TOKEN=[REDACTED]"},
			{"JSON object", `{"dbPassword": "hunter2", "note": "ok"}`, `{"dbPassword":"[REDACTED]","note":"ok"}`},
			{"JSON in prose", `bad config {"DB_PASSWORD": "hunter2"} rejected`, `bad config {"DB_PASSWORD":"[REDACTED]"} rejected`},
			{"JSON in prose spaced compound key", `bad config {" DB_PASSWORD ": "hunter2"} rejected`, `bad config {"DB_PASSWORD":"[REDACTED]"} rejected`},
			{"JSON in prose spaced state key", `bad config {" user:temp:api_key ": "sk-123", " next_page_token ": "p2"}`, `bad config {"user:temp:api_key":"[REDACTED]", " next_page_token ": "p2"}`},
			{"JSON state key", `{"user:temp:api_key": "sk-123"}`, `{"user:temp:api_key":"[REDACTED]"}`},
			{"URL query", "GET https://h/p?db_password=hunter2&x=1", "GET https://h/p?db_password=%5BREDACTED%5D&x=1"},
			{"state prefixed assignment", "user:temp:api_key=sk-123 x=1", "user:temp:api_key=[REDACTED] x=1"},
			{"quoted compound key", `'dbPassword': 'hunter2'`, `'dbPassword':'[REDACTED]'`},
			{"Go struct print", "log: {DBPassword:hunter2 Count:1} done", "log: {DBPassword:[REDACTED] Count:1} done"},
			{"struct print fails closed", "{DBPassword:hunter2 Count:1}", unparseableJSONPlaceholder},
			{"page token kept", "next_page_token=abc nextPageToken=def", "next_page_token=abc nextPageToken=def"},
			{"usage kept", "prompt_tokens=5 token_count=5 max_tokens=1024 author=bob", "prompt_tokens=5 token_count=5 max_tokens=1024 author=bob"},
			{"no word boundary kept", "xtoken=5 dbpassword=plain", "xtoken=5 dbpassword=plain"},
			{"resource names kept", "tokenCount=5 api_key_id=abc secretName=projects/p/secrets/s", "tokenCount=5 api_key_id=abc secretName=projects/p/secrets/s"},
			{"credential envelope key kept", "exchangedAuthCredential=struct", "exchangedAuthCredential=struct"},
			{"nested YAML compound key", "credentials:\n  DB_PASSWORD: SECRET123\n  region: us", "credentials:\n  DB_PASSWORD:[REDACTED]\n  region: us"},
			{"nested YAML camelCase key", "credentials:\n  dbPassword: SECRET123\n  region: us", "credentials:\n  dbPassword:[REDACTED]\n  region: us"},
			{"nested YAML quoted compound key", "credentials:\n  'DB_PASSWORD': 'SECRET123'\n  region: us", "credentials:\n  'DB_PASSWORD':'[REDACTED]'\n  region: us"},
			{"nested YAML state compound key", "credentials:\n  user:DB_PASSWORD: SECRET123\n  region: us", "credentials:\n  user:DB_PASSWORD:[REDACTED]\n  region: us"},
			{"nested YAML nested compound redacted idempotent", "credentials:\n  DB_PASSWORD:[REDACTED]\n  region: us", "credentials:\n  DB_PASSWORD:[REDACTED]\n  region: us"},
			{"same-line nested compound key", "credentials: DB_PASSWORD: SECRET123", "credentials:[REDACTED]"},
			{"value ending in underscore key name", "password=foo_password: SECRET123 ok=1", "password=[REDACTED] ok=1"},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got, _ := sanitizeRawText(tt.in); got != tt.want {
					t.Errorf("sanitizeRawText(%q) = %q, want %q", tt.in, got, tt.want)
				}
			})
		}
	})
}

// TestSensitiveKeyProperties_MonotonicNonConsumingLinear checks the three
// properties that the compound key support must keep: every text redacted
// before it is redacted the same way (monotonic), redacting one value never
// consumes the key of the next pair (non-consuming), and scrubbing stays
// linear in the length of the text, also under concurrent use.
func TestSensitiveKeyProperties_MonotonicNonConsumingLinear(t *testing.T) {
	t.Run("monotonic", func(t *testing.T) {
		// Inputs and the exact outputs of sanitizeRawText before compound
		// keys were matched, including keys that only match at a word
		// boundary inside a compound name (next-page-token, foo-credentials)
		// and an "api_key:" line, which authHeaderRe redacts to its end.
		corpus := []struct{ in, want string }{
			{"temp:access_token=sec1 x=1", "temp:access_token=[REDACTED] x=1"},
			{"user:temp:api_key=sec1 x=1", "user:temp:api_key=[REDACTED] x=1"},
			{"user:api_key=sec1", "user:api_key=[REDACTED]"},
			{`"temp:access_token": "sec1"`, `"temp:access_token":"[REDACTED]"`},
			{`"user:temp:api_key": "sec1"`, `"user:temp:api_key":"[REDACTED]"`},
			{"next-page-token=abc", "next-page-token=[REDACTED]"},
			{"X-Amz-Security-Token=sec1", "X-Amz-Security-Token=[REDACTED]"},
			{"dev-SECRET-password=sec1", "dev-SECRET-password=[REDACTED]"},
			{"foo-credentials=abc", "foo-credentials=[REDACTED]"},
			{"x.sig=abc", "x.sig=[REDACTED]"},
			{`password="SECRETQ-token" :`, `password="[REDACTED]" :`},
			{"password=SECRETQ-token :", "password=[REDACTED] :"},
			{"failed: password: \"password\"", "failed: password:\"[REDACTED]\""},
			{"Authorization: Bearer abc.def", "Authorization: [REDACTED]"},
			{"api_key: secret3 foo: bar", "api_key: [REDACTED]"},
			{"token=a api-key=b apiKey=c", "token=[REDACTED] api-key=[REDACTED] apiKey=[REDACTED]"},
			{"log: {APIKey:abc Count:1} done", "log: {APIKey:[REDACTED] Count:1} done"},
			{`log: "api_key": "a" "token": "b" done`, `log: "api_key":"[REDACTED]" "token":"[REDACTED]" done`},
			{"paſſword=abc", "paſſword=[REDACTED]"},
		}
		for _, c := range corpus {
			if got, _ := sanitizeRawText(c.in); got != c.want {
				t.Errorf("sanitizeRawText(%q) = %q, want %q as before", c.in, got, c.want)
			}
		}
	})

	t.Run("non-consuming", func(t *testing.T) {
		keys := []string{"DB_PASSWORD", "NEXT_SECRET", "dbPassword", "user:temp:refresh_token", "spring.datasource.password", "GITHUB_TOKEN", "next-page-token", "password"}
		for _, delim := range []string{"=", ": ", `": "`} {
			for _, sep := range []string{" ", ", ", "\n", ";"} {
				var in strings.Builder
				in.WriteString("log:")
				for i, k := range keys {
					if i > 0 {
						in.WriteString(sep)
					} else {
						in.WriteString(" ")
					}
					quote := ""
					if delim == `": "` {
						quote = `"`
					}
					fmt.Fprintf(&in, "%s%s%ssecret%d%s", quote, k, delim, i, quote)
				}
				got, _ := sanitizeRawText(in.String())
				for i, k := range keys {
					if !strings.Contains(got, k) {
						t.Errorf("key %q consumed in %q", k, got)
					}
					if secret := fmt.Sprintf("secret%d", i); strings.Contains(got, secret) {
						t.Errorf("value %q of %q kept in %q", secret, k, got)
					}
				}
				if n := strings.Count(got, redactedValue); n != len(keys) {
					t.Errorf("%d values redacted in %q, want %d", n, got, len(keys))
				}
			}
		}
	})

	t.Run("linear", func(t *testing.T) {
		// Each shape stresses a step of the compound key match: the lazy
		// prefix over a long name, the word boundaries read when a compound
		// name is not sensitive, many pairs that are kept, and many that
		// are redacted.
		shapes := map[string]func(n int) string{
			"long name":              func(n int) string { return strings.Repeat("a", n) + "password=x" },
			"many dashes":            func(n int) string { return strings.Repeat("a-", n/2) + "page-token=x" },
			"kept pairs":             func(n int) string { return strings.Repeat("xtoken=5 next_page_token=abc ", n/29) },
			"kept spaced json pairs": func(n int) string { return `err ` + strings.Repeat(`" next_page_token ": "p2", `, n/28) },
			"redacted pairs":         func(n int) string { return strings.Repeat("DB_PASSWORD=sec dbPassword=sec ", n/31) },
		}
		const small, large = 1 << 10, 16 << 10
		for name, shape := range shapes {
			t.Run(name, func(t *testing.T) {
				smallIn, largeIn := shape(small), shape(large)
				scrub := func(s string) {
					_, _ = sanitizeRawText(s)
					_ = redactErrorMessage(s)
				}
				scrub(smallIn)
				scrub(largeIn)
				tSmall, tLarge := minDuration(3, func() { scrub(smallIn) }), minDuration(3, func() { scrub(largeIn) })
				// A linear scrub takes about 16 times longer on the 16 times
				// longer text; a quadratic one about 256 times longer.
				if ratio := float64(tLarge) / float64(max(tSmall, 100*time.Microsecond)); ratio > 64 {
					t.Errorf("%d KB took %v, %d KB took %v: ratio %.0f, want linear", small>>10, tSmall, large>>10, tLarge, ratio)
				}

				// The result does not depend on concurrent use.
				in := shape(4 << 10)
				want, _ := sanitizeRawText(in)
				var wg sync.WaitGroup
				results := make([]string, 16)
				for i := range results {
					wg.Add(1)
					go func() {
						defer wg.Done()
						results[i], _ = sanitizeRawText(in)
					}()
				}
				wg.Wait()
				for i, got := range results {
					if got != want {
						t.Errorf("goroutine %d: result differs from the sequential one", i)
					}
				}
			})
		}
	})
}

// minDuration returns the shortest of n timed runs of f.
func minDuration(n int, f func()) time.Duration {
	best := time.Duration(-1)
	for range n {
		start := time.Now()
		f()
		if d := time.Since(start); best < 0 || d < best {
			best = d
		}
	}
	return best
}

func TestCompoundKeyEdgeCasesAndLinearBackslashRegression(t *testing.T) {
	t.Run("quoted escaped compound and state keys fail closed", func(t *testing.T) {
		unparseableInputs := []string{
			"{\\u0022dbPassword\\u0022\u00a0: \\u0022SECRET123\\u0022}",
			"{\\u0022user:api_key\\u0022\u00a0: \\u0022SECRET123\\u0022}",
			"{\\u0022user:temp:foo\\u0022\u00a0: \\u0022SECRET123\\u0022}",
			"{\\u0022app:client-secret\\u0022\u00a0= \\u0022SECRET123\\u0022}",
		}
		for _, in := range unparseableInputs {
			if got := redactErrorMessage(in); got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, unparseableJSONPlaceholder)
			}
			got, lost := sanitizeRawText(in)
			if !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, got, lost, unparseableJSONPlaceholder)
			}
		}

		prefixed := `upstream error: {\u0022dbPassword\u0022: \u0022SECRET123\u0022}`
		if got, lost := sanitizeRawText(prefixed); !lost || got != redactedSensitiveTextPlaceholder || strings.Contains(got, "SECRET123") {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", prefixed, got, lost, redactedSensitiveTextPlaceholder)
		}
		quotedNested := `"{\u0022user:temp:foo\u0022` + "\u00a0" + `: \u0022SECRET123\u0022}"`
		if got, lost := sanitizeRawText(quotedNested); !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", quotedNested, got, lost, unparseableJSONPlaceholder)
		}

		noteIn := `{"note": "a \"dbPassword\": SECRET123"}`
		if got := redactErrorMessage(noteIn); strings.Contains(got, "SECRET123") {
			t.Errorf("redactErrorMessage(%q) leaked secret: %q", noteIn, got)
		}
		if got, lost := sanitizeRawText(noteIn); !lost || strings.Contains(got, "SECRET123") {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true and redacted", noteIn, got, lost)
		}
	})

	t.Run("unicode folded compound and state keys", func(t *testing.T) {
		for _, k := range []string{"db_pa\u017f\u017fword", "api_\u212aey", "u\u017fer:temp:foo", "app:pa\u017f\u017fword"} {
			if !isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = false, want true", k)
			}
		}
		for _, k := range []string{"pas\u017fwordless", "mon\u212aey", "u\u017fer:theme"} {
			if isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = true, want false", k)
			}
		}
		cases := []struct {
			in, want string
		}{
			{"u\u017fer:temp:foo=SECRET123", "u\u017fer:temp:foo=[REDACTED]"},
			{"db_pa\u017f\u017fword=SECRET123", "db_pa\u017f\u017fword=[REDACTED]"},
			{"x-goog-api_\u212aey: SECRET123", "x-goog-api_\u212aey: [REDACTED]"},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
		}
	})

	t.Run("false positive exclusions stay untouched across formats", func(t *testing.T) {
		benignInputs := []string{
			"monkey=ok donkey=ok turkey=ok turnkey=ok keyboard=ok passkey_enabled=true passwordless=true token_count=42 next_page_token=p2 method_signature=foo request_sig=bar app:theme=dark my_xtoken=ok prod_dbpassword=ok reauthorization: ok my_xauthorization: ok benign=1",
			`{"monkey":"a","donkey":"b","turkey":"c","turnkey":"d","keyboard":"e","passkey_enabled":true,"passwordless":true,"token_count":42,"next_page_token":"p2","method_signature":"foo","request_sig":"bar","app:theme":"dark","my_xtoken":"ok","prod_dbpassword":"ok","my_xauthorization":"ok"}`,
		}
		for _, in := range benignInputs {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage modified benign input:\ngot:  %q\nwant: %q", got, in)
			}
			if got, lost := sanitizeRawText(in); got != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
			}
		}

		uriIn := "https://example.com?monkey=1&turnkey=2&next_page_token=p2&token_count=3&my_api_key=SECRET1&auth_token=SECRET2&client-secret=SECRET3&user:temp:foo=SECRET4&app:client_secret=SECRET5"
		uriGot := redactErrorMessage(uriIn)
		for _, keep := range []string{"monkey=1", "turnkey=2", "next_page_token=p2", "token_count=3"} {
			if !strings.Contains(uriGot, keep) {
				t.Errorf("redactErrorMessage(%q) dropped benign param %q: %q", uriIn, keep, uriGot)
			}
		}
		for _, secret := range []string{"SECRET1", "SECRET2", "SECRET3", "SECRET4", "SECRET5"} {
			if strings.Contains(uriGot, secret) {
				t.Errorf("redactErrorMessage(%q) leaked %q: %q", uriIn, secret, uriGot)
			}
		}
	})

	t.Run("header and compound key edge boundaries", func(t *testing.T) {
		cases := []struct {
			in, want string
		}{
			{"request failed: x-goog-api-key: SECRET123", "request failed: x-goog-api-key: [REDACTED]"},
			{"my_api_key=prodSECRET123-api-key: permission denied", "my_api_key=[REDACTED]"},
			{`password="a=b" token=SECRET123`, `password="[REDACTED]" token=[REDACTED]`},
			{
				"DB_PASSWORD=S1 my_api_key=S2 auth_token=S3 client-secret=S4 dbPassword=S5 temp:k=S6 user:temp:k=S7 app:client_secret=S8 app:region=us",
				"DB_PASSWORD=[REDACTED] my_api_key=[REDACTED] auth_token=[REDACTED] client-secret=[REDACTED] dbPassword=[REDACTED] temp:k=[REDACTED] user:temp:k=[REDACTED] app:client_secret=[REDACTED] app:region=us",
			},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
		}
	})

	t.Run("long backslash runs with compound keys stay linear", func(t *testing.T) {
		prefix := strings.Repeat("design: monkey=keyboard;\n", 4_000) + strings.Repeat(`\`, 300)
		in := prefix + " dbPassword=SECRET123"
		start := time.Now()
		got := redactErrorMessage(in)
		if d := time.Since(start); d > time.Second {
			t.Errorf("redactErrorMessage took %v, want < 1s", d)
		}
		want := prefix + " dbPassword=[REDACTED]"
		if got != want {
			t.Errorf("redactErrorMessage modified prefix or missed trailing dbPassword")
		}

		jsonIn := strings.Repeat(`"temp:a": "b", `, 500) + strings.Repeat(`\`, 300)
		start = time.Now()
		jsonGot := redactErrorMessage(jsonIn)
		if d := time.Since(start); d > time.Second {
			t.Errorf("redactErrorMessage on repeated temp JSON keys took %v, want < 1s", d)
		}
		if strings.Contains(jsonGot, `"b"`) {
			t.Errorf("redactErrorMessage did not redact temp JSON keys: %q", jsonGot[:min(len(jsonGot), 120)])
		}
	})

	t.Run("dotted multi-word compound keys do not bypass prefilter", func(t *testing.T) {
		cases := []struct {
			in, want string
		}{
			{"ssh.private.key=VAL987654", "ssh.private.key=[REDACTED]"},
			{"oauth.auth.code: VAL987654", "oauth.auth.code:[REDACTED]"},
			{"pkce.code.verifier=VAL987654", "pkce.code.verifier=[REDACTED]"},
			{`err: {"app.api.key": "VAL987654"}`, `err: {"app.api.key":"[REDACTED]"}`},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if got, lost := sanitizeRawText(tc.in); !lost || strings.Contains(got, "VAL987654") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), leaked secret", tc.in, got, lost)
			}
		}

		unparseable := "{\\u0022app.api.key\\u0022\u00a0: \\u0022VAL987654\\u0022}"
		if got, lost := sanitizeRawText(unparseable); !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "VAL987654") {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", unparseable, got, lost, unparseableJSONPlaceholder)
		}
	})

	t.Run("escaped quoted values handle odd and even backslash runs", func(t *testing.T) {
		cases := []struct {
			in, want string
		}{
			{`token=\"a\\\"SECRET\"`, `token=\"[REDACTED]\"`},
			{`token=\'a\\\'SECRET\'`, `token=\'[REDACTED]\'`},
			{`dbPassword=\"a \\\" SECRET\" monkey=ok`, `dbPassword=\"[REDACTED]\" monkey=ok`},
			{`dbPassword=\'a \\\' SECRET\' monkey=ok`, `dbPassword=\'[REDACTED]\' monkey=ok`},
			{`auth_token=\"a \\\\\\\" SECRET\" monkey=ok`, `auth_token=\"[REDACTED]\" monkey=ok`},
			{`token=\"dir\\ SECRET\" monkey=ok`, `token=\"[REDACTED]\" monkey=ok`},
			{`token=\"line1\n SECRET\" monkey=ok`, `token=\"[REDACTED]\" monkey=ok`},
			{`password=\"SECRET\\"x monkey=ok tail\"`, `password=[REDACTED] monkey=ok tail\"`},
			{`password=\'SECRET\\'x monkey=ok tail\'`, `password=[REDACTED] monkey=ok tail\'`},
			{`password=\"SECRET"x monkey=ok tail\"`, `password=[REDACTED] monkey=ok tail\"`},
			{`token=\"hello SECRET\\\\\" monkey=ok`, `token=\"[REDACTED]\" monkey=ok`},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if got, lost := sanitizeRawText(tc.in); !lost || strings.Contains(got, "SECRET") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true and secret redacted", tc.in, got, lost)
			}
		}

		// m & 3 == 3 with c != q stops escaped-quote scanning at the space
		// delimiter; if an unescaped quote later closes after unescaping, both
		// redactErrorMessage and sanitizeRawText fail closed.
		nonQuoteTripleSlash := `password=\"SECRET\\\ax monkey=ok tail\"`
		if got, lost := sanitizeRawText(nonQuoteTripleSlash); !lost || got != redactedSensitiveTextPlaceholder {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", nonQuoteTripleSlash, got, lost, redactedSensitiveTextPlaceholder)
		}
		if got := redactErrorMessage(nonQuoteTripleSlash); got != redactedSensitiveTextPlaceholder {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", nonQuoteTripleSlash, got, redactedSensitiveTextPlaceholder)
		}
		nonQuoteTripleSlashOpen := `password=\"SECRET\\\ax monkey=ok`
		if got := redactErrorMessage(nonQuoteTripleSlashOpen); got != "password=[REDACTED] monkey=ok" {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", nonQuoteTripleSlashOpen, got, "password=[REDACTED] monkey=ok")
		}
		if got, lost := sanitizeRawText(nonQuoteTripleSlashOpen); !lost || got != "password=[REDACTED] monkey=ok" {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", nonQuoteTripleSlashOpen, got, lost, "password=[REDACTED] monkey=ok")
		}
	})

	t.Run("chained quoted keys stay linear in valueStartsAt", func(t *testing.T) {
		for _, shape := range []struct {
			name, unit, suffix string
		}{
			{"repeated quoted keys", "'token'=", "SECRET123"},
			{"repeated quoted keys with mismatched quote tail", `'token'="a"'token'=`, "x"},
		} {
			t.Run(shape.name, func(t *testing.T) {
				small := strings.Repeat(shape.unit, 128) + shape.suffix
				large := strings.Repeat(shape.unit, 2048) + shape.suffix
				redactErrorMessage(small)
				redactErrorMessage(large)
				tSmall := minDuration(3, func() { redactErrorMessage(small) })
				tLarge := minDuration(3, func() { redactErrorMessage(large) })
				if ratio := float64(tLarge) / float64(max(tSmall, 100*time.Microsecond)); ratio > 64 {
					t.Errorf("128 pairs took %v, 2048 pairs took %v: ratio %.0f, want linear", tSmall, tLarge, ratio)
				}
			})
		}
	})
}
