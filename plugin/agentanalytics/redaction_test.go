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
	"fmt"
	"maps"
	"math/rand/v2"
	"regexp"
	"slices"
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
			"resync_token", "async_token", "app_sync_token",
			"temp:access_token", "temp:anything", "user:api_key", "app:client_secret", "USER:API_KEY",
			"user:temp:refresh_token", "app:temp:scratch", " user: temp:x ", "user:db_password", "app:dbPassword",
		}
		for _, k := range sensitive {
			if !isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = false, want true", k)
			}
		}
		for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
			dotted := strings.ReplaceAll(k, "_", ".")
			for _, candidate := range []string{dotted, "user:" + dotted, "app:" + dotted} {
				if !isSensitiveKey(candidate) {
					t.Errorf("isSensitiveKey(%q) = false, want true", candidate)
				}
			}
		}
		notSensitive := []string{
			"token_count", "prompt_tokens", "completion_tokens", "total_tokens", "max_tokens", "input_tokens",
			"output_tokens", "cached_tokens", "reasoning_tokens", "tool_tokens", "token_limit", "token_budget",
			"token_usage", "token_id", "token_type", "tokenCount", "author", "authority", "authorized",
			"cookie_jar", "secret_manager_project", "secretName", "api_key_id", "password_hash",
			"page_token", "next_page_token", "prev_page_token", "previous_page_token", "nextPageToken",
			"sync_token", "continuation_token", "syncToken", "continuationToken",
			"next_sync_token", "nextSyncToken", "next_continuation_token", "nextContinuationToken",
			"NextToken", "next_token", "nextToken", "prev_token", "PrevToken", "previous_token", "pad_token", "padToken",
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
			SyncToken         string
			ContinuationToken string
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
					"continuationToken": "c2", "continuation_token": "c1", "fb_page_token": "fb",
					"NextToken": "nt1", "next_token": "nt2", "nextSyncToken": "ns1", "pad_token": "pt1",
					"token_count": 3, "next_page_token": "p2", "syncToken": "s2", "sync_token": "s1",
					"nested": map[string]any{"GITHUB_TOKEN": "ghp"},
				},
				want: `{"DB_PASSWORD":"[REDACTED]","NextToken":"nt1","continuationToken":"c2","continuation_token":"c1","dbPassword":"[REDACTED]",` +
					`"fb_page_token":"[REDACTED]","my_api_key":"[REDACTED]","nested":{"GITHUB_TOKEN":"[REDACTED]"},` +
					`"nextSyncToken":"ns1","next_page_token":"p2","next_token":"nt2","pad_token":"pt1","syncToken":"s2","sync_token":"s1","token_count":3,"user:temp:refresh_token":"[REDACTED]"}`,
			},
			{
				name: "struct",
				obj:  creds{DBPassword: "pw", GithubAccessToken: "t", TokenCount: 3, NextPageToken: "p2", SyncToken: "s1", ContinuationToken: "c1"},
				want: `{"ContinuationToken":"c1","DBPassword":"[REDACTED]","GithubAccessToken":"[REDACTED]","NextPageToken":"p2","SyncToken":"s1","TokenCount":3}`,
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
			{"page token kept", "next_page_token=abc nextPageToken=def sync_token=ghi syncToken=jkl continuation_token=mno continuationToken=pqr NextToken=stu nextSyncToken=vwx next_token=yz1 pad_token=pad0", "next_page_token=abc nextPageToken=def sync_token=ghi syncToken=jkl continuation_token=mno continuationToken=pqr NextToken=stu nextSyncToken=vwx next_token=yz1 pad_token=pad0"},
			{"cursor URL query kept and fb_page_token redacted", "GET https://h/p?sync_token=s1&continuation_token=c1&syncToken=s2&continuationToken=c2&NextToken=n1&nextSyncToken=ns1&next_token=n2&pad_token=pd1&fb_page_token=fbsec", "GET https://h/p?sync_token=s1&continuation_token=c1&syncToken=s2&continuationToken=c2&NextToken=n1&nextSyncToken=ns1&next_token=n2&pad_token=pd1&fb_page_token=%5BREDACTED%5D"},
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
		// and an "api_key:" line, which redactAuthHeaders redacts to its end.
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
			"nested open containers": func(n int) string { return strings.Repeat(`"my_api_key": token:[ `, n/22) },
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
		for _, k := range []string{
			"db_pa\u017f\u017fword", "api_\u212aey", "u\u017fer:temp:foo", "app:pa\u017f\u017fword",
			"AP\u0130_KEY", "CL\u0130ENT_SECRET", "DB_CL\u0130ENT_SECRET",
		} {
			if !isSensitiveKey(k) {
				t.Errorf("isSensitiveKey(%q) = false, want true", k)
			}
		}
		for _, k := range []string{"pas\u017fwordless", "mon\u212aey", "u\u017fer:theme", "AP\u0130_KEY_ID"} {
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
			{"x\u017fecret=SECRET123", "x\u017fecret=[REDACTED]"},
			{"AP\u0130_KEY=SECRET123 CL\u0130ENT_SECRET=SECRET456", "AP\u0130_KEY=[REDACTED] CL\u0130ENT_SECRET=[REDACTED]"},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if got, lost := sanitizeRawText(tc.in); !lost || got != tc.want {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, got, lost, tc.want)
			}
		}
		truncBytes, _, err := SmartTruncate(map[string]any{"AP\u0130_KEY": "SECRET123", "CL\u0130ENT_SECRET": "SECRET456"}, 1<<20)
		if err != nil {
			t.Fatalf("SmartTruncate error: %v", err)
		}
		wantTrunc := `{"AP` + "\u0130" + `_KEY":"[REDACTED]","CL` + "\u0130" + `ENT_SECRET":"[REDACTED]"}`
		if string(truncBytes) != wantTrunc {
			t.Errorf("SmartTruncate = %s, want %s", truncBytes, wantTrunc)
		}
	})

	t.Run("false positive exclusions stay untouched across formats", func(t *testing.T) {
		benignInputs := []string{
			"monkey=ok donkey=ok turkey=ok turnkey=ok keyboard=ok passkey_enabled=true passwordless=true token_count=42 next_page_token=p2 sync_token=s1 continuation_token=c1 syncToken=s2 continuationToken=c2 method_signature=foo request_sig=bar app:theme=dark my_xtoken=ok prod_dbpassword=ok reauthorization: ok my_xauthorization: ok benign=1",
			`{"monkey":"a","donkey":"b","turkey":"c","turnkey":"d","keyboard":"e","passkey_enabled":true,"passwordless":true,"token_count":42,"next_page_token":"p2","sync_token":"s1","continuation_token":"c1","syncToken":"s2","continuationToken":"c2","method_signature":"foo","request_sig":"bar","app:theme":"dark","my_xtoken":"ok","prod_dbpassword":"ok","my_xauthorization":"ok"}`,
		}
		for _, in := range benignInputs {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage modified benign input:\ngot:  %q\nwant: %q", got, in)
			}
			if got, lost := sanitizeRawText(in); got != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
			}
		}

		uriIn := "https://example.com?monkey=1&turnkey=2&next_page_token=p2&sync_token=s1&continuation_token=c1&syncToken=s2&continuationToken=c2&token_count=3&my_api_key=SECRET1&auth_token=SECRET2&client-secret=SECRET3&user:temp:foo=SECRET4&app:client_secret=SECRET5&fb_page_token=SECRET6"
		uriGot := redactErrorMessage(uriIn)
		for _, keep := range []string{"monkey=1", "turnkey=2", "next_page_token=p2", "sync_token=s1", "continuation_token=c1", "syncToken=s2", "continuationToken=c2", "token_count=3"} {
			if !strings.Contains(uriGot, keep) {
				t.Errorf("redactErrorMessage(%q) dropped benign param %q: %q", uriIn, keep, uriGot)
			}
		}
		for _, secret := range []string{"SECRET1", "SECRET2", "SECRET3", "SECRET4", "SECRET5", "SECRET6"} {
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

	t.Run("dotted exact and multi-word sensitive keys are redacted across formats", func(t *testing.T) {
		for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
			dotted := strings.ReplaceAll(k, "_", ".")
			for _, prefix := range []string{"", "user:", "app:"} {
				key := prefix + dotted
				kvIn := key + "=VAL987654"
				wantKV := key + "=[REDACTED]"
				if got := redactErrorMessage(kvIn); got != wantKV {
					t.Errorf("redactErrorMessage(%q) = %q, want %q", kvIn, got, wantKV)
				}
				if got, lost := sanitizeRawText(kvIn); !lost || got != wantKV {
					t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", kvIn, got, lost, wantKV)
				}
				jsonIn := fmt.Sprintf(`{%q:"VAL987654"}`, key)
				wantJSON := fmt.Sprintf(`{%q:"[REDACTED]"}`, key)
				if got := redactErrorMessage(jsonIn); got != wantJSON {
					t.Errorf("redactErrorMessage(%q) = %q, want %q", jsonIn, got, wantJSON)
				}
				if got, lost := sanitizeRawText(jsonIn); got != wantJSON || lost {
					t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", jsonIn, got, lost, wantJSON)
				}
			}
		}

		cases := []struct {
			in, want string
		}{
			{"api.key=VAL987654", "api.key=[REDACTED]"},
			{"ssh.private.key=VAL987654", "ssh.private.key=[REDACTED]"},
			{"oauth.auth.code: VAL987654", "oauth.auth.code:[REDACTED]"},
			{"pkce.code.verifier=VAL987654", "pkce.code.verifier=[REDACTED]"},
			{`err: {"app.api.key": "VAL987654"}`, `err: {"app.api.key":"[REDACTED]"}`},
			{`err: {"api.key": "VAL987654", "private.key": "VAL987654"}`, `err: {"api.key":"[REDACTED]", "private.key":"[REDACTED]"}`},
			{"credentials:\n  api.key: VAL987654\n  region: us", "credentials:\n  api.key:[REDACTED]\n  region: us"},
			{"credentials: user:private.key: VAL987654", "credentials: user:private.key:[REDACTED]"},
		}
		for _, tc := range cases {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if got, lost := sanitizeRawText(tc.in); !lost || got != tc.want || strings.Contains(got, "VAL987654") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, got, lost, tc.want)
			}
		}

		jsonDoc := `{"api.key":"VAL987654","private.key":"VAL987654"}`
		wantJSONDoc := `{"api.key":"[REDACTED]","private.key":"[REDACTED]"}`
		if got := redactErrorMessage(jsonDoc); got != wantJSONDoc {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", jsonDoc, got, wantJSONDoc)
		}
		if got, lost := sanitizeRawText(jsonDoc); got != wantJSONDoc || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", jsonDoc, got, lost, wantJSONDoc)
		}

		unparseable := "{\\u0022app.api.key\\u0022\u00a0: \\u0022VAL987654\\u0022}"
		if got, lost := sanitizeRawText(unparseable); !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "VAL987654") {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", unparseable, got, lost, unparseableJSONPlaceholder)
		}
	})

	t.Run("quoted compound and state outer keys redact nested keys without leaking", func(t *testing.T) {
		const secret = "SYNTHETIC_NESTED_99"
		outerKeys := []string{
			`"app:temp:connection"`,
			`"temp:conn"`,
			`"user:api_key"`,
			`"user:temp:conn"`,
			`"my_api_key"`,
			`"DB_PASSWORD"`,
			`'app:temp:connection'`,
			`'my_api_key'`,
		}
		innerLines := []struct {
			name, line, wantLine string
		}{
			{"bare fixed", "password: " + secret, "password:[REDACTED]"},
			{"bare dotted", "api.key: " + secret, "api.key:[REDACTED]"},
			{"bare auth header", "api_key: " + secret, "api_key: [REDACTED]"},
			{"quoted fixed", `"password": "` + secret + `"`, `"password":"[REDACTED]"`},
			{"single-quoted fixed", `'password': '` + secret + `'`, `'password':'[REDACTED]'`},
			{"bare compound", "db_password: " + secret, "db_password:[REDACTED]"},
			{"bare camelCase", "dbPassword: " + secret, "dbPassword:[REDACTED]"},
			{"bare unicode folded", "db_pa\u017f\u017fword: " + secret, "db_pa\u017f\u017fword:[REDACTED]"},
			{"quoted compound", `"db_password": "` + secret + `"`, `"db_password":"[REDACTED]"`},
			{"spaced quoted compound", `" db_password ": "` + secret + `"`, `"db_password":"[REDACTED]"`},
			{"bare temp", "temp:secret: " + secret, "temp:secret:[REDACTED]"},
			{"bare user compound", "user:dbPassword: " + secret, "user:dbPassword:[REDACTED]"},
			{"quoted temp with slash", `"temp:cache/val": "` + secret + `"`, `"temp:cache/val":"[REDACTED]"`},
			{"quoted temp with colon", `"temp:a:b": "` + secret + `"`, `"temp:a:[REDACTED]`},
			{"spaced quoted state temp", `" user:temp:secret ": "` + secret + `"`, `"user:temp:secret":"[REDACTED]"`},
			{"container without space", `password:{"k": "` + secret + `"}`, "password:[REDACTED]"},
		}
		for _, outer := range outerKeys {
			for _, inner := range innerLines {
				in := fmt.Sprintf("config:\n  %s:\n    %s\n    region: us", outer, inner.line)
				want := fmt.Sprintf("config:\n  %s:\n    %s\n    region: us", outer, inner.wantLine)
				gotRaw, lost := sanitizeRawText(in)
				if !lost || gotRaw != want || strings.Contains(gotRaw, secret) {
					t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, gotRaw, lost, want)
				}
				if gotRaw2, lost2 := sanitizeRawText(gotRaw); gotRaw2 != gotRaw || lost2 {
					t.Errorf("sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", gotRaw, gotRaw2, lost2)
				}
				gotErr := redactErrorMessage(in)
				if gotErr != want || strings.Contains(gotErr, secret) {
					t.Errorf("redactErrorMessage(%q) = %q, want %q", in, gotErr, want)
				}
				if gotErr2 := redactErrorMessage(gotErr); gotErr2 != gotErr {
					t.Errorf("redactErrorMessage second pass(%q) = %q, want unchanged", gotErr, gotErr2)
				}
				truncBytes, truncated, err := SmartTruncate(map[string]any{"message": in}, 1<<20)
				if err != nil {
					t.Fatalf("SmartTruncate error: %v", err)
				}
				if !truncated || strings.Contains(string(truncBytes), secret) || !strings.Contains(string(truncBytes), "region: us") {
					t.Errorf("SmartTruncate(%q) = (%s, %v), want truncated=true without secret and keeping region: us", in, truncBytes, truncated)
				}
			}
		}

		sameLineCases := []struct {
			in, want string
		}{
			{`log: "my_api_key": "dev-` + secret + `-password": "` + secret + `_tail" region=us`, `log: "my_api_key":"[REDACTED]" region=us`},
			{`log: "app:temp:connection": "temp:` + secret + `:b": "` + secret + `_tail" region=us`, `log: "app:temp:connection":"[REDACTED]" region=us`},
			{`log: "user:api_key": password: ` + secret + ` region=us`, `log: "user:api_key": password:[REDACTED] region=us`},
		}
		for _, tc := range sameLineCases {
			if got, lost := sanitizeRawText(tc.in); !lost || got != tc.want || strings.Contains(got, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, got, lost, tc.want)
			}
			if got := redactErrorMessage(tc.in); got != tc.want || strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, got, tc.want)
			}
		}
	})

	t.Run("quoted compound and state key redaction is idempotent on second pass", func(t *testing.T) {
		alreadyRedacted := []string{
			`"X-Amz-Security-Token":"[REDACTED]" other=ok`,
			`"X-Amz-Security-Token": "[REDACTED]" other=ok`,
			`"X-Amz-Security-Token": '[REDACTED]' other=ok`,
			`"X-Amz-Security-Token"=[REDACTED] other=ok`,
			`"user:temp:api_key":"[REDACTED]" other=ok`,
			"\"X-Amz-Security-Token\":\"[REDACTED]\"\nother: ok",
			`"api.key":"[REDACTED]" other=ok`,
		}
		for _, in := range alreadyRedacted {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
			}
			if got, lost := sanitizeRawText(in); got != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
			}
		}

		twoPassCases := []struct {
			in, want string
		}{
			{"\"X-Amz-Security-Token\": \"a b\"\nother: ok", "\"X-Amz-Security-Token\":\"[REDACTED]\"\nother: ok"},
			{`"dbPassword": "secret val" other=ok`, `"dbPassword":"[REDACTED]" other=ok`},
			{"\"user:temp:api_key\": \"secret val\"\nother: ok", "\"user:temp:api_key\":\"[REDACTED]\"\nother: ok"},
			{"\"api.key\": \"secret val\"\nother: ok", "\"api.key\":\"[REDACTED]\"\nother: ok"},
		}
		for _, tc := range twoPassCases {
			gotErr := redactErrorMessage(tc.in)
			if gotErr != tc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tc.in, gotErr, tc.want)
			}
			if twice := redactErrorMessage(gotErr); twice != gotErr {
				t.Errorf("redactErrorMessage second pass(%q) = %q, want %q", gotErr, twice, gotErr)
			}
			gotRaw, lost := sanitizeRawText(tc.in)
			if !lost || gotRaw != tc.want {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tc.in, gotRaw, lost, tc.want)
			}
			if twiceRaw, lost2 := sanitizeRawText(gotRaw); twiceRaw != gotRaw || lost2 {
				t.Errorf("sanitizeRawText second pass(%q) = (%q, %v), want (%q, false)", gotRaw, twiceRaw, lost2, gotRaw)
			}
		}

		failClosedInputs := []string{
			`{"X-Amz-Security-Token": "SECRET123"`,
			`{"X-Amz-Security-Token": "[REDACTED]SECRET"`,
			`{"X-Amz-Security-Token": "[REDACTED]"SECRET`,
			`{"X-Amz-Security-Token": [REDACTED]SECRET`,
			`{" my_api_key " = [REDACTED]SECRET`,
			`{"X-Amz-Security-Token":`,
			`"X-Amz-Security-Token":"[REDACTED]" dbPassword=SECRET123`,
		}
		for _, in := range failClosedInputs {
			if got := redactErrorMessage(in); got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, unparseableJSONPlaceholder)
			}
			if got, lost := sanitizeRawText(in); !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, got, lost, unparseableJSONPlaceholder)
			}
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

	t.Run("padded nested keys with unterminated or multiline-escaped values and quote rewinds", func(t *testing.T) {
		const secret = "S3CR3T_hunter2"
		assertAllRedactors := func(t *testing.T, in, want string, wantLost bool) {
			t.Helper()
			if got := redactErrorMessage(in); got != want || strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, want)
			}
			if got, lost := sanitizeRawText(in); got != want || lost != wantLost || strings.Contains(got, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, %v)", in, got, lost, want, wantLost)
			}
			truncBytes, _, err := SmartTruncate(map[string]any{"message": in}, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(%q): %v", in, err)
			}
			if strings.Contains(string(truncBytes), secret) {
				t.Errorf("SmartTruncate(%q) leaked secret: %s", in, truncBytes)
			}
		}

		assertAllRedactors(t,
			"config:\n  \"DB_PASSWORD\":\n    password: "+secret+"\n",
			"config:\n  \"DB_PASSWORD\":\n    password:[REDACTED]\n",
			true,
		)
		assertAllRedactors(t,
			"config:\n  \"app:temp:connection\":\n    password: "+secret+"\n",
			"config:\n  \"app:temp:connection\":\n    password:[REDACTED]\n",
			true,
		)
		assertAllRedactors(t,
			`log: "my_api_key": " password ": "`+secret,
			`log: "my_api_key":"[REDACTED]"`,
			true,
		)
		assertAllRedactors(t,
			`msg="DB_PASSWORD": "password ": "`+secret+` more`,
			`msg="DB_PASSWORD":"[REDACTED]" more`,
			true,
		)
		assertAllRedactors(t,
			"log: \"my_api_key\": \" password \": \""+secret+"\\\n\"",
			"log: \"my_api_key\":\"[REDACTED]\"\n\"",
			true,
		)
		assertAllRedactors(t,
			`log: "app:temp:connection": "app: temp:conn": "`+secret,
			`log: "app:temp:connection":"[REDACTED]"`,
			true,
		)
		assertAllRedactors(t,
			`log: "user:temp:conn": "user: temp:x": '`+secret,
			`log: "user:temp:conn":"[REDACTED]"`,
			true,
		)
		assertAllRedactors(t,
			"config error: \"password\n\": \""+secret+`"`,
			`config error: "password":"[REDACTED]"`,
			true,
		)
		assertAllRedactors(t,
			`config error: "password\n": "`+secret+`"`,
			redactedSensitiveTextPlaceholder,
			true,
		)
		assertAllRedactors(t,
			`err: {"temp:a:b{c": "`+secret+`"}`,
			`err: {"temp:a:[REDACTED]`,
			true,
		)
		assertAllRedactors(t,
			"password:\n'token=\n\""+secret+"-token :token' : \"x y",
			`password:'[REDACTED]' : "x y`,
			true,
		)
		assertAllRedactors(t,
			"password:\n  "+secret+"/dbPassword: x",
			"password:[REDACTED]",
			true,
		)
		if !hasQuotedSensitiveKey("{\"temp:a:b{c\": \"x\"}") {
			t.Errorf("hasQuotedSensitiveKey rejected temp: key containing '{'")
		}
		if !hasQuotedSensitiveKey("\"password\n\": \"x\"") {
			t.Errorf("hasQuotedSensitiveKey rejected quoted key with newline padding")
		}
	})

	t.Run("punctuation-joined pairs do not consume following sensitive keys", func(t *testing.T) {
		const secret = "S3CR3T_JOIN_99"
		cases := []string{
			`log: "api.key": abc&"token": ` + secret,
			`log: "private.key": [abc];"client_secret" : "` + secret + `"`,
			`log: "my_api_key": abc/'password': '` + secret + `'`,
			`log: "dbPassword": [abc]&token: ` + secret,
			`log: "secret.key": abc+\"user:api_key\": \"` + secret + `\"`,
			`log: "X-Amz-Security-Token": abc|"api_key":"` + secret + ` more"`,
			"log: \"api.key\": [abc];\"\u00a0api_key\r\n\": \"" + secret + `"`,
		}
		for _, in := range cases {
			if got := redactErrorMessage(in); strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) leaked secret: %q", in, got)
			}
			if got, lost := sanitizeRawText(in); !lost || strings.Contains(got, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true without secret", in, got, lost)
			}
			truncBytes, _, err := SmartTruncate(map[string]any{"message": in}, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(%q): %v", in, err)
			}
			if strings.Contains(string(truncBytes), secret) {
				t.Errorf("SmartTruncate(%q) leaked secret: %s", in, truncBytes)
			}
		}
	})

	t.Run("path errors and URL route slugs stay intact while real credentials redact", func(t *testing.T) {
		preserved := []string{
			"open /etc/ssl/private.key: no such file or directory",
			`open C:\ssl\secret.key: access denied`,
			"tls: failed to load /etc/ssl/api.key: open /etc/ssl/signing.key: permission denied",
			"err: https://example.com/account/reset-password/confirm failed",
			"err: https://example.com/certs/private.key/download failed",
		}
		for _, in := range preserved {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
			}
			if got, lost := sanitizeRawText(in); got != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
			}
			truncBytes, _, err := SmartTruncate(map[string]any{"message": in}, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(%q): %v", in, err)
			}
			if !strings.Contains(string(truncBytes), "no such file") &&
				!strings.Contains(string(truncBytes), "access denied") &&
				!strings.Contains(string(truncBytes), "permission denied") &&
				!strings.Contains(string(truncBytes), "reset-password/confirm") &&
				!strings.Contains(string(truncBytes), "private.key/download") {
				t.Errorf("SmartTruncate(%q) over-redacted diagnostic text: %s", in, truncBytes)
			}
		}

		const secret = "S3CR3T_PATH_88"
		mustRedact := []string{
			"private.key: " + secret,
			"user:private.key: " + secret,
			`"/etc/ssl/private.key": "` + secret + `"`,
			"/etc/ssl/private.key=" + secret,
			"https://example.com/p?private.key=" + secret,
			"https://example.com/token/" + secret,
			"https://example.com/account/reset-password?reset_password=" + secret,
			"config:\n  \"app:temp:connection\":\n    /etc/ssl/private.key: " + secret,
		}
		for _, in := range mustRedact {
			if got := redactErrorMessage(in); strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) leaked secret: %q", in, got)
			}
			if got, lost := sanitizeRawText(in); !lost || strings.Contains(got, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true without secret", in, got, lost)
			}
		}
	})
}

// TestLegacyRegexScannerDifferential compares the hand-written scanners against
// the reference regular-expression pipeline they replaced over generated
// multi-pair inputs, verifying that quoted-key detection and secret redaction
// never miss an input that the reference expressions matched.
func TestLegacyRegexScannerDifferential(t *testing.T) {
	fixedParts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		if k == "secret_key" || k == "signing_key" {
			continue
		}
		fixedParts = append(fixedParts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[_-]?"))
	}
	fixedAlt := strings.Join(fixedParts, "|")
	refSensKey := `temp:[^"'\s\r\n,}]+|` + fixedAlt
	refBareKey := `temp:[^=:"'\s\r\n,}]+|` + fixedAlt
	escQuote := func(q string) string {
		return `\\` + q + `(?:\\\\(?:\\\\|\\` + q + `|[^\\` + q + `])|\\[^\\` + q + `]|[^\\` + q + `])*\\` + q
	}
	refQuotedVal := `"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|` + escQuote(`"`) + `|` + escQuote(`'`)
	refAuthRe := regexp.MustCompile(`(?im)\b((?:proxy[-_])?authorization|x[-_]api[-_]key|api[-_]key)[ \t]*:[ \t]*[^\r\n]*`)
	refBearerRe := regexp.MustCompile(`(?i)\b(bearer)\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]\r\n\s]+\][^\s,;)"']*|[^\s,;)"'\]]+)`)
	refBasicRe := regexp.MustCompile(`(?i)\b(basic)\s+([A-Za-z0-9+/]+={0,2})`)
	refJSONRe := regexp.MustCompile(`(?i)"[\s\p{Z}]*(` + refSensKey + `)[\s\p{Z}]*"\s*:\s*(` + refQuotedVal + `|[^\s,}]+)`)
	refTextRe := regexp.MustCompile(`(?i)\b(` + refBareKey + `)((?:\\?["'])?)[\s[:cntrl:]]*([=:])[\s[:cntrl:]]*(` + refQuotedVal + `|[^\s,;&]+)`)
	refQuotedKeyRe := regexp.MustCompile(`(?i)["'][\s\p{Z}]*(` + refSensKey + `)[\s\p{Z}]*["'][\s\p{Z}]*[:=]`)

	refScrub := func(s string) string {
		s = refAuthRe.ReplaceAllString(s, "$1: "+redactedValue)
		s = refBearerRe.ReplaceAllStringFunc(s, func(m string) string {
			sub := refBearerRe.FindStringSubmatch(m)
			if strings.EqualFold(sub[2], "of") {
				return m
			}
			return sub[1] + " " + redactedValue
		})
		s = refBasicRe.ReplaceAllStringFunc(s, func(m string) string {
			sub := refBasicRe.FindStringSubmatch(m)
			dec, err := base64.StdEncoding.DecodeString(sub[2])
			if err != nil || (!strings.Contains(sub[2], "=") && !strings.Contains(string(dec), ":")) {
				return m
			}
			return sub[1] + " " + redactedValue
		})
		s = refJSONRe.ReplaceAllString(s, `"$1":"`+redactedValue+`"`)
		s = refTextRe.ReplaceAllString(s, "$1$2$3"+redactedValue)
		return s
	}

	rng := rand.New(rand.NewPCG(20261110, 2250))
	keys := []string{
		"password", "api_key", "api-key", "apiKey", "API_KEY",
		"pa\u017f\u017fword", "api_\u212aey", "x\u017fecret",
		"token", "access_token", "client_secret", "Authorization", "x-api-key",
		"temp:abc", "temp:a:b{c", "temp:cache/val=1",
		"AP\u0130_KEY", "CL\u0130ENT_SECRET", "DB_PASSWORD", "dbPassword",
		"user:api_key", "app:temp:conn", "private.key",
	}
	pads := []string{"", " ", "  ", "\t", "\n", "\r\n", "\u00a0"}
	quotes := []string{"", `"`, `'`, `\"`, `\'`}
	delims := []string{":", "=", " : ", " = ", ":\n  ", ":\t"}
	joins := []string{" ", ", ", "; ", "&", ";", "\n  ", " | ", "/"}
	prefixes := []string{"log: ", "config:\n  ", "err: {", ""}

	for i := range 1500 {
		nParts := 1 + rng.IntN(3)
		var sb strings.Builder
		sb.WriteString(prefixes[rng.IntN(len(prefixes))])
		var secrets []string
		for p := range nParts {
			if p > 0 {
				sb.WriteString(joins[rng.IntN(len(joins))])
			}
			k := keys[rng.IntN(len(keys))]
			q := quotes[rng.IntN(len(quotes))]
			padL := pads[rng.IntN(len(pads))]
			padR := pads[rng.IntN(len(pads))]
			d := delims[rng.IntN(len(delims))]
			sec := fmt.Sprintf("S3CR3T_%d_%d_END", i, p)
			var val string
			switch rng.IntN(6) {
			case 0:
				val = sec
			case 1:
				val = `"` + sec + `"`
			case 2:
				val = `'` + sec + `'`
			case 3:
				val = `\"` + sec + `\"`
			case 4:
				val = "[" + sec + "]"
			case 5:
				val = "Bearer " + sec
			}
			if q == "" {
				sb.WriteString(k + d + val)
			} else {
				sb.WriteString(q + padL + k + padR + q + d + val)
			}
			secrets = append(secrets, sec)
		}
		text := sb.String()
		if refQuotedKeyRe.MatchString(text) && !hasQuotedSensitiveKey(text) {
			t.Fatalf("case %d: hasQuotedSensitiveKey(%q) = false while refQuotedKeyRe matched", i, text)
		}
		refOut := refScrub(text)
		gotErr := redactErrorMessage(text)
		gotSan, _ := sanitizeRawText(text)
		for _, sec := range secrets {
			if !strings.Contains(refOut, sec) {
				if strings.Contains(gotErr, sec) {
					t.Fatalf("case %d: redactErrorMessage(%q) = %q kept %s removed by refScrub", i, text, gotErr, sec)
				}
				if strings.Contains(gotSan, sec) {
					t.Fatalf("case %d: sanitizeRawText(%q) = %q kept %s removed by refScrub", i, text, gotSan, sec)
				}
			}
		}
	}
}
