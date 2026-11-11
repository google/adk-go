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
	"maps"
	"math"
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
	runInParallel(t)
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
			"prev_sync_token", "prevSyncToken", "previous_sync_token", "previousSyncToken",
			"prev_continuation_token", "prevContinuationToken", "previous_continuation_token", "previousContinuationToken",
			"NextToken", "next_token", "nextToken", "prev_token", "PrevToken", "previous_token", "pad_token", "padToken",
			"exchangedAuthCredential", "rawAuthCredential", "method_signature", "request_sig", "Sec-WebSocket-Key",
			"dbpassword", "xtoken", "user:name", "app:theme", "db/password", "db password", "",
			// Uppercase and mixed-case variants of every benign exclusion.
			"TOKEN_COUNT", "Token_Count", "PROMPT_TOKENS", "Prompt_Tokens", "COMPLETION_TOKENS", "TOTAL_TOKENS",
			"MAX_TOKENS", "INPUT_TOKENS", "OUTPUT_TOKENS", "CACHED_TOKENS", "REASONING_TOKENS", "TOOL_TOKENS",
			"TOKEN_LIMIT", "TOKEN_BUDGET", "TOKEN_USAGE", "TOKEN_ID", "TOKEN_TYPE",
			"AUTHOR", "Author", "AUTHORITY", "Authority", "AUTHORIZED", "Authorized", "COOKIE_JAR", "Cookie_Jar",
			"SECRET_MANAGER_PROJECT", "Secret_Manager_Project", "SECRETNAME", "SecretName",
			"API_KEY_ID", "ApiKeyId", "PASSWORD_HASH", "PasswordHash", "PASSWORDLESS", "Passwordless",
			"PASSKEY_ENABLED", "PasskeyEnabled", "MONKEY", "Monkey", "DONKEY", "Donkey",
			"TURKEY", "Turkey", "TURNKEY", "Turnkey", "KEYBOARD", "Keyboard",
			"PAGE_TOKEN", "PageToken", "NEXT_PAGE_TOKEN", "Next_Page_Token", "NEXTPAGETOKEN",
			"PREV_PAGE_TOKEN", "PrevPageToken", "PREVIOUS_PAGE_TOKEN", "PreviousPageToken",
			"SYNC_TOKEN", "SYNCTOKEN", "CONTINUATION_TOKEN", "CONTINUATIONTOKEN",
			"NEXT_SYNC_TOKEN", "NEXTSYNCTOKEN", "PREV_SYNC_TOKEN", "PrevSyncToken", "PREVIOUS_SYNC_TOKEN", "PreviousSyncToken",
			"NEXT_CONTINUATION_TOKEN", "NEXTCONTINUATIONTOKEN", "PREV_CONTINUATION_TOKEN", "PrevContinuationToken",
			"PREVIOUS_CONTINUATION_TOKEN", "PreviousContinuationToken",
			"NEXTTOKEN", "NEXT_TOKEN", "PREV_TOKEN", "PREVTOKEN", "PREVIOUS_TOKEN", "PreviousToken",
			"PAD_TOKEN", "PADTOKEN", "Pad_Token",
			"EXCHANGEDAUTHCREDENTIAL", "ExchangedAuthCredential", "RAWAUTHCREDENTIAL", "RawAuthCredential",
			"METHOD_SIGNATURE", "MethodSignature", "REQUEST_SIG", "RequestSig", "SEC-WEBSOCKET-KEY", "SEC_WEBSOCKET_KEY",
			"XTOKEN", "XToken", "Xtoken", "MY_XTOKEN", "My_XToken",
			"REAUTHORIZATION", "Reauthorization", "ReAuthorization",
			"PREAUTHORIZATION", "PreAuthorization", "DEAUTHORIZATION", "DeAuthorization",
			"UNAUTHORIZATION", "UnAuthorization", "XAUTHORIZATION", "MY_XAUTHORIZATION", "My_XAuthorization",
			"USER:NAME", "APP:THEME",
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
	runInParallel(t)
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
				// A linear scrub takes about 16 times longer on the 16 times
				// longer text; a quadratic one about 256 times longer.
				if ratio, tSmall, tLarge := linearRatio(64, func() { scrub(smallIn) }, func() { scrub(largeIn) }); ratio > 64 {
					t.Errorf("%d KB took %v, %d KB took %v: ratio %.0f, want linear", small>>10, tSmall, large>>10, tLarge, ratio)
				}

				// The result does not depend on concurrent use.
				in := shape(4 << 10)
				want, _ := sanitizeRawText(in)
				var wg sync.WaitGroup
				results := make([]string, 16)
				for i := range results {
					wg.Go(func() { results[i], _ = sanitizeRawText(in) })
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

// linearRatio returns the ratio of the shortest of three timed runs of large
// to the shortest of three of small, with small floored at 100µs, and those
// times. The runs of small and large alternate, so that a burst of load slows
// both alike. A ratio above limit is measured again, up to three times in
// all, and then three times more with the runs timed alone, while no other
// redaction test runs (see runInParallel), and the smallest ratio is
// returned: scheduler and garbage collector pauses on a loaded host, and
// other tests running in parallel, which take more from a long run than from
// a short one, inflate single measurements, while a quadratic scrub, whose
// ratio is far above limit, stays above it on every attempt.
func linearRatio(limit float64, small, large func()) (ratio float64, tSmall, tLarge time.Duration) {
	ratio, tSmall, tLarge = minRatio(limit, small, large)
	if ratio <= limit {
		return ratio, tSmall, tLarge
	}
	// The caller, a redaction test, holds scalingMu shared.
	scalingMu.RUnlock()
	scalingMu.Lock()
	defer func() {
		scalingMu.Unlock()
		scalingMu.RLock()
	}()
	return minRatio(limit, small, large)
}

// minRatio is the smallest ratio of up to three rounds of timed runs (see
// linearRatio), stopping at the first one at most limit.
func minRatio(limit float64, small, large func()) (ratio float64, tSmall, tLarge time.Duration) {
	ratio = math.Inf(1)
	for range 3 {
		ts, tl := time.Duration(math.MaxInt64), time.Duration(math.MaxInt64)
		for range 3 {
			ts, tl = min(ts, timed(small)), min(tl, timed(large))
		}
		if r := float64(tl) / float64(max(ts, 100*time.Microsecond)); r < ratio {
			ratio, tSmall, tLarge = r, ts, tl
		}
		if ratio <= limit {
			break
		}
	}
	return ratio, tSmall, tLarge
}

// scalingMu lets a linear-scaling check whose ratio stays above its limit
// time its runs again alone (see linearRatio): each redaction test, which runs
// in parallel with the others, holds it shared while it runs (see
// runInParallel), and the check holds it alone while it times, so that no
// other test takes the CPU from the runs that it compares.
var scalingMu sync.RWMutex

// runInParallel runs the redaction test t in parallel with the others and
// holds scalingMu shared until t and its subtests end.
func runInParallel(t *testing.T) {
	t.Parallel()
	scalingMu.RLock()
	t.Cleanup(scalingMu.RUnlock)
}

// timed returns how long a run of f takes.
func timed(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

// scalingUnits returns how many units the small and the large text of a
// linear-scaling check repeat: 128 and 2048, or 32 and 512 under the race
// detector, whose instrumentation slows these single-goroutine scans by more
// than ten times without adding concurrency for it to check. Either pair is 16
// times apart, so the ratio limit of the checks stays the same.
func scalingUnits() (small, large int) {
	if raceEnabled {
		return 32, 512
	}
	return 128, 2048
}

// largeTextLen returns how long a large text that the linear-scaling checks
// run through the entry points is: 224 KiB, or 32 KiB under the race detector
// (see scalingUnits). Texts that must run the scan budget out keep 224 KiB.
func largeTextLen() int {
	if raceEnabled {
		return 32 << 10
	}
	return 224 << 10
}

func TestCompoundKeyEdgeCasesAndLinearBackslashRegression(t *testing.T) {
	runInParallel(t)
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
			"REAUTHORIZATION=ok MY_XAUTHORIZATION: ok PREAUTHORIZATION=ok DEAUTHORIZATION=ok XTOKEN=5 MY_XTOKEN=ok NEXT_PAGE_TOKEN=p2 SYNC_TOKEN=s1 CONTINUATION_TOKEN=c1 PAD_TOKEN=pad0",
			`{"monkey":"a","donkey":"b","turkey":"c","turnkey":"d","keyboard":"e","passkey_enabled":true,"passwordless":true,"token_count":42,"next_page_token":"p2","sync_token":"s1","continuation_token":"c1","syncToken":"s2","continuationToken":"c2","method_signature":"foo","request_sig":"bar","app:theme":"dark","my_xtoken":"ok","prod_dbpassword":"ok","my_xauthorization":"ok","REAUTHORIZATION":"ok","MY_XAUTHORIZATION":"ok","XTOKEN":"ok"}`,
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
			`{"password": "[REDACTED]":SECRET123`,
			`{"password": "[REDACTED]": SECRET123`,
			`{"password": "[REDACTED]".SECRET123`,
			`{"password": "[REDACTED]")SECRET123`,
			`{"password": "[REDACTED]"]SECRET123`,
			`{"password": '[REDACTED]':SECRET123`,
			`{"password": '[REDACTED]': SECRET123`,
			`{"password": '[REDACTED]'.SECRET123`,
			`{"password": '[REDACTED]')SECRET123`,
			`{"password": '[REDACTED]']SECRET123`,
			`{"password": [REDACTED]:SECRET123`,
			`{"password": [REDACTED].SECRET123`,
			`{"token": "[REDACTED]", "note": "SECRET123`,
			`{"token": [REDACTED], "note": "SECRET123`,
			`{"password": "[REDACTED]" SECRET123}`,
			`{"password": [REDACTED] SECRET123}`,
			`[{"token": "[REDACTED]"}, "SECRET123`,
			`"token": "[REDACTED]", "note": "SECRET123`,
			`"X-Amz-Security-Token":"[REDACTED]":SECRET123`,
			`"X-Amz-Security-Token"=[REDACTED]:SECRET123`,
			`{"X-Amz-Security-Token":`,
			`"X-Amz-Security-Token":"[REDACTED]" dbPassword=SECRET123`,
		}
		wantTruncJSON := `"` + unparseableJSONPlaceholder + `"`
		for _, in := range failClosedInputs {
			if got := redactErrorMessage(in); got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, unparseableJSONPlaceholder)
			}
			if got, lost := sanitizeRawText(in); !lost || got != unparseableJSONPlaceholder || strings.Contains(got, "SECRET123") {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, got, lost, unparseableJSONPlaceholder)
			}
			if truncRaw, truncated, err := SmartTruncate(in, 1<<20); err != nil || !truncated || string(truncRaw) != wantTruncJSON || strings.Contains(string(truncRaw), "SECRET123") {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%s, true, nil)", in, truncRaw, truncated, err, wantTruncJSON)
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
				smallN, largeN := scalingUnits()
				small := strings.Repeat(shape.unit, smallN) + shape.suffix
				large := strings.Repeat(shape.unit, largeN) + shape.suffix
				redactErrorMessage(small)
				redactErrorMessage(large)
				if ratio, tSmall, tLarge := linearRatio(64, func() { redactErrorMessage(small) }, func() { redactErrorMessage(large) }); ratio > 64 {
					t.Errorf("%d pairs took %v, %d pairs took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
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
			"log: \"my_api_key\":\"[REDACTED]\"",
			true,
		)
		assertAllRedactors(t,
			"log: \"my_api_key\": \" password \": \"a\\\nb "+secret+"\"",
			"log: \"my_api_key\":\"[REDACTED]\"",
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
			`password:'[REDACTED]' y`,
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
			truncBytes, _, err := SmartTruncate(map[string]any{"message": in}, 1<<20)
			if err != nil {
				t.Fatalf("SmartTruncate(%q): %v", in, err)
			}
			if strings.Contains(string(truncBytes), secret) {
				t.Errorf("SmartTruncate(%q) leaked secret: %s", in, truncBytes)
			}
		}
		spacedContainerJoined := `dbPassword={a: b}/password: ` + secret
		wantSpacedContainerJoined := `dbPassword=[REDACTED]/password:[REDACTED]`
		if got := redactErrorMessage(spacedContainerJoined); got != wantSpacedContainerJoined {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", spacedContainerJoined, got, wantSpacedContainerJoined)
		}
		if gotSan, lost := sanitizeRawText(spacedContainerJoined); !lost || gotSan != wantSpacedContainerJoined {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", spacedContainerJoined, gotSan, lost, wantSpacedContainerJoined)
		}
	})

	t.Run("path errors and URL route slugs stay intact while real credentials redact", func(t *testing.T) {
		preserved := []string{
			"open /etc/ssl/private.key: no such file or directory",
			"open private.key: no such file or directory",
			"stat signing.key: permission denied",
			`open C:\ssl\secret.key: access denied`,
			"tls: failed to load /etc/ssl/api.key: open /etc/ssl/signing.key: permission denied",
			"tls: failed to load private.key: open signing.key: permission denied",
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

		const errPathMsg = "[ERROR] open private.key: no such file or directory"
		if got := redactErrorMessage(errPathMsg); got != errPathMsg {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", errPathMsg, got)
		}
		if got, lost := sanitizeRawText(errPathMsg); got != errPathMsg || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", errPathMsg, got, lost)
		}

		const secret = "S3CR3T_PATH_88"
		mustRedact := []string{
			"private.key: " + secret,
			"user:private.key: " + secret,
			"open private.key=" + secret,
			"kv/prod/db.password: " + secret,
			"config/github.token: " + secret,
			`C:\certs\client.secret: ` + secret,
			"/x/temp:a.b: " + secret,
			"credentials:\n  /etc/ssl/db.password: " + secret,
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
			if truncBytes, _, err := SmartTruncate(map[string]any{"message": in}, 1<<20); err != nil || strings.Contains(string(truncBytes), secret) {
				t.Errorf("SmartTruncate(%q) = (%s, %v), leaked secret", in, truncBytes, err)
			}
		}
	})

	t.Run("nested classifier fail-closed, over-budget, control-byte, inline-glued and word-joined pairs", func(t *testing.T) {
		const secret = "SYNTHETIC_NESTKEY_77"
		// Positive echo control on raw-string SmartTruncate before negative checks.
		echoBytes, echoTrunc, err := SmartTruncate("benign_probe_text", 1<<20)
		if err != nil || echoTrunc || string(echoBytes) != `"benign_probe_text"` {
			t.Fatalf("SmartTruncate positive control = (%s, %v, %v), want (\"benign_probe_text\", false, nil)", echoBytes, echoTrunc, err)
		}

		assertNoLeak := func(t *testing.T, label, in string) {
			t.Helper()
			gotSan, lost := sanitizeRawText(in)
			if !lost || strings.Contains(gotSan, secret) {
				t.Errorf("[%s] sanitizeRawText(%q) = (%q, %v), want lost=true without %q", label, in, gotSan, lost, secret)
			}
			if gotSan2, lost2 := sanitizeRawText(gotSan); gotSan2 != gotSan || lost2 {
				t.Errorf("[%s] sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", label, gotSan, gotSan2, lost2)
			}
			gotErr := redactErrorMessage(in)
			if strings.Contains(gotErr, secret) {
				t.Errorf("[%s] redactErrorMessage(%q) leaked %q: %q", label, in, secret, gotErr)
			}
			if gotErr2 := redactErrorMessage(gotErr); gotErr2 != gotErr {
				t.Errorf("[%s] redactErrorMessage second pass(%q) = %q, want unchanged", label, gotErr, gotErr2)
			}
			truncRaw, truncated, err := SmartTruncate(in, 1<<20)
			if err != nil || !truncated || strings.Contains(string(truncRaw), secret) {
				t.Errorf("[%s] SmartTruncate(%q) = (%s, %v, %v), want truncated=true without %q", label, in, truncRaw, truncated, err, secret)
			}
		}

		outers := []string{`"api.key"`, "api.key", `"DB_PASSWORD"`, "DB_PASSWORD"}
		quoteStyles := []string{`"`, `'`}
		for _, outer := range outers {
			for _, q := range quoteStyles {
				// 1. Quoted sensitive key after a non-sensitive prefix.
				assertNoLeak(t, "prefix_quoted_"+outer+"_"+q, fmt.Sprintf("log: %s: app:%spassword %s: %s%s%s", outer, q, q, q, secret, q))
				// 2. Control-byte (vertical tab) delimiter before and after inner colon.
				assertNoLeak(t, "vtab_before_"+outer+"_"+q, fmt.Sprintf("log: %s: password\v: %s%s%s", outer, q, secret, q))
				assertNoLeak(t, "vtab_after_"+outer+"_"+q, fmt.Sprintf("log: %s: password:\v%sprefix %s%s", outer, q, secret, q))
				// 3. Inline quoted candidate containing its own sensitive key and delimiter.
				assertNoLeak(t, "inline_quoted_delim_"+outer+"_"+q, fmt.Sprintf("log: %s:%stemp:c=%s%s", outer, q, q, secret))
				// 4. Word-byte joins (letter, underscore, digit) before quoted inner key, plain and padded.
				for _, join := range []string{"abcX", "abc_", "abc1"} {
					assertNoLeak(t, "word_join_"+join+"_"+outer+"_"+q, fmt.Sprintf("log: %s: %s%stoken%s: %s%s%s", outer, join, q, q, q, secret, q))
					assertNoLeak(t, "word_join_padded_"+join+"_"+outer+"_"+q, fmt.Sprintf("log: %s: %s%s token %s: %s%s%s", outer, join, q, q, q, secret, q))
				}
				// 5. Boundary key lengths and padding lengths around maxNestedKeyLen (127, 128, 129, 200 bytes).
				for _, totalQuotedLen := range []int{127, 128, 129, 200} {
					bodyLen := totalQuotedLen - 2 - len("-password")
					longKey := q + strings.Repeat("x", bodyLen) + "-password" + q
					assertNoLeak(t, fmt.Sprintf("long_key_%d_%s_%s", totalQuotedLen, outer, q), fmt.Sprintf("log: %s:\n    %s: %s%s%s", outer, longKey, q, secret, q))
					padLen := totalQuotedLen - 2 - len("password")
					paddedKey := q + strings.Repeat(" ", padLen) + "password" + q
					assertNoLeak(t, fmt.Sprintf("padded_key_%d_%s_%s", totalQuotedLen, outer, q), fmt.Sprintf("log: %s:\n    %s: %s%s%s", outer, paddedKey, q, secret, q))
					assertNoLeak(t, fmt.Sprintf("sameline_padded_%d_%s_%s", totalQuotedLen, outer, q), fmt.Sprintf("log: %s: %s: %s", outer, paddedKey, secret))
				}
				assertNoLeak(t, "long_unquoted_prefix_padded_quote_"+outer+"_"+q, fmt.Sprintf("log: %s: %s%s password %s: %s", outer, strings.Repeat("a", 130), q, q, secret))
				assertNoLeak(t, "short_join_long_padded_quote_"+outer+"_"+q, fmt.Sprintf("log: %s: abcX%s%spassword %s: %s", outer, q, strings.Repeat(" ", 130), q, secret))
			}
		}
		assertNoLeak(t, "standalone_inline_quoted_dq", `_PASSWORD:"temp:c="`+secret)
		assertNoLeak(t, "standalone_inline_quoted_sq", `_PASSWORD:'temp:c='`+secret)
		assertNoLeak(t, "escaped_quote_inside_nested_temp_body", `log: "DB_PASSWORD": "temp:\" = '`+secret+`'`)
		assertNoLeak(t, "line_start_secret_before_windows_backslash_suffix", "client_secret:\n  "+secret+"C:\\ssl\\password: x")
		assertNoLeak(t, "multiline_escaped_nl_prefix_quoted", "log: \"api.key\": app:\"password \": \"a\\\nb "+secret+"\"")
		assertNoLeak(t, "multiline_escaped_nl_over_budget_quoted", "log: \"api.key\":\n    \""+strings.Repeat("x", 130)+"-password\": \"a\\\nb "+secret+"\"")
		assertNoLeak(t, "multiline_escaped_nl_single_quoted", "log: DB_PASSWORD: app:'password ': 'a\\\nb "+secret+"'")
		assertNoLeak(t, "multiline_escaped_nl_before_next_padded_key", "log: \"api.key\": \"a\\\n}, \" my_password \": \""+secret+"\"")
	})

	t.Run("temp keys with escaped closing quotes, Windows-path inner keys, stacked bare temp keys and multiline headers", func(t *testing.T) {
		const secret = "SYNTHETIC_EXACTKEY_88"
		assertExactAllThree := func(t *testing.T, in, want string) {
			t.Helper()
			if got := redactErrorMessage(in); got != want || strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", in, got, want)
			}
			gotSan, lost := sanitizeRawText(in)
			if !lost || gotSan != want || strings.Contains(gotSan, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", in, gotSan, lost, want)
			}
			truncRaw, truncated, err := SmartTruncate(in, 1<<20)
			if err != nil || !truncated || strings.Contains(string(truncRaw), secret) {
				t.Fatalf("SmartTruncate(%q) = (%s, %v, %v), want truncated=true without secret", in, truncRaw, truncated, err)
			}
			var decoded string
			if err := json.Unmarshal(truncRaw, &decoded); err != nil || decoded != want {
				t.Errorf("SmartTruncate(%q) decoded = %q (err=%v), want %q", in, decoded, err, want)
			}
		}

		// Finding 2: temp:\":SYNTHETIC, temp:\':SYNTHETIC, and temp:a\":SYNTHETIC control.
		assertExactAllThree(t, `temp:\":`+secret, `temp:\":[REDACTED]`)
		assertExactAllThree(t, `temp:\':`+secret, `temp:\':[REDACTED]`)
		assertExactAllThree(t, `temp:a\":`+secret, `temp:a\":[REDACTED]`)

		// Finding 5: nested Windows-path inner keys under credentials:.
		assertExactAllThree(t, "credentials:\n  C:\\ssl\\password: "+secret+"\n  region: us", "credentials:\n  C:\\ssl\\password:[REDACTED]\n  region: us")
		assertExactAllThree(t, "credentials: C:\\ssl\\password: "+secret, "credentials:[REDACTED]")
		assertExactAllThree(t, "credentials:\n  \"C:\\ssl\\password\": \""+secret+"\"\n  region: us", "credentials:\n  \"C:\\ssl\\password\":\"[REDACTED]\"\n  region: us")
		assertExactAllThree(t, "credentials:\n  C:\\ssl\\secret.key: "+secret, "credentials:[REDACTED]")

		// Finding 6: multiline auth headers, padded quoted text keys, and composite escaped-newline pairs.
		// A header value on the next line keeps its line break, so that a
		// second pass does not read the text after it as the rest of the
		// header line.
		assertExactAllThree(t, "api_key:\n  "+secret, "api_key:\n  [REDACTED]")
		assertExactAllThree(t, "Authorization:\n  \""+secret+"\"", "Authorization:\n  \"[REDACTED]\"")
		assertExactAllThree(t, "' client_secret ': "+secret, "'client_secret':[REDACTED]")
		assertExactAllThree(t, `" password " = `+secret, `"password"=[REDACTED]`)
		assertExactAllThree(t, `\' client_secret \': `+secret, `\'client_secret\':[REDACTED]`)
		for _, composite := range []string{
			"\\\"temp:a:b\n\\\": \\\"first_" + secret + "\\\" api_key:\n  " + secret,
			"\\\"temp:a:b\n\\\": \\\"first_" + secret + "\\\" Authorization:\n  \"" + secret + "\"",
			"\\\"temp:a:b\n\\\": \\\"first_" + secret + "\\\" ' client_secret ': " + secret,
		} {
			if got := redactErrorMessage(composite); strings.Contains(got, secret) {
				t.Errorf("redactErrorMessage(%q) leaked secret: %q", composite, got)
			}
			if gotSan, lost := sanitizeRawText(composite); !lost || strings.Contains(gotSan, secret) {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true without secret", composite, gotSan, lost)
			}
			if truncRaw, _, err := SmartTruncate(composite, 1<<20); err != nil || strings.Contains(string(truncRaw), secret) {
				t.Errorf("SmartTruncate(%q) = (%s, %v), leaked secret", composite, truncRaw, err)
			}
		}

		// Finding 7: bare stacked temp:a:b: and user:temp:a:b: keys (nested and standalone).
		assertExactAllThree(t, "credentials:\n  temp:a:b: "+secret+"\n  region: us", "credentials:\n  temp:a:[REDACTED]\n  region: us")
		assertExactAllThree(t, "credentials:\n  user:temp:a:b: "+secret+"\n  region: us", "credentials:\n  user:temp:a:[REDACTED]\n  region: us")
		assertExactAllThree(t, "credentials: temp:a:b: "+secret, "credentials:[REDACTED]")
		assertExactAllThree(t, "temp:a:b: "+secret, "temp:a:[REDACTED]")

		// Nested inner quoted keys ending before an outer closing quote ('token'": ...)
		// and 129-byte quoted inner keys under api.key.
		assertExactAllThree(t, `log: "api_key": "'token'": password: `+secret, `log: "api_key":"[REDACTED]"`)
		assertExactAllThree(t, `api_key="'token'": password: `+secret, `api_key="[REDACTED]"`)
		assertExactAllThree(t, `log: "api_key": "abc:'token'": password: `+secret, `log: "api_key":"[REDACTED]"`)
		assertExactAllThree(t, "log: \"api_key\": \"abc:'token'\": password: \"a\\\nb "+secret+"\"", `log: "api_key":"[REDACTED]"`)
		assertExactAllThree(t, `log: "api.key": "`+strings.Repeat("a", 120)+`-password": `+secret, `log: "api.key":"[REDACTED]"`)
	})

	t.Run("over-budget nested containers and quoted inner values with glued suffixes fail closed", func(t *testing.T) {
		const secret = "S3CR3T_R3C_99"
		a130 := strings.Repeat("a", 130)

		assertAllThreeIdempotent := func(t *testing.T, label, in string) {
			t.Helper()
			gotErr := redactErrorMessage(in)
			if strings.Contains(gotErr, secret) {
				t.Errorf("[%s] redactErrorMessage(%q) leaked %q: %q", label, in, secret, gotErr)
			}
			if gotErr2 := redactErrorMessage(gotErr); gotErr2 != gotErr {
				t.Errorf("[%s] redactErrorMessage second pass(%q) = %q, want unchanged", label, gotErr, gotErr2)
			}
			gotSan, lost := sanitizeRawText(in)
			if !lost || strings.Contains(gotSan, secret) {
				t.Errorf("[%s] sanitizeRawText(%q) = (%q, %v), want lost=true without %q", label, in, gotSan, lost, secret)
			}
			if gotSan2, lost2 := sanitizeRawText(gotSan); gotSan2 != gotSan || lost2 {
				t.Errorf("[%s] sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", label, gotSan, gotSan2, lost2)
			}
			truncRaw, truncated, err := SmartTruncate(in, 1<<20)
			if err != nil || !truncated || strings.Contains(string(truncRaw), secret) {
				t.Errorf("[%s] SmartTruncate(%q) = (%s, %v, %v), want truncated=true without %q", label, in, truncRaw, truncated, err, secret)
			}
		}

		containerCases := []struct {
			label, in string
		}{
			{"json_key_overbudget_obj", `log: "api.key": credentials:{"` + a130 + `": 1, "x": "` + secret + `"}`},
			{"text_key_overbudget_arr", `DB_PASSWORD: credentials:["` + a130 + `", "` + secret + `"]`},
			{"text_key_overbudget_obj_space", `DB_PASSWORD: password:{"note":"` + a130 + ` bbb","x":"` + secret + `"}`},
			{"fixed_outer_overbudget_obj", `credentials: password:{"` + a130 + `": 1, "x": "` + secret + `"}`},
			{"joined_quoted_inner_overbudget_obj", `DB_PASSWORD: x/"credentials":{"` + a130 + `": 1, "x": "` + secret + `"}`},
			{"overbudget_arr_of_objs", `DB_PASSWORD: credentials:[{"` + a130 + `":1}, "` + secret + `"]`},
			{"closed_container_then_overbudget_obj", `DB_PASSWORD: foo:{"k":1}password:{"` + a130 + `": 1, "x": "` + secret + `"}`},
			{"closed_quote_braces_then_overbudget_obj", `DB_PASSWORD: foo"{a:[1"password:{"` + a130 + `": 1, "x": "` + secret + `"}`},
			{"unclosed_outer_brace_then_overbudget_obj", `DB_PASSWORD: a{password:{"` + a130 + `": 1, "x": "` + secret + `"}}`},
		}
		for _, tc := range containerCases {
			assertAllThreeIdempotent(t, tc.label, tc.in)
		}

		gluedQuoteCases := []struct {
			label, in string
		}{
			{"joined_unspaced_lead_space", `DB_PASSWORD=x/password=" ` + secret + `"tail`},
			{"json_outer_unspaced_lead_space", `log: "api.key": password=" ` + secret + `"tail`},
			{"joined_unspaced_lead_comma", `DB_PASSWORD=x/password=",` + secret + `"tail`},
			{"json_outer_unspaced_lead_brace", `log: "api.key": password="} ` + secret + `"tail`},
			{"json_outer_spaced_lead_space", `log: "api.key": password: " ` + secret + `"tail`},
			{"yaml_outer_spaced_lead_space", "config:\n  \"DB_PASSWORD\":\n    password: \" " + secret + "\"tail\n  region: us"},
			{"json_outer_spaced_nospace_glued", `log: "api.key": password: "` + secret + `"tail`},
			{"single_quote_spaced_lead_space", `log: "api.key": password: ' ` + secret + `'tail`},
			{"escaped_quote_spaced_lead_space", `log: "api.key": password: \" ` + secret + `\"tail`},
			{"quoted_inner_key_lead_space", `log: "api.key": 'password': ' ` + secret + `'tail`},
			{"padded_quoted_inner_key_lead_space", `log: "api.key": " password ": " ` + secret + `"tail`},
			{"prefix_padded_quoted_inner_key_lead_space", `log: "api.key": app:"password ": " ` + secret + `"tail`},
			{"overbudget_spaced_lead_space", `DB_PASSWORD=` + a130 + `/password: " ` + secret + `"tail`},
			{"redacted_prefix_joined_lead_space", `dbPassword=[REDACTED]/password=" ` + secret + `"tail`},
			{"chained_three_pairs_lead_space", `DB_PASSWORD=x-token: y/password=" ` + secret + `"tail`},
			{"stacked_bare_temp_lead_space", `temp:a:b: " ` + secret + `"tail`},
			{"secret_in_glued_tail_unspaced", `DB_PASSWORD=x/password=" x"` + secret},
			{"secret_in_glued_tail_spaced", `log: "api.key": password: " x"` + secret},
			{"secret_after_empty_quotes_spaced", `log: "api.key": password: ""` + secret},
			{"apostrophe_prose_before_single_quote", `can't connect: DB_PASSWORD=x/password=' ` + secret + `'tail`},
			{"unclosed_prev_line_quote", "note: \"unclosed\nlog: \"api.key\": password: \" " + secret + "\"tail"},
			{"enclosing_quote_nonspace_before_close", `got "DB_PASSWORD=x/password=" ` + secret + `"tail"`},
			{"colon_inside_spanning_quote_underbudget", `DB_PASSWORD: password="hello: word1 ` + secret + `"`},
			{"spaced_colon_inside_spanning_quote_underbudget", `log: "api.key": password="hello : word1 ` + secret + `"`},
			{"colon_inside_spanning_quote_overbudget", `DB_PASSWORD: ` + a130 + `/password="hello: word1 ` + secret + `"`},
			{"spaced_colon_inside_spanning_quote_overbudget", `DB_PASSWORD: ` + a130 + `/password="hello : word1 ` + secret + `"`},
			{"spaced_container_prefix_overbudget_quoted_key", "credentials:\n  {a: b}/\"" + a130 + "-password\"\v: \"hello: word1 " + secret + "\""},
		}
		for _, tc := range gluedQuoteCases {
			assertAllThreeIdempotent(t, tc.label, tc.in)
		}

		// Multiline auth header fallthrough with leading-space quoted value + glued tail.
		multilineHdr := "Authorization:\n  \" " + secret + "\"tail"
		if got := redactErrorMessage(multilineHdr); strings.Contains(got, secret) {
			t.Errorf("redactErrorMessage(%q) leaked %q: %q", multilineHdr, secret, got)
		}
		if gotSan, lost := sanitizeRawText(multilineHdr); !lost || strings.Contains(gotSan, secret) {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want lost=true without %q", multilineHdr, gotSan, lost, secret)
		}
		if truncRaw, truncated, err := SmartTruncate(multilineHdr, 1<<20); err != nil || !truncated || strings.Contains(string(truncRaw), secret) {
			t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want truncated=true without %q", multilineHdr, truncRaw, truncated, err, secret)
		}

		// Enclosing-quote prose controls must remain intact without over-redacting trailing strings.
		proseControls := []struct {
			in, want string
		}{
			{`failed: "password=SECRETP-token: " for user "bob"`, `failed: "password=[REDACTED] " for user "bob"`},
			{`got "password=SECRETP-token=", want "abc"`, `got "password=[REDACTED], want "abc"`},
			{`got "password=SECRETP-token: ", want ""`, `got "password=[REDACTED] ", want ""`},
			{`got \"password=SECRETP-token: \", want \"\"`, `got \"password=[REDACTED] \", want \"\"`},
		}
		for _, pc := range proseControls {
			if got := redactErrorMessage(pc.in); got != pc.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", pc.in, got, pc.want)
			}
		}

		// YAML sibling preservation check on spaced leading-whitespace glued quote.
		yamlIn := "config:\n  \"DB_PASSWORD\":\n    password: \" " + secret + "\"tail\n  region: us"
		if got := redactErrorMessage(yamlIn); !strings.Contains(got, "region: us") || strings.Contains(got, secret) {
			t.Errorf("redactErrorMessage(%q) = %q, want region: us preserved and secret redacted", yamlIn, got)
		}

		// 32 KB linear-scaling checks on over-budget containers and glued quotes.
		for _, shape := range []struct {
			name   string
			mkText func(n int) string
		}{
			{
				name: "chained closed containers before over-budget container",
				mkText: func(n int) string {
					return "DB_PASSWORD: " + strings.Repeat("a:{\"k\":1}", n) + " x=1"
				},
			},
			{
				name: "unclosed non-delimited braces in over-budget token",
				mkText: func(n int) string {
					return "DB_PASSWORD: a" + strings.Repeat("{", n) + " x=1"
				},
			},
			{
				name: "enclosing quote before repeated leading-space glued quotes",
				mkText: func(n int) string {
					return `"` + strings.Repeat(`password=" a"tail `, n)
				},
			},
		} {
			t.Run(shape.name, func(t *testing.T) {
				smallN, largeN := scalingUnits()
				small := shape.mkText(smallN)
				large := shape.mkText(largeN)
				redactErrorMessage(small)
				redactErrorMessage(large)
				if ratio, tSmall, tLarge := linearRatio(64, func() { redactErrorMessage(small) }, func() { redactErrorMessage(large) }); ratio > 64 {
					t.Errorf("%d units took %v, %d units took %v: ratio %.0f, want linear", smallN, tSmall, largeN, tLarge, ratio)
				}
			})
		}
	})

	t.Run("spanning quotes before quoted keys and container-prefixed over-budget idempotency", func(t *testing.T) {
		const secret = "S3CR3T_R3D_88"
		const secret2 = "S3CR3T_R3D_99"
		a130 := strings.Repeat("a", 130)

		assertNoSecretsAndIdempotent := func(t *testing.T, label, in string, forbidden ...string) {
			t.Helper()
			gotErr := redactErrorMessage(in)
			for _, f := range forbidden {
				if strings.Contains(gotErr, f) {
					t.Errorf("[%s] redactErrorMessage(%q) leaked %q: %q", label, in, f, gotErr)
				}
			}
			if gotErr2 := redactErrorMessage(gotErr); gotErr2 != gotErr {
				t.Errorf("[%s] redactErrorMessage second pass(%q) = %q, want unchanged", label, gotErr, gotErr2)
			}
			gotSan, lost := sanitizeRawText(in)
			if !lost {
				t.Errorf("[%s] sanitizeRawText(%q) = (%q, %v), want lost=true", label, in, gotSan, lost)
			}
			for _, f := range forbidden {
				if strings.Contains(gotSan, f) {
					t.Errorf("[%s] sanitizeRawText(%q) leaked %q: %q", label, in, f, gotSan)
				}
			}
			if gotSan2, lost2 := sanitizeRawText(gotSan); gotSan2 != gotSan || lost2 {
				t.Errorf("[%s] sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", label, gotSan, gotSan2, lost2)
			}
			truncRaw, truncated, err := SmartTruncate(in, 1<<20)
			if err != nil || !truncated {
				t.Errorf("[%s] SmartTruncate(%q) = (%s, %v, %v), want truncated=true without error", label, in, truncRaw, truncated, err)
			}
			for _, f := range forbidden {
				if strings.Contains(string(truncRaw), f) {
					t.Errorf("[%s] SmartTruncate(%q) leaked %q: %s", label, in, f, truncRaw)
				}
			}
			var truncDecoded string
			if err := json.Unmarshal(truncRaw, &truncDecoded); err == nil && !isKnownSentinel(truncDecoded) {
				truncRaw2, truncated2, err2 := SmartTruncate(truncDecoded, 1<<20)
				if err2 != nil || truncated2 || string(truncRaw2) != string(truncRaw) {
					t.Errorf("[%s] SmartTruncate second pass(%q) = (%s, %v, %v), want (%s, false, nil)", label, truncDecoded, truncRaw2, truncated2, err2, truncRaw)
				}
			}
		}

		spanningQuoteCases := []struct {
			label, in string
		}{
			{"overbudget_midword_sq_compound_outer", "DB_PASSWORD: " + a130 + "/x'y 'token': " + secret},
			{"overbudget_midword_sq_fixed_outer", "token: " + a130 + "/x'y 'token': " + secret},
			{"overbudget_midword_dq_compound_outer", `DB_PASSWORD: ` + a130 + `/x"y "token"= ` + secret},
			{"overbudget_midword_dq_fixed_outer", `token: ` + a130 + `/x"y "token"= ` + secret},
			{"overbudget_nonkey_delim_sq", "DB_PASSWORD: " + a130 + "='foo 'token': " + secret},
			{"overbudget_nonkey_slash_dq", `DB_PASSWORD: ` + a130 + `/" "token"= ` + secret},
			{"underbudget_midword_dq_inner_colon", `token: x"password :a"token"= ` + secret},
			{"underbudget_midword_sq_inner_colon", "token: x'password :a'token': " + secret},
			{"overbudget_senskey_delim_sq_steals_quote", "DB_PASSWORD: " + a130 + "-password='foo 'token': " + secret},
			{"underbudget_senskey_delim_sq_steals_quote", "token: password='foo 'token': " + secret},
			{"underbudget_slash_dq_inner_colon_steals_quote", `token: /"password :a"token"= ` + secret},
			{"outer_quoted_steals_padded_key_quote", "DB_PASSWORD: 'foo ' token ': " + secret},
			{"overbudget_outer_quoted_steals_padded_key_quote", "DB_PASSWORD: '" + a130 + " ' token ': " + secret},
			{"underbudget_senskey_delim_mismatched_quote_key", `DB_PASSWORD: password='foo 'token": ` + secret},
			{"overbudget_glued_secret_after_delim_steals_quote", `DB_PASSWORD: ` + a130 + `-password='foo 'token': "` + " " + secret + `"tail`},
		}
		for _, tc := range spanningQuoteCases {
			assertNoSecretsAndIdempotent(t, tc.label, tc.in, secret)
		}
		assertNoSecretsAndIdempotent(t, "double_secret_unclosed_and_stolen_quote",
			`DB_PASSWORD: `+a130+`-password="word1 `+secret+` "token"= `+secret2, secret, secret2)

		// Container-prefixed over-budget auth-header and nested-key idempotency.
		for _, hdr := range []string{"-api_key", "-x-api-key", "-authorization", "-proxy-authorization"} {
			assertNoSecretsAndIdempotent(t, "container_overbudget_hdr_"+hdr, "dbPassword={a: b}/"+a130+hdr+": "+secret, secret)
		}
		assertNoSecretsAndIdempotent(t, "container_overbudget_glued_after_redacted",
			`dbPassword=[REDACTED]/`+a130+`-password: "[REDACTED]"`+secret, secret)

		preRedactedCases := []string{
			`dbPassword=[REDACTED]/` + a130 + `-password: "[REDACTED]"`,
			`dbPassword=[REDACTED]/` + a130 + `-api_key: [REDACTED]`,
			`dbPassword=[REDACTED]/` + a130 + `-api_key : [REDACTED]`,
			`dbPassword=[REDACTED]/"` + strings.Repeat(" ", 130) + `password": "[REDACTED]"`,
			`dbPassword=[REDACTED]/"` + a130 + `-password": "[REDACTED]"`,
		}
		for _, in := range preRedactedCases {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
			}
			if gotSan, lost := sanitizeRawText(in); gotSan != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, gotSan, lost)
			}
			truncRaw, truncated, err := SmartTruncate(in, 1<<20)
			if err != nil || truncated {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want truncated=false", in, truncRaw, truncated, err)
			}
		}

		// Unclosed value quotes before quoted keys, with and without an
		// enclosing prose string, and enclosing closers glued to further text.
		for _, tc := range []struct{ label, in string }{
			{"json_unclosed_value_quote_before_key", `{"token": "a "api_key": "` + secret + `"}`},
			{"escaped_json_unclosed_value_quote_before_key", `"{\"token\": \"a \"api_key\": \"` + secret + `\"}"`},
			{"enclosing_dq_unclosed_sq_value_before_key", `msg "token: 'a 'api_key': ` + secret + `"`},
			{"enclosing_sq_unclosed_dq_value_before_key", `msg 'token: "a "api_key": ` + secret + `'`},
			{"enclosing_closer_glued_to_quoted_key", `"DB_PASSWORD: 'foo '"token": ` + secret},
			{"prefixed_enclosing_closer_glued_to_quoted_key", `err: "DB_PASSWORD: 'foo '"token": ` + secret},
			{"enclosing_closer_glued_to_word", `msg: "password='` + secret + `'"tail`},
			{"enclosing_closer_then_unspaced_colon", `parsing "password='` + secret + `'":tail`},
			{"enclosing_closer_then_sq_key", `failed: "password='a'" 'token': ` + secret},
			{"escaped_enclosing_closer_then_escaped_key", `failed: \"password='a'\" \"token\": ` + secret},
			{"same_char_value_and_enclosing_closer", `failed: "password="` + secret + `"" for user "bob"`},
			{"stolen_padded_key_after_colon", `DB_PASSWORD: "foo ": password ": ` + secret},
			{"quoted_value_then_unspaced_colon", `failed: password: "prefix":` + secret},
		} {
			assertNoSecretsAndIdempotent(t, tc.label, tc.in, secret)
		}

		// Prose controls: a value followed only by the enclosing closer and a
		// boundary keeps the closer and the trailing prose.
		for _, tc := range []struct{ label, in, want string }{
			{"dq_closer_then_space", `failed: "password='SECRETQ'" for user "bob"`, `failed: "password='[REDACTED]'" for user "bob"`},
			{"overbudget_dq_closer_then_space", `failed: "password='` + a130 + `SECRETQ'" for user "bob"`, `failed: "password='[REDACTED]'" for user "bob"`},
			{"escaped_dq_closer_then_space", `failed: \"password='SECRETQ'\" for user \"bob\"`, `failed: \"password='[REDACTED]'\" for user \"bob\"`},
			{"escaped_dq_closer_then_escaped_newline", `failed: \"password='SECRETQ'\"\nretry`, `failed: \"password='[REDACTED]'\"\nretry`},
			{"sq_closer_then_space", `failed: 'password="SECRETQ"' for user 'bob'`, `failed: 'password="[REDACTED]"' for user 'bob'`},
			{"dq_closer_at_end", `failed: "password='SECRETQ'"`, `failed: "password='[REDACTED]'"`},
			{"dq_closer_then_comma", `got "password='SECRETQ'", want ""`, `got "password='[REDACTED]'", want ""`},
			{"dq_closer_then_colon_space", `parsing "password='SECRETQ'": invalid syntax`, `parsing "password='[REDACTED]'": invalid syntax`},
			{"overbudget_quoted_then_colon_space", `failed: password: "` + a130 + `SECRETQ": invalid`, `failed: password:"[REDACTED]": invalid`},
			{"overbudget_escaped_quoted_then_escaped_newline_fails_closed", `credentials: token: \" ` + a130 + `SECRETQ\"\nretry`, redactedSensitiveTextPlaceholder},
			{"dq_closer_then_paren", `call("password='SECRETQ'") failed`, `call("password='[REDACTED]'") failed`},
		} {
			got := redactErrorMessage(tc.in)
			if got != tc.want {
				t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
			}
			if got2 := redactErrorMessage(got); got2 != got {
				t.Errorf("[%s] redactErrorMessage second pass(%q) = %q, want unchanged", tc.label, got, got2)
			}
			san, _ := sanitizeRawText(tc.in)
			if strings.Contains(san, "SECRETQ") {
				t.Errorf("[%s] sanitizeRawText(%q) leaked: %q", tc.label, tc.in, san)
			}
			if san2, lost2 := sanitizeRawText(san); san2 != san || lost2 {
				t.Errorf("[%s] sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", tc.label, san, san2, lost2)
			}
		}

		// Already-redacted text, including values that textValueEnd trims and
		// over-budget container-prefixed headers or keys, stays byte-identical.
		for _, in := range []string{
			`failed: "password='[REDACTED]'" for user "bob"`,
			`failed: \"password='[REDACTED]'\" for user \"bob\"`,
			`password: "[REDACTED]".`,
			`failed: {Password: [REDACTED]}`,
			`failed: [Password: [REDACTED]]`,
			"dbPassword=[REDACTED]/" + a130 + "-authorization: [REDACTED]\nregion=us-east-1",
			"dbPassword=[REDACTED]/" + a130 + "-x-api-key: '[REDACTED]'\nregion=us-east-1",
			`dbPassword=[REDACTED]/` + a130 + `-access_token: '[REDACTED]' region=us-east-1`,
		} {
			if got := redactErrorMessage(in); got != in {
				t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
			}
			if gotSan, lost := sanitizeRawText(in); gotSan != in || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, gotSan, lost)
			}
			if truncRaw, truncated, err := SmartTruncate(in, 1<<20); err != nil || truncated {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want truncated=false", in, truncRaw, truncated, err)
			}
		}

		// Container-prefixed values with redacted or secret tails converge in
		// one pass and keep trailing non-sensitive text.
		for _, tc := range []struct{ label, in, forbidden, keptSuffix string }{
			{"bracket_value_then_redacted_tail", `auth_token=[tk1 tk2 tk3]:[REDACTED] region=us-east-1`, "tk2", " region=us-east-1"},
			{"bearer_bracket_value_then_redacted_tail", `authorization=[Bearer bx1 bx2]:[REDACTED] region=us-east-1`, "bx1", " region=us-east-1"},
			{"nested_brace_bracket_then_redacted_tail", `cfg={secret=[sv1]:[REDACTED]} region=us-east-1`, "sv1", " region=us-east-1"},
			{"brace_nested_key_then_redacted_tail", `auth={token=av1}:[REDACTED] region=us-east-1`, "av1", " region=us-east-1"},
			{"container_overbudget_quoted_secret", `dbPassword={a: b}/` + a130 + `-password: "` + secret + `" region=us-east-1`, secret, " region=us-east-1"},
			{"bracket_container_overbudget_hdr", "dbPassword=[a b]/" + a130 + "-api_key: " + secret + "\nregion=us-east-1", secret, "\nregion=us-east-1"},
			{"bracket_container_overbudget_nonhdr", `dbPassword=[a b]/` + a130 + `-access_token: ` + secret + ` region=us-east-1`, secret, " region=us-east-1"},
			{"overbudget_bare_glued_after_redacted_nonhdr", `dbPassword=[REDACTED]/` + a130 + `-password: [REDACTED]` + secret + ` region=us-east-1`, secret, " region=us-east-1"},
		} {
			assertNoSecretsAndIdempotent(t, tc.label, tc.in, tc.forbidden)
			if got := redactErrorMessage(tc.in); !strings.Contains(got, tc.keptSuffix) {
				t.Errorf("[%s] redactErrorMessage(%q) = %q, over-redacted trailing text %q", tc.label, tc.in, got, tc.keptSuffix)
			}
		}

		// A header whose value holds a word glued to its placeholder is not
		// redacted yet, so the words of the over-budget chain before it,
		// which the expressions on main redacted with the value of
		// dbPassword, count again, and the text fails closed.
		gluedHdr := "dbPassword=[REDACTED]/" + a130 + "-api_key: [REDACTED]" + secret + "\nregion=us-east-1"
		assertNoSecretsAndIdempotent(t, "overbudget_bare_glued_after_redacted_hdr", gluedHdr, secret)
		if got := redactErrorMessage(gluedHdr); got != redactedSensitiveTextPlaceholder {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", gluedHdr, got, redactedSensitiveTextPlaceholder)
		}

		// Linear-scaling check on spanning stolen-quote chains up to 32 KB.
		mkStolenChain := func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat("password='a 'token': x ", n)
		}
		smallChain := mkStolenChain(128)
		largeChain := mkStolenChain(1536)
		redactErrorMessage(smallChain)
		redactErrorMessage(largeChain)
		if ratio, tSmall, tLarge := linearRatio(64, func() { redactErrorMessage(smallChain) }, func() { redactErrorMessage(largeChain) }); ratio > 64 {
			t.Errorf("stolen-quote chain 128 units took %v, 1536 units took %v: ratio %.0f, want linear", tSmall, tLarge, ratio)
		}
	})
}

// legacyReferenceScrub returns a reconstruction of the pre-scanner regular
// expressions for the five text-pattern passes of scrubTextPatterns (see
// TestLegacyRegexScannerDifferential) and of the quoted sensitive key pattern
// of hasQuotedSensitiveKey.
func legacyReferenceScrub() (refScrub func(string) string, refQuotedKeyRe *regexp.Regexp) {
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
	refQuotedKeyRe = regexp.MustCompile(`(?i)["'][\s\p{Z}]*(` + refSensKey + `)[\s\p{Z}]*["'][\s\p{Z}]*[:=]`)

	refScrub = func(s string) string {
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
	return refScrub, refQuotedKeyRe
}

// TestLegacyRegexScannerDifferential compares scrubTextPatterns and
// hasQuotedSensitiveKey against a reconstruction of the pre-scanner regular
// expressions for those five text-pattern passes (excluding JSON container
// parsing and backslash-escape whole-text fail-closed wrappers) over generated
// multi-pair, nested, and joined inputs.
func TestLegacyRegexScannerDifferential(t *testing.T) {
	runInParallel(t)
	refScrub, refQuotedKeyRe := legacyReferenceScrub()

	rng := rand.New(rand.NewPCG(20261110, 2250))
	keys := []string{
		"password", "api_key", "api-key", "apiKey", "API_KEY",
		"pa\u017f\u017fword", "api_\u212aey", "x\u017fecret",
		"token", "access_token", "client_secret", "Authorization", "x-api-key",
		"temp:abc", "temp:a:b{c", "temp:cache/val=1",
		"AP\u0130_KEY", "CL\u0130ENT_SECRET", "DB_PASSWORD", "dbPassword",
		"user:api_key", "app:temp:conn", "private.key",
		"kv/prod/db.password", "config/github.token", `C:\certs\client.secret`, "/x/temp:a.b",
	}
	pads := []string{"", " ", "  ", "\t", "\n", "\r\n", "\u00a0"}
	quotes := []string{"", `"`, `'`, `\"`, `\'`}
	delims := []string{":", "=", " : ", " = ", ":\n  ", ":\t", "\v: "}
	joins := []string{" ", ", ", "; ", "&", ";", "\n  ", " | ", "/", "abcX", "abc_", "[abc];", "app:"}
	prefixes := []string{"log: ", "config:\n  ", "err: {", `log: "api.key": `, "credentials:\n  ", "DB_PASSWORD: ", ""}

	for i := range 2500 {
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
		if i%10 == 0 {
			k := keys[i%len(keys)]
			unredactedMalformed := []string{
				`{"` + k + `": "S3CR3T_MALFORMED`,
			}
			if !strings.Contains(k, "=") {
				unredactedMalformed = append(unredactedMalformed, k+`: S3CR3T_MALFORMED`)
			}
			for _, malformed := range unredactedMalformed {
				if got := redactErrorMessage(malformed); strings.Contains(got, "S3CR3T_MALFORMED") {
					t.Fatalf("case %d: redactErrorMessage(%q) = %q leaked malformed payload", i, malformed, got)
				}
				if gotSan, lost := sanitizeRawText(malformed); !lost || strings.Contains(gotSan, "S3CR3T_MALFORMED") {
					t.Fatalf("case %d: sanitizeRawText(%q) = (%q, %v), leaked malformed payload", i, malformed, gotSan, lost)
				}
			}
			if !strings.ContainsAny(k, `/\`) && isSensitiveKey(k) {
				for _, malformed := range []string{
					`{"` + k + `": "[REDACTED]", "note": "S3CR3T_MALFORMED`,
					`{"` + k + `": [REDACTED] S3CR3T_MALFORMED}`,
				} {
					if refQuotedKeyRe.MatchString(malformed) && !hasQuotedSensitiveKey(malformed) {
						t.Fatalf("case %d: hasQuotedSensitiveKey(%q) = false while refQuotedKeyRe matched", i, malformed)
					}
					if got := redactErrorMessage(malformed); got != unparseableJSONPlaceholder {
						t.Fatalf("case %d: redactErrorMessage(%q) = %q, want %q", i, malformed, got, unparseableJSONPlaceholder)
					}
					if gotSan, lost := sanitizeRawText(malformed); !lost || gotSan != unparseableJSONPlaceholder {
						t.Fatalf("case %d: sanitizeRawText(%q) = (%q, %v), want (%q, true)", i, malformed, gotSan, lost, unparseableJSONPlaceholder)
					}
				}
			}
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

// assertRedactedIdempotent checks that redactErrorMessage, sanitizeRawText and
// SmartTruncate keep none of forbidden in their output for in, that
// sanitizeRawText reports the change, and that each of them leaves its own
// output unchanged on a second pass.
func assertRedactedIdempotent(t *testing.T, label, in string, forbidden ...string) {
	t.Helper()
	gotErr := redactErrorMessage(in)
	gotSan, lost := sanitizeRawText(in)
	truncRaw, _, err := SmartTruncate(in, -1)
	if err != nil {
		t.Fatalf("[%s] SmartTruncate(%q): %v", label, in, err)
	}
	var gotTrunc string
	if err := json.Unmarshal(truncRaw, &gotTrunc); err != nil {
		t.Fatalf("[%s] SmartTruncate(%q) = %s, not a JSON string: %v", label, in, truncRaw, err)
	}
	for _, f := range forbidden {
		if strings.Contains(gotErr, f) {
			t.Errorf("[%s] redactErrorMessage(%q) = %q kept %q", label, in, gotErr, f)
		}
		if strings.Contains(gotSan, f) {
			t.Errorf("[%s] sanitizeRawText(%q) = %q kept %q", label, in, gotSan, f)
		}
		if strings.Contains(gotTrunc, f) {
			t.Errorf("[%s] SmartTruncate(%q) = %q kept %q", label, in, gotTrunc, f)
		}
	}
	if len(forbidden) > 0 && !lost {
		t.Errorf("[%s] sanitizeRawText(%q) = (%q, false), want lost=true", label, in, gotSan)
	}
	if got2 := redactErrorMessage(gotErr); got2 != gotErr {
		t.Errorf("[%s] redactErrorMessage second pass(%q) = %q, want unchanged", label, gotErr, got2)
	}
	if got2, lost2 := sanitizeRawText(gotSan); got2 != gotSan || lost2 {
		t.Errorf("[%s] sanitizeRawText second pass(%q) = (%q, %v), want unchanged and lost=false", label, gotSan, got2, lost2)
	}
	if trunc2, _, err := SmartTruncate(gotTrunc, -1); err != nil || string(trunc2) != string(truncRaw) {
		t.Errorf("[%s] SmartTruncate second pass(%q) = (%s, %v), want %s", label, gotTrunc, trunc2, err, truncRaw)
	}
}

// TestCompoundKeyDifferentialAndIdempotencyRegressions covers inputs that an
// earlier revision of the scanner redacted less than the legacy regular
// expressions did, or redacted differently on a second pass: an unredacted
// assignment before a redacted over-budget tail, path- or backslash-joined
// keys after a closed quote, a quote word-joined to a key whose value starts
// on the next line, path-prefixed ".key" pairs after a container value,
// multiline Authorization headers, a key that ends one string with its value
// in the next, and quote-led text with an unbalanced quote.
func TestCompoundKeyDifferentialAndIdempotencyRegressions(t *testing.T) {
	runInParallel(t)
	const (
		diff   = "DIFF_SEC_123"
		extra  = "SECRET_EXTRA"
		adv    = "ADV_SECRET_9911"
		secret = "S3CR3T_DIFF_77"
	)
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		// Unredacted assignments before a redacted over-budget tail.
		{"overbudget_tail_earlier_assignment_padded_quoted_key", `password: [REDACTED]/` + a130 + `=` + diff + `/" password ":'` + diff + `'`, `password:[REDACTED]`, []string{diff}},
		{"overbudget_tail_earlier_assignment_redacted_pair", `password: [REDACTED]/` + a130 + `=` + extra + `/password:[REDACTED]`, `password:[REDACTED]`, []string{extra}},
		{"overbudget_tail_earlier_assignment_escaped_newline_header", "user:api_key=[a b]/" + a130 + "=" + diff + "/api_key: \"a\\\nb " + diff + "\"", `user:api_key=[REDACTED]`, []string{diff}},
		{"overbudget_tail_earlier_assignment_literal_backslashes_header", `user:api_key=[a b]/` + a130 + `=` + diff + `/api_key: "a\\\nb ` + diff + `"`, `user:api_key=[REDACTED]`, []string{diff}},
		{"overbudget_tail_compound_redacted_prefix", `dbPassword=[REDACTED]/` + a130 + `=` + adv + `/password:[REDACTED]`, `dbPassword=[REDACTED]`, []string{adv}},
		{"overbudget_tail_compound_container_prefix", `dbPassword={a: b}/` + a130 + `=` + adv + `/password:[REDACTED]`, `dbPassword=[REDACTED]`, []string{adv}},
		{"overbudget_tail_multiword_quoted_key", `dbPassword=[REDACTED]/` + a130 + `="` + adv + ` password": "[REDACTED]"`, `dbPassword=[REDACTED]`, []string{adv}},
		{"redacted_prefix_tail_earlier_assignment", `password: [REDACTED]abcX:` + secret + `|apiKey = "` + adv + `"`, ``, []string{secret, adv}},

		// Path- or backslash-joined keys after a closed quote, and a quote
		// word-joined to a key whose value starts on the next line.
		{"dangling_path_joined_key_compound_outer", `DB_PASSWORD: password='foo 'x/token": ` + adv, `DB_PASSWORD:[REDACTED]`, []string{adv}},
		{"dangling_path_joined_key_container_outer", `password : {a: b}/password='foo 'x/token": ` + diff, `password:[REDACTED]`, []string{diff}},
		{"failed_escaped_quote_before_container_value", `DB_PASSWORD: abcXa\"token:{"k": "` + extra + `"}`, `DB_PASSWORD:[REDACTED]`, []string{extra}},
		{"failed_escaped_quote_dangling_key", `DB_PASSWORD: password='foo 'a\"token': ` + adv, `DB_PASSWORD:[REDACTED]`, []string{adv}},
		{"plain_outer_failed_escaped_quote_dangling_key", `x password='foo 'a\"token': ` + adv, ``, []string{adv}},
		{"word_joined_spanning_quote_newline_value", "dbPassword:foo\"x/token:\n\" " + diff + "\"tail", `dbPassword:[REDACTED]`, []string{diff}},
		{"dangling_quote_nested_key_in_compound_value", `DB_PASSWORD: password": "` + secret + `"`, ``, []string{secret}},
		{"spanning_quote_nested_key_spaced_delimiter", `DB_PASSWORD: foo"x|/\"kv/prod/db.password\" = {"k": "` + secret + `"} x=1`, ``, []string{secret}},
		{"spanning_quote_nested_key_mid_value_delimiter", `DB_PASSWORD: client_secret':" ` + secret + `"tail 'x'=1`, ``, []string{secret}},

		// Path-prefixed ".key" pairs after a container value.
		{"container_tail_path_key_compound_outer", `DB_PASSWORD: [a b]/private.key: ` + extra, `DB_PASSWORD:[REDACTED]`, []string{extra}},
		{"container_tail_path_key_bracket", `password: [a b]/private.key: ` + extra, `password:[REDACTED]`, []string{extra}},
		{"container_tail_path_key_brace", `password: {a: b}/private.key: ` + extra, `password:[REDACTED]`, []string{extra}},
		{"container_tail_quoted_nested_key", `user:api_key={a: b}/\"dbPassword\"": '` + secret + `'`, ``, []string{secret}},

		// Multiline headers, a key that ends one string with its value in
		// the next, and quote-led text with an unbalanced quote.
		{"multiline_header_keeps_line_break", "api_key:\n  " + extra + " other=ok", "api_key:\n  [REDACTED] other=ok", []string{extra}},
		{"header_line_ending_in_nested_key", "api_key: x \"token\":\n  " + secret + " other=ok", ``, []string{secret}},
		{"header_escaped_newline_quoted_value", "x-api-key: \"a\\\nb " + secret + "\" next=1", ``, []string{secret}},
		{"key_ends_string_value_in_next_string", `msg: "password=" "` + adv + `"`, `msg: "password=" "[REDACTED]"`, []string{adv}},
		{"quote_led_unbalanced_key_text_path_key", `"api.key": failed: "` + a130 + `:` + diff + `/private.key:'` + diff + `'`, unparseableJSONPlaceholder, []string{diff}},
		{"quote_led_unbalanced_key_text_short", `"api.key": failed: "aaa:` + diff + `/private.key:'` + diff + `'`, unparseableJSONPlaceholder, []string{diff}},
		{"path_key_glued_quoted_value", `/etc/ssl/private.key:'` + secret + `'`, ``, []string{secret}},

		// Stray quotes after keys, quoted values that end in a key, and
		// quoted values whose closing quote opens a nested key or value.
		{"stray_quote_after_quoted_key", `"password" = [REDACTED],  "client_secret"':'` + secret + `'}`, ``, []string{secret}},
		{"stray_quote_after_padded_quoted_key", `DB_PASSWORD: 'a b' x/" password"": \"` + secret + `\"`, ``, []string{secret}},
		{"stray_quote_after_escaped_quoted_temp_key", "api_key:\n  \\\"temp:abc\\\"\": " + secret + " other=ok", ``, []string{secret}},
		{"stray_quote_after_quoted_state_key", `user:api_key= "app:temp:conn"": "` + secret, ``, []string{secret}},
		{"quoted_value_ends_in_key_word", "password : x\n  'kv/prod/db.password'\": \"abc; secret\": " + secret + " other=ok", ``, []string{secret}},
		{"quoted_value_opens_container_key", "failed: \"{\"k\":1}\n  \\\"kv/prod/db.password\\\"':\"abc:auth_token:{\"k\": \"" + secret + "\"} x=1", ``, []string{secret}},
		{"quoted_value_opens_next_line_value", "password : password='foo 'xx/app:temp:conn\": \"abc private.key:\n  \" " + secret + "\"tail/", ``, []string{secret}},
		{"stacked_temp_segments_stop_at_text_stop", "log: DB_PASSWORD: s0;user:temp:refresh_token: s1;spring.datasource.password: " + secret, ``, []string{secret}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if tc.want != "" {
			if got := redactErrorMessage(tc.in); got != tc.want {
				t.Errorf("[%s] redactErrorMessage(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
			}
			if got, _ := sanitizeRawText(tc.in); got != tc.want {
				t.Errorf("[%s] sanitizeRawText(%q) = %q, want %q", tc.label, tc.in, got, tc.want)
			}
		}
	}

	// Go *os.PathError text, already-redacted text and prose controls stay
	// byte-identical.
	for _, in := range []string{
		"open private.key: no such file or directory",
		"read /etc/ssl/private.key: permission denied",
		"open /etc/ssl/private.key:12: bad PEM",
		`password: [REDACTED]/` + a130 + `/token:[REDACTED]`,
		`dbPassword=[REDACTED]/` + a130 + `-password: "[REDACTED]"`,
		"api_key:\n  [REDACTED] other=ok",
		`msg: "password="[REDACTED]"`,
		`msg: "password=" "[REDACTED]"`,
		`"api.key":"[REDACTED]" other=ok`,
		`parsing "password='[REDACTED]'": invalid syntax`,
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		if truncRaw, truncated, err := SmartTruncate(in, -1); err != nil || truncated {
			t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want unchanged and truncated=false", in, truncRaw, truncated, err)
		}
	}
}

// TestLegacyRegexCompositeParity checks redactErrorMessage, sanitizeRawText
// and SmartTruncate, end to end, against the legacy regular expressions (see
// legacyReferenceScrub) over composite inputs built from the edge shapes
// that the generated inputs of TestLegacyRegexScannerDifferential do not
// combine: path- and backslash-joined keys after a closed quote, a quote
// word-joined to a key whose value starts on the next line, keys after a
// container value, an assignment before a redacted over-budget tail,
// multiline headers, a key that ends one string with its value in the next,
// and quote-led text with an unbalanced quote. No entry point may keep a
// secret that the legacy expressions remove, and each must leave its own
// output unchanged on a second pass.
func TestLegacyRegexCompositeParity(t *testing.T) {
	runInParallel(t)
	refScrub, _ := legacyReferenceScrub()
	a130 := strings.Repeat("a", 130)
	outers := []string{"", "password: ", "DB_PASSWORD: ", "dbPassword=", "user:api_key=", `msg: "`, "log: "}
	shapes := []func(k, sec string) string{
		func(k, sec string) string { return `password='foo 'x/` + k + `": ` + sec },
		func(k, sec string) string { return `abcXa\"` + k + `:{"k": "` + sec + `"}` },
		func(k, sec string) string { return `password='foo 'a\"` + k + `': ` + sec },
		func(k, sec string) string { return `foo"x/` + k + ":\n\" " + sec + `"tail` },
		func(k, sec string) string { return `[a b]/` + k + `: ` + sec },
		func(k, sec string) string { return `{a: b}/` + k + `: ` + sec },
		func(k, sec string) string { return `[REDACTED]/` + a130 + `=x/` + k + `: ` + sec },
		func(k, sec string) string { return k + ":\n  " + sec + " other=ok" },
		func(k, sec string) string { return `"` + k + `=" "` + sec + `"` },
		func(k, sec string) string { return `"` + k + `": failed: "aaa:x/` + k + `:'` + sec + `'` },
	}
	keys := []string{"password", "token", "api_key", "client_secret", "Authorization", "x-api-key", "private.key", "api.key", "DB_PASSWORD", "temp:abc"}
	n := 0
	for _, outer := range outers {
		for si, shape := range shapes {
			for ki, k := range keys {
				sec := fmt.Sprintf("S3CR3T_P%d_%d_END", si, ki)
				in := outer + shape(k, sec)
				var forbidden []string
				if !strings.Contains(refScrub(in), sec) {
					forbidden = append(forbidden, sec)
				}
				assertRedactedIdempotent(t, fmt.Sprintf("composite %d/%d/%d", len(outer), si, ki), in, forbidden...)
				n++
			}
		}
	}
	if n != len(outers)*len(shapes)*len(keys) {
		t.Fatalf("checked %d composite inputs, want %d", n, len(outers)*len(shapes)*len(keys))
	}
}

// TestAdversarialScaling checks that redactErrorMessage, sanitizeRawText and
// SmartTruncate stay linear on repeated adversarial shapes that stress each
// step added for the inputs of
// TestCompoundKeyDifferentialAndIdempotencyRegressions and
// TestCompoundKeyReviewLeakRegressions: the delimiters of a redacted
// over-budget tail, header lines left to the key-value pass, dangling
// path-joined keys, quote gaps between strings, stray quotes after keys,
// quoted values that open a container key, quote-led text with an unbalanced
// quote, escaped line breaks in header values, path-prefixed ".key" pairs
// after a container value, runs of compound key bytes, keys before unclosed
// or escaped quotes, placeholders with text glued after them, quote-led
// member lists, dangling-quote keys, quoted values that end in a key, header
// lines that end in a nested key, the legacy-pattern backstop over all of
// them, and the passes that rerun until the text no longer changes.
func TestAdversarialScaling(t *testing.T) {
	runInParallel(t)
	for _, shape := range []struct {
		name string
		mk   func(n int) string
	}{
		{"redacted over-budget tail delimiters", func(n int) string {
			return "dbPassword=[REDACTED]/" + strings.Repeat("a:[REDACTED]/", n) + "password:[REDACTED]"
		}},
		{"header line deferred to key-value pass", func(n int) string {
			return strings.Repeat("api_key:", n) + " \"token\":\n  v"
		}},
		{"multiline headers", func(n int) string { return strings.Repeat("api_key:\n  v other=ok\n", n) }},
		{"dangling path-joined keys", func(n int) string {
			return "DB_PASSWORD: " + strings.Repeat("password='foo 'x/", n) + "token\": v"
		}},
		{"quote gaps between strings", func(n int) string { return strings.Repeat(`msg: "password=" "v" `, n) }},
		{"stray quotes after keys", func(n int) string { return strings.Repeat(`"token"": v `, n) }},
		{"quoted values opening container keys", func(n int) string {
			return strings.Repeat(`password: "a:token:{"k": 1} `, n)
		}},
		{"unbalanced quote-led key text", func(n int) string { return `"api.key": failed: ` + strings.Repeat(`"x `, n) }},
		{"escaped line breaks in header values", func(n int) string { return strings.Repeat("x-api-key: \"a\\\nb\" ", n) }},
		{"container tails with path keys", func(n int) string {
			return strings.Repeat("password: [a b]/private.key: v ", n)
		}},
		{"compound key runs", func(n int) string { return "password: " + strings.Repeat("a.", n) + "token: v" }},
		{"keys before unclosed quotes", func(n int) string { return strings.Repeat(`token="a `, n) }},
		{"escaped keys before unclosed quotes", func(n int) string { return strings.Repeat(`\"token\":\"a `, n) + `\n` }},
		{"glued placeholders", func(n int) string { return strings.Repeat("password:[REDACTED]x", n) }},
		{"quote-led member lists", func(n int) string { return strings.Repeat(`"token": "a", `, n) }},
		{"dangling quote keys", func(n int) string { return "DB_PASSWORD: " + strings.Repeat(`token"=`, n) + "v" }},
		{"quoted values ending in keys", func(n int) string { return strings.Repeat(`DB_PASSWORD: "abc; /x/token": v `, n) }},
		{"header lines ending in nested keys", func(n int) string { return strings.Repeat("api_key: a b x-api-key:\n  v\n", n) }},
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

// TestCompoundKeyReviewLeakRegressions covers inputs from review of an
// earlier revision of the scanners that kept a secret the legacy regular
// expressions redacted, or one an earlier revision redacted: header lines
// that end in a nested key, dangling-quote keys before containers and after
// over-budget tails, quoted values that end in a key after a non-word joiner
// or that open a nested value after spaces, stray quotes before padded quoted
// keys, malformed quote-led JSON text with an even number of quotes, and runs
// of mixed escaped and unescaped quotes that the legacy expressions failed
// closed.
func TestCompoundKeyReviewLeakRegressions(t *testing.T) {
	runInParallel(t)
	const (
		s0 = "LEAK_SECRET_ZERO"
		s1 = "LEAK_SECRET_ONE"
		s2 = "LEAK_SECRET_TWO"
		s3 = "LEAK_SECRET_THREE"
	)
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct {
		label, in, want string
		forbidden       []string
	}{
		// A header line that ends in a nested key keeps its other words
		// redacted.
		{"header_line_nested_key_multiword_value", "api_key: " + s0 + " " + s1 + " x-api-key:\n  " + s2, "api_key: [REDACTED] x-api-key:\n  [REDACTED]", []string{s0, s1, s2}},
		{"header_line_nested_key_quoted_value", "api_key: \"" + s0 + "\" " + s1 + " x-api-key:\n  " + s2, "api_key: [REDACTED] x-api-key:\n  [REDACTED]", []string{s0, s1, s2}},

		// Dangling-quote keys after over-budget tails and before containers.
		{"overbudget_dangling_quote_key_container", "dbPassword=" + a130 + `=S0/password": {"k": "` + s1 + `"} other=ok`, "dbPassword=[REDACTED] other=ok", []string{s1}},
		{"overbudget_dangling_quote_key_glued_value", "DB_PASSWORD: " + a130 + `=S0/api_key": "` + s1 + `"tail"`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"overbudget_dangling_single_quote_key", "DB_PASSWORD: " + a130 + "=S0/api_key': '" + s1 + "'tail/password:[REDACTED]", "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"dangling_quote_key_container", `DB_PASSWORD: token":{k: "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"dangling_quote_path_key_container", `DB_PASSWORD: config/github.token":{k: "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},
		{"dangling_quote_key_equals_container", `DB_PASSWORD: password"={"k": "` + s1 + `"}`, "DB_PASSWORD:[REDACTED]", []string{s1}},

		// Quoted values that end in a key after a non-word joiner, a
		// hyphenated compound, a state prefix, or past the nested-key budget.
		{"quoted_value_slash_joined_key", `DB_PASSWORD: "abc; /x/token": ` + s0, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"quoted_value_temp_key", `DB_PASSWORD: "abc; /x/temp:a.b": ` + s0, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"quoted_value_path_key", `DB_PASSWORD: "abc; db/password": ` + s0, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"quoted_value_space_slash_key", `DB_PASSWORD: "a b/token": "` + s0 + `"`, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"quoted_value_equals_joined_key", `dbPassword="invalid key=password": "` + s0 + `"`, `dbPassword="[REDACTED]"`, []string{s0}},
		{"quoted_value_hyphen_compound_key", `my_api_key: "err next-page-token": "` + s0 + `"`, `my_api_key:"[REDACTED]"`, []string{s0}},
		{"quoted_value_overbudget_comma_key", `DB_PASSWORD: "` + a130 + `, secret": "` + s0 + `"`, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"quoted_value_overbudget_multiline_key", "DB_PASSWORD: \"" + a130 + "\ntoken\": \"" + s0 + "\"", `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"multiline_header_quoted_path_key", "api_key:\n  \"msg, x/token\": \"" + s0 + "\"", "api_key:\n  \"[REDACTED]\"", []string{s0}},
		{"multiline_header_quoted_state_key", "Authorization:\n  \"ctx: app:password\": \"" + s0 + "\"", "Authorization:\n  \"[REDACTED]\"", []string{s0}},
		{"multiline_header_quoted_temp_key", "x-api-key:\n  \"ctx: temp:session\": \"" + s0 + "\"", "x-api-key:\n  \"[REDACTED]\"", []string{s0}},

		// Quoted values whose closing quote opens a nested value after
		// spaces, including in the quoted JSON key-value pass.
		{"quoted_value_spaced_nested_value", `DB_PASSWORD: "a token: "x ` + s0 + `"`, `DB_PASSWORD:"[REDACTED]"`, []string{s0}},
		{"multiline_header_spaced_nested_value", "api_key:\n  \"a token: \"x " + s0 + "\"", "api_key:\n  \"[REDACTED]\"", []string{s0}},
		{"quote_led_spaced_nested_multiline_value", "\"S0:password= \"a\nb " + s0 + "\"\"", `"S0:password="[REDACTED]""`, []string{s0}},
		{"json_pair_quoted_value_opens_container_key", `log: "dbPassword": "a:token:{"k": "` + s0 + `"}`, `log: "dbPassword":"[REDACTED]"`, []string{s0}},

		// Stray quotes before padded quoted keys, and a nested pair whose
		// quoted value would take the pairs after it.
		{"stray_quote_padded_key_tab_next_string", "parsing \"\"api_key\"'=\"S0, ' api.key ':\t\" \"" + s1 + "\"\"", `parsing ""api_key"="[REDACTED]"`, []string{s1}},
		{"nested_pair_quote_takes_following_pairs", `DB_PASSWORD: "DB_PASSWORD"":Bearer S0, password='foo 'x//x/temp:a.b" = "` + s1 + `"`, `DB_PASSWORD:"[REDACTED]"`, []string{s1}},

		// Malformed quote-led JSON text with an even number of quotes.
		{"quote_led_member_list_glued_string", `"token": "[REDACTED]", "note": "` + s3 + ` "x`, unparseableJSONPlaceholder, []string{s3}},
		{"quote_led_adjacent_strings", `"token": "[REDACTED]" "` + s3 + `"`, unparseableJSONPlaceholder, []string{s3}},
		{"quote_led_member_list_trailing_comma", `"token": "[REDACTED]", "note": "` + s3 + `",`, unparseableJSONPlaceholder, []string{s3}},
		{"quote_led_adjacent_strings_before_redaction", `"token": "x" "` + s3 + `"`, unparseableJSONPlaceholder, []string{s3}},

		// Mixed escaped and unescaped quotes, and escapes that hide a
		// delimiter or a quoted key, which the legacy expressions failed
		// closed.
		{"mixed_quotes_escaped_key_nested_multiline_value", "dbPassword=\\\"api_key\\\"\": \"S0:password=\"a\nb " + s1 + "\"\"", "", []string{s1}},
		{"mixed_quotes_multiline_header_escaped_value", "api_key:\n  \"S0, x/token\": \\\"" + s1 + "\\\"tail", "", []string{s1}},
		{"escaped_tab_hides_delimiter", `x "client_secret"': user:temp:refresh_token\t:'` + s0 + `/private.key = '` + s1 + `'-"secret"' = ` + s2 + " " + s3, redactedSensitiveTextPlaceholder, []string{s0, s1, s2, s3}},
		{"escaped_quotes_hide_quoted_key", "'app:password:\t[a " + s0 + `]/\"monkey\" = monkey:  '` + s1 + ": invalid syntax", redactedSensitiveTextPlaceholder, []string{s0, s1}},
		{"escaped_newline_splits_compound_key", "user:temp:session = Bearer S0+apiKey=\n'secret':  " + s1 + " " + s2 + `\naccess_token="` + s3 + `\nS4`, redactedSensitiveTextPlaceholder, []string{s1, s2, s3}},
	} {
		assertRedactedIdempotent(t, tc.label, tc.in, tc.forbidden...)
		if tc.want == "" {
			continue
		}
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

	// Text that an earlier pass redacted, Go path errors and prose stay
	// byte-identical.
	for _, in := range []string{
		"api_key: [REDACTED] x-api-key:\n  [REDACTED]",
		`"client_secret=" "[REDACTED]"`,
		`failed: \"password='[REDACTED]'\"\nretry`,
		"credentials:\n  C:\\ssl\\password:[REDACTED]\n  region: us",
		`log: "dbPassword":"[REDACTED]"`,
		`parsing ""api_key"="[REDACTED]"`,
		"stat signing.key: permission denied",
		`"Hello," she said "bye"`,
		"token_count=12 max_tokens=4096 page_token=abc",
		`\"password\":\"[REDACTED]\"`,
	} {
		if got := redactErrorMessage(in); got != in {
			t.Errorf("redactErrorMessage(%q) = %q, want unchanged", in, got)
		}
		if got, lost := sanitizeRawText(in); got != in || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want unchanged and lost=false", in, got, lost)
		}
		if raw, truncated, err := SmartTruncate(in, -1); err != nil || truncated {
			t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want unchanged and truncated=false", in, raw, truncated, err)
		}
	}
}

// TestLegacyPatternBackstop checks legacyScrubTextPatterns, the linear-time
// redo of the legacy regular expressions, against the reference expressions,
// and that leavesLegacyValue flags scanner output that keeps a word they
// redact, or text they fail closed, but not a key the scanners keep while
// redacting its value or the tail of an already-redacted value.
func TestLegacyPatternBackstop(t *testing.T) {
	runInParallel(t)
	refScrub, _ := legacyReferenceScrub()
	keptWords := func(raw, redacted string) []string {
		words := legacyWords(raw, true)
		var kept []string
		for j, at := range legacyKeptWords(raw, words, redacted) {
			if at >= 0 {
				kept = append(kept, raw[words[j].start:words[j].end])
			}
		}
		return kept
	}
	for _, in := range []string{
		"password=abc other=ok",
		"api_key: v w\nnext: 1",
		"failed: Bearer abc123, retry",
		"Authorization: Basic dXNlcjpwYXNz",
		`log: "token": abc, x=1`,
		"temp:a.b=v x",
		"secret: abc; token=def&x=1",
		"x-api-key:\t k1 k2",
		"DB_PASSWORD=v1 password=v2",
		"open private.key: no such file or directory",
		`password: "a b" c`,
	} {
		if got, want := keptWords(in, legacyScrubTextPatterns(in, false)), keptWords(in, refScrub(in)); !slices.Equal(got, want) {
			t.Errorf("legacyScrubTextPatterns(%q) keeps %q, reference expressions keep %q", in, got, want)
		}
	}
	a130 := strings.Repeat("a", 130)
	for _, tc := range []struct {
		raw, scrubbed string
		want          bool
	}{
		{"password=abc def", "password=[REDACTED] def", false},
		{"password=abc", "password=abc", true},
		{`DB_PASSWORD: "abc; /x/token": v1`, `DB_PASSWORD:"[REDACTED]": v1`, true},
		{"api_key: v1 v2 x-api-key:\n  v3", "api_key:[REDACTED] v2 x-api-key:\n  [REDACTED]", true},
		{`x "client_secret"': token\t:'v1' = v2`, `x "client_secret"':[REDACTED] = v2`, true},
		{"credentials:\n  temp:a:b: v1", "credentials:\n  temp:a:[REDACTED]", false},
		{"password: [REDACTED]/" + a130 + "/token:[REDACTED]", "password: [REDACTED]/" + a130 + "/token:[REDACTED]", false},
		{"open private.key: no such file", "open private.key: no such file", false},
	} {
		if got := leavesLegacyValue(tc.raw, tc.scrubbed); got != tc.want {
			t.Errorf("leavesLegacyValue(%q, %q) = %v, want %v", tc.raw, tc.scrubbed, got, tc.want)
		}
	}
}
