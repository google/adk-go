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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// baseParityCase is a case of testdata/redaction_base_corpus.json: an input,
// the secrets planted in it, and what scrubErrorText, redactErrorMessage,
// sanitizeRawText, SmartTruncate and FormatContentParts returned for it at
// the merge base of the change that replaced the regular expressions with
// the scanners, in that order.
type baseParityCase struct {
	In      string    `json:"in"`
	Secrets []string  `json:"secrets"`
	Base    [5]string `json:"base"`
}

// TestRedactionBaseParity runs the cases of testdata/redaction_base_corpus.json
// through every entry point (see largeTextEntryPoints) and checks that no
// secret of a case occurs more often in the output than in the output that
// the merge base recorded for that entry point: every planted secret that the
// base removed, whether by redacting it or by failing the text closed, is
// removed. The cases cover what the review of the scanners found kept next to
// placeholders already in the text, punctuation-only values and values
// spelled like keys, runs of glued pairs, JSON documents whose leaves hold
// credentials, compound, state-prefixed and dotted keys, the names of
// tokenizer tokens and cursors, URI keys that hold a credential before a
// sensitive key name, URIs whose rewrite escapes bytes of a path before a
// newly sensitive segment, runs of rewritten URI pairs, JSON values whose
// leaves a later pass read as malformed, JSON documents whose text the
// expressions read otherwise than their strings decoded, as a header that an
// escaped line break ends, a quote that a later string closes, or a document
// that the redaction writes again with its keys sorted, the reproducers of
// earlier reviews, shapes left for later work, prose, and seeded random
// combinations. Each output is also checked to be unchanged by a second pass.
func TestRedactionBaseParity(t *testing.T) {
	runInParallel(t)
	b, err := os.ReadFile(filepath.Join("testdata", "redaction_base_corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		BaseCommit string           `json:"base_commit"`
		Cases      []baseParityCase `json:"cases"`
	}
	if err := json.Unmarshal(b, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) < 700 || corpus.BaseCommit == "" {
		t.Fatalf("corpus holds %d cases (base %q), want at least 700 and a base commit", len(corpus.Cases), corpus.BaseCommit)
	}
	redactedByBase := 0
	for _, c := range corpus.Cases {
		for k, ep := range largeTextEntryPoints(t) {
			got := ep.f(c.In)
			for _, s := range c.Secrets {
				if strings.Count(got, s) > strings.Count(c.Base[k], s) {
					t.Errorf("%s(%q) = %q keeps %q, which the base removed: %q", ep.name, c.In, got, s, c.Base[k])
				}
			}
			if again := ep.f(got); again != got {
				t.Errorf("%s second pass(%q) = %q, want unchanged", ep.name, got, again)
			}
		}
		if san, _ := sanitizeRawText(c.In); san != "" {
			if _, againLost := sanitizeRawText(san); againLost {
				t.Errorf("sanitizeRawText second pass(%q) reported contentLost", san)
			}
		}
		for _, s := range c.Secrets {
			if !strings.Contains(c.Base[1], s) {
				redactedByBase++
				break
			}
		}
	}
	// The corpus is only a check if the base removes secrets in it.
	if redactedByBase < 400 {
		t.Errorf("the base removes a secret in %d cases, want at least 400", redactedByBase)
	}
	// Reproducers of the reviews of this change.
	for _, in := range []string{
		"password: [REDACTED]hunter2ZQ/token: [REDACTED]",
		"DB_PASSWORD:x\v'password :[REDACTED]RVIEWz9MARK",
		"credential:\\'temp:\\\": '= [REDACTED]RVIEWz9MARK",
		"api.key:\"secret:'\" ~!@$^*~'",
		"https://example.com/cb?Bearer+SECRET_ONE+token=SECRET_TWO",
		"https://h/^/secret_key/password:#secret_hunter2&5",
		`{"nested":"{\"url\":\"https://example.com/path?a=1&access_token=SECRET\"}"}`,
		`{"a":"Authorization:\n","b":"SECRET"}`,
		`"{\"a\":\"Authorization: [REDACTED]\\n\",\"b\":\"SECRET\"}"`,
		`{"Authorization: [REDACTED]\n":"SECRET"}`,
		`["access_token='","SECRET","'"]`,
		`{"x":{"x":"proxy-authorization:  [REDACTED]"},"data":"SECRET","zz_password":"hunter2"}`,
		`{"nested":"[\"Authorization%3A token:[REDACTED]\\t\" , \"QZX7SEC\"]"}`,
		`{"temp%3aa":"QZX7SEC"}`,
		`{"m": "access_token=%22QZX7SEC;key=QZX7SEC"}`,
		"password:\n  C:\\SECRET/token:\"a b\"",
		"password:\n  /SECRET/secret_key: [REDACTED]",
		`"{\"m\":\"zqxSentinel7 Authorization:\\n;key=SECRETZQ\"}"`,
		`"{\"m\":\"Bearer \\\"SECRETZQ;key=b;token=c\"}"`,
		`{"note":"password:\n\"[REDACTED]\".hunter2ZQ"}`,
		`{"log": "x-api-key: SECRETZQ\nrequest id: 42"}`,
		`{"msg": "to\u212aen=SECRETZQ"`,
		`{"msg": "pa\u017f\u017fword=SECRETZQ"`,
		`[INFO] "pa\u017f\u017fword=SECRETZQ"`,
		`{"msg": "ap\u0130_key=SECRETZQ"`,
		"{\"log\": \"x-api-key: [REDACTED]\n\"pa\\u017f\\u017fword=SECRETZQ\"",
		`[REDACTED_SENSITIVE_URI] "pa\u017f\u017fword=SECRETZQ"`,
		`{"k": "{\"zz_a\":\"Authorization: x\",\"aa_b\":\"SECRETZQ\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`,
		`["{\"zz_a\":\"Authorization: x\",\"aa_b\":\"SECRETZQ\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"]`,
		`{"k": "{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}`,
		`["{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"]`,
		`{"a": "{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}", "b": "plain"}`,
		`{"k":"{\"inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\"}\",\"other\":\"SECRETZQ\"}"}`,
		`["{\"inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\"}\",\"other\":\"SECRETZQ\"}"]`,
		`"{\"k\":\"{\\\"inner\\\":\\\"{\\\\\\\"zz_password\\\\\\\":\\\\\\\"Authorization: x\\\\\\\"}\\\",\\\"other\\\":\\\"SECRETZQ\\\"}\"}"`,
		`{"outer":"{\"k\":\"{\\\"inner\\\":\\\"{\\\\\\\"zz_password\\\\\\\":\\\\\\\"Authorization: x\\\\\\\"}\\\",\\\"other\\\":\\\"SECRETZQ\\\"}\"}"}`,
		`{"k":"[\"{\\\"zz_password\\\":\\\"Bearer '\\\"}\",\"SECRETZQ\",\"'\"]"}`,
		`{"k":"{\"other\":\"SECRETZQ\",\"a_inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\",\\\"password\\\":\\\"p\\\"}\"}"}`,
		`{"a":"{\"other\":\"SECRETZQ\",\"a_inner\":\"{\\\"zz_password\\\":\\\"Authorization: x\\\",\\\"password\\\":\\\"p\\\"}\"}","b":"plain"}`,
		`[INFO] "unclosed: "'token'\u00a0:\u00a0SECRETZQ`,
		`{"msg": "unclosed: 'token'\u00a0:\u00a0SECRETZQ`,
		"{\"\t\\u00a0token\": \"SECRETZQ\"}",
		`{'\u00a0token': 'SECRETZQ'}`,
		`{'token'\u00a0:\u00a0'SECRETZQ'}`,
		`[INFO] 'token'\u00a0:\u00a0SECRETZQ`,
		`[REDACTED_SENSITIVE_URI] 'token'\u00a0:\u00a0SECRETZQ`,
		"{\"msg\": \"line1\n'token'\\u00a0:\\u00a0SECRETZQ\"}",
		`"\u007b\u0022DB_PASSWORD\u0022:\u0022Authorization: x\u0022\u007d" SECRETZQ`,
		`"{\" DB_PASSWORD \":\" Authorization: x\"}" SECRETZQ`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ'`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ' " }"`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ%27`,
		`"{\" DB_PASSWORD \":\" Bearer '\"}" SECRETZQ\u0027`,
		`"{\" password \":\" SECRETZQ \"}" and 'hi'`,
		`"\u007b\u0022password\u0022:\u0022SECRETZQ\u0022\u007d" ok`,
		`"[\u0022Bearer \'\u0022,\u0022SECRETZQ\u0022]" ok`,
		`"\"{\\\"DB_PASSWORD\\\":\\\"Bearer%20%22x\\\\\\\" tail\\\",\\\"b\\\":\\\"SECRETZQ\\\"}\""`,
		`"\"\\\"{\\\\\\\"DB_PASSWORD\\\\\\\":\\\\\\\"Bearer%20%22x\\\\\\\\\\\\\\\" tail\\\\\\\",\\\\\\\"b\\\\\\\":\\\\\\\"SECRETZQ\\\\\\\"}\\\"\""`,
		`{"k":"{\"k\":\"\\\"{\\\\\\\"DB_PASSWORD\\\\\\\":\\\\\\\"Bearer%20%22x\\\\\\\\\\\\\\\" tail\\\\\\\",\\\\\\\"b\\\\\\\":\\\\\\\"SECRETZQ\\\\\\\"}\\\"\"}"}`,
		`{"k":"\"\\\"{\\\\\\\"DB_PASSWORD\\\\\\\":\\\\\\\"Bearer%20%22x\\\\\\\\\\\\\\\" tail\\\\\\\",\\\\\\\"b\\\\\\\":\\\\\\\"SECRETZQ\\\\\\\"}\\\"\""}`,
		`"\"{\\\"DB_PASSWORD\\\":\\\"Bearer%20%22x\\\\\\\" tail\\\",\\\"b\\\":\\\"SECRETZQ\\\"}\"" ok`,
		`"\"{\\\"DB_PASSWORD\\\":\\\"Bearer%20%22x\\\\\\\" tail\\\",\\\"b\\\":\\\"SECRETZQ\\\",\\\"c\\\":\\\"%22\\\"}\"" ok`,
	} {
		found := false
		for _, c := range corpus.Cases {
			found = found || c.In == in
		}
		if !found {
			t.Errorf("corpus lacks %q", in)
		}
	}
}
