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
// sanitizeRawText, smartTruncate and formatContentParts returned for it at
// the merge base of the change that replaced the regular expressions with
// the scanners, in that order.
type baseParityCase struct {
	In      string    `json:"in"`
	Secrets []string  `json:"secrets"`
	Base    [5]string `json:"base"`
}

// changedByBase reports whether the base output of some entry point for c is
// neither c.In, up to whitespace, nor a placeholder for the whole text: the
// base removed part of c.In, which a secret of c must then mark for the case
// to check anything.
func changedByBase(c baseParityCase) bool {
	in := strings.Join(strings.Fields(c.In), "")
	for _, b := range c.Base {
		if !isKnownSentinel(b) && strings.Join(strings.Fields(b), "") != in {
			return true
		}
	}
	return false
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
// earlier reviews, malformed JSON text that smartTruncate fails closed, with
// or without a placeholder in it, sensitive keys that hold an object or an
// array, shapes left for later work, prose, and seeded random
// combinations. Each output is also run through a second call, which may
// change it only by failing closed what main fails closed in it (see
// assertSecondCall).
// A case whose text the base changes on some entry point, other than by
// whitespace or a placeholder for the whole text, must list a secret that
// marks what the base removed, such as the punctuation-only value of
// {"token": "!@#$%^&*()"}, so that every such case is checked.
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
	for _, c := range corpus.Cases {
		if len(c.Secrets) == 0 && changedByBase(c) {
			t.Fatalf("corpus case %q lists no secret, but the base changes it: %q", c.In, c.Base)
		}
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
			assertSecondCall(t, ep.name, ep.f, got)
		}
		if san, _ := sanitizeRawText(c.In); san != "" {
			if again, againLost := sanitizeRawText(san); againLost != (again != san) {
				t.Errorf("sanitizeRawText second call(%q) = (%q, %v), want a loss reported exactly when the text changes", san, again, againLost)
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
		`{"s":"https\\ufeff://h/a;Bearer+SECRETZQ+token=x"}`,
		`https://h/secret_key/password/SECRETZQ`,
		`https://h/signing_key/token/SECRETZQ`,
		`https://h/user:temp:a/password/SECRETZQ`,
		`https://h/secret_key/password/token/SECRETZQ`,
		`[INFO] Bearer 'SECRETZQ`,
		`[INFO] Bearer "SECRETZQ`,
		`"Bearer 'SECRETZQ`,
		`{"log":"Bearer 'SECRETZQ`,
		"[INFO] Bearer\u200bSECRETZQ",
		"[INFO] Bearer\u200b'SECRETZQ",
		`[INFO] Bearer%E2%80%8BSECRETZQ`,
		`[INFO] Bearer\u200bSECRETZQ`,
		`[INFO] Bearer%ZZSECRETZQ`,
		`[INFO] Bearer%2g'SECRETZQ`,
		`[INFO] Bearer%c0%a0SECRETZQ`,
		`[INFO] Bearer%e0%80%a0SECRETZQ`,
		`{"password"%c0%a0:%c0%a0"SECRETZQ"}`,
		`{"password"%ZZ:%ZZ"SECRETZQ"}`,
		"[INFO] password\u200b:\u200bSECRETZQ",
		`[INFO] password%c0%a0:%c0%a0SECRETZQ`,
		`[INFO] password%ZZ=%ZZSECRETZQ`,
		"[INFO] hello\u200bworld",
		`{"msg":"100% ok %80 %C2 %C2%ZZ %E0%80%80 %`,
		"{\"password\"%ZZ\n:\"SECRETZQ\"}",
		"{\"password\"%c0%a0\n:\"SECRETZQ\"}",
		"[INFO] password%ZZ\n=SECRETZQ",
		`{"log":"password%ZZ\t=SECRETZQ`,
		`{"log":"password%ZZ\\t=SECRETZQ`,
		`{"log":"password%c0%a0\f=SECRETZQ`,
		`{"temp:a:b%"%ZZ:"SECRETZQ"}`,
		`{"temp:a: b%"%ZZ:"SECRETZQ"}`,
		`{'temp:a:b%'%ZZ:'SECRETZQ'}`,
		`{"temp:a:b%2"%ZZ:"SECRETZQ"}`,
		`{"password"%2:"SECRETZQ"}`,
		`{"password"%:"SECRETZQ"}`,
		`{"log":"C:\\Bearer\t'SECRETZQ`,
		`{"log":"C:\\Basic\t'SECRETZQ`,
		`{"log":"Bearer\\t'SECRETZQ`,
		`{"log":"Bearer\\t%ZZSECRETZQ`,
		`{"log":"Bearer\\\t'SECRETZQ`,
		`{"log":"C:\\\\Bearer 'SECRETZQ`,
		`[INFO] C:\Bearer 'SECRETZQ`,
		`https://h/%5Cu00a0api_key/password/SECRETZQ`,
		`https://h/api_key%5Cu00a0/password/SECRETZQ`,
		`https://h/api_%5Cu212aey/password/SECRETZQ`,
		`https://h/api_key%5Cu3000/password/SECRETZQ`,
		`https://h/%5Cufeffapi_key/password/SECRETZQ`,
		`{"log":"C:\\Users\\alice`,
		`{"log":"C:\\Bearer of bad news`,
		"{\"msg\":\"100% ok:\nnext line",
		`{"log":"[INFO] Bearer\\u200bSECRETZQ`,
		`{"log":"Bearer\\u0020'SECRETZQ`,
		`{"log":"Bea\\u0072er 'SECRETZQ`,
		`{"log":"[INFO] Bearer\\u00a0SECRETZQ`,
		`{"log":"[INFO] password\\u00a0=SECRETZQ`,
		`{"log":"[INFO] password\\u200b=SECRETZQ`,
		`{"log":"[INFO] Bearer\\x20'SECRETZQ`,
		`{"log":"[INFO] Bearer\\u0007SECRETZQ`,
		`{"log":"[INFO] Bearer\\x07SECRETZQ`,
		`{"log":"[INFO] Bearer\\\\u200bSECRETZQ`,
		`{"log":"[INFO] Bearer\\u002520'SECRETZQ`,
		`{"log":"[INFO] Bearer\\u002507SECRETZQ`,
		"{\"\npassword\"%ZZ:\"SECRETZQ\"}",
		"{\"password\n\"%c0%a0:\"SECRETZQ\"}",
		`{"\npassword"%ZZ:"SECRETZQ"}`,
		"{\"\rpassword\"%ZZ:\"SECRETZQ\"}",
		"[INFO] Basic 'dXNlcjpwYXNz'",
		`[INFO] Basic "dXNlcjpwYXNz"`,
		"[INFO] Basic `dXNlcjpwYXNz`",
		`{"auth":"Basic 'dXNlcjpwYXNz'"`,
		`[INFO] Basic \'dXNlcjpwYXNz\'`,
		`{"log":"Basic\\u0020'dXNlcjpwYXNz'"`,
		`{"log":"Basic\\u0020` + "`dXNlcjpwYXNz`" + `"`,
		"[INFO] Basic '[REDACTED]'",
		`[INFO] Basic ""`,
		"[INFO] Basic `[REDACTED]`",
		`{"log":"Bearer\bSECRETZQ`,
		`{"log":"Bearer\u0007SECRETZQ`,
		`[INFO] Bearer\x07SECRETZQ`,
		"[INFO] Bearer%07SECRETZQ",
		`[INFO] Bearer\x00SECRETZQ`,
		"[INFO] Bearer%00SECRETZQ",
		"[INFO] Bearer%1fSECRETZQ",
		"[INFO] Bearer%7fSECRETZQ",
		"[INFO] Bearer%0bSECRETZQ",
		`{"log":"Bearer\b of SECRETZQ`,
		`{"log":"Bearer\v of SECRETZQ`,
		`{"log":"Basic\bdXNlcjpwYXNz`,
		"[INFO] Basic%07dXNlcjpwYXNz",
		`{"password"%07:"SECRETZQ"}`,
		"[INFO] password%07=SECRETZQ",
		`{"password"%ZZ\b:"SECRETZQ"}`,
		"[INFO] Bearer `my SECRETZQ`",
		`{"auth":"Bearer ` + "`SECRETZQ`" + `"`,
		`{"log":"\u2014Bearer SECRETZQ`,
		`{"log":"\u3010Bearer SECRETZQ\u3011`,
		`{"log":"\u2014Bearer 'SECRETZQ`,
		`{"log":"\u2014Basic dXNlcjpwYXNz`,
		`{"log":"\u2014password=SECRETZQ`,
		`{"log":"\u2014Authorization: SECRETZQ`,
		"{\"log\":\"\\u2014Bearer SECRETZQ\nnext line\"}",
		`"\u2014Bearer SECRETZQ`,
		`{"log":"\\u2014Bearer SECRETZQ`,
		`{"log":"\ud83d\ude00Bearer SECRETZQ`,
		`{"log":"%E2%80%94Bearer SECRETZQ`,
		`{"log":"\xe2\x80\x94Bearer SECRETZQ`,
		`{"log":"\x94Bearer SECRETZQ`,
		`{"log":"\u00e9password=SECRETZQ`,
		`{"log":"\u2014temp:a=SECRETZQ`,
		`{"log":"\u005cu2014Bearer SECRETZQ`,
		`{"log":"\ud83d\u0020Bearer SECRETZQ`,
		`{"log":"\U2014Bearer SECRETZQ`,
		"{\"log\":\"\\\\u2014Bearer\vSECRETZQ",
		`{"log":"Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZDEyMw`,
		`[INFO] Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZDEyMw`,
		`{"log":"Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZDEyM`,
		`{"log":"Basic YWRtaW46c3VwZXJzZWNyZXRwYXNzd29yZDEyMw=`,
		`{"log":"Basic dXNlcjpwYXN`,
		`{"log":"Basic dXNlcjpwYX`,
		`{"log":"Basic dXNlcjp`,
		`{"log":"Basic YWRtaW46cGFzcw===`,
		`{"log":"Basic dXNlcjphw6`,
		`{"temp: a"\b:"SECRETZQ"}`,
		"{\"temp: a\"\b:\"SECRETZQ\"}",
		"{\"temp:\"\x07:\"SECRETZQ\"}",
		`{"temp:"\u0007:"SECRETZQ"}`,
		"{\"temp:a,b\"\x07:\"SECRETZQ\"}",
		`{"temp:a,b"\u0007:"SECRETZQ"}`,
		`{"temp: a"\u200b:"SECRETZQ"}`,
		"{\"temp: a\"\u200b:\"SECRETZQ\"}",
		"{\"temp: a\"\x00=\"SECRETZQ\"}",
		`{"temp: a"\u0000="SECRETZQ"}`,
		`"hello \u007b'password':'[REDACTED]'}`,
		`"hello %257b'password':'[REDACTED]'}`,
		`{"log":"token\xe2\x80\xa8=SECRETZQ`,
		`{"log":"token\\xe2\\x80\\xa8=SECRETZQ`,
		`"'temp: a'` + "\x07" + `:'[REDACTED]' "`,
		`{"msg":"\u2014 hello world"}`,
		`{"msg":"\u3010status: ok\u3011"}`,
		`[INFO] Basic usage`,
		`[INFO] Basic Oracle setup`,
		`{"log":"Basic ''dXNlcjpwYXNz`,
		`[INFO] Basic ''dXNlcjpwYXNz`,
		"[INFO] Basic ``dXNlcjpwYXNz",
		`[INFO] Basic ""+dXNlcjpwYXNz`,
		`[INFO] Basic '[REDACTED]'dXNlcjpwYXNz`,
		"[INFO] Basic `[REDACTED]`dXNlcjpwYXNz",
		`[INFO] Basic "[REDACTED]"+dXNlcjpwYXNz`,
		`[INFO] Basic \'\'dXNlcjpwYXNz`,
		`[INFO] Basic \'[REDACTED]\'dXNlcjpwYXNz`,
		`{"log":"Basic \"\"dXNlcjpwYXNz`,
		`{"log":"Basic \"[REDACTED]\"dXNlcjpwYXNz`,
		`{"log":"Basic \u005c"\u005c"dXNlcjpwYXNz`,
		`{"log":"Basic \u005c"[REDACTED]\u005c"dXNlcjpwYXNz`,
		`[INFO] Basic '[REDACTED]'.dXNlcjpwYXNz`,
		`{"log":"Bearer '[REDACTED]'SECRETZQ`,
		`[INFO] Bearer '[REDACTED]'SECRETZQ`,
		`[INFO] Bearer "[REDACTED]"+SECRETZQ`,
		`[INFO] Bearer "[REDACTED]"SECRETZQ`,
		`[INFO] Bearer '[REDACTED]'.SECRETZQ`,
		`[INFO] Bearer '[REDACTED].'SECRETZQ`,
		`[INFO] Bearer '%5BREDACTED%5D'SECRETZQ`,
		`[INFO] Bearer \'[REDACTED]\'SECRETZQ`,
		"[INFO] Bearer `[REDACTED]`SECRETZQ",
		`[INFO] Bearer ''SECRETZQ`,
		`[INFO] Bearer ""+SECRETZQ`,
		"[INFO] Bearer ``SECRETZQ",
		`{"log":"Bearer \"\"SECRETZQ`,
		`{"log":"Bearer \"[REDACTED]\"SECRETZQ`,
		`{"log":"Bearer \u005c"\u005c"SECRETZQ`,
		`{"log":"Bearer \u005c"[REDACTED]\u005c"SECRETZQ`,
		`[INFO] Bearer '[REDACTED]'`,
		`[INFO] Bearer "[REDACTED]" ok`,
		`[INFO] Basic "" done`,
		"[INFO] Basic `[REDACTED]` done",
		`{"msg":"Use Basic '' or Bearer '[REDACTED]' in docs."}`,
		`[INFO] Bearer [REDACTED]'SECRETZQ`,
		`[INFO] Bearer [REDACTED]"SECRETZQ"`,
		`{"log":"Bearer [REDACTED]'SECRETZQ`,
		`[INFO] Bearer [REDACTED])SECRETZQ`,
		"[INFO] Bearer [REDACTED]`SECRETZQ",
		`[INFO] Bearer [REDACTED]SECRETZQ`,
		`[INFO] Bearer [REDACTED]\"SECRETZQ`,
		`[INFO] Basic [REDACTED]dXNlcjpwYXNz`,
		`{"log":"Basic [REDACTED]dXNlcjpwYXNz`,
		`[INFO] Bearer [redacted]'SECRETZQ`,
		`[INFO] Basic [redacted]dXNlcjpwYXNz`,
		`[INFO] Basic [REDACTED]'dXNlcjpwYXNz`,
		`[INFO] Basic [ReDaCtEd]dXNlcjpwYXNz`,
		`[INFO] Bearer [REDACTED_SENSITIVE_TEXT]'SECRETZQ`,
		`[INFO] Bearer [UNPARSEABLE_JSON_BLOB]'SECRETZQ`,
		`[INFO] Basic [REDACTED_SENSITIVE_TEXT]dXNlcjpwYXNz`,
		`[INFO] Bearer [REDACTED].'SECRETZQ`,
		`[INFO] Bearer [REDACTED])'SECRETZQ`,
		`[INFO] Bearer %5BREDACTED%5D'SECRETZQ`,
		`[INFO] Basic %5BREDACTED%5DdXNlcjpwYXNz`,
		`{"log":"Basic [REDACTED]\u0027dXNlcjpwYXNz`,
		"[INFO] Bearer\u00a0[REDACTED]'SECRETZQ",
		`[INFO] password'%ZZ=SECRETZQ`,
		`{"log":"password'%ZZ=SECRETZQ`,
		`[INFO] user's password'%ZZ=SECRETZQ`,
		`[INFO] "msg: password"%ZZ=SECRETZQ`,
		`[INFO] "msg: password"%c0%a0=SECRETZQ`,
		`[INFO] password\"%ZZ=SECRETZQ`,
		`[INFO] password\'%ZZ=SECRETZQ`,
		`[INFO] "msg: api_key"%ZZ:SECRETZQ`,
		"[INFO] password`%ZZ=SECRETZQ",
		`[INFO] temp:a'%ZZ=SECRETZQ`,
		`[INFO] password''%ZZ=SECRETZQ`,
		`[INFO] "client_secret"'%ZZ:SECRETZQ`,
		`[INFO] "client_secret""%ZZ:SECRETZQ`,
		`[INFO] " client_secret ""%ZZ:SECRETZQ`,
		`{"log":"password\u0027%ZZ=SECRETZQ`,
		`[INFO] Bearer [REDACTED] ok`,
		`[INFO] said "Bearer [REDACTED]" ok`,
		`[INFO] (Bearer [REDACTED]) ok`,
		`[INFO] said 'Bearer [REDACTED]' ok`,
		`{"msg":"Use Bearer [REDACTED] or Basic [REDACTED] in docs."}`,
		`[INFO] Basic [step](https://example.com)`,
		`[INFO] Basic [REDACTED]`,
		`[INFO] "status"%20: ok`,
		`[INFO] user's name'%20= bob`,
		`[INFO] "50"% done`,
		`[INFO] "msg: password "%ZZ=SECRETZQ`,
		`{"db/password "%ZZ:"SECRETZQ"}`,
		`[INFO] user's password '%ZZ=SECRETZQ`,
		`[INFO] "msg: password "'%ZZ=SECRETZQ`,
		`[INFO] \'msg: password \'%ZZ=SECRETZQ`,
		"[INFO] \"msg:\npassword \"%ZZ=SECRETZQ",
		`[INFO] temp:%xy %ZZ=SECRETZQ`,
		"[INFO] temp:%xy\n%ZZ=SECRETZQ",
		"{\"log\":\"temp:%xy\t%ZZ=SECRETZQ",
		`[INFO] temp:%C2%A0 %ZZ=SECRETZQ`,
		`[INFO] "hello world "%ZZ=foo`,
		`[INFO] temp: %ZZ=foo`,
		"password:\n  temp:SECRET123:x: bar",
		"password:\n  user:temp:SECRETZQ:x: bar",
		"password:\n  app:temp:SECRETZQ:x: bar",
		`{"startPageToken": "p1", "newStartPageToken": "p2", "resumeToken": "r1", "eot_token": "e1"}`,
		`[INFO] Bearer'SECRET`,
		`[INFO] Basic'dXNlcjpwYXNzZQ`,
		`[INFO] Bearer '[REDACTED]'&SECRET`,
		"[INFO] password\ufe0f=SECRET",
		"[INFO] password\u034f=SECRET",
		"[INFO] password\u3164=SECRET",
		`{"log":"[INFO] 'temp:a\\'x'=SECRET`,
		"{\"log\":\"[INFO] `temp:a,b\\\\`x`=SECRET",
		`[INFO] Basic mode=Okta`,
		`[CONFIG] Basic mode=Only over TLS`,
		`[INFO] password\t=[REDACTED]`,
		`[INFO] api_key\n: [REDACTED]`,
		`{"log":"Enter your password\n:`,
		"[INFO] Saisissez le token\u00a0:",
		`[INFO] Plan: Basic 'monthly'`,
		"[Basic `curl` usage](https://example.com/docs)",
		`[INFO] opened C:\Tools\Basic\bin\app.exe`,
		"[INFO] `password`: must be at least 8 characters",
		`{"Credentials":{"AccessKeyId":"EXAMPLEKEYID","SecretAccessKey":"SECRET123","SessionToken":"SECRET123"},"Message":"ok"}`,
		`{"token":{"value":"x"},"status":"ok"}`,
		`{"error":"unexpected token \":\" at line 3"}`,
		`"[object Object]" is not valid JSON`,
		"\"authorization\":\nQZX7193-authorization =8",
		"log: \"authorization\":\nQZX7193-authorization =8",
		`{"a":"[REDACTED_SENSITIVE_URI] Bearer'SECRET"}`,
		"\"api_key\":\"[REDACTED]\" password\ufe0f=SECRET",
		`"hello" "\u005bREDACTED_SENSITIVE_URI\u005d Bearer'SECRETZQ"`,
		`"password=[REDACTED]" hunter2ZQ`,
		`{"a":"\"note\" [REDACTED] hunter2ZQ"}`,
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
