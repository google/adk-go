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
	"strings"
	"testing"
	"time"
)

func promptCfg() *Config {
	return &Config{
		Owner:                     "google",
		Repo:                      "adk-go",
		StaleLabel:                "stale",
		RequestClarificationLabel: "request clarification",
		// The two thresholds must DIFFER. With both at 168h every assertion on
		// "7 days" matched either placeholder, so swapping the two entries in
		// renderPrompt's replacer was invisible and the prompt would have stated
		// the wrong policy to the model.
		StaleAfter: 168 * time.Hour,
		CloseAfter: 72 * time.Hour,
	}
}

// The rendered prompt is passed to llmagent as Instruction, which performs {}
// session-state templating. Any leftover brace would be treated as a missing
// state key and fail every run, so the render must leave none. This guards
// against adding a new {placeholder} to the prompt without a replacer entry.
func TestRenderPrompt_NoStrayBraces(t *testing.T) {
	out := renderPrompt(promptCfg())
	if i := strings.IndexAny(out, "{}"); i != -1 {
		start := i - 30
		if start < 0 {
			start = 0
		}
		end := i + 30
		if end > len(out) {
			end = len(out)
		}
		t.Errorf("rendered prompt still contains a brace near: %q", out[start:end])
	}
}

func TestRenderPrompt_StripsBracesFromConfig(t *testing.T) {
	// A brace arriving via a config value (e.g. an odd label name) must not leak
	// into the instruction, where llmagent would treat it as a missing state key.
	cfg := promptCfg()
	cfg.StaleLabel = "needs_{info}"
	if out := renderPrompt(cfg); strings.ContainsAny(out, "{}") {
		t.Errorf("renderPrompt() with a braced label left stray brace(s):\n%s", out)
	}
}

func TestRenderPrompt_SubstitutesPlaceholders(t *testing.T) {
	out := renderPrompt(promptCfg())
	for _, want := range []string{
		"google/adk-go", "'stale'", "'request clarification'",
		// The two thresholds land in different places and must not be swapped:
		// the stale threshold gates STEP 3, the close threshold gates STEP 1.
		"stale after 7 days",
		// The Go gates read the actor and author clocks, so the prompt must name
		// the same fields. Pointing the model at days_since_activity let anyone
		// who had ever commented suppress a mark-stale the Go gate would allow,
		// by re-saving their own old comment.
		"`days_since_last_actor_action` > 7",
		"`days_since_stale_label` > 3",
		// Every branch must carry the same condition the matching Go predicate
		// applies, in the same form. Two rounds of this review found the prompt
		// and the gates disagreeing, and a threshold where Go tests an ordering
		// is the shape both took.
		"`days_since_author_action` >= `days_since_last_actor_action`",
		"`days_since_last_actor_action` <= `days_since_stale_label`",
		"`days_since_author_action` < `days_since_stale_label`",
		"`days_since_author_action` <= `days_since_last_actor_action`",
		// days_since_activity is context only; no Go gate reads it.
		"NEVER make a decision from",
		// Each STEP 1 branch must still be present and reachable.
		"1a. THE USER CAME BACK AFTER THE LABEL",
		"1b. THE LABEL'S AGE IS UNKNOWN",
		"1c. STILL WAITING ON THE AUTHOR",
		"1d. ANYTHING ELSE",
		"Take the FIRST branch that matches",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered prompt missing %q", want)
		}
	}
}

// No decision in the prompt may gate on days_since_activity: no Go predicate
// reads it, so a threshold comparison there can only put the model out of step
// with the gate that will actually run.
func TestRenderPrompt_NoDecisionGatesOnTheActivityClock(t *testing.T) {
	out := renderPrompt(promptCfg())
	for _, forbidden := range []string{
		"`days_since_activity` >", "`days_since_activity` <",
		"[days_since_activity]",
	} {
		if strings.Contains(out, forbidden) {
			t.Errorf("prompt contains %q: the Go gates read days_since_last_actor_action and days_since_author_action", forbidden)
		}
	}
}

func TestFormatDays(t *testing.T) {
	cases := map[time.Duration]string{
		168 * time.Hour: "7",
		24 * time.Hour:  "1",
		12 * time.Hour:  "0.5",
		36 * time.Hour:  "1.5",
	}
	for d, want := range cases {
		if got := formatDays(d); got != want {
			t.Errorf("formatDays(%v) = %q, want %q", d, got, want)
		}
	}
}
