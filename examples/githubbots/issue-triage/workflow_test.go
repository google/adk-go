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
	"os"
	"regexp"
	"testing"
	"time"
)

// workflowPath is the workflow that runs this module in CI. Nothing else binds
// the environment variable names it sets to the ones loadConfig reads, so a
// rename on either side would leave the job silently running on defaults --
// with no gate able to see it, since actionlint checks the YAML and the Go
// suite never looks at it.
const workflowPath = "../../../.github/workflows/issue-triage-bot.yml"

// literalEnv matches the `KEY: value` lines of the workflow's env block that
// carry a literal, ignoring the ones whose value is a ${{ }} expression or a
// secret -- those are supplied by the runner and cannot be replayed here.
var literalEnv = regexp.MustCompile(`(?m)^\s{10}([A-Z][A-Z0-9_]*):\s*'?([^'$\n]+?)'?\s*$`)

// The workflow's settings must actually produce the configuration it intends.
// This reads the real file and drives the real loadConfig with it, so a renamed
// key stops taking effect and this test says so.
func TestWorkflowEnvProducesTheIntendedConfig(t *testing.T) {
	raw, err := os.ReadFile(workflowPath)
	if err != nil {
		// Deliberately fatal rather than skipped. A skip here would pass
		// silently for exactly the reason the test exists: the file moved.
		t.Fatalf("read %s: %v (the workflow that runs this module must exist)", workflowPath, err)
	}

	matches := literalEnv.FindAllStringSubmatch(string(raw), -1)
	if len(matches) == 0 {
		t.Fatalf("found no literal env assignments in %s; the pattern or the workflow changed shape", workflowPath)
	}

	setRequired(t)
	seen := make(map[string]string, len(matches))
	for _, m := range matches {
		key, value := m[1], m[2]
		seen[key] = value
		t.Setenv(key, value)
	}
	t.Logf("workflow env: %v", seen)

	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() with the workflow's own environment = %v; the workflow would not start the bot", err)
	}

	// The values the workflow deliberately overrides, and the arithmetic between
	// them. ISSUE_COUNT x ISSUE_TIMEOUT must fit inside SWEEP_TIMEOUT, or an
	// ordinary busy sweep exhausts its budget and the job goes red having
	// behaved exactly as designed.
	for key, want := range map[string]any{
		"ISSUE_COUNT":   5,
		"ISSUE_TIMEOUT": 3 * time.Minute,
		"SWEEP_TIMEOUT": 18 * time.Minute,
	} {
		if _, ok := seen[key]; !ok {
			t.Errorf("the workflow no longer sets %s; loadConfig would fall back to its default", key)
		}
		var got any
		switch key {
		case "ISSUE_COUNT":
			got = cfg.IssueCount
		case "ISSUE_TIMEOUT":
			got = cfg.IssueTimeout
		case "SWEEP_TIMEOUT":
			got = cfg.SweepTimeout
		}
		if got != want {
			t.Errorf("%s reached the config as %v, want %v", key, got, want)
		}
	}

	if budget := time.Duration(cfg.IssueCount) * cfg.IssueTimeout; budget > cfg.SweepTimeout {
		t.Errorf("ISSUE_COUNT x ISSUE_TIMEOUT = %s exceeds SWEEP_TIMEOUT %s: a full sweep cannot finish inside its own budget",
			budget, cfg.SweepTimeout)
	}
}

// timeoutMinutes is the job-level limit. The process budget has to stay below
// it, or the runner kills the job mid-sweep, which is silent.
var timeoutMinutes = regexp.MustCompile(`(?m)^\s*timeout-minutes:\s*(\d+)\s*$`)

func TestWorkflowJobTimeoutExceedsTheProcessBudget(t *testing.T) {
	raw, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read %s: %v", workflowPath, err)
	}
	m := timeoutMinutes.FindStringSubmatch(string(raw))
	if m == nil {
		t.Fatal("the workflow job declares no timeout-minutes; an overrun would run to the runner's own limit")
	}
	job, err := time.ParseDuration(m[1] + "m")
	if err != nil {
		t.Fatalf("parse timeout-minutes %q: %v", m[1], err)
	}

	sweep := literalEnv.FindAllStringSubmatch(string(raw), -1)
	var budget time.Duration
	for _, e := range sweep {
		if e[1] != "SWEEP_TIMEOUT" {
			continue
		}
		if budget, err = time.ParseDuration(e[2]); err != nil {
			t.Fatalf("parse SWEEP_TIMEOUT %q: %v", e[2], err)
		}
	}
	if budget == 0 {
		t.Fatal("the workflow sets no SWEEP_TIMEOUT, so the process has no budget below the job limit")
	}
	if budget >= job {
		t.Errorf("SWEEP_TIMEOUT %s is not below timeout-minutes %s: the job would be killed before the process could report what it left",
			budget, job)
	}
}
