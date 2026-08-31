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
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// readWorkflow returns the workflow source.
//
// Missing file, but the repository's workflow directory is there: fatal. That
// is the case this test exists for -- the workflow was renamed or moved and
// nothing else would notice. Directory absent entirely: the module has been
// copied out of adk-go, where there is no workflow to check, so skip.
func readWorkflow(t *testing.T) string {
	t.Helper()
	if _, err := os.Stat(filepath.Dir(workflowPath)); err != nil {
		t.Skipf("%s is not present, so this module is not in its adk-go tree: %v", filepath.Dir(workflowPath), err)
	}
	raw, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read %s: %v (the workflow that runs this module must exist)", workflowPath, err)
	}
	return string(raw)
}

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
	raw := readWorkflow(t)

	matches := literalEnv.FindAllStringSubmatch(raw, -1)
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
	raw := readWorkflow(t)
	m := timeoutMinutes.FindStringSubmatch(raw)
	if m == nil {
		t.Fatal("the workflow job declares no timeout-minutes; an overrun would run to the runner's own limit")
	}
	job, err := time.ParseDuration(m[1] + "m")
	if err != nil {
		t.Fatalf("parse timeout-minutes %q: %v", m[1], err)
	}

	var budget time.Duration
	for _, e := range literalEnv.FindAllStringSubmatch(raw, -1) {
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

// The keys whose values are runner expressions or shell-derived never reach the
// regex above, so the test that replays the literal env cannot see them renamed
// -- and they are the four whose absence stops the bot dead. Assert their names
// are present, which is the whole binding available without running Actions.
//
// Killing mutation: rename GITHUB_TOKEN, GEMINI_API_KEY or DRY_RUN in the
// workflow, or the `export OWNER REPO` the script derives from the repository.
func TestWorkflowSuppliesEveryKeyTheConfigRequires(t *testing.T) {
	raw := readWorkflow(t)
	for _, want := range []string{
		// Supplied by the runner as ${{ }} expressions.
		"GITHUB_TOKEN:",
		"GEMINI_API_KEY:",
		// Derived in the script from github.repository.
		"export OWNER REPO",
		// Resolved in the script, because the input's declared default only
		// applies when the key is absent from the dispatch payload.
		"export DRY_RUN",
	} {
		if !strings.Contains(raw, want) {
			t.Errorf("the workflow no longer contains %q; loadConfig would refuse to start, or the dry-run switch would not reach it", want)
		}
	}

	// And the Go side must still read them under those names.
	setRequired(t)
	if _, err := loadConfig(nil); err != nil {
		t.Fatalf("loadConfig() with the required variables set = %v", err)
	}
	for _, key := range []string{"GITHUB_TOKEN", "OWNER", "REPO"} {
		t.Run("without "+key, func(t *testing.T) {
			setRequired(t)
			t.Setenv(key, "")
			if _, err := loadConfig(nil); err == nil {
				t.Errorf("loadConfig() without %s = nil error; the workflow supplying it would be pointless", key)
			}
		})
	}
}
