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
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
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

// workflowModel returns the model the workflow pins, if it pins one.
func workflowModel(raw string) (string, bool) {
	m := regexp.MustCompile(`(?m)^\s*LLM_MODEL_NAME:\s*(\S+)\s*$`).FindStringSubmatch(raw)
	if m == nil {
		return "", false
	}
	return strings.Trim(m[1], `"'`), true
}

// The workflow must pin the model, and this must be checkable for free.
//
// A pin is the difference between a scheduled job whose model is a reviewable
// line in a diff and one that inherits whatever a floating alias resolves to
// that morning. Not hypothetical here: the alias this bot defaults to was
// measured returning 503 on every call over the credential path the workflow
// uses, so before the pin every triggered run and every six-hourly sweep would
// have failed.
//
// This began inside the e2e suite and does not belong there. It makes no model
// call -- it reads the workflow and compares against a constant -- so gating it
// behind the paid opt-in meant it never ran in CI, which is exactly where the
// drift it detects would appear. A guard nobody runs decays into the thing it
// was written to prevent. The half that genuinely needs a configured backend,
// comparing what the suite actually drove against this, stays gated.
//
// The specific model is deliberately not asserted: naming it here would make
// every model change a two-file edit and buy nothing. What must hold is that
// SOME explicit, non-floating version is chosen.
//
// Killing mutations, both verified: delete LLM_MODEL_NAME from the workflow;
// set it back to a *-latest alias.
func TestWorkflowPinsANonFloatingModel(t *testing.T) {
	raw := readWorkflow(t)

	model, ok := workflowModel(raw)
	if !ok {
		t.Fatalf("%s sets no LLM_MODEL_NAME, so the scheduled job runs whatever the code default "+
			"%q resolves to on the day, and a model change reaches production without a diff.",
			workflowPath, defaultModel)
	}
	if model == defaultModel || strings.HasSuffix(model, "-latest") {
		t.Errorf("%s pins LLM_MODEL_NAME to %q, which is a floating alias rather than a version. "+
			"It can be repointed with no change to this repository, so a model regression arrives "+
			"as a production incident and not as a broken build.", workflowPath, model)
	}
}

// The write scope belongs to the job that writes, not to the workflow.
//
// Permissions declared at workflow level apply to every job in it, so granting
// issues:write there hands the token to any job added later -- a reporting step,
// a notifier -- that has no business writing to the tracker. With one job the two
// placements behave identically, which is exactly why this drifts: nothing fails
// when the grant is in the wrong place, right up until the second job exists.
//
// Both halves are asserted. The deny-all at workflow level matters on its own:
// without it a new job falls back to the repository's default token scopes
// rather than to nothing, and that default is a repository setting this file
// cannot see or control.
func TestWorkflowGrantsWriteOnlyToTheJobThatWrites(t *testing.T) {
	raw := readWorkflow(t)

	if !regexp.MustCompile(`(?m)^permissions: \{\}\s*$`).MatchString(raw) {
		t.Errorf("%s does not declare `permissions: {}` at workflow level. Without it a job added "+
			"later inherits the repository default token scopes instead of nothing.", workflowPath)
	}
	// A workflow-level block with entries under it is the failure this guards.
	if m := regexp.MustCompile(`(?m)^permissions:[ \t]*\n((?:[ \t]+\S+:.*\n)+)`).FindStringSubmatch(raw); m != nil {
		t.Errorf("%s grants token scopes at workflow level:\n%s"+
			"Every job in the workflow inherits these, including ones added later. Move them onto "+
			"the job that needs them.", workflowPath, m[1])
	}

	_, jobs, found := strings.Cut(raw, "\njobs:\n")
	if !found {
		t.Fatalf("no jobs: block in %s; this guard no longer knows where to look", workflowPath)
	}
	// Four spaces is job level, six is a scope under it.
	if !regexp.MustCompile(`(?m)^    permissions:[ \t]*$`).MatchString(jobs) {
		t.Errorf("the job in %s declares no permissions: block of its own, so its token scopes are "+
			"whatever it inherits.", workflowPath)
	}
	for _, want := range []string{"issues: write", "contents: read"} {
		if !regexp.MustCompile(`(?m)^      ` + regexp.QuoteMeta(want) + `[ \t]*$`).MatchString(jobs) {
			t.Errorf("the job in %s does not grant %q itself. It sets an issue type and adds a label, "+
				"and checkout needs to read, so both belong on the job.", workflowPath, want)
		}
	}
}

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
	// them: ISSUE_COUNT x ISSUE_TIMEOUT plus the work-set fetch must fit inside
	// SWEEP_TIMEOUT, or an ordinary busy sweep exhausts its budget and the job
	// goes red having behaved exactly as designed.
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

	// The run budget funds choosing the work set as well as triaging it, so the
	// fetch has to be in the sum. Leaving it out is how the arithmetic came to
	// look sound while a full sweep could still exhaust its own budget.
	if budget := time.Duration(cfg.IssueCount)*cfg.IssueTimeout + fetchTimeout; budget > cfg.SweepTimeout {
		t.Errorf("ISSUE_COUNT x ISSUE_TIMEOUT + the work-set fetch = %s exceeds SWEEP_TIMEOUT %s: a full sweep cannot finish inside its own budget",
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

// runBlock extracts the workflow step's `run:` script.
//
// The script is where the two inputs an operator or an attacker controls are
// turned into behavior -- the dry-run switch and the issue number -- and no
// other gate executes it. actionlint with shellcheck checks that it parses and
// quotes correctly; neither evaluates a branch.
var runBlock = regexp.MustCompile(`(?s)\n        run: \|\n(.*?)\n\n  ?\w|(?s)\n        run: \|\n(.*)$`)

func workflowScript(t *testing.T) string {
	t.Helper()
	raw := readWorkflow(t)
	m := runBlock.FindStringSubmatch(raw)
	if m == nil {
		t.Fatal("could not find the step's run: block; the workflow changed shape")
	}
	body := m[1]
	if body == "" {
		body = m[2]
	}
	// Strip the block's YAML indentation.
	var out []string
	for line := range strings.SplitSeq(body, "\n") {
		out = append(out, strings.TrimPrefix(line, "          "))
	}
	script := strings.Join(out, "\n")
	if !strings.Contains(script, "set -euo pipefail") {
		t.Fatalf("the extracted script does not look like the workflow's:\n%s", script)
	}
	return script
}

// execScript runs the real script under bash with `go` replaced by a shim that
// prints its arguments, and returns what the shim saw plus the exit status.
func execScript(t *testing.T, env map[string]string) (goArgs, dryRun string, err error) {
	t.Helper()
	bash, lookErr := exec.LookPath("bash")
	if lookErr != nil {
		t.Skipf("bash is not available: %v", lookErr)
	}

	dir := t.TempDir()
	shim := filepath.Join(dir, "go")
	// The shim also reports DRY_RUN, which the script exports rather than
	// passing on the command line.
	if writeErr := os.WriteFile(shim, []byte("#!/bin/sh\necho \"ARGS:$*\"\necho \"DRY_RUN:$DRY_RUN\"\n"), 0o755); writeErr != nil {
		t.Fatalf("write go shim: %v", writeErr)
	}

	cmd := exec.Command(bash, "-c", workflowScript(t))
	cmd.Env = append(os.Environ(), "PATH="+dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	out, err := cmd.CombinedOutput()
	for line := range strings.SplitSeq(string(out), "\n") {
		if rest, ok := strings.CutPrefix(line, "ARGS:"); ok {
			goArgs = rest
		}
		if rest, ok := strings.CutPrefix(line, "DRY_RUN:"); ok {
			dryRun = rest
		}
	}
	t.Logf("script output:\n%s", out)
	return goArgs, dryRun, err
}

// The dry-run switch, executed. An input GitHub can deliver but the declared
// default does not cover -- the empty string a REST dispatch produces -- must
// resolve to a rehearsal, and anything unrecognised must stop the run rather
// than be guessed at.
//
// Killing mutation: change the `"")` arm of the DRY_RUN case to false, or
// delete the `*)` arm.
func TestWorkflowScriptResolvesDryRun(t *testing.T) {
	for _, tc := range []struct {
		event, input string
		want         string
		wantErr      bool
	}{
		{"workflow_dispatch", "true", "true", false},
		{"workflow_dispatch", "false", "false", false},
		// GitHub applies an input's declared default only when the key is
		// ABSENT. A REST dispatch carrying "dry_run": "" is present-but-empty.
		{"workflow_dispatch", "", "true", false},
		{"workflow_dispatch", "TRUE", "", true},
		{"workflow_dispatch", "yes", "", true},
		{"workflow_dispatch", "1", "", true},
		// The bot doing its job, not a rehearsal.
		{"issues", "", "false", false},
		{"schedule", "", "false", false},
	} {
		t.Run(tc.event+"/"+tc.input, func(t *testing.T) {
			_, got, err := execScript(t, map[string]string{
				"GH_REPOSITORY": "google/adk-go",
				"EVENT_NAME":    tc.event,
				"DRY_RUN_INPUT": tc.input,
				"ISSUE_INPUT":   "",
			})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("dry_run=%q was accepted, want the script to refuse it", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("script failed: %v", err)
			}
			if got != tc.want {
				t.Errorf("DRY_RUN = %q, want %q", got, tc.want)
			}
		})
	}
}

// The issue-number guard, executed. A grep would match per LINE, so an input
// carrying a newline would pass it.
//
// Killing mutation: replace the `case` with the original
// `printf '%s' "$ISSUE_INPUT" | grep -Eq '^[0-9]+$'`.
func TestWorkflowScriptValidatesTheIssueNumber(t *testing.T) {
	for _, tc := range []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"", "run .", false}, // no -issue: a sweep
		{"42", "run . -issue 42", false},
		{"010", "run . -issue 010", false}, // the Go side reads this as decimal 10
		{"abc", "", true},
		{"4 2", "", true},
		{"-5", "", true},
		{"42x", "", true},
		{"42\nrm -rf /", "", true},
	} {
		t.Run(strings.ReplaceAll(tc.input, "\n", `\n`), func(t *testing.T) {
			args, _, err := execScript(t, map[string]string{
				"GH_REPOSITORY": "google/adk-go",
				"EVENT_NAME":    "schedule",
				"DRY_RUN_INPUT": "",
				"ISSUE_INPUT":   tc.input,
			})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("issue input %q was accepted, want the script to refuse it", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("script failed: %v", err)
			}
			if args != tc.want {
				t.Errorf("go was invoked with %q, want %q", args, tc.want)
			}
			// Nothing else binds the flag NAME the script emits to the flag
			// loadConfig registers: both sides hardcode "-issue" independently,
			// so renaming the Go flag would leave every test green. Feed the
			// script's own argv to the real loadConfig.
			argv := strings.Fields(strings.TrimPrefix(args, "run ."))
			setRequired(t)
			cfg, cfgErr := loadConfig(argv)
			if cfgErr != nil {
				t.Fatalf("loadConfig(%q), the arguments the workflow actually passes, = %v", argv, cfgErr)
			}
			want := 0
			if tc.input != "" {
				want, _ = strconv.Atoi(tc.input)
			}
			if cfg.SingleIssue != want {
				t.Errorf("the workflow's %q reached the config as SingleIssue=%d, want %d", args, cfg.SingleIssue, want)
			}
		})
	}
}

// OWNER and REPO are derived from github.repository, not configured, so a
// change to the derivation would silently point the bot elsewhere.
func TestWorkflowScriptDerivesOwnerAndRepo(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("bash is not available: %v", err)
	}
	dir := t.TempDir()
	shim := filepath.Join(dir, "go")
	if err := os.WriteFile(shim, []byte("#!/bin/sh\necho \"OWNER:$OWNER REPO:$REPO\"\n"), 0o755); err != nil {
		t.Fatalf("write go shim: %v", err)
	}
	cmd := exec.Command(bash, "-c", workflowScript(t))
	cmd.Env = append(os.Environ(),
		"PATH="+dir+string(os.PathListSeparator)+os.Getenv("PATH"),
		"GH_REPOSITORY=google/adk-go", "EVENT_NAME=schedule", "DRY_RUN_INPUT=", "ISSUE_INPUT=")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("script failed: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "OWNER:google REPO:adk-go") {
		t.Errorf("the script derived %q, want OWNER:google REPO:adk-go", out)
	}
}
