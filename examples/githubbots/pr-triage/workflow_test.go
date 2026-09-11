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
	"strconv"
	"strings"
	"testing"
	"time"
)

// workflowPath is the shipped workflow, relative to this module.
const workflowPath = "../../../.github/workflows/pr-triage-bot.yml"

func readWorkflow(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(workflowPath)
	if err != nil {
		// Not skipped: the whole point of these tests is that the SHIPPED pair
		// of values agrees, and a skip would let the file be renamed or deleted
		// while the suite stayed green.
		t.Fatalf("read %s: %v", workflowPath, err)
	}
	return string(b)
}

// The process budget exists so an overrun is reported by the bot with a
// diagnosis, rather than the job being killed by the runner with nothing to go
// on. That only holds if the SHIPPED pair of values agrees.
//
// Asserting the Go constant against a literal copied out of the YAML does not
// do that: the workflow overrides the default with its own RUN_BUDGET, so
// editing either number leaves such a test green while breaking the invariant
// it names. This reads both from the file.
func TestShippedRunBudgetIsBelowTheShippedJobTimeout(t *testing.T) {
	wf := readWorkflow(t)

	timeoutRe := regexp.MustCompile(`(?m)^\s*timeout-minutes:\s*(\d+)`)
	tm := timeoutRe.FindStringSubmatch(wf)
	if tm == nil {
		t.Fatal("the workflow declares no timeout-minutes, so an overrun has no outer bound")
	}
	minutes, err := strconv.Atoi(tm[1])
	if err != nil {
		t.Fatalf("timeout-minutes %q: %v", tm[1], err)
	}
	jobTimeout := time.Duration(minutes) * time.Minute

	budgetRe := regexp.MustCompile(`(?m)^\s*RUN_BUDGET:\s*(\S+)`)
	bm := budgetRe.FindStringSubmatch(wf)
	if bm == nil {
		t.Fatal("the workflow sets no RUN_BUDGET, so the process cannot report its own overrun")
	}
	budget, err := time.ParseDuration(bm[1])
	if err != nil {
		t.Fatalf("RUN_BUDGET %q: %v", bm[1], err)
	}

	if budget >= jobTimeout {
		t.Errorf("RUN_BUDGET=%v is not below timeout-minutes=%v; the job would be killed "+
			"before the bot could report the overrun", budget, jobTimeout)
	}
	// The per-pull-request timeout must fit inside the run budget, or one pull
	// request can consume the whole run.
	if defaultPRTimeout > budget {
		t.Errorf("defaultPRTimeout=%v exceeds the shipped RUN_BUDGET=%v", defaultPRTimeout, budget)
	}
}

// Every third-party action must be pinned to a full 40-character commit SHA. A
// tag is mutable, and this workflow runs with the base repository's token.
func TestShippedWorkflowPinsEveryThirdPartyAction(t *testing.T) {
	wf := readWorkflow(t)
	usesRe := regexp.MustCompile(`(?m)^\s*uses:\s*(\S+)`)
	pinned := regexp.MustCompile(`^[^@]+@[0-9a-f]{40}$`)

	var found int
	for _, m := range usesRe.FindAllStringSubmatch(wf, -1) {
		ref := m[1]
		if len(ref) > 0 && ref[0] == '.' {
			// A local composite action is base-branch code, already trusted.
			continue
		}
		found++
		if !pinned.MatchString(ref) {
			t.Errorf("action %q is not pinned to a full 40-character commit SHA", ref)
		}
	}
	if found == 0 {
		t.Fatal("found no third-party action to check; the regex or the workflow changed shape")
	}
}

// The controls that make pull_request_target survivable. Each of these is one
// edit away from being lost, and none of them is covered by actionlint.
func TestShippedWorkflowKeepsItsHardening(t *testing.T) {
	wf := readWorkflow(t)
	for _, want := range []struct {
		fragment string
		why      string
	}{
		{"github.repository == 'google/adk-go'", "the fork guard stops the bot running on forks"},
		{"persist-credentials: false", "a persisted credential outlives the step that needed it"},
		{"set -euo pipefail", "an unchecked failure in the script would pass as success"},
		{"pull-requests: write", "the job's permissions must be declared, and narrowly"},
		{`=~ ^[0-9]+$`, "an unvalidated pull request number reaches a shell"},
		{"vars.PR_TRIAGE_OWNER_MAP != ''", "an unconfigured bot must skip, not fail red on a contributor's pull request"},
		{"timeout-minutes:", "the job needs an outer bound"},
	} {
		if !containsFold(wf, want.fragment) {
			t.Errorf("the workflow no longer contains %q: %s", want.fragment, want.why)
		}
	}

	// The pull request head must never be checked out: that is what would run
	// fork-controlled code with the base repository's token and secrets.
	if regexp.MustCompile(`(?m)^\s*ref:\s`).MatchString(wf) {
		t.Error("the workflow sets a checkout ref; pull_request_target must use the default (base) checkout")
	}
	// `synchronize` would re-trigger on every push to an open pull request, which
	// is how the bot would re-assign an owner a maintainer had just removed. Check
	// the trigger list itself: the prose above it names the event deliberately.
	typesRe := regexp.MustCompile(`(?m)^\s*types:\s*(\[.*\])`)
	tm := typesRe.FindStringSubmatch(wf)
	if tm == nil {
		t.Fatal("pull_request_target declares no explicit types, so it fires on every event")
	}
	if containsFold(tm[1], "synchronize") || containsFold(tm[1], "edited") {
		t.Errorf("trigger list %s re-runs the model over rewritten text; assignment must stay one-way", tm[1])
	}
	for _, want := range []string{"opened", "reopened", "ready_for_review"} {
		if !containsFold(tm[1], want) {
			t.Errorf("trigger list %s is missing %q", tm[1], want)
		}
	}
}

// The concurrency group must be CONSTANT, so at most one triage run exists
// repository-wide.
//
// This job is reachable by any stranger: pull_request_target runs from the base
// branch and is not held back by the "require approval for first-time
// contributors" gate that gates the main `pull_request` CI. A group keyed on the
// pull request number serializes same-pull-request runs but lets N open pull
// requests occupy N runners, and opening pull requests is free.
//
// The cost is real and accepted: with the default queue behavior a run queued
// while another is pending cancels that pending run, so under contention some
// pull requests are not auto-triaged. That is the right way to fail for an
// advisory job, and workflow_dispatch recovers a specific one.
func TestShippedWorkflowSerializesEveryTriageRun(t *testing.T) {
	wf := readWorkflow(t)
	groupRe := regexp.MustCompile(`(?m)^\s*group:\s*(.+)$`)
	m := groupRe.FindStringSubmatch(wf)
	if m == nil {
		t.Fatal("the workflow declares no concurrency group")
	}
	group := strings.TrimSpace(m[1])
	if strings.Contains(group, "${{") {
		t.Errorf("concurrency group %q is templated, so different pull requests land in "+
			"different groups and run concurrently. A stranger can open many pull requests, "+
			"and this job is not behind the first-time-contributor approval gate.", group)
	}
	// Cancelling in progress would drop a run mid-write.
	if !containsFold(wf, "cancel-in-progress: false") {
		t.Error("cancel-in-progress must be false: a triage run cancelled mid-write could " +
			"leave an assignment recorded with no comment, or the reverse")
	}
}

// The per-run cost ceiling. A stranger can trigger this job, so the job timeout
// is also the per-attack cost, and it must stay tight enough to be uninteresting.
func TestShippedWorkflowKeepsThePerRunCostTight(t *testing.T) {
	wf := readWorkflow(t)
	m := regexp.MustCompile(`(?m)^\s*timeout-minutes:\s*(\d+)`).FindStringSubmatch(wf)
	if m == nil {
		t.Fatal("no timeout-minutes")
	}
	minutes, err := strconv.Atoi(m[1])
	if err != nil {
		t.Fatalf("timeout-minutes %q: %v", m[1], err)
	}
	if minutes > 10 {
		t.Errorf("timeout-minutes = %d. The event path triages ONE pull request, which takes "+
			"about a minute; a large ceiling only raises what an attacker gets per pull "+
			"request they open.", minutes)
	}
}

// The deployed model must be pinned, not a floating alias. adk-python pins
// LLM_MODEL_NAME too. A floating alias changes the deployed model with no diff,
// and the alias this repo defaults to was measured shedding completely on the
// key path the workflow uses.
func TestShippedWorkflowPinsTheModel(t *testing.T) {
	wf := readWorkflow(t)
	m := regexp.MustCompile(`(?m)^\s*LLM_MODEL_NAME:\s*(\S+)`).FindStringSubmatch(wf)
	if m == nil {
		t.Fatal("the workflow does not pin LLM_MODEL_NAME, so the deployed model floats")
	}
	if strings.Contains(m[1], "latest") {
		t.Errorf("LLM_MODEL_NAME = %q is a floating alias; pin a concrete version", m[1])
	}
}

// TestShippedWorkflowNamesTheAPIKeySecretConsistently pins which secret the
// workflow asks for.
//
// Nothing pinned this before. The Go side accepts either GEMINI_API_KEY or
// GOOGLE_API_KEY, so renaming it in the workflow changes which secret an
// operator must configure while every test stays green -- and the resulting
// failure is the silent kind: the bot starts, finds no key, and does nothing on
// a repository where the other name is set. Five bots sharing one deployment
// want one name between them.
//
// Three properties, because each catches a different mistake:
//
//   - the env key and the secret expression agree, so a half-typed rename like
//     GEMINI_API_KEY: ${{ secrets.GOOGLE_API_KEY }} cannot ship;
//   - the name is the one these bots standardised on;
//   - the header comment documents the same secret, so a rename that fixes the
//     env line and leaves the prose behind fails here. Not hypothetical: the old
//     name lived in exactly those two places, and fixing only the obvious one is
//     how this job gets done wrong.
func TestShippedWorkflowNamesTheAPIKeySecretConsistently(t *testing.T) {
	const want = "GEMINI_API_KEY"
	wf := readWorkflow(t)

	m := regexp.MustCompile(`(?m)^\s*(\w*API_KEY):\s*\$\{\{\s*secrets\.(\w+)\s*\}\}`).FindStringSubmatch(wf)
	if m == nil {
		t.Fatal("the workflow passes no API-key secret to the bot; without one it can " +
			"never reach a model")
	}
	envKey, secretName := m[1], m[2]

	if envKey != secretName {
		t.Errorf("the workflow sets %s from secrets.%s. The names must match, or an "+
			"operator configures the secret the documentation names while the bot reads a "+
			"different, empty one.", envKey, secretName)
	}
	if envKey != want {
		t.Errorf("the workflow passes %s; these bots standardised on %s. Mixing them means "+
			"configuring four secrets and silently missing the fifth.", envKey, want)
	}

	// The header comment is the operator's checklist. If it still names the old
	// secret, they configure that one and the bot stays inert.
	if !regexp.MustCompile(`(?m)^#.*secrets\.` + regexp.QuoteMeta(secretName) + `\b`).MatchString(wf) {
		t.Errorf("the header comment listing required secrets does not document "+
			"secrets.%s, the one the run step actually reads. A rename that updated the env "+
			"line and left the comment behind is the likely cause.", secretName)
	}
}

func containsFold(haystack, needle string) bool {
	return regexp.MustCompile(`(?i)` + regexp.QuoteMeta(needle)).MatchString(haystack)
}
