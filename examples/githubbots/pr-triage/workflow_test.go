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

// A workflow_dispatch naming a pull request must land in the SAME concurrency
// group as that pull request's event-driven run. Keyed only on
// github.event.pull_request.number, a dispatch falls back to the run id and gets
// a group of its own, so two processes triage one pull request with independent
// claim tables.
func TestShippedWorkflowSerializesDispatchWithTheEventRun(t *testing.T) {
	wf := readWorkflow(t)
	groupRe := regexp.MustCompile(`(?m)^\s*group:\s*(.+)$`)
	m := groupRe.FindStringSubmatch(wf)
	if m == nil {
		t.Fatal("the workflow declares no concurrency group")
	}
	group := m[1]
	if !containsFold(group, "github.event.pull_request.number") {
		t.Errorf("concurrency group %q is not keyed on the pull request number", group)
	}
	if !containsFold(group, "inputs.pr_number") {
		t.Errorf("concurrency group %q ignores inputs.pr_number, so a dispatch for a pull request "+
			"runs in a different group from that pull request's own event", group)
	}
}

func containsFold(haystack, needle string) bool {
	return regexp.MustCompile(`(?i)` + regexp.QuoteMeta(needle)).MatchString(haystack)
}
