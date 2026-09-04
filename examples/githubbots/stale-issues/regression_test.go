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

// Regression tests. Each one pins a specific defect this bot has had, and each
// has a one-line change to the production code that makes it fail — that is the
// bar for adding to this file, not coverage. The comments say what went wrong
// and what it cost, because most of these are cases where the obvious
// simplification is the bug.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// The allow-list matches case-insensitively so the bot can recognize a label the
// repository stored under different casing. Go's EqualFold is Unicode simple
// folding, so the accepted set is wider than ASCII: "requeſt clarification"
// (U+017F) folds equal to "request clarification" but is a DIFFERENT label to
// GitHub. Passing the model's argument through to the API would therefore let it
// choose the bytes that get written, which is what the allow-list exists to stop.
func TestLabelWritesUseTheConfiguredNameNotTheModelArgument(t *testing.T) {
	cfg := baseCfg()
	var (
		mu      sync.Mutex
		added   []string
		removed []string
	)
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/labels"):
			var body []string
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode label body: %v", err)
			}
			mu.Lock()
			added = append(added, body...)
			mu.Unlock()
			_, _ = io.WriteString(w, `[]`)
			return
		case r.Method == http.MethodDelete:
			// go-github puts the label in the last path segment.
			idx := strings.LastIndex(r.URL.Path, "/labels/")
			mu.Lock()
			removed = append(removed, r.URL.Path[idx+len("/labels/"):])
			mu.Unlock()
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	ctx := withAuditedIssue(context.Background(), 7)

	// Add: a fold-equal spelling of the clarification label.
	c.recordObservation(7, staleReady())
	if res, err := c.doAddLabel(ctx, 7, "REQUEST CLARIFICATION"); err != nil || res.Status != "success" {
		t.Fatalf("doAddLabel = (%+v, %v), want success", res, err)
	}

	// Remove: a fold-equal spelling of the stale label. U+017F folds to "s".
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceActivity = 1
	st.DaysSinceLastActorAction = 1
	st.DaysSinceStaleLabel = 10
	c.recordObservation(7, st)
	if res, err := c.doRemoveLabel(ctx, 7, "\u017Ftale"); err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success", res, err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(added) != 1 || added[0] != cfg.RequestClarificationLabel {
		t.Errorf("added %q, want the configured %q — the model must not choose the bytes sent to GitHub", added, cfg.RequestClarificationLabel)
	}
	if len(removed) != 1 || removed[0] != cfg.StaleLabel {
		t.Errorf("removed %q, want the configured %q", removed, cfg.StaleLabel)
	}
}

// The stale warning promises the issue closes only "if no further activity
// occurs". The role check alone does not deliver that: an author who answers and
// a maintainer who then replies leave the role at "maintainer" while the label
// keeps ageing, so the issue was closed as not-planned days after it was
// answered.
func TestCloseRefusedWhenSomeoneActedAfterTheLabel(t *testing.T) {
	c := newTestClient(t)
	st := okState()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	st.DaysSinceStaleLabel = 8 // past the 7-day close threshold
	st.DaysSinceActivity = 1.5 // but the thread moved yesterday
	st.DaysSinceLastActorAction = 1.5
	c.recordObservation(7, st)

	res, err := c.doClose(withAuditedIssue(context.Background(), 7), 7)
	if err != nil {
		t.Fatalf("doClose returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "close window") {
		t.Errorf("doClose = %+v, want a refusal: the issue was active inside the close window", res)
	}
}

// The activity check must not block the ordinary close, where the last activity
// is the maintainer's question from before the label.
func TestCloseAllowedWhenNothingHappenedAfterTheLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := okState()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	st.DaysSinceStaleLabel = 8
	st.DaysSinceActivity = 22 // the maintainer's question, well before the label
	st.DaysSinceLastActorAction = 22
	c.recordObservation(7, st)

	res, err := c.doClose(withAuditedIssue(context.Background(), 7), 7)
	if err != nil || res.Status != "success" {
		t.Fatalf("doClose = (%+v, %v), want success on a genuinely abandoned issue", res, err)
	}
}

// Editing an old comment advances the activity clock without changing whose turn
// it is, so the removal check has to read the last ROLE-BEARING action. Counting
// the edit let a third party who once commented reset the author's turn and get
// a maintainer's hand-applied label stripped.
func TestThirdPartyCommentEditDoesNotCountAsTheAuthorReturning(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "maint", Time: t0, Body: "any repro?"},
		{Type: eventCommented, Actor: "author", Time: t0.Add(24 * time.Hour), Body: "not yet"},
		// A stranger who commented long ago tweaks their own comment today.
		{Type: eventEditedComment, Actor: "passer-by", Time: t0.Add(240 * time.Hour), Body: "typo"},
	}, toSet([]string{"maint"}), "author")

	if !st.LastActivity.Equal(t0.Add(240 * time.Hour)) {
		t.Errorf("LastActivity = %v, want the edit time (an edit is still activity)", st.LastActivity)
	}
	if !st.LastActorAction.Equal(t0.Add(24 * time.Hour)) {
		t.Errorf("LastActorAction = %v, want the author's comment: a stranger's edit is not the author coming back", st.LastActorAction)
	}
	if st.LastActorRole != roleAuthor {
		t.Errorf("LastActorRole = %v, want %v", st.LastActorRole, roleAuthor)
	}
}

// The same person continuing to act does advance their own clock: an author who
// edits their comment to add the logs a maintainer asked for has come back.
func TestOwnCommentEditCountsAsTheSamePersonActing(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "maint", Time: t0, Body: "any repro?"},
		{Type: eventCommented, Actor: "author", Time: t0.Add(24 * time.Hour), Body: "working on it"},
		{Type: eventEditedComment, Actor: "Author", Time: t0.Add(240 * time.Hour), Body: "working on it — here are the logs"},
	}, toSet([]string{"maint"}), "author")

	if !st.LastActorAction.Equal(t0.Add(240 * time.Hour)) {
		t.Errorf("LastActorAction = %v, want the edit time: the author edited their own comment", st.LastActorAction)
	}
}

// The removal check must read the last ROLE-BEARING action, not the activity
// clock. They diverge exactly when the newest event is a third party editing an
// old comment, which is how a stranger got a maintainer's hand-applied label
// stripped, and a predicate reading days_since_activity would allow it.
func TestRemoveStaleReadsTheActorClockNotTheActivityClock(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceStaleLabel = 10
	st.DaysSinceActivity = 1         // a passer-by edited their old comment today
	st.DaysSinceLastActorAction = 40 // but the author has not spoken in 40 days
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "nobody has come back") {
		t.Errorf("doRemoveLabel = %+v, want a refusal: the activity was not the author returning", res)
	}
}

// An action exactly as old as the label is the boundary. It must be treated as
// the user coming back, not as predating the label.
func TestRemoveStaleAtTheOrderingBoundary(t *testing.T) {
	for _, tc := range []struct {
		name            string
		lastActorAction float64
		wantSuccess     bool
	}{
		{"exactly as old as the label", 10, true},
		{"inside the race grace", 10.01, true},
		{"clearly older than the label", 40, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestClient(t)
			c.cfg.DryRun = true
			st := staleReady()
			st.IsStale = true
			st.LastActionRole = string(roleOther) // the other-user branch, not just author
			st.DaysSinceStaleLabel = 10
			st.DaysSinceActivity = tc.lastActorAction
			st.DaysSinceLastActorAction = tc.lastActorAction
			c.recordObservation(7, st)

			res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
			if err != nil {
				t.Fatalf("doRemoveLabel returned a Go error: %v", err)
			}
			if got := res.Status == "success"; got != tc.wantSuccess {
				t.Errorf("doRemoveLabel = %+v, want success=%v", res, tc.wantSuccess)
			}
		})
	}
}

// The -issue flag has to reach the work. The workflow spends a hardened
// ^[1-9][0-9]*$ guard on this input, and nothing proved the value was honoured:
// loadConfig(args) -> loadConfig(nil), and SingleIssue != 0 -> == 0, both left
// the suite green while turning a single-issue dispatch into a live full sweep.
func TestSingleIssueFlagReachesTheAuditedSet(t *testing.T) {
	t.Chdir(t.TempDir())
	setRequiredCreds(t)

	var got *Config
	err := runWith(context.Background(), discardLogger(), []string{"-issue", "123"},
		func(_ context.Context, cfg *Config, _ *slog.Logger) error {
			got = cfg
			return nil
		})
	if err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if got == nil {
		t.Fatal("the audit step never ran")
	}
	if got.SingleIssue != 123 {
		t.Fatalf("SingleIssue = %d, want 123: the -issue argument did not reach the config", got.SingleIssue)
	}

	// And the config value has to select the single-issue branch.
	gh := testClient(t, got, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("candidateIssues ran the search despite -issue being set")
		_, _ = io.WriteString(w, `{"items":[]}`)
	}))
	issues, err := candidateIssues(context.Background(), gh, got)
	if err != nil {
		t.Fatalf("candidateIssues: %v", err)
	}
	if len(issues) != 1 || issues[0] != 123 {
		t.Errorf("candidateIssues = %v, want [123]", issues)
	}
}

// With no -issue the sweep must run, or the single-issue branch could swallow
// every run.
func TestNoSingleIssueRunsTheSweep(t *testing.T) {
	cfg := baseCfg()
	searched := false
	gh := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		searched = true
		_, _ = io.WriteString(w, `{"items":[{"number":9}]}`)
	}))
	issues, err := candidateIssues(context.Background(), gh, cfg)
	if err != nil {
		t.Fatalf("candidateIssues: %v", err)
	}
	if !searched {
		t.Fatal("candidateIssues never searched with SingleIssue unset")
	}
	if len(issues) != 1 || issues[0] != 9 {
		t.Errorf("candidateIssues = %v, want [9]", issues)
	}
}

// Every optional setting must actually be read from its own environment key.
// Replacing any one of these with a constant left the suite green, because no
// test ever supplied a valid value for it.
func TestLoadConfigReadsEveryOptionalSetting(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("LLM_MODEL_NAME", "gemini-test")
	t.Setenv("STALE_HOURS_THRESHOLD", "48")
	t.Setenv("CLOSE_HOURS_AFTER_STALE_THRESHOLD", "72")
	t.Setenv("STALE_LABEL_NAME", "needs-info")
	t.Setenv("REQUEST_CLARIFICATION_LABEL", "awaiting-reply")
	t.Setenv("CONCURRENCY_LIMIT", "8")
	t.Setenv("ISSUE_TIMEOUT", "90s")
	t.Setenv("RUN_BUDGET", "11m")
	t.Setenv("MAINTAINERS", "alice,bob")

	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	for _, tc := range []struct {
		key       string
		got, want any
	}{
		{"LLM_MODEL_NAME", cfg.Model, "gemini-test"},
		{"STALE_HOURS_THRESHOLD", cfg.StaleAfter, 48 * time.Hour},
		{"CLOSE_HOURS_AFTER_STALE_THRESHOLD", cfg.CloseAfter, 72 * time.Hour},
		{"STALE_LABEL_NAME", cfg.StaleLabel, "needs-info"},
		{"REQUEST_CLARIFICATION_LABEL", cfg.RequestClarificationLabel, "awaiting-reply"},
		{"CONCURRENCY_LIMIT", cfg.Concurrency, 8},
		{"ISSUE_TIMEOUT", cfg.IssueTimeout, 90 * time.Second},
		{"RUN_BUDGET", cfg.RunBudget, 11 * time.Minute},
	} {
		if tc.got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.key, tc.got, tc.want)
		}
	}
}

// A malformed GOOGLE_GENAI_USE_VERTEXAI must be reported like every other
// setting. It sat behind a short-circuit, so with an API key present it was
// never parsed and never reported.
func TestMalformedVertexFlagIsReportedEvenWithAnAPIKey(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "yes")
	if _, err := loadConfig(nil); err == nil || !strings.Contains(err.Error(), "GOOGLE_GENAI_USE_VERTEXAI") {
		t.Errorf("loadConfig() = %v, want it to name the malformed GOOGLE_GENAI_USE_VERTEXAI", err)
	}
}

// A non-finite hours value parses without error and converts to an
// implementation-defined Duration: +Inf saturates to a ~292-year threshold that
// passes validate() and silently no-ops the sweep.
func TestLoadConfigRejectsNonFiniteHours(t *testing.T) {
	for _, v := range []string{"Inf", "+Inf", "-Inf", "NaN"} {
		t.Run(v, func(t *testing.T) {
			setRequiredCreds(t)
			t.Setenv("STALE_HOURS_THRESHOLD", v)
			if _, err := loadConfig(nil); err == nil || !strings.Contains(err.Error(), "STALE_HOURS_THRESHOLD") {
				t.Errorf("loadConfig() with STALE_HOURS_THRESHOLD=%s = %v, want a refusal naming the key", v, err)
			}
		})
	}
}

// The prompt and the tool registry must name the SAME six tools, in both
// directions. Checking only registered-to-prompt let the prompt name a tool that
// does not exist, so the model emits calls that never resolve.
//
// The prompt writes tool names and IssueState field names the same way, in
// backticks, so the field names are excluded by reading them off the struct tags
// rather than by a hand-maintained list that would drift.
func TestPromptAndToolRegistryNameTheSameTools(t *testing.T) {
	c := newTestClient(t)
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools() error = %v", err)
	}
	registered := make(map[string]bool, len(tools))
	for _, tl := range tools {
		registered[tl.Name()] = true
	}
	if len(registered) == 0 {
		t.Fatal("no tools registered: this test would otherwise pass vacuously")
	}

	stateField := make(map[string]bool)
	rt := reflect.TypeFor[IssueState]()
	for i := range rt.NumField() {
		name, _, _ := strings.Cut(rt.Field(i).Tag.Get("json"), ",")
		if name != "" {
			stateField[name] = true
		}
	}
	if len(stateField) == 0 {
		t.Fatal("no IssueState json tags found: the exclusion set would be empty")
	}

	named := make(map[string]bool)
	for _, m := range regexp.MustCompile("`([a-z][a-z0-9_]+)`").FindAllStringSubmatch(renderPrompt(promptCfg()), -1) {
		if !stateField[m[1]] {
			named[m[1]] = true
		}
	}
	if len(named) == 0 {
		t.Fatal("the prompt names no tools at all: this test would otherwise pass vacuously")
	}

	for n := range registered {
		if !named[n] {
			t.Errorf("tool %q is registered but the prompt never names it", n)
		}
	}
	for n := range named {
		if !registered[n] {
			t.Errorf("the prompt tells the model to call %q, which is not registered", n)
		}
	}
}

// Dropping repo: from the search makes the bot act on issue numbers that came
// from someone else's tracker; dropping created:< turns a bounded sweep into
// every open issue. Neither was asserted.
func TestSearchQueryScopesTheRepositoryAndTheAgeCutoff(t *testing.T) {
	var q string
	cfg := baseCfg()
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, `{"items":[]}`)
	}))
	if _, err := c.SearchOldOpenIssues(context.Background()); err != nil {
		t.Fatalf("SearchOldOpenIssues: %v", err)
	}
	for _, want := range []string{"repo:" + cfg.Owner + "/" + cfg.Repo, "is:issue", "state:open", "created:<"} {
		if !strings.Contains(q, want) {
			t.Errorf("search query = %q, want it to contain %q", q, want)
		}
	}
}

// Activity must RESTART the close clock, not cancel it. Comparing activity
// against the label age instead was permanent: both are "days since", so they
// grow in lockstep and their difference never changes, leaving an issue that
// anyone had touched once un-closeable forever while every other tool also
// refused it.
func TestActivityPostponesTheCloseRatherThanCancellingIt(t *testing.T) {
	// One issue observed on three successive days. Everything ages together;
	// only the passage of time differs.
	for _, tc := range []struct {
		day                 float64
		wantClose           bool
		activityAt, labelAt float64 // days before day 0
	}{
		{day: 0, activityAt: 0, labelAt: 8, wantClose: false}, // touched today
		{day: 6, activityAt: 0, labelAt: 8, wantClose: false}, // still inside the window
		{day: 8, activityAt: 0, labelAt: 8, wantClose: true},  // quiet for 8 > 7 days
	} {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := okState()
		st.IsStale = true
		st.LastActionRole = string(roleMaintainer)
		st.DaysSinceActivity = tc.day - tc.activityAt
		st.DaysSinceLastActorAction = st.DaysSinceActivity
		st.DaysSinceStaleLabel = tc.labelAt + tc.day
		c.recordObservation(7, st)

		res, err := c.doClose(withAuditedIssue(context.Background(), 7), 7)
		if err != nil {
			t.Fatalf("day %.0f: doClose returned a Go error: %v", tc.day, err)
		}
		if got := res.Status == "success"; got != tc.wantClose {
			t.Errorf("day %.0f (activity %.1f days ago, label %.1f days old): close=%v, want %v — %s",
				tc.day, st.DaysSinceActivity, st.DaysSinceStaleLabel, got, tc.wantClose, res.Message)
		}
	}
}

// The boundary: activity exactly at the close threshold still blocks.
func TestCloseBoundaryOnTheActivityWindow(t *testing.T) {
	for _, tc := range []struct {
		daysSinceActivity float64
		wantClose         bool
	}{
		{6.9, false},
		{7.0, false}, // "within N days" includes N
		{7.1, true},
	} {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := okState()
		st.IsStale = true
		st.LastActionRole = string(roleMaintainer)
		st.DaysSinceStaleLabel = 30
		st.DaysSinceActivity = tc.daysSinceActivity
		st.DaysSinceLastActorAction = tc.daysSinceActivity
		c.recordObservation(7, st)

		res, err := c.doClose(withAuditedIssue(context.Background(), 7), 7)
		if err != nil {
			t.Fatalf("doClose returned a Go error: %v", err)
		}
		if got := res.Status == "success"; got != tc.wantClose {
			t.Errorf("activity %.1f days ago: close=%v, want %v (%s)", tc.daysSinceActivity, got, tc.wantClose, res.Message)
		}
	}
}

// The silent-edit alert de-duplicates against the DESCRIPTION EDIT, not against
// the activity clock. A comment edit advances the activity clock without
// changing last_action_type, so comparing there let anyone who had ever
// commented re-arm the alert and make the bot re-report an edit it had already
// reported — one comment per run, indefinitely.
func TestAlertIsNotReArmedByAnUnrelatedCommentEdit(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	raw.Comments = commentNodes(
		// A passer-by commented long ago and edits it today.
		rawComment{
			Author: actor("passer-by"), Body: "ping", CreatedAt: daysAgo(35),
			LastEditedAt: func() *time.Time { t := daysAgo(1); return &t }(),
		},
		// The bot already alerted about the description edit.
		rawComment{
			Author:    actor(testSelf),
			Body:      botAlertSignature + ". Maintainers, please review.",
			CreatedAt: daysAgo(19),
		},
	)
	raw.UserContentEdits = editNodes(rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(20)})

	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.LastActionType != string(eventEditedDesc) {
		t.Fatalf("LastActionType = %q, want %q — the fixture does not reach the alert branch", got.LastActionType, eventEditedDesc)
	}
	if got.MaintainerAlertNeeded {
		t.Error("MaintainerAlertNeeded = true: the bot already alerted about this edit, and a stranger's comment edit re-armed it")
	}
}

// A genuinely NEW description edit after the last alert must still alert.
func TestAlertIsReArmedByAFreshDescriptionEdit(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	raw.Comments = commentNodes(rawComment{
		Author:    actor(testSelf),
		Body:      botAlertSignature + ". Maintainers, please review.",
		CreatedAt: daysAgo(19),
	})
	raw.UserContentEdits = editNodes(
		rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(20)},
		rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(2)}, // after the alert
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if !got.MaintainerAlertNeeded {
		t.Error("MaintainerAlertNeeded = false: a description edit after the last alert must be reported")
	}
}

// Per-session issue scoping is the primary prompt-injection defence, and only
// get_issue_state had a test that fails when its check is deleted: for the
// others the call fell through to a different refusal, which the assertions
// could not tell apart. Every tool now has to name the scoping in its refusal.
func TestEveryToolRefusesAnOutOfScopeIssueByName(t *testing.T) {
	calls := 0
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	// An observation that would satisfy every predicate, recorded for the
	// OUT-OF-SCOPE issue, so nothing but the scoping check can refuse.
	st := staleReady()
	st.IsStale = true
	st.MaintainerAlertNeeded = true
	st.DaysSinceStaleLabel = 30
	st.DaysSinceActivity = 20
	st.DaysSinceLastActorAction = 20
	c.recordObservation(8, st)
	ctx := withAuditedIssue(context.Background(), 7)

	check := func(name, status, msg string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s returned a Go error: %v", name, err)
		}
		if status != "error" {
			t.Errorf("%s on an out-of-scope issue = %q, want a refusal", name, status)
		}
		if !strings.Contains(msg, "session is scoped to issue #7") {
			t.Errorf("%s refusal = %q, want it to name the session scoping; any other refusal means the scoping check did not fire", name, msg)
		}
	}

	stRes, err := c.doGetIssueState(ctx, 8)
	check("get_issue_state", stRes.Status, stRes.Error, err)

	for _, tc := range []struct {
		name string
		call func() (actionResult, error)
	}{
		{"add_label_to_issue", func() (actionResult, error) { return c.doAddLabel(ctx, 8, c.cfg.RequestClarificationLabel) }},
		{"remove_label_from_issue", func() (actionResult, error) { return c.doRemoveLabel(ctx, 8, c.cfg.StaleLabel) }},
		{"add_stale_label_and_comment", func() (actionResult, error) { return c.doMarkStale(ctx, 8) }},
		{"alert_maintainer_of_edit", func() (actionResult, error) { return c.doAlertEdit(ctx, 8) }},
		{"close_as_stale", func() (actionResult, error) { return c.doClose(ctx, 8) }},
	} {
		res, err := tc.call()
		check(tc.name, res.Status, res.Message, err)
	}

	if calls != 0 {
		t.Errorf("%d HTTP calls were made for out-of-scope issues, want 0", calls)
	}
}

// computeIssueState has to wire the actor clock through to the field the
// predicates read. Every other test installs the field by hand on a literal, so
// setting it from days_since_activity would have gone unnoticed.
func TestComputeIssueStateWiresTheActorClock(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	raw.Comments = commentNodes(
		rawComment{Author: actor(testAuthor), Body: "here", CreatedAt: daysAgo(30)},
		rawComment{
			Author: actor("passer-by"), Body: "ping", CreatedAt: daysAgo(35),
			LastEditedAt: func() *time.Time { t := daysAgo(1); return &t }(),
		},
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.DaysSinceActivity > 2 {
		t.Errorf("DaysSinceActivity = %.1f, want ~1 (the edit is activity)", got.DaysSinceActivity)
	}
	if got.DaysSinceLastActorAction < 29 || got.DaysSinceLastActorAction > 31 {
		t.Errorf("DaysSinceLastActorAction = %.1f, want ~30 (the author's own comment), not the stranger's edit", got.DaysSinceLastActorAction)
	}
}

// buildTimeline concatenates comments, then description edits, then timeline
// items, so the history is only chronological because it is sorted. Without the
// sort, replay would read the last item of the last category as the last actor.
func TestTimelineIsSortedChronologically(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	// The comment is the NEWEST event but comes first in the concatenation.
	raw.Comments = commentNodes(rawComment{Author: actor(testAuthor), Body: "here it is", CreatedAt: daysAgo(1)})
	raw.TimelineItems = timelineNodes(
		rawTimelineItem{Typename: "RenamedTitleEvent", Actor: actor("maintainerA"), CreatedAt: daysAgo(20)},
	)
	events, _, _ := buildTimeline(raw, testSelf, testStaleLabel)
	for i := 1; i < len(events); i++ {
		if events[i].Time.Before(events[i-1].Time) {
			t.Fatalf("event %d (%v at %v) precedes event %d (%v at %v): the history is not sorted",
				i, events[i].Type, events[i].Time, i-1, events[i-1].Type, events[i-1].Time)
		}
	}
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.LastActionRole != string(roleAuthor) {
		t.Errorf("LastActionRole = %q, want author: their comment is the newest event", got.LastActionRole)
	}
}

// The search cutoff must be in the PAST. A sign flip puts it in the future, so
// `created:<cutoff` matches every open issue including ones opened minutes ago —
// the mass-marker outcome the threshold validation exists to prevent.
func TestSearchCutoffIsInThePast(t *testing.T) {
	var q string
	cfg := baseCfg()
	cfg.StaleAfter = 336 * time.Hour
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, `{"items":[]}`)
	}))
	before := time.Now().UTC()
	if _, err := c.SearchOldOpenIssues(context.Background()); err != nil {
		t.Fatalf("SearchOldOpenIssues: %v", err)
	}
	_, after, ok := strings.Cut(q, "created:<")
	if !ok {
		t.Fatalf("search query %q has no created:< clause", q)
	}
	cutoff, err := time.Parse("2006-01-02T15:04:05Z", after)
	if err != nil {
		t.Fatalf("cutoff %q does not parse: %v", after, err)
	}
	if !cutoff.Before(before) {
		t.Errorf("cutoff %v is not in the past: every open issue would match", cutoff)
	}
	if age := before.Sub(cutoff); age < cfg.StaleAfter-time.Minute || age > cfg.StaleAfter+time.Minute {
		t.Errorf("cutoff is %v old, want ~%v (StaleAfter)", age, cfg.StaleAfter)
	}
}

// Vertex AI via Application Default Credentials is a documented path: with it
// enabled the run must start without an API key. Nothing tested the USE of the
// flag, only its parse.
func TestVertexAIReplacesTheAPIKeyRequirement(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() with no key and no Vertex flag = nil error, want a refusal")
	}
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() with GOOGLE_GENAI_USE_VERTEXAI=true and no API key = %v, want success", err)
	}
	if !cfg.UseVertexAI {
		t.Error("UseVertexAI = false after GOOGLE_GENAI_USE_VERTEXAI=true")
	}
}

// A finite but huge hours value overflows the conversion to a Duration, and
// float-to-int overflow is implementation-defined: a target that saturates gets
// a ~292-year threshold that passes validate() and silently no-ops the sweep.
func TestLoadConfigRejectsOversizedHours(t *testing.T) {
	// The last entry is the bound itself, formatted so it round-trips exactly.
	// The compiled constant math.MaxInt64/float64(time.Hour) rounds UP past
	// 2^63/3.6e12, so the boundary value still overflows and the comparison has
	// to be >=, not >.
	bound := strconv.FormatFloat(math.MaxInt64/float64(time.Hour), 'g', 17, 64)
	for _, v := range []string{"1e7", "1e18", "-1e7", bound} {
		t.Run(v, func(t *testing.T) {
			setRequiredCreds(t)
			t.Setenv("STALE_HOURS_THRESHOLD", v)
			// Assert the SPECIFIC refusal. On amd64 the overflow lands on
			// MinInt64, which validate() rejects as non-positive, so a check on
			// the key name alone passes whether or not the guard exists.
			_, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig() with STALE_HOURS_THRESHOLD=%s = nil error, want a refusal", v)
			}
			if !strings.Contains(err.Error(), "too large to express as a duration") {
				t.Errorf("loadConfig() = %v, want the overflow named; a non-positive-duration error means the guard did not fire", err)
			}
		})
	}
}

// scopedToolContext is the minimum agent.Context a function tool needs: the
// handler uses it only as a context.Context. Every other method is left to the
// nil embedded interface, so a handler that reached for one would panic rather
// than quietly get a zero value.
type scopedToolContext struct {
	agent.Context
	ctx context.Context
}

func (s scopedToolContext) Value(k any) any             { return s.ctx.Value(k) }
func (s scopedToolContext) Deadline() (time.Time, bool) { return s.ctx.Deadline() }
func (s scopedToolContext) Done() <-chan struct{}       { return s.ctx.Done() }
func (s scopedToolContext) Err() error                  { return s.ctx.Err() }

// functiontool consults this before invoking the handler; nil means "no
// confirmation flow", which is what this bot runs with.
func (s scopedToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }

// The six registered closures were never invoked by any test: every
// authorization test called the do* method directly with a hand-built context,
// and tools() was only ever used for tl.Name(). So nothing checked that a tool
// passes its context through, or that it reads issue_number into the parameter
// the authorization check uses. Swapping two arguments in a closure would have
// been invisible.
//
// This drives the registered tools. It does not prove that ADK propagates the
// context value from runner.Run into a tool handler — no test here constructs a
// runner — only that these closures carry through whatever they are given.
func TestRegisteredToolsCarryTheSessionScope(t *testing.T) {
	calls := 0
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools(): %v", err)
	}
	if len(tools) == 0 {
		t.Fatal("no tools registered: this test would otherwise pass vacuously")
	}
	// An observation good enough for every predicate, recorded for the issue the
	// session is NOT scoped to, so only the scoping check can refuse.
	st := staleReady()
	st.IsStale = true
	st.MaintainerAlertNeeded = true
	st.DaysSinceStaleLabel = 30
	st.DaysSinceLastActorAction = 20
	st.DaysSinceActivity = 20
	c.recordObservation(8, st)
	ctx := scopedToolContext{ctx: withAuditedIssue(context.Background(), 7)}

	ran := 0
	for _, tl := range tools {
		callable, ok := tl.(interface {
			Run(agent.Context, any) (map[string]any, error)
		})
		if !ok {
			t.Fatalf("tool %q is not callable: the registered tool cannot be driven", tl.Name())
		}
		// The argument schema is strict, so only the label tools may carry one.
		args := map[string]any{"issue_number": 8}
		if tl.Name() == "add_label_to_issue" || tl.Name() == "remove_label_from_issue" {
			args["label"] = c.cfg.RequestClarificationLabel
		}
		res, err := callable.Run(ctx, args)
		if err != nil {
			t.Errorf("%s returned a Go error for an out-of-scope issue: %v", tl.Name(), err)
			continue
		}
		ran++
		blob, _ := res["status"].(string)
		msg, _ := res["message"].(string)
		errField, _ := res["error"].(string)
		if blob != "error" {
			t.Errorf("%s status = %q for issue #8 in a session scoped to #7, want an error", tl.Name(), blob)
		}
		if !strings.Contains(msg+errField, "session is scoped to issue #7") {
			t.Errorf("%s refusal = %q / %q, want it to name the session scoping", tl.Name(), msg, errField)
		}
	}
	if ran != len(tools) {
		t.Errorf("drove %d of %d registered tools", ran, len(tools))
	}
	if calls != 0 {
		t.Errorf("%d HTTP calls were made for an out-of-scope issue, want 0", calls)
	}
}

// A maintainer who reopens, retitles or un-labels an issue becomes the last
// actor without having asked for anything. Marking that stale posts a comment
// saying "after a maintainer requested clarification" about a request that was
// never made, and closes the issue a week later.
func TestMarkStaleRefusedWhenTheMaintainerAskedNothing(t *testing.T) {
	for _, action := range []eventType{eventReopened, eventRenamedTitle, eventUnlabeled} {
		t.Run(string(action), func(t *testing.T) {
			c := newTestClient(t)
			st := staleReady()
			st.LastActionType = string(action)
			st.LastCommentText = "" // no comment to attribute to this maintainer
			c.recordObservation(7, st)

			res, err := c.doMarkStale(withAuditedIssue(context.Background(), 7), 7)
			if err != nil {
				t.Fatalf("doMarkStale returned a Go error: %v", err)
			}
			if res.Status != "error" || !strings.Contains(res.Message, "no maintainer comment to judge") {
				t.Errorf("doMarkStale = %+v after a %s, want a refusal: nothing was asked", res, action)
			}
		})
	}
}

// Both the stale and the close gates read the actor clock. days_since_activity
// is advanced by a comment edit from anyone who ever commented — invisible and
// un-notified — so gating on it let a stranger keep an issue permanently out of
// the bot's reach by re-saving an old comment.
func TestAStrangersCommentEditCannotStallTheBot(t *testing.T) {
	ctx := withAuditedIssue(context.Background(), 7)

	t.Run("mark stale", func(t *testing.T) {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.DaysSinceLastActorAction = 30 // the maintainer asked 30 days ago
		st.DaysSinceActivity = 0.1       // a passer-by re-saved an old comment
		c.recordObservation(7, st)
		if res, err := c.doMarkStale(ctx, 7); err != nil || res.Status != "success" {
			t.Errorf("doMarkStale = (%+v, %v), want success: a stranger's edit is not the author answering", res, err)
		}
	})

	t.Run("close", func(t *testing.T) {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.IsStale = true
		st.DaysSinceStaleLabel = 30
		st.DaysSinceLastActorAction = 40
		st.DaysSinceAuthorAction = 45 // the author has not acted at all
		st.DaysSinceActivity = 0.1    // same edit
		c.recordObservation(7, st)
		if res, err := c.doClose(ctx, 7); err != nil || res.Status != "success" {
			t.Errorf("doClose = (%+v, %v), want success: a stranger's edit must not hold the issue open", res, err)
		}
	})
}

// The alert also fires when someone other than the author edits the
// description, so the comment must not say the author did it.
func TestAlertBodyDoesNotAttributeTheEditToTheAuthor(t *testing.T) {
	var posted string
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments") {
			var body struct {
				Body string `json:"body"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode comment body: %v", err)
			}
			posted = body.Body
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	st := staleReady()
	st.MaintainerAlertNeeded = true
	c.recordObservation(7, st)
	if res, err := c.doAlertEdit(withAuditedIssue(context.Background(), 7), 7); err != nil || res.Status != "success" {
		t.Fatalf("doAlertEdit = (%+v, %v), want success", res, err)
	}
	// The whole emitted body, not just the constant: a suffix naming the author
	// would otherwise slip past.
	if strings.Contains(strings.ToLower(posted), "the author") {
		t.Errorf("alert body = %q, but the alert also fires for a non-author editor; it must not say the author did it", posted)
	}
	if posted == "" {
		t.Error("no comment was posted, so this test asserted nothing")
	}
}

// The bot recognizes its own alert by PREFIX, as doAlertEdit writes it. Matching
// anywhere let another [bot] account quoting the alert suppress the next genuine
// one, and let a quoted alert be mistaken for the bot's own.
func TestBotAlertRecognitionRequiresThePrefix(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	raw.Comments = commentNodes(
		// Another bot summarising the thread, quoting our alert mid-comment.
		rawComment{
			Author:    &rawActor{Login: "summary-bot", Typename: "Bot"},
			Body:      "Recent activity: " + botAlertSignature + " — see above.",
			CreatedAt: daysAgo(19),
		},
	)
	raw.UserContentEdits = editNodes(rawEdit{Editor: actor(testAuthor), EditedAt: daysAgo(20)})

	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if !got.MaintainerAlertNeeded {
		t.Error("MaintainerAlertNeeded = false: another bot quoting the signature suppressed a genuine alert")
	}
}

// An unbounded comment body is free model spend for anyone who can comment: it
// is replayed into every turn of the issue's session, on every daily run.
func TestFencedTextIsCapped(t *testing.T) {
	huge := strings.Repeat("é", 60000) // multibyte, so a byte cut would corrupt it
	got := fenceUntrusted(huge, "abcd")
	if len([]rune(got)) > maxFencedRunes+64 {
		t.Errorf("fenced text is %d runes, want it capped near %d", len([]rune(got)), maxFencedRunes)
	}
	// The marker is nonce-bound: a fixed literal is a string anyone could write
	// into their own comment to fake an elision.
	if !strings.Contains(got, "[truncated:abcd]") {
		t.Errorf("a truncated body must say so under the issue's own marker, or the model reads a sentence that stops mid-thought as the whole comment: %q", got[:80])
	}
	if strings.Contains(got, "[truncated]") {
		t.Error("the truncation marker must not be a fixed literal: anyone can write that into a comment")
	}
	if !strings.HasPrefix(got, "[UNTRUSTED:abcd]") || !strings.HasSuffix(got, "[/UNTRUSTED:abcd]") {
		t.Errorf("truncation broke the fence: %q…%q", got[:40], got[len(got)-40:])
	}
	short := "please share a repro"
	if fenceUntrusted(short, "abcd") != "[UNTRUSTED:abcd]\n"+short+"\n[/UNTRUSTED:abcd]" {
		t.Error("a short body must pass through untouched")
	}
}

// An author who answers by EDITING their own comment has answered. The edit does
// not change whose turn it is, so it leaves the maintainer as the last actor and
// slips past the actor clock — and the bot would then mark the issue stale, or
// close it, days after the logs it asked for arrived.
func TestAnAuthorsCommentEditCountsAsAnswering(t *testing.T) {
	ctx := withAuditedIssue(context.Background(), 7)

	t.Run("blocks mark stale", func(t *testing.T) {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.DaysSinceLastActorAction = 30 // the maintainer asked 30 days ago
		st.DaysSinceAuthorAction = 2     // the author edited their comment 2 days ago
		c.recordObservation(7, st)
		res, err := c.doMarkStale(ctx, 7)
		if err != nil {
			t.Fatalf("doMarkStale returned a Go error: %v", err)
		}
		if res.Status != "error" || !strings.Contains(res.Message, "maintainers' court") {
			t.Errorf("doMarkStale = %+v, want a refusal: the author answered by editing", res)
		}
	})

	t.Run("blocks close", func(t *testing.T) {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.IsStale = true
		st.DaysSinceStaleLabel = 30
		st.DaysSinceLastActorAction = 40
		st.DaysSinceAuthorAction = 2 // they answered, whatever it did to the turn order
		c.recordObservation(7, st)
		res, err := c.doClose(ctx, 7)
		if err != nil {
			t.Fatalf("doClose returned a Go error: %v", err)
		}
		if res.Status != "error" || !strings.Contains(res.Message, "not waiting on them") {
			t.Errorf("doClose = %+v, want a refusal: the author answered", res)
		}
	})
}

// replay must record the author's own actions, edits included, without letting
// an edit change whose turn it is.
func TestReplayTracksTheAuthorsOwnClock(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "author", Time: t0, Body: "broken"},
		{Type: eventCommented, Actor: "maint", Time: t0.Add(24 * time.Hour), Body: "repro?"},
		// The author answers by editing their original comment.
		{Type: eventEditedComment, Actor: "Author", Time: t0.Add(240 * time.Hour), Body: "broken — logs attached"},
	}, toSet([]string{"maint"}), "author")

	if st.LastActorRole != roleMaintainer {
		t.Errorf("LastActorRole = %v, want maintainer: an edit does not change whose turn it is", st.LastActorRole)
	}
	if !st.LastActorAction.Equal(t0.Add(24 * time.Hour)) {
		t.Errorf("LastActorAction = %v, want the maintainer's comment", st.LastActorAction)
	}
	if !st.LastAuthorAction.Equal(t0.Add(240 * time.Hour)) {
		t.Errorf("LastAuthorAction = %v, want the edit time: the author acted, whatever it did to the turn order", st.LastAuthorAction)
	}
}

// A stranger's edit is not the author acting, so it must not hold the bot off.
func TestAStrangersEditDoesNotAdvanceTheAuthorClock(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	st := replay([]historyEvent{
		{Type: eventCommented, Actor: "author", Time: t0, Body: "broken"},
		{Type: eventCommented, Actor: "maint", Time: t0.Add(24 * time.Hour), Body: "repro?"},
		{Type: eventEditedComment, Actor: "passer-by", Time: t0.Add(240 * time.Hour), Body: "typo"},
	}, toSet([]string{"maint"}), "author")

	if !st.LastAuthorAction.Equal(t0) {
		t.Errorf("LastAuthorAction = %v, want the author's own comment at %v", st.LastAuthorAction, t0)
	}
}

// One maintainer asks for a repro and a second retitles the issue. Requiring the
// SAME login to have written the last comment wiped the request, and
// stalePredicate refuses without a comment to judge — so ordinary two-maintainer
// triage could never go stale.
func TestLastCommentSurvivesASecondMaintainersAction(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(60)}
	raw.Comments = commentNodes(
		rawComment{Author: actor("maintainerA"), Body: "Could you share a minimal repro?", CreatedAt: daysAgo(30)},
	)
	raw.TimelineItems = timelineNodes(
		rawTimelineItem{Typename: "RenamedTitleEvent", Actor: actor("maintainerB"), CreatedAt: daysAgo(25)},
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)

	if got.LastActionRole != string(roleMaintainer) {
		t.Fatalf("LastActionRole = %q, want maintainer", got.LastActionRole)
	}
	if got.LastCommentText != "Could you share a minimal repro?" {
		t.Errorf("LastCommentText = %q, want maintainerA's request retained: maintainerB holds the same role", got.LastCommentText)
	}
}

// The blanking rule still has to fire when the roles differ: the author's own
// question must never be handed to the maintainer-intent step as a maintainer's.
func TestLastCommentIsBlankedWhenTheRolesDiffer(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(60)}
	raw.Comments = commentNodes(
		rawComment{Author: actor(testAuthor), Body: "Could you please help me?", CreatedAt: daysAgo(30)},
	)
	raw.TimelineItems = timelineNodes(
		rawTimelineItem{Typename: "RenamedTitleEvent", Actor: actor("maintainerA"), CreatedAt: daysAgo(25)},
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.LastCommentText != "" {
		t.Errorf("LastCommentText = %q, want empty: the author's words must not be read as the maintainer's", got.LastCommentText)
	}
}

// GitHub's "Quote reply" pastes the previous comment as a blockquote and leaves
// the cursor underneath, so a maintainer's actual question is routinely the LAST
// thing in a long comment. Head-only truncation dropped it, and an author could
// immunize their issue by posting a description long enough that every quoted
// reply overflowed.
func TestFenceTruncationKeepsTheEndOfTheComment(t *testing.T) {
	question := "Can you reproduce this on v2.2.0?"
	body := strings.Repeat("quoted log line\n", 1000) + question
	got := fenceUntrusted(body, "abcd")

	if !strings.Contains(got, question) {
		t.Error("the maintainer's question was truncated away: it is at the END of a quoted reply, which is where GitHub puts it")
	}
	if !strings.Contains(got, "[truncated:abcd]") {
		t.Error("a truncated body must say so under the issue's own marker")
	}
	if len([]rune(got)) > maxFencedRunes+64 {
		t.Errorf("fenced text is %d runes, want it capped near %d", len([]rune(got)), maxFencedRunes)
	}
	if !strings.HasPrefix(got, "[UNTRUSTED:abcd]") || !strings.HasSuffix(got, "[/UNTRUSTED:abcd]") {
		t.Error("truncation broke the fence")
	}
}

// The author clock answers an ORDERING question, not a threshold one. Asking
// whether the author has been quiet for StaleThresholdDays marks an issue stale
// once their ANSWER ages past the threshold — a wrong write on an issue whose
// ball is in the maintainers' court.
func TestAnAnsweredIssueStaysAnsweredHoweverOldTheAnswerGets(t *testing.T) {
	ctx := withAuditedIssue(context.Background(), 7)
	// The maintainer asked, the author answered by editing one day later, and
	// nobody has spoken since. Walk that forward past every threshold.
	for _, age := range []float64{16, 40, 120, 400} {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.DaysSinceLastActorAction = age  // the maintainer's request
		st.DaysSinceAuthorAction = age - 1 // the author's answer, one day later
		c.recordObservation(7, st)

		res, err := c.doMarkStale(ctx, 7)
		if err != nil {
			t.Fatalf("doMarkStale returned a Go error: %v", err)
		}
		if res.Status != "error" || !strings.Contains(res.Message, "maintainers' court") {
			t.Errorf("request %.0f days old, answered %.0f days ago: doMarkStale = %+v, want a refusal", age, age-1, res)
		}
	}

	// The same for the close. An answer given before the label was applied, and
	// long past the close threshold, is still an answer: the ball is with the
	// maintainers, so the issue is not abandoned and must not be closed.
	for _, age := range []float64{20, 40, 400} {
		c := newTestClient(t)
		c.cfg.DryRun = true
		st := staleReady()
		st.IsStale = true
		st.LastActionRole = string(roleMaintainer)
		st.DaysSinceStaleLabel = age / 2   // well past the 7-day close threshold
		st.DaysSinceLastActorAction = age  // the maintainer's request
		st.DaysSinceAuthorAction = age - 1 // answered a day later
		c.recordObservation(7, st)

		res, err := c.doClose(ctx, 7)
		if err != nil {
			t.Fatalf("doClose returned a Go error: %v", err)
		}
		if res.Status != "error" || !strings.Contains(res.Message, "not waiting on them") {
			t.Errorf("request %.0f days old, answered %.0f days ago: doClose = %+v, want a refusal", age, age-1, res)
		}
	}
}

// The mirror: a fresh maintainer action puts the ball back in the author's court
// and clears the refusal, so this is not a permanent dead end.
func TestAFreshMaintainerRequestReopensTheStalePath(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.DaysSinceAuthorAction = 100   // the author answered long ago
	st.DaysSinceLastActorAction = 30 // and a maintainer has since asked again
	c.recordObservation(7, st)

	res, err := c.doMarkStale(withAuditedIssue(context.Background(), 7), 7)
	if err != nil || res.Status != "success" {
		t.Fatalf("doMarkStale = (%+v, %v), want success: the maintainer asked after the author last spoke", res, err)
	}
}

// An author answering by editing their own comment leaves the maintainer as the
// last actor, so the role-based removal branch refuses and the close refuses too
// — the issue would keep a stale label nobody can clear. Their action after the
// label counts as coming back regardless of the turn order.
func TestAuthorEditAfterTheLabelClearsIt(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer) // the edit did not change the turn
	st.DaysSinceStaleLabel = 10
	st.DaysSinceLastActorAction = 30
	st.DaysSinceAuthorAction = 2 // they answered after the label went on
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success: the author answered after the label", res, err)
	}
}

// But an author who has NOT acted since the label still leaves it in place.
func TestAuthorSilentSinceTheLabelKeepsIt(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	st.DaysSinceStaleLabel = 10
	st.DaysSinceLastActorAction = 30
	st.DaysSinceAuthorAction = 40 // silent since well before the label
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	// Name the refusal. Asserting only that it errored let the ROLE branch stand
	// in for the ordering branch this test is about, so deleting the ordering
	// check kept it green.
	if res.Status != "error" || !strings.Contains(res.Message, "still waiting on the author") {
		t.Errorf("doRemoveLabel = %+v, want the maintainer-turn refusal", res)
	}

	// And the ordering branch on its own, with the role out of the way.
	c2 := newTestClient(t)
	c2.cfg.DryRun = true
	st2 := staleReady()
	st2.IsStale = true
	st2.LastActionRole = string(roleOther)
	st2.DaysSinceStaleLabel = 10
	st2.DaysSinceLastActorAction = 30 // the stranger spoke before the label
	st2.DaysSinceAuthorAction = 40
	c2.recordObservation(7, st2)
	res2, err := c2.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c2.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res2.Status != "error" || !strings.Contains(res2.Message, "nobody has come back") {
		t.Errorf("doRemoveLabel = %+v, want the ordering refusal", res2)
	}
}

// computeIssueState must wire the author clock through from replay. Every other
// test hand-builds the field on a literal, so reading it off the wrong
// replayResult member compiles and passes.
func TestComputeIssueStateWiresTheAuthorClock(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(60)}
	edited := daysAgo(3)
	raw.Comments = commentNodes(
		// The author's own comment, edited three days ago: their answer.
		rawComment{Author: actor(testAuthor), Body: "logs attached", CreatedAt: daysAgo(50), LastEditedAt: &edited},
		// A maintainer asked twenty days ago, so they are still the last actor.
		rawComment{Author: actor("maintainerA"), Body: "any repro?", CreatedAt: daysAgo(20)},
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)

	if got.LastActionRole != string(roleMaintainer) {
		t.Fatalf("LastActionRole = %q, want maintainer: an edit does not change the turn", got.LastActionRole)
	}
	if got.DaysSinceLastActorAction < 19 || got.DaysSinceLastActorAction > 21 {
		t.Errorf("DaysSinceLastActorAction = %.1f, want ~20 (the maintainer's comment)", got.DaysSinceLastActorAction)
	}
	if got.DaysSinceAuthorAction < 2 || got.DaysSinceAuthorAction > 4 {
		t.Errorf("DaysSinceAuthorAction = %.1f, want ~3 (the author's edit), not the actor clock", got.DaysSinceAuthorAction)
	}
}

// The truncation marker carries the per-issue nonce, and the tail resumes at a
// line boundary so a quoted block keeps the "> " that makes it a quote.
func TestFenceTruncationIsNonceBoundAndLineAligned(t *testing.T) {
	quoted := strings.Repeat("> quoted attacker text\n", 400)
	body := quoted + "> the payload line\nAnd the maintainer's real question?"
	got := fenceUntrusted(body, "cafe1234")

	if !strings.Contains(got, "[truncated:cafe1234]") {
		t.Error("the truncation marker must carry the issue's nonce: a fixed literal is a string anyone can write")
	}
	if len([]rune(got)) > maxFencedRunes+64 {
		t.Errorf("fenced text is %d runes, want it capped near %d", len([]rune(got)), maxFencedRunes)
	}
	_, after, ok := strings.Cut(got, "[truncated:cafe1234]\n")
	if !ok {
		t.Fatal("could not find the marker")
	}
	// Whatever follows the marker must begin a line as it did in the original.
	first, _, _ := strings.Cut(after, "\n")
	if first != "" && !strings.HasPrefix(first, "> ") && !strings.HasPrefix(first, "And ") {
		t.Errorf("the tail resumes mid-line at %q: a cut through a quoted block strips the \"> \" and the attacker's words arrive looking unquoted", first)
	}
}

// Truncation must not be reachable for an ordinary comment.
func TestOrdinaryCommentsAreNotTruncated(t *testing.T) {
	body := strings.Repeat("a normal paragraph of prose.\n", 50)
	got := fenceUntrusted(body, "abcd")
	if strings.Contains(got, "[truncated") {
		t.Errorf("a %d-rune comment was truncated; the cap is %d", len([]rune(body)), maxFencedRunes)
	}
}

// replay's author clock must survive the shapes state.go's filters can produce.
func TestAuthorClockEdgeCases(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	t.Run("author is also a maintainer", func(t *testing.T) {
		st := replay([]historyEvent{
			{Type: eventCreated, Actor: "dev", Time: t0},
			{Type: eventCommented, Actor: "other", Time: t0.Add(time.Hour), Body: "hi"},
		}, toSet([]string{"dev"}), "dev")
		if !st.LastAuthorAction.Equal(t0) {
			t.Errorf("LastAuthorAction = %v, want the creation event", st.LastAuthorAction)
		}
		if st.LastActorRole != roleOther {
			t.Errorf("LastActorRole = %v, want other_user", st.LastActorRole)
		}
	})

	t.Run("issue with only its creation event", func(t *testing.T) {
		st := replay([]historyEvent{{Type: eventCreated, Actor: "author", Time: t0}}, nil, "author")
		if !st.LastAuthorAction.Equal(t0) {
			t.Errorf("LastAuthorAction = %v, want the creation time — a zero value would read as a 2000-year-old action", st.LastAuthorAction)
		}
	})

	t.Run("empty author matches only the creation event", func(t *testing.T) {
		st := replay([]historyEvent{
			{Type: eventCreated, Actor: "", Time: t0},
			{Type: eventCommented, Actor: "someone", Time: t0.Add(time.Hour), Body: "hi"},
		}, nil, "")
		if !st.LastAuthorAction.Equal(t0) {
			t.Errorf("LastAuthorAction = %v, want the creation event: an empty author must not match a named actor", st.LastAuthorAction)
		}
	})
}

// The author's escape from the role check must also require that nobody has
// spoken since. Without that, a maintainer who re-affirms the label after the
// author's edit — "still need the repro, leaving this stale" — is overridden by
// a stranger anyway, because the escape looked only at the label.
func TestAuthorCannotOverrideAMaintainerWhoReaffirmedTheLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	st.DaysSinceStaleLabel = 6      // labelled six days ago
	st.DaysSinceAuthorAction = 4    // the author edited a comment four days ago
	st.DaysSinceLastActorAction = 1 // and a maintainer re-affirmed yesterday
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res.Status != "error" {
		t.Errorf("doRemoveLabel = %+v, want a refusal: a maintainer spoke after the author's edit", res)
	}
}

// The comment bodies the bot posts carry the two thresholds, and nothing read
// them: with the test config's two thresholds equal, swapping the arguments read
// the same either way, so the bot could tell authors "stale after 7 days … will
// be closed within 14".
func TestPostedCommentBodiesCarryTheRightThresholds(t *testing.T) {
	capture := func(t *testing.T, run func(*GitHubClient, context.Context) error) []string {
		t.Helper()
		var bodies []string
		cfg := baseCfg() // StaleAfter 14d, CloseAfter 7d — deliberately different
		c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments") {
				var body struct {
					Body string `json:"body"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode comment body: %v", err)
				}
				bodies = append(bodies, body.Body)
				_, _ = io.WriteString(w, `{}`)
				return
			}
			if strings.HasSuffix(r.URL.Path, "/labels") {
				_, _ = io.WriteString(w, `[]`)
				return
			}
			_, _ = io.WriteString(w, `{}`)
		}))
		if err := run(c, withAuditedIssue(context.Background(), 7)); err != nil {
			t.Fatalf("run: %v", err)
		}
		return bodies
	}

	t.Run("stale warning", func(t *testing.T) {
		bodies := capture(t, func(c *GitHubClient, ctx context.Context) error {
			c.recordObservation(7, staleReady())
			res, err := c.doMarkStale(ctx, 7)
			if err == nil && res.Status != "success" {
				t.Fatalf("doMarkStale = %+v, want success", res)
			}
			return err
		})
		if len(bodies) != 1 {
			t.Fatalf("posted %d comments, want 1", len(bodies))
		}
		// The property is the ORDER, not the prose: the stale threshold is how
		// long they have already been silent and the close threshold is how long
		// they have left, so swapping the two arguments inverts the meaning while
		// leaving a sentence that reads perfectly well. Matched by position
		// rather than by phrase so the wording can be improved without this
		// test having to be rewritten — which is how it would come to be deleted.
		elapsed := strings.Index(bodies[0], "14 days")
		grace := strings.Index(bodies[0], "7 days")
		if elapsed < 0 {
			t.Errorf("stale warning = %q, want the STALE threshold (14 days) stated as the period already elapsed", bodies[0])
		}
		if grace < 0 {
			t.Errorf("stale warning = %q, want the CLOSE threshold (7 days) stated as the grace period", bodies[0])
		}
		if elapsed >= 0 && grace >= 0 && elapsed > grace {
			t.Errorf("stale warning = %q: the thresholds are the wrong way round, so it tells the author they have already waited the grace period and have the stale period left", bodies[0])
		}
	})

	t.Run("closing notice", func(t *testing.T) {
		bodies := capture(t, func(c *GitHubClient, ctx context.Context) error {
			st := staleReady()
			st.IsStale = true
			st.DaysSinceStaleLabel = 30
			st.DaysSinceLastActorAction = 40
			st.DaysSinceAuthorAction = 50
			c.recordObservation(7, st)
			res, err := c.doClose(ctx, 7)
			if err == nil && res.Status != "success" {
				t.Fatalf("doClose = %+v, want success", res)
			}
			return err
		})
		if len(bodies) != 1 {
			t.Fatalf("posted %d comments, want 1", len(bodies))
		}
		if !strings.Contains(bodies[0], "7 days") {
			t.Errorf("closing notice = %q, want the CLOSE threshold (7 days)", bodies[0])
		}
	})
}

// A rate limit on the alert was tried and reverted. It let the author arm the
// suppression: a throwaway edit draws the alert, the real edit lands inside the
// window and is never reported, and one comment afterwards moves
// last_action_type off the description edit so the branch can never be
// re-entered. This pins the de-duplication that replaced it — per EDIT, so an
// already-reported edit stays quiet and a genuinely new one is always reported.
func TestSilentEditAlertsAreDeduplicatedPerEditNotRateLimited(t *testing.T) {
	alertAt, editAt := daysAgo(1), daysAgo(2)
	build := func(alert, edit time.Time) *rawIssue {
		raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(60)}
		raw.Comments = commentNodes(rawComment{
			Author: actor(testSelf), Body: botAlertSignature + ". Maintainers, please review.",
			CreatedAt: alert,
		})
		raw.UserContentEdits = editNodes(rawEdit{Editor: actor(testAuthor), EditedAt: edit})
		return raw
	}

	// Already reported: the alert postdates the edit.
	got := computeIssueState(build(alertAt, editAt), testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.LastActionType != string(eventEditedDesc) {
		t.Fatalf("LastActionType = %q, want %q — the fixture misses the alert branch", got.LastActionType, eventEditedDesc)
	}
	if got.MaintainerAlertNeeded {
		t.Error("MaintainerAlertNeeded = true for an edit the bot already reported")
	}

	// A NEW edit, one day after that alert, must be reported — even though the
	// alert is recent. A window-based limit swallowed exactly this case, and an
	// author could arm it with a decoy edit.
	got = computeIssueState(build(alertAt, daysAgo(0.5)), testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if !got.MaintainerAlertNeeded {
		t.Error("MaintainerAlertNeeded = false for an edit made AFTER the last alert: a decoy edit would hide the real one")
	}
}

// A tail with no line boundary in it cannot be realigned, so it must be dropped
// rather than emitted as a fresh unprefixed line — which is exactly the quoted
// text arriving as the maintainer's own prose that the realignment prevents.
func TestFenceTruncationDropsATailItCannotRealign(t *testing.T) {
	// One unbroken line, longer than the cap.
	body := "> " + strings.Repeat("x", maxFencedRunes*2)
	got := fenceUntrusted(body, "abcd")

	if !strings.Contains(got, "[truncated:abcd]") {
		t.Fatal("expected truncation")
	}
	_, after, _ := strings.Cut(got, "[truncated:abcd]\n")
	tail := strings.TrimSuffix(after, "\n[/UNTRUSTED:abcd]")
	if tail != "" {
		t.Errorf("tail = %q, want it dropped: there was no line boundary to resume at, so keeping it strips the quote marker", tail)
	}
	if len([]rune(got)) > maxFencedRunes+64 {
		t.Errorf("fenced text is %d runes, want it capped near %d", len([]rune(got)), maxFencedRunes)
	}
}

// A cap that keeps too little is as bad as one that keeps too much: the
// maintainer's question needs enough context around it to be judged.
func TestFenceKeepsMostOfTheBudget(t *testing.T) {
	body := strings.Repeat("line of text\n", 2000)
	got := fenceUntrusted(body, "abcd")
	if n := len([]rune(got)); n < maxFencedRunes*3/4 {
		t.Errorf("fenced text is %d runes out of a %d budget: truncation is throwing away context it was given room for", n, maxFencedRunes)
	}
}

// Concurrency below 1 makes errgroup's first Go block forever on a bare channel
// send that no context can interrupt, so the run hangs until the workflow kills
// it. validate() is the only thing standing between a typo and that hang.
func TestValidateCoercesNonPositiveConcurrency(t *testing.T) {
	for _, in := range []int{0, -1, -100} {
		cfg := &Config{
			GitHubToken: "t", GeminiAPIKey: "k", Owner: "o", Repo: "r",
			StaleAfter: 336, CloseAfter: 168, IssueTimeout: 1, RunBudget: 1,
			MaxIssues: 100, MaxDestructiveActions: 20,
			Concurrency: in,
		}
		if err := cfg.validate(); err != nil {
			t.Fatalf("validate() with Concurrency=%d = %v", in, err)
		}
		if cfg.Concurrency != 1 {
			t.Errorf("Concurrency=%d was left at %d, want 1: errgroup.SetLimit(0) blocks forever", in, cfg.Concurrency)
		}
	}
}

// GOOGLE_API_KEY is a documented alternative to GEMINI_API_KEY.
func TestGoogleAPIKeyIsAcceptedAsTheModelCredential(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "from-google-api-key")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig with only GOOGLE_API_KEY = %v, want success", err)
	}
	if cfg.GeminiAPIKey != "from-google-api-key" {
		t.Errorf("GeminiAPIKey = %q, want the GOOGLE_API_KEY value", cfg.GeminiAPIKey)
	}
}

// A local .env must not reach a run under Actions: it would override the
// thresholds and label names the workflow deliberately leaves unset, inside a
// job that holds issues: write.
func TestLocalEnvFileIsIgnoredUnderActions(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(dir+"/.env", []byte("STALE_LABEL_NAME=from-dotenv\n"), 0o600); err != nil {
		t.Fatalf("write .env: %v", err)
	}
	setRequiredCreds(t)
	t.Setenv("GITHUB_ACTIONS", "true")

	var got *Config
	if err := runWith(context.Background(), discardLogger(), nil,
		func(_ context.Context, cfg *Config, _ *slog.Logger) error { got = cfg; return nil }); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if got == nil {
		t.Fatal("the audit step never ran")
	}
	if got.StaleLabel == "from-dotenv" {
		t.Error("a .env in the working directory reached a run under GITHUB_ACTIONS")
	}

	// And it IS loaded when not under Actions, or the guard would be untestable
	// by being unconditionally true. Unset rather than blank: godotenv does not
	// overwrite a variable that is already present, even as an empty string.
	t.Setenv("GITHUB_ACTIONS", "")
	if err := os.Unsetenv("STALE_LABEL_NAME"); err != nil {
		t.Fatalf("unset: %v", err)
	}
	got = nil
	if err := runWith(context.Background(), discardLogger(), nil,
		func(_ context.Context, cfg *Config, _ *slog.Logger) error { got = cfg; return nil }); err != nil {
		t.Fatalf("runWith (local): %v", err)
	}
	if got.StaleLabel != "from-dotenv" {
		t.Errorf("StaleLabel = %q locally, want the .env value: the guard is unconditional", got.StaleLabel)
	}
}

// A BOT_LOGIN naming a real participant does not mislabel one comment: the
// ignore filter drops every event by that login, so the person disappears from
// the history. Pointed at a maintainer it erases the request the decision tree
// turns on.
func TestValidateRejectsABotLoginThatNamesAMaintainer(t *testing.T) {
	base := func() *Config {
		return &Config{
			GitHubToken: "t", GeminiAPIKey: "k", Owner: "o", Repo: "r",
			StaleAfter: 336, CloseAfter: 168, IssueTimeout: 1, RunBudget: 1, Concurrency: 1,
			MaxIssues: 100, MaxDestructiveActions: 20,
			StaleLabel: "stale", RequestClarificationLabel: "request clarification",
			Maintainers: []string{"maintainerA", "MaintainerB"},
		}
	}
	for _, login := range []string{"maintainerA", "MAINTAINERA", "maintainerb"} {
		cfg := base()
		cfg.BotLogin = login
		err := cfg.validate()
		if err == nil {
			t.Errorf("validate() with BOT_LOGIN=%q = nil, want a refusal", login)
			continue
		}
		if !strings.Contains(err.Error(), "BOT_LOGIN") {
			t.Errorf("validate() = %v, want it to name BOT_LOGIN", err)
		}
	}
	// The ordinary value still passes.
	cfg := base()
	cfg.BotLogin = "github-actions[bot]"
	if err := cfg.validate(); err != nil {
		t.Errorf("validate() with the expected bot login = %v, want success", err)
	}
}

// The bot recognizes its own login case-insensitively, like every other login
// comparison. A byte-exact match let it read its own stale warning as somebody
// else's activity and clear the label it had just applied.
func TestSelfRecognitionFoldsCase(t *testing.T) {
	if !isIgnoredActor("Adk-Stale-Bot", "adk-stale-bot") {
		t.Error("isIgnoredActor did not recognize its own login in a different casing")
	}
	if !isSelfActor("Adk-Stale-Bot", "adk-stale-bot") {
		t.Error("isSelfActor did not recognize its own login in a different casing")
	}
	if isIgnoredActor("someone-else", "adk-stale-bot") {
		t.Error("isIgnoredActor dropped an unrelated human")
	}
}

// The stale label and its warning comment go on together or not at all. Leaving
// the label without the warning is invisible afterwards — the next run refuses
// to re-mark an already-stale issue, so the warning is never retried, and the
// issue is closed a week later having promised the author nothing.
func TestMarkStaleTakesTheLabelBackOffIfTheCommentFails(t *testing.T) {
	var (
		mu      sync.Mutex
		deleted []string
	)
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/labels"):
			_, _ = io.WriteString(w, `[]`)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments"):
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"message":"secondary rate limit"}`)
		case r.Method == http.MethodDelete:
			i := strings.LastIndex(r.URL.Path, "/labels/")
			mu.Lock()
			deleted = append(deleted, r.URL.Path[i+len("/labels/"):])
			mu.Unlock()
			_, _ = io.WriteString(w, `[]`)
		default:
			_, _ = io.WriteString(w, `{}`)
		}
	}))

	err := c.MarkStale(context.Background(), 7, "warning")
	if err == nil {
		t.Fatal("MarkStale = nil, want the comment failure surfaced")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(deleted) != 1 || deleted[0] != c.cfg.StaleLabel {
		t.Errorf("removed %v, want the stale label taken back off so the next run retries both", deleted)
	}
	if !strings.Contains(err.Error(), "removed again") {
		t.Errorf("error = %v, want it to say the label was taken back off", err)
	}
}

// The tools slice is shared by every per-issue agent, so it must not have spare
// capacity for a consumer to append into.
func TestSharedToolSliceHasNoSpareCapacity(t *testing.T) {
	c := newTestClient(t)
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools(): %v", err)
	}
	if cap(tools) != len(tools) {
		t.Errorf("cap=%d len=%d: an append by any consumer would write one backing array from every goroutine", cap(tools), len(tools))
	}
}

// A tool failure is reported to a public Actions log, which is parsed for ::
// workflow commands. The model chooses the argument VALUES, so only the keys go
// in the line.
func TestToolErrorLogDoesNotEchoModelSuppliedValues(t *testing.T) {
	var logged strings.Builder
	log := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelError}))
	logToolError(log, "add_label_to_issue", map[string]any{
		"issue_number": 7,
		"label":        "\n::add-mask::something",
	}, errors.New("boom"))

	out := logged.String()
	if strings.Contains(out, "add-mask") {
		t.Errorf("the model's argument value reached the log: %q", out)
	}
	if !strings.Contains(out, "issue_number") || !strings.Contains(out, "label") {
		t.Errorf("the argument names are what makes the line useful, and they are missing: %q", out)
	}
	if !strings.Contains(out, "add_label_to_issue") {
		t.Errorf("the tool name is missing: %q", out)
	}
}

// STEP 1's first branch must be reachable when the label's age is unknown.
// Comparing against a negative age made every limb false, so the "take no
// action" branch swallowed a state the Go gate deliberately permits — leaving an
// issue whose author demonstrably came back labelled stale forever.
func TestPromptClearsAnUnknownAgeLabelWhenTheUserCameBack(t *testing.T) {
	out := renderPrompt(promptCfg())
	if !strings.Contains(out, "`days_since_stale_label` is NEGATIVE (the label's age is unknown, so") {
		t.Error("branch 1a does not admit a negative label age, so 1b swallows the state and removeStalePredicate's anti-stranding path is unreachable")
	}
	// And the gate really does permit it, so the two agree.
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceStaleLabel = -1
	st.DaysSinceLastActorAction = 1
	st.DaysSinceAuthorAction = 1
	c.recordObservation(7, st)
	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success: the gate permits it, so the prompt must reach it", res, err)
	}
}

// A description edit still deserves reporting when the label is stuck.
func TestPromptAlertsOnTheStuckBranches(t *testing.T) {
	out := renderPrompt(promptCfg())
	stuck := out[strings.Index(out, "1d. ANYTHING ELSE"):]
	if !strings.Contains(stuck[:strings.Index(stuck, "STEP 2")], "alert_maintainer_of_edit") {
		t.Error("branch 1d never alerts, so a silent description edit on a stuck issue is dropped although alertPredicate allows it")
	}
	unknown := out[strings.Index(out, "1b. THE LABEL'S AGE IS UNKNOWN"):]
	if !strings.Contains(unknown[:strings.Index(unknown, "1c.")], "alert_maintainer_of_edit") {
		t.Error("branch 1b never alerts, same gap")
	}
}

// The author's escape carries the same read/write race grace its sibling does:
// an edit landing between the state read and the label write predates the label
// by seconds, and without the grace the issue is stuck with nothing logged.
func TestAuthorEscapeCarriesTheLabelRaceGrace(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleMaintainer)
	st.DaysSinceStaleLabel = 10
	// A hair OLDER than the label — the edit raced the labelling write.
	st.DaysSinceAuthorAction = 10 + labelRaceGrace/2
	st.DaysSinceLastActorAction = 30
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success: the edit raced the labelling write", res, err)
	}
}

// run must route the audit through withRunBudget. Pinning withRunBudget and
// auditAll separately left this join unpinned: replacing run's body with a bare
// audit call, deleting the budget outright, kept the whole suite green.
func TestRunAppliesTheConfiguredRunBudgetToTheAudit(t *testing.T) {
	// Run from an empty directory so a developer's local .env cannot reach
	// loadConfig and change what this test observes.
	t.Chdir(t.TempDir())
	setRequiredCreds(t)
	t.Setenv("RUN_BUDGET", "50ms")

	var (
		ran          bool
		sawBudgetCtx bool
	)
	err := runWith(context.Background(), discardLogger(), nil, func(ctx context.Context, cfg *Config, _ *slog.Logger) error {
		ran = true
		dl, ok := ctx.Deadline()
		if !ok {
			return errors.New("audit received a context with no deadline: cfg.RunBudget was never applied")
		}
		if remaining := time.Until(dl); remaining > cfg.RunBudget {
			return fmt.Errorf("deadline is %s away, more than RunBudget %s: it does not derive from the config", remaining, cfg.RunBudget)
		}
		sawBudgetCtx = true
		<-ctx.Done()
		return ctx.Err()
	})

	if !ran {
		t.Fatal("runWith never called the audit step")
	}
	if !sawBudgetCtx {
		t.Fatalf("the audit step did not receive the configured budget: %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), "run budget of 50ms") {
		t.Errorf("runWith = %v, want the exhausted budget reported by name", err)
	}
}

// A malformed DRY_RUN must fail the load, not fall back to the default. The
// default is "write for real", so a fall-back turns a requested dry run into
// live labeling, commenting and closing.
func TestLoadConfigRejectsMalformedEnvValues(t *testing.T) {
	for _, tc := range []struct{ key, value string }{
		{"DRY_RUN", "yes"},
		{"STALE_HOURS_THRESHOLD", "14d"},
		{"CLOSE_HOURS_AFTER_STALE_THRESHOLD", "one week"},
		{"CONCURRENCY_LIMIT", "three"},
		{"ISSUE_TIMEOUT", "300"},
		{"RUN_BUDGET", "30"},
	} {
		t.Run(tc.key+"="+tc.value, func(t *testing.T) {
			setRequiredCreds(t)
			t.Setenv(tc.key, tc.value)
			cfg, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig() = %+v, nil error; want a refusal naming %s", cfg, tc.key)
			}
			if !strings.Contains(err.Error(), tc.key) {
				t.Errorf("loadConfig() = %v, want it to name %s", err, tc.key)
			}
		})
	}
}

// MarkStale must apply the STALE label, not some other managed one. The ordering
// test matches the POST by URL only, so swapping the argument for
// cfg.RequestClarificationLabel left the suite green — and an issue that is
// never actually labelled stale is re-marked and re-commented on every run.
func TestMarkStaleAppliesTheStaleLabel(t *testing.T) {
	var (
		mu     sync.Mutex
		labels []string
	)
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/labels") {
			var body []string
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode label body: %v", err)
			}
			mu.Lock()
			labels = append(labels, body...)
			mu.Unlock()
			_, _ = io.WriteString(w, `[]`)
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))

	if err := c.MarkStale(context.Background(), 7, "warning"); err != nil {
		t.Fatalf("MarkStale: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(labels) != 1 || labels[0] != c.cfg.StaleLabel {
		t.Errorf("MarkStale applied %v, want exactly [%q]", labels, c.cfg.StaleLabel)
	}
}

// The alert comment must start with botAlertSignature: that prefix is the only
// way buildTimeline recognizes the bot's own alerts. Change the body and
// maintainer_alert_needed stays true forever, so a fresh alert lands on every run.
func TestAlertEditPostsTheSelfRecognizableSignature(t *testing.T) {
	var (
		mu    sync.Mutex
		posts []string
	)
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/comments") {
			var body struct {
				Body string `json:"body"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode comment body: %v", err)
			}
			mu.Lock()
			posts = append(posts, body.Body)
			mu.Unlock()
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	st := staleReady()
	st.MaintainerAlertNeeded = true
	c.recordObservation(7, st)

	res, err := c.doAlertEdit(withAuditedIssue(context.Background(), 7), 7)
	if err != nil || res.Status != "success" {
		t.Fatalf("doAlertEdit = (%+v, %v), want success", res, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(posts) != 1 {
		t.Fatalf("doAlertEdit posted %d comments, want 1", len(posts))
	}
	if !strings.HasPrefix(posts[0], botAlertSignature) {
		t.Errorf("alert body = %q, want it to start with botAlertSignature so the bot recognizes its own alert next run", posts[0])
	}
}

// The fan-out must honour cfg.Concurrency. Deleting g.SetLimit left the suite
// green, which is an unbounded fan-out over the whole backlog: hundreds of
// concurrent model and GitHub calls, and secondary rate limits.
func TestAuditAllHonoursTheConcurrencyLimit(t *testing.T) {
	cfg := baseCfg()
	cfg.Concurrency = 2
	cfg.IssueTimeout = time.Minute

	var (
		mu       sync.Mutex
		inFlight int
		peak     int
	)
	issues := []int{1, 2, 3, 4, 5, 6, 7, 8}
	// Hold every slot until cfg.Concurrency callbacks have arrived, then release
	// them together. That makes the overlap a happens-before rather than a race
	// against a sleep, so the test neither flakes on a loaded runner nor passes
	// because the goroutines happened not to interleave.
	arrived := make(chan struct{}, len(issues))
	release := make(chan struct{})
	// Wait for exactly cfg.Concurrency callbacks to pile up, which proves the
	// limit is REACHED, then watch for one more. With the limit enforced no
	// further callback can start until a slot frees, so the wait times out; with
	// it removed all eight goroutines are already running and the extra arrival
	// is immediate. Joining on the watcher keeps the read of `extra` ordered.
	watcher := make(chan bool, 1)
	go func() {
		for range cfg.Concurrency {
			<-arrived
		}
		select {
		case <-arrived:
			watcher <- true
		case <-time.After(300 * time.Millisecond):
			watcher <- false
		}
		close(release)
	}()
	err := auditAll(context.Background(), cfg, discardLogger(), issues, func(context.Context, int) error {
		mu.Lock()
		inFlight++
		if inFlight > peak {
			peak = inFlight
		}
		mu.Unlock()
		arrived <- struct{}{}
		<-release
		mu.Lock()
		inFlight--
		mu.Unlock()
		return nil
	})
	extra := <-watcher
	if err != nil {
		t.Fatalf("auditAll: %v", err)
	}
	if extra {
		t.Errorf("a %dth audit started while %d were already in flight: the limit is not enforced", cfg.Concurrency+1, cfg.Concurrency)
	}
	mu.Lock()
	defer mu.Unlock()
	if peak != cfg.Concurrency {
		t.Errorf("peak concurrency was %d, want exactly %d", peak, cfg.Concurrency)
	}
}

// The search must follow every page, drop duplicates across pages, and ask only
// for open issues. All three survived as one-line mutations: the existing test
// serves a single page with no Link header, so the loop never iterates.
func TestSearchOldOpenIssuesPaginatesDedupesAndAsksForOpenIssues(t *testing.T) {
	var gotQuery string
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		gotQuery = q.Get("q")
		w.Header().Set("Content-Type", "application/json")
		if q.Get("page") == "2" {
			// Issue 3 repeats from page 1: the result set shifts between fetches.
			_, _ = io.WriteString(w, `{"items":[{"number":3},{"number":4}]}`)
			return
		}
		w.Header().Set("Link", `<`+"http://"+r.Host+`/search/issues?page=2>; rel="next", <http://`+r.Host+`/search/issues?page=2>; rel="last"`)
		_, _ = io.WriteString(w, `{"items":[{"number":1},{"number":2},{"number":3}]}`)
	}))

	got, err := c.SearchOldOpenIssues(context.Background())
	if err != nil {
		t.Fatalf("SearchOldOpenIssues: %v", err)
	}
	want := []int{1, 2, 3, 4}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v (page 2 must be fetched and the repeated number dropped)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if !strings.Contains(gotQuery, "state:open") {
		t.Errorf("search query = %q, want it restricted to state:open", gotQuery)
	}
	if !strings.Contains(gotQuery, "is:issue") {
		t.Errorf("search query = %q, want it restricted to is:issue", gotQuery)
	}
}

// days_since_stale_label must be measured from the MOST RECENT labelling. An
// issue labelled, cleared by a human, then re-marked would otherwise be dated
// from the first labelling and closed on the very next run.
func TestDaysSinceStaleLabelUsesTheMostRecentLabelling(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(60)}
	raw.Labels = labelNodes(testStaleLabel)
	raw.TimelineItems = timelineNodes(
		rawTimelineItem{Typename: "LabeledEvent", CreatedAt: daysAgo(50), Label: &struct {
			Name string `json:"name"`
		}{Name: testStaleLabel}},
		rawTimelineItem{Typename: "LabeledEvent", CreatedAt: daysAgo(2), Label: &struct {
			Name string `json:"name"`
		}{Name: testStaleLabel}},
	)
	got := computeIssueState(raw, testSelf, testMaint, testStaleLabel, testStaleAfter, testCloseAfter, testNow)
	if got.DaysSinceStaleLabel < 1 || got.DaysSinceStaleLabel > 3 {
		t.Errorf("DaysSinceStaleLabel = %.2f, want ~2 (the re-marking), not ~50 (the first labelling)", got.DaysSinceStaleLabel)
	}
}

// The fence marker is drawn per issue, not per run: a marker the model echoes
// into a log is already spent by the time the next issue is audited. Caching one
// marker on the client survived every existing test, because the only test that
// looked called newNonce directly instead of driving doGetIssueState twice.
func TestFenceMarkerDiffersBetweenIssues(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"author":{"login":"reporter"},"createdAt":"2020-01-01T00:00:00Z",
		"labels":{"nodes":[]},
		"comments":{"nodes":[{"author":{"login":"reporter"},"body":"still broken","createdAt":"2020-02-01T00:00:00Z","lastEditedAt":null}]},
		"userContentEdits":{"nodes":[]},"timelineItems":{"nodes":[]}}}}}`
	c := testClient(t, baseCfg(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))

	marker := func(number int) string {
		t.Helper()
		st, err := c.doGetIssueState(withAuditedIssue(context.Background(), number), number)
		if err != nil {
			t.Fatalf("doGetIssueState(%d): %v", number, err)
		}
		_, rest, ok := strings.Cut(st.LastCommentText, "[UNTRUSTED:")
		if !ok {
			t.Fatalf("issue #%d: last_comment_text is not fenced: %q", number, st.LastCommentText)
		}
		m, _, ok := strings.Cut(rest, "]")
		if !ok || m == "" {
			t.Fatalf("issue #%d: could not read the fence marker from %q", number, st.LastCommentText)
		}
		return m
	}

	if a, b := marker(7), marker(8); a == b {
		t.Errorf("both issues were fenced with marker %q; a marker disclosed once would then close the fence on every later issue in the run", a)
	}
}

// STEP 1 strips the stale label only when the user came back AFTER it was
// applied. Without the ordering check, a maintainer who hand-labels an issue
// whose newest event is a months-old author comment has that triage reversed on
// the next sweep.
func TestRemoveStaleRefusedWhenNobodyRespondedAfterTheLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceActivity = 60        // the author's last comment
	st.DaysSinceLastActorAction = 60 // and it was the author who acted
	st.DaysSinceStaleLabel = 1       // the maintainer labelled it yesterday
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil {
		t.Fatalf("doRemoveLabel returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "nobody has come back") {
		t.Errorf("doRemoveLabel = %+v, want a refusal: the label postdates the author's last activity", res)
	}
}

// The ordering check must not over-correct: an author who genuinely replied
// after the label still gets it cleared.
func TestRemoveStaleAllowedWhenTheAuthorRepliedAfterTheLabel(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceActivity = 1        // the author replied yesterday
	st.DaysSinceLastActorAction = 1 // and it was the author who replied
	st.DaysSinceStaleLabel = 10     // the label went on ten days ago
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success: the author came back after the label", res, err)
	}
}

// An unknown label age must not block the removal. Refusing would strand the
// issue, whereas removal posts no comment and a maintainer can redo it.
func TestRemoveStaleAllowedWhenTheLabelAgeIsUnknown(t *testing.T) {
	c := newTestClient(t)
	c.cfg.DryRun = true
	st := staleReady()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceActivity = 60
	st.DaysSinceLastActorAction = 60
	st.DaysSinceStaleLabel = -1
	c.recordObservation(7, st)

	res, err := c.doRemoveLabel(withAuditedIssue(context.Background(), 7), 7, c.cfg.StaleLabel)
	if err != nil || res.Status != "success" {
		t.Fatalf("doRemoveLabel = (%+v, %v), want success when the label age is unknown", res, err)
	}
}
