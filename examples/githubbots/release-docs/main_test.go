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
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestNewNonce(t *testing.T) {
	a, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce(): %v", err)
	}
	b, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce(): %v", err)
	}
	if len(a) != 16 {
		t.Errorf("nonce length = %d, want 16 hex chars", len(a))
	}
	if a == b {
		t.Errorf("two nonces were identical (%q); the fence would be predictable", a)
	}
}

// stubRun records which group indexes the orchestrator actually drove.
type stubRun struct {
	mu      sync.Mutex
	indexes []int
	delay   time.Duration
	// failFrom makes every group at or after this index report failure, so the
	// accounting of a failed group can be driven rather than installed.
	failFrom int
}

func (s *stubRun) fn(_ context.Context, index int) bool {
	s.mu.Lock()
	s.indexes = append(s.indexes, index)
	s.mu.Unlock()
	time.Sleep(s.delay)
	return index < s.failFrom
}

func TestAnalyzeAllDrivesEveryGroupInOrder(t *testing.T) {
	cfg := testConfig()
	cfg.RunBudget = time.Minute
	cfg.GroupTimeout = time.Minute
	groups := [][]ChangedFile{{{Path: "a"}}, {{Path: "b"}}, {{Path: "c"}}}

	s := &stubRun{failFrom: 99}
	a := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease, s.fn)
	if !a.complete() {
		t.Errorf("a clean run reported %+v, want complete", a)
	}
	if len(s.indexes) != 3 || s.indexes[0] != 0 || s.indexes[1] != 1 || s.indexes[2] != 2 {
		t.Errorf("groups driven = %v, want [0 1 2]", s.indexes)
	}
}

// A group that fails for any reason must be counted, so the issue can say its
// files are not covered instead of silently omitting them.
//
// Mutation that must fail this test: drop the `a.Failed++` in analyzeAll.
func TestAnalyzeAllCountsFailedGroups(t *testing.T) {
	cfg := testConfig()
	cfg.RunBudget, cfg.GroupTimeout = time.Minute, time.Minute
	groups := [][]ChangedFile{{{Path: "a"}}, {{Path: "b"}}, {{Path: "c"}}}

	s := &stubRun{failFrom: 1} // groups 1 and 2 fail
	a := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease, s.fn)
	if a.Failed != 2 {
		t.Errorf("Failed = %d, want 2", a.Failed)
	}
	if a.complete() {
		t.Error("a run with two failed groups reported complete coverage")
	}
	body := buildIssueBody(&ReleaseDiff{BaseTag: "v1", HeadTag: "v2"}, []Finding{{Summary: "s"}}, a)
	if !strings.Contains(body, "2 of 3 file groups failed to complete") {
		t.Errorf("the issue does not disclose the failed groups:\n%s", body)
	}
}

// An overrun must report what it missed rather than the job being killed with
// nothing to show, and it must count as a run-level error so the exit code is
// non-zero.
//
// Mutation that must fail this test: drop the budgetCtx.Err() check from the loop.
func TestAnalyzeAllStopsAndReportsWhenTheBudgetRunsOut(t *testing.T) {
	cfg := testConfig()
	cfg.RunBudget = 30 * time.Millisecond
	cfg.GroupTimeout = time.Minute
	groups := [][]ChangedFile{{{Path: "a"}}, {{Path: "b"}}, {{Path: "c"}}, {{Path: "d"}}}

	s := &stubRun{delay: 25 * time.Millisecond, failFrom: 99}
	a := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease, s.fn)
	if !a.BudgetExhausted {
		t.Fatal("analyzeAll did not report the exhausted budget")
	}
	if a.NotAttempted == 0 {
		t.Error("the groups the budget never reached were not counted")
	}
	if len(s.indexes) >= len(groups) {
		t.Errorf("drove %d of %d groups; the budget did not stop the loop", len(s.indexes), len(groups))
	}
	// The issue must say so, otherwise a partial analysis reads as a complete one.
	body := buildIssueBody(&ReleaseDiff{BaseTag: "v1", HeadTag: "v2"}, []Finding{{Summary: "s"}}, a)
	if !strings.Contains(body, "file groups were never analyzed") {
		t.Errorf("the issue body does not disclose the exhausted budget:\n%s", body)
	}
}

// The budget can also expire while the LAST group is in flight. A loop that only
// checks at the top of each iteration reports full coverage of a release it did
// not finish reading.
//
// Mutation that must fail this test: delete the post-loop
// `a.BudgetExhausted = budgetCtx.Err() != nil`.
func TestAnalyzeAllReportsABudgetThatExpiredDuringTheLastGroup(t *testing.T) {
	cfg := testConfig()
	cfg.RunBudget = 100 * time.Millisecond
	cfg.GroupTimeout = time.Minute
	groups := [][]ChangedFile{{{Path: "a"}}, {{Path: "b"}}}

	// Deterministic rather than racing the clock: group 0 returns at once, and
	// group 1 blocks until its context (a child of the budget) is done. An
	// earlier version gave each group a 30ms sleep against a 40ms budget, which
	// depends on group 0 finishing inside 10ms of slack and flakes under -race
	// on a loaded machine.
	var indexes []int
	a := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease,
		func(gctx context.Context, i int) bool {
			indexes = append(indexes, i)
			if i == len(groups)-1 {
				<-gctx.Done()
			}
			return true
		})
	s := &stubRun{indexes: indexes}
	if len(s.indexes) != 2 {
		t.Fatalf("drove %d groups, want both attempted for this case", len(s.indexes))
	}
	if a.NotAttempted != 0 {
		t.Fatalf("NotAttempted = %d, want 0 for this case", a.NotAttempted)
	}
	if !a.BudgetExhausted {
		t.Fatal("a budget that expired during the last group was reported as not exhausted")
	}
	body := buildIssueBody(&ReleaseDiff{BaseTag: "v1", HeadTag: "v2"}, []Finding{{Summary: "s"}}, a)
	if !strings.Contains(body, "run budget was exhausted") {
		t.Errorf("the issue does not disclose the cut-off last group:\n%s", body)
	}
}

// --- The idempotency guarantee ----------------------------------------------

// releaseDocsHandler serves the endpoints runWith touches and counts the calls
// that matter: whether the diff was fetched (i.e. an analysis started) and
// whether an issue was created.
type releaseDocsHandler struct {
	mu           sync.Mutex
	existingBody string // when non-empty, an issue with this body is listed
	compares     int
	creates      int
}

func (h *releaseDocsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch {
	case strings.Contains(r.URL.Path, "/compare/"):
		h.compares++
		_, _ = io.WriteString(w, `{"html_url":"u","files":[],"commits":[]}`)
	case strings.HasPrefix(r.URL.Path, "/search/"):
		_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
	case strings.HasSuffix(r.URL.Path, "/issues") && r.Method == http.MethodPost:
		h.creates++
		_, _ = io.WriteString(w, `{"number":1}`)
	case strings.HasSuffix(r.URL.Path, "/issues"):
		if h.existingBody == "" {
			_, _ = io.WriteString(w, `[]`)
			return
		}
		_, _ = io.WriteString(w, "["+issueJSON(77, "adk-bot", "Bot", h.existingBody)+"]")
	default:
		_, _ = io.WriteString(w, `{}`)
	}
}

// This drives the REAL run sequence. It is the test the "exactly one issue per
// release tag pair" guarantee rests on: with an issue already filed for the tag
// pair, the run must reach neither the diff nor a mutation, and with none filed
// it must reach the diff -- otherwise the first half would pass for a reason
// that has nothing to do with the duplicate check.
//
// Mutation that must fail this test: delete the `if found { return nil }`
// short-circuit in runWith, or make FindExistingIssue always report absence.
func TestRunWithSkipsAReleaseThatAlreadyHasAnIssue(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"

	t.Run("already filed", func(t *testing.T) {
		h := &releaseDocsHandler{existingBody: bodyMarker("v1.0.0", "v1.1.0") + "\n\nfiled last week"}
		gh := testClient(t, cfg, h)
		if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
			t.Fatalf("runWith: %v", err)
		}
		if h.compares != 0 {
			t.Errorf("fetched the diff %d time(s); a re-run must spend nothing on an analyzed release", h.compares)
		}
		if h.creates != 0 {
			t.Errorf("created %d issue(s), want 0", h.creates)
		}
	})

	t.Run("not yet filed", func(t *testing.T) {
		h := &releaseDocsHandler{}
		gh := testClient(t, cfg, h)
		if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
			t.Fatalf("runWith: %v", err)
		}
		// Proves the skip above came from the duplicate check and not from the
		// run stopping early for some unrelated reason.
		if h.compares != 1 {
			t.Errorf("fetched the diff %d time(s), want 1 when the release is new", h.compares)
		}
		// The stub release changed no files, so there is nothing to file.
		if h.creates != 0 {
			t.Errorf("created %d issue(s) for an empty diff, want 0", h.creates)
		}
	})
}

// A duplicate probe that errored proves nothing, so the run must abort rather
// than analyze and file an issue that might be a duplicate.
func TestRunWithAbortsWhenTheDuplicateCheckFails(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	compares := 0
	gh := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/compare/") {
			compares++
		}
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	err := runWith(context.Background(), discardLogger(), cfg, gh)
	if err == nil || !strings.Contains(err.Error(), "duplicate check") {
		t.Fatalf("runWith = %v, want a duplicate-check error", err)
	}
	if compares != 0 {
		t.Errorf("analyzed the diff despite an unusable duplicate check (%d compare calls)", compares)
	}
}

// A budget so short that not one group is analyzed produces no issue at all, so
// the release is still unanalyzed. That must fail loudly: exiting 0 would leave
// a release silently skipped, and nothing suppresses a retry because no issue
// was filed.
//
// The budget expires before the first group, so no model call is made.
func TestRunWithFailsWhenTheBudgetProducedNothing(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.GeminiAPIKey = "test-key"
	cfg.Model = "gemini-flash-latest"
	cfg.RunBudget = time.Nanosecond

	creates := 0
	gh := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/compare/"):
			_, _ = io.WriteString(w, `{"html_url":"u","files":[{"filename":"a.go","patch":"p"}],"commits":[]}`)
		case strings.HasPrefix(r.URL.Path, "/search/"):
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
		case strings.HasSuffix(r.URL.Path, "/issues") && r.Method == http.MethodPost:
			creates++
			_, _ = io.WriteString(w, `{"number":1}`)
		default:
			_, _ = io.WriteString(w, `[]`)
		}
	}))

	err := runWith(context.Background(), discardLogger(), cfg, gh)
	if err == nil || !strings.Contains(err.Error(), "not one of the") {
		t.Fatalf("runWith = %v, want an error saying no group was analyzed", err)
	}
	// Filing here would mark the release done and suppress the re-run that could
	// still analyze it properly.
	if creates != 0 {
		t.Errorf("created %d issue(s) having analyzed nothing, want 0", creates)
	}
}

// A release too large for the caps produces no findings and full per-group
// coverage, which used to look identical to a clean run. The only place a
// maintainer can learn the caps need raising is an issue that says so.
//
// Mutation that must fail this test: drop `|| diff.diffTruncated()` from the
// don't-file condition in runWith.
func TestRunWithFilesWhenTheDiffWasTruncatedEvenWithNoFindings(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.MaxFiles = 1
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		// The model calls the tool and honestly reports nothing.
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{})}
	}})

	h := &filingHandler{compareFiles: `
		{"filename":"a.go","status":"modified","patch":"+x"},
		{"filename":"b.go","status":"modified","patch":"+y"},
		{"filename":"c.go","status":"modified","patch":"+z"}`}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1: a release the caps truncated must be reported", h.creates)
	}
	if !strings.Contains(h.body, "2 of 3 changed files were not analyzed") {
		t.Errorf("the issue does not say what the caps dropped:\n%s", h.body)
	}
	if !strings.Contains(h.body, "The analysis produced no suggestions") {
		t.Error("the issue does not say the analysis found nothing")
	}
}

// The counterpart: full coverage and nothing to suggest files nothing, so the
// tracker does not collect an empty issue per release.
func TestRunWithFilesNothingOnACleanReleaseWithNoFindings(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{})}
	}})
	h := &filingHandler{}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.creates != 0 {
		t.Errorf("created %d issues on a fully-analyzed release with no findings, want 0", h.creates)
	}
}

// A group whose model never calls the tool must be counted and named, not
// silently absent. The accounting line that does this was previously driven by
// no test at all: both `Unreported = 0` and dropping the subtraction survived
// the whole suite.
//
// Mutation that must fail this test: set a.Unreported = 0 in runWith.
func TestRunWithDisclosesAGroupThatNeverReported(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.FilesPerGroup = 1
	// Group 0 records; group 1's model answers in prose and never calls the tool.
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn == 0 {
			return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
				map[string]any{"kind": "new-feature", "summary": "a new exported API"},
			})}
		}
		return nil
	}})

	h := &filingHandler{compareFiles: `
		{"filename":"a.go","status":"modified","patch":"+x"},
		{"filename":"b.go","status":"modified","patch":"+y"}`}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1", h.creates)
	}
	if !strings.Contains(h.body, "1 of 2 file groups finished without reporting") {
		t.Errorf("the issue does not disclose the silent group:\n%s", h.body)
	}
}

func TestFinishTurnsARecordedErrorIntoANonZeroExit(t *testing.T) {
	gh := testClient(t, testConfig(), failIfCalled(t))
	if err := finish(gh); err != nil {
		t.Errorf("finish on a clean run = %v, want nil", err)
	}
	gh.recordError()
	if err := finish(gh); err == nil {
		t.Error("finish after a recorded error = nil; the run would exit 0 having failed")
	}
}

// A group can record its findings and THEN fail, so "failed" and "recorded
// nothing" are independent facts. Counting the silent groups by subtracting the
// failure count hides one behind the other: here group 0 recorded and then
// errored, group 1 never called the tool, and the subtraction reports zero
// silent groups while blaming a group whose files ARE covered.
//
// Mutation that must fail this test: compute Unreported as
// rec.unreported(len(groups)) - a.NotAttempted - a.Failed.
func TestRunWithSeparatesAFailedGroupFromASilentOne(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.FilesPerGroup = 1

	// Turn 0: group 0 records. Turn 1: group 0's next event is a model error.
	// Turn 2: group 1 answers in prose and never calls the tool.
	withStubModel(t, &stubModel{
		errorOnTurn: func(turn int) bool { return turn == 1 },
		reply: func(turn int) []*genai.Part {
			if turn == 0 {
				return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
					map[string]any{"kind": "new-feature", "summary": "a new exported API"},
				})}
			}
			return nil
		},
	})

	h := &filingHandler{compareFiles: `
		{"filename":"a.go","status":"modified","patch":"+x"},
		{"filename":"b.go","status":"modified","patch":"+y"}`}
	gh := testClient(t, cfg, h)
	// The model error is recorded, so the run reports failure; the issue is still
	// filed with what was found.
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err == nil {
		t.Error("runWith returned nil despite a model error; the run would exit 0")
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1", h.creates)
	}
	if !strings.Contains(h.body, "1 of 2 file groups failed to complete") {
		t.Errorf("the issue does not disclose the failed group:\n%s", h.body)
	}
	// The load-bearing assertion: group 1 is silent and must be named as such,
	// not absorbed into group 0's failure.
	if !strings.Contains(h.body, "1 of 2 file groups finished without reporting") {
		t.Errorf("the silent group was hidden behind the failed one:\n%s", h.body)
	}
}

// The mirror case: a group that fails BEFORE recording anything must be counted
// once, as failed. Counting the unreported groups without excluding the failed
// ones double-counts it, and the issue then reports two uncovered groups out of
// two when only one is.
//
// Mutation that must fail this test: stop appending to a.FailedIndexes in
// analyzeAll.
func TestRunWithCountsAGroupThatFailedBeforeRecordingOnlyOnce(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.FilesPerGroup = 1

	// Turn 0: group 0 errors immediately, so it never calls the tool.
	// Turn 1: group 1 records.
	withStubModel(t, &stubModel{
		errorOnTurn: func(turn int) bool { return turn == 0 },
		reply: func(turn int) []*genai.Part {
			if turn == 1 {
				return []*genai.Part{recordCall("v1.0.0...v1.1.0", 1, []any{
					map[string]any{"kind": "new-feature", "summary": "a new exported API"},
				})}
			}
			return nil
		},
	})

	h := &filingHandler{compareFiles: `
		{"filename":"a.go","status":"modified","patch":"+x"},
		{"filename":"b.go","status":"modified","patch":"+y"}`}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err == nil {
		t.Error("runWith returned nil despite a model error")
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1", h.creates)
	}
	if !strings.Contains(h.body, "1 of 2 file groups failed to complete") {
		t.Errorf("the issue does not disclose the failed group:\n%s", h.body)
	}
	if strings.Contains(h.body, "finished without reporting") {
		t.Errorf("the failed group was counted twice, as failed AND as silent:\n%s", h.body)
	}
}
