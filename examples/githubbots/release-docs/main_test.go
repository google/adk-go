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
	"go/parser"
	"go/token"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
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

// TestNonceIsDrawnFromACryptographicSource reads the package's own imports.
//
// The test above cannot do this job and no output-based check can. math/rand
// yields unique, correctly distributed, right-length hex, so freshness, charset
// and length assertions all pass against a generator whose next value an
// attacker can infer from earlier ones. Measured on this package: swapping
// crypto/rand for math/rand in main.go leaves the whole suite green, including
// under -race. Only the source shows it.
//
// It matters here more than in most places. The [UNTRUSTED:<nonce>] fence is
// the entire trust boundary between contributor-authored diff text and
// instructions from us, so a predictable nonce lets an attacker write the
// closing marker into a commit message and forge trusted context -- which is
// the containment the adversarial suite's results rest on.
//
// Two assertions rather than one. The ABSENCE half catches an import of
// math/rand anywhere, under any alias, because it resolves the import path and
// not the local identifier. The PRESENCE half catches the evasion absence
// cannot see: move the draw into a helper using some other weak source and
// nothing imports math/rand, while crypto/rand quietly stops being used. A pin
// written only in the absence form is satisfied vacuously the moment the code
// it guards relocates.
func TestNonceIsDrawnFromACryptographicSource(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse the package: %v", err)
	}

	cryptoRand, files := false, 0
	for _, pkg := range pkgs {
		for name, f := range pkg.Files {
			files++
			for _, imp := range f.Imports {
				path, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatalf("%s: unquote import %s: %v", name, imp.Path.Value, err)
				}
				switch path {
				case "crypto/rand":
					cryptoRand = true
				case "math/rand", "math/rand/v2":
					t.Errorf("%s imports %q. The [UNTRUSTED:<nonce>] fence is unforgeable only "+
						"while the nonce is unpredictable, and math/rand is not a cryptographic "+
						"source: an attacker who can infer the nonce closes the fence from inside "+
						"a commit message and forges trusted context.", name, path)
				}
			}
		}
	}
	if files == 0 {
		t.Fatal("parsed no non-test files, so this check proved nothing")
	}
	if !cryptoRand {
		t.Errorf("no file in this package imports crypto/rand (parsed %d files). Either the nonce "+
			"draw moved somewhere this check cannot see, or it stopped being cryptographic. The "+
			"absence half above cannot tell those apart, which is why this half exists.", files)
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
	// Generous on purpose: correctness needs only that the loop reaches group 1
	// before the budget fires, and group 1 then blocks until it does. The cost
	// is this test taking the budget's length.
	cfg.RunBudget = 300 * time.Millisecond
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
	if len(indexes) != 2 {
		t.Fatalf("drove %d groups, want both attempted for this case", len(indexes))
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
	cfg.RunBudget = time.Nanosecond
	// Install a stub even though the budget expires before any group runs:
	// without it newAgentRunner calls the real constructor, and on a machine
	// with Vertex credentials in the environment the run fails at "create
	// model" and this test's assertion breaks for a reason it is not about.
	withStubModel(t, &stubModel{reply: func(int) []*genai.Part {
		t.Error("the model was invoked despite an exhausted budget")
		return nil
	}})

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
	if err == nil || !strings.Contains(err.Error(), "1 never attempted") {
		t.Fatalf("runWith = %v, want an error naming the group the budget never reached", err)
	}
	// Filing here would mark the release done and suppress the re-run that could
	// still analyze it properly.
	if creates != 0 {
		t.Errorf("created %d issue(s) having analyzed nothing, want 0", creates)
	}
}

// A release too large for the caps, with nothing to suggest, must file NOTHING
// and must NOT turn the job red. Filing would plant the release marker and
// suppress a later run, and failing would make an ordinary release fail: one
// patch over the byte cap is enough to truncate the diff, which is near-certain
// in a real release.
//
// Mutation that must fail this test: return an error when
// `!a.complete() || diff.diffTruncated()` and there are no findings.
func TestRunWithIsQuietOnATruncatedReleaseWithNoFindings(t *testing.T) {
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
	// Under Actions the annotation is the only maintainer-facing signal on this
	// path, so drive that branch and capture it.
	t.Setenv("GITHUB_ACTIONS", "true")
	var annotated strings.Builder
	gh.out = &annotated

	var logged strings.Builder
	log := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug}))
	if err := runWith(context.Background(), log, cfg, gh); err != nil {
		t.Fatalf("runWith = %v, want nil: a release with nothing to suggest is ordinary, not a failure", err)
	}
	if h.creates != 0 {
		t.Errorf("created %d issues with nothing to suggest; the marker would suppress a later run", h.creates)
	}
	// Exiting quietly is only acceptable because what was missed is reported.
	// Without this the test would pass identically on a release that was NOT
	// truncated, i.e. as a duplicate of the clean-release test below.
	out := logged.String()
	if !strings.Contains(out, "the analysis was also incomplete") {
		t.Errorf("the run did not report that it missed part of the release:\n%s", out)
	}
	if !strings.Contains(out, "diff_truncated=true") {
		t.Errorf("the run did not report the truncation, so this test does not distinguish a truncated "+
			"release from a clean one:\n%s", out)
	}
	// A stderr line alone would leave a green check and no signal in the UI.
	if got := annotated.String(); !strings.HasPrefix(got, "::warning::") {
		t.Errorf("no workflow annotation was raised, so the run is a silent green: %q", got)
	}
}

// The BLOCKER round 5 found: text in the diff telling the model to answer in
// prose and call no tool made every group unreported, which made the run look
// incomplete, which forced the bot's only write. The filed issue carries the
// release marker, so the attacker permanently suppressed re-analysis of the
// release they poisoned.
//
// Mutation that must fail this test: restore any model-derived term to the
// filing decision, e.g. `if len(findings) == 0 && a.complete() && ...` becomes
// reachable only when complete() is true.
func TestRunWithDoesNotFileWhenASteeredModelSilencesEveryGroup(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.FilesPerGroup = 1
	// Every group answers in prose; the tool never fires.
	withStubModel(t, &stubModel{reply: func(int) []*genai.Part { return nil }})

	h := &filingHandler{compareFiles: `
		{"filename":"a.go","status":"modified","patch":"+x"},
		{"filename":"b.go","status":"modified","patch":"+y"}`}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith = %v, want nil", err)
	}
	if h.creates != 0 {
		t.Errorf("a steered model forced %d issue(s); the marker would suppress every later run", h.creates)
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

// One group, which records real findings and THEN hits a model error. Gating on
// `Attempted - Failed == 0` alone throws those findings away and reports that no
// group was analyzed, when one plainly produced output.
//
// Mutation that must fail this test: drop `&& len(findings) == 0` from the
// nothing-was-analyzed check in runWith.
func TestRunWithKeepsFindingsFromAGroupThatFailedAfterRecording(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	// One group only, so Attempted-Failed is zero.
	cfg.FilesPerGroup = 5
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

	h := &filingHandler{}
	gh := testClient(t, cfg, h)
	// The model error is recorded, so the run still exits non-zero.
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err == nil {
		t.Error("runWith returned nil despite a model error")
	} else if strings.Contains(err.Error(), "not one of the") {
		t.Errorf("runWith reported that no group produced anything, but one did: %v", err)
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1: the findings the group recorded were thrown away", h.creates)
	}
	if !strings.Contains(h.body, "a new exported API") {
		t.Errorf("the filed issue lost the finding the failed group recorded:\n%s", h.body)
	}
	if !strings.Contains(h.body, "1 of 1 file groups failed to complete") {
		t.Errorf("the issue does not disclose the failure:\n%s", h.body)
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

// godotenv fills in any variable the environment does not already set, and the
// workflow sets neither TARGET_OWNER nor TARGET_REPO — so a .env committed to
// the tree would choose which repository the bot files into. It must not be read
// under Actions.
//
// Mutation that must fail this test: remove the GITHUB_ACTIONS check from
// loadDotEnv.
func TestDotEnvIsSkippedUnderActions(t *testing.T) {
	// Written by EFFECT, not by a return value. An earlier version asserted a
	// boolean no production caller reads, so moving the Load call above the
	// guard would have left it green.
	env := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(env, []byte("TARGET_REPO=attacker-controlled\n"), 0o600); err != nil {
		t.Fatalf("write .env: %v", err)
	}

	// t.Setenv registers the restore; Unsetenv then makes the variable ABSENT,
	// which is the state godotenv fills. It leaves a variable that is merely set
	// to empty alone, so setting it to "" would not exercise the path at all.
	t.Setenv("TARGET_REPO", "")
	if err := os.Unsetenv("TARGET_REPO"); err != nil {
		t.Fatalf("unset TARGET_REPO: %v", err)
	}
	t.Setenv("GITHUB_ACTIONS", "true")
	loadDotEnv(env)
	if got := os.Getenv("TARGET_REPO"); got != "" {
		t.Errorf("under Actions a repository .env set TARGET_REPO=%q; it chooses where the issue is filed", got)
	}

	t.Setenv("GITHUB_ACTIONS", "")
	if err := os.Unsetenv("GITHUB_ACTIONS"); err != nil {
		t.Fatalf("unset GITHUB_ACTIONS: %v", err)
	}
	loadDotEnv(env)
	if got := os.Getenv("TARGET_REPO"); got != "attacker-controlled" {
		t.Errorf("a local run did not read .env (TARGET_REPO=%q), so local configuration is broken", got)
	}
}

// The two disclosure counters reach the issue only through two assignment lines
// in runWith, and every other test that "proves" disclosure hand-builds the
// analysis value — so both lines could be deleted with the suite still green.
// This produces a capped finding and a discarded one for real and reads the
// filed body.
//
// Mutations that must fail this test: delete `a.Discarded = rec.discardedCount()`
// or `a.CappedFindings = rec.cappedCount()` from runWith.
func TestRunWithReportsCappedAndDiscardedFindingsItActuallyProduced(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.MaxFindingsPerGroup = 2
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		// Four findings against a cap of two: the last two are dropped by the
		// cap, and of the two kept, one is emptied by sanitization.
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{
			map[string]any{"kind": "new-feature", "summary": "a real suggestion"},
			map[string]any{"kind": "new-feature", "summary": "\u200b\u202e"},
			map[string]any{"kind": "new-feature", "summary": "dropped by the cap"},
			map[string]any{"kind": "new-feature", "summary": "also dropped"},
		})}
	}})

	h := &filingHandler{}
	gh := testClient(t, cfg, h)
	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.creates != 1 {
		t.Fatalf("created %d issues, want 1", h.creates)
	}
	if !strings.Contains(h.body, "2 suggestions beyond the per-group cap were dropped") {
		t.Errorf("the filed issue does not disclose the capped suggestions:\n%s", h.body)
	}
	if !strings.Contains(h.body, "1 recorded suggestions were discarded") {
		t.Errorf("the filed issue does not disclose the discarded suggestion:\n%s", h.body)
	}
	if !strings.Contains(h.body, "a real suggestion") {
		t.Error("the filed issue lost the one renderable suggestion")
	}
}

// The annotation is for coverage loss an operator can act on. A patch the byte
// cap cut is routine on a large release, and annotating it would make the yellow
// banner meaningless — which is the same noise problem an earlier revision
// caused one severity higher by failing the job.
//
// Mutation that must fail this test: widen the annotation condition back to
// `!a.complete() || diff.diffTruncated()`.
func TestRunWithDoesNotAnnotateARoutineTruncatedPatch(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	cfg.MaxPatchBytes = 4 // every file's patch is cut, but no file is dropped
	withStubModel(t, &stubModel{reply: func(turn int) []*genai.Part {
		if turn > 0 {
			return nil
		}
		return []*genai.Part{recordCall("v1.0.0...v1.1.0", 0, []any{})}
	}})

	h := &filingHandler{compareFiles: `{"filename":"a.go","status":"modified","patch":"+xxxxxxxxxx"}`}
	gh := testClient(t, cfg, h)
	t.Setenv("GITHUB_ACTIONS", "true")
	var annotated strings.Builder
	gh.out = &annotated

	if err := runWith(context.Background(), discardLogger(), cfg, gh); err != nil {
		t.Fatalf("runWith: %v", err)
	}
	if h.creates != 0 {
		t.Errorf("created %d issues, want 0", h.creates)
	}
	// The premise: the diff IS truncated, so the wider condition would fire.
	diff := &ReleaseDiff{Files: []ChangedFile{{Path: "a.go", Patch: "+xxx", PatchTruncated: true}}}
	if !diff.diffTruncated() {
		t.Fatal("test premise: a cut patch must count as a truncated diff")
	}
	if got := annotated.String(); got != "" {
		t.Errorf("a routine cut patch raised a warning annotation: %q", got)
	}
}
