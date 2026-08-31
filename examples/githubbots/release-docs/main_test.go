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
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
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

// A fence draw failure must abort rather than fall back to a predictable marker:
// a guessable nonce lets a contributor pre-write the closing marker in a code
// comment and escape the fence.
func TestNonceFailureIsSurfacedRatherThanSubstituted(t *testing.T) {
	orig := newNonce
	newNonce = func() (string, error) { return "", errors.New("no entropy") }
	t.Cleanup(func() { newNonce = orig })

	if _, err := newNonce(); err == nil {
		t.Fatal("test setup: the forced failure did not take")
	}
	// An empty nonce would produce "[UNTRUSTED:]", which anybody can write. The
	// production path must never reach renderGroupPrompt with one.
	got := renderGroupPrompt(&ReleaseDiff{BaseTag: "v1", HeadTag: "v2"},
		[]ChangedFile{{Path: "a.go", Patch: "p"}}, 0, 1, "")
	if !strings.Contains(got, "[UNTRUSTED:]") {
		t.Fatal("test premise is wrong: an empty nonce no longer yields a guessable fence")
	}
}

// stubRun records which group indexes the orchestrator actually drove.
type stubRun struct {
	mu      sync.Mutex
	indexes []int
	delay   time.Duration
}

func (s *stubRun) fn(_ context.Context, index int) {
	s.mu.Lock()
	s.indexes = append(s.indexes, index)
	s.mu.Unlock()
	time.Sleep(s.delay)
}

func TestAnalyzeAllDrivesEveryGroupInOrder(t *testing.T) {
	cfg := testConfig()
	cfg.RunBudget = time.Minute
	cfg.GroupTimeout = time.Minute
	groups := [][]ChangedFile{{{Path: "a"}}, {{Path: "b"}}, {{Path: "c"}}}

	s := &stubRun{}
	if exhausted := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease, s.fn); exhausted {
		t.Error("analyzeAll reported the budget exhausted on a one-minute budget")
	}
	if len(s.indexes) != 3 || s.indexes[0] != 0 || s.indexes[1] != 1 || s.indexes[2] != 2 {
		t.Errorf("groups driven = %v, want [0 1 2]", s.indexes)
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

	s := &stubRun{delay: 25 * time.Millisecond}
	exhausted := analyzeAll(context.Background(), cfg, discardLogger(), groups, testRelease, s.fn)
	if !exhausted {
		t.Fatal("analyzeAll did not report the exhausted budget")
	}
	if len(s.indexes) >= len(groups) {
		t.Errorf("drove %d of %d groups; the budget did not stop the loop", len(s.indexes), len(groups))
	}
	// The issue must say so, otherwise a partial analysis reads as a complete one.
	body := buildIssueBody(&ReleaseDiff{BaseTag: "v1", HeadTag: "v2"}, []Finding{{Summary: "s"}}, exhausted)
	if !strings.Contains(body, "run budget was exhausted") {
		t.Error("the issue body does not disclose the exhausted budget")
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
	if err == nil || !strings.Contains(err.Error(), "run budget") {
		t.Fatalf("runWith = %v, want an error naming the exhausted budget", err)
	}
	if creates != 0 {
		t.Errorf("created %d issue(s) with no findings, want 0", creates)
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
