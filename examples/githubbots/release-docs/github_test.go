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
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner: "google", Repo: "adk-go",
		TargetOwner: "google", TargetRepo: "adk-go",
		MaxFiles: 10, MaxPatchBytes: 1000, MaxCommits: 10,
		FilesPerGroup: 2, MaxFindingsPerGroup: 5,
	}
}

func discardLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func testClient(t *testing.T, cfg *Config, h http.Handler) *GitHubClient {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &GitHubClient{
		rest:      rest,
		cfg:       cfg,
		selfLogin: "adk-bot",
		log:       discardLogger(),
		out:       io.Discard,
		filed:     make(map[string]fileOutcome),
	}
}

// respondWith answers every request with one canned body.
func respondWith(t *testing.T, cfg *Config, body string) *GitHubClient {
	t.Helper()
	return testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
}

// failIfCalled is a handler that fails the test if the code under test makes any
// request at all.
func failIfCalled(t *testing.T) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected HTTP call to %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusTeapot)
	})
}

// issueJSON renders one issue for a list or search response.
func issueJSON(number int, login, userType, body string) string {
	return fmt.Sprintf(`{"number":%d,"user":{"login":%q,"type":%q},"body":%q}`, number, login, userType, body)
}

// --- Tag resolution ---------------------------------------------------------

func TestPreviousTag(t *testing.T) {
	tags := []string{"v1.3.0", "v1.2.0", "v1.1.0"}
	got, err := previousTag(tags, "v1.3.0")
	if err != nil || got != "v1.2.0" {
		t.Errorf("previousTag(head=v1.3.0) = (%q, %v), want (v1.2.0, nil)", got, err)
	}
	if _, err := previousTag(tags, "v1.1.0"); !errors.Is(err, ErrNoPreviousRelease) {
		t.Errorf("previousTag on the earliest release = %v, want ErrNoPreviousRelease", err)
	}
	if _, err := previousTag(tags, "v9.9.9"); err == nil {
		t.Error("previousTag for an unlisted head returned no error")
	}
}

// A draft release is unpublished code, and a tag outside the allow-list would be
// interpolated into the compare path. Both must be dropped before a tag is used.
//
// Mutation that must fail this test: delete the `if r.GetDraft() { continue }` guard.
func TestPublishedTagsSkipsDraftsAndUnusableTags(t *testing.T) {
	const body = `[
		{"tag_name":"v1.3.0","draft":false},
		{"tag_name":"v1.2.9","draft":true},
		{"tag_name":"../evil","draft":false},
		{"tag_name":"v1.2.0","draft":false}
	]`
	c := respondWith(t, testConfig(), body)
	tags, err := c.publishedTags(context.Background())
	if err != nil {
		t.Fatalf("publishedTags: %v", err)
	}
	want := []string{"v1.3.0", "v1.2.0"}
	if strings.Join(tags, ",") != strings.Join(want, ",") {
		t.Errorf("publishedTags = %v, want %v", tags, want)
	}
}

func TestResolveTagsWithBothTagsMakesNoAPICall(t *testing.T) {
	cfg := testConfig()
	cfg.StartTag, cfg.EndTag = "v1.0.0", "v1.1.0"
	c := testClient(t, cfg, failIfCalled(t))
	base, head, err := c.ResolveTags(context.Background())
	if err != nil || base != "v1.0.0" || head != "v1.1.0" {
		t.Errorf("ResolveTags = (%q, %q, %v), want the configured pair", base, head, err)
	}
}

func TestResolveTagsDerivesTheRange(t *testing.T) {
	const body = `[{"tag_name":"v1.3.0"},{"tag_name":"v1.2.0"}]`
	c := respondWith(t, testConfig(), body)
	base, head, err := c.ResolveTags(context.Background())
	if err != nil {
		t.Fatalf("ResolveTags: %v", err)
	}
	if base != "v1.2.0" || head != "v1.3.0" {
		t.Errorf("ResolveTags = (%q, %q), want (v1.2.0, v1.3.0)", base, head)
	}
}

// --- Compare ----------------------------------------------------------------

func TestCompareRefusesUnvalidatedTagsWithoutHTTP(t *testing.T) {
	c := testClient(t, testConfig(), failIfCalled(t))
	if _, err := c.Compare(context.Background(), "v1", "../../etc/passwd"); err == nil {
		t.Fatal("Compare accepted a traversal tag")
	}
	if _, err := c.Compare(context.Background(), "a..b", "v2"); err == nil {
		t.Fatal("Compare accepted a tag containing the compare separator")
	}
}

func TestCompareBoundsTheDiff(t *testing.T) {
	cfg := testConfig()
	cfg.MaxFiles = 2
	cfg.MaxPatchBytes = 5
	cfg.MaxCommits = 1
	body := `{"html_url":"https://github.com/google/adk-go/compare/v1...v2","files":[
		{"filename":"a.go","status":"modified","additions":2,"deletions":1,"patch":"0123456789"},
		{"filename":"b.go","status":"added","patch":"ok"},
		{"filename":"c.go","status":"added","patch":"dropped"}
	],"commits":[
		{"sha":"abcdef1234567890","commit":{"message":"feat: one\n\nbody text"}},
		{"sha":"1234567890abcdef","commit":{"message":"fix: two"}}
	]}`
	c := respondWith(t, cfg, body)
	diff, err := c.Compare(context.Background(), "v1", "v2")
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if diff.TotalFiles != 3 || len(diff.Files) != 2 || diff.OmittedFiles != 1 {
		t.Errorf("files: total=%d kept=%d omitted=%d, want 3/2/1", diff.TotalFiles, len(diff.Files), diff.OmittedFiles)
	}
	if diff.Files[0].Patch != "01234" || !diff.Files[0].PatchTruncated {
		t.Errorf("patch = %q truncated=%v, want the byte cap applied", diff.Files[0].Patch, diff.Files[0].PatchTruncated)
	}
	if len(diff.Commits) != 1 || diff.OmittedCommits != 1 {
		t.Errorf("commits kept=%d omitted=%d, want 1/1", len(diff.Commits), diff.OmittedCommits)
	}
	// Only the subject line reaches the prompt, and the SHA is abbreviated.
	if diff.Commits[0].Subject != "feat: one" || diff.Commits[0].SHA != "abcdef12" {
		t.Errorf("commit = %+v, want subject-only and a short SHA", diff.Commits[0])
	}
	if diff.CompareURL == "" {
		t.Error("the compare URL was not captured")
	}
}

// The compare endpoint paginates commits, and whether it repeats the files array
// on each page has varied. A repeated file must be analyzed once, not once per
// page.
//
// Mutation that must fail this test: delete the seenFile guard in Compare.
func TestCompareDeduplicatesAcrossPages(t *testing.T) {
	page := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		page++
		if page == 1 {
			w.Header().Set("Link", `<http://x/?page=2>; rel="next", <http://x/?page=2>; rel="last"`)
		}
		_, _ = io.WriteString(w, `{"files":[{"filename":"a.go","patch":"p"}],`+
			`"commits":[{"sha":"aaaa","commit":{"message":"m"}}]}`)
	}))
	diff, err := c.Compare(context.Background(), "v1", "v2")
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if page < 2 {
		t.Fatalf("the test served only %d page(s); pagination was not exercised", page)
	}
	if len(diff.Files) != 1 {
		t.Errorf("got %d files across %d pages, want 1 (deduplicated)", len(diff.Files), page)
	}
	if len(diff.Commits) != 1 {
		t.Errorf("got %d commits across %d pages, want 1 (deduplicated)", len(diff.Commits), page)
	}
}

// --- Duplicate detection ----------------------------------------------------

func TestFindExistingIssueFindsTheBotsOwnIssue(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	body := "[" + issueJSON(42, "adk-bot", "Bot", marker+"\n\nbody") + "]"
	c := respondWith(t, testConfig(), body)
	n, found, err := c.FindExistingIssue(context.Background(), "v1.1.0", marker)
	if err != nil {
		t.Fatalf("FindExistingIssue: %v", err)
	}
	if !found || n != 42 {
		t.Errorf("FindExistingIssue = (%d, %v), want (42, true)", n, found)
	}
}

// Anyone can open an issue whose first line is the marker. Counting it would let
// a stranger suppress the bot's issue for a release.
//
// Mutation that must fail this test: make trustedCreator return true unconditionally.
func TestFindExistingIssueIgnoresAnImpostorsIssue(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	// A regular user account posts the exact marker; the list and search probes
	// both see it.
	body := "[" + issueJSON(7, "stranger", "User", marker+"\n\nnothing to see here") + "]"
	calls := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":1,"items":`+body+`}`)
			return
		}
		_, _ = io.WriteString(w, body)
	}))
	_, found, err := c.FindExistingIssue(context.Background(), "v1.1.0", marker)
	if err != nil {
		t.Fatalf("FindExistingIssue: %v", err)
	}
	if found {
		t.Error("an issue authored by a regular user was accepted as the bot's own")
	}
	if calls < 2 {
		t.Errorf("only %d probe(s) ran; the search probe did not run", calls)
	}
}

// A pull request body is contributor-authored and is returned by the issues list.
// Counting one would give the same suppression as an impostor issue.
func TestFindExistingIssueIgnoresAPullRequest(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	body := fmt.Sprintf(`[{"number":9,"user":{"login":"adk-bot","type":"Bot"},"body":%q,`+
		`"pull_request":{"url":"https://api.github.com/repos/google/adk-go/pulls/9"}}]`, marker)
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":1,"items":`+body+`}`)
			return
		}
		_, _ = io.WriteString(w, body)
	}))
	if _, found, err := c.FindExistingIssue(context.Background(), "v1.1.0", marker); err != nil || found {
		t.Errorf("FindExistingIssue on a marked PR = (%v, %v), want (false, nil)", found, err)
	}
}

// The list probe is bounded to the most recent issues; the search probe is what
// covers a release whose issue has scrolled past that bound.
//
// Mutation that must fail this test: make FindExistingIssue return findByList's
// result directly.
func TestFindExistingIssueFallsBackToTheSearchProbe(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	var searchQuery string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/search/") {
			searchQuery = r.URL.Query().Get("q")
			_, _ = io.WriteString(w, `{"total_count":1,"items":[`+
				issueJSON(101, "adk-bot", "Bot", marker+"\n\nold issue")+`]}`)
			return
		}
		// The list probe sees only newer, unrelated issues.
		_, _ = io.WriteString(w, "["+issueJSON(500, "adk-bot", "Bot", "unrelated")+"]")
	}))
	n, found, err := c.FindExistingIssue(context.Background(), "v1.1.0", marker)
	if err != nil {
		t.Fatalf("FindExistingIssue: %v", err)
	}
	if !found || n != 101 {
		t.Errorf("FindExistingIssue = (%d, %v), want (101, true) from the search probe", n, found)
	}
	if !strings.Contains(searchQuery, "in:title") || !strings.Contains(searchQuery, issueTitle("v1.1.0")) {
		t.Errorf("search query %q does not target the deterministic title", searchQuery)
	}
}

// A probe that errored proves nothing. Returning the error is what stops the
// caller filing an issue it could not show is new.
func TestFindExistingIssueReturnsProbeErrors(t *testing.T) {
	for _, tc := range []struct{ name, failPath string }{
		{"list probe", "/repos/"},
		{"search probe", "/search/"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			marker := bodyMarker("v1", "v2")
			c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.HasPrefix(r.URL.Path, tc.failPath) {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = io.WriteString(w, `{"message":"boom"}`)
					return
				}
				// The OTHER probe must succeed cleanly, so the error under test
				// is the only reason FindExistingIssue can fail. An unparseable
				// body here would make this test pass even with the failing
				// probe's error swallowed.
				if strings.HasPrefix(r.URL.Path, "/search/") {
					_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
					return
				}
				_, _ = io.WriteString(w, `[]`)
			}))
			if _, found, err := c.FindExistingIssue(context.Background(), "v2", marker); err == nil || found {
				t.Errorf("FindExistingIssue = (%v, %v), want an error so the caller does not file", found, err)
			}
		})
	}
}

func TestFindExistingIssueReportsAbsence(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		_, _ = io.WriteString(w, `[]`)
	}))
	n, found, err := c.FindExistingIssue(context.Background(), "v2", bodyMarker("v1", "v2"))
	if err != nil || found || n != 0 {
		t.Errorf("FindExistingIssue on an empty repo = (%d, %v, %v), want (0, false, nil)", n, found, err)
	}
}

// Without a resolved identity -- the built-in Actions token cannot read its own
// user -- authorship falls back to "written by a GitHub App", which still
// excludes every ordinary account.
func TestTrustedCreator(t *testing.T) {
	c := &GitHubClient{cfg: testConfig(), selfLogin: "adk-bot", log: discardLogger()}
	if !c.trustedCreator(&github.User{Login: github.String("ADK-Bot")}) {
		t.Error("the bot's own login was rejected (login comparison must be case-insensitive)")
	}
	if c.trustedCreator(&github.User{Login: github.String("someone-else"), Type: github.String("Bot")}) {
		t.Error("with a resolved identity, another App must not be trusted")
	}
	if c.trustedCreator(nil) {
		t.Error("a nil author was trusted")
	}

	unresolved := &GitHubClient{cfg: testConfig(), log: discardLogger()}
	if !unresolved.trustedCreator(&github.User{Login: github.String("some-app"), Type: github.String("Bot")}) {
		t.Error("without an identity, an App-authored issue should be accepted")
	}
	if unresolved.trustedCreator(&github.User{Login: github.String("stranger"), Type: github.String("User")}) {
		t.Error("without an identity, a user-authored issue must still be rejected")
	}
}

// --- The single mutation ----------------------------------------------------

// The claim is what makes "one issue per release" hold within a run: two callers
// that both passed the duplicate check must not both create.
//
// Mutation that must fail this test: make claimRelease always return true.
func TestFileReleaseIssueCreatesOnlyOnce(t *testing.T) {
	posts := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts++
		}
		_, _ = io.WriteString(w, `{"number":11}`)
	}))
	n, err := c.FileReleaseIssue(context.Background(), "v1...v2", "title", "body")
	if err != nil || n != 11 {
		t.Fatalf("first FileReleaseIssue = (%d, %v), want (11, nil)", n, err)
	}
	if _, err := c.FileReleaseIssue(context.Background(), "v1...v2", "title", "body"); err != nil {
		t.Fatalf("second FileReleaseIssue returned an error: %v", err)
	}
	if posts != 1 {
		t.Errorf("made %d create calls, want 1", posts)
	}
}

// Dry run must suppress the write AND render what it would have filed --
// rendering nothing would make the mode useless for reviewing a release.
func TestFileReleaseIssueDryRunRendersWithoutWriting(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	c := testClient(t, cfg, failIfCalled(t))
	var rendered strings.Builder
	c.out = &rendered
	n, err := c.FileReleaseIssue(context.Background(), "v1...v2", "title", "THE BODY")
	if err != nil || n != 0 {
		t.Fatalf("dry-run FileReleaseIssue = (%d, %v), want (0, nil)", n, err)
	}
	if !strings.Contains(rendered.String(), "THE BODY") {
		t.Errorf("dry run did not render the issue body, got %q", rendered.String())
	}
}

// After a failed write, a second call must report the failure rather than
// "already filed" -- otherwise the run records an issue that was never created.
//
// The failure is PRODUCED by a failing write, not installed by calling
// recordFileFailure directly, so the recording half is pinned too.
func TestSecondFileAfterFailureReportsTheFailure(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	if _, err := c.FileReleaseIssue(context.Background(), "v1...v2", "t", "b"); err == nil {
		t.Fatal("the first FileReleaseIssue must surface the write failure")
	}
	if !c.hadError() {
		t.Error("a failed write must be recorded so the run exits non-zero")
	}
	_, err := c.FileReleaseIssue(context.Background(), "v1...v2", "t", "b")
	if err == nil || !strings.Contains(err.Error(), "already failed") {
		t.Errorf("second FileReleaseIssue = %v, want an error naming the earlier failure", err)
	}
}

// The claim map is only ever built in the constructor. Nothing else initializes
// it, so a dropped assignment would panic on the first claim in production.
func TestNewGitHubClientInitializesTheClaimMap(t *testing.T) {
	// No network: NewGitHubClient builds its own REST client, so the identity
	// lookup would reach api.github.com. Cancel the context so that
	// best-effort call fails fast, and assert on the wiring.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	c := NewGitHubClient(ctx, testConfig(), discardLogger())
	if !c.claimRelease("v1...v2") {
		t.Fatal("the first claim on a fresh client was refused")
	}
	if c.claimRelease("v1...v2") {
		t.Error("the same release was claimed twice")
	}
	if c.selfLogin != "" {
		t.Errorf("selfLogin = %q, want empty after a failed identity lookup", c.selfLogin)
	}
}

func TestCommitSubjectTakesTheFirstLineOnly(t *testing.T) {
	if got := commitSubject("feat: thing\r\n\r\nlong body\nmore"); got != "feat: thing" {
		t.Errorf("commitSubject = %q, want the subject line only", got)
	}
	if got := commitSubject(strings.Repeat("x", 300)); len([]rune(got)) > 300 {
		t.Errorf("commitSubject did not bound a 300-rune subject: %d runes", len([]rune(got)))
	}
}
