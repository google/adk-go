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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner: "google", Repo: "adk-go",
		TargetOwner: "google", TargetRepo: "adk-go",
		MaxFiles: 10, MaxPatchBytes: 1000, MaxCommits: 10,
		FilesPerGroup: 2, MaxFindingsPerGroup: 5,
		RunBudget: time.Minute, GroupTimeout: time.Minute,
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

// issueJSON renders one issue for a list or search response, with the
// deterministic title the bot would have filed.
func issueJSON(number int, login, userType, body string) string {
	return issueJSONTitled(number, login, userType, issueTitle("v1.1.0"), body)
}

// issueJSONTitled is issueJSON with the title spelled out, for the cases that
// turn on it.
func issueJSONTitled(number int, login, userType, title, body string) string {
	return fmt.Sprintf(`{"number":%d,"user":{"login":%q,"type":%q},"title":%q,"body":%q}`,
		number, login, userType, title, body)
}

// --- Tag resolution ---------------------------------------------------------

// The live google/adk-go release listing, in the order ListReleases returns it.
// Maintenance-line releases interleave with the main line because GitHub orders
// by created_at -- the date of the COMMIT a tag points at, not the date the
// release was published. Any selection that trusts this order is wrong on this
// repository, today, on the bot's default path.
func liveAdkGoReleases() []Release {
	var out []Release
	// Tag with its real publication date, in the real list order.
	for _, r := range [][2]string{
		{"v2.3.0", "2026-08-31T14:44:57Z"},
		{"v1.6.0", "2026-08-12T06:36:40Z"},
		{"v2.2.0", "2026-08-10T07:30:43Z"},
		{"v2.1.0", "2026-07-23T16:16:30Z"},
		{"v1.5.1", "2026-07-22T13:53:57Z"},
		{"v2.0.0", "2026-06-30T14:24:36Z"},
		{"v1.5.0", "2026-07-01T09:40:04Z"},
		{"v1.4.0", "2026-05-29T13:45:25Z"},
		{"v1.3.0", "2026-05-19T12:49:31Z"},
		{"v1.2.0", "2026-04-23T19:13:09Z"},
		{"v1.1.0", "2026-04-10T15:08:03Z"},
		{"v1.0.0", "2026-03-23T09:38:41Z"},
	} {
		published, err := time.Parse(time.RFC3339, r[1])
		if err != nil {
			panic(err)
		}
		out = append(out, Release{Tag: r[0], Published: published})
	}
	return out
}

// Mutation that must fail this test: make previousTag return the entry after
// head in list order (`for i, r := range releases { if r.Tag == head { return
// releases[i+1].Tag, nil } }`).
func TestPreviousTagUsesVersionOrderNotListOrder(t *testing.T) {
	rels := liveAdkGoReleases()
	for _, tc := range []struct{ head, want string }{
		// List order would give v1.6.0 here: a base from the other major line.
		{"v2.3.0", "v2.2.0"},
		// List order would give v2.2.0 here: a base NEWER than the head, on the
		// other major line, producing a backwards compare.
		{"v1.6.0", "v1.5.1"},
		// Version order ALONE would give v1.6.0 here, which shipped six weeks
		// AFTER v2.0.0 and so was never the release v2.0.0 followed. The
		// publication cutoff is what excludes it. v1.5.0 published 2026-07-01,
		// after v2.0.0's 2026-06-30, so the answer is v1.4.0.
		{"v2.0.0", "v1.4.0"},
		{"v1.1.0", "v1.0.0"},
		// A head published after the listing was taken still resolves.
		{"v2.4.0", "v2.3.0"},
	} {
		got, err := previousTag(rels, tc.head)
		if err != nil {
			t.Errorf("previousTag(head=%s): %v", tc.head, err)
			continue
		}
		if got != tc.want {
			t.Errorf("previousTag(head=%s) = %s, want %s", tc.head, got, tc.want)
		}
	}
}

// A prerelease must never become a base: diffing a final release against its own
// release candidate covers only the rc-to-final delta and drops the feature set
// the release shipped.
//
// Mutation that must fail this test: delete the `r.Prerelease` term from
// previousTag's skip condition.
func TestPreviousTagSkipsPrereleases(t *testing.T) {
	// v1.3.5-rc.1 carries a version strictly BETWEEN v1.3.0 and the head, so it
	// is the answer version comparison alone would give. Only the prerelease
	// flag excludes it. (An rc of the head itself, v1.4.0-rc.1, would be
	// excluded by the version comparison anyway and so proves nothing.)
	rels := []Release{
		{Tag: "v1.4.0"},
		{Tag: "v1.4.0-rc.1", Prerelease: true},
		{Tag: "v1.3.5-rc.1", Prerelease: true},
		{Tag: "v1.3.0"},
	}
	got, err := previousTag(rels, "v1.4.0")
	if err != nil {
		t.Fatalf("previousTag: %v", err)
	}
	if got != "v1.3.0" {
		t.Errorf("previousTag(v1.4.0) = %s, want v1.3.0 (v1.3.5-rc.1 is a prerelease)", got)
	}
}

func TestPreviousTagErrors(t *testing.T) {
	rels := []Release{{Tag: "v1.0.0"}}
	if _, err := previousTag(rels, "v1.0.0"); !errors.Is(err, ErrNoPreviousRelease) {
		t.Errorf("previousTag on the earliest release = %v, want ErrNoPreviousRelease", err)
	}
	// The message must tell the operator the listing is bounded and name the way out.
	_, err := previousTag(rels, "v1.0.0")
	for _, want := range []string{"bounded", "-start-tag"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
	if _, err := previousTag(rels, "nightly"); err == nil {
		t.Error("previousTag accepted a non-version head tag")
	}
}

func TestLatestTagPicksTheGreatestVersion(t *testing.T) {
	// v1.6.0 is FIRST in list order but is not the latest release.
	rels := append(liveAdkGoReleases(), Release{Tag: "v9.9.9-rc.1", Prerelease: true})
	got, ok := latestTag(rels)
	if !ok || got != "v2.3.0" {
		t.Errorf("latestTag = (%q, %v), want (v2.3.0, true)", got, ok)
	}
	if _, ok := latestTag([]Release{{Tag: "nightly"}}); ok {
		t.Error("latestTag accepted a non-version tag")
	}
}

func TestParseAndCompareVersions(t *testing.T) {
	for _, tc := range []struct {
		tag  string
		want []int
		ok   bool
	}{
		{"v1.2.3", []int{1, 2, 3}, true},
		{"1.2", []int{1, 2}, true},
		{"v2.0.0-rc.1", []int{2, 0, 0}, true},
		{"v1.0.0+build.5", []int{1, 0, 0}, true},
		{"nightly", nil, false},
		{"v", nil, false},
	} {
		got, ok := parseVersion(tc.tag)
		if ok != tc.ok || (ok && !slices.Equal(got, tc.want)) {
			t.Errorf("parseVersion(%q) = (%v, %v), want (%v, %v)", tc.tag, got, ok, tc.want, tc.ok)
		}
	}
	// A shorter version compares as if zero-padded.
	if compareVersions([]int{1, 2}, []int{1, 2, 0}) != 0 {
		t.Error("v1.2 and v1.2.0 should compare equal")
	}
	if compareVersions([]int{1, 10, 0}, []int{1, 9, 0}) <= 0 {
		t.Error("versions must compare numerically, not lexically (1.10 > 1.9)")
	}
}

// A draft release is unpublished code, and a tag outside the allow-list would
// be interpolated into the compare API path. Both must be dropped. The
// prerelease flag and the publication time must survive, because base selection
// reads both: without the flag an rc becomes a base, and without the time a
// maintenance release that shipped later becomes the base for an older major.
//
// Mutations that must fail this test: delete the `if r.GetDraft()` guard;
// delete the `validTag` guard; hardcode `Prerelease: false`; hardcode
// `Published: time.Time{}`.
func TestPublishedReleasesMapsTheAPIFields(t *testing.T) {
	const body = `[
		{"tag_name":"v1.3.0","draft":false,"prerelease":false,"published_at":"2026-05-19T12:49:31Z"},
		{"tag_name":"v1.2.9","draft":true,"published_at":"2026-05-01T00:00:00Z"},
		{"tag_name":"../evil","draft":false,"published_at":"2026-05-01T00:00:00Z"},
		{"tag_name":"v1.2.0","draft":false,"published_at":"2026-04-23T19:13:09Z"},
		{"tag_name":"v1.1.0-rc.1","draft":false,"prerelease":true,"published_at":"2026-04-01T00:00:00Z"}
	]`
	c := respondWith(t, testConfig(), body)
	rels, err := c.publishedReleases(context.Background())
	if err != nil {
		t.Fatalf("publishedReleases: %v", err)
	}
	var tags []string
	for _, r := range rels {
		tags = append(tags, r.Tag)
	}
	want := []string{"v1.3.0", "v1.2.0", "v1.1.0-rc.1"}
	if strings.Join(tags, ",") != strings.Join(want, ",") {
		t.Fatalf("publishedReleases = %v, want %v (the draft and the unusable tag are dropped)", tags, want)
	}
	if rels[0].Prerelease {
		t.Error("a full release was marked as a prerelease")
	}
	if !rels[2].Prerelease {
		t.Error("the prerelease flag was dropped, so an rc tag could become a base")
	}
	wantPublished := time.Date(2026, 5, 19, 12, 49, 31, 0, time.UTC)
	if !rels[0].Published.Equal(wantPublished) {
		t.Errorf("Published = %v, want %v; without it base selection loses its publication cutoff",
			rels[0].Published, wantPublished)
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
	// List order is deliberately NOT version order, matching the live repo, and
	// v1.4.0 is a maintenance release published AFTER the head — so it is
	// excluded by version, and v1.3.0 by the publication cutoff. The answer is
	// v1.2.0 only if both signals are wired through from the API.
	const body = `[
		{"tag_name":"v1.3.0","published_at":"2026-08-20T00:00:00Z"},
		{"tag_name":"v1.6.0","published_at":"2026-08-12T00:00:00Z"},
		{"tag_name":"v1.2.0","published_at":"2026-04-23T00:00:00Z"}
	]`
	c := respondWith(t, testConfig(), body)
	base, head, err := c.ResolveTags(context.Background())
	if err != nil {
		t.Fatalf("ResolveTags: %v", err)
	}
	if head != "v1.6.0" {
		t.Errorf("ResolveTags head = %q, want v1.6.0 (the greatest version, not the first listed)", head)
	}
	if base != "v1.2.0" {
		t.Errorf("ResolveTags base = %q, want v1.2.0: v1.3.0 has a lower version but was published "+
			"after the head, so it was never the release v1.6.0 followed", base)
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
	body := `{"html_url":"https://github.com/google/adk-go/compare/v1...v2","total_commits":2,"files":[
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
	// The omitted count comes from the comparison's own total_commits, so a
	// release whose commits exceed the page bound does not under-report.
	//
	// Mutation that must fail this test: compute OmittedCommits from
	// len(diff.Commits) before the cap instead of from total_commits.
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
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":1,"items":`+body+`}`)
			return
		}
		_, _ = io.WriteString(w, body)
	}))
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
			_, _ = io.WriteString(w, `{"number":11}`)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		_, _ = io.WriteString(w, `[]`)
	}))
	n, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "title", "body")
	if err != nil || n != 11 {
		t.Fatalf("first FileReleaseIssue = (%d, %v), want (11, nil)", n, err)
	}
	if _, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "title", "body"); err != nil {
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
	n, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "title", "THE BODY")
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
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The duplicate re-check succeeds; only the create fails, so the failure
		// under test is the write and not the probe.
		if r.Method == http.MethodPost {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"message":"boom"}`)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		_, _ = io.WriteString(w, `[]`)
	}))
	if _, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "t", "b"); err == nil {
		t.Fatal("the first FileReleaseIssue must surface the write failure")
	}
	if !c.hadError() {
		t.Error("a failed write must be recorded so the run exits non-zero")
	}
	_, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "t", "b")
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

// The rune cap is the only bound on a commit subject, and up to MaxCommits of
// them go into the prompt, so a contributor writing one enormous subject line is
// bounded by this and nothing else.
//
// The previous assertion was `> 300` on a 300-rune input, which is false by
// construction: it passed with the cap deleted.
//
// Mutation that must fail this test: change commitSubject's truncateRunes bound
// from 200 to 100000.
func TestCommitSubjectTakesTheFirstLineOnlyAndBoundsIt(t *testing.T) {
	if got := commitSubject("feat: thing\r\n\r\nlong body\nmore"); got != "feat: thing" {
		t.Errorf("commitSubject = %q, want the subject line only", got)
	}
	const cap = 200
	got := commitSubject(strings.Repeat("x", 5000))
	if n := len([]rune(got)); n != cap+len([]rune(" …[truncated]")) {
		t.Errorf("commitSubject returned %d runes, want the %d-rune cap plus the marker", n, cap)
	}
	if !strings.HasSuffix(got, "[truncated]") {
		t.Errorf("a truncated subject does not say so: %q", got)
	}
	// A subject under the cap is returned whole.
	if got := commitSubject(strings.Repeat("y", cap)); len([]rune(got)) != cap {
		t.Errorf("a subject exactly at the cap was altered: %d runes", len([]rune(got)))
	}
}

// filingHandler serves every endpoint runWith touches and captures the create
// request, so an end-to-end test can assert what GitHub would actually receive.
type filingHandler struct {
	// compareFiles overrides the files array the compare endpoint returns, so a
	// test can drive the file caps and the grouping.
	compareFiles string

	mu sync.Mutex
	// writes counts every non-GET request, so a mutation added anywhere shows up.
	writes      int
	creates     int
	title       string
	body        string
	searchQuery string
}

func (h *filingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if r.Method != http.MethodGet {
		h.writes++
	}
	switch {
	case strings.Contains(r.URL.Path, "/compare/"):
		files := h.compareFiles
		if files == "" {
			files = `{"filename":"agent/agent.go","status":"modified","patch":"+// New exported API."}`
		}
		_, _ = io.WriteString(w, `{"html_url":"https://example.invalid/compare","total_commits":1,`+
			`"files":[`+files+`],`+
			`"commits":[{"sha":"abcdef1234","commit":{"message":"feat: a new exported API"}}]}`)
	case strings.HasPrefix(r.URL.Path, "/search/"):
		h.searchQuery = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
	case strings.HasSuffix(r.URL.Path, "/issues") && r.Method == http.MethodPost:
		h.creates++
		var req struct{ Title, Body string }
		_ = json.NewDecoder(r.Body).Decode(&req)
		h.title, h.body = req.Title, req.Body
		_, _ = io.WriteString(w, `{"number":1}`)
	default:
		_, _ = io.WriteString(w, `[]`)
	}
}

// firstLine returns s up to its first newline, for a readable failure message.
func firstLine(s string) string {
	l, _, _ := strings.Cut(s, "\n")
	return l
}

// Two runs that both passed the duplicate check must not both create. The claim
// is the only thing standing between them, and until now it was driven only
// sequentially: the -race gate was vacuous for GitHubClient because no test in
// the package ever touched one from a second goroutine.
//
// Mutation that must fail this test: make claimRelease always return true.
func TestFileReleaseIssueIsClaimedUnderConcurrency(t *testing.T) {
	var posts int32
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			atomic.AddInt32(&posts, 1)
		}
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		if r.Method == http.MethodPost {
			_, _ = io.WriteString(w, `{"number":1}`)
			return
		}
		_, _ = io.WriteString(w, `[]`)
	}))

	// Start-gated, matching the recorder's contention test: without the barrier
	// the launch skew dominates the window a check-then-act split would be
	// visible in, and detection here is probabilistic either way.
	const n = 20
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, err := c.FileReleaseIssue(context.Background(), "v1.0.0...v1.1.0", "v1.0.0", "v1.1.0", "t", "b"); err != nil {
				t.Errorf("FileReleaseIssue: %v", err)
			}
		}()
	}
	close(start)
	wg.Wait()

	if got := atomic.LoadInt32(&posts); got != 1 {
		t.Errorf("%d concurrent callers produced %d create calls, want 1", n, got)
	}
}

// The caller's duplicate check runs before the analysis loop and is minutes
// stale by the time the write happens. A concurrent run that filed in that
// window must be seen.
//
// Mutation that must fail this test: delete the FindExistingIssue re-check from
// FileReleaseIssue.
func TestFileReleaseIssueRechecksImmediatelyBeforeWriting(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	posts := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts++
			_, _ = io.WriteString(w, `{"number":2}`)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		// Another run filed it while this one was analyzing.
		_, _ = io.WriteString(w, "["+issueJSONTitled(99, "adk-bot", "Bot", issueTitle("v1.1.0"),
			marker+"\n\nfiled by the other run")+"]")
	}))

	n, err := c.FileReleaseIssue(context.Background(), "v1.0.0...v1.1.0", "v1.0.0", "v1.1.0", "t", "b")
	if err != nil {
		t.Fatalf("FileReleaseIssue: %v", err)
	}
	if posts != 0 {
		t.Errorf("made %d create calls after finding an existing issue, want 0", posts)
	}
	if n != 99 {
		t.Errorf("returned issue %d, want the existing 99", n)
	}
}

// A re-check that could not run proves nothing, so the write must not proceed.
func TestFileReleaseIssueAbortsWhenTheRecheckFails(t *testing.T) {
	posts := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts++
			_, _ = io.WriteString(w, `{"number":2}`)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	if _, err := c.FileReleaseIssue(context.Background(), "v1.0.0...v1.1.0", "v1.0.0", "v1.1.0", "t", "b"); err == nil {
		t.Fatal("FileReleaseIssue proceeded despite an unusable re-check")
	}
	if posts != 0 {
		t.Errorf("made %d create calls, want 0", posts)
	}
	if !c.hadError() {
		t.Error("the failure was not recorded, so the run would exit 0")
	}
}

// The dry-run render writes a partly model-authored body to stdout, which the
// GitHub Actions runner parses for workflow commands. A line beginning "::"
// must not reach it intact.
//
// Mutation that must fail this test: drop the escapeWorkflowCommands call.
// The dry-run render writes a partly model-authored body to stdout, which the
// GitHub Actions runner scans for workflow commands. A line beginning "::" must
// not reach it intact.
//
// The oracle here is deliberately INDEPENDENT of the implementation. An earlier
// version of this test split on "\n" and trimmed " \t" — the same predicate
// escapeWorkflowCommands used — so it could not fail for any separator or
// whitespace character the implementation had missed, and it did miss "\r",
// which the runner's line reader treats as a line terminator.
//
// Mutation that must fail this test: revert escapeWorkflowCommands to
// strings.Split(body, "\n") with strings.TrimLeft(l, " \t").
func TestDryRunRenderDefusesWorkflowCommands(t *testing.T) {
	// A hand-written table, one row per property of the runner's command
	// grammar, with the expectation written out rather than computed. The
	// previous version re-split the render with the same regexp the production
	// code compiles and re-used its TrimLeftFunc, which is frozen at today's
	// predicate: it kills a revert but cannot notice the predicate still being
	// incomplete, which is exactly the defect it was written for.
	for _, tc := range []struct {
		name        string
		body        string
		wantEscaped bool
	}{
		{"plain line", "hello\n", false},
		{"mid-line double colon", "std::vector stays\n", false},
		{"command at line start", "::add-mask::secret\n", true},
		{"after a newline", "text\n::error::x\n", true},
		{"after a bare carriage return", "text\r::error::x\n", true},
		{"after a CRLF", "text\r\n::error::x\n", true},
		{"leading spaces", "  ::error::x\n", true},
		{"leading tab", "\t::error::x\n", true},
		{"leading no-break space", "\u00a0::error::x\n", true},
		{"leading ideographic space", "\u3000::error::x\n", true},
		{"stop-commands", "::stop-commands::tok\n", true},
		{"the resuming half", "::tok::\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			cfg.DryRun = true
			c := testClient(t, cfg, failIfCalled(t))
			var rendered strings.Builder
			c.out = &rendered
			if _, err := c.FileReleaseIssue(context.Background(), "v1...v2", "v1", "v2", "t", tc.body); err != nil {
				t.Fatalf("FileReleaseIssue: %v", err)
			}
			got := strings.Contains(rendered.String(), escapedCommandPrefix)
			if got != tc.wantEscaped {
				t.Errorf("escaped=%v, want %v; render was %q", got, tc.wantEscaped, rendered.String())
			}
			// Whatever the verdict, every piece of the text must still be
			// present: the render is a preview and must not lose content.
			for _, want := range strings.FieldsFunc(tc.body, func(r rune) bool {
				return r == '\n' || r == '\r'
			}) {
				if want = strings.TrimSpace(want); want != "" && !strings.Contains(rendered.String(), want) {
					t.Errorf("the render lost %q instead of escaping it: %q", want, rendered.String())
				}
			}
		})
	}
}

// A control character inside a model-authored field is what carries a workflow
// command past a line-based escape, an ANSI sequence into the Actions log, and
// a bidi override into the issue a maintainer reads.
//
// Mutation that must fail this test: drop the stripControls call from neutralize.
func TestNeutralizeStripsControlCharacters(t *testing.T) {
	// One row per category, because unicode.IsControl covers only Cc: the
	// bidirectional marks are Cf, and the line and paragraph separators are
	// Zl/Zp, so each has to be named in the implementation and each is checked
	// here. U+061C is the one most often missed.
	for _, bad := range []string{
		"\r", "\x00", "\x1b", // C0 controls
		"\u0085",                                         // C1 next-line
		"\u061c", "\u200e", "\u200f", "\u202e", "\u2066", // bidi marks and overrides
		"\u2028", "\u2029", // line and paragraph separators
	} {
		if got := neutralize("before" + bad + "after"); strings.Contains(got, bad) {
			t.Errorf("neutralize kept %+q: %+q", bad, got)
		}
	}
	got := neutralize("ok\r::add-mask::x\x1b[31mred\u202eevil\x00")
	for _, bad := range []string{"\r", "\x1b", "\u202e", "\x00"} {
		if strings.Contains(got, bad) {
			t.Errorf("neutralize kept %q: %q", bad, got)
		}
	}
	// Newlines and tabs survive: a multi-line value is legible inside the fence.
	if n := neutralize("a\nb\tc"); n != "a\nb\tc" {
		t.Errorf("neutralize = %q, want newlines and tabs preserved", n)
	}
}

// The re-probe must look for the marker the body actually carries. Re-deriving
// the base by splitting the release key at its first "..." is not the same
// thing: validTag permits a trailing dot, so releaseKey("v1.2.", "v2.0.0") is
// "v1.2....v2.0.0" and the split yields "v1.2". The probe would then search for
// a marker no issue carries, report the release as new, and file a duplicate
// while believing it had re-checked.
//
// Mutation that must fail this test: derive base with strings.Cut(key, "...")
// instead of using the caller's baseTag.
func TestFileReleaseIssueRechecksTheCallersBase(t *testing.T) {
	const base, head = "v1.2.", "v2.0.0"
	key := releaseKey(base, head)
	if !validTag(base) {
		t.Fatalf("test premise: %q must be a tag the bot accepts", base)
	}
	if derived, _, _ := strings.Cut(key, "..."); derived == base {
		t.Fatalf("test premise: splitting %q must disagree with the real base", key)
	}

	marker := bodyMarker(base, head)
	posts := 0
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts++
			_, _ = io.WriteString(w, `{"number":2}`)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/search/") {
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		// A concurrent run already filed this exact release.
		_, _ = io.WriteString(w, "["+issueJSONTitled(88, "adk-bot", "Bot", issueTitle(head),
			marker+"\n\nfiled by the other run")+"]")
	}))

	n, err := c.FileReleaseIssue(context.Background(), key, base, head, "t", "b")
	if err != nil {
		t.Fatalf("FileReleaseIssue: %v", err)
	}
	if posts != 0 {
		t.Errorf("filed a duplicate: %d create calls, want 0", posts)
	}
	if n != 88 {
		t.Errorf("returned issue %d, want the existing 88", n)
	}
}

// GitHub caps a comparison's files array at 300 and does not signal it with a
// Link header, so pagination alone cannot detect that the count is a floor.
//
// Mutation that must fail this test: drop the `len(files) >= compareFileCap`
// check from Compare.
func TestCompareFlagsTheFileCap(t *testing.T) {
	var files []string
	for i := range compareFileCap {
		files = append(files, fmt.Sprintf(`{"filename":"f%d.go","patch":"+x"}`, i))
	}
	cfg := testConfig()
	cfg.MaxFiles = compareFileCap
	c := respondWith(t, cfg, `{"total_commits":1,"files":[`+strings.Join(files, ",")+`],"commits":[]}`)
	diff, err := c.Compare(context.Background(), "v1", "v2")
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if !diff.PageBoundHit {
		t.Error("a comparison at the file cap was not flagged, so the issue would report its count as complete")
	}
	body := buildIssueBody(diff, []Finding{{Summary: "s"}}, analysis{Groups: 1})
	if !strings.Contains(body, "at least this large rather than exact") {
		t.Error("the issue does not disclose that the totals are a floor")
	}

	// Below the cap it must stay false, or every release would claim to be truncated.
	c2 := respondWith(t, cfg, `{"total_commits":1,"files":[{"filename":"a.go","patch":"+x"}],"commits":[]}`)
	small, err := c2.Compare(context.Background(), "v1", "v2")
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if small.PageBoundHit {
		t.Error("a small comparison was flagged as hitting the cap")
	}
}

// total_commits is absent from some compare responses. Without a floor from the
// commits actually fetched, the cap would drop subjects and the issue would say
// nothing.
//
// Mutation that must fail this test: drop the `fetched > totalCommits` floor.
func TestCompareCountsOmittedCommitsWithoutTotal(t *testing.T) {
	cfg := testConfig()
	cfg.MaxCommits = 1
	// No "total_commits" field at all.
	c := respondWith(t, cfg, `{"files":[],"commits":[
		{"sha":"aaaa1111","commit":{"message":"one"}},
		{"sha":"bbbb2222","commit":{"message":"two"}},
		{"sha":"cccc3333","commit":{"message":"three"}}
	]}`)
	diff, err := c.Compare(context.Background(), "v1", "v2")
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if len(diff.Commits) != 1 || diff.OmittedCommits != 2 {
		t.Errorf("commits kept=%d omitted=%d, want 1/2 from the fetched floor",
			len(diff.Commits), diff.OmittedCommits)
	}
}

// Two releases carrying the same numeric version must not make the answer
// depend on the order the API returned them — that dependence is exactly what
// version selection exists to remove.
//
// Mutation that must fail this test: make betterCandidate return false on a tie.
func TestVersionTiesBreakDeterministically(t *testing.T) {
	forward := []Release{{Tag: "v2.0.0"}, {Tag: "v1.2"}, {Tag: "v1.2.0"}}
	reversed := []Release{{Tag: "v2.0.0"}, {Tag: "v1.2.0"}, {Tag: "v1.2"}}
	a, err := previousTag(forward, "v2.0.0")
	if err != nil {
		t.Fatalf("previousTag: %v", err)
	}
	b, err := previousTag(reversed, "v2.0.0")
	if err != nil {
		t.Fatalf("previousTag: %v", err)
	}
	if a != b {
		t.Errorf("previousTag gave %q and %q for the same releases in two orders", a, b)
	}

	tieF := []Release{{Tag: "v1.2"}, {Tag: "v1.2.0"}}
	tieR := []Release{{Tag: "v1.2.0"}, {Tag: "v1.2"}}
	lf, _ := latestTag(tieF)
	lr, _ := latestTag(tieR)
	if lf != lr {
		t.Errorf("latestTag gave %q and %q for the same releases in two orders", lf, lr)
	}
}

// Duplicate detection must survive a maintainer retitling the issue during
// triage, which is an ordinary thing to do. Requiring the title would narrow the
// App-authorship fallback, but it would break the guarantee that actually
// matters: a re-run must not file a second issue for the same release.
//
// Mutation that must fail this test: add `iss.GetTitle() == issueTitle(headTag)`
// to isOwnMarkedIssue.
func TestFindExistingIssueSurvivesARenamedIssue(t *testing.T) {
	marker := bodyMarker("v1.0.0", "v1.1.0")
	renamed := "[" + issueJSONTitled(42, "adk-bot", "Bot", "[docs] v1.1.0 follow-ups",
		marker+"\n\nfiled last week, retitled during triage") + "]"
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/search/") {
			// The search probe queries by the ORIGINAL title, so a renamed issue
			// is invisible to it. The list probe is what has to catch this.
			_, _ = io.WriteString(w, `{"total_count":0,"items":[]}`)
			return
		}
		_, _ = io.WriteString(w, renamed)
	}))
	n, found, err := c.FindExistingIssue(context.Background(), "v1.1.0", marker)
	if err != nil {
		t.Fatalf("FindExistingIssue: %v", err)
	}
	if !found || n != 42 {
		t.Errorf("FindExistingIssue = (%d, %v), want (42, true): a retitled issue would be duplicated", n, found)
	}
}

// The quiet-but-incomplete path files nothing, so coverageNotes never runs and a
// stderr line would be all a maintainer gets — a green check and no signal. The
// annotation is the one place this program writes a workflow command on purpose,
// and its text must be code-authored only.
//
// Mutation that must fail this test: make annotateWarning a no-op.
func TestAnnotateWarningReachesTheActionsUI(t *testing.T) {
	c := testClient(t, testConfig(), failIfCalled(t))
	var out strings.Builder
	c.out = &out

	// Outside Actions it stays quiet: a local run has the log in front of it.
	t.Setenv("GITHUB_ACTIONS", "")
	if err := os.Unsetenv("GITHUB_ACTIONS"); err != nil {
		t.Fatalf("unset: %v", err)
	}
	c.annotateWarning("nothing here")
	if out.Len() != 0 {
		t.Errorf("a local run emitted a workflow command: %q", out.String())
	}

	t.Setenv("GITHUB_ACTIONS", "true")
	c.annotateWarning("release v1...v2 produced no suggestions\nand the analysis was incomplete")
	got := out.String()
	if !strings.HasPrefix(got, "::warning::") {
		t.Errorf("annotation = %q, want it to start with ::warning::", got)
	}
	// One physical line, so the message cannot carry a second command.
	if n := strings.Count(strings.TrimSuffix(got, "\n"), "\n"); n != 0 {
		t.Errorf("the annotation spans %d extra lines: %q", n, got)
	}
	if !strings.Contains(got, "the analysis was incomplete") {
		t.Errorf("the annotation lost its message: %q", got)
	}
}
