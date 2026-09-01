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
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner:       "google",
		Repo:        "adk-go",
		SpamLabel:   "spam",
		IssueCount:  3,
		Concurrency: 1,
	}
}

// respondWith builds a client whose server answers every request with one
// canned body. Most tests only need that, and spelling out an http.HandlerFunc
// at each call site buried the body being asserted on.
func respondWith(t *testing.T, cfg *Config, body string) *GitHubClient {
	t.Helper()
	return testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
}

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
		selfLogin: "spam-bot",
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
}

func TestFetchIssueGraphQLErrorFailsLoud(t *testing.T) {
	// A transient GraphQL error returns issue:null WITH an error. It must surface
	// as a real error (fail loud), not be masked as ErrIssueNotFound (which would
	// silently skip the issue and exit 0).
	const body = `{"data":{"repository":{"issue":null}},"errors":[{"message":"rate limited"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 5)
	if err == nil || errors.Is(err, ErrIssueNotFound) || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("FetchIssue() error = %v, want a graphql error (not ErrIssueNotFound)", err)
	}
}

func TestFetchIssueCanonicalizesBotLogin(t *testing.T) {
	// GraphQL returns a bare bot login ("github-actions"); toIssue must canonicalize
	// it to the REST "[bot]" form so the ignore filter and self-identity match.
	const body = `{"data":{"repository":{"issue":{
		"number":5,"title":"t","body":"b",
		"author":{"login":"alice","__typename":"User"},
		"authorAssociation":"NONE","labels":{"nodes":[]},
		"comments":{"nodes":[
			{"author":{"login":"github-actions","__typename":"Bot"},"authorAssociation":"NONE","body":"beep"}
		]}}}}}`
	c := respondWith(t, testConfig(), body)
	iss, err := c.FetchIssue(context.Background(), 5)
	if err != nil {
		t.Fatalf("FetchIssue() error = %v", err)
	}
	if len(iss.Comments) != 1 || iss.Comments[0].Author != "github-actions[bot]" {
		t.Errorf("bot comment author = %q, want canonical %q", iss.Comments[0].Author, "github-actions[bot]")
	}
}

func TestSearchSpamCandidatesExcludesPRs(t *testing.T) {
	const body = `{"total_count":3,"incomplete_results":false,"items":[
		{"number":1},
		{"number":2,"pull_request":{"url":"https://api.github.com/repos/google/adk-go/pulls/2"}},
		{"number":3}
	]}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search/issues" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))
	got, err := c.SearchSpamCandidates(context.Background())
	if err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	want := []int{1, 3}
	if len(got) != len(want) || got[0] != 1 || got[1] != 3 {
		t.Errorf("SearchSpamCandidates() = %v, want %v (PR excluded)", got, want)
	}
}

func TestSearchSpamCandidatesRespectsCount(t *testing.T) {
	const body = `{"total_count":3,"incomplete_results":false,"items":[
		{"number":1},{"number":2},{"number":3}
	]}`
	cfg := testConfig()
	cfg.IssueCount = 1
	c := respondWith(t, cfg, body)
	got, err := c.SearchSpamCandidates(context.Background())
	if err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("SearchSpamCandidates() = %v, want [1] (count cap)", got)
	}
}

func TestSearchSpamCandidatesQueryExcludesLabelAndFreshness(t *testing.T) {
	cfg := testConfig()
	cfg.FreshnessWindow = 24 * time.Hour
	var gotQuery string
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, `{"items":[]}`)
	}))
	if _, err := c.SearchSpamCandidates(context.Background()); err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	if !strings.Contains(gotQuery, `-label:"spam"`) {
		t.Errorf("query %q missing -label:\"spam\"", gotQuery)
	}
	if !strings.Contains(gotQuery, "updated:>=") {
		t.Errorf("query %q missing freshness filter", gotQuery)
	}
	// Full RFC3339 timestamp, not date-only, so sub-day windows keep precision.
	if !strings.Contains(gotQuery, "T") || !strings.Contains(gotQuery, "Z") {
		t.Errorf("query %q freshness cutoff is not a full datetime", gotQuery)
	}
}

func TestFetchIssueFound(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"number":42,"title":"t","body":"b","author":{"login":"alice"},"authorAssociation":"FIRST_TIME_CONTRIBUTOR",
		"labels":{"nodes":[{"name":"bug"}]},
		"comments":{"nodes":[{"author":{"login":"bob"},"authorAssociation":"NONE","body":"hi"},{"author":null,"body":"ghost"}]}
	}}}}`
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/graphql" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))
	iss, err := c.FetchIssue(context.Background(), 42)
	if err != nil {
		t.Fatalf("FetchIssue() error = %v", err)
	}
	if iss.Number != 42 || iss.Author != "alice" || iss.Title != "t" || iss.Body != "b" {
		t.Errorf("unexpected issue: %+v", iss)
	}
	if iss.Association != "FIRST_TIME_CONTRIBUTOR" {
		t.Errorf("issue association = %q, want FIRST_TIME_CONTRIBUTOR", iss.Association)
	}
	if len(iss.Labels) != 1 || iss.Labels[0] != "bug" {
		t.Errorf("labels = %v, want [bug]", iss.Labels)
	}
	if len(iss.Comments) != 2 || iss.Comments[0].Author != "bob" || iss.Comments[1].Author != "" {
		t.Errorf("comments = %+v (want bob + empty-author ghost)", iss.Comments)
	}
	if iss.Comments[0].Association != "NONE" {
		t.Errorf("comment association = %q, want NONE", iss.Comments[0].Association)
	}
}

func TestFetchIssueNotFoundNull(t *testing.T) {
	const body = `{"data":{"repository":{"issue":null}}}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchIssue(context.Background(), 999); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("FetchIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestFetchIssueNotFoundError(t *testing.T) {
	const body = `{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND",` +
		`"message":"Could not resolve to an Issue with the number of 1005."}]}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchIssue(context.Background(), 1005); !errors.Is(err, ErrIssueNotFound) {
		t.Fatalf("FetchIssue() error = %v, want ErrIssueNotFound", err)
	}
}

func TestFetchIssueRepoNotFoundIsRealError(t *testing.T) {
	// A null repository (wrong OWNER/REPO or missing access) must surface as a
	// real error, NOT ErrIssueNotFound, so a misconfigured bot fails loudly
	// instead of silently skipping every issue and exiting 0.
	const body = `{"data":{"repository":null},"errors":[{"message":` +
		`"Could not resolve to a Repository with the name 'google/nope'."}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 1)
	if err == nil {
		t.Fatal("FetchIssue() on null repository expected error, got nil")
	}
	if errors.Is(err, ErrIssueNotFound) {
		t.Errorf("FetchIssue() on null repository = ErrIssueNotFound, want a real (fail-loud) error: %v", err)
	}
	if !strings.Contains(err.Error(), "Could not resolve to a Repository") {
		t.Errorf("error %v should carry the underlying GraphQL message", err)
	}
}

func TestFetchIssueGraphQLError(t *testing.T) {
	const body = `{"errors":[{"message":"Something went wrong"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "Something went wrong") {
		t.Fatalf("FetchIssue() error = %v, want graphql error propagated", err)
	}
}

func TestFlagSpamCommentsThenLabels(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	var commentBody string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		if strings.HasSuffix(r.URL.Path, "/comments") {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			commentBody = string(b)
			mu.Unlock()
			_, _ = io.WriteString(w, `{"id":1}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"spam"}]`)
	}))
	if err := c.FlagSpam(context.Background(), 7, buildAlertComment("promo link")); err != nil {
		t.Fatalf("FlagSpam() error = %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("made %d calls, want 2 (comment + label): %v", len(paths), paths)
	}
	// The alert comment must be posted before the label: it is the notification,
	// and posting it first keeps a failed label from silently dropping the alert.
	if !strings.HasSuffix(paths[0], "/issues/7/comments") {
		t.Errorf("first call = %s, want comments endpoint", paths[0])
	}
	if !strings.HasSuffix(paths[1], "/issues/7/labels") {
		t.Errorf("second call = %s, want labels endpoint", paths[1])
	}
	if !strings.Contains(commentBody, "Automated spam detection") {
		t.Errorf("comment body missing signature: %s", commentBody)
	}
}

func TestFlagSpamDryRunMakesNoCalls(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	var calls atomic.Int32
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{}`)
	}))
	if err := c.FlagSpam(context.Background(), 1, buildAlertComment("x")); err != nil {
		t.Fatalf("FlagSpam() dry-run error = %v", err)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", got)
	}
}

// restClient builds a go-github client pointed at an httptest server, so the
// identity lookup and the maintainer wiring can both be driven for real.
func restClient(t *testing.T, h http.Handler) *github.Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return rest
}

// A response mixing NOT_FOUND with a transient error must fail loud. Returning
// on the first NOT_FOUND anywhere in the list logged a rate-limited fetch as
// "issue not found; skipping" and let the run exit 0.
func TestFetchIssueMixedErrorsFailLoud(t *testing.T) {
	const body = `{"data":{"repository":{"issue":null}},"errors":[` +
		`{"type":"RATE_LIMITED","message":"rate limited"},` +
		`{"type":"NOT_FOUND","message":"Could not resolve to an Issue"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchIssue(context.Background(), 5)
	if err == nil {
		t.Fatal("FetchIssue() on a mixed error list returned no error")
	}
	if errors.Is(err, ErrIssueNotFound) {
		t.Errorf("FetchIssue() = ErrIssueNotFound, want a fail-loud error: %v", err)
	}
}

// An issue updated mid-pagination can appear on two pages. Processing the same
// number twice risks a duplicate alert, so the sweep dedups by number.
func TestSearchSpamCandidatesDedupsAcrossPages(t *testing.T) {
	cfg := testConfig()
	cfg.IssueCount = 10
	var page atomic.Int32
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if page.Add(1) == 1 {
			// A Link header is how go-github learns there is a second page.
			w.Header().Set("Link", `<`+r.URL.Path+`?page=2>; rel="next", <`+r.URL.Path+`?page=2>; rel="last"`)
			_, _ = io.WriteString(w, `{"items":[{"number":1},{"number":2}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"items":[{"number":2},{"number":3}]}`)
	}))
	got, err := c.SearchSpamCandidates(context.Background())
	if err != nil {
		t.Fatalf("SearchSpamCandidates() error = %v", err)
	}
	if page.Load() < 2 {
		t.Fatalf("only %d page(s) were fetched; the test never reached the duplicate", page.Load())
	}
	want := []int{1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("SearchSpamCandidates() = %v, want %v (issue 2 appears on both pages)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SearchSpamCandidates() = %v, want %v", got, want)
		}
	}
}

// GraphQL returns a bare bot login, but not always: a Bot actor whose login
// already ends in "[bot]" must not become "…[bot][bot]", which would stop
// matching selfLogin and make the bot fail to recognize its own alerts.
func TestFetchIssueDoesNotDoubleSuffixABotLogin(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"number":5,"title":"t","body":"b",
		"author":{"login":"alice","__typename":"User"},
		"authorAssociation":"NONE","labels":{"nodes":[]},
		"comments":{"nodes":[
			{"author":{"login":"github-actions[bot]","__typename":"Bot"},"authorAssociation":"NONE","body":"beep"}
		]}}}}}`
	c := respondWith(t, testConfig(), body)
	iss, err := c.FetchIssue(context.Background(), 5)
	if err != nil {
		t.Fatalf("FetchIssue() error = %v", err)
	}
	if got := iss.Comments[0].Author; got != "github-actions[bot]" {
		t.Errorf("bot comment author = %q, want %q unchanged", got, "github-actions[bot]")
	}
}

// Two invariants rest on this constructor, and neither had a test that could
// observe the real thing.
//
// "Never review a maintainer" rests entirely on the maintainers assignment:
// every other test passes its own set, so deleting that one line left a nil map
// and the whole invariant silently off.
//
// "Is this comment mine?" rests entirely on the selfLogin assignment. Setting it
// to "" used to leave the suite green, because every other test supplies the
// login by hand.
func TestNewGitHubClientResolvesIdentityAndWiresMaintainers(t *testing.T) {
	cfg := testConfig()
	cfg.Maintainers = []string{"Wolo-Lab", "dpasiukevich"}
	rest := restClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/user" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"login":"github-actions[bot]"}`)
	}))

	c, err := newGitHubClient(context.Background(), rest, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("newGitHubClient: %v", err)
	}
	if c.selfLogin != "github-actions[bot]" {
		t.Errorf("selfLogin = %q, want the login the API returned", c.selfLogin)
	}
	// Probed through the real predicate with the login as GitHub reports it.
	// Looking it up in the map after applying production's own strings.ToLower
	// would make the case-insensitivity claim vacuous.
	for _, login := range []string{"wolo-lab", "WOLO-LAB", "Wolo-Lab", "dpasiukevich"} {
		if !isIgnoredAuthor(login, c.maintainers) {
			t.Errorf("maintainer %q is not trusted; their comments would be reviewed as spam", login)
		}
	}
	if isIgnoredAuthor("someone-else", c.maintainers) {
		t.Error("a login that is not a maintainer was treated as trusted")
	}
	// The resolved identity must make the bot recognize its own alert.
	own := Issue{Comments: []Comment{{Author: "github-actions[bot]", Body: buildAlertComment("x")}}}
	if !hasBotAlert(own, c.selfLogin) {
		t.Error("the bot does not recognize its own alert, so it would re-alert on every sweep")
	}
}

// An unresolved identity is fatal. It used to be best-effort, which left
// isSelfAuthor answering "not mine" for everything: the bot would then feed its
// own past alerts back to the model as untrusted content and re-alert on an
// issue whose label a maintainer had removed.
func TestNewGitHubClientFailsWhenTheIdentityCannotBeResolved(t *testing.T) {
	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
	}{
		{"api error", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"message":"Bad credentials"}`)
		}},
		{"empty login", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, `{"login":""}`)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rest := restClient(t, tc.handler)
			c, err := newGitHubClient(context.Background(), rest, testConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))
			if err == nil {
				t.Fatalf("newGitHubClient succeeded with selfLogin=%q; the bot would not recognize its own comments", c.selfLogin)
			}
		})
	}
}

// BOT_LOGIN is the production path: the Actions GITHUB_TOKEN is an App
// installation token and GitHub refuses GET /user for it, so a bot that
// insisted on resolving its identity through the API would fail every run.
func TestNewGitHubClientPrefersTheConfiguredLogin(t *testing.T) {
	cfg := testConfig()
	cfg.BotLogin = "github-actions[bot]"
	var apiCalls atomic.Int32
	rest := restClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		apiCalls.Add(1)
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"message":"Resource not accessible by integration"}`)
	}))

	c, err := newGitHubClient(context.Background(), rest, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("newGitHubClient with BOT_LOGIN set: %v", err)
	}
	if c.selfLogin != "github-actions[bot]" {
		t.Errorf("selfLogin = %q, want the configured login", c.selfLogin)
	}
	if got := apiCalls.Load(); got != 0 {
		t.Errorf("made %d identity API call(s) with BOT_LOGIN set; GET /user is refused for the Actions token", got)
	}
}

// The lookup path is for a personal access token, and one transient failure
// must not cost the whole sweep.
func TestNewGitHubClientRetriesTheIdentityLookup(t *testing.T) {
	var attempts atomic.Int32
	rest := restClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, `{"message":"bad gateway"}`)
			return
		}
		_, _ = io.WriteString(w, `{"login":"karol-pat"}`)
	}))

	c, err := newGitHubClient(context.Background(), rest, testConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("newGitHubClient did not retry a transient failure: %v", err)
	}
	if got := attempts.Load(); got < 2 {
		t.Fatalf("made %d attempt(s); the test never reached the retry", got)
	}
	if c.selfLogin != "karol-pat" {
		t.Errorf("selfLogin = %q, want the login from the successful retry", c.selfLogin)
	}
}

// A context cancelled DURING the backoff must abort the retry loop rather than
// sleeping the rest of it out. Cancelling before the loop proves nothing: the
// request then fails instantly on every attempt, so the loop finishes fast
// either way. The cancellation has to land mid-sleep.
func TestNewGitHubClientStopsRetryingWhenTheContextIsDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	rest := restClient(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, `{"message":"bad gateway"}`)
	}))
	// Well inside the first backoff, which is identityRetryDelay.
	time.AfterFunc(identityRetryDelay/5, cancel)

	start := time.Now()
	_, err := newGitHubClient(ctx, rest, testConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("newGitHubClient succeeded on a cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error %v does not report the cancellation", err)
	}
	// The load-bearing part is that the backoff WAKES on cancellation: replace
	// the select with a plain time.Sleep and this fails. The early return
	// inside the ctx arm is not separately observable, because the select
	// already wakes on ctx.Done() and the retried request then fails instantly
	// on the cancelled context.
	// A multiple, not the delay itself: the assertion needs enough slack for one
	// httptest round trip and a goroutine wake on a loaded runner, while still
	// being far below the ~1.5s a slept-through backoff would take.
	if budget := 2 * identityRetryDelay; elapsed > budget {
		t.Errorf("returned after %v, want under %v: the retry loop slept through its backoff", elapsed, budget)
	}
}
