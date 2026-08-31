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
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner:          "google",
		Repo:           "adk-go",
		OwnerMap:       map[string]string{"core": "alice", "tools": "bob"},
		RequestContext: true,
		PRCount:        10,
		MaxFiles:       50,
		Concurrency:    1,
	}
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
		selfLogin: "adk-bot",
		log:       discardLogger(),
		eligible:  make(map[int]bool),
		claims:    make(map[claimKey]outcome),
	}
}

// respondWith builds a client whose server answers every request with one canned
// body.
func respondWith(t *testing.T, cfg *Config, body string) *GitHubClient {
	t.Helper()
	return testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
}

const fullPRBody = `{"data":{"repository":{"pullRequest":{
	"number":7,"title":"Add a thing","body":"Because.","state":"OPEN","isDraft":false,
	"author":{"login":"carol","__typename":"User"},
	"assignees":{"nodes":[{"login":"dave"}]},
	"files":{"nodes":[{"path":"agent/agent.go"},{"path":"agent/agent_test.go"}]},
	"comments":{"nodes":[{"author":{"login":"github-actions","__typename":"Bot"},"body":"beep"}]},
	"timelineItems":{"nodes":[{"actor":{"login":"maintainer","__typename":"User"}}]}
}}}}`

func TestFetchPullRequestDecodesEveryField(t *testing.T) {
	c := respondWith(t, testConfig(), fullPRBody)
	pr, err := c.FetchPullRequest(context.Background(), 7)
	if err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if pr.Number != 7 || pr.Title != "Add a thing" || pr.Body != "Because." {
		t.Errorf("basic fields = %+v", pr)
	}
	if pr.State != "OPEN" || pr.IsDraft {
		t.Errorf("state = %q, isDraft = %v, want OPEN/false", pr.State, pr.IsDraft)
	}
	if pr.Author != "carol" {
		t.Errorf("author = %q, want carol", pr.Author)
	}
	if len(pr.Assignees) != 1 || pr.Assignees[0] != "dave" {
		t.Errorf("assignees = %v, want [dave]", pr.Assignees)
	}
	if len(pr.Files) != 2 || pr.Files[0] != "agent/agent.go" {
		t.Errorf("files = %v", pr.Files)
	}
	if len(pr.AssignedBy) != 1 || pr.AssignedBy[0] != "maintainer" {
		t.Errorf("assignedBy = %v, want [maintainer]", pr.AssignedBy)
	}
	// GraphQL returns a bare bot login; the REST-resolved identity carries the
	// suffix, so without canonicalizing here the bot cannot recognize its own
	// past comment or its own past assignment.
	if len(pr.Comments) != 1 || pr.Comments[0].Author != "github-actions[bot]" {
		t.Errorf("comment author = %q, want the canonical github-actions[bot]", pr.Comments[0].Author)
	}
}

// A bot actor on the assignment timeline must be canonicalized too, or the bot
// running as github-actions[bot] would never recognize its own past assignment
// and would re-assign a pull request a maintainer had un-assigned.
func TestFetchPullRequestCanonicalizesTimelineBotActor(t *testing.T) {
	const body = `{"data":{"repository":{"pullRequest":{
		"number":7,"state":"OPEN","author":{"login":"carol","__typename":"User"},
		"timelineItems":{"nodes":[{"actor":{"login":"github-actions","__typename":"Bot"}}]}
	}}}}`
	c := respondWith(t, testConfig(), body)
	pr, err := c.FetchPullRequest(context.Background(), 7)
	if err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if len(pr.AssignedBy) != 1 || pr.AssignedBy[0] != "github-actions[bot]" {
		t.Fatalf("assignedBy = %v, want [github-actions[bot]]", pr.AssignedBy)
	}
	if reason := skipReason(pr, "github-actions[bot]", testOwnerMap()); !strings.Contains(reason, "already assigned by this bot") {
		t.Errorf("skipReason = %q, want the bot to recognize its own prior assignment", reason)
	}
}

// A timeline node with no actor (a deleted account) must not become an empty
// login that then matches an unresolved identity.
func TestFetchPullRequestDropsNullTimelineActors(t *testing.T) {
	const body = `{"data":{"repository":{"pullRequest":{
		"number":7,"state":"OPEN","author":{"login":"carol","__typename":"User"},
		"timelineItems":{"nodes":[{"actor":null},{}]}
	}}}}`
	c := respondWith(t, testConfig(), body)
	pr, err := c.FetchPullRequest(context.Background(), 7)
	if err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if len(pr.AssignedBy) != 0 {
		t.Errorf("assignedBy = %v, want empty", pr.AssignedBy)
	}
}

func TestFetchPullRequestNotFound(t *testing.T) {
	const body = `{"data":{"repository":{"pullRequest":null}}}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchPullRequest(context.Background(), 7); !errors.Is(err, ErrPullRequestNotFound) {
		t.Fatalf("FetchPullRequest() error = %v, want ErrPullRequestNotFound", err)
	}
}

func TestFetchPullRequestNotFoundTypedError(t *testing.T) {
	const body = `{"data":{"repository":{"pullRequest":null}},"errors":[{"type":"NOT_FOUND","message":"Could not resolve"}]}`
	c := respondWith(t, testConfig(), body)
	if _, err := c.FetchPullRequest(context.Background(), 7); !errors.Is(err, ErrPullRequestNotFound) {
		t.Fatalf("FetchPullRequest() error = %v, want ErrPullRequestNotFound", err)
	}
}

// A transient GraphQL error also returns pullRequest:null. Masking it as
// not-found would make a rate-limited run skip everything and exit 0.
func TestFetchPullRequestTransientErrorFailsLoud(t *testing.T) {
	const body = `{"data":{"repository":{"pullRequest":null}},"errors":[{"type":"RATE_LIMITED","message":"rate limited"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchPullRequest(context.Background(), 7)
	if err == nil || errors.Is(err, ErrPullRequestNotFound) || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("FetchPullRequest() error = %v, want a loud graphql error", err)
	}
}

// A null repository means OWNER/REPO could not be resolved at all. Treating that
// as not-found would make a misconfigured bot silently do nothing forever.
func TestFetchPullRequestNullRepositoryFailsLoud(t *testing.T) {
	const body = `{"data":{"repository":null},"errors":[{"type":"NOT_FOUND","message":"Could not resolve to a Repository"}]}`
	c := respondWith(t, testConfig(), body)
	_, err := c.FetchPullRequest(context.Background(), 7)
	if err == nil || errors.Is(err, ErrPullRequestNotFound) {
		t.Fatalf("FetchPullRequest() error = %v, want a loud repository-resolution error", err)
	}
	if !strings.Contains(err.Error(), "google/adk-go") {
		t.Errorf("error %q should name the repository it could not resolve", err)
	}
}

// The file limit is what bounds the attacker-controlled share of the prompt, so
// it has to reach the query rather than being applied only afterwards.
func TestFetchPullRequestSendsTheConfiguredLimits(t *testing.T) {
	cfg := testConfig()
	cfg.MaxFiles = 7
	var got string
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		got = string(raw)
		_, _ = io.WriteString(w, `{"data":{"repository":{"pullRequest":{"number":7,"state":"OPEN"}}}}`)
	}))
	if _, err := c.FetchPullRequest(context.Background(), 7); err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if !strings.Contains(got, `"fileLimit":7`) {
		t.Errorf("request body does not carry fileLimit 7: %s", got)
	}
	if !strings.Contains(got, `"number":7`) {
		t.Errorf("request body does not carry the pull request number: %s", got)
	}
}

func TestListUnassignedPullRequests(t *testing.T) {
	const body = `{"total_count":3,"incomplete_results":false,"items":[
		{"number":1,"pull_request":{"url":"https://api.github.com/repos/google/adk-go/pulls/1"}},
		{"number":2},
		{"number":3,"pull_request":{"url":"https://api.github.com/repos/google/adk-go/pulls/3"}}
	]}`
	var query string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search/issues" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		query = r.URL.Query().Get("q")
		_, _ = io.WriteString(w, body)
	}))
	got, err := c.ListUnassignedPullRequests(context.Background())
	if err != nil {
		t.Fatalf("ListUnassignedPullRequests() error = %v", err)
	}
	// Item 2 is an issue, not a pull request, and must be dropped.
	if len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Errorf("ListUnassignedPullRequests() = %v, want [1 3]", got)
	}
	for _, want := range []string{"is:open", "is:pr", "no:assignee", "draft:false", "repo:google/adk-go"} {
		if !strings.Contains(query, want) {
			t.Errorf("search query %q is missing %q", query, want)
		}
	}
}

func TestListUnassignedPullRequestsHonorsPRCount(t *testing.T) {
	cfg := testConfig()
	cfg.PRCount = 2
	const body = `{"items":[
		{"number":1,"pull_request":{"url":"u"}},
		{"number":2,"pull_request":{"url":"u"}},
		{"number":3,"pull_request":{"url":"u"}}
	]}`
	c := respondWith(t, cfg, body)
	got, err := c.ListUnassignedPullRequests(context.Background())
	if err != nil {
		t.Fatalf("ListUnassignedPullRequests() error = %v", err)
	}
	if len(got) != 2 {
		t.Errorf("returned %d pull requests, want the 2 PR_COUNT allows: %v", len(got), got)
	}
}

func TestIsAssignable(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   bool
	}{
		{http.StatusNoContent, true},
		{http.StatusNotFound, false},
	} {
		c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(tc.status)
		}))
		got, err := c.IsAssignable(context.Background(), "alice")
		if err != nil {
			t.Fatalf("IsAssignable() on HTTP %d error = %v", tc.status, err)
		}
		if got != tc.want {
			t.Errorf("IsAssignable() on HTTP %d = %v, want %v", tc.status, got, tc.want)
		}
	}
	// A server error is neither yes nor no, and must not be read as "assignable".
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	if got, err := c.IsAssignable(context.Background(), "alice"); err == nil || got {
		t.Errorf("IsAssignable() on HTTP 500 = %v, %v; want false and an error", got, err)
	}
}

// The claim is the anti-duplicate guard. It is granted once per (pull request,
// action) and only for a pull request Go cleared.
func TestClaim(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))

	if got := c.claim(7, actionAssign); got != claimNotEligible {
		t.Errorf("claim on an uncleared pull request = %v, want claimNotEligible", got)
	}
	c.markEligible(7)
	if got := c.claim(7, actionAssign); got != claimGranted {
		t.Errorf("first claim = %v, want claimGranted", got)
	}
	if got := c.claim(7, actionAssign); got != claimAlreadyDone {
		t.Errorf("second claim = %v, want claimAlreadyDone", got)
	}
	// The two actions have independent claims.
	if got := c.claim(7, actionComment); got != claimGranted {
		t.Errorf("comment claim = %v, want claimGranted (independent of assign)", got)
	}
	// A failure is remembered as a failure, not as "done".
	c.recordFailure(7, actionComment)
	if got := c.claim(7, actionComment); got != claimAlreadyFailed {
		t.Errorf("claim after a failure = %v, want claimAlreadyFailed", got)
	}
	// Eligibility is per pull request.
	if got := c.claim(8, actionAssign); got != claimNotEligible {
		t.Errorf("claim on a different pull request = %v, want claimNotEligible", got)
	}
}

// Both mutations must pass the dry-run chokepoint. A new mutation that forgets
// shouldSkip would fail here.
func TestDryRunSuppressesEveryMutation(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	var calls int
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	if err := c.AssignOwner(context.Background(), 7, "alice"); err != nil {
		t.Fatalf("AssignOwner() dry-run error = %v", err)
	}
	if err := c.PostComment(context.Background(), 7, "hello"); err != nil {
		t.Fatalf("PostComment() dry-run error = %v", err)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}
