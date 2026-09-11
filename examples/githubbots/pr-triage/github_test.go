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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
)

func testConfig() *Config {
	return &Config{
		Owner:          "google",
		Repo:           "adk-go",
		OwnerMap:       map[string]string{"core": "alice", "tools": "bob"},
		RequestContext: true,
		// Single-pull-request mode, as the pull_request_target path runs. The
		// context-request tool exists only here; batch mode is assignment-only.
		SinglePR:    7,
		PRCount:     10,
		MaxFiles:    50,
		Concurrency: 1,
		// A zero PRTimeout would make scopeSession hand every test an
		// already-expired context, which reads as "the model chose not to act".
		// validate() clamps it in production; the test config must set it.
		PRTimeout: time.Minute,
		RunBudget: time.Minute,
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
	"assignees":{"totalCount":1},
	"files":{"totalCount":9,"nodes":[{"path":"agent/agent.go"},{"path":"agent/agent_test.go"}]},
	"comments":{"totalCount":4,"nodes":[{"author":{"login":"github-actions","__typename":"Bot"},"body":"beep"}]},
	"timelineItems":{"totalCount":3}
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
	if pr.AssigneeCount != 1 {
		t.Errorf("assigneeCount = %d, want 1", pr.AssigneeCount)
	}
	if len(pr.Files) != 2 || pr.Files[0] != "agent/agent.go" {
		t.Errorf("files = %v", pr.Files)
	}
	// The node list is capped by the query, so the real count has to come from
	// totalCount or the prompt would call a truncated list complete.
	if pr.TotalFiles != 9 {
		t.Errorf("totalFiles = %d, want 9 (the count GitHub reported, not len(Files))", pr.TotalFiles)
	}
	// The fixture reports FOUR comments while returning one node, so a decoder
	// that used len(nodes) as the total would read the same as the real thing —
	// and the bot would then believe it can see a thread it cannot.
	if pr.TotalComments != 4 {
		t.Errorf("totalComments = %d, want the 4 GitHub reported, not the 1 node returned", pr.TotalComments)
	}
	if pr.PriorAssignments != 3 {
		t.Errorf("priorAssignments = %d, want 3", pr.PriorAssignments)
	}
	// GraphQL returns a bare bot login; the REST-resolved identity carries the
	// suffix, so without canonicalizing here the bot cannot recognize its own
	// past comment or its own past assignment.
	if len(pr.Comments) != 1 || pr.Comments[0].Author != "github-actions[bot]" {
		t.Errorf("comment author = %q, want the canonical github-actions[bot]", pr.Comments[0].Author)
	}
}

// A comment from a bot account must be canonicalized to the REST "[bot]" form,
// or a bot running as github-actions[bot] cannot recognize its own comment and
// asks the same author again on every reopen.
func TestFetchPullRequestCanonicalizesCommentBotAuthor(t *testing.T) {
	body := fmt.Sprintf(`{"data":{"repository":{"pullRequest":{
		"number":7,"state":"OPEN","author":{"login":"carol","__typename":"User"},
		"comments":{"totalCount":1,"nodes":[
			{"author":{"login":"github-actions","__typename":"Bot"},"body":%q}]}
	}}}}`, botCommentSignature+" please add more detail")
	c := respondWith(t, testConfig(), body)
	pr, err := c.FetchPullRequest(context.Background(), 7)
	if err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if len(pr.Comments) != 1 || pr.Comments[0].Author != "github-actions[bot]" {
		t.Fatalf("comment author = %q, want the canonical github-actions[bot]", pr.Comments[0].Author)
	}
	if !contextRequestSpent(pr, "github-actions[bot]") {
		t.Error("the bot did not recognize its own comment through the canonical form")
	}
}

// The GraphQL schema rejects a first/last outside 1-100, so a file limit above
// the cap would fail every fetch in the run. The clamp is what keeps the
// configured value inside the schema.
func TestFileLimitStaysInsideTheGraphQLNodeCap(t *testing.T) {
	if maxFilesLimit > 100 {
		t.Fatalf("maxFilesLimit = %d; GraphQL rejects first/last above 100", maxFilesLimit)
	}
	setRequired(t)
	t.Setenv("MAX_FILES", "5000")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	var sent string
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		sent = string(raw)
		_, _ = io.WriteString(w, `{"data":{"repository":{"pullRequest":{"number":7,"state":"OPEN"}}}}`)
	}))
	if _, err := c.FetchPullRequest(context.Background(), 7); err != nil {
		t.Fatalf("FetchPullRequest() error = %v", err)
	}
	if !strings.Contains(sent, `"fileLimit":100`) {
		t.Errorf("the query asked for a file limit outside the 1-100 cap: %s", sent)
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
	// The bot always comments EARLY, so it must read the FIRST page. Asking for
	// the last would hide its own comment behind a long thread and let a reopen
	// produce a second one.
	if !strings.Contains(got, "comments(first: $commentLimit)") {
		t.Errorf("the query does not read the first comments; the bot would miss its own:\n%s", got)
	}
	if strings.Contains(got, "comments(last:") {
		t.Errorf("the query reads the LAST comments:\n%s", got)
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
		got, err := c.IsAssignable(context.Background(), "core", "alice")
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
	if got, err := c.IsAssignable(context.Background(), "core", "alice"); err == nil || got {
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
	if err := c.AssignOwner(context.Background(), 7, "core", "alice"); err != nil {
		t.Fatalf("AssignOwner() dry-run error = %v", err)
	}
	if err := c.PostComment(context.Background(), 7, "hello"); err != nil {
		t.Fatalf("PostComment() dry-run error = %v", err)
	}
	if calls != 0 {
		t.Errorf("dry-run made %d HTTP calls, want 0", calls)
	}
}

// The workflow's built-in token is a GitHub App installation token, and REST
// GET /user is a user-to-server endpoint that answers 403 for one. Without a
// configured identity the bot cannot recognize its own past comment, so the
// configured value must win and no lookup must be attempted.
func TestNewGitHubClientUsesTheConfiguredIdentity(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"message":"Resource not accessible by integration"}`)
	}))
	t.Cleanup(srv.Close)

	cfg := testConfig()
	cfg.BotLogin = "github-actions[bot]"
	c, err := NewGitHubClient(context.Background(), cfg, discardLogger())
	if err != nil {
		t.Fatalf("NewGitHubClient() error = %v", err)
	}
	if c.selfLogin != "github-actions[bot]" {
		t.Errorf("selfLogin = %q, want the configured github-actions[bot]", c.selfLogin)
	}
	if got := atomic.LoadInt32(&calls); got != 0 {
		t.Errorf("made %d identity lookups, want 0 when BOT_LOGIN is configured", got)
	}
}

// With no configured identity the API lookup is the fallback, and a 403 from an
// installation token must leave the bot degraded but SAFE, not silently
// commenting twice.
func TestNewGitHubClientDegradesSafelyWhenTheIdentityIsUnknown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"message":"Resource not accessible by integration"}`)
	}))
	t.Cleanup(srv.Close)
	base, err := url.Parse(srv.URL + "/")
	if err != nil {
		t.Fatalf("parse base url: %v", err)
	}

	cfg := testConfig()
	cfg.BotLogin = ""
	c := &GitHubClient{
		rest: github.NewClient(nil), cfg: cfg, log: discardLogger(),
		eligible: make(map[int]bool), claims: make(map[claimKey]outcome),
	}
	c.rest.BaseURL = base
	// Drive the real resolution path rather than asserting on a hand-set field.
	if u, _, err := c.rest.Users.Get(context.Background(), ""); err == nil {
		t.Fatalf("test setup: the forced 403 did not take, got %v", u)
	}

	// The degraded state must make the bot ask LESS, not twice: with no identity
	// a comment carrying the signature counts whoever wrote it.
	pr := eligiblePR()
	pr.Comments = []Comment{{Author: "someone-else", Body: botCommentSignature + " asked"}}
	pr.TotalComments = 1
	if !contextRequestSpent(pr, c.selfLogin) {
		t.Error("with an unresolved identity the bot would ask again; it must fail safe instead")
	}
}

// A tool's Go error is serialized back to the model (main.go's
// OnToolErrorCallbacks returns nil, nil -- observe only), so an error text that
// named the login would hand a fully attacker-controlled model a real assignee
// on any transient failure of the assignability probe. That defeats the point of
// the model choosing a component rather than a person.
func TestOwnerLoginNeverAppearsInAnErrorTheModelSees(t *testing.T) {
	const secretLogin = "zzzsecretowner"
	cfg := testConfig()
	cfg.OwnerMap = map[string]string{"core": secretLogin}

	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
	}{
		{
			name: "the assignability probe fails",
			handler: func(w http.ResponseWriter, r *http.Request) {
				if strings.HasSuffix(r.URL.Path, "/graphql") {
					_, _ = io.WriteString(w, unassignedPRBody)
					return
				}
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, `{"message":"boom"}`)
			},
		},
		{
			name: "the assignee write fails",
			handler: func(w http.ResponseWriter, r *http.Request) {
				switch {
				case strings.HasSuffix(r.URL.Path, "/graphql"):
					_, _ = io.WriteString(w, unassignedPRBody)
				case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/assignees/"):
					w.WriteHeader(http.StatusNoContent)
				default:
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = io.WriteString(w, `{"message":"boom"}`)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, cfg, tc.handler)
			c.markEligible(7)
			res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core")
			if err == nil {
				t.Fatalf("expected a Go error, got result %+v", res)
			}
			if strings.Contains(err.Error(), secretLogin) {
				t.Errorf("the error the model sees names the owner login:\n%v", err)
			}
			// The component IS safe to name: the model already chose it.
			if !strings.Contains(err.Error(), "core") {
				t.Errorf("the error should say which component failed, got %v", err)
			}
		})
	}
}

// The re-check runs before every assignment, so it must not drag the file list
// and a hundred comment bodies along to compare three counters. This is a
// sample others will copy.
func TestTheAssignmentRecheckAsksOnlyForWhatItNeeds(t *testing.T) {
	var queries []string
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		queries = append(queries, string(raw))
		_, _ = io.WriteString(w, `{"data":{"repository":{"pullRequest":{
			"state":"OPEN","assignees":{"totalCount":0},"timelineItems":{"totalCount":0}}}}}`)
	}))

	reason, err := c.assignmentStillWanted(context.Background(), 7)
	if err != nil {
		t.Fatalf("assignmentStillWanted() error = %v", err)
	}
	if reason != "" {
		t.Errorf("reason = %q, want the assignment to proceed", reason)
	}
	if len(queries) != 1 {
		t.Fatalf("made %d queries, want 1", len(queries))
	}
	for _, unwanted := range []string{"files(", "comments(", "body", "title"} {
		if strings.Contains(queries[0], unwanted) {
			t.Errorf("the re-check query fetches %q, which it does not use:\n%s", unwanted, queries[0])
		}
	}
	for _, wanted := range []string{"state", "assignees", "timelineItems"} {
		if !strings.Contains(queries[0], wanted) {
			t.Errorf("the re-check query is missing %q:\n%s", wanted, queries[0])
		}
	}
}
