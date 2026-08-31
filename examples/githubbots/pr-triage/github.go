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
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v66/github"
)

const (
	// resolveIdentityTimeout bounds the one-off "who am I" lookup at startup, so
	// a hung request cannot stall the run until the workflow's own timeout.
	resolveIdentityTimeout = 10 * time.Second
	// searchPageSize bounds one page of REST search results.
	searchPageSize = 100
	// commentLimit bounds the comments fetched per pull request. They are used
	// only for the "have I already asked for context?" check.
	commentLimit = 100
	// timelineLimit bounds the assignment events fetched per pull request. They
	// establish whether the bot has already had its one assignment turn.
	timelineLimit = 100
	// assigneeLimit bounds the assignees fetched. GitHub allows at most 10.
	assigneeLimit = 10
)

// botSuffix is what the REST API appends to a GitHub App's login. GraphQL
// returns the bare name, so bot logins are canonicalized to this form before
// they are compared with the REST-resolved identity.
const botSuffix = "[bot]"

// ErrPullRequestNotFound is returned when the requested number does not exist or
// does not refer to a pull request.
var ErrPullRequestNotFound = errors.New("pull request not found")

// action is one of the two mutations this bot may perform on a pull request.
// The claim is taken per (pull request, action), so assigning does not consume
// the comment's turn or the other way round.
type action int

const (
	actionAssign action = iota
	actionComment
)

func (a action) String() string {
	if a == actionAssign {
		return "assign"
	}
	return "comment"
}

// outcome is what has happened to one (pull request, action) pair this run.
type outcome int

const (
	outcomeUnattempted outcome = iota
	outcomeClaimed             // claimed; the write is not yet known to have succeeded
	outcomeFailed              // the write returned an error
)

// claimResult is why a claim was or was not granted. The caller maps it onto a
// model-readable message.
type claimResult int

const (
	claimGranted claimResult = iota
	// claimNotEligible means the Go-side preconditions were never satisfied for
	// this pull request, so no mutation is authorized at all.
	claimNotEligible
	claimAlreadyDone
	claimAlreadyFailed
)

// claimKey identifies one (pull request, action) pair.
type claimKey struct {
	number int
	act    action
}

// GitHubClient wraps the go-github REST client and adds a raw GraphQL helper,
// the bot's resolved identity, dry-run handling, the per-action claim, and
// run-level error tracking. It carries no package-level state so it can be
// constructed per run and unit-tested with an httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger

	// mu guards errored, eligible and claims, which are touched from tool
	// handlers running concurrently across pull requests.
	mu sync.Mutex
	// errored records whether any infrastructure error (a failed GitHub
	// mutation, fetch, or agent run) occurred this run, so the program can exit
	// non-zero and a scheduled/CI run fails loudly.
	errored bool
	// eligible records the pull requests whose preconditions the bot verified in
	// Go, from GitHub API metadata, before invoking the model. A pull request
	// absent from this set can never be mutated, whatever the model says.
	eligible map[int]bool
	// claims records the single attempt allowed per (pull request, action).
	claims map[claimKey]outcome
}

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login, which is used to recognize its own past comment
// and its own past assignment.
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	rest := github.NewClient(nil).WithAuthToken(cfg.GitHubToken)
	c := &GitHubClient{
		rest:     rest,
		cfg:      cfg,
		log:      log,
		eligible: make(map[int]bool),
		claims:   make(map[claimKey]outcome),
	}

	// Resolve identity once, under a short timeout so a hanging API call cannot
	// stall startup indefinitely.
	idCtx, cancel := context.WithTimeout(ctx, resolveIdentityTimeout)
	defer cancel()
	if u, _, err := rest.Users.Get(idCtx, ""); err == nil {
		c.selfLogin = u.GetLogin()
		log.Info("resolved bot identity", "login", c.selfLogin)
	} else {
		// Without an identity the bot cannot tell its own past assignment from a
		// maintainer's, so skipReason refuses any pull request that has ever been
		// assigned. That is the fail-safe direction: it triages less, never more.
		log.Warn("could not resolve bot identity; pull requests with any prior assignment will be skipped", "error", err)
	}
	return c, nil
}

// recordError flags that an infrastructure error occurred this run.
func (c *GitHubClient) recordError() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.errored = true
}

// hadError reports whether any infrastructure error occurred this run.
func (c *GitHubClient) hadError() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.errored
}

// markEligible records that this pull request passed the Go-side preconditions
// (skipReason returned ""). It is called by the orchestrator immediately before
// the model runs, and it is the only way any mutation becomes authorized.
func (c *GitHubClient) markEligible(number int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.eligible[number] = true
}

// isEligible reports whether this pull request passed the preconditions.
func (c *GitHubClient) isEligible(number int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.eligible[number]
}

// claim atomically re-checks the Go-computed precondition and takes the single
// attempt allowed for (number, act) this run.
//
// The precondition check and the claim share one critical section on purpose. A
// check-then-act would let two concurrent tool calls both observe "unattempted"
// and both post — assignment happens to be idempotent server-side, but a second
// comment is not.
//
// A claim is deliberately NOT released on failure. Retrying inside the same run
// risks a duplicate comment if the first write actually landed and only its
// response was lost, and for assignment a retry is exactly the loop the spec
// forbids. A failed write is recorded as an error so the run exits non-zero.
func (c *GitHubClient) claim(number int, act action) claimResult {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.eligible[number] {
		return claimNotEligible
	}
	k := claimKey{number: number, act: act}
	switch c.claims[k] {
	case outcomeUnattempted:
		c.claims[k] = outcomeClaimed
		return claimGranted
	case outcomeFailed:
		return claimAlreadyFailed
	default:
		return claimAlreadyDone
	}
}

// markSpent consumes the single attempt for (number, act) without performing
// it, for an action a PREVIOUS run already performed (established from API
// metadata, e.g. the bot's own context-request comment is already on the pull
// request). Routing cross-run idempotency through the same claim table means
// the tools have exactly one place to ask "may I still do this?".
func (c *GitHubClient) markSpent(number int, act action) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.claims[claimKey{number: number, act: act}] = outcomeClaimed
}

// recordFailure notes that a claimed write failed, so a later call for the same
// pair is told the truth instead of "already done" — otherwise the model's
// transcript records an action that never happened.
func (c *GitHubClient) recordFailure(number int, act action) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.claims[claimKey{number: number, act: act}] = outcomeFailed
}

// ListUnassignedPullRequests returns up to PRCount open, non-draft pull requests
// with nobody assigned, most recently updated first. It backs the manual batch
// mode only; there is no scheduled backfill, because one would keep re-offering
// pull requests a maintainer had deliberately un-assigned. The per-pull-request
// timeline check catches those that slip through here.
func (c *GitHubClient) ListUnassignedPullRequests(ctx context.Context) ([]int, error) {
	query := fmt.Sprintf("repo:%s/%s is:open is:pr no:assignee draft:false", c.cfg.Owner, c.cfg.Repo)
	c.log.Info("searching for unassigned pull requests", "query", query)

	opts := &github.SearchOptions{
		Sort:        "updated",
		Order:       "desc",
		ListOptions: github.ListOptions{PerPage: searchPageSize},
	}
	var numbers []int
	seen := make(map[int]bool)
	for {
		result, resp, err := c.rest.Search.Issues(ctx, query, opts)
		if err != nil {
			return nil, fmt.Errorf("search pull requests: %w", err)
		}
		for _, issue := range result.Issues {
			// is:pr should make this redundant; keep it so a search-syntax change
			// cannot turn this bot loose on issues.
			if !issue.IsPullRequest() {
				continue
			}
			// Dedup: a pull request updated mid-pagination can appear on two
			// pages, and processing it twice risks a duplicate comment.
			n := issue.GetNumber()
			if seen[n] {
				continue
			}
			seen[n] = true
			numbers = append(numbers, n)
			if len(numbers) >= c.cfg.PRCount {
				return numbers, nil
			}
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	c.log.Info("found unassigned pull requests", "count", len(numbers))
	return numbers, nil
}

// --- GraphQL plumbing -------------------------------------------------------
//
// The raw shapes mirror the GraphQL response so the client can decode directly
// into them; toPullRequest maps them onto the domain type used for triage.

type ghActor struct {
	Login string `json:"login"`
	// Typename is the GraphQL __typename ("Bot" for bot accounts). GitHub's
	// GraphQL API returns a bare bot login (e.g. "github-actions") without the
	// suffix REST appends, so the type is the reliable bot signal.
	Typename string `json:"__typename"`
}

type ghComment struct {
	Author *ghActor `json:"author"`
	Body   string   `json:"body"`
}

type rawPullRequest struct {
	Number    int      `json:"number"`
	Title     string   `json:"title"`
	Body      string   `json:"body"`
	State     string   `json:"state"`
	IsDraft   bool     `json:"isDraft"`
	Author    *ghActor `json:"author"`
	Assignees struct {
		Nodes []struct {
			Login string `json:"login"`
		} `json:"nodes"`
	} `json:"assignees"`
	Files struct {
		Nodes []struct {
			Path string `json:"path"`
		} `json:"nodes"`
	} `json:"files"`
	Comments struct {
		Nodes []ghComment `json:"nodes"`
	} `json:"comments"`
	TimelineItems struct {
		Nodes []struct {
			Actor *ghActor `json:"actor"`
		} `json:"nodes"`
	} `json:"timelineItems"`
}

func login(a *ghActor) string {
	if a == nil {
		return ""
	}
	// Canonicalize a bot login to the REST form. GraphQL returns the bare login
	// ("github-actions") while selfLogin is resolved via REST
	// ("github-actions[bot]"); without this the bot would not recognize its own
	// past comment or its own past assignment.
	if a.Typename == "Bot" && !strings.HasSuffix(a.Login, botSuffix) {
		return a.Login + botSuffix
	}
	return a.Login
}

func (r *rawPullRequest) toPullRequest() PullRequest {
	assignees := make([]string, 0, len(r.Assignees.Nodes))
	for _, a := range r.Assignees.Nodes {
		assignees = append(assignees, a.Login)
	}
	files := make([]string, 0, len(r.Files.Nodes))
	for _, f := range r.Files.Nodes {
		files = append(files, f.Path)
	}
	comments := make([]Comment, 0, len(r.Comments.Nodes))
	for _, c := range r.Comments.Nodes {
		comments = append(comments, Comment{Author: login(c.Author), Body: c.Body})
	}
	assignedBy := make([]string, 0, len(r.TimelineItems.Nodes))
	for _, t := range r.TimelineItems.Nodes {
		if actor := login(t.Actor); actor != "" {
			assignedBy = append(assignedBy, actor)
		}
	}
	return PullRequest{
		Number:     r.Number,
		Title:      r.Title,
		Body:       r.Body,
		State:      r.State,
		IsDraft:    r.IsDraft,
		Author:     login(r.Author),
		Assignees:  assignees,
		Files:      files,
		Comments:   comments,
		AssignedBy: assignedBy,
	}
}

// pullRequestQuery fetches everything the preconditions and the prompt need in
// one round trip. The diff itself is deliberately not requested: the changed
// paths carry the routing signal at a fraction of the tokens and a fraction of
// the attacker-controlled text.
const pullRequestQuery = `
query($owner: String!, $name: String!, $number: Int!, $fileLimit: Int!, $commentLimit: Int!, $timelineLimit: Int!, $assigneeLimit: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      title
      body
      state
      isDraft
      author { login __typename }
      assignees(first: $assigneeLimit) { nodes { login } }
      files(first: $fileLimit) { nodes { path } }
      comments(last: $commentLimit) { nodes { author { login __typename } body } }
      timelineItems(last: $timelineLimit, itemTypes: [ASSIGNED_EVENT]) {
        nodes { ... on AssignedEvent { actor { login __typename } } }
      }
    }
  }
}`

type pullRequestResponse struct {
	Data struct {
		// Repository is a pointer so a null (could not resolve the repository:
		// wrong owner/repo, or missing access) is distinguishable from a resolved
		// repository whose pull request is null.
		Repository *struct {
			PullRequest *rawPullRequest `json:"pullRequest"`
		} `json:"repository"`
	} `json:"data"`
	Errors []struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"errors"`
}

// FetchPullRequest retrieves a pull request in a single GraphQL query, issued
// through the authenticated go-github client. It returns ErrPullRequestNotFound
// when the number does not exist or refers to an issue.
func (c *GitHubClient) FetchPullRequest(ctx context.Context, number int) (PullRequest, error) {
	body := map[string]any{
		"query": pullRequestQuery,
		"variables": map[string]any{
			"owner":         c.cfg.Owner,
			"name":          c.cfg.Repo,
			"number":        number,
			"fileLimit":     c.cfg.MaxFiles,
			"commentLimit":  commentLimit,
			"timelineLimit": timelineLimit,
			"assigneeLimit": assigneeLimit,
		},
	}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return PullRequest{}, fmt.Errorf("build graphql request: %w", err)
	}
	var out pullRequestResponse
	if _, err := c.rest.Do(ctx, req, &out); err != nil {
		return PullRequest{}, fmt.Errorf("graphql request: %w", err)
	}
	// A null repository means OWNER/REPO could not be resolved at all (misconfig
	// or missing access). That is an infrastructure error, NOT "not found":
	// mapping it to ErrPullRequestNotFound would make a misconfigured bot skip
	// everything and exit 0. Surface it so the run fails loudly.
	if out.Data.Repository == nil {
		msg := "could not resolve repository"
		if len(out.Errors) > 0 {
			msg = out.Errors[0].Message
		}
		return PullRequest{}, fmt.Errorf("resolve repository %s/%s: %s", c.cfg.Owner, c.cfg.Repo, msg)
	}
	// Inspect GraphQL errors BEFORE treating a null pull request as not-found.
	// GitHub signals a genuinely missing number with a NOT_FOUND-typed error (map
	// that to ErrPullRequestNotFound and skip), but a transient error (rate
	// limit, timeout, query complexity) also returns null with a DIFFERENT type —
	// that is infrastructure failure and must fail loud, not be silently skipped.
	if len(out.Errors) > 0 {
		for _, e := range out.Errors {
			if e.Type == "NOT_FOUND" {
				return PullRequest{}, fmt.Errorf("pull request #%d: %w", number, ErrPullRequestNotFound)
			}
		}
		return PullRequest{}, fmt.Errorf("graphql error: %s", out.Errors[0].Message)
	}
	// Repository resolved, no errors, but the pull request is null: the number
	// does not exist or refers to an issue (pullRequest() resolves only those).
	if out.Data.Repository.PullRequest == nil {
		return PullRequest{}, fmt.Errorf("pull request #%d: %w", number, ErrPullRequestNotFound)
	}
	return out.Data.Repository.PullRequest.toPullRequest(), nil
}

// --- Mutations (all honor dry-run) ------------------------------------------

// IsAssignable reports whether a login can be assigned in this repository.
// GitHub only accepts assignees with write or triage access and silently drops
// the rest from an assignee POST, so this is checked first and a negative answer
// is reported as a skip rather than a silent no-op.
func (c *GitHubClient) IsAssignable(ctx context.Context, login string) (bool, error) {
	ok, _, err := c.rest.Issues.IsAssignee(ctx, c.cfg.Owner, c.cfg.Repo, login)
	if err != nil {
		return false, fmt.Errorf("check assignability of %q: %w", login, err)
	}
	return ok, nil
}

// AssignOwner assigns one login to a pull request. Pull requests are issues as
// far as the assignee endpoint is concerned. It is a no-op under dry-run.
func (c *GitHubClient) AssignOwner(ctx context.Context, number int, login string) error {
	if c.shouldSkip(number, "assign @%s", login) {
		return nil
	}
	if _, _, err := c.rest.Issues.AddAssignees(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{login}); err != nil {
		return err
	}
	c.log.Info("assigned owner", "pr", number, "owner", login)
	return nil
}

// PostComment posts a comment on the pull request. It is a no-op under dry-run.
func (c *GitHubClient) PostComment(ctx context.Context, number int, body string) error {
	if c.shouldSkip(number, "post a context-request comment") {
		return nil
	}
	if _, _, err := c.rest.Issues.CreateComment(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueComment{Body: github.String(body)}); err != nil {
		return err
	}
	c.log.Info("posted context-request comment", "pr", number)
	return nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled. It is the single chokepoint every mutation passes
// through, so dry-run is impossible to forget.
func (c *GitHubClient) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "pr", number)
		return true
	}
	return false
}
