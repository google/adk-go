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
	// identityAttempts and identityRetryDelay bound the retry of the one-off
	// "who am I" lookup. It is fatal when it fails, so a single transient error
	// would otherwise cost a whole scheduled sweep.
	identityAttempts   = 3
	identityRetryDelay = 500 * time.Millisecond
	// searchPageSize bounds one page of REST search results. The GraphQL query
	// below carries its own literal limits; they are not derived from this.
	searchPageSize = 100
)

// botSuffix is what the REST API appends to a GitHub App's login. GraphQL
// returns the bare name, so bot logins are canonicalized to this form before the
// ignore and self filters compare against it.
const botSuffix = "[bot]"

// ErrIssueNotFound is returned when the requested issue does not exist or refers
// to a pull request.
var ErrIssueNotFound = errors.New("issue not found")

// GitHubClient wraps the go-github REST client and adds a raw GraphQL helper,
// the bot's resolved identity, dry-run handling, and run-level error tracking.
// It carries no package-level state so it can be constructed per run and
// unit-tested with an httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger

	// maintainers is the lowercased set of trusted logins, built once from
	// cfg.Maintainers so the concurrent per-issue workers don't each rebuild it.
	maintainers map[string]bool

	// mu guards errored and flagged, which may be touched from tool handlers
	// running concurrently across issues.
	mu sync.Mutex
	// errored records whether any infrastructure error (a failed GitHub
	// mutation, fetch, or agent run) occurred this run, so the program can exit
	// non-zero and a scheduled/CI run fails loudly.
	errored bool
	// flagged records the issues already flagged this run, so a model that emits
	// the flag tool twice for the same issue cannot post a duplicate comment
	// (the label add is idempotent server-side, but a second comment is not).
	flagged map[int]flagOutcome
}

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login (used to ignore the bot's own activity and to
// authenticate its prior alert comments).
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	return newGitHubClient(ctx, github.NewClient(nil).WithAuthToken(cfg.GitHubToken), cfg, log)
}

// newGitHubClient builds a client around an already-constructed REST client, so
// a test can point the identity lookup at an httptest server. Without that seam
// the one line that assigns selfLogin -- the whole basis of "is this comment
// mine?" -- had no test at all, and setting it to "" left the suite green.
func newGitHubClient(ctx context.Context, rest *github.Client, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	c := &GitHubClient{
		rest:        rest,
		cfg:         cfg,
		log:         log,
		flagged:     make(map[int]flagOutcome),
		maintainers: maintainerSet(cfg.Maintainers),
	}

	// The bot must know its own login: everything that answers "is this comment
	// mine?" -- isSelfAuthor, and through it hasBotAlert and the ignore filter
	// -- says no without it, so the bot would feed its own past alerts back to
	// the model as untrusted content and re-alert on an issue whose label a
	// maintainer had deliberately removed. Failing to establish it is therefore
	// FATAL, not best-effort: there is no cheaper guard to fall back to, since
	// trusting the bare "[bot]" suffix would let any installed App suppress
	// moderation.
	//
	// A configured BOT_LOGIN wins and costs no API call. That is the production
	// path: GET /user is user-scoped and GitHub refuses it for the Actions
	// GITHUB_TOKEN, which is an App installation token, so a bot that insisted
	// on resolving its identity that way would fail every scheduled run.
	if cfg.BotLogin != "" {
		c.selfLogin = cfg.BotLogin
		log.Info("using the configured bot identity", "login", c.selfLogin)
		return c, nil
	}

	// No configured login: resolve it, which works for a personal access token.
	// Retried, because one transient 5xx would otherwise cost a whole sweep.
	var login string
	var err error
	for attempt := range identityAttempts {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(attempt) * identityRetryDelay):
			case <-ctx.Done():
				return nil, fmt.Errorf("resolve bot identity: %w (last attempt: %v)", ctx.Err(), err)
			}
		}
		if login, err = fetchLogin(ctx, rest); err == nil {
			break
		}
		if attempt < identityAttempts-1 {
			log.Warn("could not resolve bot identity; retrying", "attempt", attempt+1, "error", err)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("resolve bot identity after %d attempts (set BOT_LOGIN to state it instead; "+
			"the Actions GITHUB_TOKEN cannot call GET /user): %w", identityAttempts, err)
	}
	c.selfLogin = login
	log.Info("resolved bot identity", "login", c.selfLogin)
	return c, nil
}

// fetchLogin returns the login the token authenticates as.
func fetchLogin(ctx context.Context, rest *github.Client) (string, error) {
	idCtx, cancel := context.WithTimeout(ctx, resolveIdentityTimeout)
	defer cancel()
	u, _, err := rest.Users.Get(idCtx, "")
	if err != nil {
		return "", err
	}
	if u.GetLogin() == "" {
		return "", errors.New("the API returned an empty login")
	}
	return u.GetLogin(), nil
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

// flagOutcome is what happened to an issue's flag attempt this run.
type flagOutcome int

const (
	flagUnattempted flagOutcome = iota
	flagInFlight                // claimed, write not yet known to have succeeded
	flagFailed                  // the write returned an error
	flagSucceeded               // the write landed
)

// markFlagged claims the single flag attempt allowed for an issue this run. It
// reports whether this call won the claim, and — when it did not — what the
// previous attempt's outcome was.
//
// Both come out of ONE critical section on purpose. Reading the outcome with a
// second call would reopen a check-then-act window: caller A wins the claim and
// is still inside FlagSpam, caller B loses it, sees flagInFlight, reports
// "already flagged" as a success, and only then does A's write fail. The
// model's transcript would record a success for a comment that was never
// posted, which is exactly what tracking the outcome exists to prevent.
//
// The claim is deliberately NOT rolled back on failure: retrying inside the same
// run risks posting the alert twice if the first write actually landed and only
// its response was lost. A failed flag is recorded as an error (fail-loud) and
// retried on the next scheduled run, by which point the label guard applies.
func (c *GitHubClient) markFlagged(number int) (won bool, previous flagOutcome) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.flagged == nil {
		c.flagged = make(map[int]flagOutcome)
	}
	if prev := c.flagged[number]; prev != flagUnattempted {
		return false, prev
	}
	c.flagged[number] = flagInFlight
	return true, flagUnattempted
}

// recordFlagFailure notes that the claimed flag write failed, so a later call
// for the same issue is told the truth instead of "already flagged".
func (c *GitHubClient) recordFlagFailure(number int) {
	c.setFlagOutcome(number, flagFailed)
}

// recordFlagSuccess notes that the claimed flag write landed, so a later call
// can report the alert as genuinely posted rather than merely claimed.
func (c *GitHubClient) recordFlagSuccess(number int) {
	c.setFlagOutcome(number, flagSucceeded)
}

func (c *GitHubClient) setFlagOutcome(number int, o flagOutcome) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.flagged == nil {
		c.flagged = make(map[int]flagOutcome)
	}
	c.flagged[number] = o
}

// flagState reports an issue's flag outcome without changing it. Read-only, so
// a test can assert the state instead of claiming it.
func (c *GitHubClient) flagState(number int) flagOutcome {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.flagged[number]
}

// SearchSpamCandidates returns up to IssueCount open issues (most recently
// updated first) that do not already carry the spam label, optionally restricted
// to a freshness window. The window filters on update time because spam often
// arrives as a comment on an older issue. Pull requests are excluded.
func (c *GitHubClient) SearchSpamCandidates(ctx context.Context) ([]int, error) {
	query := fmt.Sprintf("repo:%s/%s is:issue state:open -label:%q", c.cfg.Owner, c.cfg.Repo, c.cfg.SpamLabel)
	if c.cfg.FreshnessWindow > 0 {
		// Full RFC3339 timestamp (not date-only) so sub-day windows keep their
		// precision: the GitHub Search API honors updated:>=YYYY-MM-DDTHH:MM:SSZ.
		cutoff := time.Now().UTC().Add(-c.cfg.FreshnessWindow).Format("2006-01-02T15:04:05Z")
		query += " updated:>=" + cutoff
	}
	c.log.Info("searching for spam candidates", "query", query)

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
			return nil, fmt.Errorf("search issues: %w", err)
		}
		for _, issue := range result.Issues {
			if issue.IsPullRequest() {
				continue
			}
			// Dedup: an issue updated mid-pagination can appear on two pages;
			// processing the same number twice risks a duplicate alert.
			n := issue.GetNumber()
			if seen[n] {
				continue
			}
			seen[n] = true
			numbers = append(numbers, n)
			if len(numbers) >= c.cfg.IssueCount {
				c.log.Info("found spam candidates", "count", len(numbers), "capped", true)
				return numbers, nil
			}
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	c.log.Info("found spam candidates", "count", len(numbers), "capped", false)
	return numbers, nil
}

// --- GraphQL plumbing -------------------------------------------------------
//
// The raw shapes mirror the GraphQL response so the client can decode directly
// into them; toIssue maps them onto the domain Issue used for review.

type ghActor struct {
	Login string `json:"login"`
	// Typename is the GraphQL __typename ("Bot" for bot accounts). GitHub's
	// GraphQL API returns a bare bot login (e.g. "github-actions") without the
	// botSuffix suffix that REST appends, so the type is the reliable bot signal.
	Typename string `json:"__typename"`
}

type ghComment struct {
	Author            *ghActor `json:"author"`
	AuthorAssociation string   `json:"authorAssociation"`
	Body              string   `json:"body"`
}

type rawIssue struct {
	Number            int      `json:"number"`
	Title             string   `json:"title"`
	Body              string   `json:"body"`
	Author            *ghActor `json:"author"`
	AuthorAssociation string   `json:"authorAssociation"`
	Labels            struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	} `json:"labels"`
	Comments struct {
		// TotalCount is every comment on the issue, not just the fetched page.
		// The difference is content the model is never shown, and it has to be
		// declared to it rather than silently dropped.
		TotalCount int         `json:"totalCount"`
		Nodes      []ghComment `json:"nodes"`
	} `json:"comments"`
}

func login(a *ghActor) string {
	if a == nil {
		return ""
	}
	// Canonicalize a bot login to the REST botSuffix form. GraphQL returns the
	// bare login (e.g. "github-actions") while selfLogin is resolved via REST
	// ("github-actions[bot]"), so without this the bot would not recognize its
	// own alert comments. Nothing is skipped on the strength of the suffix
	// itself -- see isIgnoredAuthor.
	if a.Typename == "Bot" && !strings.HasSuffix(a.Login, botSuffix) {
		return a.Login + botSuffix
	}
	return a.Login
}

func (r *rawIssue) toIssue() Issue {
	labels := make([]string, 0, len(r.Labels.Nodes))
	for _, l := range r.Labels.Nodes {
		labels = append(labels, l.Name)
	}
	comments := make([]Comment, 0, len(r.Comments.Nodes))
	for _, c := range r.Comments.Nodes {
		comments = append(comments, Comment{Author: login(c.Author), Association: c.AuthorAssociation, Body: c.Body})
	}
	// Never negative: totalCount cannot be below the number of nodes returned,
	// but a malformed or absent field must not produce a nonsense count.
	unfetched := max(r.Comments.TotalCount-len(comments), 0)
	return Issue{
		Number:           r.Number,
		Title:            r.Title,
		Body:             r.Body,
		Author:           login(r.Author),
		Association:      r.AuthorAssociation,
		Labels:           labels,
		Comments:         comments,
		UnfetchedComment: unfetched,
	}
}

const issueQuery = `
query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      number
      title
      body
      author { login __typename }
      authorAssociation
      labels(first: 100) { nodes { name } }
      comments(last: $commentLimit) {
        totalCount
        nodes { author { login __typename } authorAssociation body }
      }
    }
  }
}`

type issueResponse struct {
	Data struct {
		// Repository is a pointer so a null (couldn't resolve the repository,
		// e.g. wrong owner/repo or missing access) is distinguishable from a
		// resolved repository whose issue is null (issue not found / a PR).
		Repository *struct {
			Issue *rawIssue `json:"issue"`
		} `json:"repository"`
	} `json:"data"`
	Errors []struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"errors"`
}

// FetchIssue retrieves an issue and its recent comments in a single GraphQL
// query, issued through the authenticated go-github client (no extra
// dependency). It returns ErrIssueNotFound when the number does not exist or
// refers to a pull request.
func (c *GitHubClient) FetchIssue(ctx context.Context, number int) (Issue, error) {
	body := map[string]any{
		"query": issueQuery,
		"variables": map[string]any{
			"owner":  c.cfg.Owner,
			"name":   c.cfg.Repo,
			"number": number,
			// A bounded window keeps the query cheap. The spam LABEL is the
			// primary idempotency guard (the search excludes -label:spam and
			// alreadyHandled checks it); the bot's own alert comment is only a
			// best-effort secondary signal, so on a thread with more than this
			// many comments after the alert, hasBotAlert may miss it. That can
			// cause a re-alert only if the label was also removed.
			"commentLimit": 100,
		},
	}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return Issue{}, fmt.Errorf("build graphql request: %w", err)
	}
	var out issueResponse
	if _, err := c.rest.Do(ctx, req, &out); err != nil {
		return Issue{}, fmt.Errorf("graphql request: %w", err)
	}
	// A null repository means we could not resolve OWNER/REPO at all (misconfig
	// or missing access). That is an infrastructure error, NOT "issue not found":
	// mapping it to ErrIssueNotFound would make a misconfigured bot skip every
	// issue and exit 0. Surface it so the run fails loudly.
	if out.Data.Repository == nil {
		// A rate limit or a query-complexity failure also nulls the whole data
		// object, so do not assert a cause: name the query and quote what GitHub
		// said. Reporting a transient failure as an OWNER/REPO misconfiguration
		// sends whoever reads the log after the wrong problem.
		msg := "no repository and no error in the response"
		if len(out.Errors) > 0 {
			msg = out.Errors[0].Message
		}
		return Issue{}, fmt.Errorf("query %s/%s issue #%d: %s", c.cfg.Owner, c.cfg.Repo, number, msg)
	}
	// Inspect GraphQL errors BEFORE treating a null issue as not-found. GitHub
	// signals a genuinely missing issue/PR with a NOT_FOUND-typed error (map that
	// to ErrIssueNotFound → skip), but a transient error (rate limit, timeout,
	// query-complexity) also returns issue:null with a DIFFERENT error type — that
	// is infrastructure failure and must fail loud, not be silently skipped.
	//
	// EVERY error must be NOT_FOUND for that to hold. Returning on the first
	// NOT_FOUND anywhere in the list meant a response carrying both NOT_FOUND
	// and a transient error was logged as "issue not found; skipping" with no
	// error recorded, so the run still exited 0.
	if len(out.Errors) > 0 {
		notFound := true
		for _, e := range out.Errors {
			if e.Type != "NOT_FOUND" {
				notFound = false
				break
			}
		}
		if notFound {
			return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
		}
		return Issue{}, fmt.Errorf("graphql error: %s", out.Errors[0].Message)
	}
	// Repository resolved, no errors, but the issue is null: the number does not
	// exist or refers to a pull request (issue() resolves only Issues).
	if out.Data.Repository.Issue == nil {
		return Issue{}, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
	}
	return out.Data.Repository.Issue.toIssue(), nil
}

// --- Mutations (all honor dry-run) ------------------------------------------

// FlagSpam posts the maintainer alert, then applies the spam label. The comment
// is written first because it is the notification that actually matters: if the
// label step then fails, the next run finds the issue unlabeled, hasBotAlert
// recognizes this comment and skips it, so the comment is never duplicated.
// (Labeling first would be worse: a failed comment would leave the issue
// labeled-but-unexplained and excluded from future sweeps, so maintainers would
// never be alerted at all.) Self-recognition needs a resolved identity, which
// newGitHubClient guarantees: it fails the run rather than returning a client
// that cannot tell its own comments apart.
func (c *GitHubClient) FlagSpam(ctx context.Context, number int, comment string) error {
	if err := c.postComment(ctx, number, comment); err != nil {
		return fmt.Errorf("post alert comment: %w", err)
	}
	if err := c.addLabel(ctx, number, c.cfg.SpamLabel); err != nil {
		return fmt.Errorf("add spam label: %w", err)
	}
	return nil
}

// addLabel adds a label to the issue. It is a no-op under dry-run.
func (c *GitHubClient) addLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	if _, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label}); err != nil {
		return err
	}
	c.log.Info("added label", "issue", number, "label", label)
	return nil
}

// postComment posts a comment on the issue. It is a no-op under dry-run.
func (c *GitHubClient) postComment(ctx context.Context, number int, body string) error {
	if c.shouldSkip(number, "post alert comment") {
		return nil
	}
	if _, _, err := c.rest.Issues.CreateComment(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueComment{Body: github.String(body)}); err != nil {
		return err
	}
	c.log.Info("posted alert comment", "issue", number)
	return nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled. It is the single chokepoint every mutation passes
// through, so dry-run is impossible to forget.
func (c *GitHubClient) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "issue", number)
		return true
	}
	return false
}
