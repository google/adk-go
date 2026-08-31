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
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v66/github"
)

// ErrIssueNotFound is returned when the requested issue does not exist.
var ErrIssueNotFound = errors.New("issue not found")

// GitHubClient wraps the go-github REST client and adds a raw GraphQL helper,
// the bot's resolved identity, and dry-run handling. It carries no
// package-level state so it can be constructed per run and unit-tested with an
// httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger

	// toolErrored records whether any tool hit an infrastructure error during
	// the run. Tool errors are also handed back to the model as data, so without
	// this flag the process could exit 0 despite a failed mutation; run() checks
	// it to fail loudly. Guarded by mu because audits run concurrently.
	mu          sync.Mutex
	toolErrored bool

	// claimed records the (issue, action) pairs already performed this run, so a
	// duplicate tool emission cannot repeat a non-idempotent write. The label
	// writes are idempotent; the comments that accompany them are not.
	claimed map[claimKey]bool

	// observed is the IssueState get_issue_state last computed for each issue,
	// keyed by issue number. The destructive tools re-check their mechanical
	// preconditions against it, so those preconditions hold even if the model is
	// steered by injected text in an issue comment. Guarded by mu.
	observed map[int]IssueState
}

// action names a destructive operation for claim bookkeeping.
type action string

const (
	actionMarkStale   action = "marking stale"
	actionClose       action = "closing as stale"
	actionRemoveStale action = "removing the stale label"
	actionAlertEdit   action = "alerting a maintainer"
	actionAddClarify  action = "adding the clarification label"
)

// claimKey identifies one destructive operation on one issue.
type claimKey struct {
	number int
	act    action
}

// recordObservation stores the state get_issue_state computed for an issue, so
// the destructive tools can verify their preconditions against data this
// process derived rather than against whatever the model asserts.
func (c *GitHubClient) recordObservation(number int, st IssueState) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.observed == nil {
		c.observed = make(map[int]IssueState)
	}
	c.observed[number] = st
}

// claimAction atomically tests an issue's observed state against pred and, on
// success, records the (issue, action) pair so no second call can repeat it.
//
// The test and the consume must share ONE critical section. A plain
// check-then-act lets two callers — a duplicate tool emission within a turn, or
// two audits of the same issue number — both read the same unconsumed
// observation and both proceed. The label writes are idempotent but the comments
// are not, so that posts the same stale warning or closing notice twice. This is
// the per-issue claim the siblings already have: spam-detection's markFlagged,
// issue-triage's claimType.
//
// The observation is not restored after a failed write. Retrying inside one run
// risks a duplicate comment when the first write landed and only its response
// was lost; the failure is recorded as a tool error so the run exits non-zero,
// and the next scheduled run re-derives the state from GitHub.
// The claim is per (issue, action), not per issue: STEP 1 legitimately removes
// the stale label AND posts an edit alert off one get_issue_state, and STEP 3
// marks stale AND adds the clarification label. Consuming the whole observation
// would refuse the second, correct call.
func (c *GitHubClient) claimAction(number int, act action, pred func(IssueState) (string, bool)) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	st, ok := c.observed[number]
	if !ok {
		return fmt.Sprintf("call get_issue_state for issue #%d before acting on it", number), false
	}
	key := claimKey{number: number, act: act}
	if c.claimed[key] {
		return fmt.Sprintf("%s was already performed for issue #%d this run", act, number), false
	}
	if msg, ok := pred(st); !ok {
		return msg, false
	}
	if c.claimed == nil {
		c.claimed = make(map[claimKey]bool)
	}
	c.claimed[key] = true
	return "", true
}

// recordToolError flags that a tool hit an infrastructure error this run.
func (c *GitHubClient) recordToolError() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.toolErrored = true
}

// hadToolError reports whether any tool hit an infrastructure error this run.
func (c *GitHubClient) hadToolError() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.toolErrored
}

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login (used to ignore the bot's own activity).
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) (*GitHubClient, error) {
	rest := github.NewClient(nil).WithAuthToken(cfg.GitHubToken)
	c := &GitHubClient{rest: rest, cfg: cfg, log: log}

	c.selfLogin = resolveSelfLogin(ctx, rest, cfg, log)
	return c, nil
}

// resolveSelfLogin returns the login the bot posts under, which is how it
// recognizes its own activity.
//
// The configured value wins, and when it is set no API call is made at all.
// Discovery cannot work where this bot actually runs: GET /user is a
// user-to-server endpoint and the workflow's GITHUB_TOKEN is an installation
// token, so the request is refused on every scheduled run. The empty-login
// fallback does still work, because the bot's comments arrive from a "[bot]"
// account and the suffix filter catches them — but it is load-bearing by
// accident, and it fails the moment the API reports the account without that
// suffix, at which point the bot reads its own stale warning as somebody else's
// activity and clears the label it just applied.
//
// It takes the REST client so a test can drive the real function against a
// transport instead of reconstructing the decision.
func resolveSelfLogin(ctx context.Context, rest *github.Client, cfg *Config, log *slog.Logger) string {
	if cfg.BotLogin != "" {
		log.Info("using the configured bot identity", "login", cfg.BotLogin)
		return cfg.BotLogin
	}
	// Only for a local run under a personal token. Bound the call: without a
	// deadline a hung request stalls startup until the job's own timeout.
	idCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	u, _, err := rest.Users.Get(idCtx, "")
	if err != nil {
		log.Warn("could not resolve the bot identity and BOT_LOGIN is unset, so self-recognition falls back to matching a [bot] suffix; set BOT_LOGIN to the login the bot posts under", "error", err)
		return ""
	}
	log.Info("resolved bot identity", "login", u.GetLogin())
	return u.GetLogin()
}

// newNonce returns an unguessable fence marker for untrusted text.
//
// It fails loud on a CSPRNG error rather than degrading: a predictable nonce
// would let an attacker pre-write the matching closing marker in their comment
// and escape the fence, so a weak nonce is worse than none. This mirrors the
// sibling spam-detection bot.
// newNonce is a variable so a test can force the failure path. A predictable
// nonce lets an attacker pre-write the closing marker and escape the fence, so
// every caller must treat an error as fatal rather than substituting a fallback.
var newNonce = func() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generate nonce: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// maxFencedRunes bounds how much untrusted text reaches the model.
const maxFencedRunes = 4000

// fenceUntrusted wraps user-controlled text in an unguessable fence so the
// model reads it as data.
//
// LastCommentText is the only field of IssueState an attacker controls: every
// other field is computed here from API metadata. STEP 3 of the decision tree
// asks the model to judge that comment's intent, so the text has to reach the
// model — fencing it is what keeps an instruction inside it inert. Empty text
// is left empty so the prompt does not see an empty fence.
// The marker is drawn per issue, not per run. A run-scoped marker is shared by
// every audit in the sweep, so one disclosure -- the model echoing it into its
// final text, which is logged, and Actions logs on a public repo are readable
// live -- would let an attacker close the fence on a later issue in the same
// run. Per issue, a leaked marker is already spent. This matches both siblings.
func fenceUntrusted(s, nonce string) string {
	if s == "" {
		return ""
	}
	if r := []rune(s); len(r) > maxFencedRunes {
		// GitHub allows a 65,536-character comment and the fenced text is
		// replayed into every turn of the issue's session, so an unbounded body
		// is free model spend for anyone who can comment.
		//
		// Keep the head AND the tail, not just the head. GitHub's "Quote reply"
		// pastes the previous comment as a blockquote and puts the cursor
		// underneath, so a maintainer's actual question is routinely the LAST
		// thing in a long comment. Head-only truncation would drop it, and an
		// author could immunize their issue against ever being marked stale by
		// posting a description long enough that every quoted reply overflows.
		// Cut by rune: slicing bytes splits a multibyte rune at the boundary.
		marker := "\n[truncated:" + nonce + "]\n"
		half := (maxFencedRunes - len([]rune(marker))) / 2
		head, tail := string(r[:half]), string(r[len(r)-half:])
		// Resume at a line boundary. Cutting mid-line strips the "> " that makes
		// a quoted block a quote, so the attacker's own words would arrive
		// looking like the maintainer's unquoted prose.
		if i := strings.IndexByte(tail, '\n'); i >= 0 {
			tail = tail[i+1:]
		} else {
			// No line boundary to resume at, so there is no way to keep the tail
			// without possibly stripping the "> " that marked it as quoted. Drop
			// it: a missing tail is a smaller lie than an unquoted one.
			tail = ""
		}
		// The marker carries the per-issue nonce for the same reason the fence
		// does: a fixed literal is a string anyone can write into a comment.
		s = head + marker + tail
	}
	return "[UNTRUSTED:" + nonce + "]\n" + s + "\n[/UNTRUSTED:" + nonce + "]"
}

// SearchOldOpenIssues returns the numbers of open issues created before the
// stale threshold, using the Search API to avoid scanning recent issues. PRs
// are excluded.
//
// Because candidates are restricted to issues older than the stale threshold,
// the silent-edit alert path only ever runs on older issues; a description edit
// on a very recent issue is not detected. A transient Search rate-limit error
// surfaces as an error and aborts this run (the next scheduled run retries).
func (c *GitHubClient) SearchOldOpenIssues(ctx context.Context) ([]int, error) {
	cutoff := time.Now().UTC().Add(-c.cfg.StaleAfter).Format("2006-01-02T15:04:05Z")
	query := fmt.Sprintf("repo:%s/%s is:issue state:open created:<%s", c.cfg.Owner, c.cfg.Repo, cutoff)
	c.log.Info("searching for stale candidates", "query", query)

	opts := &github.SearchOptions{
		Sort:        "created",
		Order:       "asc",
		ListOptions: github.ListOptions{PerPage: 100},
	}
	// Dedupe across pages, as the spam-detection sibling does. The result set can
	// shift between page fetches -- an issue reopened mid-pagination re-enters the
	// state:open set and pushes the ordering along -- so the last item of one page
	// can reappear on the next. Auditing one number twice fans two concurrent
	// goroutines onto the same issue, and while the label writes are idempotent
	// the comments that accompany them are not.
	seen := make(map[int]bool)
	var numbers []int
	for {
		result, resp, err := c.rest.Search.Issues(ctx, query, opts)
		if err != nil {
			return nil, fmt.Errorf("search issues: %w", err)
		}
		for _, issue := range result.Issues {
			if issue.IsPullRequest() {
				continue
			}
			n := issue.GetNumber()
			if seen[n] {
				continue
			}
			seen[n] = true
			numbers = append(numbers, n)
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	c.log.Info("found stale candidates", "count", len(numbers))
	return numbers, nil
}

// graphQLResponse is the envelope returned by the GitHub GraphQL API.
type graphQLResponse struct {
	Data struct {
		Repository struct {
			Issue *rawIssue `json:"issue"`
		} `json:"repository"`
	} `json:"data"`
	Errors []struct {
		Message string `json:"message"`
	} `json:"errors"`
}

const issueHistoryQuery = `
query($owner: String!, $name: String!, $number: Int!, $commentLimit: Int!, $editLimit: Int!, $timelineLimit: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      author { login __typename }
      createdAt
      labels(first: 100) { nodes { name } }
      comments(last: $commentLimit) {
        nodes { author { login __typename } body createdAt lastEditedAt }
      }
      userContentEdits(last: $editLimit) {
        nodes { editor { login __typename } editedAt }
      }
      timelineItems(itemTypes: [LABELED_EVENT, UNLABELED_EVENT, RENAMED_TITLE_EVENT, REOPENED_EVENT], last: $timelineLimit) {
        nodes {
          __typename
          ... on LabeledEvent { createdAt actor { login __typename } label { name } }
          ... on UnlabeledEvent { createdAt actor { login __typename } label { name } }
          ... on RenamedTitleEvent { createdAt actor { login __typename } }
          ... on ReopenedEvent { createdAt actor { login __typename } }
        }
      }
    }
  }
}`

// FetchIssueHistory retrieves an issue's full history in a single GraphQL query,
// issued through the authenticated go-github client (no extra dependency). The
// response decodes into rawIssue (defined in state.go).
func (c *GitHubClient) FetchIssueHistory(ctx context.Context, number int) (*rawIssue, error) {
	body := map[string]any{
		"query": issueHistoryQuery,
		"variables": map[string]any{
			"owner":  c.cfg.Owner,
			"name":   c.cfg.Repo,
			"number": number,
			// Bounded windows keep the query cheap. They are generous enough
			// that the stale LabeledEvent and the bot's own alert comment are
			// usually still in view; computeIssueState degrades gracefully when
			// they are not.
			"commentLimit":  50,
			"editLimit":     10,
			"timelineLimit": 50,
		},
	}
	req, err := c.rest.NewRequest("POST", "graphql", body)
	if err != nil {
		return nil, fmt.Errorf("build graphql request: %w", err)
	}
	var out graphQLResponse
	if _, err := c.rest.Do(ctx, req, &out); err != nil {
		return nil, fmt.Errorf("graphql request: %w", err)
	}
	if len(out.Errors) > 0 {
		return nil, fmt.Errorf("graphql error: %s", out.Errors[0].Message)
	}
	if out.Data.Repository.Issue == nil {
		return nil, fmt.Errorf("issue #%d: %w", number, ErrIssueNotFound)
	}
	return out.Data.Repository.Issue, nil
}

// GetIssueState fetches and analyzes an issue, returning the structured summary
// consumed by the agent.
func (c *GitHubClient) GetIssueState(ctx context.Context, number int) (IssueState, error) {
	raw, err := c.FetchIssueHistory(ctx, number)
	if err != nil {
		return IssueState{}, err
	}
	st := computeIssueState(raw, c.selfLogin, c.cfg.Maintainers, c.cfg.StaleLabel, c.cfg.StaleAfter, c.cfg.CloseAfter, time.Now().UTC())
	// A stale issue whose last actor is not a maintainer and who acted BEFORE
	// the label is a dead end: the close needs a maintainer to have acted last,
	// and the removal needs someone to have come back since the label. Both
	// comparisons are differences of two "days since" values, so they grow in
	// lockstep and the refusal never lapses. The realistic route is a maintainer
	// leaving MAINTAINERS or renaming, which reclassifies their old comment.
	// Nothing is written either way; say so rather than refusing in silence.
	if st.IsStale && st.LastActionRole != string(roleMaintainer) &&
		st.DaysSinceStaleLabel >= 0 &&
		st.DaysSinceLastActorAction > st.DaysSinceStaleLabel+labelRaceGrace &&
		st.DaysSinceAuthorAction >= st.DaysSinceStaleLabel {
		c.log.Warn("stale issue is stuck: nobody has acted since the label and the last actor is not a maintainer, so it can be neither closed nor un-staled; a maintainer must clear the label",
			"issue", number, "last_action_role", st.LastActionRole)
	}
	if st.DaysSinceStaleLabel < 0 {
		// The labelling event is outside the history window, so the close is
		// refused and this issue can never close while that stays true. Say so:
		// otherwise it silently lingers and nobody knows why.
		c.log.Warn("stale label age unknown (labelling event outside the history window); this issue cannot be closed until it re-enters",
			"issue", number)
	}
	return st, nil
}

// --- Mutations (all honor dry-run) ------------------------------------------

// AddLabel adds a label to the issue. It is a no-op under dry-run.
func (c *GitHubClient) AddLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "add label %q", label) {
		return nil
	}
	_, _, err := c.rest.Issues.AddLabelsToIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, []string{label})
	return err
}

// RemoveLabel removes a label from the issue. It is a no-op under dry-run.
func (c *GitHubClient) RemoveLabel(ctx context.Context, number int, label string) error {
	if c.shouldSkip(number, "remove label %q", label) {
		return nil
	}
	_, err := c.rest.Issues.RemoveLabelForIssue(ctx, c.cfg.Owner, c.cfg.Repo, number, label)
	return err
}

// Comment posts a comment on the issue. It is a no-op under dry-run.
func (c *GitHubClient) Comment(ctx context.Context, number int, body string) error {
	if c.shouldSkip(number, "comment") {
		return nil
	}
	_, _, err := c.rest.Issues.CreateComment(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueComment{Body: github.String(body)})
	return err
}

// MarkStale adds the stale label, then posts the warning comment. The label is
// applied first so a failure mid-operation cannot leave the issue commented but
// unlabeled (which would cause a duplicate comment on the next run).
func (c *GitHubClient) MarkStale(ctx context.Context, number int, comment string) error {
	if err := c.AddLabel(ctx, number, c.cfg.StaleLabel); err != nil {
		return fmt.Errorf("add stale label: %w", err)
	}
	if err := c.Comment(ctx, number, comment); err != nil {
		// The label is on and the warning is not, and nothing downstream would
		// ever notice: the next run sees is_stale and refuses to re-mark, so the
		// warning is never retried and the issue is closed CloseAfter days later
		// having promised the author nothing. Take the label back off so the next
		// run redoes both, and report the removal if it also fails — an issue
		// left labelled without its warning is the state to shout about.
		if rmErr := c.RemoveLabel(ctx, number, c.cfg.StaleLabel); rmErr != nil {
			return fmt.Errorf("post stale comment: %w (and the label could not be taken back off: %v)", err, rmErr)
		}
		return fmt.Errorf("post stale comment: %w (the stale label was removed again so the next run can retry)", err)
	}
	return nil
}

// CloseAsStale closes the issue as "not planned" (rather than the default
// "completed") and then posts a closing comment.
//
// The issue is closed before the comment is posted: closing is idempotent and,
// once closed, the issue drops out of the next run's open-issue search, so a
// failed comment can never produce a duplicate closing comment on retry.
func (c *GitHubClient) CloseAsStale(ctx context.Context, number int, comment string) error {
	if !c.shouldSkip(number, "close as not_planned") {
		if _, _, err := c.rest.Issues.Edit(ctx, c.cfg.Owner, c.cfg.Repo, number, &github.IssueRequest{
			State:       github.String("closed"),
			StateReason: github.String("not_planned"),
		}); err != nil {
			return fmt.Errorf("close issue: %w", err)
		}
	}
	if err := c.Comment(ctx, number, comment); err != nil {
		return fmt.Errorf("post closing comment: %w", err)
	}
	return nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled.
func (c *GitHubClient) shouldSkip(number int, format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would "+fmt.Sprintf(format, args...), "issue", number)
		return true
	}
	return false
}
