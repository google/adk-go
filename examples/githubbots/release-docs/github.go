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
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v66/github"
)

const (
	// resolveIdentityTimeout bounds the one-off "who am I" lookup at startup, so
	// a hung request cannot stall the run until the workflow's own timeout.
	resolveIdentityTimeout = 10 * time.Second
	// listPageSize is one page of a REST list.
	listPageSize = 100
	// dedupPages bounds the immediately-consistent duplicate scan. The bot files
	// at most one issue per release, so this covers the last dedupPages*100
	// issues in the target repository; older ones are covered by the search
	// probe instead. See FindExistingIssue.
	dedupPages = 3
	// releasePages bounds the release listing used to derive the previous tag.
	releasePages = 2
	// comparePages bounds the paginated compare result. GitHub returns at most
	// 300 files across the pages of one comparison.
	comparePages = 3
	// createIssueTimeout bounds the single mutation. It is separate from the
	// analysis budget so an exhausted budget still gets to file its partial
	// findings instead of losing the whole run.
	createIssueTimeout = 60 * time.Second
)

// botTypeLogin is the GitHub account type of a GitHub App, as reported by the
// REST API's user object.
const botTypeLogin = "Bot"

// ErrNoPreviousRelease is returned when a base tag was not supplied and no
// earlier release exists to diff against.
var ErrNoPreviousRelease = errors.New("no previous release to compare against")

// GitHubClient wraps the go-github REST client and adds the bot's resolved
// identity, dry-run handling, the per-release claim, and run-level error
// tracking. It carries no package-level state so it can be constructed per run
// and unit-tested with an httptest server.
type GitHubClient struct {
	rest      *github.Client
	cfg       *Config
	selfLogin string
	log       *slog.Logger
	// out is where a dry run renders the issue it would have filed. It is a
	// field so a test can capture the render and assert it happened.
	out io.Writer

	// mu guards errored and filed.
	mu sync.Mutex
	// errored records whether any infrastructure error occurred this run, so the
	// program can exit non-zero and a scheduled/CI run fails loudly.
	errored bool
	// filed records the release keys already claimed this run, so two callers
	// cannot both pass the duplicate check on one observation and file twice.
	filed map[string]fileOutcome
}

// fileOutcome is what happened to a release's issue-creation attempt this run.
type fileOutcome int

const (
	fileUnattempted fileOutcome = iota
	fileInFlight                // claimed, write not yet known to have succeeded
	fileFailed                  // the write returned an error
)

// NewGitHubClient builds a client authenticated with the configured token and
// resolves the bot's own login, which the duplicate probes use to tell the bot's
// own issues from anybody else's.
func NewGitHubClient(ctx context.Context, cfg *Config, log *slog.Logger) *GitHubClient {
	c := &GitHubClient{
		rest:  github.NewClient(nil).WithAuthToken(cfg.GitHubToken),
		cfg:   cfg,
		log:   log,
		out:   os.Stdout,
		filed: make(map[string]fileOutcome),
	}

	// Resolve identity once, under a short timeout so a hanging API call cannot
	// stall startup. The built-in Actions GITHUB_TOKEN is an app installation
	// token and this call fails for it; that is expected and handled by
	// trustedCreator, which then falls back to the account type.
	idCtx, cancel := context.WithTimeout(ctx, resolveIdentityTimeout)
	defer cancel()
	if u, _, err := c.rest.Users.Get(idCtx, ""); err == nil {
		c.selfLogin = u.GetLogin()
		log.Info("resolved bot identity", "login", c.selfLogin)
	} else {
		log.Warn("could not resolve bot identity; duplicate detection will accept any App-authored issue "+
			"carrying the marker", "error", err)
	}
	return c
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

// claimRelease claims the single issue-creation attempt allowed for a release
// key this run, and reports whether this call won the claim.
//
// The claim is taken in the same critical section that reads the previous
// outcome: a check-then-act would let two callers pass on one observation and
// file two issues. It is deliberately NOT rolled back on failure -- retrying
// inside the same run risks a duplicate if the first write actually landed and
// only its response was lost. A failed attempt is recorded as an error
// (fail-loud) and retried on the next run, by which point the duplicate probes
// see the issue that did land.
func (c *GitHubClient) claimRelease(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.filed[key] != fileUnattempted {
		return false
	}
	c.filed[key] = fileInFlight
	return true
}

// recordFileFailure notes that the claimed write failed, so a later call for the
// same release is told the truth instead of "already filed".
func (c *GitHubClient) recordFileFailure(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.filed[key] = fileFailed
}

// fileAttemptFailed reports whether this run already tried and failed to file
// the issue for a release.
func (c *GitHubClient) fileAttemptFailed(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.filed[key] == fileFailed
}

// --- Release resolution -----------------------------------------------------

// ResolveTags determines the tag pair to diff. An explicitly configured tag is
// used as given; an empty EndTag resolves to the most recent published release
// and an empty StartTag to the release published immediately before it.
//
// Draft releases are skipped: they are not published and diffing against one
// would compare code nobody has shipped.
func (c *GitHubClient) ResolveTags(ctx context.Context) (base, head string, err error) {
	if c.cfg.StartTag != "" && c.cfg.EndTag != "" {
		return c.cfg.StartTag, c.cfg.EndTag, nil
	}
	tags, err := c.publishedTags(ctx)
	if err != nil {
		return "", "", err
	}
	head = c.cfg.EndTag
	if head == "" {
		if len(tags) == 0 {
			return "", "", errors.New("no published releases found")
		}
		head = tags[0]
	}
	base = c.cfg.StartTag
	if base == "" {
		base, err = previousTag(tags, head)
		if err != nil {
			return "", "", err
		}
	}
	return base, head, nil
}

// publishedTags lists non-draft release tags, most recently created first, and
// drops any tag that fails the allow-list so a malformed tag from the API can
// never reach a URL path.
func (c *GitHubClient) publishedTags(ctx context.Context) ([]string, error) {
	opts := &github.ListOptions{PerPage: listPageSize}
	var tags []string
	for page := 0; page < releasePages; page++ {
		releases, resp, err := c.rest.Repositories.ListReleases(ctx, c.cfg.Owner, c.cfg.Repo, opts)
		if err != nil {
			return nil, fmt.Errorf("list releases: %w", err)
		}
		for _, r := range releases {
			if r.GetDraft() {
				continue
			}
			tag := r.GetTagName()
			if !validTag(tag) {
				c.log.Warn("skipping release with an unusable tag name", "tag", tag)
				continue
			}
			tags = append(tags, tag)
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	return tags, nil
}

// previousTag returns the tag listed immediately after head, i.e. the release
// published before it. tags must be ordered most-recent-first.
func previousTag(tags []string, head string) (string, error) {
	for i, t := range tags {
		if t == head {
			if i+1 >= len(tags) {
				return "", fmt.Errorf("%q is the earliest listed release: %w", head, ErrNoPreviousRelease)
			}
			return tags[i+1], nil
		}
	}
	return "", fmt.Errorf("release %q not found among the listed releases", head)
}

// --- Diff -------------------------------------------------------------------

// Compare fetches the diff between two release tags and applies the configured
// bounds. Both tags must already have passed validTag; Compare re-checks rather
// than trusting its caller, because the tags are interpolated into the request
// path.
func (c *GitHubClient) Compare(ctx context.Context, base, head string) (*ReleaseDiff, error) {
	if !validTag(base) || !validTag(head) {
		return nil, fmt.Errorf("refusing to compare unvalidated tags %q...%q", base, head)
	}
	diff := &ReleaseDiff{BaseTag: base, HeadTag: head}
	var files []ChangedFile
	// The compare endpoint paginates commits, and whether it also paginates the
	// files array has varied. Deduplicating by path is correct either way; without
	// it a repeated files array would have the same file analyzed once per page.
	seenFile := make(map[string]bool)
	seenCommit := make(map[string]bool)
	opts := &github.ListOptions{PerPage: listPageSize}
	for page := 0; page < comparePages; page++ {
		cmp, resp, err := c.rest.Repositories.CompareCommits(ctx, c.cfg.Owner, c.cfg.Repo, base, head, opts)
		if err != nil {
			return nil, fmt.Errorf("compare %s...%s: %w", base, head, err)
		}
		if page == 0 {
			diff.CompareURL = cmp.GetHTMLURL()
		}
		for _, f := range cmp.Files {
			path := f.GetFilename()
			if seenFile[path] {
				continue
			}
			seenFile[path] = true
			files = append(files, ChangedFile{
				Path:      path,
				Status:    f.GetStatus(),
				Additions: f.GetAdditions(),
				Deletions: f.GetDeletions(),
				Patch:     f.GetPatch(),
			})
		}
		for _, cm := range cmp.Commits {
			sha := cm.GetSHA()
			if seenCommit[sha] {
				continue
			}
			seenCommit[sha] = true
			diff.Commits = append(diff.Commits, Commit{
				SHA:     shortSHA(sha),
				Subject: commitSubject(cm.GetCommit().GetMessage()),
			})
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}

	diff.TotalFiles = len(files)
	diff.Files, diff.OmittedFiles = boundFiles(files, c.cfg.MaxFiles, c.cfg.MaxPatchBytes)
	if len(diff.Commits) > c.cfg.MaxCommits {
		diff.OmittedCommits = len(diff.Commits) - c.cfg.MaxCommits
		diff.Commits = diff.Commits[:c.cfg.MaxCommits]
	}
	return diff, nil
}

// shortSHA abbreviates a commit SHA for display, tolerating a short or empty one.
func shortSHA(sha string) string {
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}

// commitSubject returns the first line of a commit message, bounded. The rest of
// the message is dropped: it is contributor-authored text that adds little to a
// documentation analysis and a lot to the prompt.
func commitSubject(message string) string {
	first, _, _ := strings.Cut(strings.ReplaceAll(message, "\r\n", "\n"), "\n")
	return truncateRunes(strings.TrimSpace(first), 200)
}

// --- Duplicate detection ----------------------------------------------------

// FindExistingIssue reports whether this bot has already filed an issue for a
// tag pair, and its number if so.
//
// It runs two probes with different failure modes, and treats a hit from either
// as a duplicate:
//
//   - The LIST probe reads the target repository's issues directly. It is
//     immediately consistent, so a re-run seconds after a successful run sees
//     the issue that run created -- which is the case that actually produces
//     duplicates. It is bounded to the most recent dedupPages*100 issues.
//   - The SEARCH probe queries the search index by the deterministic title. It
//     covers issues older than the list bound, but the index is only eventually
//     consistent, so it cannot be relied on alone.
//
// An error from either probe is returned rather than swallowed: the caller must
// not file an issue it could not prove is new.
func (c *GitHubClient) FindExistingIssue(ctx context.Context, headTag, marker string) (int, bool, error) {
	if n, found, err := c.findByList(ctx, marker); err != nil || found {
		return n, found, err
	}
	return c.findBySearch(ctx, headTag, marker)
}

// findByList scans the target repository's most recent issues for the marker.
func (c *GitHubClient) findByList(ctx context.Context, marker string) (int, bool, error) {
	opts := &github.IssueListByRepoOptions{
		State:       "all",
		Sort:        "created",
		Direction:   "desc",
		ListOptions: github.ListOptions{PerPage: listPageSize},
	}
	for page := 0; page < dedupPages; page++ {
		issues, resp, err := c.rest.Issues.ListByRepo(ctx, c.cfg.TargetOwner, c.cfg.TargetRepo, opts)
		if err != nil {
			return 0, false, fmt.Errorf("list issues for duplicate check: %w", err)
		}
		for _, iss := range issues {
			if c.isOwnMarkedIssue(iss, marker) {
				return iss.GetNumber(), true, nil
			}
		}
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}
	return 0, false, nil
}

// findBySearch queries the search index for the deterministic issue title and
// confirms each candidate by its marker.
func (c *GitHubClient) findBySearch(ctx context.Context, headTag, marker string) (int, bool, error) {
	query := fmt.Sprintf("repo:%s/%s is:issue in:title %q",
		c.cfg.TargetOwner, c.cfg.TargetRepo, issueTitle(headTag))
	result, _, err := c.rest.Search.Issues(ctx, query, &github.SearchOptions{
		ListOptions: github.ListOptions{PerPage: listPageSize},
	})
	if err != nil {
		return 0, false, fmt.Errorf("search issues for duplicate check: %w", err)
	}
	for _, iss := range result.Issues {
		if c.isOwnMarkedIssue(iss, marker) {
			return iss.GetNumber(), true, nil
		}
	}
	return 0, false, nil
}

// isOwnMarkedIssue reports whether an issue is one THIS bot filed for the tag
// pair the marker names.
//
// Authorship is checked as well as the marker: without it, anyone could open an
// issue whose first line is the marker and suppress the bot's issue for that
// release. Pull requests are excluded because a PR body is contributor-authored
// and would give the same suppression for free.
func (c *GitHubClient) isOwnMarkedIssue(iss *github.Issue, marker string) bool {
	if iss == nil || iss.IsPullRequest() {
		return false
	}
	return c.trustedCreator(iss.GetUser()) && hasBodyMarker(iss.GetBody(), marker)
}

// trustedCreator reports whether an issue's author may be treated as this bot.
//
// With a resolved identity the check is exact. Without one -- the built-in
// Actions token cannot read its own user -- it falls back to "some GitHub App
// wrote this", which still excludes every ordinary user account. The residual
// gap is an App installed on the target repository, which is a far higher bar
// than opening an issue, and it costs a suppressed issue rather than a wrong
// mutation.
func (c *GitHubClient) trustedCreator(u *github.User) bool {
	if u == nil {
		return false
	}
	if c.selfLogin != "" {
		return strings.EqualFold(u.GetLogin(), c.selfLogin)
	}
	return u.GetType() == botTypeLogin
}

// --- The single mutation ----------------------------------------------------

// FileReleaseIssue creates the release's documentation issue in the target
// repository. It is the bot's only mutation.
//
// The claim is taken before the write so a second call for the same release is
// a no-op rather than a second issue. A failed write is reported as such rather
// than as "already filed", so the run does not record an issue that was never
// created.
func (c *GitHubClient) FileReleaseIssue(ctx context.Context, key, title, body string) (int, error) {
	if !c.claimRelease(key) {
		if c.fileAttemptFailed(key) {
			return 0, fmt.Errorf("filing the issue for %s already failed this run", key)
		}
		c.log.Info("issue for this release was already filed in this run; not filing again", "release", key)
		return 0, nil
	}
	if c.shouldSkip("create issue %q in %s/%s", title, c.cfg.TargetOwner, c.cfg.TargetRepo) {
		c.log.Info("[dry-run] rendering the issue body it would have filed", "release", key, "bytes", len(body))
		if _, err := fmt.Fprintln(c.out, body); err != nil {
			c.log.Warn("could not render the dry-run issue body", "error", err)
		}
		return 0, nil
	}
	iss, _, err := c.rest.Issues.Create(ctx, c.cfg.TargetOwner, c.cfg.TargetRepo, &github.IssueRequest{
		Title: github.String(title),
		Body:  github.String(body),
	})
	if err != nil {
		c.recordFileFailure(key)
		c.recordError()
		return 0, fmt.Errorf("create issue: %w", err)
	}
	c.log.Info("filed release documentation issue",
		"issue", iss.GetNumber(), "repo", c.cfg.TargetOwner+"/"+c.cfg.TargetRepo)
	return iss.GetNumber(), nil
}

// shouldSkip logs an intended mutation and reports whether it should be skipped
// because dry-run is enabled. It is the single chokepoint every mutation passes
// through, so dry-run is impossible to forget.
func (c *GitHubClient) shouldSkip(format string, args ...any) bool {
	if c.cfg.DryRun {
		c.log.Info("[dry-run] would " + fmt.Sprintf(format, args...))
		return true
	}
	return false
}
