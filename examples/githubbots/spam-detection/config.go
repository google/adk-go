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
	"flag"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

// Config holds all runtime configuration for the spam-detection bot. It is
// parsed once from environment variables and command-line flags and then
// injected into the rest of the program; there is deliberately no
// package-level mutable state.
type Config struct {
	// Owner and Repo identify the target repository (e.g. "google"/"adk-go").
	Owner string
	Repo  string

	// GitHubToken authenticates GitHub REST and GraphQL calls. In GitHub
	// Actions this is the auto-provided github-actions[bot] token
	// (${{ secrets.GITHUB_TOKEN }}), authorized via the workflow permissions
	// block.
	GitHubToken string

	// BotLogin is the GitHub login this bot posts as. When set it is used
	// directly and no identity lookup is made.
	//
	// It exists because the identity lookup cannot work under GitHub Actions:
	// GET /user is a user-scoped endpoint and the built-in GITHUB_TOKEN is a
	// GitHub App installation token, which GitHub refuses on it. The workflow
	// therefore states the identity, which for that token is documented and
	// constant. A run with a personal access token can leave this empty and let
	// the lookup resolve it.
	BotLogin string

	// GeminiAPIKey authenticates the Gemini (AI Studio) model.
	GeminiAPIKey string

	// Model is the Gemini model name used for reasoning.
	Model string

	// SpamLabel is the label applied to issues judged to be spam. It must
	// already exist in the repository.
	SpamLabel string

	// Maintainers is the set of GitHub logins whose comments are trusted and
	// therefore never reviewed for spam. The default GITHUB_TOKEN cannot list
	// collaborators, so the set is supplied explicitly via the MAINTAINERS env
	// var (comma-separated). When empty, maintainer comments are reviewed like
	// anyone else's (see maintainersWarning).
	Maintainers []string

	// IssueCount caps how many candidate issues a single scheduled sweep
	// processes (most-recently-updated first).
	IssueCount int

	// FreshnessWindow optionally restricts the sweep to issues updated within
	// the window. Zero disables the restriction (the full open backlog). Spam
	// frequently arrives as a comment on an older issue, so the window filters
	// on last-updated time rather than creation time.
	FreshnessWindow time.Duration

	// Concurrency bounds how many issues are reviewed in parallel.
	Concurrency int

	// IssueTimeout bounds how long a single issue review may take.
	IssueTimeout time.Duration

	// RunTimeout bounds the whole run. The workflow sets its job timeout above
	// this so an overrun is reported by the bot rather than the runner killing
	// the job partway through a write.
	RunTimeout time.Duration

	// DryRun, when true, logs intended mutations without performing them.
	DryRun bool

	// SingleIssue, when non-zero, reviews only that issue and skips the search
	// step. Useful for local testing and workflow_dispatch.
	SingleIssue int
}

// defaultIssueTimeout bounds one issue's agent run. It is both the default and
// the value a non-positive ISSUE_TIMEOUT is clamped back to, so the two must not
// drift apart.
const defaultIssueTimeout = 5 * time.Minute

// defaultRunTimeout bounds the whole run, and is likewise both the default and
// the clamp for a non-positive RUN_TIMEOUT.
const defaultRunTimeout = 15 * time.Minute

// loadConfig parses configuration from flags (args) and environment variables
// and validates required fields. args is injectable so tests can exercise flag
// parsing.
func loadConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("githubspambot", flag.ContinueOnError)
	dryRunDefault, err := envBoolStrict("DRY_RUN", false)
	if err != nil {
		return nil, err
	}
	dryRun := fs.Bool("dry-run", dryRunDefault,
		"log intended actions without labeling or commenting")
	// A string rather than fs.Int, parsed below with strconv.Atoi. fs.Int uses
	// strconv.ParseInt(s, 0, ...), which reads a leading zero as octal: -issue
	// 0123 would silently review issue 83, and consistently so -- the session
	// scope, the fetch and the write would all agree on the wrong number, so no
	// downstream check could catch it.
	singleIssue := fs.String("issue", "",
		"review only this issue number and skip the search step (empty = sweep candidates)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	issueNumber, err := parseIssueNumber(*singleIssue)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:           os.Getenv("OWNER"),
		Repo:            os.Getenv("REPO"),
		GitHubToken:     os.Getenv("GITHUB_TOKEN"),
		BotLogin:        strings.TrimSpace(os.Getenv("BOT_LOGIN")),
		GeminiAPIKey:    firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:           envString("LLM_MODEL_NAME", "gemini-flash-latest"),
		SpamLabel:       envString("SPAM_LABEL_NAME", "spam"),
		Maintainers:     splitList(os.Getenv("MAINTAINERS")),
		IssueCount:      envInt("ISSUE_COUNT", 3),
		FreshnessWindow: envDays("FRESHNESS_WINDOW_DAYS", 0),
		Concurrency:     envInt("CONCURRENCY_LIMIT", 3),
		IssueTimeout:    envDuration("ISSUE_TIMEOUT", defaultIssueTimeout),
		RunTimeout:      envDuration("RUN_TIMEOUT", defaultRunTimeout),
		DryRun:          *dryRun,
		SingleIssue:     issueNumber,
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	var missing []string
	if c.GitHubToken == "" {
		missing = append(missing, "GITHUB_TOKEN")
	}
	// A Gemini API key is the simplest path, but Vertex AI via ADC is also
	// supported; in that case the genai SDK reads its configuration from the
	// environment (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, ...).
	if c.GeminiAPIKey == "" && !envBool("GOOGLE_GENAI_USE_VERTEXAI", false) {
		missing = append(missing, "GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)")
	}
	if c.Owner == "" {
		missing = append(missing, "OWNER")
	}
	if c.Repo == "" {
		missing = append(missing, "REPO")
	}
	if c.SpamLabel == "" {
		missing = append(missing, "SPAM_LABEL_NAME")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}
	// The label goes into a GitHub search query as -label:"…", and that syntax
	// has no escape for a quote or a backslash. Rejecting such a label is better
	// than silently building a query the Search API parses differently, which
	// would drop the "already labeled" exclusion without saying so.
	// %q escapes a quote, a backslash, a non-printable rune AND an invalid UTF-8
	// byte, and a GitHub search term has no escape for any of them. Checking the
	// quoted form directly covers all four without enumerating them: an escape
	// in the output means the value cannot survive interpolation as itself.
	if strings.Contains(strconv.Quote(c.SpamLabel), `\`) {
		return fmt.Errorf("SPAM_LABEL_NAME %q contains a character that cannot be expressed in a GitHub search query", c.SpamLabel)
	}
	// Owner and Repo are interpolated into the search query AND into every REST
	// path the client builds, so a value carrying a slash or a space could point
	// a write at another repository. The workflow supplies them from
	// GITHUB_REPOSITORY, but nothing in the program required that.
	if !validRepoName(c.Owner) {
		return fmt.Errorf("OWNER %q is not a valid GitHub owner name", c.Owner)
	}
	if !validRepoName(c.Repo) {
		return fmt.Errorf("REPO %q is not a valid GitHub repository name", c.Repo)
	}
	// The identity is never interpolated anywhere -- it is only ever compared
	// with EqualFold -- but it decides which comments the bot treats as its own,
	// so a wrong value is not merely useless: pointed at a real account it would
	// let that account suppress moderation by posting the alert marker. A shape
	// check cannot tell a right value from a wrong one, but it rejects the ones
	// that could only be a mistake, including the invisible characters that make
	// a login look correct and never match.
	if c.BotLogin != "" && !botLoginPattern.MatchString(c.BotLogin) {
		return fmt.Errorf("BOT_LOGIN %q is not a valid GitHub login", c.BotLogin)
	}
	if c.IssueCount < 1 {
		c.IssueCount = 1
	}
	if c.Concurrency < 1 {
		c.Concurrency = 1
	}
	if c.FreshnessWindow < 0 {
		c.FreshnessWindow = 0
	}
	if c.IssueTimeout <= 0 {
		c.IssueTimeout = defaultIssueTimeout
	}
	if c.RunTimeout <= 0 {
		c.RunTimeout = defaultRunTimeout
	}
	// The run budget has to be able to cover the work the run is told to do.
	//
	// The sweep processes IssueCount issues in waves of Concurrency, and each
	// issue may take up to IssueTimeout, so the worst case is
	// ceil(IssueCount/Concurrency) * IssueTimeout. Where that exceeds RunTimeout
	// the configuration is unsatisfiable by arithmetic: the run cannot finish
	// its own declared work set even when nothing goes wrong. Expiry is
	// deliberately fail-loud, so every sweep that takes the slow path reports a
	// failure that is not one -- and an attacker only has to open enough issues
	// whose text keeps the model busy to make that the normal outcome, which
	// trains maintainers to ignore the bot.
	//
	// Rejected at startup rather than clamped. Clamping would review fewer
	// issues than the operator asked for without saying so, and quietly doing
	// less moderation than configured is the worse failure of the two.
	if waves := (c.IssueCount + c.Concurrency - 1) / c.Concurrency; c.IssueTimeout*time.Duration(waves) > c.RunTimeout {
		return fmt.Errorf(
			"RUN_TIMEOUT %s cannot cover the configured work: ISSUE_COUNT %d at CONCURRENCY_LIMIT %d is %d wave(s), "+
				"and at ISSUE_TIMEOUT %s each that is %s in the worst case. Raise RUN_TIMEOUT, lower ISSUE_TIMEOUT "+
				"or ISSUE_COUNT, or raise CONCURRENCY_LIMIT",
			c.RunTimeout, c.IssueCount, c.Concurrency, waves, c.IssueTimeout, c.IssueTimeout*time.Duration(waves))
	}
	return nil
}

// parseIssueNumber reads the -issue flag in base 10 and rejects anything that
// is not a plain positive integer. An empty value means "sweep candidates".
//
// Rejecting a leading zero is the point: base-10 parsing already makes "0123"
// mean 123 rather than 83, but a caller who typed a leading zero has more
// likely mistyped the number than intended it, and this bot writes to whatever
// issue it is pointed at.
func parseIssueNumber(s string) (int, error) {
	if s == "" {
		return 0, nil
	}
	// Shape first: strconv.Atoi accepts a sign, and "+05" would otherwise slip
	// past a leading-zero check that only looks at s[0].
	if s[0] < '1' || s[0] > '9' {
		return 0, fmt.Errorf("-issue %q must start with a digit 1-9", s)
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("-issue %q is not a positive issue number: %w", s, err)
	}
	return n, nil
}

// GitHub owner and repository names are ASCII alphanumerics plus a few
// separators. "." and ".." are excluded specifically, because those become path
// segments in every REST URL the client builds -- but a leading dot is fine, and
// forbidding it would reject ".github", a real repository name. A login is
// alphanumerics, hyphens and (for older accounts) underscores, with an optional
// "[bot]" suffix for a GitHub App.
var (
	repoNamePattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,100}$`)
	dotOnlyName     = regexp.MustCompile(`^\.{1,2}$`)
	botLoginPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,38}(\[bot\])?$`)
)

// validRepoName reports whether s can appear as one path segment of a GitHub
// REST URL as itself.
func validRepoName(s string) bool {
	return repoNamePattern.MatchString(s) && !dotOnlyName.MatchString(s)
}

// loadDotEnv reads a local .env for local runs only, and reports whether it
// tried. Under Actions the configuration comes from the workflow, and a .env
// committed to the repository would otherwise set anything the workflow leaves
// unset -- inside a job holding issues: write. CI is set to "true" by GitHub
// Actions and by every other CI system worth naming.
func loadDotEnv() bool {
	if os.Getenv("CI") != "" {
		return false
	}
	_ = godotenv.Load()
	return true
}

// Environment helpers.

func envString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// splitList splits a comma-separated list, trimming whitespace and dropping
// empty entries.
func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

// envBoolStrict reads a boolean environment variable and rejects a value that
// is set but unparseable, instead of falling back to the default.
//
// DRY_RUN goes through this rather than envBool because it is the switch that
// suppresses every mutation. Reading an unrecognized value (DRY_RUN=yes, say,
// which strconv.ParseBool does not accept) as "false" would turn a requested
// dry run into a live one that labels and comments on real issues.
func envBoolStrict(key string, def bool) (bool, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false, fmt.Errorf("%s=%q is not a boolean: use true or false", key, v)
	}
	return b, nil
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

// envDays reads a (possibly fractional) number of days and returns a Duration.
func envDays(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if days, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(days * float64(24*time.Hour))
		}
	}
	return def
}
