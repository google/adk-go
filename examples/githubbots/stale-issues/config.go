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
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all runtime configuration for the stale-issue bot. It is parsed
// once from environment variables and command-line flags and then injected into
// the rest of the program; there is deliberately no package-level mutable state.
type Config struct {
	// Owner and Repo identify the target repository (e.g. "google"/"adk-go").
	Owner string
	Repo  string

	// GitHubToken authenticates GitHub REST and GraphQL calls. In GitHub
	// Actions this is the auto-provided github-actions[bot] token
	// (${{ secrets.GITHUB_TOKEN }}), authorized via the workflow permissions
	// block.
	GitHubToken string

	// GeminiAPIKey authenticates the Gemini (AI Studio) model.
	GeminiAPIKey string

	// Model is the Gemini model name used for reasoning.
	Model string

	// StaleAfter is how long an issue may sit waiting on the author (after a
	// maintainer's request) before it is marked stale. Default: 14 days.
	StaleAfter time.Duration

	// CloseAfter is how long an issue may remain stale (after the warning
	// comment) before it is closed. Default: 7 days.
	CloseAfter time.Duration

	// StaleLabel and RequestClarificationLabel are the label names the bot
	// manages. They must already exist in the repository.
	StaleLabel                string
	RequestClarificationLabel string

	// BotLogin is the login the bot posts under, used to recognize its own
	// activity. It is configured rather than discovered because the discovery
	// call cannot work where the bot actually runs: GET /user is a
	// user-to-server endpoint and the workflow's GITHUB_TOKEN is an installation
	// token, so the request is refused. Set it to "github-actions[bot]" in
	// Actions; leave it empty locally to have the login resolved from the API.
	BotLogin string

	// Maintainers is the set of GitHub logins treated as maintainers. The
	// default GITHUB_TOKEN cannot list collaborators, so the maintainer set is
	// supplied explicitly via the MAINTAINERS env var (comma-separated).
	Maintainers []string

	// Concurrency bounds how many issues are audited in parallel.
	Concurrency int

	// IssueTimeout bounds how long a single issue audit may take.
	IssueTimeout time.Duration

	// RunBudget bounds the whole run. Per-issue timeouts do not: a sweep is
	// len(issues)/Concurrency of them long, so a large backlog can outlast the
	// workflow's timeout-minutes and be killed mid-write with nothing in the log
	// to say why. The budget must stay below timeout-minutes so the overrun is
	// this process's diagnosis rather than the runner's SIGKILL.
	RunBudget time.Duration

	// UseVertexAI reports whether the genai SDK should authenticate against
	// Vertex AI via Application Default Credentials instead of an API key. It is
	// read here rather than at the point of use so a malformed
	// GOOGLE_GENAI_USE_VERTEXAI is reported with every other bad setting instead
	// of being skipped by a short-circuit when an API key happens to be set.
	UseVertexAI bool

	// DryRun, when true, logs intended mutations without performing them.
	DryRun bool

	// SingleIssue, when non-zero, audits only that issue and skips the search
	// step. Useful for local testing and workflow_dispatch.
	SingleIssue int
}

// loadConfig parses configuration from flags and environment variables and
// validates required fields.
func loadConfig(args []string) (*Config, error) {
	// A malformed value is an error, never a silent fall-back to the default.
	// DRY_RUN is the case that forced this: its default is "write for real", so
	// DRY_RUN=yes parsing as an error and degrading to the default turned a
	// requested dry run into live labeling, commenting and closing. The
	// thresholds get the same treatment, because a run that quietly uses 14 days
	// when the operator wrote 14d is not the run they asked for.
	e := &envErrors{}
	dryRunDefault := e.boolean("DRY_RUN", false)
	fs := flag.NewFlagSet("githubstalebot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", dryRunDefault,
		"log intended actions without commenting, labeling, or closing")
	singleIssue := fs.Int("issue", 0,
		"audit only this issue number and skip the search step (0 = audit all stale candidates)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:                     os.Getenv("OWNER"),
		Repo:                      os.Getenv("REPO"),
		GitHubToken:               os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:              firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:                     envString("LLM_MODEL_NAME", "gemini-flash-latest"),
		StaleAfter:                e.hours("STALE_HOURS_THRESHOLD", 14*24*time.Hour),
		CloseAfter:                e.hours("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 7*24*time.Hour),
		StaleLabel:                envString("STALE_LABEL_NAME", "stale"),
		RequestClarificationLabel: envString("REQUEST_CLARIFICATION_LABEL", "request clarification"),
		BotLogin:                  strings.TrimSpace(os.Getenv("BOT_LOGIN")),
		Maintainers:               splitList(os.Getenv("MAINTAINERS")),
		Concurrency:               e.integer("CONCURRENCY_LIMIT", 3),
		IssueTimeout:              e.duration("ISSUE_TIMEOUT", 5*time.Minute),
		RunBudget:                 e.duration("RUN_BUDGET", 30*time.Minute),
		UseVertexAI:               e.boolean("GOOGLE_GENAI_USE_VERTEXAI", false),
		DryRun:                    *dryRun,
		SingleIssue:               *singleIssue,
	}

	if err := e.err(); err != nil {
		return nil, err
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
	if c.GeminiAPIKey == "" && !c.UseVertexAI {
		missing = append(missing, "GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)")
	}
	if c.Owner == "" {
		missing = append(missing, "OWNER")
	}
	if c.Repo == "" {
		missing = append(missing, "REPO")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}
	// Reject non-positive durations rather than coercing them. A negative
	// STALE_HOURS_THRESHOLD — a plausible sign typo, plumbed straight from a
	// workflow input — puts the search cutoff in the future, so `created:<cutoff`
	// matches every open issue including ones opened minutes ago, and the
	// rendered threshold makes every comparison true. One character would turn
	// the bot into a mass-marker, so this has to fail loudly.
	if c.StaleAfter <= 0 {
		return fmt.Errorf("STALE_HOURS_THRESHOLD must be positive, got %s", c.StaleAfter)
	}
	if c.CloseAfter <= 0 {
		return fmt.Errorf("CLOSE_HOURS_AFTER_STALE_THRESHOLD must be positive, got %s", c.CloseAfter)
	}
	if c.IssueTimeout <= 0 {
		return fmt.Errorf("ISSUE_TIMEOUT must be positive, got %s", c.IssueTimeout)
	}
	if c.RunBudget <= 0 {
		return fmt.Errorf("RUN_BUDGET must be positive, got %s", c.RunBudget)
	}
	// A BOT_LOGIN that names a real participant does not merely mislabel one
	// comment: isIgnoredActor drops every event by that login, so the person
	// disappears from the history entirely. Pointed at a maintainer it erases the
	// request the whole decision tree turns on; pointed at the author it erases
	// their replies and the bot marks stale and closes an issue that was being
	// answered.
	for _, m := range c.Maintainers {
		if c.BotLogin != "" && strings.EqualFold(c.BotLogin, m) {
			return fmt.Errorf("BOT_LOGIN (%q) is also listed in MAINTAINERS; the bot ignores its own activity, so that maintainer's comments would be dropped from every issue's history", c.BotLogin)
		}
	}
	if c.StaleLabel != "" && sameLabel(c.StaleLabel, c.RequestClarificationLabel) {
		return fmt.Errorf("STALE_LABEL_NAME (%q) and REQUEST_CLARIFICATION_LABEL (%q) must differ; GitHub label names are case-insensitively unique, so a collision would route clarification-label writes through the staleness gate", c.StaleLabel, c.RequestClarificationLabel)
	}
	if c.Concurrency < 1 {
		c.Concurrency = 1
	}
	return nil
}

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

func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// envErrors reads typed values from the environment, collecting parse failures
// so loadConfig can report every one of them at once rather than dying on the
// first. Each reader returns the default on failure, which keeps the Config
// well-formed for the rest of the load — err() is what stops the run.
type envErrors struct{ errs []error }

func (e *envErrors) err() error { return errors.Join(e.errs...) }

func (e *envErrors) bad(key, value string, err error) {
	e.errs = append(e.errs, fmt.Errorf("%s=%q is not valid: %w", key, value, err))
}

func (e *envErrors) integer(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		e.bad(key, v, err)
		return def
	}
	return n
}

func (e *envErrors) boolean(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		e.bad(key, v, err)
		return def
	}
	return b
}

// hours reads a float number of hours and returns it as a time.Duration.
// Thresholds are expressed in hours for easy configuration in the workflow
// (e.g. STALE_HOURS_THRESHOLD=168).
func (e *envErrors) hours(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	hours, err := strconv.ParseFloat(v, 64)
	if err != nil {
		e.bad(key, v, err)
		return def
	}
	// ParseFloat accepts "Inf" and "NaN" without error, and converting either —
	// or any finite value past ~2.56e6 hours — to a Duration overflows int64.
	// Float-to-int overflow is implementation-defined in Go: amd64 yields
	// MinInt64, which validate() rejects with a baffling message, and a target
	// that saturates yields MaxInt64, a ~292-year threshold that passes
	// validate() and silently turns the sweep into a no-op.
	if math.IsNaN(hours) || math.IsInf(hours, 0) {
		e.bad(key, v, errors.New("must be a finite number of hours"))
		return def
	}
	// >=, not >: the compiled constant rounds UP past 2^63/3.6e12, so the
	// boundary value itself still overflows the conversion.
	if math.Abs(hours) >= math.MaxInt64/float64(time.Hour) {
		e.bad(key, v, errors.New("is too large to express as a duration"))
		return def
	}
	return time.Duration(hours * float64(time.Hour))
}

func (e *envErrors) duration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		e.bad(key, v, err)
		return def
	}
	return d
}
