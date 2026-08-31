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

// defaultAllowedLabels are the categorization labels the bot may apply. They
// must already exist in the target repository.
var defaultAllowedLabels = []string{"bug", "enhancement", "documentation", "question"}

// allowedTypes are the GitHub issue types the bot may set. These must be
// enabled at the organization level.
var allowedTypes = []string{"Bug", "Feature", "Task"}

// Config holds all runtime configuration. It is parsed once and injected; there
// is no package-level mutable state.
type Config struct {
	Owner string
	Repo  string

	GitHubToken  string
	GeminiAPIKey string
	Model        string

	// AllowedLabels is the set of categorization labels the agent may apply.
	AllowedLabels []string

	// IssueCount caps how many untriaged issues a single scheduled sweep
	// processes (newest first).
	IssueCount int
	// FreshnessWindow optionally restricts the sweep to issues created within
	// the window. Zero disables the restriction (full backlog).
	FreshnessWindow time.Duration
	// IssueTimeout bounds a single issue's agent run.
	IssueTimeout time.Duration
	// SweepTimeout bounds the WHOLE run, so N issues cannot multiply into
	// N x IssueTimeout and overrun the workflow's own timeout-minutes. The job
	// gets killed at that point regardless, and a kill mid-sweep is silent, so
	// the process needs its own bound to stop first and report.
	SweepTimeout time.Duration

	// DryRun logs intended actions without performing any mutation.
	DryRun bool
	// SingleIssue, when > 0, triages only that issue instead of sweeping.
	SingleIssue int
	// UseVertexAI reports whether the genai SDK should reach Vertex AI through
	// Application Default Credentials instead of a Gemini API key.
	UseVertexAI bool
}

// loadConfig parses configuration from flags (args) and environment variables.
// args is injectable so tests can exercise flag parsing.
func loadConfig(args []string) (*Config, error) {
	// Read the environment through a reader that collects parse failures rather
	// than substituting the default. Falling back on a malformed value turns
	// DRY_RUN=yes -- which strconv.ParseBool rejects -- into "act for real",
	// the opposite of what the operator asked for, and it turns
	// FRESHNESS_WINDOW_DAYS=7d into a sweep of the entire backlog instead of a
	// week. A control that degrades to "off" on a typo is not a control.
	env := &envReader{}
	fs := flag.NewFlagSet("issue-triage", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", env.boolean("DRY_RUN", false), "Log intended actions without modifying any issues.")
	// Not fs.Int: that parses with strconv.ParseInt base 0, so "-issue 010"
	// would be read as octal 8 and the bot would triage a different issue than
	// the operator named. strconv.Atoi is base 10. A value below 1 is rejected
	// rather than ignored, because only SingleIssue > 0 selects single-issue
	// mode -- "-issue 0" would otherwise become a full backlog sweep.
	var singleIssue int
	fs.Func("issue", "Triage only this issue number (omit to sweep untriaged issues).", func(v string) error {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("issue number %q is not a base-10 integer: %w", v, err)
		}
		if n < 1 {
			return fmt.Errorf("issue number must be at least 1, got %d", n)
		}
		singleIssue = n
		return nil
	})
	parseErr := fs.Parse(args)
	// Report the environment failures too rather than only the first problem:
	// a bad flag and a DRY_RUN typo are independent, and hiding the second
	// behind the first is how the second survives a fix for the first.
	if err := errors.Join(parseErr, env.err()); err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:           os.Getenv("OWNER"),
		Repo:            os.Getenv("REPO"),
		GitHubToken:     os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:    firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:           envString("LLM_MODEL_NAME", "gemini-flash-latest"),
		AllowedLabels:   splitList(envString("ALLOWED_LABELS", strings.Join(defaultAllowedLabels, ","))),
		IssueCount:      env.integer("ISSUE_COUNT", 3),
		FreshnessWindow: env.days("FRESHNESS_WINDOW_DAYS", 0),
		IssueTimeout:    env.duration("ISSUE_TIMEOUT", 5*time.Minute),
		SweepTimeout:    env.duration("SWEEP_TIMEOUT", 20*time.Minute),
		DryRun:          *dryRun,
		SingleIssue:     singleIssue,
		UseVertexAI:     env.boolean("GOOGLE_GENAI_USE_VERTEXAI", false),
	}
	// The typed reads above happen after fs.Parse, so collect their failures too.
	if err := env.err(); err != nil {
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
	if c.Owner == "" {
		missing = append(missing, "OWNER")
	}
	if c.Repo == "" {
		missing = append(missing, "REPO")
	}
	// A Gemini API key is the simplest path, but Vertex AI via ADC is also
	// supported; in that case the genai SDK reads its configuration from the
	// environment (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, ...).
	if c.GeminiAPIKey == "" && !c.UseVertexAI {
		missing = append(missing, "GEMINI_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}

	if c.IssueTimeout <= 0 {
		return fmt.Errorf("ISSUE_TIMEOUT must be positive, got %s", c.IssueTimeout)
	}
	if c.SweepTimeout <= 0 {
		return fmt.Errorf("SWEEP_TIMEOUT must be positive, got %s", c.SweepTimeout)
	}
	if c.SweepTimeout < c.IssueTimeout {
		return fmt.Errorf("SWEEP_TIMEOUT (%s) must be at least ISSUE_TIMEOUT (%s)", c.SweepTimeout, c.IssueTimeout)
	}
	// The run budget funds choosing the work set as well as triaging it. A
	// configuration that cannot fit its own worst case exhausts the budget on
	// an ordinary busy sweep and reports issues untriaged, which reads as a
	// failure on a run that behaved exactly as designed.
	//
	// A single-issue run triages one issue whatever ISSUE_COUNT says, so
	// charging it the sweep's worst case would refuse budgets that are provably
	// sufficient -- and every `issues: opened` run takes that path.
	issues := c.IssueCount
	if c.SingleIssue > 0 {
		issues = 1
	}
	if worst := time.Duration(issues)*c.IssueTimeout + fetchTimeout; worst > c.SweepTimeout {
		return fmt.Errorf("%d issue(s) x ISSUE_TIMEOUT (%s) plus the %s work-set fetch is %s, which exceeds SWEEP_TIMEOUT (%s)",
			issues, c.IssueTimeout, fetchTimeout, worst, c.SweepTimeout)
	}
	if c.IssueCount < 1 {
		return fmt.Errorf("ISSUE_COUNT must be at least 1, got %d", c.IssueCount)
	}
	// Bounded before it is multiplied below: the product is a time.Duration, so
	// an absurd count would wrap int64 and land back inside the accepted range,
	// letting a configuration through that the budget check exists to reject.
	// The sweep cannot see more than this many issues anyway.
	if maxSweep := maxSearchPages * searchPageSize; c.IssueCount > maxSweep {
		return fmt.Errorf("ISSUE_COUNT (%d) exceeds the %d issues one sweep can read", c.IssueCount, maxSweep)
	}
	if c.FreshnessWindow < 0 {
		return fmt.Errorf("FRESHNESS_WINDOW_DAYS must not be negative, got %s", c.FreshnessWindow)
	}
	if len(c.AllowedLabels) == 0 {
		return errors.New("ALLOWED_LABELS is set but contains no usable label")
	}
	return nil
}

// Environment helpers.

func envString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
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

// envReader reads typed environment variables, recording a parse failure rather
// than silently returning the default. Every value it reads is either a safety
// switch (DRY_RUN) or a bound on how much the bot does, so "malformed" must not
// collapse into "absent".
type envReader struct{ errs []error }

func (e *envReader) fail(key, value string, err error) {
	e.errs = append(e.errs, fmt.Errorf("%s=%q is not valid: %w", key, value, err))
}

func (e *envReader) err() error { return errors.Join(e.errs...) }

func (e *envReader) boolean(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		e.fail(key, v, err)
		return def
	}
	return b
}

func (e *envReader) integer(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		e.fail(key, v, err)
		return def
	}
	return n
}

func (e *envReader) duration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		e.fail(key, v, err)
		return def
	}
	return d
}

// days reads a (possibly fractional) number of days and returns a Duration.
//
// ParseFloat accepts "NaN" and "Inf", and converting either to an int64 is
// implementation-defined in Go: on amd64 NaN becomes a large negative value,
// on a saturating target it becomes 0 -- which reads as "no freshness window"
// and silently widens the sweep to the whole backlog. Both are rejected, as is
// a day count too large to be a Duration.
func (e *envReader) days(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	days, err := strconv.ParseFloat(v, 64)
	if err != nil {
		e.fail(key, v, err)
		return def
	}
	hours := days * 24
	if math.IsNaN(hours) || math.IsInf(hours, 0) || math.Abs(hours) > maxDurationHours {
		e.fail(key, v, errors.New("not a finite number of days a duration can represent"))
		return def
	}
	return time.Duration(hours * float64(time.Hour))
}

// maxDurationHours is how many hours fit in a time.Duration, with margin.
const maxDurationHours = float64(math.MaxInt64/int64(time.Hour)) - 1
