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
	singleIssue := fs.Int("issue", 0, "Triage only this issue number (0 = sweep untriaged issues).")
	if err := fs.Parse(args); err != nil {
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
		SweepTimeout:    env.duration("SWEEP_TIMEOUT", 15*time.Minute),
		DryRun:          *dryRun,
		SingleIssue:     *singleIssue,
		UseVertexAI:     env.boolean("GOOGLE_GENAI_USE_VERTEXAI", false),
	}
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
	if c.IssueCount < 1 {
		return fmt.Errorf("ISSUE_COUNT must be at least 1, got %d", c.IssueCount)
	}
	if c.FreshnessWindow < 0 {
		return fmt.Errorf("FRESHNESS_WINDOW_DAYS must not be negative, got %s", c.FreshnessWindow)
	}
	// A negative -issue is neither a valid issue number nor a request to sweep.
	// Only SingleIssue > 0 selects single-issue mode, so letting a negative
	// through would quietly turn "triage issue -5" into a full backlog sweep.
	if c.SingleIssue < 0 {
		return fmt.Errorf("-issue must not be negative, got %d", c.SingleIssue)
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
	return time.Duration(days * float64(24*time.Hour))
}
