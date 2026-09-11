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
	"sort"
	"strconv"
	"strings"
	"time"
)

// Config holds all runtime configuration for the pr-triage bot. It is parsed
// once from environment variables and command-line flags and then injected into
// the rest of the program; there is deliberately no package-level mutable state.
type Config struct {
	// Owner and Repo identify the target repository (e.g. "google"/"adk-go").
	Owner string
	Repo  string

	// GitHubToken authenticates GitHub REST and GraphQL calls. In GitHub Actions
	// this is the auto-provided github-actions[bot] token
	// (${{ secrets.GITHUB_TOKEN }}), authorized via the workflow permissions
	// block.
	GitHubToken string

	// GeminiAPIKey authenticates the Gemini (AI Studio) model.
	GeminiAPIKey string

	// Model is the Gemini model name used for reasoning.
	Model string

	// OwnerMap maps a component name to the GitHub login that shepherds it. It
	// is the complete set of logins this bot may ever assign: the model picks a
	// key, never a value, so it cannot invent an assignee. Keys are lowercased
	// at parse time; lookups lowercase the model's choice to match.
	OwnerMap map[string]string

	// BotLogin is the login this bot posts under. It exists because the identity
	// lookup the obvious way -- REST GET /user -- is a user-to-server endpoint,
	// and the workflow authenticates with an installation token, for which it
	// returns 403. The bot needs its own login to recognize its own past
	// comment, so the workflow states it explicitly and the API lookup is only
	// the fallback for a personal access token.
	BotLogin string

	// UseVertexAI reports whether the genai SDK should read its configuration
	// from the environment (Vertex AI via ADC) instead of using an API key.
	UseVertexAI bool

	// RequestContext enables the second tool, which posts one comment asking the
	// author for missing context. When false the tool is not registered at all
	// and the prompt says so, so the bot's only power is assignment.
	RequestContext bool

	// SinglePR, when non-zero, triages only that pull request and skips the
	// search step. It is how the pull_request_target path runs.
	SinglePR int

	// PRCount caps how many unassigned pull requests a batch run processes
	// (most-recently-updated first). It applies only when SinglePR is zero.
	PRCount int

	// MaxFiles caps how many changed-file paths are shown to the model. Paths
	// come from the fork's branch, so they are attacker-controlled and are
	// fenced with everything else.
	MaxFiles int

	// Concurrency bounds how many pull requests are triaged in parallel.
	Concurrency int

	// PRTimeout bounds how long a single pull request's triage may take.
	PRTimeout time.Duration

	// RunBudget bounds the whole run. It exists so an overrun is reported by the
	// process rather than the job being killed by the workflow's
	// timeout-minutes, which would leave no diagnosis behind. Keep it below the
	// workflow timeout.
	RunBudget time.Duration

	// DryRun, when true, logs intended mutations without performing them.
	DryRun bool
}

const (
	// defaultPRTimeout bounds one pull request's agent run. It is both the
	// default and the value a non-positive PR_TIMEOUT is clamped back to, so the
	// two must not drift apart.
	defaultPRTimeout = 5 * time.Minute
	// defaultRunBudget bounds the whole run. The shipped workflow sets
	// timeout-minutes above this so the process reports the overrun itself.
	defaultRunBudget = 10 * time.Minute

	// maxPRCount and maxConcurrency bound what a workflow_dispatch input can ask
	// for, so a mistyped batch size cannot turn one manual run into hundreds of
	// mutations or hundreds of parallel API calls.
	maxPRCount     = 100
	maxConcurrency = 10
	// maxFilesLimit bounds the changed-file list even if MAX_FILES is raised, so
	// the attacker-controlled part of the prompt stays bounded. It is 100 because
	// that is a hard limit of GitHub's GraphQL schema, not a taste choice: "values
	// of first and last must be within 1-100", and a larger value fails schema
	// validation, so every fetch in the run would error.
	maxFilesLimit = 100
)

// componentPattern constrains a component name from the owner map. The charset
// deliberately excludes braces: the rendered component list is interpolated into
// the agent instruction, and llmagent treats {token} as a session-state
// reference, so a brace here would fail every run. It also keeps the name
// readable in the prompt.
var componentPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9 ._-]{0,39}$`)

// loginPattern is GitHub's own rule for a user login: alphanumeric with single
// internal hyphens. Validating it here means the set of strings this bot can
// ever POST as an assignee is fixed at startup.
//
// The length bound is checked separately, in validLogin: each repetition of the
// group consumes one OR two characters, so the pattern by itself admits 77 --
// long enough that an over-long login would pass startup validation, spend a
// pull request's single assignment attempt, and then be rejected by GitHub.
var loginPattern = regexp.MustCompile(`^[A-Za-z0-9](?:-?[A-Za-z0-9]){0,38}$`)

// maxLoginRunes is GitHub's limit on a user login.
const maxLoginRunes = 39

// botSuffixPattern matches the "[bot]" a GitHub App login carries. BOT_LOGIN may
// have it; an assignable owner may not.
var botSuffixPattern = regexp.MustCompile(`\[bot\]$`)

// validLogin reports whether s is a syntactically valid GitHub login. allowBot
// permits the "[bot]" suffix a GitHub App identity carries.
func validLogin(s string, allowBot bool) bool {
	if allowBot {
		s = botSuffixPattern.ReplaceAllString(s, "")
	}
	return len([]rune(s)) <= maxLoginRunes && loginPattern.MatchString(s)
}

// loadConfig parses configuration from flags (args) and environment variables
// and validates required fields. args is injectable so tests can exercise flag
// parsing.
func loadConfig(args []string) (*Config, error) {
	// PULL_REQUEST_NUMBER is parsed before the flag set so a malformed value
	// fails loudly. Defaulting it to 0 would silently turn a workflow_dispatch
	// for one pull request into a batch run over many.
	envPR, err := parsePRNumber(os.Getenv("PULL_REQUEST_NUMBER"))
	if err != nil {
		return nil, err
	}

	var ee envErrs
	fs := flag.NewFlagSet("githubprtriagebot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", envBool(&ee, "DRY_RUN", false),
		"log intended actions without assigning or commenting")
	singlePR := fs.Int("pr", envPR,
		"triage only this pull request number and skip the search step (0 = batch)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	ownerMap, err := parseOwnerMap(os.Getenv("OWNER_MAP"))
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target a
		// concrete repository if a caller forgot to set them (validate() rejects an
		// empty value instead, so misconfiguration fails loudly).
		Owner:          os.Getenv("OWNER"),
		Repo:           os.Getenv("REPO"),
		GitHubToken:    os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:   firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:          envString("LLM_MODEL_NAME", "gemini-flash-latest"),
		OwnerMap:       ownerMap,
		BotLogin:       strings.TrimSpace(os.Getenv("BOT_LOGIN")),
		UseVertexAI:    envBool(&ee, "GOOGLE_GENAI_USE_VERTEXAI", false),
		RequestContext: envBool(&ee, "REQUEST_CONTEXT", true),
		SinglePR:       *singlePR,
		PRCount:        envInt(&ee, "PR_COUNT", 10),
		MaxFiles:       envInt(&ee, "MAX_FILES", 50),
		Concurrency:    envInt(&ee, "CONCURRENCY_LIMIT", 3),
		PRTimeout:      envDuration(&ee, "PR_TIMEOUT", defaultPRTimeout),
		RunBudget:      envDuration(&ee, "RUN_BUDGET", defaultRunBudget),
		DryRun:         *dryRun,
	}
	if err := ee.err(); err != nil {
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
	// Assignment is this bot's whole purpose, so an empty map is a
	// misconfiguration rather than a quiet no-op run.
	if len(c.OwnerMap) == 0 {
		missing = append(missing, "OWNER_MAP")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}
	if c.SinglePR < 0 {
		return fmt.Errorf("pull request number must not be negative, got %d", c.SinglePR)
	}
	if c.BotLogin != "" && !validLogin(c.BotLogin, true) {
		return fmt.Errorf("BOT_LOGIN %q is not a valid GitHub login", c.BotLogin)
	}

	c.PRCount = clamp(c.PRCount, 1, maxPRCount)
	c.Concurrency = clamp(c.Concurrency, 1, maxConcurrency)
	c.MaxFiles = clamp(c.MaxFiles, 1, maxFilesLimit)
	if c.PRTimeout <= 0 {
		c.PRTimeout = defaultPRTimeout
	}
	if c.RunBudget <= 0 {
		c.RunBudget = defaultRunBudget
	}
	return nil
}

// contextRequestsEnabled reports whether the bot may ask an author for more
// context on this run.
//
// It is off in batch mode even when REQUEST_CONTEXT is set. Batch mode is a
// manual sweep over a backlog, and asking the author of a months-old pull
// request for a better description is noise: one operator action becomes a
// burst of notifications to people who did not ask for them. Assignment does
// not have that problem -- it notifies one person, and it is the point of the
// sweep.
//
// This carried a second reason until the workflow changed: that a batch run and
// a per-pull-request run sat in DIFFERENT concurrency groups, so letting both
// comment was a way two comments could reach one pull request. That is no
// longer true. Every run now shares one constant group, so they serialize and
// cannot overlap. The noise argument is the one that survives, and it is
// sufficient by itself.
func (c *Config) contextRequestsEnabled() bool {
	return c.RequestContext && c.SinglePR != 0
}

// components returns the configured component names in a stable order, for the
// prompt and for log lines.
func (c *Config) components() []string {
	names := make([]string, 0, len(c.OwnerMap))
	for name := range c.OwnerMap {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// parsePRNumber reads a pull request number from an environment string. An
// empty value means "no single pull request" (batch mode); anything that is not
// a non-negative integer is an error rather than a silent fallback.
func parsePRNumber(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(s)
	// Zero is rejected rather than accepted: it is not a pull request number, and
	// letting it through would make a maintainer who typed 0 into the dispatch
	// input get a batch run over the backlog instead.
	if err != nil || n < 1 {
		return 0, fmt.Errorf("PULL_REQUEST_NUMBER must be a positive integer, got %q", s)
	}
	return n, nil
}

// parseOwnerMap parses "component=login,component=login" into the map of
// components the model may choose from.
//
// Both halves are validated: a component name must be prompt-safe (no braces,
// which llmagent would read as a session-state reference) and a login must
// match GitHub's own login rule. Validating here rather than at assign time
// means the complete set of strings this bot can ever POST as an assignee is
// fixed before the model is constructed.
func parseOwnerMap(s string) (map[string]string, error) {
	out := make(map[string]string)
	for _, entry := range strings.Split(s, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, login, ok := strings.Cut(entry, "=")
		if !ok {
			return nil, fmt.Errorf("OWNER_MAP entry %q is not in component=login form", entry)
		}
		name = strings.ToLower(strings.TrimSpace(name))
		login = strings.TrimSpace(login)
		if !componentPattern.MatchString(name) {
			return nil, fmt.Errorf("OWNER_MAP component %q is not a valid component name", name)
		}
		if !validLogin(login, false) {
			return nil, fmt.Errorf("OWNER_MAP owner %q for component %q is not a valid GitHub login", login, name)
		}
		if prev, dup := out[name]; dup {
			return nil, fmt.Errorf("OWNER_MAP defines component %q twice (%q and %q)", name, prev, login)
		}
		out[name] = login
	}
	return out, nil
}

// Environment helpers.
//
// A malformed value is an error, never a silent fall back to the default. The
// default is not a safe guess: DRY_RUN=yes would mean a LIVE run, and
// MAX_FILES=1OO would quietly change how much of the pull request the model
// sees. envErrs collects them so one run reports every one of them.

// envErrs accumulates malformed environment values.
type envErrs struct{ msgs []string }

func (e *envErrs) bad(key, val, want string) {
	e.msgs = append(e.msgs, fmt.Sprintf("%s=%q is not %s", key, val, want))
}

func (e *envErrs) err() error {
	if len(e.msgs) == 0 {
		return nil
	}
	return fmt.Errorf("invalid configuration: %s", strings.Join(e.msgs, "; "))
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

func envInt(e *envErrs, key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		e.bad(key, v, "an integer")
		return def
	}
	return n
}

func envBool(e *envErrs, key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		e.bad(key, v, "a boolean (true/false/1/0)")
		return def
	}
	return b
}

func envDuration(e *envErrs, key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		e.bad(key, v, "a duration (for example 90s or 10m)")
		return def
	}
	return d
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
