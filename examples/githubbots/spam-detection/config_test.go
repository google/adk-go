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
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// setRequired sets the minimum credentials and clears optional env vars so
// defaults are observable.
func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("GITHUB_TOKEN", "test-token")
	t.Setenv("GEMINI_API_KEY", "test-key")
	// OWNER/REPO are required (no default), so they are part of the minimum.
	t.Setenv("OWNER", "google")
	t.Setenv("REPO", "adk-go")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	for _, k := range []string{
		"LLM_MODEL_NAME", "SPAM_LABEL_NAME", "MAINTAINERS",
		"ISSUE_COUNT", "FRESHNESS_WINDOW_DAYS", "CONCURRENCY_LIMIT",
		"ISSUE_TIMEOUT", "RUN_TIMEOUT", "DRY_RUN",
	} {
		t.Setenv(k, "")
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	setRequired(t)
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.Owner != "google" || cfg.Repo != "adk-go" {
		t.Errorf("default owner/repo = %s/%s, want google/adk-go", cfg.Owner, cfg.Repo)
	}
	if cfg.Model != "gemini-flash-latest" {
		t.Errorf("default model = %q, want gemini-flash-latest", cfg.Model)
	}
	if cfg.SpamLabel != "spam" {
		t.Errorf("default SpamLabel = %q, want spam", cfg.SpamLabel)
	}
	if cfg.IssueCount != 3 {
		t.Errorf("default IssueCount = %d, want 3", cfg.IssueCount)
	}
	if cfg.Concurrency != 3 {
		t.Errorf("default Concurrency = %d, want 3", cfg.Concurrency)
	}
	if cfg.FreshnessWindow != 0 {
		t.Errorf("default FreshnessWindow = %v, want 0", cfg.FreshnessWindow)
	}
	if cfg.IssueTimeout != 5*time.Minute {
		t.Errorf("default IssueTimeout = %v, want 5m", cfg.IssueTimeout)
	}
	// The workflow sets its job timeout above this budget, so an overrun is
	// reported by the bot rather than the runner killing the job. Nothing
	// asserted the default, and setRequired did not even clear RUN_TIMEOUT, so
	// a developer's own value leaked into every config test.
	if cfg.RunTimeout != 15*time.Minute {
		t.Errorf("default RunTimeout = %v, want 15m", cfg.RunTimeout)
	}
	if cfg.DryRun {
		t.Error("default DryRun = true, want false")
	}
	if len(cfg.Maintainers) != 0 {
		t.Errorf("default Maintainers = %v, want empty", cfg.Maintainers)
	}
}

func TestLoadConfigMissingOwnerRepo(t *testing.T) {
	for _, unset := range []string{"OWNER", "REPO"} {
		t.Run(unset, func(t *testing.T) {
			setRequired(t)
			t.Setenv(unset, "")
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig() with empty %s = nil error, want error (no silent default)", unset)
			}
		})
	}
}

func TestLoadConfigFlagsAndEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("OWNER", "acme")
	t.Setenv("REPO", "widgets")
	t.Setenv("SPAM_LABEL_NAME", "junk")
	t.Setenv("MAINTAINERS", "alice, bob ,carol")
	t.Setenv("ISSUE_COUNT", "7")
	t.Setenv("CONCURRENCY_LIMIT", "5")
	t.Setenv("FRESHNESS_WINDOW_DAYS", "30")

	cfg, err := loadConfig([]string{"-dry-run", "-issue", "42"})
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.Owner != "acme" || cfg.Repo != "widgets" {
		t.Errorf("owner/repo = %s/%s", cfg.Owner, cfg.Repo)
	}
	if cfg.SpamLabel != "junk" {
		t.Errorf("SpamLabel = %q, want junk", cfg.SpamLabel)
	}
	if !cfg.DryRun {
		t.Error("DryRun = false, want true from flag")
	}
	if cfg.SingleIssue != 42 {
		t.Errorf("SingleIssue = %d, want 42", cfg.SingleIssue)
	}
	if cfg.IssueCount != 7 {
		t.Errorf("IssueCount = %d, want 7", cfg.IssueCount)
	}
	if cfg.Concurrency != 5 {
		t.Errorf("Concurrency = %d, want 5", cfg.Concurrency)
	}
	if cfg.FreshnessWindow != 30*24*time.Hour {
		t.Errorf("FreshnessWindow = %v, want 720h", cfg.FreshnessWindow)
	}
	want := []string{"alice", "bob", "carol"}
	if !reflect.DeepEqual(cfg.Maintainers, want) {
		t.Errorf("Maintainers = %v, want %v", cfg.Maintainers, want)
	}
}

func TestLoadConfigMissingToken(t *testing.T) {
	setRequired(t)
	t.Setenv("GITHUB_TOKEN", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() expected error for missing GITHUB_TOKEN, got nil")
	}
}

func TestLoadConfigVertexFallback(t *testing.T) {
	setRequired(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	// No API key, but Vertex enabled -> should succeed.
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	if _, err := loadConfig(nil); err != nil {
		t.Fatalf("loadConfig() with Vertex fallback error = %v", err)
	}
}

func TestLoadConfigMissingModelCreds(t *testing.T) {
	setRequired(t)
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() expected error for missing model credentials, got nil")
	}
}

// The clamps are the last line of defence on a hostile or mistyped value.
// CONCURRENCY_LIMIT=0 in particular would make errgroup.SetLimit(0) block the
// first goroutine forever, and a negative value would remove the bound entirely.
func TestLoadConfigClampsOutOfRangeValues(t *testing.T) {
	for _, tc := range []struct {
		env, value string
		check      func(*testing.T, *Config)
	}{
		{"CONCURRENCY_LIMIT", "0", func(t *testing.T, c *Config) {
			if c.Concurrency < 1 {
				t.Errorf("Concurrency = %d; SetLimit(%d) would hang the sweep", c.Concurrency, c.Concurrency)
			}
		}},
		{"CONCURRENCY_LIMIT", "-1", func(t *testing.T, c *Config) {
			if c.Concurrency < 1 {
				t.Errorf("Concurrency = %d; a negative limit removes the bound on parallel writes", c.Concurrency)
			}
		}},
		{"ISSUE_COUNT", "0", func(t *testing.T, c *Config) {
			if c.IssueCount < 1 {
				t.Errorf("IssueCount = %d; the sweep would review nothing", c.IssueCount)
			}
		}},
		{"ISSUE_TIMEOUT", "0s", func(t *testing.T, c *Config) {
			if c.IssueTimeout <= 0 {
				t.Errorf("IssueTimeout = %v; every review would start already expired", c.IssueTimeout)
			}
		}},
		{"RUN_TIMEOUT", "-5m", func(t *testing.T, c *Config) {
			if c.RunTimeout <= 0 {
				t.Errorf("RunTimeout = %v; the budget would be exhausted before the sweep began", c.RunTimeout)
			}
		}},
		{"FRESHNESS_WINDOW_DAYS", "-3", func(t *testing.T, c *Config) {
			if c.FreshnessWindow < 0 {
				t.Errorf("FreshnessWindow = %v; the search cutoff would be in the future", c.FreshnessWindow)
			}
		}},
	} {
		t.Run(tc.env+"="+tc.value, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.env, tc.value)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("loadConfig: %v", err)
			}
			tc.check(t, cfg)
		})
	}
}

func TestParseIssueNumber(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    int
		wantErr bool
	}{
		{"", 0, false},
		{"123", 123, false},
		// Go's flag.Int would read this as octal 83 and review the wrong issue.
		{"0123", 0, true},
		{"08", 0, true},
		{"0", 0, true},
		{"-4", 0, true},
		{"12x", 0, true},
		{"0x10", 0, true},
		// strconv.Atoi accepts a sign, so the leading-zero guard has to look at
		// the string's shape rather than at s[0] alone.
		{"+5", 0, true},
		{"+05", 0, true},
		{" 7", 0, true},
	} {
		t.Run("issue="+tc.in, func(t *testing.T) {
			got, err := parseIssueNumber(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseIssueNumber(%q) = %d, want an error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseIssueNumber(%q) error = %v", tc.in, err)
			}
			if got != tc.want {
				t.Errorf("parseIssueNumber(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

// The label is interpolated into a search query with %q, which escapes a quote,
// a backslash, a non-printable rune and an invalid UTF-8 byte alike. Any of
// those silently changes what the query means, so all four are rejected.
func TestLoadConfigRejectsALabelThatCannotSurviveQuoting(t *testing.T) {
	for name, label := range map[string]string{
		"quote":        `spam"bot`,
		"backslash":    `spam\bot`,
		"tab":          "spam\tbot",
		"newline":      "spam\nbot",
		"invalid utf8": "spam\xffbot",
	} {
		t.Run(name, func(t *testing.T) {
			setRequired(t)
			t.Setenv("SPAM_LABEL_NAME", label)
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("the spam label %q was accepted; the -label: exclusion would silently stop matching", label)
			}
		})
	}
}

// A .env in the checked-out repository must not reach a CI run: the job holds
// issues: write, and .env would otherwise set anything the workflow leaves
// unset. Locally it must still be read, which is the whole point of the file.
func TestLoadDotEnvSkipsCI(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".env"),
		[]byte("SPAM_DETECTION_PROBE_CI=from-dotenv\nSPAM_DETECTION_PROBE_LOCAL=from-dotenv\n"), 0o600); err != nil {
		t.Fatalf("write .env: %v", err)
	}
	t.Chdir(dir)
	// godotenv never overrides a variable that is already set, so each probe must
	// be genuinely absent rather than set to "" -- and each subtest needs its own,
	// because -shuffle does not permute subtests and sharing one would make their
	// order load-bearing.
	t.Cleanup(func() {
		_ = os.Unsetenv("SPAM_DETECTION_PROBE_CI")
		_ = os.Unsetenv("SPAM_DETECTION_PROBE_LOCAL")
	})

	t.Run("CI set", func(t *testing.T) {
		t.Setenv("CI", "true")
		if loadDotEnv() {
			t.Error("loadDotEnv read a .env inside CI")
		}
		if got, ok := os.LookupEnv("SPAM_DETECTION_PROBE_CI"); ok {
			t.Errorf("the .env leaked into a CI run: SPAM_DETECTION_PROBE=%q", got)
		}
	})
	t.Run("CI unset", func(t *testing.T) {
		t.Setenv("CI", "")
		if !loadDotEnv() {
			t.Fatal("loadDotEnv skipped the .env outside CI")
		}
		if got := os.Getenv("SPAM_DETECTION_PROBE_LOCAL"); got != "from-dotenv" {
			t.Errorf("SPAM_DETECTION_PROBE_LOCAL=%q, want the value from .env", got)
		}
	})
}

// A legitimate non-ASCII label must be accepted. The quoting check rejects
// anything strconv.Quote escapes, and without a positive case nothing pins that
// it is not simply rejecting everything unusual.
func TestLoadConfigAcceptsAPrintableNonASCIILabel(t *testing.T) {
	for _, label := range []string{"spam-böt", "спам", "🐛 spam"} {
		t.Run(label, func(t *testing.T) {
			setRequired(t)
			t.Setenv("SPAM_LABEL_NAME", label)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("the label %q was rejected: %v", label, err)
			}
			if cfg.SpamLabel != label {
				t.Errorf("SpamLabel = %q, want %q", cfg.SpamLabel, label)
			}
		})
	}
}

// OWNER, REPO and BOT_LOGIN are interpolated into the search query and into
// every REST path the client builds, and BOT_LOGIN additionally decides which
// comments the bot treats as its own. A shape check cannot tell a right value
// from a wrong one, but it rejects the ones that could only be a mistake.
func TestLoadConfigRejectsMalformedRepositoryCoordinates(t *testing.T) {
	for name, env := range map[string]struct{ key, value string }{
		"owner with a slash":    {"OWNER", "google/adk-go"},
		"owner with a space":    {"OWNER", "goo gle"},
		"owner with a query op": {"OWNER", "google is:issue"},
		"repo with a slash":     {"REPO", "adk-go/../victim"},
		"repo with a quote":     {"REPO", `adk"go`},
		"login with a slash":    {"BOT_LOGIN", "org/bot"},
		"login with a space":    {"BOT_LOGIN", "github actions"},
		"login with zero width": {"BOT_LOGIN", "github-actions\u200b[bot]"},
	} {
		t.Run(name, func(t *testing.T) {
			setRequired(t)
			t.Setenv(env.key, env.value)
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("%s=%q was accepted", env.key, env.value)
			}
		})
	}
}

func TestLoadConfigAcceptsTheActionsBotLogin(t *testing.T) {
	setRequired(t)
	t.Setenv("BOT_LOGIN", "github-actions[bot]")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("the login the workflow ships was rejected: %v", err)
	}
	if cfg.BotLogin != "github-actions[bot]" {
		t.Errorf("BotLogin = %q, want the configured value", cfg.BotLogin)
	}
}

func TestLoadConfigRejectsPathSegmentRepositoryNames(t *testing.T) {
	// "." and ".." become path segments in every REST URL the client builds.
	for _, tc := range []struct{ key, value string }{
		{"OWNER", "."},
		{"OWNER", ".."},
		{"REPO", ".."},
		{"BOT_LOGIN", "-leading-hyphen"},
	} {
		t.Run(tc.key+"="+tc.value, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.key, tc.value)
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("%s=%q was accepted", tc.key, tc.value)
			}
		})
	}
}

// ".github" is a real repository name, and rejecting every leading dot to keep
// "." and ".." out of a URL path would lock it out.
func TestLoadConfigAcceptsALeadingDotRepositoryName(t *testing.T) {
	setRequired(t)
	t.Setenv("REPO", ".github")
	if _, err := loadConfig(nil); err != nil {
		t.Fatalf("REPO=.github was rejected: %v", err)
	}
}

// An older GitHub account may carry an underscore, and the check must not lock
// such a login out of BOT_LOGIN.
func TestLoadConfigAcceptsALegacyUnderscoreLogin(t *testing.T) {
	setRequired(t)
	t.Setenv("BOT_LOGIN", "some_bot")
	if _, err := loadConfig(nil); err != nil {
		t.Fatalf("a legacy login with an underscore was rejected: %v", err)
	}
}
