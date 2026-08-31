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
)

// repoPartPattern is the allow-list for a GitHub owner or repository name.
// GitHub itself permits only these characters, so anything else is a
// misconfiguration rather than a name that could not be matched.
var repoPartPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$`)

// Config holds all runtime configuration for the release-docs bot. It is parsed
// once from environment variables and command-line flags and then injected into
// the rest of the program; there is deliberately no package-level mutable state.
type Config struct {
	// Owner and Repo identify the SOURCE repository whose releases are diffed
	// (e.g. "google"/"adk-go").
	Owner string
	Repo  string

	// TargetOwner and TargetRepo identify the repository the issue is filed in.
	// They default to the source repository, because the built-in Actions
	// GITHUB_TOKEN can only write to the repository it was issued for. Pointing
	// them at another repository (e.g. google/adk-docs) requires a cross-repo
	// token that this repository may not have; see README.md.
	TargetOwner string
	TargetRepo  string

	// GitHubToken authenticates GitHub REST calls. In GitHub Actions this is
	// the auto-provided github-actions[bot] token, authorized via the workflow
	// permissions block.
	GitHubToken string

	// GeminiAPIKey authenticates the Gemini (AI Studio) model.
	GeminiAPIKey string

	// Model is the Gemini model name used for the analysis.
	Model string

	// StartTag and EndTag bound the diff. EndTag empty means "the most recent
	// release"; StartTag empty means "the release published before EndTag".
	// Both are validated against tagPattern before use.
	StartTag string
	EndTag   string

	// MaxFiles caps how many changed files are analyzed. MaxPatchBytes caps how
	// much of one file's patch is read. MaxCommits caps the commit subjects
	// included. Whatever these drop is reported in the issue.
	MaxFiles      int
	MaxPatchBytes int
	MaxCommits    int

	// FilesPerGroup is how many files one model call analyzes, so a large
	// release does not blow the context window.
	FilesPerGroup int

	// MaxFindingsPerGroup caps how many suggestions one group may record, so a
	// steered model cannot produce an unbounded issue body.
	MaxFindingsPerGroup int

	// RunBudget bounds the whole analysis loop. It is deliberately below the
	// workflow's timeout-minutes so an overrun is reported in the issue rather
	// than the job being killed with nothing to show.
	RunBudget time.Duration

	// GroupTimeout bounds one group's model call.
	GroupTimeout time.Duration

	// DryRun, when true, renders the issue that would be filed and suppresses
	// every mutation.
	DryRun bool
}

// Defaults. Each is both the initial value and the value a non-positive
// override is clamped back to, so the two cannot drift apart.
const (
	defaultModel               = "gemini-flash-latest"
	defaultMaxFiles            = 60
	defaultMaxPatchBytes       = 8000
	defaultMaxCommits          = 100
	defaultFilesPerGroup       = 5
	defaultMaxFindingsPerGroup = 10
	defaultRunBudget           = 15 * time.Minute
	defaultGroupTimeout        = 4 * time.Minute
)

// loadConfig parses configuration from flags (args) and environment variables
// and validates required fields. args is injectable so tests can exercise flag
// parsing.
func loadConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("releasedocsbot", flag.ContinueOnError)
	dryRun := fs.Bool("dry-run", envBool("DRY_RUN", false),
		"render the issue that would be filed without creating it")
	startTag := fs.String("start-tag", os.Getenv("START_TAG"),
		"older release tag (base); empty = the release before -end-tag")
	endTag := fs.String("end-tag", os.Getenv("END_TAG"),
		"newer release tag (head); empty = the most recent release")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	owner, repo := os.Getenv("OWNER"), os.Getenv("REPO")
	cfg := &Config{
		// OWNER/REPO have no default on purpose: a default would silently target
		// a concrete repository if a caller forgot to set them (validate rejects
		// an empty value instead, so misconfiguration fails loudly).
		Owner: owner,
		Repo:  repo,
		// The target defaults to the source, so the built-in token suffices.
		TargetOwner:         envString("TARGET_OWNER", owner),
		TargetRepo:          envString("TARGET_REPO", repo),
		GitHubToken:         os.Getenv("GITHUB_TOKEN"),
		GeminiAPIKey:        firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")),
		Model:               envString("LLM_MODEL_NAME", defaultModel),
		StartTag:            strings.TrimSpace(*startTag),
		EndTag:              strings.TrimSpace(*endTag),
		MaxFiles:            envInt("MAX_FILES", defaultMaxFiles),
		MaxPatchBytes:       envInt("MAX_PATCH_BYTES", defaultMaxPatchBytes),
		MaxCommits:          envInt("MAX_COMMITS", defaultMaxCommits),
		FilesPerGroup:       envInt("FILES_PER_GROUP", defaultFilesPerGroup),
		MaxFindingsPerGroup: envInt("MAX_FINDINGS_PER_GROUP", defaultMaxFindingsPerGroup),
		RunBudget:           envDuration("RUN_BUDGET", defaultRunBudget),
		GroupTimeout:        envDuration("GROUP_TIMEOUT", defaultGroupTimeout),
		DryRun:              *dryRun,
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
	for _, f := range []struct{ name, value string }{
		{"OWNER", c.Owner},
		{"REPO", c.Repo},
		{"TARGET_OWNER", c.TargetOwner},
		{"TARGET_REPO", c.TargetRepo},
	} {
		if f.value == "" {
			missing = append(missing, f.name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required configuration: %s", strings.Join(missing, ", "))
	}

	// Owner and repository names are interpolated into API paths and into the
	// duplicate-check search query, where a stray space or colon would change
	// which repository is queried. They come from workflow configuration rather
	// than from a contributor, but validating them is a line of code.
	for _, f := range []struct{ name, value string }{
		{"OWNER", c.Owner},
		{"REPO", c.Repo},
		{"TARGET_OWNER", c.TargetOwner},
		{"TARGET_REPO", c.TargetRepo},
	} {
		if !repoPartPattern.MatchString(f.value) {
			return fmt.Errorf("%s %q is not a valid GitHub owner or repository name", f.name, f.value)
		}
	}

	// Reject a malformed tag here rather than at the point of use. A tag is
	// interpolated into the compare API path and into the issue marker, so an
	// unvalidated one could reshape either.
	for _, t := range []struct{ name, value string }{
		{"START_TAG", c.StartTag}, {"END_TAG", c.EndTag},
	} {
		if t.value != "" && !validTag(t.value) {
			return fmt.Errorf("%s %q is not a valid release tag", t.name, t.value)
		}
	}

	clampMin(&c.MaxFiles, 1, defaultMaxFiles)
	clampMin(&c.MaxPatchBytes, 1, defaultMaxPatchBytes)
	clampMin(&c.MaxCommits, 0, defaultMaxCommits)
	clampMin(&c.FilesPerGroup, 1, defaultFilesPerGroup)
	clampMin(&c.MaxFindingsPerGroup, 1, defaultMaxFindingsPerGroup)
	if c.RunBudget <= 0 {
		c.RunBudget = defaultRunBudget
	}
	if c.GroupTimeout <= 0 {
		c.GroupTimeout = defaultGroupTimeout
	}
	return nil
}

// crossRepoWarning returns a warning when the issue target differs from the
// source repository. That configuration needs a token with issues:write on the
// target, which the built-in Actions GITHUB_TOKEN does not have.
func crossRepoWarning(c *Config) string {
	if c.TargetOwner == c.Owner && c.TargetRepo == c.Repo {
		return ""
	}
	return fmt.Sprintf(
		"filing into %s/%s but diffing %s/%s: this needs a token with issues:write on the target repository; "+
			"the built-in GITHUB_TOKEN cannot write across repositories",
		c.TargetOwner, c.TargetRepo, c.Owner, c.Repo,
	)
}

// clampMin replaces a value below floor with def, so a nonsensical override
// falls back to the documented default rather than disabling a cap.
func clampMin(v *int, floor, def int) {
	if *v < floor {
		*v = def
	}
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

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
