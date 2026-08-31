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
	"strings"
	"testing"
	"time"
)

// setRequired sets the minimum credentials and clears optional env vars so
// defaults are observable.
func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("GITHUB_TOKEN", "test-token")
	t.Setenv("GEMINI_API_KEY", "test-key")
	t.Setenv("OWNER", "google")
	t.Setenv("REPO", "adk-go")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	for _, k := range []string{
		"TARGET_OWNER", "TARGET_REPO", "LLM_MODEL_NAME", "START_TAG", "END_TAG",
		"MAX_FILES", "MAX_PATCH_BYTES", "MAX_COMMITS", "FILES_PER_GROUP",
		"MAX_FINDINGS_PER_GROUP", "RUN_BUDGET", "GROUP_TIMEOUT", "DRY_RUN",
	} {
		t.Setenv(k, "")
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	setRequired(t)
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig(): %v", err)
	}
	// The issue target defaults to the source repo, which is the only one the
	// built-in Actions token can write to.
	if cfg.TargetOwner != "google" || cfg.TargetRepo != "adk-go" {
		t.Errorf("target = %s/%s, want it defaulted to the source repo", cfg.TargetOwner, cfg.TargetRepo)
	}
	if cfg.Model != defaultModel {
		t.Errorf("model = %q, want %q", cfg.Model, defaultModel)
	}
	if cfg.MaxFiles != defaultMaxFiles || cfg.MaxPatchBytes != defaultMaxPatchBytes {
		t.Errorf("caps = %d files / %d bytes, want %d / %d",
			cfg.MaxFiles, cfg.MaxPatchBytes, defaultMaxFiles, defaultMaxPatchBytes)
	}
	if cfg.FilesPerGroup != defaultFilesPerGroup || cfg.MaxFindingsPerGroup != defaultMaxFindingsPerGroup {
		t.Errorf("grouping = %d per group / %d findings, want %d / %d",
			cfg.FilesPerGroup, cfg.MaxFindingsPerGroup, defaultFilesPerGroup, defaultMaxFindingsPerGroup)
	}
	if cfg.RunBudget != defaultRunBudget || cfg.GroupTimeout != defaultGroupTimeout {
		t.Errorf("budgets = %v / %v, want %v / %v", cfg.RunBudget, cfg.GroupTimeout, defaultRunBudget, defaultGroupTimeout)
	}
	if cfg.DryRun {
		t.Error("DryRun defaulted to true")
	}
	if cfg.StartTag != "" || cfg.EndTag != "" {
		t.Errorf("tags = %q...%q, want both empty (derived at run time)", cfg.StartTag, cfg.EndTag)
	}
}

func TestLoadConfigRequiresCredentialsAndRepo(t *testing.T) {
	for _, unset := range []string{"GITHUB_TOKEN", "OWNER", "REPO"} {
		t.Run(unset, func(t *testing.T) {
			setRequired(t)
			t.Setenv(unset, "")
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig with an empty %s returned no error (a silent default would target the wrong repo)", unset)
			}
		})
	}
	t.Run("model credentials", func(t *testing.T) {
		setRequired(t)
		t.Setenv("GEMINI_API_KEY", "")
		if _, err := loadConfig(nil); err == nil {
			t.Fatal("loadConfig with no model credentials returned no error")
		}
	})
	t.Run("vertex instead of a key", func(t *testing.T) {
		setRequired(t)
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
		if _, err := loadConfig(nil); err != nil {
			t.Fatalf("loadConfig with Vertex configured: %v", err)
		}
	})
}

// A tag reaches the compare API path and the issue marker. Rejecting a bad one
// at load time is what keeps every later use from having to.
//
// Mutation that must fail this test: delete the validTag loop from validate().
func TestLoadConfigRejectsAMalformedTag(t *testing.T) {
	for _, tc := range []struct{ env, value string }{
		{"START_TAG", "../../etc/passwd"},
		{"END_TAG", "v1 v2"},
		{"END_TAG", "a..b"},
	} {
		t.Run(tc.env+"="+tc.value, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.env, tc.value)
			_, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig accepted %s=%q", tc.env, tc.value)
			}
			if !strings.Contains(err.Error(), "not a valid release tag") {
				t.Errorf("error = %v, want it to name the invalid tag", err)
			}
		})
	}
}

// An owner or repository name is interpolated into API paths and into the
// duplicate-check search query, where a space or a colon would change which
// repository is queried.
func TestLoadConfigRejectsAMalformedRepositoryName(t *testing.T) {
	for _, tc := range []struct{ env, value string }{
		{"OWNER", "google adk"},
		{"REPO", "adk-go OR repo:evil/x"},
		{"TARGET_REPO", "../other"},
	} {
		t.Run(tc.env+"="+tc.value, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.env, tc.value)
			_, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig accepted %s=%q", tc.env, tc.value)
			}
			if !strings.Contains(err.Error(), "valid GitHub owner or repository name") {
				t.Errorf("error = %v, want it to name the invalid repository part", err)
			}
		})
	}
}

func TestLoadConfigFlagsAndEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("TARGET_OWNER", "google")
	t.Setenv("TARGET_REPO", "adk-docs")
	t.Setenv("MAX_FILES", "7")
	t.Setenv("MAX_PATCH_BYTES", "123")
	t.Setenv("FILES_PER_GROUP", "2")
	t.Setenv("RUN_BUDGET", "90s")

	cfg, err := loadConfig([]string{"-dry-run", "-start-tag", "v1.0.0", "-end-tag", "v1.1.0"})
	if err != nil {
		t.Fatalf("loadConfig(): %v", err)
	}
	if cfg.TargetOwner != "google" || cfg.TargetRepo != "adk-docs" {
		t.Errorf("target = %s/%s, want google/adk-docs", cfg.TargetOwner, cfg.TargetRepo)
	}
	if !cfg.DryRun {
		t.Error("DryRun = false, want true from the flag")
	}
	if cfg.StartTag != "v1.0.0" || cfg.EndTag != "v1.1.0" {
		t.Errorf("tags = %q...%q, want v1.0.0...v1.1.0 from the flags", cfg.StartTag, cfg.EndTag)
	}
	if cfg.MaxFiles != 7 || cfg.MaxPatchBytes != 123 || cfg.FilesPerGroup != 2 {
		t.Errorf("caps = %d/%d/%d, want 7/123/2", cfg.MaxFiles, cfg.MaxPatchBytes, cfg.FilesPerGroup)
	}
	if cfg.RunBudget != 90*time.Second {
		t.Errorf("RunBudget = %v, want 90s", cfg.RunBudget)
	}
}

// A nonsensical override must fall back to the documented default rather than
// disabling a cap: MAX_FILES=0 with no clamp would analyze nothing, and a
// negative value would panic the slice.
func TestLoadConfigClampsNonsensicalOverrides(t *testing.T) {
	setRequired(t)
	for _, k := range []string{"MAX_FILES", "MAX_PATCH_BYTES", "FILES_PER_GROUP", "MAX_FINDINGS_PER_GROUP"} {
		t.Setenv(k, "-5")
	}
	t.Setenv("RUN_BUDGET", "-1m")
	t.Setenv("GROUP_TIMEOUT", "0s")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig(): %v", err)
	}
	if cfg.MaxFiles != defaultMaxFiles || cfg.MaxPatchBytes != defaultMaxPatchBytes ||
		cfg.FilesPerGroup != defaultFilesPerGroup || cfg.MaxFindingsPerGroup != defaultMaxFindingsPerGroup {
		t.Errorf("negative caps were not clamped: %+v", cfg)
	}
	if cfg.RunBudget != defaultRunBudget || cfg.GroupTimeout != defaultGroupTimeout {
		t.Errorf("non-positive budgets were not clamped: %v / %v", cfg.RunBudget, cfg.GroupTimeout)
	}
}

func TestCrossRepoWarning(t *testing.T) {
	same := &Config{Owner: "google", Repo: "adk-go", TargetOwner: "google", TargetRepo: "adk-go"}
	if w := crossRepoWarning(same); w != "" {
		t.Errorf("warning for a same-repo target = %q, want none", w)
	}
	cross := &Config{Owner: "google", Repo: "adk-go", TargetOwner: "google", TargetRepo: "adk-docs"}
	w := crossRepoWarning(cross)
	if !strings.Contains(w, "adk-docs") || !strings.Contains(w, "GITHUB_TOKEN") {
		t.Errorf("cross-repo warning = %q, want it to name the target and the token limitation", w)
	}
}
