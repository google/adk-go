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
	"reflect"
	"testing"
	"time"
)

// setRequired sets the minimum required configuration and clears optional env
// vars so defaults are observable. OWNER/REPO are required (no default), so they
// are part of the minimum here.
func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("GITHUB_TOKEN", "test-token")
	t.Setenv("GEMINI_API_KEY", "test-key")
	t.Setenv("OWNER", "google")
	t.Setenv("REPO", "adk-go")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	for _, k := range []string{
		"LLM_MODEL_NAME", "ALLOWED_LABELS",
		"ISSUE_COUNT", "FRESHNESS_WINDOW_DAYS", "ISSUE_TIMEOUT", "SWEEP_TIMEOUT", "DRY_RUN",
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
		t.Errorf("owner/repo = %s/%s, want google/adk-go", cfg.Owner, cfg.Repo)
	}
	if cfg.Model != "gemini-flash-latest" {
		t.Errorf("default model = %q, want gemini-flash-latest", cfg.Model)
	}
	if cfg.IssueCount != 3 {
		t.Errorf("default IssueCount = %d, want 3", cfg.IssueCount)
	}
	if cfg.FreshnessWindow != 0 {
		t.Errorf("default FreshnessWindow = %v, want 0", cfg.FreshnessWindow)
	}
	if cfg.DryRun {
		t.Error("default DryRun = true, want false")
	}
	if !reflect.DeepEqual(cfg.AllowedLabels, defaultAllowedLabels) {
		t.Errorf("default AllowedLabels = %v, want %v", cfg.AllowedLabels, defaultAllowedLabels)
	}
}

func TestLoadConfigFlagsAndEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("OWNER", "acme")
	t.Setenv("REPO", "widgets")
	t.Setenv("ALLOWED_LABELS", "bug, enhancement ,docs")
	t.Setenv("ISSUE_COUNT", "7")
	// The budget has to fit 7 issues plus the work-set fetch.
	t.Setenv("SWEEP_TIMEOUT", "40m")
	t.Setenv("FRESHNESS_WINDOW_DAYS", "30")

	cfg, err := loadConfig([]string{"-dry-run", "-issue", "42"})
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.Owner != "acme" || cfg.Repo != "widgets" {
		t.Errorf("owner/repo = %s/%s", cfg.Owner, cfg.Repo)
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
	if cfg.FreshnessWindow != 30*24*time.Hour {
		t.Errorf("FreshnessWindow = %v, want 720h", cfg.FreshnessWindow)
	}
	want := []string{"bug", "enhancement", "docs"}
	if !reflect.DeepEqual(cfg.AllowedLabels, want) {
		t.Errorf("AllowedLabels = %v, want %v", cfg.AllowedLabels, want)
	}
}

func TestLoadConfigMissingToken(t *testing.T) {
	setRequired(t)
	t.Setenv("GITHUB_TOKEN", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() expected error for missing GITHUB_TOKEN, got nil")
	}
}

func TestLoadConfigMissingOwnerRepo(t *testing.T) {
	tests := []struct {
		name  string
		unset string
	}{
		{"missing owner", "OWNER"},
		{"missing repo", "REPO"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.unset, "")
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig() with empty %s = nil error, want error (no silent default)", tc.unset)
			}
		})
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

// The -issue flag must read its argument as the decimal number the operator
// typed, and must refuse anything that is not a usable issue number.
//
// flag.Int would parse with strconv.ParseInt base 0, so "-issue 010" would be
// octal 8 and the bot would triage a different issue than the one named -- and
// "-issue 0" would fall through the SingleIssue > 0 check in selectIssues and
// silently become a full backlog sweep.
//
// Killing mutation: go back to fs.Int for the flag.
func TestLoadConfigIssueFlag(t *testing.T) {
	for _, tc := range []struct {
		arg     string
		want    int
		wantErr bool
	}{
		{"42", 42, false},
		{"010", 10, false}, // decimal ten, not octal eight
		{"0", 0, true},     // not an issue number, and not a request to sweep
		{"-5", 0, true},
		{"1e3", 0, true},
		{"abc", 0, true},
	} {
		t.Run("-issue="+tc.arg, func(t *testing.T) {
			setRequired(t)
			cfg, err := loadConfig([]string{"-issue", tc.arg})
			if tc.wantErr {
				if err == nil {
					t.Fatalf("loadConfig(-issue %s) = %+v, nil error; want a rejection", tc.arg, cfg)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadConfig(-issue %s) error = %v", tc.arg, err)
			}
			if cfg.SingleIssue != tc.want {
				t.Errorf("SingleIssue = %d, want %d", cfg.SingleIssue, tc.want)
			}
		})
	}
}

// Omitting the flag means "sweep", which must stay distinguishable from a bad
// value rather than collapsing into it.
func TestLoadConfigNoIssueFlagMeansSweep(t *testing.T) {
	setRequired(t)
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.SingleIssue != 0 {
		t.Errorf("SingleIssue = %d, want 0 (sweep)", cfg.SingleIssue)
	}
}

// envReader.days must reject a value ParseFloat accepts but a Duration cannot
// represent. Tested on the reader directly rather than through loadConfig,
// because the two failure directions are architecture-dependent: converting NaN
// to int64 is implementation-defined in Go, so on amd64 it lands negative and
// validate() catches it, while on a saturating target it lands on 0 -- which
// reads as "no freshness window" and silently sweeps the whole backlog. Only
// the reader itself can be asserted the same way on both.
//
// Killing mutation: drop the finiteness check from envReader.days.
func TestEnvReaderDaysRejectsWhatADurationCannotHold(t *testing.T) {
	for _, v := range []string{"NaN", "Inf", "+Inf", "-Inf", "1e300", "-1e300"} {
		t.Run(v, func(t *testing.T) {
			t.Setenv("FRESHNESS_WINDOW_DAYS", v)
			e := &envReader{}
			got := e.days("FRESHNESS_WINDOW_DAYS", 0)
			if e.err() == nil {
				t.Fatalf("days(%q) recorded no failure, so loadConfig would accept it", v)
			}
			if got != 0 {
				t.Errorf("days(%q) = %v, want the default 0 when the value is rejected", v, got)
			}
		})
	}
}

// The ordinary values must still work, or the guard above is too strict.
func TestEnvReaderDaysAcceptsARealWindow(t *testing.T) {
	for v, want := range map[string]time.Duration{
		"7":   7 * 24 * time.Hour,
		"0.5": 12 * time.Hour,
		"0":   0,
	} {
		t.Run(v, func(t *testing.T) {
			t.Setenv("FRESHNESS_WINDOW_DAYS", v)
			e := &envReader{}
			got := e.days("FRESHNESS_WINDOW_DAYS", time.Hour)
			if err := e.err(); err != nil {
				t.Fatalf("days(%q) = %v, want it accepted", v, err)
			}
			if got != want {
				t.Errorf("days(%q) = %v, want %v", v, got, want)
			}
		})
	}
}
