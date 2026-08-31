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
	t.Setenv("OWNER_MAP", "core=alice,tools=bob")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	for _, k := range []string{
		"LLM_MODEL_NAME", "REQUEST_CONTEXT", "PULL_REQUEST_NUMBER", "PR_COUNT",
		"MAX_FILES", "CONCURRENCY_LIMIT", "PR_TIMEOUT", "RUN_BUDGET", "DRY_RUN",
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
	if !cfg.RequestContext {
		t.Error("default RequestContext = false, want true")
	}
	if cfg.SinglePR != 0 {
		t.Errorf("default SinglePR = %d, want 0 (batch mode)", cfg.SinglePR)
	}
	if cfg.PRCount != 10 {
		t.Errorf("default PRCount = %d, want 10", cfg.PRCount)
	}
	if cfg.MaxFiles != 50 {
		t.Errorf("default MaxFiles = %d, want 50", cfg.MaxFiles)
	}
	if cfg.Concurrency != 3 {
		t.Errorf("default Concurrency = %d, want 3", cfg.Concurrency)
	}
	if cfg.PRTimeout != 5*time.Minute {
		t.Errorf("default PRTimeout = %v, want 5m", cfg.PRTimeout)
	}
	if cfg.RunBudget != 10*time.Minute {
		t.Errorf("default RunBudget = %v, want 10m", cfg.RunBudget)
	}
	if cfg.DryRun {
		t.Error("default DryRun = true, want false")
	}
	if want := map[string]string{"core": "alice", "tools": "bob"}; !reflect.DeepEqual(cfg.OwnerMap, want) {
		t.Errorf("OwnerMap = %v, want %v", cfg.OwnerMap, want)
	}
}

// The run budget exists so the process reports its own overrun instead of being
// killed by the workflow. It must therefore stay strictly below the workflow's
// timeout-minutes, which the shipped workflow sets to 15.
func TestDefaultRunBudgetLeavesRoomBelowTheWorkflowTimeout(t *testing.T) {
	const workflowTimeout = 15 * time.Minute
	if defaultRunBudget >= workflowTimeout {
		t.Errorf("defaultRunBudget = %v, must be below the workflow timeout of %v", defaultRunBudget, workflowTimeout)
	}
	if defaultPRTimeout > defaultRunBudget {
		t.Errorf("defaultPRTimeout = %v exceeds defaultRunBudget = %v; one pull request could consume the whole run",
			defaultPRTimeout, defaultRunBudget)
	}
}

func TestLoadConfigMissingRequired(t *testing.T) {
	for _, unset := range []string{"OWNER", "REPO", "GITHUB_TOKEN", "OWNER_MAP"} {
		t.Run(unset, func(t *testing.T) {
			setRequired(t)
			t.Setenv(unset, "")
			_, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig() with %s unset returned no error", unset)
			}
			if !strings.Contains(err.Error(), unset) {
				t.Errorf("error %q does not name the missing %s", err, unset)
			}
		})
	}
}

// A workflow_dispatch typo in the pull request number must fail loudly. Falling
// back to 0 would silently turn a request to triage one pull request into a
// batch run over ten.
func TestLoadConfigRejectsMalformedPullRequestNumber(t *testing.T) {
	for _, bad := range []string{"abc", "12x", "-3", "1 2", "3.0"} {
		t.Run(bad, func(t *testing.T) {
			setRequired(t)
			t.Setenv("PULL_REQUEST_NUMBER", bad)
			cfg, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig() accepted PULL_REQUEST_NUMBER=%q and produced SinglePR=%d", bad, cfg.SinglePR)
			}
		})
	}
}

func TestLoadConfigReadsPullRequestNumberFromEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("PULL_REQUEST_NUMBER", "417")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.SinglePR != 417 {
		t.Errorf("SinglePR = %d, want 417", cfg.SinglePR)
	}
	// The flag must still win, so a local run can override the environment.
	cfg, err = loadConfig([]string{"-pr", "9"})
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if cfg.SinglePR != 9 {
		t.Errorf("SinglePR with -pr=9 = %d, want 9", cfg.SinglePR)
	}
}

func TestLoadConfigClampsBounds(t *testing.T) {
	for _, tc := range []struct {
		env, val                                     string
		wantCount, wantConc, wantFiles               int
		wantPRTimeout, wantBudget                    time.Duration
		name                                         string
		checkCount, checkConc, checkFiles, checkTime bool
	}{
		{env: "PR_COUNT", val: "0", wantCount: 1, checkCount: true, name: "PR_COUNT floor"},
		{env: "PR_COUNT", val: "5000", wantCount: maxPRCount, checkCount: true, name: "PR_COUNT ceiling"},
		{env: "CONCURRENCY_LIMIT", val: "-4", wantConc: 1, checkConc: true, name: "concurrency floor"},
		{env: "CONCURRENCY_LIMIT", val: "999", wantConc: maxConcurrency, checkConc: true, name: "concurrency ceiling"},
		{env: "MAX_FILES", val: "0", wantFiles: 1, checkFiles: true, name: "max files floor"},
		{env: "MAX_FILES", val: "100000", wantFiles: maxFilesLimit, checkFiles: true, name: "max files ceiling"},
		{
			env: "PR_TIMEOUT", val: "0s", wantPRTimeout: defaultPRTimeout, wantBudget: defaultRunBudget,
			checkTime: true, name: "non-positive timeout falls back to the default",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.env, tc.val)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("loadConfig() error = %v", err)
			}
			if tc.checkCount && cfg.PRCount != tc.wantCount {
				t.Errorf("PRCount = %d, want %d", cfg.PRCount, tc.wantCount)
			}
			if tc.checkConc && cfg.Concurrency != tc.wantConc {
				t.Errorf("Concurrency = %d, want %d", cfg.Concurrency, tc.wantConc)
			}
			if tc.checkFiles && cfg.MaxFiles != tc.wantFiles {
				t.Errorf("MaxFiles = %d, want %d", cfg.MaxFiles, tc.wantFiles)
			}
			if tc.checkTime {
				if cfg.PRTimeout != tc.wantPRTimeout {
					t.Errorf("PRTimeout = %v, want %v", cfg.PRTimeout, tc.wantPRTimeout)
				}
				if cfg.RunBudget != tc.wantBudget {
					t.Errorf("RunBudget = %v, want %v", cfg.RunBudget, tc.wantBudget)
				}
			}
		})
	}
}

func TestLoadConfigVertexInsteadOfAPIKey(t *testing.T) {
	setRequired(t)
	t.Setenv("GEMINI_API_KEY", "")
	if _, err := loadConfig(nil); err == nil {
		t.Fatal("loadConfig() with no key and no Vertex flag returned no error")
	}
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	if _, err := loadConfig(nil); err != nil {
		t.Errorf("loadConfig() with Vertex enabled error = %v", err)
	}
}

// The owner map is the entire bound on who this bot can assign. Every rejection
// here is a login it can therefore never POST.
func TestParseOwnerMap(t *testing.T) {
	for _, tc := range []struct {
		name    string
		in      string
		want    map[string]string
		wantErr string
	}{
		{name: "simple", in: "core=alice,tools=bob", want: map[string]string{"core": "alice", "tools": "bob"}},
		{name: "spaces and case", in: " Core = Alice , TOOLS=bob ", want: map[string]string{"core": "Alice", "tools": "bob"}},
		{name: "shared owner", in: "core=alice,web=alice", want: map[string]string{"core": "alice", "web": "alice"}},
		{name: "empty", in: "", want: map[string]string{}},
		{name: "trailing comma", in: "core=alice,", want: map[string]string{"core": "alice"}},
		{name: "no equals", in: "core", wantErr: "component=login form"},
		{name: "empty login", in: "core=", wantErr: "not a valid GitHub login"},
		{name: "login with slash", in: "core=al/ice", wantErr: "not a valid GitHub login"},
		{name: "login with at", in: "core=@alice", wantErr: "not a valid GitHub login"},
		{name: "login too long", in: "core=" + strings.Repeat("a", 40), wantErr: "not a valid GitHub login"},
		{name: "login with double hyphen", in: "core=al--ice", wantErr: "not a valid GitHub login"},
		{name: "empty component", in: "=alice", wantErr: "not a valid component name"},
		{name: "braced component", in: "co{re}=alice", wantErr: "not a valid component name"},
		{name: "newline in component", in: "co\nre=alice", wantErr: "not a valid component name"},
		{name: "duplicate component", in: "core=alice,CORE=bob", wantErr: "twice"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseOwnerMap(tc.in)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("parseOwnerMap(%q) = %v, want error containing %q", tc.in, got, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("parseOwnerMap(%q) error = %q, want it to contain %q", tc.in, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseOwnerMap(%q) error = %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseOwnerMap(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestConfigComponentsIsSorted(t *testing.T) {
	cfg := &Config{OwnerMap: map[string]string{"web": "c", "auth": "a", "core": "b"}}
	if got, want := cfg.components(), []string{"auth", "core", "web"}; !reflect.DeepEqual(got, want) {
		t.Errorf("components() = %v, want %v", got, want)
	}
}
