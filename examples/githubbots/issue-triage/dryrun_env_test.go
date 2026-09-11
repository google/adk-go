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
)

// DRY_RUN is the ONLY channel the workflow uses to request a dry run: it sets
// the environment variable and never passes the -dry-run flag. The rest of the
// config suite asserts dry-run through the flag, so severing the env var from
// the flag default would leave those tests green while a `dry_run: true`
// dispatch silently mutated live issues. This test pins the env path.
func TestLoadConfigDryRunFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"true", true},
		{"false", false},
		{"", false},
		{"1", true},
		{"0", false},
	} {
		t.Run("DRY_RUN="+tc.env, func(t *testing.T) {
			setRequired(t)
			t.Setenv("DRY_RUN", tc.env)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("loadConfig: %v", err)
			}
			if cfg.DryRun != tc.want {
				t.Errorf("DRY_RUN=%q gave DryRun=%v, want %v", tc.env, cfg.DryRun, tc.want)
			}
		})
	}
}

// A malformed DRY_RUN must abort, not fall back to the default. The default is
// "act for real", so an operator who writes DRY_RUN=yes -- which
// strconv.ParseBool rejects -- would otherwise get live writes having asked for
// a rehearsal. The same reasoning covers the bounds: a malformed
// FRESHNESS_WINDOW_DAYS would silently widen the sweep to the whole backlog.
//
// Killing mutation: return the default instead of recording the error in any
// envReader method.
func TestLoadConfigRejectsAMalformedEnvValue(t *testing.T) {
	for _, tc := range []struct{ key, value string }{
		{"DRY_RUN", "yes"},
		{"DRY_RUN", "on"},
		{"ISSUE_COUNT", "many"},
		{"FRESHNESS_WINDOW_DAYS", "7d"},
		{"ISSUE_TIMEOUT", "5min"},
		{"SWEEP_TIMEOUT", "15"},
		{"GOOGLE_GENAI_USE_VERTEXAI", "yes"},
		// ParseFloat accepts these, and converting either to an int64 is
		// implementation-defined in Go: on a saturating target NaN becomes 0,
		// which reads as "no freshness window" and silently widens the sweep to
		// the whole backlog. Killing mutation: drop the finiteness check from
		// envReader.days.
		{"FRESHNESS_WINDOW_DAYS", "NaN"},
		{"FRESHNESS_WINDOW_DAYS", "Inf"},
		{"FRESHNESS_WINDOW_DAYS", "-Inf"},
		{"FRESHNESS_WINDOW_DAYS", "1e300"},
	} {
		t.Run(tc.key+"="+tc.value, func(t *testing.T) {
			setRequired(t)
			t.Setenv(tc.key, tc.value)
			cfg, err := loadConfig(nil)
			if err == nil {
				t.Fatalf("loadConfig() = %+v, nil error; want a rejection of %s=%q", cfg, tc.key, tc.value)
			}
			if !strings.Contains(err.Error(), tc.key) {
				t.Errorf("loadConfig() = %v, want the error to name %s", err, tc.key)
			}
		})
	}
}
