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

import "testing"

// DRY_RUN is the ONLY channel the workflow uses to request a dry run: it sets
// the environment variable and never passes the -dry-run flag. The rest of the
// config suite asserts dry-run through the flag, so severing the env var from
// the flag default would leave those tests green while a caller's
// `dry_run: true` silently filed a real issue. This test pins the env path.
func TestLoadConfigDryRunFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"true", true},
		{"false", false},
		{"", false},
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

// START_TAG and END_TAG are likewise only ever supplied through the environment
// by the workflow. The flag defaults read them, and a test that only exercises
// the flags would not notice that wiring being severed.
func TestLoadConfigTagsFromEnv(t *testing.T) {
	setRequired(t)
	t.Setenv("START_TAG", "v1.0.0")
	t.Setenv("END_TAG", "v1.1.0")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.StartTag != "v1.0.0" || cfg.EndTag != "v1.1.0" {
		t.Errorf("tags from the environment = %q...%q, want v1.0.0...v1.1.0", cfg.StartTag, cfg.EndTag)
	}
}

// DRY_RUN is the one control whose entire job is to suppress the write, so it
// must fail CLOSED. envBool returns the default on a value it cannot parse, and
// the default is "perform the mutation" -- so `DRY_RUN=yes` would silently file
// a real issue for an operator who asked for a preview.
//
// Mutation that must fail this test: use envBool("DRY_RUN", false) instead of
// envBoolStrict.
func TestLoadConfigRejectsAnUnparseableDryRun(t *testing.T) {
	for _, v := range []string{"yes", "on", "1 ", "maybe"} {
		t.Run("DRY_RUN="+v, func(t *testing.T) {
			setRequired(t)
			t.Setenv("DRY_RUN", v)
			if _, err := loadConfig(nil); err == nil {
				t.Fatalf("loadConfig accepted DRY_RUN=%q; the run would have written for real", v)
			}
		})
	}
	// The values the workflow actually sends still work.
	for _, v := range []string{"true", "false", "1", "0", "TRUE"} {
		setRequired(t)
		t.Setenv("DRY_RUN", v)
		if _, err := loadConfig(nil); err != nil {
			t.Errorf("loadConfig rejected DRY_RUN=%q: %v", v, err)
		}
	}
}
