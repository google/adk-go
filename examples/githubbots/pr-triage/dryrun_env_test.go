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
// the environment variable and never passes the -dry-run flag. Severing the env
// var from the flag default would leave the flag-based tests green while a
// dispatch with dry_run: true silently mutated live pull requests.
func TestLoadConfigDryRunFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"true", true},
		{"1", true},
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

// REQUEST_CONTEXT is likewise env-only, and it decides whether a whole tool
// exists. Defaulting it wrong in either direction changes what the bot can do.
func TestLoadConfigRequestContextFromEnv(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"", true},
		{"true", true},
		{"false", false},
		{"0", false},
	} {
		t.Run("REQUEST_CONTEXT="+tc.env, func(t *testing.T) {
			setRequired(t)
			t.Setenv("REQUEST_CONTEXT", tc.env)
			cfg, err := loadConfig(nil)
			if err != nil {
				t.Fatalf("loadConfig: %v", err)
			}
			if cfg.RequestContext != tc.want {
				t.Errorf("REQUEST_CONTEXT=%q gave RequestContext=%v, want %v", tc.env, cfg.RequestContext, tc.want)
			}
		})
	}
}
