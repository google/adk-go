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

package launcher_test

import (
	"strings"
	"testing"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/session/compaction"
)

// TestConfigValidateRejectsUnusableCompaction checks that a launcher refuses to
// start on a compaction config that cannot work.
//
// The config is validated inside runner.New, and a runner is built per request,
// so without this the process starts cleanly and then fails every request with
// an error that names nothing an operator can act on.
func TestConfigValidateRejectsUnusableCompaction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  *compaction.Config
		ok   bool
	}{
		{name: "nil is fine", cfg: nil, ok: true},
		{name: "valid sliding window", cfg: &compaction.Config{CompactionInterval: 2}, ok: true},
		{name: "overlap with no interval", cfg: &compaction.Config{OverlapSize: 2}},
		{name: "threshold with no retention", cfg: &compaction.Config{TokenThreshold: 100}},
		{name: "no strategy at all", cfg: &compaction.Config{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := (&launcher.Config{EventsCompactionConfig: tc.cfg}).Validate()
			if tc.ok {
				if err != nil {
					t.Errorf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Validate() accepted a config that cannot work")
			}
			if !strings.Contains(err.Error(), "EventsCompactionConfig") {
				t.Errorf("error %q does not name the field", err)
			}
		})
	}
}

// TestFullLauncherRefusesUnusableCompaction drives the entry point a program
// actually reaches, rather than Validate on its own.
//
// Only a launcher.Launcher has Execute, and full.NewLauncher and
// universal.NewLauncher are the only two. console.NewLauncher and
// web.NewLauncher return a launcher.SubLauncher, whose interface has Run and
// no Execute, so the Execute methods on those two concrete types cannot be
// called through the exported API at all: the universal launcher dispatches to
// Run. That makes this the one place the check has to hold, and the arguments
// are rejected before any of them are parsed.
func TestFullLauncherRefusesUnusableCompaction(t *testing.T) {
	t.Parallel()

	cfg := &launcher.Config{EventsCompactionConfig: &compaction.Config{OverlapSize: 2}}
	err := full.NewLauncher().Execute(t.Context(), cfg, []string{"console"})
	if err == nil {
		t.Fatal("Execute() started on a compaction config that cannot work")
	}
	if !strings.Contains(err.Error(), "EventsCompactionConfig") {
		t.Errorf("error %q does not name the field an operator has to change", err)
	}
}
