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

package agentengine_test

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/server/agentengine"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestNewHandlerRejectsUnusableCompaction checks that Agent Engine refuses a
// compaction config it cannot serve, at construction.
//
// The config is validated inside runner.New, and this surface builds a runner
// per request, so without a check here the handler is created, the process
// reports healthy, and every request fails with the same error instead.
//
// This pins that the check runs, not how. NewHandler delegates to
// launcher.Config.Validate rather than reaching for the compaction field, so
// that a check added there later reaches this surface too, but a hand-rolled
// copy of today's check would satisfy this test just as well.
func TestNewHandlerRejectsUnusableCompaction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  *compaction.Config
		ok   bool
	}{
		{name: "nil compaction is fine", cfg: nil, ok: true},
		{name: "overlap with no interval", cfg: &compaction.Config{OverlapSize: 2}},
		{name: "no strategy at all", cfg: &compaction.Config{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg := &launcher.Config{
				SessionService:         session.InMemoryService(),
				EventsCompactionConfig: tc.cfg,
			}
			_, err := agentengine.NewHandler(cfg, time.Second, 1<<20, "engine")
			if tc.ok {
				if err != nil {
					t.Errorf("NewHandler() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("NewHandler() accepted a compaction config it cannot serve")
			}
			if !strings.Contains(err.Error(), "EventsCompactionConfig") {
				t.Errorf("error %q does not name the field an operator has to change", err)
			}
		})
	}
}
