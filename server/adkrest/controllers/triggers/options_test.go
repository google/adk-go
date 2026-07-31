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

package triggers

import (
	"testing"

	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session/compaction"
)

// TestControllerOptionsAreBackwardCompatible pins that the trigger controller
// constructors still accept their original argument list. The compaction option
// was added variadically precisely so existing callers keep compiling; a switch
// to a required parameter would break this file.
func TestControllerOptionsAreBackwardCompatible(t *testing.T) {
	t.Parallel()

	if got := NewPubSubController(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewPubSubController() with no options returned nil")
	}
	if got := NewEventarcController(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewEventarcController() with no options returned nil")
	}
}

func TestWithEventsCompactionConfig(t *testing.T) {
	t.Parallel()

	cfg := &compaction.Config{CompactionInterval: 3, OverlapSize: 1}
	tc := TriggerConfig{MaxConcurrentRuns: 1}

	tests := []struct {
		name   string
		runner *RetriableRunner
	}{
		{
			name:   "pubsub",
			runner: NewPubSubController(nil, nil, nil, nil, runner.PluginConfig{}, tc, WithEventsCompactionConfig(cfg)).runner,
		},
		{
			name:   "eventarc",
			runner: NewEventarcController(nil, nil, nil, nil, runner.PluginConfig{}, tc, WithEventsCompactionConfig(cfg)).runner,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if tt.runner.eventsCompactionConfig != cfg {
				t.Errorf("eventsCompactionConfig = %v, want the config passed to the option", tt.runner.eventsCompactionConfig)
			}
		})
	}
}

func TestWithEventsCompactionConfigDefaultsToNil(t *testing.T) {
	t.Parallel()

	c := NewPubSubController(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1})
	if c.runner.eventsCompactionConfig != nil {
		t.Errorf("eventsCompactionConfig = %v, want nil when the option is not supplied", c.runner.eventsCompactionConfig)
	}
}
