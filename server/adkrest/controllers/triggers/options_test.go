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
	"bytes"
	"context"
	"log"
	"os"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestControllerConstructorTypesAreUnchanged pins the exported *type* of the
// trigger constructors, not merely that a call compiles.
//
// This is the assertion the previous version of this test was missing. A plain
// call expression still compiles after a trailing variadic parameter is added,
// so it cannot catch the one change that actually breaks downstream code:
// anything that referenced the constructor as a value, or stored it in a field
// of that function type, stops compiling. Assigning to an explicit function
// type is what makes the signature part of the contract.
func TestControllerConstructorTypesAreUnchanged(t *testing.T) {
	t.Parallel()

	if got := pubSubCtor(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewPubSubController() returned nil")
	}
	if got := eventarcCtor(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewEventarcController() returned nil")
	}
}

// The declared types are the assertion: assigning each constructor to an
// explicit function type fails to compile if its signature changes, including
// by gaining a trailing variadic parameter, which an ordinary call expression
// would still accept.
var (
	pubSubCtor   NewPubSubControllerFunc   = NewPubSubController
	eventarcCtor NewEventarcControllerFunc = NewEventarcController
)

// NewPubSubControllerFunc is the released signature of [NewPubSubController].
type NewPubSubControllerFunc = func(session.Service, agent.Loader, memory.Service, artifact.Service, runner.PluginConfig, TriggerConfig) *PubSubController

// NewEventarcControllerFunc is the released signature of [NewEventarcController].
type NewEventarcControllerFunc = func(session.Service, agent.Loader, memory.Service, artifact.Service, runner.PluginConfig, TriggerConfig) *EventarcController

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
			runner: mustPubSub(t, nil, tc, WithEventsCompactionConfig(cfg)).runner,
		},
		{
			name:   "eventarc",
			runner: mustEventarc(t, nil, tc, WithEventsCompactionConfig(cfg)).runner,
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

// TestControllerOptionsToleratesNil checks that a nil option is skipped rather
// than dereferenced.
//
// Options are commonly assembled by a helper that returns nil when it has
// nothing to apply, and a variadic parameter makes passing one easy. Panicking
// during construction is a poor way to report that.
func TestControllerOptionsToleratesNil(t *testing.T) {
	t.Parallel()

	if got, _ := NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil); got == nil {
		t.Error("NewPubSubController() with a nil option returned nil")
	}
	if got, _ := NewEventarcControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil); got == nil {
		t.Error("NewEventarcController() with a nil option returned nil")
	}
	// A nil option alongside a real one must not stop the real one applying.
	cfg := &compaction.Config{CompactionInterval: 2}
	c, _ := NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil, WithEventsCompactionConfig(cfg))
	if c.runner.eventsCompactionConfig != cfg {
		t.Error("a nil option prevented a later option from applying")
	}
}

// TestWithEventsCompactionConfigWarnsWhenItCannotFire checks that a
// sliding-window-only config on a trigger controller says so.
//
// Each delivery runs in a session of its own, so history never accumulates and
// the sliding window, which counts completed invocations within one session,
// can never reach its interval. Silently doing nothing is the bad outcome here:
// the operator has configured compaction and will believe it is working.
func TestWithEventsCompactionConfigWarnsAboutSlidingWindows(t *testing.T) {
	tc := TriggerConfig{MaxConcurrentRuns: 1}

	tests := []struct {
		name string
		cfg  *compaction.Config
		want string // a phrase the warning must contain, or "" for silence
	}{
		{
			name: "interval above one cannot reach its interval in one attempt",
			cfg:  &compaction.Config{CompactionInterval: 2},
			want: "will not reach its interval",
		},
		{
			// It does fire here, on every delivery, which the old warning
			// denied. The waste is the summary, not the silence: the session it
			// is written into is discarded when the delivery ends.
			name: "interval of one fires and is wasted",
			cfg:  &compaction.Config{CompactionInterval: 1},
			want: "discarded when the delivery ends",
		},
		{
			name: "tail retention works here and must not warn",
			cfg:  &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2},
		},
		{
			// The sliding window is just as inert alongside tail retention, so
			// enabling both must not buy silence about the half that cannot run.
			name: "a sliding window still warns when tail retention is also set",
			cfg:  &compaction.Config{CompactionInterval: 2, TokenThreshold: 1000, EventRetentionSize: 2},
			want: "will not reach its interval",
		},
		{
			name: "no config at all",
			cfg:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			log.SetOutput(&buf)
			t.Cleanup(func() { log.SetOutput(os.Stderr) })

			// The warning is emitted by the option, before any validation, so
			// the controller itself does not matter here.
			_, _ = NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, tc,
				WithEventsCompactionConfig(tt.cfg))

			got := buf.String()
			if tt.want == "" {
				if strings.Contains(got, "adk: sliding-window compaction") {
					t.Errorf("warned about a configuration that works here; log was %q", got)
				}
				return
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("warning does not mention %q; log was %q", tt.want, got)
			}
		})
	}
}

// mustPubSub builds a PubSub controller and fails the test if it is refused.
func mustPubSub(t *testing.T, loader agent.Loader, tc TriggerConfig, opts ...ControllerOption) *PubSubController {
	t.Helper()
	c, err := NewPubSubControllerWithOptions(nil, loader, nil, nil, runner.PluginConfig{}, tc, opts...)
	if err != nil {
		t.Fatalf("NewPubSubControllerWithOptions() error = %v", err)
	}
	return c
}

// mustEventarc is mustPubSub for the Eventarc controller.
func mustEventarc(t *testing.T, loader agent.Loader, tc TriggerConfig, opts ...ControllerOption) *EventarcController {
	t.Helper()
	c, err := NewEventarcControllerWithOptions(nil, loader, nil, nil, runner.PluginConfig{}, tc, opts...)
	if err != nil {
		t.Fatalf("NewEventarcControllerWithOptions() error = %v", err)
	}
	return c
}

// TestControllerRefusesACompactionConfigItCannotServe pins that an unusable
// compaction config is rejected at construction.
//
// A trigger controller returned only a controller, so it had no way to refuse
// one: an empty &compaction.Config{} constructed fine and then failed every
// delivery with a 500. On Pub/Sub push a 500 is a NACK, so the message comes
// back, fails again, and the subscription spins.
func TestControllerRefusesACompactionConfigItCannotServe(t *testing.T) {
	tc := TriggerConfig{MaxConcurrentRuns: 1}

	tests := []struct {
		name   string
		cfg    *compaction.Config
		loader agent.Loader
		wantOK bool
	}{
		{
			// Enables no strategy at all.
			name: "a config that enables nothing",
			cfg:  &compaction.Config{},
		},
		{
			name:   "a config with no summarizer over an agent with no model",
			cfg:    &compaction.Config{CompactionInterval: 2},
			loader: agent.NewSingleLoader(mustWorkflowAgent(t)),
		},
		{
			name:   "a usable config",
			cfg:    &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2, Summarizer: stubSummarizer{}},
			loader: agent.NewSingleLoader(mustWorkflowAgent(t)),
			wantOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPubSubControllerWithOptions(session.InMemoryService(), tt.loader, nil, nil,
				runner.PluginConfig{}, tc, WithEventsCompactionConfig(tt.cfg))
			if gotOK := err == nil; gotOK != tt.wantOK {
				t.Errorf("NewPubSubControllerWithOptions() error = %v, want an error: %t", err, !tt.wantOK)
			}
		})
	}
}

// mustWorkflowAgent returns an agent with no model of its own.
func mustWorkflowAgent(t *testing.T) agent.Agent {
	t.Helper()
	a, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "wf_app"}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}
	return a
}

// stubSummarizer stands in for a configured summarizer.
type stubSummarizer struct{}

func (stubSummarizer) SummarizeEvents(context.Context, []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	return nil, nil, nil
}
