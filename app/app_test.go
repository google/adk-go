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

package app

import (
	"strings"
	"testing"

	"google.golang.org/adk/agent"
)

func newTestAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Description: "test"})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

func TestNew_OK(t *testing.T) {
	a, err := New(App{Name: "myapp", RootAgent: newTestAgent(t, "root")})
	if err != nil {
		t.Fatalf("New err: %v", err)
	}
	if a.Name != "myapp" {
		t.Errorf("Name = %q, want myapp", a.Name)
	}
}

func TestNew_RejectsInvalidNames(t *testing.T) {
	cases := []string{"", "user", "1leading-digit", "has space", "has-dash", "has.dot"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := New(App{Name: name, RootAgent: newTestAgent(t, "root")})
			if err == nil {
				t.Errorf("expected error for name %q", name)
			}
		})
	}
}

func TestNew_AcceptsValidNames(t *testing.T) {
	cases := []string{"app", "_app", "app_2", "App", "_", "my_app1"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := New(App{Name: name, RootAgent: newTestAgent(t, "root")})
			if err != nil {
				t.Errorf("unexpected error for name %q: %v", name, err)
			}
		})
	}
}

func TestNew_RequiresRootAgent(t *testing.T) {
	_, err := New(App{Name: "myapp"})
	if err == nil {
		t.Fatal("expected error when RootAgent is nil")
	}
	if !strings.Contains(err.Error(), "RootAgent") {
		t.Errorf("err = %v, want mention of RootAgent", err)
	}
}

func TestEventsCompactionConfigValidation(t *testing.T) {
	thr := 1024
	ret := 4
	bad := -1
	zero := 0
	cases := []struct {
		name    string
		cfg     *EventsCompactionConfig
		wantErr bool
	}{
		{"interval-zero", &EventsCompactionConfig{CompactionInterval: 0, OverlapSize: 0}, true},
		{"overlap-negative", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: -1}, true},
		{"token-only", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, TokenThreshold: &thr}, true},
		{"retention-only", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, EventRetentionSize: &ret}, true},
		{"both-set-ok", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, TokenThreshold: &thr, EventRetentionSize: &ret}, false},
		{"token-bad-value", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, TokenThreshold: &bad, EventRetentionSize: &ret}, true},
		{"retention-bad-value", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, TokenThreshold: &thr, EventRetentionSize: &bad}, true},
		{"retention-zero-ok", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 0, TokenThreshold: &thr, EventRetentionSize: &zero}, false},
		{"defaults-ok", &EventsCompactionConfig{CompactionInterval: 5, OverlapSize: 2}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(App{Name: "a", RootAgent: newTestAgent(t, "root"), EventsCompactionConfig: tc.cfg})
			if tc.wantErr && err == nil {
				t.Error("expected error")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
