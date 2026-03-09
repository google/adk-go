// Copyright 2025 Google LLC
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

package controllers

import (
	"context"
	"testing"
	"time"

	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func TestNewRuntimeAPIController_PluginsAssignment(t *testing.T) {
	p1, err := plugin.New(plugin.Config{Name: "plugin1"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin1: %v", err)
	}

	p2, err := plugin.New(plugin.Config{Name: "plugin2"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin2: %v", err)
	}

	tc := []struct {
		name        string
		plugins     []*plugin.Plugin
		wantPlugins int
	}{
		{
			name:        "with no plugins",
			plugins:     nil,
			wantPlugins: 0,
		},
		{
			name:        "with empty plugin list",
			plugins:     []*plugin.Plugin{},
			wantPlugins: 0,
		},
		{
			name:        "with single plugin",
			plugins:     []*plugin.Plugin{p1},
			wantPlugins: 1,
		},
		{
			name:        "with multiple plugins",
			plugins:     []*plugin.Plugin{p1, p2},
			wantPlugins: 2,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			controller := NewRuntimeAPIController(nil, nil, nil, nil, 10*time.Second, runner.PluginConfig{
				Plugins: tt.plugins,
			})

			if controller == nil {
				t.Fatal("NewRuntimeAPIController returned nil")
			}

			if got := len(controller.pluginConfig.Plugins); got != tt.wantPlugins {
				t.Errorf("NewRuntimeAPIController() plugins count = %v, want %v", got, tt.wantPlugins)
			}
		})
	}
}

// recordingSessionService wraps a session.Service and records each AppendEvent call.
type recordingSessionService struct {
	session.Service
	appendEventCalls []*session.Event
}

func (r *recordingSessionService) AppendEvent(ctx context.Context, s session.Session, ev *session.Event) error {
	r.appendEventCalls = append(r.appendEventCalls, ev)
	return r.Service.AppendEvent(ctx, s, ev)
}

func TestRuntimeAPIController_applyStateDeltaIfPresent(t *testing.T) {
	ctx := context.Background()
	base := session.InMemoryService()
	rec := &recordingSessionService{Service: base}

	createResp, err := base.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sess",
	})
	if err != nil {
		t.Fatalf("Create session: %v", err)
	}
	ssn := createResp.Session

	c := NewRuntimeAPIController(rec, nil, nil, nil, 10*time.Second, runner.PluginConfig{})

	t.Run("nil stateDelta does not call AppendEvent", func(t *testing.T) {
		rec.appendEventCalls = nil
		if err := c.applyStateDeltaIfPresent(ctx, ssn, nil); err != nil {
			t.Errorf("applyStateDeltaIfPresent(nil): %v", err)
		}
		if n := len(rec.appendEventCalls); n != 0 {
			t.Errorf("AppendEvent called %d times, want 0", n)
		}
		// Validate from session perspective: no new events.
		getResp, err := rec.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
		if err != nil {
			t.Fatalf("Get session: %v", err)
		}
		if n := getResp.Session.Events().Len(); n != 0 {
			t.Errorf("session Events().Len() = %d, want 0 after nil stateDelta", n)
		}
	})

	t.Run("empty stateDelta does not call AppendEvent", func(t *testing.T) {
		rec.appendEventCalls = nil
		empty := map[string]any{}
		if err := c.applyStateDeltaIfPresent(ctx, ssn, &empty); err != nil {
			t.Errorf("applyStateDeltaIfPresent(empty): %v", err)
		}
		if n := len(rec.appendEventCalls); n != 0 {
			t.Errorf("AppendEvent called %d times, want 0", n)
		}
		// Validate from session perspective: no new events.
		getResp, err := rec.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
		if err != nil {
			t.Fatalf("Get session: %v", err)
		}
		if n := getResp.Session.Events().Len(); n != 0 {
			t.Errorf("session Events().Len() = %d, want 0 after empty stateDelta", n)
		}
	})

	t.Run("non-empty stateDelta appends event with Author system and shallow-copied delta", func(t *testing.T) {
		rec.appendEventCalls = nil
		delta := map[string]any{"user:key": "value", "user:other": 42}
		if err := c.applyStateDeltaIfPresent(ctx, ssn, &delta); err != nil {
			t.Errorf("applyStateDeltaIfPresent: %v", err)
		}
		if n := len(rec.appendEventCalls); n != 1 {
			t.Fatalf("AppendEvent called %d times, want 1", n)
		}
		ev := rec.appendEventCalls[0]
		if ev.Author != "system" {
			t.Errorf("event.Author = %q, want %q", ev.Author, "system")
		}
		if ev.Actions.StateDelta == nil {
			t.Fatal("event.Actions.StateDelta is nil")
		}
		if got := ev.Actions.StateDelta["user:key"]; got != "value" {
			t.Errorf("StateDelta[user:key] = %v, want value", got)
		}
		if got := ev.Actions.StateDelta["user:other"]; got != 42 {
			t.Errorf("StateDelta[user:other] = %v, want 42", got)
		}
		// Shallow copy: mutating the request map after the call must not change the stored event.
		delta["user:key"] = "mutated"
		if got := ev.Actions.StateDelta["user:key"]; got != "value" {
			t.Errorf("after mutating request map, StateDelta[user:key] = %v, want value (stored event must be a copy)", got)
		}
		// Validate from session perspective: state was merged.
		getResp, err := rec.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
		if err != nil {
			t.Fatalf("Get session: %v", err)
		}
		state := getResp.Session.State()
		if got, err := state.Get("user:key"); err != nil || got != "value" {
			t.Errorf("session state user:key = %v, err = %v; want value", got, err)
		}
		if got, err := state.Get("user:other"); err != nil || got != 42 {
			t.Errorf("session state user:other = %v, err = %v; want 42", got, err)
		}
	})
}
