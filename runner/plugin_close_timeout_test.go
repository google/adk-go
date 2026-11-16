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

package runner

import (
	"context"
	"errors"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestRunner_Run_WrapsPluginErrorsOfAgentStage(t *testing.T) {
	errBoom := errors.New("boom")
	tests := []struct {
		name string
		cfg  plugin.Config
		want string
	}{
		{
			name: "before_agent_callback",
			cfg: plugin.Config{Name: "p", BeforeAgentCallback: func(agent.Context) (*genai.Content, error) {
				return nil, errBoom
			}},
			want: "Error in plugin 'p' during 'before_agent_callback' callback: boom",
		},
		{
			name: "after_agent_callback",
			cfg: plugin.Config{Name: "p", AfterAgentCallback: func(agent.Context) (*genai.Content, error) {
				return nil, errBoom
			}},
			want: "Error in plugin 'p' during 'after_agent_callback' callback: boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, _ := newLifecycleRunner(t, Config{Agent: scriptedAgent(t, "custom", step{text: "ok"})}, mustPlugin(t, tt.cfg))

			var err error
			for _, e := range r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}) {
				if e != nil {
					err = e
					break
				}
			}

			if err == nil {
				t.Fatalf("Run() yielded no error, want one containing %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Run() error = %q, want it to contain %q", err.Error(), tt.want)
			}
			if !errors.Is(err, errBoom) {
				t.Errorf("errors.Is(Run() error, errBoom) = false, want true; err = %v", err)
			}
		})
	}
}

func TestNew_RefusesANegativePluginCloseTimeout(t *testing.T) {
	tests := []struct {
		name         string
		closeTimeout time.Duration
		wantErr      string // empty when New succeeds
	}{
		{name: "one nanosecond below zero", closeTimeout: -1, wantErr: "failed to create plugin manager: plugin close timeout must not be negative, got -1ns"},
		{name: "minus one second", closeTimeout: -time.Second, wantErr: "failed to create plugin manager: plugin close timeout must not be negative, got -1s"},
		{name: "zero", closeTimeout: 0},
		{name: "one nanosecond", closeTimeout: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(Config{
				AppName:        "close_test",
				SessionService: session.InMemoryService(),
				Agent:          scriptedAgent(t, "custom"),
				PluginConfig:   PluginConfig{CloseTimeout: tt.closeTimeout},
			})

			if tt.wantErr == "" {
				if err != nil {
					t.Errorf("New() error = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("New() error = nil, want %q", tt.wantErr)
			}
			if got := err.Error(); got != tt.wantErr {
				t.Errorf("New() error = %q, want %q", got, tt.wantErr)
			}
		})
	}
}

func TestRunner_Close_AppliesPluginConfigCloseTimeout(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseHung := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseHung)
	var okCloses atomic.Int32
	plugins := []*plugin.Plugin{
		newClosePlugin(t, "hung", func() error { <-release; return nil }),
		newClosePlugin(t, "ok", func() error { okCloses.Add(1); return nil }),
	}
	r := newCloseTestRunner(t, Config{
		Agent:        newCloseTestLLM(t, "root", nil),
		PluginConfig: PluginConfig{Plugins: plugins, CloseTimeout: 30 * time.Millisecond},
	})
	const labelKey = "plugin-close-timeout-test"
	labelValue := t.Name()

	var err error
	pprof.Do(t.Context(), pprof.Labels(labelKey, labelValue), func(context.Context) {
		_, err = closeWithin(t, r, 2*time.Second)
	})

	if err == nil {
		t.Fatal("Close() error = nil, want the timeout of hung")
	}
	want := "Failed to close plugins: 'hung'\nerror closing plugin 'hung': no result after 30ms: context deadline exceeded"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(Close(), context.DeadlineExceeded) = false, want true")
	}
	if got := okCloses.Load(); got != 1 {
		t.Errorf("closes of ok = %d, want 1", got)
	}
	if !labeledGoroutineAlive(t, labelKey, labelValue) {
		t.Fatal("no goroutine started by Close is alive after the timeout, want the one running hung's CloseFunc")
	}
	releaseHung()
	deadline := time.Now().Add(2 * time.Second)
	for labeledGoroutineAlive(t, labelKey, labelValue) {
		if time.Now().After(deadline) {
			t.Fatal("a goroutine started by Close is still alive 2s after hung's CloseFunc returned")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
