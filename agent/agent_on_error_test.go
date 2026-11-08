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

package agent_test

import (
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestAgentRunOnAgentErrorCallback(t *testing.T) {
	runErr := errors.New("run failed")
	beforeErr := errors.New("before failed")

	tests := []struct {
		name                string
		before              []agent.BeforeAgentCallback
		run                 func(agent.InvocationContext) iter.Seq2[*session.Event, error]
		wantCalls           int
		wantRunErr          bool
		wantCallbackErr     error
		wantCallbackErrText string
	}{
		{
			name: "error from run is reported",
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, runErr)
				}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: runErr,
		},
		{
			name: "error from before agent callback is reported",
			before: []agent.BeforeAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return nil, beforeErr
				},
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: beforeErr,
		},
		{
			name: "successful run is not reported",
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls: 0,
		},
		{
			name:                "missing run function is reported",
			wantCalls:           1,
			wantRunErr:          true,
			wantCallbackErrText: `agent "failing" has no Run function`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls int
			var callbackErr error
			var callbackAgent string
			p, err := plugin.New(plugin.Config{
				Name: "observer",
				OnAgentErrorCallback: func(_ agent.Context, a agent.Agent, err error) {
					calls++
					callbackErr = err
					callbackAgent = a.Name()
				},
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			pm, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
				Plugins: []*plugin.Plugin{p},
			})
			if err != nil {
				t.Fatalf("NewPluginManager() error = %v", err)
			}
			a, err := agent.New(agent.Config{
				Name:                 "failing",
				BeforeAgentCallbacks: test.before,
				Run:                  test.run,
			})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}

			ctx := plugininternal.ToContext(t.Context(), pm)
			ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Agent:        a,
				InvocationID: "inv-1",
			})

			var runErr error
			for _, err := range a.Run(ic) {
				if err != nil {
					runErr = err
				}
			}
			if got := runErr != nil; got != test.wantRunErr {
				t.Fatalf("Run() error = %v, want error = %v", runErr, test.wantRunErr)
			}
			if calls != test.wantCalls {
				t.Fatalf("OnAgentErrorCallback calls = %d, want %d", calls, test.wantCalls)
			}
			if calls == 0 {
				return
			}
			if test.wantCallbackErr != nil && !errors.Is(callbackErr, test.wantCallbackErr) {
				t.Errorf("callback error = %v, want %v", callbackErr, test.wantCallbackErr)
			}
			if test.wantCallbackErrText != "" && (callbackErr == nil || callbackErr.Error() != test.wantCallbackErrText) {
				t.Errorf("callback error = %v, want text %q", callbackErr, test.wantCallbackErrText)
			}
			if callbackAgent != "failing" {
				t.Errorf("callback ctx.Agent().Name() = %q, want %q", callbackAgent, "failing")
			}
		})
	}
}

func TestAgentRunOnAgentErrorCallbackNotifiesEveryPlugin(t *testing.T) {
	var notified []string
	newPlugin := func(name string) *plugin.Plugin {
		p, err := plugin.New(plugin.Config{
			Name: name,
			OnAgentErrorCallback: func(agent.Context, agent.Agent, error) {
				notified = append(notified, name)
			},
		})
		if err != nil {
			t.Fatalf("plugin.New(%q) error = %v", name, err)
		}
		return p
	}
	pm, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins: []*plugin.Plugin{newPlugin("first"), newPlugin("second")},
	})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	a, err := agent.New(agent.Config{
		Name: "failing",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("run failed"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	ctx := plugininternal.ToContext(t.Context(), pm)
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: a})
	for range a.Run(ic) {
	}

	if len(notified) != 2 || notified[0] != "first" || notified[1] != "second" {
		t.Errorf("notified plugins = %v, want [first second]", notified)
	}
}
