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
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestAgentRunOnAgentErrorCallback(t *testing.T) {
	runErr := errors.New("run failed")
	beforeErr := errors.New("before failed")
	afterErr := errors.New("after failed")

	tests := []struct {
		name                string
		before              []agent.BeforeAgentCallback
		after               []agent.AfterAgentCallback
		invocationCtx       func(context.Context) context.Context
		run                 func(agent.InvocationContext) iter.Seq2[*session.Event, error]
		wantCalls           int
		wantRunErr          bool
		wantRunEvent        bool
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
			name: "error from after agent callback is reported",
			after: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return nil, afterErr
				},
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: afterErr,
		},
		{
			name: "successful run is not reported",
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls: 0,
		},
		{
			name: "successful before agent callback event is not reported",
			before: []agent.BeforeAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return genai.NewContentFromText("short circuit", genai.RoleModel), nil
				},
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls:    0,
			wantRunEvent: true,
		},
		{
			name: "successful run event is not reported",
			run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(session.NewEvent(ctx, ctx.InvocationID()), nil)
				}
			},
			wantCalls:    0,
			wantRunEvent: true,
		},
		{
			name: "successful after agent callback event is not reported",
			after: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return genai.NewContentFromText("after callback", genai.RoleModel), nil
				},
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			},
			wantCalls:    0,
			wantRunEvent: true,
		},
		{
			name: "caller cancellation is not reported",
			invocationCtx: func(ctx context.Context) context.Context {
				ctx, cancel := context.WithCancel(ctx)
				cancel()
				return ctx
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.Canceled))
				}
			},
			wantCalls:  0,
			wantRunErr: true,
		},
		{
			name: "caller deadline is not reported",
			invocationCtx: func(ctx context.Context) context.Context {
				ctx, cancel := context.WithDeadline(ctx, time.Now().Add(-time.Second))
				t.Cleanup(cancel)
				return ctx
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.DeadlineExceeded))
				}
			},
			wantCalls:  0,
			wantRunErr: true,
		},
		{
			name: "caller cancellation with cause is not reported",
			invocationCtx: func(ctx context.Context) context.Context {
				cause := errors.New("caller gave up")
				ctx, cancel := context.WithCancelCause(ctx)
				cancel(cause)
				return ctx
			},
			run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.Cause(ctx)))
				}
			},
			wantCalls:  0,
			wantRunErr: true,
		},
		{
			name: "caller deadline with cause is not reported",
			invocationCtx: func(ctx context.Context) context.Context {
				cause := errors.New("caller gave up")
				ctx, cancel := context.WithTimeoutCause(ctx, -time.Second, cause)
				t.Cleanup(cancel)
				return ctx
			},
			run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.Cause(ctx)))
				}
			},
			wantCalls:  0,
			wantRunErr: true,
		},
		{
			name: "unrelated error while cancelled with cause is reported",
			invocationCtx: func(ctx context.Context) context.Context {
				cause := errors.New("caller gave up")
				ctx, cancel := context.WithCancelCause(ctx)
				cancel(cause)
				return ctx
			},
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", runErr))
				}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: runErr,
		},
		{
			name: "live invocation deadline is reported",
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.DeadlineExceeded))
				}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: context.DeadlineExceeded,
		},
		{
			name: "live invocation wrapped cancellation is reported",
			run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, fmt.Errorf("wrapped: %w", context.Canceled))
				}
			},
			wantCalls:       1,
			wantRunErr:      true,
			wantCallbackErr: context.Canceled,
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
			var callbackCtx agent.Context
			var callbackErr error
			var callbackAgent string
			p, err := plugin.New(plugin.Config{
				Name: "observer",
				OnAgentErrorCallback: func(ctx agent.Context, a agent.Agent, err error) {
					calls++
					callbackCtx = ctx
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
				AfterAgentCallbacks:  test.after,
				Run:                  test.run,
			})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}

			ctx := plugininternal.ToContext(t.Context(), pm)
			if test.invocationCtx != nil {
				ctx = test.invocationCtx(ctx)
			}
			ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Agent:        a,
				InvocationID: "inv-1",
			})

			var runErr error
			var runEvent *session.Event
			for event, err := range a.Run(ic) {
				if event != nil {
					runEvent = event
				}
				if err != nil {
					runErr = err
				}
			}
			if got := runErr != nil; got != test.wantRunErr {
				t.Fatalf("Run() error = %v, want error = %v", runErr, test.wantRunErr)
			}
			if got := runEvent != nil; got != test.wantRunEvent {
				t.Fatalf("Run() event = %v, want event = %v", runEvent, test.wantRunEvent)
			}
			if calls != test.wantCalls {
				t.Fatalf("OnAgentErrorCallback calls = %d, want %d", calls, test.wantCalls)
			}
			if calls == 0 {
				return
			}
			if callbackCtx == nil {
				t.Error("callback ctx = nil")
			}
			if callbackErr != runErr {
				t.Errorf("callback error = %v, Run() yielded %v; want the same error", callbackErr, runErr)
			}
			if test.wantCallbackErr != nil && !errors.Is(callbackErr, test.wantCallbackErr) {
				t.Errorf("callback error = %v, want %v", callbackErr, test.wantCallbackErr)
			}
			if test.wantCallbackErrText != "" && (callbackErr == nil || callbackErr.Error() != test.wantCallbackErrText) {
				t.Errorf("callback error = %v, want text %q", callbackErr, test.wantCallbackErrText)
			}
			if callbackAgent != "failing" {
				t.Errorf("callback agent name = %q, want %q", callbackAgent, "failing")
			}
		})
	}
}

func TestAgentRunOnAgentErrorCallbackRecoversPanic(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, trace.NewTracerProvider(trace.WithSyncer(exporter)))

	runErr := errors.New("run failed")
	var firstCalled bool
	var notified []string

	newPlugin := func(name string, callback plugin.OnAgentErrorCallback) *plugin.Plugin {
		p, err := plugin.New(plugin.Config{
			Name:                 name,
			OnAgentErrorCallback: callback,
		})
		if err != nil {
			t.Fatalf("plugin.New(%q) error = %v", name, err)
		}
		return p
	}
	pm, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins: []*plugin.Plugin{
			newPlugin("panicking", func(agent.Context, agent.Agent, error) {
				firstCalled = true
				panic("plugin blew up")
			}),
			newPlugin("observer", func(agent.Context, agent.Agent, error) {
				notified = append(notified, "observer")
			}),
		},
	})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	a, err := agent.New(agent.Config{
		Name: "failing",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, runErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	ctx := plugininternal.ToContext(t.Context(), pm)
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: a})
	var gotErr error
	for _, err := range a.Run(ic) {
		if err != nil {
			gotErr = err
		}
	}

	if !firstCalled {
		t.Error("panicking callback was not called")
	}
	if len(notified) != 1 || notified[0] != "observer" {
		t.Errorf("notified plugins = %v, want [observer]", notified)
	}
	if gotErr != runErr {
		t.Errorf("Run() error = %v, want original error %v", gotErr, runErr)
	}
	recorded := false
	for _, span := range exporter.GetSpans() {
		for _, event := range span.Events {
			if event.Name != "exception" {
				continue
			}
			for _, attr := range event.Attributes {
				if attr.Key == "exception.message" && strings.Contains(attr.Value.AsString(), "OnAgentErrorCallback panicked") {
					recorded = true
				}
			}
		}
	}
	if !recorded {
		t.Error("panic was not recorded on the span")
	}
}

func TestAgentRunOnAgentErrorCallbackRunsBeforeYield(t *testing.T) {
	runErr := errors.New("run failed")
	var order []string
	p, err := plugin.New(plugin.Config{
		Name: "observer",
		OnAgentErrorCallback: func(agent.Context, agent.Agent, error) {
			order = append(order, "callback")
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
		Name: "failing",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, runErr)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	ctx := plugininternal.ToContext(t.Context(), pm)
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: a})
	for _, err := range a.Run(ic) {
		if err != nil {
			order = append(order, "yield")
		}
	}

	if len(order) != 2 || order[0] != "callback" || order[1] != "yield" {
		t.Errorf("order = %v, want [callback yield]", order)
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
	noCallback, err := plugin.New(plugin.Config{Name: "no-callback"})
	if err != nil {
		t.Fatalf("plugin.New(%q) error = %v", "no-callback", err)
	}
	pm, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins: []*plugin.Plugin{newPlugin("first"), noCallback, newPlugin("second")},
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
	var gotErr error
	for _, err := range a.Run(ic) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Error("Run() error = nil, want error")
	}

	if len(notified) != 2 || notified[0] != "first" || notified[1] != "second" {
		t.Errorf("notified plugins = %v, want [first second]", notified)
	}
}
