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

package plugininternal

import (
	"errors"
	"slices"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/plugin"
)

// tracedInvocation returns an invocation context whose span records, and a
// function that ends that span and returns the messages of its exception
// events.
func tracedInvocation(t *testing.T) (agent.InvocationContext, func() []string) {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	ctx, span := tp.Tracer("test").Start(t.Context(), "invocation")
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{})
	return ictx, func() []string {
		span.End()
		var messages []string
		for _, s := range exporter.GetSpans() {
			for _, event := range s.Events {
				if event.Name != "exception" {
					continue
				}
				for _, attr := range event.Attributes {
					if attr.Key == "exception.message" {
						messages = append(messages, attr.Value.AsString())
					}
				}
			}
		}
		return messages
	}
}

func newTestManager(t *testing.T, cfgs ...plugin.Config) *PluginManager {
	t.Helper()
	plugins := make([]*plugin.Plugin, 0, len(cfgs))
	for _, cfg := range cfgs {
		p, err := plugin.New(cfg)
		if err != nil {
			t.Fatalf("plugin.New(%q) error = %v", cfg.Name, err)
		}
		plugins = append(plugins, p)
	}
	pm, err := NewPluginManager(PluginConfig{Plugins: plugins})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	return pm
}

func TestPluginManager_RunOnRunErrorCallback_RegistrationOrderSkipsPluginsWithoutHook(t *testing.T) {
	errBoom := errors.New("boom")
	var called []string
	var received []error
	record := func(name string) plugin.OnRunErrorCallback {
		return func(_ agent.InvocationContext, err error) {
			called = append(called, name)
			received = append(received, err)
		}
	}
	pm := newTestManager(t,
		plugin.Config{Name: "p1", OnRunErrorCallback: record("p1")},
		plugin.Config{Name: "p2"},
		plugin.Config{Name: "p3", OnRunErrorCallback: record("p3")},
	)
	ictx, exceptions := tracedInvocation(t)

	pm.RunOnRunErrorCallback(ictx, errBoom)

	if want := []string{"p1", "p3"}; !slices.Equal(called, want) {
		t.Errorf("plugins called = %v, want %v", called, want)
	}
	for i, err := range received {
		if err != errBoom {
			t.Errorf("plugin %d received error %v, want %v", i, err, errBoom)
		}
	}
	if got := exceptions(); len(got) != 0 {
		t.Errorf("span exception events = %q, want none", got)
	}
}

func TestPluginManager_RunOnRunErrorCallback_PanicRecordedOnSpanLaterPluginsStillRun(t *testing.T) {
	var laterCalls int
	pm := newTestManager(t,
		plugin.Config{Name: "p1", OnRunErrorCallback: func(agent.InvocationContext, error) {
			panic("plugin blew up")
		}},
		plugin.Config{Name: "p2", OnRunErrorCallback: func(agent.InvocationContext, error) {
			laterCalls++
		}},
	)
	ictx, exceptions := tracedInvocation(t)

	pm.RunOnRunErrorCallback(ictx, errors.New("run failed"))

	if laterCalls != 1 {
		t.Errorf("p2 calls = %d, want 1", laterCalls)
	}
	want := []string{`plugin "p1" OnRunErrorCallback panicked: plugin blew up`}
	if got := exceptions(); !slices.Equal(got, want) {
		t.Errorf("span exception events = %q, want %q", got, want)
	}
}

// The name dates from the time the manager returned the callback's error as it
// was. The test now requires the error to name the plugin and the callback and
// to wrap the callback's error.
func TestPluginManager_RunAfterRunCallback_StopsAtFirstErrorAndReturnsItUnwrapped(t *testing.T) {
	errB := errors.New("p2 cleanup failed")
	var log []string
	plain := func(name string) plugin.AfterRunCallback {
		return func(agent.InvocationContext) { log = append(log, name) }
	}
	failable := func(name string, err error) plugin.FailableAfterRunCallback {
		return func(agent.InvocationContext) error {
			log = append(log, name)
			return err
		}
	}

	tests := []struct {
		name    string
		cfgs    []plugin.Config
		wantErr error
		wantMsg string
		wantLog []string
	}{
		{
			name: "failable callback fails",
			cfgs: []plugin.Config{
				{Name: "p1", AfterRunCallback: plain("p1")},
				{Name: "p2", FailableAfterRunCallback: failable("p2", errB)},
				{Name: "p3", AfterRunCallback: plain("p3")},
			},
			wantErr: errB,
			wantMsg: "Error in plugin 'p2' during 'after_run_callback' callback: p2 cleanup failed",
			wantLog: []string{"p1", "p2"},
		},
		{
			name: "every callback succeeds",
			cfgs: []plugin.Config{
				{Name: "p1", AfterRunCallback: plain("p1")},
				{Name: "p2", FailableAfterRunCallback: failable("p2", nil)},
				{Name: "p3", FailableAfterRunCallback: failable("p3", nil)},
			},
			wantLog: []string{"p1", "p2", "p3"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log = nil
			pm := newTestManager(t, tt.cfgs...)

			err := pm.RunAfterRunCallback(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("RunAfterRunCallback() error = %v, want one that wraps %v", err, tt.wantErr)
			}
			if err != nil && err.Error() != tt.wantMsg {
				t.Errorf("RunAfterRunCallback() error text = %q, want %q", err.Error(), tt.wantMsg)
			}
			if !slices.Equal(log, tt.wantLog) {
				t.Errorf("callbacks run = %v, want %v", log, tt.wantLog)
			}
		})
	}
}
