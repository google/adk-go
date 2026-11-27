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

package console

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel/sdk/resource"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/telemetry"
)

// closingPlugin returns a plugin whose CloseFunc records that it ran, then runs
// closeFunc when it is not nil.
func closingPlugin(t *testing.T, closeFunc func() error) (*plugin.Plugin, *atomic.Bool) {
	t.Helper()
	var closed atomic.Bool
	p, err := plugin.New(plugin.Config{
		Name: "flushing",
		CloseFunc: func() error {
			closed.Store(true)
			if closeFunc != nil {
				return closeFunc()
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p, &closed
}

func captureLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	orig := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(orig) })
	return &buf
}

// runConsoleCtx runs the console launcher with args, ctx and input on stdin,
// and returns Run's error. It fails the test if Run does not return in time.
func runConsoleCtx(t *testing.T, ctx context.Context, config *launcher.Config, input string, args ...string) error {
	t.Helper()

	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	oldIn, oldOut := os.Stdin, os.Stdout
	os.Stdin, os.Stdout = inR, outW
	t.Cleanup(func() { os.Stdin, os.Stdout = oldIn, oldOut })
	go func() { _, _ = io.Copy(io.Discard, outR) }()

	if _, err := inW.WriteString(input); err != nil {
		t.Fatalf("writing stdin: %v", err)
	}
	if err := inW.Close(); err != nil {
		t.Fatalf("closing stdin: %v", err)
	}

	l := NewLauncher()
	if _, err := l.Parse(args); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	result := make(chan error, 1)
	go func() { result <- l.Run(ctx, config) }()
	select {
	case err := <-result:
		_ = outW.Close()
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("Run() did not return within 10s")
		return nil
	}
}

// TestConsoleClosesPluginsOnExit pins that Run closes the configured plugins
// when the interaction loop ends, and logs a close failure.
func TestConsoleClosesPluginsOnExit(t *testing.T) {
	buf := captureLog(t)
	p, closed := closingPlugin(t, func() error { return errors.New("simulated flush error") })
	config := testConfig(t, nil)
	config.PluginConfig = runner.PluginConfig{Plugins: []*plugin.Plugin{p}}

	if err := runConsoleCtx(t, t.Context(), config, "hello\n"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if !closed.Load() {
		t.Error("Run() returned without closing the configured plugin")
	}
	if got := buf.String(); !strings.Contains(got, "plugin shutdown failed") || !strings.Contains(got, "simulated flush error") {
		t.Errorf("log output = %q, want the plugin close failure logged", got)
	}
}

// TestConsoleClosesPluginsAfterInterrupt pins that plugins are closed when Run
// ends because ctx was canceled, as it is after an interrupt.
func TestConsoleClosesPluginsAfterInterrupt(t *testing.T) {
	buf := captureLog(t)
	p, closed := closingPlugin(t, nil)
	config := testConfig(t, nil)
	config.PluginConfig = runner.PluginConfig{Plugins: []*plugin.Plugin{p}}

	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	oldIn, oldOut := os.Stdin, os.Stdout
	os.Stdin, os.Stdout = inR, outW
	// Closing stdin last lets the console's reader goroutine see EOF and exit.
	t.Cleanup(func() { _ = inW.Close() })
	t.Cleanup(func() { os.Stdin, os.Stdout = oldIn, oldOut })

	var mu sync.Mutex
	var out strings.Builder
	go func() {
		chunk := make([]byte, 4096)
		for {
			n, err := outR.Read(chunk)
			mu.Lock()
			out.Write(chunk[:n])
			mu.Unlock()
			if err != nil {
				return
			}
		}
	}()

	l := NewLauncher()
	if _, err := l.Parse(nil); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- l.Run(ctx, config) }()

	// Wait for one full turn, so the reader goroutine has started and the loop
	// is idle when the interrupt arrives.
	if _, err := inW.WriteString("hello\n"); err != nil {
		t.Fatalf("writing stdin: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		prompts := strings.Count(out.String(), "User -> ")
		mu.Unlock()
		if prompts >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("console did not finish the first turn")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()

	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Run() did not return after ctx was canceled")
	}
	_ = outW.Close()
	if !closed.Load() {
		t.Errorf("Run() returned without closing the configured plugin, log: %q", buf.String())
	}
}

// TestConsoleClosesPluginsWhenTelemetryInitFails pins that plugins are closed
// even when Run fails during setup.
func TestConsoleClosesPluginsWhenTelemetryInitFails(t *testing.T) {
	p, closed := closingPlugin(t, nil)
	config := testConfig(t, nil)
	config.PluginConfig = runner.PluginConfig{Plugins: []*plugin.Plugin{p}}
	// A resource whose schema URL conflicts with resource.Default()'s makes
	// telemetry init fail without touching the network.
	config.TelemetryOptions = []telemetry.Option{
		telemetry.WithResource(resource.NewWithAttributes("https://conflicting.invalid/schema/v1")),
	}

	if err := runConsoleCtx(t, t.Context(), config, ""); err == nil {
		t.Fatal("Run() succeeded, want telemetry initialization failure")
	}

	if !closed.Load() {
		t.Error("Run() returned without closing the configured plugin")
	}
}

// TestConsoleBoundsPluginCloseByShutdownTimeout pins that a plugin that never
// finishes closing cannot hold Run open past --shutdown-timeout.
func TestConsoleBoundsPluginCloseByShutdownTimeout(t *testing.T) {
	buf := captureLog(t)
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	p, _ := closingPlugin(t, func() error {
		<-release
		return nil
	})
	config := testConfig(t, nil)
	config.PluginConfig = runner.PluginConfig{Plugins: []*plugin.Plugin{p}}

	if err := runConsoleCtx(t, t.Context(), config, "", "--shutdown-timeout", "100ms"); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got := buf.String(); !strings.Contains(got, "did not finish closing") {
		t.Errorf("log output = %q, want the unfinished plugin close logged", got)
	}
}
