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
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/plugin"
)

// closeRecorder records the order in which plugins are closed.
type closeRecorder struct {
	mu     sync.Mutex
	closed []string
}

func (r *closeRecorder) plugin(t *testing.T, name string, closeFunc func() error) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: name,
		CloseFunc: func() error {
			r.mu.Lock()
			r.closed = append(r.closed, name)
			r.mu.Unlock()
			if closeFunc != nil {
				return closeFunc()
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New(%q) error = %v", name, err)
	}
	return p
}

func (r *closeRecorder) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.closed...)
}

// blockingClose returns a CloseFunc that does not return until the test ends.
func blockingClose(t *testing.T) func() error {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	return func() error {
		<-release
		return nil
	}
}

// closeWithin runs c.Close and fails the test if it does not return in time,
// so a missing bound shows up as a failure rather than a hung test.
func closeWithin(t *testing.T, ctx context.Context, c PluginConfig) error {
	t.Helper()
	result := make(chan error, 1)
	go func() { result <- c.Close(ctx) }()
	select {
	case err := <-result:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("PluginConfig.Close did not return within 5s")
		return nil
	}
}

func TestPluginConfigCloseZeroValue(t *testing.T) {
	if err := (PluginConfig{}).Close(t.Context()); err != nil {
		t.Errorf("PluginConfig{}.Close() error = %v, want nil", err)
	}
}

func TestPluginConfigCloseContinuesPastAFailure(t *testing.T) {
	errMiddle := errors.New("flush failed")
	var rec closeRecorder
	cfg := PluginConfig{Plugins: []*plugin.Plugin{
		rec.plugin(t, "first", nil),
		nil,
		rec.plugin(t, "middle", func() error { return errMiddle }),
		rec.plugin(t, "last", nil),
	}}

	err := closeWithin(t, t.Context(), cfg)

	if !errors.Is(err, errMiddle) {
		t.Errorf("Close() error = %v, want it to wrap %v", err, errMiddle)
	}
	if err != nil && !strings.Contains(err.Error(), `"middle"`) {
		t.Errorf("Close() error = %q, want it to name the failing plugin", err)
	}
	if got, want := strings.Join(rec.names(), ","), "first,middle,last"; got != want {
		t.Errorf("closed plugins = %s, want %s", got, want)
	}
}

func TestPluginConfigCloseTimeoutAppliesPerPlugin(t *testing.T) {
	var rec closeRecorder
	cfg := PluginConfig{
		Plugins: []*plugin.Plugin{
			rec.plugin(t, "slow", blockingClose(t)),
			rec.plugin(t, "after", nil),
		},
		CloseTimeout: 20 * time.Millisecond,
	}

	err := closeWithin(t, t.Context(), cfg)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Close() error = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if err != nil && !strings.Contains(err.Error(), `"slow"`) {
		t.Errorf("Close() error = %q, want it to name the plugin that timed out", err)
	}
	if got, want := strings.Join(rec.names(), ","), "slow,after"; got != want {
		t.Errorf("closed plugins = %s, want %s: a timeout must not stop the rest", got, want)
	}
}

func TestPluginConfigCloseZeroTimeoutLeavesTheBoundToContext(t *testing.T) {
	var rec closeRecorder
	cfg := PluginConfig{Plugins: []*plugin.Plugin{
		rec.plugin(t, "fast", nil),
		rec.plugin(t, "slow", blockingClose(t)),
		rec.plugin(t, "unreached", nil),
		rec.plugin(t, "also-unreached", nil),
	}}
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()

	err := closeWithin(t, ctx, cfg)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Close() error = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if err != nil && strings.Contains(err.Error(), `"fast"`) {
		t.Errorf("Close() error = %q, want no error for the plugin that closed in time", err)
	}
	for _, name := range []string{`"slow"`, `"unreached"`, `"also-unreached"`} {
		if err != nil && !strings.Contains(err.Error(), name) {
			t.Errorf("Close() error = %q, want an entry for plugin %s", err, name)
		}
	}
	if got, want := strings.Join(rec.names(), ","), "fast,slow"; got != want {
		t.Errorf("closed plugins = %s, want %s: no plugin starts after ctx is done", got, want)
	}
}

func TestPluginConfigCloseCanceledContextStartsNothing(t *testing.T) {
	var rec closeRecorder
	cfg := PluginConfig{Plugins: []*plugin.Plugin{rec.plugin(t, "only", nil)}}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	err := closeWithin(t, ctx, cfg)

	if !errors.Is(err, context.Canceled) {
		t.Errorf("Close() error = %v, want it to wrap context.Canceled", err)
	}
	// A close started in the background would record itself shortly after
	// Close returns, so give it time to show up before checking.
	time.Sleep(100 * time.Millisecond)
	if got := rec.names(); len(got) != 0 {
		t.Errorf("closed plugins = %v, want none after ctx was canceled", got)
	}
}
