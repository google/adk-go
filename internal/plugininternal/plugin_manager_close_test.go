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
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/plugin"
)

// newTimeoutManager is newTestManager with a close timeout.
func newTimeoutManager(t *testing.T, closeTimeout time.Duration, cfgs ...plugin.Config) *PluginManager {
	t.Helper()
	plugins := make([]*plugin.Plugin, 0, len(cfgs))
	for _, cfg := range cfgs {
		p, err := plugin.New(cfg)
		if err != nil {
			t.Fatalf("plugin.New(%q) error = %v", cfg.Name, err)
		}
		plugins = append(plugins, p)
	}
	pm, err := NewPluginManager(PluginConfig{Plugins: plugins, CloseTimeout: closeTimeout})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	return pm
}

// hangingClose returns a CloseFunc that blocks until the test ends.
func hangingClose(t *testing.T) func() error {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	return func() error {
		<-release
		return nil
	}
}

// closeManagerWithin calls pm.Close in a goroutine and fails the test when it
// has not returned after limit, so a missing timeout fails instead of hanging.
func closeManagerWithin(t *testing.T, pm *PluginManager, limit time.Duration) (time.Duration, error) {
	t.Helper()
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- pm.Close() }()
	select {
	case err := <-done:
		return time.Since(start), err
	case <-time.After(limit):
		t.Fatalf("Close() has not returned after %v", limit)
		return 0, nil
	}
}

func TestPluginManager_Close_ClosesEachPluginOnceInOrderOneAtATime(t *testing.T) {
	var mu sync.Mutex
	var log []string
	add := func(entry string) {
		mu.Lock()
		defer mu.Unlock()
		log = append(log, entry)
	}
	logged := func(name string) func() error {
		return func() error {
			add(name + ":start")
			time.Sleep(10 * time.Millisecond)
			add(name + ":end")
			return nil
		}
	}
	pm := newTestManager(t,
		plugin.Config{Name: "p1", CloseFunc: logged("p1")},
		plugin.Config{Name: "nilclose"},
		plugin.Config{Name: "p2", CloseFunc: logged("p2")},
		plugin.Config{Name: "p3", CloseFunc: logged("p3")},
	)

	err := pm.Close()
	if err != nil {
		t.Errorf("Close() error = %v, want nil", err)
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"p1:start", "p1:end", "p2:start", "p2:end", "p3:start", "p3:end"}
	if !slices.Equal(log, want) {
		t.Errorf("close log = %v, want %v", log, want)
	}
}

func TestPluginManager_Close_ReportsEveryFailureAndStillClosesTheRest(t *testing.T) {
	errP2 := errors.New("p2 failed")
	errP4 := errors.New("p4 failed")
	var closes [4]atomic.Int32
	closeFunc := func(i int, err error) func() error {
		return func() error {
			closes[i].Add(1)
			return err
		}
	}
	pm := newTestManager(t,
		plugin.Config{Name: "p1", CloseFunc: closeFunc(0, nil)},
		plugin.Config{Name: "p2", CloseFunc: closeFunc(1, errP2)},
		plugin.Config{Name: "p3", CloseFunc: closeFunc(2, nil)},
		plugin.Config{Name: "p4", CloseFunc: closeFunc(3, errP4)},
	)

	err := pm.Close()

	if err == nil {
		t.Fatal("Close() error = nil, want the failures of p2 and p4")
	}
	want := "Failed to close plugins: 'p2', 'p4'\nerror closing plugin 'p2': p2 failed\nerror closing plugin 'p4': p4 failed"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
	if !errors.Is(err, errP2) {
		t.Errorf("errors.Is(Close(), errP2) = false, want true")
	}
	if !errors.Is(err, errP4) {
		t.Errorf("errors.Is(Close(), errP4) = false, want true")
	}
	for i := range closes {
		if got := closes[i].Load(); got != 1 {
			t.Errorf("closes of p%d = %d, want 1", i+1, got)
		}
	}
}

func TestPluginManager_Close_RecoversAPanicAndReportsOnlyItsType(t *testing.T) {
	var p2Closes atomic.Int32
	pm := newTestManager(t,
		plugin.Config{Name: "p1", CloseFunc: func() error { panic("secret user data") }},
		plugin.Config{Name: "p2", CloseFunc: func() error { p2Closes.Add(1); return nil }},
	)

	err := pm.Close()

	if err == nil {
		t.Fatal("Close() error = nil, want the panic of p1")
	}
	want := "Failed to close plugins: 'p1'\nerror closing plugin 'p1': panic of type string"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
	if strings.Contains(err.Error(), "secret") {
		t.Errorf("Close() error %q contains the panic value", err.Error())
	}
	if got := p2Closes.Load(); got != 1 {
		t.Errorf("closes of p2 = %d, want 1", got)
	}
}

func TestPluginManager_Close_AbandonsEachPluginAfterCloseTimeout(t *testing.T) {
	var okCloses atomic.Int32
	pm := newTimeoutManager(t, 50*time.Millisecond,
		plugin.Config{Name: "h1", CloseFunc: hangingClose(t)},
		plugin.Config{Name: "h2", CloseFunc: hangingClose(t)},
		plugin.Config{Name: "ok", CloseFunc: func() error { okCloses.Add(1); return nil }},
	)

	elapsed, err := closeManagerWithin(t, pm, 2*time.Second)

	if err == nil {
		t.Fatal("Close() error = nil, want the timeouts of h1 and h2")
	}
	want := "Failed to close plugins: 'h1', 'h2'\n" +
		"error closing plugin 'h1': no result after 50ms: context deadline exceeded\n" +
		"error closing plugin 'h2': no result after 50ms: context deadline exceeded"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(Close(), context.DeadlineExceeded) = false, want true")
	}
	if elapsed < 100*time.Millisecond {
		t.Errorf("Close() took %v, want at least 100ms (50ms for each hung plugin)", elapsed)
	}
	if got := okCloses.Load(); got != 1 {
		t.Errorf("closes of ok = %d, want 1", got)
	}
}

func TestPluginManager_Close_AppliesANanosecondCloseTimeout(t *testing.T) {
	pm := newTimeoutManager(t, time.Nanosecond, plugin.Config{Name: "hung", CloseFunc: hangingClose(t)})

	_, err := closeManagerWithin(t, pm, 2*time.Second)

	if err == nil {
		t.Fatal("Close() error = nil, want the timeout of hung")
	}
	want := "Failed to close plugins: 'hung'\nerror closing plugin 'hung': no result after 1ns: context deadline exceeded"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
}

func TestPluginManager_Close_ZeroCloseTimeoutMeansFiveSeconds(t *testing.T) {
	t.Parallel()
	pm := newTimeoutManager(t, 0, plugin.Config{Name: "hung", CloseFunc: hangingClose(t)})

	elapsed, err := closeManagerWithin(t, pm, 7*time.Second)

	if err == nil {
		t.Fatal("Close() error = nil, want the timeout of hung")
	}
	want := "Failed to close plugins: 'hung'\nerror closing plugin 'hung': no result after 5s: context deadline exceeded"
	if got := err.Error(); got != want {
		t.Errorf("Close() error = %q, want %q", got, want)
	}
	if elapsed < 5*time.Second {
		t.Errorf("Close() took %v, want at least 5s", elapsed)
	}
}

func TestPluginManager_Close_ConcurrentCallsEachCloseEveryPlugin(t *testing.T) {
	var aCloses, bCloses atomic.Int32
	pm := newTestManager(t,
		plugin.Config{Name: "a", CloseFunc: func() error { aCloses.Add(1); return nil }},
		plugin.Config{Name: "b", CloseFunc: func() error { bCloses.Add(1); return errors.New("b failed") }},
	)
	const calls = 8

	errs := make([]error, calls)
	var wg sync.WaitGroup
	for i := range calls {
		wg.Go(func() { errs[i] = pm.Close() })
	}
	wg.Wait()

	want := "Failed to close plugins: 'b'\nerror closing plugin 'b': b failed"
	for i, err := range errs {
		if err == nil {
			t.Errorf("Close() call %d error = nil, want %q", i, want)
			continue
		}
		if got := err.Error(); got != want {
			t.Errorf("Close() call %d error = %q, want %q", i, got, want)
		}
	}
	if got := aCloses.Load(); got != calls {
		t.Errorf("closes of a = %d, want %d", got, calls)
	}
	if got := bCloses.Load(); got != calls {
		t.Errorf("closes of b = %d, want %d", got, calls)
	}
}
