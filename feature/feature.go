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

// Package feature provides runtime feature flags. Mirrors adk-python's
// `google.adk.features` subsystem.
//
// Priority (highest to lowest) when resolving IsEnabled:
//  1. Programmatic override (Override / WithOverride)
//  2. Environment variables: ADK_ENABLE_<NAME>=true or ADK_DISABLE_<NAME>=true
//  3. Registry default (Config.DefaultOn)
//
// Stage classifies a feature lifecycle. Non-Stable features that resolve
// enabled emit a one-time WARNING via the standard log package.
package feature

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
)

// Name identifies a feature. Use Register to add a new one before any
// IsEnabled call.
type Name string

// Stage is a feature's lifecycle classification.
type Stage int

const (
	// WIP — work-in-progress; not functioning completely. ADK internal only.
	WIP Stage = iota
	// Experimental — feature works but API may change.
	Experimental
	// Stable — production-ready; no breaking changes without a major bump.
	Stable
)

func (s Stage) String() string {
	switch s {
	case WIP:
		return "wip"
	case Experimental:
		return "experimental"
	case Stable:
		return "stable"
	default:
		return fmt.Sprintf("stage(%d)", int(s))
	}
}

// Config configures a feature.
type Config struct {
	Stage     Stage
	DefaultOn bool
}

var (
	mu        sync.RWMutex
	registry  = map[Name]Config{}
	overrides = map[Name]bool{}
	warned    = map[Name]struct{}{}
)

// Register adds a feature to the registry. Re-registering an existing name
// with a different stage is an error and returns it; this matches the
// Python behavior in `_make_feature_decorator`.
func Register(name Name, cfg Config) error {
	mu.Lock()
	defer mu.Unlock()
	if existing, ok := registry[name]; ok && existing.Stage != cfg.Stage {
		return fmt.Errorf("feature %q already registered with stage %s; cannot redeclare with stage %s",
			name, existing.Stage, cfg.Stage)
	}
	registry[name] = cfg
	return nil
}

// MustRegister is Register that panics on error. Convenient for package init.
func MustRegister(name Name, cfg Config) {
	if err := Register(name, cfg); err != nil {
		panic(err)
	}
}

// Override programmatically sets a feature on/off. Highest priority.
func Override(name Name, enabled bool) error {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := registry[name]; !ok {
		return fmt.Errorf("feature %q is not registered", name)
	}
	overrides[name] = enabled
	return nil
}

// ClearOverride removes a programmatic override (env var / default takes
// over again).
func ClearOverride(name Name) {
	mu.Lock()
	defer mu.Unlock()
	delete(overrides, name)
}

// WithOverride applies an override and returns a restore function suitable
// for defer. Useful in tests.
//
//	defer feature.WithOverride(MyFeature, true)()
func WithOverride(name Name, enabled bool) (restore func()) {
	mu.Lock()
	// Two-value lookup: distinguish a missing key from a key that
	// happens to be set to false. Without this, restoring an explicit
	// override of `false` would incorrectly delete the entry instead of
	// putting the false value back.
	prev, had := overrides[name]
	if _, ok := registry[name]; !ok {
		mu.Unlock()
		// Match Python: invalid feature name is a programmer error.
		panic(fmt.Sprintf("feature %q is not registered", name))
	}
	overrides[name] = enabled
	mu.Unlock()
	return func() {
		mu.Lock()
		defer mu.Unlock()
		if had {
			overrides[name] = prev
		} else {
			delete(overrides, name)
		}
	}
}

// IsEnabled returns whether the feature is on at this moment, applying the
// documented priority. Unregistered features panic — same as Python's
// ValueError.
func IsEnabled(name Name) bool {
	mu.RLock()
	cfg, ok := registry[name]
	if !ok {
		mu.RUnlock()
		panic(fmt.Sprintf("feature %q is not registered", name))
	}
	if v, has := overrides[name]; has {
		mu.RUnlock()
		if v && cfg.Stage != Stable {
			warnOnce(name, cfg.Stage)
		}
		return v
	}
	mu.RUnlock()

	if envEnabled("ADK_ENABLE_" + string(name)) {
		if cfg.Stage != Stable {
			warnOnce(name, cfg.Stage)
		}
		return true
	}
	if envEnabled("ADK_DISABLE_" + string(name)) {
		return false
	}

	if cfg.DefaultOn && cfg.Stage != Stable {
		warnOnce(name, cfg.Stage)
	}
	return cfg.DefaultOn
}

// Check is the error-returning form of IsEnabled. It returns nil if the
// feature is enabled, an error otherwise. Use to gate code paths that
// require a feature.
func Check(name Name) error {
	if IsEnabled(name) {
		return nil
	}
	return fmt.Errorf("feature %q is not enabled", name)
}

// IsRegistered reports whether name is in the registry without panicking.
func IsRegistered(name Name) bool {
	mu.RLock()
	defer mu.RUnlock()
	_, ok := registry[name]
	return ok
}

func envEnabled(varName string) bool {
	v, ok := os.LookupEnv(varName)
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func warnOnce(name Name, stage Stage) {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := warned[name]; ok {
		return
	}
	warned[name] = struct{}{}
	log.Printf("[%s] feature %q is enabled.", strings.ToUpper(stage.String()), name)
}

// reset is for tests in this package.
func reset() {
	mu.Lock()
	defer mu.Unlock()
	registry = map[Name]Config{}
	overrides = map[Name]bool{}
	warned = map[Name]struct{}{}
}
