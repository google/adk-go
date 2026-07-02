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

package model

import (
	"context"
	"fmt"
	"regexp"
	"sync"
)

// Factory constructs an [LLM] for the given model name.
//
// The name is the same string that was matched against the registered
// pattern, allowing a single factory to serve a family of models (for
// example, all "gemini-*" names).
type Factory func(ctx context.Context, name string) (LLM, error)

// registration pairs a compiled name pattern with the factory that builds the
// corresponding [LLM].
type registration struct {
	pattern *regexp.Regexp
	factory Factory
}

// registry holds all model registrations in registration order, guarded by mu.
var (
	mu       sync.RWMutex
	registry []registration
)

// Register associates a model name pattern with a [Factory].
//
// namePattern is a regular expression compiled with [regexp.MustCompile];
// it is matched against candidate model names using [regexp.Regexp.MatchString]
// (i.e. an unanchored, partial match). Registrations are retained in
// registration order, and [NewLLM] uses the first pattern that matches a given
// name (first-match-wins). To match a name exactly, anchor the pattern with
// "^" and "$".
//
// Register is intended to be called at initialization time (typically from a
// provider package's init function). It panics if namePattern is not a valid
// regular expression, surfacing the programming error immediately.
//
// Register is safe for concurrent use.
func Register(namePattern string, f Factory) {
	re := regexp.MustCompile(namePattern)
	mu.Lock()
	defer mu.Unlock()
	registry = append(registry, registration{pattern: re, factory: f})
}

// NewLLM builds an [LLM] for the given model name.
//
// It consults the registrations in registration order and invokes the
// [Factory] of the first one whose pattern matches name (first-match-wins),
// returning that factory's result. If no registered pattern matches name, it
// returns an error.
//
// NewLLM is safe for concurrent use.
func NewLLM(ctx context.Context, name string) (LLM, error) {
	mu.RLock()
	// Copy the matching factory out before releasing the lock so we do not hold
	// the registry lock while the factory (which may do I/O) runs.
	var f Factory
	for _, reg := range registry {
		if reg.pattern.MatchString(name) {
			f = reg.factory
			break
		}
	}
	mu.RUnlock()

	if f == nil {
		return nil, fmt.Errorf("model: no registered LLM matches %q", name)
	}
	return f(ctx, name)
}
