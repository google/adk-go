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

// The name-based model registry lets an [LLM] be constructed from just a
// model-name string via [NewLLM], mirroring Python ADK's
// LLMRegistry.new_llm(name).
//
// Registration is opt-in: provider packages do not register themselves on
// import. A user who wants name-based Gemini resolution registers it
// explicitly, using an anchored pattern:
//
//	model.Register("^(?i)gemini-.*", func(ctx context.Context, name string) (model.LLM, error) {
//		return gemini.NewModel(ctx, name, nil)
//	})
//
// [NewLLM] then requires exactly one registered pattern to match a given name:
// zero matches or two-or-more matches are reported as errors, so resolution
// never depends on registration or import order.
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

// registry holds all model registrations, guarded by mu.
var (
	mu       sync.RWMutex
	registry []registration
)

// Register associates a model name pattern with a [Factory].
//
// namePattern is a regular expression compiled with [regexp.MustCompile];
// it is matched against candidate model names using [regexp.Regexp.MatchString]
// (i.e. an unanchored, partial match). To match a name exactly, anchor the
// pattern with "^" and "$".
//
// Registration order is not significant: [NewLLM] requires exactly one
// registered pattern to match a name, so patterns must not overlap for names a
// caller intends to resolve. Register is intended to be called at
// initialization time. It panics if namePattern is not a valid regular
// expression, or if namePattern has already been registered — surfacing the
// programming error at the registration site rather than later in [NewLLM].
//
// Register is safe for concurrent use.
func Register(namePattern string, f Factory) {
	re := regexp.MustCompile(namePattern)
	mu.Lock()
	defer mu.Unlock()
	for _, reg := range registry {
		if reg.pattern.String() == namePattern {
			panic(fmt.Sprintf("model: LLM pattern %q is already registered", namePattern))
		}
	}
	registry = append(registry, registration{pattern: re, factory: f})
}

// NewLLM builds an [LLM] for the given model name.
//
// It scans all registrations and requires exactly one whose pattern matches
// name:
//
//   - If no registered pattern matches, NewLLM returns an error reporting that
//     name was not found.
//   - If exactly one matches, NewLLM invokes its [Factory] and returns the
//     result.
//   - If two or more match, NewLLM returns an error naming the ambiguous name
//     and listing the matching patterns, rather than silently picking one.
//
// Because a match must be unique, the result never depends on registration or
// import order.
//
// NewLLM is safe for concurrent use.
func NewLLM(ctx context.Context, name string) (LLM, error) {
	mu.RLock()
	// Copy the matching factory out before releasing the lock so we do not hold
	// the registry lock while the factory (which may do I/O) runs.
	var f Factory
	var matched []string
	for _, reg := range registry {
		if reg.pattern.MatchString(name) {
			f = reg.factory
			matched = append(matched, reg.pattern.String())
		}
	}
	mu.RUnlock()

	switch len(matched) {
	case 0:
		return nil, fmt.Errorf("model: no registered LLM matches %q", name)
	case 1:
		return f(ctx, name)
	default:
		return nil, fmt.Errorf("model: %q matches multiple registered LLM patterns %q; patterns must be unambiguous", name, matched)
	}
}
