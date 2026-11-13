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

package llminternal

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

// The patterns are registered once per test binary, because model.Register
// panics for a pattern registered twice (go test -count=2).
var registerDefaultModelPatterns sync.Once

func registerDefaultModelTestPatterns() {
	registerDefaultModelPatterns.Do(func() {
		model.Register("^fr-llmi-model-.*$", func(_ context.Context, name string) (model.LLM, error) {
			return &namedTestLLM{name: name}, nil
		})
		model.Register("^fr-llmi-nilmodel-.*$", func(context.Context, string) (model.LLM, error) {
			return nil, nil
		})
		model.Register("^fr-llmi-ctx-.*$", func(ctx context.Context, name string) (model.LLM, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return &namedTestLLM{name: name}, nil
		})
		model.Register("^fr-llmi-barrier-.*$", func(_ context.Context, name string) (model.LLM, error) {
			return activeBarrier.Load().build(name)
		})
	})
}

// barrierFactory counts the models it builds and holds the first two builds
// until both have started, so two resolutions are inside the factory at once.
type barrierFactory struct {
	calls   atomic.Int32
	release chan struct{}
}

// activeBarrier is the barrierFactory the fr-llmi-barrier- pattern builds
// with; a test stores a fresh one before it resolves.
var activeBarrier atomic.Pointer[barrierFactory]

func (b *barrierFactory) build(name string) (model.LLM, error) {
	if b.calls.Add(1) == 2 {
		close(b.release)
	}
	select {
	case <-b.release:
		return &namedTestLLM{name: name}, nil
	case <-time.After(10 * time.Second):
		return nil, errors.New("the second resolution never reached the factory")
	}
}

func TestDefaultModel_ConcurrentFirstResolutionsShareOneModel(t *testing.T) {
	registerDefaultModelTestPatterns()
	factory := &barrierFactory{release: make(chan struct{})}
	activeBarrier.Store(factory)
	d := &defaultModel{kind: "test default", name: "fr-llmi-barrier-a"}

	var first, second model.LLM
	var firstErr, secondErr error
	var wg sync.WaitGroup
	wg.Go(func() { first, firstErr = d.resolve(t.Context()) })
	wg.Go(func() { second, secondErr = d.resolve(t.Context()) })
	wg.Wait()
	third, thirdErr := d.resolve(t.Context())

	for _, err := range []error{firstErr, secondErr, thirdErr} {
		if err != nil {
			t.Fatalf("resolve() error = %v, want nil", err)
		}
	}
	if first != second {
		t.Errorf("concurrent first resolutions returned %p and %p, want the same model", first, second)
	}
	if third != first {
		t.Errorf("resolve() after the first resolutions = %p, want %p, the model they share", third, first)
	}
	if got := factory.calls.Load(); got != 2 {
		t.Errorf("factory calls = %d, want 2 (one per concurrent first resolution, none after)", got)
	}
}

func TestDefaultModel_ResolutionStartedBeforeSetterDoesNotOverwriteIt(t *testing.T) {
	registerDefaultModelTestPatterns()
	d := &defaultModel{kind: "test default", name: "fr-llmi-model-old"}

	// A resolution reads the name and generation, then a setter runs before
	// the resolution stores what it built.
	_, name, gen := d.current()
	if err := d.setName("fr-llmi-model-new"); err != nil {
		t.Fatal(err)
	}
	stale := &namedTestLLM{name: name}
	if got := d.keep(gen, stale); got != stale {
		t.Errorf("keep() for the stale generation = %s, want the stale model returned to its caller", got.Name())
	}

	m, err := d.resolve(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if m.Name() != "fr-llmi-model-new" {
		t.Errorf("resolve() after the setter = %s, want fr-llmi-model-new", m.Name())
	}
}

func TestDefaultModel_CanceledFirstCallerDoesNotFailResolution(t *testing.T) {
	registerDefaultModelTestPatterns()
	d := &defaultModel{kind: "test default", name: "fr-llmi-ctx-a"}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	m, err := d.resolve(ctx)
	if err != nil {
		t.Fatalf("resolve() with a canceled context: %v, want the model", err)
	}
	if m.Name() != "fr-llmi-ctx-a" {
		t.Errorf("resolve() = %s, want fr-llmi-ctx-a", m.Name())
	}
}

func TestDefaultModel_NilModelFromFactoryIsAnError(t *testing.T) {
	registerDefaultModelTestPatterns()
	d := &defaultModel{kind: "test default", name: "fr-llmi-nilmodel-a"}

	m, err := d.resolve(t.Context())
	if !errors.Is(err, errNilModelFromRegistry) {
		t.Errorf("resolve() = %v, %v; want an error wrapping %v", m, err, errNilModelFromRegistry)
	}
}

func TestResolveModel_NonLLMAgentWrapsErrModelNotConfigured(t *testing.T) {
	custom, err := agent.New(agent.Config{Name: "custom"})
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []agent.Agent{nil, custom} {
		if _, err := ResolveModel(t.Context(), a); !errors.Is(err, ErrModelNotConfigured) {
			t.Errorf("ResolveModel(%v) error = %v, want one wrapping ErrModelNotConfigured", a, err)
		}
		if _, err := ResolveLiveModel(t.Context(), a); !errors.Is(err, ErrModelNotConfigured) {
			t.Errorf("ResolveLiveModel(%v) error = %v, want one wrapping ErrModelNotConfigured", a, err)
		}
	}
}
