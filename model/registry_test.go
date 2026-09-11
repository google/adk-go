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
	"iter"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// stubLLM is a minimal [LLM] implementation used by the registry tests. It
// carries the name it was constructed with so tests can assert which factory
// produced it.
type stubLLM struct {
	name string
}

func (s *stubLLM) Name() string { return s.name }

func (s *stubLLM) GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error] {
	return func(yield func(*LLMResponse, error) bool) {}
}

// The registry is a global shared across every test in this package, so each
// test registers its own uniquely prefixed patterns and never relies on
// provider packages (e.g. gemini) self-registering.

func TestNewLLMSingleMatch(t *testing.T) {
	const pattern = "^registry-test-single-.*$"
	Register(pattern, func(_ context.Context, name string) (LLM, error) {
		return &stubLLM{name: name}, nil
	})

	llm, err := NewLLM(t.Context(), "registry-test-single-001")
	if err != nil {
		t.Fatalf("NewLLM returned unexpected error: %v", err)
	}
	stub, ok := llm.(*stubLLM)
	if !ok {
		t.Fatalf("NewLLM returned %T, want *stubLLM", llm)
	}
	if got, want := stub.Name(), "registry-test-single-001"; got != want {
		t.Errorf("stub.Name() = %q, want %q (factory should receive the matched name)", got, want)
	}
}

func TestRegisterDuplicatePatternPanics(t *testing.T) {
	// Registering the same pattern twice is a programming error and must panic
	// at the registration site rather than surfacing later in NewLLM.
	const pattern = "^registry-test-dup-.*$"
	Register(pattern, func(_ context.Context, name string) (LLM, error) {
		return &stubLLM{name: name}, nil
	})
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Register did not panic when the same pattern was registered twice")
		}
	}()
	Register(pattern, func(_ context.Context, name string) (LLM, error) {
		return &stubLLM{name: name}, nil
	})
}

func TestNewLLMNoMatch(t *testing.T) {
	// A name that no registered pattern can match.
	const name = "registry-test-no-such-provider-xyzzy-0000"
	_, err := NewLLM(t.Context(), name)
	if err == nil {
		t.Fatalf("NewLLM(%q) returned nil error, want no-match error", name)
	}
	if !strings.Contains(err.Error(), name) {
		t.Errorf("NewLLM error = %q, want it to mention the unmatched name %q", err.Error(), name)
	}
}

func TestNewLLMMultipleMatchesError(t *testing.T) {
	// Two patterns that both match the same name. NewLLM must refuse to guess
	// and instead return an error, regardless of registration order.
	const (
		name     = "registry-test-ambiguous-001"
		broad    = "^registry-test-ambiguous-.*$"
		specific = "^registry-test-ambiguous-001$"
	)
	Register(broad, func(_ context.Context, _ string) (LLM, error) {
		return &stubLLM{name: "broad"}, nil
	})
	Register(specific, func(_ context.Context, _ string) (LLM, error) {
		return &stubLLM{name: "specific"}, nil
	})

	llm, err := NewLLM(t.Context(), name)
	if err == nil {
		t.Fatalf("NewLLM(%q) returned %v, want ambiguous-match error", name, llm)
	}
	// The error must name the ambiguous input and list the matching patterns.
	for _, want := range []string{name, broad, specific} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("NewLLM error = %q, want it to mention %q", err.Error(), want)
		}
	}
}

func TestRegistryConcurrentAccess(t *testing.T) {
	// Exercise Register and NewLLM concurrently so `go test -race` can detect
	// data races on the package-level registry. Each goroutine uses a distinct
	// pattern so no name matches more than one of them.
	const goroutines = 16
	var wg sync.WaitGroup
	wg.Add(goroutines * 2)

	for i := 0; i < goroutines; i++ {
		suffix := strconv.Itoa(i)
		// Writer: register a unique pattern.
		go func() {
			defer wg.Done()
			Register("^registry-test-concurrent-"+suffix+"-.*$", func(_ context.Context, name string) (LLM, error) {
				return &stubLLM{name: name}, nil
			})
		}()
		// Reader: look up a name. It may or may not match depending on
		// scheduling; either outcome is fine, we only care about race safety.
		go func() {
			defer wg.Done()
			_, _ = NewLLM(t.Context(), "registry-test-concurrent-"+suffix+"-x")
		}()
	}

	wg.Wait()
}
