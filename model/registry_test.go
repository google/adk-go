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

func TestRegisterAndNewLLM(t *testing.T) {
	// Use a unique, test-specific pattern: the registry is global and shared
	// across all tests in the package.
	const pattern = "^registry-test-stub-.*$"
	Register(pattern, func(_ context.Context, name string) (LLM, error) {
		return &stubLLM{name: name}, nil
	})

	llm, err := NewLLM(context.Background(), "registry-test-stub-001")
	if err != nil {
		t.Fatalf("NewLLM returned unexpected error: %v", err)
	}
	stub, ok := llm.(*stubLLM)
	if !ok {
		t.Fatalf("NewLLM returned %T, want *stubLLM", llm)
	}
	if got, want := stub.Name(), "registry-test-stub-001"; got != want {
		t.Errorf("stub.Name() = %q, want %q (factory should receive the matched name)", got, want)
	}
}

func TestNewLLMNoMatch(t *testing.T) {
	// A name that no registered pattern can match.
	const name = "registry-test-no-such-provider-xyzzy-0000"
	_, err := NewLLM(context.Background(), name)
	if err == nil {
		t.Fatalf("NewLLM(%q) returned nil error, want no-match error", name)
	}
	if !strings.Contains(err.Error(), name) {
		t.Errorf("NewLLM error = %q, want it to mention the unmatched name %q", err.Error(), name)
	}
}

func TestNewLLMFirstMatchWins(t *testing.T) {
	// Two patterns that both match the same name; the first one registered must
	// win. Tag each factory's output via the name prefix it hard-codes so we can
	// tell them apart.
	const name = "registry-test-precedence-001"

	Register("^registry-test-precedence-.*$", func(_ context.Context, _ string) (LLM, error) {
		return &stubLLM{name: "first"}, nil
	})
	Register("^registry-test-precedence-001$", func(_ context.Context, _ string) (LLM, error) {
		return &stubLLM{name: "second"}, nil
	})

	llm, err := NewLLM(context.Background(), name)
	if err != nil {
		t.Fatalf("NewLLM returned unexpected error: %v", err)
	}
	if got, want := llm.Name(), "first"; got != want {
		t.Errorf("first-match-wins violated: NewLLM(%q) used factory %q, want %q", name, got, want)
	}
}

func TestRegistryConcurrentAccess(t *testing.T) {
	// Exercise Register and NewLLM concurrently so `go test -race` can detect
	// data races on the package-level registry. Each goroutine uses a distinct
	// pattern to avoid relying on registration order across goroutines.
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
			_, _ = NewLLM(context.Background(), "registry-test-concurrent-"+suffix+"-x")
		}()
	}

	wg.Wait()
}
