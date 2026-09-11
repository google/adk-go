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

package skill

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWithFrontmatterPreloadSource_Success(t *testing.T) {
	// Test Init.
	mock := &mockFrontmatterPreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "B"}, {Name: "A"}},
	}
	source, reload, err := WithFrontmatterPreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithFrontmatterPreloadSource() returned unexpected error: %v", err)
	}
	if got, want := mock.listFrontmattersCalls, 1; got != want {
		t.Errorf("mock.listFrontmattersCalls = %v, want %v", got, want)
	}
	got, err := source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("source.ListFrontmatters() returned unexpected error: %v", err)
	}
	want := []*Frontmatter{{Name: "A"}, {Name: "B"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ListFrontmatters mismatch (-want +got):\n%s", diff)
	}

	// Test Reload.
	mock.frontmatters = []*Frontmatter{{Name: "C"}}
	err = reload(t.Context())
	if err != nil {
		t.Fatalf("reload() returned unexpected error: %v", err)
	}
	if got, want := mock.listFrontmattersCalls, 2; got != want {
		t.Errorf("mock.listFrontmattersCalls = %v, want %v after reload", got, want)
	}
	got, err = source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("source.ListFrontmatters() returned unexpected error: %v", err)
	}
	want = []*Frontmatter{{Name: "C"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ListFrontmatters mismatch after reload (-want +got):\n%s", diff)
	}
}

func TestWithFrontmatterPreloadSource_DuplicateSkill(t *testing.T) {
	mock := &mockFrontmatterPreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "A"}, {Name: "A"}},
	}

	_, _, err := WithFrontmatterPreloadSource(t.Context(), mock)

	if !errors.Is(err, ErrDuplicateSkill) {
		t.Errorf("WithFrontmatterPreloadSource error = %v, want %v", err, ErrDuplicateSkill)
	}
}

func TestWithFrontmatterPreloadSource_ReloadFailureDoesNotInvalidateCache(t *testing.T) {
	mock := &mockFrontmatterPreloadBaseSource{frontmatters: []*Frontmatter{{Name: "A"}}}
	source, reload, err := WithFrontmatterPreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithFrontmatterPreloadSource failed: %v", err)
	}
	testErr := errors.New("reload error")
	mock.listFrontmattersErr = testErr

	err = reload(t.Context())

	// If reload fails, the cache should still be valid.
	if !errors.Is(err, testErr) {
		t.Fatalf("Reload error = %v, want %v", err, testErr)
	}
	got, err := source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters failed: %v", err)
	}
	if diff := cmp.Diff([]*Frontmatter{{Name: "A"}}, got); diff != "" {
		t.Errorf("ListFrontmatters mismatch (-want +got):\n%s", diff)
	}
}

func TestFrontmatterPreloadSource_ListFrontmatters(t *testing.T) {
	mock := &mockFrontmatterPreloadBaseSource{frontmatters: []*Frontmatter{{Name: "A"}}}

	source, _, err := WithFrontmatterPreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithFrontmatterPreloadSource failed: %v", err)
	}

	// Should be called once on init.
	if got, want := mock.listFrontmattersCalls, 1; got != want {
		t.Errorf("mock.listFrontmattersCalls = %v, want %v", got, want)
	}
	// Calls should be cached.
	got1, err := source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters failed: %v", err)
	}
	got2, err := source.ListFrontmatters(t.Context())
	if err != nil {
		t.Fatalf("ListFrontmatters failed: %v", err)
	}
	// Should not change since list is cached.
	if got, want := mock.listFrontmattersCalls, 1; got != want {
		t.Errorf("mock.listFrontmattersCalls = %v, want %v", got, want)
	}
	if diff := cmp.Diff(got1, got2); diff != "" {
		t.Errorf("ListFrontmatters results differ (-got1 +got2):\n%s", diff)
	}
}

func TestFrontmatterPreloadSource_LoadFrontmatter(t *testing.T) {
	mock := &mockFrontmatterPreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "A", Description: "desc A"}},
	}

	source, _, err := WithFrontmatterPreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithFrontmatterPreloadSource failed: %v", err)
	}

	// Success case.
	got, err := source.LoadFrontmatter(t.Context(), "A")
	if err != nil {
		t.Fatalf("source.LoadFrontmatter() returned unexpected error: %v", err)
	}
	want := &Frontmatter{Name: "A", Description: "desc A"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LoadFrontmatter mismatch (-want +got):\n%s", diff)
	}

	// NotFound case.
	_, err = source.LoadFrontmatter(t.Context(), "B")
	if !errors.Is(err, ErrSkillNotFound) {
		t.Errorf("LoadFrontmatter error = %v, want %v", err, ErrSkillNotFound)
	}

	// Check LoadFrontmatter was not called for existing or non-existing skill -
	// they were preloaded.
	if got, want := mock.loadFrontmatterCalls, 0; got != want {
		t.Errorf("mock.loadFrontmatterCalls = %v, want %v (cached)", got, want)
	}
}

type mockFrontmatterPreloadBaseSource struct {
	Source
	frontmatters          []*Frontmatter
	listFrontmattersErr   error
	listFrontmattersCalls int
	loadFrontmatterCalls  int
}

func (m *mockFrontmatterPreloadBaseSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	m.listFrontmattersCalls++
	if m.listFrontmattersErr != nil {
		return nil, m.listFrontmattersErr
	}
	return m.frontmatters, nil
}

func (m *mockFrontmatterPreloadBaseSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	m.loadFrontmatterCalls++
	for _, fm := range m.frontmatters {
		if fm.Name == name {
			return fm, nil
		}
	}
	return nil, ErrSkillNotFound
}
