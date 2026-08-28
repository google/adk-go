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
	"bytes"
	"context"
	"errors"
	"io"
	"maps"
	"path"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestMergedSource_ListFrontmatters(t *testing.T) {
	testErr := errors.New("test error")

	tests := []struct {
		name    string
		sources []Source
		want    []*Frontmatter
		wantErr error
	}{
		{
			name: "duplicates in a single source",
			sources: []Source{
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill1"}, {Name: "skill1"}}},
			},
			wantErr: ErrDuplicateSkill,
		},
		{
			name: "duplicates among sources",
			sources: []Source{
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill1"}}},
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill1"}}},
			},
			wantErr: ErrDuplicateSkill,
		},
		{
			name: "error in source is returned",
			sources: []Source{
				&mockMergedBaseSource{listFrontmattersErr: testErr},
			},
			wantErr: testErr,
		},
		{
			name: "frontmatters are in the order of sources",
			sources: []Source{
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "D"}, {Name: "B"}}},
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "A"}, {Name: "C"}}},
			},
			want: []*Frontmatter{{Name: "D"}, {Name: "B"}, {Name: "A"}, {Name: "C"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			merged := NewMergedSource(tc.sources...)
			got, err := merged.ListFrontmatters(t.Context())
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("ListFrontmatters() error = %v, want %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ListFrontmatters mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergedSource_LoadFrontmatter(t *testing.T) {
	testErr := errors.New("test error")

	tests := []struct {
		name      string
		sources   []Source
		skillName string
		want      *Frontmatter
		wantErr   error
	}{
		{
			name: "returns first success and stops search",
			sources: []Source{
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill A"}}},
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill B", Description: "desc B"}}},
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill B", Description: "desc B dup"}}},
				&mockMergedBaseSource{loadFrontmatterErr: testErr},
			},
			skillName: "skill B",
			want:      &Frontmatter{Name: "skill B", Description: "desc B"},
		},
		{
			name: "error precedence",
			sources: []Source{
				&mockMergedBaseSource{loadFrontmatterErr: testErr},
				&mockMergedBaseSource{frontmatters: []*Frontmatter{{Name: "skill A"}}},
			},
			skillName: "skill A",
			wantErr:   testErr,
		},
		{
			name:      "skill not found",
			sources:   []Source{&mockMergedBaseSource{}, &mockMergedBaseSource{}},
			skillName: "skill C",
			wantErr:   ErrSkillNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			merged := NewMergedSource(tc.sources...)
			got, err := merged.LoadFrontmatter(t.Context(), tc.skillName)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("LoadFrontmatter(%q) error = %v, want %v", tc.skillName, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("LoadFrontmatter mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergedSource_LoadInstructions(t *testing.T) {
	testErr := errors.New("test error")

	tests := []struct {
		name      string
		sources   []Source
		skillName string
		want      string
		wantErr   error
	}{
		{
			name: "returns first success and stops search",
			sources: []Source{
				&mockMergedBaseSource{instructions: map[string]string{"skill A": "inst A"}},
				&mockMergedBaseSource{instructions: map[string]string{"skill B": "inst B"}},
				&mockMergedBaseSource{instructions: map[string]string{"skill B": "inst B dup"}},
				&mockMergedBaseSource{loadInstructionsErr: testErr},
			},
			skillName: "skill B",
			want:      "inst B",
		},
		{
			name: "error precedence",
			sources: []Source{
				&mockMergedBaseSource{loadInstructionsErr: testErr},
				&mockMergedBaseSource{instructions: map[string]string{"skill A": "inst A"}},
			},
			skillName: "skill A",
			wantErr:   testErr,
		},
		{
			name:      "skill not found",
			sources:   []Source{&mockMergedBaseSource{}, &mockMergedBaseSource{}},
			skillName: "skill C",
			wantErr:   ErrSkillNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			merged := NewMergedSource(tc.sources...)
			got, err := merged.LoadInstructions(t.Context(), tc.skillName)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("LoadInstructions(%q) error = %v, want %v", tc.skillName, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if got != tc.want {
				t.Errorf("LoadInstructions(%q) = %q, want %q", tc.skillName, got, tc.want)
			}
		})
	}
}

func TestMergedSource_ListResources(t *testing.T) {
	testErr := errors.New("test error")

	tests := []struct {
		name      string
		sources   []Source
		skillName string
		want      []string
		wantErr   error
	}{
		{
			name: "returns first success and stops search",
			sources: []Source{
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill B": {"p2": "c2"}}},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill B": {"p3": "c3"}}},
				&mockMergedBaseSource{listResourcesErr: testErr},
			},
			skillName: "skill B",
			want:      []string{"p2"},
		},
		{
			name: "error precedence",
			sources: []Source{
				&mockMergedBaseSource{listResourcesErr: testErr},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
			},
			skillName: "skill A",
			wantErr:   testErr,
		},
		{
			name: "ResourceNotFound stops query",
			sources: []Source{
				&mockMergedBaseSource{
					resources:        map[string]map[string]string{"skill A": {}},
					listResourcesErr: ErrResourceNotFound,
				},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
			},
			skillName: "skill A",
			wantErr:   ErrResourceNotFound,
		},
		{
			name:      "skill not found",
			sources:   []Source{&mockMergedBaseSource{}, &mockMergedBaseSource{}},
			skillName: "skill C",
			wantErr:   ErrSkillNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			merged := NewMergedSource(tc.sources...)
			got, err := merged.ListResources(t.Context(), tc.skillName, ".")
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("ListResources(%q, %q) error = %v, want %v", tc.skillName, ".", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ListResources mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergedSource_LoadResource(t *testing.T) {
	testErr := errors.New("test error")

	tests := []struct {
		name         string
		sources      []Source
		skillName    string
		resourcePath string
		want         string
		wantErr      error
	}{
		{
			name: "returns first success and stops search",
			sources: []Source{
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill B": {"p2": "c2"}}},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill B": {"p3": "c3"}}},
				&mockMergedBaseSource{loadResourceErr: testErr},
			},
			skillName:    "skill B",
			resourcePath: "p2",
			want:         "c2",
		},
		{
			name: "error precedence",
			sources: []Source{
				&mockMergedBaseSource{loadResourceErr: testErr},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
			},
			skillName:    "skill A",
			resourcePath: "p1",
			wantErr:      testErr,
		},
		{
			name: "ResourceNotFound stops query",
			sources: []Source{
				&mockMergedBaseSource{
					resources:       map[string]map[string]string{"skill A": {}},
					loadResourceErr: ErrResourceNotFound,
				},
				&mockMergedBaseSource{resources: map[string]map[string]string{"skill A": {"p1": "c1"}}},
			},
			skillName:    "skill A",
			resourcePath: "p1",
			wantErr:      ErrResourceNotFound,
		},
		{
			name: "skill not found",
			sources: []Source{
				&mockMergedBaseSource{},
				&mockMergedBaseSource{},
			},
			skillName:    "skill C",
			resourcePath: "p1",
			wantErr:      ErrSkillNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			merged := NewMergedSource(tc.sources...)
			got, err := merged.LoadResource(t.Context(), tc.skillName, tc.resourcePath)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("LoadResource(%q, %q) error = %v, want %v", tc.skillName, tc.resourcePath, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			data, err := io.ReadAll(got)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != tc.want {
				t.Errorf("LoadResource(%q, %q) = %q, want %q", tc.skillName, tc.resourcePath, string(data), tc.want)
			}
		})
	}
}

type mockMergedBaseSource struct {
	frontmatters []*Frontmatter
	instructions map[string]string
	resources    map[string]map[string]string

	listFrontmattersErr error
	loadFrontmatterErr  error
	loadInstructionsErr error
	listResourcesErr    error
	loadResourceErr     error
}

func (m *mockMergedBaseSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	if m.listFrontmattersErr != nil {
		return nil, m.listFrontmattersErr
	}
	return m.frontmatters, nil
}

func (m *mockMergedBaseSource) ListResources(ctx context.Context, name, _ string) ([]string, error) {
	if m.listResourcesErr != nil {
		return nil, m.listResourcesErr
	}
	skillResource, found := m.resources[name]
	if !found {
		return nil, ErrSkillNotFound
	}
	return slices.SortedFunc(maps.Keys(skillResource), strings.Compare), nil
}

func (m *mockMergedBaseSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	if m.loadFrontmatterErr != nil {
		return nil, m.loadFrontmatterErr
	}
	for _, frontmatter := range m.frontmatters {
		if frontmatter.Name == name {
			return frontmatter, nil
		}
	}
	return nil, ErrSkillNotFound
}

func (m *mockMergedBaseSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	if m.loadInstructionsErr != nil {
		return "", m.loadInstructionsErr
	}
	if v, ok := m.instructions[name]; ok {
		return v, nil
	}
	return "", ErrSkillNotFound
}

func (m *mockMergedBaseSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if m.loadResourceErr != nil {
		return nil, m.loadResourceErr
	}
	skillResource, found := m.resources[name]
	if !found {
		return nil, ErrSkillNotFound
	}
	content, found := skillResource[path.Clean(resourcePath)]
	if !found {
		return nil, ErrResourceNotFound
	}
	return io.NopCloser(bytes.NewReader([]byte(content))), nil
}
