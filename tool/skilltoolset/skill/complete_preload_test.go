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
	"fmt"
	"io"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestWithCompletePreloadSource(t *testing.T) {
	// Test initial load.
	mock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{
			{Name: "skill1", Description: "desc1"},
			{Name: "skill2", Description: "desc2"},
		},
		instructions: map[string]string{
			"skill1": "instructions1",
			"skill2": "instructions2",
		},
		resources: map[string]map[string][]byte{
			"skill1": {
				"assets/image.png":   []byte("image-data"),
				"references/doc.txt": []byte("doc-data"),
				"scripts/script.py":  []byte("script-data"),
			},
			"skill2": {},
		},
	}

	source, reload, err := WithCompletePreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithCompletePreloadSource failed: %v", err)
	}

	if err := mock.verifyCallCount(1, 2, 2, 3); err != nil {
		t.Fatal(err)
	}
	collected, err := collectSourceData(t.Context(), source)
	if err != nil {
		t.Fatalf("collectSourceData failed: %v", err)
	}
	if err := mock.verifyCallCount(1, 2, 2, 3); err != nil {
		t.Errorf("the base mock should not have been called: %v", err)
	}
	wantCollected := &collectedSourceData{
		Frontmatters: []*Frontmatter{
			{Name: "skill1", Description: "desc1"},
			{Name: "skill2", Description: "desc2"},
		},
		Instructions: map[string]string{
			"skill1": "instructions1",
			"skill2": "instructions2",
		},
		Resources: map[string]map[string][]byte{
			"skill1": {
				"assets/image.png":   []byte("image-data"),
				"references/doc.txt": []byte("doc-data"),
				"scripts/script.py":  []byte("script-data"),
			},
		},
	}
	if diff := cmp.Diff(wantCollected, collected); diff != "" {
		t.Errorf("collectedSourceData mismatch (-want +got):\n%s", diff)
	}

	// Test reload.
	mock.frontmatters = []*Frontmatter{{Name: "skill1", Description: "desc1_updated"}}
	mock.instructions["skill1"] = "instructions1_updated"
	mock.resources = map[string]map[string][]byte{
		"skill1": {"assets/image.png": []byte("image-data-2")},
	}

	err = reload(t.Context())
	if err != nil {
		t.Fatalf("Reload failed: %v", err)
	}

	if err := mock.verifyCallCount(1+1, 2+1, 2+1, 3+1); err != nil {
		t.Error(err)
	}
	collected, err = collectSourceData(t.Context(), source)
	if err != nil {
		t.Fatalf("collectSourceData failed: %v", err)
	}
	if err := mock.verifyCallCount(1+1, 2+1, 2+1, 3+1); err != nil {
		t.Errorf("the base mock should not have been called: %v", err)
	}
	wantCollected = &collectedSourceData{
		Frontmatters: []*Frontmatter{{Name: "skill1", Description: "desc1_updated"}},
		Instructions: map[string]string{"skill1": "instructions1_updated"},
		Resources: map[string]map[string][]byte{
			"skill1": {"assets/image.png": []byte("image-data-2")},
		},
	}
	if diff := cmp.Diff(wantCollected, collected); diff != "" {
		t.Errorf("collectedSourceData mismatch (-want +got):\n%s", diff)
	}
}

func TestWithCompletePreloadSource_ReloadFailureDoesNotInvalidateCache(t *testing.T) {
	mock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "skill1"}},
		instructions: map[string]string{"skill1": "instructions1"},
		resources: map[string]map[string][]byte{
			"skill1": {"assets/image.png": []byte("image-data")},
		},
	}

	source, reload, err := WithCompletePreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithCompletePreloadSource failed: %v", err)
	}
	collected, err := collectSourceData(t.Context(), source)
	if err != nil {
		t.Fatalf("collectSourceData failed: %v", err)
	}
	wantCollected := &collectedSourceData{
		Frontmatters: []*Frontmatter{{Name: "skill1"}},
		Instructions: map[string]string{"skill1": "instructions1"},
		Resources: map[string]map[string][]byte{
			"skill1": {"assets/image.png": []byte("image-data")},
		},
	}
	if diff := cmp.Diff(wantCollected, collected); diff != "" {
		t.Errorf("collectedSourceData mismatch (-want +got):\n%s", diff)
	}

	mock.frontmatters = []*Frontmatter{{Name: "skill2"}}
	mock.instructions = map[string]string{"skill2": "instructions2"}
	mock.resources = map[string]map[string][]byte{
		"skill2": {"assets/image2.png": []byte("image-data-2")},
	}
	mock.listResourcesErr = fmt.Errorf("list resources error")

	err = reload(t.Context())
	if err == nil {
		t.Fatalf("Reload should have failed")
	}

	collected, err = collectSourceData(t.Context(), source)
	if err != nil {
		t.Fatalf("collectSourceData failed: %v", err)
	}
	// should be unchanged.
	if diff := cmp.Diff(wantCollected, collected); diff != "" {
		t.Errorf("collectedSourceData mismatch (-want +got):\n%s", diff)
	}
}

func TestWithCompletePreloadSource_ListResources_Filtering(t *testing.T) {
	mock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "skill1"}},
		instructions: map[string]string{"skill1": "instructions1"},
		resources: map[string]map[string][]byte{
			"skill1": {
				"assets/image.png":              []byte("image-data"),
				"references/doc.txt":            []byte("doc-data"),
				"scripts/script.py.js":          []byte("js-script-data"),
				"scripts/script.py":             []byte("py-script-data"),
				"scripts/script":                []byte("script-data"),
				"scripts/script/another_script": []byte("another-script-data"),
			},
		},
	}

	source, _, err := WithCompletePreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithCompletePreloadSource failed: %v", err)
	}

	tests := []struct {
		name      string
		subpath   string
		wantPaths []string
	}{
		{
			name:      "Folder match",
			subpath:   "assets",
			wantPaths: []string{"assets/image.png"},
		},
		{
			name:      "Folder match with trailing slash",
			subpath:   "assets/",
			wantPaths: []string{"assets/image.png"},
		},
		{
			name:      "Another folder match",
			subpath:   "references",
			wantPaths: []string{"references/doc.txt"},
		},
		{
			name:      "Specific resource request",
			subpath:   "scripts/script.py",
			wantPaths: []string{"scripts/script.py"},
		},
		{
			// Whether or not it is possible to have both a file and a directory
			// with the same name is dependent on the base source implementation.
			name:      "Match both directory and file with the same name",
			subpath:   "scripts/script",
			wantPaths: []string{"scripts/script/another_script", "scripts/script"},
		},
		{
			name:      "No match",
			subpath:   "nonexistent",
			wantPaths: nil,
		},
		{
			name:      "Root match",
			subpath:   ".",
			wantPaths: []string{"assets/image.png", "references/doc.txt", "scripts/script", "scripts/script.py", "scripts/script/another_script", "scripts/script.py.js"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			paths, err := source.ListResources(t.Context(), "skill1", tc.subpath)
			if err != nil {
				t.Fatalf("ListResources failed: %v", err)
			}
			if diff := cmp.Diff(tc.wantPaths, paths, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
				t.Errorf("ListResources mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestWithCompletePreloadSource_LoadResource(t *testing.T) {
	mock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "skill1"}},
		instructions: map[string]string{"skill1": "instructions1"},
		resources: map[string]map[string][]byte{
			"skill1": {
				"assets/image1.png":          []byte("image-data-1"),
				"assets/dir/.././image2.png": []byte("image-data-2"),
			},
		},
	}
	source, _, err := WithCompletePreloadSource(t.Context(), mock)
	if err != nil {
		t.Fatalf("WithCompletePreloadSource failed: %v", err)
	}

	tests := []struct {
		name    string
		path    string
		want    string
		wantErr error
	}{
		{
			name: "Direct path",
			path: "assets/image1.png",
			want: "image-data-1",
		},
		{
			name: "Complex path",
			path: "assets/../././assets/./image2.png",
			want: "image-data-2",
		},
		{
			name:    "Not existing resource",
			path:    "assets/nonexistent.png",
			wantErr: ErrResourceNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resource, err := source.LoadResource(t.Context(), "skill1", tc.path)

			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("LoadResource(%q, %q) error: %v, wantErr: %v", "skill1", tc.path, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			data, err := io.ReadAll(resource)
			if err != nil {
				t.Fatalf("LoadResource(%q, %q): io.ReadAll failed: %v", "skill1", tc.path, err)
			}
			if tc.want != string(data) {
				t.Fatalf("LoadResource(%q, %q) = %q, want %q", "skill1", tc.path, string(data), tc.want)
			}
		})
	}
}

func TestWithCompletePreloadSource_ResourceSizeLimit(t *testing.T) {
	maximumSizeData := make([]byte, maxResourceSize)
	mock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "skill1"}},
		instructions: map[string]string{"skill1": "instructions1"},
		resources: map[string]map[string][]byte{
			"skill1": {"assets/justRight.png": maximumSizeData},
		},
	}
	exceedMaximumSizeData := make([]byte, maxResourceSize+1)
	exceedMock := &mockCompletePreloadBaseSource{
		frontmatters: []*Frontmatter{{Name: "skill1"}},
		instructions: map[string]string{"skill1": "instructions1"},
		resources: map[string]map[string][]byte{
			"skill1": {"assets/tooLarge.png": exceedMaximumSizeData},
		},
	}

	_, _, err := WithCompletePreloadSource(t.Context(), mock)
	_, _, exceedErr := WithCompletePreloadSource(t.Context(), exceedMock)

	if err != nil {
		t.Fatalf("WithCompletePreloadSource() error for maximum size data: %v", err)
	}
	if exceedErr == nil {
		t.Fatalf("WithCompletePreloadSource() should have failed for exceed maximum size data: %v", exceedErr)
	}
}

func TestWithCompletePreloadSource_Errors(t *testing.T) {
	goodSource := func() *mockCompletePreloadBaseSource {
		return &mockCompletePreloadBaseSource{
			frontmatters: []*Frontmatter{{Name: "skill1"}},
			instructions: map[string]string{"skill1": "instructions1"},
			resources: map[string]map[string][]byte{
				"skill1": {"assets/image.png": []byte("image-data")},
			},
		}
	}
	testError := fmt.Errorf("test error")

	successSource := goodSource()
	listfrontmattersFailedSource := goodSource()
	loadInstructionsFailedSource := goodSource()
	listResourcesFailedSource := goodSource()
	loadResourceFailedSource := goodSource()
	duplicateSkillSource := goodSource()

	listfrontmattersFailedSource.listFrontmattersErr = testError
	loadInstructionsFailedSource.loadInstructionsErr = testError
	listResourcesFailedSource.listResourcesErr = testError
	loadResourceFailedSource.loadResourceErr = testError
	duplicateSkillSource.frontmatters = append(duplicateSkillSource.frontmatters, duplicateSkillSource.frontmatters...)

	tests := []struct {
		name string
		mock *mockCompletePreloadBaseSource
		want error
	}{
		{
			name: "Succeed", // for completeness
			mock: successSource,
			want: nil,
		},
		{
			name: "ListFrontmatters error",
			mock: listfrontmattersFailedSource,
			want: testError,
		},
		{
			name: "LoadInstructions error",
			mock: loadInstructionsFailedSource,
			want: testError,
		},
		{
			name: "ListResources error",
			mock: listResourcesFailedSource,
			want: testError,
		},
		{
			name: "LoadResource error",
			mock: loadResourceFailedSource,
			want: testError,
		},
		{
			name: "Duplicate skill error",
			mock: duplicateSkillSource,
			want: ErrDuplicateSkill,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := WithCompletePreloadSource(t.Context(), tc.mock)
			if !errors.Is(err, tc.want) {
				t.Errorf("WithCompletePreloadSource() error = %v, want error %v", err, tc.want)
			}
		})
	}
}

type mockCompletePreloadBaseSource struct {
	Source
	frontmatters          []*Frontmatter
	instructions          map[string]string
	resources             map[string]map[string][]byte
	listFrontmattersCalls int
	listResourcesCalls    int
	loadInstructionsCalls int
	loadResourceCalls     int
	listFrontmattersErr   error
	loadInstructionsErr   error
	listResourcesErr      error
	loadResourceErr       error
}

func (m *mockCompletePreloadBaseSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	m.listFrontmattersCalls++
	if m.listFrontmattersErr != nil {
		return nil, m.listFrontmattersErr
	}
	return m.frontmatters, nil
}

func (m *mockCompletePreloadBaseSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	m.loadInstructionsCalls++
	if m.loadInstructionsErr != nil {
		return "", m.loadInstructionsErr
	}
	if v, ok := m.instructions[name]; ok {
		return v, nil
	}
	return "", ErrSkillNotFound
}

func (m *mockCompletePreloadBaseSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	m.listResourcesCalls++
	if m.listResourcesErr != nil {
		return nil, m.listResourcesErr
	}
	res, ok := m.resources[name]
	if !ok {
		return nil, ErrSkillNotFound
	}
	var paths []string
	for p := range res {
		paths = append(paths, p)
	}
	slices.Sort(paths)
	if subpath != "." {
		return nil, fmt.Errorf("unexpected subpath %q: mock only supports '.' as required by complete preload", subpath)
	}
	return paths, nil
}

func (m *mockCompletePreloadBaseSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	m.loadResourceCalls++
	if m.loadResourceErr != nil {
		return nil, m.loadResourceErr
	}
	res, ok := m.resources[name]
	if !ok {
		return nil, ErrSkillNotFound
	}
	data, ok := res[resourcePath]
	if !ok {
		return nil, ErrResourceNotFound
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

func (m *mockCompletePreloadBaseSource) verifyCallCount(listFrontmattersCalls, listResourcesCalls, loadInstructionsCalls, loadResourceCalls int) error {
	if m.listFrontmattersCalls != listFrontmattersCalls {
		return fmt.Errorf("listFrontmattersCalls: %d, want %d", m.listFrontmattersCalls, listFrontmattersCalls)
	}
	if m.listResourcesCalls != listResourcesCalls {
		return fmt.Errorf("listResourcesCalls: %d, want %d", m.listResourcesCalls, listResourcesCalls)
	}
	if m.loadInstructionsCalls != loadInstructionsCalls {
		return fmt.Errorf("loadInstructionsCalls: %d, want %d", m.loadInstructionsCalls, loadInstructionsCalls)
	}
	if m.loadResourceCalls != loadResourceCalls {
		return fmt.Errorf("loadResourceCalls: %d, want %d", m.loadResourceCalls, loadResourceCalls)
	}
	return nil
}

type collectedSourceData struct {
	// Fields are exported to enable comparison using cmp.Diff.
	Frontmatters []*Frontmatter
	Instructions map[string]string
	Resources    map[string]map[string][]byte
}

func collectSourceData(ctx context.Context, source Source) (*collectedSourceData, error) {
	collected := &collectedSourceData{}
	frontmatters, err := source.ListFrontmatters(ctx)
	if err != nil {
		return nil, err
	}
	collected.Frontmatters = frontmatters
	for _, fm := range frontmatters {
		loadedFrontmatter, err := source.LoadFrontmatter(ctx, fm.Name)
		if err != nil {
			return nil, err
		}
		if diff := cmp.Diff(fm, loadedFrontmatter); diff != "" {
			return nil, fmt.Errorf("frontmatter mismatch for %q:%s", fm.Name, diff)
		}
		instructions, err := source.LoadInstructions(ctx, fm.Name)
		if err != nil {
			return nil, err
		}
		if collected.Instructions == nil {
			collected.Instructions = make(map[string]string)
		}
		collected.Instructions[fm.Name] = instructions
		resources, err := source.ListResources(ctx, fm.Name, ".")
		if err != nil {
			return nil, err
		}
		if len(resources) == 0 {
			continue
		}
		if collected.Resources == nil {
			collected.Resources = make(map[string]map[string][]byte)
		}
		collected.Resources[fm.Name] = make(map[string][]byte)
		for _, resourcePath := range resources {
			resource, err := source.LoadResource(ctx, fm.Name, resourcePath)
			if err != nil {
				return nil, fmt.Errorf("loading resource %q:%s: %w", fm.Name, resourcePath, err)
			}
			data, err := io.ReadAll(resource)
			_ = resource.Close()
			if err != nil {
				return nil, fmt.Errorf("reading resource %q:%s: %w", fm.Name, resourcePath, err)
			}
			collected.Resources[fm.Name][resourcePath] = data
		}
	}
	return collected, nil
}
