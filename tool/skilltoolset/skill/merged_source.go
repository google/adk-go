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
	"fmt"
	"io"
	"slices"
)

type mergedSource struct {
	sources []Source
}

// NewMergedSource creates a Source that combines multiple underlying
// Source implementations. The sources are queried in the order they are
// provided.
func NewMergedSource(sources ...Source) Source {
	return &mergedSource{sources: slices.Clone(sources)}
}

func (m *mergedSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	var allFrontmatters []*Frontmatter
	names := make(map[string]bool) // Track skill names to detect duplicates.

	for _, source := range m.sources {
		frontmatters, err := source.ListFrontmatters(ctx)
		if err != nil {
			return nil, err
		}
		for _, fm := range frontmatters {
			if names[fm.Name] {
				return nil, fmt.Errorf("%w: %q", ErrDuplicateSkill, fm.Name)
			}
			names[fm.Name] = true
		}
		allFrontmatters = append(allFrontmatters, frontmatters...)
	}

	return allFrontmatters, nil
}

func (m *mergedSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	for _, source := range m.sources {
		res, err := source.ListResources(ctx, name, subpath)
		if err == nil {
			return res, nil
		}
		if !errors.Is(err, ErrSkillNotFound) {
			return nil, err
		}
	}
	return nil, ErrSkillNotFound
}

func (m *mergedSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	for _, source := range m.sources {
		frontmatter, err := source.LoadFrontmatter(ctx, name)
		if err == nil {
			return frontmatter, nil
		}
		if !errors.Is(err, ErrSkillNotFound) {
			return nil, err
		}
	}
	return nil, ErrSkillNotFound
}

func (m *mergedSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	for _, source := range m.sources {
		instructions, err := source.LoadInstructions(ctx, name)
		if err == nil {
			return instructions, nil
		}
		if !errors.Is(err, ErrSkillNotFound) {
			return "", err
		}
	}
	return "", ErrSkillNotFound
}

func (m *mergedSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	for _, source := range m.sources {
		resource, err := source.LoadResource(ctx, name, resourcePath)
		if err == nil {
			return resource, nil
		}
		if !errors.Is(err, ErrSkillNotFound) {
			return nil, err
		}
	}
	return nil, ErrSkillNotFound
}
