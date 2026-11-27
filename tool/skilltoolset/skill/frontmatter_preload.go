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
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
)

// frontmatterPreloadSource serves the frontmatters from an in-memory copy of
// the wrapped source's, which reload replaces as a whole, and everything else
// from the wrapped source.
type frontmatterPreloadSource struct {
	Source
	preload atomic.Pointer[inMemorySource]
}

// WithFrontmatterPreloadSource returns a Source proxy that wraps the given
// source and eagerly preloads all skill Frontmatters into memory upon creation.
// This optimizes future calls to ListFrontmatters and LoadFrontmatter.
// Additionally, 'reload' callback is returned: calling it actualizes
// data by loading frontmatters anew.
func WithFrontmatterPreloadSource(ctx context.Context, source Source) (Source, func(context.Context) error, error) {
	s := &frontmatterPreloadSource{Source: source}
	if err := s.reload(ctx); err != nil {
		return nil, nil, err
	}
	return s, s.reload, nil
}

func (s *frontmatterPreloadSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	return s.preload.Load().ListFrontmatters(ctx)
}

func (s *frontmatterPreloadSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	return s.preload.Load().LoadFrontmatter(ctx, name)
}

func (s *frontmatterPreloadSource) reload(ctx context.Context) error {
	sorted, err := s.Source.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("preload: %w", err)
	}
	sorted = slices.Clone(sorted)
	slices.SortFunc(sorted, func(lhs, rhs *Frontmatter) int {
		return strings.Compare(lhs.Name, rhs.Name)
	})

	skills := make([]InMemorySkill, 0, len(sorted))
	for i, frontmatter := range sorted {
		// The frontmatters are sorted by name, so a repeated name follows
		// its first use.
		if i > 0 && sorted[i-1].Name == frontmatter.Name {
			return fmt.Errorf("%w: %q", ErrDuplicateSkill, frontmatter.Name)
		}
		skills = append(skills, InMemorySkill{Frontmatter: frontmatter})
	}

	s.preload.Store(newInMemorySource(skills))
	return nil
}
