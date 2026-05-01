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
	"sync"
)

type frontmatterPreloadSource struct {
	Source
	mu           sync.RWMutex
	frontmatters map[string]*Frontmatter // for fast lookup.
	sorted       []*Frontmatter          // for deterministic listing.
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
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sorted, nil
}

func (s *frontmatterPreloadSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	frontmatter, exists := s.frontmatters[name]
	if !exists {
		return nil, ErrSkillNotFound
	}
	return frontmatter, nil
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

	frontmatters := make(map[string]*Frontmatter, len(sorted))
	for _, frontmatter := range sorted {
		if _, exists := frontmatters[frontmatter.Name]; exists {
			return fmt.Errorf("%w: %q", ErrDuplicateSkill, frontmatter.Name)
		}
		frontmatters[frontmatter.Name] = frontmatter
	}

	s.mu.Lock()
	s.frontmatters = frontmatters
	s.sorted = sorted
	s.mu.Unlock()
	return nil
}
