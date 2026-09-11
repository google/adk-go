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
	"fmt"
	"io"
	"path"
	"slices"
	"strings"
	"sync"
)

const maxResourceSize = 10 * 1024 * 1024 // 10MB

type completePreloadSkillData struct {
	frontmatter  *Frontmatter
	instructions string
	// resources enables fast lookup.
	resources map[string][]byte
	// sortedResourcePaths enables fast filtered listing of resources in
	// deterministic order.
	sortedResourcePaths []string
}

type completePreloadSource struct {
	base   Source
	mu     sync.RWMutex
	skills map[string]*completePreloadSkillData
	// sortedFrontmatters enables listing of frontmatters in deterministic order.
	sortedFrontmatters []*Frontmatter
}

// WithCompletePreloadSource returns a Source proxy that wraps the given source
// and fully preloads all skill data (Frontmatters, Instructions, and Resources)
// into memory upon creation.
// This offers the fastest access speed after initialization, at the expense of
// higher initial load time and memory usage.
// Additionally, 'reload' callback is returned: calling it actualizes data by
// loading all the data anew.
//
// NOTE: source.ListResources must list all resources for subpath ".".
func WithCompletePreloadSource(ctx context.Context, source Source) (Source, func(context.Context) error, error) {
	s := &completePreloadSource{base: source}
	if err := s.reload(ctx); err != nil {
		return nil, nil, err
	}
	return s, s.reload, nil
}

func (s *completePreloadSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sortedFrontmatters, nil
}

func (s *completePreloadSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.skills[name]
	if !ok {
		return nil, ErrSkillNotFound
	}
	return data.frontmatter, nil
}

func (s *completePreloadSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.skills[name]
	if !ok {
		return "", ErrSkillNotFound
	}
	return data.instructions, nil
}

func (s *completePreloadSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	skill, ok := s.skills[name]
	if !ok {
		return nil, ErrSkillNotFound
	}
	resource, ok := skill.resources[path.Clean(resourcePath)]
	if !ok {
		return nil, ErrResourceNotFound
	}
	return io.NopCloser(bytes.NewReader(resource)), nil
}

func (s *completePreloadSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.skills[name]
	if !ok {
		return nil, ErrSkillNotFound
	}

	cleanPath := path.Clean(subpath)
	if cleanPath == "." || cleanPath == "/" {
		return data.sortedResourcePaths, nil
	}

	var result []string
	if !strings.HasSuffix(subpath, "/") {
		if _, matchesSpecificResource := data.resources[cleanPath]; matchesSpecificResource {
			result = append(result, cleanPath)
		}
	}
	// Optimization for skills with many resources: binary search to find the
	// first resource path that could match the prefix.
	prefix := cleanPath + "/" // clean path is guaranteed to not end with "/".
	startIdx, _ := slices.BinarySearch(data.sortedResourcePaths, prefix)
	for i := startIdx; i < len(data.sortedResourcePaths); i++ {
		resourcePath := data.sortedResourcePaths[i]
		if !strings.HasPrefix(resourcePath, prefix) {
			break // Nothing else will match since resourcePaths is sorted.
		}
		result = append(result, resourcePath)
	}
	return result, nil
}

func (s *completePreloadSource) reload(ctx context.Context) error {
	frontmatters, err := s.base.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("list frontmatters: %w", err)
	}
	frontmatters = slices.Clone(frontmatters)
	slices.SortFunc(frontmatters, func(a, b *Frontmatter) int {
		return strings.Compare(a.Name, b.Name)
	})

	skills := make(map[string]*completePreloadSkillData, len(frontmatters))
	for _, frontmatter := range frontmatters {
		if err := ctx.Err(); err != nil {
			return err
		}

		if _, exists := skills[frontmatter.Name]; exists {
			return fmt.Errorf("%w: %q", ErrDuplicateSkill, frontmatter.Name)
		}

		instructions, err := s.base.LoadInstructions(ctx, frontmatter.Name)
		if err != nil {
			return fmt.Errorf("load instructions for skill %q: %w", frontmatter.Name, err)
		}

		resourcePaths, err := s.base.ListResources(ctx, frontmatter.Name, ".")
		if err != nil {
			return fmt.Errorf("list resources for skill %q: %w", frontmatter.Name, err)
		}

		var cleanResourcePaths []string
		resources := make(map[string][]byte)
		for _, resourcePath := range resourcePaths {
			if err := ctx.Err(); err != nil {
				return err
			}
			resource, err := s.base.LoadResource(ctx, frontmatter.Name, resourcePath)
			if err != nil {
				return fmt.Errorf("resource path %q in skill %q: %w", resourcePath, frontmatter.Name, err)
			}
			data, err := io.ReadAll(io.LimitReader(resource, maxResourceSize+1))
			_ = resource.Close()
			if err != nil {
				return fmt.Errorf("read resource path %q in skill %q: %w", resourcePath, frontmatter.Name, err)
			}
			if len(data) > maxResourceSize {
				return fmt.Errorf("resource %q in skill %q exceeds %d bytes limit", resourcePath, frontmatter.Name, maxResourceSize)
			}
			cleanPath := path.Clean(resourcePath)
			resources[cleanPath] = data
			cleanResourcePaths = append(cleanResourcePaths, cleanPath)
		}
		slices.Sort(cleanResourcePaths)

		skills[frontmatter.Name] = &completePreloadSkillData{
			frontmatter:         frontmatter,
			instructions:        instructions,
			sortedResourcePaths: cleanResourcePaths,
			resources:           resources,
		}
	}

	s.mu.Lock()
	s.skills = skills
	s.sortedFrontmatters = frontmatters
	s.mu.Unlock()
	return nil
}
