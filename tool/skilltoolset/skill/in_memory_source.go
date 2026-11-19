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
	"maps"
	"path"
	"slices"
	"strings"
)

// InMemorySkill is one skill of a Source created by [NewInMemorySource].
type InMemorySkill struct {
	// Frontmatter is the skill's metadata. Required. Its Name identifies the
	// skill.
	Frontmatter *Frontmatter

	// Instructions is the skill's instruction body. LoadInstructions returns
	// it as given.
	Instructions string

	// Resources holds the content of each resource by its path relative to
	// the skill, such as "references/guide.md". A path must be within
	// "references/", "assets/" or "scripts/", the directories a skill keeps
	// its resources in, and must be written in its shortest form: with no
	// "." or ".." elements, repeated slashes or trailing slash.
	Resources map[string][]byte
}

// NewInMemorySource creates a Source that serves the given skills from
// memory. Its ListFrontmatters returns them in the order given.
//
// It returns an error that wraps
//   - ErrInvalidFrontmatter for a skill whose Frontmatter is nil or does not
//     pass [Validate];
//   - ErrDuplicateSkill for two skills of the same name;
//   - ErrInvalidResourcePath for a resource path that InMemorySkill.Resources
//     does not allow.
//
// The Source keeps the frontmatters and the resource contents it is given, so
// they must not be modified afterwards.
func NewInMemorySource(skills ...InMemorySkill) (Source, error) {
	// The caller may pass a slice of its own, so set the copied maps on a
	// copy of it.
	skills = slices.Clone(skills)
	names := make(map[string]bool, len(skills))
	for i, s := range skills {
		if err := Validate(s.Frontmatter); err != nil {
			return nil, fmt.Errorf("%w: skills[%d]: %w", ErrInvalidFrontmatter, i, err)
		}
		name := s.Frontmatter.Name
		if names[name] {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateSkill, name)
		}
		names[name] = true
		for resourcePath := range s.Resources {
			inResourceDirectory := strings.HasPrefix(resourcePath, "references/") || strings.HasPrefix(resourcePath, "assets/") || strings.HasPrefix(resourcePath, "scripts/")
			if !inResourceDirectory || resourcePath != path.Clean(resourcePath) {
				return nil, fmt.Errorf("%w: %q in skill %q must be a path within 'references/', 'assets/', or 'scripts/' in its shortest form", ErrInvalidResourcePath, resourcePath, name)
			}
		}
		skills[i].Resources = maps.Clone(s.Resources)
	}
	return newInMemorySource(skills), nil
}

// inMemorySource serves skills from memory. It never changes after it is
// created, so it needs no locking.
type inMemorySource struct {
	frontmatters []*Frontmatter
	skills       map[string]*inMemorySkillData
}

type inMemorySkillData struct {
	InMemorySkill
	// sortedResourcePaths enables fast filtered listing of resources in
	// deterministic order.
	sortedResourcePaths []string
}

// newInMemorySource creates a source of skills without checking them. It
// keeps the skills' maps. The names of the skills must be unique and the
// resource paths clean.
func newInMemorySource(skills []InMemorySkill) *inMemorySource {
	source := &inMemorySource{skills: make(map[string]*inMemorySkillData, len(skills))}
	for _, s := range skills {
		source.frontmatters = append(source.frontmatters, s.Frontmatter)
		source.skills[s.Frontmatter.Name] = &inMemorySkillData{
			InMemorySkill:       s,
			sortedResourcePaths: slices.Sorted(maps.Keys(s.Resources)),
		}
	}
	return source
}

func (s *inMemorySource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	// A copy, so that a caller that reorders the result does not reorder
	// the source's skills.
	return slices.Clone(s.frontmatters), nil
}

func (s *inMemorySource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	data, ok := s.skills[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}
	return data.Frontmatter, nil
}

func (s *inMemorySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	data, ok := s.skills[name]
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}
	return data.Instructions, nil
}

func (s *inMemorySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	data, ok := s.skills[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}
	cleanPath := path.Clean(resourcePath)
	resource, ok := data.Resources[cleanPath]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrResourceNotFound, cleanPath)
	}
	return io.NopCloser(bytes.NewReader(resource)), nil
}

// ListResources returns the skill's resource paths under subpath, sorted.
// "." and "/" list every resource. A subpath that names a resource lists that
// resource.
//
// The source holds resources, not directories, so it cannot tell a directory
// that is empty from one that does not exist: a subpath with no resource
// under it lists nothing, and is not an error.
func (s *inMemorySource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	data, ok := s.skills[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
	}

	cleanPath := path.Clean(subpath)
	if cleanPath == "." || cleanPath == "/" {
		// A copy, so that a caller that reorders the result does not
		// break the binary search below.
		return slices.Clone(data.sortedResourcePaths), nil
	}

	var result []string
	if !strings.HasSuffix(subpath, "/") {
		if _, matchesSpecificResource := data.Resources[cleanPath]; matchesSpecificResource {
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
