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

package vertexai

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"

	skills "google.golang.org/adk/tool/skilltoolset/skill"
)

type vertexAIRegistrySource struct {
	client *vertexAIClient
	skills []string
}

// RegistryConfig configures the SkillsRegistry.
type RegistryConfig struct {
	Project  string
	Location string
	Skills   []string // Optionally makes explicitly named registry skills available without need for search.
}

// NewRegistrySource creates a new source based on the VertexAI SkillsRegistry.
//
// Each skill in SkillsRegistry is stored as a remote zipped file system containing a "SKILL.md" file
// with valid YAML frontmatter and supported directories.

// Expected zip file system layout example:
//
// SKILL.md
// references/
// scripts/
func NewRegistrySource(ctx context.Context, config RegistryConfig) (skills.Source, error) {
	client, err := newVertexAIClient(ctx, &vertexAIClientConfig{
		ProjectID: config.Project,
		Location:  config.Location,
	})
	if err != nil {
		return nil, fmt.Errorf("newVertexAIClient failed: %w", err)
	}
	return &vertexAIRegistrySource{
		client: client,
		skills: config.Skills,
	}, nil
}

var _ skills.Source = &vertexAIRegistrySource{}

// ListFrontmatters retrieves named skill frontmatters (if applicable) from the registry.
//
// Any named skill not found will result in an error.
// As the skills registry may contain a large number of available skills, skills may instead be fetched dynamically via the 'Search' method.
func (s *vertexAIRegistrySource) ListFrontmatters(ctx context.Context) ([]*skills.Frontmatter, error) {
	if len(s.skills) == 0 {
		return nil, nil
	}
	var frontmatters []*skills.Frontmatter
	for _, skillName := range s.skills {
		frontmatter, err := s.LoadFrontmatter(ctx, skillName)
		if err != nil {
			return nil, err
		}
		frontmatters = append(frontmatters, frontmatter)
	}
	return frontmatters, nil
}

// LoadFrontmatter fetches and parses the SKILL.md file.
func (s *vertexAIRegistrySource) LoadFrontmatter(ctx context.Context, name string) (*skills.Frontmatter, error) {
	zipBytes, err := s.fetchAndDecodeZip(ctx, name)
	if err != nil {
		return nil, err
	}
	frontmatter, _, closer, err := s.readSkill(name, zipBytes)
	if err != nil {
		return nil, err
	}
	_ = closer.Close()
	return frontmatter, nil
}

// LoadInstructions fetches the SKILL.md file and returns the markdown content
// immediately following the frontmatter delimiter.
func (s *vertexAIRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	zipBytes, err := s.fetchAndDecodeZip(ctx, name)
	if err != nil {
		return "", err
	}
	_, reader, closer, err := s.readSkill(name, zipBytes)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = closer.Close() // Ignore error as read success is what matters.
	}()

	instructions, err := io.ReadAll(reader)
	if err != nil {
		return "", fmt.Errorf("read instructions: %w", err)
	}
	return string(instructions), nil
}

// LoadResource reads a specific file from the zipped file system.
//
// Access is strictly limited to files within the 'references/', 'assets/', or 'scripts/' subdirectories.
func (s *vertexAIRegistrySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	zipBytes, err := s.fetchAndDecodeZip(ctx, name)
	if err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourcePath)
	if !strings.HasPrefix(cleanPath, "references/") && !strings.HasPrefix(cleanPath, "assets/") && !strings.HasPrefix(cleanPath, "scripts/") {
		return nil, fmt.Errorf("%w: %q must be within 'references/', 'assets/', or 'scripts/' (relative to skill directory)", skills.ErrInvalidResourcePath, resourcePath)
	}

	zipReader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to open in-memory zip archive: %w", err)
	}

	file, err := zipReader.Open(cleanPath)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %q", skills.ErrResourceNotFound, cleanPath)
		}
		return nil, fmt.Errorf("open resource file %q: %w", cleanPath, err)
	}
	return file, nil
}

// ListResources walks the specified zipped file system directory of the skill.
//
// If resourceDirectoryPath is empty or ".", it walks the 'references/',
// 'assets/', and 'scripts/' directories. It restricts traversal to these
// approved directories and returns sanitized paths relative to the skill root.
func (s *vertexAIRegistrySource) ListResources(ctx context.Context, name, resourceDirectoryPath string) ([]string, error) {
	zipBytes, err := s.fetchAndDecodeZip(ctx, name)
	if err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourceDirectoryPath)
	isRoot := cleanPath == "." || cleanPath == ""

	if !isRoot {
		switch strings.SplitN(cleanPath, "/", 2)[0] {
		case "references", "assets", "scripts": // Valid top level directories.
		default:
			return nil, fmt.Errorf("%w: %q must be empty, root (.), or within 'references/', 'assets/', or 'scripts/'", skills.ErrInvalidResourcePath, resourceDirectoryPath)
		}
	}

	zipReader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to open in-memory zip archive: %w", err)
	}

	targets := []string{cleanPath}
	if isRoot { // Limit the walk to these top-level directories.
		targets = []string{"references", "assets", "scripts"}
	}

	var resources []string
	for _, target := range targets {
		err := fs.WalkDir(zipReader, target, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					return nil
				}
				return err
			}
			if !d.IsDir() {
				resources = append(resources, p)
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("walk target %q: %w", target, err)
		}
	}
	return resources, nil
}

// Search performs a semantic and key word query search of the registry and returns the partial frontmatters of matched skills.
func (s *vertexAIRegistrySource) Search(ctx context.Context, query string) ([]*skills.Frontmatter, error) {
	return s.searchSkills(ctx, query)
}

// readSkill reads and validates the frontmatter from the SKILL.md file and
// returns the frontmatter, a buffered reader for the rest of the file, and a
// closer for the file.
func (s *vertexAIRegistrySource) readSkill(name string, zipBytes []byte) (*skills.Frontmatter, *bufio.Reader, io.Closer, error) {
	zipReader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to open zip: %w", err)
	}
	var targetFile *zip.File
	for _, file := range zipReader.File {
		if slices.Contains([]string{"SKILL.md", "skill.md"}, file.Name) {
			targetFile = file
			break
		}
	}
	if targetFile == nil {
		return nil, nil, nil, fmt.Errorf("neither SKILL.md or skill.md found in zip archive")
	}
	rc, err := targetFile.Open()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open fro zip failed: %v", err)
	}
	reader := bufio.NewReader(rc)
	frontmatter, err := skills.Parse(reader)
	if err != nil {
		_ = rc.Close() // Clean up on failure
		return nil, nil, nil, fmt.Errorf("%w: parse frontmatter: %w", skills.ErrInvalidFrontmatter, err)
	}
	if frontmatter.Name != name {
		_ = rc.Close() // Clean up on failure
		return nil, nil, nil, fmt.Errorf("%w: name in SKILL.md (%q) does not match directory name (%q)", skills.ErrInvalidSkillName, frontmatter.Name, name)
	}
	return frontmatter, reader, rc, nil
}

// searchSkill searches the registry for skills which match the query and returns partial frontmatters.
func (s *vertexAIRegistrySource) searchSkills(ctx context.Context, query string) ([]*skills.Frontmatter, error) {
	res, err := s.client.searchSkills(ctx, query)
	if err != nil {
		return nil, err
	}
	var results []*skills.Frontmatter
	for _, result := range res.Skills {
		parts := strings.Split(result.Name, "/")
		results = append(results, &skills.Frontmatter{
			Name:        parts[len(parts)-1],
			Description: result.Description,
		})
	}
	return results, nil
}

// fetchAndDecodeZip fetches and decodes the skill's zipped file contents.
func (s *vertexAIRegistrySource) fetchAndDecodeZip(ctx context.Context, name string) ([]byte, error) {
	skill, err := s.client.getSkill(ctx, name)
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(skill.ZippedFilesystem)
}
