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
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"
)

// NewFileSystemSource creates a Source implementation backed by an fs.FS.
//
// The provided filesystem is expected to have skills organized as immediate
// subdirectories. A valid skill directory MUST contain a "SKILL.md" file
// with valid YAML frontmatter, and the directory name must exactly match
// the name defined in that frontmatter.
//
// Expected layout example:
//
//	  skill-1/
//		   SKILL.md
//		   assets/
//	  skill-2/
//		   SKILL.md
//		   references/
//		   scripts/
func NewFileSystemSource(filesystem fs.FS) Source {
	return &fileSystemSource{filesystem: filesystem}
}

type fileSystemSource struct {
	filesystem fs.FS
}

// ListFrontmatters scans the immediate subdirectories of the root filesystem.
// It does not traverse recursively. Directories without a valid SKILL.md
// are silently ignored.
func (f *fileSystemSource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	var frontmatters []*Frontmatter

	entries, err := fs.ReadDir(f.filesystem, ".")
	if err != nil {
		return nil, fmt.Errorf("read root directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue // Skills must be directories
		}
		frontmatter, _, closer, err := f.readSkill(entry.Name())
		if err != nil {
			if errors.Is(err, ErrSkillNotFound) {
				continue // Directory doesn't contain SKILL.md - not a skill.
			}
			return nil, err
		}
		// Avoid holding files open in the loop by closing without defer.
		_ = closer.Close() // Ignore error as read success is what matters.
		frontmatters = append(frontmatters, frontmatter)
	}
	return frontmatters, nil
}

// LoadFrontmatter opens and parses the SKILL.md file located at the root
// of the specified skill directory.
func (f *fileSystemSource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	frontmatter, _, closer, err := f.readSkill(name)
	if err != nil {
		return nil, err
	}
	_ = closer.Close() // Ignore error as read success is what matters.
	return frontmatter, nil
}

// LoadInstructions parses the SKILL.md file and returns the markdown content
// immediately following the frontmatter delimiter.
func (f *fileSystemSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	_, reader, closer, err := f.readSkill(name)
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

// LoadResource reads a specific file from the skill's directory.
//
// For security, the resourcePath is sanitized using path.Clean. Access is
// strictly limited to files within the 'references/', 'assets/', or
// 'scripts/' subdirectories to prevent path traversal attacks.
func (f *fileSystemSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if err := f.validateSkill(name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourcePath)
	if !strings.HasPrefix(cleanPath, "references/") && !strings.HasPrefix(cleanPath, "assets/") && !strings.HasPrefix(cleanPath, "scripts/") {
		return nil, fmt.Errorf("%w: %q must be within 'references/', 'assets/', or 'scripts/' (relative to skill directory)", ErrInvalidResourcePath, resourcePath)
	}

	fullPath := path.Join(name, cleanPath)
	file, err := f.filesystem.Open(fullPath)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %q", ErrResourceNotFound, cleanPath)
		}
		return nil, fmt.Errorf("open resource file %q: %w", fullPath, err)
	}
	return file, nil
}

// ListResources walks the specified resource directory within the skill.
//
// If resourceDirectoryPath is empty or ".", it walks the 'references/',
// 'assets/', and 'scripts/' directories. It restricts traversal to these
// approved directories and returns sanitized paths relative to the skill root.
func (f *fileSystemSource) ListResources(ctx context.Context, name, resourceDirectoryPath string) ([]string, error) {
	if err := f.validateSkill(name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourceDirectoryPath)
	isRoot := cleanPath == "." || cleanPath == ""

	if !isRoot {
		switch strings.SplitN(cleanPath, "/", 2)[0] {
		case "references", "assets", "scripts": // Valid top level directories.
		default:
			return nil, fmt.Errorf("%w: %q must be empty, root (.), or within 'references/', 'assets/', or 'scripts/'", ErrInvalidResourcePath, resourceDirectoryPath)
		}
	}

	skillFS, err := fs.Sub(f.filesystem, name)
	if err != nil {
		return nil, fmt.Errorf("create sub-filesystem for %q: %w", name, err)
	}

	if _, err := fs.Stat(skillFS, cleanPath); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %q", ErrResourceNotFound, cleanPath)
		}
		return nil, fmt.Errorf("stat %q: %w", cleanPath, err)
	}

	targets := []string{cleanPath}
	if isRoot { // Limit the walk to these top-level directories.
		targets = []string{"references", "assets", "scripts"}
	}

	var resources []string
	for _, target := range targets {
		err := fs.WalkDir(skillFS, target, func(p string, d fs.DirEntry, err error) error {
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

func (f *fileSystemSource) validateSkill(name string) error {
	_, _, closer, err := f.readSkill(name)
	if err != nil {
		return err
	}
	_ = closer.Close() // Ignore error as read success is what matters.
	return nil
}

// readSkill reads and validates the frontmatter from the SKILL.md file and
// returns the frontmatter, a buffered reader for the rest of the file, and a
// closer for the file.
func (f *fileSystemSource) readSkill(name string) (*Frontmatter, *bufio.Reader, io.Closer, error) {
	skillFilePath := path.Join(name, "SKILL.md")
	file, err := f.filesystem.Open(skillFilePath)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil, nil, fmt.Errorf("%w: %q", ErrSkillNotFound, name)
		}
		return nil, nil, nil, fmt.Errorf("open %q: %w", skillFilePath, err)
	}
	reader := bufio.NewReader(file)
	frontmatter, err := Parse(reader)
	if err != nil {
		_ = file.Close()
		return nil, nil, nil, fmt.Errorf("%w: parse frontmatter: %w", ErrInvalidFrontmatter, err)
	}
	if frontmatter.Name != name {
		_ = file.Close()
		return nil, nil, nil, fmt.Errorf("%w: name in SKILL.md (%q) does not match directory name (%q)", ErrInvalidSkillName, frontmatter.Name, name)
	}
	return frontmatter, reader, file, nil
}
