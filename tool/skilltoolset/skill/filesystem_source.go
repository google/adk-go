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
//	skill-1/
//	         SKILL.md
//	         assets/
//	skill-2/
//	         SKILL.md
//	         references/
//	         scripts/
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

// rejectSymlinkPath walks every path component of p, from the filesystem
// root down, and refuses the whole path if any component is a symbolic
// link.
//
// Checking each component -- rather than only the final target -- closes
// the gap path.Clean leaves open: a resourcePath that is syntactically
// inside references/, assets/, or scripts/ can still resolve, through a
// symlink shipped inside the skill itself, to a file outside the skill's
// own directory. fs.Open ultimately reaches the operating system's open(2),
// which dereferences symlinks transparently, so the string-level prefix
// check alone never sees the real target.
//
// This requires the underlying filesystem to implement fs.ReadLinkFS
// (added in Go 1.23; os.DirFS satisfies it). A filesystem that does not --
// such as the in-memory filesystems common in tests -- has no notion of
// symlinks at all, so it cannot contain one to escape through; this follows
// the same convention as fs.Lstat itself, which falls back to the ordinary,
// symlink-following Stat when ReadLinkFS is unavailable.
func (f *fileSystemSource) rejectSymlinkPath(p string) error {
	rlfs, ok := f.filesystem.(fs.ReadLinkFS)
	if !ok {
		return nil
	}

	clean := path.Clean(p)
	if clean == "." {
		return nil
	}

	var cur string
	for _, part := range strings.Split(clean, "/") {
		if cur == "" {
			cur = part
		} else {
			cur = cur + "/" + part
		}
		info, err := rlfs.Lstat(cur)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return fmt.Errorf("%w: %q", ErrResourceNotFound, p)
			}
			return fmt.Errorf("lstat %q: %w", cur, err)
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return fmt.Errorf("%w: %q contains a symbolic link (%q); a skill resource path must not pass through a symlink", ErrInvalidResourcePath, p, cur)
		}
	}
	return nil
}

// LoadResource reads a specific file from the skill's directory.
//
// For security, the resourcePath is sanitized using path.Clean, access is
// strictly limited to files within the 'references/', 'assets/', or
// 'scripts/' subdirectories, and -- on a filesystem that supports symlinks
// -- every path component the resolved path walks through is required to
// be a real directory or file rather than a symbolic link, which together
// prevent path traversal and symlink-escape attacks.
func (f *fileSystemSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if err := f.validateSkill(name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourcePath)
	if !strings.HasPrefix(cleanPath, "references/") && !strings.HasPrefix(cleanPath, "assets/") && !strings.HasPrefix(cleanPath, "scripts/") {
		return nil, fmt.Errorf("%w: %q must be within 'references/', 'assets/', or 'scripts/' (relative to skill directory)", ErrInvalidResourcePath, resourcePath)
	}

	fullPath := path.Join(name, cleanPath)
	if err := f.rejectSymlinkPath(fullPath); err != nil {
		return nil, err
	}

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
//
// Note: fs.WalkDir lists a symlinked entry by name but does not follow it
// to walk its target's contents, so this method does not leak file content
// across the same symlink LoadResource is hardened against above; it can
// only reveal that a symlink with a given name exists. LoadResource is the
// path that actually opens file content and is the one this change fixes.
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
