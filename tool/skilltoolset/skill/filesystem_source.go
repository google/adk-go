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
//		 SKILL.md
//		 assets/
//	skill-2/
//		 SKILL.md
//		 references/
//		 scripts/
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

// resolveWithinSkill follows every symlink encountered while walking p (a
// path within the source's filesystem, e.g. "evilskill/assets/link") and
// returns the fully resolved, symlink-free path, or an error if resolution
// fails or the result lands outside the skill directory named by name.
//
// Unlike a check that refuses any symlink component outright, this allows a
// skill to ship a symlink that stays inside its own directory -- for
// example assets/shared.md -> ../references/shared.md, so a skill can share
// one file between its resource directories -- while still refusing one
// that points at a sibling skill or anywhere else on disk, such as
// assets/link -> /home/user/.ssh or assets/link -> ../../other-skill.
// A symlink whose target is itself an absolute path (a leading "/", or on
// Windows a drive letter or UNC prefix) is always rejected outright, since
// such a target cannot be reinterpreted as a path inside this filesystem.
//
// This requires the underlying filesystem to implement fs.ReadLinkFS with
// an Lstat that reports a symlink without following it. A filesystem whose
// Lstat falls back to the ordinary, link-following Stat -- which includes
// one that wraps a ReadLinkFS-capable filesystem behind a plain fs.FS and
// so hides the interface, not only one that predates it -- has no way to
// report a symlink, so its symlinks are followed without being checked.
//
// Two gaps are not closed here. On Windows, a directory junction is
// reported by Go's os package as fs.ModeIrregular rather than
// fs.ModeSymlink and so passes through this check unexamined; closing that
// needs Windows-specific reparse-point handling this change does not add.
// And this check and the Open that follows it are two separate filesystem
// operations, so a process that can write into the skill's own directory
// while it is being served could, in principle, swap a checked path for a
// symlink in the gap between them; closing that would need the two to
// happen as one atomic operation, which a generic fs.FS does not offer.
func (f *fileSystemSource) resolveWithinSkill(name, p string) (string, error) {
	rlfs, ok := f.filesystem.(fs.ReadLinkFS)
	if !ok {
		return path.Clean(p), nil
	}

	const maxLinks = 40 // matches the ELOOP convention most OSes use.
	links := 0

	pending := strings.Split(path.Clean(p), "/")
	var resolved []string

	for len(pending) > 0 {
		seg := pending[0]
		pending = pending[1:]

		switch seg {
		case "", ".":
			continue
		case "..":
			if len(resolved) == 0 {
				return "", fmt.Errorf("%w: %q escapes the filesystem root", ErrInvalidResourcePath, p)
			}
			resolved = resolved[:len(resolved)-1]
			continue
		}

		resolved = append(resolved, seg)
		cur := strings.Join(resolved, "/")

		info, err := rlfs.Lstat(cur)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return "", fmt.Errorf("%w: %q", ErrResourceNotFound, strings.TrimPrefix(p, name+"/"))
			}
			return "", fmt.Errorf("lstat %q: %w", cur, err)
		}
		if info.Mode()&fs.ModeSymlink == 0 {
			continue // Ordinary file or directory component; keep walking.
		}

		links++
		if links > maxLinks {
			return "", fmt.Errorf("%w: %q: too many levels of symbolic links", ErrInvalidResourcePath, p)
		}

		target, err := rlfs.ReadLink(cur)
		if err != nil {
			return "", fmt.Errorf("readlink %q: %w", cur, err)
		}
		// A target read back from the filesystem may use OS-native
		// separators -- a Windows symlink created via filepath.Join, for
		// instance, stores a target like "..\references\shared.md". fs.FS
		// paths are always slash-separated (io/fs's own convention), so
		// normalize before treating the target as one; this also turns a
		// UNC target such as "\\host\share\x" into "//host/share/x",
		// which path.IsAbs below already recognizes as absolute.
		target = strings.ReplaceAll(target, `\`, "/")
		if path.IsAbs(target) || isWindowsDriveAbsolutePath(target) {
			return "", fmt.Errorf("%w: %q: %q is an absolute symlink target", ErrInvalidResourcePath, p, cur)
		}

		// Drop the symlink itself, splice its target's components in
		// (resolved relative to the symlink's own directory, which is
		// what every POSIX symlink target is relative to) ahead of
		// whatever of the original path still remains to process.
		resolved = resolved[:len(resolved)-1]
		pending = append(strings.Split(path.Clean(target), "/"), pending...)
	}

	resolvedPath := strings.Join(resolved, "/")
	if resolvedPath != name && !strings.HasPrefix(resolvedPath, name+"/") {
		return "", fmt.Errorf("%w: %q resolves to %q, outside skill %q's own directory", ErrInvalidResourcePath, p, resolvedPath, name)
	}
	return resolvedPath, nil
}

// isWindowsDriveAbsolutePath reports whether target -- already normalized
// to forward slashes -- starts with a Windows drive letter, such as
// "C:/Users", which path.IsAbs does not recognize since it only knows the
// slash-separated paths fs.FS uses.
func isWindowsDriveAbsolutePath(target string) bool {
	if len(target) < 2 || target[1] != ':' {
		return false
	}
	c := target[0]
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// LoadResource reads a specific file from the skill's directory.
//
// For security, the resourcePath is sanitized using path.Clean, access is
// strictly limited to files within the 'references/', 'assets/', or
// 'scripts/' subdirectories, and -- on a filesystem that supports symlinks
// -- every symlink the resolved path passes through is followed and
// checked rather than trusted, so a resource may still be reached through
// a symlink that stays inside the skill's own directory, but not through
// one that resolves anywhere else. See resolveWithinSkill's own doc for
// what this does and does not protect against.
func (f *fileSystemSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if err := f.validateSkill(name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourcePath)
	if !strings.HasPrefix(cleanPath, "references/") && !strings.HasPrefix(cleanPath, "assets/") && !strings.HasPrefix(cleanPath, "scripts/") {
		return nil, fmt.Errorf("%w: %q must be within 'references/', 'assets/', or 'scripts/' (relative to skill directory)", ErrInvalidResourcePath, resourcePath)
	}

	fullPath := path.Join(name, cleanPath)
	resolvedPath, err := f.resolveWithinSkill(name, fullPath)
	if err != nil {
		return nil, err
	}

	file, err := f.filesystem.Open(resolvedPath)
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
// Note: fs.WalkDir skips a symlink it finds inside a directory without
// following it, but it does follow one if the walk's own root is a
// symlink, and the fs.Stat this method runs before the walk follows links
// too. So if resourceDirectoryPath (or, when it is empty, assets/,
// references/, or scripts/ itself) is a symlink, this method walks and
// lists whatever it points at, by name, the same way LoadResource used to
// before this change. That behavior is unchanged by this change and is not
// this method's to fix here -- see the symlink-escape issue's follow-up
// notes -- but it means this method is not a safe source of truth for
// which paths stay inside the skill the way LoadResource now is.
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
