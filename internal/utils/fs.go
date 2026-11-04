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

package utils

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"syscall"
)

var (
	// ErrNotValidRelativePath means that the reference, read as a string, is not
	// a relative path below the base directory.
	ErrNotValidRelativePath = errors.New("path is not a relative path inside the base directory")
	// ErrSymlinkInRelativePath means that a link on the way leads outside the
	// base directory. A link that stays inside is not an error.
	ErrSymlinkInRelativePath = errors.New("path leads outside the base directory through a link")
)

// SafeSubpath resolves rel below base and checks that the result stays inside it.
//
// The base path must be absolute and is trusted. rel is not. On success the
// result is the absolute path the check followed rel to, every link on the way
// replaced by its target, and the base-relative spelling of rel. A caller that
// opens the returned path opens the file the check judged; opening another
// spelling of it, or resolving it again by other rules, is what lets a check
// and a read disagree.
//
// Returns ErrNotValidRelativePath iff rel is not a local path below base:
// absolute, empty, escaping through "..", effectively empty, or (on Windows)
// carrying a drive, a UNC prefix, a reserved device name or a trailing dot or
// space.
// Returns ErrSymlinkInRelativePath iff following the links on the way (symlinks,
// and on Windows junctions and other reparse points that name a target) lands
// outside base, itself followed the same way. A link that lands inside base is
// accepted, which keeps Kubernetes ConfigMap and Secret mounts (a "..data" link
// and per-key links) working, as in adk-python. A link whose target does not
// exist is judged by where it points: outside is refused, inside is returned
// so the caller's read reports the missing file.
// A component that does not exist, or that sits below a non-directory, is not
// an error here: nothing below it can exist, so the path is returned and the
// caller's read reports the real problem.
func SafeSubpath(base, rel string) (absPath, relPath string, resErr error) {
	if !filepath.IsAbs(base) {
		return "", "", fmt.Errorf("base path is not absolute")
	}
	absBase := filepath.Clean(base)

	if rel == "" {
		return "", "", fmt.Errorf("%w: rel path is empty", ErrNotValidRelativePath)
	}
	// NUL char is not allowed both on linux and windows
	if strings.Contains(rel, "\x00") {
		return "", "", fmt.Errorf("%w: rel path contains null byte", ErrNotValidRelativePath)
	}
	// IsLocal settles absolute paths and any ".." that escapes. On Windows it also
	// rejects drive-relative and UNC spellings and the reserved device names,
	// matching them case-insensitively and with trailing spaces and dots.
	if !filepath.IsLocal(rel) {
		return "", "", fmt.Errorf("%w: rel path is not local", ErrNotValidRelativePath)
	}
	// On Windows IsLocal already refuses any ':' (a stream or a drive) and the
	// leading separators of a UNC path. What it leaves to this block is the
	// other characters a file name may not contain, and a trailing " " or "."
	// on any component, the last included: IsLocal refuses that trailing dot
	// or space only when the name is a reserved device name such as NUL or CON.
	if runtime.GOOS == "windows" {
		if strings.ContainsAny(rel, windowsForbiddenStandardChars) {
			return "", "", fmt.Errorf("%w: rel path contains windows-specific invalid chars: %v", ErrNotValidRelativePath, windowsForbiddenStandardChars)
		}
		if strings.ContainsAny(rel, windowsForbiddenLowChar) {
			return "", "", fmt.Errorf("%w: rel path contains windows-specific invalid chars 1-31", ErrNotValidRelativePath)
		}
		for _, f := range splitPathToFragments(rel) {
			if strings.HasSuffix(f, " ") {
				return "", "", fmt.Errorf("%w: rel path contains trailing space", ErrNotValidRelativePath)
			}
			if strings.HasSuffix(f, ".") && f != "." && f != ".." {
				return "", "", fmt.Errorf("%w: rel path contains trailing dot", ErrNotValidRelativePath)
			}
		}
	}

	// check if rel stays below absBase. Rel + IsLocal, not a string prefix: the
	// prefix absBase+separator is wrong for a filesystem root ("//").
	res := filepath.Join(absBase, rel)
	r, err := filepath.Rel(absBase, res)
	if err != nil {
		return "", "", fmt.Errorf("%w: filepath.Rel failed: %w", ErrNotValidRelativePath, err)
	}
	if r == "." { // rel is effectively empty
		return "", "", fmt.Errorf("%w: rel is effectively empty", ErrNotValidRelativePath)
	}
	if !filepath.IsLocal(r) {
		return "", "", fmt.Errorf("%w: not a subpath", ErrNotValidRelativePath)
	}

	real, err := resolveInside(absBase, res)
	if err != nil {
		return "", "", err
	}
	return real, r, nil
}

// resolveInside follows the links in target and reports where it really lands.
// It returns ErrSymlinkInRelativePath when that is not below base, itself
// followed the same way so a base reached through a link is trusted.
func resolveInside(base, target string) (string, error) {
	realBase, err := followLinks(base)
	if err != nil {
		return "", fmt.Errorf("cannot resolve links in base: %w", err)
	}
	realTarget, err := followLinks(target)
	if err != nil {
		return "", fmt.Errorf("cannot resolve links: %w", err)
	}
	rel, err := filepath.Rel(realBase, realTarget)
	if err != nil || (rel != "." && !filepath.IsLocal(rel)) {
		return "", fmt.Errorf("%w: it resolves to %q", ErrSymlinkInRelativePath, realTarget)
	}
	return realTarget, nil
}

// maxLinkFollows bounds the number of links followed on one path, as
// filepath.EvalSymlinks does, so a link loop is an error rather than a hang.
const maxLinkFollows = 255

// followLinks walks the absolute path p one component at a time and replaces
// every link by its target. It follows links the way filepath.EvalSymlinks
// does: a link's target is spliced into the path uncleaned and then walked
// component by component, so a ".." applies to whatever the walk has reached
// so far. In the target "s/../y.yaml" that is what s points at, not the
// directory the link sits in; cleaning the target as text first would drop
// "s/.." before looking at s and judge a different file. The result is what
// the caller then opens, so the check and the read agree by construction.
//
// Two differences from EvalSymlinks matter here:
//
//   - A link is anything os.Readlink can read. On Windows that includes a
//     directory junction, which since Go 1.23 os.Lstat reports as ModeIrregular
//     and which EvalSymlinks therefore does not follow (it fails with ENOTDIR
//     when a junction sits in the middle of a path).
//   - A link whose target does not exist is still followed, by name, and a
//     component that does not exist, or sits below a non-directory, ends the
//     walk with the remaining components appended and cleaned. Nothing below
//     such a component can exist, so it can hide no link, and the caller's
//     read is what reports the missing file.
func followLinks(p string) (string, error) {
	sep := string(filepath.Separator)
	vol := filepath.VolumeName(p)
	cur := vol + sep
	rest := strings.Split(strings.Trim(p[len(vol):], sep), sep)
	follows := 0
	for i := 0; i < len(rest); i++ {
		c := rest[i]
		switch c {
		case "", ".":
			continue
		case "..":
			cur = filepath.Dir(cur)
			continue
		}
		next := filepath.Join(cur, c)
		fi, err := os.Lstat(next)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
				return filepath.Join(append([]string{next}, rest[i+1:]...)...), nil
			}
			return "", err
		}
		if !isLinkLike(fi.Mode()) {
			cur = next
			continue
		}
		dest, err := os.Readlink(next)
		if err != nil {
			// A reparse point that names no target (a cloud placeholder, say)
			// redirects the read nowhere, so it is an ordinary entry here.
			cur = next
			continue
		}
		follows++
		if follows > maxLinkFollows {
			return "", fmt.Errorf("too many links in %q", p)
		}
		// Continue the walk from the target's components, then whatever followed
		// the link. The target is not cleaned: a ".." in it is applied by the
		// walk, after the component before it has been followed.
		dest = filepath.FromSlash(dest)
		switch {
		case filepath.IsAbs(dest):
			vol := filepath.VolumeName(dest)
			cur = vol + sep
			dest = dest[len(vol):]
		case dest != "" && os.IsPathSeparator(dest[0]):
			// Windows only: a target such as \outside\x.yaml is rooted but has
			// no drive, so IsAbs is false. The operating system reads it from the
			// root of the drive the link sits on, not below the link's directory.
			cur = filepath.VolumeName(cur) + sep
		default:
			// A relative target is read from the directory the link sits in,
			// which is cur.
		}
		rest = append(strings.Split(strings.Trim(dest, sep), sep), rest[i+1:]...)
		i = -1
	}
	return filepath.Clean(cur), nil
}

// return a slice of path elements for a given path
func splitPathToFragments(p string) []string {
	p = filepath.Clean(p)
	res := make([]string, 0)
	cont := true
	for cont {
		p = filepath.Clean(p)
		d, f := filepath.Split(p)
		res = append(res, f)
		if p == d || d == "" {
			cont = false
		}
		p = d
	}
	slices.Reverse(res)
	return res
}

const (
	// windowsForbiddenStandardChars contains the printable characters a Windows
	// file or directory name may not contain. ':' is not listed because
	// filepath.IsLocal refuses it first.
	windowsForbiddenStandardChars string = `<>"|?*`
	// windowsForbiddenLowChar contains low <32 chars which are not allowed to be used in a file/dir name
	windowsForbiddenLowChar string = "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
)

// isLinkLike reports whether a component may redirect the read somewhere other
// than where its own name sits in the tree. followLinks asks os.Readlink where,
// and treats a reparse point that names no target as an ordinary entry.
//
// ModeSymlink alone is not enough on Windows. A directory junction is a reparse
// point with tag IO_REPARSE_TAG_MOUNT_POINT, and since Go 1.23 (godebug
// winsymlink=1) os.Lstat reports it as ModeIrregular, not ModeSymlink: see
// os/types_windows.go, where IO_REPARSE_TAG_SYMLINK sets ModeSymlink and mount
// points fall through to ModeIrregular. Before Go 1.23 mount points did carry
// ModeSymlink, which is why testing for it alone looks sufficient. This module
// declares go 1.26, so it gets the newer mapping and a junction would walk
// straight past a ModeSymlink-only test.
//
// ModeIrregular is a broad term: it covers reparse tags that do not redirect
// anywhere, such as cloud-provider placeholder files. isLinkLike lets them
// through to followLinks, which asks os.Readlink and, when it cannot read a
// target, walks into the entry as an ordinary one. It is not universal either — IO_REPARSE_TAG_AF_UNIX maps
// to ModeSocket and IO_REPARSE_TAG_DEDUP is reported as an ordinary file — but
// neither of those redirects a read out of the directory, so neither matters
// here.
func isLinkLike(mode fs.FileMode) bool {
	return mode&(fs.ModeSymlink|fs.ModeIrregular) != 0
}
