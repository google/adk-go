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
	// ErrNotValidRelativePath means that a path is considered not relative or invalid
	ErrNotValidRelativePath = errors.New("the path is not valid relative path in the subtree")
	// ErrSymlinkInRelativePath means that a path contains a link that resolves outside the base
	ErrSymlinkInRelativePath = errors.New("relative path contains a symlink")
)

// SafeSubpath resolves rel below base and checks that the result stays inside it.
//
// The base path must be absolute and is trusted. rel is not. On success the
// result is the normalized absolute path and its base-relative equivalent.
//
// Returns ErrNotValidRelativePath iff rel is not a local path below base:
// absolute, empty, escaping through "..", effectively empty, or (on Windows)
// carrying a drive, a stream, a reserved device name or a trailing dot/space.
// Returns ErrSymlinkInRelativePath iff a component of rel below base is a
// symlink or another link-like entry that resolves outside base. A link that
// resolves inside base is accepted, which keeps Kubernetes ConfigMap and Secret
// mounts (a "..data" link and per-key links) working, as in adk-python.
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
	// on windows IsLocal does not look at alternate data streams ("test.txt:aaa") and
	// does not refuse trailing " " and "." on every component
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

	inside, err := resolvesInside(absBase, res)
	if err != nil {
		return "", "", fmt.Errorf("cannot resolve links: %w", err)
	}
	if !inside {
		return "", "", ErrSymlinkInRelativePath
	}
	return res, r, nil
}

// resolvesInside reports whether target, with every link on the way resolved,
// is still below base (itself resolved, so a symlinked base is trusted).
func resolvesInside(base, target string) (bool, error) {
	realBase, err := resolveExisting(base)
	if err != nil {
		return false, err
	}
	realTarget, err := resolveExisting(target)
	if err != nil {
		if errors.Is(err, errDangling) {
			return false, nil
		}
		return false, err
	}
	rel, err := filepath.Rel(realBase, realTarget)
	if err != nil {
		return false, nil
	}
	return rel == "." || filepath.IsLocal(rel), nil
}

var errDangling = errors.New("dangling link")

// resolveExisting resolves the links in the deepest existing prefix of p and
// appends the rest as is: a component that does not exist, or sits below a
// non-directory, cannot hide a link. A dangling link is errDangling, because
// its destination is unknown.
func resolveExisting(p string) (string, error) {
	existing, rest := p, ""
	for {
		real, err := filepath.EvalSymlinks(existing)
		if err == nil {
			return filepath.Join(real, rest), nil
		}
		if !errors.Is(err, fs.ErrNotExist) && !errors.Is(err, syscall.ENOTDIR) {
			return "", err
		}
		if _, lerr := os.Lstat(existing); lerr == nil {
			return "", errDangling
		}
		parent := filepath.Dir(existing)
		if parent == existing {
			return filepath.Join(existing, rest), nil
		}
		rest = filepath.Join(filepath.Base(existing), rest)
		existing = parent
	}
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

// ContainsSymlink checks the rel path for symlinks
// The check is performed regardless the rel path is valid or not
// The check takes into the consideration the actual FS
func ContainsSymlink(base, rel string) (bool, error) {
	if !filepath.IsAbs(base) {
		return false, fmt.Errorf("the base path has to be absolute")
	}
	if filepath.IsAbs(rel) {
		return false, fmt.Errorf("the rel path must not be absolute")
	}

	p := base
	frags := splitPathToFragments(rel)
	for _, fr := range frags {
		p = filepath.Join(p, fr)
		fi, err := os.Lstat(p)
		if err != nil {
			return false, fmt.Errorf("cannot Lstat: %w", err)
		}
		if isLinkLike(fi.Mode()) {
			return true, nil
		}
	}

	return false, nil
}

const (
	// windowsForbiddenStandardChars contains standard char which are not allowed to be used in a file/dir name
	windowsForbiddenStandardChars string = `:<>"|?*`
	// windowsForbiddenLowChar contains low <32 chars which are not allowed to be used in a file/dir name
	windowsForbiddenLowChar string = "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
)

// isLinkLike reports whether a component may redirect the read somewhere other
// than where its own name sits in the tree.
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
// anywhere, such as cloud-provider placeholder files, and those are refused
// along with the rest. It is not universal either — IO_REPARSE_TAG_AF_UNIX maps
// to ModeSocket and IO_REPARSE_TAG_DEDUP is reported as an ordinary file — but
// neither of those redirects a read out of the directory, so neither matters
// here.
func isLinkLike(mode fs.FileMode) bool {
	return mode&(fs.ModeSymlink|fs.ModeIrregular) != 0
}
