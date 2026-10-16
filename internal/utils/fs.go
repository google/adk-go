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
)

var (
	// ErrNotValidRelativePath means that a path is considered not relative or invalid
	ErrNotValidRelativePath = errors.New("the path is not valid relative path in the subtree")
	// ErrSymlinkInRelativePath means that a path contains a symlink when it should not
	ErrSymlinkInRelativePath = errors.New("relative path contains a symlink")
)

// SafeSubpath should be given a base path and a relative path (rel).
// The base path should be absolute and is considered to be safe
// The rel path is not considered to be safe.
// On success, the result is a normalized absolute path and the base-path-relative equivalent path.
// Returns an error if the result is not a valid subpath or the path contains unsafe elements (OS-specific)
// Returns ErrNotValidRelativePath iff the path is invalid or is not relative
// Returns ErrSymlinkInRelativePath iff you set acceptSymlinks to false and evalPath to true and the path being check contains symlinks.
func SafeSubpath(base, rel string, acceptSymlinks, evalPath bool) (absPath, relPath string, resErr error) {
	// check the base path.
	if !filepath.IsAbs(base) {
		return "", "", fmt.Errorf("base path is not absolute")
	}
	absBase := filepath.Clean(base)

	// validate rel
	// on windows do not allow rel to start with "\\" - it will catch \\host\dir, \\.\long, \\?\ etc
	if runtime.GOOS == "windows" && strings.HasPrefix(rel, "\\") {
		return "", "", fmt.Errorf("%w: rel path should not start with \\", ErrNotValidRelativePath)
	}
	// do not allow rel to start with "/"
	if strings.HasPrefix(rel, "/") {
		return "", "", fmt.Errorf("%w: rel path should not start with /", ErrNotValidRelativePath)
	}
	// on windows we need additional tests for drives, ADS, special files, trailing " " and "."
	if runtime.GOOS == "windows" {
		// this will prevent explicit drive like "c:\" and ADS ("test.txt:aaa")
		if strings.ContainsAny(rel, windowsForbiddenStandardChars) {
			return "", "", fmt.Errorf("%w: rel path contains windows-specific invalid chars: %v", ErrNotValidRelativePath, windowsForbiddenStandardChars)
		}
		if strings.ContainsAny(rel, windowsForbiddenLowChar) {
			return "", "", fmt.Errorf("%w: rel path contains windows-specific invalid chars 1-31", ErrNotValidRelativePath)
		}
		fn := filepath.Base(rel)
		// look at the file name and its extension
		if noext, ext, found := strings.Cut(fn, "."); found {
			// catches a.txt. and .....
			if strings.HasSuffix(ext, ".") {
				return "", "", fmt.Errorf("%w: rel path: file extension contains '.' at the end", ErrNotValidRelativePath)
			}
			if _, ok := windowsForbiddenFileNames[noext]; ok {
				return "", "", fmt.Errorf("%w: rel path has a windows-specific forbidden file: %v", ErrNotValidRelativePath, noext)
			}
		} else {
			if _, ok := windowsForbiddenFileNames[fn]; ok {
				return "", "", fmt.Errorf("%w: rel path has a windows-specific forbidden file: %v", ErrNotValidRelativePath, noext)
			}
		}

		// check for trailing " " and "." all along the path
		p := rel
		cont := true
		for cont {
			p = filepath.Clean(p)
			d, f := filepath.Split(p)
			if d == "" {
				cont = false
			}
			if strings.HasSuffix(f, " ") {
				return "", "", fmt.Errorf("%w: rel path contains trailing space", ErrNotValidRelativePath)
			}
			if strings.HasSuffix(f, ".") {
				return "", "", fmt.Errorf("%w: rel path contains trailing dot", ErrNotValidRelativePath)
			}
			if p == d {
				cont = false
			}
			p = d
		}
	}

	// NUL char is not allowed both on linux and windows
	if strings.Contains(rel, "\x00") {
		return "", "", fmt.Errorf("%w: rel path contains null byte", ErrNotValidRelativePath)
	}

	cleanRel := filepath.Clean(rel)
	if (runtime.GOOS == "windows" && strings.HasPrefix(cleanRel, "..\\")) || strings.HasPrefix(cleanRel, "../") {
		return "", "", fmt.Errorf("%w: rel path after cleaning should not start with '..'", ErrNotValidRelativePath)
	}

	// check if rel escapes absBase
	p := filepath.Join(absBase, rel)
	r, err := filepath.Rel(absBase, p)
	if err != nil {
		return "", "", fmt.Errorf("filepath.Rel failed: %w", err)
	}
	res := filepath.Join(absBase, r)
	// now compare res and absBase
	if absBase == res { // r effectively is .
		return "", "", fmt.Errorf("%w: rel is effectively empty", ErrNotValidRelativePath)
	}
	if !strings.HasPrefix(res, absBase+string(filepath.Separator)) {
		return "", "", fmt.Errorf("%w: not a subpath", ErrNotValidRelativePath)
	}

	if acceptSymlinks {
		if evalPath {
			fp := filepath.Join(absBase, r)
			ev, err := filepath.EvalSymlinks(fp)
			if err != nil {
				return "", "", fmt.Errorf("%w: cannot evaluate symlinks: %w", ErrNotValidRelativePath, err)
			}
			if ev != fp {
				// got symlink in rel
				// make it relative to absBase
				evr, err := filepath.Rel(absBase, ev)
				if err != nil {
					return "", "", fmt.Errorf("filepath.Rel failed: %w", err)
				}

				// make sure we are still safe - recheck everything as symlink-free
				return SafeSubpath(absBase, evr, false, true)
			}
		} else {
			return absBase, r, nil
		}
	} else {
		if evalPath {
			hasSymlink, err := ContainsSymlink(absBase, rel)
			if err != nil {
				return "", "", fmt.Errorf("%w: cannot check for symlinks: %w", ErrNotValidRelativePath, err)
			}
			if hasSymlink {
				return "", "", ErrSymlinkInRelativePath
			}
		}
	}

	return res, r, nil
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

// windowsForbiddenFileNames lists names of files reserved on windows. Also file names with such suffixes are reserved.
// please refer to https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file
var windowsForbiddenFileNames = map[string]bool{
	"CON":     true,
	"CONIN$":  true,
	"CONOUT$": true,
	"PRN":     true,
	"AUX":     true,
	"NUL":     true,
	"COM1":    true,
	"COM2":    true,
	"COM3":    true,
	"COM4":    true,
	"COM5":    true,
	"COM6":    true,
	"COM7":    true,
	"COM8":    true,
	"COM9":    true,
	"COM¹":    true,
	"COM²":    true,
	"COM³":    true,
	"LPT1":    true,
	"LPT2":    true,
	"LPT3":    true,
	"LPT4":    true,
	"LPT5":    true,
	"LPT6":    true,
	"LPT7":    true,
	"LPT8":    true,
	"LPT9":    true,
	"LPT¹":    true,
	"LPT²":    true,
	"LPT³":    true,
}

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
