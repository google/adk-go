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
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

var (
	ErrNotValidRelativePath  = errors.New("the path is not valid relative path in the subtree")
	ErrSymlinkInRelativePath = errors.New("relative path contains a symlink")
)

// SafeSubpath returns a normalized absolute path and the base-path-relative equivalent path.
// The result is a valid and safe subpath for a given base path.
// The rel path is not considered to be safe.
// The base path should be absolute and is considered to be safe
// Returns on error if the result is not a valid subpath or the path contains unsafe elements (especially for windows)
func SafeSubpath(base, rel string, acceptSymlinks bool, evalPath bool) (absPath, relPath string, resErr error) {
	// validate rel
	// do not allow rel to start with "\\"
	// on windows it will catch \\host\dir, \\.\long, \\?\ etc
	if runtime.GOOS == "windows" && strings.HasPrefix(rel, "\\") {
		return "", "", fmt.Errorf("%w: rel path should not start with \\", ErrNotValidRelativePath)
	}
	// do not allow rel to start with "/"
	if strings.HasPrefix(rel, "/") {
		return "", "", fmt.Errorf("%w: rel path should not start with /", ErrNotValidRelativePath)
	}
	// on windows we need additional tests
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
		// cleanRel := filepath.Clean(rel)
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

	// now check the base path.
	if !filepath.IsAbs(base) {
		return "", "", fmt.Errorf("base file is not absolute")
	}
	absBase := filepath.Clean(base)

	cleanRel := filepath.Clean(rel)
	if strings.HasPrefix(cleanRel, "..\\") || strings.HasPrefix(cleanRel, "../") {
		return "", "", fmt.Errorf("%w: rel path after cleaning should not start with ..", ErrNotValidRelativePath)
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

				// make sure we are still safe - recheck everything
				return SafeSubpath(absBase, evr, false, true)
			}
		} else {
			return absBase, r, nil
		}
	} else {
		if evalPath {
			hasSymlink, err := ContainsSymlink(absBase, rel)
			if err != nil {
				return "", "", fmt.Errorf("%w: cannot check for symlinks: %v", ErrNotValidRelativePath, err)
			}
			if hasSymlink {
				return "", "", ErrSymlinkInRelativePath
			}
		}
	}

	return res, r, nil
}

// // evalExistingSymlinks tries to resolve symlinks in the whole base and then symlinks in existing part of rel
// func evalExistingSymlinks(base, rel string) (existing, absent string, err error) {
// 	cleanbase, err := filepath.EvalSymlinks(base)
// 	if err != nil {
// 		return "", "", fmt.Errorf("cannot evaluate symlinks in base: %w", err)
// 	}
// 	cleanrel, err := filepath.EvalSymlinks(rel)
// 	if err == nil {
// 		return filepath.Join(cleanbase, cleanrel), "", nil
// 	}

// 	// why?
// 	if !errors.Is(err, os.ErrNotExist) {
// 		// if any other reason that NotExist - return error
// 		return "", "", fmt.Errorf("cannot evaluate symlinks in rel: %w", err)
// 	}

// 	// a part of ref path doesn't exist. Find the existing part
// 	// don't worry about path traversal right now
// 	frags:= splitPathToFragments(rel)
// 	currentDir:= cleanbase
// 	for i, fr := range frags {
// 		p:= filepath.Join(currentDir, fr)
// 		fi, err := os.Lstat(p)
// 		if fi

// 	return cleanbase, cleanrel, nil
// }

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
		return false, fmt.Errorf("The base path has to be absolute")
	}
	if filepath.IsAbs(rel) {
		return false, fmt.Errorf("The rel path musn't be absolute")
	}

	p := base
	frags := splitPathToFragments(rel)
	for _, fr := range frags {
		p = filepath.Join(p, fr)
		fi, err := os.Lstat(p)
		if err != nil {
			return false, fmt.Errorf("cannot Lstat: %w", err)
		}
		if fi.Mode()&os.ModeSymlink != 0 {
			return true, nil
		}
		if fi.Mode()&os.ModeIrregular != 0 {
			return true, nil
		}
	}

	return false, nil
}

const (
	windowsForbiddenStandardChars string = `:<>"|?*`
	windowsForbiddenLowChar       string = "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
)

// please refer to https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file
var windowsForbiddenFileNames = map[string]bool{
	"CON":  true,
	"PRN":  true,
	"AUX":  true,
	"NUL":  true,
	"COM1": true,
	"COM2": true,
	"COM3": true,
	"COM4": true,
	"COM5": true,
	"COM6": true,
	"COM7": true,
	"COM8": true,
	"COM9": true,
	"COM¹": true,
	"COM²": true,
	"COM³": true,
	"LPT1": true,
	"LPT2": true,
	"LPT3": true,
	"LPT4": true,
	"LPT5": true,
	"LPT6": true,
	"LPT7": true,
	"LPT8": true,
	"LPT9": true,
	"LPT¹": true,
	"LPT²": true,
	"LPT³": true,
}
