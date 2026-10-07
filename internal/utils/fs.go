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
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
)

// SafeSubpath returns a normalized absolute path which is a valid and safe subpath for a given base path.
// The rel path is not considered to be safe.
// The base path should be absolute and is considered to be safe
// Returns on error if the result is not a valid subpath or the path contains unsafe elements (especially for windows)
func SafeSubpath(base, rel string) (string, error) {
	// validate rel
	// do not allow rel to start with "\\"
	// on windows it will catch \\host\dir, \\.\long, \\?\ etc
	if strings.HasPrefix(rel, "\\") {
		return "", fmt.Errorf("rel path should not start with \\")
	}
	// do not allow rel to start with "/"
	if strings.HasPrefix(rel, "/") {
		return "", fmt.Errorf("rel path should not start with /")
	}
	// on windows we need additional tests
	if runtime.GOOS == "windows" {
		// this will prevent explicit drive like "c:\" and ADS ("test.txt:aaa")
		if strings.ContainsAny(rel, windowsForbiddenStandardChars) {
			return "", fmt.Errorf("rel path contains windows-specific invalid chars: %v", windowsForbiddenStandardChars)
		}
		if strings.ContainsAny(rel, windowsForbiddenLowChar) {
			return "", fmt.Errorf("rel path contains windows-specific invalid chars 1-31")
		}
		fn := filepath.Base(rel)
		// look at the file name and its extension
		if noext, ext, found := strings.Cut(fn, "."); found {
			// catches a.txt. and .....
			if strings.HasSuffix(ext, ".") {
				return "", fmt.Errorf("rel path: file extension contains '.' at the end")
			}
			if _, ok := windowsForbiddenFileNames[noext]; ok {
				return "", fmt.Errorf("rel path has a windows-specific forbidden file: %v", noext)
			}
		} else {
			if _, ok := windowsForbiddenFileNames[fn]; ok {
				return "", fmt.Errorf("rel path has a windows-specific forbidden file: %v", noext)
			}
		}

		// check for trailing " " and "." all along the path
		for _, s := range filepath.SplitList(rel) {
			if strings.HasSuffix(s, " ") {
				return "", fmt.Errorf("rel path contains trailing space")
			}
			if strings.HasSuffix(s, ".") {
				return "", fmt.Errorf("rel path contains trailing dot")
			}
		}
	}

	// NUL char is not allowed both on linux and windows
	if strings.Contains(rel, "\x00") {
		return "", fmt.Errorf("rel path contains null byte")
	}

	// now check the base path.
	if !filepath.IsAbs(base) {
		return "", fmt.Errorf("base file is not absolute")
	}
	absBase := filepath.Clean(base)

	cleanRel := filepath.Clean(rel)
	if strings.HasPrefix(cleanRel, "..\\") || strings.HasPrefix(cleanRel, "../") {
		return "", fmt.Errorf("rel path after cleaning should not start with ..")
	}

	// check if rel escapes absBase
	p := filepath.Join(absBase, rel)
	r, err := filepath.Rel(absBase, p)
	if err != nil {
		return "", fmt.Errorf("filepath.Rel failed: %w", err)
	}
	res := filepath.Join(absBase, r)
	// now compare res and absBase
	if absBase == res { // r effectively is .
		return "", fmt.Errorf("rel is effectively empty")
	}
	if !strings.HasPrefix(res, absBase+string(filepath.Separator)) {
		return "", fmt.Errorf("not a subpath")
	}

	return res, nil
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
