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
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

// var (
// 	ErrBaseIsNotAbsolute = errors.New("base path must be absolute")
// 	ErrRelIsForbidden    = errors.New("rel path is forbidden")
// )

// func CandidateResolveSubPath(base, rel string) {
// 	// Assumption: base is a valid path local-fs path, trusted to be valid
// 	// rel is user-supplied and can be malicious

// 	// checks for rel:
// 	//  - doesn't start with slash / backslash
// 	//  - ..
// 	//  - not restricted for fs  (NUL on windows)

// 	isAbs := IsPlatformAbs(rel)

// }

// topLevel

// NormalizeRelativePath return true and the normalized path on success.
// On failure returns false.
func NormalizeRelativePath(p string) (bool, string) {
	p = filepath.Clean(p)
	if runtime.GOOS == "windows" {
		return windowsNormalizeRelativePath(p)
	}
	return defaultNormalizeRelativePath(p)
}

func defaultNormalizeRelativePath(p string) (bool, string) {
	p = filepath.Clean(p)
	if path.IsAbs(p) {
		return false, ""
	}

	return true, p
}

func SafeSubpath(base, rel string) (string, error) {
	if !filepath.IsAbs(base) {
		return "", fmt.Errorf("base file is not absolute")
	}
	absBase := filepath.Clean(base)
	p := filepath.Join(absBase, rel)
	r, err := filepath.Rel(absBase, p)
	if err != nil {
		return "", err
	}
	res := filepath.Join(absBase, r)
	if !strings.HasPrefix(res, absBase) {
		return "", fmt.Errorf("not a subpath")
	}
	return res, nil
}

const (
	windowsInvalidChars string = `:<>"|?*`
)

func windowsNormalizeRelativePath(p string) (bool, string) {
	if len(p) == 0 {
		return false, ""
	}
	// non-empty
	if p[0] == '\\' || p[0] == '/' {
		// please mind this also includes long paths \\?\ and physical path \\.\
		return false, ""
	}
	if len(p) >= 2 {
		// check for drives
		// no ':' as the second char
		if p[1] == ':' {
			// no matter what the drive is, it's not valid
			return false, ""
		}
	}
	// check for not allowed chars
	if strings.ContainsAny(p, windowsInvalidChars) {
		// please note that this also forbids file streams
		return false, ""
	}
	// convert to upper and use backslashes only
	p = strings.ToUpper(p)
	p = strings.ReplaceAll(p, "/", "\\")

	// validate path fragments
	frags := strings.Split(p, "\\")
	for i, f := range frags {
		if len(f) == 0 {
			return false, ""
		}
		if f == "." {
			return false, "" // . should be skipped - just to be safe
		}
		if f == ".." {
			return false, "" // .. should be skipped - path traversal
		}
		if i == len(frags)-1 {
			// the last fragment (final file/dir name)
			for fn := range windowsForbiddenFileNames {
				if f == fn {
					return false, "" // explicit forbidden file name
				}
				if strings.HasPrefix(f, fn+".") {
					return false, "" // like 'NUL.txt'
				}
			}
		}
	}

	p = strings.ToLower(p)

	return true, p
}

// type windowsPathClassification struct {
// 	isLong           bool
// 	isPhys           bool
// 	usesCurrentDrive bool
// 	usesCurrentDir   bool
// 	canBeAppended    bool
// }

// func PlatformClassify(platform, path string) (isLong bool, isPhys bool, isAbs bool, isEmpty bool, usesParent bool, canBeAppended bool, usesDefaultDrive bool, usesDefaultDir bool, isInvalid bool) {
// 	if platform == "windows" {
// 		if strings.HasPrefix(path, windowsLongPathPrefix) {
// 			isLong = true
// 			isAbs = true
// 			return
// 		}
// 		if strings.HasPrefix(path, windowsPhysicalDrive) {
// 			isPhys = true
// 			isAbs = true
// 			return
// 		}
// 		if len(path) == 0 {
// 			isEmpty = true
// 			return
// 		}
// 		path = strings.ToLower(path)
// 		if len(path) <= 1 {
// 			// 1
// 			if path[0] == '\\' || path[0] == '/' {
// 				usesDefaultDrive = true

// 			}
// 		} else { // >= 2
// 			// check for drive
// 			if path[1] == ':' {
// 				if path[0] >= 'A' && path[0] <= 'Z' {
// 					// got drive
// 					if len(path) == 2 {
// 						// just drive
// 						usesDefaultDir = true
// 						return
// 					}
// 					// longer than 2
// 					if path[2] == '\\' || path[2] == '/' {
// 						isAbs = true
// 						// look for \..\
// 					} else {
// 						usesDefaultDir = true
// 					}

// 				} else {
// 					isInvalid = true
// 					return
// 				}
// 			}
// 		}

// 	}
// }

// // func IsPlatformAbs(platform, rel string) bool {
// // 	if platform == "windows" {
// // 		if strings.HasPrefix(rel, windowsLongPathPrefix)
// // 	}
// // 	// not windows

// // }

// // ResolveSubPath tries to add rel to base path. The result is a valid absolute path under the base path.
// // Returns an error if the base path is not absolute or the rel path is not a valid sub-path.
// // Depending on OS returns and error for paths which should not be used.
// func ResolveSubPath(base, rel string) (cleanBase, cleanRes string, err error) {

// 	// ensure base is absolute
// 	if !path.IsAbs(base) {
// 		return "", "", ErrBaseIsNotAbsolute
// 	}

// 	//

// 	return "", "", nil
// }

// func resolvePath(p string) {
// 	filepath.Abs(p)
// }

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
