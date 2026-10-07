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
	"cmp"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type fsTest struct {
	name         string
	p            string
	wantErr      bool
	wantBase     string
	wantPath     string
	relOnWindows bool
}

type testingEntry struct {
	p    string
	base string
}

func TestNorm(t *testing.T) {

	tests := []struct {
		p                string // to be appended to a base absolute path with `/` or `\\` at the end
		wantDefault      string // after normalization, what should be appended to a base absolute path with `/` or `\` at the end - may be ``
		wantWindows      string
		wantDefaultValid bool
		wantWindowsValid bool
	}{
		{
			p:                `a`,
			wantDefault:      `a`,
			wantWindows:      `a`,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
		{
			p:                `.`,
			wantDefault:      ``,
			wantWindows:      ``,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
		{
			p: `..`,
		},
		{
			p: `../`,
		},
		{
			p: `..\`,
		},
		{
			p: `../a`,
		},
		{
			p: `../z`, // the same like the final path element for the base path
		},
		{
			p: `../../../..`,
		},
		{
			p: `../../../../a/b/c`,
		},
		{
			p: "a\x00/../../etc/passwd",
		},
		{
			p: `/`,
		},
		{
			p: `/a`,
		},
		{
			p: `/a/../b`,
		},
		{
			p: `\\`,
		},
		{
			p: `\\host`,
		},
		{
			p: `\\host\d`,
		},
		{
			p: `\\host\d\a\b`,
		},
		{
			p:                `../a`,
			wantDefault:      ``,
			wantWindows:      ``,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
		{
			p:                `../c`,
			wantDefault:      ``,
			wantWindows:      ``,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
		{
			p:                `../a/b`,
			wantDefault:      `b`,
			wantWindows:      `b`,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
		{
			p:                `a/../b`,
			wantDefault:      `b`,
			wantWindows:      `b`,
			wantDefaultValid: true,
			wantWindowsValid: true,
		},
	}

	for _, tc := range tests {
		// gotDefValid, gotDef := defaultNormalizeRelativePath(tc.p)
		// gotWinValid, gotWin := windowsNormalizeRelativePath(tc.p)
		ssp, err := SafeSubpath("/x/y/z", tc.p)
		t.Logf("ssp: %30v => %30v", tc.p, ssp)
		if err != nil {
			t.Errorf("SafeSubpath failed: %v", err)
		}

		// if gotDefValid != tc.wantDefaultValid {
		// 	t.Errorf("default for %q: error got: %v want %v", tc.p, gotDefValid, tc.wantDefaultValid)
		// }
		// if gotWinValid != tc.wantWindowsValid {
		// 	t.Errorf("windows for %q: error got: %v want %v", tc.p, gotWinValid, tc.wantWindowsValid)
		// }
		// if gotDefValid && (gotDef != tc.wantDefault) {
		// 	t.Errorf("default for %q: error got: %v want %v", tc.p, gotDef, tc.wantDefault)
		// }
		// if gotWinValid && (gotWin != tc.wantWindows) {
		// 	t.Errorf("default for %q: error got: %v want %v", tc.p, gotWin, tc.wantWindows)
		// }
	}
	t.Fail()
}

func addPrefixes(paths []string) []string {
	res := []string{}
	prefs := []string{
		`/`,
		`./`,
		`../`,
		`a/./`,
		`a/../`,
		`\\?\`,
		`\\.\`,
		`\\?\UNC\`,
		`\\host\dir`,
	}
	for _, pref := range prefs {
		for _, p := range paths {
			res = append(res, pref+p)
		}
	}
	return res
}

func addSuffixes(paths []string) []string {
	res := []string{}
	sufs := []string{
		``,
		`.`,
		`..`,
		`...`,
		`.txt`,
		`:stream`,
	}
	for _, suf := range sufs {
		for _, p := range paths {
			res = append(res, p+suf)
		}
	}
	return res
}

func addBackslashes(paths []string) []string {
	res := []string{}
	res = append(res, paths...)
	for _, p := range paths {
		bs := strings.ReplaceAll(p, "/", "\\")
		if bs != p {
			res = append(res)
		}
	}
	return res
}

func toBeTested() []string {
	res := []string{
		``,
		`a`,
		`a/b`,
		`a/b.txt`,
		`a/b/`,
		`/a`,
		`CON`,
		`CONIN$`,
		`CONOUT$`,
		`LPT`,
		`LPT1`,
		`NUL`,
		`c:`,
		`c:\`,
		`c:\a`,
		`c:a`,
		`.`,
		`..`,
		`....`,
	}
	res = addSuffixes(res)
	res = addPrefixes(res)
	res = addBackslashes(res)
	return res
}

func allTests(allin, nonein string) []testingEntry {
	tests := toBeTested()

	res := []testingEntry{}

	for _, p := range tests {
		res = append(res, testingEntry{p: p, base: allin})
		res = append(res, testingEntry{p: p, base: nonein})
	}
	slices.SortFunc(res, func(a, b testingEntry) int { return cmp.Compare(a.p+"|"+a.base, b.p+"|"+b.base) })
	res = slices.Compact(res)
	return res
}

func TestComb2(t *testing.T) {
	format := "%-20v %-6v %-6v %-6v %-6v %-10v %-10v %-10v"
	t.Logf(format, "path", "isAbs", "isLoc", "open?", "stat?", "clean", "volume", "base")

	dir, err := os.Getwd()
	if err != nil {
		t.Errorf("os.Getwd failed %v", err)
	}

	defer os.Chdir(dir)

	os.Chdir("./_testDir")

	tests := allTests(`C:\Users\kdroste\Projects\test\files\_testDir\allin\tp`, `C:\Users\kdroste\Projects\test\files\_testDir\nonein\tp`)

	for _, tc := range tests {
		resPath := filepath.Join(tc.base, tc.p)

		isAbs := filepath.IsAbs(tc.p)
		isLocal := filepath.IsLocal(tc.p)
		clean := filepath.Clean(tc.p)
		volume := filepath.VolumeName(tc.p)
		statable := true
		openable := true

		_, err := os.Stat(resPath)
		if err != nil {
			statable = false
			// t.Errorf("os.Stat failed for %v: %v", p, err)
			// continue
		}
		f, errOpen := os.Open(resPath)
		if errOpen != nil {
			openable = false
		} else {
			defer f.Close()
		}

		// t.Logf(format, "path", "isAbs", "isLoc", "open?", "stat?", "clean", "volume")
		t.Logf(format, tc.p, isAbs, isLocal, openable, statable, clean, volume, tc.base)

	}
	t.Fail()
}

func TestComb(t *testing.T) {
	format := "%-20v %-6v %-6v %-6v %-6v %-10v %-10v"
	t.Logf(format, "path", "isAbs", "isLoc", "open?", "stat?", "clean", "volume")

	dir, err := os.Getwd()
	if err != nil {
		t.Errorf("os.Getwd failed %v", err)
	}

	defer os.Chdir(dir)

	os.Chdir("./_testDir")

	tests := toBeTested()
	slices.Sort(tests)

	for _, p := range tests {

		isAbs := filepath.IsAbs(p)
		isLocal := filepath.IsLocal(p)
		clean := filepath.Clean(p)
		volume := filepath.VolumeName(p)
		statable := true
		openable := true

		_, err := os.Stat(p)
		if err != nil {
			statable = false
			// t.Errorf("os.Stat failed for %v: %v", p, err)
			// continue
		}
		f, errOpen := os.Open(p)
		if errOpen != nil {
			openable = false
		} else {
			defer f.Close()
		}

		// t.Logf(format, "path", "isAbs", "isLoc", "open?", "stat?", "clean", "volume")
		t.Logf(format, p, isAbs, isLocal, openable, statable, clean, volume)

		// if fi.IsDir() {
		// 	t.Logf("Dir: "+format, p, isAbs, isLocal, "", clean, volume, 0, err, "")
		// } else {
		// 	f, errOpen := os.Open(p)
		// 	nRead := -1
		// 	if errOpen == nil {
		// 		defer f.Close()

		// 		b := make([]byte, 10)
		// 		// err = f.SetReadDeadline(time.Now().Add(time.Second))
		// 		// if err != nil {
		// 		// 	t.Errorf("f.SetReadDeadline failed: %v", err)
		// 		// }

		// 		skip := false
		// 		if p == `\\.\CON` || p == `\\.\CON.` || p == `\\.\CON..` || p == `\\.\CON...` ||
		// 			p == `\\.\CONIN$` || p == `\\.\CONIN$.` || p == `\\.\CONIN$..` || p == `\\.\CONIN$...` ||
		// 			p == `\\.\CONOUT$` || p == `\\.\CONOUT$.` || p == `\\.\CONOUT$..` || p == `\\.\CONOUT$...` ||
		// 			p == `\\?\CONIN$` || p == `\\?\CONIN$.` || p == `\\?\CONIN$..` || p == `\\?\CONIN$...` ||
		// 			p == `\\?\CONOUT$` || p == `\\?\CONOUT$.` || p == `\\?\CONOUT$..` || p == `\\?\CONOUT$...` ||
		// 			p == `\\?\CON` {
		// 			skip = true
		// 		}

		// 		if skip {
		// 			fmt.Fprintf(os.Stderr, "Will read: %v  - SKIPPED\n", p)
		// 		} else {
		// 			fmt.Fprintf(os.Stderr, "Will read: %v\n", p)
		// 			nRead, err = f.Read(b)
		// 			if err != nil && !errors.Is(err, io.EOF) {
		// 				t.Errorf("f.Read failed: %v", err)
		// 			}
		// 		}
		// 	}
		// t.Logf("File: "+format, p, isAbs, isLocal, errOpen == nil, clean, volume, nRead, err, errOpen)
		// }
	}
	t.Fail()

}

func listFSTestCases(basePath string) []fsTest {
	return []fsTest{
		{
			name:         "existing rel file path",
			p:            `a`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:         "existing rel file backslash sub path ",
			p:            `a\b`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:         "existing rel file slash sub path",
			p:            `a/b`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     "long existing rel file path",
			p:        `\\?\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         "parent slash",
			p:            `../a`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     "long parent slash",
			p:        `\\?\../a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         "parent backslash",
			p:            `..\a`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     "long parent backslash",
			p:        `\\?\..\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "root backslash",
			p:        `\`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long root backslash",
			p:        `\\?\\`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "root slash",
			p:        `/`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long root slash",
			p:        `\\?\/`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "c slash",
			p:        `c:/`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long c slash",
			p:        `\\?\c:/`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "c backslash",
			p:        `c:\`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long c backslash",
			p:        `\\?\c:\`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "just c",
			p:        `c:`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long just c",
			p:        `\\.\c:`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     "long c backslash",
			p:        `\\.\c:\`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		// {
		// 	name:     `\\host`,
		// 	p:        `\\host`,
		// 	wantErr:  false,
		// 	wantBase: basePath,
		// 	wantPath: filepath.Join(basePath, "a"),
		// },
		{
			name:     `\\localhost`,
			p:        `\\localhost`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		// {
		// 	name:     `\\host\dir`,
		// 	p:        `\\host\dir`,
		// 	wantErr:  false,
		// 	wantBase: basePath,
		// 	wantPath: filepath.Join(basePath, "a"),
		// },
		{
			name:     `long UNC`,
			p:        `\\?\UNC\host\dir`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `\\localhost\dir`,
			p:        `\\localhost\dir`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `\\localhost\c$`,
			p:        `\\localhost\c$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `only NUL`,
			p:        `NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long only NUL`,
			p:        `\\?\NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `abs C:\NUL`,
			p:        `C:\NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long abs C:\NUL`,
			p:        `\\?\C:\NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `default C:NUL`,
			p:        `C:NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long default C:NUL`,
			p:        `\\?\C:NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         `relative parent NUL`,
			p:            `..\NUL`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     `long relative parent NUL`,
			p:        `\\?\..\NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         `relative sub NUL`,
			p:            `a\NUL`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:         `relative current NUL`,
			p:            `.\NUL`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     `long relative sub NUL`,
			p:        `\\?\a\NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `decent abs path`,
			p:        `c:\temp\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `decent abs path mixed slashes`,
			p:        `c:\temp/a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long decent abs path`,
			p:        `\\?\c:\temp\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `decent backslashed abs path`,
			p:        `c:/temp/a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long backslashed decent abs path`,
			p:        `\\?\c:/temp/a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long fully backslashed decent abs path`,
			p:        `//?/c:/temp/a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},

		{
			name:     `decent default path`,
			p:        `c:a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long decent default path`,
			p:        `\\?\c:a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},

		{
			name:         `only CON`,
			p:            `CON`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     `long only CON`,
			p:        `\\?\CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `abs C:\CON`,
			p:        `C:\CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long abs C:\CON`,
			p:        `\\?\C:\CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `default C:CON`,
			p:        `C:CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `long default C:CON`,
			p:        `\\?\C:CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         `relative parent CON`,
			p:            `..\CON`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     `long relative parent CON`,
			p:        `\\?\..\CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:         `relative sub CON`,
			p:            `a\CON`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:         `relative current CON`,
			p:            `.\CON`,
			wantErr:      false,
			wantBase:     basePath,
			wantPath:     filepath.Join(basePath, "a"),
			relOnWindows: true,
		},
		{
			name:     `long relative sub CON`,
			p:        `\\?\a\CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `current dir`,
			p:        `.`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `current dir \ a`,
			p:        `.\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `current dir / a`,
			p:        `./a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `parent dir`,
			p:        `..`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `parent dir / a`,
			p:        `../a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `parent dir \ a`,
			p:        `..\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `b\..\a`,
			p:        `b\..\a`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `b\..\a\..\..\c`,
			p:        `b\..\a\..\..\c`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CON`,
			p:        `CON`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CON.`,
			p:        `CON.`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CON.txt`,
			p:        `CON.txt`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `NUL.txt`,
			p:        `NUL.txt`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `./NUL`,
			p:        `./NUL`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `./NUL.`,
			p:        `./NUL.`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `./NUL.txt`,
			p:        `./NUL.txt`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CONIN`,
			p:        `CONIN`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CONIN$`,
			p:        `CONIN$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `CONIN$.txt`,
			p:        `CONIN$.txt`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `./CONIN$`,
			p:        `./CONIN$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `.\CONIN$`,
			p:        `.\CONIN$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `\\?\.\CONIN$`,
			p:        `\\?\.\CONIN$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
		{
			name:     `\\?\CONIN$`,
			p:        `\\?\CONIN$`,
			wantErr:  false,
			wantBase: basePath,
			wantPath: filepath.Join(basePath, "a"),
		},
	}
}

func TestValues(t *testing.T) {
	pwd, err := os.Getwd()
	if err != nil {
		t.Errorf("cannot os.Getwd()")
	}
	tests := listFSTestCases(pwd)

	format := "%-20v %-6v %-6v %-10v %-10v %10v %-6v %20v   %50v"

	t.Logf(format, "path", "isAbs", "isLoc", "openable", "clean", "volume", "RES", "Norm", "errOpen")

	for _, tc := range tests {
		isAbs := filepath.IsAbs(tc.p)
		isLocal := filepath.IsLocal(tc.p)
		clean := filepath.Clean(tc.p)
		volume := filepath.VolumeName(tc.p)
		f, errOpen := os.Open(tc.p)
		if errOpen == nil {
			defer f.Close()
		}

		valid, norm := windowsNormalizeRelativePath(tc.p)

		t.Logf(format, tc.p, isAbs, isLocal, errOpen == nil, clean, volume, valid, norm, errOpen)
	}
	t.Fail()
}

// func TestWindowsPaths(t *testing.T) {
// 	pwd, err := os.Getwd()
// 	if err != nil {
// 		t.Errorf("cannot os.Getwd()")
// 	}

// 	tests := listFSTestCases(pwd)

// 	for _, tc := range tests {
// 		t.Run(tc.name, func(t *testing.T) {
// 			cleanBase, cleanPath, err := ResolveSubPath(".", tc.p)
// 			if (err != nil) != tc.wantErr {
// 				t.Errorf("TestWindowsPaths() name=%v, error = %v, wantErr %v", tc.name, err, tc.wantErr)
// 			}
// 			if cleanBase != tc.wantBase {
// 				t.Errorf("TestWindowsPaths() name=%v, cleanBase got %q, want %q", tc.name, cleanBase, tc.wantBase)
// 			}
// 			if cleanPath != tc.wantPath {
// 				t.Errorf("TestWindowsPaths() name=%v, cleanPath got %q, want %q", tc.name, cleanPath, tc.wantPath)
// 			}

// 		})
// 	}

// }
