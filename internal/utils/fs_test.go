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
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type normalizeCase struct {
	name       string
	rel        string
	wantErr    bool
	wantSuffix string
}

// runTests runs every case against [SafeSubpath] for the given base.
func runTests(t *testing.T, base string, cases []normalizeCase) {
	t.Helper()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, gotRel, err := SafeSubpath(base, tc.rel)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("Normalize(%q, %q) = %q, nil; want error (escape/invalid)", base, tc.rel, got)
				}
				return
			}

			if err != nil {
				t.Fatalf("Normalize(%q, %q) returned error %v; want success", base, tc.rel, err)
			}

			// Defense-in-depth: a success must never land outside base.
			if !withinBase(base, got) {
				t.Fatalf("SafeSubpath(%q, %q) = %q; result escapes base %q", base, tc.rel, got, base)
			}

			if tc.wantSuffix != gotRel {
				t.Fatalf("SafeSubpath(%q, %q) = (%q,%q); want suffix %q", base, tc.rel, got, gotRel, tc.wantSuffix)
			}

			// want := base
			// if tc.wantSuffix != "" {
			// 	want = filepath.Join(base, filepath.FromSlash(tc.wantSuffix))
			// }
			// if got != want {
			// 	t.Errorf("Normalize(%q, %q) = %q; want %q", base, tc.rel, got, want)
			// }
		})
	}
}

// withinBase reports whether p is base or a descendant of base, using the
// platform's path semantics.
func withinBase(base, p string) bool {
	rel, err := filepath.Rel(base, p)
	if err != nil {
		return false
	}
	if rel == ".." || rel == "." {
		return rel == "."
	}
	return !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// commonSubpathCases apply identically on all platforms.
var commonSubpathCases = []normalizeCase{
	// Accepted
	{name: "simple_file", rel: "file.txt", wantSuffix: "file.txt"},
	{name: "nested_descendant", rel: "sub/dir/file.txt", wantSuffix: "sub/dir/file.txt"},
	{name: "leading_dot_slash", rel: "./file.txt", wantSuffix: "file.txt"},
	{name: "interior_dotdot_stays_inside", rel: "sub/../file.txt", wantSuffix: "file.txt"},
	{name: "interior_single_dot", rel: "sub/./nested/file.txt", wantSuffix: "sub/nested/file.txt"},
	{name: "collapsed_double_slash", rel: "sub//double.txt", wantSuffix: "sub/double.txt"},
	{name: "trailing_slash", rel: "sub/", wantSuffix: "sub"},

	// Rejected
	// We don't allow that, it allows to test for the physical path (brute-force)
	{name: "dotdot_nets_back_into_base", rel: "../data/inside.txt", wantErr: true},

	// Empty input
	{name: "empty_resolves_to_base", rel: "", wantErr: true},
	{name: "dot_resolves_to_base", rel: ".", wantErr: true},

	// Escapes
	{name: "parent_escape", rel: "../x", wantErr: true},
	{name: "deep_escape", rel: "../../../../etc/secret", wantErr: true},
	{name: "interior_escape", rel: "sub/../../x", wantErr: true},
	{name: "mixed_valid_then_escape", rel: "foo/../../../bar", wantErr: true},
	{name: "bare_dotdot", rel: "..", wantErr: true},
	{name: "sibling_prefix_dash", rel: "../data-evil/x", wantErr: true},
	{name: "sibling_prefix_word", rel: "../database/x", wantErr: true},

	// NUL rune
	{name: "embedded_null_byte", rel: "a\x00.txt", wantErr: true},
}

func TestSafeSubpath_Windows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Skipped - will be run only on Windows")
	}

	const base = `C:\srv\app\data`

	cases := append([]normalizeCase(nil), commonSubpathCases...)
	cases = append(cases, []normalizeCase{
		// Allowed: mixed separators
		{name: "backslash_separators", rel: `logs\2025\app.log`, wantSuffix: "logs/2025/app.log"},
		{name: "forward_slash_separators", rel: "logs/2025/app.log", wantSuffix: "logs/2025/app.log"},
		{name: "mixed_separators", rel: `sub/nested\mixed.txt`, wantSuffix: "sub/nested/mixed.txt"},

		// .. in the middle, stays within the base
		{name: "mixed_sep_interior_dotdot", rel: `a\b\..\c.txt`, wantSuffix: "a/c.txt"},

		// Traversal
		{name: "backslash_traversal", rel: `..\..\Windows\System32\drivers\etc\hosts`, wantErr: true},
		{name: "mixed_sep_traversal", rel: `../..\Windows`, wantErr: true},
		{name: "interior_escape_backslash", rel: `sub\..\..\Windows`, wantErr: true},

		// base has data at the end, simple suffix check is not sufficient
		{name: "sibling_prefix_backslash", rel: `..\data-evil\x`, wantErr: true},

		// ADS
		{name: "ads_with_traversal", rel: `..\..\secret.txt::$DATA`, wantErr: true},

		// Absolute
		{name: "drive_absolute", rel: `C:\Windows\System32`, wantErr: true},
		{name: "drive_absolute_forward_slash", rel: "C:/Windows", wantErr: true},
		{name: "other_drive_absolute", rel: `D:\data\x`, wantErr: true},
		{name: "unc_path", rel: `\\server\share\x`, wantErr: true},
		{name: "extended_length_prefix", rel: `\\?\C:\Windows`, wantErr: true},
		{name: "device_namespace", rel: `\\.\PhysicalDrive0`, wantErr: true},

		// Current drive - relative
		{name: "drive_rooted_current_drive", rel: `\Windows\x`, wantErr: true},
		// Specified drive, current dir-related
		{name: "drive_relative", rel: "C:file.txt", wantErr: true},

		// Misc
		{name: "reserved_name_con", rel: "CON", wantErr: true},
		{name: "reserved_name_nul", rel: "NUL", wantErr: true},
		{name: "reserved_name_com1", rel: "COM1", wantErr: true},
		{name: "ads_on_inside_file", rel: "file.txt:hidden", wantErr: true},
		{name: "control_char", rel: "bad\x01name.txt", wantErr: true},

		// Trailing ' ' and '.'
		{name: "trailing_dot", rel: "secret.txt.", wantErr: true},
		{name: "trailing_space", rel: "secret.txt ", wantErr: true},

		// ' ' or '.' suffix within the path
		{name: "dir_trailing_dot", rel: "./dir./secret.txt", wantErr: true},
		{name: "dir_trailing_space", rel: "./dir /secret.txt", wantErr: true},
	}...)

	runTests(t, base, cases)
}

func TestSafeSubpath_NonWindows(t *testing.T) {
	const base = "/srv/app/data"

	cases := append([]normalizeCase(nil), commonSubpathCases...)
	cases = append(cases, []normalizeCase{
		// ---- Allowed: characters that are LITERAL on POSIX filesystems ----
		// Backslash is a normal filename byte on POSIX, NOT a separator: this
		// must be a single file inside base, not treated as traversal.
		{name: "backslash_is_literal_not_separator", rel: `weird\name.txt`, wantSuffix: `weird\name.txt`},
		{name: "colon_is_literal", rel: "file:with:colons.txt", wantSuffix: "file:with:colons.txt"},
		{name: "dotfile", rel: ".hiddenfile", wantSuffix: ".hiddenfile"},
		{name: "unicode_name", rel: "naïve-名前.txt", wantSuffix: "naïve-名前.txt"},

		// ---- Rejected: absolute override ----
		// An absolute rel must not silently replace the base.
		{name: "absolute_override", rel: "/etc/passwd", wantErr: true},
		{name: "leading_double_slash_absolute", rel: "//etc/passwd", wantErr: true},

		// ---- Rejected: traversal ----
		{name: "classic_traversal", rel: "../../etc/passwd", wantErr: true},
		{name: "traversal_into_proc", rel: "../../../proc/self/environ", wantErr: true},
		{name: "null_byte_with_traversal", rel: "foo\x00/../../etc/passwd", wantErr: true},
	}...)

	runTests(t, base, cases)
}
