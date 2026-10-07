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

// normalizeCase is one table-driven case for Normalize(base, rel).
//
// Success case (wantErr == false):
//   - the runner always asserts the result stays inside base;
//   - unless withinBaseOnly is set, it also asserts the result equals
//     filepath.Join(base, filepath.FromSlash(wantSuffix)). wantSuffix is
//     slash-form and relative to base; "" means the result must equal base.
//
// Failure case (wantErr == true): the runner only asserts a non-nil error.
type normalizeCase struct {
	name           string
	rel            string
	wantErr        bool
	wantSuffix     string
	withinBaseOnly bool
}

func Normalize(base, rel string) (string, error) {
	return SafeSubpath(base, rel)
}

// runNormalizeCases runs every case against Normalize for the given base.
func runNormalizeCases(t *testing.T, base string, cases []normalizeCase) {
	t.Helper()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Normalize(base, tc.rel)

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
				t.Fatalf("Normalize(%q, %q) = %q; result escapes base %q", base, tc.rel, got, base)
			}

			if tc.withinBaseOnly {
				return
			}

			want := base
			if tc.wantSuffix != "" {
				want = filepath.Join(base, filepath.FromSlash(tc.wantSuffix))
			}
			if got != want {
				t.Errorf("Normalize(%q, %q) = %q; want %q", base, tc.rel, got, want)
			}
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

// commonNormalizeCases apply identically on all platforms.
//
// The fixture base in every suite ends in the component "data" so the
// sibling-prefix cases below exercise the classic boundary bug (a raw
// strings.HasPrefix(result, base) check wrongly accepts ".../data-evil" and
// ".../database").
var commonNormalizeCases = []normalizeCase{
	// ---- Allowed: resolves to base or a descendant ----
	{name: "simple_file", rel: "file.txt", wantSuffix: "file.txt"},
	{name: "nested_descendant", rel: "sub/dir/file.txt", wantSuffix: "sub/dir/file.txt"},
	{name: "leading_dot_slash", rel: "./file.txt", wantSuffix: "file.txt"},
	{name: "interior_dotdot_stays_inside", rel: "sub/../file.txt", wantSuffix: "file.txt"},
	{name: "interior_single_dot", rel: "sub/./nested/file.txt", wantSuffix: "sub/nested/file.txt"},
	{name: "collapsed_double_slash", rel: "sub//double.txt", wantSuffix: "sub/double.txt"},
	{name: "trailing_slash", rel: "sub/", wantSuffix: "sub"},
	// Nets back inside after a leading "..": proves Normalize normalizes THEN
	// checks, rather than rejecting any path containing "..".
	{name: "dotdot_nets_back_into_base", rel: "../data/inside.txt", wantSuffix: "inside.txt"},

	// Policy A4: a path that resolves exactly to base is allowed. Flip these to
	// wantErr:true if your contract forbids returning the root itself.
	{name: "empty_resolves_to_base", rel: "", wantSuffix: ""},
	{name: "dot_resolves_to_base", rel: ".", wantSuffix: ""},

	// ---- Rejected: escapes base ----
	{name: "parent_escape", rel: "../x", wantErr: true},
	{name: "deep_escape", rel: "../../../../etc/secret", wantErr: true},
	{name: "interior_escape", rel: "sub/../../x", wantErr: true},
	{name: "mixed_valid_then_escape", rel: "foo/../../../bar", wantErr: true},
	{name: "bare_dotdot", rel: "..", wantErr: true},
	{name: "sibling_prefix_dash", rel: "../data-evil/x", wantErr: true},
	{name: "sibling_prefix_word", rel: "../database/x", wantErr: true},

	// ---- Rejected: invalid input ----
	// Assumes Normalize validates input and rejects an embedded NUL. If your
	// Normalize is purely lexical and does not validate, move/flip this case.
	{name: "embedded_null_byte", rel: "a\x00.txt", wantErr: true},
}

func TestNormalize_Windows(t *testing.T) {

	if runtime.GOOS != "windows" {
		t.Skip("Skipped - will be run only on Windows")
	}

	const base = `C:\srv\app\data`

	cases := append([]normalizeCase(nil), commonNormalizeCases...)
	cases = append(cases, []normalizeCase{
		// ---- Allowed: both separators are valid on Windows ----
		{name: "backslash_separators", rel: `logs\2025\app.log`, wantSuffix: "logs/2025/app.log"},
		{name: "forward_slash_separators", rel: "logs/2025/app.log", wantSuffix: "logs/2025/app.log"},
		{name: "mixed_separators", rel: `sub/nested\mixed.txt`, wantSuffix: "sub/nested/mixed.txt"},
		{name: "mixed_sep_interior_dotdot", rel: `a\b\..\c.txt`, wantSuffix: "a/c.txt"},

		// ---- Rejected: traversal (both separator styles) ----
		{name: "backslash_traversal", rel: `..\..\Windows\System32\drivers\etc\hosts`, wantErr: true},
		{name: "mixed_sep_traversal", rel: `../..\Windows`, wantErr: true},
		{name: "interior_escape_backslash", rel: `sub\..\..\Windows`, wantErr: true},
		{name: "sibling_prefix_backslash", rel: `..\data-evil\x`, wantErr: true},
		{name: "ads_with_traversal", rel: `..\..\secret.txt::$DATA`, wantErr: true},

		// ---- Rejected: absolute / drive / UNC / device forms ----
		{name: "drive_absolute", rel: `C:\Windows\System32`, wantErr: true},
		{name: "drive_absolute_forward_slash", rel: "C:/Windows", wantErr: true},
		{name: "other_drive_absolute", rel: `D:\data\x`, wantErr: true},
		{name: "unc_path", rel: `\\server\share\x`, wantErr: true},
		{name: "extended_length_prefix", rel: `\\?\C:\Windows`, wantErr: true},
		{name: "device_namespace", rel: `\\.\PhysicalDrive0`, wantErr: true},

		// ---- Semantics-dependent: verify against the intended contract ----
		// `\Windows` is drive-rooted (root of the CURRENT drive) and `C:file`
		// is drive-relative (CWD on C:). A secure Normalize should reject both,
		// but a naive filepath.Join-based impl treats `\Windows` as an in-base
		// subdir. Flip to a success case if that is the intended behavior.
		{name: "drive_rooted_current_drive", rel: `\Windows\x`, wantErr: true},
		{name: "drive_relative", rel: "C:file.txt", wantErr: true},

		// ---- Policy-dependent: a purely lexical Normalize may NOT implement
		// these. Flip wantErr / delete if your Normalize does not special-case
		// Windows device names, alternate data streams, or control chars. ----
		{name: "reserved_name_con", rel: "CON", wantErr: true},
		{name: "reserved_name_nul", rel: "NUL", wantErr: true},
		{name: "reserved_name_com1", rel: "COM1", wantErr: true},
		{name: "ads_on_inside_file", rel: "file.txt:hidden", wantErr: true},
		{name: "control_char", rel: "bad\x01name.txt", wantErr: true},

		// ---- Canonicalization quirks: stays inside base, so exact output is
		// implementation-defined (Windows strips trailing dots/spaces). Assert
		// containment only. ----
		{name: "trailing_dot", rel: "secret.txt.", withinBaseOnly: true},
		{name: "trailing_space", rel: "secret.txt ", withinBaseOnly: true},
	}...)

	runNormalizeCases(t, base, cases)
}

func TestNormalize_NonWindows(t *testing.T) {
	const base = "/srv/app/data"

	cases := append([]normalizeCase(nil), commonNormalizeCases...)
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

	runNormalizeCases(t, base, cases)
}
