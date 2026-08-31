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

package internal_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// wholePackageAck is the marker a directive uses to acknowledge that it
// deliberately claims every cassette in a multi-cassette package. It is
// recognized only as the leading token of a `//` comment line above the
// directive (see ackedAbove).
const wholePackageAck = "httprecord:whole-package"

// TestHTTPRecordDirectivesPartitionCassettes checks that in every package which
// declares at least one `//go:generate go test -httprecord=…` directive, each
// of the package's testdata/*.httprr cassettes is claimed by exactly one
// directive.
//
// The -httprecord flag is a regexp matched against the cassette's file path
// (see httprr.Recording in internal/httprr), not a -run test-name filter. That
// makes three failure modes possible, all silent:
//
//   - Two directives in one package match the same cassette, so
//     `go generate ./<pkg>/...` re-records it twice against a live model in a
//     single invocation, and re-recording any one cassette rewrites unrelated
//     ones.
//   - A directive matches none of its cassettes, so `go generate` appears to
//     succeed while recording nothing.
//   - A single directive matches every cassette in a package that has more
//     than one, so `go generate ./<pkg>/...` re-records the whole package --
//     against live models whose responses differ run to run -- whenever
//     someone means to refresh just one. This is sometimes unavoidable: a
//     package whose cassettes all come from a single test function has no
//     per-function partition to make. So it is not forbidden; it requires a
//     `// httprecord:whole-package` comment above the directive stating why,
//     so the breadth is a deliberate, reviewed choice and not a silent
//     default (see #1330, reopened after #1362 fixed only the first mode
//     above: four packages still had an unreviewed wildcard directive then).
//
// Requiring exactly one directive per cassette rules out the first failure
// mode, at least one cassette per directive rules out the second, and the
// `httprecord:whole-package` acknowledgment rules out the third. The marker
// detection itself is exercised by TestAckedAbove.
func TestHTTPRecordDirectivesPartitionCassettes(t *testing.T) {
	// Start test from the parent directory, root of the module.
	t.Chdir("..")

	ignore := map[string]bool{
		// Copied from golang.org/x/oscar; defines the flag, does not use it.
		"internal/httprr": true,
		// Contains vendored dependencies.
		"vendor": true,
	}

	type directive struct {
		file    string
		lineNo  int // 1-based line of the //go:generate directive
		text    string
		pattern string
		re      *regexp.Regexp
		claims  int  // cassettes in the same package this directive matched
		acked   bool // comment block above carries the wholePackageAck marker
	}
	byPkg := make(map[string][]directive)
	total := 0

	err := filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if ignore[filepath.ToSlash(path)] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		lines := strings.Split(string(content), "\n")
		for i, raw := range lines {
			line := strings.TrimSpace(raw)
			if !strings.HasPrefix(line, "//go:generate") || !strings.Contains(line, "-httprecord") {
				continue
			}
			pattern := ""
			for _, field := range strings.Fields(line) {
				if p, ok := strings.CutPrefix(field, "-httprecord="); ok {
					pattern = p
					break
				}
			}
			if pattern == "" {
				// Not `-httprecord=<regexp>`: either an empty pattern (which
				// disables recording) or a form this test cannot read, in which
				// case the checks below would pass vacuously.
				t.Errorf("%s: cannot read -httprecord pattern from directive: %s", path, line)
				continue
			}
			re, err := regexp.Compile(pattern)
			if err != nil {
				t.Errorf("%s: -httprecord=%s is not a valid regexp: %v", path, pattern, err)
				continue
			}
			byPkg[filepath.Dir(path)] = append(byPkg[filepath.Dir(path)], directive{
				file:    path,
				lineNo:  i + 1,
				text:    line,
				pattern: pattern,
				re:      re,
				acked:   ackedAbove(lines, i),
			})
			total++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking module: %v", err)
	}
	if total == 0 {
		t.Fatal("no //go:generate -httprecord directives found; this test is no longer checking anything")
	}

	for pkg, directives := range byPkg {
		cassettes, err := filepath.Glob(filepath.Join(pkg, "testdata", "*.httprr"))
		if err != nil {
			t.Errorf("globbing cassettes in %s: %v", pkg, err)
			continue
		}
		for _, cassette := range cassettes {
			// The path httprr matches against is the one the test passes to
			// httprr.Open, which is filepath.Join("testdata", <name>.httprr)
			// relative to the package directory. Checked in slash form only:
			// on Windows that path uses a backslash, and directives written as
			// `testdata/…` would not match it, but no cassette in this repo has
			// ever been recorded on Windows. Directives added by this change use
			// `testdata[/\\]…` so they work either way.
			rel := filepath.ToSlash(filepath.Join("testdata", filepath.Base(cassette)))
			var matched []string
			for i := range directives {
				if directives[i].re.MatchString(rel) {
					directives[i].claims++
					matched = append(matched, directives[i].file+": -httprecord="+directives[i].pattern)
				}
			}
			switch {
			case len(matched) == 0:
				t.Errorf("%s/%s is claimed by no -httprecord directive; `go generate ./%s/...` will not re-record it", pkg, rel, pkg)
			case len(matched) > 1:
				t.Errorf("%s/%s is claimed by %d -httprecord directives, want exactly 1:\n\t%s",
					pkg, rel, len(matched), strings.Join(matched, "\n\t"))
			}
		}

		// A directive that claims nothing is a dead pattern: `go generate` runs
		// it, records nothing, and reports success. The per-cassette check above
		// misses this whenever a sibling directive already covers every cassette.
		//
		// A directive that claims EVERY cassette in a package with more than
		// one is the opposite problem (#1330): `go generate ./<pkg>/...`
		// silently re-records the whole package, against live models whose
		// responses differ run to run, whenever someone means to refresh just
		// one. Unavoidable only when every cassette comes from a single test
		// function, so this does not forbid it -- it requires the
		// `httprecord:whole-package` comment above the directive, so the
		// breadth is a visible, deliberate choice in the diff that introduces
		// or keeps it, not a silent default.
		for _, d := range directives {
			if d.claims == 0 {
				t.Errorf("%s: -httprecord=%s claims none of the %d cassettes in %s/testdata; `go generate` would record nothing",
					d.file, d.pattern, len(cassettes), pkg)
				continue
			}
			if d.claims == len(cassettes) && len(cassettes) > 1 && !d.acked {
				t.Errorf("%s:%d: -httprecord=%s claims all %d cassettes in %s/testdata, so re-recording any one "+
					"re-records the whole package. Give each recording test function its own directive; only if "+
					"they all come from one function, add a `// %s` comment directly above the directive explaining "+
					"why (see TestHTTPRecordDirectivesPartitionCassettes):\n\t%s",
					d.file, d.lineNo, d.pattern, len(cassettes), pkg, wholePackageAck, d.text)
			}
		}
	}

	t.Logf("checked %d -httprecord directives across %d packages", total, len(byPkg))
}

// ackedAbove reports whether the contiguous block of `//` comment lines
// immediately above lines[directiveIdx] carries the whole-package
// acknowledgment marker.
//
// The marker is recognized only as the leading token of a comment line --
// `// httprecord:whole-package …` -- so a line that merely mentions it in
// prose, including one that forbids it, does not satisfy the guard. The block
// is walked upward from the directive and stops at the first line that is
// blank, is not a `//` comment, or is itself a //go:generate directive (so an
// ack never bleeds onto a sibling directive stacked below it). The marker may
// sit anywhere in the block, so a longer explanation can precede it.
func ackedAbove(lines []string, directiveIdx int) bool {
	for j := directiveIdx - 1; j >= 0; j-- {
		line := strings.TrimSpace(lines[j])
		if !strings.HasPrefix(line, "//") || strings.HasPrefix(line, "//go:generate") {
			return false
		}
		text := strings.TrimSpace(strings.TrimPrefix(line, "//"))
		if strings.HasPrefix(text, wholePackageAck) {
			return true
		}
	}
	return false
}

func TestAckedAbove(t *testing.T) {
	// The directive is always the last line; the lines before it are the
	// candidate comment block. Placement is what matters here, not the regexp.
	const directive = "//go:generate go test -httprecord=.*"

	tests := []struct {
		name  string
		above []string
		want  bool
	}{
		{
			name:  "marker on the line touching the directive",
			above: []string{"// httprecord:whole-package -- one test function"},
			want:  true,
		},
		{
			name: "marker higher in the block, explanation below it",
			above: []string{
				"// httprecord:whole-package",
				"// every cassette is a subtest of the one integration test.",
			},
			want: true,
		},
		{
			name:  "no comment at all",
			above: []string{""},
			want:  false,
		},
		{
			name:  "blank line between marker and directive",
			above: []string{"// httprecord:whole-package", ""},
			want:  false,
		},
		{
			name:  "marker only mentioned in prose",
			above: []string{"// Do NOT use httprecord:whole-package in this package."},
			want:  false,
		},
		{
			name:  "block comment, not a line comment",
			above: []string{"/* httprecord:whole-package */"},
			want:  false,
		},
		{
			name:  "marker appended to a go:generate line above",
			above: []string{"//go:generate go test -httprecord=x // httprecord:whole-package"},
			want:  false,
		},
		{
			name: "ack does not bleed past a sibling directive",
			above: []string{
				"// httprecord:whole-package",
				"//go:generate go test -httprecord=other",
			},
			want: false,
		},
		{
			name:  "non-comment code line directly above",
			above: []string{"const modelName = \"gemini\""},
			want:  false,
		},
		{
			name:  "marker indented inside the comment",
			above: []string{"//   httprecord:whole-package -- indented"},
			want:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			lines := append(append([]string{}, tc.above...), directive)
			if got := ackedAbove(lines, len(lines)-1); got != tc.want {
				t.Errorf("ackedAbove(%q) = %v, want %v", tc.above, got, tc.want)
			}
		})
	}
}
