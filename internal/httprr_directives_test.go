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
//     than one, so refreshing one recording re-records the whole package.
//     This is not always wrong -- a package with one directive per test file
//     has nothing to partition -- but it must be a deliberate choice, not an
//     accident (see #1330, reopened after #1362 fixed only the first failure
//     mode above: five packages still had this at the time, unreviewed).
//
// Requiring exactly one directive per cassette rules out the first failure
// mode, at least one cassette per directive rules out the second, and a
// `httprecord:whole-package` acknowledgment comment immediately above any
// directive that claims every cassette in a multi-cassette package rules out
// the third -- it does not change behavior, only forces the choice to be
// visible in the diff that makes it, and readable at the directive itself
// rather than only in an issue thread.
func TestHTTPRecordDirectivesPartitionCassettes(t *testing.T) {
	// Start test from the parent directory, root of the module.
	t.Chdir("..")

	ignore := map[string]bool{
		// Copied from golang.org/x/oscar; defines the flag, does not use it.
		"internal/httprr": true,
		// Contains vendored dependencies.
		"vendor": true,
	}

	const wholePackageAck = "httprecord:whole-package"

	type directive struct {
		file       string
		pattern    string
		re         *regexp.Regexp
		claims     int  // cassettes in the same package this directive matched
		acked      bool // preceding line carries the wholePackageAck marker
		lineForAck string
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
			// The marker must be in a comment line above the directive, never
			// appended to the //go:generate line itself -- go:generate treats
			// everything after the directive name as the shell command to
			// run, so trailing text there would be passed to `go test` as
			// bogus extra arguments instead of being a comment. It can be
			// anywhere in the contiguous `//`-comment block directly above
			// the directive (walking up through consecutive plain "//" lines,
			// stopping at the first blank or non-comment line), not only the
			// single line touching the directive, so an explanation can
			// precede the marker within the same block.
			acked := false
			for j := i - 1; j >= 0; j-- {
				prev := strings.TrimSpace(lines[j])
				if !strings.HasPrefix(prev, "//") || strings.HasPrefix(prev, "//go:generate") {
					break
				}
				if strings.Contains(prev, wholePackageAck) {
					acked = true
					break
				}
			}
			byPkg[filepath.Dir(path)] = append(byPkg[filepath.Dir(path)], directive{
				file: path, pattern: pattern, re: re, acked: acked, lineForAck: line,
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
		// one. Not necessarily wrong -- a package with one directive covering
		// one test file has nothing to partition -- so this does not forbid
		// it, only requires the `httprecord:whole-package` comment directly
		// above the directive, so the breadth is a visible, deliberate choice
		// in the diff that introduces or keeps it, not a silent default.
		for _, d := range directives {
			if d.claims == 0 {
				t.Errorf("%s: -httprecord=%s claims none of the %d cassettes in %s/testdata; `go generate` would record nothing",
					d.file, d.pattern, len(cassettes), pkg)
				continue
			}
			if d.claims == len(cassettes) && len(cassettes) > 1 && !d.acked {
				t.Errorf("%s: -httprecord=%s claims all %d cassettes in %s/testdata, re-recording any one re-records "+
					"the whole package -- if that's intended, add a `// %s` comment directly above the directive "+
					"(see TestHTTPRecordDirectivesPartitionCassettes); if not, narrow the pattern:\n\t%s",
					d.file, d.pattern, len(cassettes), pkg, wholePackageAck, d.lineForAck)
			}
		}
	}

	t.Logf("checked %d -httprecord directives across %d packages", total, len(byPkg))
}
