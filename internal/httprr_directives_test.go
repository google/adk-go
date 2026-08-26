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
// makes two failure modes possible, and both are silent:
//
//   - Two directives in one package match the same cassette, so
//     `go generate ./<pkg>/...` re-records it twice against a live model in a
//     single invocation, and re-recording any one cassette rewrites unrelated
//     ones.
//   - A directive matches none of its cassettes, so `go generate` appears to
//     succeed while recording nothing.
//
// Requiring exactly one directive per cassette rules out the first, and at
// least one cassette per directive rules out the second.
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
		pattern string
		re      *regexp.Regexp
		claims  int // cassettes in the same package this directive matched
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
		for _, line := range strings.Split(string(content), "\n") {
			line = strings.TrimSpace(line)
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
			byPkg[filepath.Dir(path)] = append(byPkg[filepath.Dir(path)], directive{file: path, pattern: pattern, re: re})
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
		for _, d := range directives {
			if d.claims == 0 {
				t.Errorf("%s: -httprecord=%s claims none of the %d cassettes in %s/testdata; `go generate` would record nothing",
					d.file, d.pattern, len(cassettes), pkg)
			}
		}
	}

	t.Logf("checked %d -httprecord directives across %d packages", total, len(byPkg))
}
