// Copyright 2025 Google LLC
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
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const copyrightHeader = `// Copyright 2025 Google LLC
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
`

var fixError = flag.Bool("fix", false, "fix detected problems (e.g. add missing copyright headers)")

func TestCopyrightHeader(t *testing.T) {
	// Start test from the parent directory, root of the module.
	t.Chdir("..")

	ignore := map[string]bool{
		// Skip directories that are not relevant for copyright checks.
		// The followings were copied from golang.org/x/tools.
		"internal/jsonschema": true,
		"internal/util":       true,
		// The following was copied from golang.org/x/oscar.
		"internal/httprr": true,
	}
	_ = filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if ignore[path] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		hasHeader, err := hasCopyrightHeader(path)
		switch {
		case err != nil:
			t.Errorf("failed to check file %q: %v", path, err)
		case !hasHeader && !*fixError:
			t.Errorf("file %q does not have the copyright header", path)
		case !hasHeader && *fixError:
			t.Logf("updating file %q with copyright header", path)
			if err := addCopyrightHeader(path); err != nil {
				t.Errorf("failed to update file %q: %v", path, err)
			}
		}
		return nil
	})
}

func hasCopyrightHeader(path string) (bool, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}
	return strings.HasPrefix(string(content), copyrightHeader), nil
}

func addCopyrightHeader(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	newContent := []byte(copyrightHeader)
	newContent = append(newContent, content...)
	return os.WriteFile(path, newContent, 0o644)
}
