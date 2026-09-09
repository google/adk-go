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
	"strings"
	"testing"
)

// TestBackportConflictCommentMatchesBothBotSpellings pins the author filter that
// stops scripts/backport.sh re-posting its conflict comment on every run.
//
// gh's two API surfaces disagree about what a bot is called. `gh pr view --json
// comments` is GraphQL and reports the author of a bot comment as
// "github-actions"; the REST API reports "github-actions[bot]". The filter runs
// against the GraphQL shape, so a filter naming only the REST spelling selects
// nothing, the script concludes it has never commented, and it comments again.
// That shipped: three contributors' pull requests collected one comment per run
// for nine runs before it was noticed.
//
// A behavioural test would need a live pull request, so this pins the property
// that actually broke: whichever spelling gh returns, the filter matches it.
func TestBackportConflictCommentMatchesBothBotSpellings(t *testing.T) {
	path := filepath.Join("..", "scripts", "backport.sh")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	script := string(data)

	// The one jq program that decides whether the comment was already posted.
	const filterAnchor = "select(.body | contains("
	if !strings.Contains(script, filterAnchor) {
		t.Fatalf("%s no longer contains the conflict-comment filter (looked for %q); "+
			"update this test to point at wherever the dedup moved", path, filterAnchor)
	}

	for _, login := range []string{
		`.author.login == "github-actions"`,      // GraphQL, what gh pr view returns
		`.author.login == "github-actions[bot]"`, // REST, and what a human reads in the UI
	} {
		if !strings.Contains(script, login) {
			t.Errorf("%s: conflict-comment filter does not match %s.\n"+
				"Both spellings are required: gh reports the bot differently through "+
				"GraphQL and REST, and matching only one makes the dedup select nothing, "+
				"so the script re-comments on every run.", path, login)
		}
	}
}
