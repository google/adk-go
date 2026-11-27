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
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
// A behavioral test would need a live pull request, so this pins the property
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

// TestBackportSkipsLeftoverBranch runs scripts/backport.sh against a local bare
// repository with two queued PRs. #7 already has a backport/v1/pr-7 branch and
// no open backport pull request, which is what a backport closed unmerged, or a
// run killed before `gh pr create`, leaves behind. A fresh replay cannot be
// pushed over it, and that used to fail every run until the branch was deleted.
// #8 has no branch yet and must still be backported.
//
// gh is a stub that answers only the calls the script is expected to make, so
// the test needs no network and no credentials. Any other call fails the test,
// including one the script only warns about.
func TestBackportSkipsLeftoverBranch(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("scripts/backport.sh needs bash and a POSIX gh stub")
	}
	realGit, err := exec.LookPath("git")
	if err != nil {
		t.Skipf("git is not installed: %v", err)
	}
	if err := exec.Command("bash", "-c", "((BASH_VERSINFO[0]*100 + BASH_VERSINFO[1] >= 404))").Run(); err != nil {
		t.Skipf("scripts/backport.sh needs bash >= 4.4: %v", err)
	}
	script, err := filepath.Abs(filepath.Join("..", "scripts", "backport.sh"))
	if err != nil {
		t.Fatal(err)
	}

	tmp := t.TempDir()
	remote := filepath.Join(tmp, "remote.git")
	work := filepath.Join(tmp, "work")
	bin := filepath.Join(tmp, "bin")

	// Dropped rather than overridden: a GIT_DIR or GIT_INDEX_FILE inherited from
	// a git hook that runs the tests would point every command below at the
	// developer's repository.
	var env []string
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "GIT_") {
			env = append(env, kv)
		}
	}
	// git reads ignore and attributes files from $XDG_CONFIG_HOME/git or
	// ~/.config/git even when no config file names them, so HOME and
	// XDG_CONFIG_HOME point at the temporary directory, which has neither.
	env = append(env,
		"HOME="+tmp,
		"XDG_CONFIG_HOME="+tmp,
		"GIT_CONFIG_GLOBAL="+os.DevNull,
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_ATTR_NOSYSTEM=1",
	)

	git := func(t *testing.T, dir string, args ...string) string {
		t.Helper()
		cmd := exec.CommandContext(t.Context(), "git", args...)
		cmd.Dir = dir
		cmd.Env = env
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	write := func(name, content string, perm os.FileMode) {
		t.Helper()
		if err := os.WriteFile(name, []byte(content), perm); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(file, content, message string) string {
		t.Helper()
		write(filepath.Join(work, file), content, 0o644)
		git(t, work, "add", file)
		git(t, work, "commit", "-q", "-m", message)
		return git(t, work, "rev-parse", "HEAD")
	}

	git(t, tmp, "init", "-q", "--bare", remote)
	git(t, tmp, "init", "-q", "-b", "main", work)
	git(t, work, "config", "user.name", "Test")
	git(t, work, "config", "user.email", "test@example.com")
	git(t, work, "remote", "add", "origin", remote)

	base := commit("a.txt", "one\n", "base")
	git(t, work, "push", "-q", "origin", "main", "main:v1")
	// The squash commits of #7 and #8, each of which applies cleanly to v1.
	pr7 := commit("a.txt", "two\n", "fix: change a (#7)")
	pr8 := commit("c.txt", "new\n", "feat: add c (#8)")
	git(t, work, "push", "-q", "origin", "main")

	git(t, work, "checkout", "-q", "--detach", base)
	leftover := commit("b.txt", "stale\n", "earlier backport of #7")
	git(t, work, "push", "-q", "origin", "HEAD:refs/heads/backport/v1/pr-7")
	git(t, work, "checkout", "-q", "main")

	if err := os.Mkdir(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(bin, "gh"), `#!/bin/sh
case "$*" in
"auth status") ;;
*"--state merged"*) printf '7\t%s\tfix: change a\n8\t%s\tfeat: add c\n' "$STUB_PR7" "$STUB_PR8" ;;
*"--state open"*) echo 0 ;;
*changedFiles*) echo 1 ;;
"pr create "*) echo https://github.com/google/adk-go/pull/100 ;;
"pr edit 100 "*) ;;
*) echo "gh stub: unexpected call: $*" | tee -a "$STUB_UNEXPECTED" >&2; exit 1 ;;
esac
`, 0o755)

	// A git that fails ls-remote and passes everything else to the real one.
	brokenLsRemote := filepath.Join(tmp, "broken-ls-remote")
	if err := os.Mkdir(brokenLsRemote, 0o755); err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(brokenLsRemote, "git"), `#!/bin/sh
if [ "$1" = ls-remote ]; then
  echo "git stub: ls-remote failed" >&2
  exit 128
fi
exec "$REAL_GIT" "$@"
`, 0o755)

	// run runs the script with dirs ahead of PATH.
	run := func(t *testing.T, dirs string, args ...string) (string, error) {
		t.Helper()
		unexpected := filepath.Join(t.TempDir(), "unexpected")
		cmd := exec.CommandContext(t.Context(), "bash", append([]string{script}, args...)...)
		cmd.Dir = work
		cmd.Env = append(env,
			"PATH="+dirs+string(os.PathListSeparator)+os.Getenv("PATH"),
			"TMPDIR="+tmp,
			"ADK_REMOTE=origin",
			"REAL_GIT="+realGit,
			"STUB_PR7="+pr7,
			"STUB_PR8="+pr8,
			"STUB_UNEXPECTED="+unexpected,
		)
		out, err := cmd.CombinedOutput()
		// The script only warns on some gh failures, such as the label call, so
		// an unexpected call can leave the run green.
		if calls, readErr := os.ReadFile(unexpected); readErr == nil {
			t.Errorf("backport.sh %v made gh calls the stub does not expect:\n%s\noutput:\n%s", args, calls, out)
		} else if !errors.Is(readErr, fs.ErrNotExist) {
			t.Fatal(readErr)
		}
		return string(out), err
	}

	t.Run("pr", func(t *testing.T) {
		out, err := run(t, bin, "--pr")
		if err != nil {
			t.Fatalf("backport.sh --pr failed the run: %v\n%s", err, out)
		}
		for _, want := range []string{
			"backported 1, conflicted 0, empty 0, stale 1, failed 0",
			"backport branch already pushed with no pull request open when this run started: #7\n",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("backport.sh --pr output does not contain %q:\n%s", want, out)
			}
		}
		if got := git(t, tmp, "ls-remote", remote, "refs/heads/backport/v1/pr-7"); !strings.HasPrefix(got, leftover) {
			t.Errorf("backport/v1/pr-7 = %q, want it left at %s", got, leftover)
		}
		if got := git(t, tmp, "ls-remote", remote, "refs/heads/backport/v1/pr-8"); got == "" {
			t.Error("backport/v1/pr-8 was not pushed")
		}
	})

	// A local replay pushes nothing, so the leftover branch is no reason to skip.
	t.Run("local", func(t *testing.T) {
		out, err := run(t, bin)
		if err != nil {
			t.Fatalf("backport.sh failed the run: %v\n%s", err, out)
		}
		if want := "backported 2, conflicted 0, empty 0, stale 0, failed 0"; !strings.Contains(out, want) {
			t.Errorf("backport.sh output does not contain %q:\n%s", want, out)
		}
	})

	// A branch check that got no answer must fail the run, not pass as stale.
	t.Run("ls-remote fails", func(t *testing.T) {
		out, err := run(t, brokenLsRemote+string(os.PathListSeparator)+bin, "--pr")
		if err == nil {
			t.Errorf("backport.sh --pr succeeded with ls-remote failing:\n%s", out)
		}
		for _, want := range []string{
			"could not check whether backport/v1/pr-7 already exists",
			"backported 0, conflicted 0, empty 0, stale 0, failed 2",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("backport.sh --pr output does not contain %q:\n%s", want, out)
			}
		}
	})
}
