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
	"fmt"
	"os"
	"os/exec"
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

type backportFixture struct {
	t         *testing.T
	script    string
	bareDir   string
	cloneDir  string
	binDir    string
	tmpDir    string
	ghLogPath string
}

func newBackportFixture(t *testing.T, baseBranch, baseModule string) *backportFixture {
	t.Helper()

	script, err := filepath.Abs(filepath.Join("..", "scripts", "backport.sh"))
	if err != nil {
		t.Fatalf("resolve script path: %v", err)
	}

	root := t.TempDir()
	bareDir := filepath.Join(root, "remote.git")
	cloneDir := filepath.Join(root, "clone")
	binDir := filepath.Join(root, "bin")
	tmpDir := filepath.Join(root, "tmp")
	ghLogPath := filepath.Join(root, "gh.log")

	for _, dir := range []string{binDir, tmpDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}

	f := &backportFixture{
		t:         t,
		script:    script,
		bareDir:   bareDir,
		cloneDir:  cloneDir,
		binDir:    binDir,
		tmpDir:    tmpDir,
		ghLogPath: ghLogPath,
	}

	f.git(root, "init", "--bare", "--initial-branch="+baseBranch, bareDir)
	f.git(root, "clone", bareDir, cloneDir)
	f.git(cloneDir, "config", "user.name", "Test User")
	f.git(cloneDir, "config", "user.email", "test@example.com")
	f.git(cloneDir, "config", "commit.gpgsign", "false")
	f.git(cloneDir, "checkout", "-B", baseBranch)

	f.writeFile("go.mod", fmt.Sprintf("module %s\n\ngo 1.26.0\n", baseModule))
	f.writeFile("pkg/hello.go", fmt.Sprintf("package pkg\n\nimport _ %q\n\nfunc Hello() string { return \"base\" }\n", baseModule+"/agent"))
	f.git(cloneDir, "add", "-A")
	f.git(cloneDir, "commit", "-m", "chore: initial commit")
	f.git(cloneDir, "push", "-u", "origin", baseBranch)

	return f
}

func (f *backportFixture) cleanEnv(extra ...string) []string {
	env := make([]string, 0, len(os.Environ())+len(extra)+2)
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, "GIT_") || k == "ADK_REPO" || k == "ADK_REMOTE" {
			continue
		}
		env = append(env, kv)
	}
	env = append(env,
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_NOSYSTEM=1",
	)
	return append(env, extra...)
}

func (f *backportFixture) git(dir string, args ...string) string {
	f.t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = f.cleanEnv()
	out, err := cmd.CombinedOutput()
	if err != nil {
		f.t.Fatalf("git %v in %s failed: %v\n%s", args, dir, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (f *backportFixture) gitBare(args ...string) string {
	f.t.Helper()
	full := append([]string{"--git-dir=" + f.bareDir}, args...)
	cmd := exec.Command("git", full...)
	cmd.Env = f.cleanEnv()
	out, err := cmd.CombinedOutput()
	if err != nil {
		f.t.Fatalf("git %v failed: %v\n%s", full, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (f *backportFixture) writeFile(rel, content string) {
	f.t.Helper()
	full := filepath.Join(f.cloneDir, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		f.t.Fatalf("mkdir %s: %v", filepath.Dir(full), err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		f.t.Fatalf("write %s: %v", full, err)
	}
}

func (f *backportFixture) installFakeGH(prNum int, mergeSHA, title string, changedFiles int) {
	f.t.Helper()
	script := fmt.Sprintf(`#!/usr/bin/env bash
set -euo pipefail
printf '%%s\n' "$*" >> %q
case "${1:-}" in
  auth)
    exit 0
    ;;
  api)
    if [[ "${2:-}" == "user" ]]; then
      exit 1
    fi
    exit 0
    ;;
  pr)
    sub="${2:-}"
    shift 2
    case "$sub" in
      list)
        if [[ " $* " == *" --head "* ]]; then
          printf '0\n'
        else
          printf '%%s\t%%s\t%%s\n' %q %q %q
        fi
        ;;
      view)
        if [[ " $* " == *" changedFiles "* ]]; then
          printf '%%d\n' %d
        elif [[ " $* " == *" comments "* ]]; then
          printf '0\n'
        fi
        ;;
      create)
        printf 'https://github.com/google/adk-go/pull/999\n'
        ;;
      edit|comment)
        exit 0
        ;;
      *)
        printf 'unexpected gh pr subcommand: %%s\n' "$sub" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    printf 'unexpected gh command: %%s\n' "$*" >&2
    exit 1
    ;;
esac
`, f.ghLogPath, fmt.Sprintf("%d", prNum), mergeSHA, title, changedFiles)

	ghPath := filepath.Join(f.binDir, "gh")
	if err := os.WriteFile(ghPath, []byte(script), 0o755); err != nil {
		f.t.Fatalf("write fake gh: %v", err)
	}
}

func (f *backportFixture) run(args ...string) (string, error) {
	f.t.Helper()
	cmd := exec.Command(f.script, args...)
	cmd.Dir = f.cloneDir
	cmd.Env = f.cleanEnv(
		"PATH="+f.binDir+":"+os.Getenv("PATH"),
		"ADK_REMOTE=origin",
		"TMPDIR="+f.tmpDir,
	)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func (f *backportFixture) ghLog() string {
	f.t.Helper()
	data, err := os.ReadFile(f.ghLogPath)
	if err != nil {
		f.t.Fatalf("read gh log: %v", err)
	}
	return string(data)
}

func TestBackportTargetMainCleanReplay(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"from-v3\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat(pkg): update greeting on v3")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")
	sha := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.installFakeGH(42, sha, "feat(pkg): update greeting on v3", 1)

	out, err := f.run("--target", "main", "--pr")
	if err != nil {
		t.Fatalf("backport --target main --pr failed: %v\n%s", err, out)
	}

	ghLog := f.ghLog()
	for _, want := range []string{
		"--base v3 --state merged --label feature-parity-main",
		"--head backport/main/pr-42 --state open",
		"pr create --repo google/adk-go --base main --head backport/main/pr-42",
		"Backports #42 from `v3` to `main`.",
		"pr edit 999 --repo google/adk-go --add-label v2",
		"pr edit 999 --repo google/adk-go --add-label feature-parity",
	} {
		if !strings.Contains(ghLog, want) {
			t.Errorf("gh log missing %q\nfull log:\n%s", want, ghLog)
		}
	}
	if strings.Contains(ghLog, "Import paths were rewritten") {
		t.Errorf("PR body should not claim import paths were rewritten when v3 and main share module path\nlog:\n%s", ghLog)
	}

	replayedMsg := f.gitBare("log", "-1", "--format=%B", "refs/heads/backport/main/pr-42")
	wantTrailer := fmt.Sprintf("(cherry picked from commit %s)", sha)
	if !strings.Contains(replayedMsg, wantTrailer) {
		t.Errorf("replayed commit message missing trailer %q:\n%s", wantTrailer, replayedMsg)
	}

	replayedContent := f.gitBare("show", "refs/heads/backport/main/pr-42:pkg/hello.go")
	if !strings.Contains(replayedContent, `import _ "google.golang.org/adk/v2/agent"`) ||
		!strings.Contains(replayedContent, `return "from-v3"`) {
		t.Errorf("unexpected replayed content:\n%s", replayedContent)
	}

	// Merging the backport onto main makes a subsequent run a no-op.
	f.git(f.cloneDir, "fetch", "origin", "backport/main/pr-42")
	f.git(f.cloneDir, "checkout", "main")
	f.git(f.cloneDir, "merge", "--ff-only", "FETCH_HEAD")
	f.git(f.cloneDir, "push", "origin", "main")

	out2, err := f.run("--target", "main", "--list")
	if err != nil {
		t.Fatalf("second --list failed: %v\n%s", err, out2)
	}
	if !strings.Contains(out2, "nothing pending") {
		t.Errorf("expected nothing pending after merge to main, got:\n%s", out2)
	}
}

func TestBackportTargetMainConflict(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"v3-change\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "fix(pkg): v3 change")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")
	sha := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"main-conflict\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "fix(pkg): conflicting main change")
	f.git(f.cloneDir, "push", "origin", "main")

	f.installFakeGH(42, sha, "fix(pkg): v3 change", 1)

	out, err := f.run("--target", "main", "--pr")
	if err != nil {
		t.Fatalf("conflict run should exit 0, got %v\n%s", err, out)
	}

	ghLog := f.ghLog()
	for _, want := range []string{
		"pr comment 42 --repo google/adk-go --body <!-- adk-backport-conflict:main -->",
		"scripts/backport.sh --target main 42",
		"gh pr create --repo google/adk-go --base main --head backport/main/pr-42",
	} {
		if !strings.Contains(ghLog, want) {
			t.Errorf("conflict comment missing %q\nfull log:\n%s", want, ghLog)
		}
	}
	if strings.Contains(ghLog, "pr create --repo google/adk-go --base main --head backport/main/pr-42 --title") {
		t.Errorf("should not open a PR on conflict\nlog:\n%s", ghLog)
	}
}

func TestBackportTargetMainRewritesWhenV3ModuleDiffers(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.writeFile("go.mod", "module google.golang.org/adk/v3\n\ngo 1.26.0\n")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v3/agent\"\n\nfunc Hello() string { return \"base\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "chore: bump v3 module path")

	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v3/agent\"\n\nfunc Hello() string { return \"v3-mod\" }\n")
	f.writeFile("sub/go.mod", "module google.golang.org/adk/sub\n\nrequire google.golang.org/adk/v3 v3.0.0\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat(pkg): update greeting after v3 bump\n\nUpdates google.golang.org/adk/v3/agent callers.")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")
	sha := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.installFakeGH(43, sha, "feat(pkg): update greeting after v3 bump", 2)

	out, err := f.run("--target", "main", "--pr")
	if err != nil {
		t.Fatalf("backport --target main --pr failed: %v\n%s", err, out)
	}

	replayedContent := f.gitBare("show", "refs/heads/backport/main/pr-43:pkg/hello.go")
	if !strings.Contains(replayedContent, `import _ "google.golang.org/adk/v2/agent"`) ||
		!strings.Contains(replayedContent, `return "v3-mod"`) {
		t.Errorf("expected module path rewrite from v3 to v2, got:\n%s", replayedContent)
	}

	replayedSubMod := f.gitBare("show", "refs/heads/backport/main/pr-43:sub/go.mod")
	if !strings.Contains(replayedSubMod, "require google.golang.org/adk/v3 v3.0.0") {
		t.Errorf("expected v3 go.mod require line to stay intact (not downgraded to invalid v2 v3.0.0), got:\n%s", replayedSubMod)
	}

	replayedMsg := f.gitBare("log", "-1", "--format=%B", "refs/heads/backport/main/pr-43")
	if !strings.Contains(replayedMsg, "Updates google.golang.org/adk/v2/agent callers.") {
		t.Errorf("expected commit message body module path to be rewritten, got:\n%s", replayedMsg)
	}

	ghLog := f.ghLog()
	if !strings.Contains(ghLog, "Import paths were rewritten from `google.golang.org/adk/v3` to `google.golang.org/adk/v2`") {
		t.Errorf("PR body should note module rewrite when v3 and main differ\nlog:\n%s", ghLog)
	}
}

func TestBackportTargetMainSkipsDirectAncestor(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"merged-directly\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat(pkg): landed on v3")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")
	sha := f.git(f.cloneDir, "rev-parse", "HEAD")

	// Merging v3 directly into main brings sha in as an ancestor without a
	// cherry-pick trailer.
	f.git(f.cloneDir, "checkout", "main")
	f.git(f.cloneDir, "merge", "--no-ff", "v3", "-m", "chore: merge v3 into main")
	f.git(f.cloneDir, "push", "origin", "main")

	f.installFakeGH(44, sha, "feat(pkg): landed on v3", 1)

	out, err := f.run("--target", "main", "--list")
	if err != nil {
		t.Fatalf("--target main --list failed: %v\n%s", err, out)
	}
	if !strings.Contains(out, "nothing pending") {
		t.Errorf("expected ancestor commit on main to be skipped, got:\n%s", out)
	}
}

func TestBackportGoModParsingAndValidation(t *testing.T) {
	t.Run("strips inline comment without running sed command", func(t *testing.T) {
		f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.writeFile("go.mod", "module google.golang.org/adk/v2//|e;#/v3\n\ngo 1.26.0\n")
		f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"safe\" }\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat(pkg): inline comment in go.mod")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")
		sha := f.git(f.cloneDir, "rev-parse", "HEAD")

		f.installFakeGH(45, sha, "feat(pkg): inline comment in go.mod", 2)

		out, err := f.run("--target", "main", "--list")
		if err != nil {
			t.Fatalf("--list should succeed when inline comment is stripped: %v\n%s", err, out)
		}
		if !strings.Contains(out, "#45") {
			t.Errorf("expected PR #45 in pending queue, got:\n%s", out)
		}
	})

	t.Run("fails closed on invalid module path", func(t *testing.T) {
		f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.writeFile("go.mod", "module google.golang.org/adk|e;id\n\ngo 1.26.0\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "chore: malformed go.mod")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")
		sha := f.git(f.cloneDir, "rev-parse", "HEAD")

		f.installFakeGH(46, sha, "chore: malformed go.mod", 1)

		out, err := f.run("--target", "main", "--list")
		if err == nil {
			t.Fatalf("expected error on invalid module path, got nil\n%s", out)
		}
		if !strings.Contains(out, "invalid or missing module path") {
			t.Errorf("expected invalid module path error, got:\n%s", out)
		}
	})
}

func TestBackportHelpShowsConfiguredTarget(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	out, err := f.run("--target", "main", "--help")
	if err != nil {
		t.Fatalf("--target main --help failed: %v\n%s", err, out)
	}
	for _, want := range []string{"'v3'", "'feature-parity-main'", "'main'"} {
		if !strings.Contains(out, want) {
			t.Errorf("--target main --help output missing %s:\n%s", want, out)
		}
	}
}

func TestBackportDefaultTargetV1Preserved(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")

	f.git(f.cloneDir, "checkout", "-b", "v1")
	f.writeFile("go.mod", "module google.golang.org/adk\n\ngo 1.26.0\n")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/agent\"\n\nfunc Hello() string { return \"base\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "chore: v1 module path")
	f.git(f.cloneDir, "push", "-u", "origin", "v1")

	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("pkg/hello.go", "package pkg\n\nimport _ \"google.golang.org/adk/v2/agent\"\n\nfunc Hello() string { return \"from-main\" }\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "fix(pkg): update greeting on main")
	f.git(f.cloneDir, "push", "origin", "main")
	sha := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.installFakeGH(10, sha, "fix(pkg): update greeting on main", 1)

	out, err := f.run("--pr")
	if err != nil {
		t.Fatalf("default backport --pr failed: %v\n%s", err, out)
	}

	ghLog := f.ghLog()
	for _, want := range []string{
		"--base main --state merged --label v1-needed",
		"pr create --repo google/adk-go --base v1 --head backport/v1/pr-10",
		"Import paths were rewritten from `google.golang.org/adk/v2` to `google.golang.org/adk`",
		"pr edit 999 --repo google/adk-go --add-label v1",
	} {
		if !strings.Contains(ghLog, want) {
			t.Errorf("gh log missing %q\nfull log:\n%s", want, ghLog)
		}
	}

	replayedContent := f.gitBare("show", "refs/heads/backport/v1/pr-10:pkg/hello.go")
	if !strings.Contains(replayedContent, `import _ "google.golang.org/adk/agent"`) ||
		!strings.Contains(replayedContent, `return "from-main"`) {
		t.Errorf("expected v1 module path rewrite, got:\n%s", replayedContent)
	}
}

func TestBackportInvalidTarget(t *testing.T) {
	f := newBackportFixture(t, "main", "google.golang.org/adk/v2")
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{name: "unknown target", args: []string{"--target", "v99"}, want: "unknown target"},
		{name: "missing target value", args: []string{"--target"}, want: "--target requires a value"},
		{name: "empty equals value", args: []string{"--target="}, want: "--target requires a value"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := f.run(tc.args...)
			if err == nil {
				t.Fatalf("expected error for %v, got nil\n%s", tc.args, out)
			}
			if !strings.Contains(out, tc.want) {
				t.Errorf("expected error containing %q, got:\n%s", tc.want, out)
			}
		})
	}
}

func TestBackportMainWorkflow(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "backport-main.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	wf := string(data)

	for _, want := range []string{
		"branches: [v3]",
		"schedule:",
		"workflow_dispatch:",
		"group: backport-main",
		"if: github.repository == 'google/adk-go'",
		`scripts/backport.sh --target main "${args[@]}" --pr`,
	} {
		if !strings.Contains(wf, want) {
			t.Errorf("%s missing %q", path, want)
		}
	}
	for _, forbidden := range []string{"pull_request_target", "workflow_run"} {
		for line := range strings.SplitSeq(wf, "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "#") {
				continue
			}
			if strings.Contains(trimmed, forbidden) {
				t.Errorf("%s must not use %q (line: %s)", path, forbidden, trimmed)
			}
		}
	}
}
