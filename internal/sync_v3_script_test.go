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

type syncV3Fixture struct {
	t         *testing.T
	script    string
	bareDir   string
	cloneDir  string
	binDir    string
	tmpDir    string
	ghLogPath string
	stateDir  string
}

func newSyncV3Fixture(t *testing.T) *syncV3Fixture {
	t.Helper()

	script, err := filepath.Abs(filepath.Join("..", "scripts", "sync-v3.sh"))
	if err != nil {
		t.Fatalf("resolve script path: %v", err)
	}

	root := t.TempDir()
	bareDir := filepath.Join(root, "remote.git")
	cloneDir := filepath.Join(root, "clone")
	binDir := filepath.Join(root, "bin")
	tmpDir := filepath.Join(root, "tmp")
	stateDir := filepath.Join(root, "gh-state")
	ghLogPath := filepath.Join(root, "gh.log")

	for _, dir := range []string{binDir, tmpDir, stateDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}

	f := &syncV3Fixture{
		t:         t,
		script:    script,
		bareDir:   bareDir,
		cloneDir:  cloneDir,
		binDir:    binDir,
		tmpDir:    tmpDir,
		ghLogPath: ghLogPath,
		stateDir:  stateDir,
	}

	f.git(root, "init", "--bare", "--initial-branch=main", bareDir)
	f.git(root, "clone", bareDir, cloneDir)
	f.git(cloneDir, "config", "user.name", "Test User")
	f.git(cloneDir, "config", "user.email", "test@example.com")
	f.git(cloneDir, "config", "commit.gpgsign", "false")
	f.git(cloneDir, "checkout", "-B", "main")

	f.writeFile("go.mod", "module google.golang.org/adk/v2\n\ngo 1.26.0\n")
	f.writeFile("shared.go", "package root\n\nconst Version = \"v0\"\n")
	f.git(cloneDir, "add", "-A")
	f.git(cloneDir, "commit", "-m", "chore: initial commit")
	f.git(cloneDir, "push", "-u", "origin", "main")

	f.installFakeGH()
	return f
}

func (f *syncV3Fixture) cleanEnv(extra ...string) []string {
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

func (f *syncV3Fixture) git(dir string, args ...string) string {
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

func (f *syncV3Fixture) gitBare(args ...string) string {
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

func (f *syncV3Fixture) writeFile(rel, content string) {
	f.t.Helper()
	full := filepath.Join(f.cloneDir, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		f.t.Fatalf("mkdir %s: %v", filepath.Dir(full), err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		f.t.Fatalf("write %s: %v", full, err)
	}
}

func (f *syncV3Fixture) setOpenPR(prNum string) {
	f.t.Helper()
	path := filepath.Join(f.stateDir, "open_pr")
	if prNum == "" {
		_ = os.Remove(path)
		return
	}
	if err := os.WriteFile(path, []byte(prNum+"\n"), 0o644); err != nil {
		f.t.Fatalf("write open_pr: %v", err)
	}
}

func (f *syncV3Fixture) installFakeGH() {
	f.t.Helper()
	script := fmt.Sprintf(`#!/usr/bin/env bash
set -euo pipefail
LOG=%q
STATE=%q
printf '%%s\n' "$*" >> "${LOG}"

case "${1:-}" in
  auth)
    exit 0
    ;;
  api)
    shift
    endpoint="${1:-}"
    if [[ "${endpoint}" == "user" ]]; then
      exit 1
    fi
    # Capture --input payload if provided.
    input_file=""
    prev=""
    for arg in "$@"; do
      if [[ "${prev}" == "--input" ]]; then
        input_file="${arg}"
      fi
      prev="${arg}"
    done
    if [[ -n "${input_file}" ]]; then
      cp "${input_file}" "${STATE}/last_patch.json"
    fi
    printf '{"html_url":"https://github.com/google/adk-go/pull/777"}\n'
    ;;
  pr)
    sub="${2:-}"
    shift 2
    case "${sub}" in
      list)
        if [[ -f "${STATE}/open_pr" ]]; then
          cat "${STATE}/open_pr"
        else
          printf '\n'
        fi
        ;;
      create)
        body_file=""
        prev=""
        for arg in "$@"; do
          if [[ "${prev}" == "--body-file" ]]; then
            body_file="${arg}"
          fi
          prev="${arg}"
        done
        if [[ -n "${body_file}" ]]; then
          cp "${body_file}" "${STATE}/last_create_body.md"
        fi
        printf '777\n' > "${STATE}/open_pr"
        printf 'https://github.com/google/adk-go/pull/777\n'
        ;;
      view)
        if [[ " $* " == *" comments "* ]]; then
          if [[ -f "${STATE}/comments.json" ]]; then
            jq_filter=""
            prev=""
            for arg in "$@"; do
              if [[ "${prev}" == "--jq" ]]; then
                jq_filter="${arg}"
              fi
              prev="${arg}"
            done
            jq -r "${jq_filter}" "${STATE}/comments.json"
          else
            printf '0\n'
          fi
        fi
        ;;
      comment)
        body=""
        prev=""
        for arg in "$@"; do
          if [[ "${prev}" == "--body" ]]; then
            body="${arg}"
          fi
          prev="${arg}"
        done
        jq -n --arg body "${body}" '{comments: [{author: {login: "github-actions"}, body: $body}]}' > "${STATE}/comments.json"
        printf '%%s\n' "${body}" > "${STATE}/last_comment.md"
        ;;
      *)
        printf 'unexpected gh pr subcommand: %%s\n' "${sub}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    printf 'unexpected gh command: %%s\n' "$*" >&2
    exit 1
    ;;
esac
`, f.ghLogPath, f.stateDir)

	ghPath := filepath.Join(f.binDir, "gh")
	if err := os.WriteFile(ghPath, []byte(script), 0o755); err != nil {
		f.t.Fatalf("write fake gh: %v", err)
	}
}

func (f *syncV3Fixture) run(args ...string) (string, error) {
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

func (f *syncV3Fixture) ghLog() string {
	f.t.Helper()
	data, err := os.ReadFile(f.ghLogPath)
	if err != nil {
		if os.IsNotExist(err) {
			return ""
		}
		f.t.Fatalf("read gh log: %v", err)
	}
	return string(data)
}

func (f *syncV3Fixture) clearGHLog() {
	f.t.Helper()
	_ = os.Remove(f.ghLogPath)
}

func (f *syncV3Fixture) readState(name string) string {
	f.t.Helper()
	data, err := os.ReadFile(filepath.Join(f.stateDir, name))
	if err != nil {
		f.t.Fatalf("read state %s: %v", name, err)
	}
	return string(data)
}

func TestSyncV3BootstrapFromSquashedHistory(t *testing.T) {
	f := newSyncV3Fixture(t)

	// Cut v3 from initial main commit (c0).
	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")

	// Advance main with two commits (c1, c2).
	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("shared.go", "package root\n\nconst Version = \"v1\"\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: step 1")

	f.writeFile("shared.go", "package root\n\nconst Version = \"v2\"\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: step 2")
	c2Tree := f.git(f.cloneDir, "rev-parse", "HEAD^{tree}")

	// Simulate an earlier manual squash-merge of main (up to c2) onto v3 without
	// any Synced-from-main trailer, followed by a v3-only commit.
	f.git(f.cloneDir, "checkout", "v3")
	squashSHA := f.git(f.cloneDir, "commit-tree", c2Tree, "-p", "HEAD", "-m", "V3 rebase (#2431)")
	f.git(f.cloneDir, "reset", "--hard", squashSHA)
	f.writeFile("v3only.go", "package root\n\nconst V3Only = true\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat(v3): add v3-only file")
	f.git(f.cloneDir, "push", "origin", "v3")

	// Advance main past c2, modifying shared.go again (which would conflict if
	// merge-base stayed at c0!).
	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("shared.go", "package root\n\nconst Version = \"v3-from-main\"\n")
	f.writeFile("newmain.go", "package root\n\nconst NewOnMain = true\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: step 3 after squash")
	f.git(f.cloneDir, "push", "origin", "main")
	mainSHA := f.git(f.cloneDir, "rev-parse", "HEAD")

	out, err := f.run("--pr")
	if err != nil {
		t.Fatalf("sync-v3 --pr failed: %v\n%s", err, out)
	}

	if got := f.gitBare("show", "refs/heads/sync/main-to-v3:shared.go"); !strings.Contains(got, `"v3-from-main"`) {
		t.Errorf("expected shared.go to carry main's v3-from-main change, got:\n%s", got)
	}
	if got := f.gitBare("show", "refs/heads/sync/main-to-v3:v3only.go"); !strings.Contains(got, "V3Only = true") {
		t.Errorf("expected v3only.go to be preserved, got:\n%s", got)
	}
	if got := f.gitBare("show", "refs/heads/sync/main-to-v3:newmain.go"); !strings.Contains(got, "NewOnMain = true") {
		t.Errorf("expected newmain.go to be synced, got:\n%s", got)
	}

	msg := f.gitBare("log", "-1", "--format=%B", "refs/heads/sync/main-to-v3")
	wantTrailer := fmt.Sprintf("Synced-from-main: %s", mainSHA)
	if !strings.Contains(msg, wantTrailer) {
		t.Errorf("sync commit message missing trailer %q:\n%s", wantTrailer, msg)
	}
	if !strings.HasPrefix(msg, fmt.Sprintf("chore(v3): sync main (%s)", mainSHA[:9])) {
		t.Errorf("unexpected commit subject:\n%s", msg)
	}

	ghLog := f.ghLog()
	if !strings.Contains(ghLog, "pr create --repo google/adk-go --base v3 --head sync/main-to-v3") {
		t.Errorf("expected gh pr create targeting v3, got log:\n%s", ghLog)
	}
	body := f.readState("last_create_body.md")
	if !strings.Contains(body, "feat: step 3 after squash") || !strings.Contains(body, mainSHA[:9]) {
		t.Errorf("unexpected PR body:\n%s", body)
	}
}

func TestSyncV3SubsequentSyncViaTrailerAndIdempotency(t *testing.T) {
	f := newSyncV3Fixture(t)

	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")

	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("a.go", "package root\n\nconst A = 1\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: add A")
	f.git(f.cloneDir, "push", "origin", "main")

	// First sync opens PR #777.
	if out, err := f.run("--pr"); err != nil {
		t.Fatalf("first sync failed: %v\n%s", err, out)
	}

	// Re-running with nothing changed must be a complete no-op (no push, no PATCH, no create).
	f.clearGHLog()
	beforeSyncSHA := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3")
	out, err := f.run("--pr")
	if err != nil {
		t.Fatalf("idempotent re-run failed: %v\n%s", err, out)
	}
	if !strings.Contains(out, "already up to date") {
		t.Errorf("expected already up to date message, got:\n%s", out)
	}
	if afterSyncSHA := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3"); afterSyncSHA != beforeSyncSHA {
		t.Errorf("idempotent re-run changed sync branch SHA from %s to %s", beforeSyncSHA, afterSyncSHA)
	}
	if ghLog := f.ghLog(); strings.Contains(ghLog, "pr create") || strings.Contains(ghLog, "PATCH") {
		t.Errorf("idempotent re-run should not create or PATCH PR, got log:\n%s", ghLog)
	}

	// While PR #777 is still open, main advances again -> updates PR #777 via REST PATCH.
	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("b.go", "package root\n\nconst B = 2\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: add B")
	f.git(f.cloneDir, "push", "origin", "main")
	mainBSHA := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.clearGHLog()
	out, err = f.run("--pr")
	if err != nil {
		t.Fatalf("rolling PR update failed: %v\n%s", err, out)
	}
	ghLog := f.ghLog()
	if !strings.Contains(ghLog, "api repos/google/adk-go/pulls/777 --method PATCH") {
		t.Errorf("expected REST PATCH of existing PR #777, got log:\n%s", ghLog)
	}
	if strings.Contains(ghLog, "pr create") || strings.Contains(ghLog, "pr edit") {
		t.Errorf("rolling update must not call pr create or pr edit, got log:\n%s", ghLog)
	}
	patchJSON := f.readState("last_patch.json")
	if !strings.Contains(patchJSON, "feat: add B") || !strings.Contains(patchJSON, mainBSHA[:9]) {
		t.Errorf("PATCH payload missing new commit info:\n%s", patchJSON)
	}

	// Simulate human merging PR #777 into v3 (fast-forward/squash preserves message).
	f.git(f.cloneDir, "fetch", "origin", "sync/main-to-v3")
	f.git(f.cloneDir, "checkout", "v3")
	f.git(f.cloneDir, "merge", "--ff-only", "FETCH_HEAD")
	f.git(f.cloneDir, "push", "origin", "v3")
	f.setOpenPR("")

	// With main == last_synced, next run is a no-op.
	f.clearGHLog()
	out, err = f.run("--pr")
	if err != nil {
		t.Fatalf("no-op sync after merge failed: %v\n%s", err, out)
	}
	if !strings.Contains(out, "already up to date") {
		t.Errorf("expected already up to date after merge, got:\n%s", out)
	}
	if strings.Contains(f.ghLog(), "pr create") {
		t.Errorf("should not open PR when already synced:\n%s", f.ghLog())
	}

	// Add a new commit on both main and v3 -> next sync uses the Synced-from-main trailer.
	f.writeFile("v3extra.go", "package root\n\nconst V3Extra = 3\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat(v3): extra")
	f.git(f.cloneDir, "push", "origin", "v3")

	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("c.go", "package root\n\nconst C = 4\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: add C")
	f.git(f.cloneDir, "push", "origin", "main")
	mainCSHA := f.git(f.cloneDir, "rev-parse", "HEAD")

	f.clearGHLog()
	out, err = f.run("--pr")
	if err != nil {
		t.Fatalf("subsequent sync failed: %v\n%s", err, out)
	}
	if got := f.gitBare("show", "refs/heads/sync/main-to-v3:c.go"); !strings.Contains(got, "const C = 4") {
		t.Errorf("expected c.go on sync branch, got:\n%s", got)
	}
	if got := f.gitBare("show", "refs/heads/sync/main-to-v3:v3extra.go"); !strings.Contains(got, "const V3Extra = 3") {
		t.Errorf("expected v3extra.go preserved, got:\n%s", got)
	}
	if msg := f.gitBare("log", "-1", "--format=%B", "refs/heads/sync/main-to-v3"); !strings.Contains(msg, "Synced-from-main: "+mainCSHA) {
		t.Errorf("expected updated trailer for %s, got:\n%s", mainCSHA, msg)
	}
}

func TestSyncV3ConflictHandling(t *testing.T) {
	setupConflict := func(t *testing.T) *syncV3Fixture {
		t.Helper()
		f := newSyncV3Fixture(t)
		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.writeFile("shared.go", "package root\n\nconst Version = \"v3-edit\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(v3): v3 version")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")

		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("shared.go", "package root\n\nconst Version = \"main-edit\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(main): main version")
		f.git(f.cloneDir, "push", "origin", "main")
		return f
	}

	t.Run("pr mode without open PR fails loudly and does not push", func(t *testing.T) {
		f := setupConflict(t)
		out, err := f.run("--pr")
		if err == nil {
			t.Fatalf("expected conflict to exit non-zero in --pr mode\n%s", out)
		}
		if !strings.Contains(out, "shared.go") {
			t.Errorf("expected conflicting file shared.go in output:\n%s", out)
		}
		if refs := f.gitBare("for-each-ref", "refs/heads/sync/main-to-v3"); refs != "" {
			t.Errorf("must not push sync/main-to-v3 on conflict, found: %s", refs)
		}
	})

	t.Run("pr mode with open PR comments once and deduplicates", func(t *testing.T) {
		f := setupConflict(t)
		f.setOpenPR("777")

		out, err := f.run("--pr")
		if err == nil {
			t.Fatalf("expected conflict to exit non-zero\n%s", out)
		}
		comment := f.readState("last_comment.md")
		if !strings.Contains(comment, "shared.go") || !strings.Contains(comment, "<!-- adk-sync-v3-conflict:") {
			t.Errorf("unexpected conflict comment:\n%s", comment)
		}

		// Second run at the same (v3, main) pair must not post another comment.
		f.clearGHLog()
		_, err = f.run("--pr")
		if err == nil {
			t.Fatalf("expected second conflict run to exit non-zero")
		}
		if strings.Contains(f.ghLog(), "pr comment") {
			t.Errorf("expected conflict comment to be deduplicated, got log:\n%s", f.ghLog())
		}
	})

	t.Run("local mode populates unmerged index stages and explicit push lease", func(t *testing.T) {
		f := setupConflict(t)

		// Also add a modify/delete conflict (v3 deletes legacy.go, main edits it).
		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("legacy.go", "package root\n\nconst Legacy = \"base\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "chore: add legacy.go")
		baseWithLegacy := f.git(f.cloneDir, "rev-parse", "HEAD")

		f.git(f.cloneDir, "checkout", "v3")
		f.git(f.cloneDir, "reset", "--hard", baseWithLegacy)
		f.git(f.cloneDir, "rm", "legacy.go")
		f.writeFile("shared.go", "package root\n\nconst Version = \"v3-edit2\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(v3): drop legacy and edit shared")
		f.git(f.cloneDir, "push", "--force", "origin", "v3")

		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("legacy.go", "package root\n\nconst Legacy = \"main-edit\"\n")
		f.writeFile("shared.go", "package root\n\nconst Version = \"main-edit2\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(main): edit legacy and shared")
		f.git(f.cloneDir, "push", "origin", "main")

		// Seed an existing remote sync/main-to-v3 branch to confirm the printed
		// explicit lease matches it and works without stale-info rejection.
		f.git(f.cloneDir, "push", "origin", "v3:refs/heads/sync/main-to-v3")
		oldSyncSHA := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3")

		out, err := f.run()
		if err == nil {
			t.Fatalf("expected local conflict run to exit non-zero\n%s", out)
		}
		wantLease := fmt.Sprintf("--force-with-lease=refs/heads/sync/main-to-v3:%s", oldSyncSHA)
		if !strings.Contains(out, wantLease) {
			t.Errorf("expected explicit lease %q in output:\n%s", wantLease, out)
		}
		if !strings.Contains(out, "Synced-from-main:") {
			t.Errorf("expected manual resolution instructions with Synced-from-main trailer:\n%s", out)
		}

		wtDir := filepath.Join(f.tmpDir, "adk-sync-v3")
		statusOut := f.git(wtDir, "status", "--short")
		if !strings.Contains(statusOut, "DU legacy.go") || !strings.Contains(statusOut, "UU shared.go") {
			t.Errorf("expected unmerged DU legacy.go and UU shared.go in git status, got:\n%s", statusOut)
		}

		// Attempting to commit before resolving must be refused by git.
		commitCmd := exec.Command("git", "commit", "-m", "should fail")
		commitCmd.Dir = wtDir
		commitCmd.Env = f.cleanEnv()
		if cOut, cErr := commitCmd.CombinedOutput(); cErr == nil {
			t.Fatalf("git commit should fail while unmerged entries exist, output:\n%s", cOut)
		}

		// Resolve conflicts, commit, and verify the printed explicit lease pushes cleanly.
		f.git(wtDir, "rm", "-f", "legacy.go")
		if err := os.WriteFile(filepath.Join(wtDir, "shared.go"), []byte("package root\n\nconst Version = \"resolved\"\n"), 0o644); err != nil {
			t.Fatalf("resolve shared.go: %v", err)
		}
		f.git(wtDir, "add", "shared.go")
		f.git(wtDir, "commit", "-m", "chore(v3): sync main (resolved)\n\nSynced-from-main: "+f.git(f.cloneDir, "rev-parse", "origin/main"))
		f.git(wtDir, "push", wantLease, "origin", "HEAD:refs/heads/sync/main-to-v3")
	})
}

func TestSyncV3GuardsAndEdgeCases(t *testing.T) {
	t.Run("does not clobber manual resolution or fixup commits on open PR", func(t *testing.T) {
		f := newSyncV3Fixture(t)
		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.writeFile("shared.go", "package root\n\nconst Version = \"v3-edit\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(v3): v3 version")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")

		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("shared.go", "package root\n\nconst Version = \"main-edit\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "fix(main): main version")
		f.git(f.cloneDir, "push", "origin", "main")
		main1 := f.git(f.cloneDir, "rev-parse", "HEAD")

		// Maintainer manually resolves the conflict on sync/main-to-v3 and opens PR #777.
		f.git(f.cloneDir, "checkout", "v3")
		f.git(f.cloneDir, "checkout", "-b", "sync/main-to-v3")
		f.writeFile("shared.go", "package root\n\nconst Version = \"manual-resolution\"\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", fmt.Sprintf("chore(v3): sync main (%s)\n\nSynced-from-main: %s", main1[:9], main1))
		f.git(f.cloneDir, "push", "origin", "sync/main-to-v3")
		manualSHA := f.git(f.cloneDir, "rev-parse", "HEAD")
		f.setOpenPR("777")

		// Re-run at same main1 -> leaves manual resolution untouched and exits 0.
		out, err := f.run("--pr")
		if err != nil {
			t.Fatalf("expected manual resolution at current main to exit 0, got %v\n%s", err, out)
		}
		if !strings.Contains(out, "manual resolution") {
			t.Errorf("expected manual resolution notice, got:\n%s", out)
		}
		if got := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3"); got != manualSHA {
			t.Errorf("manual resolution SHA changed from %s to %s", manualSHA, got)
		}

		// Main advances to main2 -> script must refuse to overwrite manual commit.
		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("other.go", "package root\n\nconst Other = 1\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat: other")
		f.git(f.cloneDir, "push", "origin", "main")

		out, err = f.run("--pr")
		if err == nil {
			t.Fatalf("expected script to refuse clobbering manual commits when main advances\n%s", out)
		}
		if !strings.Contains(out, "refusing to force-overwrite") {
			t.Errorf("expected refusal message, got:\n%s", out)
		}
		if got := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3"); got != manualSHA {
			t.Errorf("sync branch was clobbered: got %s, want %s", got, manualSHA)
		}
	})

	t.Run("rebases open sync PR when v3 advances", func(t *testing.T) {
		f := newSyncV3Fixture(t)
		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")

		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("a.go", "package root\n\nconst A = 1\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat: add A")
		f.git(f.cloneDir, "push", "origin", "main")

		if out, err := f.run("--pr"); err != nil {
			t.Fatalf("first sync failed: %v\n%s", err, out)
		}

		// v3 advances with a non-conflicting commit while PR #777 is still open.
		f.git(f.cloneDir, "checkout", "v3")
		f.writeFile("v3new.go", "package root\n\nconst V3New = 1\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat(v3): add v3new")
		f.git(f.cloneDir, "push", "origin", "v3")
		newV3SHA := f.git(f.cloneDir, "rev-parse", "HEAD")

		out, err := f.run("--pr")
		if err != nil {
			t.Fatalf("sync after v3 advance failed: %v\n%s", err, out)
		}
		parent := f.gitBare("rev-parse", "refs/heads/sync/main-to-v3^1")
		if parent != newV3SHA {
			t.Errorf("expected sync branch parent to rebase onto new v3 %s, got %s\n%s", newV3SHA, parent, out)
		}
	})

	t.Run("ignores spoofed trailer in ordinary v3 commit and skips empty merge", func(t *testing.T) {
		f := newSyncV3Fixture(t)
		f.git(f.cloneDir, "checkout", "-b", "v3")
		f.git(f.cloneDir, "push", "-u", "origin", "v3")

		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("a.go", "package root\n\nconst A = 1\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat: add A")
		m1 := f.git(f.cloneDir, "rev-parse", "HEAD")

		f.writeFile("b.go", "package root\n\nconst B = 2\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat: add B")
		f.git(f.cloneDir, "push", "origin", "main")
		m2 := f.git(f.cloneDir, "rev-parse", "HEAD")

		// Sync up to m2 and merge onto v3.
		if out, err := f.run("--pr"); err != nil {
			t.Fatalf("sync m2 failed: %v\n%s", err, out)
		}
		f.git(f.cloneDir, "fetch", "origin", "sync/main-to-v3")
		f.git(f.cloneDir, "checkout", "v3")
		f.git(f.cloneDir, "merge", "--ff-only", "FETCH_HEAD")

		// Land an ordinary v3 commit that quotes an older Synced-from-main: m1
		// AND already contains main's next commit content (empty merge test).
		f.writeFile("c.go", "package root\n\nconst C = 3\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", fmt.Sprintf("docs(v3): note about sync\n\nSynced-from-main: %s", m1))
		f.git(f.cloneDir, "push", "origin", "v3")
		f.setOpenPR("")

		// Same c.go lands on main (m3) -> last_synced must stay m2 (not rewind to m1),
		// and merged_tree == v3_tree must skip opening an empty PR.
		f.git(f.cloneDir, "checkout", "main")
		f.writeFile("c.go", "package root\n\nconst C = 3\n")
		f.git(f.cloneDir, "add", "-A")
		f.git(f.cloneDir, "commit", "-m", "feat: add C identically")
		f.git(f.cloneDir, "push", "origin", "main")

		f.clearGHLog()
		out, err := f.run("--pr")
		if err != nil {
			t.Fatalf("sync after spoofed trailer + empty merge failed: %v\n%s", err, out)
		}
		if !strings.Contains(out, m2[:9]) {
			t.Errorf("expected last synced to stay at %s (ignoring spoofed %s), got:\n%s", m2[:9], m1[:9], out)
		}
		if !strings.Contains(out, "already matches") {
			t.Errorf("expected empty-merge skip message, got:\n%s", out)
		}
		if strings.Contains(f.ghLog(), "pr create") {
			t.Errorf("must not open an empty PR, got log:\n%s", f.ghLog())
		}
	})
}

func TestSyncV3StatusMode(t *testing.T) {
	f := newSyncV3Fixture(t)
	f.git(f.cloneDir, "checkout", "-b", "v3")
	f.git(f.cloneDir, "push", "-u", "origin", "v3")

	f.git(f.cloneDir, "checkout", "main")
	f.writeFile("a.go", "package root\n\nconst A = 1\n")
	f.git(f.cloneDir, "add", "-A")
	f.git(f.cloneDir, "commit", "-m", "feat: add A")
	f.git(f.cloneDir, "push", "origin", "main")

	out, err := f.run("--status")
	if err != nil {
		t.Fatalf("--status failed: %v\n%s", err, out)
	}
	for _, want := range []string{"1 commit(s)", "feat: add A", "clean"} {
		if !strings.Contains(out, want) {
			t.Errorf("--status output missing %q:\n%s", want, out)
		}
	}
	if refs := f.gitBare("for-each-ref", "refs/heads/sync/main-to-v3"); refs != "" {
		t.Errorf("--status must not push anything, found: %s", refs)
	}
}

func TestSyncV3ConflictCommentMatchesBothBotSpellings(t *testing.T) {
	path := filepath.Join("..", "scripts", "sync-v3.sh")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	script := string(data)
	for _, login := range []string{
		`.author.login == "github-actions"`,
		`.author.login == "github-actions[bot]"`,
	} {
		if !strings.Contains(script, login) {
			t.Errorf("%s missing bot login match %s", path, login)
		}
	}
}

func TestSyncV3Workflow(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "sync-v3.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	wf := string(data)

	for _, want := range []string{
		"schedule:",
		"workflow_dispatch:",
		"group: sync-v3",
		"if: github.repository == 'google/adk-go'",
		"actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
		"scripts/sync-v3.sh --pr",
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
