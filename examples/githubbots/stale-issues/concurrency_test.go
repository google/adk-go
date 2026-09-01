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

package main

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// stubLLM answers immediately with a final text part and calls no tools, so an
// audit completes without a network round trip.
type stubLLM struct{}

func (stubLLM) Name() string { return "stub" }

func (stubLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("Analysis for Issue: no action.", genai.RoleModel),
		}, nil)
	}
}

// The agent, session service and runner are built per audit because sharing
// them races. runner.Run writes the root agent's mode on its first call, so two
// concurrent audits on one agent are a write/write on that field — reproduced
// here under -race before the fix, at runner.go's LlmAgent node path.
//
// This is live on any real sweep: CONCURRENCY_LIMIT defaults to 3 and the search
// routinely returns more than one candidate.
func TestConcurrentAuditsDoNotShareAgentState(t *testing.T) {
	// 32 workers, not 8. The racing window is two instructions — a read of the
	// agent's mode and the write that follows it — so the odds rest on how many
	// pairs hit it at once. Measured with the factory hoisted out of
	// perIssueRunFn's closure: 8 workers caught it in 4 of 5 fresh processes,
	// 32 in 10 of 10. The work is a stub model and no I/O, so the extra workers
	// cost nothing.
	issues := make([]int, 32)
	for i := range issues {
		issues[i] = i + 1
	}
	cfg := baseCfg()
	cfg.Concurrency = len(issues) // every worker must be able to run at once
	cfg.IssueTimeout = time.Minute
	log := discardLogger()

	// Hold every worker until all of them hold a runner, then release them into
	// Run together. The mode write is the first thing Run does, so without the
	// barrier the workers file past it one at a time and the detector sees
	// nothing: measured with the fix reverted, 1 reproduction in 5 fresh
	// processes without the barrier against 5 in 5 with it. A pin that fires a
	// fifth of the time is not a pin.
	var (
		ran     atomic.Int64
		arrived = make(chan struct{}, len(issues))
		start   = make(chan struct{})
	)
	go func() {
		for range issues {
			select {
			case <-arrived:
			case <-time.After(30 * time.Second):
				close(start) // never wedge the suite if a worker dies
				return
			}
		}
		close(start)
	}()

	// perIssueRunFn is what audit() actually fans out over. Wrapping it rather
	// than rebuilding it is what makes this a pin: a test that assembles its own
	// equivalent closure keeps passing after production's regresses, which is
	// how an identical race went unpinned in a sibling bot.
	production := perIssueRunFn(cfg, stubLLM{}, nil, log)
	err := auditAll(context.Background(), cfg, log, issues,
		func(ictx context.Context, n int) error {
			ran.Add(1)
			arrived <- struct{}{}
			<-start
			return production(ictx, n)
		})
	if err != nil {
		t.Fatalf("auditAll: %v", err)
	}
	if got := ran.Load(); got != int64(len(issues)) {
		t.Fatalf("drove %d audits, want %d: the fan-out did not exercise the shared state", got, len(issues))
	}
}

// And the factory must actually produce a fresh agent each time. Memoizing it
// into a singleton would compile, pass every other test, and put the race back.
func TestNewAuditRunnerReturnsAFreshRunnerEachCall(t *testing.T) {
	cfg := baseCfg()
	log := discardLogger()
	r1, ss1, err := newAuditRunner(cfg, stubLLM{}, nil, log)
	if err != nil {
		t.Fatalf("newAuditRunner: %v", err)
	}
	r2, ss2, err := newAuditRunner(cfg, stubLLM{}, nil, log)
	if err != nil {
		t.Fatalf("newAuditRunner: %v", err)
	}
	if r1 == r2 {
		t.Error("both calls returned the same *runner.Runner: concurrent audits would share the agent it wraps")
	}
	if ss1 == ss2 {
		t.Error("both calls returned the same session service")
	}
}

// Both pins above hold only while audit() still fans out over perIssueRunFn.
// Hoisting the construction ABOVE that call — build one runner in audit(), hand
// auditAll a closure over it — shares one agent across every issue and puts the
// write/write race back. Nothing catches it: measured, the full suite plus vet,
// staticcheck and golangci-lint stay green in 5 of 5 fresh processes, and the
// `unused` check stays quiet because the test above still calls perIssueRunFn.
//
// No runtime test can reach that edit. audit() is the composition root: it
// builds a real GitHub client and a real Gemini model before it reaches the
// fan-out, so a test cannot call it, and there is no seam left to inject at.
// The wiring is therefore pinned where it is written rather than where it runs.
//
// This test is deliberately brittle. Restructuring this call is exactly the
// moment someone should have to re-read why the runner is built per issue.
func TestAuditFansOutOverPerIssueRunFn(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parse main.go: %v", err)
	}

	var auditFn *ast.FuncDecl
	for _, d := range file.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "audit" {
			auditFn = fd
			break
		}
	}
	if auditFn == nil {
		t.Fatal("main.go has no func audit: the fan-out moved, so re-point this pin at its new home rather than deleting it")
	}

	var found bool
	ast.Inspect(auditFn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); !ok || id.Name != "auditAll" {
			return true
		}
		if len(call.Args) == 0 {
			return false
		}
		arg, ok := call.Args[len(call.Args)-1].(*ast.CallExpr)
		if !ok {
			return false
		}
		if id, ok := arg.Fun.(*ast.Ident); ok && id.Name == "perIssueRunFn" {
			found = true
		}
		return false
	})

	if !found {
		t.Error("audit() no longer passes perIssueRunFn(...) as auditAll's run function: whatever it passes instead is not what TestConcurrentAuditsDoNotShareAgentState exercises, and a runner built once above the fan-out is shared by every issue in the sweep")
	}
}

// GET /user is a user-to-server endpoint and the workflow's GITHUB_TOKEN is an
// installation token, so identity discovery is refused on every scheduled run.
// The login therefore has to be configured, and configuring it must skip the
// call entirely rather than merely overriding its result.
func TestConfiguredBotLoginSkipsIdentityDiscovery(t *testing.T) {
	var calls int
	cfg := baseCfg()
	cfg.BotLogin = "github-actions[bot]"
	c := testClient(t, cfg, forbiddenHandler(&calls))

	if got := resolveSelfLogin(context.Background(), c.rest, cfg, discardLogger()); got != "github-actions[bot]" {
		t.Errorf("resolveSelfLogin = %q, want the configured login", got)
	}
	if calls != 0 {
		t.Errorf("made %d API calls with BOT_LOGIN set, want 0: GET /user is refused for an installation token", calls)
	}
}

// Without it the bot must still start — a local run under a personal token can
// discover the login — but the run has to say the fallback is in use, and name
// the setting that fixes it.
func TestUnconfiguredBotLoginFallsBackAndSaysSo(t *testing.T) {
	var calls int
	cfg := baseCfg()
	cfg.BotLogin = ""
	var logged strings.Builder
	log := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))
	c := testClient(t, cfg, forbiddenHandler(&calls))

	if got := resolveSelfLogin(context.Background(), c.rest, cfg, log); got != "" {
		t.Errorf("resolveSelfLogin = %q, want empty after a refused discovery call", got)
	}
	if calls == 0 {
		t.Error("no API call was made with BOT_LOGIN unset: the local discovery path is gone")
	}
	if !strings.Contains(logged.String(), "BOT_LOGIN") {
		t.Errorf("the warning does not name BOT_LOGIN, so nobody reading the log learns the fix: %q", logged.String())
	}
}

// And a login that IS discoverable is used, so the fallback is not the only path.
func TestDiscoveredBotLoginIsUsed(t *testing.T) {
	cfg := baseCfg()
	cfg.BotLogin = ""
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"login":"local-dev"}`)
	}))
	if got := resolveSelfLogin(context.Background(), c.rest, cfg, discardLogger()); got != "local-dev" {
		t.Errorf("resolveSelfLogin = %q, want the discovered login", got)
	}
}

// BOT_LOGIN must reach the config from the environment.
func TestBotLoginFromEnv(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("BOT_LOGIN", "  github-actions[bot]  ")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.BotLogin != "github-actions[bot]" {
		t.Errorf("BotLogin = %q, want it read and trimmed", cfg.BotLogin)
	}
}

func forbiddenHandler(calls *int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		*calls++
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"message":"Resource not accessible by integration"}`)
	})
}

// The claim's test and its consume must share ONE critical section. Splitting
// them — read the observation and the claimed flag under the lock, evaluate the
// predicate outside it, re-lock to record — is the check-then-act the design
// exists to prevent, and it is a plausible refactor: it looks like shortening a
// lock hold.
//
// TestClaimIsAtomicUnderConcurrency does not catch it: its workers race a
// window two instructions wide, so catching the split needs two of them to read
// "unclaimed" before either writes. Measured against that mutant it killed it in
// 1 of 15 fresh processes. This test drops the timing dependency by making the
// PREDICATE itself the barrier, and killed the same mutant in 10 of 10.
//
// If the predicate runs under the claim's lock, as it must, only one worker can
// ever be inside it, the arrival count never reaches the total, that worker
// waits out the timeout alone and wins, and the rest find the key claimed and
// return without ever calling the predicate. If the predicate has been moved
// outside the lock, every worker enters it together, the count is satisfied at
// once, they all see an unclaimed key, and they all win — which is a duplicate
// stale warning on a real issue.
func TestClaimTestAndConsumeShareOneCriticalSection(t *testing.T) {
	const workers = 8
	c := newTestClient(t)
	c.recordObservation(7, staleReady())

	var (
		inPredicate atomic.Int64
		release     = make(chan struct{})
		closeOnce   sync.Once
	)
	barrierPred := func(st IssueState) (string, bool) {
		if inPredicate.Add(1) >= workers {
			closeOnce.Do(func() { close(release) })
		}
		select {
		case <-release:
		case <-time.After(200 * time.Millisecond):
		}
		return stalePredicate(7)(st)
	}

	var (
		wins  atomic.Int64
		wg    sync.WaitGroup
		start = make(chan struct{})
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, ok := c.claimAction(7, actionMarkStale, barrierPred); ok {
				wins.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if got := wins.Load(); got != 1 {
		t.Errorf("%d of %d workers claimed the same action, want exactly 1: the predicate and the consume are not in one critical section, so a duplicate stale comment reaches the issue", got, workers)
	}
	if n := inPredicate.Load(); n != 1 {
		t.Errorf("the predicate ran %d times, want 1: a worker that lost the claim must never reach it", n)
	}
}
