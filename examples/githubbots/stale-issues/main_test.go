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
	"errors"
	"io"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// An empty MAINTAINERS list silently disables stale-marking (no comment can be
// classified as a maintainer action), so the bot must surface a warning.
func TestMaintainersWarning(t *testing.T) {
	if w := maintainersWarning(&Config{Maintainers: nil}); w == "" {
		t.Error("expected a warning when MAINTAINERS is empty")
	}
	if w := maintainersWarning(&Config{Maintainers: []string{"alice"}}); w != "" {
		t.Errorf("expected no warning when maintainers are configured, got %q", w)
	}
}

func TestSummarize(t *testing.T) {
	if got := summarize("line one\nline two"); got != "line one line two" {
		t.Errorf("summarize collapsed newlines wrong: %q", got)
	}
	long := strings.Repeat("x", 500)
	if got := summarize(long); len(got) > 210 || !strings.HasSuffix(got, "...") {
		t.Errorf("summarize did not truncate long text: len=%d", len(got))
	}
}

// The run budget is what keeps a long sweep from being killed by the workflow's
// timeout-minutes with nothing in the log to say why. Drives the real function:
// the budget must come from the config, reach the work, and be reported.
func TestWithRunBudgetBoundsTheRunAndReportsTheOverrun(t *testing.T) {
	cfg := &Config{RunBudget: 20 * time.Millisecond}
	var (
		ran         bool
		sawDeadline bool
	)
	err := withRunBudget(context.Background(), cfg, discardLogger(), func(ctx context.Context) error {
		ran = true
		select {
		case <-ctx.Done():
			sawDeadline = true
			return ctx.Err()
		case <-time.After(10 * time.Second):
			return errors.New("no deadline arrived within 10s: cfg.RunBudget never reached the work")
		}
	})
	if !ran {
		t.Fatal("withRunBudget never called fn")
	}
	if !sawDeadline {
		t.Fatalf("the work saw no deadline, so the budget was not applied: %v", err)
	}
	if err == nil {
		t.Fatal("withRunBudget = nil, want an error reporting the exhausted budget")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if !strings.Contains(err.Error(), "run budget of 20ms") {
		t.Errorf("err = %v, want it to name the exhausted budget so the overrun is diagnosable", err)
	}
}

// A run that finishes inside its budget must be reported exactly as it ended --
// the budget check must not manufacture an error, nor swallow a real one.
func TestWithRunBudgetPassesThroughWhenNotExhausted(t *testing.T) {
	cfg := &Config{RunBudget: time.Minute}
	log := discardLogger()

	if err := withRunBudget(context.Background(), cfg, log, func(context.Context) error { return nil }); err != nil {
		t.Errorf("withRunBudget on a clean run = %v, want nil", err)
	}

	sentinel := errors.New("audit failed")
	err := withRunBudget(context.Background(), cfg, log, func(context.Context) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("withRunBudget = %v, want the underlying failure preserved", err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("withRunBudget = %v, want no spurious budget diagnosis on a run that finished in time", err)
	}
}

// Every candidate must be audited, each in a session scoped to its OWN number,
// and a failure on one must neither abort the batch nor be swallowed. A closure
// capturing the wrong number here would hand one issue's session the authority
// to mutate another.
func TestAuditAllAuditsEveryIssueUnderItsOwnScope(t *testing.T) {
	cfg := baseCfg()
	cfg.Concurrency = 2
	cfg.IssueTimeout = time.Minute

	var (
		mu       sync.Mutex
		seen     []int
		misscope []int
	)
	failing := errors.New("issue 2 blew up")
	err := auditAll(context.Background(), cfg, discardLogger(), []int{1, 2, 3}, func(ictx context.Context, n int) error {
		mu.Lock()
		seen = append(seen, n)
		if _, ok := authorizeIssue(ictx, n); !ok {
			misscope = append(misscope, n)
		}
		mu.Unlock()
		if n == 2 {
			return failing
		}
		return nil
	})

	mu.Lock()
	defer mu.Unlock()
	sort.Ints(seen)
	if len(seen) != 3 || seen[0] != 1 || seen[1] != 2 || seen[2] != 3 {
		t.Errorf("audited %v, want every candidate [1 2 3] even though one failed", seen)
	}
	if len(misscope) != 0 {
		t.Errorf("issues %v ran in a session scoped to a different number", misscope)
	}
	if !errors.Is(err, failing) {
		t.Errorf("auditAll = %v, want the per-issue failure surfaced so the run exits non-zero", err)
	}
}
