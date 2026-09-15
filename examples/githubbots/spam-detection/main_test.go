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
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestTools(t *testing.T) {
	c := &GitHubClient{cfg: testConfig(), log: discardLogger()}
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(tools))
	}
	if tools[0].Name() != "flag_issue_as_spam" {
		t.Errorf("tool name = %q, want flag_issue_as_spam", tools[0].Name())
	}
}

func TestCandidateIssuesSingle(t *testing.T) {
	cfg := testConfig()
	cfg.SingleIssue = 5
	// The single-issue path must not touch the client (nil here proves it).
	got, err := candidateIssues(context.Background(), nil, cfg)
	if err != nil {
		t.Fatalf("candidateIssues() error = %v", err)
	}
	if len(got) != 1 || got[0] != 5 {
		t.Errorf("candidateIssues() = %v, want [5]", got)
	}
}

func TestMaintainersWarning(t *testing.T) {
	if w := maintainersWarning(&Config{}); w == "" {
		t.Error("expected a warning when MAINTAINERS is empty")
	}
	if w := maintainersWarning(&Config{Maintainers: []string{"alice"}}); w != "" {
		t.Errorf("expected no warning when maintainers are set, got %q", w)
	}
}

func TestNewNonce(t *testing.T) {
	a, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce() error = %v", err)
	}
	b, err := newNonce()
	if err != nil {
		t.Fatalf("newNonce() error = %v", err)
	}
	if len(a) != 16 {
		t.Errorf("nonce length = %d, want 16 hex chars", len(a))
	}
	if a == b {
		t.Errorf("two nonces were identical (%q); fence would be predictable", a)
	}
}

// The workflow's job timeout sits above the run budget so that an overrun is
// reported by the bot instead of the runner killing the job. That only holds if
// an exhausted budget actually becomes a non-nil error: swallowing it would
// make an overrunning sweep exit 0 and look like a clean run.
func TestRunWithBudgetReportsAnExhaustedBudget(t *testing.T) {
	ran := false
	err := runWithBudget(context.Background(), 20*time.Millisecond, func(ctx context.Context) error {
		ran = true
		select {
		case <-ctx.Done(): // the budget reached the work, as production relies on
		case <-time.After(2 * time.Second): // bounded so a missing deadline fails instead of hanging
		}
		return nil
	})
	if !ran {
		t.Fatal("runWithBudget never called fn: the sweep would not run at all")
	}
	if err == nil {
		t.Fatal("an exhausted budget produced no error, so the run would exit 0")
	}
	if !strings.Contains(err.Error(), "budget") {
		t.Errorf("error %q does not say the budget ran out", err)
	}
}

func TestRunWithBudgetPassesThroughTheWorkError(t *testing.T) {
	want := errors.New("one or more issues failed to process")
	got := runWithBudget(context.Background(), time.Minute, func(context.Context) error { return want })
	if !errors.Is(got, want) {
		t.Errorf("runWithBudget() = %v, want the work's own error", got)
	}
}

// A deadline the caller imposed is not this budget running out. Both surface as
// context.DeadlineExceeded on the derived context, so only the caller's own
// state tells them apart, and blaming the budget sends whoever reads the log
// after the wrong problem.
func TestRunWithBudgetDoesNotBlameTheBudgetForTheCallersDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	err := runWithBudget(ctx, time.Hour, func(ctx context.Context) error { return ctx.Err() })
	if err == nil {
		t.Fatal("a run stopped by the caller's deadline reported success")
	}
	if strings.Contains(err.Error(), "budget") {
		t.Errorf("the caller's own deadline was reported as a budget overrun: %v", err)
	}
}

func TestSummarize(t *testing.T) {
	if got := summarize("line one\nline two"); got != "line one line two" {
		t.Errorf("summarize() = %q, want newlines collapsed", got)
	}
	long := strings.Repeat("x", 250)
	got := summarize(long)
	if len([]rune(got)) != 203 || !strings.HasSuffix(got, "...") {
		t.Errorf("summarize() rune length = %d, want 203 ending in ellipsis", len([]rune(got)))
	}
	// Truncation must be rune-safe: a multibyte rune at the boundary must not be
	// split into invalid UTF-8.
	multibyte := strings.Repeat("界", 250)
	got = summarize(multibyte)
	if !utf8.ValidString(got) {
		t.Errorf("summarize() produced invalid UTF-8: %q", got)
	}
	if len([]rune(got)) != 203 {
		t.Errorf("summarize() rune length = %d, want 203", len([]rune(got)))
	}
}
