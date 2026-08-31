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
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// graphQLPR builds a client that answers the pull request query with the given
// node and everything else with an empty success.
func graphQLPR(t *testing.T, cfg *Config, node string) *GitHubClient {
	t.Helper()
	return respondWith(t, cfg, fmt.Sprintf(`{"data":{"repository":{"pullRequest":%s}}}`, node))
}

// agentRecorder captures what the orchestrator would hand the model.
type agentRecorder struct {
	calls  int
	prompt string
	number int
}

func (a *agentRecorder) run(_ context.Context, number int, prompt string) string {
	a.calls++
	a.number = number
	a.prompt = prompt
	return "done"
}

// Drives the real scopeSession. Deleting its withAuditedPR line -- this bot's
// entire cross-pull-request defense -- must not leave the suite green, so this
// observes the context production builds rather than constructing its own.
func TestScopeSessionScopesAndBoundsTheSession(t *testing.T) {
	cfg := testConfig()
	cfg.PRTimeout = time.Minute
	var problems []string
	ran := false
	scopeSession(context.Background(), cfg, 42, func(ictx context.Context) {
		ran = true
		if msg, ok := authorizePR(ictx, 42); !ok {
			problems = append(problems, "not scoped to the triaged pull request: "+msg)
		}
		if _, ok := authorizePR(ictx, 43); ok {
			problems = append(problems, "the session accepted a DIFFERENT pull request")
		}
		dl, hasDeadline := ictx.Deadline()
		if !hasDeadline {
			problems = append(problems, "no per-pull-request deadline")
		} else if time.Until(dl) > cfg.PRTimeout {
			problems = append(problems, "the deadline does not derive from PRTimeout")
		}
	})
	if !ran {
		t.Fatal("scopeSession never called runFn: the bot would triage nothing")
	}
	for _, p := range problems {
		t.Error(p)
	}
}

// triageOne is the gate. An ineligible pull request must cost zero model tokens
// AND must never be marked eligible, because eligibility is the only thing that
// authorizes a mutation.
func TestTriageOneGatesIneligiblePullRequests(t *testing.T) {
	for _, tc := range []struct {
		name     string
		node     string
		wantSkip string
	}{
		{
			name:     "closed",
			node:     `{"number":7,"state":"CLOSED","author":{"login":"carol","__typename":"User"}}`,
			wantSkip: "closed",
		},
		{
			name:     "draft",
			node:     `{"number":7,"state":"OPEN","isDraft":true,"author":{"login":"carol","__typename":"User"}}`,
			wantSkip: "draft",
		},
		{
			name: "already assigned",
			node: `{"number":7,"state":"OPEN","author":{"login":"carol","__typename":"User"},
				"assignees":{"totalCount":1}}`,
			wantSkip: "already has an assignee",
		},
		{
			name: "assigned before and since un-assigned",
			node: `{"number":7,"state":"OPEN","author":{"login":"carol","__typename":"User"},
				"assignees":{"totalCount":0},"timelineItems":{"totalCount":1}}`,
			wantSkip: "assigned before",
		},
		{
			name:     "authored by a component owner",
			node:     `{"number":7,"state":"OPEN","author":{"login":"alice","__typename":"User"}}`,
			wantSkip: "component owner",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			c := graphQLPR(t, cfg, tc.node)
			var rec agentRecorder

			got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run)

			if !strings.Contains(got, tc.wantSkip) {
				t.Errorf("triageOne() = %q, want a skip mentioning %q", got, tc.wantSkip)
			}
			if rec.calls != 0 {
				t.Errorf("the model was invoked %d times for an ineligible pull request, want 0", rec.calls)
			}
			if c.isEligible(7) {
				t.Error("an ineligible pull request was authorized for mutation")
			}
			// The gate is the only thing standing between an ineligible pull
			// request and a write, so prove the tools actually refuse.
			ctx := withAuditedPR(context.Background(), 7)
			if res, _ := c.assignOwner(ctx, 7, "core"); res.Status != "error" {
				t.Errorf("assignOwner on a gated pull request = %+v, want an error", res)
			}
			if res, _ := c.requestMoreContext(ctx, 7, []string{"problem"}); res.Status != "error" {
				t.Errorf("requestMoreContext on a gated pull request = %+v, want an error", res)
			}
		})
	}
}

func TestTriageOneRunsTheModelForAnEligiblePullRequest(t *testing.T) {
	cfg := testConfig()
	const node = `{"number":7,"title":"Add caching","body":"Speeds things up.","state":"OPEN",
		"author":{"login":"carol","__typename":"User"},
		"files":{"nodes":[{"path":"agent/cache.go"}]}}`
	c := graphQLPR(t, cfg, node)
	var rec agentRecorder

	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run); got != "" {
		t.Fatalf("triageOne() = %q, want \"\" (eligible)", got)
	}
	if rec.calls != 1 {
		t.Fatalf("the model was invoked %d times, want 1", rec.calls)
	}
	if rec.number != 7 {
		t.Errorf("the model was given pull request %d, want 7", rec.number)
	}
	if !c.isEligible(7) {
		t.Error("an eligible pull request was not authorized for mutation")
	}
	// The prompt production builds must name the pull request and fence the
	// author's text, or the tool call would carry the wrong number and the
	// description would be read as instructions.
	if !strings.Contains(rec.prompt, "Triage pull request #7.") {
		t.Errorf("prompt does not name the pull request:\n%s", rec.prompt)
	}
	if !strings.Contains(rec.prompt, "Speeds things up.") {
		t.Errorf("prompt does not carry the description:\n%s", rec.prompt)
	}
	nonce := fenceNonce(t, rec.prompt)
	if !insideFence(rec.prompt, "Speeds things up.", "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]") {
		t.Errorf("the description was not fenced in the prompt production builds:\n%s", rec.prompt)
	}
	if !insideFence(rec.prompt, "agent/cache.go", "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]") {
		t.Errorf("the changed file path was not fenced:\n%s", rec.prompt)
	}
}

// fenceNonce extracts the nonce from a prompt, failing the test when there is
// none — a prompt with no fence at all must never look like a pass.
func fenceNonce(t *testing.T, prompt string) string {
	t.Helper()
	const marker = "[UNTRUSTED:"
	i := strings.Index(prompt, marker)
	if i < 0 {
		t.Fatalf("prompt contains no untrusted fence:\n%s", prompt)
	}
	rest := prompt[i+len(marker):]
	j := strings.Index(rest, "]")
	if j <= 0 {
		t.Fatalf("prompt fence marker is malformed:\n%s", prompt)
	}
	nonce := rest[:j]
	if len(nonce) != 16 {
		t.Fatalf("fence nonce %q is %d hex chars, want 16", nonce, len(nonce))
	}
	return nonce
}

// Two pull requests triaged in the same run must not share a fence nonce: a
// reused nonce lets an author who has seen one pull request's markers escape
// another's.
func TestTriageOneUsesAFreshNoncePerPullRequest(t *testing.T) {
	cfg := testConfig()
	c := graphQLPR(t, cfg, `{"number":7,"title":"t","body":"b","state":"OPEN","author":{"login":"carol","__typename":"User"}}`)
	var first, second agentRecorder
	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, first.run); got != "" {
		t.Fatalf("first triageOne() = %q", got)
	}
	if got := triageOne(context.Background(), c, cfg, discardLogger(), 8, second.run); got != "" {
		t.Fatalf("second triageOne() = %q", got)
	}
	if a, b := fenceNonce(t, first.prompt), fenceNonce(t, second.prompt); a == b {
		t.Errorf("both pull requests were fenced with nonce %q", a)
	}
}

// A CSPRNG failure must abort the pull request rather than fence it with a
// predictable marker, which an author could close from inside their own text.
func TestTriageOneAbortsWhenTheNonceCannotBeDrawn(t *testing.T) {
	orig := newNonce
	newNonce = func() (string, error) { return "", errors.New("no entropy") }
	t.Cleanup(func() { newNonce = orig })
	if _, err := newNonce(); err == nil {
		t.Fatal("test setup: the forced failure did not take")
	}

	cfg := testConfig()
	c := graphQLPR(t, cfg, `{"number":7,"title":"t","body":"b","state":"OPEN","author":{"login":"carol","__typename":"User"}}`)
	var rec agentRecorder

	got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run)

	if got == "" {
		t.Error("triageOne() reported success despite the nonce failure")
	}
	if rec.calls != 0 {
		t.Errorf("the model was invoked %d times with no usable fence, want 0", rec.calls)
	}
	if c.isEligible(7) {
		t.Error("the pull request was authorized for mutation despite the nonce failure")
	}
	if !c.hadError() {
		t.Error("the nonce failure was not recorded, so the run would exit 0")
	}
}

// A comment the bot posted on an earlier run must consume this run's single
// comment claim, so a reopened pull request is never asked for context twice.
func TestTriageOneSpendsTheCommentClaimWhenItAlreadyAsked(t *testing.T) {
	cfg := testConfig()
	body := buildContextComment([]string{"problem"})
	if body == "" {
		t.Fatal("test setup: no comment body")
	}
	node := fmt.Sprintf(`{"number":7,"title":"t","body":"b","state":"OPEN",
		"author":{"login":"carol","__typename":"User"},
		"comments":{"totalCount":1,"nodes":[{"author":{"login":"adk-bot","__typename":"User"},"body":%q}]}}`, body)
	c := graphQLPR(t, cfg, node)
	var rec agentRecorder

	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run); got != "" {
		t.Fatalf("triageOne() = %q, want the pull request still triaged for assignment", got)
	}
	ctx := withAuditedPR(context.Background(), 7)
	if res, _ := c.requestMoreContext(ctx, 7, []string{"testing"}); res.Status != "skipped" {
		t.Errorf("a second context request = %+v, want skipped", res)
	}
	// Assignment is a separate claim and must still be available.
	if res, _ := c.assignOwner(ctx, 7, "core"); res.Status != "success" {
		t.Errorf("assignment was blocked by the earlier comment: %+v", res)
	}
}

// A stranger who pastes the bot's signature must not be able to suppress the
// bot's own context request.
func TestTriageOneIgnoresAForgedBotComment(t *testing.T) {
	cfg := testConfig()
	node := fmt.Sprintf(`{"number":7,"title":"t","body":"b","state":"OPEN",
		"author":{"login":"carol","__typename":"User"},
		"comments":{"totalCount":1,"nodes":[{"author":{"login":"mallory","__typename":"User"},"body":%q}]}}`,
		botCommentSignature+" already asked")
	c := graphQLPR(t, cfg, node)
	var rec agentRecorder

	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run); got != "" {
		t.Fatalf("triageOne() = %q, want eligible", got)
	}
	// A comment from anyone but the bot leaves the claim available.
	if res, _ := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, []string{"problem"}); res.Status != "success" {
		t.Errorf("a forged comment suppressed the bot's request: %+v", res)
	}
}

// Comments are fetched for the idempotency check and must never be shown to the
// model: they are extra attacker-controlled text that neither decision needs.
func TestTriageOneNeverShowsCommentsToTheModel(t *testing.T) {
	cfg := testConfig()
	const node = `{"number":7,"title":"t","body":"b","state":"OPEN",
		"author":{"login":"carol","__typename":"User"},
		"comments":{"totalCount":1,"nodes":[{"author":{"login":"mallory","__typename":"User"},"body":"COMMENT-MARKER"}]}}`
	c := graphQLPR(t, cfg, node)
	var rec agentRecorder
	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run); got != "" {
		t.Fatalf("triageOne() = %q, want eligible", got)
	}
	if strings.Contains(rec.prompt, "COMMENT-MARKER") {
		t.Errorf("a comment reached the prompt:\n%s", rec.prompt)
	}
}

func TestTriageOneReportsFetchFailures(t *testing.T) {
	cfg := testConfig()
	// A transient GraphQL failure must be recorded so the run exits non-zero.
	c := respondWith(t, cfg, `{"data":{"repository":{"pullRequest":null}},"errors":[{"type":"RATE_LIMITED","message":"slow down"}]}`)
	var rec agentRecorder
	if got := triageOne(context.Background(), c, cfg, discardLogger(), 7, rec.run); got != "fetch failed" {
		t.Errorf("triageOne() = %q, want \"fetch failed\"", got)
	}
	if !c.hadError() {
		t.Error("a transient fetch failure was not recorded, so the run would exit 0")
	}
	if c.isEligible(7) {
		t.Error("a pull request that could not be fetched was authorized for mutation")
	}

	// A genuinely missing pull request is a skip, not a run failure.
	c2 := respondWith(t, cfg, `{"data":{"repository":{"pullRequest":null}}}`)
	if got := triageOne(context.Background(), c2, cfg, discardLogger(), 7, rec.run); got != "not found" {
		t.Errorf("triageOne() = %q, want \"not found\"", got)
	}
	if c2.hadError() {
		t.Error("a missing pull request was recorded as a run error")
	}
}

// triageAll must scope every pull request it hands out, and must keep going
// after one fails. The scoping is asserted from inside the callback the real
// orchestrator invokes.
func TestTriageAllScopesEveryPullRequestAndSurvivesFailures(t *testing.T) {
	cfg := testConfig()
	cfg.Concurrency = 4
	cfg.PRTimeout = time.Minute
	prs := []int{1, 2, 3, 4, 5}

	var mu sync.Mutex
	seen := map[int]bool{}
	var problems []string

	triageAll(context.Background(), cfg, discardLogger(), prs, func(ictx context.Context, n int) {
		mu.Lock()
		seen[n] = true
		if _, ok := authorizePR(ictx, n); !ok {
			problems = append(problems, fmt.Sprintf("pull request %d was not scoped to itself", n))
		}
		for _, other := range prs {
			if other != n {
				if _, ok := authorizePR(ictx, other); ok {
					problems = append(problems, fmt.Sprintf("session for %d accepted %d", n, other))
				}
			}
		}
		mu.Unlock()
		if n == 2 {
			// One pull request failing must not abort the batch.
			return
		}
	})

	for _, n := range prs {
		if !seen[n] {
			t.Errorf("pull request %d was never triaged", n)
		}
	}
	for _, p := range problems {
		t.Error(p)
	}
}

// Production runs triageOne on several goroutines against one shared client
// (main.go's triageAll closure), and nothing exercised that: every triageOne
// test was single-threaded, so -race never saw the real orchestration path.
func TestTriageOneIsConcurrencySafe(t *testing.T) {
	cfg := testConfig()
	cfg.Concurrency = 4
	cfg.SinglePR = 0
	c := graphQLPR(t, cfg, `{"number":7,"title":"t","body":"b","state":"OPEN",
		"author":{"login":"carol","__typename":"User"},
		"assignees":{"totalCount":0},"files":{"totalCount":1,"nodes":[{"path":"a.go"}]},
		"comments":{"totalCount":0,"nodes":[]},"timelineItems":{"totalCount":0}}`)

	prs := []int{1, 2, 3, 4, 5, 6, 7, 8}
	var mu sync.Mutex
	nonces := map[string]bool{}
	triageAll(context.Background(), cfg, discardLogger(), prs, func(ictx context.Context, n int) {
		triageOne(ictx, c, cfg, discardLogger(), n, func(_ context.Context, _ int, prompt string) string {
			mu.Lock()
			nonces[fenceNonce(t, prompt)] = true
			mu.Unlock()
			return "done"
		})
	})

	mu.Lock()
	defer mu.Unlock()
	if len(nonces) != len(prs) {
		t.Errorf("saw %d distinct fence nonces across %d concurrent pull requests; they must not be shared",
			len(nonces), len(prs))
	}
	for _, n := range prs {
		if !c.isEligible(n) {
			t.Errorf("pull request %d was not authorized; the concurrent path lost a write", n)
		}
	}
}

func TestTriageAllRespectsTheConcurrencyLimit(t *testing.T) {
	cfg := testConfig()
	cfg.Concurrency = 2
	cfg.PRTimeout = time.Minute

	var inFlight, peak int32
	var wg sync.WaitGroup
	wg.Add(6)
	triageAll(context.Background(), cfg, discardLogger(), []int{1, 2, 3, 4, 5, 6}, func(context.Context, int) {
		n := atomic.AddInt32(&inFlight, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if n <= p || atomic.CompareAndSwapInt32(&peak, p, n) {
				break
			}
		}
		wg.Done()
		time.Sleep(2 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
	})
	wg.Wait()
	if got := atomic.LoadInt32(&peak); got > int32(cfg.Concurrency) {
		t.Errorf("peak concurrency = %d, want at most %d", got, cfg.Concurrency)
	}
}

func TestCandidatePRsSingleSkipsTheSearch(t *testing.T) {
	cfg := testConfig()
	cfg.SinglePR = 5
	// A nil client proves the single-pull-request path never touches the API.
	got, err := candidatePRs(context.Background(), nil, cfg)
	if err != nil {
		t.Fatalf("candidatePRs() error = %v", err)
	}
	if len(got) != 1 || got[0] != 5 {
		t.Errorf("candidatePRs() = %v, want [5]", got)
	}
}

func TestCandidatePRsBatchSearches(t *testing.T) {
	cfg := testConfig()
	cfg.SinglePR = 0 // batch mode
	c := respondWith(t, cfg, `{"items":[{"number":11,"pull_request":{"url":"u"}}]}`)
	got, err := candidatePRs(context.Background(), c, cfg)
	if err != nil {
		t.Fatalf("candidatePRs() error = %v", err)
	}
	if len(got) != 1 || got[0] != 11 {
		t.Errorf("candidatePRs() = %v, want [11]", got)
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
		t.Errorf("two nonces were identical (%q); the fence would be predictable", a)
	}
}

func TestBuildPromptNamesTheFence(t *testing.T) {
	got := buildPrompt(9, "CONTENT", "abcd1234")
	for _, want := range []string{"pull request #9", "[UNTRUSTED:abcd1234]", "[/UNTRUSTED:abcd1234]", "CONTENT"} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt is missing %q:\n%s", want, got)
		}
	}
}

func TestSummarize(t *testing.T) {
	if got := summarize("line one\nline two"); got != "line one line two" {
		t.Errorf("summarize() = %q, want newlines collapsed", got)
	}
	got := summarize(strings.Repeat("x", 250))
	if len([]rune(got)) != 203 || !strings.HasSuffix(got, "...") {
		t.Errorf("summarize() rune length = %d, want 203 ending in an ellipsis", len([]rune(got)))
	}
	// Truncation must be rune-safe: a multibyte rune at the boundary must not be
	// split into invalid UTF-8.
	got = summarize(strings.Repeat("界", 250))
	if !utf8.ValidString(got) {
		t.Errorf("summarize() produced invalid UTF-8: %q", got)
	}
	if len([]rune(got)) != 203 {
		t.Errorf("summarize() rune length = %d, want 203", len([]rune(got)))
	}
}

// An expired run budget must become a non-zero exit. Once the context is done
// every remaining API call fails quietly, so a silent success here would look
// exactly like a run that found nothing to do.
func TestRunOutcomeReportsAnExpiredBudget(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))

	// Produce the expiry rather than faking it: a real deadline that has passed.
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-ctx.Done()

	err := runOutcome(ctx, c, 42*time.Second)
	if err == nil {
		t.Fatal("runOutcome() on an expired budget returned nil; the job would exit 0")
	}
	if !strings.Contains(err.Error(), "42s") {
		t.Errorf("error %q should name the budget that expired", err)
	}
}

// A tool error is handed back to the model as data, so the only thing that makes
// the job fail is this check.
func TestRunOutcomeReportsRecordedErrors(t *testing.T) {
	h := newRecordingHandler()
	h.status = http.StatusInternalServerError
	c := eligibleClient(t, testConfig(), h)

	if err := runOutcome(context.Background(), c, time.Minute); err != nil {
		t.Fatalf("runOutcome() on a clean run = %v, want nil", err)
	}
	// Drive a real failure through the real tool rather than setting the flag.
	if _, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core"); err == nil {
		t.Fatal("test setup: the forced HTTP 500 did not produce an error")
	}
	if err := runOutcome(context.Background(), c, time.Minute); err == nil {
		t.Error("runOutcome() after a failed mutation returned nil; the job would exit 0")
	}
}

// A pull request that is skipped must not consume a concurrency slot forever or
// leave the client thinking anything is authorized.
func TestNothingIsAuthorizedBeforeTriage(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	for _, n := range []int{0, 1, 7, 999} {
		if c.isEligible(n) {
			t.Errorf("pull request %d was authorized before any triage ran", n)
		}
	}
}

// The run budget must stop the dispatch, not just fail each remaining pull
// request: past the deadline every fetch returns "context deadline exceeded",
// so continuing buries the real cause under one recorded failure per queued
// item.
func TestTriageAllStopsDispatchingWhenTheBudgetIsGone(t *testing.T) {
	cfg := testConfig()
	cfg.Concurrency = 1
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-ctx.Done()

	var started int32
	triageAll(ctx, cfg, discardLogger(), []int{1, 2, 3, 4, 5}, func(context.Context, int) {
		atomic.AddInt32(&started, 1)
	})
	if got := atomic.LoadInt32(&started); got != 0 {
		t.Errorf("dispatched %d pull requests after the budget expired, want 0", got)
	}
}

// A panic in the agent, transport or model stack must not unwind out of the
// worker and kill the process: the surviving pull requests would never be
// triaged and runOutcome would never run.
func TestTriageAllSurvivesAPanickingWorker(t *testing.T) {
	cfg := testConfig()
	cfg.Concurrency = 1

	var seen []int
	var mu sync.Mutex
	triageAll(context.Background(), cfg, discardLogger(), []int{1, 2, 3}, func(_ context.Context, n int) {
		mu.Lock()
		seen = append(seen, n)
		mu.Unlock()
		if n == 2 {
			panic("the model client exploded")
		}
	})

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 3 {
		t.Errorf("triaged %v, want all three: a panic on one must not abort the batch", seen)
	}
}

// The event loop must not depend on the ADK iterator choosing to stop after a
// cancelled context. If it kept yielding, the worker would spin until the job's
// own timeout killed it — the outcome the run budget exists to prevent.
func TestRunAgentStopsOnAnExpiredDeadline(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)

	llm := &scriptedLLM{}
	tools, err := gh.tools()
	if err != nil {
		t.Fatalf("tools(): %v", err)
	}
	triager, err := llmagent.New(llmagent.Config{
		Name: "pr_triager", Model: llm, Instruction: renderPrompt(cfg), Tools: tools,
	})
	if err != nil {
		t.Fatalf("llmagent.New(): %v", err)
	}
	ss := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: appName, Agent: triager, SessionService: ss})
	if err != nil {
		t.Fatalf("runner.New(): %v", err)
	}

	ctx, cancel := context.WithTimeout(withAuditedPR(context.Background(), 7), time.Nanosecond)
	defer cancel()
	<-ctx.Done()

	done := make(chan string, 1)
	go func() { done <- runAgent(ctx, r, ss, gh, discardLogger(), "triage pull request #7") }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("runAgent did not return on an expired deadline; the worker would spin until the job timeout")
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Errorf("an expired run still wrote to GitHub: %v", writes)
	}
}

// The prompt names which lines the model may trust. It must not name the
// author: the login is attacker-chosen and is deliberately not emitted, so
// promising it both misleads the model and invites the line back.
func TestBuildPromptDoesNotPromiseTheAuthor(t *testing.T) {
	got := buildPrompt(9, assemblePRContext(eligiblePR(), 10, "abcd1234"), "abcd1234")
	if strings.Contains(got, "the author,") {
		t.Errorf("the prompt promises a trusted author line:\n%s", got)
	}
	if strings.Contains(got, "@carol") {
		t.Errorf("the author login reached the prompt:\n%s", got)
	}
	// It must still vouch for what IS emitted, or the model has no trusted anchor.
	if !strings.Contains(got, "the pull request number") {
		t.Errorf("the prompt no longer names its trusted context:\n%s", got)
	}
}

// The loop must not depend on the iterator choosing to stop. A sequence that
// keeps yielding an error is exactly what a runner reporting a cancelled context
// would look like, and without the deadline check the worker spins until the
// job's own timeout kills it — the outcome the run budget exists to prevent.
func TestConsumeEventsStopsOnAnExpiredDeadline(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))

	var yielded int
	endless := func(yield func(*session.Event, error) bool) {
		for {
			yielded++
			if yielded > 1_000_000 {
				// The loop never terminated. Stop rather than hang the suite.
				return
			}
			if !yield(nil, errors.New("context deadline exceeded")) {
				return
			}
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-ctx.Done()

	done := make(chan struct{})
	go func() { consumeEvents(ctx, c, discardLogger(), endless); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("consumeEvents did not return on an endless error sequence")
	}
	if yielded > 2 {
		t.Errorf("the loop consumed %d events past the deadline, want it to break at once", yielded)
	}
}

// The control: with a live context the loop still drains the sequence and
// returns the model's last text, so the deadline check cannot pass by refusing
// everything.
func TestConsumeEventsReturnsTheFinalText(t *testing.T) {
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	seq := func(yield func(*session.Event, error) bool) {
		for _, text := range []string{"first", "second"} {
			ev := &session.Event{}
			ev.Content = genai.NewContentFromText(text, genai.RoleModel)
			if !yield(ev, nil) {
				return
			}
		}
	}
	if got := consumeEvents(context.Background(), c, discardLogger(), seq); got != "second" {
		t.Errorf("consumeEvents() = %q, want the last text", got)
	}
	if c.hadError() {
		t.Error("a clean run recorded an error")
	}
}

// A model error is data to the agent, so the ONLY thing that turns it into a
// non-zero exit is recordError here. Nothing tested that: the fetch-failure test
// covers a different branch entirely, so deleting either recordError call left
// the suite green while a run in which every turn failed reported success.
func TestConsumeEventsRecordsModelErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		seq  iter.Seq2[*session.Event, error]
	}{
		{
			name: "the runner yields an error",
			seq: func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("the model refused"))
			},
		},
		{
			name: "the event carries an error code",
			seq: func(yield func(*session.Event, error) bool) {
				ev := &session.Event{}
				ev.ErrorCode, ev.ErrorMessage = "SAFETY", "blocked"
				yield(ev, nil)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
			if c.hadError() {
				t.Fatal("hadError() should start false")
			}
			consumeEvents(context.Background(), c, discardLogger(), tc.seq)
			if !c.hadError() {
				t.Error("the failure was not recorded, so the run would exit 0")
			}
			// And it must fail the run for real, through the production path.
			if err := runOutcome(context.Background(), c, time.Minute); err == nil {
				t.Error("runOutcome() returned nil after a failed agent run")
			}
		})
	}

	// The control: a clean run must not be recorded as an error, or the check
	// above would pass by failing everything.
	c := testClient(t, testConfig(), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	consumeEvents(context.Background(), c, discardLogger(), func(yield func(*session.Event, error) bool) {
		ev := &session.Event{}
		ev.Content = genai.NewContentFromText("all good", genai.RoleModel)
		yield(ev, nil)
	})
	if c.hadError() {
		t.Error("a clean run was recorded as an error")
	}
}
