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
	"bytes"
	"context"
	"encoding/json"
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

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// The orchestration layer -- reviewAll and the filtering and error recording in
// runReviewFor -- was the largest hole in this suite: every one of the following
// mutations left it green, because the tests all called the pure helpers with
// hand-built arguments instead of driving the real path.
//
//	assembleSuspectText(iss, gh.selfLogin, nil, …)   the maintainer set
//	alreadyHandled(iss, "", cfg.SpamLabel)           the bot's own identity
//	deleting gh.recordError() on the fetch path      exit 0 on a rate limit
//	deleting gh.recordError() in the event loop      exit 0 on a model error
//	deleting g.SetLimit(cfg.Concurrency)             unbounded parallel writes
//	for _, n := range issues[:1]                     only the first candidate

// issueWith builds a GraphQL issue body with a chosen author, labels and
// comments, so a test can produce the state it wants instead of asserting on
// state it installed by hand.
func issueWith(number int, author, body string, labels []string, comments ...Comment) string {
	labelNodes := make([]any, 0, len(labels))
	for _, l := range labels {
		labelNodes = append(labelNodes, map[string]any{"name": l})
	}
	commentNodes := make([]any, 0, len(comments))
	for _, c := range comments {
		commentNodes = append(commentNodes, map[string]any{
			"author":            map[string]any{"login": c.Author, "__typename": "User"},
			"authorAssociation": c.Association,
			"body":              c.Body,
		})
	}
	payload := map[string]any{"data": map[string]any{"repository": map[string]any{"issue": map[string]any{
		"number": number, "title": "Free followers", "body": body,
		"author":            map[string]any{"login": author, "__typename": "User"},
		"authorAssociation": "NONE",
		"labels":            map[string]any{"nodes": labelNodes},
		"comments":          map[string]any{"nodes": commentNodes},
	}}}}
	b, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// A maintainer's own issue and comments must never reach the model. The set is
// wired once in NewGitHubClient and read here through gh.maintainers; every
// other test builds its own set and passes it directly, which cannot observe
// whether the real call site still uses the wired one.
func TestReviewSkipsAnIssueWrittenEntirelyByMaintainers(t *testing.T) {
	body := issueWith(7, "Maint", "here is a link to our docs", nil,
		Comment{Author: "maint", Association: "MEMBER", Body: "and another"})
	// No scripted turns: reaching the model is itself the failure.
	h := newReviewHarness(t, body)
	h.gh.maintainers = maintainerSet([]string{"maint"})

	h.review(t, 7)

	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) on maintainer-authored content", got)
	}
	if got := h.writes.recorded(); len(got) != 0 {
		t.Errorf("a maintainer's issue was written to: %v", got)
	}
}

// An unlisted account is reviewed even when a maintainer set exists, so the
// skip above is the filter working rather than the review being switched off.
func TestReviewStillReviewsAStrangerWhenMaintainersAreConfigured(t *testing.T) {
	body := issueWith(7, "spammer", "buy followers cheap", nil)
	h := newReviewHarness(t, body, textTurn("No spam detected."))
	h.gh.maintainers = maintainerSet([]string{"maint"})

	h.review(t, 7)

	if got := h.model.calls(); got != 1 {
		t.Errorf("the model was invoked %d time(s), want 1: a stranger's issue must be reviewed", got)
	}
}

// An issue the bot has already alerted on is skipped without a model call, even
// though the spam label was removed. Recognizing its own alert needs the
// resolved identity, so this is what pins gh.selfLogin reaching the real check.
func TestReviewSkipsAnIssueTheBotAlreadyAlerted(t *testing.T) {
	body := issueWith(7, "spammer", "buy followers cheap", nil,
		Comment{Author: "spam-bot", Body: buildAlertComment("promo link")})
	h := newReviewHarness(t, body) // no turns: a model call is the failure

	h.review(t, 7)

	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) on an issue already alerted; a duplicate alert would follow", got)
	}
	if got := h.writes.recorded(); len(got) != 0 {
		t.Errorf("an already-alerted issue was written to again: %v", got)
	}
}

// A fetch that fails for any reason other than "not found" must be recorded, or
// a rate limit mid-sweep skips every issue and the run still exits 0.
func TestReviewRecordsAFetchFailure(t *testing.T) {
	const rateLimited = `{"data":{"repository":{"issue":null}},"errors":[{"type":"RATE_LIMITED","message":"rate limited"}]}`
	h := newReviewHarness(t, rateLimited) // no turns: a model call is the failure

	h.review(t, 7)

	if !h.gh.hadError() {
		t.Error("a failed fetch was not recorded, so a rate-limited sweep would exit 0 looking clean")
	}
	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) after the fetch failed", got)
	}
}

// A genuinely missing issue is the one fetch failure that must NOT be recorded:
// it is a skip, not an infrastructure problem.
func TestReviewDoesNotRecordAMissingIssue(t *testing.T) {
	const notFound = `{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND","message":"no such issue"}]}`
	h := newReviewHarness(t, notFound)

	h.review(t, 7)

	if h.gh.hadError() {
		t.Error("a missing issue was recorded as an infrastructure error; every deleted issue would fail the run")
	}
	// Without this the assertion above would also hold if runReviewFor returned
	// before doing anything at all.
	if got := h.writes.recorded(); len(got) != 0 {
		t.Errorf("a missing issue was written to: %v", got)
	}
	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) for an issue that does not exist", got)
	}
}

// errorModel yields a transport-level error, the way a model call that fails
// mid-stream does.
type errorModel struct{ calls atomic.Int64 }

func (m *errorModel) Name() string { return "error-test-model" }

func (m *errorModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("model unavailable"))
	}
}

// A model error must be recorded. Without it a sweep in which every model call
// failed reviews nothing and still exits 0.
func TestReviewRecordsAModelError(t *testing.T) {
	body := issueWith(7, "spammer", "buy followers cheap", nil)
	mdl := &errorModel{}
	h := newHarnessWith(t, func(int) string { return body }, mdl, nil)

	h.review(t, 7)

	if mdl.calls.Load() == 0 {
		t.Fatal("the model was never called, so this test cannot observe the error path")
	}
	if !h.gh.hadError() {
		t.Error("a failed model run was not recorded, so the run would exit 0 having reviewed nothing")
	}
}

// failingSessions is a session.Service whose Create always fails.
type failingSessions struct{ session.Service }

func (failingSessions) Create(context.Context, *session.CreateRequest) (*session.CreateResponse, error) {
	return nil, errors.New("session store unavailable")
}

// The session-create failure path had no test at all, and no reviewer lane
// found it. Without the recording, a broken session store silently reviews
// nothing and exits 0.
func TestReviewRecordsASessionCreateFailure(t *testing.T) {
	body := issueWith(7, "spammer", "buy followers cheap", nil)
	h := newReviewHarness(t, body) // no turns: a model call is the failure

	// Wrap the real factory so the runner is genuine and only Create fails.
	broken := func() (*runner.Runner, session.Service, error) {
		r, ss, err := h.reviewe()
		return r, failingSessions{ss}, err
	}
	reviewIssue(context.Background(), h.cfg, 7, func(ictx context.Context) {
		runReviewFor(ictx, broken, h.gh, h.cfg, discardLogger(), 7)
	})

	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) despite the session failing to open", got)
	}
	if !h.gh.hadError() {
		t.Error("a failed session create was not recorded, so the run would exit 0 having reviewed nothing")
	}
}

// gatedModel blocks every call until the test releases it, and announces each
// arrival. That makes the concurrency limit observable deterministically:
// counting how many calls are in flight at a moment in time is a race with the
// scheduler, and a version of this test that did so let `g.SetLimit` be deleted
// without failing.
type gatedModel struct {
	arrived chan string
	release chan struct{}
}

func newGatedModel(capacity int) *gatedModel {
	return &gatedModel{arrived: make(chan string, capacity), release: make(chan struct{})}
}

func (m *gatedModel) Name() string { return "gated-test-model" }

func (m *gatedModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var text strings.Builder
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p != nil {
				text.WriteString(p.Text)
			}
		}
	}
	m.arrived <- text.String()
	<-m.release

	// Text only. Driving the tool from here as well would also put the shared
	// []tool.Tool values under concurrency, but a tool call makes the agent take
	// a second turn and the resulting loop did not terminate reliably. That
	// sharing is therefore an assumption this suite does not verify -- recorded
	// rather than left implied.
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(textTurn("No spam detected."), nil)
	}
}

// waitArrival returns the next call's prompt, or fails if none comes.
func (m *gatedModel) waitArrival(t *testing.T, why string) string {
	t.Helper()
	select {
	case s := <-m.arrived:
		return s
	case <-time.After(10 * time.Second):
		t.Fatalf("no model call arrived: %s", why)
		return ""
	}
}

// reviewAll had no test of any kind. This one drives the real errgroup over
// several issues -- which is also the only place -race observes the agent,
// runner and session service being used by concurrent reviews. Sharing one
// runner across them races inside ADK, and this is the test that shows it.
func TestReviewAllReviewsEveryIssueUpToTheConcurrencyLimit(t *testing.T) {
	// A limit of 3 rather than 2: with 2, replacing g.SetLimit(cfg.Concurrency)
	// by any small literal would still pass, so the config-to-limit link would
	// not be pinned. 3 is also what the workflow ships.
	const (
		total = 7
		limit = 3
	)
	mdl := newGatedModel(total * 2) // headroom, so a send can never block
	h := newHarnessWith(t, func(n int) string {
		return issueWith(n, fmt.Sprintf("user%d", n), fmt.Sprintf("spam text for %d", n), nil)
	}, mdl, nil)
	h.cfg.Concurrency = limit

	issues := make([]int, 0, total)
	for i := 1; i <= total; i++ {
		issues = append(issues, i)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.reviewAll(context.Background(), issues)
	}()
	// Released here too, so a t.Fatal below cannot leave that goroutine blocked
	// on <-m.release forever, outliving the test and racing the package-level
	// newNonce that another test swaps.
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(mdl.release) }) }
	t.Cleanup(func() {
		release()
		<-done
	})

	// Exactly `limit` reviews may be in flight. Every one of them blocks in the
	// model until released, so with the limit applied the next arrival cannot
	// happen until one is let go.
	seen := make([]string, 0, total)
	for range limit {
		seen = append(seen, mdl.waitArrival(t, "the concurrency limit is starving the sweep"))
	}
	select {
	case extra := <-mdl.arrived:
		seen = append(seen, extra)
		t.Errorf("a %d+1st review started while %d were already in flight: the concurrency limit is not applied", limit, limit)
	case <-time.After(300 * time.Millisecond):
		// Nothing more started, which is the limit holding.
	}

	release()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("reviewAll did not finish")
	}
	for len(mdl.arrived) > 0 {
		seen = append(seen, <-mdl.arrived)
	}

	if len(seen) != total {
		t.Errorf("reviewed %d issue(s), want %d: %d candidates were dropped", len(seen), total, total-len(seen))
	}
	// Every issue's own text reached the model, so the loop is not reviewing one
	// issue repeatedly.
	for i := 1; i <= total; i++ {
		want := fmt.Sprintf("spam text for %d", i)
		found := false
		for _, s := range seen {
			if strings.Contains(s, want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("issue %d was never sent to the model", i)
		}
	}
}

// Once the run budget is gone, queued issues must not start. Each one would
// otherwise fail its fetch on the dead context and record an error, burying the
// budget error under one spurious line per issue.
func TestReviewAllSkipsQueuedIssuesOnceTheBudgetIsGone(t *testing.T) {
	h := newReviewHarness(t, issueJSON(1, "spam")) // no turns: a model call is the failure
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	h.reviewAll(ctx, []int{1, 2, 3, 4, 5})

	if got := h.model.calls(); got != 0 {
		t.Errorf("the model was invoked %d time(s) after the budget expired", got)
	}
	if h.gh.hadError() {
		t.Error("skipping an issue the budget never allowed was recorded as a failure")
	}
}

// Two tool calls for the same issue, concurrently. Exactly one may write, and
// the one that loses the claim while the winner's write is still in flight must
// not report a success it cannot vouch for.
//
// The interleaving is forced rather than hoped for: the winner's first HTTP
// call blocks until the loser has been observed arriving. Racing two goroutines
// and asserting on whatever happened made the outcome depend on the scheduler,
// because a loser that arrives AFTER the winner finished legitimately gets
// "already flagged this run", which is a success.
func TestFlagAsSpamConcurrentSameIssue(t *testing.T) {
	var calls atomic.Int32
	secondArrived := make(chan struct{})
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			// Hold the winner inside FlagSpam until the loser has tried to claim.
			select {
			case <-secondArrived:
			case <-time.After(10 * time.Second):
			}
		}
		if strings.HasSuffix(r.URL.Path, "/comments") {
			_, _ = io.WriteString(w, `{"id":1}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"spam"}]`)
	}))
	ctx := withAuditedIssue(context.Background(), 7)

	type outcome struct {
		res actionResult
		err error
	}
	// The result travels over the channel rather than through t.Errorf: on the
	// t.Fatal paths below the test returns while this goroutine is still inside
	// flagAsSpam, and logging from it then panics with "Log in goroutine after
	// test has completed", turning a clear failure into an unrelated crash.
	winner := make(chan outcome, 1)
	go func() {
		res, err := c.flagAsSpam(ctx, 7, "promo")
		winner <- outcome{res, err}
	}()

	// Wait until the winner holds the claim, then race the loser against it.
	deadline := time.Now().Add(10 * time.Second)
	for c.flagState(7) != flagInFlight {
		if time.Now().After(deadline) {
			t.Fatal("the first caller never claimed the issue")
		}
		time.Sleep(time.Millisecond)
	}
	loser, err := c.flagAsSpam(ctx, 7, "promo")
	close(secondArrived)
	if err != nil {
		t.Fatalf("the losing flagAsSpam returned a Go error: %v", err)
	}
	if loser.Status != "error" {
		t.Errorf("the loser reported %+v while the winner's write was still in flight; that write can still fail", loser)
	}
	if !strings.Contains(loser.Message, "in progress") {
		t.Errorf("loser message = %q, want it to say the write is still in progress", loser.Message)
	}

	w := <-winner
	if w.err != nil {
		t.Errorf("the winning flagAsSpam returned an error: %v", w.err)
	}
	if w.res.Status != "success" {
		t.Errorf("the winner reported %q, want success", w.res.Status)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d HTTP calls, want 2 (one comment and one label, once)", got)
	}
	if got := c.flagState(7); got != flagSucceeded {
		t.Errorf("flag state = %v, want flagSucceeded", got)
	}
}

// errorCodeModel yields an event carrying an ErrorCode, the way a model that
// refuses (safety, quota) does. That is a different branch from a transport
// error and had no test.
type errorCodeModel struct{ calls atomic.Int64 }

func (m *errorCodeModel) Name() string { return "errorcode-test-model" }

func (m *errorCodeModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{ErrorCode: "SAFETY", ErrorMessage: "blocked"}, nil)
	}
}

func TestReviewRecordsAModelErrorCode(t *testing.T) {
	body := issueWith(7, "spammer", "buy followers cheap", nil)
	mdl := &errorCodeModel{}
	h := newHarnessWith(t, func(int) string { return body }, mdl, nil)

	h.review(t, 7)

	if mdl.calls.Load() == 0 {
		t.Fatal("the model was never called, so this test cannot observe the error path")
	}
	if !h.gh.hadError() {
		t.Error("a model that returned an error code was not recorded, so the run would exit 0 having reviewed nothing")
	}
}

// A panic in one review must not take the process down: a sibling goroutine may
// be between the alert comment and the label, and the process dying there
// leaves an issue commented but unlabeled with nothing in the log to say why.
//
// The panic is injected at newNonce, which runs inside runReviewFor before the
// agent is built. A panic in the MODEL would not exercise this: ADK recovers
// those itself and turns them into an error event, so a version of this test
// that used a panicking model passed with the recover deleted.
func TestReviewAllSurvivesAPanicInOneReview(t *testing.T) {
	h := newHarnessWith(t, func(n int) string {
		return issueWith(n, fmt.Sprintf("user%d", n), "buy followers cheap", nil)
	}, &scriptedModel{t: t, turns: []*model.LLMResponse{textTurn("No spam detected.")}}, nil)
	h.cfg.Concurrency = 1

	orig := newNonce
	newNonce = func() (string, error) { panic("entropy source exploded") }
	t.Cleanup(func() { newNonce = orig })

	// If the panic escapes, the test binary dies here rather than failing.
	h.reviewAll(context.Background(), []int{1, 2})

	if !h.gh.hadError() {
		t.Error("a panic during a review was not recorded, so the run would exit 0")
	}
}

// sweep is the only place that maps recorded errors onto a non-zero exit, and
// the only place the real instruction reaches the agent. Neither had a test:
// deleting the hadError check, or replacing renderPrompt(cfg) with "", left the
// whole suite green while turning a rate-limited sweep into a clean exit 0 and
// a moderation bot into one with no classification rules.
func sweepHarness(t *testing.T, graphQL func(int) string, mdl model.LLM, searchBody string) (*Config, sweepDeps, *writeRecorder) {
	t.Helper()
	writes := &writeRecorder{graphQ: graphQL}
	cfg := testConfig()
	cfg.IssueTimeout = 30 * time.Second
	cfg.RunTimeout = time.Minute

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/user":
			_, _ = io.WriteString(w, `{"login":"spam-bot"}`)
		case "/search/issues":
			_, _ = io.WriteString(w, searchBody)
		default:
			writes.handler()(w, r)
		}
	})
	rest := restClient(t, handler)
	deps := sweepDeps{
		newClient: func(ctx context.Context, c *Config, l *slog.Logger) (*GitHubClient, error) {
			return newGitHubClient(ctx, rest, c, l)
		},
		newModel: func(context.Context, *Config) (model.LLM, error) { return mdl, nil },
	}
	return cfg, deps, writes
}

func TestSweepReviewsTheCandidatesAndExitsCleanly(t *testing.T) {
	mdl := &scriptedModel{t: t, turns: []*model.LLMResponse{textTurn("No spam detected.")}}
	cfg, deps, writes := sweepHarness(t,
		func(n int) string { return issueWith(n, "spammer", "buy followers", nil) },
		mdl, `{"items":[{"number":4}]}`)

	if err := sweep(context.Background(), cfg, discardLogger(), deps); err != nil {
		t.Fatalf("sweep() on a clean run = %v, want nil", err)
	}
	if got := mdl.calls(); got != 1 {
		t.Errorf("the model was called %d time(s), want 1", got)
	}
	if got := writes.recorded(); len(got) != 0 {
		t.Errorf("a clean run wrote to GitHub: %v", got)
	}

	// The agent must receive the RENDERED instruction, not a placeholder: this
	// is the only assertion anywhere that the prompt reaches the model at all.
	mdl.mu.Lock()
	req := mdl.requests[0]
	mdl.mu.Unlock()
	if req.Config == nil || req.Config.SystemInstruction == nil {
		t.Fatal("the agent ran with no system instruction")
	}
	var instruction strings.Builder
	for _, p := range req.Config.SystemInstruction.Parts {
		if p != nil {
			instruction.WriteString(p.Text)
		}
	}
	for _, want := range []string{"automated spam-moderation agent", "google/adk-go", "'spam'"} {
		if !strings.Contains(instruction.String(), want) {
			t.Errorf("the instruction the model received is missing %q", want)
		}
	}
	// The output cap is the Go-side half of bounding what the model can write
	// into a public comment.
	if req.Config.MaxOutputTokens != 512 {
		t.Errorf("MaxOutputTokens = %d, want 512", req.Config.MaxOutputTokens)
	}
}

// A recorded infrastructure error must become a non-zero exit. Every
// recordError test in this package terminates in this one line.
func TestSweepFailsWhenAnIssueCouldNotBeProcessed(t *testing.T) {
	const rateLimited = `{"data":{"repository":{"issue":null}},"errors":[{"type":"RATE_LIMITED","message":"rate limited"}]}`
	mdl := &scriptedModel{t: t} // no turns: a model call is the failure
	cfg, deps, _ := sweepHarness(t, func(int) string { return rateLimited }, mdl, `{"items":[{"number":4}]}`)

	err := sweep(context.Background(), cfg, discardLogger(), deps)
	if err == nil {
		t.Fatal("sweep() returned nil after every issue failed to process; the run would exit 0 looking clean")
	}
	if !strings.Contains(err.Error(), "failed to process") {
		t.Errorf("sweep() error = %v, want it to name the failure", err)
	}
}

// A sweep that finds nothing is not a failure.
func TestSweepWithNoCandidatesIsNotAnError(t *testing.T) {
	mdl := &scriptedModel{t: t}
	cfg, deps, _ := sweepHarness(t, func(int) string { return "" }, mdl, `{"items":[]}`)

	if err := sweep(context.Background(), cfg, discardLogger(), deps); err != nil {
		t.Errorf("sweep() with no candidates = %v, want nil", err)
	}
	if got := mdl.calls(); got != 0 {
		t.Errorf("the model was called %d time(s) with no candidates", got)
	}
}

// A tool that returns a Go error must reach the OnToolError callback as an
// observation, not a replacement: returning a non-nil map there would swap the
// error the model sees for something the bot made up. The callback is wired in
// newReviewer, so only a run through the real agent loop exercises it.
func TestAgentLoopObservesAToolErrorWithoutReplacingIt(t *testing.T) {
	writes := &writeRecorder{graphQ: func(int) string { return issueWith(7, "spammer", "buy followers cheap", nil) }}
	cfg := testConfig()
	cfg.IssueTimeout = 30 * time.Second
	// Every write fails, so FlagSpam returns a Go error and the tool does too.
	gh := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/graphql") {
			writes.handler()(w, r)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))

	mdl := &scriptedModel{t: t, turns: []*model.LLMResponse{
		flagCallTurn(7, "promotional link"),
		textTurn("The flag failed."),
	}}
	tools, err := gh.tools()
	if err != nil {
		t.Fatalf("tools: %v", err)
	}
	// A capturing logger, so the callback's execution is observable. Without it
	// the test passes with OnToolErrorCallbacks deleted outright: ADK serializes
	// the tool's Go error to the model either way, so every other assertion here
	// holds whether or not the callback ever runs.
	var logged bytes.Buffer
	capture := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug}))
	factory := reviewerFor(cfg, mdl, tools, "test", capture)
	reviewIssue(context.Background(), cfg, 7, func(ictx context.Context) {
		runReviewFor(ictx, factory, gh, cfg, discardLogger(), 7)
	})

	if !strings.Contains(logged.String(), "tool call failed") {
		t.Errorf("the OnToolError callback never ran:\n%s", logged.String())
	}
	if !strings.Contains(logged.String(), "flag_issue_as_spam") {
		t.Errorf("the callback ran without naming the tool:\n%s", logged.String())
	}
	if !gh.hadError() {
		t.Error("a failed write inside the tool was not recorded, so the run would exit 0")
	}
	// The model must be handed the real failure, not a substituted result.
	got := mdl.functionResponses()
	if len(got) != 1 {
		t.Fatalf("the tool produced %d response(s), want 1", len(got))
	}
	// "500" specifically: the substring "error" alone matches ADK's own error
	// envelope key and would hold whatever the tool returned.
	if !strings.Contains(got[0], "500") {
		t.Errorf("the model was handed %q, which does not carry the write failure", got[0])
	}
	if got := gh.flagState(7); got != flagFailed {
		t.Errorf("flag state = %v, want flagFailed so a second call is told the truth", got)
	}
}

// Two reviews in one run must not share a fence marker: a nonce reused across
// issues is one an attacker who saw the first prompt could pre-write into the
// second. The generator is tested separately; this pins the call site.
func TestEachReviewGetsItsOwnNonce(t *testing.T) {
	mdl := &scriptedModel{t: t, turns: []*model.LLMResponse{
		textTurn("No spam detected."), textTurn("No spam detected."),
	}}
	h := newHarnessWith(t, func(n int) string {
		return issueWith(n, fmt.Sprintf("user%d", n), fmt.Sprintf("spam text for %d", n), nil)
	}, mdl, mdl)

	h.review(t, 1)
	first := h.model.prompt()
	h.review(t, 2)

	h.model.mu.Lock()
	second := promptOf(h.model.requests[len(h.model.requests)-1])
	h.model.mu.Unlock()

	n1, n2 := nonceIn(t, first), nonceIn(t, second)
	if n1 == n2 {
		t.Errorf("both reviews used the fence marker %q; one issue's prompt reveals the next issue's fence", n1)
	}
}

func promptOf(req *model.LLMRequest) string {
	var b strings.Builder
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p != nil {
				b.WriteString(p.Text)
			}
		}
	}
	return b.String()
}

func nonceIn(t *testing.T, prompt string) string {
	t.Helper()
	i := strings.Index(prompt, "\n[UNTRUSTED:")
	if i < 0 {
		t.Fatalf("no fence in the prompt:\n%s", prompt)
	}
	rest := prompt[i+len("\n[UNTRUSTED:"):]
	end := strings.Index(rest, "]")
	if end < 0 {
		t.Fatalf("unterminated fence marker:\n%s", prompt)
	}
	return rest[:end]
}

// The reviewer factory must hand every concurrent review its OWN runner.
//
// ADK's runner.Run lazily initializes mutable state on the agent it is given
// (a read at runner.go:210 against a write at :212 in v2.2.0), so a shared
// runner races as soon as two reviews overlap — which the shipped
// CONCURRENCY_LIMIT=3 does on any sweep with two candidates.
//
// Two things make this catch a regression where an earlier attempt did not.
//
// It calls reviewerFor, the function sweep uses, rather than assembling an
// equivalent closure. While sweep built its factory inline and the tests built
// their own, memoizing sweep's closure into one shared runner reintroduced the
// race and FIVE fresh `go test -race` processes reported it zero times.
//
// And the barrier sits before Run, not inside the model. The racing access is
// at the top of Run, so a model that blocks has already let the workers past it
// one at a time; holding each worker once it HOLDS a runner and releasing them
// together is what puts them on the write at once.
//
// Two distinct regressions, measured separately in fresh processes, because
// `-count=N` is not N attempts at a once-per-process initialization:
//
//	reviewerFor memoized into a shared singleton   DATA RACE 10/10, test fails 10/10
//	the factory call hoisted out of reviewAll's    DATA RACE  7-9/10, test fails 10/10
//	  per-review path
//
// The hoist is caught every time despite the race detector missing some runs,
// because it also stops the factory being called once per review, and
// the holding count below asserts that outright. Do not drop that assertion in
// favour of the race detector.
//
// 32 workers rather than a handful: the racing window is two instructions, so
// detection scales with how many worker pairs collide. Measured at 4 workers
// the memoize regression was still 10/10 on the machine this was written on,
// but that is one machine, and the extra workers run a stub model with no I/O.
func TestReviewAllGivesEachConcurrentReviewItsOwnRunner(t *testing.T) {
	const workers = 32

	writes := &writeRecorder{graphQ: func(n int) string {
		return issueWith(n, fmt.Sprintf("user%d", n), fmt.Sprintf("spam text for %d", n), nil)
	}}
	cfg := testConfig()
	cfg.IssueTimeout = 30 * time.Second
	cfg.Concurrency = workers // every worker must be in flight for the barrier to fill
	gh := testClient(t, cfg, writes.handler())

	mdl := &countingModel{}
	tools, err := gh.tools()
	if err != nil {
		t.Fatalf("tools: %v", err)
	}
	production := reviewerFor(cfg, mdl, tools, "test", discardLogger())

	// Hold each worker once it holds a runner, and release them together.
	barrier := make(chan struct{})
	barrierTimeout := time.After(5 * time.Second)
	var holding atomic.Int32
	var released sync.Once
	factory := func() (*runner.Runner, session.Service, error) {
		r, ss, err := production()
		if err != nil {
			return nil, nil, err
		}
		if holding.Add(1) >= workers {
			released.Do(func() { close(barrier) })
		}
		select {
		case <-barrier:
		case <-barrierTimeout:
			// Reached when the factory is called fewer than `workers` times,
			// which is what hoisting the call out of the per-review path looks
			// like from here. Short, because a healthy run fills the barrier in
			// milliseconds and this timeout is otherwise pure dead wait on a
			// failure.
			t.Error("the barrier never filled: the factory was not called once per review")
		}
		return r, ss, nil
	}
	t.Cleanup(func() { released.Do(func() { close(barrier) }) })

	issues := make([]int, 0, workers)
	for i := 1; i <= workers; i++ {
		issues = append(issues, i)
	}
	reviewAll(context.Background(), factory, gh, cfg, discardLogger(), issues)

	if got := int(holding.Load()); got != workers {
		t.Errorf("%d review(s) reached the factory, want %d", got, workers)
	}
	if got := mdl.calls.Load(); got != workers {
		t.Errorf("the model ran %d time(s), want %d: some review never entered Run", got, workers)
	}
	if gh.hadError() {
		t.Error("a review failed, so this test may not have exercised what it claims")
	}
}

// countingModel answers immediately and counts. The blocking belongs at the
// barrier before Run, not here.
type countingModel struct{ calls atomic.Int64 }

func (m *countingModel) Name() string { return "counting-test-model" }

func (m *countingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(textTurn("No spam detected."), nil)
	}
}

// The same invariant one level up. This drives sweep itself, so it reaches a
// hoist introduced between reviewerFor and reviewAll — which the test above,
// calling reviewAll directly with its own factory, cannot see.
//
// It is NOT a pin, and should not be read as one. sweep builds the factory
// internally, which is the point, so there is nowhere to put a barrier and
// detection rests entirely on the race detector catching two workers on the
// same two instructions. Measured with sweep hoisting one shared reviewer:
// DATA RACE in 7 of 10 fresh processes at 32 issues. Raising the count makes it
// WORSE, not better — 4 of 10 at 96 — because each worker fetches its issue
// before reaching Run, and more workers queueing through one httptest server
// spreads their arrivals apart rather than colliding them. So the usual "add
// workers until it is deterministic" move does not apply here; what makes the
// test above deterministic is the barrier, and that is unavailable here.
//
// It earns its place as a probabilistic backstop on a variant nothing else
// covers. The deterministic guarantee lives in the test above.
func TestSweepGivesEachConcurrentReviewItsOwnRunner(t *testing.T) {
	const issues = 32

	var items []string
	for i := 1; i <= issues; i++ {
		items = append(items, fmt.Sprintf(`{"number":%d}`, i))
	}
	mdl := &countingModel{}
	cfg, deps, _ := sweepHarness(t,
		func(n int) string {
			return issueWith(n, fmt.Sprintf("user%d", n), fmt.Sprintf("spam text for %d", n), nil)
		},
		mdl, `{"items":[`+strings.Join(items, ",")+`]}`)
	cfg.IssueCount = issues
	cfg.Concurrency = issues

	if err := sweep(context.Background(), cfg, discardLogger(), deps); err != nil {
		t.Fatalf("sweep() = %v, want nil", err)
	}
	if got := mdl.calls.Load(); got != issues {
		t.Errorf("the model ran %d time(s), want %d", got, issues)
	}
}
