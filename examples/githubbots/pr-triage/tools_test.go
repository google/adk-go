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
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// recordingHandler answers the two endpoints the mutating tools use and records
// every request, so a test can assert both that nothing was written and exactly
// what was written.
type recordingHandler struct {
	mu sync.Mutex
	// requests holds "METHOD path" for every call.
	requests []string
	// bodies holds the decoded JSON body of every POST.
	bodies []map[string]any
	// assignable controls what the assignee-check endpoint answers.
	assignable bool
	// status, when non-zero, is returned for every mutation.
	status int
}

func newRecordingHandler() *recordingHandler {
	return &recordingHandler{assignable: true}
}

func (h *recordingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	h.requests = append(h.requests, r.Method+" "+r.URL.Path)
	if r.Method == http.MethodPost {
		var body map[string]any
		if raw, err := io.ReadAll(r.Body); err == nil {
			_ = json.Unmarshal(raw, &body)
		}
		h.bodies = append(h.bodies, body)
	}
	assignable, status := h.assignable, h.status
	h.mu.Unlock()

	// GET /repos/{o}/{r}/assignees/{login}: 204 assignable, 404 not.
	if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/assignees/") {
		if assignable {
			w.WriteHeader(http.StatusNoContent)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
		return
	}
	if status != 0 {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
		return
	}
	if strings.HasSuffix(r.URL.Path, "/comments") {
		_, _ = io.WriteString(w, `{"id":1}`)
		return
	}
	_, _ = io.WriteString(w, `{"number":7}`)
}

func (h *recordingHandler) calls() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.requests...)
}

// writes returns only the mutating calls. The assignability probe is a read and
// must not be counted as a write.
func (h *recordingHandler) writes() []string {
	var out []string
	for _, c := range h.calls() {
		if strings.HasPrefix(c, "POST ") {
			out = append(out, c)
		}
	}
	return out
}

func (h *recordingHandler) postedBodies() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]map[string]any(nil), h.bodies...)
}

// eligibleClient builds a client whose pull request #7 has already passed the
// Go-side preconditions, which is the state every tool call assumes.
func eligibleClient(t *testing.T, cfg *Config, h http.Handler) *GitHubClient {
	t.Helper()
	c := testClient(t, cfg, h)
	c.markEligible(7)
	return c
}

func TestAuthorizePR(t *testing.T) {
	ctx := withAuditedPR(context.Background(), 7)
	if _, ok := authorizePR(ctx, 7); !ok {
		t.Error("authorizePR(7) on a session scoped to 7 = not ok, want ok")
	}
	if _, ok := authorizePR(ctx, 8); ok {
		t.Error("authorizePR(8) on a session scoped to 7 = ok, want not ok")
	}
	if _, ok := authorizePR(context.Background(), 7); ok {
		t.Error("authorizePR on an unscoped session = ok, want not ok")
	}
}

// --- assign_owner_to_pull_request -------------------------------------------

func TestAssignOwnerHappyPath(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core")
	if err != nil {
		t.Fatalf("assignOwner() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q (%s), want success", res.Status, res.Message)
	}
	writes := h.writes()
	if len(writes) != 1 || !strings.HasSuffix(writes[0], "/issues/7/assignees") {
		t.Fatalf("writes = %v, want one POST to /issues/7/assignees", writes)
	}
	// The login must be the one Go resolved from the map, not anything the
	// caller supplied.
	bodies := h.postedBodies()
	if len(bodies) != 1 {
		t.Fatalf("got %d posted bodies, want 1", len(bodies))
	}
	assignees, _ := bodies[0]["assignees"].([]any)
	if len(assignees) != 1 || assignees[0] != "alice" {
		t.Errorf("posted assignees = %v, want [alice] from the owner map", assignees)
	}
}

// The model names a component, never a person. Anything that is not a key of
// the configured map must be refused before any HTTP call, so injected text has
// no path to a login of its own choosing.
func TestAssignOwnerRejectsAnythingButAConfiguredComponent(t *testing.T) {
	for _, component := range []string{
		"", "mallory", "alice", "not-a-component", "core; tools", "co\nre", "core tools", "*",
	} {
		t.Run("component="+component, func(t *testing.T) {
			h := newRecordingHandler()
			c := eligibleClient(t, testConfig(), h)
			res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, component)
			if err != nil {
				t.Fatalf("assignOwner() unexpected Go error = %v", err)
			}
			if res.Status != "error" {
				t.Errorf("status = %q, want error for component %q", res.Status, component)
			}
			if calls := h.calls(); len(calls) != 0 {
				t.Errorf("made %d HTTP calls for an unknown component, want 0: %v", len(calls), calls)
			}
		})
	}
	// A configured component still resolves whatever the casing or surrounding
	// whitespace, so the model is not tripped by its own formatting.
	for _, ok := range []string{"  CORE  ", "core\n", "\tTools"} {
		h := newRecordingHandler()
		c := eligibleClient(t, testConfig(), h)
		if res, _ := c.assignOwner(withAuditedPR(context.Background(), 7), 7, ok); res.Status != "success" {
			t.Errorf("status = %q for component %q, want success", res.Status, ok)
		}
	}
}

func TestAssignOwnerRejectsWrongPullRequestWithoutHTTP(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	c.markEligible(8) // even an eligible OTHER pull request must be refused
	res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 8, "core")
	if err != nil {
		t.Fatalf("assignOwner() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (session is scoped to #7)", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
}

func TestAssignOwnerRejectsUnscopedSessionWithoutHTTP(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	res, err := c.assignOwner(context.Background(), 7, "core")
	if err != nil {
		t.Fatalf("assignOwner() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (no session scope)", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
}

// Eligibility is decided in Go from API metadata. A pull request that never
// passed that gate cannot be mutated, whatever the model asks for.
func TestAssignOwnerRefusesAnIneligiblePullRequest(t *testing.T) {
	h := newRecordingHandler()
	c := testClient(t, testConfig(), h) // deliberately NOT marked eligible
	res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core")
	if err != nil {
		t.Fatalf("assignOwner() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (never cleared for triage)", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
}

// One assignment per pull request, ever. A model that calls the tool twice must
// not reach the API a second time.
func TestAssignOwnerSecondCallIsRefused(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)
	if res, _ := c.assignOwner(ctx, 7, "core"); res.Status != "success" {
		t.Fatalf("first assign status = %q, want success", res.Status)
	}
	res, err := c.assignOwner(ctx, 7, "tools")
	if err != nil {
		t.Fatalf("second assignOwner() error = %v", err)
	}
	if res.Status != "skipped" {
		t.Errorf("second assign status = %q, want skipped", res.Status)
	}
	if got := len(h.writes()); got != 1 {
		t.Errorf("made %d writes, want 1 (the second assign must not reach the API)", got)
	}
}

// A non-assignable owner is a configuration problem for a human. The tool must
// report it and stop: releasing the claim would let the model walk the component
// map until it found an assignable login, which is both a retry loop and a way
// for injected text to steer the assignee.
func TestAssignOwnerNonAssignableIsReportedAndEndsTheAttempt(t *testing.T) {
	h := newRecordingHandler()
	h.assignable = false
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)

	res, err := c.assignOwner(ctx, 7, "core")
	if err != nil {
		t.Fatalf("assignOwner() error = %v", err)
	}
	if res.Status != "skipped" {
		t.Fatalf("status = %q (%s), want skipped", res.Status, res.Message)
	}
	if writes := h.writes(); len(writes) != 0 {
		t.Errorf("a non-assignable owner still produced writes: %v", writes)
	}
	// A non-assignable owner is not an infrastructure failure, so the run must
	// still be able to exit 0.
	if c.hadError() {
		t.Error("a non-assignable owner was recorded as a run error")
	}

	// The retry the spec forbids.
	h.assignable = true
	res, err = c.assignOwner(ctx, 7, "tools")
	if err != nil {
		t.Fatalf("retry assignOwner() error = %v", err)
	}
	if res.Status == "success" {
		t.Error("the model walked to a second component after a non-assignable owner")
	}
	if writes := h.writes(); len(writes) != 0 {
		t.Errorf("the retry reached the API: %v", writes)
	}
}

func TestAssignOwnerDryRunWritesNothing(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	h := newRecordingHandler()
	c := eligibleClient(t, cfg, h)
	res, err := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core")
	if err != nil {
		t.Fatalf("assignOwner() dry-run error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if writes := h.writes(); len(writes) != 0 {
		t.Errorf("dry-run performed %d writes, want 0: %v", len(writes), writes)
	}
}

func TestAssignOwnerRESTErrorIsGoErrorAndRecorded(t *testing.T) {
	h := newRecordingHandler()
	h.status = http.StatusInternalServerError
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)
	if c.hadError() {
		t.Fatal("hadError() should start false")
	}
	if _, err := c.assignOwner(ctx, 7, "core"); err == nil {
		t.Fatal("assignOwner() expected a Go error on HTTP 500, got nil")
	}
	if !c.hadError() {
		t.Error("the infrastructure error was not recorded, so the run would exit 0")
	}
	// A second call must report the failure rather than "already done":
	// otherwise the model's transcript records an assignment that never landed.
	res, err := c.assignOwner(ctx, 7, "core")
	if err != nil {
		t.Fatalf("second assignOwner() returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "already failed") {
		t.Errorf("second assign = %+v, want an error saying the attempt already failed", res)
	}
}

// --- request_more_context ---------------------------------------------------

func TestRequestMoreContextHappyPath(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, []string{"problem", "testing"})
	if err != nil {
		t.Fatalf("requestMoreContext() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q (%s), want success", res.Status, res.Message)
	}
	writes := h.writes()
	if len(writes) != 1 || !strings.HasSuffix(writes[0], "/issues/7/comments") {
		t.Fatalf("writes = %v, want one POST to /issues/7/comments", writes)
	}
	body, _ := h.postedBodies()[0]["body"].(string)
	want, _ := contextItemText("problem")
	if !strings.Contains(body, want) {
		t.Errorf("posted body does not contain the allow-listed text:\n%s", body)
	}
}

// The comment body is assembled from constants. The model supplies keys, and a
// key that is not in the allow-list must never reach GitHub — this is what keeps
// attacker-influenced prose out of anything posted under the repo's identity.
func TestRequestMoreContextRejectsUnknownItemsWithoutHTTP(t *testing.T) {
	for _, items := range [][]string{
		nil,
		{},
		{"problem", "<script>alert(1)</script>"},
		{"@everyone please review"},
		{"PROBLEM!"},
		{"testing", "testing", "testing", "testing", "testing", "testing", "testing"},
	} {
		t.Run(strings.Join(items, "|"), func(t *testing.T) {
			h := newRecordingHandler()
			c := eligibleClient(t, testConfig(), h)
			res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, items)
			if err != nil {
				t.Fatalf("requestMoreContext() unexpected Go error = %v", err)
			}
			if res.Status != "error" {
				t.Errorf("status = %q, want error for items %v", res.Status, items)
			}
			if calls := h.calls(); len(calls) != 0 {
				t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
			}
		})
	}
}

// A rejected item list must not burn the pull request's single comment: the
// model should be able to correct a typo and try again with valid keys.
func TestRequestMoreContextTypoDoesNotSpendTheClaim(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)
	if res, _ := c.requestMoreContext(ctx, 7, []string{"problemm"}); res.Status != "error" {
		t.Fatalf("a typo was accepted: %+v", res)
	}
	if res, _ := c.requestMoreContext(ctx, 7, []string{"problem"}); res.Status != "success" {
		t.Errorf("the corrected call was refused: %+v", res)
	}
	if got := len(h.writes()); got != 1 {
		t.Errorf("made %d writes, want 1", got)
	}
}

func TestRequestMoreContextSecondCallIsRefused(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)
	if res, _ := c.requestMoreContext(ctx, 7, []string{"problem"}); res.Status != "success" {
		t.Fatal("the first request was refused")
	}
	res, err := c.requestMoreContext(ctx, 7, []string{"testing"})
	if err != nil {
		t.Fatalf("second requestMoreContext() error = %v", err)
	}
	if res.Status != "skipped" {
		t.Errorf("second request status = %q, want skipped", res.Status)
	}
	if got := len(h.writes()); got != 1 {
		t.Errorf("made %d writes, want 1 (no duplicate comment)", got)
	}
}

// markSpent is how a comment posted on an EARLIER run consumes this run's
// claim, so a reopened pull request is never asked twice.
func TestRequestMoreContextRefusedWhenAlreadySpent(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	c.markSpent(7, actionComment)
	res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, []string{"problem"})
	if err != nil {
		t.Fatalf("requestMoreContext() error = %v", err)
	}
	if res.Status != "skipped" {
		t.Errorf("status = %q, want skipped", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
	// Spending the comment claim must not spend the assignment claim.
	if res, _ := c.assignOwner(withAuditedPR(context.Background(), 7), 7, "core"); res.Status != "success" {
		t.Errorf("assignment was blocked by the comment claim: %+v", res)
	}
}

func TestRequestMoreContextRejectsWrongPullRequestWithoutHTTP(t *testing.T) {
	h := newRecordingHandler()
	c := eligibleClient(t, testConfig(), h)
	c.markEligible(8)
	res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 8, []string{"problem"})
	if err != nil {
		t.Fatalf("requestMoreContext() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (session is scoped to #7)", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
}

func TestRequestMoreContextRefusesAnIneligiblePullRequest(t *testing.T) {
	h := newRecordingHandler()
	c := testClient(t, testConfig(), h)
	res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, []string{"problem"})
	if err != nil {
		t.Fatalf("requestMoreContext() unexpected Go error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (never cleared for triage)", res.Status)
	}
	if calls := h.calls(); len(calls) != 0 {
		t.Errorf("made %d HTTP calls, want 0: %v", len(calls), calls)
	}
}

func TestRequestMoreContextDryRunWritesNothing(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	h := newRecordingHandler()
	c := eligibleClient(t, cfg, h)
	res, err := c.requestMoreContext(withAuditedPR(context.Background(), 7), 7, []string{"problem"})
	if err != nil {
		t.Fatalf("requestMoreContext() dry-run error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if writes := h.writes(); len(writes) != 0 {
		t.Errorf("dry-run performed %d writes, want 0: %v", len(writes), writes)
	}
}

func TestRequestMoreContextRESTErrorIsGoErrorAndRecorded(t *testing.T) {
	h := newRecordingHandler()
	h.status = http.StatusInternalServerError
	c := eligibleClient(t, testConfig(), h)
	ctx := withAuditedPR(context.Background(), 7)
	if _, err := c.requestMoreContext(ctx, 7, []string{"problem"}); err == nil {
		t.Fatal("requestMoreContext() expected a Go error on HTTP 500, got nil")
	}
	if !c.hadError() {
		t.Error("the infrastructure error was not recorded, so the run would exit 0")
	}
	res, err := c.requestMoreContext(ctx, 7, []string{"problem"})
	if err != nil {
		t.Fatalf("second requestMoreContext() returned a Go error: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "already failed") {
		t.Errorf("second request = %+v, want an error saying the attempt already failed", res)
	}
}

// --- concurrency ------------------------------------------------------------

// Pull requests are triaged concurrently against one shared client. Each session
// is scoped to its own number and must only ever act on that one. Run under
// -race to catch data races on the shared claim table.
func TestToolsAreIsolatedAcrossConcurrentSessions(t *testing.T) {
	var posts int32
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/assignees/") {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		atomic.AddInt32(&posts, 1)
		_, _ = io.WriteString(w, `{"id":1}`)
	})
	c := testClient(t, testConfig(), h)

	const n = 20
	for i := 1; i <= n; i++ {
		c.markEligible(i)
	}
	var wg sync.WaitGroup
	for i := 1; i <= n; i++ {
		wg.Add(1)
		go func(pr int) {
			defer wg.Done()
			ctx := withAuditedPR(context.Background(), pr)
			// Acting on a DIFFERENT (and equally eligible) pull request must be
			// refused without any HTTP call.
			if res, _ := c.assignOwner(ctx, 1+pr%n, "core"); pr != 1+pr%n && res.Status != "error" {
				t.Errorf("pr %d: cross-pull-request assign status = %q, want error", pr, res.Status)
			}
			if res, _ := c.requestMoreContext(ctx, 1+pr%n, []string{"problem"}); pr != 1+pr%n && res.Status != "error" {
				t.Errorf("pr %d: cross-pull-request comment status = %q, want error", pr, res.Status)
			}
			// Its own pull request must work, exactly once each.
			if res, _ := c.assignOwner(ctx, pr, "core"); res.Status != "success" {
				t.Errorf("pr %d: self assign status = %q, want success", pr, res.Status)
			}
			if res, _ := c.requestMoreContext(ctx, pr, []string{"problem"}); res.Status != "success" {
				t.Errorf("pr %d: self comment status = %q, want success", pr, res.Status)
			}
		}(i)
	}
	wg.Wait()

	// One assign and one comment per pull request, and nothing else.
	if got := atomic.LoadInt32(&posts); got != 2*n {
		t.Errorf("made %d writes, want %d (one assign + one comment per pull request)", got, 2*n)
	}
}

// --- tool inventory ---------------------------------------------------------

// The inventory is pinned so a new tool cannot be added without this failing.
// Every tool here is gated by authorizePR, the claim, and shouldSkip; an
// ungated one reaching the same state is the failure shape this catches.
func TestToolInventoryIsPinned(t *testing.T) {
	for _, tc := range []struct {
		name           string
		requestContext bool
		want           []string
	}{
		{name: "default", requestContext: true, want: []string{"assign_owner_to_pull_request", "request_more_context"}},
		{name: "context requests disabled", requestContext: false, want: []string{"assign_owner_to_pull_request"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			cfg.RequestContext = tc.requestContext
			c := &GitHubClient{cfg: cfg, log: discardLogger()}
			tools, err := c.tools()
			if err != nil {
				t.Fatalf("tools() error = %v", err)
			}
			var got []string
			for _, tool := range tools {
				got = append(got, tool.Name())
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Errorf("tool inventory = %v, want exactly %v", got, tc.want)
			}
		})
	}
}
