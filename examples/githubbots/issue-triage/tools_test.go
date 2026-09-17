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
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-github/v66/github"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// countingClient returns a client whose handler counts every HTTP call, so
// tests can assert that rejected actions make no network calls.
func countingClient(t *testing.T, cfg *Config, status int, body string) (*Client, *atomic.Int64) {
	t.Helper()
	// Atomic because the count is written on the server goroutine and read on
	// the test goroutine, whatever the concurrency of the test using it.
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	base, _ := url.Parse(srv.URL + "/")
	rest := github.NewClient(nil)
	rest.BaseURL = base
	return &Client{
		rest:       rest,
		cfg:        cfg,
		log:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		authorized: make(map[int]need),
	}, &calls
}

// scoped returns a context authorizing a session for exactly one issue, which
// the mutating tools now require.
func scoped(number int) context.Context { return withAuditedIssue(context.Background(), number) }

func TestDoChangeTypeRejectsWithoutHTTP(t *testing.T) {
	tests := []struct {
		name       string
		number     int
		issueType  string
		authorize  bool
		wantStatus string
	}{
		{"disallowed type", 7, "Epic", true, "error"},
		{"unauthorized issue", 7, "Bug", false, "error"},
		{"invalid number", 0, "Bug", true, "error"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
			if tc.authorize {
				c.authorize(tc.number, need{typ: true})
			}
			res, err := c.doChangeType(withAuditedIssue(context.Background(), tc.number), tc.number, tc.issueType)
			if err != nil {
				t.Fatalf("doChangeType() unexpected Go error = %v", err)
			}
			if res.Status != tc.wantStatus {
				t.Errorf("status = %q, want %q", res.Status, tc.wantStatus)
			}
			if got := calls.Load(); got != 0 {
				t.Errorf("made %d HTTP calls, want 0", got)
			}
		})
	}
}

func TestDoChangeTypeAuthorizedSucceeds(t *testing.T) {
	var writes int
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		writes++
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	c.authorize(7, need{typ: true})
	res, err := c.doChangeType(scoped(7), 7, "Bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "success" {
		t.Errorf("status = %q, want success", res.Status)
	}
	if writes != 1 {
		t.Errorf("made %d mutating calls, want 1", writes)
	}
}

// The need is computed when the work set is fetched, and a sweep can reach the
// last issue up to SweepTimeout later. A maintainer who sets the type inside
// that window must not be clobbered, so the write re-reads first.
//
// Killing mutation: delete the confirmStillNeeded block from doChangeType.
func TestDoChangeTypeRefusesAFieldFilledSinceTheFetch(t *testing.T) {
	const nowTyped = `{"data":{"repository":{"issue":` +
		`{"number":7,"title":"t","body":"b","issueType":{"name":"Feature"},"labels":{"nodes":[]}}}}}`
	var writes int
	c := writeClient(t, testConfig(), nowTyped, func(w http.ResponseWriter, _ *http.Request) {
		writes++
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	// Authorized from the stale snapshot: at fetch time the type was empty.
	c.authorize(7, need{typ: true})

	res, err := c.doChangeType(scoped(7), 7, "Bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "now has a type") {
		t.Errorf("doChangeType = %+v, want a refusal naming the freshly-set type", res)
	}
	if writes != 0 {
		t.Errorf("made %d mutating calls, want 0: the human's type must not be overwritten", writes)
	}
}

// Same window, the additive half: a label added since the fetch must not be
// joined by a second one.
//
// Killing mutation: delete the confirmStillNeeded block from doAddLabel.
func TestDoAddLabelRefusesALabelAddedSinceTheFetch(t *testing.T) {
	const nowLabelled = `{"data":{"repository":{"issue":` +
		`{"number":7,"title":"t","body":"b","issueType":null,"labels":{"nodes":[{"name":"bug"}]}}}}}`
	var writes int
	c := writeClient(t, testConfig(), nowLabelled, func(w http.ResponseWriter, _ *http.Request) {
		writes++
		_, _ = io.WriteString(w, `[{"name":"enhancement"}]`)
	})
	c.authorize(7, need{label: true})

	res, err := c.doAddLabel(scoped(7), 7, "enhancement")
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "now has a categorization label") {
		t.Errorf("doAddLabel = %+v, want a refusal naming the freshly-added label", res)
	}
	if writes != 0 {
		t.Errorf("made %d mutating calls, want 0: a second label must not be added", writes)
	}
}

func TestDoAddLabelRejectsWithoutHTTP(t *testing.T) {
	tests := []struct {
		name      string
		number    int
		label     string
		authorize bool
	}{
		{"disallowed label", 7, "good first issue", true},
		{"unauthorized issue", 7, "bug", false},
		{"invalid number", 0, "bug", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c, calls := countingClient(t, testConfig(), http.StatusOK, `[]`)
			if tc.authorize {
				c.authorize(tc.number, need{label: true})
			}
			res, err := c.doAddLabel(withAuditedIssue(context.Background(), tc.number), tc.number, tc.label)
			if err != nil {
				t.Fatalf("doAddLabel() unexpected Go error = %v", err)
			}
			if res.Status != "error" {
				t.Errorf("status = %q, want error", res.Status)
			}
			if got := calls.Load(); got != 0 {
				t.Errorf("made %d HTTP calls, want 0", got)
			}
		})
	}
}

func TestDoChangeTypeRESTErrorIsGoError(t *testing.T) {
	// Infrastructure failures (non-2xx) must surface as a Go error (not an
	// errResult) AND be recorded so the run fails loudly.
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	})
	c.authorize(7, need{typ: true})
	if c.hadToolError() {
		t.Fatal("hadToolError() should start false")
	}
	if _, err := c.doChangeType(scoped(7), 7, "Bug"); err == nil {
		t.Fatal("doChangeType() expected Go error on HTTP 500, got nil")
	}
	if !c.hadToolError() {
		t.Error("doChangeType() did not record the infrastructure error")
	}
}

func TestAuthorizeDoesNotResurrectConsumedNeed(t *testing.T) {
	// triageOne is today the only caller and runs once per issue, so the merge is
	// defense in depth: a future second call site for an issue whose need was
	// already satisfied must not resurrect the consumed field, or a second
	// overwrite becomes possible.
	c := &Client{cfg: testConfig(), log: discardLogger(), authorized: make(map[int]need)}
	c.authorize(7, need{typ: true, label: true})
	if claimed, _ := c.claimType(7); !claimed {
		t.Fatal("claimType(7) = false on first claim, want true")
	}
	// Re-authorize as if a later list returned the (stale) issue again.
	c.authorize(7, need{typ: true, label: true})
	if claimed, _ := c.claimType(7); claimed {
		t.Error("claimType(7) = true after re-authorize, want false (consumed need must not resurrect)")
	}
}

func TestDoChangeTypeConcurrentSingleWrite(t *testing.T) {
	// ADK executes a turn's tool calls concurrently: handleFunctionCalls builds
	// one task per call and hands them to platform.RunTasks, whose default
	// runner is a goroutine per task (adk/v2 internal/llminternal/base_flow.go
	// and platform/exec.go). So if the model emits change_issue_type twice for
	// one issue, exactly one call must reach the API.
	//
	// What this test does and does not establish. It establishes the OUTCOME
	// under real contention: 64 goroutines released together produce one write
	// and one success, through the real tool function and a live server. It does
	// NOT kill a check-then-act split of claimType -- measured 0 kills in 10
	// fresh processes against exactly that mutation, because the window between
	// an unlocked read and an unlocked write is two instructions wide and racing
	// for it is a coin flip. The mechanism is pinned separately, by forcing that
	// interleaving rather than racing it: see
	// TestAClaimsCriticalSectionAdmitsOneCallerAtATime.
	var calls atomic.Int64
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	c.authorize(7, need{typ: true})
	// The attempt cap would otherwise refuse all but the first
	// maxAttemptsPerIssue goroutines before they reach the claim, quietly
	// reducing the contention this test exists to create.
	c.liftAttemptCap(7, "type")

	// Released together from a barrier, and enough of them to make the
	// interleaving likely: without the barrier the goroutines are staggered by
	// their own startup and a check-then-act split can go unobserved.
	const goroutines = 64
	var (
		wg        sync.WaitGroup
		successes atomic.Int64
		start     = make(chan struct{})
	)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			res, err := c.doChangeType(scoped(7), 7, "Bug")
			if err != nil {
				return
			}
			if res.Status == "success" {
				successes.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if got := calls.Load(); got != 1 {
		t.Errorf("doChangeType(7, \"Bug\") x%d concurrent = %d API writes, want 1", goroutines, got)
	}
	if got := successes.Load(); got != 1 {
		t.Errorf("doChangeType(7, \"Bug\") x%d concurrent = %d successes, want 1", goroutines, got)
	}
}

// The same for the label field. See TestDoChangeTypeConcurrentSingleWrite for
// what this establishes and what it does not.
func TestDoAddLabelConcurrentSingleWrite(t *testing.T) {
	var calls atomic.Int64
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	})
	c.authorize(7, need{label: true})
	c.liftAttemptCap(7, "label")

	const goroutines = 64
	var (
		wg        sync.WaitGroup
		successes atomic.Int64
		start     = make(chan struct{})
	)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			res, err := c.doAddLabel(scoped(7), 7, "bug")
			if err != nil {
				return
			}
			if res.Status == "success" {
				successes.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if got := calls.Load(); got != 1 {
		t.Errorf("doAddLabel(7, \"bug\") x%d concurrent = %d API writes, want 1", goroutines, got)
	}
	if got := successes.Load(); got != 1 {
		t.Errorf("doAddLabel(7, \"bug\") x%d concurrent = %d successes, want 1", goroutines, got)
	}
}

func TestDoChangeTypeConsumesNeed(t *testing.T) {
	// After a successful type set, a second set on the same issue this run must
	// be refused (don't overwrite what we just set) without an HTTP call.
	var calls int
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	c.authorize(7, need{typ: true})
	if res, err := c.doChangeType(scoped(7), 7, "Bug"); err != nil || res.Status != "success" {
		t.Fatalf("first doChangeType() = (%+v, %v), want success", res, err)
	}
	res, err := c.doChangeType(scoped(7), 7, "Feature")
	if err != nil {
		t.Fatalf("second doChangeType() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("second set status = %q, want error (need already consumed)", res.Status)
	}
	if calls != 1 {
		t.Errorf("made %d HTTP calls, want 1 (second set must not call the API)", calls)
	}
}

func TestDoAddLabelConsumesNeed(t *testing.T) {
	var calls int
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	})
	c.authorize(7, need{label: true})
	if res, err := c.doAddLabel(scoped(7), 7, "bug"); err != nil || res.Status != "success" {
		t.Fatalf("first doAddLabel() = (%+v, %v), want success", res, err)
	}
	res, err := c.doAddLabel(scoped(7), 7, "enhancement")
	if err != nil {
		t.Fatalf("second doAddLabel() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("second add status = %q, want error (need already consumed)", res.Status)
	}
	if calls != 1 {
		t.Errorf("made %d HTTP calls, want 1 (second add must not call the API)", calls)
	}
}

func TestDoChangeTypeCanonicalizesCasing(t *testing.T) {
	// The model may emit a lowercase type; GitHub must still receive the
	// canonical "Bug".
	var gotType any
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotType = body["type"]
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	c.authorize(7, need{typ: true})
	res, err := c.doChangeType(scoped(7), 7, "bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q, want success", res.Status)
	}
	if gotType != "Bug" {
		t.Errorf("GitHub received type %v, want canonical \"Bug\"", gotType)
	}
}

func TestDoAddLabelCanonicalizesCasing(t *testing.T) {
	// The model may emit a differently-cased label; GitHub must receive the
	// allowlist's exact spelling so it matches an existing label.
	var gotBody string
	c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = io.WriteString(w, `[{"name":"bug"}]`)
	})
	c.authorize(7, need{label: true})
	res, err := c.doAddLabel(scoped(7), 7, "BUG")
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "success" {
		t.Fatalf("status = %q, want success", res.Status)
	}
	if !strings.Contains(gotBody, `"bug"`) || strings.Contains(gotBody, `"BUG"`) {
		t.Errorf("GitHub received label body %q, want canonical \"bug\"", gotBody)
	}
}

func TestDoChangeTypeDoesNotOverwriteExistingField(t *testing.T) {
	// An issue can appear in the triage set because it needs a *label* while
	// already having a type. The type tool must refuse to overwrite it, in code.
	c, calls := countingClient(t, testConfig(), http.StatusOK, `{}`)
	c.authorize(7, need{label: true}) // type already set
	res, err := c.doChangeType(scoped(7), 7, "Bug")
	if err != nil {
		t.Fatalf("doChangeType() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (must not overwrite existing type)", res.Status)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("made %d HTTP calls, want 0", got)
	}
}

func TestDoAddLabelDoesNotOverwriteExistingField(t *testing.T) {
	c, calls := countingClient(t, testConfig(), http.StatusOK, `[]`)
	c.authorize(7, need{typ: true}) // label already set
	res, err := c.doAddLabel(scoped(7), 7, "bug")
	if err != nil {
		t.Fatalf("doAddLabel() error = %v", err)
	}
	if res.Status != "error" {
		t.Errorf("status = %q, want error (must not add a second label)", res.Status)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("made %d HTTP calls, want 0", got)
	}
}

// A claim is one-shot per (issue, field) per run: no failure re-opens it.
//
// Deciding that a write did not land is a judgement about a response, and every
// version of that judgement has a false positive -- the label read-back reports
// one page of an issue's labels, and an empty response body decodes to the same
// zero value as a dropped type. Re-opening on a write that DID land lets the
// model claim the need again with a different value, which is two labels on one
// issue. Both failure shapes are pinned here.
//
// Killing mutation: re-add a release call inside writeFailed.
func TestAFailedWriteDoesNotReopenTheClaim(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mutate  http.HandlerFunc
		act     func(*Client) (actionResult, error)
		claimed func(*Client) bool
	}{
		{
			// 200 carrying an issue with no type: GitHub accepted the mutation
			// and dropped it, which is what happens without push access.
			name: "type, GitHub dropped the write",
			mutate: func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, `{"number":7}`)
			},
			act:     func(c *Client) (actionResult, error) { return c.doChangeType(scoped(7), 7, "Bug") },
			claimed: func(c *Client) bool { claimed, _ := c.claimType(7); return claimed },
		},
		{
			name: "type, transport failure",
			mutate: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, `{"message":"boom"}`)
			},
			act:     func(c *Client) (actionResult, error) { return c.doChangeType(scoped(7), 7, "Bug") },
			claimed: func(c *Client) bool { claimed, _ := c.claimType(7); return claimed },
		},
		{
			name: "label, GitHub dropped the write",
			mutate: func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, `[]`)
			},
			act:     func(c *Client) (actionResult, error) { return c.doAddLabel(scoped(7), 7, "bug") },
			claimed: func(c *Client) bool { claimed, _ := c.claimLabel(7); return claimed },
		},
		{
			name: "label, transport failure",
			mutate: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, `{"message":"boom"}`)
			},
			act:     func(c *Client) (actionResult, error) { return c.doAddLabel(scoped(7), 7, "bug") },
			claimed: func(c *Client) bool { claimed, _ := c.claimLabel(7); return claimed },
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := writeClient(t, testConfig(), untriagedIssueJSON, tc.mutate)
			c.authorize(7, need{typ: true, label: true})
			if _, err := tc.act(c); err == nil {
				t.Fatal("the failed write returned a nil error, want it surfaced")
			}
			if !c.hadToolError() {
				t.Error("the failed write was not recorded, so the run would exit 0")
			}
			if tc.claimed(c) {
				t.Error("the claim was re-opened; a retry could write a different value")
			}
		})
	}
}

// The pre-write revalidation read must be reported when it fails, and must cost
// nothing. A read is unambiguous -- if it failed, no mutation was attempted --
// so unlike a failed write it must not burn the issue's one write for the run.
// That is why the read happens before the claim, in BOTH tools: pinning only
// the type path left "move claimLabel above confirmStillNeeded" surviving the
// whole suite.
//
// Killing mutation: move the claim above the confirmStillNeeded call, in either
// doChangeType or doAddLabel.
func TestARevalidationFailureIsReportedAndCostsNoClaim(t *testing.T) {
	for _, tc := range []struct {
		name    string
		act     func(*Client) (actionResult, error)
		claimed func(*Client) bool
	}{
		{
			"doChangeType",
			func(c *Client) (actionResult, error) { return c.doChangeType(scoped(7), 7, "Bug") },
			func(c *Client) bool { claimed, _ := c.claimType(7); return claimed },
		},
		{
			"doAddLabel",
			func(c *Client) (actionResult, error) { return c.doAddLabel(scoped(7), 7, "bug") },
			func(c *Client) bool { claimed, _ := c.claimLabel(7); return claimed },
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var writes int
			c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/graphql" {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = io.WriteString(w, `{"message":"boom"}`)
					return
				}
				writes++
			}))
			c.authorized = make(map[int]need)
			c.authorize(7, need{typ: true, label: true})

			if _, err := tc.act(c); err == nil {
				t.Fatalf("%s = nil error, want the revalidation failure surfaced", tc.name)
			}
			if !c.hadToolError() {
				t.Error("the failed read was not recorded, so the run would exit 0")
			}
			if writes != 0 {
				t.Errorf("made %d mutating calls, want 0: the write must not proceed on an unverified read", writes)
			}
			if !tc.claimed(c) {
				t.Error("a failed READ consumed the claim; nothing was written, so the field must stay open")
			}
		})
	}
}

// The label add-response is one page, so a label missing from a FULL page
// proves nothing about whether the add landed. The code asks GraphQL, which
// fetches labelPageSize labels, rather than guessing in either direction.
//
// Killing mutation: delete the full-page branch from AddLabel, or make it
// `return nil` without confirming.
func TestAFullPageLabelResponseIsConfirmedNotGuessed(t *testing.T) {
	// Thirty labels, none of them ours: exactly the ambiguous case.
	var page []string
	for i := range labelResponsePage {
		page = append(page, fmt.Sprintf(`{"name":"area/%d"}`, i))
	}
	fullPage := "[" + strings.Join(page, ",") + "]"

	for _, tc := range []struct {
		name      string
		confirmed string
		wantErr   bool
	}{
		{
			"the add did land, GraphQL sees it",
			`{"data":{"repository":{"issue":{"number":7,"issueType":null,"labels":{"nodes":[{"name":"bug"}]}}}}}`,
			false,
		},
		{
			"the add did not land, GraphQL agrees",
			`{"data":{"repository":{"issue":{"number":7,"issueType":null,"labels":{"nodes":[{"name":"area/1"}]}}}}}`,
			true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/graphql" {
					_, _ = io.WriteString(w, tc.confirmed)
					return
				}
				_, _ = io.WriteString(w, fullPage)
			}))
			err := c.AddLabel(context.Background(), 7, "bug")
			if tc.wantErr {
				if err == nil {
					t.Fatal("AddLabel = nil, want the unconfirmed add reported as not applied")
				}
				if !errors.Is(err, errNotApplied) {
					t.Errorf("AddLabel = %v, want errNotApplied", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("AddLabel = %v, want nil: GraphQL confirmed the label is present", err)
			}
		})
	}
}

// The confirming read is a network call and can fail. That must surface, not be
// read as either outcome.
//
// Killing mutation: drop the error return from the confirmation branch.
func TestAFailedLabelConfirmationIsReported(t *testing.T) {
	var page []string
	for i := range labelResponsePage {
		page = append(page, fmt.Sprintf(`{"name":"area/%d"}`, i))
	}
	fullPage := "[" + strings.Join(page, ",") + "]"

	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/graphql" {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"message":"boom"}`)
			return
		}
		_, _ = io.WriteString(w, fullPage)
	}))
	err := c.AddLabel(context.Background(), 7, "bug")
	if err == nil {
		t.Fatal("AddLabel = nil, want the failed confirmation surfaced")
	}
	if errors.Is(err, errNotApplied) {
		t.Error("a failed confirmation was reported as a demonstrable non-application")
	}
	if !strings.Contains(err.Error(), "confirm label") {
		t.Errorf("AddLabel = %v, want it to name the failed confirmation", err)
	}
}

// A short response is decisive on its own; the confirming read must not happen.
func TestAShortLabelResponseNeedsNoConfirmation(t *testing.T) {
	var reads int
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/graphql" {
			reads++
			_, _ = io.WriteString(w, `{"data":{"repository":{"issue":null}}}`)
			return
		}
		_, _ = io.WriteString(w, `[{"name":"go"}]`)
	}))
	if err := c.AddLabel(context.Background(), 7, "bug"); !errors.Is(err, errNotApplied) {
		t.Fatalf("AddLabel = %v, want errNotApplied", err)
	}
	if reads != 0 {
		t.Errorf("made %d confirming reads for a response that was already decisive", reads)
	}
}

// toolCtx adapts a plain context.Context to the agent.Context a tool is called
// with, so a test can drive the REAL functiontool wrapper. It delegates exactly
// the context.Context half of the interface; agent.ContextMock supplies the
// rest as zero values, which is what an unused invocation context looks like.
type toolCtx struct {
	*agent.ContextMock
	ctx context.Context
}

func (c toolCtx) Deadline() (time.Time, bool) { return c.ctx.Deadline() }
func (c toolCtx) Done() <-chan struct{}       { return c.ctx.Done() }
func (c toolCtx) Err() error                  { return c.ctx.Err() }
func (c toolCtx) Value(key any) any           { return c.ctx.Value(key) }

func scopedTool(number int) agent.Context {
	return toolCtx{ContextMock: &agent.ContextMock{}, ctx: scoped(number)}
}

// runnableTool is how the framework actually calls a tool. tool.Tool itself
// exposes only Name/Description/IsLongRunning, so this layer is reachable from a
// test only through the assertion below -- which is exactly why nothing was
// covering it.
type runnableTool interface {
	Run(ctx agent.Context, args any) (map[string]any, error)
	Declaration() *genai.FunctionDeclaration
}

func runnable(t *testing.T, tools []tool.Tool, name string) runnableTool {
	t.Helper()
	for _, tl := range tools {
		if tl.Name() != name {
			continue
		}
		r, ok := tl.(runnableTool)
		if !ok {
			t.Fatalf("tool %q is not runnable (%T)", name, tl)
		}
		return r
	}
	t.Fatalf("tool %q not found", name)
	return nil
}

// Everything else exercises doChangeType/doAddLabel directly, with a context the
// test built. That leaves the whole layer the model actually reaches untested:
// the closure that forwards the agent.Context, and the argument struct whose
// json tags ARE the schema the model fills.
//
// Two one-line mutations survived the entire suite before this test existed:
//   - tools.go: `return c.doChangeType(context.Background(), ...)` — every tool
//     call then hits "no issue is authorized" and the bot silently triages
//     nothing, forever.
//   - tools.go: rename the `issue_type` json tag — the model's argument stops
//     binding, IssueType is "", and every type set is refused.
//
// Both fail here.
func TestToolsEnforceTheSessionScopeThroughTheRealWrapper(t *testing.T) {
	for _, tc := range []struct {
		tool string
		args map[string]any
	}{
		{"change_issue_type", map[string]any{"issue_number": 7, "issue_type": "Bug"}},
		{"add_label_to_issue", map[string]any{"issue_number": 7, "label": "bug"}},
	} {
		t.Run(tc.tool, func(t *testing.T) {
			c := writeClient(t, testConfig(), untriagedIssueJSON, func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPatch {
					_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
					return
				}
				_, _ = io.WriteString(w, `[{"name":"bug"}]`)
			})
			c.authorize(7, need{typ: true, label: true})
			c.authorize(9, need{typ: true, label: true})

			tools, err := c.tools()
			if err != nil {
				t.Fatalf("tools() error = %v", err)
			}
			tl := runnable(t, tools, tc.tool)

			// The schema the model fills is built by reflection over the argument
			// struct, so a renamed json tag silently changes the wire contract.
			schema, err := json.Marshal(tl.Declaration().ParametersJsonSchema)
			if err != nil {
				t.Fatalf("marshal declaration: %v", err)
			}
			for arg := range tc.args {
				if !strings.Contains(string(schema), `"`+arg+`"`) {
					t.Errorf("declared schema omits %q: %s", arg, schema)
				}
			}

			// A session scoped to #7, asked to act on #9, must refuse. This is the
			// control the whole design rests on, driven through the real wrapper.
			elsewhere := map[string]any{"issue_number": 9}
			for k, v := range tc.args {
				if k != "issue_number" {
					elsewhere[k] = v
				}
			}
			res, err := tl.Run(scopedTool(7), elsewhere)
			if err != nil {
				t.Fatalf("Run() out-of-scope returned a Go error: %v", err)
			}
			if got, _ := res["status"].(string); got != "error" {
				t.Errorf("out-of-scope status = %q, want error (result %v)", got, res)
			}
			if msg, _ := res["message"].(string); !strings.Contains(msg, "scoped to issue #7") {
				t.Errorf("out-of-scope message = %q, want it to name the session scope", msg)
			}

			// The same session acting on its own issue must succeed, which is what
			// proves the context and the arguments both arrived.
			res, err = tl.Run(scopedTool(7), tc.args)
			if err != nil {
				t.Fatalf("Run() in-scope returned a Go error: %v", err)
			}
			if got, _ := res["status"].(string); got != "success" {
				t.Errorf("in-scope status = %q, want success (result %v)", got, res)
			}
		})
	}
}

// The full-page confirmation re-reads the issue, so it too can find the issue
// gone between the add and the confirmation. That is the same benign race as
// the pre-write read and must not red the run.
//
// Killing mutation: drop the ErrIssueNotFound branch after c.AddLabel in
// doAddLabel.
func TestAVanishedIssueDuringLabelConfirmationIsNotAFailure(t *testing.T) {
	var page []string
	for i := range labelResponsePage {
		page = append(page, fmt.Sprintf(`{"name":"area/%d"}`, i))
	}
	fullPage := "[" + strings.Join(page, ",") + "]"

	// The pre-write read finds the issue needing a label; the confirming read,
	// after the add, finds it gone.
	var graphQLCalls int
	c := testClient(t, testConfig(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/graphql" {
			graphQLCalls++
			if graphQLCalls == 1 {
				_, _ = io.WriteString(w, untriagedIssueJSON)
				return
			}
			_, _ = io.WriteString(w, `{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND",`+
				`"path":["repository","issue"],"message":"gone"}]}`)
			return
		}
		_, _ = io.WriteString(w, fullPage)
	}))
	c.authorized = make(map[int]need)
	c.attempts = make(map[attemptKey]int)
	c.authorize(7, need{label: true})

	res, err := c.doAddLabel(scoped(7), 7, "bug")
	if err != nil {
		t.Fatalf("doAddLabel returned a Go error for an issue that vanished mid-confirmation: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "no longer exists") {
		t.Errorf("doAddLabel = %+v, want a model-readable refusal", res)
	}
	if c.hadToolError() {
		t.Error("a vanished issue was recorded as an infrastructure failure; the scheduled sweep would go red")
	}
	if graphQLCalls != 2 {
		t.Errorf("made %d GraphQL calls, want 2: the confirming read must have run", graphQLCalls)
	}
}

// A NOT_FOUND about the REPOSITORY is a configuration failure, not a vanished
// issue, and must stay loud. GitHub uses the same error type for both, so only
// the path distinguishes them -- and reading a wrong OWNER/REPO as "the issue
// vanished" would leave the bot exiting 0 forever while triaging nothing.
//
// Killing mutation: drop the path check from graphQLError.isIssueNotFound.
func TestAMissingRepositoryIsAFailure(t *testing.T) {
	var writes int
	c := writeClient(t, testConfig(),
		`{"data":{"repository":null},"errors":[{"type":"NOT_FOUND",`+
			`"path":["repository"],"message":"Could not resolve to a Repository with the name 'acme/typo'"}]}`,
		func(http.ResponseWriter, *http.Request) { writes++ })
	c.authorize(7, need{typ: true})

	res, err := c.doChangeType(scoped(7), 7, "Bug")
	if err == nil {
		t.Fatalf("doChangeType = (%+v, nil); a missing repository must surface as an error", res)
	}
	if !c.hadToolError() {
		t.Error("a missing repository was not recorded, so the run would exit 0")
	}
	if writes != 0 {
		t.Errorf("made %d mutating calls against a repository that does not resolve", writes)
	}
}

// An issue deleted, transferred or converted to a discussion between the sweep
// selecting it and the tool acting on it is not an infrastructure failure. It
// must be reported to the model as data and leave the run green -- otherwise
// every such event reds a scheduled sweep.
//
// Killing mutation: remove the ErrIssueNotFound branch from doChangeType, so
// the read error routes through toolFailed.
func TestAVanishedIssueIsNotAFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		act  func(*Client) (actionResult, error)
	}{
		{"doChangeType", func(c *Client) (actionResult, error) { return c.doChangeType(scoped(7), 7, "Bug") }},
		{"doAddLabel", func(c *Client) (actionResult, error) { return c.doAddLabel(scoped(7), 7, "bug") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var writes int
			c := writeClient(t, testConfig(),
				`{"data":{"repository":{"issue":null}},"errors":[{"type":"NOT_FOUND",`+
					`"path":["repository","issue"],"message":"Could not resolve to an Issue"}]}`,
				func(http.ResponseWriter, *http.Request) { writes++ })
			c.authorize(7, need{typ: true, label: true})

			res, err := tc.act(c)
			if err != nil {
				t.Fatalf("%s returned a Go error for a vanished issue: %v", tc.name, err)
			}
			if res.Status != "error" || !strings.Contains(res.Message, "no longer exists") {
				t.Errorf("%s = %+v, want a model-readable refusal naming the missing issue", tc.name, res)
			}
			if c.hadToolError() {
				t.Error("a vanished issue was recorded as an infrastructure failure; the run would go red")
			}
			if writes != 0 {
				t.Errorf("made %d mutating calls for an issue that does not exist", writes)
			}
		})
	}
}

// The model is attacker-controlled by assumption and nothing bounds how many
// tool calls it emits in a turn. Each call that clears the allow-list spends a
// GitHub read before the claim refuses it, so an unbounded fan-out burns the
// repository's shared API budget and the resulting rate-limit error reds the
// job. The per-issue cap bounds that, and refusals past it cost no network.
//
// Killing mutation: delete the c.attempt check from checkIssueArg.
func TestAnIssueGetsABoundedNumberOfAttempts(t *testing.T) {
	// The reads the cap exists to bound are only spendable CONCURRENTLY: once
	// one call claims the field, later sequential calls are refused at peek
	// with no network. So release the fan-out together, the way the framework
	// does, and count the reads that actually reach GitHub.
	var reads, writes atomic.Int64
	c := writeClientAtomic(t, testConfig(), &reads, func(w http.ResponseWriter, _ *http.Request) {
		writes.Add(1)
		_, _ = io.WriteString(w, `{"number":7,"type":{"name":"Bug"}}`)
	})
	c.authorize(7, need{typ: true, label: true})

	const callers = 64
	var (
		wg       sync.WaitGroup
		refusals atomic.Int64
		start    = make(chan struct{})
	)
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			res, err := c.doChangeType(scoped(7), 7, "Bug")
			if err != nil {
				return
			}
			if strings.Contains(res.Message, "attempts for this run") {
				refusals.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if refusals.Load() == 0 {
		t.Fatal("the attempt cap never fired; a model can spend unbounded GitHub reads")
	}
	// This is the property the cap is for: reads are bounded by the cap, not by
	// how many calls the model chose to emit.
	if got := reads.Load(); got > maxAttemptsPerIssue {
		t.Errorf("%d callers spent %d GitHub reads, want at most the cap of %d",
			callers, got, maxAttemptsPerIssue)
	}
	if got := writes.Load(); got != 1 {
		t.Errorf("made %d writes, want exactly 1: the claim is one-shot", got)
	}

	// The cap must stay small enough to be a bound. A well-behaved run makes one
	// call per field, so anything past a handful is not bounding anything.
	if maxAttemptsPerIssue > 16 {
		t.Errorf("maxAttemptsPerIssue = %d, which is too high to bound read amplification", maxAttemptsPerIssue)
	}

	// The cap is per field, so a burst on the type must not starve the label.
	// Only the refusal matters here, so it is enough that the label call gets
	// past checkIssueArg -- an out-of-budget call never reaches the network.
	if msg, ok := c.checkIssueArg(scoped(7), 7, "label"); !ok {
		t.Errorf("a burst of type calls exhausted the label's budget too (%s); the cap must be per field", msg)
	}
}

// A refused out-of-scope call must leave a server-side trace. The model's own
// summary is the only other record and it is attacker-influenced, so without
// this an operator reading the Actions log sees nothing.
//
// Killing mutation: delete the c.log.Warn from checkIssueArg's scope branch.
func TestAnOutOfScopeRefusalIsLogged(t *testing.T) {
	var buf bytes.Buffer
	c := &Client{
		cfg:        testConfig(),
		log:        slog.New(slog.NewTextHandler(&buf, nil)),
		authorized: make(map[int]need),
		attempts:   make(map[attemptKey]int),
	}
	c.authorize(99, need{typ: true})

	if _, err := c.doChangeType(scoped(5), 99, "Bug"); err != nil {
		t.Fatalf("doChangeType returned a Go error: %v", err)
	}
	got := buf.String()
	if !strings.Contains(got, "outside the session's issue scope") {
		t.Errorf("the refusal left no trace in the log; got %q", got)
	}
	if !strings.Contains(got, "requested=99") {
		t.Errorf("the log does not name the issue the model asked for; got %q", got)
	}
}
