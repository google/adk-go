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
	"strings"
	"sync"
	"testing"
	"time"
)

const testRelease = "v1.0.0...v1.1.0"

func scoped(release string, index int) context.Context {
	return withAuditedGroup(context.Background(), release, index)
}

// The tool inventory is pinned. A new tool added without a gate would otherwise
// slip in silently, which is exactly how the authority limit gets lost: the
// gated tool is rarely the hole, an ungated sibling reaching the same state is.
func TestToolInventoryIsPinned(t *testing.T) {
	tools, err := newRecorder(testConfig()).tools()
	if err != nil {
		t.Fatalf("tools(): %v", err)
	}
	got := make([]string, 0, len(tools))
	for _, tl := range tools {
		got = append(got, tl.Name())
	}
	// Exactly this set, and nothing else. The model's whole authority is here:
	// this tool writes to an in-memory collector and performs no mutation.
	want := []string{"record_documentation_findings"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("tool inventory = %v, want exactly %v", got, want)
	}
}

func TestAuthorizeGroup(t *testing.T) {
	ctx := scoped(testRelease, 2)
	if _, ok := authorizeGroup(ctx, testRelease, 2); !ok {
		t.Error("the session refused its own release and group")
	}
	if _, ok := authorizeGroup(ctx, testRelease, 3); ok {
		t.Error("the session accepted a DIFFERENT group index")
	}
	if _, ok := authorizeGroup(ctx, "v9...v9", 2); ok {
		t.Error("the session accepted a DIFFERENT release")
	}
	if _, ok := authorizeGroup(context.Background(), testRelease, 2); ok {
		t.Error("an unscoped session was authorized")
	}
}

func TestRecordFindingsRejectsAnUnscopedOrMismatchedSession(t *testing.T) {
	for _, tc := range []struct {
		name    string
		ctx     context.Context
		release string
		index   int
	}{
		{"unscoped", context.Background(), testRelease, 0},
		{"wrong group", scoped(testRelease, 0), testRelease, 1},
		{"wrong release", scoped(testRelease, 0), "v8.0.0...v9.0.0", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newRecorder(testConfig())
			res := r.recordFindings(tc.ctx, recordArgs{
				Release: tc.release, GroupIndex: tc.index,
				Findings: []Finding{{Kind: "new-feature", Summary: "s"}},
			})
			if res.Status != "error" {
				t.Errorf("status = %q, want error", res.Status)
			}
			if got := len(r.findings()); got != 0 {
				t.Errorf("%d findings were recorded by a rejected call, want 0", got)
			}
		})
	}
}

func TestRecordFindingsStoresSanitizedFindings(t *testing.T) {
	r := newRecorder(testConfig())
	res := r.recordFindings(scoped(testRelease, 0), recordArgs{
		Release: testRelease, GroupIndex: 0,
		Findings: []Finding{
			{Kind: "made-up-kind", DocFile: "../etc/passwd", Summary: "```escape```"},
			{Kind: "new-feature"}, // no content: dropped
		},
	})
	if res.Status != "success" {
		t.Fatalf("status = %q, want success: %s", res.Status, res.Message)
	}
	got := r.findings()
	if len(got) != 1 {
		t.Fatalf("recorded %d findings, want 1 (the contentless one is dropped)", len(got))
	}
	if got[0].Kind != unclassifiedKind || got[0].DocFile != "" || strings.Contains(got[0].Summary, "```") {
		t.Errorf("the stored finding was not sanitized: %+v", got[0])
	}
}

// A steered model must not be able to make the issue body unbounded.
//
// Mutation that must fail this test: remove the MaxFindingsPerGroup slice.
func TestRecordFindingsCapsTheGroupsFindings(t *testing.T) {
	cfg := testConfig()
	cfg.MaxFindingsPerGroup = 3
	r := newRecorder(cfg)
	var many []Finding
	for i := range 50 {
		many = append(many, Finding{Kind: "new-feature", Summary: string(rune('a' + i%26))})
	}
	res := r.recordFindings(scoped(testRelease, 0), recordArgs{Release: testRelease, Findings: many})
	if res.Status != "success" {
		t.Fatalf("status = %q, want success", res.Status)
	}
	if got := len(r.findings()); got != 3 {
		t.Errorf("recorded %d findings, want the cap of 3", got)
	}
	if !strings.Contains(res.Message, "dropped") {
		t.Errorf("the result does not tell the model findings were dropped: %q", res.Message)
	}
}

// A model that emits the tool twice for one group must not double the group's
// findings in the issue.
//
// Mutation that must fail this test: make recorder.record overwrite unconditionally.
func TestRecordFindingsRejectsASecondCallForTheSameGroup(t *testing.T) {
	r := newRecorder(testConfig())
	ctx := scoped(testRelease, 0)
	args := recordArgs{Release: testRelease, Findings: []Finding{{Kind: "new-feature", Summary: "first"}}}
	if res := r.recordFindings(ctx, args); res.Status != "success" {
		t.Fatalf("first call status = %q, want success", res.Status)
	}
	args.Findings = []Finding{{Kind: "new-feature", Summary: "second"}}
	res := r.recordFindings(ctx, args)
	if res.Status != "error" {
		t.Errorf("second call status = %q, want error", res.Status)
	}
	got := r.findings()
	if len(got) != 1 || got[0].Summary != "first" {
		t.Errorf("findings = %+v, want only the first call's", got)
	}
}

func TestRecorderOrdersFindingsByGroup(t *testing.T) {
	r := newRecorder(testConfig())
	// Record out of order: the issue body must not depend on completion order.
	for _, i := range []int{2, 0, 1} {
		res := r.recordFindings(scoped(testRelease, i), recordArgs{
			Release: testRelease, GroupIndex: i,
			Findings: []Finding{{Kind: "new-feature", Summary: string(rune('0' + i))}},
		})
		if res.Status != "success" {
			t.Fatalf("group %d: %s", i, res.Message)
		}
	}
	got := r.findings()
	if len(got) != 3 || got[0].Summary != "0" || got[1].Summary != "1" || got[2].Summary != "2" {
		t.Errorf("findings = %+v, want them ordered by group index", got)
	}
}

// Concurrent sessions share one recorder. Each may write only its own group, and
// -race must find no data race on the shared map.
func TestRecordFindingsConcurrentIsolation(t *testing.T) {
	r := newRecorder(testConfig())
	const n = 20
	var wg sync.WaitGroup
	var mu sync.Mutex
	var problems []string
	note := func(s string) { mu.Lock(); problems = append(problems, s); mu.Unlock() }

	for i := range n {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			ctx := scoped(testRelease, index)
			// Recording for a DIFFERENT group must be refused.
			cross := r.recordFindings(ctx, recordArgs{
				Release: testRelease, GroupIndex: index + n,
				Findings: []Finding{{Kind: "new-feature", Summary: "cross"}},
			})
			if cross.Status != "error" {
				note("a session recorded for another group")
			}
			own := r.recordFindings(ctx, recordArgs{
				Release: testRelease, GroupIndex: index,
				Findings: []Finding{{Kind: "new-feature", Summary: "own"}},
			})
			if own.Status != "success" {
				note("a session could not record for its own group")
			}
		}(i)
	}
	wg.Wait()
	for _, p := range problems {
		t.Error(p)
	}
	if got := len(r.findings()); got != n {
		t.Errorf("recorded %d findings, want %d (one per group, no cross-group writes)", got, n)
	}
}

// Drives the real analyzeGroup. Deleting its withAuditedGroup line -- the whole
// cross-group defense -- would leave a suite green if every test built its own
// scoped context instead of observing this one.
//
// Mutation that must fail this test: delete `gctx = withAuditedGroup(gctx, key, index)`.
func TestAnalyzeGroupScopesTheSession(t *testing.T) {
	cfg := testConfig()
	cfg.GroupTimeout = time.Minute
	var problems []string
	ran := false
	gotIndex := -1

	got := analyzeGroup(context.Background(), cfg, testRelease, 3, func(gctx context.Context, index int) bool {
		ran = true
		gotIndex = index
		if msg, ok := authorizeGroup(gctx, testRelease, 3); !ok {
			problems = append(problems, "not scoped to the analyzed group: "+msg)
		}
		if _, ok := authorizeGroup(gctx, testRelease, 4); ok {
			problems = append(problems, "the session accepted a DIFFERENT group")
		}
		if _, ok := authorizeGroup(gctx, "v9...v9", 3); ok {
			problems = append(problems, "the session accepted a DIFFERENT release")
		}
		dl, hasDeadline := gctx.Deadline()
		if !hasDeadline {
			problems = append(problems, "no per-group deadline")
		} else if time.Until(dl) > cfg.GroupTimeout {
			problems = append(problems, "the deadline does not derive from GroupTimeout")
		}
		return true
	})

	if !got {
		t.Error("analyzeGroup did not return runFn's result, so a failed group would count as clean")
	}
	if !ran {
		t.Fatal("analyzeGroup never called runFn: the bot would analyze nothing")
	}
	if gotIndex != 3 {
		t.Errorf("runFn received index %d, want 3", gotIndex)
	}
	for _, p := range problems {
		t.Error(p)
	}
}
