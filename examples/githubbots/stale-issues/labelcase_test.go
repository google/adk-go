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
	"io"
	"net/http"
	"strings"
	"testing"
)

// GitHub label names are case-insensitively unique but keep the casing they were
// created with, so a repository whose label reads "Stale" against the default
// STALE_LABEL_NAME of "stale" must still be recognized as stale.
//
// Every assertion here failed before the fix. The read side compared byte-exactly
// while the write side used EqualFold, so is_stale was false forever: the bot
// re-marked the issue and re-posted its warning comment on every run, and the
// close and un-stale branches — both of which require is_stale — were dead.
func TestStaleLabelIsRecognizedRegardlessOfCasing(t *testing.T) {
	for _, actual := range []string{"Stale", "STALE", "stale"} {
		t.Run(actual, func(t *testing.T) {
			raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(30)}
			raw.Labels = labelNodes(actual)
			raw.TimelineItems = timelineNodes(rawTimelineItem{
				Typename: "LabeledEvent", CreatedAt: daysAgo(10),
				Label: &struct {
					Name string `json:"name"`
				}{Name: actual},
			})
			got := computeIssueState(raw, testSelf, testMaint, "stale", testStaleAfter, testCloseAfter, testNow)
			if !got.IsStale {
				t.Errorf("IsStale = false with the label present as %q and STALE_LABEL_NAME %q", actual, "stale")
			}
			if got.DaysSinceStaleLabel < 9 || got.DaysSinceStaleLabel > 11 {
				t.Errorf("DaysSinceStaleLabel = %.2f, want ~10; a label whose age is unknown can never be closed on", got.DaysSinceStaleLabel)
			}
		})
	}
}

// A human clearing a differently-cased stale label must still reset the clock,
// or the bot re-marks on the next sweep what the human just cleared.
func TestUnlabelingIsRecognizedRegardlessOfCasing(t *testing.T) {
	raw := &rawIssue{Author: actor(testAuthor), CreatedAt: daysAgo(40)}
	raw.TimelineItems = timelineNodes(rawTimelineItem{
		Typename: "UnlabeledEvent", Actor: actor("maintainerB"), CreatedAt: daysAgo(1),
		Label: &struct {
			Name string `json:"name"`
		}{Name: "Stale"},
	})
	got := computeIssueState(raw, testSelf, testMaint, "stale", testStaleAfter, testCloseAfter, testNow)
	if got.DaysSinceActivity > 2 {
		t.Errorf("DaysSinceActivity = %.2f, want ~1", got.DaysSinceActivity)
	}
	if got.LastActionType != string(eventUnlabeled) {
		t.Errorf("LastActionType = %q, want %q", got.LastActionType, eventUnlabeled)
	}
}

// The same defect, driven through the production entry point rather than through
// computeIssueState: doGetIssueState over a real transport, then the destructive
// tool whose gate must refuse. Before the fix doMarkStale SUCCEEDED here, which
// is the daily duplicate warning comment.
func TestProductionPathRefusesToReMarkAnAlreadyStaleIssue(t *testing.T) {
	const body = `{"data":{"repository":{"issue":{
		"author":{"login":"reporter"},"createdAt":"2020-01-01T00:00:00Z",
		"labels":{"nodes":[{"name":"Stale"}]},
		"comments":{"nodes":[{"author":{"login":"maintainerA"},"body":"Can you share a repro?","createdAt":"2020-02-01T00:00:00Z","lastEditedAt":null}]},
		"userContentEdits":{"nodes":[]},
		"timelineItems":{"nodes":[{"__typename":"LabeledEvent","createdAt":"2020-02-02T00:00:00Z","label":{"name":"Stale"}}]}}}}}`

	cfg := baseCfg()
	cfg.Maintainers = []string{"maintainerA"}
	cfg.DryRun = true // the write itself is not the point; the gate is
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	ctx := withAuditedIssue(context.Background(), 7)

	st, err := c.doGetIssueState(ctx, 7)
	if err != nil {
		t.Fatalf("doGetIssueState: %v", err)
	}
	if !st.IsStale {
		t.Errorf("is_stale = false though the issue carries the stale label as %q", "Stale")
	}

	res, err := c.doMarkStale(ctx, 7)
	if err != nil {
		t.Fatalf("doMarkStale: %v", err)
	}
	if res.Status != "error" || !strings.Contains(res.Message, "already stale") {
		t.Errorf("doMarkStale = %+v on an already-stale issue, want a refusal; otherwise the warning comment is posted again on every run", res)
	}
}

// The allow-list has to match the same way, or the bot can mark an issue stale
// and then be refused when it tries to clear the label the author's return
// earned them.
func TestManagedLabelMatchingIsCaseInsensitive(t *testing.T) {
	c := &GitHubClient{cfg: &Config{
		StaleLabel:                "stale",
		RequestClarificationLabel: "request clarification",
	}}
	for _, tc := range []struct {
		label string
		want  bool
	}{
		{"stale", true},
		{"Stale", true},
		{"STALE", true},
		{"request clarification", true},
		{"Request Clarification", true},
		{"security", false},
		{"staleness", false},
		{"", false},
	} {
		if got := c.isManagedLabel(tc.label); got != tc.want {
			t.Errorf("isManagedLabel(%q) = %t, want %t", tc.label, got, tc.want)
		}
	}

	// An empty label is never managed, even against an empty configured name.
	// sameLabel("", "") is true, so without the explicit guard a Config that left
	// one of the two names unset would admit the empty label into the tool body.
	unset := &GitHubClient{cfg: &Config{StaleLabel: "stale"}}
	if unset.isManagedLabel("") {
		t.Error(`isManagedLabel("") = true with RequestClarificationLabel unset, want false`)
	}
}

// Widening the allow-list must not widen authority: a differently-cased stale
// label now reaches the refusal instead of being turned away as unmanaged, and
// it must still be refused, naming the gated tool.
func TestAddLabelRefusesTheStaleLabelInAnyCasing(t *testing.T) {
	c := newTestClient(t)
	c.cfg.RequestClarificationLabel = "request clarification"
	ctx := withAuditedIssue(context.Background(), 7)

	for _, label := range []string{"stale", "Stale", "STALE"} {
		res, err := c.doAddLabel(ctx, 7, label)
		if err != nil {
			t.Fatalf("doAddLabel(%q) returned a Go error: %v", label, err)
		}
		if res.Status != "error" || !strings.Contains(res.Message, "add_stale_label_and_comment") {
			t.Errorf("doAddLabel(%q) = %+v, want a refusal pointing at the gated tool", label, res)
		}
	}
}
