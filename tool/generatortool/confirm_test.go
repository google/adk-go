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

package generatortool_test

import (
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// confirmedTool is a tool named analyze that requires confirmation, counts
// its runs, emits the message "m" and returns "ok".
func confirmedTool(t *testing.T, runs *atomic.Int32) tool.Tool {
	return mustNew(t, functiontool.Config{Name: "analyze", RequireConfirmation: true}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (string, error) {
		runs.Add(1)
		if err := emit(message("m")); err != nil {
			return "", err
		}
		return "ok", nil
	})
}

// confirmation is the user's answer to the one confirmation request in events.
func confirmation(t *testing.T, events []*session.Event, confirmed bool) *genai.Content {
	t.Helper()
	requests := functionCalls(events, toolconfirmation.FunctionCallName)
	if len(requests) != 1 {
		t.Fatalf("confirmation requests = %d, want 1", len(requests))
	}
	return userResponse(requests[0].ID, toolconfirmation.FunctionCallName, map[string]any{"confirmed": confirmed})
}

var confirmedToolEvents = []evSummary{
	{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Text: "m"},
	{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Output: "ok", NoContent: true},
}

// A17. Fails if the confirmed call's re-dispatch has no workflow node context
// (it would answer ErrInvalidRunNodeContext), or if the body starts before the
// confirmation.
func TestRun_ConfirmationApprovedRunsBody(t *testing.T) {
	var runs atomic.Int32
	h := newHarness(t, harnessConfig{tools: []tool.Tool{confirmedTool(t, &runs)}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))

	if got := runs.Load(); got != 0 {
		t.Errorf("handler runs after the first turn = %d, want 0", got)
	}
	turn1 := h.events()
	want1 := []*genai.FunctionResponse{{ID: "call_1", Name: "analyze", Response: map[string]any{"error": `error tool "analyze" requires confirmation, please approve or reject`}}}
	if diff := cmp.Diff(want1, responses(turn1, "analyze")); diff != "" {
		t.Fatalf("first turn function responses mismatch (-want +got):\n%s", diff)
	}

	h.mustRun(confirmation(t, turn1, true))

	if got := runs.Load(); got != 1 {
		t.Errorf("handler runs after the confirmation = %d, want 1", got)
	}
	events := h.events()
	if diff := cmp.Diff(confirmedToolEvents, summarize(authoredBy(events, "analyze"))); diff != "" {
		t.Errorf("tool events mismatch (-want +got):\n%s", diff)
	}
	got := responses(events, "analyze")
	if diff := cmp.Diff(map[string]any{"result": "ok"}, got[len(got)-1].Response); diff != "" {
		t.Errorf("last function response mismatch (-want +got):\n%s", diff)
	}
}

// A18. Fails if the body starts despite a rejection.
func TestRun_ConfirmationRejectedSkipsBody(t *testing.T) {
	var runs atomic.Int32
	h := newHarness(t, harnessConfig{tools: []tool.Tool{confirmedTool(t, &runs)}, responses: []*genai.Content{analyzeCall(), modelText("done")}})

	h.mustRun(userText("go"))
	h.mustRun(confirmation(t, h.events(), false))

	if got := runs.Load(); got != 0 {
		t.Errorf("handler runs = %d, want 0", got)
	}
	got := responses(h.events(), "analyze")
	last, _ := got[len(got)-1].Response["error"].(string)
	if !strings.Contains(last, "rejected") {
		t.Errorf("last function response = %v, want an error that contains %q", got[len(got)-1].Response, "rejected")
	}
}

// A33. Fails if the confirmation placeholder of a resumed invocation counts as
// an answered call, which moves the confirmed call to the path
// root_agent/analyze@~call_1~1.
func TestRun_ConfirmationApprovedOnResumedInvocationKeepsPath(t *testing.T) {
	var runs atomic.Int32
	h := newHarness(t, harnessConfig{
		tools:     []tool.Tool{confirmedTool(t, &runs)},
		responses: []*genai.Content{analyzeCall(), modelText("done")},
		resumable: true,
	})

	h.mustRun(userText("go"))
	turn1 := h.events()
	h.mustRun(confirmation(t, turn1, true), runner.WithInvocationID(turn1[0].InvocationID))

	if got := runs.Load(); got != 1 {
		t.Errorf("handler runs = %d, want 1", got)
	}
	events := h.events()
	if diff := cmp.Diff(confirmedToolEvents, summarize(authoredBy(events, "analyze"))); diff != "" {
		t.Errorf("tool events mismatch (-want +got):\n%s", diff)
	}
	got := responses(events, "analyze")
	if diff := cmp.Diff(map[string]any{"result": "ok"}, got[len(got)-1].Response); diff != "" {
		t.Errorf("last function response mismatch (-want +got):\n%s", diff)
	}
	if got := h.modelRequests(); got != 2 {
		t.Errorf("model requests = %d, want 2", got)
	}
}
