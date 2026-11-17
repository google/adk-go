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
	"sync"
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
	"google.golang.org/adk/v2/workflow"
)

// askRecorder counts the runs of the ask tool and records the answer it saw.
type askRecorder struct {
	runs atomic.Int32
	mu   sync.Mutex
	seen any
}

// askTool is a tool named ask that asks "Which region?" under the interrupt id
// clarify_region and returns the query with the answer.
func askTool(t *testing.T, rec *askRecorder, cfg functiontool.Config) tool.Tool {
	cfg.Name = "ask"
	return mustNew(t, cfg, func(ctx agent.Context, args queryArgs, emit func(*session.Event) error) (map[string]any, error) {
		rec.runs.Add(1)
		answer, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "clarify_region", Message: "Which region?"})
		if err != nil {
			return nil, err
		}
		rec.mu.Lock()
		rec.seen = answer
		rec.mu.Unlock()
		return map[string]any{"q": args.Query, "region": answer}, nil
	})
}

func askCall() *genai.Content {
	return calls(call("call_1", "ask", map[string]any{"query": "sales"}))
}

// regionAnswer is the user's answer to the request for input.
func regionAnswer() *genai.Content {
	return userResponse("clarify_region", workflow.WorkflowInputFunctionCallName, map[string]any{"result": "APAC"})
}

// A7. Fails if the request-input event carries the agent's author or branch
// (the emit wrapper does not stamp it), or if the pause becomes an
// {"error": ...} response and the model is called again.
func TestRun_PauseEventShape(t *testing.T) {
	var rec askRecorder
	h := newHarness(t, harnessConfig{tools: []tool.Tool{askTool(t, &rec, functiontool.Config{})}, responses: []*genai.Content{askCall(), modelText("done")}})

	h.mustRun(userText("go"))

	events := h.events()
	var requests []*session.Event
	for _, ev := range events {
		if len(functionCalls([]*session.Event{ev}, workflow.WorkflowInputFunctionCallName)) > 0 {
			requests = append(requests, ev)
		}
	}
	if len(requests) != 1 {
		t.Fatalf("request-input events = %d, want 1; events: %+v", len(requests), summarize(events))
	}
	ev := requests[0]
	wantCall := &genai.FunctionCall{ID: "clarify_region", Name: workflow.WorkflowInputFunctionCallName, Args: map[string]any{
		"interruptId": "clarify_region", "message": "Which region?", "payload": nil,
	}}
	if diff := cmp.Diff(wantCall, ev.Content.Parts[0].FunctionCall); diff != "" {
		t.Errorf("request-input call mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"clarify_region"}, ev.LongRunningToolIDs); diff != "" {
		t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
	}
	if ev.Author != "ask" || ev.Branch != "ask@call_1" || pathOf(ev) != "root_agent/ask@call_1" {
		t.Errorf("request-input event author, branch, path = %q, %q, %q; want ask, ask@call_1, root_agent/ask@call_1", ev.Author, ev.Branch, pathOf(ev))
	}
	if got := responses(events, "ask"); len(got) != 0 {
		t.Errorf("function responses = %v, want none", got)
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
}

// A8. Fails if the replay of a paused call needs a non-resumable runner, or if
// the body sees the answer still wrapped (map[result:APAC]).
func TestRun_PauseThenResumeResumable(t *testing.T) {
	var rec askRecorder
	h := newHarness(t, harnessConfig{
		tools:     []tool.Tool{askTool(t, &rec, functiontool.Config{})},
		responses: []*genai.Content{askCall(), modelText("done")},
		resumable: true,
	})

	h.mustRun(userText("go"))
	h.mustRun(regionAnswer(), runner.WithInvocationID(h.events()[0].InvocationID))

	if got := rec.runs.Load(); got != 2 {
		t.Errorf("handler runs = %d, want 2", got)
	}
	if rec.seen != "APAC" {
		t.Errorf("answer seen by the handler = %#v, want \"APAC\"", rec.seen)
	}
	events := h.events()
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "ask", Response: map[string]any{"q": "sales", "region": "APAC"}}}
	if diff := cmp.Diff(want, responses(events, "ask")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	var texts []string
	for _, ev := range authoredBy(events, "root_agent") {
		if text := textOf(ev); text != "" {
			texts = append(texts, text)
		}
	}
	if diff := cmp.Diff([]string{"done"}, texts); diff != "" {
		t.Errorf("agent texts mismatch (-want +got):\n%s", diff)
	}
	if got := functionCalls(events, "ask"); len(got) != 1 {
		t.Errorf("ask call events = %d, want 1", len(got))
	}
}

// A19. Fails if the pause of a call that was re-dispatched after its
// confirmation is dropped or shown to the user as an error, or if the replay
// after the user's answer reuses that approval. As in adk-python, the replay
// runs the call without its confirmation
// (flows/llm_flows/core/_function_call_postprocessor.py), so the gate asks
// again, the body does not run and the model is not called.
func TestRun_ConfirmedCallPauses(t *testing.T) {
	var rec askRecorder
	h := newHarness(t, harnessConfig{
		tools:     []tool.Tool{askTool(t, &rec, functiontool.Config{RequireConfirmation: true})},
		responses: []*genai.Content{askCall(), modelText("done")},
	})

	h.mustRun(userText("go"))
	h.mustRun(confirmation(t, h.events(), true))

	requests := functionCalls(h.events(), workflow.WorkflowInputFunctionCallName)
	if len(requests) != 1 || requests[0].ID != "clarify_region" {
		t.Fatalf("request-input calls after the confirmation = %v, want one with id clarify_region", requests)
	}

	h.mustRun(regionAnswer())

	if got := rec.runs.Load(); got != 1 {
		t.Errorf("handler runs = %d, want 1", got)
	}
	got := responses(h.events(), "ask")
	if diff := cmp.Diff(map[string]any{"error": `error tool "ask" requires confirmation, please approve or reject`}, got[len(got)-1].Response); diff != "" {
		t.Errorf("last function response mismatch (-want +got):\n%s", diff)
	}
	if got := functionCalls(h.events(), toolconfirmation.FunctionCallName); len(got) != 2 {
		t.Errorf("confirmation requests = %d, want 2", len(got))
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
}

// A28. Fails if the replayed call_1, which finished before its sibling
// paused, is answered from the sub-scheduler's record: its body would run
// once, its SkipSummarization would be lost and the model would be called
// again.
func TestRun_FinishedSiblingRunsAgainAfterPause(t *testing.T) {
	var runs1, runs2 atomic.Int32
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if ctx.FunctionCallID() == "call_1" {
			runs1.Add(1)
			if err := emit(message("p1")); err != nil {
				return nil, err
			}
			ctx.Actions().SkipSummarization = true
			return "r1", nil
		}
		runs2.Add(1)
		return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "region"})
	})
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{calls(
		call("call_1", "analyze", map[string]any{"query": "a"}),
		call("call_2", "analyze", map[string]any{"query": "b"}),
	)}})

	h.mustRun(userText("go"))
	h.mustRun(userResponse("region", workflow.WorkflowInputFunctionCallName, map[string]any{"result": "APAC"}))

	if runs1.Load() != 2 || runs2.Load() != 2 {
		t.Errorf("handler runs of call_1, call_2 = %d, %d; want 2, 2", runs1.Load(), runs2.Load())
	}
	events := h.events()
	var p1 []evSummary
	for _, s := range summarize(authoredBy(events, "analyze")) {
		if s.Text == "p1" {
			p1 = append(p1, s)
		}
	}
	wantP1 := []evSummary{
		{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@call_1", Text: "p1"},
		{Author: "analyze", Branch: "analyze@call_1", Path: "root_agent/analyze@~call_1~1", Text: "p1"},
	}
	if diff := cmp.Diff(wantP1, p1); diff != "" {
		t.Errorf("p1 events mismatch (-want +got):\n%s", diff)
	}
	var merged []*session.Event
	for _, ev := range events {
		if len(responses([]*session.Event{ev}, "analyze")) > 0 {
			merged = append(merged, ev)
		}
	}
	if len(merged) != 1 {
		t.Fatalf("response events = %d, want 1", len(merged))
	}
	want := []*genai.FunctionResponse{
		{ID: "call_1", Name: "analyze", Response: map[string]any{"result": "r1"}},
		{ID: "call_2", Name: "analyze", Response: map[string]any{"result": "APAC"}},
	}
	if diff := cmp.Diff(want, responses(merged, "analyze")); diff != "" {
		t.Errorf("merged responses mismatch (-want +got):\n%s", diff)
	}
	if !merged[0].Actions.SkipSummarization {
		t.Error("merged response event SkipSummarization = false, want true")
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
}
